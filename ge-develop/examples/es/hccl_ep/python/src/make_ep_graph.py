#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import os
import sys
import numpy as np

from ge.es import GraphBuilder
from ge.graph import Tensor, DumpFormat
from ge.graph.types import DataType, Format
from ge.ge_global import GeApi
from ge.session import Session
from ge.es.all import *

# 定义常量
LOCAL_BATCH = 4      # 本地batch size
HIDDEN_SIZE = 512    # 隐藏层大小
NUM_EXPERTS = 8      # 专家数量
TOP_K = 2            # top-k专家
RANK_SIZE = 2        # 双卡并行


def build_ep_graph():
    # 1、创建图构建器
    builder = GraphBuilder("MakeEPGraph")

    # 2、定义输入配置
    input_configs = [
        ("hidden_states", [LOCAL_BATCH, HIDDEN_SIZE], DataType.DT_FLOAT),
        ("gate_weight", [HIDDEN_SIZE, NUM_EXPERTS], DataType.DT_FLOAT),
        ("finished", [LOCAL_BATCH], DataType.DT_BOOL),
        ("shared_output", [LOCAL_BATCH, HIDDEN_SIZE], DataType.DT_FLOAT),
        ("expert_weight", [NUM_EXPERTS, HIDDEN_SIZE, HIDDEN_SIZE], DataType.DT_INT8),
        ("group_list", [NUM_EXPERTS], DataType.DT_INT32)
    ]

    # 3、创建输入节点
    inputs = []
    for idx, (name, shape, dtype) in enumerate(input_configs):
        input_node = builder.create_input(
            index=idx,
            name=name,
            data_type=dtype,
            shape=shape
        )
        inputs.append(input_node)

    hidden_states, gate_weight, finished, shared_output, expert_weight, group_list = inputs

    # 4、构建计算图
    quant_result = DynamicQuant(
        hidden_states,
        smooth_scales=None,
        group_index=None,
        dst_type=DataType.DT_INT8
    )
    hidden_states_int8 = quant_result.y
    pertoken_scale = quant_result.scale

    global_hidden_states = HcomAllGather(
        hidden_states_int8,
        rank_size=RANK_SIZE,
        group="hccl_world_group",
        fusion=0,
        fusion_id=-1
    )

    router_logits = MatMul(hidden_states, gate_weight)
    topk_result = MoeGatingTopKSoftmax(router_logits, finished, k=TOP_K)
    topk_weights = topk_result.y
    topk_ids = topk_result.expert_idx
    row_idx = topk_result.row_idx

    topk_ids_float = Cast(topk_ids, dst_type=DataType.DT_FLOAT)
    row_idx_float = Cast(row_idx, dst_type=DataType.DT_FLOAT)
    pertoken_scale_unsqueezed = Unsqueeze(pertoken_scale, axes=[-1])
    topk_cat = ConcatV2([topk_weights, topk_ids_float, row_idx_float, pertoken_scale_unsqueezed], concat_dim=-1, N=4)

    topk_all = HcomAllGather(
        topk_cat,
        rank_size=RANK_SIZE,
        group="hccl_world_group",
        fusion=0,
        fusion_id=-1
    )

    size_splits = builder.create_vector_int64([TOP_K, TOP_K, TOP_K, 1])
    split_results = SplitV(topk_all, size_splits, -1, 4, num_split=4)
    global_topk_weights = split_results[0]
    global_topk_ids_float = split_results[1]
    global_row_idx_float = split_results[2]
    global_pertoken_scale = split_results[3]

    topk_ids_rounded = Round(global_topk_ids_float)
    row_idx_rounded = Round(global_row_idx_float)
    global_row_idx = Cast(row_idx_rounded, dst_type=DataType.DT_INT64)
    global_row_idx_flat = Reshape(global_row_idx, builder.create_vector_int64([-1]))

    axis_const = builder.create_const_int32([0], [])
    dispatched_hidden_states = GatherV2(global_hidden_states, global_row_idx_flat, axis_const)
    global_pertoken_scale_squeezed = Reshape(global_pertoken_scale, builder.create_vector_int64([-1]))
    dispatched_pertoken_scale = GatherV2(global_pertoken_scale_squeezed, global_row_idx_flat, axis_const)
    global_topk_weights_flat = Reshape(global_topk_weights, builder.create_vector_int64([-1]))
    group_list_int64 = Cast(group_list, dst_type=DataType.DT_INT64)
    scale_data = np.ones([NUM_EXPERTS, HIDDEN_SIZE], dtype=np.float32)
    weight_scale = builder.create_const_float(scale_data.flatten().tolist(), [NUM_EXPERTS, HIDDEN_SIZE])

    moe_output = GroupedMatmulFinalizeRouting(
        dispatched_hidden_states,        # x (int8)
        expert_weight,                   # w (int8)
        scale=weight_scale,              # scale
        bias=None,                       # bias（可选）
        pertoken_scale=dispatched_pertoken_scale,  # pertoken_scale
        group_list=group_list_int64,     # group_list
        shared_input=None,               # shared_input（可选）
        logit=global_topk_weights_flat,  # logit
        row_index=global_row_idx_flat,   # row_index
        offset=None,                     # offset（可选）
        dtype=DataType.DT_FLOAT,         # 输出数据类型
        shared_input_weight=0.0,         # 共享输入权重
        output_bs=LOCAL_BATCH * RANK_SIZE  # 输出batch size
    )

    final_hidden_states = HcomReduceScatter(
        moe_output,
        reduction="sum",
        group="hccl_world_group",
        rank_size=RANK_SIZE,
        fusion=0,
        fusion_id=-1
    )

    # Add shared_output
    final_output = Add(final_hidden_states, shared_output)

    # 5、设置图输出节点
    builder.set_graph_output(final_output, 0)

    # 6、构建图
    return builder.build_and_reset()


def dump_ep_graph(graph):
    graph.dump_to_file(format=DumpFormat.kOnnx, suffix="make_ep_graph")


def run_graph(graph, device_id="0") -> int:
    # 1. 初始化GE环境
    # 获取环境变量
    rank_id = os.environ.get("RANK_ID", None)
    rank_table_file = os.environ.get("RANK_TABLE_FILE", None)

    # 构建配置字典
    config = {
        "ge.exec.deviceId": str(device_id),
        "ge.graphRunMode": "0"  
    }

    if rank_id is not None and rank_table_file is not None:
        config["ge.exec.rankTableFile"] = rank_table_file
        config["ge.exec.rankId"] = rank_id
        print(f"[Info] 多卡模式 - RANK_ID: {rank_id}, RANK_TABLE_FILE: {rank_table_file}")
    else:
        print("[Info] 单卡模式 - 未检测到RANK_ID和RANK_TABLE_FILE环境变量")

    ge_api = GeApi()
    ret = ge_api.ge_initialize(config)
    if ret != 0:
        print(f"[Error] GE初始化失败，返回码: {ret}")
        return ret

    print(f"[Info] GE环境初始化成功 (Device ID: {device_id}, RANK_ID: {rank_id if rank_id else 'N/A'})")

    try:
        # 2. 创建Session
        session = Session()

        # 3. 添加图到Session
        graph_id = 1
        ret = session.add_graph(graph_id, graph)
        if ret != 0:
            print(f"[Error] 添加图失败，返回码: {ret}")
            return ret
        print(f"[Info] 图已添加到Session (Graph ID: {graph_id})")

        # 4. 准备输入数据
        # hidden_states
        hidden_states_data = np.full([LOCAL_BATCH, HIDDEN_SIZE], 1.0, dtype=np.float32)
        hidden_states_tensor = Tensor(
            hidden_states_data.flatten().tolist(),
            None,
            DataType.DT_FLOAT,
            Format.FORMAT_ND,
            [LOCAL_BATCH, HIDDEN_SIZE]
        )

        # gate_weight
        gate_weight_data = np.full([HIDDEN_SIZE, NUM_EXPERTS], 0.1, dtype=np.float32)
        gate_weight_tensor = Tensor(
            gate_weight_data.flatten().tolist(),
            None,
            DataType.DT_FLOAT,
            Format.FORMAT_ND,
            [HIDDEN_SIZE, NUM_EXPERTS]
        )

        # finished (BOOL 类型)
        finished_data = [False] * LOCAL_BATCH
        finished_tensor = Tensor(
            finished_data,
            None,
            DataType.DT_BOOL,
            Format.FORMAT_ND,
            [LOCAL_BATCH]
        )

        # shared_output
        shared_output_data = np.full([LOCAL_BATCH, HIDDEN_SIZE], 0.5, dtype=np.float32)
        shared_output_tensor = Tensor(
            shared_output_data.flatten().tolist(),
            None,
            DataType.DT_FLOAT,
            Format.FORMAT_ND,
            [LOCAL_BATCH, HIDDEN_SIZE]
        )

        # expert_weight
        expert_weight_data = np.ones([NUM_EXPERTS, HIDDEN_SIZE, HIDDEN_SIZE], dtype=np.int8)
        expert_weight_tensor = Tensor(
            expert_weight_data.flatten().tolist(),
            None,
            DataType.DT_INT8,
            Format.FORMAT_ND,
            [NUM_EXPERTS, HIDDEN_SIZE, HIDDEN_SIZE]
        )

        # group_list
        group_list_data = np.full([NUM_EXPERTS], 2, dtype=np.int32)
        group_list_tensor = Tensor(
            group_list_data.tolist(),
            None,
            DataType.DT_INT32,
            Format.FORMAT_ND,
            [NUM_EXPERTS]
        )

        inputs = [
            hidden_states_tensor,
            gate_weight_tensor,
            finished_tensor,
            shared_output_tensor,
            expert_weight_tensor,
            group_list_tensor
        ]

        print(f"[Info] 输入数据已准备，共{len(inputs)}个输入tensor")

        # 5. 运行图
        ret = session.run_graph(graph_id, inputs)
        print("[Info] 图运行成功！")
        for idx, tensor in enumerate(ret, start=1):
            print(f"Tensor{idx}详情：{tensor}")
        return 0

    except Exception as e:
        print(f"[Error] 执行过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return -1

    finally:
        # 6. 清理GE环境
        print("[Info] 清理GE环境...")
        ge_api.ge_finalize()
        print("[Success] GE环境已清理")


if __name__ == "__main__":
    # 构建图
    graph = build_ep_graph()

    # 先dump图（生成pbtxt文件用于可视化）
    dump_ep_graph(graph)

    # 运行图（从命令行参数获取device_id，默认为"0"）
    device_id = sys.argv[1] if len(sys.argv) > 1 else "0"
    ret = run_graph(graph, device_id)
    sys.exit(ret)
