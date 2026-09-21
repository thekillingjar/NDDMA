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

BATCH_SIZE = 2
SEQ_LEN = 128
HIDDEN_SIZE = 512


def build_pfa_hcom_graph():
    # 1、创建图构建器
    builder = GraphBuilder("MakePfaHcomGraph")

    # 2、定义输入配置 
    input_configs = [
        ("query", [BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE]),
        ("key", [BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE]),
        ("value", [BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE]),
        ("atten_mask", [BATCH_SIZE, SEQ_LEN, SEQ_LEN]),
        ("quant_scale2", [1]),
        ("quant_offset2", [1]),
        ("mm_x2", [BATCH_SIZE, HIDDEN_SIZE, HIDDEN_SIZE]),
        ("arn_x1", [BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE]),
        ("arn_gamma", [HIDDEN_SIZE])
    ]

    # 3、创建输入节点并转换为FP16
    inputs_fp32 = []
    inputs_fp16 = []
    for idx, (name, shape) in enumerate(input_configs):
        input_fp32 = builder.create_input(
            index=idx,
            name=name,
            data_type=DataType.DT_FLOAT,
            shape=shape
        )
        inputs_fp32.append(input_fp32)
        inputs_fp16.append(Cast(input_fp32, dst_type=DataType.DT_FLOAT16))

    query_fp16, key_fp16, value_fp16, atten_mask_fp16, quant_scale2_fp16, \
    quant_offset2_fp16, mm_x2_fp16, arn_x1_fp16, arn_gamma_fp16 = inputs_fp16

    # 4、构建计算图
    pfa_output = PromptFlashAttention(
        query_fp16,           # query
        key_fp16,             # key
        value_fp16,           # value
        None,                 # pse_shift（可选）
        atten_mask_fp16,      # atten_mask
        None,                 # actual_seq_lengths（可选）
        None,                 # actual_seq_lengths_kv（可选）
        None,                 # dequant_scale1（可选）
        None,                 # quant_scale1（可选）
        None,                 # dequant_scale2（可选）
        quant_scale2_fp16,    # quant_scale2
        quant_offset2_fp16,   # quant_offset2
        num_heads=8,          # 注意力头数
        scale_value=1.0,      # 缩放因子
        pre_tokens=214748647, # 预分配token数
        next_tokens=0,        # 下一轮token数
        input_layout="BSH",   # 输入布局：Batch, Sequence, Hidden
        num_key_value_heads=8,# KV头数
        sparse_mode=0,        # 稀疏模式
        inner_precise=1       # 内部精度
    )

    # Reshape变换
    reshape_shape = builder.create_vector_int64([2, 128, 512])
    reshape_output = Reshape(pfa_output, reshape_shape)

    # BatchMatMul矩阵乘法
    mm_output = BatchMatMul(reshape_output, mm_x2_fp16)

    # HcomAllReduce多卡通信
    hcom_output = HcomAllReduce(
        mm_output,
        reduction="sum",           
        group="hccl_world_group"   
    )

    # Cast为FP16
    cast_output = Cast(hcom_output, dst_type=DataType.DT_FLOAT16)

    # AddRmsNorm归一化
    arn_output = AddRmsNorm(arn_x1_fp16, cast_output, arn_gamma_fp16)

    # 5、将中间结果转换回FP32作为输出
    mm_output_fp32 = Cast(mm_output, dst_type=DataType.DT_FLOAT)
    hcom_output_fp32 = Cast(hcom_output, dst_type=DataType.DT_FLOAT)
    arn_output_fp32 = Cast(arn_output.y, dst_type=DataType.DT_FLOAT)

    # 6、设置图输出节点
    builder.set_graph_output(mm_output_fp32, 0)
    builder.set_graph_output(hcom_output_fp32, 1)
    builder.set_graph_output(arn_output_fp32, 2)

    # 7、构建图
    return builder.build_and_reset()


def dump_pfa_hcom_graph(graph):
    graph.dump_to_file(format=DumpFormat.kOnnx, suffix="make_pfa_hcom_graph")


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

    # 添加HCCL相关配置
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
        input_data_configs = [
            ([BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE], 1.0),       # query
            ([BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE], 1.0),       # key
            ([BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE], 1.0),       # value
            ([BATCH_SIZE, SEQ_LEN, SEQ_LEN], 0.0),           # atten_mask
            ([1], 1.0),                                      # quant_scale2
            ([1], 0.0),                                      # quant_offset2
            ([BATCH_SIZE, HIDDEN_SIZE, HIDDEN_SIZE], 1.0),   # mm_x2
            ([BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE], 1.0),       # arn_x1
            ([HIDDEN_SIZE], 1.0)                             # arn_gamma
        ]

        # 创建Tensor对象
        inputs = []
        for shape, init_value in input_data_configs:
            data = np.full(shape, init_value, dtype=np.float32)
            tensor = Tensor(
                data.flatten().tolist(),
                None,
                DataType.DT_FLOAT,
                Format.FORMAT_ND,
                shape
            )
            inputs.append(tensor)

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
    graph = build_pfa_hcom_graph()

    # 先dump图（生成pbtxt文件用于可视化）
    dump_pfa_hcom_graph(graph)

    # 运行图（从命令行参数获取device_id，默认为"0"）
    device_id = sys.argv[1] if len(sys.argv) > 1 else "0"
    ret = run_graph(graph, device_id)
    sys.exit(ret)