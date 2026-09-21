#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import numpy as np

from ge.es.graph_builder import GraphBuilder, TensorHolder
from ge.graph import Tensor
from ge.graph.types import DataType, Format
from ge.graph import Graph, DumpFormat
from ge.ge_global import GeApi
from ge.session import Session
from ge.es.all import ConcatV2


def build_concat_graph():
    # 1、创建图构建器
    builder = GraphBuilder("MakeConcatV2Graph")
    # 2、创建图输入节点
    tensor_holder1 = builder.create_input(
        index=0,
        name="tensor1",
        data_type=DataType.DT_FLOAT,
        shape=[8, 64, 128]
    )
    tensor_holder2 = builder.create_input(
        index=1,
        name="tensor2",
        data_type=DataType.DT_FLOAT,
        shape=[2, 64, 128]
    )
    tensor_holder3 = builder.create_input(
        index=2,
        name="tensor3",
        data_type=DataType.DT_FLOAT,
        shape=[6, 64, 128]
    )
    tensor_holder_list = [tensor_holder1, tensor_holder2, tensor_holder3]
    concat_tensor_holder = ConcatV2(tensor_holder_list, 0, N=3)
    # 3、设置图输出节点
    builder.set_graph_output(concat_tensor_holder, 0)
    # 4、构建图
    return builder.build_and_reset()


def dump_concat_graph(graph):
    graph.dump_to_file(format=DumpFormat.kOnnx, suffix="make_concat_graph")


def run_graph(graph) -> int:
    config = {
        "ge.exec.deviceId": "0",
        "ge.graphRunMode": "0"  # 0: 图模式, 1: 单算子模式
    }
    ge_api = GeApi()
    ret = ge_api.ge_initialize(config)
    if ret != 0:
        print(f"GE初始化失败，返回码: {ret}")
        return ret
    print("GE环境初始化成功 (Device ID: 0)")

    try:
        # 2. 创建Session
        session = Session()
        # 3. 添加图到Session
        graph_id = 1
        ret = session.add_graph(graph_id, graph)
        if ret != 0:
            print(f"添加图失败，返回码: {ret}")
            return ret
        print(f"图已添加到Session (Graph ID: {graph_id})")

        # 4. 准备输入数据
        tensor1_data = np.full(8 * 64 * 128, 1.0, dtype=np.float32)
        tensor2_data = np.full(2 * 64 * 128, 2.0, dtype=np.float32)
        tensor3_data = np.full(6 * 64 * 128, 3.0, dtype=np.float32)
        tensor1 = Tensor(
            tensor1_data.flatten().tolist(),
            None,
            DataType.DT_FLOAT,
            Format.FORMAT_ND,
            shape=[8, 64, 128]
        )
        tensor2 = Tensor(
            tensor2_data.flatten().tolist(),
            None,
            DataType.DT_FLOAT,
            Format.FORMAT_ND,
            shape=[2, 64, 128]
        )
        tensor3 = Tensor(
            tensor3_data.flatten().tolist(),
            None,
            DataType.DT_FLOAT,
            Format.FORMAT_ND,
            shape=[6, 64, 128]
        )
        input_tensor = [tensor1, tensor2, tensor3]
        # 5. 运行图
        ret = session.run_graph(graph_id, input_tensor)
        print("[Info] 图运行成功！")
        for idx, tensor in enumerate(ret, start=1):
            print(f"Tensor{idx}详情： {tensor}")
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


graph = build_concat_graph()
dump_concat_graph(graph)
run_graph(graph)
