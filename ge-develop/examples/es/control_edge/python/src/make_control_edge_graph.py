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
from ge.es.all import Add
from ge.es.graph_builder import control_dependency_scope


def build_control_edge_graph():
    # 1. 创建图构建器
    builder = GraphBuilder("control_dep_scope_example")
    # 2. 创建依赖源节点
    input_tensor_holder0 = builder.create_input(
        index=0,
        name="input",
        data_type=DataType.DT_FLOAT,
        shape=[2, 2]
    )
    input_tensor_holder1 = builder.create_input(
        index=1,
        name="input1",
        data_type=DataType.DT_FLOAT,
        shape=[2, 2]
    )
    # 3. 使用 scope：在 scope 内创建的所有节点自动依赖 input_tensor_holder0 和 input_tensor_holder1
    with control_dependency_scope([input_tensor_holder0, input_tensor_holder1]):
        # 在此 scope 内创建的节点会自动添加控制依赖
        input_tensor_holder2 = builder.create_input(
            index=2,
            name="input2",
            data_type=DataType.DT_FLOAT,
            shape=[2, 2]
        )
        input_tensor_holder3 = input_tensor_holder0 + input_tensor_holder2
    # 4. 设置输出并构建
    builder.set_graph_output(input_tensor_holder3, 0)
    return builder.build_and_reset()


def dump_control_edge_graph(graph):
    graph.dump_to_file(format=DumpFormat.kOnnx, suffix="make_control_edge_graph")


def run_graph(graph) -> None:
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
        session = Session()
        graph_id = 1
        ret = session.add_graph(graph_id, graph)
        if ret != 0:
            print(f"添加图失败，返回码: {ret}")
            return ret
        print(f"图已添加到Session (Graph ID: {graph_id})")

        # 4. 准备输入数据
        tensor1 = Tensor([2.0, 2.0, 2.0, 2.0], None, DataType.DT_FLOAT, Format.FORMAT_ND, [2, 2])
        tensor2 = Tensor([1.0, 1.0, 1.0, 1.0], None, DataType.DT_FLOAT, Format.FORMAT_ND, [2, 2])
        tensor3 = Tensor([1.0, 1.0, 1.0, 1.0], None, DataType.DT_FLOAT, Format.FORMAT_ND, [2, 2])
        # 创建Tensor对象
        inputs = [tensor1, tensor2, tensor3]
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


graph = build_control_edge_graph()
dump_control_edge_graph(graph)
run_graph(graph)
