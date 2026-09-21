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
from ge.es.all import If


def build_if_graph():
    # 1、创建图构建器
    builder = GraphBuilder("MakeIfGraph")
    # 2、MakeIfGraph实例创建输入节点
    input_tensor_holder = builder.create_input(
        index=0,
        name="test_input",
        data_type=DataType.DT_FLOAT,
        shape=[2]
    )
    cond = builder.create_input(
        index=1,
        name="cond",
        data_type=DataType.DT_INT32,
        shape=[]
    )
    # 初始化图构建器实例，用于构建子图then_branch
    then_branch_builder = GraphBuilder("then_branch")
    then_branch_input_tensor = then_branch_builder.create_input(
        index=0,
        name="then_input",
        data_type=DataType.DT_FLOAT,
        shape=[2]
    )
    const_tensor_then_branch = then_branch_builder.create_const_int64(5)
    then_result = const_tensor_then_branch + then_branch_input_tensor
    # 将 then_result设置为then_graph_builder的输出节点
    then_branch_builder.set_graph_output(then_result, 0)
    then_graph = then_branch_builder.build_and_reset()

    else_branch_builder = GraphBuilder("else_branch")
    else_branch_input_tensor = else_branch_builder.create_input(
        index=0,
        name="then_input",
        data_type=DataType.DT_FLOAT,
        shape=[2]
    )
    const_tensor_else_branch = else_branch_builder.create_const_int64(2)
    else_result = else_branch_input_tensor + const_tensor_else_branch
    # 3、设置图输出节点
    else_branch_builder.set_graph_output(else_result, 0)
    else_graph = else_branch_builder.build_and_reset()
    inputs_tensor_holder_list = [input_tensor_holder]
    If(cond, inputs_tensor_holder_list, 1, then_graph, else_graph)
    # 4、构建图
    return builder.build_and_reset()


def dump_if_graph(graph):
    graph.dump_to_file(format=DumpFormat.kOnnx, suffix="make_if_graph")


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
        tensor1 = Tensor([1.0, 2.0], None, DataType.DT_FLOAT, Format.FORMAT_ND, [2])
        tensor_const = Tensor([1], None, DataType.DT_INT32, Format.FORMAT_ND, [])
        # 创建Tensor对象
        inputs = [tensor1, tensor_const]
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


graph = build_if_graph()
dump_if_graph(graph)
run_graph(graph)
