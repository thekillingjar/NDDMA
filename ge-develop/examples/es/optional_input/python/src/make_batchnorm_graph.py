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

import os
import sys
import numpy as np

from ge.es.graph_builder import GraphBuilder, TensorHolder
from ge.graph import Tensor
from ge.graph.types import DataType, Format
from ge.graph import Graph, DumpFormat
from ge.ge_global import GeApi
from ge.session import Session
from ge.es.all import BatchNorm


def build_batch_norm_graph():
    # 1、创建图构建器
    builder = GraphBuilder("MakeBatchNormGarph")
    # 2、创建图输入节点 
    input_tensor_holder = builder.create_input(
        index=0,
        name="input",
        data_type=DataType.DT_FLOAT,
        format=Format.FORMAT_NCHW,
        shape=[1, 3, 1, 2]
    )
    mean = builder.create_input(
        index=1,
        name="mean",
        data_type=DataType.DT_FLOAT,
        shape=[3]
    )
    variance = builder.create_input(
        index=2,
        name="variance",
        data_type=DataType.DT_FLOAT,
        shape=[3]
    )
    scale = builder.create_const_float([1.0, 1.0, 1.0], shape=[3])
    offset = builder.create_const_float([0.0, 0.0, 0.0], shape=[3])
    batchNorm_tensor_holder = BatchNorm(input_tensor_holder, scale, offset, mean, variance,
                                        epsilon=1e-4, data_format="NCHW", is_training=False)
    # 3、设置图输出节点
    builder.set_graph_output(batchNorm_tensor_holder.y, 0)
    # 4、构建图
    return builder.build_and_reset()


def dump_batch_norm_graph(graph):
    graph.dump_to_file(format=DumpFormat.kOnnx, suffix="make_batch_norm_graph")


def run_graph(graph, device_id="0") -> int:
    # 1. 初始化GE环境
    config = {
        "ge.exec.deviceId": str(device_id),
        "ge.graphRunMode": "0"
    }

    ge_api = GeApi()
    ret = ge_api.ge_initialize(config)
    if ret != 0:
        print(f"[Error] GE初始化失败，返回码: {ret}")
        return ret

    print(f"[Info] GE环境初始化成功 (Device ID: {device_id})")

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
        input_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
        mean_data = np.array([2.0, 4.0, 6.0], dtype=np.float32)
        variance_data = np.array([0.5, 0.5, 0.5], dtype=np.float32)

        input_tensor = Tensor(
            input_data.tolist(),
            None,
            DataType.DT_FLOAT,
            Format.FORMAT_NCHW,
            [1, 3, 1, 2]
        )
        mean_tensor = Tensor(
            mean_data.tolist(),
            None,
            DataType.DT_FLOAT,
            Format.FORMAT_ND,
            [3]
        )
        variance_tensor = Tensor(
            variance_data.tolist(),
            None,
            DataType.DT_FLOAT,
            Format.FORMAT_ND,
            [3]
        )

        inputs = [input_tensor, mean_tensor, variance_tensor]
        print(f"[Info] 输入数据已准备，共{len(inputs)}个输入tensor (input, mean, variance)")

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
    graph = build_batch_norm_graph()

    # 先dump图（生成pbtxt文件用于可视化）
    dump_batch_norm_graph(graph)

    # 运行图（从命令行参数获取device_id，默认为"0"）
    run_graph(graph)
