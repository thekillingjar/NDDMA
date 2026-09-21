#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
import torch
import torch.nn as nn
import dataflow as df


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv_relu = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.conv_relu(x)
        return out


class MatMulNetwork(nn.Module):
    def __init__(self, num_matrices):
        super(MatMulNetwork, self).__init__()
        self.matrices = nn.ModuleList([
            nn.Linear(32, 32) for _ in range(num_matrices)
        ])

    def forward(self, x):
        result = x
        for matrix in self.matrices:
            result = torch.matmul(result, matrix.weight.T)
        return result


@df.pyflow()
class ConvFunc:
    def __init__(self):
        self.model = SimpleNet().eval()

    @df.method(num_returns=1)
    @torch.inference_mode()
    def compute(self, data):
        return torch.mean(data.to(torch.float), dim=(0, 2, 3))


@df.pyflow()
class MatmulFunc:
    def __init__(self):
        self.model = MatMulNetwork(num_matrices=1).eval()

    @df.method(num_returns=1)
    @torch.inference_mode()
    def compute(self, data):
        return torch.mean(data.to(torch.float), dim=(0, 2, 3))


@df.pyflow(num_returns=1)
def convert_data(data):
    return df.Tensor(data.numpy().astype(np.int32))


# 目标构图如下所示
'''
     FlowData     FlowData     FlowData 
       |             |             /       
       |             |            /
       |             |           / 
    SimpleNet MatMulNetwork     /  
       \             /         /   
        \           /         /    
         \         /         /     
          \       /         /      
         FlowNode(onnx)    /       
       GraphProcessPoint  /        
	            \        /
			     \      /
				  \    /
				   \  /
				    \/
				FlowNode(pb)
              GraphProcessPoint  
'''


class SampleFlowGraph:
    def __init__(self):
        self.data0 = None
        self.data1 = None
        self.data2 = None
        self.flow_graph = None
        self.options = {
            "ge.exec.deviceId": "0",
            "ge.experiment.data_flow_deploy_info_path": "./config/multi_model_deploy.json"
        }

    def init(self):
        df.init(self.options)
        self.build_flow_graph()

    def build_flow_graph(self):
        # 定义输入
        self.data0 = df.FlowData()
        self.data1 = df.FlowData()
        self.data2 = df.FlowData()
        # python 类对象直接构造UDF节点
        conv_node = ConvFunc.fnode().set_alias('conv_model')
        matmul_node = MatmulFunc.fnode().set_alias('matmul_model')
        # 构造Graph process point，该节点支持执行onnx模型，运行时将模型下沉到device上执行
        graph_pp0 = df.GraphProcessPoint(df.Framework.ONNX, "config/simple_model.onnx",
                                         load_params={"input_data_names": "X1,X2"},
                                         compile_config_path="config/add_graph.json")
        onnx_node = df.FlowNode(input_num=2, output_num=1, name='onnx_node')
        onnx_node.add_process_point(graph_pp0)
        # 构造Graph process point，该节点支持执行onnx模型，运行时将模型下沉到device上执行
        graph_pp1 = df.GraphProcessPoint(df.Framework.TENSORFLOW, "config/add.pb",
                                         load_params={"input_data_names": "Placeholder,Placeholder_1"},
                                         compile_config_path="config/add_graph.json")
        pb_node = df.FlowNode(input_num=2, output_num=1, name='pb_node')
        pb_node.add_process_point(graph_pp1)

        # 构建连边关系
        conv_result = conv_node(self.data0)
        matmul_result = matmul_node(self.data1)
        convert_data0 = convert_data.fnode().set_alias('convert_data0')(conv_result)
        convert_data1 = convert_data.fnode().set_alias('convert_data1')(matmul_result)
        onnx_result = onnx_node(convert_data0, convert_data1)
        pb_result = pb_node(onnx_result, self.data2)
        self.flow_graph = df.FlowGraph([pb_result])

    def feed(self, inputs):
        feed_dict = {self.data0: inputs['data0'], self.data1: inputs['data1'], self.data2: inputs['data2']}
        self.flow_graph.feed(feed_dict, timeout=-1)

    def fetch(self, indexes=None, timeout=1000 * 6 * 5):
        return self.flow_graph.fetch(indexes=indexes, timeout=timeout)

    def finalize(self):
        df.finalize()


def main():
    runner = SampleFlowGraph()
    runner.init()
    shape = (1, 3, 32, 32)
    low = 0
    high = 101
    input0_tensor = torch.randint(low, high, shape, dtype=torch.int32)
    input1_tensor = torch.randint(low, high, shape, dtype=torch.int32)
    input2_tensor = df.Tensor(np.ones([3], dtype=np.int32))

    inputs = {"data0": input0_tensor, "data1": input1_tensor, "data2": input2_tensor}
    runner.feed(inputs)
    output = runner.fetch()
    print(f'Flow graph execute success. result is: {output}', flush=True)
    runner.finalize()


if __name__ == "__main__":
    main()
