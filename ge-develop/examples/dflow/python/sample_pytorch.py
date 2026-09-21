#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import dataflow as df

'''模型执行流程
             ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
  模型输入--->│    preprocess    │--->│   model1 infer   │--->│   model2 infer   │--->│    postprocess   │--->模型输出
             └──────────────────┘    └──────────────────┘    └──────────────────┘    └──────────────────┘
'''


############## 模型定义 ##############
class FakeModel1(nn.Module):
    def __init__(self):
        super().__init__()

    # 模拟模型推理
    def forward(self, input_image):
        return F.interpolate(input_image, size=(256, 256), mode='bilinear')


class FakeModel2(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = 0.5
        self.std = 0.5

    # 模拟模型推理
    def forward(self, input_image):
        return (input_image - self.mean) / self.std


############## UDF定义 ##############
# 预处理
@df.pyflow
def preprocess(input_image):
    # 模拟图片裁切
    transform = transforms.Compose([transforms.CenterCrop(512)])
    return transform(input_image)


@df.pyflow
class Model1Runner:
    def __init__(self):
        self.model = FakeModel1()

    @df.method()
    def run(self, input_image):
        return self.model(input_image)


@df.pyflow
class Model2Runner:
    def __init__(self):
        self.model = FakeModel2()

    @df.method()
    def run(self, input_image):
        return self.model(input_image)


@df.pyflow
def postprocess(input_image):
    mean = 0.5
    std = 0.5
    img = input_image * std + mean
    return F.interpolate(img, size=(512, 512), mode='bilinear')


if __name__ == '__main__':
    options = {
        "ge.experiment.data_flow_deploy_info_path": "./config/sample_pytorch_deploy_info.json"
    }
    df.init(options)

    ############## FlowGraph构图 ##############
    # 定义图的输入
    data0 = df.FlowData()

    # 构造flow node
    preprocess_node = preprocess.fnode()
    model1_node = Model1Runner.fnode()
    model2_node = Model2Runner.fnode()
    postprocess_node = postprocess.fnode()

    # 节点连边
    preprocess_output = preprocess_node(data0)
    model1_output = model1_node.run(preprocess_output)
    model2_output = model2_node.run(model1_output)
    postprocess_output = postprocess_node(model2_output)

    # 构造FlowGraph
    dag = df.FlowGraph([postprocess_output])

    # FlowGraph执行
    image = torch.randn(1, 3, 768, 768)
    dag.feed({data0: image})
    output = dag.fetch()

    print("TEST-OUTPUT:", output)
    df.finalize()
