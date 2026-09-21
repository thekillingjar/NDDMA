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

import dataflow as df
import time

# dataflow 初始化参数设置
options = {
    "ge.exec.deviceId": "0",
    "ge.exec.logicalDeviceClusterDeployMode": "SINGLE",
    "ge.exec.logicalDeviceId": "[0:0]"
}
df.init(options)

# 定义输入
data0 = df.FlowData()
data1 = df.FlowData()

# 定义FuncProcessPoint
pp0 = df.FuncProcessPoint(compile_config_path='config/add_func.json')
pp0.set_init_param("out_type", df.DT_INT32)

# 创建计算节点
flow_node0 = df.FlowNode(input_num=2, output_num=1)
flow_node0.add_process_point(pp0)

# 构建连边关系
flow_node0_out = flow_node0(data0, data1)

# 构建FlowGraph
dag = df.FlowGraph([flow_node0_out])

# feed
feed_data = df.Tensor([4, 7, 5], tensor_desc=df.TensorDesc(df.DT_INT32, [3]))

flow_info = df.FlowInfo()

# warm up
for i in range(3):
    dag.feed_data({data0: feed_data, data1: feed_data}, flow_info)

s = time.time()
for i in range(10):
    dag.feed_data({data0: feed_data, data1: feed_data}, flow_info)
e = time.time()

print(f"TEST-TIME: fetch cost {(e - s * 1000000)} us")
print("TEST SUCCESS")

# 释放dataflow资源
df.finalize()
