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
import dataflow as df

# dataflow 初始化参数按需设置
options = {
    "ge.exec.deviceId": "0",
    "ge.exec.logicalDeviceClusterDeployMode": "SINGLE",
    "ge.exec.logicalDeviceId": "[0:0]"
}
df.init(options)

# 构图
'''
      FlowData     FlowData
       |    \       /  |
       |     \     /   |
       |      \   /    |
       |       \ /     |
       |        /\     |
       |       /  \    |
       |      /    \   |
      FlowNode     FlowNode
FuncProcessPoint   FuncProcessPoint
        \            /
          \         /
           \       /
           FlowNode
      GraphProcessPoint
'''

# 定义输入
data0 = df.FlowData()
data1 = df.FlowData()

# 定义FuncProcessPoint实现Add功能的FlowNode
pp0 = df.FuncProcessPoint(compile_config_path="config/add_func.json")
pp0.set_init_param("out_type", df.DT_INT32)  # 按UDF实际实现设置
pp0.set_init_param("enableExceptionCatch", True)  # 按UDF实际实现设置

flow_node0 = df.FlowNode(input_num=2, output_num=1)
flow_node0.add_process_point(pp0)
# 给flowNode设置flowAttr相关属性，按需设置
flow_node0.set_attr("_flow_attr", True)
flow_node0.set_attr("_flow_attr_depth", 108)
flow_node0.set_attr("_flow_attr_enqueue_policy", "FIFO")

# 定义FuncProcessPoint调用实现add功能的GraphProcessPoint
pp1 = df.FuncProcessPoint(compile_config_path="config/invoke_func.json")
pp1.set_init_param("enableExceptionCatch", True)  # 按UDF实际实现设置
pp2 = df.GraphProcessPoint(df.Framework.TENSORFLOW, "config/add.pb",
                           load_params={"input_data_names": "Placeholder,Placeholder_1"},
                           compile_config_path="config/add_graph.json")
pp1.add_invoked_closure("invoke_graph", pp2)

flow_node1 = df.FlowNode(input_num=2, output_num=1)
flow_node1.add_process_point(pp1)

# 定义GraphProcessPoint实现Add功能的FlowNode
pp3 = df.GraphProcessPoint(df.Framework.TENSORFLOW, "config/add.pb",
                           load_params={"input_data_names": "Placeholder,Placeholder_1"},
                           compile_config_path="config/add_graph.json")
flow_node2 = df.FlowNode(input_num=2, output_num=1)
flow_node2.add_process_point(pp3)

# 构建连边关系
flow_node0_out = flow_node0(data0, data1)
flow_node1_out = flow_node1(data0, data1)
flow_node2_out = flow_node2(flow_node0_out, flow_node1_out)

# 构建FlowGraph
dag = df.FlowGraph([flow_node2_out])
# 使能异常上报，按需设置
dag.set_inputs_align_attrs(100, 1000, False)
dag.set_exception_catch(True)
# feed data
feed_data0 = np.ones([3], dtype=np.int32)
feed_data1 = np.array([1, 2, 3], dtype=np.int32)

flow_info = df.FlowInfo()
flow_info.start_time = 0
flow_info.end_time = 5

# 异步喂数据
for i in range(10):
    dag.feed_data({data0: feed_data0, data1: feed_data1}, flow_info)

# 获取输出
for i in range(10):
    result = dag.fetch_data()
    print("TEST-OUTPUT:", result)

# 释放
df.finalize()
