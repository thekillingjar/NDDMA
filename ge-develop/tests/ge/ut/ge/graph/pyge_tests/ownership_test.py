#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
所有权转移测试 - 使用 pytest 框架
测试 GraphBuilder.build_and_reset() 返回的 Graph 对象的所有权转移
"""

import pytest
import sys
import os

# 需要添加 ge 到 Python 路径
try:
    from ge.es.graph_builder import GraphBuilder
    from ge.graph import Graph
    from ge.graph.types import DataType
except ImportError as e:
    pytest.skip(f"无法导入 pyge 模块: {e}", allow_module_level=True)


class TestOwnershipTransfer:
    """所有权转移测试类"""

    @pytest.fixture
    def builder(self):
        """创建 GraphBuilder 实例的 fixture"""
        return GraphBuilder("ownership_test_graph")

    def test_graph_builder_build_ownership(self, builder):
        """测试 GraphBuilder.build_and_reset() 返回的 Graph 对象拥有所有权"""
        # 创建输入和常量
        input_tensor = builder.create_input(0, data_type=DataType.DT_FLOAT, shape=[1, 3, 224, 224])

        # 设置输出
        builder.set_graph_output(input_tensor, 0)

        # 构建图
        graph = builder.build_and_reset()

        # 验证 Graph 对象拥有所有权
        assert isinstance(graph, Graph)
        assert graph._handle is not None, "Graph 对象应该有有效的句柄"

    def test_graph_ownership_with_manual_creation(self):
        """测试手动创建的 Graph 对象的所有权"""
        # 手动创建 Graph 对象
        graph = Graph("manual_graph")

        # 验证手动创建的 Graph 对象拥有所有权
        assert graph._handle is not None, "手动创建的 Graph 对象应该有有效的句柄"
