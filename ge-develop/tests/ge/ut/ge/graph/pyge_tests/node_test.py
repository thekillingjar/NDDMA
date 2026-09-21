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
Node 功能测试 - 使用 pytest 框架
测试 ge.graph.node.py 中的 Node 类的各种功能
"""

from typing import List
import pytest
import sys
import os
import ctypes

# 需要添加 ge 到 Python 路径，否则会报错
try:
    from ge.es.graph_builder import GraphBuilder, TensorHolder
    from ge.graph import Graph, Node
    from ge.graph.types import DataType, Format
    from ge._capi.pyes_graph_builder_wrapper import is_generated_lib_available
except ImportError as e:
    pytest.skip(f"无法导入 ge 模块: {e}", allow_module_level=True)


class TestNode:
    """Node 功能测试类"""

    @pytest.fixture
    def builder(self):
        """创建 GraphBuilder 实例的 fixture"""
        return GraphBuilder("test_graph")

    @pytest.fixture
    def graph(self, builder):
        """创建 Graph 实例的 fixture"""
        # 创建输入节点
        input_tensor = builder.create_input(
            index=0,
            name="test_input",
            data_type=DataType.DT_FLOAT,
            shape=[1, 3, 224, 224]
        )

        # 创建常量节点
        const_tensor = builder.create_const_float(1.0)

        # 设置输出
        builder.set_graph_output(input_tensor, 0)
        builder.set_graph_output(const_tensor, 1)

        # 构建图
        return builder.build_and_reset()

    @pytest.fixture
    def node(self, graph):
        """创建 Node 实例的 fixture"""
        # 从图中获取所有节点
        nodes = graph.get_all_nodes()
        assert isinstance(nodes, list)
        assert len(nodes) > 0

        # 返回最后一个节点用于测试
        return nodes[-1]

    @pytest.fixture
    def node_first(self, graph):
        """创建 Node 实例的 fixture"""
        # 从图中获取所有节点
        nodes = graph.get_all_nodes()
        assert isinstance(nodes, list)
        assert len(nodes) > 0

        # 返回第一个节点用于测试
        return nodes[0]


    def test_node_create_from_array(self, graph):
        """测试从数组创建 Node 对象"""
        # 从图中获取所有节点
        nodes = graph.get_all_nodes()  # 内部调用create_from_array
        # 验证节点列表
        assert isinstance(nodes, list)
        assert len(nodes) > 0
        for node in nodes:
            assert isinstance(node, Node)

    def test_node_name_property(self, node):
        """测试 Node name 属性"""
        name = node.name
        assert isinstance(name, str)
        assert len(name) > 0  # 节点名称不应为空

    def test_node_type_property(self, node):
        """测试 Node type 属性"""
        node_type = node.type
        assert isinstance(node_type, str)
        assert len(node_type) > 0  # 节点类型不应为空

    def test_get_in_control_nodes(self, node):
        """测试获取输入控制节点"""
        control_nodes = node.get_in_control_nodes()
        assert isinstance(control_nodes, list)
        assert len(control_nodes) == 0  # 控制节点当前为空

    def test_get_out_control_nodes(self, node):
        """测试获取输出控制节点"""
        control_nodes = node.get_out_control_nodes()
        assert isinstance(control_nodes, list)
        assert len(control_nodes) == 0  # 控制节点当前为空

    def test_get_in_data_nodes_and_port_indexes(self, node):
        """测试获取输入数据节点和端口索引"""
        in_node, port_index = node.get_in_data_nodes_and_port_indexes(0)
        assert isinstance(in_node, Node)
        assert isinstance(port_index, int)
        assert in_node.type == "Data"
        assert port_index == 0

    def test_get_in_data_nodes_and_port_indexes_invalid_index(self, node):
        """测试获取输入数据节点无效索引"""
        with pytest.raises(TypeError, match="Input index must be an integer"):
            node.get_in_data_nodes_and_port_indexes("invalid")

    def test_get_out_data_nodes_and_port_indexes(self, node_first):
        """测试获取输出数据节点和端口索引"""
        out_data_nodes_and_port_indexes = node_first.get_out_data_nodes_and_port_indexes(0)
        assert isinstance(out_data_nodes_and_port_indexes, List)
        out_node, port_index = out_data_nodes_and_port_indexes[0]
        assert isinstance(port_index, int)
        assert isinstance(out_node, Node)
        assert out_node.type == "NetOutput"
        assert port_index == 0


    def test_get_out_data_nodes_and_port_indexes_error(self, node_first):
        handle = node_first._handle
        try:
            with pytest.raises(RuntimeError, match="Failed to get output data node for index 0"):
                node_first._handle = ctypes.c_void_p(None)
                out_data_nodes_and_port_indexes = node_first.get_out_data_nodes_and_port_indexes(0)
        finally:
            node_first._handle = handle
        
    def test_get_out_data_nodes_and_port_indexes_invalid_index(self, node):
        """测试获取输出数据节点无效索引"""
        with pytest.raises(TypeError, match="Output index must be an integer"):
            node.get_out_data_nodes_and_port_indexes("invalid")


    def test_get_attr(self, node):
        """测试获取属性"""
        # 先设置一个属性
        node.set_attr("test_attr", "test_value")

        # 获取属性
        retrieved_attr = node.get_attr("test_attr")
        assert retrieved_attr == "test_value"

    def test_get_attr_invalid_key(self, node):
        """测试获取无效键属性"""
        with pytest.raises(TypeError, match="Attribute key must be a string"):
            node.get_attr(123)

    def test_set_attr(self, node):
        """测试设置属性"""

        # 设置属性
        node.set_attr("test_attr", "test_value")

        # 验证设置成功
        retrieved_attr = node.get_attr("test_attr")
        assert retrieved_attr == "test_value"

    def test_set_attr_invalid_key(self, node):
        """测试设置无效键属性"""

        with pytest.raises(TypeError, match="Attribute key must be a string"):
            node.set_attr(123, "test_value")

    def test_get_input_attr(self, node):
        """测试获取输入属性"""
        # 先设置一个输入属性
        node.set_input_attr("input_attr", 0, "input_value")

        # 获取输入属性
        retrieved_attr = node.get_input_attr("input_attr", 0)
        assert retrieved_attr == "input_value"

    def test_get_input_attr_invalid_attr_name(self, node):
        """测试获取输入属性无效属性名"""
        with pytest.raises(TypeError, match="Attribute name must be a string"):
            node.get_input_attr(123, 0)

    def test_get_input_attr_invalid_index(self, node):
        """测试获取输入属性无效索引"""
        with pytest.raises(TypeError, match="Input index must be an integer"):
            node.get_input_attr("attr", "invalid")

    def test_set_input_attr(self, node):
        """测试设置输入属性"""

        # 设置输入属性
        node.set_input_attr("input_attr", 0, "input_value")

        # 验证设置成功
        retrieved_attr = node.get_input_attr("input_attr", 0)
        assert retrieved_attr == "input_value"

    def test_set_input_attr_invalid_attr_name(self, node):
        """测试设置输入属性无效属性名"""

        with pytest.raises(TypeError, match="Attribute name must be a string"):
            node.set_input_attr(123, 0, "input_value")

    def test_set_input_attr_invalid_index(self, node):
        """测试设置输入属性无效索引"""

        with pytest.raises(TypeError, match="Input index must be an integer"):
            node.set_input_attr("attr", "invalid", "input_value")


    def test_get_output_attr(self, node):
        """测试获取输出属性"""
        # 先设置一个输出属性
        node.set_output_attr("output_attr", 0, "output_value")

        # 获取输出属性
        retrieved_attr = node.get_output_attr("output_attr", 0)
        assert retrieved_attr == "output_value"

    def test_get_output_attr_invalid_attr_name(self, node):
        """测试获取输出属性无效属性名"""
        with pytest.raises(TypeError, match="Attribute name must be a string"):
            node.get_output_attr(123, 0)

    def test_get_output_attr_invalid_index(self, node):
        """测试获取输出属性无效索引"""
        with pytest.raises(TypeError, match="Output index must be an integer"):
            node.get_output_attr("attr", "invalid")

    def test_set_output_attr(self, node):
        """测试设置输出属性"""

        # 设置输出属性
        node.set_output_attr("output_attr", 0, "output_value")

        # 验证设置成功
        assert node.get_output_attr("output_attr", 0) == "output_value"

    def test_set_output_attr_invalid_attr_name(self, node):
        """测试设置输出属性无效属性名"""

        with pytest.raises(TypeError, match="Attribute name must be a string"):
            node.set_output_attr(123, 0, "output_value")

    def test_set_output_attr_invalid_index(self, node):
        """测试设置输出属性无效索引"""

        with pytest.raises(TypeError, match="Output index must be an integer"):
            node.set_output_attr("attr", "invalid", "output_value")


    def test_node_attr_operations_with_different_types(self, node):
        """测试不同属性类型的操作"""
        # 测试字符串属性
        node.set_attr("str_attr", "hello")
        retrieved_str = node.get_attr("str_attr")
        assert retrieved_str == "hello"

        # 测试整数属性
        node.set_attr("int_attr", 42)
        retrieved_int = node.get_attr("int_attr")
        assert retrieved_int == 42

        # 测试浮点数属性
        node.set_attr("float_attr", 3.14)
        retrieved_float = node.get_attr("float_attr")
        assert abs(retrieved_float - 3.14) < 1e-6

        # 测试布尔属性
        node.set_attr("bool_attr", True)
        retrieved_bool = node.get_attr("bool_attr")
        assert retrieved_bool is True
    
    def test_get_inputs_size(self, node):
        # 获取输入数量
        inputs_size = node.get_inputs_size()

        # 测试输入数量
        assert inputs_size > 0

    def test_get_outputs_size(self, node):
        # 获取输出数量
        outputs_size = node.get_outputs_size()

        # 测试输出数量
        assert outputs_size > 0

    def test_has_attr(self, node):
        # 设置属性
        node.set_attr("test_attr", "test_value")

        # 测试HasAttr
        has_attr = node.has_attr("test_attr")
        assert has_attr is True
    
    def test_has_attr_invalid(self, node):
        with pytest.raises(TypeError, match="attr_name must be a string"):
            node.has_attr(True)
