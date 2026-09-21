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
Graph 功能测试 - 使用 pytest 框架
测试 graph.py 中的 Graph 类的各种功能
"""

import pytest
import sys
import os
import ctypes

# 添加 ge 到 Python 路径
try:
    from ge.graph import Graph
    from ge.graph._attr import _AttrValue as AttrValue
    from ge.graph.types import AttrValueType, DataType
except ImportError as e:
    pytest.skip(f"无法导入 ge 模块: {e}", allow_module_level=True)


class TestGraph:
    """Graph 功能测试类"""

    @pytest.fixture
    def graph(self):
        """创建 Graph 实例的 fixture"""
        return Graph("test_graph")

    @pytest.fixture
    def graph_with_subgraph(self):
        """创建 含有子图的 Graph 实例的 fixture"""
        from ge.es.graph_builder import GraphBuilder
        from es_ut_test import phony_If
        builder = GraphBuilder("graph_with_subgraph")
        input_tensor = builder.create_input(0)

        then_graph_builder = GraphBuilder("then_graph")
        then_graph_const = then_graph_builder.create_const_int64(1)
        then_graph_builder.set_graph_output(then_graph_const, 0)
        then_graph = then_graph_builder.build_and_reset()

        else_graph_builder = GraphBuilder("else_graph")
        else_graph_const = else_graph_builder.create_const_int64(1)
        else_graph_builder.set_graph_output(else_graph_const, 0)
        else_graph = else_graph_builder.build_and_reset()

        phony_if_res = phony_If(input_tensor, [1, 2], 1, then_graph, else_graph)

        builder.set_graph_output(phony_if_res[0], 0)
        return builder.build_and_reset()

    @pytest.fixture
    def graph_with_data(self):
        """创建 含有实际数据的的 Graph 实例的 fixture"""
        from ge.es.graph_builder import GraphBuilder
        builder = GraphBuilder()
        input_tensor = builder.create_input(
            index=0,
            name="test_input",
            data_type=DataType.DT_FLOAT,
            shape=[1, 3, 224, 224]
        )

        # 创建常量节点
        const_tensor1 = builder.create_const_float(1.0)
        const_tensor2 = builder.create_const_int64(10)
        
        # 设置输出
        builder.set_graph_output(input_tensor, 0)
        builder.set_graph_output(const_tensor1, 1)
        builder.set_graph_output(const_tensor2, 2)

        return builder.build_and_reset()

    def test_graph_creation(self):
        """测试 Graph 创建"""
        graph = Graph("my_graph")
        assert graph is not None
        assert hasattr(graph, '_handle')
        assert hasattr(graph, 'name')

    def test_graph_creation_with_default_name(self):
        """测试使用默认名称创建 Graph"""
        graph = Graph()
        assert graph is not None
        assert graph.name == "graph"

    def test_graph_creation_invalid_name(self):
        """测试无效名称创建 Graph"""
        with pytest.raises(TypeError, match="Graph name must be a string"):
            Graph(123)

    def test_graph_create_from_invalid(self):
        """测试从无效指针创建 Graph"""
        with pytest.raises(ValueError, match="Graph pointer cannot be None"):
            Graph._create_from(None)

    def test_graph_name_property(self, graph):
        """测试 Graph name 属性"""
        name = graph.name
        assert isinstance(name, str)
        assert name == "test_graph"

    def test_get_all_nodes(self, graph):
        """测试获取所有节点"""
        nodes = graph.get_all_nodes()
        assert isinstance(nodes, list)
        # 新创建的图应该没有节点
        assert len(nodes) == 0

    def test_get_attr(self, graph):
        """测试获取属性"""
        # 先设置一个属性
        graph.set_attr("test_attr", "test_value")
        
        # 获取属性
        retrieved_attr = graph.get_attr("test_attr")
        assert isinstance(retrieved_attr, str)
        assert retrieved_attr == "test_value"

    def test_get_attr_invalid_key(self, graph):
        """测试获取无效键属性"""
        with pytest.raises(TypeError, match="Attribute key must be a string"):
            graph.get_attr(123)

    def test_set_attr(self, graph):
        """测试设置属性"""
        # 设置属性
        graph.set_attr("test_attr", "test_value")
        
        # 验证设置成功
        retrieved_attr = graph.get_attr("test_attr")
        assert retrieved_attr == "test_value"

    def test_set_attr_invalid_key(self, graph):
        """测试设置无效键属性"""
        with pytest.raises(TypeError, match="Attribute key must be a string"):
            graph.set_attr(123, "test_value")

    def test_copy_not_supported(self, graph):
        """测试复制不支持"""
        with pytest.raises(RuntimeError, match="Graph does not support copy"):
            graph.__copy__()

    def test_deepcopy_not_supported(self, graph):
        """测试深复制不支持"""
        with pytest.raises(RuntimeError, match="Graph does not support deepcopy"):
            graph.__deepcopy__({})

    def test_attr_operations_with_different_types(self, graph):
        """测试不同属性类型的操作"""
        # 测试字符串属性
        graph.set_attr("str_attr", "hello")
        retrieved_str = graph.get_attr("str_attr")
        assert retrieved_str == "hello"

        # 测试整数属性
        graph.set_attr("int_attr", 42)
        retrieved_int = graph.get_attr("int_attr")
        assert retrieved_int == 42

        # 测试浮点数属性
        graph.set_attr("float_attr", 3.14)
        retrieved_float = graph.get_attr("float_attr")
        assert abs(retrieved_float - 3.14) < 1e-6

        # 测试布尔属性
        graph.set_attr("bool_attr", True)
        retrieved_bool = graph.get_attr("bool_attr")
        assert retrieved_bool is True

    def test_attr_operations_with_lists(self, graph):
        """测试列表属性操作"""
        # 测试整数列表
        graph.set_attr("int_list_attr", [1, 2, 3, 4])
        retrieved_int_list = graph.get_attr("int_list_attr")
        assert retrieved_int_list == [1, 2, 3, 4]

        # 测试浮点数列表
        graph.set_attr("float_list_attr", [1.0, 2.0, 3.0])
        retrieved_float_list = graph.get_attr("float_list_attr")
        assert len(retrieved_float_list) == 3

        # 测试字符串列表
        graph.set_attr("str_list_attr", ["hello", "world"])
        retrieved_str_list = graph.get_attr("str_list_attr")
        assert retrieved_str_list == ["hello", "world"]

    def test_multiple_attr_operations(self, graph):
        """测试多个属性操作"""
        attrs = {}
        
        # 设置多个属性
        for i in range(10):
            attr_name = f"attr_{i}"
            graph.set_attr(attr_name, f"value_{i}")
            attrs[attr_name] = f"value_{i}"
        
        # 验证所有属性
        for attr_name, expected_value in attrs.items():
            retrieved_attr = graph.get_attr(attr_name)
            assert retrieved_attr == expected_value

    def test_transfer_ownership(self):
        """测试所有权转移"""
        from ge.es.graph_builder import GraphBuilder
        builder = GraphBuilder()
        graph = builder.build_and_reset()
        assert graph.name == "graph"
        assert graph._owner is None
        assert graph._owns_handle is True
        assert graph._handle is not None
        # 测试所有权转移, 当graph作为子图参数传递给builder时, 所有权转移给builder, 
        # 因转移是在es gen的If, While, Case等操作中C接口自动完成的, 所以这里仅仅是覆盖率测试
        graph._transfer_ownership_when_pass_as_subgraph(builder)
        assert graph._owner is not None
        assert graph._owner == builder
        assert graph._owns_handle is False
        assert graph._handle is not None
        assert graph.name == "graph"
        with pytest.raises(RuntimeError, match="Graph :graph already has an new owner builder :graph, cannot transfer ownership again"):
            graph._transfer_ownership_when_pass_as_subgraph(builder)
        # 还原为了避免内存泄漏
        graph._owns_handle = True

    def test_get_direct_nodes(self, graph):
        nodes = graph.get_direct_nodes()
        assert isinstance(nodes, list)
        # 新创建的图没有节点
        assert len(nodes) == 0

    def test_remove_node(self, graph_with_data):
        """测试移除节点"""
        
        # 获取所有节点
        nodes = graph_with_data.get_all_nodes()
        node_count_before = len(nodes)
        print(f"node_count_before = '{node_count_before}'")
        assert node_count_before > 0 
        
        # 选取单个节点
        node = nodes[-1]
        graph_with_data.remove_node(node)
        node_count_after = len(graph_with_data.get_all_nodes())
        assert node_count_before - node_count_after == 1

    def test_add_data_edge_invalid(self, graph_with_data):
        """测试添加数据边"""
        with pytest.raises(RuntimeError, match="Failed to add DataEdge from Node test_input, Port Index 0 to Node Const_0, Port Index 0"):
            nodes = graph_with_data.get_all_nodes()
            in_node = nodes[0]
            out_node = nodes[1]
            # 添加数据边
            graph_with_data.add_data_edge(in_node, 0 , out_node, 0)

    def test_add_control_edge(self, graph_with_data):
        """测试添加控制边"""
        nodes = graph_with_data.get_all_nodes()
        for node in nodes:
            print(f"Found node: name='{node.name}', type='{node.type}'")

        in_node = nodes[0]
        out_node = nodes[-1]
        # 添加控制边
        graph_with_data.add_control_edge(in_node, out_node)

        # 移除控制边
        graph_with_data.remove_edge(in_node, -1, out_node, -1)
        assert True
        
    def test_find_node_by_name(self, graph_with_data):
        """测试通过名字获取Node"""
        # 获取单个节点
        node = graph_with_data.get_all_nodes()[-1]
        # 根据节点名字获取节点
        node_found = graph_with_data.find_node_by_name(node.name)
        assert node_found is not None

    def test_find_node_by_name_invalid(self, graph_with_data):
        """测试找不到Node名字"""
        # 根据节点名字获取节点        
        with pytest.raises(RuntimeError, match="Failed to find Node name invalid"):
            graph_with_data.find_node_by_name("invalid")
    
    def test_find_node_by_name_invalid_name(self, graph_with_data):
        """测试name必须为string"""
        # 根据节点名字获取节点        
        with pytest.raises(TypeError, match="name must be a string"):
            graph_with_data.find_node_by_name(122)

    def test_save_to_air_invalid_path(self, graph_with_data):
        """测试path必须为string"""
        with pytest.raises(TypeError, match="file_path must be a string"):
            graph_with_data.save_to_air(233)

    def test_save_to_air_error(self, graph_with_data):
        """测试存成AIR时错误情况"""
        handle = graph_with_data._handle
        try:
            with pytest.raises(RuntimeError, match="Failed to save graph to AIR format"):
                graph_with_data._handle = ctypes.c_void_p(None)
                graph_with_data.save_to_air("")
        finally:
            graph_with_data._handle = handle

    def test_load_from_air_invalid_path(self, graph_with_data):
        """测试path必须为string"""
        with pytest.raises(TypeError, match="file_path must be a string"):
            graph_with_data.load_from_air(233)

    def test_load_from_air_error(self, graph_with_data):
        """测试加载AIR时错误情况"""
        handle = graph_with_data._handle
        try:
            with pytest.raises(RuntimeError, match="Failed to load graph from AIR format"):
                graph_with_data._handle = ctypes.c_void_p(None)
                graph_with_data.load_from_air("")
        finally:
            graph_with_data._handle = handle

    def test_remove_edge_invalid_src_node(self, graph_with_data):
        """测试src_node必须为Node"""
        with pytest.raises(TypeError, match="src_node must be a Node"):
            node = graph_with_data.get_all_nodes()[-1]
            graph_with_data.remove_edge(1, 2, node, 1)

    def test_remove_edge_invalid_src_port_index(self, graph_with_data):
        """测试src_port_index必须为string"""
        with pytest.raises(TypeError, match="src_port_index must be an integer"):
            node = graph_with_data.get_all_nodes()[-1]
            graph_with_data.remove_edge(node, "string", node, 1)
            
    def test_remove_edge_invalid_dst_node(self, graph_with_data):
        """测试dst_node必须为Node"""
        with pytest.raises(TypeError, match="dst_node must be a Node"):
            node = graph_with_data.get_all_nodes()[-1]
            graph_with_data.remove_edge(node, 1, 1, 1)

    def test_remove_edge_invalid_dst_port_index(self, graph_with_data):
        """测试dst_port_index必须为string"""
        with pytest.raises(TypeError, match="dst_port_index must be an integer"):
            node = graph_with_data.get_all_nodes()[-1]
            graph_with_data.remove_edge(node, 1, node, "string")
    

    def test_add_data_edge_invalid_src_node(self, graph_with_data):
        with pytest.raises(TypeError, match="src_node must be a Node"):
            node = graph_with_data.get_all_nodes()[-1]
            graph_with_data.add_data_edge(1, 2, node, 1)

    def test_add_data_edge_invalid_src_port_index(self, graph_with_data):
        with pytest.raises(TypeError, match="src_port_index must be an integer"):
            node = graph_with_data.get_all_nodes()[-1]
            graph_with_data.add_data_edge(node, "string", node, 1)
            
    def test_add_data_edge_invalid_dst_node(self, graph_with_data):
        with pytest.raises(TypeError, match="dst_node must be a Node"):
            node = graph_with_data.get_all_nodes()[-1]
            graph_with_data.add_data_edge(node, 1, 1, 1)

    def test_add_data_edge_invalid_dst_port_index(self, graph_with_data):
        with pytest.raises(TypeError, match="dst_port_index must be an integer"):
            node = graph_with_data.get_all_nodes()[-1]
            graph_with_data.add_data_edge(node, 1, node, "string")
        

    def test_add_control_edge_invalid_src_node(self, graph_with_data):
        with pytest.raises(TypeError, match="src_node must be a Node"):
            node = graph_with_data.get_all_nodes()[-1]
            graph_with_data.add_control_edge(1, node)

    def test_add_data_edge_invalid_dst_node(self, graph_with_data):
        with pytest.raises(TypeError, match="dst_node must be a Node"):
            node = graph_with_data.get_all_nodes()[-1]
            graph_with_data.add_control_edge(node, 1)

    def test_get_all_subgraphs_empty(self, graph):
        """测试获取空子图列表"""
        subgraphs = graph.get_all_subgraphs()
        assert len(subgraphs) == 0
        assert isinstance(subgraphs, list)

    def test_get_and_add_subgraph(self, graph, graph_with_subgraph):
        """测试获取并添加子图"""
        # 获取并添加第一个子图
        then_graph = graph_with_subgraph.get_subgraph("then_graph")
        assert then_graph is not None
        assert then_graph.name == "then_graph"

        graph.add_subgraph(then_graph)
        subgraphs = graph.get_all_subgraphs()
        assert len(subgraphs) == 1
        assert subgraphs[0].name == "then_graph"

        # 获取并添加第二个子图
        else_graph = graph_with_subgraph.get_subgraph("else_graph")
        assert else_graph is not None
        assert else_graph.name == "else_graph"

        graph.add_subgraph(else_graph)
        subgraphs = graph.get_all_subgraphs()
        assert len(subgraphs) == 2
        subgraph_names = [sg.name for sg in subgraphs]
        assert "then_graph" in subgraph_names
        assert "else_graph" in subgraph_names

        # 获取不存在的子图
        not_found = graph.get_subgraph("nonexistent")
        assert not_found is None

    def test_add_duplicate_subgraph_name_error(self, graph, graph_with_subgraph):
        """测试添加同名子图"""
        # 添加第一个子图
        then_graph = graph_with_subgraph.get_subgraph("then_graph")
        graph.add_subgraph(then_graph)
        subgraphs = graph.get_all_subgraphs()
        assert len(subgraphs) == 1
        assert subgraphs[0].name == "then_graph"

        # 尝试添加同名子图，应该失败
        with pytest.raises(RuntimeError, match="Failed to add subgraph 'then_graph' to graph 'test_graph'"):
            graph.add_subgraph(then_graph)

        # 验证只有一个子图
        subgraphs = graph.get_all_subgraphs()
        assert len(subgraphs) == 1
        assert subgraphs[0].name == "then_graph"

    def test_add_subgraph_type_error(self, graph):
        """测试添加子图时类型错误"""
        with pytest.raises(TypeError, match="subgraph must be a Graph"):
            graph.add_subgraph("not_a_graph")

    def test_get_subgraph_type_error(self, graph):
        """测试获取子图时类型错误"""
        with pytest.raises(TypeError, match="name must be a string"):
            graph.get_subgraph(123)

    def test_remove_subgraph(self, graph, graph_with_subgraph):
        """测试移除子图"""
        # 添加子图
        then_graph = graph_with_subgraph.get_subgraph("then_graph")
        graph.add_subgraph(then_graph)
        else_graph = graph_with_subgraph.get_subgraph("else_graph")
        graph.add_subgraph(else_graph)

        # 验证子图存在
        subgraphs = graph.get_all_subgraphs()
        assert len(subgraphs) == 2

        # 移除一个子图
        graph.remove_subgraph("then_graph")
        subgraphs = graph.get_all_subgraphs()
        assert len(subgraphs) == 1
        assert subgraphs[0].name == "else_graph"

        # 移除另一个子图
        graph.remove_subgraph("else_graph")
        subgraphs = graph.get_all_subgraphs()
        assert len(subgraphs) == 0

    def test_remove_subgraph_type_error(self, graph):
        """测试移除子图时类型错误"""
        with pytest.raises(TypeError, match="name must be a string"):
            graph.remove_subgraph(123)
