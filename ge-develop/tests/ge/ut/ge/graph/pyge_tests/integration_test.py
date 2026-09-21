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
集成测试 - 使用 pytest 框架
测试 ge 模块各个组件之间的集成功能
"""

import pytest
import sys
import os

# 需要添加 ge 到 Python 路径，否则会报错
try:
    from ge.es.graph_builder import GraphBuilder, TensorHolder
    from ge.graph import Graph, Node, DumpFormat
    from ge.graph.types import DataType, Format
except ImportError as e:
    pytest.skip(f"无法导入 ge 模块: {e}", allow_module_level=True)


class TestIntegration:
    """集成测试类"""

    @pytest.fixture
    def builder(self):
        """创建 GraphBuilder 实例的 fixture"""
        return GraphBuilder("integration_test_graph")

    def test_graph_builder_to_graph_integration(self, builder):
        """测试 GraphBuilder 到 Graph 的基本集成"""
        # 创建输入
        input_tensor = builder.create_input(
            index=0,
            name="input",
            data_type=DataType.DT_FLOAT,
            shape=[1, 3, 224, 224]
        )

        # 创建常量
        const_tensor = builder.create_const_float(1.0)

        # 设置输出
        builder.set_graph_output(input_tensor, 0)

        # 构建图
        graph = builder.build_and_reset()

        # 验证图
        assert isinstance(graph, Graph)
        assert graph.name == "integration_test_graph"

    def test_graph_nodes_validation(self, builder):
        """测试图中节点的验证"""
        # 创建输入
        input_tensor = builder.create_input(
            index=0,
            name="input",
            data_type=DataType.DT_FLOAT,
            shape=[1, 3, 224, 224]
        )

        # 创建常量
        const_tensor = builder.create_const_float(1.0)

        # 设置输出
        builder.set_graph_output(input_tensor, 0)

        # 构建图
        graph = builder.build_and_reset()

        # 获取图中的所有节点
        nodes = graph.get_all_nodes()
        assert isinstance(nodes, list)
        assert len(nodes) > 0  # 应该至少有一个节点

        # 验证每个节点的基本信息
        for node in nodes:
            assert isinstance(node, Node)
            assert node.name is not None  # 节点名称不应为空
            assert node.type is not None  # 节点类型不应为空
            assert isinstance(node.name, str)  # 节点名称应该是字符串
            assert isinstance(node.type, str)  # 节点类型应该是字符串

        # 验证特定节点（如果存在）
        node_names = [node.name for node in nodes]
        node_types = [node.type for node in nodes]

        # 检查是否包含预期的节点名称
        if "input" in node_names:
            input_node = next(node for node in nodes if node.name == "input")
            assert input_node.type is not None
            print(f"Found input node: name='{input_node.name}', type='{input_node.type}'")

        # 打印所有节点信息用于调试
        print(f"Graph contains {len(nodes)} nodes:")
        for i, node in enumerate(nodes):
            print(f"  Node {i}: name='{node.name}', type='{node.type}'")

    def test_graph_dump_to_onnx(self, builder):
        """测试图dump到ONNX功能"""
        # 创建输入
        input_tensor = builder.create_input(
            index=0,
            name="input",
            data_type=DataType.DT_FLOAT,
            shape=[1, 3, 224, 224]
        )

        # 创建常量
        const_tensor = builder.create_const_float(1.0)

        # 设置输出
        builder.set_graph_output(input_tensor, 0)

        # 构建图
        graph = builder.build_and_reset()

        # 测试 dump 到 ONNX 功能
        import tempfile
        import os
        import re

        # 创建临时目录进行测试
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = os.getcwd()
            os.chdir(temp_dir)
            try:
                test_suffix = "test_suffix"
                graph.dump_to_file(format=DumpFormat.kOnnx, suffix=test_suffix)

                # 在指定目录中查找文件
                temp_dir_files = [f for f in os.listdir(temp_dir) if f.startswith('ge_onnx_') and f.endswith('.pbtxt')]
                assert len(
                    temp_dir_files) > 0, f"Expected ge_onnx_*.pbtxt file not found in {temp_dir}. Found: {os.listdir(temp_dir)}"

                # 验证文件名格式（包含suffix）
                expected_pattern_with_suffix = r'ge_onnx_\d+_graph_\d+_' + re.escape(test_suffix) + r'\.pbtxt$'
                found_valid_file_with_suffix = False
                for file in temp_dir_files:
                    if re.match(expected_pattern_with_suffix, file):
                        found_valid_file_with_suffix = True
                        expected_file = os.path.join(temp_dir, file)
                        print(f"Found valid ONNX file with suffix: {file}")
                        break

                assert found_valid_file_with_suffix, f"No file matching pattern {expected_pattern_with_suffix} found. Files: {temp_dir_files}"

                # 验证文件内容不为空
                with open(expected_file, 'r') as f:
                    content = f.read()
                    assert len(content) > 0, "Dumped ONNX file should not be empty"
                    assert "node" in content.lower() or "graph" in content.lower(), "Dumped file should contain graph structure"

                print(f"Successfully dumped graph to ONNX format: {expected_file}")
                print(f"File size: {len(content)} bytes")
            finally:
                os.chdir(original_cwd)

    def test_attr_value_integration_with_graph(self, builder):
        """测试 AttrValue 与 Graph 的集成"""
        # 创建图
        input_tensor = builder.create_input(0)
        builder.set_graph_output(input_tensor, 0)
        graph = builder.build_and_reset()

        # 设置图属性
        graph.set_attr("str_attr", "test_string")
        graph.set_attr("int_attr", 42)
        graph.set_attr("float_attr", 3.14)
        graph.set_attr("bool_attr", True)

        # 获取并验证属性
        retrieved_str = graph.get_attr("str_attr")
        assert retrieved_str == "test_string"

        retrieved_int = graph.get_attr("int_attr")
        assert retrieved_int == 42

        retrieved_float = graph.get_attr("float_attr")
        assert abs(retrieved_float - 3.14) < 1e-6

        retrieved_bool = graph.get_attr("bool_attr")
        assert retrieved_bool is True

    def test_attr_value_list_integration_with_graph(self, builder):
        """测试 AttrValue 列表与 Graph 的集成"""
        # 创建图
        input_tensor = builder.create_input(0)
        builder.set_graph_output(input_tensor, 0)
        graph = builder.build_and_reset()

        # 设置图属性
        graph.set_attr("int_list_attr", [1, 2, 3, 4, 5])
        graph.set_attr("float_list_attr", [1.0, 2.0, 3.0, 4.0, 5.0])
        graph.set_attr("bool_list_attr", [True, False, True, False])
        graph.set_attr("str_list_attr", ["hello", "world", "test"])

        # 获取并验证属性
        retrieved_int_list = graph.get_attr("int_list_attr")
        assert retrieved_int_list == [1, 2, 3, 4, 5]

        retrieved_float_list = graph.get_attr("float_list_attr")
        assert len(retrieved_float_list) == 5

        retrieved_bool_list = graph.get_attr("bool_list_attr")
        assert retrieved_bool_list == [True, False, True, False]

        retrieved_str_list = graph.get_attr("str_list_attr")
        assert retrieved_str_list == ["hello", "world", "test"]

    def test_data_type_integration(self, builder):
        """测试 DataType 集成"""
        # 测试不同数据类型创建输入
        float_input = builder.create_input(0, data_type=DataType.DT_FLOAT, shape=[1, 3, 224, 224])
        int32_input = builder.create_input(1, data_type=DataType.DT_INT32, shape=[1, 3, 224, 224])
        int64_input = builder.create_input(2, data_type=DataType.DT_INT64, shape=[1, 3, 224, 224])
        bool_input = builder.create_input(3, data_type=DataType.DT_BOOL, shape=[1, 3, 224, 224])
        string_input = builder.create_input(4, data_type=DataType.DT_STRING, shape=[1, 3, 224, 224])

        # 验证输入创建成功
        assert isinstance(float_input, TensorHolder)
        assert isinstance(int32_input, TensorHolder)
        assert isinstance(int64_input, TensorHolder)
        assert isinstance(bool_input, TensorHolder)
        assert isinstance(string_input, TensorHolder)

        # 测试设置数据类型
        float_input.set_data_type(DataType.DT_INT32)
        int32_input.set_data_type(DataType.DT_FLOAT)
        int64_input.set_data_type(DataType.DT_BOOL)

    def test_complex_graph_construction(self, builder):
        """测试复杂图构建"""
        # 创建多个输入
        input1 = builder.create_input(0, name="input1", data_type=DataType.DT_FLOAT, shape=[1, 3, 224, 224])
        input2 = builder.create_input(1, name="input2", data_type=DataType.DT_FLOAT, shape=[1, 3, 224, 224])
        input3 = builder.create_input(2, name="input3", data_type=DataType.DT_FLOAT, shape=[1, 3, 224, 224])

        # 创建常量
        const1 = builder.create_const_float(1.0)
        const2 = builder.create_const_float(2.0)
        const3 = builder.create_const_float(3.0)

        # 设置输出
        builder.set_graph_output(input1, 0)
        builder.set_graph_output(input2, 1)
        builder.set_graph_output(input3, 2)

        # 构建图
        graph = builder.build_and_reset()

        # 验证图
        assert isinstance(graph, Graph)
        assert graph.name == "integration_test_graph"

        # 设置图属性
        graph.set_attr("graph_name", "complex_graph")

        # 验证属性
        retrieved_attr = graph.get_attr("graph_name")
        assert retrieved_attr == "complex_graph"

    def test_tensor_properties_integration(self, builder):
        """测试 Tensor 属性集成"""
        # 创建输入tensor
        input_tensor = builder.create_input(
            index=0,
            name="test_input",
            data_type=DataType.DT_FLOAT,
            shape=[1, 3, 224, 224]
        )

        # 测试设置方法
        input_tensor.set_data_type(DataType.DT_INT32)
        input_tensor.set_format(Format.FORMAT_NCHW)
        input_tensor.set_shape([2, 3, 112, 112])

    def test_error_handling_integration(self, builder):
        """测试错误处理集成"""
        # 测试无效参数  
        with pytest.raises(TypeError):
            builder.create_input("invalid")

        with pytest.raises(TypeError):
            builder.create_const_float("invalid")

        with pytest.raises(TypeError):
            builder.set_graph_output("invalid", 0)

        # 测试无效属性操作
        graph = Graph("test")
        with pytest.raises(TypeError):
            graph.set_attr(123, "test")

        with pytest.raises(TypeError):
            graph.get_attr(123)

    def test_memory_management_integration(self, builder):
        """测试内存管理集成"""
        # 创建多个对象
        input_tensor = builder.create_input(0)
        const_tensor = builder.create_const_float(1.0)

        # 设置输出
        builder.set_graph_output(input_tensor, 0)

        # 构建图
        graph = builder.build_and_reset()

        # 验证对象存在
        assert input_tensor._handle is not None
        assert const_tensor._handle is not None
        assert graph._handle is not None

    def test_dump_to_file_error_handling(self, builder):
        """测试 dump_to_file 错误处理"""
        # 创建简单的图
        input_tensor = builder.create_input(0)
        builder.set_graph_output(input_tensor, 0)
        graph = builder.build_and_reset()

        # 确保图有节点才进行dump测试
        nodes = graph.get_all_nodes()
        assert len(nodes) != 0

        # 测试无效参数类型
        with pytest.raises(TypeError):
            graph.dump_to_file(format=123, suffix="test_suffix")  # format 应该是 DumpFormat 枚举类型
        with pytest.raises(TypeError):
            graph.dump_to_file(format=DumpFormat.kOnnx, suffix=123)  # suffix 应该是字符串

        original_handle = graph._handle
        graph._handle = None
        with pytest.raises(RuntimeError):
            graph.dump_to_file(format=DumpFormat.kOnnx, suffix="test_suffix")
        graph._handle = original_handle

    def test_dump_to_stream_error_handling(self, builder):
        input_tensor = builder.create_input(0)
        builder.set_graph_output(input_tensor, 0)
        graph = builder.build_and_reset()

        # 1) 非法参数类型：format 不是 DumpFormat
        with pytest.raises(TypeError):
            graph.dump_to_stream(format=123)  # 非法类型
        with pytest.raises(TypeError):
            graph.dump_to_stream(format="kReadable")  # 非法类型

        original_handle = graph._handle
        graph._handle = None
        with pytest.raises(RuntimeError):
            graph.dump_to_stream(format=DumpFormat.kReadable)
        graph._handle = original_handle

        # 2) 合法参数：可读串应非空，且 print(graph) 不抛异常
        readable = graph.dump_to_stream(format=DumpFormat.kReadable)
        assert isinstance(readable, str)
        assert len(readable) > 0

        # 确认 __str__ 使用可读 dump，不抛异常
        s = str(graph)
        assert isinstance(s, str)
        assert len(s) > 0
        assert ("graph" in s.lower())

    def test_save_to_air(self, builder):
        # 创建简单的图
        input_tensor = builder.create_input(0)
        builder.set_graph_output(input_tensor, 0)
        graph = builder.build_and_reset()
        

        # 测试 dump 到 AIR 功能
        import tempfile
        import os
        import re
          # 创建临时目录进行测试
        with tempfile.TemporaryDirectory() as temp_dir:
            # 测试指定路径和后缀
            filename = f"{temp_dir}/ut_graph1.txt"
            graph.save_to_air(filename)

            # 在指定目录中查找文件
            temp_dir_files = [f for f in os.listdir(temp_dir) if f.endswith("ut_graph1.txt")]
            assert len(
                temp_dir_files) > 0, f"Expected file not found in {temp_dir}. Found: {os.listdir(temp_dir)}"
            # 验证文件内容不为空
            with open(filename, 'rb') as f:
                content = f.read()
                assert len(content) > 0, "Dumped AIR file should not be empty"
                
    def test_load_from_air(self, builder):
        # 创建简单的图
        input_tensor = builder.create_input(0)
        builder.set_graph_output(input_tensor, 0)
        graph = builder.build_and_reset()
        

        # 测试 dump 到 AIR 功能
        import tempfile
        import os
        import re
          # 创建临时目录进行测试
        with tempfile.TemporaryDirectory() as temp_dir:
            # 测试指定路径和后缀
            filename = f"{temp_dir}/ut_graph2.txt"
            graph.save_to_air(filename)

            # 在指定目录中查找文件
            temp_dir_files = [f for f in os.listdir(temp_dir) if f.endswith("ut_graph2.txt")]
            assert len(
                temp_dir_files) > 0, f"Expected file not found in {temp_dir}. Found: {os.listdir(temp_dir)}"
            with open(filename, 'rb') as f:
                content = f.read()
                assert len(content) > 0, "Dumped AIR file should not be empty"
            tmp_graph = Graph("tmp_graph_name")
            tmp_graph.load_from_air(filename)
            assert tmp_graph.name == "integration_test_graph"