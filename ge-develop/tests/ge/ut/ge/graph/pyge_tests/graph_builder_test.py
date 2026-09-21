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
GraphBuilder 功能测试 - 使用 pytest 框架
测试 graph_builder.py 中的 GraphBuilder 类的各种功能
"""

import pytest
import sys
import os

# 需要添加 ge 到 Python 路径，否则会报错
try:
    from ge.es.graph_builder import GraphBuilder, TensorHolder
    from ge.graph.types import DataType, Format
    from ge.graph import Graph
except ImportError as e:
    pytest.skip(f"无法导入 ge 模块: {e}", allow_module_level=True)


class TestGraphBuilder:
    """GraphBuilder 功能测试类"""

    @pytest.fixture
    def builder(self):
        """创建 GraphBuilder 实例的 fixture"""
        return GraphBuilder("test_graph")

    def test_graph_builder_creation(self):
        """测试 GraphBuilder 创建"""
        builder = GraphBuilder("my_graph")
        assert builder is not None
        assert hasattr(builder, '_handle')

    def test_graph_builder_creation_with_default_name(self):
        """测试使用默认名称创建 GraphBuilder"""
        builder = GraphBuilder()
        assert builder is not None
        assert builder.name == "graph"

    def test_graph_builder_invalid_name(self):
        """测试无效名称创建 GraphBuilder"""
        with pytest.raises(TypeError, match="Graph name must be a string"):
            GraphBuilder(123)

    def test_create_input_simple(self, builder):
        """测试创建简单输入"""
        input_tensor = builder.create_input(0)
        assert isinstance(input_tensor, TensorHolder)
        assert input_tensor._handle is not None

    def test_create_inputs_simple(self, builder):
        """测试创建多个简单输入"""
        input_tensors = builder.create_inputs(3)
        assert isinstance(input_tensors, list)
        assert len(input_tensors) == 3
        for input_tensor in input_tensors:
            assert isinstance(input_tensor, TensorHolder)
            assert input_tensor._handle is not None

    def test_create_inputs_invalid_args(self, builder):
        """测试创建多个简单输入无效参数"""
        with pytest.raises(TypeError, match="Number of inputs must be a positive integer"):
            builder.create_inputs(0)
        with pytest.raises(TypeError, match="Start index must be a non-negative integer"):
            builder.create_inputs(1, start_index=-1)

    def test_create_input_with_details(self, builder):
        """测试创建详细输入"""
        input_tensor = builder.create_input(
            index=0,
            name="input_tensor",
            data_type=DataType.DT_FLOAT,
            format=Format.FORMAT_NCHW,
            shape=[1, 3, 224, 224]
        )
        assert isinstance(input_tensor, TensorHolder)
        assert input_tensor._handle is not None

    def test_create_input_invalid_index(self, builder):
        """测试无效索引创建输入"""
        with pytest.raises(TypeError, match="Input index must be an integer"):
            builder.create_input("invalid")

    def test_create_input_invalid_shape(self, builder):
        """测试无效形状创建输入"""
        with pytest.raises(TypeError, match="Shape must be a list of integers"):
            builder.create_input(0, shape="invalid")

    def test_create_const_int64_scalar(self, builder):
        """测试创建 int64 标量常量"""
        const_tensor = builder.create_const_int64(42)
        assert isinstance(const_tensor, TensorHolder)
        assert const_tensor._handle is not None

    def test_create_const_int64_list(self, builder):
        """测试创建 int64 列表常量"""
        const_tensor = builder.create_const_int64([1, 2, 3, 4], shape=[2, 2])
        assert isinstance(const_tensor, TensorHolder)
        assert const_tensor._handle is not None

    def test_create_const_int64_invalid_value(self, builder):
        """测试无效值创建 int64 常量"""
        with pytest.raises(TypeError, match="Value must be an integer or list of integers"):
            builder.create_const_int64("invalid")

    def test_create_const_int64_shape_mismatch(self, builder):
        """测试 value 数量与 shape 不匹配的情况"""
        # value 有 3 个元素，但 shape [2,2] 需要 4 个元素
        with pytest.raises(ValueError, match="Value count \\(3\\) doesn't match shape"):
            builder.create_const_int64([1, 2, 3], shape=[2, 2])
        
        # value 有 5 个元素，但 shape [2,2] 需要 4 个元素
        with pytest.raises(ValueError, match="Value count \\(5\\) doesn't match shape"):
            builder.create_const_int64([1, 2, 3, 4, 5], shape=[2, 2])
        
        # value 有 4 个元素，但 shape [3,3] 需要 9 个元素
        with pytest.raises(ValueError, match="Value count \\(4\\) doesn't match shape"):
            builder.create_const_int64([1, 2, 3, 4], shape=[3, 3])

    def test_create_const_float_scalar(self, builder):
        """测试创建 float 标量常量"""
        const_tensor = builder.create_const_float(3.14)
        assert isinstance(const_tensor, TensorHolder)
        assert const_tensor._handle is not None

    def test_create_const_float_list(self, builder):
        """测试创建 float 列表常量"""
        const_tensor = builder.create_const_float([1.0, 2.0, 3.0], shape=[3])
        assert isinstance(const_tensor, TensorHolder)
        assert const_tensor._handle is not None

    def test_create_const_float_invalid_value(self, builder):
        """测试无效值创建 float 常量"""
        with pytest.raises(TypeError, match="Value must be a float or list of floats"):
            builder.create_const_float("invalid")

    def test_create_const_uint64_scalar(self, builder):
        """测试创建 uint64 标量常量"""
        const_tensor = builder.create_const_uint64(42)
        assert isinstance(const_tensor, TensorHolder)
        assert const_tensor._handle is not None

    def test_create_const_uint64_list(self, builder):
        """测试创建 uint64 列表常量"""
        const_tensor = builder.create_const_uint64([1, 2, 3, 4], shape=[2, 2])
        assert isinstance(const_tensor, TensorHolder)
        assert const_tensor._handle is not None

    def test_create_const_int32_scalar(self, builder):
        """测试创建 int32 标量常量"""
        const_tensor = builder.create_const_int32(42)
        assert isinstance(const_tensor, TensorHolder)
        assert const_tensor._handle is not None

    def test_create_const_uint32_scalar(self, builder):
        """测试创建 uint32 标量常量"""
        const_tensor = builder.create_const_uint32(42)
        assert isinstance(const_tensor, TensorHolder)
        assert const_tensor._handle is not None

    def test_create_vector_int64(self, builder):
        """测试创建 int64 向量"""
        vector_tensor = builder.create_vector_int64([1, 2, 3, 4])
        assert isinstance(vector_tensor, TensorHolder)
        assert vector_tensor._handle is not None

    def test_create_vector_int64_invalid_value(self, builder):
        """测试无效值创建 int64 向量"""
        with pytest.raises(TypeError, match="Value must be a list of integers"):
            builder.create_vector_int64("invalid")

    def test_create_scalar_int64(self, builder):
        """测试创建 int64 标量"""
        scalar_tensor = builder.create_scalar_int64(42)
        assert isinstance(scalar_tensor, TensorHolder)
        assert scalar_tensor._handle is not None

    def test_create_scalar_int32(self, builder):
        """测试创建 int32 标量"""
        scalar_tensor = builder.create_scalar_int32(42)
        assert isinstance(scalar_tensor, TensorHolder)
        assert scalar_tensor._handle is not None

    def test_create_scalar_float(self, builder):
        """测试创建 float 标量"""
        scalar_tensor = builder.create_scalar_float(3.14)
        assert isinstance(scalar_tensor, TensorHolder)
        assert scalar_tensor._handle is not None

    def test_create_scalar_uint64(self, builder):
        """测试创建 uint64 标量"""
        scalar_tensor = builder.create_scalar_uint64(42)
        assert isinstance(scalar_tensor, TensorHolder)
        assert scalar_tensor._handle is not None

    def test_create_scalar_uint64_zero(self, builder):
        """测试创建 uint64 标量值为 0"""
        scalar_tensor = builder.create_scalar_uint64(0)
        assert isinstance(scalar_tensor, TensorHolder)
        assert scalar_tensor._handle is not None

    def test_create_scalar_uint64_large_value(self, builder):
        """测试创建 uint64 标量大值"""
        scalar_tensor = builder.create_scalar_uint64(2**63 - 1)
        assert isinstance(scalar_tensor, TensorHolder)
        assert scalar_tensor._handle is not None

    def test_create_scalar_uint64_invalid_value(self, builder):
        """测试无效值创建 uint64 标量"""
        with pytest.raises(TypeError, match="Value must be an integer"):
            builder.create_scalar_uint64("invalid")
        
        with pytest.raises(ValueError, match="Value must be non-negative"):
            builder.create_scalar_uint64(-1)

    def test_create_scalar_uint32(self, builder):
        """测试创建 uint32 标量"""
        scalar_tensor = builder.create_scalar_uint32(42)
        assert isinstance(scalar_tensor, TensorHolder)
        assert scalar_tensor._handle is not None

    def test_create_scalar_uint32_zero(self, builder):
        """测试创建 uint32 标量值为 0"""
        scalar_tensor = builder.create_scalar_uint32(0)
        assert isinstance(scalar_tensor, TensorHolder)
        assert scalar_tensor._handle is not None

    def test_create_scalar_uint32_max_value(self, builder):
        """测试创建 uint32 标量最大值"""
        scalar_tensor = builder.create_scalar_uint32(0xFFFFFFFF)
        assert isinstance(scalar_tensor, TensorHolder)
        assert scalar_tensor._handle is not None

    def test_create_scalar_uint32_invalid_value(self, builder):
        """测试无效值创建 uint32 标量"""
        with pytest.raises(TypeError, match="Value must be an integer"):
            builder.create_scalar_uint32("invalid")
        
        with pytest.raises(ValueError, match="Value must be in range"):
            builder.create_scalar_uint32(-1)
        
        with pytest.raises(ValueError, match="Value must be in range"):
            builder.create_scalar_uint32(0xFFFFFFFF + 1)

    def test_create_const_uint64_scalar_equivalence(self, builder):
        """测试 create_const_uint64(0) 等价于 create_scalar_uint64(0)"""
        # 当 dims 为空且只有一个值时，应该调用 scalar 方法
        const_tensor = builder.create_const_uint64(0)
        scalar_tensor = builder.create_scalar_uint64(0)
        
        assert isinstance(const_tensor, TensorHolder)
        assert isinstance(scalar_tensor, TensorHolder)
        assert const_tensor._handle is not None
        assert scalar_tensor._handle is not None

    def test_create_const_uint64_with_empty_shape(self, builder):
        """测试 create_const_uint64 带空 shape 的情况"""
        # 显式指定空 shape，应该调用 scalar 方法
        const_tensor = builder.create_const_uint64(42, shape=[])
        assert isinstance(const_tensor, TensorHolder)
        assert const_tensor._handle is not None

    def test_create_const_uint64_non_scalar(self, builder):
        """测试 create_const_uint64 非标量情况（列表）"""
        # 多个值或非空 shape，应该使用常量创建方法
        const_tensor = builder.create_const_uint64([1, 2, 3, 4], shape=[2, 2])
        assert isinstance(const_tensor, TensorHolder)
        assert const_tensor._handle is not None

    def test_create_const_uint64_list_without_shape(self, builder):
        """测试 create_const_uint64 列表不带 shape（自动推断）"""
        const_tensor = builder.create_const_uint64([1, 2, 3])
        assert isinstance(const_tensor, TensorHolder)
        assert const_tensor._handle is not None

    def test_create_const_uint32_scalar_equivalence(self, builder):
        """测试 create_const_uint32(0) 等价于 create_scalar_uint32(0)"""
        # 当 dims 为空且只有一个值时，应该调用 scalar 方法
        const_tensor = builder.create_const_uint32(0)
        scalar_tensor = builder.create_scalar_uint32(0)
        
        assert isinstance(const_tensor, TensorHolder)
        assert isinstance(scalar_tensor, TensorHolder)
        assert const_tensor._handle is not None
        assert scalar_tensor._handle is not None

    def test_create_const_uint32_with_empty_shape(self, builder):
        """测试 create_const_uint32 带空 shape 的情况"""
        # 显式指定空 shape，应该调用 scalar 方法
        const_tensor = builder.create_const_uint32(42, shape=[])
        assert isinstance(const_tensor, TensorHolder)
        assert const_tensor._handle is not None

    def test_create_const_uint32_non_scalar(self, builder):
        """测试 create_const_uint32 非标量情况（列表）"""
        # 多个值或非空 shape，应该使用常量创建方法
        const_tensor = builder.create_const_uint32([1, 2, 3, 4], shape=[2, 2])
        assert isinstance(const_tensor, TensorHolder)
        assert const_tensor._handle is not None

    def test_create_const_uint32_list_without_shape(self, builder):
        """测试 create_const_uint32 列表不带 shape（自动推断）"""
        const_tensor = builder.create_const_uint32([1, 2, 3])
        assert isinstance(const_tensor, TensorHolder)
        assert const_tensor._handle is not None

    def test_create_const_int64_scalar_equivalence(self, builder):
        """测试 create_const_int64(0) 等价于 create_scalar_int64(0)"""
        # 当 dims 为空且只有一个值时，应该调用 scalar 方法
        const_tensor = builder.create_const_int64(0)
        scalar_tensor = builder.create_scalar_int64(0)
        
        assert isinstance(const_tensor, TensorHolder)
        assert isinstance(scalar_tensor, TensorHolder)
        assert const_tensor._handle is not None
        assert scalar_tensor._handle is not None

    def test_create_const_int64_with_empty_shape(self, builder):
        """测试 create_const_int64 带空 shape 的情况"""
        # 显式指定空 shape，应该调用 scalar 方法
        const_tensor = builder.create_const_int64(42, shape=[])
        assert isinstance(const_tensor, TensorHolder)
        assert const_tensor._handle is not None

    def test_create_const_int32_scalar_equivalence(self, builder):
        """测试 create_const_int32(0) 等价于 create_scalar_int32(0)"""
        # 当 dims 为空且只有一个值时，应该调用 scalar 方法
        const_tensor = builder.create_const_int32(0)
        scalar_tensor = builder.create_scalar_int32(0)
        
        assert isinstance(const_tensor, TensorHolder)
        assert isinstance(scalar_tensor, TensorHolder)
        assert const_tensor._handle is not None
        assert scalar_tensor._handle is not None

    def test_create_const_int32_with_empty_shape(self, builder):
        """测试 create_const_int32 带空 shape 的情况"""
        # 显式指定空 shape，应该调用 scalar 方法
        const_tensor = builder.create_const_int32(42, shape=[])
        assert isinstance(const_tensor, TensorHolder)
        assert const_tensor._handle is not None

    def test_create_const_float_scalar_equivalence(self, builder):
        """测试 create_const_float(0.0) 等价于 create_scalar_float(0.0)"""
        # 当 dims 为空且只有一个值时，应该调用 scalar 方法
        const_tensor = builder.create_const_float(0.0)
        scalar_tensor = builder.create_scalar_float(0.0)
        
        assert isinstance(const_tensor, TensorHolder)
        assert isinstance(scalar_tensor, TensorHolder)
        assert const_tensor._handle is not None
        assert scalar_tensor._handle is not None

    def test_create_const_float_with_empty_shape(self, builder):
        """测试 create_const_float 带空 shape 的情况"""
        # 显式指定空 shape，应该调用 scalar 方法
        const_tensor = builder.create_const_float(3.14, shape=[])
        assert isinstance(const_tensor, TensorHolder)
        assert const_tensor._handle is not None

    def test_create_variable(self, builder):
        """测试创建变量"""
        var_tensor = builder.create_variable(0, "my_variable")
        assert isinstance(var_tensor, TensorHolder)
        assert var_tensor._handle is not None

    def test_create_variable_invalid_args(self, builder):
        """测试无效参数创建变量"""
        with pytest.raises(TypeError, match="Index must be an integer"):
            builder.create_variable("invalid", "name")

        with pytest.raises(TypeError, match="Name must be a string"):
            builder.create_variable(0, 123)

    def test_set_graph_output(self, builder):
        """测试设置图输出"""
        input_tensor = builder.create_input(0)
        builder.set_graph_output(input_tensor, 0)

    def test_set_graph_output_invalid_tensor(self, builder):
        """测试无效tensor设置图输出"""
        with pytest.raises(TypeError, match="Tensor must be a TensorHolder"):
            builder.set_graph_output("invalid", 0)

    def test_set_graph_output_invalid_index(self, builder):
        """测试无效索引设置图输出"""
        input_tensor = builder.create_input(0)
        with pytest.raises(TypeError, match="Output index must be an integer"):
            builder.set_graph_output(input_tensor, "invalid")

    # ============ 图属性设置测试 ============
    def test_set_graph_attributes(self, builder):
        """测试设置图属性"""
        # 测试设置 int64 属性
        builder.set_graph_attr_int64("int_attr", 42)
        builder.set_graph_attr_string("str_attr", "hello")
        builder.set_graph_attr_bool("bool_attr", True)

        graph = builder.build_and_reset()
        assert graph.get_attr("int_attr") == 42
        assert graph.get_attr("str_attr") == "hello"
        assert graph.get_attr("bool_attr") == True

    def test_set_graph_attributes_invalid_args(self, builder):
        """测试无效参数设置图属性"""
        # 测试 int64 属性无效参数
        with pytest.raises(TypeError, match="Attribute name must be a string"):
            builder.set_graph_attr_int64(123, 42)
        with pytest.raises(TypeError, match="Value must be an integer"):
            builder.set_graph_attr_int64("attr", "invalid")

        # 测试 string 属性无效参数
        with pytest.raises(TypeError, match="Value must be a string"):
            builder.set_graph_attr_string("attr", 123)

        # 测试 bool 属性无效参数
        with pytest.raises(TypeError, match="Value must be a boolean"):
            builder.set_graph_attr_bool("attr", "invalid")

    # ============ Tensor属性设置测试 ============
    def test_set_tensor_attributes(self, builder):
        """测试设置tensor属性"""
        tensor = builder.create_const_float(1.0)
        producer = tensor._get_node_snapshot()

        # 测试设置 int64 属性
        builder.set_tensor_attr_int64(tensor, "int_attr", 42)
        assert producer.get_output_attr("int_attr", 0) == 42

        # 测试设置 string 属性
        builder.set_tensor_attr_string(tensor, "str_attr", "hello")
        assert producer.get_output_attr("str_attr", 0) == "hello"

        # 测试设置 bool 属性
        builder.set_tensor_attr_bool(tensor, "bool_attr", True)
        assert producer.get_output_attr("bool_attr", 0) == True

    def test_set_tensor_attributes_invalid_args(self, builder):
        """测试无效参数设置tensor属性"""
        tensor = builder.create_const_float(1.0)

        # 测试 int64 属性无效参数
        with pytest.raises(TypeError, match="Tensor must be a TensorHolder"):
            builder.set_tensor_attr_int64("invalid", "attr", 42)
        with pytest.raises(TypeError, match="Attribute name must be a string"):
            builder.set_tensor_attr_int64(tensor, 123, 42)
        with pytest.raises(TypeError, match="Value must be an integer"):
            builder.set_tensor_attr_int64(tensor, "attr", "invalid")

        # 测试 string 属性无效参数
        with pytest.raises(TypeError, match="Value must be a string"):
            builder.set_tensor_attr_string(tensor, "attr", 123)

        # 测试 bool 属性无效参数
        with pytest.raises(TypeError, match="Value must be a boolean"):
            builder.set_tensor_attr_bool(tensor, "attr", "invalid")

    # ============ 节点属性设置测试 ============
    def test_set_node_attributes(self, builder):
        """测试设置节点属性"""
        tensor = builder.create_const_float(1.0)
        producer = tensor._get_node_snapshot()

        # 测试设置 int64 属性
        builder.set_node_attr_int64(tensor, "int_attr", 42)
        assert producer.get_attr("int_attr") == 42

        # 测试设置 string 属性
        builder.set_node_attr_string(tensor, "str_attr", "hello")
        assert producer.get_attr("str_attr") == "hello"

        # 测试设置 bool 属性
        builder.set_node_attr_bool(tensor, "bool_attr", True)
        assert producer.get_attr("bool_attr") == True

    def test_set_node_attributes_invalid_args(self, builder):
        """测试无效参数设置节点属性"""
        tensor = builder.create_const_float(1.0)

        # 测试 int64 属性无效参数
        with pytest.raises(TypeError, match="Tensor must be a TensorHolder"):
            builder.set_node_attr_int64("invalid", "attr", 42)
        with pytest.raises(TypeError, match="Attribute name must be a string"):
            builder.set_node_attr_int64(tensor, 123, 42)
        with pytest.raises(TypeError, match="Value must be an integer"):
            builder.set_node_attr_int64(tensor, "attr", "invalid")

        # 测试 string 属性无效参数
        with pytest.raises(TypeError, match="Value must be a string"):
            builder.set_node_attr_string(tensor, "attr", 123)

        # 测试 bool 属性无效参数
        with pytest.raises(TypeError, match="Value must be a boolean"):
            builder.set_node_attr_bool(tensor, "attr", "invalid")

    def test_build_graph(self, builder):
        """测试构建图"""
        # 创建输入和常量
        input_tensor = builder.create_input(0)

        # 设置输出
        builder.set_graph_output(input_tensor, 0)

        # 构建图
        graph = builder.build_and_reset()
        assert isinstance(graph, Graph)

    def test_build_and_reset_with_outputs(self, builder):
        """测试使用 outputs 参数构建图"""
        # 创建多个输入
        input_tensor1 = builder.create_input(0)
        input_tensor2 = builder.create_input(1)

        # 使用 outputs 参数构建图
        graph = builder.build_and_reset([input_tensor1, input_tensor2])
        assert isinstance(graph, Graph)

    def test_build_and_reset_with_outputs_invalid_type(self, builder):
        """测试 outputs 参数类型错误"""
        input_tensor = builder.create_input(0)

        # outputs 不是 list
        with pytest.raises(TypeError, match="Outputs must be a list"):
            builder.build_and_reset(input_tensor)

        # outputs 是 list 但元素不是 TensorHolder
        with pytest.raises(TypeError, match="All outputs must be TensorHolder objects"):
            builder.build_and_reset([input_tensor, "invalid"])

        # outputs 是 list 但包含 None
        with pytest.raises(TypeError, match="All outputs must be TensorHolder objects"):
            builder.build_and_reset([input_tensor, None])

    def test_copy_not_supported(self, builder):
        """测试复制不支持"""
        with pytest.raises(RuntimeError, match="GraphBuilder does not support copy"):
            builder.__copy__()

    def test_deepcopy_not_supported(self, builder):
        """测试深复制不支持"""
        with pytest.raises(RuntimeError, match="GraphBuilder does not support deepcopy"):
            builder.__deepcopy__({})

    @pytest.mark.parametrize("data_type", [
        DataType.DT_FLOAT,
        DataType.DT_INT32,
        DataType.DT_INT64,
        DataType.DT_BOOL,
        DataType.DT_STRING
    ])
    def test_create_input_with_different_data_types(self, builder, data_type):
        """测试使用不同数据类型创建输入"""
        input_tensor = builder.create_input(0, data_type=data_type)
        assert isinstance(input_tensor, TensorHolder)

    @pytest.mark.parametrize("format", [
        Format.FORMAT_NCHW,
        Format.FORMAT_NHWC,
        Format.FORMAT_ND,
        Format.FORMAT_NC1HWC0,
        Format.FORMAT_FRACTAL_Z
    ])
    def test_create_input_with_different_formats(self, builder, format):
        """测试使用不同格式创建输入"""
        input_tensor = builder.create_input(0, format=format)
        assert isinstance(input_tensor, TensorHolder)

    @pytest.mark.parametrize("shape", [
        [1],
        [1, 2],
        [1, 2, 3],
        [1, 2, 3, 4],
        [224, 224, 3]
    ])
    def test_create_input_with_different_shapes(self, builder, shape):
        """测试使用不同形状创建输入"""
        input_tensor = builder.create_input(0, shape=shape)
        assert isinstance(input_tensor, TensorHolder)

    def test_complex_graph_construction(self, builder):
        """测试复杂图构建"""
        # 创建多个输入
        input1 = builder.create_input(0, name="input1", data_type=DataType.DT_FLOAT, format=Format.FORMAT_NCHW,
                                      shape=[1, 3, 224, 224])
        input2 = builder.create_input(1, name="input2", data_type=DataType.DT_FLOAT, format=Format.FORMAT_NHWC,
                                      shape=[1, 3, 224, 224])

        # 创建常量
        const1 = builder.create_const_float(1.0)
        const2 = builder.create_const_float(2.0)

        # 创建标量和向量
        scalar = builder.create_scalar_int64(100)
        vector = builder.create_vector_int64([1, 2, 3, 4])

        # 创建变量
        variable = builder.create_variable(0, "my_var")

        # 设置图属性
        builder.set_graph_attr_string("graph_name", "complex_graph")
        builder.set_graph_attr_int64("version", 1)
        builder.set_graph_attr_bool("optimized", True)

        # 设置tensor属性
        builder.set_tensor_attr_string(const1, "tensor_name", "const1")
        builder.set_tensor_attr_int64(const1, "tensor_id", 1)
        builder.set_tensor_attr_bool(const1, "trainable", False)

        # 设置节点属性
        builder.set_node_attr_string(const1, "node_name", "const_node")
        builder.set_node_attr_int64(const1, "node_id", 1)
        builder.set_node_attr_bool(const1, "optimized", True)

        # 设置输出
        builder.set_graph_output(input1, 0)
        builder.set_graph_output(input2, 1)

        # 构建图
        graph = builder.build_and_reset()
        assert isinstance(graph, Graph)

    # ============ 属性作用域测试 ============
    def test_attr_scope_non_nested(self, builder):
        """测试非嵌套属性作用域"""
        from ge.es.graph_builder import attr_scope

        # 在属性作用域外创建tensor，应该没有属性
        tensor1 = builder.create_const_float(1.0)
        producer1 = tensor1._get_node_snapshot()

        # 在属性作用域内创建tensor，应该有属性
        with attr_scope({
            "node_type": "custom_node",
            "priority": 100,
            "enabled": True,
            "description": "test node",
            "dimensions": [1, 2, 3],
            "flags": [True, False, True]
        }):
            tensor2 = builder.create_const_float(2.0)
            producer2 = tensor2._get_node_snapshot()

            # 验证属性已设置
            assert producer2.get_attr("node_type") == "custom_node"
            assert producer2.get_attr("priority") == 100
            assert producer2.get_attr("enabled") == True
            assert producer2.get_attr("description") == "test node"
            assert producer2.get_attr("dimensions") == [1, 2, 3]
            assert producer2.get_attr("flags") == [True, False, True]

        # 作用域外创建的tensor不应该有这些属性
        try:
            producer1.get_attr("node_type")
            assert False, "Should have raised RuntimeError"
        except RuntimeError:
            pass  # 期望的行为

    def test_attr_scope_nested(self, builder):
        """测试嵌套属性作用域"""
        from ge.es.graph_builder import attr_scope

        # 外层作用域
        with attr_scope({
            "outer_attr": "outer_value",
            "common_attr": "outer_common",
            "outer_int": 10,
            "outer_bool": True
        }):
            # 在外层作用域创建tensor
            tensor1 = builder.create_const_float(1.0)
            producer1 = tensor1._get_node_snapshot()

            # 验证外层属性
            assert producer1.get_attr("outer_attr") == "outer_value"
            assert producer1.get_attr("common_attr") == "outer_common"
            assert producer1.get_attr("outer_int") == 10
            assert producer1.get_attr("outer_bool") == True

            # 内层作用域
            with attr_scope({
                "inner_attr": "inner_value",
                "common_attr": "inner_common",  # 覆盖外层属性
                "inner_int": 20,
                "inner_bool": False,
                "inner_list": [1, 2, 3, 4]
            }):
                # 在内层作用域创建tensor
                tensor2 = builder.create_const_float(2.0)
                producer2 = tensor2._get_node_snapshot()

                # 验证内层属性（包括继承的外层属性）
                assert producer2.get_attr("outer_attr") == "outer_value"
                assert producer2.get_attr("common_attr") == "inner_common"  # 被覆盖
                assert producer2.get_attr("outer_int") == 10
                assert producer2.get_attr("outer_bool") == True
                assert producer2.get_attr("inner_attr") == "inner_value"
                assert producer2.get_attr("inner_int") == 20
                assert producer2.get_attr("inner_bool") == False
                assert producer2.get_attr("inner_list") == [1, 2, 3, 4]

            # 回到外层作用域，创建新的tensor
            tensor3 = builder.create_const_float(3.0)
            producer3 = tensor3._get_node_snapshot()

            # 验证只有外层属性，内层属性应该不存在
            assert producer3.get_attr("outer_attr") == "outer_value"
            assert producer3.get_attr("common_attr") == "outer_common"  # 恢复原值
            assert producer3.get_attr("outer_int") == 10
            assert producer3.get_attr("outer_bool") == True

            # 内层属性应该不存在
            try:
                producer3.get_attr("inner_attr")
                assert False, "Should have raised RuntimeError"
            except RuntimeError:
                pass  # 期望的行为

    def test_attr_scope_multiple_types(self, builder):
        """测试属性作用域中的多种数据类型"""
        from ge.es.graph_builder import attr_scope

        with attr_scope({
            # 基本类型
            "int_attr": 42,
            "bool_attr": True,
            "string_attr": "hello world",

            # 列表类型
            "int_list": [1, 2, 3, 4, 5],
            "float_list": [1.1, 2.2, 3.3],
            "bool_list": [True, False, True, False],
            "string_list": ["a", "b", "c"]
        }):
            # 创建不同类型的tensor
            tensor1 = builder.create_const_float(1.0)
            tensor2 = builder.create_const_int64(100)
            tensor4 = builder.create_vector_int64([1, 2, 3])

            # 验证所有tensor的producer都有相同的属性
            for tensor in [tensor1, tensor2, tensor4]:
                producer = tensor._get_node_snapshot()

                # 基本类型
                assert producer.get_attr("int_attr") == 42
                assert producer.get_attr("bool_attr") == True
                assert producer.get_attr("string_attr") == "hello world"

                # 列表类型
                assert producer.get_attr("int_list") == [1, 2, 3, 4, 5]
                # 浮点数列表需要特殊处理精度问题
                float_list = producer.get_attr("float_list")
                expected_float_list = [1.1, 2.2, 3.3]
                assert len(float_list) == len(expected_float_list)
                for expected, actual in zip(expected_float_list, float_list):
                    assert abs(expected - actual) < 1e-6, f"Float list mismatch: expected {expected}, got {actual}"
                assert producer.get_attr("bool_list") == [True, False, True, False]
                assert producer.get_attr("string_list") == ["a", "b", "c"]

    def test_attr_scope_unsupported_types(self, builder):
        """测试不支持的属性类型"""
        from ge.es.graph_builder import attr_scope

        # 测试字典类型
        with pytest.raises(ValueError, match="Unsupported attribute type: <class 'dict'> for value: {'key': 'value'}"):
            with attr_scope({"dict_attr": {"key": "value"}}):
                builder.create_const_float(1.0)

        # 测试混合类型列表 - 会抛出ValueError因为不匹配任何特定的列表类型
        with pytest.raises(ValueError,
                           match="Unsupported attribute type: <class 'list'> for value: \\[1, 'hello', True, 3.14\\]"):
            with attr_scope({"mixed_list": [1, "hello", True, 3.14]}):
                builder.create_const_float(1.0)

    def test_control_dependency_scope_non_nested(self, builder, monkeypatch):
        """测试非嵌套控制依赖作用域"""
        from ge.es.graph_builder import control_dependency_scope
        from ge._capi.pyes_graph_builder_wrapper import get_generated_lib
        def mock_get_math_lib(self):
            return get_generated_lib("libes_ut_test.so")

        monkeypatch.setattr(TensorHolder, '_get_math_operator_lib', mock_get_math_lib)

        # 在控制依赖作用域外创建tensor，应该没有控制依赖
        tensor0 = builder.create_const_float(0.0)
        producer0 = tensor0._get_node_snapshot()
        tensor1 = builder.create_const_float(1.0)
        producer1 = tensor1._get_node_snapshot()


        # 在控制依赖作用域内创建tensor，应该有控制依赖
        with control_dependency_scope([tensor0, tensor1]):
            tensor2 = builder.create_const_float(2.0) + 1.0
            producer2 = tensor2._get_node_snapshot()

        # 验证控制依赖
        assert len(producer1.get_in_control_nodes()) == 0
        assert len(producer2.get_in_control_nodes()) == 2
        assert producer2.type == "Add"
        assert producer2.get_in_control_nodes()[0].name == producer0.name
        assert producer2.get_in_control_nodes()[1].name == producer1.name

    def test_control_dependency_scope_nested(self, builder):
        """测试嵌套控制依赖作用域"""
        from ge.es.graph_builder import control_dependency_scope
        tensor1 = builder.create_const_float(1.0)
        tensor2 = builder.create_const_float(2.0)
        producer1 = tensor1._get_node_snapshot()
        producer2 = tensor2._get_node_snapshot()

        # 外层作用域
        with control_dependency_scope([tensor1]):
            tensor3 = builder.create_const_float(3.0)
            producer3 = tensor3._get_node_snapshot()
            with control_dependency_scope([tensor2]):
                tensor4 = builder.create_const_float(4.0)
                producer4 = tensor4._get_node_snapshot()

        # 验证控制依赖
        assert len(producer3.get_in_control_nodes()) == 1
        assert producer3.get_in_control_nodes()[0].name == producer1.name
        assert len(producer4.get_in_control_nodes()) == 2
        assert producer4.get_in_control_nodes()[0].name == producer1.name
        assert producer4.get_in_control_nodes()[1].name == producer2.name

    def test_generated_lib_cache_hit(self):
        """测试生成库的缓存命中"""
        from ge._capi.pyes_graph_builder_wrapper import get_generated_lib

        # 使用 libes_ut_test.so 测试缓存
        lib1 = get_generated_lib("libes_ut_test.so")
        lib2 = get_generated_lib("libes_ut_test.so")
        assert lib1 is lib2

        # 再次调用应该返回相同的缓存库
        lib3 = get_generated_lib("libes_ut_test.so")
        assert lib1 is lib3

    def test_generated_lib_with_absolute_path(self):
        """测试使用绝对路径加载库"""
        from ge._capi.pyes_graph_builder_wrapper import get_generated_lib, _dir, _lib_cache
        import os

        # 构造 libes_ut_test.so 的绝对路径
        abs_path = os.path.abspath(os.path.join(_dir, "libes_ut_test.so"))

        if not os.path.exists(abs_path):
            pytest.skip("libes_ut_test.so not found")

        # 确保这个绝对路径不在缓存中
        if abs_path in _lib_cache:
            del _lib_cache[abs_path]

        # 使用绝对路径加载
        lib = get_generated_lib(abs_path)
        assert lib is not None

        # 验证使用绝对路径也能成功加载
        assert abs_path in _lib_cache

    def test_generated_lib_not_found(self):
        """测试加载不存在的库"""
        from ge._capi.pyes_graph_builder_wrapper import get_generated_lib

        # 尝试加载一个不存在的库，应该抛出 RuntimeError
        with pytest.raises(RuntimeError, match="is not available"):
            get_generated_lib("libes_nonexistent.so")

    def test_generated_lib_configure_cache(self):
        """测试库配置函数的缓存机制"""
        from ge._capi.pyes_graph_builder_wrapper import (
            _configure_generated_lib,
            _configured_lib_ids,
            get_generated_lib
        )

        # 加载 libes_ut_test.so
        lib = get_generated_lib("libes_ut_test.so")
        lib_id = id(lib)

        # 验证库ID已经被添加到配置缓存中
        assert lib_id in _configured_lib_ids

        # 再次配置同一个库，应该直接返回（命中缓存）
        _configure_generated_lib(lib)

    def test_get_default_lib_from_cache(self):
        """测试从缓存中获取默认库"""
        from ge._capi.pyes_graph_builder_wrapper import (
            get_generated_lib,
            DEFAULT_GENERATED_LIB_NAME
        )
        import ge._capi.pyes_graph_builder_wrapper as wrapper

        # 保存原始状态
        orig_lib = wrapper._default_lib
        orig_available = wrapper._default_lib_available

        try:
            # 模拟默认库已经被加载并缓存的场景
            # 使用 libes_ut_test.so 作为模拟的默认库
            mock_lib = get_generated_lib("libes_ut_test.so")
            wrapper._default_lib = mock_lib
            wrapper._default_lib_available = True

            # 现在调用 get_generated_lib() 或 get_generated_lib(DEFAULT_GENERATED_LIB_NAME)
            # 应该返回缓存的 _default_lib
            cached_lib = get_generated_lib()
            assert cached_lib is mock_lib

            # 也可以显式使用 DEFAULT_GENERATED_LIB_NAME
            cached_lib2 = get_generated_lib(DEFAULT_GENERATED_LIB_NAME)
            assert cached_lib2 is mock_lib

        finally:
            # 恢复原始状态
            wrapper._default_lib = orig_lib
            wrapper._default_lib_available = orig_available

    def test_lib_cache_utilities(self):
        """测试库缓存工具函数"""
        from ge._capi.pyes_graph_builder_wrapper import (
            get_generated_lib,
            get_loaded_lib_names,
            clear_lib_cache
        )
        import ge._capi.pyes_graph_builder_wrapper as wrapper

        # 保存原始缓存
        orig_cache = wrapper._lib_cache.copy()

        try:
            # 加载一个库
            lib = get_generated_lib("libes_ut_test.so")
            assert lib is not None

            # 获取已加载的库名称列表
            loaded_libs = get_loaded_lib_names()
            assert isinstance(loaded_libs, list)
            assert "libes_ut_test.so" in loaded_libs

            # 清除缓存
            clear_lib_cache()

            # 验证缓存已清空
            loaded_libs_after_clear = get_loaded_lib_names()
            assert len(loaded_libs_after_clear) == 0

        finally:
            # 恢复原始缓存
            wrapper._lib_cache.clear()
            wrapper._lib_cache.update(orig_cache)

    def test_geapi_and_session_lib_utilities(self):
        """测试 geapi 和 session 库工具函数"""
        # 测试 geapi_wrapper
        from ge._capi.pygeapi_wrapper import get_geapi_lib, is_geapi_lib_loaded
        # 检查库是否加载
        assert is_geapi_lib_loaded() == True
        # 获取库句柄
        lib = get_geapi_lib()
        assert lib is not None

        # 测试 session_wrapper
        from ge._capi.pysession_wrapper import get_session_lib, is_session_lib_loaded
        # 检查库是否加载
        assert is_session_lib_loaded() == True
        # 获取库句柄
        lib = get_session_lib()
        assert lib is not None
