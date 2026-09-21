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
TensorHolder 功能测试 - 使用 pytest 框架
测试 tensor_holder.py 中的 TensorHolder 类的各种功能
"""

import pytest
import sys
import os

# 需要添加 ge 到 Python 路径，否则会报错
try:
    from ge.es.graph_builder import GraphBuilder, TensorHolder
    from ge.graph.types import DataType, Format
    from ge._capi.pyes_graph_builder_wrapper import is_generated_lib_available
except ImportError as e:
    pytest.skip(f"无法导入 ge 模块: {e}", allow_module_level=True)


class TestTensorHolder:
    """TensorHolder 功能测试类"""

    @pytest.fixture
    def builder(self):
        """创建 GraphBuilder 实例的 fixture"""
        return GraphBuilder("test_graph")

    @pytest.fixture
    def tensor_holder(self, builder):
        """创建 TensorHolder 实例的 fixture"""
        return builder.create_const_float(1.0)

    def test_tensor_holder_creation(self, builder):
        """测试 TensorHolder 创建"""
        tensor = builder.create_const_float(3.14)
        assert isinstance(tensor, TensorHolder)
        assert tensor._handle is not None

    def test_tensor_holder_invalid_handle(self, builder):
        """测试无效句柄创建 TensorHolder"""
        with pytest.raises(ValueError, match="Tensor handle cannot be None"):
            TensorHolder._create_from(None, builder)
        with pytest.raises(ValueError, match="Owner builder cannot be None"):
            from ge._capi.pyes_graph_builder_wrapper import EsCTensorHolderPtr, EsCTensorHolder
            TensorHolder._create_from(EsCTensorHolderPtr(EsCTensorHolder()), None)

    def test_set_data_type(self, tensor_holder):
        """测试设置数据类型"""
        result = tensor_holder.set_data_type(DataType.DT_INT32)
        assert isinstance(result, TensorHolder)

    def test_set_data_type_invalid(self, tensor_holder):
        """测试无效数据类型设置"""
        with pytest.raises(TypeError, match="Data type must be a DataType enum"):
            tensor_holder.set_data_type("invalid")

    def test_set_format(self, tensor_holder):
        """测试设置格式"""
        result = tensor_holder.set_format(Format.FORMAT_NCHW)
        assert isinstance(result, TensorHolder)

    def test_set_format_invalid(self, tensor_holder):
        """测试无效格式设置"""
        with pytest.raises(TypeError, match="Format must be a Format enum"):
            tensor_holder.set_format("invalid")

    def test_set_shape(self, tensor_holder):
        """测试设置形状"""
        result = tensor_holder.set_shape([1, 3, 224, 224])
        assert isinstance(result, TensorHolder)

    def test_set_shape_invalid(self, tensor_holder):
        """测试无效形状设置"""
        with pytest.raises(TypeError, match="Shape must be a list of integers"):
            tensor_holder.set_shape("invalid")

        with pytest.raises(TypeError, match="All shape dimensions must be integers"):
            tensor_holder.set_shape([1, "invalid", 3])

    @pytest.mark.skipif(not is_generated_lib_available(),
                        reason="Generated library not available")
    def test_operator_overloading(self, builder):
        """测试操作符重载（需要生成的操作符库）"""
        tensor1 = builder.create_const_float(2.0)
        tensor2 = builder.create_const_float(3.0)

        # 测试 + 操作符
        result = tensor1 + tensor2
        assert isinstance(result, TensorHolder)

    def test_arithmetic_operations_without_lib(self, tensor_holder):
        """测试没有生成库时的算术运算错误"""
        if is_generated_lib_available():
            pytest.skip("Generated library is available, test not applicable")

        tensor2 = tensor_holder

        # 测试加法错误
        with pytest.raises(RuntimeError, match="not available"):
            tensor_holder.add(tensor2)

        # 测试操作符重载错误
        with pytest.raises(RuntimeError, match="not available"):
            var = tensor_holder + tensor2

    @pytest.mark.parametrize("data_type", [
        DataType.DT_FLOAT,
        DataType.DT_INT32,
        DataType.DT_INT64,
        DataType.DT_BOOL,
        DataType.DT_STRING
    ])
    def test_set_different_data_types(self, tensor_holder, data_type):
        """测试设置不同数据类型"""
        result = tensor_holder.set_data_type(data_type)
        assert isinstance(result, TensorHolder)

    @pytest.mark.parametrize("format", [
        Format.FORMAT_NCHW,
        Format.FORMAT_NHWC,
        Format.FORMAT_ND,
        Format.FORMAT_NC1HWC0,
        Format.FORMAT_FRACTAL_Z
    ])
    def test_set_different_formats(self, tensor_holder, format):
        """测试设置不同格式"""
        result = tensor_holder.set_format(format)
        assert isinstance(result, TensorHolder)

    @pytest.mark.parametrize("shape", [
        [],
        [1],
        [1, 2],
        [1, 2, 3],
        [1, 2, 3, 4],
        [224, 224, 3]
    ])
    def test_set_different_shapes(self, tensor_holder, shape):
        """测试设置不同形状"""
        result = tensor_holder.set_shape(shape)
        assert isinstance(result, TensorHolder)

    def test_complex_tensor_operations(self, builder):
        """测试复杂tensor操作"""
        # 创建tensor
        tensor = builder.create_const_float([1.0, 2.0, 3.0, 4.0], shape=[2, 2])

        # 设置数据类型和格式
        tensor.set_data_type(DataType.DT_FLOAT)
        tensor.set_format(Format.FORMAT_NCHW)

        # 设置形状
        tensor.set_shape([2, 2])

        assert isinstance(tensor, TensorHolder)

    def test_get_node_snapshot(self, builder):
        """测试获取producer节点"""
        # 创建一个tensor
        tensor = builder.create_const_float(1.0)

        # 获取producer节点
        producer_node = tensor._get_node_snapshot()

        # 验证返回的是Node对象
        from ge.graph import Node
        assert isinstance(producer_node, Node)
        assert producer_node.name == tensor.name

        # 验证Node对象不拥有所有权（快照模式）
        assert producer_node._owns_handle == False

        # 验证Node对象有基本属性和方法
        assert hasattr(producer_node, 'name')
        assert hasattr(producer_node, 'type')
        assert hasattr(producer_node, 'set_attr')
        assert hasattr(producer_node, 'get_attr')
        assert producer_node._handle is not None

    def test_tensor_holder_strong_reference(self):
        """测试tensor holder的强引用"""

        def create_tensor_and_release_builder():
            builder = GraphBuilder("test_graph")
            tensor = builder.create_const_float(1.0)
            assert tensor.name == "Const_0"
            return tensor, builder.build_and_reset()

        tensor, graph = create_tensor_and_release_builder()
        assert graph is not None
        assert tensor.name == "Const_0"
        assert tensor._builder is not None
        with pytest.raises(RuntimeError, match="Cannot set data type: GraphBuilder has already been built. Create a new GraphBuilder to build another graph"):
            tensor.set_data_type(DataType.DT_FLOAT)


    def test_different_builder_operations(self, builder):
        """测试不同builder的操作"""
        builder1 = GraphBuilder("test_graph1")
        builder2 = GraphBuilder("test_graph2")
        tensor1 = builder1.create_const_float(1.0)
        tensor2 = builder2.create_const_float(2.0)
        with pytest.raises(ValueError, match="Cannot perform add: tensors from different GraphBuilders"):
            tensor1 + tensor2

    def test_arithmetic_operations_with_mocked_lib(self, builder, monkeypatch):
        """测试模拟生成库时的算术运算"""
        from unittest.mock import Mock, MagicMock

        # 创建两个tensor
        tensor1 = builder.create_const_float(2.0)
        tensor2 = builder.create_const_float(3.0)

        # 创建一个新的tensor作为mock返回值的基础
        mock_result_tensor = builder.create_const_float(5.0)
        mock_result_handle = mock_result_tensor._handle

        # Mock生成的库
        mock_lib = Mock()
        mock_lib.EsAdd = MagicMock(return_value=mock_result_handle)
        mock_lib.EsSub = MagicMock(return_value=mock_result_handle)
        mock_lib.EsMul = MagicMock(return_value=mock_result_handle)
        mock_lib.EsDiv = MagicMock(return_value=mock_result_handle)

        # Mock _get_math_operator_lib
        def mock_get_math_lib(self):
            return mock_lib
        
        monkeypatch.setattr(TensorHolder, '_get_math_operator_lib', mock_get_math_lib)

        # 测试加法
        result_add = tensor1.add(tensor2)
        assert isinstance(result_add, TensorHolder)
        assert result_add._builder is tensor1._builder
        assert mock_lib.EsAdd.call_count >= 1

        # 测试减法
        result_sub = tensor1.sub(tensor2)
        assert isinstance(result_sub, TensorHolder)
        assert result_sub._builder is tensor1._builder
        assert mock_lib.EsSub.call_count >= 1

        # 测试乘法
        result_mul = tensor1.mul(tensor2)
        assert isinstance(result_mul, TensorHolder)
        assert result_mul._builder is tensor1._builder
        assert mock_lib.EsMul.call_count >= 1

        # 测试除法
        result_div = tensor1.div(tensor2)
        assert isinstance(result_div, TensorHolder)
        assert result_div._builder is tensor1._builder
        assert mock_lib.EsDiv.call_count >= 1

        # 测试操作符重载
        result_add_op = tensor1 + tensor2
        assert isinstance(result_add_op, TensorHolder)

        result_sub_op = tensor1 - tensor2
        assert isinstance(result_sub_op, TensorHolder)

        result_mul_op = tensor1 * tensor2
        assert isinstance(result_mul_op, TensorHolder)

        result_div_op = tensor1 / tensor2
        assert isinstance(result_div_op, TensorHolder)

        # 测试操作符重载支持数值
        result_add = 3.0 + tensor1 + 3.0
        assert isinstance(result_add, TensorHolder)

        result_sub = 3.0 - tensor1 - 3.0
        assert isinstance(result_sub, TensorHolder)

        result_mul = 3.0 * tensor1 * 3.0
        assert isinstance(result_mul, TensorHolder)

        result_div = 3.0 / tensor1 / 3.0
        assert isinstance(result_div, TensorHolder)

        result_div = [[3.0, 2], [1, 1.0]] / tensor1
        assert isinstance(result_div, TensorHolder)

        with pytest.raises(ValueError, match="Irregular nested list: all sub-lists must have the same shape"):
            [[3.0, 2], [1]] / tensor1

    def test_set_data_type_and_format_and_shape(self, builder):
        """测试设置数据类型、格式和形状"""
        tensor = builder.create_const_float(1.0).set_data_type(DataType.DT_FLOAT).set_format(
            Format.FORMAT_NCHW).set_shape([1, 2, 3, 4])
        assert isinstance(tensor, TensorHolder)
