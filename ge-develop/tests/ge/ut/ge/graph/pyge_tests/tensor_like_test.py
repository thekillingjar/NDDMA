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
TensorLike 功能测试 - 使用 pytest 框架
测试 tensor_like.py 中的参数转换功能
"""

import pytest

# 添加 ge 到 Python 路径
try:
    from ge.es.graph_builder import GraphBuilder, TensorHolder
    from ge.es.tensor_like import (
        convert_to_tensor_holder,
        resolve_builder,
        _flatten_and_infer_shape,
        _unflatten
    )
except ImportError as e:
    pytest.skip(f"无法导入 ge 模块: {e}", allow_module_level=True)


class TestTensorLike:
    @pytest.fixture
    def builder(self):
        """创建 GraphBuilder 实例的 fixture"""
        return GraphBuilder("test_graph")

    @staticmethod
    def test_flatten_and_infer_shape_nested_list():
        """验证不同维度的嵌套列表能正确展平并推断形状"""
        flat1, shape1 = _flatten_and_infer_shape([1, 2, 3, 4])
        assert flat1 == [1, 2, 3, 4]
        assert shape1 == [4]

        flat2, shape2 = _flatten_and_infer_shape([[1, 2], [3, 4]])
        assert flat2 == [1, 2, 3, 4]
        assert shape2 == [2, 2]

        flat2, shape2 = _flatten_and_infer_shape([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        assert flat2 == [1, 2, 3, 4, 5, 6, 7, 8]
        assert shape2 == [2, 2, 2]

    @staticmethod
    def test_flatten_and_infer_shape_empty_top_level_returns_zero_length():
        """验证顶层空列表展平后返回空值与零长度形状"""
        flat, shape = _flatten_and_infer_shape([])
        assert flat == []
        assert shape == [0]

    @staticmethod
    def test_flatten_and_infer_shape_ragged_list_raises():
        """验证不规则嵌套或非法元素会触发异常"""
        with pytest.raises(ValueError, match="Irregular nested list: all sub-lists must have the same shape"):
            _flatten_and_infer_shape([[1, 2], [3]])

        with pytest.raises(TypeError, match="Value must be int/float or nested lists of int|float"):
            _flatten_and_infer_shape(["1"])

        with pytest.raises(TypeError, match="Value must be int/float or nested lists of int|float"):
            _flatten_and_infer_shape([[]])

    @staticmethod
    def test_convert_tensor_to_tensor_holder():
        """验证TensorHolder对象可转换为 TensorHolder"""
        builder = GraphBuilder("test_graph")
        const_tensor = builder.create_const_int64(1)
        result = convert_to_tensor_holder(const_tensor, builder)
        assert isinstance(result, TensorHolder)
        assert result is const_tensor

    def test_convert_scalar_int_to_tensor_holder(self, builder):
        """验证整型标量可转换为 TensorHolder"""
        result = convert_to_tensor_holder(7, builder)
        assert isinstance(result, TensorHolder)
        assert result._handle is not None

        result = convert_to_tensor_holder(0, builder)
        assert isinstance(result, TensorHolder)
        assert result._handle is not None

    def test_convert_scalar_float_to_tensor_holder(self, builder):
        """验证浮点标量可转换为 TensorHolder"""
        result = convert_to_tensor_holder(3.5, builder)
        assert isinstance(result, TensorHolder)
        assert result._handle is not None

    def test_convert_int_list_to_tensor_holder(self, builder):
        """验证整型列表可转换为 TensorHolder"""
        result = convert_to_tensor_holder([[0, 1, 2], [0, 3, 4]], builder)
        assert isinstance(result, TensorHolder)
        assert result._handle is not None

    def test_convert_float_list_to_tensor_holder(self, builder):
        """验证浮点列表可转换为 TensorHolder"""
        result = convert_to_tensor_holder([[1.0, 2.0], [3.0, 4.5]], builder)
        assert isinstance(result, TensorHolder)
        assert result._handle is not None

    def test_convert_none_to_none(self, builder):
        """验证None可转换为 None"""
        result = convert_to_tensor_holder(None, builder)
        assert result is None

    def test_convert_invalid_type_raises_type_error(self, builder):
        """验证非数值或非数值列表会触发 TypeError"""
        with pytest.raises(TypeError, match="Value must be int/float or nested lists of int|float"):
            convert_to_tensor_holder("invalid", builder)

        with pytest.raises(TypeError, match="Value must be int/float or nested lists of int|float"):
            convert_to_tensor_holder(["invalid"], builder)

    @staticmethod
    def test_unflatten():
        """验证去扁平化"""
        res = _unflatten([1, 2, 3, 4, 5, 6], [2, 3])
        assert len(res) == 2
        assert len(res[0]) == 3

    @staticmethod
    def test_unflatten_error():
        """验证去扁平化错误"""
        with pytest.raises(ValueError, match="Length of list 5 does not match shape 6"):
            _unflatten([1, 2, 3, 4, 5], [2, 3])

    @staticmethod
    def test_unflatten_invalid_shape():
        """验证去扁平化非法Shape"""
        with pytest.raises(ValueError, match="Shape cannot be empty"):
            _unflatten([1, 2, 3, 4, 5], None)

    def test_resolve_builder_from_args_success(self, builder):
        """验证能从混合入参中解析出builder"""
        tensor1 = builder.create_const_float(2.0)

        result = resolve_builder(1, [2, 3], tensor1)
        assert result is builder

        tensor2 = builder.create_const_float(2.0)
        tensor_list = [tensor1, tensor2]
        tensor_list1 = []
        tensor_list2 = [None]
        result = resolve_builder(
            1,
            *(tensor_list if tensor_list else [None]),
            *(tensor_list1 if tensor_list1 else [None]),
            *(tensor_list2 if tensor_list2 else [None]))
        assert result is builder

        result = resolve_builder(1, builder)
        assert result is builder

    @staticmethod
    def test_resolve_builder_raises_without_tensor_holder():
        """验证未提供 TensorHolder 时抛出异常"""
        with pytest.raises(ValueError,
                           match="Please ensure at least one input tensor or an explicit owner_builder is provided when supported"):
            resolve_builder(1, [2, 3.0])
