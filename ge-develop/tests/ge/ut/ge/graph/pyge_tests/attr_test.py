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
AttrValue 功能测试 - 使用 pytest 框架
测试 attr.py 中的 AttrValue 类的各种功能
"""

import pytest
import sys
import os

# 添加 ge 到 Python 路径
try:
    from ge.graph._attr import _AttrValue as AttrValue
    from ge.graph.types import AttrValueType, DataType
except ImportError as e:
    pytest.skip(f"无法导入 ge 模块: {e}", allow_module_level=True)


class TestAttrValue:
    """AttrValue 功能测试类"""

    @pytest.fixture
    def attr_value(self):
        """创建 AttrValue 实例的 fixture"""
        return AttrValue()

    def test_attr_value_creation(self):
        """测试 AttrValue 创建"""
        av = AttrValue()
        assert av is not None
        assert hasattr(av, '_av_ptr')
        assert hasattr(av, '_value_type')
        assert hasattr(av, '_cached_value')

    def test_string_operations(self, attr_value):
        """测试字符串操作"""
        # 设置字符串
        result = attr_value.set_string("Hello, World!")
        assert result is True

        # 获取字符串
        value = attr_value.get_string()
        assert value == "Hello, World!"

        # 测试空字符串
        attr_value.set_string("")
        assert attr_value.get_string() == ""

        # 测试中文字符串
        attr_value.set_string("你好，世界！")
        assert attr_value.get_string() == "你好，世界！"

    def test_numeric_operations(self, attr_value):
        """测试数值操作"""
        # 测试整数
        attr_value.set_int(42)
        assert attr_value.get_int() == 42

        # 测试负数
        attr_value.set_int(-100)
        assert attr_value.get_int() == -100

        # 测试浮点数
        attr_value.set_float(3.14)
        assert abs(attr_value.get_float() - 3.14) < 1e-6

        # 测试负数浮点数
        attr_value.set_float(-2.718)
        assert abs(attr_value.get_float() - (-2.718)) < 1e-6

        # 测试布尔值
        attr_value.set_bool(True)
        assert attr_value.get_bool() is True

        attr_value.set_bool(False)
        assert attr_value.get_bool() is False

    def test_list_operations(self, attr_value):
        """测试列表操作"""
        # 测试整数列表
        test_int_list = [1, 2, 3, 4, 5]
        attr_value.set_list_int(test_int_list)
        assert attr_value.get_list_int() == test_int_list

        # 测试浮点数列表
        test_float_list = [1.1, 2.2, 3.3]
        attr_value.set_list_float(test_float_list)
        result_float_list = attr_value.get_list_float()
        assert len(result_float_list) == len(test_float_list)
        for i, (expected, actual) in enumerate(zip(test_float_list, result_float_list)):
            assert abs(expected - actual) < 1e-6, f"Index {i}: expected {expected}, got {actual}"

        # 测试布尔列表
        test_bool_list = [True, False, True, False]
        attr_value.set_list_bool(test_bool_list)
        assert attr_value.get_list_bool() == test_bool_list

        # 测试字符串列表
        test_string_list = ["hello", "world", "test"]
        attr_value.set_list_string(test_string_list)
        assert attr_value.get_list_string() == test_string_list

    def test_empty_list_validation(self, attr_value):
        """测试空列表验证"""
        # 测试空列表应该抛出 ValueError
        with pytest.raises(ValueError, match="Empty list is not supported"):
            attr_value.set_list_int([])

        with pytest.raises(ValueError, match="Empty list is not supported"):
            attr_value.set_list_float([])

        with pytest.raises(ValueError, match="Empty list is not supported"):
            attr_value.set_list_bool([])

        with pytest.raises(ValueError, match="Empty list is not supported"):
            attr_value.set_list_string([])

        with pytest.raises(ValueError, match="Empty list is not supported"):
            attr_value.set_list_data_type([])

    def test_data_type_operations(self, attr_value):
        """测试数据类型操作"""
        # 测试数据类型设置和获取
        test_data_type = DataType.DT_FLOAT
        attr_value.set_data_type(test_data_type)
        assert attr_value.get_data_type() == test_data_type

        # 测试数据类型列表
        test_data_type_list = [DataType.DT_FLOAT, DataType.DT_INT32, DataType.DT_BOOL, DataType.DT_STRING]
        attr_value.set_list_data_type(test_data_type_list)
        assert attr_value.get_list_data_type() == test_data_type_list

    def test_type_validation(self, attr_value):
        """测试类型验证"""
        # 测试字符串类型错误
        with pytest.raises(TypeError):
            attr_value.set_string(123)

        with pytest.raises(TypeError):
            attr_value.set_string(None)

        # 测试整数类型错误
        with pytest.raises(TypeError):
            attr_value.set_int("not_a_number")

        with pytest.raises(TypeError):
            attr_value.set_int(3.14)  # 浮点数不能直接设置给整数

        # 测试布尔类型错误
        with pytest.raises(TypeError):
            attr_value.set_bool("not_a_bool")

        with pytest.raises(TypeError):
            attr_value.set_bool(1)  # 整数不能直接设置给布尔值

        # 测试列表类型错误
        with pytest.raises(TypeError):
            attr_value.set_list_int("not_a_list")

        with pytest.raises(TypeError):
            attr_value.set_list_string(123)

    def test_caching_behavior(self, attr_value):
        """测试缓存行为"""
        # 设置值
        attr_value.set_string("cached_value")

        # 第一次获取（应该缓存）
        val1 = attr_value.get_string()

        # 第二次获取（应该从缓存）
        val2 = attr_value.get_string()

        assert val1 == val2 == "cached_value"

        # 设置新值（应该清除缓存）
        attr_value.set_string("new_value")
        val3 = attr_value.get_string()

        assert val3 == "new_value"

    def test_value_type_property(self, attr_value):
        """测试 value_type 属性"""
        # 初始状态
        initial_type = attr_value.value_type

        # 设置字符串后检查类型
        attr_value.set_string("test")
        assert attr_value.value_type == AttrValueType.VT_STRING

        # 设置整数后检查类型
        attr_value.set_int(42)
        assert attr_value.value_type == AttrValueType.VT_INT

        # 设置浮点数后检查类型
        attr_value.set_float(3.14)
        assert attr_value.value_type == AttrValueType.VT_FLOAT

        # 设置布尔值后检查类型
        attr_value.set_bool(True)
        assert attr_value.value_type == AttrValueType.VT_BOOL

    def test_str_representation(self, attr_value):
        """测试字符串表示"""
        # 测试字符串值的表示
        attr_value.set_string("test_string")
        str_repr = str(attr_value)
        assert "AttrValue" in str_repr
        assert "test_string" in str_repr

        # 测试整数值的表示
        attr_value.set_int(42)
        str_repr = str(attr_value)
        assert "AttrValue" in str_repr
        assert "42" in str_repr

    def test_multiple_instances(self):
        """测试多个实例的独立性"""
        av1 = AttrValue()
        av2 = AttrValue()

        # 设置不同的值
        av1.set_string("instance1")
        av2.set_string("instance2")

        # 验证独立性
        assert av1.get_string() == "instance1"
        assert av2.get_string() == "instance2"

        # 修改一个实例不应该影响另一个
        av1.set_int(100)
        assert av2.get_string() == "instance2"  # 应该保持不变

    def test_edge_cases(self, attr_value):
        """测试边界情况"""
        # 测试极大整数
        large_int = 2 ** 63 - 1
        attr_value.set_int(large_int)
        assert attr_value.get_int() == large_int

        # 测试极小整数
        small_int = -2 ** 63
        attr_value.set_int(small_int)
        assert attr_value.get_int() == small_int

        # 测试极大浮点数
        large_float = 1e10
        attr_value.set_float(large_float)
        assert abs(attr_value.get_float() - large_float) < 1e-6

        # 测试极小浮点数
        small_float = 1e-10
        attr_value.set_float(small_float)
        assert abs(attr_value.get_float() - small_float) < 1e-15


# 参数化测试
class TestAttrValueParametrized:
    """使用参数化测试的 AttrValue 测试类"""

    @pytest.fixture
    def attr_value(self):
        return AttrValue()

    @pytest.mark.parametrize("test_string", [
        "simple",
        "with spaces",
        "with\nnewlines",
        "with\ttabs",
        "with special chars: !@#$%^&*()",
        "中文测试",
        "",  # 空字符串
    ])
    def test_string_values(self, attr_value, test_string):
        """参数化测试各种字符串值"""
        attr_value.set_string(test_string)
        assert attr_value.get_string() == test_string

    @pytest.mark.parametrize("test_int", [
        0, 1, -1, 42, -100, 2 ** 31 - 1, -2 ** 31, 2 ** 63 - 1, -2 ** 63
    ])
    def test_int_values(self, attr_value, test_int):
        """参数化测试各种整数值"""
        attr_value.set_int(test_int)
        assert attr_value.get_int() == test_int

    @pytest.mark.parametrize("test_float", [
        0.0, 1.0, -1.0, 3.14159, -2.71828, 1e10, 1e-10, float('inf'), -float('inf')
    ])
    def test_float_values(self, attr_value, test_float):
        """参数化测试各种浮点数值"""
        if not (test_float == float('inf') or test_float == -float('inf')):
            attr_value.set_float(test_float)
            assert abs(attr_value.get_float() - test_float) < 1e-6

    @pytest.mark.parametrize("test_bool", [True, False])
    def test_bool_values(self, attr_value, test_bool):
        """参数化测试布尔值"""
        attr_value.set_bool(test_bool)
        assert attr_value.get_bool() == test_bool

    @pytest.mark.parametrize("test_list", [
        [1, 2, 3],
        [1.1, 2.2, 3.3],
        [True, False, True],
        ["a", "b", "c"],
        [1],  # 单元素列表
    ])
    def test_list_values(self, attr_value, test_list):
        """参数化测试各种列表值"""
        if test_list and isinstance(test_list[0], int):
            attr_value.set_list_int(test_list)
            assert attr_value.get_list_int() == test_list
        elif test_list and isinstance(test_list[0], float):
            attr_value.set_list_float(test_list)
            result = attr_value.get_list_float()
            assert len(result) == len(test_list)
            for expected, actual in zip(test_list, result):
                assert abs(expected - actual) < 1e-6
        elif test_list and isinstance(test_list[0], bool):
            attr_value.set_list_bool(test_list)
            assert attr_value.get_list_bool() == test_list
        elif test_list and isinstance(test_list[0], str):
            attr_value.set_list_string(test_list)
            assert attr_value.get_list_string() == test_list
