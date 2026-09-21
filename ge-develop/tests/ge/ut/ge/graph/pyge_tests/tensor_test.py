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

import pytest
import sys
import os
import ctypes

# 添加 ge 到 Python 路径
try:
    from ge.graph import Tensor
    from ge.graph.tensor import _parse_str_list, unflatten_tensor_data
    from ge.graph.types import DataType, Format
except ImportError as e:
    pytest.skip(f"无法导入 ge 模块: {e}", allow_module_level=True)

class TestTensor:
    """Tensor 功能测试类"""

    @pytest.fixture
    def tensor(self):
        """创建 Tensor 实例的 fixture"""
        try:
            tensor = Tensor([1, 2, 3, 4, 5, 6], None, DataType.DT_INT8, Format.FORMAT_ND, [1, 2, 3])
            return tensor
        except Exception as e:
            print(f"Exception occurs {e}")
            
    def test_tensor_none_args_init(self):
        """测试无参初始化"""
        tensor = Tensor()
        assert tensor._handle is not None
        assert tensor.set_data_type(DataType.DT_FLOAT).get_data_type() == DataType.DT_FLOAT


    def test_tensor_shape_invalid(self):
        """测试shape传入错误"""
        with pytest.raises(TypeError, match="Shape must be a list of integers"):  
            tensor = Tensor([1.0, 2.1, 3.2, 4.3, 5.2, 6.0], shape=1)

    def test_tensor_all_args_invalid(self):
        """测试所有data和file_path同时传入时错误"""
        with pytest.raises(RuntimeError, match="Tensor should be created either by data or by file"):  
            tensor = Tensor([1.0, 2.1, 3.2, 4.3, 5.2, 6.0], "test.txt")

    def test_tensor_data_invalid(self):
        """测试data类型必须为List"""
        with pytest.raises(TypeError, match="data should be List"):  
            tensor = Tensor(1, shape=[1,2,3])

    def test_scalar_tensor_data_invalid(self):
        """测试data类型必须为List"""
        with pytest.raises(TypeError, match="data should be List"):  
            tensor = Tensor(1)

    def test_create_tensor_double_data_invalid(self):
        """测试通过Double构造Tensor"""
        with pytest.raises(RuntimeError, match="DT_DOUBLE is not supported in python Tensor constructor"):
            tensor = Tensor([1.0, 2.1, 3.2, 4.3, 5.2], data_type=DataType.DT_DOUBLE, shape=[5])

    def test_create_tensor_from_file_invalid(self):
        """测试test.txt文件读取"""
        with pytest.raises(RuntimeError, match="Failed to create Tensor"):  
            tensor = Tensor(file_path="test.txt")

    def test_create_tensor_float_data(self):
        """测试通过Float构造Tensor"""
        tensor = Tensor([1.0, 2.1, 3.2, 4.3, 5.2, 6.0], shape=[1,2,3])
        assert tensor._handle is not None

    def test_create_tensor_int_data(self):
        """测试通过Int构造Tensor"""
        tensor = Tensor([1, 2, 3, 4, 5, 6], data_type=DataType.DT_INT32, shape=[1,2,3])
        assert tensor._handle is not None

    def test_create_tensor_bool_data(self):
        """测试通过Bool构造Tensor"""
        tensor = Tensor([True, True, False, False, False, False], None, DataType.DT_BOOL, Format.FORMAT_ND, [1,2,3])
        assert tensor._handle is not None

    def test_create_tensor_invalid_data(self):
        """测试无效构造Tensor"""
        with pytest.raises(RuntimeError, match="Failed to create Tensor"):  
            tensor = Tensor([1.0, bool, 3.2, 4.3, 5.2, 3], None, DataType.DT_BOOL, Format.FORMAT_ND, [1,2,3])

    def test_copy_not_supported(self, tensor):
        """测试复制不支持"""
        with pytest.raises(RuntimeError, match="Tensor does not support copy"):
            tensor.__copy__()

    def test_deepcopy_not_supported(self, tensor):
        """测试深复制不支持"""
        with pytest.raises(RuntimeError, match="Tensor does not support deepcopy"):
            tensor.__deepcopy__({})
    
    def test_tensor_create_from_invalid(self):
        """测试从无效指针创建 Tensor"""
        with pytest.raises(ValueError, match="Tensor pointer cannot be None"):
            Tensor._create_from(None)

    def test_tensor_create_from(self):
        """测试从有效指针创建 Tensor"""
        tensosr1 = Tensor()
        tensor2 = Tensor._create_from(tensosr1._handle)
        assert tensor2 is not None
        tensor2._owns_handle = False

    def test_get_format(self, tensor):
        """测试获取当前Format"""
        format = tensor.get_format()
        assert format == Format.FORMAT_ND

    def test_set_format_invalid(self, tensor):
        """测试无效Format"""
        with pytest.raises(TypeError, match="Format must be a Format"):
            tensor.set_format("")

    def test_set_format_error(self, tensor):
        """测试set_format错误"""
        handle = tensor._handle
        try:
            with pytest.raises(RuntimeError, match="Failed to set format 0"):
                tensor._handle = ctypes.c_void_p(None)
                tensor.set_format(Format.FORMAT_NCHW)
        finally:
            tensor._handle = handle
        
    def test_set_format(self, tensor):
        """测试set_format功能"""
        # 获取当前Format
        current_format = tensor.get_format()
        assert current_format == Format.FORMAT_ND
        
        # 获取修改后的Format
        tensor.set_format(Format.FORMAT_NCHW)
        current_format = tensor.get_format()
        assert current_format == Format.FORMAT_NCHW

    def test_get_data_type(self, tensor):
        # 获取当前DataType
        current_data_type = tensor.get_data_type()
        assert current_data_type == DataType.DT_INT8

    def test_set_data_type(self, tensor):
        """测试set_data_type功能"""
        # 获取当前DataType
        current_data_type = tensor.get_data_type()
        assert current_data_type == DataType.DT_INT8
        
        # 获取修改后的DataType
        tensor.set_data_type(DataType.DT_INT64)
        current_format = tensor.get_data_type()
        assert current_format == DataType.DT_INT64

    def test_set_data_type_invalid(self, tensor):
        """测试必须传入DataType类型"""
        with pytest.raises(TypeError, match="Data_type must be a DataType"):
            tensor.set_data_type("")

    def test_set_data_type_error(self, tensor):
        """测试必须传入DataType类型"""
        handle = tensor._handle
        try:
            with pytest.raises(RuntimeError, match="Failed to set datatype 9"):
                tensor._handle = ctypes.c_void_p(None)
                tensor.set_data_type(DataType.DT_INT64)
        finally:
            tensor._handle = handle
            
    def test_transfer_ownership(self):
        """测试Tensor作为Attr时的所有权转移"""
        from ge.es.graph_builder import GraphBuilder
        builder = GraphBuilder()
        tensor = Tensor()
        assert tensor._owner is None
        assert tensor._owns_handle is True
        assert tensor._handle is not None
        # 测试所有权转移, 当tensor作为Attr参数传递给builder时, 所有权转移给builder, 
        tensor._transfer_ownership_when_pass_as_attr(builder)
        assert tensor._owner is not None
        assert tensor._owner == builder
        assert tensor._owns_handle is False
        assert tensor._handle is not None
        with pytest.raises(RuntimeError, match="Tensor already has an new owner builder :graph, cannot transfer ownership again"):
            tensor._transfer_ownership_when_pass_as_attr(builder)
        # 还原为了避免内存泄漏
        tensor._owns_handle = True

    def test_get_shape(self, tensor):
        # 获取当前Shape
        tensor_shape = tensor.get_shape()
        assert len(tensor_shape) == 3
        assert tensor_shape[0] == 1
        assert tensor_shape[1] == 2
        assert tensor_shape[2] == 3

    def test_get_data(self, tensor):
        # 获取当前Data
        tensor_data = tensor.get_data()
        assert len(tensor_data) == 1
        assert len(tensor_data[0]) == 2
        assert len(tensor_data[0][0]) == 3

    def test_get_data_error(self, tensor):
        # 获取当前Data
        handle = tensor._handle
        try:
            with pytest.raises(RuntimeError, match="Failed to get Tensor data"):
                tensor._handle = ctypes.c_void_p(None)
                tensor_data = tensor.get_data()
        finally:
            tensor._handle = handle

    def test_tensor_print(self, tensor): 
        val = tensor.__str__()
        print(tensor)
        assert val == f'''
        Tensor format is 2, 
        data_type is 2, 
        shape is [1, 2, 3], 
        data is [[[1, 2, 3], [4, 5, 6]]]
        '''

    def test_parse_str_list_invalid(self):
        """验证解析无效 list字符串"""
        with pytest.raises(ValueError, match="Input must start with '[' and end with ']'"):
            _parse_str_list("")

    def test_parse_str_list_empty(self):
        """验证解析空 list字符串"""
        res = _parse_str_list("[]")
        assert len(res) == 0

    def test_parse_str_list_int(self):
        """验证解析int list字符串"""
        res = _parse_str_list("[1, 2, -3, 5, 7]")
        assert len(res) == 5
        assert res[2] == -3

    def test_parse_str_list_float(self):
        """验证解析float list字符串"""
        res = _parse_str_list("[1.0, 2.0, -3.2, 5.7, 7.22, 9.0]")
        assert len(res) == 6
        assert res[2] == -3.2


    def test_parse_str_list_mix(self):
        """验证解析list字符串无效"""
        res = _parse_str_list("[1, 2.0, 3.3, 5, 7]")
        assert len(res) == 5
        assert res[2] == 3.3

    def test_parse_str_list_invalid(self):
        """验证解析list字符串无效"""
        with pytest.raises(ValueError, match="Invalid item: '2;23'"):
            res = _parse_str_list("[1, 2;23 , 3.3, 5, 7]")

    def test_unflatten_tensor_data(self):
        """验证正常TensorData去扁平化"""
        res = unflatten_tensor_data("[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]", [3, 2, 1])
        assert res.__str__() == "[[[1.0], [2.0]], [[3.0], [4.0]], [[5.0], [6.0]]]"