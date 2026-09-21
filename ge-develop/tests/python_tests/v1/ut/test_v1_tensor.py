#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#-------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# content of test_sample.py
import unittest
from llm_datadist_v1 import Tensor, TensorDesc, DataType
import numpy as np


class TensorUt(unittest.TestCase):
    def setUp(self) -> None:
        print("Begin ", self._testMethodName)

    def tearDown(self) -> None:
        print("End ", self._testMethodName)

    def test_str_tensor(self):
        arr1 = np.array([["aaaa", "bbbb"], ["ccc", "ddd"]])
        src_type = arr1.dtype
        tensor = Tensor(arr1)
        self.assertEqual(tensor.numpy(True).dtype, src_type)
        print(tensor.numpy(True))

    def test_tensor(self):
        arr1 = np.random.rand(2, 3)
        arr1 = np.array(arr1, np.float32)
        print("src numpy:", arr1)
        tensor = Tensor(arr1)
        print("generated:", tensor)
        self.assertEqual(tensor.numpy().dtype, np.float32)

    def test_tensor_with_tensor_desc(self):
        arr1 = np.array([[1 for i in range(256)]], np.int32)
        tensor_desc = TensorDesc(DataType.DT_INT32, (1, 256))
        print(tensor_desc)
        tensor = Tensor(arr1, tensor_desc)
        self.assertEqual(tensor.numpy().dtype, np.int32)

    def test_tensor_bf16(self):
        arr1 = np.array([1.875], np.float16)
        tensor_desc = TensorDesc(DataType.DT_BF16, [1])
        tensor = Tensor(arr1, tensor_desc)
        print("generated numpy:", tensor.numpy())
        self.assertEqual(tensor.numpy().dtype, np.float32)
        self.assertEqual(int(tensor.numpy()[0]), 1)

    def test_tensor_foat16(self):
        arr1 = np.array([1.875], np.float16)
        tensor_desc = TensorDesc(DataType.DT_FLOAT16, [1])
        tensor = Tensor(arr1, tensor_desc)
        print("generated numpy:", tensor.numpy())
        self.assertEqual(tensor.numpy().dtype, np.float16)
    
    def test_tensor_foat32_copy_true(self):
        arr1 = np.array([[1.0, 2.0], [3.0, 4.0]], np.float32)
        tensor_desc = TensorDesc(DataType.DT_FLOAT, (2, 2))
        tensor = Tensor(arr1, tensor_desc)
        res = tensor.numpy(copy=True)
        print("generated numpy:", res)
        np.testing.assert_array_equal(res, arr1)
        self.assertEqual(res.dtype, np.float32)
        self.assertTrue(res.flags.c_contiguous)
        self.assertTrue(res.flags.writeable)
        arr1[0, 0] = 999.0
        self.assertNotEqual(res[0, 0], 999.0)
