#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

__all__ = [
    "DType",
    "DT_FLOAT",
    "DT_FLOAT16",
    "DT_INT8",
    "DT_INT16",
    "DT_UINT16",
    "DT_UINT8",
    "DT_INT32",
    "DT_INT64",
    "DT_UINT32",
    "DT_UINT64",
    "DT_BOOL",
    "DT_DOUBLE",
    "DT_STRING",
]

import numpy as np
from . import data_wrapper as dwrapper

_dwrapper_dtype_to_python_dtype_str = {
    dwrapper.DataType.DT_FLOAT: "DT_FLOAT",
    dwrapper.DataType.DT_FLOAT16: "DT_FLOAT16",
    dwrapper.DataType.DT_INT8: "DT_INT8",
    dwrapper.DataType.DT_INT16: "DT_INT16",
    dwrapper.DataType.DT_UINT16: "DT_UINT16",
    dwrapper.DataType.DT_UINT8: "DT_UINT8",
    dwrapper.DataType.DT_INT32: "DT_INT32",
    dwrapper.DataType.DT_INT64: "DT_INT64",
    dwrapper.DataType.DT_UINT32: "DT_UINT32",
    dwrapper.DataType.DT_UINT64: "DT_UINT64",
    dwrapper.DataType.DT_BOOL: "DT_BOOL",
    dwrapper.DataType.DT_DOUBLE: "DT_DOUBLE",
    dwrapper.DataType.DT_STRING: "DT_STRING",
}


class DType(object):
    def __init__(self, dtype: dwrapper.DataType):
        self.dtype = dtype

    def __str__(self):
        return str(_dwrapper_dtype_to_python_dtype_str.get(self.dtype, None))

    def __repr__(self):
        return self.__str__()


DT_FLOAT = DType(dwrapper.DataType.DT_FLOAT)
DT_FLOAT16 = DType(dwrapper.DataType.DT_FLOAT16)
DT_BF16 = DType(dwrapper.DataType.DT_BF16)
DT_INT8 = DType(dwrapper.DataType.DT_INT8)
DT_INT16 = DType(dwrapper.DataType.DT_INT16)
DT_UINT16 = DType(dwrapper.DataType.DT_UINT16)
DT_UINT8 = DType(dwrapper.DataType.DT_UINT8)
DT_INT32 = DType(dwrapper.DataType.DT_INT32)
DT_INT64 = DType(dwrapper.DataType.DT_INT64)
DT_UINT32 = DType(dwrapper.DataType.DT_UINT32)
DT_UINT64 = DType(dwrapper.DataType.DT_UINT64)
DT_BOOL = DType(dwrapper.DataType.DT_BOOL)
DT_DOUBLE = DType(dwrapper.DataType.DT_DOUBLE)
DT_STRING = DType(dwrapper.DataType.DT_STRING)

_dwrapper_dtype_to_python_dtype = {
    dwrapper.DataType.DT_FLOAT: DT_FLOAT,
    dwrapper.DataType.DT_FLOAT16: DT_FLOAT16,
    dwrapper.DataType.DT_INT8: DT_INT8,
    dwrapper.DataType.DT_INT16: DT_INT16,
    dwrapper.DataType.DT_UINT16: DT_UINT16,
    dwrapper.DataType.DT_UINT8: DT_UINT8,
    dwrapper.DataType.DT_INT32: DT_INT32,
    dwrapper.DataType.DT_INT64: DT_INT64,
    dwrapper.DataType.DT_UINT32: DT_UINT32,
    dwrapper.DataType.DT_UINT64: DT_UINT64,
    dwrapper.DataType.DT_BOOL: DT_BOOL,
    dwrapper.DataType.DT_DOUBLE: DT_DOUBLE,
    dwrapper.DataType.DT_STRING: DT_STRING,
}


def get_python_dtype_from_dwrapper_dtype(dwrapper_dtype):
    dtype = _dwrapper_dtype_to_python_dtype.get(dwrapper_dtype, None)
    if not dtype:
        raise ValueError(f"The data type {dwrapper_dtype} is not support.")
    return dtype


_dflow_dtype_to_np_dtype = {
    DT_FLOAT: np.float32,
    DT_FLOAT16: np.float16,
    DT_INT8: np.int8,
    DT_INT16: np.int16,
    DT_UINT16: np.uint16,
    DT_UINT8: np.uint8,
    DT_INT32: np.int32,
    DT_INT64: np.int64,
    DT_UINT32: np.uint32,
    DT_UINT64: np.uint64,
    DT_BOOL: np.bool_,
    DT_DOUBLE: np.float64,
    DT_STRING: np.bytes_,
}

_np_dtype_to_dflow_dtype = {
    np.dtype(np.float32): DT_FLOAT,
    np.dtype(np.float16): DT_FLOAT16,
    np.dtype(np.int8): DT_INT8,
    np.dtype(np.int16): DT_INT16,
    np.dtype(np.uint16): DT_UINT16,
    np.dtype(np.uint8): DT_UINT8,
    np.dtype(np.int32): DT_INT32,
    np.dtype(np.int64): DT_INT64,
    np.dtype(np.uint32): DT_UINT32,
    np.dtype(np.uint64): DT_UINT64,
    np.dtype(np.bool_): DT_BOOL,
    np.dtype(np.float64): DT_DOUBLE,
}
