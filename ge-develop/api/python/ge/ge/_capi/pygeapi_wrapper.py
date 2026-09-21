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

import ctypes
import os
import sys

from ._lib_loader import load_lib_from_path

LIB_NAME = "libgraph_wrapper.so"
_dir = os.path.dirname(os.path.abspath(__file__))
geapi_lib = load_lib_from_path(LIB_NAME, _dir)

c_char_p = ctypes.c_char_p
c_int = ctypes.c_int

# 初始化阶段
geapi_lib.GeApiWrapper_GEInitialize.argtypes = [ctypes.POINTER(c_char_p), ctypes.POINTER(c_char_p), c_int]
geapi_lib.GeApiWrapper_GEInitialize.restype = c_int

geapi_lib.GeApiWrapper_GEFinalize.argtypes = []
geapi_lib.GeApiWrapper_GEFinalize.restype = c_int


def get_geapi_lib():
    """Get the GE API wrapper library handle.
    
    Returns:
        ctypes.CDLL: The loaded libgraph_wrapper.so library handle.
    """
    return geapi_lib


def is_geapi_lib_loaded():
    """Check if GE API library is loaded.
    
    Returns:
        bool: True if library is loaded successfully.
    """
    return geapi_lib is not None
