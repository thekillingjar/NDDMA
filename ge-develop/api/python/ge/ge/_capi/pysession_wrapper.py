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
session_lib = load_lib_from_path(LIB_NAME, _dir)

# 常用C类型别名
c_void_p = ctypes.c_void_p
# c_char_p 会自动转换为 Python 字符串
c_char_p = ctypes.c_char_p
c_int = ctypes.c_int
c_int32 = ctypes.c_int32
c_uint32 = ctypes.c_uint32
c_size_t = ctypes.c_size_t
c_p_to_char_p = ctypes.POINTER(ctypes.c_char_p)
c_p_to_void_p = ctypes.POINTER(ctypes.c_void_p)

# ============ Session C API ============

session_lib.GeApiWrapper_Session_CreateSession.argtypes = []
session_lib.GeApiWrapper_Session_CreateSession.restype = c_void_p

session_lib.GeApiWrapper_Session_CreateSessionWithOptions.argtypes = [c_p_to_char_p, c_p_to_char_p, c_int]
session_lib.GeApiWrapper_Session_CreateSessionWithOptions.restype = c_void_p

session_lib.GeApiWrapper_Session_AddGraph.argtypes = [c_void_p, c_uint32, c_void_p]
session_lib.GeApiWrapper_Session_AddGraph.restype = c_int

session_lib.GeApiWrapper_Session_AddGraphWithOptions.argtypes = [
    c_void_p, c_uint32, c_void_p, c_p_to_char_p, c_p_to_char_p, c_int
]
session_lib.GeApiWrapper_Session_AddGraphWithOptions.restype = c_int

session_lib.GeApiWrapper_Session_RunGraph.argtypes = [
    c_void_p, c_uint32, c_p_to_void_p, c_int, ctypes.POINTER(c_size_t)
]
session_lib.GeApiWrapper_Session_RunGraph.restype = ctypes.POINTER(c_void_p)

session_lib.GeApiWrapper_Session_DestroySession.argtypes = [c_void_p]
session_lib.GeApiWrapper_Session_DestroySession.restype = None


def get_session_lib():
    """Get the session wrapper library handle.

    Returns:
        ctypes.CDLL: The loaded libgraph_wrapper.so library handle.
    """
    return session_lib


def is_session_lib_loaded():
    """Check if session library is loaded.

    Returns:
        bool: True if library is loaded successfully.
    """
    return session_lib is not None
