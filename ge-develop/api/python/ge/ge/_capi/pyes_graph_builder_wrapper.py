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
from typing import Dict, Optional

from ._lib_loader import load_lib_from_path

# 常用C类型别名
c_void_p = ctypes.c_void_p
c_char_p = ctypes.c_char_p
c_int = ctypes.c_int
c_int32 = ctypes.c_int32
c_int64 = ctypes.c_int64
c_float = ctypes.c_float
c_bool = ctypes.c_bool
c_uint32 = ctypes.c_uint32
c_uint64 = ctypes.c_uint64
c_double = ctypes.c_double


# C结构体定义
class EsCTensorHolder(ctypes.Structure):
    """C层 struct EsCTensorHolder"""
    pass


class EsCGraphBuilder(ctypes.Structure):
    """C层 struct EsCGraphBuilder"""
    pass


class EsCGraph(ctypes.Structure):
    """C层 struct EsCGraph"""
    pass


class EsCTensor(ctypes.Structure):
    """C层 struct EsCTensor"""
    pass


# 指针类型定义
EsCTensorHolderPtr = ctypes.POINTER(EsCTensorHolder)
EsCGraphBuilderPtr = ctypes.POINTER(EsCGraphBuilder)
EsCTensorPtr = ctypes.POINTER(EsCTensor)

# 基础库
BASE_LIB_NAME = "libeager_style_graph_builder_base.so"
# 默认生成的es api的全量库
DEFAULT_GENERATED_LIB_NAME = "libes_all.so"
# 生成的数学库
MATH_LIB_NAME = "libes_math.so"

# 优先使用 GLOBAL|NOW，确保符号可见并尽早完成重定位
_dir = os.path.dirname(os.path.abspath(__file__))
_dlopen_mode = getattr(os, "RTLD_GLOBAL", 0) | getattr(os, "RTLD_NOW", 0)

# 尝试在当前py的安装路径下，加载生成的es api的C库
_lib_cache: Dict[str, ctypes.CDLL] = {}
_configured_lib_ids = set()
_default_lib_available = False
_default_lib = None
esb_lib = None


def _configure_generated_lib(lib: ctypes.CDLL) -> None:
    lib_id = id(lib)
    if lib_id in _configured_lib_ids:
        return

    # 二元操作符：加减乘除（仅在符号存在时配置）
    for op_name in ['EsAdd', 'EsSub', 'EsMul', 'EsDiv']:
        if hasattr(lib, op_name):
            getattr(lib, op_name).restype = EsCTensorHolderPtr
            getattr(lib, op_name).argtypes = [EsCTensorHolderPtr, EsCTensorHolderPtr]

    _configured_lib_ids.add(lib_id)


# Try to load generated libraries first so base weak-symbol calls can bind to strong impls.
for lib_name in [DEFAULT_GENERATED_LIB_NAME, MATH_LIB_NAME]:
    try:
        lib = load_lib_from_path(lib_name, _dir, mode=_dlopen_mode)
        _configure_generated_lib(lib)
        _lib_cache[lib_name] = lib
        if lib_name == DEFAULT_GENERATED_LIB_NAME:
            _default_lib = lib
            _default_lib_available = True
    except OSError:
        continue

# Load base library after generated libraries.
esb_lib = load_lib_from_path(BASE_LIB_NAME, _dir, mode=_dlopen_mode)
if esb_lib is None:
    raise RuntimeError(f"Failed to load {BASE_LIB_NAME}")


def is_generated_lib_available():
    """Check if default generated operator library is available"""
    return _default_lib_available


def get_loaded_lib_names():
    """Get list of currently loaded library names.

    Returns:
        List of library names that have been loaded and cached.
    """
    return list(_lib_cache.keys())


def clear_lib_cache():
    """Clear the library cache.

    Warning: This only clears Python references. The actual shared libraries
    remain loaded in the process memory and cannot be unloaded.
    """
    _lib_cache.clear()


def get_generated_lib(lib_name: str = None):
    """Get specified generated operator library.

    Args:
        lib_name: Library name. If None, returns default library.

    Returns:
        Loaded ctypes.CDLL object.

    Raises:
        RuntimeError: If library is not available.
    """
    global _default_lib, _default_lib_available
    target = lib_name or DEFAULT_GENERATED_LIB_NAME

    # Return cached default library if available
    if target == DEFAULT_GENERATED_LIB_NAME and _default_lib_available and _default_lib is not None:
        return _default_lib

    # Return cached library if already loaded
    if target in _lib_cache:
        return _lib_cache[target]

    # Load library from file system
    try:
        lib = load_lib_from_path(target, _dir, mode=_dlopen_mode)
        _configure_generated_lib(lib)
        _lib_cache[target] = lib
    except OSError as exc:
        raise RuntimeError(f"Generated library {target} is not available: {exc}") from exc

    # Update default library reference if loading default lib name
    if target == DEFAULT_GENERATED_LIB_NAME:
        _default_lib, _default_lib_available = lib, True

    return lib

# ============ GraphBuilder C API ============
# 创建/销毁
esb_lib.EsCreateGraphBuilder.restype = EsCGraphBuilderPtr
esb_lib.EsCreateGraphBuilder.argtypes = [c_char_p]

esb_lib.EsDestroyGraphBuilder.restype = None
esb_lib.EsDestroyGraphBuilder.argtypes = [EsCGraphBuilderPtr]

# 图输入创建
esb_lib.EsCreateGraphInputWithDetails.restype = EsCTensorHolderPtr
esb_lib.EsCreateGraphInputWithDetails.argtypes = [EsCGraphBuilderPtr, c_int64, c_char_p, c_char_p, c_int, c_int,
                                                  ctypes.POINTER(c_int64), c_int64]

esb_lib.EsCreateGraphInput.restype = EsCTensorHolderPtr
esb_lib.EsCreateGraphInput.argtypes = [EsCGraphBuilderPtr, c_int64]

# 常量创建 - 根据 esb_funcs.h 中的实际接口定义
esb_lib.EsCreateConstInt64.restype = EsCTensorHolderPtr
esb_lib.EsCreateConstInt64.argtypes = [EsCGraphBuilderPtr, ctypes.POINTER(c_int64), ctypes.POINTER(c_int64), c_int64]

esb_lib.EsCreateConstFloat.restype = EsCTensorHolderPtr
esb_lib.EsCreateConstFloat.argtypes = [EsCGraphBuilderPtr, ctypes.POINTER(c_float), ctypes.POINTER(c_int64), c_int64]

esb_lib.EsCreateConstUInt64.restype = EsCTensorHolderPtr
esb_lib.EsCreateConstUInt64.argtypes = [EsCGraphBuilderPtr, ctypes.POINTER(c_uint64), ctypes.POINTER(c_int64), c_int64]

esb_lib.EsCreateConstInt32.restype = EsCTensorHolderPtr
esb_lib.EsCreateConstInt32.argtypes = [EsCGraphBuilderPtr, ctypes.POINTER(c_int32), ctypes.POINTER(c_int64), c_int64]

esb_lib.EsCreateConstUInt32.restype = EsCTensorHolderPtr
esb_lib.EsCreateConstUInt32.argtypes = [EsCGraphBuilderPtr, ctypes.POINTER(c_uint32), ctypes.POINTER(c_int64), c_int64]

# 向量和标量创建
esb_lib.EsCreateVectorInt64.restype = EsCTensorHolderPtr
esb_lib.EsCreateVectorInt64.argtypes = [EsCGraphBuilderPtr, ctypes.POINTER(c_int64), c_int64]

esb_lib.EsCreateScalarInt64.restype = EsCTensorHolderPtr
esb_lib.EsCreateScalarInt64.argtypes = [EsCGraphBuilderPtr, c_int64]

esb_lib.EsCreateScalarInt32.restype = EsCTensorHolderPtr
esb_lib.EsCreateScalarInt32.argtypes = [EsCGraphBuilderPtr, c_int32]

esb_lib.EsCreateScalarFloat.restype = EsCTensorHolderPtr
esb_lib.EsCreateScalarFloat.argtypes = [EsCGraphBuilderPtr, c_float]

esb_lib.EsCreateScalarUInt64.restype = EsCTensorHolderPtr
esb_lib.EsCreateScalarUInt64.argtypes = [EsCGraphBuilderPtr, c_uint64]

esb_lib.EsCreateScalarUInt32.restype = EsCTensorHolderPtr
esb_lib.EsCreateScalarUInt32.argtypes = [EsCGraphBuilderPtr, c_uint32]

# 变量创建
esb_lib.EsCreateVariable.restype = EsCTensorHolderPtr
esb_lib.EsCreateVariable.argtypes = [EsCGraphBuilderPtr, c_int32, c_char_p]

# 图构建
esb_lib.EsBuildGraphAndReset.restype = c_void_p
esb_lib.EsBuildGraphAndReset.argtypes = [EsCGraphBuilderPtr]

# 获取拥有者
esb_lib.EsGetProducer.restype = c_void_p
esb_lib.EsGetProducer.argtypes = [EsCTensorHolderPtr]

# 获取构建器
esb_lib.EsGetOwnerBuilder.restype = EsCGraphBuilderPtr
esb_lib.EsGetOwnerBuilder.argtypes = [EsCTensorHolderPtr]

# ============ TensorHolder C API ============
# 数据类型和格式设置
esb_lib.EsSetDataType.restype = c_uint32
esb_lib.EsSetDataType.argtypes = [EsCTensorHolderPtr, c_int]

esb_lib.EsSetFormat.restype = c_uint32
esb_lib.EsSetFormat.argtypes = [EsCTensorHolderPtr, c_int]

# 形状设置
esb_lib.EsSetShape.restype = c_uint32
esb_lib.EsSetShape.argtypes = [EsCTensorHolderPtr, ctypes.POINTER(c_int64), c_int64]

# 图输出设置
esb_lib.EsSetGraphOutput.restype = c_uint32
esb_lib.EsSetGraphOutput.argtypes = [EsCTensorHolderPtr, c_int64]

# ============ 属性设置 API ============
# 图属性设置
esb_lib.EsSetInt64AttrForGraph.restype = c_uint32
esb_lib.EsSetInt64AttrForGraph.argtypes = [EsCGraphBuilderPtr, c_char_p, c_int64]

esb_lib.EsSetStringAttrForGraph.restype = c_uint32
esb_lib.EsSetStringAttrForGraph.argtypes = [EsCGraphBuilderPtr, c_char_p, c_char_p]

esb_lib.EsSetBoolAttrForGraph.restype = c_uint32
esb_lib.EsSetBoolAttrForGraph.argtypes = [EsCGraphBuilderPtr, c_char_p, c_bool]

# Tensor属性设置
esb_lib.EsSetInt64AttrForTensor.restype = c_uint32
esb_lib.EsSetInt64AttrForTensor.argtypes = [EsCTensorHolderPtr, c_char_p, c_int64]

esb_lib.EsSetStringAttrForTensor.restype = c_uint32
esb_lib.EsSetStringAttrForTensor.argtypes = [EsCTensorHolderPtr, c_char_p, c_char_p]

esb_lib.EsSetBoolAttrForTensor.restype = c_uint32
esb_lib.EsSetBoolAttrForTensor.argtypes = [EsCTensorHolderPtr, c_char_p, c_bool]

# 节点属性设置
esb_lib.EsSetInt64AttrForNode.restype = c_uint32
esb_lib.EsSetInt64AttrForNode.argtypes = [EsCTensorHolderPtr, c_char_p, c_int64]

esb_lib.EsSetStringAttrForNode.restype = c_uint32
esb_lib.EsSetStringAttrForNode.argtypes = [EsCTensorHolderPtr, c_char_p, c_char_p]

esb_lib.EsSetBoolAttrForNode.restype = c_uint32
esb_lib.EsSetBoolAttrForNode.argtypes = [EsCTensorHolderPtr, c_char_p, c_bool]

# 控制边设置
esb_lib.EsAddControlEdge.restype = c_uint32
esb_lib.EsAddControlEdge.argtypes = [EsCTensorHolderPtr, ctypes.POINTER(EsCTensorHolderPtr), c_int64]

# ============ Tensor C API ============
esb_lib.EsCreateEsCTensor.restype = EsCTensorPtr
esb_lib.EsCreateEsCTensor.argtypes = [c_void_p, ctypes.POINTER(c_int64), c_int64, c_int, c_int]

esb_lib.EsCreateEsCTensorFromFile.restype = EsCTensorPtr
esb_lib.EsCreateEsCTensorFromFile.argtypes = [c_char_p, ctypes.POINTER(c_int64), c_int64, c_int, c_int]
