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
from .pyes_graph_builder_wrapper import EsCTensorPtr

from ._lib_loader import load_lib_from_path

LIB_NAME = "libgraph_wrapper.so"
_dir = os.path.dirname(os.path.abspath(__file__))
graph_lib = load_lib_from_path(LIB_NAME, _dir)
if graph_lib is None:
    raise RuntimeError(f"Failed to load library {LIB_NAME}")

# 常用C类型别名
c_void_p = ctypes.c_void_p
# c_char_p 会自动转换为 Python 字符串
c_char_p = ctypes.c_char_p
c_char_ptr = ctypes.POINTER(ctypes.c_char)
c_int = ctypes.c_int
c_int32 = ctypes.c_int32
c_int64 = ctypes.c_int64
c_float = ctypes.c_float
c_bool = ctypes.c_bool
c_uint32 = ctypes.c_uint32
c_size_t = ctypes.c_size_t

# ============ AttrValue C API ============
# 创建/销毁
graph_lib.GeApiWrapper_AttrValue_Create.restype = c_void_p
graph_lib.GeApiWrapper_AttrValue_Create.argtypes = []

graph_lib.GeApiWrapper_AttrValue_Destroy.restype = None
graph_lib.GeApiWrapper_AttrValue_Destroy.argtypes = [c_void_p]

# 类型获取
graph_lib.GeApiWrapper_AttrValue_GetValueType.restype = c_int32
graph_lib.GeApiWrapper_AttrValue_GetValueType.argtypes = [c_void_p]

# 标量值获取
graph_lib.GeApiWrapper_AttrValue_GetString.restype = c_char_ptr
graph_lib.GeApiWrapper_AttrValue_GetString.argtypes = [c_void_p]

graph_lib.GeApiWrapper_AttrValue_GetFloat.restype = c_int
graph_lib.GeApiWrapper_AttrValue_GetFloat.argtypes = [c_void_p, ctypes.POINTER(c_float)]

graph_lib.GeApiWrapper_AttrValue_GetBool.restype = c_int
graph_lib.GeApiWrapper_AttrValue_GetBool.argtypes = [c_void_p, ctypes.POINTER(c_bool)]

graph_lib.GeApiWrapper_AttrValue_GetInt.restype = c_int
graph_lib.GeApiWrapper_AttrValue_GetInt.argtypes = [c_void_p, ctypes.POINTER(c_int64)]

graph_lib.GeApiWrapper_AttrValue_GetDataType.restype = c_int
graph_lib.GeApiWrapper_AttrValue_GetDataType.argtypes = [c_void_p, ctypes.POINTER(c_int32)]

# 列表值获取
graph_lib.GeApiWrapper_AttrValue_GetListString.restype = ctypes.POINTER(c_char_p)
graph_lib.GeApiWrapper_AttrValue_GetListString.argtypes = [c_void_p, ctypes.POINTER(c_int64)]

graph_lib.GeApiWrapper_AttrValue_GetListFloat.restype = ctypes.POINTER(c_float)
graph_lib.GeApiWrapper_AttrValue_GetListFloat.argtypes = [c_void_p, ctypes.POINTER(c_int64)]

graph_lib.GeApiWrapper_AttrValue_GetListBool.restype = ctypes.POINTER(c_bool)
graph_lib.GeApiWrapper_AttrValue_GetListBool.argtypes = [c_void_p, ctypes.POINTER(c_int64)]

graph_lib.GeApiWrapper_AttrValue_GetListInt.restype = ctypes.POINTER(c_int64)
graph_lib.GeApiWrapper_AttrValue_GetListInt.argtypes = [c_void_p, ctypes.POINTER(c_int64)]

graph_lib.GeApiWrapper_AttrValue_GetListDataType.restype = ctypes.POINTER(c_int32)
graph_lib.GeApiWrapper_AttrValue_GetListDataType.argtypes = [c_void_p, ctypes.POINTER(c_int64)]

# 标量值设置
graph_lib.GeApiWrapper_AttrValue_SetString.restype = c_int
graph_lib.GeApiWrapper_AttrValue_SetString.argtypes = [c_void_p, c_char_p]

graph_lib.GeApiWrapper_AttrValue_SetFloat.restype = c_int
graph_lib.GeApiWrapper_AttrValue_SetFloat.argtypes = [c_void_p, c_float]

graph_lib.GeApiWrapper_AttrValue_SetBool.restype = c_int
graph_lib.GeApiWrapper_AttrValue_SetBool.argtypes = [c_void_p, c_bool]

graph_lib.GeApiWrapper_AttrValue_SetInt.restype = c_int
graph_lib.GeApiWrapper_AttrValue_SetInt.argtypes = [c_void_p, c_int64]

graph_lib.GeApiWrapper_AttrValue_SetDataType.restype = c_int
graph_lib.GeApiWrapper_AttrValue_SetDataType.argtypes = [c_void_p, c_int32]

# 列表值设置
graph_lib.GeApiWrapper_AttrValue_SetListString.restype = c_int
graph_lib.GeApiWrapper_AttrValue_SetListString.argtypes = [c_void_p, ctypes.POINTER(c_char_p), c_int64]

graph_lib.GeApiWrapper_AttrValue_SetListFloat.restype = c_int
graph_lib.GeApiWrapper_AttrValue_SetListFloat.argtypes = [c_void_p, ctypes.POINTER(c_float), c_int64]

graph_lib.GeApiWrapper_AttrValue_SetListBool.restype = c_int
graph_lib.GeApiWrapper_AttrValue_SetListBool.argtypes = [c_void_p, ctypes.POINTER(c_bool), c_int64]

graph_lib.GeApiWrapper_AttrValue_SetListInt.restype = c_int
graph_lib.GeApiWrapper_AttrValue_SetListInt.argtypes = [c_void_p, ctypes.POINTER(c_int64), c_int64]

graph_lib.GeApiWrapper_AttrValue_SetListDataType.restype = c_int
graph_lib.GeApiWrapper_AttrValue_SetListDataType.argtypes = [c_void_p, ctypes.POINTER(c_int32), c_int64]

# 内存释放函数
graph_lib.GeApiWrapper_FreeString.restype = None
graph_lib.GeApiWrapper_FreeString.argtypes = [c_char_ptr]

graph_lib.GeApiWrapper_FreeListString.restype = None
graph_lib.GeApiWrapper_FreeListString.argtypes = [ctypes.POINTER(c_char_p)]

graph_lib.GeApiWrapper_FreeListFloat.restype = None
graph_lib.GeApiWrapper_FreeListFloat.argtypes = [ctypes.POINTER(c_float)]

graph_lib.GeApiWrapper_FreeListBool.restype = None
graph_lib.GeApiWrapper_FreeListBool.argtypes = [ctypes.POINTER(c_bool)]

graph_lib.GeApiWrapper_FreeListInt.restype = None
graph_lib.GeApiWrapper_FreeListInt.argtypes = [ctypes.POINTER(c_int64)]

graph_lib.GeApiWrapper_FreeListDataType.restype = None
graph_lib.GeApiWrapper_FreeListDataType.argtypes = [ctypes.POINTER(c_int32)]

# ============ Graph C API ============
# 创建/销毁
graph_lib.GeApiWrapper_Graph_CreateGraph.restype = c_void_p
graph_lib.GeApiWrapper_Graph_CreateGraph.argtypes = [c_char_p]

graph_lib.GeApiWrapper_Graph_DestroyGraph.restype = None
graph_lib.GeApiWrapper_Graph_DestroyGraph.argtypes = [c_void_p]

graph_lib.GeApiWrapper_Graph_FreeGraphArray.restype = None
graph_lib.GeApiWrapper_Graph_FreeGraphArray.argtypes = [ctypes.POINTER(c_void_p)]

# 属性操作
graph_lib.GeApiWrapper_Graph_GetName.restype = c_char_ptr
graph_lib.GeApiWrapper_Graph_GetName.argtypes = [c_void_p]

graph_lib.GeApiWrapper_Graph_GetAllNodes.restype = ctypes.POINTER(c_void_p)
graph_lib.GeApiWrapper_Graph_GetAllNodes.argtypes = [c_void_p, ctypes.POINTER(c_size_t)]

graph_lib.GeApiWrapper_Graph_GetAttr.restype = c_int
graph_lib.GeApiWrapper_Graph_GetAttr.argtypes = [c_void_p, c_char_p, c_void_p]

graph_lib.GeApiWrapper_Graph_SetAttr.restype = c_int
graph_lib.GeApiWrapper_Graph_SetAttr.argtypes = [c_void_p, c_char_p, c_void_p]

graph_lib.GeApiWrapper_Graph_Dump_To_Onnx.restype = c_int
graph_lib.GeApiWrapper_Graph_Dump_To_Onnx.argtypes = [c_void_p, c_char_p, c_char_p]

graph_lib.GeApiWrapper_Graph_Dump_To_File.restype = c_int
graph_lib.GeApiWrapper_Graph_Dump_To_File.argtypes = [c_void_p, c_int32, c_char_p]

graph_lib.GeApiWrapper_Graph_Dump_To_Stream.restype = c_char_ptr
graph_lib.GeApiWrapper_Graph_Dump_To_Stream.argtypes = [c_void_p, c_int32]

graph_lib.GeApiWrapper_Graph_SaveToAir.restype = c_int
graph_lib.GeApiWrapper_Graph_SaveToAir.argtypes = [c_void_p, c_char_p]


graph_lib.GeApiWrapper_Graph_LoadFromAir.restype = c_int
graph_lib.GeApiWrapper_Graph_LoadFromAir.argtypes = [c_void_p, c_char_ptr]
 
graph_lib.GeApiWrapper_Graph_GetDirectNode.restype = ctypes.POINTER(c_void_p)
graph_lib.GeApiWrapper_Graph_GetDirectNode.argtypes = [c_void_p, ctypes.POINTER(c_size_t)]

graph_lib.GeApiWrapper_Graph_RemoveEdge.restype = c_int
graph_lib.GeApiWrapper_Graph_RemoveEdge.argtypes = [c_void_p, c_void_p, c_int32, c_void_p, c_int32]

graph_lib.GeApiWrapper_Graph_AddDataEdge.restype = c_int
graph_lib.GeApiWrapper_Graph_AddDataEdge.argtypes =  [c_void_p, c_void_p, c_int32, c_void_p, c_int32]

graph_lib.GeApiWrapper_Graph_AddControlEdge.restype = c_int
graph_lib.GeApiWrapper_Graph_AddControlEdge.argtypes = [c_void_p, c_void_p, c_void_p]

graph_lib.GeApiWrapper_Graph_RemoveNode.restype = c_int
graph_lib.GeApiWrapper_Graph_RemoveNode.argtypes = [c_void_p, c_void_p]


graph_lib.GeApiWrapper_Graph_FindNodeByName.restype = int
graph_lib.GeApiWrapper_Graph_FindNodeByName.argtypes = [c_void_p, c_char_p, ctypes.POINTER(c_void_p)]

# 子图相关 API
graph_lib.GeApiWrapper_Graph_GetAllSubgraphs.restype = ctypes.POINTER(c_void_p)
graph_lib.GeApiWrapper_Graph_GetAllSubgraphs.argtypes = [c_void_p, ctypes.POINTER(c_size_t)]

graph_lib.GeApiWrapper_Graph_GetSubGraph.restype = c_void_p
graph_lib.GeApiWrapper_Graph_GetSubGraph.argtypes = [c_void_p, c_char_p]

graph_lib.GeApiWrapper_Graph_AddSubGraph.restype = c_int
graph_lib.GeApiWrapper_Graph_AddSubGraph.argtypes = [c_void_p, c_void_p]

graph_lib.GeApiWrapper_Graph_RemoveSubgraph.restype = c_int
graph_lib.GeApiWrapper_Graph_RemoveSubgraph.argtypes = [c_void_p, c_char_p]

# ============ GNode C API ============
# 销毁
graph_lib.GeApiWrapper_GNode_DestroyGNode.restype = None
graph_lib.GeApiWrapper_GNode_DestroyGNode.argtypes = [c_void_p]

graph_lib.GeApiWrapper_GNode_FreeGNodeArray.restype = None
graph_lib.GeApiWrapper_GNode_FreeGNodeArray.argtypes = [ctypes.POINTER(c_void_p)]

graph_lib.GeApiWrapper_GNode_FreeIntArray.restype = None
graph_lib.GeApiWrapper_GNode_FreeIntArray.argtypes = [ctypes.POINTER(c_int32)]

# 基本信息
graph_lib.GeApiWrapper_GNode_GetName.restype = c_char_ptr
graph_lib.GeApiWrapper_GNode_GetName.argtypes = [c_void_p]

graph_lib.GeApiWrapper_GNode_GetType.restype = c_char_ptr
graph_lib.GeApiWrapper_GNode_GetType.argtypes = [c_void_p]

# 节点关系
graph_lib.GeApiWrapper_GNode_GetInControlNodes.restype = ctypes.POINTER(c_void_p)
graph_lib.GeApiWrapper_GNode_GetInControlNodes.argtypes = [c_void_p, ctypes.POINTER(c_size_t)]

graph_lib.GeApiWrapper_GNode_GetInDataNodesAndPortIndexes.restype = c_int
graph_lib.GeApiWrapper_GNode_GetInDataNodesAndPortIndexes.argtypes = [c_void_p, c_int32, ctypes.POINTER(c_void_p),
                                                                      ctypes.POINTER(c_int32)]

graph_lib.GeApiWrapper_GNode_GetOutDataNodesAndPortIndexes.restype = c_int
graph_lib.GeApiWrapper_GNode_GetOutDataNodesAndPortIndexes.argtypes = [c_void_p, c_int32, ctypes.POINTER(ctypes.POINTER(c_void_p)),
ctypes.POINTER(ctypes.POINTER(c_int32)), ctypes.POINTER(c_int)]    

graph_lib.GeApiWrapper_GNode_GetOutControlNodes.restype = ctypes.POINTER(c_void_p)
graph_lib.GeApiWrapper_GNode_GetOutControlNodes.argtypes = [c_void_p, ctypes.POINTER(c_size_t)]   

graph_lib.GeApiWrapper_GNode_GetInputsSize.restype = c_size_t
graph_lib.GeApiWrapper_GNode_GetInputsSize.argtypes = [c_void_p]  

graph_lib.GeApiWrapper_GNode_GetOutputsSize.restype = c_size_t
graph_lib.GeApiWrapper_GNode_GetOutputsSize.argtypes = [c_void_p]                    

graph_lib.GeApiWrapper_GNode_HasAttr.restype = c_bool
graph_lib.GeApiWrapper_GNode_HasAttr.argtypes = [c_void_p, c_char_p]                      

# 属性操作
graph_lib.GeApiWrapper_GNode_GetAttr.restype = c_int
graph_lib.GeApiWrapper_GNode_GetAttr.argtypes = [c_void_p, c_char_p, c_void_p]

graph_lib.GeApiWrapper_GNode_SetAttr.restype = c_int
graph_lib.GeApiWrapper_GNode_SetAttr.argtypes = [c_void_p, c_char_p, c_void_p]

graph_lib.GeApiWrapper_GNode_GetInputAttr.restype = c_int
graph_lib.GeApiWrapper_GNode_GetInputAttr.argtypes = [c_void_p, c_char_p, c_uint32, c_void_p]

graph_lib.GeApiWrapper_GNode_SetInputAttr.restype = c_int
graph_lib.GeApiWrapper_GNode_SetInputAttr.argtypes = [c_void_p, c_char_p, c_uint32, c_void_p]

graph_lib.GeApiWrapper_GNode_GetOutputAttr.restype = c_int
graph_lib.GeApiWrapper_GNode_GetOutputAttr.argtypes = [c_void_p, c_char_p, c_uint32, c_void_p]

graph_lib.GeApiWrapper_GNode_SetOutputAttr.restype = c_int
graph_lib.GeApiWrapper_GNode_SetOutputAttr.argtypes = [c_void_p, c_char_p, c_uint32, c_void_p]


# ============ Tensor C API ============
graph_lib.GeApiWrapper_Tensor_SetFormat.restype = c_int
graph_lib.GeApiWrapper_Tensor_SetFormat.argtypes = [c_void_p, ctypes.POINTER(c_int)]

graph_lib.GeApiWrapper_Tensor_GetFormat.restype = c_int
graph_lib.GeApiWrapper_Tensor_GetFormat.argtypes = [c_void_p]

graph_lib.GeApiWrapper_Tensor_SetDataType.restype = c_int
graph_lib.GeApiWrapper_Tensor_SetDataType.argtypes = [c_void_p, ctypes.POINTER(c_int)]

graph_lib.GeApiWrapper_Tensor_GetDataType.restype = c_int
graph_lib.GeApiWrapper_Tensor_GetDataType.argtypes = [c_void_p]

graph_lib.GeApiWrapper_Tensor_CreateTensor.restype = EsCTensorPtr
graph_lib.GeApiWrapper_Tensor_CreateTensor.argtypes = []

graph_lib.GeApiWrapper_Tensor_DestroyEsCTensor.restype = None
graph_lib.GeApiWrapper_Tensor_DestroyEsCTensor.argtypes = [c_void_p]

graph_lib.GeApiWrapper_Tensor_GetShape.restype = c_int
graph_lib.GeApiWrapper_Tensor_GetShape.argtypes = [c_void_p, ctypes.POINTER(ctypes.POINTER(c_int64)), ctypes.POINTER(c_size_t)]

graph_lib.GeApiWrapper_Tensor_FreeDimsArray.restype = None
graph_lib.GeApiWrapper_Tensor_FreeDimsArray.argtypes = [ctypes.POINTER(c_int64)]

graph_lib.GeApiWrapper_Tensor_GetData.restype = c_char_ptr
graph_lib.GeApiWrapper_Tensor_GetData.argtypes = [c_void_p]


