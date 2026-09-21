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

"""Graph module for GraphEngine graph operations."""

import ctypes
from typing import List, Optional
from ge._capi.pygraph_wrapper import graph_lib
from ge._capi.pysession_wrapper import session_lib
from ge.graph.graph import Graph
from ge.graph.tensor import Tensor


def _str_list_to_c_array(python_list: list):
    """
        Convert python list to c array

        Parameters:
            python_list: python list.

        Returns:
            c_array: c array_ptr.
    """
    size = len(python_list)
    c_array = (ctypes.c_char_p * size)()
    for i, item in enumerate(python_list):
        c_array[i] = item.encode('utf-8')
    return c_array


class Session:
    """Session class for session operations.

    This class provides a Pythonic interface for session operations
    using the session C API.
    """

    def __init__(self, options: Optional[dict] = None) -> None:
        """Session Initialize a Session"""
        self._handle = None
        self._owns_handle = False
        if options is None:
            self._handle = session_lib.GeApiWrapper_Session_CreateSession()
        elif isinstance(options, dict):
            keys = [k for k in options.keys()]
            values = [v for v in options.values()]
            c_array_key = _str_list_to_c_array(keys)
            c_array_value = _str_list_to_c_array(values)
            c_array_key_ptr = ctypes.cast(c_array_key, ctypes.POINTER(ctypes.c_char_p))
            c_value_key_ptr = ctypes.cast(c_array_value, ctypes.POINTER(ctypes.c_char_p))
            self._handle = session_lib.GeApiWrapper_Session_CreateSessionWithOptions(c_array_key_ptr, c_value_key_ptr,
                                                                                     len(keys))
        else:
            raise TypeError("option must be a dictionary")
        if not self._handle:
            raise RuntimeError("Failed to create session")
        self._owns_handle = True

    def __del__(self) -> None:
        """Clean up resources."""
        if self._owns_handle:
            session_lib.GeApiWrapper_Session_DestroySession(self._handle)
            self._handle = None

    def __copy__(self) -> None:
        """Copy is not supported."""
        raise RuntimeError("Session does not support copy")

    def __deepcopy__(self, session) -> None:
        """Deep copy is not supported."""
        raise RuntimeError("Session does not support deepcopy")

    def add_graph(self, graph_id: int, add_graph: Graph, options: dict = None) -> None:
        if not isinstance(graph_id, int):
            raise TypeError("Graph_id must be an integer")
        if not isinstance(add_graph, Graph):
            raise TypeError("Add_graph must be a Graph")
        if options is None:
            ret = session_lib.GeApiWrapper_Session_AddGraph(self._handle, ctypes.c_uint32(graph_id), add_graph._handle)
        elif not isinstance(options, dict):
            raise TypeError("options must be a dictionary")
        else:
            keys = [k for k in options.keys()]
            values = [v for v in options.values()]
            c_array_key = _str_list_to_c_array(keys)
            c_array_value = _str_list_to_c_array(values)
            c_array_key_ptr = ctypes.cast(c_array_key, ctypes.POINTER(ctypes.c_char_p))
            c_value_key_ptr = ctypes.cast(c_array_value, ctypes.POINTER(ctypes.c_char_p))
            ret = session_lib.GeApiWrapper_Session_AddGraphWithOptions(self._handle, ctypes.c_uint32(graph_id),
                                                                       add_graph._handle,
                                                                       c_array_key_ptr, c_value_key_ptr, len(keys))
        if ret != 0:
            raise RuntimeError(f"Failed to add graph, graph_id is {graph_id}")
        return ret

    def run_graph(self, graph_id: int, inputs: List[Tensor]) -> List[Tensor]:
        if not isinstance(graph_id, int):
            raise TypeError("Graph_id must be an integer")
        if not all(isinstance(input_tensor, Tensor) for input_tensor in inputs):
            raise TypeError("All elements in inputs must be the type of Tensor")
        inputs_handle = []
        for inputs_tensor in inputs:
            c_inputs_tensor_ptr = ctypes.cast(inputs_tensor._handle, ctypes.c_void_p)
            inputs_handle.append(c_inputs_tensor_ptr)
        arr_type = ctypes.c_void_p * len(inputs_handle)
        arr = arr_type(*inputs_handle)
        tensor_num = ctypes.c_size_t()
        output_tensors = session_lib.GeApiWrapper_Session_RunGraph(self._handle, ctypes.c_uint32(graph_id), arr,
                                                                   len(inputs_handle), ctypes.byref(tensor_num))
        if not output_tensors:
            raise RuntimeError(f"Failed to run graph, graph_id is {graph_id}")
        return [Tensor._create_from(output_tensors[i]) for i in range(tensor_num.value)]
