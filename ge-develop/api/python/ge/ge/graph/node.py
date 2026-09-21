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

"""Node module for GraphEngine node operations."""

import ctypes
from typing import List, Optional, Tuple, Any
from ge._capi.pygraph_wrapper import graph_lib
from ._attr import _AttrValue


class Node:
    """Node class for GraphEngine node operations.
    
    This class provides a Pythonic interface for node operations
    using the GraphEngine C API.
    
    Example:
        >>> # Node objects are created internally by Graph operations
        >>> nodes = graph.get_all_nodes()
        >>> for node in nodes:
        ...     name = node.name
        ...     node_type = node.type
        ...     attr = node.get_attr("some_attr")
        ...     node.set_attr("new_attr", attr_value)
    """

    def __init__(self) -> None:
        """Prevent direct instantiation of Node objects."""
        raise RuntimeError("Node objects should not be created directly.")

    def __del__(self) -> None:
        """Clean up resources."""
        if self._owns_handle:
            graph_lib.GeApiWrapper_GNode_DestroyGNode(self._handle)
            self._handle = None

    def __copy__(self) -> None:
        """Copy is not supported."""
        raise RuntimeError("Node does not support copy")

    def __deepcopy__(self, memodict) -> None:
        """Deep copy is not supported."""
        raise RuntimeError("Node does not support deepcopy")

    @classmethod
    def _create_from(cls, node_handle: ctypes.c_void_p, owns_handle: bool = True) -> 'Node':
        """Create Node object from C++ handle.(internal use only by e.g Graph.get_all_nodes() or TensorHolder.get_producer(),
         do not use this method directly)

        Args:
            node_handle: C++ GNode object handle.
            owns_handle: Whether to own the handle.

        Returns:
            Node object.

        Raises:
            ValueError: If node_handle is None.
        """
        if not node_handle:
            raise ValueError("Node handle cannot be None")
        instance = cls.__new__(cls)
        instance._handle = node_handle
        instance._owns_handle = owns_handle
        return instance

    @property
    def name(self) -> str:
        """Get node name.
        
        Returns:
            Node name.
            
        Raises:
            RuntimeError: If name retrieval fails.
        """
        c_str = graph_lib.GeApiWrapper_GNode_GetName(self._handle)
        if not c_str:
            raise RuntimeError("Failed to get Node name")

        try:
            return ctypes.string_at(c_str).decode('utf-8')
        finally:
            graph_lib.GeApiWrapper_FreeString(c_str)

    @property
    def type(self) -> str:
        """Get node type.

        Returns:
            Node type.

        Raises:
            RuntimeError: If type retrieval fails.
        """
        c_str = graph_lib.GeApiWrapper_GNode_GetType(self._handle)
        if not c_str:
            raise RuntimeError("Failed to get Node type")

        try:
            return ctypes.string_at(c_str).decode('utf-8')
        finally:
            graph_lib.GeApiWrapper_FreeString(c_str)

    def get_in_control_nodes(self) -> List['Node']:
        """Get input control nodes.
        
        Returns:
            List of input control Node objects.
        """
        node_num = ctypes.c_size_t()
        nodes = graph_lib.GeApiWrapper_GNode_GetInControlNodes(self._handle, ctypes.byref(node_num))
        if not nodes:
            return []

        try:
            return [Node._create_from(nodes[i]) for i in range(node_num.value)]
        finally:
            graph_lib.GeApiWrapper_GNode_FreeGNodeArray(nodes)

    def get_in_data_nodes_and_port_indexes(self, in_index: int) -> Tuple['Node', int]:
        """Get input data node and port index.
        
        Args:
            in_index: Input index.
            
        Returns:
            Tuple of (input Node, port index).
            
        Raises:
            TypeError: If in_index is not an integer.
            RuntimeError: If retrieval fails.
        """
        if not isinstance(in_index, int):
            raise TypeError("Input index must be an integer")

        in_node = ctypes.c_void_p()
        index = ctypes.c_int32()

        ret = graph_lib.GeApiWrapper_GNode_GetInDataNodesAndPortIndexes(
            self._handle, in_index, ctypes.byref(in_node), ctypes.byref(index)
        )
        if ret != 0:  # GRAPH_SUCCESS
            raise RuntimeError(f"Failed to get input data node for index {in_index}")

        return Node._create_from(in_node), index.value

    def get_out_data_nodes_and_port_indexes(self, out_index: int) -> List[Tuple['Node', int]]:
        
        """Get output data nodes and port indexes.
        
        Args:
            out_index: Output index.
            
        Returns:
            List of Tuple of (output Node, port index).
            
        Raises:
            TypeError: If out_index is not an integer.
            RuntimeError: If retrieval fails.
        """
        if not isinstance(out_index, int):
            raise TypeError("Output index must be an integer")
        out_nodes = ctypes.POINTER(ctypes.c_void_p)()
        out_indexes = ctypes.POINTER(ctypes.c_int32)()
        size = ctypes.c_int()
        ret = graph_lib.GeApiWrapper_GNode_GetOutDataNodesAndPortIndexes(
            self._handle, out_index, ctypes.byref(out_nodes), ctypes.byref(out_indexes), ctypes.byref(size)
        )
        if ret != 0:  # GRAPH_SUCCESS
            raise RuntimeError(f"Failed to get output data node for index {out_index}")
        try: 
            return [(Node._create_from(out_nodes[i]), out_indexes[i]) for i in range(size.value)]
        finally:
            graph_lib.GeApiWrapper_GNode_FreeGNodeArray(out_nodes)
            graph_lib.GeApiWrapper_GNode_FreeIntArray(out_indexes)

    def get_attr(self, key: str) -> Any:
        """Get node attribute.
        
        Args:
            key: Attribute name.
            
        Returns:
            Attribute value.
            
        Raises:
            TypeError: If key is not a string.
            RuntimeError: If attribute retrieval fails.
        """
        if not isinstance(key, str):
            raise TypeError("Attribute key must be a string")

        attr_value = _AttrValue()
        key_bytes = key.encode('utf-8')

        ret = graph_lib.GeApiWrapper_GNode_GetAttr(self._handle, key_bytes, attr_value._av_ptr)
        if ret != 0:  # GRAPH_SUCCESS
            raise RuntimeError(f"Failed to get attribute '{key}' from Node")

        return attr_value.get_value()

    def set_attr(self, key: str, value: Any) -> None:
        """Set node attribute.
        
        Args:
            key: Attribute name.
            value: Attribute value.
            
        Raises:
            TypeError: If arguments have wrong types.
            RuntimeError: If attribute setting fails.
        """
        if not isinstance(key, str):
            raise TypeError("Attribute key must be a string")

        key_bytes = key.encode('utf-8')
        attr_value = _AttrValue()
        attr_value.set_value(value)
        ret = graph_lib.GeApiWrapper_GNode_SetAttr(self._handle, key_bytes, attr_value._av_ptr)
        if ret != 0:  # GRAPH_SUCCESS
            raise RuntimeError(f"Failed to set attribute '{key}' on Node {self.name}")

    def get_input_attr(self, attr_name: str, input_index: int) -> Any:
        """Get input attribute.

        Args:
            attr_name: Attribute name.
            input_index: Input index.

        Returns:
            Attribute value.

        Raises:
            TypeError: If arguments have wrong types.
            RuntimeError: If attribute retrieval fails.
        """
        if not isinstance(attr_name, str):
            raise TypeError("Attribute name must be a string")
        if not isinstance(input_index, int):
            raise TypeError("Input index must be an integer")

        attr_value = _AttrValue()
        attr_name_bytes = attr_name.encode('utf-8')

        ret = graph_lib.GeApiWrapper_GNode_GetInputAttr(
            self._handle, attr_name_bytes, ctypes.c_uint32(input_index), attr_value._av_ptr
        )
        if ret != 0:  # GRAPH_SUCCESS
            raise RuntimeError(f"Failed to get Node {self.name} input attribute '{attr_name}' for index {input_index}")

        return attr_value.get_value()

    def set_input_attr(self, attr_name: str, input_index: int, value: Any) -> None:
        """Set input attribute.
        
        Args:
            attr_name: Attribute name.
            input_index: Input index.
            value: Attribute value.
            
        Raises:
            TypeError: If arguments have wrong types.
            RuntimeError: If attribute setting fails.
        """
        if not isinstance(attr_name, str):
            raise TypeError("Attribute name must be a string")
        if not isinstance(input_index, int):
            raise TypeError("Input index must be an integer")

        attr_name_bytes = attr_name.encode('utf-8')
        attr_value = _AttrValue()
        attr_value.set_value(value)
        ret = graph_lib.GeApiWrapper_GNode_SetInputAttr(
            self._handle, attr_name_bytes, ctypes.c_uint32(input_index), attr_value._av_ptr
        )
        if ret != 0:  # GRAPH_SUCCESS
            raise RuntimeError(f"Failed to set Node {self.name} input attribute '{attr_name}' for index {input_index}")

    def get_output_attr(self, attr_name: str, output_index: int) -> Any:
        """Get output attribute.
        
        Args:
            attr_name: Attribute name.
            output_index: Output index.
            
        Returns:
            Attribute value.
            
        Raises:
            TypeError: If arguments have wrong types.
            RuntimeError: If attribute retrieval fails.
        """
        if not isinstance(attr_name, str):
            raise TypeError("Attribute name must be a string")
        if not isinstance(output_index, int):
            raise TypeError("Output index must be an integer")

        attr_value = _AttrValue()
        attr_name_bytes = attr_name.encode('utf-8')

        ret = graph_lib.GeApiWrapper_GNode_GetOutputAttr(
            self._handle, attr_name_bytes, ctypes.c_uint32(output_index), attr_value._av_ptr
        )
        if ret != 0:  # GRAPH_SUCCESS
            raise RuntimeError(
                f"Failed to get Node {self.name} output attribute '{attr_name}' for index {output_index}")

        return attr_value.get_value()

    def set_output_attr(self, attr_name: str, output_index: int, value: Any) -> None:
        """Set output attribute.
        
        Args:
            attr_name: Attribute name.
            output_index: Output index.
            value: Attribute value.
            
        Raises:
            TypeError: If arguments have wrong types.
            RuntimeError: If attribute setting fails.
        """
        if not isinstance(attr_name, str):
            raise TypeError("Attribute name must be a string")
        if not isinstance(output_index, int):
            raise TypeError("Output index must be an integer")

        attr_name_bytes = attr_name.encode('utf-8')
        attr_value = _AttrValue()
        attr_value.set_value(value)
        ret = graph_lib.GeApiWrapper_GNode_SetOutputAttr(
            self._handle, attr_name_bytes, ctypes.c_uint32(output_index), attr_value._av_ptr
        )
        if ret != 0:  # GRAPH_SUCCESS
            raise RuntimeError(
                f"Failed to set Node {self.name} output attribute '{attr_name}' for index {output_index}")

    def get_out_control_nodes(self) -> List['Node']:
        """Get output control nodes.
        
        Returns:
            List of output control Node objects.
        """
        node_num = ctypes.c_size_t()
        nodes = graph_lib.GeApiWrapper_GNode_GetOutControlNodes(self._handle, ctypes.byref(node_num))
        if not nodes:
            return []

        try:
            return [Node._create_from(nodes[i]) for i in range(node_num.value)]
        finally:
            graph_lib.GeApiWrapper_GNode_FreeGNodeArray(nodes)

    def get_inputs_size(self) -> int:
        """Get input size
        
        Returns:
            Number of input size
        """
        return graph_lib.GeApiWrapper_GNode_GetInputsSize(self._handle)
        
    def get_outputs_size(self) -> int:
        """Get output size
        
        Returns:
            Number of output size
        """
        return graph_lib.GeApiWrapper_GNode_GetOutputsSize(self._handle)

    def has_attr(self, attr_name: str) -> bool:
        """Has attribute
        
        Args:
            attr_name: Attribute name.
            
        Returns:
            If node has attribute of this name.
        """
        if not isinstance(attr_name, str):
            raise TypeError("attr_name must be a string")

        attr_name_bytes = attr_name.encode("utf-8")
        return  graph_lib.GeApiWrapper_GNode_HasAttr(self._handle, attr_name_bytes)
