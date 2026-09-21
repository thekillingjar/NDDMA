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

"""Graph module for GraphEngine graph operations."""

import ctypes
from enum import Enum
from typing import List, Optional, Any, TYPE_CHECKING
from ge._capi.pygraph_wrapper import graph_lib
from .node import Node
from ._attr import _AttrValue

if TYPE_CHECKING:
    from ge.es.graph_builder import GraphBuilder

class DumpFormat(Enum):
    kOnnx = 0
    kTxt = 1
    kReadable = 2

class Graph:
    """Graph class for GraphEngine graph operations.

    This class provides a Pythonic interface for graph operations
    using the GraphEngine C API.
    
    OWNERSHIP MANAGEMENT:
    Graph objects can be in two ownership states:
    1. Python-owned (default): Python is responsible for releasing C++ resources
    2. C++-owned: C++ side manages the resource, Python is only a snapshot of the graph
    
    When a Graph is passed as a subgraph parameter to operators (e.g., If, While, Case),
    ownership is automatically transferred to the C++ side to avoid double-free issues.

    Example:
        >>> graph = Graph("my_graph")
        >>> nodes = graph.get_all_nodes()
        >>> attr = graph.get_attr("some_attr")
        >>> graph.set_attr("new_attr", attr_value)
    """

    def __init__(self,
                 name: Optional[str] = "graph") -> None:
        """Initialize a Graph.

        Args:
            name: Graph name (optional).

        Raises:
            TypeError: If name is not a string.
            RuntimeError: If graph creation fails.
        """
        # Init 
        self._owns_handle = False
        self._owner = None  
        self._handle = None
        if not isinstance(name, str):
            raise TypeError("Graph name must be a string")

        name_bytes = name.encode('utf-8')
        self._handle = graph_lib.GeApiWrapper_Graph_CreateGraph(name_bytes)
        if not self._handle:
            raise RuntimeError("Failed to create Graph")
        self._owns_handle = True  # Python owns the resource by default
        self._owner = None  # Track the owner if ownership is transferred

    def __del__(self) -> None:
        """Clean up resources only if we own them."""
        # Only destroy the graph if Python owns the handle
        if self._owns_handle:
            graph_lib.GeApiWrapper_Graph_DestroyGraph(self._handle)
            self._handle = None

    def __copy__(self) -> None:
        """Copy is not supported."""
        raise RuntimeError("Graph does not support copy")

    def __deepcopy__(self, memodict) -> None:
        """Deep copy is not supported."""
        raise RuntimeError("Graph does not support deepcopy")

    def __str__(self) -> str:
        """Dump graph to stream in readable format."""
        return self.dump_to_stream(DumpFormat.kReadable)

    @classmethod
    def _create_from(cls, handle: ctypes.c_void_p) -> 'Graph':
        """Create Graph object from C++ pointer.(internal use only by e.g GraphBuilder.build_and_reset(),
         do not use this method directly)

        Args:
            handle: C++ Graph object pointer.

        Returns:
            Graph object.

        Raises:
            ValueError: If pointer is None.
        """
        if not handle:
            raise ValueError("Graph pointer cannot be None")
        instance = cls.__new__(cls)
        instance._handle = handle
        instance._owns_handle = True
        instance._owner = None
        return instance

    def _transfer_ownership_when_pass_as_subgraph(self, new_owner: 'GraphBuilder') -> None:
        """Transfer ownership of the C++ resource to new_owner.
        
        After calling this method, Python will no longer destroy the underlying
        C++ Graph object. This is called automatically when a Graph is passed
        as a subgraph parameter to operators(i.e., If, While, Case inner_graph).
        
        Args:
            new_owner: The object that will manage the C++ resource (typically a GraphBuilder).
                       this Graph will hold a reference to keep the new_owner alive.
           
        Example:
            >>> sub_graph = create_subgraph()
            >>> result = If(..., then_graph=sub_graph, ...)
            >>>  # sub_graph._transfer_ownership_when_pass_as_subgraph(main_builder) is called automatically
            >>>  # sub_graph._owner now holds a reference to main_builder
            >>>  # As long as sub_graph exists, main_builder won't be GC'd
        """
        if self._owner is not None:
            raise RuntimeError("Graph :{} already has an new owner builder :{}, cannot transfer ownership again"
                               .format(self.name, self._owner.name))
        self._owns_handle = False
        self._owner = new_owner  # Keep reference to prevent premature GC

    @property
    def name(self) -> str:
        """Get graph name.

        Returns:
            Graph name.

        Raises:
            RuntimeError: If name retrieval fails.
        """
        c_str = graph_lib.GeApiWrapper_Graph_GetName(self._handle)
        if not c_str:
            raise RuntimeError("Failed to get Graph name")

        try:
            return ctypes.string_at(c_str).decode('utf-8')
        finally:
            graph_lib.GeApiWrapper_FreeString(c_str)

    def get_all_nodes(self) -> List[Node]:
        """Get all nodes in the graph.

        Returns:
            List of Node objects.
        """
        node_num = ctypes.c_size_t()
        nodes = graph_lib.GeApiWrapper_Graph_GetAllNodes(self._handle, ctypes.byref(node_num))
        if not nodes:
            return []

        try:
            return [Node._create_from(nodes[i]) for i in range(node_num.value)]
        finally:
            graph_lib.GeApiWrapper_GNode_FreeGNodeArray(nodes)

    def get_direct_nodes(self) -> List[Node]:
        """Get direct nodes in the graph.

        Returns:
            List of Node objects.
        """
        node_num = ctypes.c_size_t()
        nodes = graph_lib.GeApiWrapper_Graph_GetDirectNode(self._handle, ctypes.byref(node_num))
        if not nodes:
            return []
            
        try:
            return [Node._create_from(nodes[i]) for i in range(node_num.value)]
        finally:
            graph_lib.GeApiWrapper_GNode_FreeGNodeArray(nodes)

    def get_attr(self, key: str) -> Any:
        """Get graph attribute.

        Args:
            key: Attribute name.

        Returns:
            AttrValue object.

        Raises:
            TypeError: If key is not a string.
            RuntimeError: If attribute retrieval fails.
        """
        if not isinstance(key, str):
            raise TypeError("Attribute key must be a string")

        attr_value = _AttrValue()
        key_bytes = key.encode('utf-8')

        ret = graph_lib.GeApiWrapper_Graph_GetAttr(self._handle, key_bytes, attr_value._av_ptr)
        if ret != 0:  # GRAPH_SUCCESS
            raise RuntimeError(f"Failed to get attribute '{key}' from Graph {self.name}")
        return attr_value.get_value()

    def set_attr(self, key: str, value: Any) -> None:
        """Set graph attribute.

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
        ret = graph_lib.GeApiWrapper_Graph_SetAttr(self._handle, key_bytes, attr_value._av_ptr)
        if ret != 0:  # GRAPH_SUCCESS
            raise RuntimeError(f"Failed to set attribute '{key}' on Graph {self.name}")


    def dump_to_file(self, format: DumpFormat = DumpFormat.kReadable, suffix: str = "") -> None:
        """Dump graph to file.

        Args:
            format: Format of the file. Defaults to "kReadable". Can be "kOnnx" or "kTxt" or "kReadable".
            suffix: Suffix to append to the filename. Defaults to empty string. eg: xxxx
            If path and suffix is not empty, the file name will be like: path/ge_<onnx/txt/readable>_00000_graph_0_xxxx.<txt/pbtxt>
        Note:
            pbtxt has only graph structure, no weights data or other attributes.
        Raises:
            TypeError: If format is not in [DumpFormat.kOnnx, DumpFormat.kTxt, DumpFormat.kReadable].
            TypeError: If suffix is not a string.
            RuntimeError: If dump operation fails.
        """

        if not isinstance(suffix, str):
            raise TypeError("Suffix must be a string")

        if format not in [DumpFormat.kOnnx, DumpFormat.kTxt, DumpFormat.kReadable]:
            raise TypeError("Format must be in [DumpFormat.kOnnx, DumpFormat.kTxt, DumpFormat.kReadable]")

        suffix_bytes = suffix.encode('utf-8')
        ret = graph_lib.GeApiWrapper_Graph_Dump_To_File(
            self._handle,
            format.value,
            suffix_bytes
        )
        if ret != 0:
            raise RuntimeError(f"Failed to dump graph: {self.name} to file format {format} with suffix {suffix}")

    def dump_to_stream(self, format: DumpFormat = DumpFormat.kReadable) -> str:
        """Dump graph to stream.

        Args:
            format: Format of the stream. Defaults to "kReadable". Can be "kOnnx" or "kTxt" or "kReadable".

        Returns:
            Stream of the graph.
        """
        if format not in [DumpFormat.kOnnx, DumpFormat.kTxt, DumpFormat.kReadable]:
            raise TypeError("Format must be in [DumpFormat.kOnnx, DumpFormat.kTxt, DumpFormat.kReadable]")

        c_str = graph_lib.GeApiWrapper_Graph_Dump_To_Stream(
            self._handle,
            format.value
        )
        if not c_str:
            raise RuntimeError(f"Failed to dump graph: {self.name} to stream")

        try:
            return ctypes.string_at(c_str).decode('utf-8')
        finally:
            graph_lib.GeApiWrapper_FreeString(c_str)


    def save_to_air(self, file_path: str) -> None:
        """Save graph to AIR format.

        Args:
            path: file path to save the AIR file.
        
        Raises:
            TypeError: If file_path is not a string.
            RuntimeError: If save operation fails.
        """
        if not isinstance(file_path, str):
            raise TypeError("file_path must be a string")
        ret = graph_lib.GeApiWrapper_Graph_SaveToAir(self._handle, file_path.encode('utf-8'))
        if ret != 0:
            raise RuntimeError(f"Failed to save graph to AIR format")

    def load_from_air(self, file_path: str) -> None:
        """Load graph from AIR format.

        Args:
            path: file path to load the AIR file.
        
        Raises:
            TypeError: If file_path is not a string.
            RuntimeError: If load operation fails.
        """
        if not isinstance(file_path, str):
            raise TypeError("file_path must be a string")
        ret = graph_lib.GeApiWrapper_Graph_LoadFromAir(self._handle, file_path.encode('utf-8'))
        if ret != 0:
            raise RuntimeError(f"Failed to load graph from AIR format")

    def remove_node(self, node: Node) -> None:
        """Remove node.

        Args:
            node: node to be removed.
        
        Raises:
            TypeError: If node is not a Node.
            RuntimeError: If removing node operation fails.
        """
        if not isinstance(node, Node):
            raise TypeError("node must be a Node")
        ret = graph_lib.GeApiWrapper_Graph_RemoveNode(
            self._handle, node._handle)
        if ret != 0:  # GRAPH_SUCCESS
            raise RuntimeError(
                f"Failed to remove Node {node.name} from Graph {self.name}")

    def remove_edge(self, src_node: Node, src_port_index: int, dst_node: Node, dst_port_index: int) -> None:
        """Remove edge.

        Args:
            src_node: source node of edge
            src_port_index: source port index of edge. If removing control edge, should be set to -1
            dst_node: destination node of edge
            dst_port_index: destination port index of edge. If removing control edge, should be set to -1
            
        
        Raises:
            TypeError: If src_node, dst_node are not nodes or src_port_index,dst_port_index are not integers.
            RuntimeError: If removing edge operation fails.

        Example:
        >>> remove_edge(src_node, -1, dst_node, -1)
        >>> remove_edge()
        """
        if not isinstance(src_node, Node):
            raise TypeError("src_node must be a Node")
        if not isinstance(src_port_index, int):
            raise TypeError("src_port_index must be an integer")
        if not isinstance(dst_node, Node):
            raise TypeError("dst_node must be a Node")
        if not isinstance(dst_port_index, int):
            raise TypeError("dst_port_index must be an integer")

        ret = graph_lib.GeApiWrapper_Graph_RemoveEdge(
            self._handle, src_node._handle, src_port_index, dst_node._handle, dst_port_index)
        if ret != 0:  # GRAPH_SUCCESS
            raise RuntimeError(
                f"Failed to remove Edge from Node {src_node.name}, Port Index {src_port_index} to Node {dst_node.name}, Port Index {dst_port_index}")

    def add_data_edge(self, src_node: Node, src_port_index: int, dst_node: Node, dst_port_index: int) -> None:
        """Add data edge.

        Args:
            src_node: source node of edge
            src_port_index: source port index of edge. 
            dst_node: destination node of edge
            dst_port_index: destination port index of edge. 
            
        
        Raises:
            TypeError: If src_node, dst_node are not nodes or src_port_index,dst_port_index are not integers.
            RuntimeError: If adding data edge operation fails.
        """
        if not isinstance(src_node, Node):
            raise TypeError("src_node must be a Node")
        if not isinstance(src_port_index, int):
            raise TypeError("src_port_index must be an integer")
        if not isinstance(dst_node, Node):
            raise TypeError("dst_node must be a Node")
        if not isinstance(dst_port_index, int):
            raise TypeError("dst_port_index must be an integer")

        ret = graph_lib.GeApiWrapper_Graph_AddDataEdge(
            self._handle, src_node._handle, src_port_index, dst_node._handle, dst_port_index)
        if ret != 0:  # GRAPH_SUCCESS
            raise RuntimeError(
                f"Failed to add DataEdge from Node {src_node.name}, Port Index {src_port_index} to Node {dst_node.name}, Port Index {dst_port_index}")

    def add_control_edge(self, src_node: Node, dst_node: Node) -> None:
        """Add control edge.

        Args:
            src_node: source node of edge
            dst_node: destination node of edge
            
        
        Raises:
            TypeError: If src_node, dst_node are not nodes .
            RuntimeError: If adding control edge operation fails.
        """
        if not isinstance(src_node, Node):
            raise TypeError("src_node must be a Node")
        if not isinstance(dst_node, Node):
            raise TypeError("dst_node must be a Node")
               
        ret = graph_lib.GeApiWrapper_Graph_AddControlEdge(
            self._handle, src_node._handle, dst_node._handle)
        if ret != 0:  # GRAPH_SUCCESS
            raise RuntimeError(
                f"Failed to add Control from Node {src_node.name} to Node {dst_node.name}")

    def find_node_by_name(self, name: str) -> Node:
        """Find node by name

        Args:
            name: node name.
            
        Returns:
            Node found.

        Raises:
            TypeError: If name is not string.
            RuntimeError: If finding node by name operation fails.
        """
        if not isinstance(name, str):
            raise TypeError("name must be a string")
            
        name_bytes = name.encode('utf-8')
        node = ctypes.c_void_p()
        ret = graph_lib.GeApiWrapper_Graph_FindNodeByName(self._handle, name_bytes, ctypes.byref(node))
        if ret != 0:  # GRAPH_SUCCESS
            raise RuntimeError(
                f"Failed to find Node name {name}")
        return Node._create_from(node)

    def get_all_subgraphs(self) -> List['Graph']:
        """Get all subgraphs in the graph.

        Returns:
            List of Graph objects representing subgraphs.
        """
        subgraph_num = ctypes.c_size_t()
        subgraphs = graph_lib.GeApiWrapper_Graph_GetAllSubgraphs(self._handle, ctypes.byref(subgraph_num))
        if not subgraphs:
            return []

        try:
            return [Graph._create_from(subgraphs[i]) for i in range(subgraph_num.value)]
        finally:
            graph_lib.GeApiWrapper_Graph_FreeGraphArray(subgraphs)

    def get_subgraph(self, name: str) -> Optional['Graph']:
        """Get subgraph by name.

        Args:
            name: Subgraph name.

        Returns:
            Graph object representing the subgraph, or None if not found.

        Raises:
            TypeError: If name is not a string.
            RuntimeError: If getting subgraph fails.
        """
        if not isinstance(name, str):
            raise TypeError("name must be a string")

        name_bytes = name.encode('utf-8')
        subgraph_handle = graph_lib.GeApiWrapper_Graph_GetSubGraph(self._handle, name_bytes)
        if not subgraph_handle:
            return None

        return Graph._create_from(subgraph_handle)

    def add_subgraph(self, subgraph: 'Graph') -> None:
        """Add a subgraph to the graph.

        The subgraph is indexed by its name. Subgraph names must be unique within the parent graph;
        attempting to add a subgraph whose name already exists will fail.

        Args:
            subgraph: Graph object to be added as a subgraph.

        Raises:
            TypeError: If subgraph is not a Graph.
            RuntimeError: If adding subgraph fails.
        """
        if not isinstance(subgraph, Graph):
            raise TypeError("subgraph must be a Graph")

        ret = graph_lib.GeApiWrapper_Graph_AddSubGraph(self._handle, subgraph._handle)
        if ret != 0:  # GRAPH_SUCCESS
            raise RuntimeError(
                f"Failed to add subgraph '{subgraph.name}' to graph '{self.name}'")

    def remove_subgraph(self, name: str) -> None:
        """Remove subgraph by name.

        Args:
            name: Subgraph name.

        Raises:
            TypeError: If name is not a string.
            RuntimeError: If removing subgraph fails.
        """
        if not isinstance(name, str):
            raise TypeError("name must be a string")

        name_bytes = name.encode('utf-8')
        ret = graph_lib.GeApiWrapper_Graph_RemoveSubgraph(self._handle, name_bytes)
        if ret != 0:  # GRAPH_SUCCESS
            raise RuntimeError(
                f"Failed to remove subgraph '{name}' from graph '{self.name}'")
