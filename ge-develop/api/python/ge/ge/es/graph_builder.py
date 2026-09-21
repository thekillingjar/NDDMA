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

"""GraphBuilder module for eager-style graph construction."""

import ctypes
import contextlib
import threading
from enum import Enum
from typing import List, Optional, Union
from ge._capi.pyes_graph_builder_wrapper import (
    esb_lib,
    EsCGraphBuilderPtr,
    EsCTensorHolderPtr,
    c_int64,
    c_float,
    c_int32,
    c_uint64,
    c_uint32,
    c_bool,
    c_char_p
)
from ge.graph.types import DataType, Format
from ge.graph import Graph
from .tensor_holder import TensorHolder

# use thread local storage to manage attribute scope
_local = threading.local()
slot_name_control_dependency_nodes = "control_dependency_nodes"
slot_name_custom_node_attrs = "custom_node_attrs"


class InputType(Enum):
    DATA = "Data"
    REF_DATA = "RefData"
    AIPP_DATA = "AippData"
    ANY_DATA = "AnyData"


class GraphBuilder:
    """GraphBuilder for eager-style graph construction.
    
    This class provides a Pythonic interface for building computation graphs
    using the eager-style graph builder C API.

    IMPORTANT LIFECYCLE NOTES:
    **Keep builder alive**: All TensorHolder objects created by this builder
       maintain a reference to it. The builder will not be garbage collected
       as long as any of its tensors are still referenced.
    
    
    Example:
        >>> builder = GraphBuilder("my_graph")
        >>> input_tensor = builder.create_input(0)
        >>> const_tensor = builder.create_const_float(1.0)
        >>> builder.set_graph_output(input_tensor, 0)
        >>> graph = builder.build_and_reset()
        >>> builder.create_const_float(2.0)  # Would raise error
    """

    def __init__(self, name: Optional[str] = None) -> None:
        """Initialize a GraphBuilder.
        
        Args:
            name: Graph name. If None, defaults to "graph".
            
        Raises:
            TypeError: If name is not a string or None.
            RuntimeError: If graph builder creation fails.
        """
        self._handle = None
        # Never store strong references to TensorHolder in GraphBuilder, now or in the future,
        # to avoid reference cycles.
        if name is not None and not isinstance(name, str):
            raise TypeError("Graph name must be a string")

        name_bytes = name.encode('utf-8') if name else "graph".encode('utf-8')
        self._name = name_bytes.decode('utf-8')
        self._handle = esb_lib.EsCreateGraphBuilder(name_bytes)
        if not self._handle:
            raise RuntimeError("Failed to create graph builder")
        self._is_built = False

    def _check_usable(self, operation: str) -> None:
        """Check if the graph builder is usable"""
        if self._is_built:
            raise RuntimeError(f"Cannot {operation}: GraphBuilder has already been built. "
                               "Create a new GraphBuilder to build another graph.")

    def __del__(self) -> None:
        """Destroy the graph builder."""
        esb_lib.EsDestroyGraphBuilder(self._handle)
        self._handle = None

    def __copy__(self):
        """Copy not supported."""
        raise RuntimeError("GraphBuilder does not support copy")

    def __deepcopy__(self, memodict):
        """Deepcopy not supported."""
        raise RuntimeError("GraphBuilder does not support deepcopy")

    @property
    def name(self) -> str:
        """Get graph builder name.
        
        Returns:
            Graph builder name.
        """
        return self._name

    def _get_current_attrs(self):
        """Get current scope attributes"""
        return getattr(_local, slot_name_custom_node_attrs, {})

    def _apply_attrs_to_node(self, tensor_holder: TensorHolder) -> TensorHolder:
        """Apply current scope attributes to tensor's producer node"""
        attrs = self._get_current_attrs()
        if attrs:
            producer_node = tensor_holder._get_node_snapshot()

            for attr_name, attr_value in attrs.items():
                # set attribute to node
                producer_node.set_attr(attr_name, attr_value)

        return tensor_holder

    def _get_current_control_dependency_nodes(self):
        """Get current scope control dependency nodes"""
        return getattr(_local, slot_name_control_dependency_nodes, [])

    def _apply_control_dependencies_to_node(self, tensor_holder: TensorHolder) -> TensorHolder:
        """Apply current scope control dependency nodes to tensor's producer node"""
        control_dependency_nodes = self._get_current_control_dependency_nodes()
        if control_dependency_nodes:
            self.add_control_dependency(tensor_holder, control_dependency_nodes)

        return tensor_holder

    def _apply_scope_infos_to_node(self, tensor_holder: TensorHolder) -> TensorHolder:
        """Apply current scope attributes and control dependency nodes to tensor's producer node"""
        self._apply_attrs_to_node(tensor_holder)
        self._apply_control_dependencies_to_node(tensor_holder)
        return tensor_holder

    @staticmethod
    def _validate_const_shape(values: List, dims: List[int]) -> None:
        """Validate that the number of values matches the shape dimensions.
        
        Args:
            values: List of values.
            dims: Shape dimensions (must be non-empty, scalar case handled separately).
            
        Raises:
            ValueError: If the number of values doesn't match the shape.
        """
        expected_count = 1
        for dim in dims:
            expected_count *= dim
        if len(values) != expected_count:
            raise ValueError(
                f"Value count ({len(values)}) doesn't match shape {dims} "
                f"(expected {expected_count} elements)"
            )

    def create_input(self, index: int, *, name: Optional[str] = None,
                     type_str: Optional[InputType] = InputType.DATA, data_type: Optional[DataType] = DataType.DT_FLOAT,
                     format: Optional[Format] = Format.FORMAT_ND, shape: Optional[List[int]] = None) -> TensorHolder:
        """Create a graph input.
        
        Args:
            index: Input index, means the index of the input in the graph.
            name: Input name. If None, defaults to "input_{index}".
            type_str: Input type using InputType enum, defaults to InputType.DATA, means the type of the input is Data.
            data_type: Data type using DataType enum, defaults to DataType.DT_FLOAT.
            format: Data format using Format enum, defaults to Format.FORMAT_ND.
            shape: List of shape dimensions, If None, means scalar.
            
        Returns:
            TensorHolder representing the input.
            
        Raises:
            TypeError: If arguments have incorrect types.
            RuntimeError: If input creation fails.
        """
        self._check_usable("create input")

        if not isinstance(index, int):
            raise TypeError("Input index must be an integer")

        if shape is not None:
            if not isinstance(shape, list) or not all(isinstance(dim, int) for dim in shape):
                raise TypeError("Shape must be a list of integers")

        # use detailed input creation
        name = name or f"input_{index}"
        name_bytes = name.encode('utf-8')
        type_bytes = type_str.value.encode('utf-8')
        dt = data_type.value
        fmt = format.value

        dim_num = len(shape) if shape is not None else 0
        shape_ptr = (c_int64 * dim_num)(*shape) if shape is not None else None

        tensor_handle = esb_lib.EsCreateGraphInputWithDetails(
            self._handle, c_int64(index), name_bytes, type_bytes,
            ctypes.c_int(dt), ctypes.c_int(fmt), shape_ptr, c_int64(dim_num)
        )

        if not tensor_handle:
            raise RuntimeError("Failed to create input")

        tensor_holder = TensorHolder._create_from(tensor_handle, self)
        self._apply_scope_infos_to_node(tensor_holder)
        return tensor_holder

    def create_inputs(self, num: int, start_index: int = 0) -> List[TensorHolder]:
        """Create multiple inputs.
        
        Args:
            num: Number of inputs to create.
            start_index: Start index of the inputs, if not 0 means other inputs have been created, 
             the overall input node index should start from 0 and be continuous increment.

        Returns:
            List of TensorHolder representing the inputs. 
            TensorHolder is DataType.DT_FLOAT and Format.FORMAT_ND and shape is [].

        Raises:
            TypeError: If arguments have incorrect types.
            RuntimeError: If input creation fails.
        """
        self._check_usable("create inputs")

        if not isinstance(num, int) or num <= 0:
            raise TypeError("Number of inputs must be a positive integer")

        if not isinstance(start_index, int) or start_index < 0:
            raise TypeError("Start index must be a non-negative integer")

        return [self.create_input(i) for i in range(start_index, start_index + num)]

    def create_const_int64(self, value: Union[int, List[int]], shape: Optional[List[int]] = None) -> TensorHolder:
        """Create int64 constant tensor.
        
        Args:
            value: Single integer or list of integers. If list, the number of elements
                   must match the product of shape dimensions when shape is provided.
            shape: Shape dimensions. If None: for single integer creates scalar (shape=[]),
                   for list creates 1D tensor (shape=[len(value)]). When provided, the product
                   of dimensions must equal len(value) if value is list, or [] if value is int.
            
        Returns:
            TensorHolder representing the constant.
            
        Raises:
            TypeError: If value is not int or list of ints.
            ValueError: If value count doesn't match shape dimensions.
            RuntimeError: If constant creation fails.
        """
        self._check_usable("create constant")

        if isinstance(value, int):
            values = [value]
            dims = [] if shape is None else shape
        elif isinstance(value, list):
            if not all(isinstance(v, int) for v in value):
                raise TypeError("Value must be an integer or list of integers")
            values = value
            dims = shape if shape is not None else [len(value)]
        else:
            raise TypeError("Value must be an integer or list of integers")

        # If dims is empty and there's only one value, use scalar creation for consistency
        if len(dims) == 0 and len(values) == 1:
            return self.create_scalar_int64(values[0])

        self._validate_const_shape(values, dims)
        c_values = (c_int64 * len(values))(*values)
        c_dims = (c_int64 * len(dims))(*dims)

        tensor_handle = esb_lib.EsCreateConstInt64(
            self._handle, c_values, c_dims, c_int64(len(dims))
        )

        if not tensor_handle:
            raise RuntimeError("Failed to create int64 constant")

        tensor_holder = TensorHolder._create_from(tensor_handle, self)
        return self._apply_scope_infos_to_node(tensor_holder)

    def create_const_float(self, value: Union[float, List[float]], shape: Optional[List[int]] = None) -> TensorHolder:
        """Create float constant tensor.
        
        Args:
            value: Single float or list of floats. If list, the number of elements
                   must match the product of shape dimensions when shape is provided.
            shape: Shape dimensions. If None: for single float creates scalar (shape=[]),
                   for list creates 1D tensor (shape=[len(value)]). When provided, the product
                   of dimensions must equal len(value) if value is list, or [] if value is float.
            
        Returns:
            TensorHolder representing the constant.
            
        Raises:
            TypeError: If value is not float or list of floats.
            ValueError: If value count doesn't match shape dimensions.
            RuntimeError: If constant creation fails.
        """
        self._check_usable("create constant")

        if isinstance(value, (int, float)):
            values = [float(value)]
            dims = [] if shape is None else shape
        elif isinstance(value, list):
            if not all(isinstance(v, (int, float)) for v in value):
                raise TypeError("Value must be a float or list of floats")
            values = [float(v) for v in value]
            dims = shape if shape is not None else [len(value)]
        else:
            raise TypeError("Value must be a float or list of floats")

        # If dims is empty and there's only one value, use scalar creation for consistency
        if len(dims) == 0 and len(values) == 1:
            return self.create_scalar_float(values[0])

        self._validate_const_shape(values, dims)
        c_values = (c_float * len(values))(*values)
        c_dims = (c_int64 * len(dims))(*dims)

        tensor_handle = esb_lib.EsCreateConstFloat(
            self._handle, c_values, c_dims, c_int64(len(dims))
        )

        if not tensor_handle:
            raise RuntimeError("Failed to create float constant")

        tensor_holder = TensorHolder._create_from(tensor_handle, self)
        return self._apply_scope_infos_to_node(tensor_holder)

    def create_const_uint64(self, value: Union[int, List[int]], shape: Optional[List[int]] = None) -> TensorHolder:
        """Create uint64 constant tensor.
        
        Args:
            value: Single integer or list of integers. If list, the number of elements
                   must match the product of shape dimensions when shape is provided.
            shape: Shape dimensions. If None: for single integer creates scalar (shape=[]),
                   for list creates 1D tensor (shape=[len(value)]). When provided, the product
                   of dimensions must equal len(value) if value is list, or [] if value is int.
            
        Returns:
            TensorHolder representing the constant.
            
        Raises:
            TypeError: If value is not int or list of ints.
            ValueError: If value count doesn't match shape dimensions.
            RuntimeError: If constant creation fails.
        """
        self._check_usable("create constant")

        if isinstance(value, int):
            values = [value]
            dims = [] if shape is None else shape
        elif isinstance(value, list):
            if not all(isinstance(v, int) for v in value):
                raise TypeError("Value must be an integer or list of integers")
            values = value
            dims = shape if shape is not None else [len(value)]
        else:
            raise TypeError("Value must be an integer or list of integers")

        # If dims is empty and there's only one value, use scalar creation for consistency
        if len(dims) == 0 and len(values) == 1:
            return self.create_scalar_uint64(values[0])

        self._validate_const_shape(values, dims)
        c_values = (c_uint64 * len(values))(*values)
        c_dims = (c_int64 * len(dims))(*dims)

        tensor_handle = esb_lib.EsCreateConstUInt64(
            self._handle, c_values, c_dims, c_int64(len(dims))
        )

        if not tensor_handle:
            raise RuntimeError("Failed to create uint64 constant")

        tensor_holder = TensorHolder._create_from(tensor_handle, self)
        return self._apply_scope_infos_to_node(tensor_holder)

    def create_const_int32(self, value: Union[int, List[int]], shape: Optional[List[int]] = None) -> TensorHolder:
        """Create int32 constant tensor.
        
        Args:
            value: Single integer or list of integers. If list, the number of elements
                   must match the product of shape dimensions when shape is provided.
            shape: Shape dimensions. If None: for single integer creates scalar (shape=[]),
                   for list creates 1D tensor (shape=[len(value)]). When provided, the product
                   of dimensions must equal len(value) if value is list, or [] if value is int.
            
        Returns:
            TensorHolder representing the constant.
            
        Raises:
            TypeError: If value is not int or list of ints.
            ValueError: If value count doesn't match shape dimensions.
            RuntimeError: If constant creation fails.
        """
        self._check_usable("create constant")

        if isinstance(value, int):
            values = [value]
            dims = [] if shape is None else shape
        elif isinstance(value, list):
            if not all(isinstance(v, int) for v in value):
                raise TypeError("Value must be an integer or list of integers")
            values = value
            dims = shape if shape is not None else [len(value)]
        else:
            raise TypeError("Value must be an integer or list of integers")

        # If dims is empty and there's only one value, use scalar creation for consistency
        if len(dims) == 0 and len(values) == 1:
            return self.create_scalar_int32(values[0])

        self._validate_const_shape(values, dims)
        c_values = (c_int32 * len(values))(*values)
        c_dims = (c_int64 * len(dims))(*dims)

        tensor_handle = esb_lib.EsCreateConstInt32(
            self._handle, c_values, c_dims, c_int64(len(dims))
        )

        if not tensor_handle:
            raise RuntimeError("Failed to create int32 constant")

        tensor_holder = TensorHolder._create_from(tensor_handle, self)
        return self._apply_scope_infos_to_node(tensor_holder)

    def create_const_uint32(self, value: Union[int, List[int]], shape: Optional[List[int]] = None) -> TensorHolder:
        """Create uint32 constant tensor.
        
        Args:
            value: Single integer or list of integers. If list, the number of elements
                   must match the product of shape dimensions when shape is provided.
            shape: Shape dimensions. If None: for single integer creates scalar (shape=[]),
                   for list creates 1D tensor (shape=[len(value)]). When provided, the product
                   of dimensions must equal len(value) if value is list, or [] if value is int.
            
        Returns:
            TensorHolder representing the constant.
            
        Raises:
            TypeError: If value is not int or list of ints.
            ValueError: If value count doesn't match shape dimensions.
            RuntimeError: If constant creation fails.
        """
        self._check_usable("create constant")

        if isinstance(value, int):
            values = [value]
            dims = [] if shape is None else shape
        elif isinstance(value, list):
            if not all(isinstance(v, int) for v in value):
                raise TypeError("Value must be an integer or list of integers")
            values = value
            dims = shape if shape is not None else [len(value)]
        else:
            raise TypeError("Value must be an integer or list of integers")

        # If dims is empty and there's only one value, use scalar creation for consistency
        if len(dims) == 0 and len(values) == 1:
            return self.create_scalar_uint32(values[0])

        self._validate_const_shape(values, dims)
        c_values = (c_uint32 * len(values))(*values)
        c_dims = (c_int64 * len(dims))(*dims)

        tensor_handle = esb_lib.EsCreateConstUInt32(
            self._handle, c_values, c_dims, c_int64(len(dims))
        )

        if not tensor_handle:
            raise RuntimeError("Failed to create uint32 constant")

        tensor_holder = TensorHolder._create_from(tensor_handle, self)
        return self._apply_scope_infos_to_node(tensor_holder)

    def create_vector_int64(self, value: List[int]) -> TensorHolder:
        """Create int64 vector tensor.
        
        Args:
            value: List of integers.
            
        Returns:
            TensorHolder representing the vector.
            
        Raises:
            TypeError: If value is not a list of ints.
            RuntimeError: If vector creation fails.
        """
        self._check_usable("create vector")

        if not isinstance(value, list) or not all(isinstance(v, int) for v in value):
            raise TypeError("Value must be a list of integers")

        c_values = (c_int64 * len(value))(*value)

        tensor_handle = esb_lib.EsCreateVectorInt64(
            self._handle, c_values, c_int64(len(value))
        )

        if not tensor_handle:
            raise RuntimeError("Failed to create int64 vector")

        tensor_holder = TensorHolder._create_from(tensor_handle, self)
        return self._apply_scope_infos_to_node(tensor_holder)

    def create_scalar_int64(self, value: int) -> TensorHolder:
        """Create int64 scalar tensor.
        
        Args:
            value: Integer value.
            
        Returns:
            TensorHolder representing the scalar.
            
        Raises:
            TypeError: If value is not an integer.
            RuntimeError: If scalar creation fails.
        """
        self._check_usable("create scalar")

        if not isinstance(value, int):
            raise TypeError("Value must be an integer")

        tensor_handle = esb_lib.EsCreateScalarInt64(self._handle, c_int64(value))

        if not tensor_handle:
            raise RuntimeError("Failed to create int64 scalar")

        tensor_holder = TensorHolder._create_from(tensor_handle, self)
        return self._apply_scope_infos_to_node(tensor_holder)

    def create_scalar_int32(self, value: int) -> TensorHolder:
        """Create int32 scalar tensor.
        
        Args:
            value: Integer value.
            
        Returns:
            TensorHolder representing the scalar.
            
        Raises:
            TypeError: If value is not an integer.
            RuntimeError: If scalar creation fails.
        """
        self._check_usable("create scalar")

        if not isinstance(value, int):
            raise TypeError("Value must be an integer")

        tensor_handle = esb_lib.EsCreateScalarInt32(self._handle, c_int32(value))

        if not tensor_handle:
            raise RuntimeError("Failed to create int32 scalar")

        tensor_holder = TensorHolder._create_from(tensor_handle, self)
        return self._apply_scope_infos_to_node(tensor_holder)

    def create_scalar_float(self, value: float) -> TensorHolder:
        """Create float scalar tensor.
        
        Args:
            value: Float value.
            
        Returns:
            TensorHolder representing the scalar.
            
        Raises:
            TypeError: If value is not a number.
            RuntimeError: If scalar creation fails.
        """
        self._check_usable("create scalar")

        if not isinstance(value, (int, float)):
            raise TypeError("Value must be a number")

        tensor_handle = esb_lib.EsCreateScalarFloat(self._handle, c_float(float(value)))

        if not tensor_handle:
            raise RuntimeError("Failed to create float scalar")

        tensor_holder = TensorHolder._create_from(tensor_handle, self)
        return self._apply_scope_infos_to_node(tensor_holder)

    def create_scalar_uint64(self, value: int) -> TensorHolder:
        """Create uint64 scalar tensor.
        
        Args:
            value: Integer value (must be non-negative).
            
        Returns:
            TensorHolder representing the scalar.
            
        Raises:
            TypeError: If value is not an integer.
            RuntimeError: If scalar creation fails.
        """
        self._check_usable("create scalar")

        if not isinstance(value, int):
            raise TypeError("Value must be an integer")
        if value < 0:
            raise ValueError("Value must be non-negative for uint64")

        tensor_handle = esb_lib.EsCreateScalarUInt64(self._handle, c_uint64(value))

        if not tensor_handle:
            raise RuntimeError("Failed to create uint64 scalar")

        tensor_holder = TensorHolder._create_from(tensor_handle, self)
        return self._apply_scope_infos_to_node(tensor_holder)

    def create_scalar_uint32(self, value: int) -> TensorHolder:
        """Create uint32 scalar tensor.
        
        Args:
            value: Integer value (must be non-negative and fit in uint32 range).
            
        Returns:
            TensorHolder representing the scalar.
            
        Raises:
            TypeError: If value is not an integer.
            RuntimeError: If scalar creation fails.
        """
        self._check_usable("create scalar")

        if not isinstance(value, int):
            raise TypeError("Value must be an integer")
        if value < 0 or value > 0xFFFFFFFF:
            raise ValueError("Value must be in range [0, 2^32-1] for uint32")

        tensor_handle = esb_lib.EsCreateScalarUInt32(self._handle, c_uint32(value))

        if not tensor_handle:
            raise RuntimeError("Failed to create uint32 scalar")

        tensor_holder = TensorHolder._create_from(tensor_handle, self)
        return self._apply_scope_infos_to_node(tensor_holder)

    def create_variable(self, index: int, name: str) -> TensorHolder:
        """Create a variable tensor.
        
        Args:
            index: Variable index.
            name: Variable name.
            
        Returns:
            TensorHolder representing the variable.
            
        Raises:
            TypeError: If arguments have incorrect types.
            RuntimeError: If variable creation fails.
        """
        self._check_usable("create variable")

        if not isinstance(index, int):
            raise TypeError("Index must be an integer")
        if not isinstance(name, str):
            raise TypeError("Name must be a string")

        tensor_handle = esb_lib.EsCreateVariable(
            self._handle, c_int32(index), name.encode('utf-8')
        )

        if not tensor_handle:
            raise RuntimeError("Failed to create variable")

        tensor_holder = TensorHolder._create_from(tensor_handle, self)
        return self._apply_scope_infos_to_node(tensor_holder)

    def set_graph_output(self, tensor: TensorHolder, output_index: int) -> None:
        """Set graph output.
        
        Args:
            tensor: TensorHolder to set as output.
            output_index: Output index.
            
        Raises:
            TypeError: If arguments have incorrect types.
        """
        self._check_usable("set graph output")

        if not isinstance(tensor, TensorHolder):
            raise TypeError("Tensor must be a TensorHolder")
        if not isinstance(output_index, int):
            raise TypeError("Output index must be an integer")

        if esb_lib.EsSetGraphOutput(tensor._handle, c_int64(output_index)) != 0:
            raise RuntimeError(f"Failed to set graph output for graph {self.name} output index {output_index}")

    def set_graph_attr_int64(self, attr_name: str, value: int) -> None:
        """Set int64 attribute for graph.
        
        Args:
            attr_name: Attribute name.
            value: Integer value.
            
        Raises:
            TypeError: If arguments have incorrect types.
        """
        self._check_usable("set graph attribute")

        if not isinstance(attr_name, str):
            raise TypeError("Attribute name must be a string")
        if not isinstance(value, int):
            raise TypeError("Value must be an integer")

        if esb_lib.EsSetInt64AttrForGraph(
                self._handle, attr_name.encode('utf-8'), c_int64(value)
        ) != 0:
            raise RuntimeError(f"Failed to set graph attribute {attr_name} for graph {self.name}")

    def set_graph_attr_string(self, attr_name: str, value: str) -> None:
        """Set string attribute for graph.
        
        Args:
            attr_name: Attribute name.
            value: String value.
            
        Raises:
            TypeError: If arguments have incorrect types.
        """
        self._check_usable("set graph attribute")

        if not isinstance(attr_name, str):
            raise TypeError("Attribute name must be a string")
        if not isinstance(value, str):
            raise TypeError("Value must be a string")

        if esb_lib.EsSetStringAttrForGraph(
                self._handle, attr_name.encode('utf-8'), value.encode('utf-8')
        ) != 0:
            raise RuntimeError(f"Failed to set graph attribute {attr_name} for graph {self.name}")

    def set_graph_attr_bool(self, attr_name: str, value: bool) -> None:
        """Set bool attribute for graph.
        
        Args:
            attr_name: Attribute name.
            value: Boolean value.
            
        Raises:
            TypeError: If arguments have incorrect types.
        """
        self._check_usable("set graph attribute")

        if not isinstance(attr_name, str):
            raise TypeError("Attribute name must be a string")
        if not isinstance(value, bool):
            raise TypeError("Value must be a boolean")

        if esb_lib.EsSetBoolAttrForGraph(
                self._handle, attr_name.encode('utf-8'), c_bool(value)
        ) != 0:
            raise RuntimeError(f"Failed to set graph attribute {attr_name} for graph {self.name}")

    def set_tensor_attr_int64(self, tensor: TensorHolder, attr_name: str, value: int) -> None:
        """Set int64 attribute for tensor.
        
        Args:
            tensor: TensorHolder object.
            attr_name: Attribute name.
            value: Integer value.
            
        Raises:
            TypeError: If arguments have incorrect types.
        """
        self._check_usable("set tensor attribute")

        if not isinstance(tensor, TensorHolder):
            raise TypeError("Tensor must be a TensorHolder")
        if not isinstance(attr_name, str):
            raise TypeError("Attribute name must be a string")
        if not isinstance(value, int):
            raise TypeError("Value must be an integer")

        if esb_lib.EsSetInt64AttrForTensor(
                tensor._handle, attr_name.encode('utf-8'), c_int64(value)
        ) != 0:
            raise RuntimeError(f"Failed to set tensor attribute {attr_name} for tensor {tensor.name}")

    def set_tensor_attr_string(self, tensor: TensorHolder, attr_name: str, value: str) -> None:
        """Set string attribute for tensor.
        
        Args:
            tensor: TensorHolder object.
            attr_name: Attribute name.
            value: String value.
            
        Raises:
            TypeError: If arguments have incorrect types.
        """
        self._check_usable("set tensor attribute")

        if not isinstance(tensor, TensorHolder):
            raise TypeError("Tensor must be a TensorHolder")
        if not isinstance(attr_name, str):
            raise TypeError("Attribute name must be a string")
        if not isinstance(value, str):
            raise TypeError("Value must be a string")

        if esb_lib.EsSetStringAttrForTensor(
                tensor._handle, attr_name.encode('utf-8'), value.encode('utf-8')
        ) != 0:
            raise RuntimeError(f"Failed to set tensor attribute {attr_name} for tensor {tensor.name}")

    def set_tensor_attr_bool(self, tensor: TensorHolder, attr_name: str, value: bool) -> None:
        """Set bool attribute for tensor.
        
        Args:
            tensor: TensorHolder object.
            attr_name: Attribute name.
            value: Boolean value.
            
        Returns:
            Operation result status code.
            
        Raises:
            TypeError: If arguments have incorrect types.
        """
        self._check_usable("set tensor attribute")

        if not isinstance(tensor, TensorHolder):
            raise TypeError("Tensor must be a TensorHolder")
        if not isinstance(attr_name, str):
            raise TypeError("Attribute name must be a string")
        if not isinstance(value, bool):
            raise TypeError("Value must be a boolean")

        if esb_lib.EsSetBoolAttrForTensor(
                tensor._handle, attr_name.encode('utf-8'), c_bool(value)
        ) != 0:
            raise RuntimeError(f"Failed to set tensor attribute {attr_name} for tensor {tensor.name}")

    def set_node_attr_int64(self, tensor: TensorHolder, attr_name: str, value: int) -> None:
        """Set int64 attribute for node.
        
        Args:
            tensor: TensorHolder object.
            attr_name: Attribute name.
            value: Integer value.   
            
        Raises:
            TypeError: If arguments have incorrect types.
        """
        self._check_usable("set node attribute")

        if not isinstance(tensor, TensorHolder):
            raise TypeError("Tensor must be a TensorHolder")
        if not isinstance(attr_name, str):
            raise TypeError("Attribute name must be a string")
        if not isinstance(value, int):
            raise TypeError("Value must be an integer")

        if esb_lib.EsSetInt64AttrForNode(
                tensor._handle, attr_name.encode('utf-8'), c_int64(value)
        ) != 0:
            raise RuntimeError(f"Failed to set node attribute {attr_name} for node {tensor.name}")

    def set_node_attr_string(self, tensor: TensorHolder, attr_name: str, value: str) -> None:
        """Set string attribute for node.
        
        Args:
            tensor: TensorHolder object.
            attr_name: Attribute name.
            value: String value.
            
        Raises:
            TypeError: If arguments have incorrect types.
        """
        self._check_usable("set node attribute")

        if not isinstance(tensor, TensorHolder):
            raise TypeError("Tensor must be a TensorHolder")
        if not isinstance(attr_name, str):
            raise TypeError("Attribute name must be a string")
        if not isinstance(value, str):
            raise TypeError("Value must be a string")

        if esb_lib.EsSetStringAttrForNode(
                tensor._handle, attr_name.encode('utf-8'), value.encode('utf-8')
        ) != 0:
            raise RuntimeError(f"Failed to set node attribute {attr_name} for node {tensor.name}")

    def set_node_attr_bool(self, tensor: TensorHolder, attr_name: str, value: bool) -> None:
        """Set bool attribute for node.
        
        Args:
            tensor: TensorHolder object.
            attr_name: Attribute name.
            value: Boolean value.
            
        Raises:
            TypeError: If arguments have incorrect types.
        """
        self._check_usable("set node attribute")

        if not isinstance(tensor, TensorHolder):
            raise TypeError("Tensor must be a TensorHolder")
        if not isinstance(attr_name, str):
            raise TypeError("Attribute name must be a string")
        if not isinstance(value, bool):
            raise TypeError("Value must be a boolean")

        if esb_lib.EsSetBoolAttrForNode(
                tensor._handle, attr_name.encode('utf-8'), c_bool(value)
        ) != 0:
            raise RuntimeError(f"Failed to set node attribute {attr_name} for node {tensor.name}")

    def add_control_dependency(self, dst_tensor: TensorHolder, src_tensors: List[TensorHolder]) -> None:
        """Add control dependency from src_tensors to dst_tensor.

        Args:
            dst_tensor: TensorHolder to add control dependency to.
            src_tensors: List of TensorHolder to add control dependency from.
        
        Raises:
            TypeError: If arguments have incorrect types.
        """
        self._check_usable("add control dependency")

        if not isinstance(dst_tensor, TensorHolder):
            raise TypeError("dst_tensor must be a TensorHolder")
        if not isinstance(src_tensors, list):
            raise TypeError("src_tensors must be a list")
        if not all(isinstance(tensor, TensorHolder) for tensor in src_tensors):
            raise TypeError("src_tensors must be a list of TensorHolder")
        raw_src_tensors = [tensor._handle for tensor in src_tensors]
        raw_src_array = (EsCTensorHolderPtr * len(raw_src_tensors))(*raw_src_tensors)
        raw_dst_tensor = dst_tensor._handle
        if esb_lib.EsAddControlEdge(raw_dst_tensor, raw_src_array, len(raw_src_tensors)) != 0:
            raise RuntimeError(f"Failed to add control dependency for nodes in graph {self.name}")

    def build_and_reset(self, outputs: Optional[List[TensorHolder]] = None) -> Graph:
        """Build the graph.
           After calling build_and_reset(), the builder enters a built state
           and cannot be used to create new tensors. Create a new GraphBuilder
           for building another graph.
        Args:
            outputs: Optional list of TensorHolder objects to set as graph outputs.
                    If provided, automatically sets all outputs before building.
                    Output indices are assigned sequentially starting from 0.
                    If None (default), builds the graph with previously set outputs.
        Returns:
            Graph object representing the built graph.
            
        Raises:
            TypeError: If outputs is not a list of TensorHolder objects.
            RuntimeError: If graph building fails.
        """
        self._check_usable("build_and_reset")
        if outputs is not None:
            if not isinstance(outputs, list):
                raise TypeError("Outputs must be a list")
            if not all(isinstance(tensor, TensorHolder) for tensor in outputs):
                raise TypeError("All outputs must be TensorHolder objects")
            for i, tensor in enumerate(outputs):
                self.set_graph_output(tensor, i)
        graph_ptr = esb_lib.EsBuildGraphAndReset(self._handle)
        if not graph_ptr:
            raise RuntimeError("Failed to build graph")
        self._is_built = True
        return Graph._create_from(ctypes.c_void_p(graph_ptr))


@contextlib.contextmanager
def attr_scope(attr_maps):
    """Attribute scope context manager"""
    current_attrs = getattr(_local, slot_name_custom_node_attrs, {})
    new_attrs = {**current_attrs, **attr_maps}
    try:
        setattr(_local, slot_name_custom_node_attrs, new_attrs)
        yield
    finally:
        setattr(_local, slot_name_custom_node_attrs, current_attrs)


@contextlib.contextmanager
def control_dependency_scope(tensors: List[TensorHolder]) -> None:
    """Control dependency scope context manager"""
    current_control_dependency_nodes = getattr(_local, slot_name_control_dependency_nodes, [])
    new_control_dependency_nodes = current_control_dependency_nodes + tensors
    try:
        setattr(_local, slot_name_control_dependency_nodes, new_control_dependency_nodes)
        yield
    finally:
        setattr(_local, slot_name_control_dependency_nodes, current_control_dependency_nodes)
