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

"""TensorHolder module for tensor operations in eager-style graph construction."""

import ctypes
from typing import List, Optional, TYPE_CHECKING, Union
from ge._capi.pyes_graph_builder_wrapper import (
    esb_lib,
    EsCTensorHolderPtr,
    get_generated_lib,
    c_int64,
    DEFAULT_GENERATED_LIB_NAME,
    MATH_LIB_NAME
)

from ge.es.tensor_like import convert_to_tensor_holder
from ge.graph.types import DataType, Format

if TYPE_CHECKING:
    from ge.es.graph_builder import GraphBuilder
    from ge.es.tensor_like import TensorLike
    from ge.graph.node import Node


class TensorHolder:
    """TensorHolder for tensor operations in eager-style graph construction.
    
    This class provides a Pythonic interface for tensor operations using
    the eager-style graph builder C API.
    
    The TensorHolder automatically resolves and maintains a strong reference to its
    GraphBuilder to ensure that the underlying C++ resources remain valid as long
    as the TensorHolder object exists. This prevents dangling references when the
    GraphBuilder is garbage collected while TensorHolder objects are still in use.


    Note:
        # TensorHolder cannot call setter methods after GraphBuilder.build_and_reset() is called.

    Example:
        >>> builder = GraphBuilder("my_graph")
        >>> tensor1 = builder.create_const_float(1.0)
        >>> tensor2 = builder.create_const_float(2.0)
        >>> result = tensor1 + tensor2  # Uses operator overloading
        >>> result = Add(tensor1, tensor2)  # Or explicit method call
        >>> # Even if 'builder' goes out of scope, 'tensor1' and 'tensor2' remain valid
    """

    def __init__(self):
        """Prevent direct instantiation of TensorHolder objects."""
        raise RuntimeError("TensorHolder objects should not be created directly")

    @classmethod
    def _create_from(cls, handle: EsCTensorHolderPtr, owner_builder: 'GraphBuilder') -> 'TensorHolder':
        """Create TensorHolder object from C++ pointer. (internal use only by e.g GraphBuilder.create_input(),
         do not use this method directly)
        
        Args:
            handle: C++ EsCTensorHolder object pointer.
            owner_builder: The GraphBuilder that owns this tensor.

        Returns:
            TensorHolder object.

        Raises:
            ValueError: If handle or owner_builder is None.
        """
        if not handle:
            raise ValueError("Tensor handle cannot be None")
        if not owner_builder:
            raise ValueError("Owner builder cannot be None")

        instance = cls.__new__(cls)
        instance._handle = handle
        instance._builder = owner_builder

        return instance

    def _check_usable(self, operation: str) -> None:
        """Check if tensor holder is usable.
        
        Args:
            operation: Operation name.
        """
        self._builder._check_usable(operation)

    def _validate_operation(self, other: 'TensorHolder', op_name: str) -> None:
        """Validate operation between two tensor holders.

        Args:
            other: Another TensorHolder object.
            op_name: Operation name.
        """
        if self._builder is not other._builder:
            raise ValueError(
                f"Cannot perform {op_name}: tensors from different GraphBuilders "
                f"('{self._builder.name}' vs '{other._builder.name}')"
            )
        other._check_usable(op_name)
        self._check_usable(op_name)

    @property
    def name(self) -> str:
        """Get node name.
        
        Returns:
            Producer node name.
        """
        return self._get_node_snapshot().name

    def set_data_type(self, data_type: DataType) -> 'TensorHolder':
        """Set tensor data type.
        
        Args:
            data_type: Data type using DataType enum.

        Returns:
            TensorHolder object.
            
        Raises:
            TypeError: If data_type is not a DataType enum.
            RuntimeError: If operation fails.
        """
        self._check_usable("set data type")

        if not isinstance(data_type, DataType):
            raise TypeError("Data type must be a DataType enum")

        if esb_lib.EsSetDataType(self._handle, ctypes.c_int(data_type.value)) != 0:
            raise RuntimeError(f"Failed to set data type")

        return self

    def set_format(self, format: Format) -> 'TensorHolder':
        """Set tensor data format.
        
        Args:
            format: Data format using Format enum.

        Returns:
            TensorHolder object.
            
        Raises:
            TypeError: If format is not a Format enum.
            RuntimeError: If operation fails.
        """
        self._check_usable("set format")

        if not isinstance(format, Format):
            raise TypeError("Format must be a Format enum")

        if esb_lib.EsSetFormat(self._handle, ctypes.c_int(format.value)) != 0:
            raise RuntimeError(f"Failed to set format")

        return self

    def set_shape(self, shape: List[int]) -> 'TensorHolder':
        """Set tensor shape.
        
        Args:
            shape: List of shape dimensions.
            
        Returns:
            TensorHolder object.
            
        Raises:
            TypeError: If shape is not a list of integers.
            RuntimeError: If operation fails.
        """
        self._check_usable("set shape")

        if not isinstance(shape, list):
            raise TypeError("Shape must be a list of integers")

        if not all(isinstance(dim, int) for dim in shape):
            raise TypeError("All shape dimensions must be integers")

        dim_num = len(shape)
        shape_array = (c_int64 * dim_num)(*shape)
        if esb_lib.EsSetShape(self._handle, shape_array, c_int64(dim_num)) != 0:
            raise RuntimeError(f"Failed to set shape")

        return self

    def _get_math_operator_lib(self):
        """Get library containing math operators (Add/Sub/Mul/Div).
        
        Tries libes_math.so first, falls back to default library if unavailable.
        
        Returns:
            ctypes library object containing EsAdd/EsSub/EsMul/EsDiv.
            
        Raises:
            RuntimeError: If neither library is available.
        """
        try:
            return get_generated_lib()
        except RuntimeError:
            try:
                return get_generated_lib(MATH_LIB_NAME)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"Math operators (Add/Sub/Mul/Div) not available: neither {MATH_LIB_NAME} "
                    f"nor {DEFAULT_GENERATED_LIB_NAME} could be loaded. Please ensure at least one is accessible."
                ) from exc

    # Operator overloading support
    def __add__(self, other: Union['TensorHolder', 'TensorLike']) -> 'TensorHolder':
        """Support + operator."""
        return self.add(other)

    def __sub__(self, other: Union['TensorHolder', 'TensorLike']) -> 'TensorHolder':
        """Support - operator."""
        return self.sub(other)

    def __mul__(self, other: Union['TensorHolder', 'TensorLike']) -> 'TensorHolder':
        """Support * operator."""
        return self.mul(other)

    def __truediv__(self, other: Union['TensorHolder', 'TensorLike']) -> 'TensorHolder':
        """Support / operator."""
        return self.div(other)

    def __radd__(self, other: Union['TensorHolder', 'TensorLike']) -> 'TensorHolder':
        """Support right addition operator."""
        return self.add(other)

    def __rsub__(self, other: Union['TensorHolder', 'TensorLike']) -> 'TensorHolder':
        """Support right subtraction operator."""
        other = convert_to_tensor_holder(other, self._builder)
        return other.sub(self)

    def __rmul__(self, other: Union['TensorHolder', 'TensorLike']) -> 'TensorHolder':
        """Support right multiplication operator."""
        return self.mul(other)

    def __rtruediv__(self, other: Union['TensorHolder', 'TensorLike']) -> 'TensorHolder':
        """Support right division operator."""
        other = convert_to_tensor_holder(other, self._builder)
        return other.div(self)

    def add(self, other: Union['TensorHolder', 'TensorLike']) -> 'TensorHolder':
        """Add two tensors.

        Args:
            other: Another TensorHolder object.

        Returns:
            New TensorHolder representing the result.

        Raises:
            TypeError: If other is not a TensorHolder.
            RuntimeError: If operation fails or library is not available.
        """
        other = convert_to_tensor_holder(other, self._builder)

        self._validate_operation(other, "add")
        generated_lib = self._get_math_operator_lib()

        result_handle = generated_lib.EsAdd(self._handle, other._handle)
        if not result_handle:
            raise RuntimeError("Failed to create add operation")

        return self._builder._apply_scope_infos_to_node(TensorHolder._create_from(result_handle, self._builder))

    def sub(self, other: Union['TensorHolder', 'TensorLike']) -> 'TensorHolder':
        """Subtract two tensors.

        Args:
            other: Another TensorHolder object.

        Returns:
            New TensorHolder representing the result.

        Raises:
            TypeError: If other is not a TensorHolder.
            RuntimeError: If operation fails or library is not available.
        """
        other = convert_to_tensor_holder(other, self._builder)

        self._validate_operation(other, "sub")
        generated_lib = self._get_math_operator_lib()

        result_handle = generated_lib.EsSub(self._handle, other._handle)
        if not result_handle:
            raise RuntimeError("Failed to create sub operation")

        return self._builder._apply_scope_infos_to_node(TensorHolder._create_from(result_handle, self._builder))

    def mul(self, other: Union['TensorHolder', 'TensorLike']) -> 'TensorHolder':
        """Multiply two tensors.

        Args:
            other: Another TensorHolder object.

        Returns:
            New TensorHolder representing the result.

        Raises:
            TypeError: If other is not a TensorHolder.
            RuntimeError: If operation fails or library is not available.
        """
        other = convert_to_tensor_holder(other, self._builder)

        self._validate_operation(other, "mul")
        generated_lib = self._get_math_operator_lib()

        result_handle = generated_lib.EsMul(self._handle, other._handle)
        if not result_handle:
            raise RuntimeError("Failed to create mul operation")

        return self._builder._apply_scope_infos_to_node(TensorHolder._create_from(result_handle, self._builder))

    def div(self, other: Union['TensorHolder', 'TensorLike']) -> 'TensorHolder':
        """Divide two tensors.

        Args:
            other: Another TensorHolder object.

        Returns:
            New TensorHolder representing the result.

        Raises:
            TypeError: If other is not a TensorHolder.
            RuntimeError: If operation fails or library is not available.
        """
        other = convert_to_tensor_holder(other, self._builder)

        self._validate_operation(other, "div")
        generated_lib = self._get_math_operator_lib()

        result_handle = generated_lib.EsDiv(self._handle, other._handle)
        if not result_handle:
            raise RuntimeError("Failed to create div operation")

        return self._builder._apply_scope_infos_to_node(TensorHolder._create_from(result_handle, self._builder))

    def get_owner_builder(self):
        return self._builder

    def _get_node_snapshot(self) -> 'Node':
        """Get a snapshot of the producer node of this tensor. (internal use by internal api)
        
        Returns:
            Node object representing the producer node snapshot.
            The returned Node object is a snapshot and does not own the underlying pointer.
            
        Raises:
            RuntimeError: If getting node snapshot fails.
        """
        from ge.graph.node import Node

        node_ptr = esb_lib.EsGetProducer(self._handle)
        if not node_ptr:
            raise RuntimeError("Failed to get producer node")

        return Node._create_from(ctypes.c_void_p(node_ptr), owns_handle=False)
