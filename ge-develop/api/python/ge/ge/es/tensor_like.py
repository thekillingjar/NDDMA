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

"""Handling tensor-like operands in eager-style graph construction."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from ge.es import GraphBuilder, TensorHolder

Number = Union[int, float]
TensorLike = Union[Number, List['TensorLike']]


def convert_to_tensor_holder(value: Optional[Union['TensorHolder', 'TensorLike']],
                             owner_builder: 'GraphBuilder') -> Optional['TensorHolder']:
    """
    Convert value to TensorHolder object.
    Args:
        value: TensorHolder or int|float or nested lists of int|float or None
        owner_builder: The GraphBuilder that owns this value.
    Returns:
        TensorHolder object or None if value is None
    Raises:
        TypeError: If value is not TensorHolder or TensorLike or None.
    """
    from ge.es.tensor_holder import TensorHolder
    if value is None or isinstance(value, TensorHolder):
        return value
    if isinstance(value, (int, float)):
        return _convert_scalar_to_tensor_holder(value, owner_builder)

    if isinstance(value, list):
        return _convert_list_to_tensor_holder(value, owner_builder)

    raise TypeError("Value must be TensorHolder or int|float or nested lists of int|float or None")


def resolve_builder(*values: Union['TensorLike', 'TensorHolder', 'GraphBuilder']) -> 'GraphBuilder':
    """
    Resolve the owning GraphBuilder from inputs.

    Args:
        *values: TensorHolder or TensorLike or GraphBuilder arguments.

    Returns:
        The owner GraphBuilder.

    Raises:
        ValueError: If no TensorHolder is provided to infer the GraphBuilder.
    """
    if not values:
        raise ValueError("At least one argument is required to resolve GraphBuilder")

    from ge.es import TensorHolder, GraphBuilder
    for value in values:
        if isinstance(value, GraphBuilder):
            return value
        if isinstance(value, TensorHolder):
            return value.get_owner_builder()

    raise ValueError("Please ensure at least one input tensor or an explicit owner_builder is provided when supported")


def _flatten_and_infer_shape(value: List['TensorLike']) -> Tuple[List['Number'], List[int]]:
    """
    Flatten nested numeric lists and infer their tensor shape.

    Args:
        value: List whose leaves are ints/floats.

    Returns:
        A pair of (flat_values, shape) (e.g., [[1, 2], [3, 4]] -> ([1, 2, 3, 4], [2, 2])).

    Raises:
        TypeError: value or any nested element is neither list nor numeric.
        ValueError: Nested shapes differ.
    """
    if not value:
        return value, [0]

    def _recurse(current: 'TensorLike') -> Tuple[List['Number'], List[int]]:
        if isinstance(current, (int, float)):
            return [current], []

        if not isinstance(current, list) or not current:
            raise TypeError("Value must be int|float or nested lists of int|float")

        first_flat, inner_shape = _recurse(current[0])
        flat_acc: List[Number] = list(first_flat)

        for elem in current[1:]:
            sub_flat, sub_shape = _recurse(elem)
            if sub_shape != inner_shape:
                raise ValueError("Irregular nested list: all sub-lists must have the same shape")
            flat_acc.extend(sub_flat)

        return flat_acc, [len(current)] + inner_shape

    return _recurse(value)


def _unflatten(value: List['Number'], shape: List[int]) -> List['TensorLike']:
    """
    Unflatten data

    Args:
        value: flattened data like [1,2,3,4,5,6]
        shape: list of shape like [2,3]
    Returns:
        List of TensorLike

    Raises:
        ValueError: If shape is empty
        ValueError: If shape and value does not match
    """

    # shape check
    if not shape:
        raise ValueError("Shape cannot be empty")

    # length check 
    total_size = 1
    for dim in shape:
        total_size *= dim

    if len(value) != total_size:
        raise ValueError(
            f"Length of list {len(value)} does not match shape {total_size}"
        )

    # recurse
    def _recurse(flat_values, shape):
        # last dimension
        if len(shape) == 1:
            return flat_values[:shape[0]]

        size = shape[0]
        step = int(len(flat_values) / size)
        result = []
        index = 0

        for _ in range(size):
            block = flat_values[index:index + step]
            result.append(_recurse(block, shape[1:]))
            index += step

        return result

    return _recurse(value, shape)

def _convert_scalar_to_tensor_holder(value: 'Number', owner_builder: 'GraphBuilder') -> 'TensorHolder':
    """
    Convert scalar to TensorHolder object.

    Args:
        value: int or float scaler
        owner_builder: The GraphBuilder that owns this value.

    Returns:
        TensorHolder representing the scalar.

    Raises:
        TypeError: If value is not a number.
    """
    if isinstance(value, int):
        return owner_builder.create_scalar_int64(value)

    if isinstance(value, float):
        return owner_builder.create_scalar_float(value)

    raise TypeError("Value must be a number")


def _convert_list_to_tensor_holder(value: List[Any], owner_builder: 'GraphBuilder') -> 'TensorHolder':
    """
    Convert list to TensorHolder object.

    Args:
        value: nested lists of int/float
        owner_builder: The GraphBuilder that owns this value.

    Returns:
        TensorHolder representing the list.
    """
    flat_values, shape = _flatten_and_infer_shape(value)
    has_float = any(isinstance(v, float) for v in flat_values)
    if has_float:
        return owner_builder.create_const_float(flat_values, shape=shape)
    return owner_builder.create_const_int64(flat_values, shape=shape)
