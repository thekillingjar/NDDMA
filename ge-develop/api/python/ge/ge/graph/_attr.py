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

"""AttrValue module for attribute value operations in GraphEngine."""

import ctypes
from typing import List, Union, Any, Callable, TypeVar, Type
from ge._capi.pygraph_wrapper import graph_lib
from .types import DataType, AttrValueType


def _clear_cache_on_success(method: Callable[..., int]) -> Callable[..., bool]:
    """Decorator: clear cache and return True on success (ret == 0), else return False."""

    def _wrapped(self, *args, **kwargs):
        ret = method(self, *args, **kwargs)
        if ret == 0:
            self._clear_cache()
            return True
        return False

    return _wrapped


def _cache_with_type_check(expected_type: AttrValueType):
    """Decorator factory: create a cache decorator with type checking."""

    def decorator(method: Callable[..., Any]) -> Callable[..., Any]:
        def _wrapped(self, *args, **kwargs):
            # 检查是否有缓存且类型匹配
            if (self._cached_value is not None and
                    self._value_type is not None and
                    self._value_type == expected_type):
                return self._cached_value

            # 没有缓存或类型不匹配，调用原方法获取值
            try:
                result = method(self, *args, **kwargs)
                # 设置缓存和类型
                self._cached_value = result
                self._value_type = expected_type
                return result
            except Exception:
                # 如果获取失败，不设置缓存
                raise

        return _wrapped

    return decorator


def _validate_list_input(values: List[Any], expected_type: Type, type_name: str) -> None:
    """Validate list input parameters.
    
    Args:
        values: List of values to validate.
        expected_type: Expected type for list elements.
        type_name: Human-readable type name for error messages.
        
    Raises:
        TypeError: If values is not a list or elements are wrong type.
        ValueError: If values is empty.
    """
    if not isinstance(values, list):
        raise TypeError("Values must be a list")

    if not values:
        raise ValueError("Empty list is not supported")

    if not all(isinstance(v, expected_type) for v in values):
        raise TypeError(f"All values must be {type_name}")


def _create_list_setter(c_type, setter_func, expected_type, type_name):
    """Factory function to create list setter methods.
    
    Args:
        c_type: C type for the array (e.g., ctypes.c_float).
        setter_func: C API function for setting the list.
        expected_type: Python type for validation.
        type_name: Human-readable type name for error messages.
        
    Returns:
        Decorated setter method.
    """

    @_clear_cache_on_success
    def setter(self, values: List[Any]) -> bool:
        """Set list of values.
        
        Args:
            values: List of values.
            
        Returns:
            True if successful, False otherwise.
        """
        _validate_list_input(values, expected_type, type_name)

        # Convert values to the expected type
        if expected_type == (int, float):  # Special case for float lists
            converted_values = [float(v) for v in values]
        else:
            converted_values = values

        arr = (c_type * len(converted_values))(*converted_values)
        return setter_func(self._av_ptr, arr, len(converted_values))

    return setter


def _create_list_getter(c_type, getter_func, free_func, expected_type):
    """Factory function to create list getter methods.
    
    Args:
        c_type: C type for the array (e.g., ctypes.c_float).
        getter_func: C API function for getting the list.
        free_func: C API function for freeing the list.
        expected_type: Python type for the returned values.
        
    Returns:
        Decorated getter method.
    """

    @_cache_with_type_check(expected_type)
    def getter(self) -> List[Any]:
        """Get list of values.
        
        Returns:
            List of values.
            
        Raises:
            RuntimeError: If retrieval fails.
        """
        size = ctypes.c_int64()
        c_array = getter_func(self._av_ptr, ctypes.byref(size))
        if not c_array:
            raise RuntimeError(f"Failed to get list value")

        try:
            return [c_array[i] for i in range(size.value)]
        finally:
            free_func(c_array)

    return getter


class _AttrValue:
    """AttrValue for attribute value operations in GraphEngine.
    
    This class provides a Pythonic interface for attribute value operations
    using the GraphEngine C API.
    
    Example:
        >>> attr = _AttrValue()
        >>> attr.set_string("hello")
        >>> value = attr.get_string()
        >>> attr.set_list_float([1.0, 2.0, 3.0])
        >>> values = attr.get_list_float()
    """

    def __init__(self) -> None:
        """Initialize an AttrValue.
        
        Creates a new AttrValue instance using the GraphEngine C API.
        """
        # Create new AttrValue
        self._av_ptr = graph_lib.GeApiWrapper_AttrValue_Create()

        self._value_type = None
        self._cached_value = None

    def __del__(self) -> None:
        """Clean up resources."""
        graph_lib.GeApiWrapper_AttrValue_Destroy(self._av_ptr)
        self._av_ptr = None

    def __copy__(self) -> None:
        """Copy is not supported."""
        raise RuntimeError("AttrValue does not support copy")

    def __deepcopy__(self, memodict) -> None:
        """Deep copy is not supported."""
        raise RuntimeError("AttrValue does not support deepcopy")

    def _clear_cache(self) -> None:
        """Clear cached value and type."""
        self._cached_value = None
        self._value_type = None

    def _type_name(self, avt: AttrValueType) -> str:
        """Get human-readable type name.
        
        Args:
            avt: AttrDataType enum value.
            
        Returns:
            Human-readable type name.
        """
        type_names = {
            AttrValueType.VT_STRING: "string",
            AttrValueType.VT_FLOAT: "float",
            AttrValueType.VT_BOOL: "bool",
            AttrValueType.VT_INT: "int",
            AttrValueType.VT_DATA_TYPE: "data_type",
            AttrValueType.VT_LIST_FLOAT: "list_float",
            AttrValueType.VT_LIST_BOOL: "list_bool",
            AttrValueType.VT_LIST_INT: "list_int",
            AttrValueType.VT_LIST_DATA_TYPE: "list_data_type",
            AttrValueType.VT_LIST_STRING: "list_string",
        }
        return type_names.get(avt, f"unknown({avt})")

    @property
    def value_type(self) -> AttrValueType:
        """Get the current value type.
        
        Returns:
            AttrDataType enum value.
        """
        if self._value_type is None:
            self._value_type = AttrValueType(graph_lib.GeApiWrapper_AttrValue_GetValueType(self._av_ptr))
        return self._value_type

    def get_value_type(self) -> AttrValueType:
        """Get the current value type.
        
        Returns:
            AttrDataType enum value.
        """
        return self.value_type

    def get_value(self) -> Any:
        """Get the current value.
        
        Returns:
            Value.
        """
        if self.value_type == AttrValueType.VT_STRING:
            return self.get_string()
        elif self.value_type == AttrValueType.VT_FLOAT:
            return self.get_float()
        elif self.value_type == AttrValueType.VT_BOOL:
            return self.get_bool()
        elif self.value_type == AttrValueType.VT_INT:
            return self.get_int()
        elif self.value_type == AttrValueType.VT_DATA_TYPE:
            return self.get_data_type()
        elif self.value_type == AttrValueType.VT_LIST_FLOAT:
            return self.get_list_float()
        elif self.value_type == AttrValueType.VT_LIST_BOOL:
            return self.get_list_bool()
        elif self.value_type == AttrValueType.VT_LIST_INT:
            return self.get_list_int()
        elif self.value_type == AttrValueType.VT_LIST_DATA_TYPE:
            return self.get_list_data_type()
        elif self.value_type == AttrValueType.VT_LIST_STRING:
            return self.get_list_string()
        else:
            raise RuntimeError(f"Unsupported attribute type: {self.value_type}")

    def set_value(self, value: Any) -> None:
        """Set the current value.
        
        Args:
            value: Value to set.
        """
        if isinstance(value, str):
            self.set_string(value)
        elif isinstance(value, float):
            self.set_float(value)
        elif isinstance(value, bool):
            self.set_bool(value)
        elif isinstance(value, DataType):
            self.set_data_type(value)
        elif isinstance(value, int):
            self.set_int(value)
        elif isinstance(value, list) and all(isinstance(v, float) for v in value):
            self.set_list_float(value)
        elif isinstance(value, list) and all(isinstance(v, bool) for v in value):
            self.set_list_bool(value)
        elif isinstance(value, list) and all(isinstance(v, int) for v in value):
            self.set_list_int(value)
        elif isinstance(value, list) and all(isinstance(v, DataType) for v in value):
            self.set_list_data_type(value)
        elif isinstance(value, list) and all(isinstance(v, str) for v in value):
            self.set_list_string(value)
        else:
            raise ValueError(f"Unsupported attribute type: {type(value)} for value: {value}")

    # String operations
    @_clear_cache_on_success
    def set_string(self, value: str) -> bool:
        """Set string value.
        
        Args:
            value: String value to set.
            
        Returns:
            True if successful, False otherwise.
        """
        if not isinstance(value, str):
            raise TypeError("Value must be a string")

        value_bytes = value.encode('utf-8')
        return graph_lib.GeApiWrapper_AttrValue_SetString(self._av_ptr, value_bytes)

    @_cache_with_type_check(AttrValueType.VT_STRING)
    def get_string(self) -> str:
        """Get string value.
        
        Returns:
            String value.
            
        Raises:
            RuntimeError: If value is not a string or retrieval fails.
        """
        c_str = graph_lib.GeApiWrapper_AttrValue_GetString(self._av_ptr)
        if not c_str:
            raise RuntimeError("Failed to get string value")

        try:
            return ctypes.string_at(c_str).decode('utf-8')
        finally:
            graph_lib.GeApiWrapper_FreeString(c_str)

    # Float operations
    @_clear_cache_on_success
    def set_float(self, value: float) -> bool:
        """Set float value.
        
        Args:
            value: Float value to set.
            
        Returns:
            True if successful, False otherwise.
        """
        if not isinstance(value, (int, float)):
            raise TypeError("Value must be a number")

        return graph_lib.GeApiWrapper_AttrValue_SetFloat(self._av_ptr, ctypes.c_float(float(value)))

    @_cache_with_type_check(AttrValueType.VT_FLOAT)
    def get_float(self) -> float:
        """Get float value.
        
        Returns:
            Float value.
            
        Raises:
            RuntimeError: If value is not a float or retrieval fails.
        """
        value = ctypes.c_float()
        ret = graph_lib.GeApiWrapper_AttrValue_GetFloat(self._av_ptr, ctypes.byref(value))
        if ret != 0:
            raise RuntimeError("Failed to get float value")
        return value.value

    # Bool operations
    @_clear_cache_on_success
    def set_bool(self, value: bool) -> bool:
        """Set bool value.
        
        Args:
            value: Bool value to set.
            
        Returns:
            True if successful, False otherwise.
        """
        if not isinstance(value, bool):
            raise TypeError("Value must be a boolean")

        return graph_lib.GeApiWrapper_AttrValue_SetBool(self._av_ptr, ctypes.c_bool(value))

    @_cache_with_type_check(AttrValueType.VT_BOOL)
    def get_bool(self) -> bool:
        """Get bool value.
        
        Returns:
            Bool value.
            
        Raises:
            RuntimeError: If value is not a bool or retrieval fails.
        """
        value = ctypes.c_bool()
        ret = graph_lib.GeApiWrapper_AttrValue_GetBool(self._av_ptr, ctypes.byref(value))
        if ret != 0:
            raise RuntimeError("Failed to get bool value")
        return value.value

    # Int operations
    @_clear_cache_on_success
    def set_int(self, value: int) -> bool:
        """Set int value.
        
        Args:
            value: Int value to set.
            
        Returns:
            True if successful, False otherwise.
        """
        if not isinstance(value, int):
            raise TypeError("Value must be an integer")

        return graph_lib.GeApiWrapper_AttrValue_SetInt(self._av_ptr, ctypes.c_int64(value))

    @_cache_with_type_check(AttrValueType.VT_INT)
    def get_int(self) -> int:
        """Get int value.
        
        Returns:
            Int value.
            
        Raises:
            RuntimeError: If value is not an int or retrieval fails.
        """
        value = ctypes.c_int64()
        ret = graph_lib.GeApiWrapper_AttrValue_GetInt(self._av_ptr, ctypes.byref(value))
        if ret != 0:
            raise RuntimeError("Failed to get int value")
        return value.value

    # DataType operations
    @_clear_cache_on_success
    def set_data_type(self, value: DataType) -> bool:
        """Set DataType value.
        
        Args:
            value: DataType value to set.
            
        Returns:
            True if successful, False otherwise.
        """
        if not isinstance(value, DataType):
            raise TypeError("Value must be a DataType")

        return graph_lib.GeApiWrapper_AttrValue_SetDataType(self._av_ptr, ctypes.c_int(value.value))

    @_cache_with_type_check(AttrValueType.VT_DATA_TYPE)
    def get_data_type(self) -> DataType:
        """Get DataType value.
        
        Returns:
            DataType value.
            
        Raises:
            RuntimeError: If value is not a DataType or retrieval fails.
        """
        value = ctypes.c_int()
        ret = graph_lib.GeApiWrapper_AttrValue_GetDataType(self._av_ptr, ctypes.byref(value))
        if ret != 0:
            raise RuntimeError("Failed to get DataType value")
        return DataType(value.value)

    # List operations
    set_list_float = _create_list_setter(
        ctypes.c_float,
        graph_lib.GeApiWrapper_AttrValue_SetListFloat,
        (int, float),
        "numbers"
    )

    get_list_float = _create_list_getter(
        ctypes.c_float,
        graph_lib.GeApiWrapper_AttrValue_GetListFloat,
        graph_lib.GeApiWrapper_FreeListFloat,
        AttrValueType.VT_LIST_FLOAT
    )

    set_list_int = _create_list_setter(
        ctypes.c_int64,
        graph_lib.GeApiWrapper_AttrValue_SetListInt,
        int,
        "integers"
    )

    get_list_int = _create_list_getter(
        ctypes.c_int64,
        graph_lib.GeApiWrapper_AttrValue_GetListInt,
        graph_lib.GeApiWrapper_FreeListInt,
        AttrValueType.VT_LIST_INT
    )

    set_list_bool = _create_list_setter(
        ctypes.c_bool,
        graph_lib.GeApiWrapper_AttrValue_SetListBool,
        bool,
        "booleans"
    )

    get_list_bool = _create_list_getter(
        ctypes.c_bool,
        graph_lib.GeApiWrapper_AttrValue_GetListBool,
        graph_lib.GeApiWrapper_FreeListBool,
        AttrValueType.VT_LIST_BOOL
    )

    set_list_data_type = _create_list_setter(
        ctypes.c_int,
        graph_lib.GeApiWrapper_AttrValue_SetListDataType,
        DataType,
        "DataTypes"
    )

    get_list_data_type = _create_list_getter(
        ctypes.c_int,
        graph_lib.GeApiWrapper_AttrValue_GetListDataType,
        graph_lib.GeApiWrapper_FreeListDataType,
        AttrValueType.VT_LIST_DATA_TYPE
    )

    # String list needs special handling due to encoding
    @_clear_cache_on_success
    def set_list_string(self, values: List[str]) -> bool:
        """Set list of string values.
        
        Args:
            values: List of string values.
            
        Returns:
            True if successful, False otherwise.
        """
        _validate_list_input(values, str, "strings")

        arr = (ctypes.c_char_p * len(values))(*[v.encode('utf-8') for v in values])
        return graph_lib.GeApiWrapper_AttrValue_SetListString(self._av_ptr, arr, len(values))

    @_cache_with_type_check(AttrValueType.VT_LIST_STRING)
    def get_list_string(self) -> List[str]:
        """Get list of string values.
        
        Returns:
            List of string values.
            
        Raises:
            RuntimeError: If value is not a list of strings or retrieval fails.
        """
        size = ctypes.c_int64()
        c_array = graph_lib.GeApiWrapper_AttrValue_GetListString(self._av_ptr, ctypes.byref(size))
        if not c_array:
            raise RuntimeError("Failed to get list_string value")

        try:
            return [ctypes.string_at(c_array[i]).decode('utf-8') for i in range(size.value)]
        finally:
            graph_lib.GeApiWrapper_FreeListString(c_array)

    def __str__(self) -> str:
        """String representation of AttrValue."""
        try:
            value_type = self.get_value_type()
            type_name = self._type_name(value_type)

            if value_type == AttrValueType.VT_STRING:
                return f"AttrValue(type: {type_name}, value: '{self.get_string()}')"
            elif value_type == AttrValueType.VT_FLOAT:
                return f"AttrValue(type: {type_name}, value: {self.get_float()})"
            elif value_type == AttrValueType.VT_BOOL:
                return f"AttrValue(type: {type_name}, value: {self.get_bool()})"
            elif value_type == AttrValueType.VT_INT:
                return f"AttrValue(type: {type_name}, value: {self.get_int()})"
            elif value_type == AttrValueType.VT_DATA_TYPE:
                return f"AttrValue(type: {type_name}, value: {self.get_data_type()})"
            elif value_type == AttrValueType.VT_LIST_FLOAT:
                return f"AttrValue(type: {type_name}, value: {self.get_list_float()})"
            elif value_type == AttrValueType.VT_LIST_BOOL:
                return f"AttrValue(type: {type_name}, value: {self.get_list_bool()})"
            elif value_type == AttrValueType.VT_LIST_INT:
                return f"AttrValue(type: {type_name}, value: {self.get_list_int()})"
            elif value_type == AttrValueType.VT_LIST_DATA_TYPE:
                return f"AttrValue(type: {type_name}, value: {self.get_list_data_type()})"
            elif value_type == AttrValueType.VT_LIST_STRING:
                return f"AttrValue(type: {type_name}, value: {self.get_list_string()})"
            else:
                return f"AttrValue(type: {type_name}, value: <unknown>)"
        except Exception:
            return "AttrValue(<error reading value>)"

    def __repr__(self) -> str:
        """Detailed string representation of AttrValue."""
        return self.__str__()
