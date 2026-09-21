#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import ctypes
from typing import Optional, List, Any, Union, TYPE_CHECKING
from ge._capi.pygraph_wrapper import graph_lib
from ge._capi.pyes_graph_builder_wrapper import esb_lib
from .types import DataType, Format
from ._numeric import float_list_to_fp16_bits


if TYPE_CHECKING:
    from ge.es.graph_builder import GraphBuilder
    from ge.es.tensor_like import TensorLike, _unflatten
UnionTensorDataType = Union[List[int], List[float], List[bool]]

class Tensor:
    def __init__(
        self,
        data: Optional[UnionTensorDataType] = None,
        file_path: Optional[str] = None,
        data_type: Optional[DataType] = DataType.DT_FLOAT,
        format: Optional[Format] = Format.FORMAT_ND,
        shape: Optional[List[int]] = None
        ) -> None:
        """
        Args:
            data: Data to read from.
            file_path: File path to read from.
            data_type: Data type using DataType enum, defaults to DataType.DT_FLOAT.
            format: Data format using Format enum, defaults to Format.FORMAT_ND.
            shape: List of shape dimensions, If None, means scalar.
        Example
        >>> Tensor(data, None, data_type, format, shape)
        >>> Tensor(None, file_path, data_type, format, shape)
        """
        self._handle = None
        self._owns_handle = False
        self._owner = None
        if shape is not None:
            if not isinstance(shape, list) or not all(isinstance(dim, int) for dim in shape):
                raise TypeError("Shape must be a list of integers")
        if data is not None and file_path is not None:
            raise RuntimeError("Tensor should be created either by data or by file")
        elif data is not None and file_path is None:
            self._create_from_data(data, data_type, format, shape)
        elif data is None and file_path is not None:
            self._create_from_file(file_path, data_type, format, shape)
        else:
            self._handle = graph_lib.GeApiWrapper_Tensor_CreateTensor()
        if not self._handle:
            raise RuntimeError("Failed to create Tensor")
        self._owns_handle = True
        self._owner = None

    @staticmethod
    def _prepare_ctypes_array(data: List[Any], data_type: DataType):
        int_type_map = {
            DataType.DT_INT64: ctypes.c_int64,
            DataType.DT_UINT64: ctypes.c_uint64,
            DataType.DT_INT32: ctypes.c_int32,
            DataType.DT_UINT32: ctypes.c_uint32,
            DataType.DT_INT16: ctypes.c_int16,
            DataType.DT_UINT16: ctypes.c_uint16,
            DataType.DT_INT8: ctypes.c_int8,
            DataType.DT_UINT8: ctypes.c_uint8,
        }
        if all(isinstance(x, bool) for x in data):
            return ctypes.c_bool, [bool(x) for x in data]

        if data_type in int_type_map:
            return int_type_map[data_type], [int(x) for x in data]

        if data_type == DataType.DT_FLOAT:
            return ctypes.c_float, [float(x) for x in data]

        if data_type == DataType.DT_FLOAT16:
            return ctypes.c_uint16, float_list_to_fp16_bits([float(x) for x in data])

        if data_type == DataType.DT_DOUBLE:
            raise RuntimeError("DT_DOUBLE is not supported in python Tensor constructor")

        raise RuntimeError("Failed to create Tensor with data type: {}".format(data_type))

    def _create_from_data(
        self,
        data: Optional[UnionTensorDataType],
        data_type: Optional[DataType],
        format: Optional[Format], 
        shape: Optional[List[int]]):
        """Create Tensor from data."""
        if not isinstance(data, list):
            raise TypeError("data should be List")

        c_type, normalized = self._prepare_ctypes_array(data, data_type)
        c_data_array = (c_type * len(normalized))(*normalized)

        dim_num = len(shape) if shape is not None else 0
        c_dims_array = (ctypes.c_int64 * dim_num)(*shape) if shape is not None else None
        self._handle = esb_lib.EsCreateEsCTensor(c_data_array, c_dims_array, dim_num, data_type, format)

    def _create_from_file(
        self,
        file_path: Optional[str],
        data_type: Optional[DataType],
        format: Optional[Format], 
        shape: Optional[List[int]]):
        """Create Tensor from file."""
        dim_num = len(shape) if shape is not None else 0
        c_dims_array = (ctypes.c_int64 * dim_num)(*shape) if shape is not None else None
        file_path_bytes = file_path.encode("utf-8")
        self._handle = esb_lib.EsCreateEsCTensorFromFile(file_path_bytes, c_dims_array, dim_num, data_type, format)

    def __del__(self) -> None:
        """Clean up resources."""
        if self._owns_handle:
            graph_lib.GeApiWrapper_Tensor_DestroyEsCTensor(self._handle)  # _handle must be valid
            self._handle = None

    def __copy__(self) -> None:
        """Copy is not supported."""
        raise RuntimeError("Tensor does not support copy")

    def __deepcopy__(self, memodict) -> None:
        """Deep copy is not supported."""
        raise RuntimeError("Tensor does not support deepcopy")
        
    def __str__(self) -> str:
        return f'''
        Tensor format is {self.get_format()}, 
        data_type is {self.get_data_type()}, 
        shape is {self.get_shape()}, 
        data is {self.get_data()}
        '''
        
    @classmethod
    def _create_from(cls, handle: ctypes.c_void_p) -> 'Tensor':
        """Create Tensor object from C++ pointer.

        Args:
            handle: C++ Tensor object pointer.

        Returns:
            Tensor object.

        Raises:
            ValueError: If pointer is None.
        """
        if not handle:
            raise ValueError("Tensor pointer cannot be None")
        instance = cls.__new__(cls)
        instance._handle = handle
        instance._owns_handle = True
        instance._owner = None
        return instance

    def _transfer_ownership_when_pass_as_attr(self, new_owner: 'GraphBuilder') -> None:
        """Transfer ownership of the C++ resource to new_owner.
        
        After calling this method, Python will no longer destroy the underlying
        C++ Tensor object. This is called automatically when a Tenensor is passed
        as an attribute.
        
        Args:
            new_owner: The object that will manage the C++ resource (typically a GraphBuilder).
                       this Tensor will hold a reference to keep the new_owner alive.
        """
        if self._owner is not None:
            raise RuntimeError("Tensor already has an new owner builder :{}, cannot transfer ownership again"
                               .format(self._owner.name))
        self._owns_handle = False
        self._owner = new_owner  # Keep reference to prevent premature GC

    def set_format(self, format: Format) -> 'Tensor':
        """Set format of tensor.

        Args:
            format: format to be set.

        Raises:
            TypeError: If format is not a Format.
            RuntimeError: If setting format operation fails.
        """
        if not isinstance(format, Format):
            raise TypeError("Format must be a Format")

        format_ref = ctypes.byref(ctypes.c_int(format.value))
        ret = graph_lib.GeApiWrapper_Tensor_SetFormat(self._handle, format_ref)
        if ret != 0:  # GRAPH_SUCCESS
            raise RuntimeError(f"Failed to set format {format}")
        return self

    def get_format(self) -> Format:
        """Get format of tensor.

        Returns:
            Format of tensor

        Raises:
            ValueError: If return format is not a Format.
        """
        res = graph_lib.GeApiWrapper_Tensor_GetFormat(self._handle)
        return Format(res)

    @property
    def format(self) -> Format:
        return self.get_format()

    def set_data_type(self, data_type: DataType) -> 'Tensor':
        """Set datatype of tensor.

        Args:
            data_type: datatype to be set.

        Raises:
            TypeError: If datatype is not a DataType.
            RuntimeError: If setting datatype operation fails.
        """
        if not isinstance(data_type, DataType):
            raise TypeError("Data_type must be a DataType")

        datatype_ref = ctypes.byref(ctypes.c_int(data_type.value))
        ret = graph_lib.GeApiWrapper_Tensor_SetDataType(self._handle, datatype_ref)
        if ret != 0:  # GRAPH_SUCCESS
            raise RuntimeError(f"Failed to set datatype {data_type}")
        return self

    def get_data_type(self) -> DataType:
        """Get dataype of tensor.

        Returns:
            DataType of tensor

        Raises:
            ValueError: If return dataype is not a DataType.
        """
        res = graph_lib.GeApiWrapper_Tensor_GetDataType(self._handle)
        return DataType(res)

    @property
    def data_type(self) -> DataType:
        return self.get_data_type()

    def get_shape(self) -> List[int]:
        """Get shape of tensor.

        Returns:
            Shape of tensor
        """
        dims_num = ctypes.c_size_t()
        dims_arr = ctypes.POINTER(ctypes.c_int64)()
        ret = graph_lib.GeApiWrapper_Tensor_GetShape(self._handle, ctypes.byref(dims_arr), ctypes.byref(dims_num))

        if ret != 0:  # GRAPH_SUCCESS
            raise RuntimeError(f"Failed to get shape of tensor")
        try: 
            return [dims_arr[i] for i in range(dims_num.value)]
        finally:
            graph_lib.GeApiWrapper_Tensor_FreeDimsArray(dims_arr)
    
    @property
    def shape(self) -> List[int]:
        return self.get_shape()

    def get_data(self) -> 'TensorLike':
        """Get tensor data.

        Returns:
            Tensor data.

        Raises:
            RuntimeError: If data retrieval fails.
        """
        c_str = graph_lib.GeApiWrapper_Tensor_GetData(self._handle)
        if not c_str:
            raise RuntimeError("Failed to get Tensor data")
        
        try:
            data_str = ctypes.string_at(c_str).decode('utf-8')
            return unflatten_tensor_data(data_str, self.shape)
        finally:
            graph_lib.GeApiWrapper_FreeString(c_str)
    
    @property
    def data(self) -> 'TensorLike':
        return self.get_data()


def _parse_str_list(list_str: str) -> List['Number']:
    """
    Convert string like '[1, 2, 3]' into python list.

    Args:
        list_str: string format of list

    Returns:
        List of Number
    """
    list_str = list_str.strip()
    if not (list_str.startswith("[") and list_str.endswith("]")):
        raise ValueError("Input must start with '[' and end with ']'")
    inner = list_str[1:-1].strip()
    if not inner:
        return []
    items = [x.strip() for x in inner.split(",")]
    result = []
    for item in items:
        if item.isdigit() or (item.startswith("-") and item[1:].isdigit()):
            result.append(int(item))
        else:
            try:
                result.append(float(item))
            except ValueError:
                raise ValueError(f"Invalid item: '{item}'")
    return result


def unflatten_tensor_data(tensor_data: str, shape: List[int]) -> 'TensorLike':
    from ge.es.tensor_like import _unflatten
    tensor_data_list = _parse_str_list(tensor_data)
    return _unflatten(tensor_data_list, shape)