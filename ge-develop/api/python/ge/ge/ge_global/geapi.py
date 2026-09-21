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

"""Initialize GE for GraphEngine graph operations."""

import ctypes

from ge._capi.pygeapi_wrapper import geapi_lib

class GeApi:

    def __init__(self):
        pass
    @classmethod
    def ge_initialize(cls, config: dict) -> None:

        """Initialize GE, prepare for execution

        Args:
            config: Config.

        Returns:
            Initialize GE result (int res).
        """
        if not isinstance(config, dict):
            raise TypeError("Ge init config must be a dictionary")
        keys = [k for k in config.keys()]
        values = [v for v in config.values()]
        _self = cls()
        c_array_key = _self._python_list_to_c_array(keys)
        c_array_value = _self._python_list_to_c_array(values)
        if len(c_array_key) == 0 or len(c_array_value) == 0:
            raise TypeError("Ge init config must not be empty")
        ret = geapi_lib.GeApiWrapper_GEInitialize(ctypes.cast(c_array_key, ctypes.POINTER(ctypes.c_char_p)), ctypes.cast(c_array_value, ctypes.POINTER(ctypes.c_char_p)), len(keys))
        if ret != 0:
            raise RuntimeError("GE initialization failed")
        return ret

    @classmethod
    def ge_finalize(cls) -> None:
        """
        GE finalize, releasing all resources
        """
        ret = geapi_lib.GeApiWrapper_GEFinalize()
        if ret != 0:
            raise RuntimeError("GE finalization failed")
        return ret

    def _python_list_to_c_array(self, python_list: list):
        """
            Convert python list to c array

            Parameters:
                python_list: python list.

            Returns:
                c_array: c array.
        """
        size = len(python_list)
        c_array = (ctypes.c_char_p * size)()
        for i , item in enumerate(python_list):
            c_array[i] = item.encode('utf-8')
        return c_array