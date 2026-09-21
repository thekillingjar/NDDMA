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

"""
auto tune function manager
"""
import functools

_tune_space = {}
_tune_param_check = {}


def register_tune_space(op_type):
    """
    Parameters
    ----------
    op_type : string
        op type

    Returns
    -------
    decorator : decorator
        decorator to set fusion build config
    """

    if op_type is None:
        raise RuntimeError("register tune space failed, op_type is none")

    def decorator(func):
        """
        Parameters
        ----------
        func : function
            func
        Returns
        -------
        wrapper : wrapper
            wrapper to exce func
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            """
            Parameters
            ----------
            args : list
            kwargs : dict
            Returns
            -------
            wrapper : func
                func to exce func
            """

            return func(*args, **kwargs)
        _tune_space[op_type] = wrapper
        return wrapper
    return decorator


def get_tune_space(op_type):
    """
    :return:
    """

    if op_type is None:
        raise RuntimeError("register tune space failed, op_type is none")
    return _tune_space.get(op_type)


def register_tune_param_check_supported(op_type):
    """
    register tune param check supported

    Parameters
    ----------
    op_type : string
        op type

    Returns
    -------
    decorator : decorator
        decorator to set fusion build config
    """

    if op_type is None:
        raise RuntimeError("register tune param check supported failed, op_type is none")

    def decorator(func):
        """
        Parameters
        ----------
        func : function
            func
        Returns
        -------
        wrapper : wrapper
            wrapper to exce func
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            """
            Parameters
            ----------
            args : list
            kwargs : dict
            Returns
            -------
            wrapper : func
                func to exce func
            """

            return func(*args, **kwargs)
        _tune_param_check[op_type] = wrapper
        return wrapper
    return decorator


def get_tune_param_check_supported(op_type):
    """
    :return:
    """

    if op_type is None:
        raise RuntimeError("register tune param check supported failed, op_type is none")
    return _tune_param_check.get(op_type)
