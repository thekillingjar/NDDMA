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

__all__ = ["assert_args_type"]

import inspect
from inspect import signature
from functools import wraps
from typing import Dict, Union, List, Tuple, Optional
from dataflow.flow_func import flow_func as ff
from dataflow.utils.msg_type_register import msg_type_register
import dataflow.dflow_wrapper as dwrapper

_global_type_checker_functions = {}

_global_running_device_id = None
_global_running_instance_id = None
_global_running_instance_num = None
_global_running_in_udf = False
_global_msg_type_register = msg_type_register


class DfException(Exception):
    def __init__(self, obj, error_code: int = None):
        super().__init__()
        self.message = obj
        self.error_code = (
            error_code if error_code is not None else dwrapper.PARAM_INVALID
        )

    def __str__(self):
        return f"DfException:{self.error_code}:{self.message}"

    def __repr__(self):
        return f"DfException:{self.error_code}:{self.message}"


class DfAbortException(DfException):
    def __init__(self, obj, error_code: int = None):
        super().__init__(obj, error_code)


def _register_type_checker_function(type_key, type_check_function):
    _global_type_checker_functions[type_key] = type_check_function


def _is_instance(value, dtype):
    if isinstance(dtype, type):
        if not isinstance(value, dtype):
            return False
    elif hasattr(dtype, "__origin__") and (
            dtype.__origin__ in _global_type_checker_functions
    ):
        return _global_type_checker_functions[dtype.__origin__](value, dtype)
    else:
        raise TypeError(f"The type {value} check of data {dtype} is not supported")
    return True


def _check_dict_key_value_type(arg, dtype):
    if not isinstance(arg, Dict):
        return False
    dict_key_value_type = dtype.__args__
    for key, value in arg.items():
        if not _is_instance(key, dict_key_value_type[0]) or not _is_instance(
                value, dict_key_value_type[1]
        ):
            return False
    return True


_register_type_checker_function(Dict.__origin__, _check_dict_key_value_type)


def _check_union_type(arg, dtype):
    union_types = dtype.__args__
    for union_type in union_types:
        if _is_instance(arg, union_type):
            return True
    return False


_register_type_checker_function(Union, _check_union_type)


def _check_list_value_type(arg, dtype):
    if not isinstance(arg, List):
        return False
    list_types = dtype.__args__
    if len(list_types) < 1:
        return False
    list_type = list_types[0]
    for value in arg:
        if not _is_instance(value, list_type):
            return False
    return True


_register_type_checker_function(List.__origin__, _check_list_value_type)


def assert_args_type(*type_args, **type_kwargs):
    def decorate(func):
        sig = signature(func)
        args_types = sig.bind_partial(*type_args, **type_kwargs).arguments

        @wraps(func)
        def wrapper(*args, **kwargs):
            args_values = sig.bind(*args, **kwargs)
            for arg, value in args_values.arguments.items():
                if arg in args_types:
                    if not _is_instance(value, args_types[arg]):
                        raise DfException(f"Argument {arg} must be {args_types[arg]}")
            return func(*args, **kwargs)

        return wrapper

    return decorate


def set_running_device_id(running_device_id):
    global _global_running_device_id
    _global_running_device_id = running_device_id


def get_running_device_id():
    logger = ff.FlowFuncLogger()
    if _global_running_device_id is None:
        logger.error("running device id is not set")
    return _global_running_device_id


def set_running_instance_id(running_instance_id):
    global _global_running_instance_id
    _global_running_instance_id = running_instance_id


def get_running_instance_id():
    logger = ff.FlowFuncLogger()
    if _global_running_instance_id is None:
        logger.error("running instance id is not set")
    return _global_running_instance_id


def set_running_instance_num(running_instance_num):
    global _global_running_instance_num
    _global_running_instance_num = running_instance_num


def get_running_instance_num():
    logger = ff.FlowFuncLogger()
    if _global_running_instance_num is None:
        logger.error("running instance num is not set")
    return _global_running_instance_num


def set_running_in_udf():
    global _global_running_in_udf
    _global_running_in_udf = True


def get_running_in_udf():
    return _global_running_in_udf


def set_msg_type_register(type_register):
    global _global_msg_type_register
    _global_msg_type_register = type_register


def get_msg_type_register():
    return _global_msg_type_register


def convert_flow_msg_to_object(flow_msg):
    if int(flow_msg.get_msg_type()) == int(ff.MSG_TYPE_TENSOR_DATA):
        return flow_msg.get_tensor()
    elif get_msg_type_register().registered(flow_msg.get_msg_type()):
        deserialize_func = get_msg_type_register().get_deserialize_func(
            flow_msg.get_msg_type()
        )
        obj = deserialize_func(flow_msg.get_raw_data())
        return obj
    else:
        return flow_msg.get_raw_data()


def get_param_count(func):
    sig = inspect.signature(func)
    return len([param for param in sig.parameters if param not in ("self", "cls")])


def get_typing_num_returns(func):
    sig = inspect.signature(func)
    if sig.return_annotation != inspect.Signature.empty:
        if hasattr(sig.return_annotation, "__args__"):
            return len(sig.return_annotation.__args__)
        else:
            return 1
    else:
        return -1
