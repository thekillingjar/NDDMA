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
operation function manager
"""
import functools

from tbe import tvm
import tbe.dsl as tbe
import tbe.common.platform as tbe_platform
from tbe.common.register.class_manager import Operator
from tbe.common.register.class_manager import OpCompute
from tbe.common.register.class_manager import OpClassify


# op compute func dict
_op_computes = {}
# op realization func dict
_operators = {}
# op param generalization func dict
# 'pylint: disable=C0103
_generalization = {}
_op_register_pattern = {}
_op_type_classifys = {}
_op_pattern_classifys = {}
_op_no_trans_bool_to_s8 = {}


def cast_in_to_fp32(ori_inputs):
    """
    cast input dtype from bfloat16 to float32
    """
    def need_cast_to_fp32(tensor):
        """
        check platform support bfloat16
        """
        tensor_dtype = tensor.dtype.lower()
        return  tensor_dtype == "bfloat16" and \
                not tbe_platform.api_check_support("tbe.dsl.vadd", "bfloat16") and \
                tbe_platform.intrinsic_check_support("Intrinsic_vconv", "bf162f32")

    from tbe.dsl.base import operation

    inputs = []
    for ori_input in ori_inputs:
        # support single inputs
        if operation.is_generic_tensor(ori_input) and need_cast_to_fp32(ori_input):
            ori_input = tbe.cast_to(ori_input, "float32")
            inputs.append(ori_input)
        # support multi inputs
        elif isinstance(ori_input, list):
            new_input = []
            for _input in ori_input:
                if isinstance(_input, tvm.Tensor) and need_cast_to_fp32(_input):
                    _input = tbe.cast_to(_input, "float32")
                    new_input.append(_input)
                else:
                    new_input.append(_input)
            inputs.append(new_input)
        elif isinstance(ori_input, dict) and "dtype" in ori_input and ori_input["dtype"] == "bfloat16":
            new_input = ori_input.copy()
            new_input["dtype"] = "float32"
            inputs.append(new_input)
        else:
            inputs.append(ori_input)
    return inputs


def cast_res_to_bf16(ori_outputs, ori_args):
    """
    cast output dtype from float32 to bfloat16
    """
    def float32_process(data):
        """
        deal with src dtype float32 case
        """
        data_dtype = data.dtype.lower()
        if data_dtype == "float32":
            data = tbe.round(data, "bfloat16")
        elif data_dtype == "float16":
            data = tbe.cast_to(data, "float32")
            data = tbe.round(data, "bfloat16")

        return data

    from tbe.dsl.base import operation

    single_res = False
    # check tensor and list case
    if operation.is_generic_tensor(ori_outputs) or not hasattr(ori_outputs, "index"):
        output_tensors = [ori_outputs]
        single_res = True
    else:
        output_tensors = list(ori_outputs)

    output_dtypes = []
    for param in ori_args:
        if isinstance(param, dict) and "dtype" in param:
            output_dtypes.append(param["dtype"])

    if len(output_tensors) != len(output_dtypes):
        return ori_outputs

    res_tensors = []
    for idx, output_tensor in enumerate(output_tensors):
        dtype = output_dtypes[idx]
        if dtype == "bfloat16" and output_tensor.dtype != dtype:
            res_tensors.append(float32_process(output_tensor))
        else:
            res_tensors.append(output_tensor)

    if single_res:
        return res_tensors[0]
    else:
        return res_tensors


def register_op_compute(op_type, op_mode="dynamic", support_fusion=True, **kwargus):
    """
    register op compute func

    Parameters
    ----------
    op_type: string
        op_func_name(old process) or op type(new process)
    op_mode: string
        dynamic or static shape
    support_fusion: bool
        support dynamic shape UB fusion
    fusion_pattern: string
        undefined(default) or func(diff shape, diff pattern) or  Elewise/Conv/...
    **kwargus:
        - support_bfp16: support bfloat16 dtype

    Returns
    -------
    decorator : decorator
        decorator to register compute func
    support_bfp16 case:
        decorator to return output_tensor
    """
    if op_type is None:
        raise RuntimeError("register op compute failed, op_type is none")
    global _op_computes
    global _op_register_pattern
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if "support_bfp16" in kwargus.keys():
                new_args = cast_in_to_fp32(args)
                tmp_res = func(*new_args, **kwargs)
                return cast_res_to_bf16(tmp_res, args)
            else:
                return func(*args, **kwargs)

        if "fusion_pattern" in kwargus.keys():
            _op_register_pattern[op_type] = kwargus["fusion_pattern"]
        elif op_type in _op_register_pattern:
            _op_register_pattern.pop(op_type)
        _op_computes[(op_type, op_mode)] = OpCompute(support_fusion, wrapper)
        return wrapper
    return decorator


def get_op_compute(op_type, op_mode="dynamic"):
    """
    :return:
    """
    if op_type is None:
        raise RuntimeError("get op compute failed, op_type is none")
    return _op_computes.get((op_type, op_mode))


def register_operator(op_type, pattern=None, trans_bool_to_s8=True):
    """
    register op realization func

    Parameters
    ----------
    op_type : string
        op type
    pattern: string
        op fusion pattern
    trans_bool_to_s8: bool
        if need trans bool to int8
    Returns
    -------
    decorator : decorator
        decorator to register realization func
    """
    if op_type is None:
        raise RuntimeError("register operator failed, op_type is none")
    global _op_no_trans_bool_to_s8
    if not trans_bool_to_s8:
        _op_no_trans_bool_to_s8[op_type] = "True"
    global _operators
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        _operators[op_type] = Operator(pattern, wrapper)
        return wrapper
    return decorator


def get_operator(op_type):
    """
    :return:
    """
    if op_type is None:
        raise RuntimeError("get operator failed, op_type is none")
    return _operators.get(op_type)


def register_param_generalization(op_type):
    """
    register op param generalization func

    Parameters
    ----------
    op_type : string
        op type

    Returns
    -------
    decorator : decorator
        decorator to register generalization func
    """
    if op_type is None:
        raise RuntimeError("register generalization func failed, op_type is none")
    global _generalization
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        _generalization[op_type] = wrapper
        return wrapper
    return decorator


def get_param_generalization(op_type):
    """
    :return:
    """
    if op_type is None:
        raise RuntimeError("get generalization func failed, op_type is none")
    return _generalization.get(op_type)


def register_classify_processor(pattern=None, op_type=None, support_type="input"):
    """
    register op classify func

    Parameters
    ----------
    pattern : fusion_patern
    op_type : op_type
    support_type: string
        input or all(include input, attrs, options)

    Returns
    -------
    decorator : decorator
        decorator to register classify func
    """
    if pattern is None and op_type is None:
        raise RuntimeError("Register op classify processor failed, pattern or op_type is none!")
    support_types = ("input", "all")
    if support_type not in support_types:
        raise RuntimeError("Unsupported type, it must be "
                           "one of ('input', 'all') and the data type "
                           "must be string ")
    global _op_pattern_classifys
    global _op_type_classifys
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        if pattern is not None:
            _op_pattern_classifys[pattern] = OpClassify(pattern, support_type, wrapper)
        if op_type is not None:
            _op_type_classifys[op_type] = OpClassify(op_type, support_type, wrapper)
        return wrapper
    return decorator


def get_classify_processor(pattern, op_type):
    """
    :return:
    """
    if pattern is None and op_type is None:
        raise RuntimeError("Get op classify processor failed, pattern or op_type is none")

    classify_func = None
    if op_type is not None:
        classify_func = _op_type_classifys.get(op_type)
    if classify_func is not None:
        return classify_func

    if pattern is not None:
        classify_func = _op_pattern_classifys.get(pattern)
    return classify_func


def get_op_register_pattern():
    global _op_register_pattern
    return _op_register_pattern


def get_op_no_trans_bool_to_s8():
    global _op_no_trans_bool_to_s8
    return _op_no_trans_bool_to_s8


def is_no_need_trans_bool_to_s8(op_type):
    global _op_no_trans_bool_to_s8
    if op_type in _op_no_trans_bool_to_s8.keys():
        return _op_no_trans_bool_to_s8[op_type] == "True"
    else:
        return False