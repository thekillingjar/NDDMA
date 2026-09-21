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
tbe register API:
To make it easy to manage operator registration information, TBE provides a set of register APIs.
"""
from tbe.common.register import fusion_build_config_mgr
from tbe.common.register import fusion_pass_mgr
from tbe.common.register import operation_func_mgr
from tbe.common.register.class_manager import InvokeStage
from tbe.common.register.class_manager import Priority
from tbe.common.register import auto_tune_func_mgr


def register_tune_param_check_supported(op_type):
    """
    register op tune func

    Parameters
    ----------
    op_type : string

    Returns
    -------
    decorator : decorator
        decorator to register tune func
    """

    return auto_tune_func_mgr.register_tune_param_check_supported(op_type)


def get_tune_param_check_supported(op_type):
    """
    get op tune func info

    Parameters
    ----------
    op_type : string

    Returns
    -------
    func
    """

    return auto_tune_func_mgr.get_tune_param_check_supported(op_type)


def register_tune_space(op_type):
    """
    register op tune func

    Parameters
    ----------
    op_type : string

    Returns
    -------
    decorator : decorator
        decorator to register tune func
    """

    return auto_tune_func_mgr.register_tune_space(op_type)


def get_tune_space(op_type):
    """
    get op tune func info

    Parameters
    ----------
    op_type : string

    Returns
    -------
    func
    """

    return auto_tune_func_mgr.get_tune_space(op_type)


def register_op_compute(op_type, op_mode="dynamic", support_fusion=True, **kwargs):
    """
    register op compute func

    Parameters
    ----------
    op_type : string
        op_func_name(old process) or op type(new process)
    op_mode: string
        dynamic or static shape
    support_fusion: bool
        support dynamic shape UB fusion
    **kwargus:
        - support_bfp16: support bfloat16 dtype

    Returns
    -------
    decorator : decorator
        decorator to register compute func
    support_bfp16 case:
        decorator to return output_tensor
    """

    return operation_func_mgr.register_op_compute(op_type, op_mode, support_fusion, **kwargs)


def get_op_compute(op_type, op_mode="dynamic"):
    """
    get op compute func info

    Parameters
    ----------
    op_type : string
        op_func_name(old process) or op type(new process)
    op_mode: string
        dynamic or static shape

    Returns
    -------
    object of class OpCompute
    """

    return operation_func_mgr.get_op_compute(op_type, op_mode)


def register_operator(op_type, pattern=None, trans_bool_to_s8=True):
    """
    register op realization func

    Parameters
    ----------
    op_type : string
        op type
    pattern: string
        op fusion pattern

    Returns
    -------
    decorator : decorator
        decorator to register realization func
    """

    return operation_func_mgr.register_operator(op_type, pattern, trans_bool_to_s8)


def get_operator(op_type):
    """
    get op realization func info

    Parameters
    ----------
    op_type : string
        op_func_name(old process) or op type(new process)

    Returns
    -------
    object of class Operator
    """

    return operation_func_mgr.get_operator(op_type)


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
        decorator to register realization func
    """

    return operation_func_mgr.register_param_generalization(op_type)


def get_param_generalization(op_type):
    """
    get op param generalization func

    Parameters
    ----------
    op_type : string

    Returns
    -------
    op param generalization func
    """

    return operation_func_mgr.get_param_generalization(op_type)


def register_fusion_pass(stage=InvokeStage.STAGE_BUILD, priority=Priority.NORMAL):
    """
    register op UB fusion pass func info

    Parameters
    ----------
    stage : int
        op function executed stage.
    priority : int
        op function executed priority.

    Returns
    -------
    decorator : decorator
        decorator to register fusion pass func
    """

    return fusion_pass_mgr.register_fusion_pass(stage, priority)


def get_all_fusion_pass():
    """
    get all fusion_pass func info

    Parameters
    ----------

    Returns
    -------
    object of class FusionPassItem
    """

    return fusion_pass_mgr.get_all_fusion_pass()


def register_pass_for_fusion(match_condition, stage=InvokeStage.STAGE_BUILD, priority=Priority.NORMAL):
    return fusion_pass_mgr.register_pass_for_fusion(match_condition, stage, priority)


def register_op_param_pass(match_condition, stage=InvokeStage.STAGE_BUILD):
    return fusion_pass_mgr.register_op_param_pass(match_condition, stage)


def set_fusion_buildcfg(op_type, build_config):
    """
    register fusion build config

    Parameters
    ----------
    op_type : string
        op type
    config: dict
        build configs

    Returns
    -------
    decorator : decorator
        decorator to set fusion build config
    """

    fusion_build_config_mgr.set_fusion_buildcfg(op_type, build_config)


def get_fusion_buildcfg(op_type=None):
    """
    get fusion build config

    Parameters
    ----------
    op_type : string
        op type

    Returns
    -------
    object of class FusionBuildCfg
    """

    return fusion_build_config_mgr.get_fusion_buildcfg(op_type)


def reset():
    """
    reset register info

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    fusion_build_config_mgr.reset_fusion_buildcfg()


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
    return operation_func_mgr.register_classify_processor(pattern, op_type, support_type)


def get_classify_processor(pattern=None, op_type=None):
    """
    :return:
    """
    return operation_func_mgr.get_classify_processor(pattern, op_type)


def get_op_register_pattern():
    """
    :return:
    """
    return operation_func_mgr.get_op_register_pattern()
