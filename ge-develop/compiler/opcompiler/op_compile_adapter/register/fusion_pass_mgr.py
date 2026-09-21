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
fusion pass manager
"""
import functools

from tbe.common.register.class_manager import InvokeStage
from tbe.common.register.class_manager import Priority
from tbe.common.register.class_manager import FusionPassItem

# ub fusion pass list, input is op list json
_fusion_passes = []

# fusion pass list for ub fusion, input is graph info
g_fusion_pass_list = []

# pass list for op param
g_op_param_pass_list = []

def register_fusion_pass(stage=InvokeStage.STAGE_BUILD, priority=Priority.NORMAL):
    """
    register op UB fusion pass

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
    global _fusion_passes
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        func_name = func.__name__
        fusion_pass = FusionPassItem(func_name, stage, priority, wrapper)
        import tbe.common.utils.log as logger
        logger.info("Fusion pass[%s] of stage[%s] priority[%s] has been registered.",
                    func_name, stage, priority)
        _fusion_passes.append(fusion_pass)
        return wrapper
    return decorator


def get_all_fusion_pass():
    """
    :return:
    """
    _fusion_passes.sort(key=lambda mgr: mgr.get_priority())
    return _fusion_passes


def register_pass_for_fusion(match_condition, stage, priority):
    """
    register fusion pass

    Parameters
    ----------
    match_condition : dict
        contain op_type and op_pattern
    stage : int
        op function executed stage.
    priority : int
        op function executed priority.

    Returns
    -------
    decorator : decorator
        decorator to register fusion pass func
    """

    if stage is None or priority is None:
        raise RuntimeError("Fail to register fusion pass, stage or priority should not be none.")

    op_type = match_condition.get("op_type")
    op_pattern = match_condition.get("op_pattern")
    if op_type is None and op_pattern is None:
        raise RuntimeError("Fail to register fusion pass, match condition should contain op type or op pattern")

    global g_fusion_pass_list
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        func_name = func.__name__
        pass_item = FusionPassItem(func_name, stage, priority, wrapper)
        pass_item.set_op_type(op_type)
        pass_item.set_op_pattern(op_pattern)
        pass_item.set_op_type_list(op_type)
        pass_item.set_op_pattern_list(op_pattern)
        g_fusion_pass_list.append(pass_item)
        import tbe.common.utils.log as logger
        logger.info("Fusion pass[%s] of op type[%s] op pattern[%s] stage[%s] priority[%s] has been registered.",
                    func_name, op_type, op_pattern, stage, priority)
        return wrapper
    return decorator


def get_fusion_pass_list_by_stage(stage=InvokeStage.STAGE_BUILD):
    """
    get fusion pass list of specified stage

    Parameters
    ----------
    stage : int
        op function executed stage.

    Returns
    -------
    pass_list : list
        fusion pass list of specified stage
    """
    import tbe.common.utils.log as logger
    pass_list = []
    if stage is None:
        logger.warn("Stage should not be none.")
        return pass_list
    for pass_item in g_fusion_pass_list:
        if pass_item.get_stage() == stage:
            pass_list.append(pass_item)

    pass_list.sort(key=lambda item: item.get_priority())
    import tbe.common.utils.log as logger
    logger.info("Get fusion pass list, size of stage[%s] is [%d].", stage, len(pass_list))
    return pass_list


def register_op_param_pass(match_condition, stage):
    """
    register op param pass

    Parameters
    ----------
    match_condition : dict
        contain op_type and op_pattern
    stage : int
        op function executed stage.

    Returns
    -------
    decorator : decorator
        decorator to register op param pass func
    """

    if stage is None:
        raise RuntimeError("Stage should not be none.")

    op_type = match_condition.get("op_type")
    op_pattern = match_condition.get("op_pattern")
    if op_type is None and op_pattern is None:
        raise RuntimeError("Fail to register op param pass, match condition should contain op type or op pattern")

    if op_type is not None and get_op_param_pass_cond(op_type, None, stage) is not None:
        raise RuntimeError("This pass will not be registered for op type[%s] stage[%s] has already been registered.",
                           op_type, stage)

    if op_pattern is not None and get_op_param_pass_cond(None, op_pattern, stage) is not None:
        raise RuntimeError("This pass will not be registered for op pattern[%s] stage[%s] has already been registered.",
                           op_pattern, stage)

    global g_op_param_pass_list
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        func_name = func.__name__
        pass_item = FusionPassItem(func_name, stage, Priority.NORMAL, wrapper)
        pass_item.set_op_type(op_type)
        pass_item.set_op_pattern(op_pattern)
        pass_item.set_op_type_list(op_type)
        pass_item.set_op_pattern_list(op_pattern)
        g_op_param_pass_list.append(pass_item)
        import tbe.common.utils.log as logger
        logger.info("Op param pass[%s] of op type[%s] op pattern[%s] stage[%s] has been registered.",
                    func_name, op_type, op_pattern, stage)
        return wrapper
    return decorator


def get_op_param_pass_cond(op_type, op_pattern, stage=InvokeStage.STAGE_BUILD):
    """
    get op param pass list by some conditions, op type, op pattern and stage

    Parameters
    ----------
    op_type : str
        op type
    op_pattern : str
        op pattern
    stage : int
        op function executed stage.

    Returns
    -------
    pass_list : list
        op param pass list
    """
    import tbe.common.utils.log as logger
    if stage is None:
        logger.warn("Stage should not be none.")
        return None

    if op_type is not None:
        for pass_item in g_op_param_pass_list:
            if pass_item.get_stage() == stage and pass_item.get_op_type() == op_type:
                logger.debug("Op param pass is found by op type[%s] and stage[%s].", op_type, stage)
                return pass_item

    if op_pattern is not None:
        for pass_item in g_op_param_pass_list:
            if pass_item.get_stage() == stage and pass_item.get_op_pattern() == op_pattern:
                logger.debug("Op param pass is found by op pattern[%s] and stage[%s].", op_pattern, stage)
                return pass_item

    logger.debug("Op param pass is not found by op type[%s] op pattern[%s] and stage[%s].",
                 op_type, op_pattern, stage)
    return None
