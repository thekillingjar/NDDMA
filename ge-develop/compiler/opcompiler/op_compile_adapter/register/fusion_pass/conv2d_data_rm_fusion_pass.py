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
conv2d_data_rm_build_pass
"""
import copy
import time
from tbe.common import register
from tbe.common.platform import platform_info

CONV = "Conv2D"
QUANTCONV = "QuantConv2D"
DATA = "Data"
STRIDED_READ = "StridedRead"
DATA_RM = "conv2d_data_rm"
TRANS_DATA = "TransData"

DEQUANT_PATTERN = "dequant"
QUANT_PATTERN = "quant"
ELTWISE_PATTERN = "ElemWise"
STRIDED_WRITE_PATTERN = "strided_write"
STRIDED_READ_PATTERN = "strided_read"

INPUT_DESC = "input_desc"
OUTPUT_DESC = "output_desc"
PATTERN_STR = "pattern"
TYPE_STR = "type"
NAME_STR = "name"
L1_FUSION_TYPE = "L1_fusion_type"
FUNC_NAME = "func_name"
INDEX = "id"
OUTPUT_INDEX = "output_index"
ORIGINAL_NAME = "ori_name"
INVALID_PATTERN = "Opaque"
PREBUILD_ATTRS = "prebuild_outs_attrs"
KWDS_ARGS = "kwds_args"
LIST_ARGS = "list_args"
MODULE_NAME = "module_name"
OP_MODULE = "impl.conv2d_data_rm"
OP_LIST_KEY_OPTIONS = "options"
PARAM_KEY_DATA_RM = "invalid_data_rm"
MULTI_REFERED_FLAG = ":"
IS_DYNAMIC_IMPL = "is_dynamic_impl"
DYN_FLAG = "dyn_flag"

NAME_PREFIX_FORMAT_STR = "ori_{}"
NAME_SUFFIX_FORMAT_STR = "{}__0"
MULTI_REFERED_FLAG_FORMAT_STR = "{}:{}"
DATA_RM_FORMAT_STR = "conv2d_data_rm_{}"

L1_FUSION_DISABLE = -1
ELTWISE_INPUT_SIZE_MAX = 1
OUTPUT_DESC_SIZE_MAX = 2
HEAD_STRUCT_OP_SIZE_ONE = 1
HEAD_STRUCT_OP_SIZE_TWO = 2
DOUBLE_OUTPUT = 2
SINGLE_OUTPUT = 1
INDEX_0 = 0

RET_CONTINUE = 0
RET_FALSE = 1
RET_TRUE = 2


def get_next_op(op_list, current_op):
    """
    get next node based on the current node

    Parameters
    ----------
    op_list:list
    op info of fusion op
    current_op:dict
    one op in op_list

    Returns
    -------
    next op of current op
    """
    for output_desc in current_op.get(OUTPUT_DESC, []):
        if not output_desc:
            continue
        for op_node in op_list:
            if op_node.get(TYPE_STR) == DATA:
                continue
            for input_desc in op_node.get(INPUT_DESC, []):
                if output_desc.get(NAME_STR) is not None and output_desc.get(NAME_STR) == input_desc.get(NAME_STR):
                    return op_node
    return {}


def get_pre_op(op_list, current_op):
    """
    get pre node based on the current node

    Parameters
    ----------
    op_list:list
    op info of fusion op
    current_op:dict
    one op in op_list

    Returns
    -------
    pre op of current op
    """
    for input_desc in current_op.get(INPUT_DESC, []):
        for op_node in op_list:
            for output_desc in op_node.get(OUTPUT_DESC, []):
                if not output_desc:
                    continue
                if output_desc.get(NAME_STR) is not None and output_desc[NAME_STR] == input_desc[NAME_STR] and\
                    op_node.get(TYPE_STR) is not None and op_node[TYPE_STR] != DATA:
                    return op_node
    return {}


def is_tail_op(op_list, current_op):
    """
    get next node based on the current node

    Parameters
    ----------
    op_list:list
    op info of fusion op
    current_op:dict
    one op in op_list

    Returns
    -------
    next op of current op
    """
    tail_op_flag = True
    for output_desc in current_op.get(OUTPUT_DESC, []):
        if not output_desc:
            continue
        for op_node in op_list:
            if op_node.get(TYPE_STR) == DATA:
                continue
            for input_desc in op_node.get(INPUT_DESC, []):
                if output_desc.get(NAME_STR) is not None and output_desc[NAME_STR] == input_desc[NAME_STR]:
                    tail_op_flag = False
    return tail_op_flag


def is_support_fixpipe():
    """
    Check fixpipe support.
    """
    return platform_info.intrinsic_check_support("Intrinsic_fix_pipe_unit_list")


def validity_pre_check(op_list, current_op, strided_read_flag, tail_op, tail_op_list):
    """
    op validity preCheck whether it meets the conditions

    Parameters
    ----------
    op_list:list
    op info of fusion op
    current_op:dict
    one op in op_list
    strided_read_flag:bool
    whether has strided_read in struct
    tail_op:dict
    first op of common tail_struct
    tail_op_list:list
    all last ops in op_list

    Returns
    -------
    True or False
    """
    if is_support_fixpipe():
        return RET_FALSE

    double_output_oplist = (DEQUANT_PATTERN, ELTWISE_PATTERN)
    for output_desc in current_op.get(OUTPUT_DESC, []):
        if not output_desc:
            continue
        if L1_FUSION_TYPE in output_desc and output_desc[L1_FUSION_TYPE] != L1_FUSION_DISABLE:
            return RET_FALSE
    if current_op.get(TYPE_STR) == DATA:
        return RET_CONTINUE
    if current_op.get(PATTERN_STR) == ELTWISE_PATTERN and len(current_op.get(INPUT_DESC, [])) > ELTWISE_INPUT_SIZE_MAX:
        return RET_FALSE
    if len(current_op.get(OUTPUT_DESC, [])) == OUTPUT_DESC_SIZE_MAX:
        if current_op.get(PATTERN_STR, INVALID_PATTERN) not in double_output_oplist or strided_read_flag:
            return RET_FALSE
        tail_op_list.append(current_op)
        if is_tail_op(op_list, current_op):
            tail_op.update(current_op)
        return RET_CONTINUE
    return RET_TRUE


def validity_final_check(head_struct_oplist, tail_op_list, strided_write_flag):
    """
    op validity final Check whether it meets the conditions

    Parameters
    ----------
    head_struct_oplist:list
    head struct ops in op_list
    tail_op_list:list
    all last ops in op_list
    strided_read_flag:bool
    whether has strided_read in struct

    Returns
    -------
    True or False
    """
    head_oplist = (CONV, STRIDED_READ)
    if strided_write_flag and len(tail_op_list) == DOUBLE_OUTPUT:
        return False
    if len(head_struct_oplist) == HEAD_STRUCT_OP_SIZE_ONE and\
        head_struct_oplist[0].get(TYPE_STR) == CONV:
        return True
    if len(head_struct_oplist) == HEAD_STRUCT_OP_SIZE_TWO and\
        head_struct_oplist[0].get(TYPE_STR) in head_oplist and\
        head_struct_oplist[1].get(TYPE_STR) in head_oplist:
        return True
    return False


# 'pylint: disable=invalid-name, too-many-return-statements, too-many-branches
def total_struct_check(op_list, head_op, tail_op, tail_op_list):
    """
    check struct of op_list whether it meets the conditions

    Parameters
    ----------
    op_list:list
    op info of fusion op
    head_op:dict
    last op of common head_struct
    tail_op:dict
    first op of common tail_struct
    head_struct_oplist:list
    head struct ops in op_list
    tail_op_list:list
    all last ops in op_list

    Returns
    -------
    True or False
    """
    tail_oplist = (QUANT_PATTERN, ELTWISE_PATTERN, STRIDED_WRITE_PATTERN)
    strided_read_flag = False
    strided_write_flag = False
    head_struct_oplist = []
    for op in op_list:
        if OUTPUT_DESC in op:
            ret = validity_pre_check(op_list, op, strided_read_flag, tail_op, tail_op_list)
            if ret == RET_CONTINUE:
                continue
            if ret == RET_FALSE:
                return False
            next_op = get_next_op(op_list, op)
            if op.get(TYPE_STR) == STRIDED_READ:
                strided_read_flag = True
                if next_op and next_op.get(TYPE_STR) == CONV:
                    head_op.update(op)
                    head_struct_oplist.append(op)
                else:
                    return False
            if op.get(TYPE_STR) == CONV:
                head_op.update(op)
                head_struct_oplist.append(op)
            if not next_op:
                if op.get(PATTERN_STR, INVALID_PATTERN) not in tail_oplist and op.get(TYPE_STR) != TRANS_DATA:
                    return False
                if op.get(PATTERN_STR, INVALID_PATTERN) == STRIDED_WRITE_PATTERN:
                    strided_write_flag = True
                tail_op_list.append(op)
                tail_op.update(op)
    return validity_final_check(head_struct_oplist, tail_op_list, strided_write_flag)


def pattern1_match(op_list, head_op, tail_op):
    """
    check pattern: (stridedread) + conv + dequant + eltwise*N + (stridedwrite)

    Parameters
    ----------
    op_list:list
    op info of fusion op
    head_op:dict
    last op of common head_struct
    tail_op:dict
    first op of common tail_struct

    Returns
    -------
    True or False
    """
    if head_op.get(PATTERN_STR, INVALID_PATTERN) != DEQUANT_PATTERN or\
        tail_op.get(PATTERN_STR, INVALID_PATTERN) != ELTWISE_PATTERN:
        return False
    while 1:
        next_op = get_next_op(op_list, head_op)
        if not next_op or next_op.get(PATTERN_STR, INVALID_PATTERN) != ELTWISE_PATTERN:
            return False
        if next_op.get(NAME_STR) is not None and next_op.get(NAME_STR) == tail_op.get(NAME_STR):
            return True
        head_op = next_op


def pattern2_match(op_list, head_op, tail_op):
    """
    check pattern: (stridedread) + conv + (dequant) + (eltwise*N) + quant + (stridedwrite)

    Parameters
    ----------
    op_list:list
    op info of fusion op
    head_op:dict
    last op of common head_struct
    tail_op:dict
    first op of common tail_struct

    Returns
    -------
    True or False
    """
    if tail_op.get(PATTERN_STR, INVALID_PATTERN) != QUANT_PATTERN:
        return False
    while 1:
        next_op = get_next_op(op_list, head_op)
        if not next_op or (next_op.get(PATTERN_STR, INVALID_PATTERN) != ELTWISE_PATTERN and\
                           next_op.get(PATTERN_STR, INVALID_PATTERN) != QUANT_PATTERN):
            return False
        if next_op.get(NAME_STR) is not None and next_op.get(NAME_STR) == tail_op.get(NAME_STR):
            return True
        head_op = next_op


def pattern3_match(op_list, head_op, tail_op):
    """
    check pattern: conv + dequant + (eltwise*N) + quant
    double output: one output from quant, the other from dequant or eltwise

    Parameters
    ----------
    op_list:list
    op info of fusion op
    head_op:dict
    last op of common head_struct
    tail_op:dict
    first op of common tail_struct

    Returns
    -------
    True or False
    """
    if head_op.get(PATTERN_STR, INVALID_PATTERN) != DEQUANT_PATTERN or\
        tail_op.get(PATTERN_STR, INVALID_PATTERN) != QUANT_PATTERN:
        return False
    while 1:
        next_op = get_next_op(op_list, head_op)
        if not next_op or (next_op.get(PATTERN_STR, INVALID_PATTERN) != ELTWISE_PATTERN and\
                           next_op.get(PATTERN_STR, INVALID_PATTERN) != QUANT_PATTERN):
            return False
        if next_op.get(NAME_STR) is not None and next_op.get(NAME_STR) == tail_op.get(NAME_STR):
            return True
        head_op = next_op


def pattern4_match(op_list:list, head_op:dict, tail_op:dict) -> bool:
    """
    check pattern: conv + transdata

    Parameters
    ----------
    op_list:list
    op info of fusion op
    head_op:dict
    last op of common head_struct
    tail_op:dict
    first op of common tail_struct

    Returns
    -------
    True or False
    """
    if head_op.get(TYPE_STR) != CONV or tail_op.get(TYPE_STR) != TRANS_DATA:
        return False

    op_num = 1
    while 1:
        next_op = get_next_op(op_list, head_op)
        if next_op:
            head_op = next_op
            op_num += 1
        else:
            break
    if op_num == 2:
        return True
    return False


def can_user_define_compute(op_list, tail_op_list):
    """
    check the op_list whether it meets the conditions

    Parameters
    ----------
    op_list:list
    op info of fusion op
    tail_op_list:list
    all last ops in op_list

    Returns
    -------
    True or False
    """
    head_op = {}
    tail_op = {}
    if total_struct_check(op_list, head_op, tail_op, tail_op_list):
        if head_op:
            next_op = get_next_op(op_list, head_op)
            if next_op.get(PATTERN_STR, INVALID_PATTERN) == DEQUANT_PATTERN:
                head_op = next_op
        if tail_op and tail_op.get(PATTERN_STR, INVALID_PATTERN) == STRIDED_WRITE_PATTERN:
            tail_op = get_pre_op(op_list, tail_op)

        if len(tail_op_list) == SINGLE_OUTPUT:
            if pattern1_match(op_list, head_op, tail_op):
                return True
            if pattern2_match(op_list, head_op, tail_op):
                return True
            if pattern4_match(op_list, head_op, tail_op):
                return True
        if len(tail_op_list) == DOUBLE_OUTPUT:
            if pattern3_match(op_list, head_op, tail_op):
                return True
    return False


def set_rm_in_options(op_list, rm_flag=False):
    """
    set invalid_data_rm in options as True or False

    Parameters
    ----------
    op_list:list
    op info of fusion op
    rm_flag:bool
    whether user define compute

    Returns
    -------
    None
    """
    for op_node in op_list:
        if op_node.get(TYPE_STR) in [CONV, QUANTCONV]:
            op_node[OP_LIST_KEY_OPTIONS] = {PARAM_KEY_DATA_RM: rm_flag}
            return


# 'pylint: disable=invalid-name, too-many-nested-blocks, too-many-branches
def add_rm_op_in_op_list(op_list, tail_op_list):
    """
    add remove op in oplist

    Parameters
    ----------
    op_list:list
    op info of fusion op
    tail_op_list:list
    all last ops in op_list

    Returns
    -------
    None
    """
    for last_op in tail_op_list:
        rm_op = copy.deepcopy(last_op)
        del rm_op[INPUT_DESC]
        del rm_op[OUTPUT_DESC]
        name = DATA_RM_FORMAT_STR.format(str(time.time()))
        if len(last_op.get(OUTPUT_DESC, [])) == DOUBLE_OUTPUT:
            last_op_desc_flag = True
            for output_desc in last_op.get(OUTPUT_DESC):
                if not output_desc:
                    continue
                for op in op_list:
                    if INPUT_DESC not in op:
                        continue
                    for input_desc in op[INPUT_DESC]:
                        if output_desc.get(NAME_STR) == input_desc.get(NAME_STR):
                            last_op_desc_flag = False
                            break
                if last_op_desc_flag:
                    rm_op[INPUT_DESC] = [copy.deepcopy(output_desc)]
                    rm_op[OUTPUT_DESC] = [copy.deepcopy(output_desc)]
                    if output_desc.get(NAME_STR).find(MULTI_REFERED_FLAG) == -1:
                        output_desc[NAME_STR] = NAME_SUFFIX_FORMAT_STR.format(name)
                    else:
                        _, name_suffix = output_desc.get(NAME_STR).split(MULTI_REFERED_FLAG, 1)
                        output_desc[NAME_STR] = MULTI_REFERED_FLAG_FORMAT_STR.format(name, name_suffix)
                    rm_op[INPUT_DESC][0][NAME_STR] = output_desc[NAME_STR]
                last_op_desc_flag = True
        else:
            rm_op[INPUT_DESC] = copy.deepcopy(last_op.get(OUTPUT_DESC))
            rm_op[OUTPUT_DESC] = copy.deepcopy(last_op.get(OUTPUT_DESC))
            last_op[OUTPUT_DESC][0][NAME_STR] = NAME_SUFFIX_FORMAT_STR.format(name)
            rm_op[INPUT_DESC][0][NAME_STR] = last_op[OUTPUT_DESC][0][NAME_STR]
        # op no dynamic impl, only support dynamic and static shape
        rm_op[IS_DYNAMIC_IMPL] = rm_op.get(DYN_FLAG, True)
        rm_op[FUNC_NAME] = DATA_RM
        rm_op[INDEX] = rm_op[INDEX] * 2
        del rm_op[INPUT_DESC][0][OUTPUT_INDEX]
        rm_op[OUTPUT_DESC][0][OUTPUT_INDEX] = INDEX_0
        rm_op[MODULE_NAME] = OP_MODULE
        rm_op[NAME_STR] = name
        if not rm_op.get(ORIGINAL_NAME):
            rm_op[ORIGINAL_NAME] = [NAME_PREFIX_FORMAT_STR.format(name)]
        else:
            rm_op[ORIGINAL_NAME][0] = NAME_PREFIX_FORMAT_STR.format(name)
        rm_op[PATTERN_STR] = INVALID_PATTERN
        rm_op[TYPE_STR] = DATA_RM
        op_list.append(rm_op)


@register.register_fusion_pass()
def conv2d_data_rm_build_pass(op_list):
    """
    user-defined compute if necessary

    Parameters
    ----------
    op_list:list
    op info of fusion op

    Returns
    -------
    None
    """
    tail_op_list = []
    res = can_user_define_compute(op_list, tail_op_list)
    if res:
        set_rm_in_options(op_list, True)
        add_rm_op_in_op_list(op_list, tail_op_list)
    else:
        set_rm_in_options(op_list, False)
