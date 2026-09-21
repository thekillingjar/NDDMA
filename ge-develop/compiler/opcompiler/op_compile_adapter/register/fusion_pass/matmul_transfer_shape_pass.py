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
matmul trans shape fusion pass
"""


import functools
from tbe.common.register import register_pass_for_fusion
from tbe.common.platform import platform_info


def reshape_bias_shape(shape_bias):
    """
    tansform matmul bias op shape
    :param shape_bias:
    """
    bias_length = 1
    for i in shape_bias:
        bias_length *= i
    support_out2l1_nd2nz = platform_info.intrinsic_check_support("Intrinsic_data_move_out2l1_nd2nz")
    if not support_out2l1_nd2nz:
        return [(bias_length + 15) // 16 * 16]
    return [bias_length]


def trans_shape_by_index(index, op_info, graph_info):
    if len(op_info.get_input_list()) <= index:
        return
    input_info = op_info.get_input_list()[index]
    peer_output_info = graph_info.get_data_output_by_id(input_info.get_edge_id())

    if peer_output_info is None:
        return
    input_shape = peer_output_info.get_shape()
    if len(input_shape) > 0:
        new_shape = reshape_bias_shape(input_shape)
        peer_output_info.set_shape(new_shape)


@register_pass_for_fusion(match_condition={"op_type":"MatMul"})
@register_pass_for_fusion(match_condition={"op_type":"MatMulV2"})
def trans_matmul_bias_shape(graph_info):
    """
    tansform matmul bias op shape if necessary
    original func :
    Parameters
    ----------
    graph_info : GraphInfo
        graph info

    Returns
    -------
    """
    if graph_info.is_dynamic_shape():
        return
    for op_info in graph_info.get_op_list():
        if op_info is None:
            continue
        if op_info.get_op_type() in ["MatMul", "MatMulV2"]:
            trans_shape_by_index(2, op_info, graph_info)


@register_pass_for_fusion(match_condition={"op_type":"MatMulV2Compress"})
def trans_matmulcompress_bias_shape(graph_info):
    """
    tansform matmulcompress bias op shape if necessary. input tensor order is
    x, w, compress_index, b, and so on
    original func :
    Parameters
    ----------
    graph_info : GraphInfo
        graph info

    Returns
    -------
    """
    if graph_info.is_dynamic_shape():
        return
    for op_info in graph_info.get_op_list():
        if op_info is None:
            continue
        if op_info.get_op_type() == "MatMulV2Compress":
            trans_shape_by_index(3, op_info, graph_info)


@register_pass_for_fusion(match_condition={"op_type":"BatchMatMul"})
@register_pass_for_fusion(match_condition={"op_type":"BatchMatMulV2"})
def trans_batch_matmul_shape(graph_info):
    """
    transform batch_matmul op shape if necessary which is not supported by dynamic
    original func :
    Parameters
    ----------
    graph_info : GraphInfo
        graph info

    Returns
    -------
    """
    if graph_info.is_dynamic_shape():
        return

    for op_info in graph_info.get_op_list():
        if op_info is None:
            continue
        if op_info.get_op_type() in ["BatchMatMul", "BatchMatMulV2"]:
            for idx, input_info in enumerate(op_info.get_input_list()):
                if idx > 3:
                    continue
                peer_output_info = graph_info.get_data_output_by_id(input_info.get_edge_id())
                if peer_output_info is None:
                    continue
                peer_out_shape = peer_output_info.get_shape()
                peer_out_format = peer_output_info.get_format()
                shape_len = 3 if peer_out_format == "ND" else 5
                if len(peer_out_shape) > shape_len:
                    new_shape = [functools.reduce(
                        lambda x, y: x * y, peer_out_shape[:-(shape_len-1)])] + peer_out_shape[-(shape_len-1):]
                    peer_output_info.set_shape(new_shape)
                    peer_output_info.set_only_update_data()


def trans_shape_of_transdata(op_info, graph_info):
    """
    transform trans_data shape before batchmatmul if necessary which is not supported by dynamic
    :param op_info:
    :param graph_info:
    """
    if len(op_info.get_input_list()) == 0:
        return
    input_info = op_info.get_input_list()[0]
    peer_output_info = graph_info.get_data_output_by_id(input_info.get_edge_id())
    if peer_output_info is None:
        return
    peer_out_shape = peer_output_info.get_shape()
    if len(peer_out_shape) > 3:
        new_shape = [functools.reduce(lambda x, y: x * y, peer_out_shape[:-2])] + peer_out_shape[-2:]
        peer_output_info.set_shape(new_shape)


def trans_shape_of_fixpipe(op_info, graph_info):
    """
    transform fixpipe shape after batchmatmul if necessary which is not supported by dynamic
    :param data_node:
    :param node:
    """
    for input_info in op_info.get_input_list():
        peer_output_info = graph_info.get_data_output_by_id(input_info.get_edge_id())
        if peer_output_info is None:
            continue
        peer_out_shape = peer_output_info.get_shape()
        if len(peer_out_shape) > 5:
            new_shape = [functools.reduce(lambda x, y: x * y, peer_out_shape[:-4])] + peer_out_shape[-4:]
            peer_output_info.set_shape(new_shape)


@register_pass_for_fusion(match_condition={"op_pattern":"BatchMatmul"})
def trans_shape_transdata_fixpipe(graph_info):
    """
    transform shape of TransData and Fixpipe while BatchMatmul is in op list
    original func :
    Parameters
    ----------
    graph_info : GraphInfo
        graph info

    Returns
    -------
    """
    if graph_info.is_dynamic_shape():
        return

    for op_info in graph_info.get_op_list():
        if op_info is None:
            continue
        if op_info.get_op_type() == "TransData":
            trans_shape_of_transdata(op_info, graph_info)
        if op_info.get_op_type() == "FixPipe":
            trans_shape_of_fixpipe(op_info, graph_info)
