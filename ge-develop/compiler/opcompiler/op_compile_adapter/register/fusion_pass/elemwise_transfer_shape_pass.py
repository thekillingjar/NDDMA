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
elemwise trans shape fusion pass
"""


from tbe.common.register import register_pass_for_fusion


def trans_shape_by_pattern(graph_info, pattern):
    """
    broadcast elemwise input shape if necessary
    original func : trans_elemwise_shape
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
        if op_info.get_pattern() in ("QuantConvolution", "Convolution", "Conv2d_backprop_input", 'Conv3D'):
            return

    for op_info in graph_info.get_op_list():
        if op_info.get_pattern() != pattern:
            continue
        max_input_dim = 0
        for input_info in op_info.get_input_list():
            peer_output_info = graph_info.get_peer_output_by_id(input_info.get_edge_id())
            if peer_output_info is not None:
                max_input_dim = max(len(peer_output_info.get_shape()), max_input_dim)
        for input_info in op_info.get_input_list():
            peer_data_output_info = graph_info.get_data_output_by_id(input_info.get_edge_id())
            if peer_data_output_info is not None and not peer_data_output_info.is_shape_null():
                data_shape_size = len(peer_data_output_info.get_shape())
                if data_shape_size <= 1:
                    new_shape = [1] * (max_input_dim - data_shape_size) + peer_data_output_info.get_shape()
                    peer_data_output_info.set_shape(new_shape)


@register_pass_for_fusion(match_condition={"op_pattern":"ElemWise"})
def trans_shape_for_elemwise(graph_info):
    trans_shape_by_pattern(graph_info, "ElemWise")


@register_pass_for_fusion(match_condition={"op_pattern":"Broadcast"})
def trans_shape_for_broadcast(graph_info):
    trans_shape_by_pattern(graph_info, "Broadcast")


def trans_shape_input_bias(op_info, graph_info):
    """
    transform shape for biasadd
    """
    input_x = op_info.get_input_list()[0]
    input_bias = op_info.get_input_list()[1]
    if input_bias.get_edge_type() != "Data":
        return
    peer_output_x = graph_info.get_peer_output_by_id(input_x.get_edge_id())
    peer_output_bias = graph_info.get_peer_output_by_id(input_bias.get_edge_id())
    attr_data_format = op_info.get_attr_by_name("data_format")
    is_param_invalid = peer_output_x is None or peer_output_bias is None or attr_data_format is None
    if is_param_invalid:
        return
    shape_x = peer_output_x.get_shape()
    format_x = peer_output_x.get_format()
    data_format = attr_data_format.get_value().upper()
    shape_bias = ()
    if format_x is not None and format_x == "NC1HWC0":
        shape_bias = (1, shape_x[1], 1, 1, shape_x[4])
    elif format_x is not None and format_x == "NDHWC":
        shape_bias = (1, 1, 1, 1, shape_x[4])
    elif format_x is not None and format_x == "NCDHW":
        shape_bias = (1, shape_x[1], 1, 1, 1)
    elif format_x is not None and format_x == "NDC1HWC0":
        shape_bias = (1, 1, shape_x[2], 1, 1, shape_x[5])
    elif data_format == "NCHW":
        shape_bias = (1, shape_x[1],)
        for _ in range(2, len(shape_x)):
            shape_bias = shape_bias + (1,)
    else:
        shape_bias = ()
        for _ in range(0, len(shape_x) - 1):
            shape_bias = shape_bias + (1,)
        shape_bias = shape_bias + (shape_x[-1],)

    peer_output_bias.set_shape(shape_bias)


@register_pass_for_fusion(match_condition={"op_type":"BiasAdd"})
def trans_shape_of_biasadd(graph_info):
    """
    broadcast elemwise input shape if necessary
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

    # all nodes must be elemwise
    for op_info in graph_info.get_op_list():
        if op_info.get_op_type() != "Data" and op_info.get_pattern() not in ["ElemWise", "Broadcast", "quant"]:
            return

    for op_info in graph_info.get_op_list():
        if op_info.get_op_type() != "BiasAdd":
            continue
        if len(op_info.get_input_list()) < 2:
            continue
        trans_shape_input_bias(op_info, graph_info)


@register_pass_for_fusion(match_condition={"op_type":"AscendRequantS16"})
def trans_shape_of_requant_s16(graph_info):
    """
    transform AscendRequantS16 shape
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
        if op_info.get_op_type() != "AscendRequantS16":
            continue
        for idx, input_info in enumerate(op_info.get_input_list()):
            if input_info.get_edge_type() != "Data":
                continue
            if idx in [0, 2]:
                peer_output_info = graph_info.get_peer_output_by_id(input_info.get_edge_id())
                if peer_output_info is None:
                    continue
                peer_shape = peer_output_info.get_shape()
                if len(peer_shape) == 5:
                    new_shape = [peer_shape[0], peer_shape[1], peer_shape[2] * peer_shape[3], peer_shape[4]]
                    peer_output_info.set_shape(new_shape)


@register_pass_for_fusion(match_condition={"op_type":"BNInferenceD"})
def trans_shape_of_bninference(graph_info):
    """
    transform AscendRequantS16 shape
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
        if op_info.get_op_type() != "BNInferenceD":
            continue
        for idx, input_info in enumerate(op_info.get_input_list()):
            peer_output_info = graph_info.get_data_output_by_id(input_info.get_edge_id())
            if peer_output_info is None:
                continue
            peer_shape = peer_output_info.get_shape()
            if idx in [1, 2]:
                new_shape = []
                if peer_output_info.get_format() == "NHWC":
                    new_shape = [1, 1, 1] + peer_shape
                    peer_output_info.set_shape(new_shape)
                if peer_output_info.get_format() == "NCHW":
                    new_shape = [1] + peer_shape + [1, 1]
                    peer_output_info.set_shape(new_shape)
            if idx == 0:
                if peer_output_info.get_format() == "NCHW" and len(peer_shape) == 3:
                    new_shape = peer_shape + [1]
                    peer_output_info.set_shape(new_shape)

