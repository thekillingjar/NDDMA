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
# ----------------------------------------------------------------------------------------------

"""
FullyConnection trans shape fusion pass
"""


from tbe.common.register import register_pass_for_fusion


def trans_shape_by_index(index, op_info, graph_info):
    if index >= len(op_info.get_input_list()):
        return
    input_info = op_info.get_input_list()[index]
    peer_output_info = graph_info.get_data_output_by_id(input_info.get_edge_id())
    if peer_output_info is None:
        return
    input_shape = peer_output_info.get_shape()
    if len(input_shape) == 5:
        if index == 0:
            new_shape = [input_shape[0], input_shape[1] * input_shape[2] * input_shape[3] * input_shape[4]]
            peer_output_info.set_shape(new_shape)
        if index == 2 or index == 3:
            new_shape = [input_shape[1] * input_shape[4]]
            peer_output_info.set_shape(new_shape)


@register_pass_for_fusion(match_condition={"op_type":"FullyConnection"})
def trans_shape_fully_connection(graph_info):
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
        if op_info is None or op_info.get_op_type() != "FullyConnection":
            continue
        trans_shape_by_index(0, op_info, graph_info)
        trans_shape_by_index(2, op_info, graph_info)


@register_pass_for_fusion(match_condition={"op_type":"FullyConnectionCompress"})
def trans_shape_fully_connection_compress(graph_info):
    """
    tansform fullyconnectioncompress op shape, input tensor order is
    x, w, compress_index, b, and so on
    original func : trans_shape_fullycompress
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
        if op_info is None or op_info.get_op_type() != "FullyConnectionCompress":
            continue
        trans_shape_by_index(0, op_info, graph_info)
        trans_shape_by_index(3, op_info, graph_info)
