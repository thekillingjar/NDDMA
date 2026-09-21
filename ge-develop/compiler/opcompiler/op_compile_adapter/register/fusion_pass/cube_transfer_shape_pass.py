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
cube trans shape fusion pass
"""


from tbe.common.register import register_pass_for_fusion
from tbe.common.register import register_fusion_pass


@register_pass_for_fusion(match_condition={"op_type":"Conv3D"})
def trans_shape_of_conv3d(graph_info):
    """
    tranform shape for Conv3D
    original func : trans_conv3d_shape
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
        if op_info.get_op_type() == "Conv3D":
            if len(op_info.get_input_list()) < 2:
                continue
            input_info = op_info.get_input_list()[1]
            if not input_info.is_data_edge():
                continue
            peer_output_info = graph_info.get_peer_output_by_id(input_info.get_edge_id())
            if peer_output_info is None:
                continue
            peer_out_shape = peer_output_info.get_shape()
            if len(peer_out_shape) == 4:
                new_shape = [peer_out_shape[0], peer_out_shape[1], peer_out_shape[2], peer_out_shape[3]]
                peer_output_info.set_shape(new_shape)


def is_dynamic_shape(op_list):
    if op_list is None or len(op_list) == 0:
        return False
    for op_info in op_list:
        if "output_desc" not in op_info:
            continue
        for output_desc in op_info["output_desc"]:
            output_shape = output_desc.get("shape")
            if output_shape is None or len(output_shape) == 0 or output_shape == "NULL":
                continue
            for dim in output_shape:
                if int(dim) < 0:
                    return True

    return False


@register_fusion_pass()
def trans_shape_depthwise_conv2d(op_list):
    """
    tranform shape for depthwise_conv2d
    original func : trans_depthwise_conv2d
    Parameters
    ----------
    op_list : dict
        op list dict

    Returns
    -------
    """
    if is_dynamic_shape(op_list):
        return

    data_output_dict = {}
    for op_info in op_list:
        if op_info["type"] == "Data":
            for output_desc in op_info["output_desc"]:
                data_output_dict[output_desc["name"]] = output_desc

    for op_info in op_list:
        if op_info["type"] == "DepthwiseConv2D":
            for idx, input_desc in enumerate(op_info["input_desc"]):
                data_output = data_output_dict.get(input_desc["name"])
                if data_output is None:
                    continue
                shape = data_output["shape"]
                total_shape = data_output["total_shape"]
                valid_shape = data_output["valid_shape"]
                if idx == 0 and len(shape) == 5:
                    data_output["shape"] = [shape[0], shape[1], shape[2], shape[3], shape[4]]
                    data_output["total_shape"] = [total_shape[0], total_shape[1], 1,
                                                  total_shape[2], total_shape[3], total_shape[4]]
                    input_desc["shape"] = data_output["shape"]
                    input_desc["total_shape"] = data_output["total_shape"]
                    if not any(valid_shape):
                        data_output["valid_shape"] = [valid_shape[0], valid_shape[1], 1,
                                                      valid_shape[2], valid_shape[3], valid_shape[4]]
                        input_desc["valid_shape"] = data_output["valid_shape"]
                if idx == 1 and len(shape) == 6:
                    data_output["shape"] = [shape[0], shape[1], shape[2], shape[4], shape[5]]
                    data_output["total_shape"] = [total_shape[0], total_shape[1], total_shape[2],
                                                  total_shape[4], total_shape[5]]
                    input_desc["shape"] = data_output["shape"]
                    input_desc["total_shape"] = data_output["total_shape"]
                    if not any(valid_shape):
                        data_output["valid_shape"] = [valid_shape[0], valid_shape[1],
                                                      valid_shape[2], valid_shape[4], valid_shape[5]]
                        input_desc["valid_shape"] = data_output["valid_shape"]
