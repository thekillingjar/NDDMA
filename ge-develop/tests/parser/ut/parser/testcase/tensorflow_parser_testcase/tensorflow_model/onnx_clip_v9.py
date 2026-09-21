#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#-------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto


def make_clip_V9():
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [3, 4, 5])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3, 4, 5])
    node_def = helper.make_node('Clip',
                                inputs=['X'],
                                outputs=['Y'],
                                max = 1.0,
                                min = -1.0,
                                )
    graph = helper.make_graph(
        [node_def],
        "test_clip_case_V9",
        [X],
        [Y],
    )

    model = helper.make_model(graph, producer_name="onnx-mul_test")
    model.opset_import[0].version = 9
    onnx.save(model, "./onnx_clip_v9.onnx")


if __name__ == '__main__':
    make_clip_V9()
