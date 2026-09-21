#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

# 生成用于构图的add.pb
def generate_tensorflow_graph():
    import tensorflow as tf
    import os

    tf.compat.v1.disable_eager_execution()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    graph_path = os.path.join(current_dir, "add.pb")
    try:
        os.remove(graph_path)
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"delete {graph_path} occurred exception: {e}", flush=True)

    input_shape = [3]
    input1 = tf.compat.v1.placeholder(tf.float32, shape=input_shape, name='Placeholder')
    input2 = tf.compat.v1.placeholder(tf.float32, shape=input_shape, name='Placeholder_1')
    output = tf.add(input1, input2, name="add")
    with tf.compat.v1.Session() as sess:
        tf.io.write_graph(tf.compat.v1.get_default_graph(), current_dir, 'add.pb', as_text=False)
        print(f"tensor model saved to {graph_path}", flush=True)


# 生成用于构图的简单的onnx模型
def generate_onnx_model():
    import onnx
    from onnx import helper, TensorProto
    import os

    current_dir = os.path.dirname(os.path.abspath(__file__))
    onnx_path = os.path.join(current_dir, "simple_model.onnx")
    try:
        os.remove(onnx_path)
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"delete {onnx_path} occurred exception: {e}", flush=True)

    input_shape = [3]
    X1 = helper.make_tensor_value_info('X1', TensorProto.INT32, input_shape)
    X2 = helper.make_tensor_value_info('X2', TensorProto.INT32, input_shape)
    add_node = helper.make_node(
        'Add',
        inputs=['X1', 'X2'],
        outputs=['Y'])
    output_shape = input_shape  # 输出形状与输入相同
    Y = helper.make_tensor_value_info('Y', TensorProto.INT32, output_shape)
    graph_def = helper.make_graph(
        [add_node],
        'simple_add_model',
        [X1, X2],
        [Y], )
    model_def = helper.make_model(graph_def, producer_name='onnx-example', opset_imports=[helper.make_opsetid("", 11)])
    onnx.checker.check_model(model_def)
    onnx.save(model_def, onnx_path)
    print(f"ONNX model saved to {onnx_path}", flush=True)


if __name__ == "__main__":
    generate_tensorflow_graph()
    generate_onnx_model()
