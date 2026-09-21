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
    except Exception as e:
        print(f"delete {graph_path} occurred exception: {e}", flush=True)

    input_shape = [3]
    input1 = tf.compat.v1.placeholder(tf.float32, shape=input_shape, name='Placeholder')
    input2 = tf.compat.v1.placeholder(tf.float32, shape=input_shape, name='Placeholder_1')
    output = tf.add(input1, input2, name="add")
    with tf.compat.v1.Session() as sess:
        tf.io.write_graph(tf.compat.v1.get_default_graph(), current_dir, 'add.pb', as_text=False)
        print(f"tensor model saved to {graph_path}", flush=True)


if __name__ == "__main__":
    generate_tensorflow_graph()
