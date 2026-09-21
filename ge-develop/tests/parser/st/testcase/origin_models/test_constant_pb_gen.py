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

import tensorflow as tf
import numpy as np
from tensorflow.python.framework import graph_util

def generate_constant_pb():
    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
        grads_1 = tf.compat.v1.placeholder(dtype="float16", shape=(1,1,2,2,2))
        grads_2 = tf.compat.v1.placeholder(dtype="float16", shape=(1,1,2,2,2))
        grads = tf.add(grads_1, grads_2)
        orig_input_shape = tf.constant(np.array([1,1,3,3,3]).astype("int32"), )
        op = tf.raw_ops.AvgPool3DGrad(orig_input_shape=orig_input_shape,
                                      grad=grads,
                                      ksize=[1,1,2,2,2],
                                      strides=[1,1,1,1,1],
                                      padding="VALID",
                                      data_format='NCDHW',
                                      name='AvgPool3DGrad')
        tf.io.write_graph(sess.graph, logdir="./", name="test_constant.pb", as_text=False)

if __name__ == "__main__":
    generate_constant_pb()
