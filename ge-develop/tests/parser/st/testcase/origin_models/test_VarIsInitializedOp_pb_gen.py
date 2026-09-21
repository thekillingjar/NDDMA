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

def generate_VarIsInitializedOp_pb():
    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
        x = tf.compat.v1.placeholder(dtype="int32", shape=())
        y = tf.Variable(tf.compat.v1.random_normal(shape=[4,3],mean=0,stddev=1), dtype="float32", name='y')
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)
        op = tf.compat.v1.raw_ops.VarIsInitializedOp(resource=y, name="VarIsInitializedOp")
    tf.io.write_graph(sess.graph, logdir="./", name="test_VarIsInitializedOp.pb", as_text=False)

if __name__=='__main__':
    generate_VarIsInitializedOp_pb()
