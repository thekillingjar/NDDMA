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

def generate_identity_pb():
    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
        x = tf.compat.v1.placeholder(dtype="int32", shape=())
        x_plus_1 = tf.add(x, 1, name='x_plus')
        with tf.control_dependencies([x_plus_1]):
            y = x
            z = tf.identity(x,name='identity')
    tf.io.write_graph(sess.graph, logdir="./", name="test_identity.pb", as_text=False)

if __name__=='__main__':
    generate_identity_pb()
