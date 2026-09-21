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
from tensorflow.python.framework import graph_util
import numpy as np

def generate_merge_pb():
    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
        dist = tf.compat.v1.placeholder(tf.float32, [100])
        tf.compat.v1.summary.histogram(name="Merge", values=dist)
        writer = tf.compat.v1.summary.FileWriter("./tf_summary_merge_pb")
        op = tf.compat.v1.summary.merge_all()
        for step in range(10):
            mean_moving_normal = np.random.normal(loc=step, scale=1, size=[100])
            summ = sess.run(op, feed_dict = {dist : mean_moving_normal})
            writer.add_summary(summ, global_step=step)
        tf.io.write_graph(sess.graph, logdir="./", name="merge.pb", as_text=False)

if __name__=='__main__':
    generate_merge_pb()
