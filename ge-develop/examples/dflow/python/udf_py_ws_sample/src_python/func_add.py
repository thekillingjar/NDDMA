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

import dataflow.flow_func as ff
import dataflow.data_type as dt
import numpy as np


class Add():
    def __init__(self):
        self.count = 0

    @ff.init_wrapper()
    def init_flow_func(self, meta_params):
        return ff.FLOW_FUNC_SUCCESS

    @ff.proc_wrapper("i0,i1,o0")
    def add1(self, run_context, input_flow_msgs):
        return ff.FLOW_FUNC_SUCCESS

    @ff.proc_wrapper("i2,i3,o1")
    def add2(self, run_context, input_flow_msgs):
        return ff.FLOW_FUNC_SUCCESS
