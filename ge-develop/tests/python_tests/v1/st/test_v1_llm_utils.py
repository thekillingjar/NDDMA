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

import unittest
from llm_datadist_v1.llm_utils import layer_range_to_tensor_indices, parse_listen_ip_info


class LlmUtilsSt(unittest.TestCase):
    def setUp(self):
        print("Begin ", self._testMethodName)
    
    def tearDown(self):
        print("End ", self._testMethodName)

    def testLayerRangeToTensorIndices(self):
        src_layer_range = range(1, 2)
        dst_layer_range = range(1, 2)
        result = layer_range_to_tensor_indices(src_layer_range, dst_layer_range)
        self.assertEqual(result[0], [2, 3])
        self.assertEqual(result[1], [2, 3])
    
    def testParseListenIpInfo(self):
        listen_ip_info = "22:33"
        result = parse_listen_ip_info(listen_ip_info)
        self.assertEqual(result[0], "22")
        self.assertEqual(result[1], 33)