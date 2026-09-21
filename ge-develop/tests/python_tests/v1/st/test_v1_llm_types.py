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
from llm_datadist_v1 import MemInfo, Memtype, CacheKeyByIdAndIndex, TransferWithCacheKeyConfig, TransferConfig


class MemInfoSt(unittest.TestCase):
    def setUp(self):
        print("Begin ", self._testMethodName)
        self._memoinfo = MemInfo(Memtype.MEM_TYPE_DEVICE, 0, 1)
    
    def tearDown(self) -> None:
        print("End ", self._testMethodName)

    def testMemType(self):
        self.assertEqual(self._memoinfo.mem_type, Memtype.MEM_TYPE_DEVICE)

    def testAddr(self):
        self.assertEqual(self._memoinfo.addr, 0)
    
    def testSize(self):
        self.assertEqual(self._memoinfo.size, 1)

class TransferWithCacheKeyConfigSt(unittest.TestCase):
    def setUp(self):
        print("Begin ", self._testMethodName)
        cache_key = CacheKeyByIdAndIndex(1, 1)
        src_layer_range = range(1, 5)
        dst_layer_range = range(1, 5)
        self._twckc = TransferWithCacheKeyConfig(cache_key, src_layer_range, dst_layer_range)

    def tearDown(self) -> None:
        print("End ", self._testMethodName)
    
    def testCacheKey(self):
        self.assertEqual(self._twckc.cache_key.cluster_id, 1)
        self._twckc.cache_key = CacheKeyByIdAndIndex(2, 1)
        self.assertEqual(self._twckc.cache_key.cluster_id, 2)

    def testSrcLayerRange(self):
        self.assertEqual(self._twckc.src_layer_range.start, 1)
        self._twckc.src_layer_range = range(2, 5)
        self.assertEqual(self._twckc.src_layer_range.start, 2)
    
    def testDstLayerRange(self):
        self.assertEqual(self._twckc.dst_layer_range.start, 1)
        self._twckc.dst_layer_range = range(2, 5)
        self.assertEqual(self._twckc.dst_layer_range.start, 2)

    def testSrcBatchIndex(self):
        self.assertEqual(self._twckc.src_batch_index, 0)
        self._twckc.src_batch_index = 1
        self.assertEqual(self._twckc.src_batch_index, 1)

class TransferConfigSt(unittest.TestCase):
    def setUp(self):
        print("Begin ", self._testMethodName)
        dst_addrs = [1, 2, 3, 4]
        self._transfer_config = TransferConfig(1, dst_addrs, range(1, 5), 0)

    def tearDown(self) -> None:
        print("End ", self._testMethodName)

    def testDstClusterId(self):
        self._transfer_config.dst_cluster_id = 5
        self.assertEqual(self._transfer_config.dst_cluster_id, 5)
    
    def testSrcBatchIndex(self):
        self._transfer_config.src_batch_index = 10
        self.assertEqual(self._transfer_config.src_batch_index, 10)
    
