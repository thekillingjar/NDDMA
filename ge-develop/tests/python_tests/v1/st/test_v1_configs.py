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
from llm_datadist_v1.configs import LlmConfig


class LlmConfigSt(unittest.TestCase):
    def setUp(self):
        print("Begin ", self._testMethodName)
        self._llm_config = LlmConfig()
    
    def tearDown(self):
        print("End ", self._testMethodName)

    def testEnableSwitchRole(self):
        self._llm_config.enable_switch_role = True
        self.assertEqual(self._llm_config.enable_switch_role, True)

    def testEnableCacheManager(self):
        self._llm_config.enable_cache_manager = True
        self.assertEqual(self._llm_config.enable_cache_manager, True)

    def testEnableRemoteCacheAccessible(self):
        self._llm_config.enable_remote_cache_accessible = True
        self.assertEqual(self._llm_config.enable_remote_cache_accessible, True)
    
    def testMemPoolCfg(self):
        self._llm_config.mem_pool_cfg = "mem_pool_cfg"
        self.assertEqual(self._llm_config.mem_pool_cfg, "mem_pool_cfg")
    
    def testHostMemPoolCfg(self):
        self._llm_config.host_mem_pool_cfg = "host_mem_pool_cfg"
        self.assertEqual(self._llm_config.host_mem_pool_cfg, "host_mem_pool_cfg")
