#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import time
import unittest
import os
from typing import List, Optional

import numpy as np

from llm_datadist_v1 import *
from llm_datadist_v1.llm_datadist import _shutdown_handler
from llm_datadist_v1.llm_utils import TransferCacheJob, TransferCacheParameters
import json

from llm_datadist_v1.llm_types import KvCache, BlocksCacheKey, Placement
from llm_datadist_v1.config import EngineConfig

_INVALID_ID = 2 ** 64 - 1

_TEST_BASE_DIR = '../tests/dflow/llm_datadist/st/testcase/llm_datadist'

class LlmEngineV2Ut(unittest.TestCase):
    def setUp(self) -> None:
        os.environ['ASCEND_GLOBAL_LOG_LEVEL'] = '1'
        print("Begin ", self._testMethodName)

    def tearDown(self) -> None:
        os.environ.pop('RESOURCE_CONFIG_PATH', None)
        _shutdown_handler()
        print("End ", self._testMethodName)

    @staticmethod
    def _engine_options(is_prompt: bool, cluster_id: int = 0, rank_id: int = -1, resource_path: str = ''):
        cluster_info = {
            'cluster_id': cluster_id, 'logic_device_id': ['0:0:0:0', '0:0:1:0', '0:0:2:0', '0:0:3:0'],
        }
        if is_prompt:
            cluster_info['listen_ip_info'] = [{'ip': 0, 'port': 26000}, {'ip': 1, 'port': 26000},
                                              {'ip': 2, 'port': 26000},
                                              {'ip': 3, 'port': 26000}]
        engine_options = {'llm.ClusterInfo': json.dumps(cluster_info)}
        if rank_id != -1:
            engine_options['ge.exec.rankId'] = str(rank_id)
        if resource_path != '':
            engine_options['ge.resourceConfigPath'] = resource_path
        return engine_options

    def test_simple_option(self):
        cluster_id = 0
        prompt_engine = LLMDataDist(LLMRole.PROMPT, cluster_id)
        llm_config = LLMConfig()
        llm_config.device_id = 1
        llm_config.listen_ip_info = "127.0.0.1:26000"
        llm_config.deploy_res_path = "./"
        llm_config.ge_options = {
            "ge.flowGraphMemMaxSize": "10000000"
        }
        engine_options = llm_config.generate_options()
        print("engine_options:", engine_options)
        prompt_engine.init(engine_options)
        
    def test_check_flow_graph_mem_max_size(self):
        cluster_id = 0
        llm_engine = LLMDataDist(LLMRole.PROMPT, cluster_id)
        llm_config = LLMConfig()
        llm_config.device_id = 0
        llm_config.listen_ip_info = "0.0.0.0:26000"
        llm_config.ge_options = {"ge.flowGraphMemMaxSize": "-1"}
        init_options = llm_config.generate_options()

        has_err = False
        try:
            llm_engine.init(init_options)
        except LLMException:
            has_err = True
        self.assertEqual(has_err, True)

    def test_check_flow_graph_mem_max_size2(self):
        cluster_id = 0
        llm_engine = LLMDataDist(LLMRole.PROMPT, cluster_id)
        llm_config = LLMConfig()
        llm_config.device_id = 0
        llm_config.listen_ip_info = "0.0.0.0:26000"
        llm_config.ge_options = {"llm.EnableCacheManager": "0"}
        init_options = llm_config.generate_options()
        has_err = False
        try:
            llm_engine.init(init_options)
        except LLMException:
            has_err = True
        self.assertEqual(has_err, False)