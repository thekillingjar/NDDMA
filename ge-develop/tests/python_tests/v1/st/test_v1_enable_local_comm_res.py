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
import time

from llm_datadist_v1 import *


class LlmCacheManagerSt(unittest.TestCase):
    def setUp(self) -> None:
        print("Begin ", self._testMethodName)
        config = LlmConfig()
        config.device_id = 0
        config.enable_switch_role = True
        config.rdma_service_level = 100
        config.rdma_traffic_class = 100
        config.listen_ip_info = "127.0.0.1:26008"
        config.local_comm_res = '''
        {
            "server_count": "1",
            "server_list": [{
                "device": [{
                "device_id": "0",
                "device_ip": "1.1.1.1"
                }],
                "server_id": "127.0.0.1"
            }],
            "status": "completed",
            "version": "1.0"
        }
        '''
        engine_options = config.generate_options()
        self.llm_datadist = LLMDataDist(LLMRole.PROMPT, 1)
        self.llm_datadist.init(engine_options)
        time.sleep(1) # wait listen
        self.has_exception = False

    def tearDown(self) -> None:
        print("End ", self._testMethodName)
        self.llm_datadist.finalize()

    def create_link_cluster(self):
        cluster = LLMClusterInfo()
        cluster.remote_cluster_id = 1
        cluster.append_local_ip_info("127.0.0.1", 26008)
        cluster.append_remote_ip_info("127.0.0.1", 26008)
        ret, rets = self.llm_datadist.link_clusters([cluster], 5000)
        self.assertEqual(ret, LLMStatusCode.LLM_SUCCESS)

    def test_unlink_cluster(self):
        self.llm_datadist.switch_role(LLMRole.DECODER)
        self.llm_datadist._enable_cache_mgr = False
        self.create_link_cluster()
        cluster = LLMClusterInfo()
        cluster.remote_cluster_id = 1
        cluster.append_local_ip_info("127.0.0.1", 26008)
        cluster.append_remote_ip_info("127.0.0.1", 26008)
        ret, rets = self.llm_datadist.unlink_clusters([cluster], 5000)
        self.assertEqual(ret, LLMStatusCode.LLM_SUCCESS)

    def test_local_comm_res_switch_role(self):
        try:
            self.llm_datadist.switch_role(LLMRole.DECODER)
            options = { 'llm.listenIpInfo': '127.0.0.1:26008'}
            self.llm_datadist.switch_role(LLMRole.PROMPT, options)
        except Exception as e:
            print(f"{type(e).__name__} - {str(e)}")
            import traceback
            print(traceback.format_exc())
            self.has_exception = True
        self.assertEqual(self.has_exception, False)