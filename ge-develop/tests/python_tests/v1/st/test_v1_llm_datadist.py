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

class LayerSynchronizerImpl(LayerSynchronizer):
    def __init__(self, ret=True):
        self._ret = ret

    def synchronize_layer(self, layer_index: int, timeout_in_millis: Optional[int]) -> bool:
        if layer_index == 0:
            time.sleep(0.3)
        return self._ret


class MockTransferCacheJob(TransferCacheJob):
    def __init__(self, params: TransferCacheParameters) -> None:
        super().__init__(params, LayerSynchronizerImpl(True), None)

    def transfer_layer(self, src_layer_index: int, dst_layer_idx, transfer_config: TransferConfig) -> LLMStatusCode:
        return LLMStatusCode.LLM_WAIT_PROCESS_TIMEOUT

class LlmEngineV2St(unittest.TestCase):
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

    def test_prompt_cache_ops(self):
        cluster_id = 0
        engine = LLMDataDist(LLMRole.PROMPT, cluster_id)
        engine.init(LlmEngineV2St._engine_options(True, cluster_id))
        kv_cache_manager = engine.kv_cache_manager
        cache_desc = CacheDesc(80, [2, 8], DataType.DT_FLOAT16)
        cache_key = CacheKey(0, 0, 1)
        kv_cache = kv_cache_manager.allocate_cache(cache_desc, [cache_key])
        with self.assertRaises(LLMException):
            tensors = kv_cache_manager.get_cache_tensors(kv_cache, -1)
        tensors = kv_cache_manager.get_cache_tensors(kv_cache, 0)
        print(f'kv_cache: {kv_cache}')
        print(f'tensors: {Tensor(tensors[0]).numpy()}')
        kv_cache_manager.deallocate_cache(kv_cache)
        kv_cache_manager.remove_cache_key(cache_key)
        engine.finalize()

    def test_decoder_cache_ops(self):
        cluster_id = 0
        llm_engine = LLMDataDist(LLMRole.DECODER, cluster_id)
        llm_engine.init(LlmEngineV2St._engine_options(False, cluster_id))
        kv_cache_manager = llm_engine.kv_cache_manager
        cache_desc = CacheDesc(80, [2, 8], DataType.DT_FLOAT16)
        cache_key = CacheKey(0, 1, 1)
        # mock prompt allocate kv
        kv_cache_manager._role = LLMRole.PROMPT
        kv_cache = kv_cache_manager.allocate_cache(cache_desc, [cache_key])
        kv_cache_manager._role = LLMRole.DECODER

        dst_kv_cache = kv_cache_manager.allocate_cache(cache_desc)
        print(f'kv_cache: {kv_cache}')
        with self.assertRaises(LLMException):
            kv_cache_manager.pull_cache(cache_key, dst_kv_cache, 0, -2)
        kv_cache_manager.pull_cache(cache_key, dst_kv_cache, src_cache_offset=0, dst_cache_offset=0)
        cache_key_id = CacheKeyByIdAndIndex(cluster_id, kv_cache.cache_id, 0)
        kv_cache_manager.pull_cache(cache_key_id, dst_kv_cache, 0)
        kv_cache_manager.copy_cache(dst_kv_cache, kv_cache)
        with self.assertRaises(LLMException):
            kv_cache_manager.copy_cache(dst_kv_cache, kv_cache, 0, 0, 0, -2)
        with self.assertRaises(LLMException):
            kv_cache_manager.copy_cache(dst_kv_cache, kv_cache, 0, 0, 0, 0)
        kv_cache_manager.deallocate_cache(kv_cache)
        # test use after deallocated
        with self.assertRaises(LLMException) as ex:
            kv_cache_manager.pull_cache(cache_key, kv_cache)
        self.assertEqual(ex.exception.status_code, LLMStatusCode.LLM_KV_CACHE_NOT_EXIST)
        kv_cache_manager.deallocate_cache(dst_kv_cache)
        llm_engine.finalize()

    def test_cluster_ops(self):
        cluster_id = 0
        llm_engine = LLMDataDist(LLMRole.DECODER, cluster_id)
        cluster_info = {
            "cluster_id": 1, "logic_device_id": ["0:0:0:0"]
        }
        engine_options = {'llm.ClusterInfo': json.dumps(cluster_info)}
        llm_engine.init(engine_options)

        cluster = LLMClusterInfo()
        cluster.remote_cluster_id = 1
        cluster.append_local_ip_info(1, 26000)
        cluster.append_remote_ip_info(1, 26000)
        ret, rets = llm_engine.link_clusters([cluster], 5000)
        self.assertEqual(ret, LLMStatusCode.LLM_SUCCESS)
        ret, rets = llm_engine.unlink_clusters([cluster], 5000)
        self.assertEqual(ret, LLMStatusCode.LLM_SUCCESS)
        llm_engine.check_link_status(1)

    def test_init_and_finalize(self):
        cluster_id = 0
        decoder_engine = LLMDataDist(LLMRole.DECODER, cluster_id)
        decoder_engine.finalize()
        decoder_engine.init(self._engine_options(False, cluster_id, 1))
        decoder_engine.init(self._engine_options(False, cluster_id, 1))
        decoder_engine.finalize()
        decoder_engine.finalize()
        decoder_engine_2 = LLMDataDist(LLMRole.DECODER, cluster_id)
        decoder_engine_2.init(self._engine_options(False, cluster_id, 1))

    def test_init_witch_device_id_and_rank_id(self):
        llm_config = LLMConfig()
        llm_config.device_id = 1
        llm_config.listen_ip_info = "127.0.0.1:26000"
        llm_config.ge_options = {
            "ge.exec.rankId": "1"
        }
        engine_options = llm_config.generate_options()
        print("engine_options:", engine_options)
        prompt_engine = LLMDataDist(LLMRole.PROMPT, 0)
        prompt_engine.init(engine_options)
        EngineConfig.from_engine_options(True, engine_options)

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

    def test_shutdown_failed(self):
        cluster_id = 0
        decoder_engine = LLMDataDist(LLMRole.DECODER, cluster_id)
        decoder_engine.init(self._engine_options(False, cluster_id, 1))


        # 模拟被其它框架Finalize
        decoder_engine._llm_datadist.finalize()
        _shutdown_handler()
        decoder_engine._is_initialized = False
        LLMDataDist.llm_engine_instance = None

    def test_switch_role_failed_option_not_set(self):
        engine = LLMDataDist(LLMRole.MIX, 0)
        options = {'ge.exec.deviceId': '0'}
        engine.init(options)
        try:
            engine.switch_role(LLMRole.DECODER)
        except LLMException as e:
            self.assertEqual(e.status_code, LLMStatusCode.LLM_FEATURE_NOT_ENABLED)

    def test_switch_role_failed_identical_role(self):
        engine = LLMDataDist(LLMRole.MIX, 0)
        options = {'llm.EnableSwitchRole': '1', 'ge.exec.deviceId': '0'}
        engine.init(options)
        with self.assertRaises(LLMException):
            engine.switch_role(LLMRole.MIX)

    def test_switch_role(self):
        engine = LLMDataDist(LLMRole.MIX, 0)
        llm_config = LLMConfig()
        llm_config.device_id = 0
        llm_config.enable_switch_role = True
        options = llm_config.generate_options()
        os.environ['RESOURCE_CONFIG_PATH'] = _TEST_BASE_DIR + '/json_file/numa_config.json'
        engine.init(options)
        engine.switch_role(LLMRole.DECODER)
        switch_options = {
            'llm.listenIpInfo': '127.0.0.1:1111',
        }
        engine.switch_role(LLMRole.PROMPT, switch_options)

    @staticmethod
    def _allocate_npu_cache(kv_cache_manager, block_size, num_block, num_tensors):
        npu_cache_desc = CacheDesc(num_tensors=num_tensors, shape=[num_block, block_size],
                                   data_type=DataType.DT_FLOAT16)
        npu_cache_key = BlocksCacheKey(0, 0)
        cache = kv_cache_manager.allocate_blocks_cache(npu_cache_desc, npu_cache_key)
        return cache, npu_cache_key

    @staticmethod
    def _allocate_cpu_cache(kv_cache_manager, block_size, num_block, num_tensors):
        # DT没有友好的方式创建cpu tensor，用npu接口模拟
        cpu_cache_desc = CacheDesc(num_tensors=num_tensors, shape=[num_block, block_size],
                                   data_type=DataType.DT_FLOAT16)
        cpu_cache_key = BlocksCacheKey(1, 1)
        cache = kv_cache_manager.allocate_blocks_cache(cpu_cache_desc, cpu_cache_key)
        cpu_cache_desc = CacheDesc(num_tensors=num_tensors, shape=[num_block, block_size],
                                   data_type=DataType.DT_FLOAT16, placement=Placement.HOST)
        return KvCache.create_cpu_cache(cpu_cache_desc, cache.per_device_tensor_addrs[0]), cache

    def test_swap_blocks(self):
        cluster_id = 0
        llm_engine = LLMDataDist(LLMRole.PROMPT, cluster_id)
        llm_config = LLMConfig()
        llm_config.device_id = 0
        llm_config.listen_ip_info = "0.0.0.0:26000"
        llm_config.ge_options = {"ge.flowGraphMemMaxSize": "10240"}
        init_options = llm_config.generate_options()
        llm_engine.init(init_options)

        kv_cache_manager = llm_engine.kv_cache_manager
        # allocate npu cache
        npu_cache, npu_cache_key = self._allocate_npu_cache(kv_cache_manager, 64*1024, 10, 10)
        cpu_cache, tmp_cache = self._allocate_cpu_cache(kv_cache_manager, 64*1024, 20, 10)
        src_to_dst = {3:4, 0:0, 1:1, 2:2, 5:6, 6:7, 7:8, 9:9}
        kv_cache_manager.swap_blocks(npu_cache, cpu_cache, src_to_dst)
        kv_cache_manager.swap_blocks(cpu_cache, npu_cache, src_to_dst)

    def test_create_cpu_cache_failed(self):
        cluster_id = 0
        llm_engine = LLMDataDist(LLMRole.PROMPT, cluster_id)
        llm_config = LLMConfig()
        llm_config.device_id = 0
        llm_config.listen_ip_info = "0.0.0.0:26000"
        llm_config.ge_options = {"ge.flowGraphMemMaxSize": "10240"}
        init_options = llm_config.generate_options()
        llm_engine.init(init_options)

        has_err = False
        try:
            cpu_cache_desc = CacheDesc(num_tensors=1, shape=[1, 1],
                                       data_type=DataType.DT_FLOAT16, placement=Placement.HOST)
            KvCache.create_cpu_cache(cpu_cache_desc, [[1], 1])
        except LLMException as ex:
            self.assertEqual("should be consistent" in ex.__str__(), True)
            has_err = True
        self.assertEqual(has_err, True)


    def test_copy_blocks_validate(self):
        cluster_id = 0
        llm_datadist = LLMDataDist(LLMRole.DECODER, cluster_id)
        llm_datadist.init(self._engine_options(False, cluster_id))

        kv_cache_manager = llm_datadist.kv_cache_manager
        cache_desc = CacheDesc(1, [2, 8], DataType.DT_FLOAT16)
        kv_cache = kv_cache_manager.allocate_blocks_cache(cache_desc)
        try:
            kv_cache_manager.copy_blocks(kv_cache, {0: "1"})
        except Exception as ex:
            self.assertEqual("only support" in ex.__str__(), True)
        print("copy_blocks validate2")
        try:
            kv_cache_manager.copy_blocks(kv_cache, {0: ["1"]})
        except Exception as ex:
            self.assertEqual("inner type only support" in ex.__str__(), True)

        llm_datadist.finalize()

    def test_transfer_cache(self):
        cluster_id = 0
        engine = LLMDataDist(LLMRole.PROMPT, cluster_id)
        engine.init(LlmEngineV2St._engine_options(True, cluster_id))
        kv_cache_manager = engine.kv_cache_manager
        cache_desc = CacheDesc(10, [2, 8], DataType.DT_FLOAT16)
        cache_key = CacheKey(0, 0, 1)
        kv_cache = kv_cache_manager.allocate_cache(cache_desc, [cache_key])
        dst_addrs_1 = [10000000, 20000000, 30000000, 40000000, 50000000, 60000000]
        dst_addrs_2 = [10000000, 20000000, 30000000, 40000000]
        transfer_config_1 = TransferConfig(1, dst_addrs_1, range(0, 3))
        print(transfer_config_1)
        transfer_config_2 = TransferConfig(2, dst_addrs_2, range(2, 4))
        transfer_configs = (transfer_config_1, transfer_config_2)
        cache_task = kv_cache_manager.transfer_cache_async(kv_cache, LayerSynchronizerImpl(True), transfer_configs)
        ret = cache_task.synchronize(0)
        self.assertEqual(ret, LLMStatusCode.LLM_WAIT_PROCESS_TIMEOUT)
        rets = cache_task.get_results(0)
        self.assertEqual(rets, [LLMStatusCode.LLM_WAIT_PROCESS_TIMEOUT] * 2)
        ret = cache_task.synchronize(1000)
        rets = cache_task.get_results(1000)
        self.assertEqual(ret, LLMStatusCode.LLM_SUCCESS)
        self.assertEqual(rets[0], LLMStatusCode.LLM_SUCCESS)
        self.assertEqual(rets[1], LLMStatusCode.LLM_SUCCESS)

        transfer_config_3 = TransferConfig(2, dst_addrs_1 + dst_addrs_2)
        cache_task = kv_cache_manager.transfer_cache_async(kv_cache, LayerSynchronizerImpl(True), [transfer_config_3])
        ret = cache_task.synchronize()
        self.assertEqual(ret, LLMStatusCode.LLM_SUCCESS)

        # test blocks suc
        kv_cache = kv_cache_manager.allocate_blocks_cache(cache_desc)
        dst_addrs_1 = [10000000, 20000000, 30000000, 40000000, 50000000, 60000000]
        transfer_config_1 = TransferConfig(1, dst_addrs_1, range(0, 3))
        transfer_configs = [transfer_config_1]
        block_indices = [0, 1]
        cache_task = kv_cache_manager.transfer_cache_async(kv_cache, LayerSynchronizerImpl(True), transfer_configs,
                                                           block_indices, block_indices, dst_block_memory_size=32)
        ret = cache_task.synchronize()
        rets = cache_task.get_results()
        self.assertEqual(ret, LLMStatusCode.LLM_SUCCESS)
        self.assertEqual(rets[0], LLMStatusCode.LLM_SUCCESS)

        # test sync layer failed
        transfer_config_1 = TransferConfig(1, dst_addrs_1, range(0, 3))
        transfer_config_2 = TransferConfig(2, dst_addrs_2, range(2, 4))
        transfer_configs = (transfer_config_1, transfer_config_2)
        cache_task = kv_cache_manager.transfer_cache_async(kv_cache, LayerSynchronizerImpl(False), transfer_configs)
        ret = cache_task.synchronize()
        rets = cache_task.get_results()
        self.assertNotEqual(ret, LLMStatusCode.LLM_SUCCESS)
        self.assertNotEqual(rets[0], LLMStatusCode.LLM_SUCCESS)
        self.assertEqual(rets[1], None)

        # test transfer failed
        params = TransferCacheParameters(kv_cache, transfer_configs, None, None, None)
        job = MockTransferCacheJob(params)
        job.init()
        job.transfer_layers()
        self.assertEqual(job.get_results()[0], LLMStatusCode.LLM_WAIT_PROCESS_TIMEOUT)
        self.assertEqual(job.get_results()[1], None)

        src_block_indices = [1, 2]
        with self.assertRaisesRegex(LLMException, "transfer from blocks to cache is not supported"):
            _ = kv_cache_manager.transfer_cache_async(kv_cache, LayerSynchronizerImpl(False), transfer_configs,
                                                      src_block_indices)
        with self.assertRaises(TypeError):
            _ = kv_cache_manager.transfer_cache_async(kv_cache, LayerSynchronizerImpl(False), transfer_configs,
                                                      [0, 1], [0, "1"])
        with self.assertRaises(ValueError):
            _ = kv_cache_manager.transfer_cache_async(kv_cache, LayerSynchronizerImpl(False), transfer_configs,
                                                      [-1, 1], [0, 1])
        with self.assertRaises(ValueError):
            _ = kv_cache_manager.transfer_cache_async(kv_cache, LayerSynchronizerImpl(False), transfer_configs,
                                                      [0, 1], [-1, 1])
        with self.assertRaises(LLMException):
            _ = kv_cache_manager.transfer_cache_async(kv_cache, LayerSynchronizerImpl(False), transfer_configs,
                                                      [0, 1], [0])
        with self.assertRaises(LLMException):
            _ = kv_cache_manager.transfer_cache_async(kv_cache, LayerSynchronizerImpl(False), transfer_configs,
                                                      [], [], 1)
        with self.assertRaises(ValueError):
            _ = kv_cache_manager.transfer_cache_async(kv_cache, LayerSynchronizerImpl(False), transfer_configs,
                                                      [0, 1], [0, 1], -1)
        with self.assertRaises(TypeError):
            _ = kv_cache_manager.transfer_cache_async('cache', LayerSynchronizerImpl(False), transfer_configs)
        with self.assertRaises(TypeError):
            _ = kv_cache_manager.transfer_cache_async(None, LayerSynchronizerImpl(False), transfer_configs)
        kv_cache_manager.deallocate_cache(kv_cache)
        kv_cache_manager.remove_cache_key(cache_key)
        engine.finalize()

    def test_transfer_config(self):
        config = TransferConfig(2, [1000, 2000])
        self.assertEqual(config.dst_cluster_id, 2)
        self.assertEqual(config.dst_addrs, [1000, 2000])
        self.assertIsNone(config.src_layer_range)
        self.assertEqual(config.src_batch_index, 0)

        with self.assertRaises(TypeError):
            config.dst_cluster_id = None
        with self.assertRaises(TypeError):
            config.dst_cluster_id = '123'
        with self.assertRaises(TypeError):
            config.dst_addrs = None
        with self.assertRaises(TypeError):
            config.dst_addrs = '123'
        with self.assertRaises(TypeError):
            config.dst_addrs = ['123']
        with self.assertRaises(ValueError):
            config.dst_addrs = [-1]
        with self.assertRaises(TypeError):
            config.src_layer_range = [0, 1]
        with self.assertRaises(LLMException):
            config.src_layer_range = range(0, 5, 2)
        with self.assertRaises(LLMException):
            config.src_layer_range = range(0, 1, -1)
        with self.assertRaises(LLMException):
            config.src_layer_range = range(4, 1)
        with self.assertRaises(LLMException):
            config.src_layer_range = range(0, 0)
        with self.assertRaises(TypeError):
            config.src_batch_index = None
        with self.assertRaises(TypeError):
            config.src_batch_index = '1'
        with self.assertRaises(ValueError):
            config.src_batch_index = -1
        with self.assertRaises(ValueError):
            config.src_batch_index = 2 ** 32

        with self.assertRaises(TypeError):
            _ = TransferConfig(2, [1000, 2000], None, '1')
        with self.assertRaises(TypeError):
            _ = TransferConfig(2, [1000, 2000], [0, 1])
        with self.assertRaises(TypeError):
            _ = TransferConfig(2, [1000, '2000'])
        with self.assertRaises(TypeError):
            _ = TransferConfig(2, None)
        with self.assertRaises(TypeError):
            _ = TransferConfig('1', [1000, 2000])

    def test_multiple_devices(self):
        llm_config = LLMConfig()
        llm_config.device_id = [1, 2]
        llm_config.listen_ip_info = "127.0.0.1:26000;127.0.0.1:26000"
        engine_options = llm_config.generate_options()
        print("engine_options:", engine_options)
        prompt_engine = LLMDataDist(LLMRole.PROMPT, 0)
        prompt_engine.init(engine_options)
        print(engine_options['llm.ClusterInfo'])

    def test_tensor_bf16(self):
        arr1 = np.array([1.875], np.float16)
        tensor_desc = TensorDesc(DataType.DT_BF16, [1])
        tensor = Tensor(arr1, tensor_desc)
        print("generated numpy:", tensor.numpy())
        self.assertEqual(tensor.numpy().dtype, np.float32)
        self.assertEqual(int(tensor.numpy()[0]), 1)
    
    def test_tensor_foat32_copy_true(self):
        arr1 = np.array([[1.0, 2.0], [3.0, 4.0]], np.float32)
        tensor_desc = TensorDesc(DataType.DT_FLOAT, (2, 2))
        tensor = Tensor(arr1, tensor_desc)
        res = tensor.numpy(copy=True)
        print("generated numpy:", res)
        np.testing.assert_array_equal(res, arr1)
        self.assertEqual(res.dtype, np.float32)
        self.assertTrue(res.flags.c_contiguous)
        self.assertTrue(res.flags.writeable)
        arr1[0, 0] = 999.0
        self.assertNotEqual(res[0, 0], 999.0)

    def test_str_tensor(self):
        arr1 = np.array([["aaaa", "bbbb"], ["ccc", "ddd"]])
        src_type = arr1.dtype
        tensor = Tensor(arr1)
        self.assertEqual(tensor.numpy(True).dtype, src_type)
        print(tensor.numpy(True))

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