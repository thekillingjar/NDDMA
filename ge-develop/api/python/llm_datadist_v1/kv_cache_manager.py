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

from typing import List, Optional, Tuple, Union, Dict

try:
    from llm_datadist.configs import LLMRole
    from llm_datadist.utils import log
    from llm_datadist.utils.utils import check_isinstance, check_dict, check_int64, check_uint64,\
        check_uint32, check_int32, check_positive_or_set_default
    from llm_datadist.status import handle_llm_status, raise_if_false, raise_if_true, LLMStatusCode
    from llm_datadist.v2.llm_types import CacheDesc, KvCache, CacheKey, CacheKeyByIdAndIndex, BlocksCacheKey, Placement, \
        LayerSynchronizer, TransferConfig, CacheTask
    from llm_datadist.v2.llm_utils import (
        is_invalid_id, is_valid_id, pack_cache_desc, pack_cache_key, pack_block_cache_key, \
        pack_cache_key_by_id, transfer_cache_async, TransferCacheParameters, layer_range_to_tensor_indices
    )
except ModuleNotFoundError:
    from llm_datadist_v1.configs import LLMRole
    from llm_datadist_v1.utils import log
    from llm_datadist_v1.utils.utils import check_isinstance, check_dict, check_int64, check_uint64,\
        check_uint32, check_int32, check_positive_or_set_default
    from llm_datadist_v1.status import handle_llm_status, raise_if_false, raise_if_true, LLMStatusCode
    from llm_datadist_v1.llm_types import CacheDesc, KvCache, CacheKey, CacheKeyByIdAndIndex, BlocksCacheKey, Placement, \
        LayerSynchronizer, TransferConfig, CacheTask
    from llm_datadist_v1.llm_utils import (
        is_invalid_id, is_valid_id, pack_cache_desc, pack_cache_key, pack_block_cache_key, \
        pack_cache_key_by_id, transfer_cache_async, TransferCacheParameters, layer_range_to_tensor_indices
    )
    
from llm_datadist_v1.tensor import Tensor

_NUM_TENSORS_PER_LAYER = 2
_INVALID_ID = 2 ** 64 - 1


class KvCacheManager(object):
    """
    提供了一组KvCache的操作函数, 在LLMEngine初始化后通过LLMEngine获取实例

    Examples:
        >>> from llm_datadist import LLMRole, LLMDataDist
        >>> # init LLMDataDist
        >>> cluster_id = 0
        >>> datadist = LLMDataDist(LLMRole.PROMPT, cluster_id)
        >>> engine_options = {}
        >>> datadist.init(engine_options)
        >>> # get KvCacheManager
        >>> kv_cache_manager = datadist.kv_cache_manager
    """

    def __init__(self, llm_engine, role: LLMRole) -> None:
        self._llm_engine = llm_engine
        self._role = role
        self._initialized = True

    def is_initialized(self) -> bool:
        return self._initialized

    def allocate_blocks_cache(self,
                              cache_desc: CacheDesc,
                              blocks_cache_key: Optional[BlocksCacheKey] = None) -> KvCache:
        """
        分配Blocks Cache, cache分配成功后
        需通过deallocate_cache释放

        Args:
            cache_desc: Cache描述
            blocks_cache_key(Optional): 仅当LLMRole为PROMPT时可设置, 用于在DECODER拉取KV
        Returns:
            KvCache

        Examples:
            >>> from llm_datadist import LLMRole, CacheDesc, DataType, LLMDataDist
            >>> # init LLMDataDist
            >>> cluster_id = 0
            >>> datadist = LLMDataDist(LLMRole.PROMPT, cluster_id)
            >>> engine_options = {} # 按需填写
            >>> datadist.init(engine_options)
            >>> # get KvCacheManager
            >>> kv_cache_manager = datadist.kv_cache_manager
            >>> # allocate_cache
            >>> kv_cache_desc = CacheDesc(num_tensors=80, shape=[4, 256], data_type=DataType.DT_FLOAT16)
            >>> # case 1: batch_size = 4, 只有第二个有效, 此时后两个cache_key可以省略
            >>> block_cache_key = BlocksCacheKey(prompt_cluster_id=0, model_id=0)
            >>> cache = kv_cache_manager.allocate_blocks_cache(kv_cache_desc, block_cache_key)
            >>> kv_cache_manager.deallocate_cache(cache)
        """
        check_isinstance("cache_desc", cache_desc, CacheDesc)
        check_isinstance("blocks_cache_key", blocks_cache_key, BlocksCacheKey)
        raise_if_false(cache_desc.num_tensors > 0, "num_tensors should be bigger than zero.")
        raise_if_false(cache_desc.placement == Placement.DEVICE, "Only support allocate device cache")
        if self._role == LLMRole.DECODER:
            raise_if_false(blocks_cache_key is None, 'blocks_cache_key is not supported by DECODER')
        wrapped_cache_keys = [pack_block_cache_key(blocks_cache_key)] if blocks_cache_key is not None else []
        ret, cache_id_and_addr = self._llm_engine.allocate_cache(pack_cache_desc(cache_desc),
                                                                 wrapped_cache_keys)
        handle_llm_status(ret, '[allocate_blocks_cache]', f'cache_desc = {cache_desc}')
        kv_cache = KvCache(cache_id_and_addr[0], cache_desc, cache_id_and_addr[1], self)
        log.info('[allocate_blocks_cache] success, cache_id = %d', kv_cache.cache_id)
        return kv_cache

    def allocate_cache(self,
                       cache_desc: CacheDesc,
                       cache_keys: Union[Tuple[CacheKey], List[CacheKey]] = ()) -> KvCache:
        """
        分配Cache, cache分配成功后, 会同时被cache_id与cache_keys(如果传了)引用, 只有当这些引用都解除后, cache所占用的资源才会实际释放
        cache_id的引用需通过deallocate_cache解除
        cache_keys的引用则可以通过以下2种方式解除:
            1. DECODER调用pull_kv接口, pull_kv成功后解除
            2. PROMPT调用remove_cache_key接口

        Args:
            cache_desc: Cache描述
            cache_keys(Optional): 仅当LLMRole为PROMPT时可设置, 用于在DECODER拉取KV
                如果Cache的batch size > 1, 则需要提供相同数量的CacheKey, 分别引用一组kv tensor
                如果当次推理的batch未占用满，即存在无效batch index，则需要插入特殊的CacheKey(req_id = UINT64_MAX)占位,
                如果空闲的batch_index在末尾，则可以省略
        Returns:
            KvCache

        Examples:
            >>> from llm_datadist import LLMRole, CacheDesc, DataType, LLMDataDist
            >>> # init LLMDataDist
            >>> cluster_id = 0
            >>> datadist = LLMDataDist(LLMRole.PROMPT, cluster_id)
            >>> engine_options = {} # 按需填写
            >>> datadist.init(engine_options)
            >>> # get KvCacheManager
            >>> kv_cache_manager = datadist.kv_cache_manager
            >>> # allocate_cache
            >>> kv_cache_desc = CacheDesc(num_tensors=80, shape=[4, 256], data_type=DataType.DT_FLOAT16)
            >>> # case 1: batch_size = 4, 只有第二个有效, 此时后两个cache_key可以省略
            >>> kv_cache_key_1 = CacheKey(prompt_cluster_id=0, req_id=1, model_id=0)
            >>> padding_cache_key = CacheKey(prompt_cluster_id=0, req_id=2 ** 64 - 1, model_id=0)
            >>> kv_cache_keys = [padding_cache_key, kv_cache_key_1]
            >>> cache = kv_cache_manager.allocate_cache(kv_cache_desc, kv_cache_keys)
            >>> # case 2: batch_size = 4, 只有最后一个有效, 此时所有cache_key都不能省略
            >>> kv_cache_key_3 = CacheKey(prompt_cluster_id=0, req_id=3, model_id=0)
            >>> kv_cache_keys = [padding_cache_key, padding_cache_key, padding_cache_key, kv_cache_key_3]
            >>> cache_1 = kv_cache_manager.allocate_cache(kv_cache_desc, kv_cache_keys)
            >>> # 释放cache_id对cache的引用
            >>> kv_cache_manager.deallocate_cache(cache)
            >>> kv_cache_manager.deallocate_cache(cache_1)
            >>> # 释放cache_key对cache的引用
            >>> kv_cache_manager.remove_cache_key(kv_cache_key_1)
            >>> kv_cache_manager.remove_cache_key(kv_cache_key_3)
        """
        check_isinstance("cache_desc", cache_desc, CacheDesc)
        check_isinstance("cache_keys", cache_keys, [list, tuple], CacheKey)
        raise_if_false(cache_desc.num_tensors > 0, "num_tensors should be bigger than zero.")
        raise_if_false(cache_desc.placement == Placement.DEVICE, "Only support allocate device cache")
        log.info('[allocate_cache] start, cache_desc = %s, cache_keys = %s', cache_desc, cache_keys)
        if self._role != LLMRole.PROMPT:
            raise_if_false(len(cache_keys) == 0, 'cache_keys is not supported by {0}', self._role.name)
        wrapped_cache_keys = [pack_cache_key(cache_key) for cache_key in cache_keys]
        ret, cache_id_and_addr = self._llm_engine.allocate_cache(pack_cache_desc(cache_desc),
                                                                 wrapped_cache_keys)
        handle_llm_status(ret, '[allocate_cache]', f'cache_desc = {cache_desc}')
        kv_cache = KvCache(cache_id_and_addr[0], cache_desc, cache_id_and_addr[1], self)
        log.info('[allocate_cache] success, cache_id = %d', kv_cache.cache_id)
        return kv_cache

    def deallocate_cache(self, cache: KvCache) -> None:
        """
        释放Cache, 如果该Cache在Allocate时关联了CacheKey, 则实际的释放会延后到所有的CacheKey被拉取或执行了remove_cache_key
        释放之后，不应再对该KvCache做任何操作

        Args:
            cache: Cache

        Examples:
            see examples of allocate_cache
        """
        check_isinstance("cache", cache, KvCache)
        raise_if_true(self._is_cpu_cache(cache), "Only support device cache")
        log.info('[deallocate_cache] start, cache_id = %d', cache.cache_id)
        ret = self._llm_engine.deallocate_cache(cache.cache_id)
        handle_llm_status(ret, '[deallocate_cache]', f'cache_id = {cache.cache_id}')
        cache._per_device_tensor_addrs = []
        cache._valid = False
        log.info('[deallocate_cache] success')

    def remove_cache_key(self, cache_key: CacheKey) -> None:
        """
        移除CacheKey, 仅当LLMRole为PROMPT时可调用
        移除CacheKey后, 该Cache将无法再被pull_cache拉取

        Args:
            cache_key: CacheKey

        Examples:
            see examples of allocate_cache
        """
        self._check_role("[remove_cache_key]", LLMRole.PROMPT)
        check_isinstance("cache_key", cache_key, CacheKey)
        log.info('[remove_cache_key] start, cache_key = %s', cache_key)
        ret = self._llm_engine.remove_cache_key(pack_cache_key(cache_key))
        handle_llm_status(ret, '[remove_cache_key]', f'cache_key = {cache_key}')
        log.info('[remove_cache_key] success')

    def pull_blocks(self, prompt_cache_key: BlocksCacheKey, decoder_kv_cache: KvCache, prompt_blocks: List[int],
                    decoder_blocks: List[int], **kwargs):
        """
        PA模式下拉取KV
        Args:
            prompt_cache_key: prompt缓存key
            decoder_kv_cache: decoder目标缓存
            prompt_blocks: prompt block列表
            decoder_blocks: decoder block列表
            **kwargs:
                src_layer_range: 源层范围
                dst_layer_range: 目标层范围
                tensor_num_per_layer: 每层tensor数量
        """
        src_layer_range = kwargs.get("src_layer_range")
        dst_layer_range = kwargs.get("dst_layer_range")
        tensor_num_per_layer = kwargs.get("tensor_num_per_layer", _NUM_TENSORS_PER_LAYER)
        self._check_role("[pull_blocks]", LLMRole.DECODER)
        check_isinstance("prompt_cache_key", prompt_cache_key, BlocksCacheKey)
        check_isinstance("decoder_kv_cache", decoder_kv_cache, KvCache)
        check_isinstance("prompt_blocks", prompt_blocks, list, int)
        check_isinstance("decoder_blocks", decoder_blocks, list, int)
        raise_if_true(self._is_cpu_cache(decoder_kv_cache), "Only support device cache")
        raise_if_false(len(prompt_blocks) > 0, "prompt_blocks can not be empty.")
        raise_if_false(len(decoder_blocks) > 0, "decoder_blocks can not be empty.")
        raise_if_false(len(prompt_blocks) == len(decoder_blocks),
                       "Param prompt_blocks and decoder_blocks size should be same.")
        check_uint32("tensor_num_per_layer", tensor_num_per_layer)
        raise_if_false(tensor_num_per_layer > 0,
                       '[pull_blocks] param check failed, tensor_num_per_layer ({0}) is invalid, should [1, {1}]',
                       tensor_num_per_layer, decoder_kv_cache.cache_desc.num_tensors)

        log.info('[pull_blocks] start, target cache_id = %d, cache_key = %s, '
                 'src_layer_range = %s, dst_layer_range = %s, tensor_num_per_layer = %d',
                 decoder_kv_cache.cache_id, prompt_cache_key, src_layer_range, dst_layer_range, tensor_num_per_layer)
        src_tensor_indices, dst_tensor_indices = layer_range_to_tensor_indices(src_layer_range, dst_layer_range, tensor_num_per_layer)
        param = (-1, 0, prompt_blocks, decoder_blocks, src_tensor_indices, dst_tensor_indices, -1, -1, tensor_num_per_layer)
        ret = self._llm_engine.pull_cache(decoder_kv_cache.cache_id, pack_block_cache_key(prompt_cache_key),
                                          param)
        handle_llm_status(ret, '[pull_blocks]', f'prompt_cache_key = {prompt_cache_key}')
        log.info('[pull_blocks] success')

    def pull_cache(self,
                   cache_key: Union[CacheKey, CacheKeyByIdAndIndex],
                   kv_cache: KvCache,
                   batch_index: int = 0,
                   size: int = -1,
                   **kwargs) -> None:
        """
        拉取KV, 仅当LLMRole为DECODER时可调用

        Args:
            cache_key: CacheKey或CacheKeyByIdAndIndex
            kv_cache: 目标KvCache
            batch_index: batch index
            size: 拉取的tensor大小, -1表示拉取全部大小
            **kwargs:
                src_layer_range: 源层范围
                dst_layer_range: 目标层范围
                src_cache_offset: 源cache偏移
                dst_cache_offset: 目标cache偏移
                tensor_num_per_layer: 每层tensor数量

        Examples:
            >>> from llm_datadist import LLMRole, CacheDesc, DataType, LLMDataDist
            >>> # init LLMDataDist
            >>> cluster_id = 0
            >>> datadist = LLMDataDist(LLMRole.DECODER, cluster_id)
            >>> engine_options = {} # 按需填写
            >>> datadist.init(engine_options)
            >>> # get KvCacheManager
            >>> kv_cache_manager = datadist.kv_cache_manager
            >>> # allocate_cache
            >>> kv_cache_desc = CacheDesc(num_tensors=80, shape=[4, 256], data_type=DataType.DT_FLOAT16)
            >>> cache = kv_cache_manager.allocate_cache(kv_cache_desc)
            >>> # pull prompt kv to allocated cache
            >>> prompt_cache_key = CacheKey(prompt_cluster_id=0, req_id = 1, model_id = 0)
            >>> kv_cache_manager.pull_cache(prompt_cache_key, cache, 0)
        """
        src_layer_range = kwargs.get("src_layer_range")
        dst_layer_range = kwargs.get("dst_layer_range")
        src_cache_offset = kwargs.get("src_cache_offset")
        dst_cache_offset = kwargs.get("dst_cache_offset")
        tensor_num_per_layer = kwargs.get("tensor_num_per_layer", _NUM_TENSORS_PER_LAYER)
        self._check_role("[pull_cache]", LLMRole.DECODER)
        check_isinstance("cache_key", cache_key, [CacheKey, CacheKeyByIdAndIndex])
        check_isinstance("kv_cache", kv_cache, KvCache)
        raise_if_true(self._is_cpu_cache(kv_cache), "Only support device cache")
        check_int64("size", size)
        check_uint32("batch_index", batch_index)
        raise_if_false(size == -1 or size > 0,
                       '[pull_cache] param check failed, size ({0}) is invalid, should be = -1 or > 0', size)
        src_cache_offset = check_positive_or_set_default("src_cache_offset", src_cache_offset)
        dst_cache_offset = check_positive_or_set_default("dst_cache_offset", dst_cache_offset)
        if isinstance(cache_key, CacheKey):
            KvCacheManager.check_cache_key(cache_key)
            packed_cache_key = pack_cache_key(cache_key)
        else:
            packed_cache_key = pack_cache_key_by_id(cache_key)
        check_uint32("tensor_num_per_layer", tensor_num_per_layer)
        raise_if_false(tensor_num_per_layer > 0,
                       '[pull_cache] param check failed, tensor_num_per_layer ({0}) is invalid, should [1, {1}]',
                       tensor_num_per_layer, kv_cache.cache_desc.num_tensors)

        log.info('[pull_cache] start, cache_id = %d, batch_index = %d, size = %d, cache_key = %s, '
                 'src_layer_range = %s, dst_layer_range = %s, src_cache_offset = %d, dst_cache_offset = %d, tensor_num_per_layer = %d',
                 kv_cache.cache_id, batch_index, size, cache_key, src_layer_range, dst_layer_range,
                 src_cache_offset, dst_cache_offset, tensor_num_per_layer)
        src_tensor_indices, dst_tensor_indices = layer_range_to_tensor_indices(src_layer_range, dst_layer_range, tensor_num_per_layer)

        param = (size, batch_index, [], [], src_tensor_indices, dst_tensor_indices, src_cache_offset,
                 dst_cache_offset, tensor_num_per_layer)
        ret = self._llm_engine.pull_cache(kv_cache.cache_id, packed_cache_key, param)
        handle_llm_status(ret, '[pull_cache]', f'cache_key = {cache_key}')
        log.info('[pull_cache] success')

    @staticmethod
    def check_cache_key(cache_key: CacheKey) -> None:
        raise_if_true(is_invalid_id(cache_key.req_id) and is_invalid_id(cache_key.prefix_id),
                      f'one of req id and prefix id should contain valid value:[0, 2**64-1), '
                      f'req id:{cache_key.req_id},prefix id{cache_key.prefix_id}.')
        raise_if_true(is_valid_id(cache_key.req_id) and is_valid_id(cache_key.prefix_id),
                      'only one of req id and prefix id should contain valid value:[0, 2**64-1), '
                      f'req id:{cache_key.req_id}, prefix id{cache_key.prefix_id}.')

    @staticmethod
    def _verify_caches(src: KvCache, dst: KvCache, src_to_dst: Dict[int, int]):
        check_isinstance("src", src, KvCache)
        check_isinstance("dst", dst, KvCache)
        check_isinstance("src_to_dst", src_to_dst, dict, int)

        src_block_size = src.cache_desc.size // src.cache_desc.batch_size
        dst_block_size = dst.cache_desc.size // dst.cache_desc.batch_size
        raise_if_false(src_block_size == dst_block_size,
                       f"src block size:{src_block_size} and dst block size:{dst_block_size} must be equal")
        log.info("src and dst cache block size:%d", src_block_size)

        src_num_tensors = src.cache_desc.num_tensors
        dst_num_tensors = dst.cache_desc.num_tensors
        raise_if_false(src_num_tensors == dst_num_tensors,
                       f"src num_tensors:{src_num_tensors} and dst num_tensors:{dst_num_tensors} must be equal")
        log.info("src and dst cache num:%d", src_num_tensors)

        src_block_num = src.cache_desc.batch_size
        dst_block_num = dst.cache_desc.batch_size
        log.info("src num block:%d, dst num block:%d", src_block_num, dst_block_num)
        for src_block_index, dst_block_index in src_to_dst.items():
            raise_if_false(0 <= src_block_index < src_block_num,
                           f"src_block_index:{src_block_index} must be in [0, {src_block_num})")
            raise_if_false(0 <= dst_block_index < dst_block_num,
                           f"dst_block_index:{dst_block_index} must be in [0, {dst_block_num})")

    @staticmethod
    def _is_cpu_cache(cache: KvCache):
        return cache.cache_desc.placement == Placement.HOST

    def copy_blocks(self, cache: KvCache, copy_block_info: Dict[int, List[int]]):
        """
        Args:
            cache: 目标缓存
            copy_block_info: 拷贝信息, (int, List[int])代表(原始block index, 目标block index)
        """
        check_isinstance("cache", cache, KvCache)
        check_isinstance("copy_block_info", copy_block_info, dict)
        check_dict("copy_block_info", copy_block_info, int, list, int)
        raise_if_true(self._is_cpu_cache(cache), "Only support device cache")
        copy_block_infos = []
        for src_block, dst_blocks in copy_block_info.items() :
            check_uint64("src_block", src_block)
            for dst_block in dst_blocks :
                check_uint64("dst_block", dst_block)
                copy_block_infos.append((src_block, dst_block))
        param = (cache.cache_id, cache.cache_id, 0, 0, 0, -1, _INVALID_ID, copy_block_infos)
        ret = self._llm_engine.copy_cache(param)
        handle_llm_status(ret, '[copy_blocks]', "cache id is:%s" % cache.cache_id)
        log.info('[copy_blocks] success')

    def copy_cache(self,
                   dst: KvCache,
                   src: KvCache,
                   dst_batch_index: int = 0,
                   src_batch_index: int = 0,
                   offset: int = 0,
                   size: int = -1,
                   req_id: Optional[int] = None) -> None:
        """
        拷贝KV, src/dst的CacheDesc需要匹配

        Args:
            dst: 目标Cache
            src: 源Cache
            dst_batch_index: 目标Cache的batch_index
            src_batch_index: 源Cache的batch_index
            offset: 每个tensor的偏移
            size: 每个tensor拷贝的大小
            req_id(Optional): 本次操作关联的req_id, 仅用于维测

        Examples:
            >>> from llm_datadist import LLMRole, CacheDesc, DataType, LLMDataDist
            >>> # init LLMDataDist
            >>> cluster_id = 0
            >>> datadist = LLMDataDist(LLMRole.DECODER, cluster_id)
            >>> engine_options = {} # 按需填写
            >>> datadist.init(engine_options)
            >>> # get KvCacheManager
            >>> kv_cache_manager = datadist.kv_cache_manager
            >>> # allocate caches
            >>> tmp_kv_cache_desc = CacheDesc(num_tensors=80, shape=[1, 256], data_type=DataType.DT_FLOAT16)
            >>> model_kv_cache_desc = CacheDesc(num_tensors=80, shape=[4, 256], data_type=DataType.DT_FLOAT16)
            >>> tmp_cache = kv_cache_manager.allocate_cache(tmp_kv_cache_desc)
            >>> model_cache = kv_cache_manager.allocate_cache(model_kv_cache_desc)
            >>> # pull prompt kv to tmp cache
            >>> prompt_cache_key = CacheKey(prompt_cluster_id=0, req_id = 1, model_id = 0)
            >>> kv_cache_manager.pull_cache(prompt_cache_key, tmp_cache)
            >>> # copy cache from tmp cache to model cache
            >>> kv_cache_manager.copy_cache(model_cache, tmp_cache)
        """
        check_isinstance("dst", dst, KvCache)
        raise_if_true(self._is_cpu_cache(dst), "Only support device cache")
        check_isinstance("src", src, KvCache)
        raise_if_true(self._is_cpu_cache(src), "Only support device cache")
        check_uint32("dst_batch_index", dst_batch_index)
        check_uint32("src_batch_index", src_batch_index)
        check_uint64("offset", offset)
        check_int64("size", size)
        raise_if_false(size == -1 or size > 0,
                       '[copy_cache] param check failed, size ({0}) is invalid, should be = -1 or > 0', size)
        user_param = (dst.cache_id, src.cache_id, dst_batch_index, src_batch_index, offset, size, req_id, [])
        log.info('[copy_cache] start, param = %s', user_param)
        if req_id is not None:
            check_isinstance("req_id", req_id, int)
        else:
            req_id = _INVALID_ID
        param = (dst.cache_id, src.cache_id, dst_batch_index, src_batch_index, offset, size, req_id, [])
        ret = self._llm_engine.copy_cache(param)
        handle_llm_status(ret, '[copy_cache]', f'param = {param}')
        log.info('[copy_cache] success')

    def get_cache_tensors(self, cache: KvCache, tensor_index: int = 0) -> List:
        """
        获取cache tensor

        Args:
            cache: KvCache
            tensor_index: tensor index
        """
        check_isinstance("cache", cache, KvCache)
        check_int32("tensor_index", tensor_index)
        raise_if_false(0 <= tensor_index < cache.cache_desc.num_tensors,
                       '[get_cache_tensors] param check failed, tensor_index ({0}) out of range, [0, {1})',
                       tensor_index, cache.cache_desc.num_tensors)
        ret, outputs = self._llm_engine.get_tensor(cache.cache_id, tensor_index)
        handle_llm_status(ret,
                          '[get_cache_tensors]',
                          {'cache_id': cache.cache_id, 'tensor_index': tensor_index})
        tensors = [Tensor.from_tensor_tuple(output) for output in outputs]
        return tensors

    def swap_blocks(self, src: KvCache, dst: KvCache, src_to_dst: Dict[int, int]) -> None:
        """
        交换blocks

        Args:
            src: 源KvCache
            dst: 目的KvCache
            src_to_dst: block index的字典
        """
        self._verify_caches(src, dst, src_to_dst)
        src_placement = src.cache_desc.placement
        dst_placement = dst.cache_desc.placement
        is_swap_in = (src_placement == Placement.HOST) and (dst_placement == Placement.DEVICE)
        is_swap_out = (src_placement == Placement.DEVICE) and (dst_placement == Placement.HOST)
        raise_if_false(is_swap_in or is_swap_out,
                       f"swap src placement:{src_placement} to dst placement:{dst_placement} is not support")

        # 0标识swap in，1标识swap out
        swap_type = 0 if is_swap_in else 1
        block_size = src.cache_desc.size // src.cache_desc.batch_size
        default_cache_id = -1
        src_cache = (default_cache_id, src.per_device_tensor_addrs)
        dst_cache = (default_cache_id, dst.per_device_tensor_addrs)
        ret = self._llm_engine.swap_blocks(src_cache, dst_cache, block_size, swap_type,
                                           self._llm_engine.dict_to_vector(src_to_dst))
        handle_llm_status(ret, '[swap_blocks]', 'swap blocks failed')
        log.info('[swap_blocks] success')

    def transfer_cache_async(self,
                             src_cache: KvCache,
                             layer_synchronizer: LayerSynchronizer,
                             transfer_configs: Union[List[TransferConfig], Tuple[TransferConfig]],
                             src_block_indices: Optional[Union[List[int], Tuple[int]]] = None,
                             dst_block_indices: Optional[Union[List[int], Tuple[int]]] = None,
                             dst_block_memory_size: Optional[int] = None) -> CacheTask:
        self._check_role("[transfer_cache_async]", LLMRole.PROMPT)
        check_isinstance("src_cache", src_cache, KvCache, allow_none=False)
        params = TransferCacheParameters(src_cache,
                                         transfer_configs,
                                         src_block_indices,
                                         dst_block_indices,
                                         dst_block_memory_size)
        log.info('[transfer_cache_async] start, params = %s', params)
        return transfer_cache_async(params, layer_synchronizer, self._llm_engine.transfer_cache,
                                    LLMStatusCode.LLM_WAIT_PROCESS_TIMEOUT)

    def _check_role(self, func_name, role: LLMRole) -> None:
        raise_if_false(self._role == role,
                       '{0} is not supported by {1}',
                       func_name, self._role)

    def _switch_role(self, role: LLMRole) -> None:
        log.info(f'[switch_role] [{self._role.name}->{role.name}] success')
        self._role = role
