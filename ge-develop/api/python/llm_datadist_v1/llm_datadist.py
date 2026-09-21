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
from typing import Any, Dict, List, Tuple, Optional, Union
import atexit
from llm_datadist_v1 import llm_wrapper

try:
    from llm_datadist.utils.utils import check_isinstance, check_uint64, check_int32
except ModuleNotFoundError:
    from llm_datadist_v1.utils.utils import check_isinstance, check_uint64, check_int32

try:
    from llm_datadist.utils import log
except ModuleNotFoundError:
    from llm_datadist_v1.utils import log

try:
    from llm_datadist.status import (code_2_status, handle_llm_status, raise_if_false, LLMStatusCode,
                                    LLMException)
except ModuleNotFoundError:
    from llm_datadist_v1.status import (code_2_status, handle_llm_status, raise_if_false, LLMStatusCode,
                                    LLMException)

try:
    from llm_datadist.configs import LLMRole, LLMClusterInfo
except ModuleNotFoundError:
    from llm_datadist_v1.configs import LLMRole, LLMClusterInfo

try:
    from llm_datadist.v2.config import EngineConfig
except ModuleNotFoundError:
    from llm_datadist_v1.config import EngineConfig

try:
    from llm_datadist.v2.llm_utils import parse_listen_ip_info
except ModuleNotFoundError:
    from llm_datadist_v1.llm_utils import parse_listen_ip_info

from llm_datadist_v1.kv_cache_manager import KvCacheManager

__all__ = ['LLMDataDist', 'KvCacheManager']

class LLMDataDist(object):
    llm_engine_instance = None
    """
    LLMDataDist

    Args:
        role: role of LLMDataDist
        cluster_id: cluster_id of LLMDataDist
    """

    def __init__(self, role: LLMRole, cluster_id: int):
        check_isinstance("role", role, LLMRole)
        self._kv_cache_manager = None
        self._cache_manager = None
        self._role = role
        check_uint64("cluster_id", cluster_id)
        self._cluster_id = cluster_id
        self._llm_datadist = None
        self._engine_config = None
        self._is_initialized = False
        self._engine_options: Dict[str, str] = {}
        self._enable_cache_mgr = False
        self._enable_free_comm = False
        self._enable_local_comm_res = False

    def _check_flow_graph_max_size(self, options: Dict[str, str]) -> None:
        mem_utilization = float(options.get("llm.MemoryUtilization", "0.95"))
        value = options.get("ge.flowGraphMemMaxSize", None)
        if value is None:
            return
        check_isinstance('ge.flowGraphMemMaxSize', value, str)
        raise_if_false(len(value.split(",")) == 1, "ge.flowGraphMemMaxSize only support one mem pool in llm datadist")
        raise_if_false(value.isdigit(), "ge.flowGraphMemMaxSize must be digit")

    def init(self, options: Dict[str, str]) -> None:
        """
        初始化LLM Engine

        Args:
            options: Engine相关options
        """
        if self._is_initialized:
            return
        raise_if_false(LLMDataDist.llm_engine_instance is None, 'Cannot init multiple LLM engines',
                       status_code=LLMStatusCode.LLM_FAILED)
        check_isinstance("options", options, dict)
        self._check_flow_graph_max_size(options)
        self._engine_options = options
        self._engine_options['llm.Role'] = self._role_to_str(self._role)
        self._enable_local_comm_res = "llm.LocalCommRes" in options
        if self._enable_local_comm_res and "llm.EnableCacheManager" not in options:
            self._engine_options["llm.EnableCacheManager"] = "1"
        if self._enable_local_comm_res and "llm.EnableRemoteCacheAccessible" not in options:
            self._engine_options["llm.EnableRemoteCacheAccessible"] = "1"
        self._enable_cache_mgr = (
            "llm.EnableCacheManager" in self._engine_options and
            self._engine_options["llm.EnableCacheManager"] == "1"
        )
        log.info('options = %s', self._engine_options)
        self._llm_datadist = llm_wrapper
        EngineConfig.gen_cluster_info_if_not_exist(self._cluster_id, self._role, self._engine_options)
        ret = self._llm_datadist.initialize(self._cluster_id, self._engine_options)
        handle_llm_status(ret, '[LLMDataDist.init]', f'Failed to initialize llm datadist, options = {options}')
        self._kv_cache_manager = KvCacheManager(self._llm_datadist, self._role)
        LLMDataDist.llm_engine_instance = self
        self._is_initialized = True

    def _check_is_cache_mgr_mode(self, func_name):
        raise_if_false(self._enable_cache_mgr,
                       '{0} is not supported when llm.EnableCacheManager is not configured.',
                       func_name)

    def _check_is_not_cache_mgr_mode(self, func_name):
        raise_if_false(not self._enable_cache_mgr,
                       '{0} is not supported when llm.EnableCacheManager is configured.',
                       func_name)

    def finalize(self) -> None:
        """
        释放LLM Engine相关资源
        """
        if not self._is_initialized:
            return
        self._llm_datadist.finalize()
        if self._kv_cache_manager is not None:
            self._kv_cache_manager._initialized = False
        self._is_initialized = False
        LLMDataDist.llm_engine_instance = None

    def _cluster_config(self):
        if self._engine_config is None:
            self._engine_config = EngineConfig.from_engine_options(self._role == LLMRole.PROMPT, self._engine_options)
        return self._engine_config.cluster_config


    def check_link_status(self, remote_cluster_id: int):
        self._check_is_inited()
        self._check_is_not_cache_mgr_mode('check_link_status')
        check_uint64("remote_cluster_id", remote_cluster_id)
        ret = self._llm_datadist.check_link_status(remote_cluster_id)
        handle_llm_status(ret, '[check_link_status]', f"remote_cluster_id is {remote_cluster_id}")
        log.info('[check_link_status] success')


    def link_clusters(self, clusters: Union[List[LLMClusterInfo], Tuple[LLMClusterInfo]], timeout=3000):
        self._check_is_inited()
        check_int32("timeout", timeout)
        raise_if_false(timeout > 0, "Param timeout should be greater than 0.")
        check_isinstance("clusters", clusters, [list, tuple], LLMClusterInfo)
        cluster_list = [(cluster.remote_cluster_id,
                         0,
                         cluster.local_ip_info_list,
                         cluster.remote_ip_info_list) for cluster in clusters]
        ret, rets = self._llm_datadist.link_clusters(cluster_list, timeout)
        return code_2_status(ret), [code_2_status(cluster_ret) for cluster_ret in rets]

    def unlink_clusters(self, clusters: Union[List[LLMClusterInfo], Tuple[LLMClusterInfo]], timeout=3000, force=False):
        self._check_is_inited()
        self._check_is_not_cache_mgr_mode('unlink_clusters')
        check_int32("timeout", timeout)
        raise_if_false(timeout > 0, "Param timeout should be greater than 0.")
        check_isinstance("clusters", clusters, [list, tuple], LLMClusterInfo)
        check_isinstance("force", force, bool)
        cluster_list = [(cluster.remote_cluster_id,
                         0,
                         cluster.local_ip_info_list,
                         cluster.remote_ip_info_list) for cluster in clusters]
        ret, rets = self._llm_datadist.unlink_clusters(cluster_list, timeout, force)
        return code_2_status(ret), [code_2_status(cluster_ret) for cluster_ret in rets]

    def switch_role(self, role: LLMRole, switch_options: Optional[Dict[str, str]] = None):
        self._check_is_inited()
        check_isinstance('role', role, LLMRole)
        raise_if_false(self._role != role, f'role not changed, role = {role.name}')
        role_str = self._role_to_str(role)
        log.info(f'[switch_role] [{self._role.name}->{role.name}] start, switch_options = {switch_options}')
        check_isinstance('switch_options', switch_options, dict)
        options = switch_options.copy() if switch_options is not None else {}
        if role == LLMRole.PROMPT:
            raise_if_false('llm.listenIpInfo' in switch_options,
                           'Failed to switch to Prompt, option "llm.listenIpInfo" was specified')
            listen_ip_info = switch_options['llm.listenIpInfo']
            ip, port = EngineConfig.parse_listen_ip_info(listen_ip_info)
            options['llm.ListenIp'] = str(ip)
            options['llm.ListenPort'] = str(port)
        ret = self._llm_datadist.switch_role(role_str, options)
        handle_llm_status(ret,
                          '[LLMEngine.switch_role]',
                          f'Failed to switch role, role = {role}, options = {options}')
        self._kv_cache_manager._switch_role(role)
        log.info(f'[switch_role] [{self._role.name}->{role.name}] success')
        self._role = role

    @staticmethod
    def _role_to_str(role: LLMRole) -> str:
        role_mapping = {
            LLMRole.PROMPT: 'Prompt',
            LLMRole.DECODER: 'Decoder',
            LLMRole.MIX: 'Mix',
        }
        return role_mapping[role]

    def _check_is_inited(self):
        if not self._is_initialized:
            raise RuntimeError('llm datadist is not initialized')

    @property
    def kv_cache_manager(self) -> KvCacheManager:
        """
        获取KvCacheManager

        Returns:
            KvCacheManager
        """
        self._check_is_inited()
        self._check_is_not_cache_mgr_mode('kv_cache_manager')
        return self._kv_cache_manager

    @property
    def cluster_id(self):
        return self._cluster_id


def _shutdown_handler():
    if LLMDataDist.llm_engine_instance is not None:
        log.info('[shutdown_handler] finalize llm datadist')
        try:
            LLMDataDist.llm_engine_instance.finalize()
        except LLMException as e:
            log.warn(f'error occurred while finalize llm datadist: {e} '
                     f'may cause by already finalized by another framework')


atexit.register(_shutdown_handler)
