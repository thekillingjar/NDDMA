/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_PYTHON_LLM_WRAPPER_LLM_DATADIST_WRAPPER_H
#define AIR_PYTHON_LLM_WRAPPER_LLM_DATADIST_WRAPPER_H

#include "llm_datadist.h"
#include "llm_tensor.h"

namespace llm {
using ClusterInfoTuple = std::tuple<uint64_t,
                                    int32_t,
                                    std::vector<std::pair<uint32_t, uint16_t>>,
                                    std::vector<std::pair<uint32_t, uint16_t>>>;

using CacheTuple = std::tuple<int64_t, std::vector<std::vector<uintptr_t>>>;

using CacheDescTuple = std::tuple<uint32_t, int32_t, int32_t, std::vector<int64_t>, uint32_t, uint32_t>;

using PullCacheParamTuple = std::tuple<int64_t, uint32_t, std::vector<uint64_t>, std::vector<uint64_t>,
                                       std::vector<uint64_t>, std::vector<uint64_t>, int64_t, int64_t, uint64_t>;

using CacheKeyTuple = std::tuple<uint64_t, int64_t, uint64_t, uint64_t, uint64_t, uint64_t, bool>;

using MemInfoTuple = std::tuple<uint32_t, uint64_t, uint64_t>;

using CopyCacheParamTuple = std::tuple<int64_t,
                                       int64_t,
                                       uint32_t,
                                       uint32_t,
                                       uint64_t,
                                       int64_t,
                                       uint64_t,
std::vector<std::pair<uint64_t, uint64_t>>>;

using TransferBlockConfigTuple = std::tuple<uint64_t, std::vector<uint64_t>, std::vector<uint64_t>>;

using TransferCacheConfigTuple =
  std::tuple<uint64_t, uint64_t, uint64_t, std::vector<uintptr_t>, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
             uint64_t>;

class LLMDataDistWrapper {
 public:
  static ge::Status Init(uint64_t cluster_id, const std::map<std::string, std::string> &options);

  static void Finalize();

  static CacheDesc UnpackCacheDesc(const CacheDescTuple &cache_desc_tuple);
  static CopyCacheParam UnpackCopyCacheParam(CopyCacheParamTuple cache_param_tuple);
  static CacheKey UnpackCacheKey(const CacheKeyTuple &cache_key_tuple);
  static Cache UnpackCacheTuple(const CacheTuple &cache_tuple);
  static std::vector<CacheKey> UnpackCacheKeys(const std::vector<CacheKeyTuple> &cache_key_tuples);
  static PullCacheParam UnpackPullCacheParam(const PullCacheParamTuple &pull_cache_param_tuple);
  static TransferCacheConfig UnpackTransferCacheConfig(const TransferCacheConfigTuple &transfer_cache_config_tuple);
  static TransferBlockConfig UnpackTransferBlockConfig(const TransferBlockConfigTuple &transfer_block_config_tuple);
  static LLMMemInfo UnpackMemInfo(const MemInfoTuple &mem_info_tuple);
  static std::vector<LLMMemInfo> UnpackMemInfos(const std::vector<MemInfoTuple> &mem_info_tuples);

  static ge::Status CheckLinkStatus(uint64_t remote_cluster_id);
  static std::pair<ge::Status, std::vector<ge::Status>> LinkClusters(
      const std::vector<ClusterInfoTuple> &clusters,
      int32_t timeout);

  static std::pair<ge::Status, std::vector<ge::Status>> UnlinkClusters(
      const std::vector<ClusterInfoTuple> &clusters,
      int32_t timeout, bool force_flag);

  static std::pair<ge::Status, CacheTuple> AllocateCache(const CacheDescTuple &cache_desc,
                                                         const std::vector<CacheKeyTuple> &cache_keys);

  static ge::Status DeallocateCache(int64_t cache_id);

  static ge::Status PullCache(int64_t cache_id,
                              const CacheKeyTuple &cache_key,
                              const PullCacheParamTuple &pull_cache_param);

  static ge::Status TransferCache(uint64_t task_id, const TransferCacheConfigTuple &transfer_cache_config_tuple,
                                  const TransferBlockConfigTuple &transfer_block_config_tuple);

  static ge::Status CopyCache(CopyCacheParamTuple copy_cache_param);

  static ge::Status RemoveCacheKey(const CacheKeyTuple &cache_key_tuple);

  static std::pair<ge::Status, std::vector<TensorIdAndDesc>> GetCachedTensor(int64_t cache_id, int32_t tensor_index);

  static ge::Status SwapBlocks(const CacheTuple &src, const CacheTuple &dst, const uint64_t block_size,
                               const uint32_t type, const std::vector<std::pair<int64_t, int64_t>> &block_mapping);

  static ge::Status SwitchRole(const std::string &role, std::map<std::string, std::string> &options);

  static std::vector<llm::ClusterInfo> UnpackClusterInfos(const std::vector<ClusterInfoTuple> &clusters);

 private:
  static std::unique_ptr<LLMDataDist> llm_data_dist;
};
}  // namespace llm

#endif  // AIR_PYTHON_LLM_WRAPPER_LLM_DATADIST_WRAPPER_H
