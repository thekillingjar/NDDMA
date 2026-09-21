/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_RUNTIME_LLM_ENGINE_INC_LLM_DATADIST_H
#define AIR_RUNTIME_LLM_ENGINE_INC_LLM_DATADIST_H

#include "ge/ge_ir_build.h"
#include "llm_datadist/llm_engine_types.h"
#include "llm_datadist/llm_error_codes.h"
#include "common/llm_inner_types.h"
#include "common/common.h"

namespace llm {
class LLMDataDist {
 public:
  explicit LLMDataDist(const uint64_t cluster_id);
  ~LLMDataDist();
  ge::Status LLMDataDistInitialize(const std::map<ge::AscendString, ge::AscendString> &options);

  void LLMDataDistFinalize();

  ge::Status CheckLinkStatus(uint64_t remote_cluster_id);

  // @brief 进行device间建链
  // @param [in] clusters 需要建链的cluster信息
  // @param [in] timeout 超时时间，单位ms
  // @param [out] rets 每个cluster建链结果
  // @return Status result of function
  //         SUCCESS: 成功
  //         其他做错误码: 执行推理失败
  ge::Status LinkClusters(const std::vector<ClusterInfo> &clusters, std::vector<ge::Status> &rets,
                          const int32_t timeout);

  // @brief 进行device间断链
  // @param [in] clusters 需要建链的cluster信息
  // @param [in] timeout 超时时间，单位ms
  // @param [out] rets 每个cluster断链结果
  // @return Status result of function
  //         SUCCESS: 成功
  //         其他做错误码: 执行推理失败
  ge::Status UnlinkClusters(const std::vector<ClusterInfo> &clusters, std::vector<ge::Status> &rets,
                            const int32_t timeout, bool force_flag = false);

  ge::Status AllocateCache(const CacheDesc &cache_desc,
                           Cache &cache,
                           const std::vector<CacheKey> &cache_keys = {});

  ge::Status DeallocateCache(int64_t cache_id);

  ge::Status RemoveCacheKey(const CacheKey &cache_key);

  ge::Status PullCache(int64_t cache_id, const CacheKey &cache_key, const PullCacheParam &pull_cache_param = {});

  ge::Status PullBlocks(int64_t cache_id, const CacheKey &cache_key, const PullCacheParam &pull_cache_param = {});

  ge::Status TransferCache(uint64_t task_id, const TransferCacheConfig &transfer_cache_config,
                           const TransferBlockConfig &transfer_block_config);

  ge::Status CopyCache(const CacheEntry &src_cache_entry, const CacheEntry &dst_cache_entry,
                       const CopyCacheParam &copy_cache_param, const std::vector<int32_t> &device_ids);

  ge::Status CopyCache(const CopyCacheParam &copy_cache_param);

  ge::Status GetCacheTensors(int64_t cache_id,
                             std::vector<ge::Tensor> &outputs,
                             int32_t tensor_index = -1);

  ge::Status SwitchRole(const std::string &role, const std::map<std::string, std::string> &options);

  ge::Status SwapBlocks(const Cache &src, const Cache &dst, const uint64_t block_size, const uint32_t type,
                        const std::vector<std::pair<int64_t, int64_t>> &block_mapping);

  bool IsInitialized() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace llm
#endif  // AIR_RUNTIME_LLM_ENGINE_INC_LLM_DATADIST_H
