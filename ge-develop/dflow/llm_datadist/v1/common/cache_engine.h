/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_RUNTIME_LLM_ENGINE_COMMON_CACHE_ENGINE_H_
#define AIR_RUNTIME_LLM_ENGINE_COMMON_CACHE_ENGINE_H_

#include "common/llm_flow_service.h"

namespace llm {
class CacheEngine {
 public:
  explicit CacheEngine(LlmFlowService *llm_flow_service) : llm_flow_service_(llm_flow_service) {}
  ~CacheEngine() = default;
  ge::Status Allocate(const CacheDesc &cache_desc,
                      const std::vector<CacheKey> &cache_keys,
                      Cache &cache);
  ge::Status Deallocate(int64_t cache_id);
  ge::Status RemoveCacheKey(const CacheKey &cache_key) const;
  ge::Status PullCache(int64_t cache_id, const CacheKey &cache_key, const PullCacheParam &pull_cache_param);
  ge::Status TransferCache(uint64_t task_id, const TransferCacheConfig &transfer_cache_config,
                           const TransferBlockConfig &transfer_block_config);
  ge::Status CopyCache(const CopyCacheParam &copy_cache_param);
  ge::Status GetCacheTensors(int64_t cache_id,
                             std::vector<ge::Tensor> &outputs,
                             int32_t tensor_index);

 private:
  std::mutex mu_;
  std::atomic_int64_t cache_id_gen_{1};
  LlmFlowService *llm_flow_service_;
  std::map<int64_t, CacheDesc> cache_id_to_desc_;
};
}  // namespace llm

#endif  // AIR_RUNTIME_LLM_ENGINE_COMMON_CACHE_ENGINE_H_
