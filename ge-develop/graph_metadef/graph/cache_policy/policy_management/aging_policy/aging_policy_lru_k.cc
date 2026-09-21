/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/cache_policy/aging_policy_lru_k.h"
namespace ge {
std::vector<CacheItemId> AgingPolicyLruK::DoAging(const CacheState &cache_state) const {
  size_t cur_depth = cache_state.GetCacheInfoNum();
  const auto &cc_state = cache_state.GetState();
  GELOGD("[CACHE][AGING] current depth[%zu] cache queue capacity[%zu].", cur_depth, depth_);
  if (cur_depth <= depth_) {
    return {};
  }
  std::pair<CacheItemId, uint64_t> delete_item({KInvalidCacheItemId, UINT64_MAX});
  for (const auto &each_cc_state : cc_state) {
    for (const auto &cache_info : each_cc_state.second) {
      if (cache_info.GetTimerCount() <= delete_item.second) {
        delete_item = {cache_info.GetItemId(), cache_info.GetTimerCount()};
      }
    }
  }
  if (delete_item.first == KInvalidCacheItemId) {
    return {};
  }
  return {delete_item.first};
}

bool AgingPolicyLruK::IsCacheDescAppearKTimes(const CacheHashKey hash_key, const CacheDescPtr &cache_desc) {
  const std::lock_guard<std::mutex> lock(hash_2_cache_descs_and_count_mu_);
  if (hash_2_cache_descs_and_count_.count(hash_key) > 0U) {
    auto &cache_descs_and_count = hash_2_cache_descs_and_count_[hash_key];
    for (auto &cache_desc_and_count : cache_descs_and_count) {
      if (cache_desc->IsEqual(cache_desc_and_count.first)) {
        ++cache_desc_and_count.second;
        return cache_desc_and_count.second >= k_times_;
      }
    }
  }
  hash_2_cache_descs_and_count_[hash_key].emplace_back(std::make_pair(cache_desc, 1U));
  return false;
}
}  // namespace ge
