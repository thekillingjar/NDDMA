/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/cache_policy/match_policy_for_exactly_the_same.h"

namespace ge {
CacheItemId MatchPolicyForExactlyTheSame::GetCacheItemId(const CCStatType &cc_state,
                                                         const CacheDescPtr &cache_desc) const {
  const CacheHashKey hash_key = cache_desc->GetCacheDescHash();
  const auto &iter = cc_state.find(hash_key);
  if (iter == cc_state.end() || iter->second.empty()) {
    GELOGD("[CACHE] hash [%lu] does not exist.", hash_key);
    return KInvalidCacheItemId;
  }
  const auto &info_vec = iter->second;
  const auto cached_info = std::find_if(info_vec.begin(), info_vec.end(), [&cache_desc](const CacheInfo &cached) {
    return (cache_desc->IsEqual(cached.GetCacheDesc()));
  });
  if (cached_info != info_vec.cend()) {
    return cached_info->GetItemId();
  } else {
    GELOGD("[CACHE] hash [%lu] collision occurred, the same cached desc not found.", hash_key);
    return KInvalidCacheItemId;
  }
}
}  // namespace ge
