/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/cache_policy/aging_policy_lru.h"
namespace ge {
std::vector<CacheItemId> AgingPolicyLru::DoAging(const CacheState &cache_state) const {
  const auto &cc_state = cache_state.GetState();
  if (cache_state.GetCurTimerCount() <= delete_interval_) {
    GELOGE(ge::PARAM_INVALID, "[Aging][Lru]Delete interval param is invalid. Delete interval is %lu, expect[0, %lu].",
           delete_interval_, cache_state.GetCurTimerCount());
    return {};
  }
  const uint64_t timer_count_lower_bound = cache_state.GetCurTimerCount() - delete_interval_;
  std::vector<CacheItemId> delete_item;
  for (const auto &cache_item : cc_state) {
    const std::vector<CacheInfo> &cache_vec = cache_item.second;
    for (auto iter = cache_vec.begin(); iter != cache_vec.end(); iter++) {
      if ((*iter).GetTimerCount() < timer_count_lower_bound) {
          delete_item.emplace_back((*iter).GetItemId());
      }
    }
  }
  return delete_item;
}
}
