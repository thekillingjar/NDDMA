/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/cache_policy/cache_state.h"
#include "framework/common/debug/ge_log.h"
namespace ge {
CacheItemId CacheState::GetNextCacheItemId() {
  const std::lock_guard<std::mutex> lock(cache_item_mu_);
  if (cache_item_queue_.empty()) {
    return cache_item_counter_++;
  } else {
    const CacheItemId next_item_id = cache_item_queue_.front();
    cache_item_queue_.pop();
    return next_item_id;
  }
}

void CacheState::RecoveryCacheItemId(const std::vector<CacheItemId> &cache_items) {
  const std::lock_guard<std::mutex> lock(cache_item_mu_);
  for (auto &item_id : cache_items) {
    cache_item_queue_.push(item_id);
  }
}

CacheItemId CacheState::AddCache(const CacheHashKey main_hash_key, const CacheDescPtr &cache_desc) {
  const std::lock_guard<std::mutex> lock(cache_info_queue_mu_);
  const auto iter = cache_info_queue.cc_state_.find(main_hash_key);
  if (iter == cache_info_queue.cc_state_.end()) {
    const CacheItemId next_item_id = GetNextCacheItemId();
    const CacheInfo cache_info = CacheInfo(GetNextTimerCount(), next_item_id, cache_desc);
    std::vector<CacheInfo> info = {cache_info};
    cache_info_queue.Insert(main_hash_key, info);
    return next_item_id;
  }
  auto &cache_infos = iter->second;
  for (auto &cache_info : cache_infos) {
    if (cache_desc->IsEqual(cache_info.desc_)) {
      cache_info.RefreshTimerCount(GetNextTimerCount());
      GELOGW("[AddCache] Same CacheDesc has already been added, whose cache_item is %" PRIu64, cache_info.item_id_);
      return cache_info.item_id_;
    }
  }
  // hash collision may happened
  const CacheItemId next_item_id = GetNextCacheItemId();
  CacheInfo cache_info = CacheInfo(GetNextTimerCount(), next_item_id, cache_desc);
  cache_info_queue.EmplaceBack(main_hash_key, cache_info);
  return next_item_id;
}

std::vector<CacheItemId> CacheState::DelCache(const DelCacheFunc &func) {
  std::vector<CacheItemId> delete_item;
  const std::lock_guard<std::mutex> lock(cache_info_queue_mu_);
  cache_info_queue.Erase(delete_item, func);

  RecoveryCacheItemId(delete_item);
  return delete_item;
}

std::vector<CacheItemId> CacheState::DelCache(const std::vector<CacheItemId> &delete_item) {
  const DelCacheFunc lamb = [&delete_item] (const CacheInfo &info) -> bool {
    const auto iter = std::find(delete_item.begin(), delete_item.end(), info.GetItemId());
    return iter != delete_item.end();
  };
  return DelCache(lamb);
}

void CacheInfoQueue::Insert(const CacheHashKey main_hash_key, std::vector<CacheInfo> &cache_info) {
  (void) cc_state_.insert({main_hash_key, std::move(cache_info)});
  ++cache_info_num_;
}
void CacheInfoQueue::EmplaceBack(const CacheHashKey main_hash_key, CacheInfo &cache_info) {
  cc_state_[main_hash_key].emplace_back(std::move(cache_info));
  ++cache_info_num_;
}
void CacheInfoQueue::Erase(std::vector<CacheItemId> &delete_ids, const DelCacheFunc &is_need_delete_func) {
  for (auto &item : cc_state_) {
    std::vector<CacheInfo> &cache_vec = item.second;
    for (auto iter = cache_vec.begin(); iter != cache_vec.end();) {
      if (is_need_delete_func(*iter)) {
        delete_ids.emplace_back((*iter).GetItemId());
        iter = cache_vec.erase(iter);
        --cache_info_num_;
      } else {
        iter++;
      }
    }
  }
}
}  // namespace ge
