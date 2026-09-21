/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GRAPH_CACHE_POLICY_CACHE_STATE_H
#define GRAPH_CACHE_POLICY_CACHE_STATE_H

#include <vector>
#include <functional>
#include <unordered_map>
#include <chrono>
#include <queue>
#include <mutex>

#include "compile_cache_desc.h"

namespace ge {
class CacheInfo;
using CacheItemId = uint64_t;
constexpr CacheItemId KInvalidCacheItemId = std::numeric_limits<uint64_t>::max();

using DelCacheFunc = std::function<bool(CacheInfo &)>;
using CCStatType = std::unordered_map<uint64_t, std::vector<CacheInfo>>;

class CacheInfo {
friend class CacheState;
public:
  CacheInfo(const uint64_t timer_count, const CacheItemId item_id, const CacheDescPtr &desc)
     : item_id_(item_id), desc_(desc), timer_count_(timer_count) {}
  CacheInfo(const CacheInfo &other)
     : item_id_(other.item_id_), desc_(other.desc_), timer_count_(other.timer_count_) {}
  CacheInfo &operator=(const CacheInfo &other) {
    timer_count_ = other.timer_count_;
    item_id_ = other.item_id_;
    desc_ = other.desc_;
    return *this;
  }
  CacheInfo() = delete;
  ~CacheInfo() = default;

  void RefreshTimerCount(uint64_t time_count) {
    timer_count_ = time_count;
  }

  uint64_t GetTimerCount() const noexcept {
    return timer_count_;
  }

  CacheItemId GetItemId() const noexcept {
    return item_id_;
  }

  const CacheDescPtr &GetCacheDesc() const noexcept {
    return desc_;
  }

private:
  CacheItemId item_id_;
  CacheDescPtr desc_;
  uint64_t timer_count_;
};

struct CacheInfoQueue {
  void Insert(const CacheHashKey main_hash_key, std::vector<CacheInfo> &cache_info);
  void EmplaceBack(const CacheHashKey main_hash_key, CacheInfo &cache_info);
  void Erase(std::vector<CacheItemId> &delete_ids, const DelCacheFunc &is_need_delete_func);

  CCStatType cc_state_;
  uint64_t cache_info_num_ = 0U;
};

class CacheState {
public:
  CacheState() = default;
  ~CacheState() = default;

  CacheItemId AddCache(const CacheHashKey main_hash_key, const CacheDescPtr &cache_desc);

  std::vector<CacheItemId> DelCache(const DelCacheFunc &func);

  std::vector<CacheItemId> DelCache(const std::vector<CacheItemId> &delete_item);

  const CCStatType &GetState() const {
    return cache_info_queue.cc_state_;
  }

  uint64_t GetCacheInfoNum() const {
    return cache_info_queue.cache_info_num_;
  }

  uint64_t GetCurTimerCount() const {
    return cache_timer_count_;
  }
private:
  CacheItemId GetNextCacheItemId();
  void RecoveryCacheItemId(const std::vector<CacheItemId> &cache_items);
  uint64_t GetNextTimerCount() {
    const std::lock_guard<std::mutex> lock(cache_timer_count_mu_);
    return cache_timer_count_++;
  }

  std::mutex cache_info_queue_mu_;
  std::mutex cache_item_mu_;

  int64_t cache_item_counter_ = 0L;
  std::queue<int64_t> cache_item_queue_;
  CacheInfoQueue cache_info_queue;

  uint64_t cache_timer_count_ = 0U;
  std::mutex cache_timer_count_mu_;
};
}  // namespace ge
#endif
