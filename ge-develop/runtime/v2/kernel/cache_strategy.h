/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_CORE_CACHE_CACHE_STRATEGY_H
#define AIR_CXX_RUNTIME_V2_CORE_CACHE_CACHE_STRATEGY_H
#include <list>
#include <unordered_map>

namespace gert {
template <typename KeyT, typename ValueT>
class CacheStrategy {
 public:
  virtual ~CacheStrategy() = default;
  virtual ValueT *Put(const KeyT &key, ValueT value) = 0;
  virtual ValueT *Get(const KeyT &key) = 0;
  virtual bool Exist(const KeyT &key) const = 0;
};

template <typename KeyT, typename ValueT, typename HashFunc>
class LruCacheStrategy : public CacheStrategy<KeyT, ValueT> {
 public:
  using KvPairT = std::pair<KeyT, ValueT>;
  using ListIteratorT = typename std::list<KvPairT>::iterator;

  explicit LruCacheStrategy(const size_t capacity, const size_t evict_num)
      : capacity_(capacity), evict_num_(evict_num) {}

  auto Put(const KeyT &key, ValueT value) -> ValueT* override {
    auto ret = cache_map_.emplace(key, cache_list_.end());
    if (ret.second) {
      ret.first->second = cache_list_.emplace(cache_list_.begin(), key, std::move(value));
    }
    if (Size() > capacity_) {
      Evict();
    }
    return &(ret.first->second->second);
  }

  ValueT *Get(const KeyT &key) override {
    auto it = cache_map_.find(key);
    if (it == cache_map_.end()) {
      return nullptr;
    }
    cache_list_.splice(cache_list_.begin(), cache_list_, it->second);
    return &(it->second->second);
  }

  bool Exist(const KeyT &key) const override {
    return (cache_map_.find(key) != cache_map_.end());
  }

 private:
  void Evict() {
    for (size_t i = 0UL; i < evict_num_; i++) {
      // 当 capacity_ <= evict_num_ 时，至少保留1个缓存
      if (Size() > 1UL) {
        auto last_it = cache_list_.rbegin();
        cache_map_.erase(last_it->first);
        cache_list_.pop_back();
      } else {
        break;
      }
    }
  }

  size_t Size() const {
    return cache_map_.size();
  }

 private:
  size_t capacity_;
  size_t evict_num_;
  std::list<KvPairT> cache_list_;
  std::unordered_map<KeyT, ListIteratorT, HashFunc> cache_map_;
};
}  // namespace gert

#endif  // AIR_CXX_RUNTIME_V2_CORE_CACHE_CACHE_STRATEGY_H
