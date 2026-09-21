/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_CORE_CACHE_TILING_CACHE_H
#define AIR_CXX_RUNTIME_V2_CORE_CACHE_TILING_CACHE_H
#include <memory>
#include "cache_strategy.h"
#include "graph/types.h"
#include "graph/node.h"
#include "exe_graph/runtime/tensor.h"
#include "exe_graph/runtime/shape.h"
#include "lowering/lowering_global_data.h"

namespace gert {
constexpr size_t kMaxHashBufSize = 8192UL;
constexpr size_t kInvalidHashBufOffset = kMaxHashBufSize + 1024UL;

struct TilingCacheKey {
  uint8_t *buf;
  size_t len;

  TilingCacheKey() : buf(nullptr), len(0) {}
  TilingCacheKey(uint8_t *buffer, const size_t length) : buf(buffer), len(length) {}
  bool IsValid() const;
  bool operator==(const TilingCacheKey &other) const;
};

struct TilingCacheKeyHash {
  size_t operator()(const TilingCacheKey &key) const;
};

class HashBuffer {
 public:
  HashBuffer();
  ~HashBuffer();
  void AddParamToBuf(const Shape &shape);
  void AddParamToBuf(const Tensor &tensor);
  void AddParamToBuf(const int64_t &dim);
  void AddShapeToBuf(const Shape &shape, const size_t &dim_num) const;
  TilingCacheKey GetTilingCacheKey() const;

 private:
  static void AddSeparator();

 private:
  static const int64_t sep_;
  thread_local static HashBuffer *occupier_;
  thread_local static size_t offset_;
  thread_local static uint8_t hash_buf_[kMaxHashBufSize];
};

struct TilingCacheValue {
  bool atomic_clean_flag;
  int32_t tiling_cond;
  uint32_t local_mem_size;
  uint64_t block_dim;
  uint64_t tiling_key;
  size_t ori_tiling_data_size;
  size_t dfx_dump_data_num;
  std::unique_ptr<uint8_t[]> workspace_sizes_holder;  // TypedContinuousVector<size_t>
  std::unique_ptr<uint8_t[]> launch_arg_holder;
  std::unique_ptr<uint64_t[]> dfx_dump_data_holder;  // memory length = dfx_dump_data_num * sizeof(uint64_t)
};

class TilingCache {
 public:
  TilingCache(const TilingCacheKey &key, TilingCacheValue value);
  TilingCache(TilingCache &&other) noexcept;
  TilingCache &operator=(TilingCache &&other) noexcept;
  ~TilingCache();

  TilingCacheKey GetTilingCacheKey() const;
  const TilingCacheValue &GetTilingCacheValue() const;
  TilingCacheValue &GetTilingCacheValue();
  void *GetLaunchArgPtr() const;

 private:
  void InitCacheKey(const TilingCacheKey &key);
  void ReleaseCacheKey();

 private:
  TilingCacheKey cache_key_;
  TilingCacheValue cache_value_;
};

using TilingCacheStrategy = CacheStrategy<TilingCacheKey, TilingCache>;
using TilingCacheLruStrategy = LruCacheStrategy<TilingCacheKey, TilingCache, TilingCacheKeyHash>;

class TilingCacheManager {
 public:
  // strategy is guaranteed to be a non-null pointer when created
  explicit TilingCacheManager(std::unique_ptr<TilingCacheStrategy> strategy) : cache_strategy_(std::move(strategy)) {}
  TilingCache *AddNewCache(const TilingCacheKey &key, TilingCacheValue value);
  const TilingCache *TryFetchCache(const TilingCacheKey &key) const;
  bool Exist(const TilingCacheKey &key) const;

 private:
  std::unique_ptr<TilingCacheStrategy> cache_strategy_;
};

class TilingCacheUtils {
 public:
  static constexpr size_t kByteBitCount = 8U;
  static bool IsOpSupportTilingCache(const ge::NodePtr &node, LoweringGlobalData &global_data, size_t &data_dependency);
};
}  // namespace gert

#endif  // AIR_CXX_RUNTIME_V2_CORE_CACHE_TILING_CACHE_H
