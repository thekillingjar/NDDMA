/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tiling_cache.h"

#include "framework/common/debug/ge_log.h"
#include "graph/utils/hash_utils.h"
#include "exe_graph/lowering/data_dependent_interpreter.h"
#include "mmpa/mmpa_api.h"

namespace {
// should ensure buf is not null
size_t InnerHashFunc(const uint8_t *const buf, const size_t len) {
  constexpr size_t block_len = sizeof(int64_t);
  const auto *block_ptr = static_cast<const int64_t *>(static_cast<const void *>(&buf[0]));
  const size_t num_block = len / block_len;
  size_t seed = ge::HashUtils::MultiHash();
  for (size_t i = 0; i < num_block; i++) {
    seed = ge::HashUtils::HashCombine(seed, block_ptr[i]);
  }
  // process tail block, because the size of a block is 8
  const auto *tail_block = &(buf[num_block * block_len]);
  const size_t tail_len = static_cast<uint64_t>(len) & (block_len - 1);
  for (size_t i = 0; i < tail_len; i++) {
    seed = ge::HashUtils::HashCombine(seed, tail_block[i]);
  }
  return seed;
}
}  // namespace

namespace gert {
const int64_t HashBuffer::sep_ = 0x2323232323232323;
thread_local HashBuffer* HashBuffer::occupier_ = nullptr;
thread_local size_t HashBuffer::offset_ = 0UL;
thread_local uint8_t HashBuffer::hash_buf_[kMaxHashBufSize];

HashBuffer::HashBuffer() {
  // 在同一个作用域内, 只有第一个实例可以使用,
  if (occupier_ == nullptr) {
    occupier_ = this;
    offset_ = 0UL;
  }
}

HashBuffer::~HashBuffer() {
  if (occupier_ == this) {
    occupier_ = nullptr;
  }
}

void HashBuffer::AddParamToBuf(const Shape &shape) {
  if (occupier_ != this) {
    return;
  }
  const auto dim_num = shape.GetDimNum();
  if (offset_ + dim_num * sizeof(int64_t) > kMaxHashBufSize) {
    offset_ = kInvalidHashBufOffset;
    return;
  }
  AddShapeToBuf(shape, dim_num);
}

void HashBuffer::AddShapeToBuf(const Shape &shape, const size_t &dim_num) const {
  for (size_t i = 0; i < dim_num; ++i) {
    *static_cast<int64_t *>(static_cast<void *>(&hash_buf_[offset_])) = shape.GetDim(i);
    offset_ += sizeof(int64_t);
  }
  AddSeparator();
}

void HashBuffer::AddParamToBuf(const Tensor &tensor) {
  if (occupier_ != this) {
    return;
  }
  const auto& origin_shape = tensor.GetOriginShape();
  const auto dim_num = origin_shape.GetDimNum();
  const auto param_size = offset_ + dim_num * sizeof(int64_t) + tensor.GetSize();
  if (param_size > kMaxHashBufSize || tensor.GetAddr() == nullptr ||
      !TensorPlacementUtils::IsOnHost(tensor.GetPlacement())) {
    offset_ = kInvalidHashBufOffset;
    return;
  }
  AddShapeToBuf(origin_shape, dim_num);
  if (offset_ == kInvalidHashBufOffset) {
    return;
  }

  const auto ret = memcpy_s(&hash_buf_[offset_], kMaxHashBufSize - offset_, tensor.GetAddr(), tensor.GetSize());
  if (ret != EOK) {
    GELOGW("Failed to add param to hash buffer, offset %zu, size %zu", offset_, tensor.GetSize());
    offset_ = kInvalidHashBufOffset;
    return;
  }
  offset_ += tensor.GetSize();
  AddSeparator();
}

void HashBuffer::AddParamToBuf(const int64_t &dim) {
  if (occupier_ != this) {
    return;
  }
  if (offset_ + sizeof(int64_t) > kMaxHashBufSize) {
    offset_ = kInvalidHashBufOffset;
    return;
  }
  *static_cast<int64_t *>(static_cast<void *>(&hash_buf_[offset_])) = dim;
  offset_ += sizeof(int64_t);
  AddSeparator();
}

TilingCacheKey HashBuffer::GetTilingCacheKey() const {
  return (occupier_ == this) ? TilingCacheKey{hash_buf_, offset_} : TilingCacheKey{nullptr, 0};
}

void HashBuffer::AddSeparator() {
  if (offset_ + sizeof(sep_) > kMaxHashBufSize) {
    offset_ = kInvalidHashBufOffset;
    return;
  }
  *static_cast<int64_t *>(static_cast<void *>(&hash_buf_[offset_])) = sep_;
  offset_ += sizeof(sep_);
}

bool TilingCacheKey::IsValid() const {
  return (buf != nullptr) && (len > 0) && (len <= kMaxHashBufSize);
}

bool TilingCacheKey::operator==(const TilingCacheKey &other) const {
  return (len == other.len) && (std::memcmp(buf, other.buf, len) == 0);
}

size_t TilingCacheKeyHash::operator()(const TilingCacheKey &key) const {
  return InnerHashFunc(key.buf, key.len);
}

TilingCache::TilingCache(const TilingCacheKey &key, TilingCacheValue value) {
  InitCacheKey(key);
  cache_value_ = std::move(value);
}

TilingCache::TilingCache(TilingCache &&other) noexcept {
  cache_key_ = other.cache_key_;
  cache_value_ = std::move(other.cache_value_);
  // reset old TilingCache object
  other.cache_key_ = {};
  other.cache_value_ = {};
}

TilingCache &TilingCache::operator=(TilingCache &&other) noexcept {
  if (this != &other) {
    ReleaseCacheKey();
    cache_key_ = other.cache_key_;
    cache_value_ = std::move(other.cache_value_);
    other.cache_key_ = {};
    other.cache_value_ = {};
  }
  return *this;
}

TilingCache::~TilingCache() {
  ReleaseCacheKey();
}

TilingCacheKey TilingCache::GetTilingCacheKey() const {
  return cache_key_;
}

const TilingCacheValue &TilingCache::GetTilingCacheValue() const {
  return cache_value_;
}

TilingCacheValue &TilingCache::GetTilingCacheValue() {
  return cache_value_;
}

void *TilingCache::GetLaunchArgPtr() const {
  return cache_value_.launch_arg_holder.get();
}

void TilingCache::InitCacheKey(const TilingCacheKey &key) {
  cache_key_.len = key.len;
  cache_key_.buf = new (std::nothrow) uint8_t[cache_key_.len];
  if (cache_key_.buf == nullptr) {
    GELOGW("Failed to create buffer for cache_key_");
    return;
  }
  auto ret = memcpy_s(cache_key_.buf, cache_key_.len, key.buf, key.len);
  if (ret != EOK) {
    GELOGW("Failed to copy hash buffer");
    ReleaseCacheKey();
  }
}

void TilingCache::ReleaseCacheKey() {
  if (cache_key_.buf != nullptr) {
    delete[] cache_key_.buf;
    cache_key_.buf = nullptr;
  }
  cache_key_.len = 0;
}

TilingCache *TilingCacheManager::AddNewCache(const TilingCacheKey &key, TilingCacheValue value) {
  // move语义会清空原value的key, 此处new_cache_key不能为引用
  TilingCache new_cache(key, std::move(value));
  const auto new_cache_key = new_cache.GetTilingCacheKey();
  return cache_strategy_->Put(new_cache_key, std::move(new_cache));
}

const TilingCache *TilingCacheManager::TryFetchCache(const TilingCacheKey &key) const {
  if (!key.IsValid()) {
    return nullptr;
  }
  return cache_strategy_->Get(key);
}

bool TilingCacheManager::Exist(const TilingCacheKey &key) const {
  return cache_strategy_->Exist(key);
}

bool TilingCacheUtils::IsOpSupportTilingCache(const ge::NodePtr &node, LoweringGlobalData &global_data,
                                              size_t &data_dependency) {
  data_dependency = 0U;
  GE_ASSERT_NOTNULL(node);
  GE_ASSERT_NOTNULL(node->GetOpDesc());
  DataDependentInterpreter ddi(node->GetOpDesc(), global_data.GetSpaceRegistriesV2());
  const size_t input_size = node->GetInDataNodesSize();
  for (size_t i = 0; i < input_size; ++i) {
    bool is_data_dependent = false;
    auto ret = ddi.IsDataDependent(static_cast<int32_t>(i), is_data_dependent);
    GE_ASSERT_TRUE(ret == ge::SUCCESS);
    bool is_tiling_dependent = false;
    if (!is_data_dependent) {
      ret = ddi.IsTilingInputDataDependent(static_cast<int32_t>(i), is_tiling_dependent);
      GE_ASSERT_TRUE(ret == ge::SUCCESS);
    }
    GELOGD("Node:%s, input:%zu, data/tiling depend flag:%d/%d.", node->GetName().c_str(), i, is_data_dependent,
           is_tiling_dependent);
    if (is_data_dependent || is_tiling_dependent) {
      if (input_size >= sizeof(data_dependency) * kByteBitCount) {
        return false;
      }
      data_dependency |= (static_cast<size_t>(1) << i);
    }
  }
  return true;
}
}  // namespace gert