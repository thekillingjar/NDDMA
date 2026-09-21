/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cache_desc_stub//runtime_cache_desc.h"
#include "common/checker.h"

namespace ge {
bool RuntimeCacheDesc::IsEqual(const CacheDescPtr &other) const {
  auto rht = dynamic_cast<const RuntimeCacheDesc *>(other.get());
  GE_ASSERT_NOTNULL(rht, "Type error, expect type ge::RuntimeCacheDesc.");
  return (*this == *rht);
}

bool RuntimeCacheDesc::IsMatch(const CacheDescPtr &other) const {
  return IsEqual(other);
}

CacheHashKey RuntimeCacheDesc::GetCacheDescHash() const {
  CacheHashKey hash_key = 0U;
  const char_t sep = ',';
  for (const auto &shape : shapes_) {
    auto seed = HashUtils::MultiHash();
    for (size_t idx = 0U; idx < shape.GetDimNum(); ++idx) {
      seed = HashUtils::HashCombine(seed, shape.GetDim(idx));
    }
    hash_key = HashUtils::HashCombine(seed, sep);
  }
  return hash_key;
}

bool RuntimeCacheDesc::operator==(const RuntimeCacheDesc &rht) const {
  if (shapes_.size() != rht.shapes_.size()) {
    return false;
  }
  for (size_t idx = 0U; idx < shapes_.size(); ++idx) {
    if (shapes_[idx] != rht.shapes_[idx]) {
      return false;
    }
  }
  return true;
}
}  // namespace ge
