/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/large_bitmap.h"
namespace fe {
constexpr size_t kBitsEachValue = 64;

LargeBitMap::LargeBitMap(size_t size)
    : size_(size), bits_((size + kBitsEachValue - 1) / kBitsEachValue, 0) {}

bool LargeBitMap::operator==(const LargeBitMap &another_bm) const {
  return bits_ == another_bm.bits_;
}

bool LargeBitMap::operator!=(const LargeBitMap &another_bm) const {
  return bits_ != another_bm.bits_;
}

void LargeBitMap::SetValues(uint64_t value) {
  std::fill(bits_.begin(), bits_.end(), value);
}

void LargeBitMap::SetBit(size_t index) {
  if (index < size_) {
    bits_[index / kBitsEachValue] |= 1ull << (index % kBitsEachValue);
  } else {
    FE_LOGW("Index %zu is invalid. Total size is %zu.", index, size_);
    return;
  }
}

bool LargeBitMap::GetBit(size_t index) const {
  if (index < size_) {
    return bits_[index / kBitsEachValue] & (1ull << (index % kBitsEachValue));
  } else {
    FE_LOGW("Index %zu is invalid. Total size is %zu.", index, size_);
    return false;
  }
}

void LargeBitMap::Or(const LargeBitMap &another_bm) {
  size_t index = 0;
  size_t another_size = another_bm.bits_.size();
  for (auto &bit : bits_) {
    if (index >= another_size) {
      return;
    }
    bit |= another_bm.bits_[index];
    ++index;
  }
}
}
