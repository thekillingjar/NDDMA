/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/large_bm.h"
#include "framework/common/debug/ge_log.h"

namespace ge {
constexpr size_t kBitsEachValue = 64UL;

constexpr size_t AlignBitSize(size_t bit_size) {
  return bit_size + kBitsEachValue - 1;
}

constexpr size_t AlignArraySize(size_t bit_size) {
  return AlignBitSize(bit_size) >> 6; // shifting right by 6 bits
}

void LargeBitmap::ResizeBits(size_t new_size) {
  if (new_size < size_) {
    return;
  }

  size_t new_byte_size = AlignArraySize(new_size);
  if (new_byte_size == AlignArraySize(size_)) {
    size_ = new_size;
    return;
  }

  this->bits_.resize(new_byte_size, 0);
  for (size_t i = size_; i < AlignBitSize(size_); ++i) {
    ClearBit(i);
  }

  size_ = new_size;
}

// Shifting right by 6 bits is equivalent to dividing by 64
void LargeBitmap::ClearBit(size_t bit_idx) {
  bits_[bit_idx >> 6] &= ~(1UL << (bit_idx % kBitsEachValue));
}

LargeBitmap::LargeBitmap(const size_t &size)
    : size_(size), bits_(AlignArraySize(size), 0UL) {}

bool LargeBitmap::operator==(const LargeBitmap &another_bm) const {
  return bits_ == another_bm.bits_;
}

bool LargeBitmap::operator!=(const LargeBitmap &another_bm) const {
  return bits_ != another_bm.bits_;
}

void LargeBitmap::SetValues(const uint64_t &value) {
  std::fill(bits_.begin(), bits_.end(), value);
}

void LargeBitmap::SetBit(const size_t &index) {
  if (index < size_) {
    bits_[index / kBitsEachValue] |= 1UL << (index % kBitsEachValue);
  } else {
    GE_LOGE("index %zu is not valid. Total size is %zu", index, size_);
    return;
  }
}

bool LargeBitmap::GetBit(const size_t &index) const {
  if (index < size_) {
    return static_cast<bool>(bits_[index / kBitsEachValue] & (1UL << (index % kBitsEachValue)));
  } else {
    GE_LOGE("index %zu is not valid. Total size is %zu", index, size_);
    return false;
  }
}

void LargeBitmap::Or(const LargeBitmap &another_bm) {
  size_t index = 0UL;
  const size_t another_size = another_bm.bits_.size();
  for (auto &bit : bits_) {
    if (index >= another_size) {
      return;
    }
    bit |= another_bm.bits_[index];
    ++index;
  }
}

void LargeBitmap::And(const LargeBitmap &another_bm) {
  size_t index = 0UL;
  const size_t another_size = another_bm.bits_.size();
  for (auto &bit : bits_) {
    if (index >= another_size) {
      return;
    }
    bit &= another_bm.bits_[index];
    ++index;
  }
}
}
