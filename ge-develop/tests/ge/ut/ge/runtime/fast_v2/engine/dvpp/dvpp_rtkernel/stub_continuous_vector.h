/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef STUB_EXE_GRAPH_RUNTIME_CONTINUOUS_VECTOR_H_
#define STUB_EXE_GRAPH_RUNTIME_CONTINUOUS_VECTOR_H_
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <string.h>
#include "stub_ge_error_codes.h"

namespace gert {
class ContinuousVector {
 public:
  /**
   * 创建一个ContinuousVector实例，ContinuousVector不支持动态扩容
   * @tparam T 实例中包含的元素类型
   * @param capacity 实例的最大容量
   * @param total_size 本实例的总长度
   * @return 指向本实例的指针
   */
  size_t GetSize() const {
    return size_;
  }

  uint32_t SetSize(const size_t size) {
    if (size > capacity_) {
      return -1;
    }
    size_ = size;
    return 0;
  }

  const void *GetData() const {
    return elements;
  }

  void *MutableData() {
    return elements;
  }

  template<typename T>
  static std::unique_ptr<uint8_t[]> Create(size_t capacity, size_t &total_size) {
      return nullptr;
  }

  template<typename T>
  static std::unique_ptr<uint8_t[]> Create(const size_t capacity) {
    size_t total_size;
    return Create<T>(capacity, total_size);
  }

 private:
  size_t capacity_;
  size_t size_;
  uint8_t reserved_[40]; // Reserved field, 32+8, do not directly use when only 8-byte left
  uint8_t elements[8];
};

/*
 * memory layout: |size_|offset1|offset2|...|ContinuousVector1|ContinuousVector1|...|
 * size_ : number of ContinuousVector
 * offset1 : offset of ContinuousVector1u
 */
class ContinuousVectorVector {
 public:

  const ContinuousVector *Get(const size_t index) const {
    return nullptr;
  }

  size_t GetSize() const {
    return size_;
  }

 private:
  size_t capacity_ = 0U;
  size_t size_ = 0U;
  uint8_t reserved_[40];  // Reserved field, 32+8, do not directly use when only 8-byte left
  size_t offset_[1U];
};

} // namespace gert

#endif // STUB_EXE_GRAPH_RUNTIME_CONTINUOUS_VECTOR_H_