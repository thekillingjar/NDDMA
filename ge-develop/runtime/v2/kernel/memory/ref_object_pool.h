/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_KERNEL_MEMORY_REF_OBJECT_POOL_H_
#define AIR_CXX_RUNTIME_V2_KERNEL_MEMORY_REF_OBJECT_POOL_H_
#include <cstdlib>
#include <cstdint>
#include <array>
#include <vector>
#include <list>
#include "common/checker.h"
namespace gert {
/**
 * base class, do not use it directly
 * @tparam T
 * @tparam ExtendSize
 */
template <typename T, size_t ExtendSize = 100>
class BaseRefObjectPool {
 public:
  ~BaseRefObjectPool() {
    auto free_set = GetFreeElements();
    for (auto &segment : holder_) {
      auto element = reinterpret_cast<T *>(&(segment.at(0)));
      for (size_t i = 0U; i < ExtendSize; ++i) {
        if (free_set.count(element + i) == 0) {
          element[i].~T();
        }
      }
    }
  }

  std::set<T *> GetFreeElements() {
    return {free_.begin(), free_.end()};
  }

  std::vector<T *> GetAllElements() {
    std::vector<T *> ret;
    ret.reserve(ExtendSize * holder_.size());
    for (auto &segment : holder_) {
      auto element = reinterpret_cast<T *>(&(segment.at(0)));
      for (size_t i = 0U; i < ExtendSize; ++i) {
        ret.emplace_back(element + i);
      }
    }
    return ret;
  }

 protected:
  std::list<std::array<uint8_t, sizeof(T) * ExtendSize>> holder_;
  std::vector<T *> free_;
};

template <typename T, size_t ExtendSize = 100>
class RefObjectPool : public BaseRefObjectPool<T, ExtendSize> {
 public:
  template <typename... Args>
  T *Acquire(Args &&...args) {
    if (this->free_.empty()) {
      Extend();
    }
    GE_ASSERT_TRUE(!this->free_.empty());
    auto element = *(this->free_.rbegin());
    this->free_.pop_back();
    new (element) T(args...);
    return element;
  }
  void Release(T *element) {
    element->~T();
    this->free_.push_back(element);
  }

 private:
  void Extend() {
    this->holder_.push_back({});
    auto new_place = reinterpret_cast<T *>(&(this->holder_.rbegin()->at(0)));
    for (size_t i = 0U; i < ExtendSize; ++i) {
      this->free_.push_back(new_place + i);
    }
  }
};


template <typename T, size_t ExtendSize = 100>
class NoDestructRefObjectPool : public BaseRefObjectPool<T, ExtendSize> {
 public:
  template <typename... Args>
  T *Acquire(Args &&...args) {
    if (this->free_.empty()) {
      Extend();
    }
    GE_ASSERT_TRUE(!this->free_.empty());
    auto element = *(this->free_.rbegin());
    this->free_.pop_back();
    GE_ASSERT_SUCCESS(element->ReInit(args...));
    return element;
  }
  void Release(T *element) {
    this->free_.push_back(element);
  }

 private:
  void Extend() {
    this->holder_.push_back({});
    auto new_place = reinterpret_cast<T *>(&(this->holder_.rbegin()->at(0)));
    for (size_t i = 0U; i < ExtendSize; ++i) {
      new (new_place + i) T();
      this->free_.push_back(new_place + i);
    }
  }
};
}  // namespace gert
#endif  // AIR_CXX_RUNTIME_V2_KERNEL_MEMORY_REF_OBJECT_POOL_H_
