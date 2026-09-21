/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_CXX_REPEATED_ITERATOR_H
#define METADEF_CXX_REPEATED_ITERATOR_H
#include <iterator>
#include <cstddef>

namespace ge {
template<typename T>
class RepeatedIterator {
 public:
  using iterator_category = std::forward_iterator_tag;
  using difference_type = std::ptrdiff_t;
  using value_type = T;
  using pointer = T *;
  using reference = T &;
  using size_type = size_t;

  RepeatedIterator(size_type index, reference value) : index_(index), value_(value) {}

  reference operator*() const {
    return value_;
  }

  pointer operator->() const {
    return &value_;
  }

  RepeatedIterator &operator++() {
    ++index_;
    return *this;
  }
  RepeatedIterator operator++(int) {
    RepeatedIterator ret = *this;
    ++*this;
    return ret;
  }

  friend bool operator==(const RepeatedIterator &lhs, const RepeatedIterator &rhs) {
      return (lhs.index_ == rhs.index_) && (&lhs.value_ == &rhs.value_);
  }
  friend bool operator!=(const RepeatedIterator &lhs, const RepeatedIterator &rhs) {
    return !(lhs == rhs);
  };

 private:
  size_type index_;
  reference value_;
};
}  // namespace ge
#endif  // METADEF_CXX_REPEATED_ITERATOR_H
