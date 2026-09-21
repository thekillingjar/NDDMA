/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_TABLE_DRIVEN_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_TABLE_DRIVEN_H_

#include <cstddef>
#include <utility>
#include <array>

namespace gert {
template <size_t DIM1, size_t DIM2, typename TE>
class TableDriven2 {
 public:
  explicit TableDriven2(const TE &default_value) noexcept : default_value_(default_value) {
    for (auto &row : elements_) {
      for (auto &element : row) {
        element = default_value;
      }
    }
  }
  TE Find(size_t src, size_t dst) const {
    if ((src >= DIM1) || (dst >= DIM2)) {
      return default_value_;
    }
    return elements_[src][dst];
  }
  const TE *FindPointer(size_t src, size_t dst) const {
    if ((src >= DIM1) || (dst >= DIM2)) {
      return nullptr;
    }
    return &elements_[src][dst];
  }
  TE *FindPointer(size_t src, size_t dst) {
    if ((src >= DIM1) || (dst >= DIM2)) {
      return nullptr;
    }
    return &elements_[src][dst];
  }
  template <typename... Arg>
  TableDriven2 &Add(size_t src, size_t dst, Arg &&...arg) {
    auto &element = elements_[src][dst];
    element = TE(std::forward<Arg>(arg)...);
    return *this;
  }

 private:
  TE default_value_;
  std::array<std::array<TE, DIM2>, DIM1> elements_;
};
}  // namespace gert
#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_TABLE_DRIVEN_H_
