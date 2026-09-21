/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UTILS_H_
#define UTILS_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <string>
#include <tuple>
#include <vector>

namespace wrapper {
inline bool CheckInt64MulOverflow(int64_t a, int64_t b) {
  if ((a == 0) || (b == 0)) {
    return false;
  }
  const auto a_abs = std::abs(a);
  const auto b_abs = std::abs(b);
  if (INT64_MAX / a_abs < b_abs) {
    return true;
  }
  return false;
}

inline std::string ComputeStrides(ssize_t item_size, const std::vector<int64_t> &dims, std::vector<ssize_t> &strides) {
  if (dims.empty()) {
    return "";
  }
  strides.emplace_back(item_size);
  int64_t stride = static_cast<int64_t>(item_size);
  for (auto it = dims.crbegin(); it != (dims.crend() - 1); it++) {
    if (CheckInt64MulOverflow(stride, *it)) {
      return std::to_string(stride) + " mul " + std::to_string(*it) + "is overflow.";
    }
    stride *= *it;
    strides.emplace_back(static_cast<ssize_t>(stride));
  }
  std::reverse(strides.begin(), strides.end());
  return "";
}
}  // namespace wrapper
#endif  // UTILS_H_