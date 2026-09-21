/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <functional>
#include <algorithm>
#include <vector>
#include "framework/common/debug/ge_log.h"

namespace ge {
namespace {
static graphStatus BroadCastRankAndDim(
    const std::vector<int64_t> &x1_shape, const std::vector<int64_t> &x2_shape, const int64_t len_diff,
    const std::function<void(const std::vector<int64_t> &out_shape)> &set_out_shape) {
  std::vector<int64_t> y_shape;
  y_shape.reserve(x1_shape.size());
  for (size_t i = 0UL; i < static_cast<size_t>(len_diff); i++) {
    y_shape.push_back(x1_shape[i]);
  }
  for (size_t i = 0UL; i < x2_shape.size(); i++) {
    const size_t idx_diff = i + static_cast<size_t>(len_diff);
    if ((x1_shape[idx_diff] != x2_shape[i]) && (std::min(x1_shape[idx_diff], x2_shape[i]) != 1)) {
      GE_LOGE("operands could not be broadcast together");
      return GRAPH_FAILED;
    }
    y_shape.push_back(std::max(x1_shape[idx_diff], x2_shape[i]));
  }
  set_out_shape(y_shape);
  return GRAPH_SUCCESS;
}
}  // namespace

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
BroadCastInfer(const std::function<std::vector<int64_t>()> &get_in1_shape,
               const std::function<std::vector<int64_t>()> &get_in2_shape,
               const std::function<void(const std::vector<int64_t> &out_shape)> &set_out_shape) {
  const auto x1_shape = get_in1_shape();
  const auto x2_shape = get_in2_shape();

  if ((x1_shape.size() >= static_cast<size_t>(std::numeric_limits<int64_t>::max())) ||
      (x2_shape.size() >= static_cast<size_t>(std::numeric_limits<int64_t>::max()))) {
    return GRAPH_FAILED;
  }

  if (x1_shape.empty()) {
    set_out_shape(x2_shape);
    return GRAPH_SUCCESS;
  }
  if (x2_shape.empty()) {
    set_out_shape(x1_shape);
    return GRAPH_SUCCESS;
  }

  const int64_t len_diff = static_cast<int64_t>(x1_shape.size()) - static_cast<int64_t>(x2_shape.size());
  if (len_diff >= 0) {
    return BroadCastRankAndDim(x1_shape, x2_shape, len_diff, set_out_shape);
  } else {
    return BroadCastRankAndDim(x2_shape, x1_shape, std::abs(len_diff), set_out_shape);
  }
}
}  // namespace ge
