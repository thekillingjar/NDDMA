/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMMON_UTILS_TRANSFORMER_INC_AXIS_UTIL_H_
#define COMMON_UTILS_TRANSFORMER_INC_AXIS_UTIL_H_

#include <memory.h>
#include <array>
#include "graph/types.h"
#include "graph/utils/math_util.h"
#include "exe_graph/runtime/shape.h"

namespace transformer {
#define CHECK(cond, log_func, return_expr) \
  do {                                     \
    if (cond) {                            \
      log_func;                            \
      return_expr;                         \
    }                                      \
  } while (0)

#define INT64_ZEROCHECK(a)                 \
  if (a == 0) {                            \
    return false;                          \
  }

#define MUL_OVERFLOW(x, y, z)              \
  if (ge::MulOverflow((x), (y), (z))) {    \
    return false;                          \
  }                                        \

enum AxisValueType {
  AXIS_N = 0,
  AXIS_C = 1,
  AXIS_H = 2,
  AXIS_W = 3,
  AXIS_C1 = 4,
  AXIS_C0 = 5,
  AXIS_Co = 6,
  AXIS_D = 7,
  AXIS_G = 8,
  AXIS_M0 = 9,
  AXIS_INPUT_SIZE = 10,
  AXIS_HIDDEN_SIZE = 11,
  AXIS_STATE_SIZE = 12,
  AXIS_BOTTOM = 13
};

using AxisValue = std::array<int64_t, static_cast<size_t>(AXIS_BOTTOM)>;

inline int64_t DivisionCeiling(int64_t dividend, int64_t divisor) {
  if (divisor == 0) {
    return 0;
  } else if (dividend < 0) {
    return -1;
  } else {
    return (dividend + divisor - 1) / divisor;
  }
}

class AxisUtil {
 public:
  AxisUtil() {};
  ~AxisUtil() {};
  static bool GetAxisValueByOriginFormat(const ge::Format &format, const gert::Shape &shape, AxisValue &axis_value);
  static int32_t GetAxisIndexByFormat(const ge::Format &format, const std::string &axis);
  static int32_t GetAxisIndexByFormat(const ge::Format &format, const std::string &axis,
                                      const std::map<std::string, int32_t> &valid_axis_map);
  static std::vector<std::string> GetAxisVecByFormat(const ge::Format &format);
  static std::vector<std::string> GetReshapeTypeAxisVec(const ge::Format &format, const int64_t &reshape_type_mask);
  static std::map<std::string, int32_t> GetReshapeTypeAxisMap(const ge::Format &format,
                                                              const int64_t &reshape_type_mask);
  static std::vector<std::string> GetSplitOrConcatAxisByFormat(const ge::Format &format, const std::string &axis);
 private:
  static bool GetAxisValueByNCHW(const gert::Shape &shape, AxisValue &axis_value);

  static bool GetAxisValueByNHWC(const gert::Shape &shape, AxisValue &axis_value);

  static bool GetAxisValueByHWCN(const gert::Shape &shape, AxisValue &axis_value);

  static bool GetAxisValueByND(const gert::Shape &shape, AxisValue &axis_value);

  static bool GetAxisValueByNDHWC(const gert::Shape &shape, AxisValue &axis_value);

  static bool GetAxisValueByNCDHW(const gert::Shape &shape, AxisValue &axis_value);

  static bool GetAxisValueByDHWCN(const gert::Shape &shape, AxisValue &axis_value);

  static bool GetAxisValueByDHWNC(const gert::Shape &shape, AxisValue &axis_value);

  static bool GetAxisValueByNC1HWC0(const gert::Shape &shape, AxisValue &axis_value);

  static bool GetAxisValueByC1HWNCoC0(const gert::Shape &shape, AxisValue &axis_value);
};
} // namespace transformer
#endif // COMMON_UTILS_TRANSFORMER_INC_AXIS_UTIL_H_
 