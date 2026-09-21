/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_CXX_INC_COMMON_UTIL_TILING_UTILS_H_
#define METADEF_CXX_INC_COMMON_UTIL_TILING_UTILS_H_
#include <cstdint>
#include "graph/types.h"

namespace optiling {
union Fp32 {
  uint32_t u;
  ge::float32_t f;
};

inline uint16_t Float32ToFloat16(const ge::float32_t value) {
  constexpr Fp32 f32infty = {static_cast<uint32_t>(255) << static_cast<uint32_t>(23)};
  constexpr uint32_t sign_mask = 0x80000000U;
  constexpr uint32_t right_shift_16 = 16U;

  Fp32 temp;
  uint16_t out;
  temp.f = value;
  const uint32_t sign = temp.u & sign_mask;
  temp.u ^= sign;

  if (temp.u >= f32infty.u) {
    constexpr uint32_t round_max = 0x7FFFU;
    constexpr uint32_t dst_addr = 0x7C00U;
    out = (temp.u > f32infty.u) ? round_max : dst_addr;
  } else {
    constexpr uint32_t right_shift_13 = 13U;
    constexpr Fp32 f16infty = {static_cast<uint32_t>(31) << static_cast<uint32_t>(23)};
    constexpr Fp32 magic = {static_cast<uint32_t>(15) << static_cast<uint32_t>(23)};
    constexpr uint32_t round_mask = static_cast<uint32_t>(~0xFFFU);

    temp.u &= round_mask;
    temp.f *= magic.f;
    temp.u -= round_mask;
    if (temp.u > f16infty.u) {
      temp.u = f16infty.u;
    }
    out = uint16_t(temp.u >> right_shift_13);
  }

  out = uint16_t(out | (sign >> right_shift_16));
  return out;
}

inline uint16_t Float32ToBfloat16(const ge::float32_t value) {
  Fp32 temp;
  temp.f = value;
  constexpr uint32_t right_shift_16 = 16U;
  return uint16_t(temp.u >> right_shift_16);
}

template<typename T>
inline uint16_t OtherToFloat16(const T value) {
  const ge::float32_t value_f32 = static_cast<ge::float32_t>(value);
  return Float32ToFloat16(value_f32);
}

template<typename T>
inline uint16_t OtherToBfloat16(const T value) {
  const ge::float32_t value_f32 = static_cast<ge::float32_t>(value);
  return Float32ToBfloat16(value_f32);
}
}  // namespace optiling
#endif
