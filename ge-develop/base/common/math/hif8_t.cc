/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/math/hif8_t.h"

#include <cmath>
#include <array>
#include <limits>
#include "securec.h"
#include "common/checker.h"

namespace {
constexpr std::array<uint8_t, 24> kHiF8ExpToMantissaWidth = {{
  3, 3, 3, 3,                   // [0, 3]
  2, 2, 2, 2,                   // [4, 7]
  1, 1, 1, 1, 1, 1, 1, 1,       // [8, 15]
  0, 0, 0, 0, 0, 0, 0,          // [16, 22]
  static_cast<uint8_t>(-1),     // [23]
}};
constexpr std::array<uint8_t, 16> kHiF8ExpToDot = {{
  0b0001,                                                               // [0]
  0b0010,                                                               // [1]
  0b0100, 0b0100,                                                       // [2, 3]
  0b1000, 0b1000, 0b1000, 0b1000,                                       // [4, 7]
  0b1100, 0b1100, 0b1100, 0b1100, 0b1100, 0b1100, 0b1100, 0b1100,       // [8, 15]
}};

constexpr int8_t kHiF8DmlExpMax = -16;
constexpr int8_t kHiF8DmlExpBias = 23;
constexpr uint8_t kHiF8DotOffset = 3;

constexpr int8_t kHiF8NmlExpMax = 15;
constexpr int8_t kHiF8DmlExpMin = -22;

constexpr uint8_t kHiF8SignMask = 0b10000000;

constexpr uint8_t kIeeeFp32ExpBits = 8;
constexpr uint8_t kIeeeFp32ManBits = 32 - 1 - kIeeeFp32ExpBits;
constexpr uint8_t kIeeeFp16ExpBits = 5;
constexpr uint8_t kIeeeFp16ManBits = 16 - 1 - kIeeeFp16ExpBits;

uint32_t Extract32(uint32_t u32, uint8_t width, uint8_t offset) {
  return (u32 >> offset) & ((1U << width) - 1U);
}
uint32_t Deposit32(uint32_t u32, uint8_t width, uint8_t offset, uint32_t value) {
  const uint32_t mask = (1U << width) - 1U;
  return (u32 & ~(mask << offset)) | ((value & mask) << offset);
}

uint8_t EncodeHiF8(int8_t exp, uint8_t mantissa) {
  if (exp <= kHiF8DmlExpMax) {
    return static_cast<uint8_t>(exp + kHiF8DmlExpBias);
  }
  const uint8_t mag = static_cast<uint8_t>(exp < 0 ? -exp : exp);
  auto encode_em = [](uint8_t m, uint8_t se_bit) -> uint8_t {
    if (m == 0) {
      return 0;
    }
    // Replace the most-significant bit in `m` to `se_bit`.
    return Deposit32(m, 1, 7 - (__builtin_clz(m) - 24), se_bit);
  };
  GE_ASSERT(mag < sizeof(kHiF8ExpToDot));
  return static_cast<uint8_t>(kHiF8ExpToDot[mag] << kHiF8DotOffset) |
         static_cast<uint8_t>(encode_em(mag, exp < 0) << kHiF8ExpToMantissaWidth[mag]) |
         mantissa;
}

enum class IeeeType : int8_t {
  kZero,
  kNaN,
  kInf,
  kOrdinary,
};

template <uint8_t exp_bits, uint8_t mantissa_bits>
IeeeType UnpackIeeeFp(uint32_t bits, uint8_t *sign_bit, int8_t *exp, uint32_t *mantissa) {
  constexpr uint32_t ieee_sign_bit = 1U << (exp_bits + mantissa_bits);
  constexpr int8_t bias = static_cast<int8_t>((1U << (exp_bits - 1U)) - 1U);

  if ((bits & ~ieee_sign_bit) == 0) {
    return IeeeType::kZero;
  }

  *sign_bit = (bits & ieee_sign_bit) != 0 ? kHiF8SignMask : 0U;
  *exp = static_cast<int8_t>(Extract32(bits, exp_bits, mantissa_bits)) - bias;
  *mantissa = Extract32(bits, mantissa_bits, 0);

  // Handle NaN or Inf.
  if (static_cast<uint8_t>(*exp) == (1U << (exp_bits - 1U))) {
    return *mantissa != 0 ? IeeeType::kNaN : IeeeType::kInf;
  }
  // Handle IEEE subnormal values.
  if (*exp == -bias && *mantissa != 0) {
    const uint8_t shift = mantissa_bits - (32 - __builtin_clz(*mantissa)) + 1;
    *exp = -bias + 1 - shift;
    *mantissa = Extract32(*mantissa << shift, mantissa_bits, 0);
  }
  return IeeeType::kOrdinary;
}

template <uint8_t mantissa_bits>
bool RoundMantissa(uint32_t *mantissa, uint32_t probe_bit) {
  const uint32_t tmp = *mantissa + probe_bit;
  *mantissa = Extract32(tmp, mantissa_bits, 0);
  return (tmp & (1U << mantissa_bits)) != 0;
}

template <uint8_t exp_bits, uint8_t mantissa_bits>
uint8_t HiF8FromIeeeBits(uint32_t bits) {
  uint8_t sign_bit = 0;
  int8_t exp = 0;
  uint32_t mantissa = 0;
  const auto type = UnpackIeeeFp<exp_bits, mantissa_bits>(bits, &sign_bit, &exp, &mantissa);
  if (type == IeeeType::kZero || (type == IeeeType::kOrdinary && exp < kHiF8DmlExpMin - 1)) {
    return 0;
  } else if (type == IeeeType::kNaN) {
    return 0b10000000;
  } else if (type == IeeeType::kInf || (type == IeeeType::kOrdinary && exp > kHiF8NmlExpMax)) {
    return sign_bit | 0b01101111U;
  }

  // TA rounding.
  const uint8_t bits_to_discard = mantissa_bits - kHiF8ExpToMantissaWidth[static_cast<ssize_t>(std::abs(exp))];
  if (RoundMantissa<mantissa_bits>(&mantissa, 1U << (bits_to_discard - 1U))) {
    exp += 1;
    // Handle overflow after TA rounding.
    if (exp > kHiF8NmlExpMax) {
      return sign_bit | 0b01101111U;
    }
  }
  return sign_bit | EncodeHiF8(exp, mantissa >> bits_to_discard);
}

template <typename To, typename From>
auto BitCast(const From &src) -> To {
  static_assert(sizeof(To) == sizeof(From), "");
  To dst = 0;
  (void)memcpy_s(&dst, sizeof(To), &src, sizeof(From));
  return dst;
}
} // namespace anonymous

namespace ge {
HiF8 HiF8::FromRawBits(uint8_t bits) {
  HiF8 hif8;
  hif8.u8_ = bits;
  return hif8;
}

HiF8::HiF8(float f32) : u8_(BitsFromFp32(BitCast<uint32_t>(f32))) {}
HiF8::HiF8(fp16_t f16) : u8_(BitsFromFp16(BitCast<uint16_t>(f16))) {}

uint8_t HiF8::BitsFromFp32(uint32_t f32) {
  return HiF8FromIeeeBits<kIeeeFp32ExpBits, kIeeeFp32ManBits>(f32);
}
uint8_t HiF8::BitsFromFp16(uint16_t f16) {
  return HiF8FromIeeeBits<kIeeeFp16ExpBits, kIeeeFp16ManBits>(f16);
}

bool HiF8::IsNaN() const {
  return u8_ == kHiF8SignMask;
}

bool HiF8::IsInf() const {
  return (u8_ & static_cast<uint8_t>(~kHiF8SignMask)) == 0b01101111U;
}

HiF8::operator float() const {
  constexpr uint8_t kHiF8DmlMask = 0b01111000;
  constexpr uint8_t kHiF8DmlFlag = static_cast<uint8_t>(-1);
  constexpr float kExpBase = 2.0F;
  if (IsNaN()) {
    return NAN;
  } else if (IsInf()) {
    return (u8_ & kHiF8SignMask) != 0 ? -std::numeric_limits<float>::infinity()
                                      : std::numeric_limits<float>::infinity();
  }
  static const std::array<std::pair<uint8_t, uint8_t>, 16> kDotTable = {{
    { kHiF8DmlFlag, 4 },                        // 0b0000
    { 0, 4 },                                   // 0b0001
    { 1, 3 }, { 1, 3 },                         // 0b001.
    { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 },     // 0b01..
    { 3, 2 }, { 3, 2 }, { 3, 2 }, { 3, 2 },     // 0b10..
    { 4, 2 }, { 4, 2 }, { 4, 2 }, { 4, 2 },     // 0b11..
  }};
  const float sign = (u8_ & kHiF8SignMask) != 0 ? -1.0F : 1.0F;
  const auto pair = kDotTable[(u8_ & kHiF8DmlMask) >> kHiF8DotOffset];
  const uint8_t dot = pair.first;
  const uint8_t width = pair.second;
  if (dot == kHiF8DmlFlag) {
    const int8_t dml_exp = static_cast<int8_t>(u8_ & static_cast<uint8_t>(~(kHiF8SignMask | kHiF8DmlMask)));
    if (dml_exp == 0) {
      return 0.0F;
    }
    return sign * powf(kExpBase, static_cast<float>(dml_exp - kHiF8DmlExpBias));
  }
  const uint8_t mantissa_width = 7 - (dot + width);
  float mantissa_value = 0.0F;
  for (uint8_t n = 0; n < mantissa_width; n ++) {
    if ((u8_ & static_cast<uint8_t>(1U << n)) == 0) {
      continue;
    }
    mantissa_value += powf(kExpBase, -static_cast<float>(mantissa_width - n));
  }

  auto ei = [mantissa_width](uint8_t d, uint8_t raw) -> int8_t {
    if (d == 0) {
      return 0;
    }
    const uint8_t em = Extract32(raw, d, mantissa_width);
    const uint8_t se = em & static_cast<uint8_t>(1U << (d - 1U));
    const uint8_t mag = Deposit32(em, 1, d - 1U, 1);
    return (se != 0 ? -1 : 1) * static_cast<int8_t>(mag);
  };
  return sign * powf(kExpBase, static_cast<float>(ei(dot, u8_))) * (1.0F + mantissa_value);
}

HiF8::operator fp16_t() const {
  if (IsNaN()) {
    // Conversion from float to fp16 are implemented in saturation mode,
    // which transforms even NaN into kFp16Max, breaking many tests.
    return fp16_t(Fp16Constructor(0, kFp16MaxExp, 1));
  }
  fp16_t tmp;
  // Conversion from fp32 to fp16 is implemented in fp16_t::operator=(const float&).
  tmp = static_cast<float>(*this);
  return tmp;
}
} // namespace ge
