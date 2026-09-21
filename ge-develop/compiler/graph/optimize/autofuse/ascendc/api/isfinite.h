/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCENDC_API_ISFINITE_H__
#define __ASCENDC_API_ISFINITE_H__

const int16_t NEG_INF_NUM_HALF = -0x7c00;
const int32_t NEG_INF_NUM_FLOAT = -0x7f800000;
constexpr int16_t SIGN_MASK_HALF = 0x7fff;
constexpr int32_t SIGN_MASK_FLOAT = 0x7fffffff;

template <typename T>
struct InferCalcType {
  using Type = T;
};

template <>
struct InferCalcType<half> {
  using Type = int16_t;
};

template <>
struct InferCalcType<float> {
  using Type = int32_t;
};

template<typename T1, typename T2>
inline __aicore__ void IsFiniteCompute(
  const AscendC::LocalTensor<uint8_t> &dst, const AscendC::LocalTensor<int16_t> &src,
  const LocalTensor<T2> &calc_buf, const AscendC::LocalTensor<int16_t> &sign_mask,
  const BinaryRepeatParams &rpt_params, const int32_t calc_cnt, const int32_t mask, const int32_t rpt_times)
{
  if constexpr (SupportType<T1, int8_t, uint8_t, uint16_t, int16_t, bool, uint32_t, int32_t, uint64_t, int64_t>()) {
    AscendC::Duplicate(dst.ReinterpretCast<uint16_t>(), (uint16_t)257, calc_cnt / 2);  
  } else {
    AscendC::LocalTensor<int16_t> calc_buf_int16 = calc_buf.template ReinterpretCast<int16_t>();
    AscendC::And(calc_buf_int16, src, sign_mask, mask, rpt_times, rpt_params);
    constexpr T2 NEG_INF_NUM = AscendC::IsSameType<T2, int32_t>::value ? NEG_INF_NUM_FLOAT : NEG_INF_NUM_HALF;
    AscendC::Adds(calc_buf, calc_buf, NEG_INF_NUM, calc_cnt);
    Mins(calc_buf, calc_buf, static_cast<T2>(0), calc_cnt);
    LocalTensor<T1> abs_buf = calc_buf.template ReinterpretCast<T1>();
    Abs(abs_buf, abs_buf, calc_cnt);
    Mins(calc_buf, calc_buf, static_cast<T2>(1), calc_cnt);
    LocalTensor<half> half_tmp = calc_buf.template ReinterpretCast<half>();
    if constexpr (AscendC::IsSameType<T2, int32_t>::value) {
      AscendC::SetDeqScale(half(1.0));
    }
    AscendC::PipeBarrier<PIPE_V>();
    Cast(half_tmp, calc_buf, RoundMode::CAST_NONE, calc_cnt);
    AscendC::PipeBarrier<PIPE_V>();
    Cast(dst, half_tmp, RoundMode::CAST_NONE, calc_cnt);
  }
}

template<typename T>
inline __aicore__ void IsFiniteExtend(const AscendC::LocalTensor<uint8_t> &dst, const AscendC::LocalTensor<T> &src,
  const uint32_t size, LocalTensor<uint8_t> &tmp_buf) {
  // Init local tensor from tmp_buf
  using T2 = typename InferCalcType<T>::Type;
  LocalTensor<T2> sign_mask = tmp_buf.ReinterpretCast<T2>();
  sign_mask.SetSize(ONE_BLK_SIZE / sizeof(T2));
  constexpr T2 SIGN_MASK = AscendC::IsSameType<T2, int32_t>::value ? SIGN_MASK_FLOAT : SIGN_MASK_HALF;
  Duplicate(sign_mask, SIGN_MASK, ONE_BLK_SIZE / sizeof(T2));
  LocalTensor<int16_t> sign_mask_int16 = sign_mask.template ReinterpretCast<int16_t>();
  LocalTensor<T2> calc_buf = tmp_buf[ONE_BLK_SIZE].ReinterpretCast<T2>();
  calc_buf.SetSize((tmp_buf.GetSize() - ONE_BLK_SIZE) / sizeof(T2));
  // Prepare binary repeat params for and operation
  BinaryRepeatParams and_repeat (1, 1, 0, 8, 8, 0);
  int32_t calc_size = 0;

  // Calc when size > max_repeat
  int32_t max_repeat = calc_buf.GetSize() / (ONE_REPEAT_BYTE_SIZE / sizeof(T));
  max_repeat = max_repeat > MAX_REPEAT_TIME ? MAX_REPEAT_TIME : max_repeat;
  max_repeat = max_repeat == 0 ? 1 : max_repeat;
  const int32_t max_repeat_calc_size = max_repeat * ONE_REPEAT_BYTE_SIZE / sizeof(T);
  constexpr uint32_t RATIO = sizeof(T2) / sizeof(int16_t);
  LocalTensor<int16_t> src_int16 = src.template ReinterpretCast<int16_t>();
  for (; calc_size + max_repeat_calc_size < size; calc_size += max_repeat_calc_size) {
    IsFiniteCompute<T, T2>(dst[calc_size], src_int16[calc_size * RATIO], calc_buf, sign_mask_int16,
      and_repeat, max_repeat_calc_size, ONE_REPEAT_BYTE_SIZE / sizeof(int16_t), max_repeat);
  }

  constexpr int one_repeat_calc_size = ONE_REPEAT_BYTE_SIZE / sizeof(T);
  // Calc max_repeat > size > one_repeat
  if (calc_size + one_repeat_calc_size <= size) {
    int repeat = (size - calc_size) / one_repeat_calc_size;
    IsFiniteCompute<T, T2>(dst[calc_size], src_int16[calc_size * RATIO], calc_buf, sign_mask_int16, and_repeat,
      repeat * one_repeat_calc_size, ONE_REPEAT_BYTE_SIZE / sizeof(int16_t), repeat);
    calc_size += repeat * one_repeat_calc_size;
  }

  // Calc when one_repeat > size
  if (calc_size < size) {
    IsFiniteCompute<T, T2>(dst[calc_size], src_int16[calc_size * RATIO], calc_buf, sign_mask_int16, and_repeat,
      size - calc_size, (size - calc_size) * RATIO, 1);
  }
}
#endif // __ASCENDC_API_ISFINITE_H__
