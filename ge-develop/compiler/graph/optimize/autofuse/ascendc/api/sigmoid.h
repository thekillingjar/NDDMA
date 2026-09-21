/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCENDC_API_SIGMOID_H__
#define __ASCENDC_API_SIGMOID_H__

template <typename T>
__aicore__ inline void CalcDenominator(const LocalTensor<T> &dst, const LocalTensor<T> &src, const uint32_t cal_cnt) {
  struct UnaryRepeatParams repeatParams;
  AscendC::SetMaskCount();
  AscendC::SetVectorMask<T, MaskMode::COUNTER>(cal_cnt);
  AscendC::PipeBarrier<PIPE_V>();
  AscendC::Muls<T, false>(dst, src, static_cast<T>(-1.0), MASK_PLACEHOLDER, 1, repeatParams);
  AscendC::PipeBarrier<PIPE_V>();
  AscendC::Exp<T, false>(dst, dst, MASK_PLACEHOLDER, 1, repeatParams);
  AscendC::PipeBarrier<PIPE_V>();
  AscendC::Adds<T, false>(dst, dst, static_cast<T>(1), MASK_PLACEHOLDER, 1, repeatParams);
  AscendC::SetMaskNorm();
  AscendC::ResetMask();
  AscendC::PipeBarrier<PIPE_V>();
}

template <typename T>
inline __aicore__ void SigmoidExtend(const LocalTensor<T> &dst, const LocalTensor<T> &src, const uint32_t cal_cnt,
                                     LocalTensor<uint8_t> &tmp_buf) {
  uint32_t one_blk_num = KernelUtils::BlkSize<T>();
  LocalTensor<T> ones = tmp_buf.template ReinterpretCast<T>();
  Duplicate(ones, (T)1.0, one_blk_num);

  // Prepare binary repeat params for div operation
  BinaryRepeatParams div_repeat;
  div_repeat.src0BlkStride = 0;
  div_repeat.src0RepStride = 0;

  uint32_t calc_size = 0;

  // Calc when size > max_repeat
  const uint32_t one_repeat_calc_size = ONE_REPEAT_BYTE_SIZE / sizeof(T);
  uint32_t max_repeat = cal_cnt / one_repeat_calc_size;
  max_repeat = max_repeat > MAX_REPEAT_TIME ? MAX_REPEAT_TIME : (max_repeat == 0 ? 1 : max_repeat);
  const uint32_t max_repeat_calc_size = max_repeat * ONE_REPEAT_BYTE_SIZE / sizeof(T);
  for (; calc_size + max_repeat_calc_size < cal_cnt; calc_size += max_repeat_calc_size) {
    CalcDenominator(dst[calc_size], src[calc_size], max_repeat_calc_size);
    AscendC::Div(dst[calc_size], ones, dst[calc_size], ONE_REPEAT_BYTE_SIZE / sizeof(T), max_repeat, div_repeat);
  }

  // Calc max_repeat > size > one_repeat
  if (calc_size + one_repeat_calc_size <= cal_cnt) {
    int repeat = (cal_cnt - calc_size) / one_repeat_calc_size;
    CalcDenominator(dst[calc_size], src[calc_size], repeat * one_repeat_calc_size);
    AscendC::Div(dst[calc_size], ones, dst[calc_size], ONE_REPEAT_BYTE_SIZE / sizeof(T), repeat, div_repeat);
    calc_size += repeat * one_repeat_calc_size;
  }

  // Calc when one_repeat > size
  if (calc_size < cal_cnt) {
    CalcDenominator(dst[calc_size], src[calc_size], cal_cnt - calc_size);
    div_repeat.blockNumber = (cal_cnt - calc_size + one_blk_num - 1) / one_blk_num;
    AscendC::Div(dst[calc_size], ones, dst[calc_size], div_repeat.blockNumber * one_blk_num, 1, div_repeat);
  }
}
#endif  // __ASCENDC_API_SIGMOID_H__
