/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCENDC_API_RECIPROCAL_H__
#define __ASCENDC_API_RECIPROCAL_H__

template <typename T>
inline __aicore__ void ReciprocalExtend(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<T> &src,
                                        const uint32_t size, LocalTensor<uint8_t> &tmp_buf) {
  // Initialize ones tensor with 1.0
  LocalTensor<T> ones = tmp_buf.template ReinterpretCast<T>();
  uint32_t one_blk_num = KernelUtils::BlkSize<T>();
  Duplicate(ones, (T)1.0, one_blk_num);

  // Prepare binary repeat params for div operation
  BinaryRepeatParams div_repeat;
  div_repeat.blockNumber = size / one_blk_num;
  div_repeat.src0BlkStride = 0;
  div_repeat.src0RepStride = 0;

  uint32_t calc_size = 0;

  // Calc when size > max_repeat
  const int one_repeat_calc_size = ONE_REPEAT_BYTE_SIZE / sizeof(T);
  int32_t max_repeat = size / one_repeat_calc_size;
  max_repeat = max_repeat > MAX_REPEAT_TIME ? MAX_REPEAT_TIME : (max_repeat == 0 ? 1 : max_repeat);
  const int max_repeat_calc_size = max_repeat * ONE_REPEAT_BYTE_SIZE / sizeof(T);
  for (; calc_size + max_repeat_calc_size < size; calc_size += max_repeat_calc_size) {
    Div(dst[calc_size], ones, src[calc_size], ONE_REPEAT_BYTE_SIZE / sizeof(T), max_repeat, div_repeat);
  }

  // Calc max_repeat > size > one_repeat
  if (calc_size + one_repeat_calc_size <= size) {
    int repeat = (size - calc_size) / one_repeat_calc_size;
    Div(dst[calc_size], ones, src[calc_size], ONE_REPEAT_BYTE_SIZE / sizeof(T), repeat, div_repeat);
    calc_size += repeat * one_repeat_calc_size;
  }

  // Calc when one_repeat > size
  if (calc_size < size) {
    div_repeat.blockNumber = (size - calc_size + one_blk_num - 1) / one_blk_num;
    Div(dst[calc_size], ones, src[calc_size], div_repeat.blockNumber * one_blk_num, 1, div_repeat);
  }
}

#endif  // __ASCENDC_API_RECIPROCAL_H__