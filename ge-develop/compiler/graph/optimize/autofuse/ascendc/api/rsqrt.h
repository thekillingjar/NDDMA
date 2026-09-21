/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCENDC_API_RSQRT_H__
#define __ASCENDC_API_RSQRT_H__

template <typename T>
inline __aicore__ void RsqrtExtend(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<T> &src,
                                   const LocalTensor<float> &blk_tensor_with_value_1,
                                   const uint32_t size, LocalTensor<uint8_t> &tmp_buf) {
  uint32_t offset = 0;

  // Prepare sqrt in float type tensor
  LocalTensor<float> float_sqrt = tmp_buf[offset].template ReinterpretCast<float>();
  float_sqrt.SetSize((tmp_buf.GetSize() - offset) / sizeof(float));

  // Prepare binary repeat params for div operation
  BinaryRepeatParams div_repeat;
  div_repeat.blockNumber = size / (ONE_BLK_SIZE / sizeof(float));
  div_repeat.src0BlkStride = 0;
  div_repeat.src0RepStride = 0;

  int calc_size = 0;

  // Calc when size > max_repeat
  int max_repeat = float_sqrt.GetSize() / (ONE_REPEAT_BYTE_SIZE / sizeof(float));
  max_repeat = max_repeat > MAX_REPEAT_TIME ? MAX_REPEAT_TIME : max_repeat;
  const int max_repeat_calc_size = max_repeat * ONE_REPEAT_BYTE_SIZE / sizeof(float);
  for (; calc_size + max_repeat_calc_size < size; calc_size += max_repeat_calc_size) {
    if constexpr (std::is_same<T, float>::value) {
      Sqrt(float_sqrt[0], src[calc_size], max_repeat_calc_size);
      PipeBarrier<PIPE_V>();
      Div(dst[calc_size], blk_tensor_with_value_1[0], float_sqrt[0],
          ONE_REPEAT_BYTE_SIZE / sizeof(float), max_repeat, div_repeat);
    } else {
      Cast(float_sqrt[0], src[calc_size], RoundMode::CAST_NONE, max_repeat_calc_size);
      PipeBarrier<PIPE_V>();
      Sqrt(float_sqrt[0], float_sqrt[0], max_repeat_calc_size);
      PipeBarrier<PIPE_V>();
      Div(float_sqrt[0], blk_tensor_with_value_1[0], float_sqrt[0],
          ONE_REPEAT_BYTE_SIZE / sizeof(float), max_repeat, div_repeat);
      PipeBarrier<PIPE_V>();
      Cast(dst[calc_size], float_sqrt[0], RoundMode::CAST_NONE, max_repeat_calc_size);
    }
  }

  // Calc max_repeat > size > one_repeat
  const int one_repeat_calc_size = ONE_REPEAT_BYTE_SIZE / sizeof(float);
  if (calc_size + one_repeat_calc_size <= size) {
    int repeat = (size - calc_size) / one_repeat_calc_size;
    if constexpr (std::is_same<T, float>::value) {
      Sqrt(float_sqrt[0], src[calc_size], repeat * one_repeat_calc_size);
      PipeBarrier<PIPE_V>();
      Div(dst[calc_size], blk_tensor_with_value_1[0], float_sqrt[0], ONE_REPEAT_BYTE_SIZE / sizeof(float), repeat, div_repeat);
    } else {
      Cast(float_sqrt[0], src[calc_size], RoundMode::CAST_NONE, repeat * one_repeat_calc_size);
      PipeBarrier<PIPE_V>();
      Sqrt(float_sqrt[0], float_sqrt[0], repeat * one_repeat_calc_size);
      PipeBarrier<PIPE_V>();
      Div(float_sqrt[0], blk_tensor_with_value_1[0], float_sqrt[0],
          ONE_REPEAT_BYTE_SIZE / sizeof(float), repeat, div_repeat);
      PipeBarrier<PIPE_V>();
      Cast(dst[calc_size], float_sqrt[0], RoundMode::CAST_NONE, repeat * one_repeat_calc_size);
    }
    calc_size += repeat * one_repeat_calc_size;
  }

  // Calc when one_repeat > size
  if (calc_size < size) {
    div_repeat.blockNumber = (size - calc_size + ONE_BLK_SIZE / sizeof(float) - 1) / (ONE_BLK_SIZE / sizeof(float));
    if constexpr (std::is_same<T, float>::value) {
      Sqrt(float_sqrt[0], src[calc_size], size - calc_size);
      PipeBarrier<PIPE_V>();
      Div(dst[calc_size], blk_tensor_with_value_1[0], float_sqrt[0],
          div_repeat.blockNumber * (ONE_BLK_SIZE / sizeof(float)), 1, div_repeat);
    } else {
      Cast(float_sqrt[0], src[calc_size], RoundMode::CAST_NONE, size - calc_size);
      PipeBarrier<PIPE_V>();
      Sqrt(float_sqrt[0], float_sqrt[0], size - calc_size);
      PipeBarrier<PIPE_V>();
      Div(float_sqrt[0], blk_tensor_with_value_1[0], float_sqrt[0],
          div_repeat.blockNumber * (ONE_BLK_SIZE / sizeof(float)), 1,div_repeat);
      PipeBarrier<PIPE_V>();
      Cast(dst[calc_size], float_sqrt[0], RoundMode::CAST_NONE, size - calc_size);
    }
  }
  return;
}

#endif  // __ASCENDC_API_RSQRT_H__