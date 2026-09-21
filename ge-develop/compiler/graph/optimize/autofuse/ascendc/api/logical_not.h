/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCENDC_API_LOGICAL_NOT_H__
#define __ASCENDC_API_LOGICAL_NOT_H__

inline __aicore__ void do_logical_not(LocalTensor<half> ones, LocalTensor<half> half_not, BinaryRepeatParams sub_repeat,
                                      int repeat, const int calc_size) {
  Abs(half_not[0], half_not[0], calc_size);
  AscendC::Min(half_not[0], ones[0], half_not[0], ONE_REPEAT_BYTE_SIZE / sizeof(half), repeat, sub_repeat);
  Sub(half_not[0], ones[0], half_not[0], ONE_REPEAT_BYTE_SIZE / sizeof(half), repeat, sub_repeat);
  Abs(half_not[0], half_not[0], calc_size);
}

template <typename T>
inline __aicore__ void SpecialTypeLogicalNot(const LocalTensor<T> &dst, const LocalTensor<T> &src,
                                             const LocalTensor<half> &blk_tensor_with_value_1, const uint32_t size,
                                             LocalTensor<uint8_t> &tmp_buf) {
  /*
   * 原数据类型是int64，需要将数据先转成float，然后再转成half计算，因此tmp_buf要分成3份：
   * 1）存储数字1
   * 2）存储src转成float后的结果
   * 3）存储2转成half后的结果
   * 由于float的大小是half的两倍，相同数量下，2分配的空间应该是3的两倍，因此2占tmp_buf扣除1后的1/3，3占tmp_buf扣除1后的2/3
   */
  uint32_t offset = 0;
  constexpr uint32_t tmp_buf_splits = 3;
  constexpr uint32_t cast_dst_align = 32;  // cast算子要求目标地址32字节对齐
  const uint32_t cast_buf_size =
      (tmp_buf.GetSize() - offset) / tmp_buf_splits / cast_dst_align * cast_dst_align / sizeof(half);

  // Prepare logicalNot in float type tensor
  LocalTensor<float> float_not = tmp_buf[offset].template ReinterpretCast<float>();
  float_not.SetSize(cast_buf_size);
  offset += cast_buf_size * sizeof(float);

  // Prepare logicalNot in half type tensor
  LocalTensor<half> half_not = tmp_buf[offset].template ReinterpretCast<half>();
  half_not.SetSize(cast_buf_size);

  // Prepare binary repeat params for min/sub operation
  BinaryRepeatParams sub_repeat;
  sub_repeat.blockNumber = size / (ONE_BLK_SIZE / sizeof(half));
  sub_repeat.src0BlkStride = 0;
  sub_repeat.src0RepStride = 0;

  int calc_size = 0;
  // Calc when size > max_repeat
  int max_repeat = half_not.GetSize() / (ONE_REPEAT_BYTE_SIZE / sizeof(half));
  max_repeat = max_repeat > MAX_REPEAT_TIME ? MAX_REPEAT_TIME : max_repeat;
  const int max_repeat_calc_size = max_repeat * ONE_REPEAT_BYTE_SIZE / sizeof(half);
  for (; calc_size + max_repeat_calc_size < size; calc_size += max_repeat_calc_size) {
    Cast(float_not[0], src[calc_size], RoundMode::CAST_RINT, max_repeat_calc_size);
    Cast(half_not[0], float_not[0], RoundMode::CAST_NONE, max_repeat_calc_size);
    do_logical_not(blk_tensor_with_value_1, half_not, sub_repeat, max_repeat, max_repeat_calc_size);
    Cast(float_not[0], half_not[0], RoundMode::CAST_NONE, max_repeat_calc_size);
    Cast(dst[calc_size], float_not[0], RoundMode::CAST_RINT, max_repeat_calc_size);
  }

  // Calc max_repeat > size > one_repeat
  const int one_repeat_calc_size = ONE_REPEAT_BYTE_SIZE / sizeof(half);
  if (calc_size + one_repeat_calc_size <= size) {
    int repeat = (size - calc_size) / one_repeat_calc_size;
    Cast(float_not[0], src[calc_size], RoundMode::CAST_RINT, repeat * one_repeat_calc_size);
    Cast(half_not[0], float_not[0], RoundMode::CAST_NONE, repeat * one_repeat_calc_size);
    do_logical_not(blk_tensor_with_value_1, half_not, sub_repeat, repeat, repeat * one_repeat_calc_size);
    Cast(float_not[0], half_not[0], RoundMode::CAST_NONE, repeat * one_repeat_calc_size);
    Cast(dst[calc_size], float_not[0], RoundMode::CAST_RINT, repeat * one_repeat_calc_size);
    calc_size += repeat * one_repeat_calc_size;
  }

  // Calc when one_repeat > size
  if (calc_size < size) {
    sub_repeat.blockNumber = (size - calc_size + ONE_BLK_SIZE / sizeof(half) - 1) / (ONE_BLK_SIZE / sizeof(half));
    Cast(float_not[0], src[calc_size], RoundMode::CAST_RINT, size - calc_size);
    Cast(half_not[0], float_not[0], RoundMode::CAST_NONE, size - calc_size);
    do_logical_not(blk_tensor_with_value_1, half_not, sub_repeat, 1, size - calc_size);
    Cast(float_not[0], half_not[0], RoundMode::CAST_NONE, size - calc_size);
    Cast(dst[calc_size], float_not[0], RoundMode::CAST_RINT, size - calc_size);
  }
}

template <typename T>
inline __aicore__ void LogicalNot(const LocalTensor<T> &dst, const LocalTensor<T> &src, const LocalTensor<half> &blk_tensor_with_value_1,
                                  const uint32_t size, LocalTensor<uint8_t> &tmp_buf) {
  if constexpr (std::is_same_v<T, int64_t>) {
    SpecialTypeLogicalNot<T>(dst, src, blk_tensor_with_value_1, size, tmp_buf);
    return;
  }
  uint32_t offset = 0;

  // Prepare logicalNot in half type tensor
  LocalTensor<half> half_not = tmp_buf[offset].template ReinterpretCast<half>();
  half_not.SetSize((tmp_buf.GetSize() - offset) / sizeof(half));

  // Prepare binary repeat params for min/sub operation
  BinaryRepeatParams sub_repeat;
  sub_repeat.blockNumber = size / (ONE_BLK_SIZE / sizeof(half));
  sub_repeat.src0BlkStride = 0;
  sub_repeat.src0RepStride = 0;

  int calc_size = 0;

  // Calc when size > max_repeat
  int max_repeat = half_not.GetSize() / (ONE_REPEAT_BYTE_SIZE / sizeof(half));
  max_repeat = max_repeat > MAX_REPEAT_TIME ? MAX_REPEAT_TIME : max_repeat;
  const int max_repeat_calc_size = max_repeat * ONE_REPEAT_BYTE_SIZE / sizeof(half);
  auto round_mode =
      std::is_same_v<T, int16_t> || std::is_same_v<T, int32_t> ? RoundMode::CAST_RINT : RoundMode::CAST_NONE;
  for (; calc_size + max_repeat_calc_size < size; calc_size += max_repeat_calc_size) {
    if constexpr (std::is_same_v<T, half>) {
      Abs(half_not[0], src[calc_size], max_repeat_calc_size);
      AscendC::Min(half_not[0], blk_tensor_with_value_1[0], half_not[0], ONE_REPEAT_BYTE_SIZE / sizeof(half), max_repeat, sub_repeat);
      Sub(half_not[0], blk_tensor_with_value_1[0], half_not[0], ONE_REPEAT_BYTE_SIZE / sizeof(half), max_repeat, sub_repeat);
      Abs(dst[calc_size], half_not[0], max_repeat_calc_size);
    } else if constexpr (!std::is_same_v<T, int64_t>) {
      Cast(half_not[0], src[calc_size], round_mode, max_repeat_calc_size);
      do_logical_not(blk_tensor_with_value_1, half_not, sub_repeat, max_repeat, max_repeat_calc_size);
      Cast(dst[calc_size], half_not[0], round_mode, max_repeat_calc_size);
    }
  }

  // Calc max_repeat > size > one_repeat
  const int one_repeat_calc_size = ONE_REPEAT_BYTE_SIZE / sizeof(half);
  if (calc_size + one_repeat_calc_size <= size) {
    int repeat = (size - calc_size) / one_repeat_calc_size;
    if constexpr (std::is_same_v<T, half>) {
      Abs(half_not[0], src[calc_size], repeat * one_repeat_calc_size);
      AscendC::Min(half_not[0], blk_tensor_with_value_1[0], half_not[0], ONE_REPEAT_BYTE_SIZE / sizeof(half), repeat, sub_repeat);
      Sub(half_not[0], blk_tensor_with_value_1[0], half_not[0], ONE_REPEAT_BYTE_SIZE / sizeof(half), repeat, sub_repeat);
      Abs(dst[calc_size], half_not[0], repeat * one_repeat_calc_size);
    } else if constexpr (!std::is_same_v<T, int64_t>) {
      Cast(half_not[0], src[calc_size], round_mode, repeat * one_repeat_calc_size);
      do_logical_not(blk_tensor_with_value_1, half_not, sub_repeat, repeat, repeat * one_repeat_calc_size);
      Cast(dst[calc_size], half_not[0], round_mode, repeat * one_repeat_calc_size);
    }
    calc_size += repeat * one_repeat_calc_size;
  }

  // Calc when one_repeat > size
  if (calc_size < size) {
    sub_repeat.blockNumber = (size - calc_size + ONE_BLK_SIZE / sizeof(half) - 1) / (ONE_BLK_SIZE / sizeof(half));
    if constexpr (std::is_same_v<T, half>) {
      Abs(half_not[0], src[calc_size], size - calc_size);
      AscendC::Min(half_not[0], blk_tensor_with_value_1[0], half_not[0], ONE_REPEAT_BYTE_SIZE / sizeof(half), 1, sub_repeat);
      Sub(half_not[0], blk_tensor_with_value_1[0], half_not[0], ONE_REPEAT_BYTE_SIZE / sizeof(half), 1, sub_repeat);
      Abs(dst[calc_size], half_not[0], size - calc_size);
    } else if constexpr (!std::is_same_v<T, int64_t>) {
      Cast(half_not[0], src[calc_size], round_mode, size - calc_size);
      do_logical_not(blk_tensor_with_value_1, half_not, sub_repeat, 1, size - calc_size);
      Cast(dst[calc_size], half_not[0], round_mode, size - calc_size);
    }
  }
}

#endif  // __ASCENDC_API_LOGICAL_NOT_H__
