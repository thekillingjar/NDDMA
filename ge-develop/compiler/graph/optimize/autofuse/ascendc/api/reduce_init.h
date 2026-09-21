/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCENDC_API_REDUCEINIT_H__
#define __ASCENDC_API_REDUCEINIT_H__

#ifndef INFINITY
#define INFINITY (1.0f / 0.0f)
#endif

constexpr int32_t kReduceOpMin = 0;
constexpr int32_t kReduceOpMax = 1;
constexpr int32_t kReduceOpSum = 2;
constexpr int32_t kReduceOpProd = 3;
constexpr int32_t kReduceOpAny = 4;
constexpr int32_t kReduceOpAll = 5;
constexpr int32_t kReduceOpMean = 6;
constexpr uint8_t kMaxRepeatTimes = 255U;
constexpr uint64_t kOneBlockSize = 32U;
constexpr uint64_t kOneRepeatMaxSize = 256U;
constexpr uint8_t kMaxRepeatStride = 255U;

template <typename T, int reduce_type>
inline __aicore__ T GetPaddingValue() {
  T paddingValue = 1;
  if constexpr (reduce_type == kReduceOpMin) {
    paddingValue = INFINITY;
  } else if constexpr (reduce_type == kReduceOpMax) {
    paddingValue = -INFINITY;
  } else if constexpr (reduce_type == kReduceOpSum) {
    paddingValue = T(0);
  } else if constexpr (reduce_type == kReduceOpProd) {
    paddingValue = T(1);
  } else if constexpr (reduce_type == kReduceOpAny) {
    paddingValue = T(0);
  } else if constexpr (reduce_type == kReduceOpAll) {
    paddingValue = T(1);
  } else if constexpr (reduce_type == kReduceOpMean) {
    paddingValue = T(0);
  }
  return paddingValue;
}

template <typename T, int ReduceType, bool isTailLast>
__aicore__ inline void ReduceInit(const LocalTensor<T> &dst, const uint64_t dim_a, const uint64_t dim_r,
                                  const uint64_t dim_r_current, const uint64_t inner_r) {
  const uint64_t align_num = kOneBlockSize / sizeof(T);
  const uint64_t inner_r_align = (inner_r + align_num - 1) / align_num * align_num;
  const uint64_t inner_r_tail = inner_r % align_num;
  const uint64_t elem_num_one_repeat = kOneRepeatMaxSize / sizeof(T);
  T pad_value = GetPaddingValue<T, ReduceType>();
  if (inner_r_tail != 0) {
    uint64_t one_mask = (((uint64_t)1 << (align_num - inner_r_tail)) - 1) << inner_r_tail;
    uint64_t tail_mask[2] = {one_mask, 0};
    uint64_t all_mask = one_mask + (one_mask << 8) + (one_mask << 16) + (one_mask << 24);
    all_mask += (all_mask << 32);
    uint64_t mask[2] = {all_mask, 0};
    uint64_t repeat_time = 0;
    uint64_t reminder = 0;
    uint64_t block_stride = 1;
    if constexpr (isTailLast) {
      repeat_time = dim_a / 8;
      reminder = dim_a % 8;
      block_stride = dim_r / align_num;
    } else {
      repeat_time = dim_r / inner_r_align * dim_a / 8;
      reminder = dim_r / inner_r_align * dim_a % 8;
      block_stride = inner_r_align / align_num;
    }
    uint64_t repeat_stride = block_stride * 8;
    uint64_t repeat_stride_tail = block_stride;
    uint64_t loop_times = (repeat_time + kMaxRepeatTimes - 1) / kMaxRepeatTimes;
    uint64_t loop_times_tail = repeat_time - (loop_times - 1) * kMaxRepeatTimes;
    if (repeat_stride <= kMaxRepeatStride) {
      for (uint64_t i = 0; i < loop_times; i++) {
        uint64_t one_repeat_time = (i == loop_times - 1) ? loop_times_tail : kMaxRepeatTimes;
        Duplicate(dst[i * kMaxRepeatTimes * repeat_stride * align_num + inner_r_align - align_num], pad_value, mask,
                  one_repeat_time, block_stride, repeat_stride);
      }
    } else {
      for (uint64_t i = 0; i < repeat_time; i++) {
        Duplicate(dst[i * repeat_stride * align_num + inner_r_align - align_num], pad_value, mask, 1, block_stride,
                  repeat_stride);
      }
    }
    
    if (reminder != 0) {
      if (repeat_stride_tail <= kMaxRepeatStride) {
        Duplicate(dst[repeat_time * repeat_stride * align_num + inner_r_align - align_num], pad_value, tail_mask,
                  reminder, 1, repeat_stride_tail);
      } else {
        for (uint64_t i = 0; i < reminder; i++) {
          Duplicate(dst[repeat_time * repeat_stride * align_num + (i + 1) * inner_r_align - align_num], pad_value,
                    tail_mask, 1, 1, repeat_stride_tail);
        }
      }
    }
  }
  if (dim_r_current != dim_r) {
    uint64_t dim_r_diff = dim_r - dim_r_current;
    uint64_t repeat_count = (dim_r_diff + elem_num_one_repeat - 1) / elem_num_one_repeat;
    if (repeat_count >= dim_a) {
      uint64_t dim_a_loop_times = dim_a / kMaxRepeatTimes;
      uint64_t dim_a_reminder = dim_a - dim_a_loop_times * kMaxRepeatTimes;
      for (uint64_t i = 0; i < dim_a_loop_times; i++) {
        Duplicate(dst[i * kMaxRepeatTimes * dim_r + dim_r_current], pad_value, dim_r_diff, kMaxRepeatTimes, 1,
                  dim_r / align_num);
      }
      if (dim_a_reminder != 0) {
        for (int64_t i = 0; i < dim_a_reminder; i++) {
          Duplicate(dst[(dim_a_loop_times * kMaxRepeatTimes + i) * dim_r + dim_r_current], pad_value, dim_r_diff);
        }
      }
      return;
    }
    uint64_t reminder = dim_r_diff - (repeat_count - 1) * elem_num_one_repeat;
    for (uint64_t i = 0; i < repeat_count - 1; i++) {
      Duplicate(dst[dim_r_current + i * elem_num_one_repeat], pad_value, elem_num_one_repeat, dim_a, 1,
                dim_r / align_num);
    }
    Duplicate(dst[dim_r_current + (repeat_count - 1) * elem_num_one_repeat], pad_value, reminder, dim_a, 1,
              dim_r / align_num);
  }
}

template <typename T, bool isReuseSource = false, int32_t operatorType = 0>
inline __aicore__ void AREntireTailFold(const LocalTensor<T> &dst, const LocalTensor<T> &src0,
                                        const LocalTensor<T> &src1, const int32_t dim_a, const int32_t total_dim_r) {
  if constexpr (operatorType == kReduceOpSum || operatorType == kReduceOpMean) {
    Add(dst, src0, src1, dim_a * total_dim_r);
  } else if constexpr (operatorType == kReduceOpProd) {
    Mul(dst, src0, src1, dim_a * total_dim_r);
  }
}

template <typename T, bool isReuseSource = false, int32_t operatorType = 0>
inline __aicore__ void RAEntireTailFold(const LocalTensor<T> &dst, const LocalTensor<T> &src0,
                                        const LocalTensor<T> &src1, const int32_t dim_a, const int32_t total_dim_r,
                                        const int32_t tail_dim_r) {
  if constexpr (isReuseSource) {
    if constexpr (operatorType == kReduceOpSum || operatorType == kReduceOpMean) {
      Add(dst, src0, src1, tail_dim_r * dim_a);
    } else if constexpr (operatorType == kReduceOpProd) {
      Mul(dst, src0, src1, tail_dim_r * dim_a);
    }
    if (total_dim_r > tail_dim_r) {
      DataCopy(dst[tail_dim_r * dim_a], src0[tail_dim_r * dim_a], (total_dim_r - tail_dim_r) * dim_a);
    }
  } else {
    if constexpr (operatorType == kReduceOpSum || operatorType == kReduceOpMean) {
      Add(dst, src0, src1, tail_dim_r * dim_a);
      if (total_dim_r > tail_dim_r) {
        DataCopy(dst[tail_dim_r * dim_a], src0[tail_dim_r * dim_a], (total_dim_r - tail_dim_r) * dim_a);
      }
    } else if constexpr (operatorType == kReduceOpProd) {
      Mul(dst, src0, src1, tail_dim_r * dim_a);
      if (total_dim_r > tail_dim_r) {
        DataCopy(dst[tail_dim_r * dim_a], src0[tail_dim_r * dim_a], (total_dim_r - tail_dim_r) * dim_a);
      }
    }
  }
}

template <typename T, bool isReuseSource = false, bool isAr = false, int32_t operatorType = 0>
inline __aicore__ void EntireTailFold(const LocalTensor<T> &dst, const LocalTensor<T> &src0, const LocalTensor<T> &src1,
                                      const int32_t dim_a, const int32_t total_dim_r, const int32_t tail_dim_r) {
  if constexpr (isAr) {
    return AREntireTailFold<T, isReuseSource, operatorType>(dst, src0, src1, dim_a, total_dim_r);
  } else {
    return RAEntireTailFold<T, isReuseSource, operatorType>(dst, src0, src1, dim_a, total_dim_r, tail_dim_r);
  }
}

#endif  // __ASCENDC_API_REDUCEINIT_H__