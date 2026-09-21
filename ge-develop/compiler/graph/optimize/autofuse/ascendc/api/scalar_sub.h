
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCENDC_API_SCALAR_SUB_H__
#define __ASCENDC_API_SCALAR_SUB_H__

template <typename T, bool IS_SCALAR_LATTER = true>
inline __aicore__ void Subs(const LocalTensor<T> &dst, const LocalTensor<T> &src, const T constant_x,
                            const uint32_t calc_cnt, LocalTensor<uint8_t> &tmp_buf) {
  LocalTensor<T> tmp = tmp_buf.template ReinterpretCast<T>();

  uint32_t calc_size = 0U;
  uint32_t one_blk_num = KernelUtils::BlkSize<T>();
  uint32_t one_rpt_num = KernelUtils::RptSize<T>();
  Duplicate(tmp, constant_x, one_blk_num);
  AscendC::PipeBarrier<PIPE_V>();
  uint32_t repeat_times = calc_cnt / one_rpt_num;
  BinaryRepeatParams binary_param(1U, 1U, 0U, 8U, 8U, 0U);
  if constexpr (!IS_SCALAR_LATTER) {
    binary_param.src0BlkStride = 0U;
    binary_param.src1BlkStride = 1U;
    binary_param.src0RepStride = 0U;
    binary_param.src1RepStride = 8U;
  }
  uint32_t max_rpt_loop_cnt = repeat_times / MAX_REPEAT_TIME;
  SetMaskNorm();
  AscendC::SetVectorMask<T, MaskMode::NORMAL>(one_rpt_num);
  for (uint32_t idx = 0U; idx < max_rpt_loop_cnt; idx++) {
    if constexpr (IS_SCALAR_LATTER) {
      AscendC::Sub<T, false>(dst[calc_size], src[calc_size], tmp, MASK_PLACEHOLDER, MAX_REPEAT_TIME, binary_param);
    } else {
      AscendC::Sub<T, false>(dst[calc_size], tmp, src[calc_size], MASK_PLACEHOLDER, MAX_REPEAT_TIME, binary_param);
    }
    calc_size += MAX_REPEAT_TIME * one_rpt_num;
  }
  uint32_t remain_rpt_times = repeat_times - max_rpt_loop_cnt * MAX_REPEAT_TIME;
  if (remain_rpt_times != 0U) {
    if constexpr (IS_SCALAR_LATTER) {
      AscendC::Sub<T, false>(dst[calc_size], src[calc_size], tmp, MASK_PLACEHOLDER, remain_rpt_times, binary_param);
    } else {
      AscendC::Sub<T, false>(dst[calc_size], tmp, src[calc_size], MASK_PLACEHOLDER, remain_rpt_times, binary_param);
    }
    calc_size += remain_rpt_times * one_rpt_num;
  }
  uint32_t calc_tail = calc_cnt - calc_size;
  if (calc_tail != 0U) {
    SetMaskNorm();
    AscendC::SetVectorMask<T, MaskMode::NORMAL>(calc_tail);
    if constexpr (IS_SCALAR_LATTER) {
      AscendC::Sub<T, false>(dst[calc_size], src[calc_size], tmp, MASK_PLACEHOLDER, 1, binary_param);
    } else {
      AscendC::Sub<T, false>(dst[calc_size], tmp, src[calc_size], MASK_PLACEHOLDER, 1, binary_param);
    }
  }
  AscendC::SetMaskNorm();
  AscendC::ResetMask();
}

template <typename T>
inline __aicore__ void Subs(const LocalTensor<T>& dst, const T x, const T y) {
  T res = x - y;
  AscendC::Duplicate(dst, res, dst.GetSize());
}

#endif  // __ASCENDC_API_SCALAR_SUB_H__
