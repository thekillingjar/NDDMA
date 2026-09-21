
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCENDC_API_FLOOR_DIV_H__
#define __ASCENDC_API_FLOOR_DIV_H__

template <typename T>
inline __aicore__ void FloorDivExtend(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<T> &src1,
                                      const AscendC::LocalTensor<T> &src2, const uint32_t size,
                                      AscendC::LocalTensor<uint8_t> &tmp_buf) {
  uint32_t buf_size = tmp_buf.GetSize() / 2 / ONE_BLK_SIZE * ONE_BLK_SIZE;
  LocalTensor<T> div_res = tmp_buf.ReinterpretCast<T>();
  LocalTensor<uint8_t> remain_tmp_buf = tmp_buf[buf_size];
  constexpr uint32_t ONE_RPT_SIZE = ONE_REPEAT_BYTE_SIZE / sizeof(T);

  uint32_t max_buf_size = buf_size / sizeof(T);
  uint32_t max_buf_rpt_num = max_buf_size / ONE_RPT_SIZE;
  uint32_t max_do_rpt_num = MAX_REPEAT_TIME > max_buf_rpt_num ? max_buf_rpt_num : MAX_REPEAT_TIME;
  uint32_t max_do_size = max_do_rpt_num * ONE_RPT_SIZE;

  uint32_t do_size = 0;                              // will calculate number size
  uint64_t mask = 0;                                 // Select api mask
  uint8_t repeat_times = 0;                          // Select api repeatTimes
  uint32_t calc_size = 0;                            // has calculated number size
  // do max repeat once time
  if (max_do_size <= size) {
    do_size = max_do_size;
    mask = ONE_RPT_SIZE;
    repeat_times = max_do_rpt_num;
    AscendC::SetMaskNorm();
    AscendC::SetVectorMask<T, AscendC::MaskMode::NORMAL>(mask);
    for (; calc_size + do_size < size; calc_size += do_size) {
      AscendC::Div<T, false>(div_res, src1[calc_size], src2[calc_size], AscendC::MASK_PLACEHOLDER, repeat_times, {1, 1, 1, 8, 8, 8});
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Floor(dst[calc_size], div_res, remain_tmp_buf, do_size);
      AscendC::PipeBarrier<PIPE_V>();
    }
    AscendC::ResetMask();
  }
  // do left repeats
  if (calc_size + ONE_RPT_SIZE <= size) {
    uint32_t left_rpt_num = (size - calc_size) / ONE_RPT_SIZE;
    do_size = left_rpt_num * ONE_RPT_SIZE;
    mask = ONE_RPT_SIZE;
    repeat_times = left_rpt_num;
    AscendC::Div(div_res, src1[calc_size], src2[calc_size], mask, repeat_times, {1, 1, 1, 8, 8, 8});
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Floor(dst[calc_size], div_res, remain_tmp_buf, do_size);
    AscendC::PipeBarrier<PIPE_V>();
    calc_size += do_size;
  }
  // do left blocks
  if (calc_size < size) {
    uint32_t left_size = size - calc_size;
    AscendC::Div(div_res, src1[calc_size], src2[calc_size], left_size);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Floor(dst[calc_size], div_res, remain_tmp_buf, left_size);
  }
}

#endif  // __ASCENDC_API_FLOOR_DIV_H__
