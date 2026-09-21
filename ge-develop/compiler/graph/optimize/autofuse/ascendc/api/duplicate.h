
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCENDC_API_DUPLICATE_H__
#define __ASCENDC_API_DUPLICATE_H__

template <typename T>
inline __aicore__ void DuplicateInt64(const AscendC::LocalTensor<T> &dst, T src, const uint32_t size,
                                      AscendC::LocalTensor<uint8_t> &tmp_buf) {
  uint16_t bit_00 = src & 0xFFFF;                            // 最低16位
  uint16_t bit_16 = src >> 16 & 0xFFFF;                      // 次低16位
  uint16_t bit_32 = src >> 32 & 0xFFFF;                      // 次高16位
  uint16_t bit_48 = src >> 48 & 0xFFFF;                      // 最高16位
  uint32_t cal_cnt = size * (sizeof(T) / sizeof(uint16_t));  // int64 -> int16

  constexpr uint32_t TRANSPOSE_MIN_NUM = 16;
  constexpr uint32_t TRANSPOSE_MIN_CUBE_BYTE_SIZE = TRANSPOSE_MIN_NUM * TRANSPOSE_MIN_NUM * sizeof(uint16_t);

  AscendC::LocalTensor<uint16_t> init_buf = KernelUtils::NewTensor<uint16_t>(tmp_buf, 0, TRANSPOSE_MIN_CUBE_BYTE_SIZE);
  Duplicate(init_buf[TRANSPOSE_MIN_NUM * 0], bit_00, TRANSPOSE_MIN_NUM);
  Duplicate(init_buf[TRANSPOSE_MIN_NUM * 1], bit_16, TRANSPOSE_MIN_NUM);
  Duplicate(init_buf[TRANSPOSE_MIN_NUM * 2], bit_32, TRANSPOSE_MIN_NUM);
  Duplicate(init_buf[TRANSPOSE_MIN_NUM * 3], bit_48, TRANSPOSE_MIN_NUM);
  Duplicate(init_buf[TRANSPOSE_MIN_NUM * 4], bit_00, TRANSPOSE_MIN_NUM);
  Duplicate(init_buf[TRANSPOSE_MIN_NUM * 5], bit_16, TRANSPOSE_MIN_NUM);
  Duplicate(init_buf[TRANSPOSE_MIN_NUM * 6], bit_32, TRANSPOSE_MIN_NUM);
  Duplicate(init_buf[TRANSPOSE_MIN_NUM * 7], bit_48, TRANSPOSE_MIN_NUM);
  Duplicate(init_buf[TRANSPOSE_MIN_NUM * 8], bit_00, TRANSPOSE_MIN_NUM);
  Duplicate(init_buf[TRANSPOSE_MIN_NUM * 9], bit_16, TRANSPOSE_MIN_NUM);
  Duplicate(init_buf[TRANSPOSE_MIN_NUM * 10], bit_32, TRANSPOSE_MIN_NUM);
  Duplicate(init_buf[TRANSPOSE_MIN_NUM * 11], bit_48, TRANSPOSE_MIN_NUM);
  Duplicate(init_buf[TRANSPOSE_MIN_NUM * 12], bit_00, TRANSPOSE_MIN_NUM);
  Duplicate(init_buf[TRANSPOSE_MIN_NUM * 13], bit_16, TRANSPOSE_MIN_NUM);
  Duplicate(init_buf[TRANSPOSE_MIN_NUM * 14], bit_32, TRANSPOSE_MIN_NUM);
  Duplicate(init_buf[TRANSPOSE_MIN_NUM * 15], bit_48, TRANSPOSE_MIN_NUM);
  AscendC::PipeBarrier<PIPE_V>();
  Transpose(init_buf, init_buf);
  AscendC::PipeBarrier<PIPE_V>();
  AscendC::LocalTensor<uint16_t> new_dst = dst.template ReinterpretCast<uint16_t>();
  CopyRepeatParams params;
  params.srcStride = 0;
  params.dstStride = DEFAULT_BLK_STRIDE;
  params.srcRepeatSize = 0;
  params.dstRepeatSize = DEFAULT_REPEAT_STRIDE;
  constexpr uint32_t ONE_RPT_SIZE = KernelUtils::RptSize<uint16_t>();
  constexpr uint32_t MAX_RPT_SIZE = MAX_REPEAT_TIME * ONE_RPT_SIZE;

  uint32_t calc_size = 0;
  // do max repeat once time
  for (; calc_size + MAX_RPT_SIZE <= cal_cnt; calc_size += MAX_RPT_SIZE) {
    Copy(new_dst[calc_size], init_buf, ONE_RPT_SIZE, MAX_REPEAT_TIME, params);
    AscendC::PipeBarrier<PIPE_V>();
  }

  // do left repeats
  if (calc_size + ONE_RPT_SIZE <= cal_cnt) {
    uint32_t left_rpt_num = KernelUtils::RptNum<uint16_t>(cal_cnt - calc_size);
    uint32_t do_size = left_rpt_num * KernelUtils::RptSize<uint16_t>();
    Copy(new_dst[calc_size], init_buf, ONE_RPT_SIZE, left_rpt_num, params);
    calc_size += do_size;
    AscendC::PipeBarrier<PIPE_V>();
  }

  // do left blocks
  if (calc_size < cal_cnt) {
    uint32_t left_size = cal_cnt - calc_size;
    uint32_t mask = KernelUtils::SizeAlign(left_size, KernelUtils::BlkSize<uint16_t>());
    Copy(new_dst[calc_size], init_buf, mask, 1, params);
    AscendC::PipeBarrier<PIPE_V>();
  }
}

inline __aicore__ void Duplicate(const AscendC::LocalTensor<int64_t> &dst, int64_t src, const uint32_t size,
                                 AscendC::LocalTensor<uint8_t> &tmp_buf) {
  DuplicateInt64(dst, src, size, tmp_buf);
}

inline __aicore__ void Duplicate(const AscendC::LocalTensor<uint64_t> &dst, uint64_t src, const uint32_t size,
                                 AscendC::LocalTensor<uint8_t> &tmp_buf) {
  DuplicateInt64(dst, src, size, tmp_buf);
}

inline __aicore__ void Duplicate(const AscendC::LocalTensor<uint8_t> &dst, uint8_t src, const uint32_t size) {
  uint16_t tmp_src = src;
  tmp_src <<= 8;
  tmp_src |= src;
  const AscendC::LocalTensor<uint16_t> tmp_dst = dst.ReinterpretCast<uint16_t>();
  Duplicate(tmp_dst, tmp_src, static_cast<int32_t>(dst.GetSize() + 1) / 2);
}

inline __aicore__ void Duplicate(const AscendC::LocalTensor<uint8_t> &dst, uint8_t src, const uint32_t size,
                                 AscendC::LocalTensor<uint8_t> &tmp_buf) {
  Duplicate(dst, src, size);
}

inline __aicore__ void Duplicate(const AscendC::LocalTensor<int8_t> &dst, int8_t src, const uint32_t size) {
  uint16_t tmp_src = static_cast<uint16_t>((uint8_t)src);
  tmp_src <<= 8;
  tmp_src |= (uint8_t)src;
  const AscendC::LocalTensor<uint16_t> tmp_dst = dst.ReinterpretCast<uint16_t>();
  AscendC::Duplicate(tmp_dst, tmp_src, static_cast<int32_t>(size + 1) / 2);
}

inline __aicore__ void Duplicate(const AscendC::LocalTensor<int8_t> &dst, int8_t src, const uint32_t size,
                                 AscendC::LocalTensor<uint8_t> &tmp_buf) {
  Duplicate(dst, src, size);
}

#endif  // __ASCENDC_API_DUPLICATE_H__