/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCENDC_API_CLIPBYVALUE_H__
#define __ASCENDC_API_CLIPBYVALUE_H__

// 场景1，x, min, max, 是tensor
template <typename T>
inline __aicore__ void ClipByValue(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<T> &src_x,
                                   const AscendC::LocalTensor<T> &src_min, const AscendC::LocalTensor<T> &src_max,
                                   const uint32_t calCount, AscendC::LocalTensor<uint8_t> &tmp_buf) {
  const uint32_t max_rpt_cnt = KernelUtils::MaxRptSize<T>();
  const uint32_t loop_cnt = calCount / max_rpt_cnt;
  uint32_t offset = 0;
  for (uint32_t loop = 0; loop < loop_cnt; loop++) {
    AscendC::Min(dst[offset], src_x[offset], src_max[offset], max_rpt_cnt);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Max(dst[offset], dst[offset], src_min[offset], max_rpt_cnt);
    offset += max_rpt_cnt;
  }
  const uint32_t tail_cnt = calCount - offset;
  if (tail_cnt != 0) {
    AscendC::Min(dst[offset], src_x[offset], src_max[offset], tail_cnt);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Max(dst[offset], dst[offset], src_min[offset], tail_cnt);
  }
}

// 场景2，x是tensor，min, max是scalar
template <typename T>
inline __aicore__ void ClipByValue(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<T> &src_x,
                                   const T &src_min, const T &src_max, const uint32_t calCount,
                                   AscendC::LocalTensor<uint8_t> &tmp_buf) {
  constexpr auto minSize = sizeof(T) * 32;
  const auto maxTmpSize = (tmp_buf.GetSize() - minSize) / sizeof(T);
  constexpr auto repeatSize = 256 / sizeof(T);
  const auto calsize = (calCount * sizeof(T) + 31) / 32 * 32;
  if (maxTmpSize * sizeof(T) > calsize) {
    auto tmp0 = tmp_buf.ReinterpretCast<T>();
    auto sharedTmpBuffer = tmp_buf[calsize];
    AscendC::ClampMax(tmp0, src_x, sharedTmpBuffer, src_max, calCount);
    AscendC::ClampMin(dst, tmp0, sharedTmpBuffer, src_min, calCount);
  } else {
    const int32_t repeatTime = DivCeil(calCount, repeatSize);
    const int32_t maxRepeatTime = maxTmpSize / repeatSize;
    const int32_t loopTime = repeatTime / maxRepeatTime;
    const int32_t loopSize = maxRepeatTime * repeatSize;
    const int32_t tailSize = calCount - (loopSize * loopTime);
    auto tmp0 = tmp_buf.ReinterpretCast<T>();
    auto sharedTmpBuffer = tmp_buf[loopSize * sizeof(T)];
    for (int32_t i = 0; i < loopTime; i++) {
      AscendC::ClampMax(tmp0, src_x[loopSize * i], sharedTmpBuffer, src_max, loopSize);
      AscendC::ClampMin(dst[loopSize * i], tmp0, sharedTmpBuffer, src_min, loopSize);
    }
    if (tailSize > 0) {
      AscendC::ClampMax(tmp0, src_x[loopSize * loopTime], sharedTmpBuffer, src_max, tailSize);
      AscendC::ClampMin(dst[loopSize * loopTime], tmp0, sharedTmpBuffer, src_min, tailSize);
    }
  }
}

// 场景3，x、min是tensor，max是scalar
template <typename T>
inline __aicore__ void ClipByValue(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<T> &src_x,
                                   const AscendC::LocalTensor<T> &src_min, const T &src_max, const uint32_t calCount,
                                   AscendC::LocalTensor<uint8_t> &tmp_buf) {
  const uint32_t max_rpt_cnt = KernelUtils::MaxRptSize<T>();
  const uint32_t loop_cnt = calCount / max_rpt_cnt;
  uint32_t offset = 0;
  for (uint32_t loop = 0; loop < loop_cnt; loop++) {
    AscendC::Mins(dst[offset], src_x[offset], src_max, max_rpt_cnt);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Max(dst[offset], dst[offset], src_min[offset], max_rpt_cnt);
    offset += max_rpt_cnt;
  }
  const uint32_t tail_cnt = calCount - offset;
  if (tail_cnt != 0) {
    AscendC::Mins(dst[offset], src_x[offset], src_max, tail_cnt);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Max(dst[offset], dst[offset], src_min[offset], tail_cnt);
  }
}

// 场景4，x、max是tensor，min是scalar
template <typename T>
inline __aicore__ void ClipByValue(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<T> &src_x,
                                   const T &src_min, const AscendC::LocalTensor<T> &src_max, const uint32_t calCount,
                                   AscendC::LocalTensor<uint8_t> &tmp_buf) {
  const uint32_t max_rpt_cnt = KernelUtils::MaxRptSize<T>();
  const uint32_t loop_cnt = calCount / max_rpt_cnt;
  uint32_t offset = 0;
  for (uint32_t loop = 0; loop < loop_cnt; loop++) {
    AscendC::Min(dst[offset], src_x[offset], src_max[offset], max_rpt_cnt);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Maxs(dst[offset], dst[offset], src_min, max_rpt_cnt);
    offset += max_rpt_cnt;
  }
  const uint32_t tail_cnt = calCount - offset;
  if (tail_cnt != 0) {
    AscendC::Min(dst[offset], src_x[offset], src_max[offset], tail_cnt);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Maxs(dst[offset], dst[offset], src_min, tail_cnt);
  }
}

// 场景4， x，min，max都是scalar
template <typename T>
inline __aicore__ void ClipByValue(const AscendC::LocalTensor<T> &dst, const T &src_x, const T &src_min,
                                   const T &src_max) {
  T val;
  uint32_t count = dst.GetSize();
  if (static_cast<float>(src_x) < static_cast<float>(src_min)) {
    val = src_min;
  } else if (static_cast<float>(src_x) > static_cast<float>(src_max)) {
    val = src_max;
  } else {
    val = src_x;
  }
  AscendC::Duplicate(dst, val, count);
}
#endif  // __ASCENDC_API_CLIPBYVALUE_H__
