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
                                   const uint32_t cal_count) {
  Clamp(dst, src_x, src_min, src_max, cal_count);
}

// 场景2，x是tensor，min, max是scalar
template <typename T>
inline __aicore__ void ClipByValue(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<T> &src_x,
                                   const T &src_min, const T &src_max, const uint32_t cal_count) {
  Clamp(dst, src_x, src_min, src_max, cal_count);
}

// 场景3，x、min是tensor，max是scalar
template <typename T>
inline __aicore__ void ClipByValue(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<T> &src_x,
                                   const AscendC::LocalTensor<T> &src_min, const T &src_max, const uint32_t cal_count) {
  Clamp(dst, src_x, src_min, src_max, cal_count);
}

// 场景4，x、max是tensor，min是scalar
template <typename T>
inline __aicore__ void ClipByValue(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<T> &src_x,
                                   const T &src_min, const AscendC::LocalTensor<T> &src_max, const uint32_t cal_count) {
  Clamp(dst, src_x, src_min, src_max, cal_count);
}

// 场景5， x，min，max都是scalar
template <typename T>
inline __aicore__ void ClipByValue(const AscendC::LocalTensor<T> &dst, const T &src_x, const T &src_min,
                                   const T &src_max) {
  T val;
  uint32_t count = dst.GetSize();
  if (src_x < src_min) {
    val = src_min;
  } else if (src_x > src_max) {
    val = src_max;
  } else {
    val = src_x;
  }
  AscendC::Duplicate(dst, val, count);
}
#endif  // __ASCENDC_API_CLIPBYVALUE_H__
