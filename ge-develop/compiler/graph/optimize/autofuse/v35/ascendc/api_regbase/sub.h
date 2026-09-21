
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCENDC_REGBASE_API_SUB_H__
#define __ASCENDC_REGBASE_API_SUB_H__

template <typename T>
inline __aicore__ void SubExtend(const LocalTensor<T> &dst, const LocalTensor<T> &src1, const LocalTensor<T> &src2,
                            const uint32_t calc_cnt) {
  AscendC::Sub(dst, src1, src2, calc_cnt);
}

template <typename T, bool IS_SCALAR_LATTER = true>
inline __aicore__ void SubExtends(const LocalTensor<T> &dst, const LocalTensor<T> &src, const T constant_x,
                            const uint32_t calc_cnt) {
  if constexpr (IS_SCALAR_LATTER) {
    AscendC::Subs(dst, src, constant_x, calc_cnt);
  } else {
    AscendC::Subs(dst, constant_x, src, calc_cnt);
  }
}

template <typename T>
inline __aicore__ void SubExtends(const LocalTensor<T>& dst, const T x, const T y) {
  T res = x - y;
  AscendC::Duplicate(dst, res, dst.GetSize());
}

#endif  // __ASCENDC_REGBASE_API_SUB_H__
