/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCENDC_API_BITWISE_AND_H__
#define __ASCENDC_API_BITWISE_AND_H__

template <typename T>
inline __aicore__ void BitwiseAndExtend(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<T> &src_0,
                                        const AscendC::LocalTensor<T> &src_1, const uint32_t size,
                                        LocalTensor<uint8_t> &tmp_buf) {
  if constexpr (AscendC::SupportType<T, uint16_t, int16_t>()) {
    AscendC::And(dst, src_0, src_1, size);
  } else {
    auto src_0_tmp = src_0.template ReinterpretCast<int16_t>();
    auto src_1_tmp = src_1.template ReinterpretCast<int16_t>();
    auto dst_tmp = dst.template ReinterpretCast<int16_t>();
    AscendC::And(dst_tmp, src_0_tmp, src_1_tmp, (size * sizeof(T) + sizeof(int16_t) - 1) / sizeof(int16_t));
  }
}
#endif  // __ASCENDC_API_BITWISE_AND_H__
