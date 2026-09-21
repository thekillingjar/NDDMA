/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCENDC_API_AXPY_H__
#define __ASCENDC_API_AXPY_H__

template <typename T>
inline __aicore__ void AxpyExtend(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<T> &src_0,
                                  const AscendC::LocalTensor<T> &src_1, const float alpha,
                                  const uint32_t count, const AscendC::LocalTensor<uint8_t> &tmp_buf) {

    if constexpr (AscendC::IsSameType<T, half>::value) {
      // T如果是half类型，需要把tmp_buf分成2份
      constexpr uint32_t tmp_buf_splits = 2U;
      constexpr uint32_t cast_dst_align = 32U; // cast算子要求目标地址32B对齐
      auto offset = tmp_buf.GetSize() / tmp_buf_splits / cast_dst_align * cast_dst_align;

      LocalTensor<float> cast_float_src_0 = tmp_buf[0].template ReinterpretCast<float>();
      LocalTensor<float> cast_float_src_1 = tmp_buf[offset].template ReinterpretCast<float>();

      AscendC::Cast(cast_float_src_0, src_0, RoundMode::CAST_NONE, count);
      AscendC::Cast(cast_float_src_1, src_1, RoundMode::CAST_NONE, count);
      AscendC::PipeBarrier<PIPE_V>();

      AscendC::Muls(cast_float_src_1, cast_float_src_1, alpha, count);
      AscendC::Add(cast_float_src_0, cast_float_src_0, cast_float_src_1, count);
      AscendC::PipeBarrier<PIPE_V>();

      AscendC::Cast(dst, cast_float_src_0, RoundMode::CAST_RINT, count);
    } else {
      AscendC::Muls(src_1, src_1, alpha, count);
      AscendC::Add(dst, src_0, src_1, count);
    }
}
#endif  // __ASCENDC_API_AXPY_H__