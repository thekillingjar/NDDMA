/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __ASCENDC_API_REGBASE_GELU_H__
#define __ASCENDC_API_REGBASE_GELU_H__

template<typename T>
__simd_vf__ inline void GeluCompute(__ubuf__ T* dst, __ubuf__ T* src, uint32_t count) 
{
    constexpr float kTanhApproxFactor = 1 / 0.044715;
    constexpr float kNegSqrtEightOverPi = -1.595769121 * 0.044715;
    constexpr uint32_t kOneRepeatElem = static_cast<uint32_t>(GetVecLen() / sizeof(T));
    uint16_t repeat_times = static_cast<uint16_t>(CeilDivision(count, kOneRepeatElem));
    MicroAPI::RegTensor<T> vreg_input;
    MicroAPI::RegTensor<T> vreg_input_sqr;
    MicroAPI::RegTensor<T> vreg_input_cube;
    MicroAPI::RegTensor<T> vreg_output;
    MicroAPI::MaskReg mask;

    for (uint16_t i = 0; i < repeat_times; i++) {
        mask = MicroAPI::UpdateMask<T>(count);
        // OpCopyIn
        MicroAPI::DataCopy(vreg_input, src + i * kOneRepeatElem);
        MicroAPI::Mul(vreg_input_sqr, vreg_input, vreg_input, mask);
        MicroAPI::Mul(vreg_input_cube, vreg_input_sqr, vreg_input, mask);
        MicroAPI::Axpy(vreg_input_cube, vreg_input, kTanhApproxFactor, mask);
        MicroAPI::Muls(vreg_input_cube, vreg_input_cube, kNegSqrtEightOverPi, mask);
        MicroAPI::Exp(vreg_input_cube, vreg_input_cube, mask);
        MicroAPI::Adds(vreg_input_cube, vreg_input_cube, 1.0f, mask);
        MicroAPI::Div(vreg_output, vreg_input, vreg_input_cube, mask);

        // OpCopyOut
        MicroAPI::DataCopy(dst + i * kOneRepeatElem, vreg_output, mask);
    }
}

template<typename T>
__aicore__ inline void GeluCustomExtend(
    const LocalTensor<T>& dst, const LocalTensor<T>& src, const uint32_t count) 
{
    static_assert(SupportType<T, float>(), "Type must be float in GeluCustomExtend API");
    __ubuf__ T* dst_addr = (__ubuf__ T *)dst.GetPhyAddr();
    __ubuf__ T* src_addr = (__ubuf__ T *)src.GetPhyAddr();

    VF_CALL<GeluCompute<T>>(dst_addr, src_addr, count);
}

#endif // __ASCENDC_API_REGBASE_GELU_H__
