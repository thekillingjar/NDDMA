/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __ASCENDC_API_REGBASE_WHERE_V2_H__
#define __ASCENDC_API_REGBASE_WHERE_V2_H__

using namespace AscendC;

/**
 * 场景1： src0和src1都是标量，输出Shape与mask相同
 */
template <typename T>
inline __aicore__ void WhereExtend(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<uint8_t> &mask, T src0,
                                   T src1, const uint32_t size) {
  auto mask_tmp = mask.template ReinterpretCast<bool>();
  Where(dst, src0, src1, mask_tmp, size);
}

/**
 * 场景2： src0是标量，src1及输出的Shape与mask相同
 */
template <typename T>
inline __aicore__ void WhereExtend(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<uint8_t> &mask, T src0,
                                   const AscendC::LocalTensor<T> &src1, const uint32_t size) {
  auto mask_tmp = mask.template ReinterpretCast<bool>();
  Where(dst, src0, src1, mask_tmp, size);
}

inline __aicore__ void WhereExtend(const AscendC::LocalTensor<int64_t> &dst, const AscendC::LocalTensor<uint8_t> &mask,
                                   float src0, const AscendC::LocalTensor<int64_t> &src1, const uint32_t size) {
  auto mask_tmp = mask.template ReinterpretCast<bool>();
  Where(dst, static_cast<int64_t>(src0), src1, mask_tmp, size);
}

/**
 * 场景3： src1是标量，src0及输出的Shape与mask相同
 */
template <typename T>
inline __aicore__ void WhereExtend(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<uint8_t> &mask,
                                   const AscendC::LocalTensor<T> &src0, T src1, const uint32_t size) {
  auto mask_tmp = mask.template ReinterpretCast<bool>();
  Where(dst, src0, src1, mask_tmp, size);
}

/**
 * 场景4: src0和src1都不是标量，且Shape均与mask相同，且不需要广播。
 */
template <typename T>
inline __aicore__ void WhereExtend(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<uint8_t> &mask,
                                   const AscendC::LocalTensor<T> &src0, const AscendC::LocalTensor<T> &src1,
                                   const uint32_t size) {
  auto mask_tmp = mask.template ReinterpretCast<bool>();
  Where(dst, src0, src1, mask_tmp, size);
}

template <bool isBcastSrc0 = false, bool isBcastSrc1 = false, typename T, const MicroAPI::RegTrait& regTrait = MicroAPI::RegTraitNumOne>
__simd_vf__ inline void WhereNormal2DVecImpl(__local_mem__ T *dstUb,__local_mem__ uint8_t *maskUb,
    __local_mem__ T *src0Ub,__local_mem__ T *src1Ub, const uint16_t dstStride, const uint16_t maskStride,
    const uint16_t srcStride, const uint16_t repeatTime, const uint16_t counterFirst, const uint16_t counterLast, 
    T scalar0, T scalar1) 
{
    constexpr uint32_t repeatElm = regTrait.REG_NUM * GetVecLen() / sizeof(T);
    MicroAPI::RegTensor<T, regTrait> src0Reg, src1Reg, dstReg;
    MicroAPI::RegTensor<uint8_t> selReg;
    MicroAPI::MaskReg maskReg, selMask;
    MicroAPI::MaskReg maskFull = MicroAPI::CreateMask<uint8_t>();

    if constexpr (isBcastSrc0) {
        MicroAPI::Duplicate(src0Reg, scalar0);
    }
    if constexpr (isBcastSrc1) {
        MicroAPI::Duplicate(src1Reg, scalar1);
    }
    uint32_t count = 0;
    for (uint16_t j = 0U; j < counterFirst; ++j) {
        count = static_cast<uint32_t>(counterLast);
        for (uint16_t i = 0U; i < repeatTime; ++i) {
            maskReg = MicroAPI::UpdateMask<T, regTrait>(count);
            MicroAPI::DataCopy(selReg, maskUb + j * maskStride + i * repeatElm);
            MicroAPI::CompareScalar<uint8_t, CMPMODE::NE>(selMask, selReg, static_cast<uint8_t>(0), maskFull);
            if constexpr (sizeof(T) == 2) {
                MicroAPI::MaskUnPack(selMask, selMask);
            } else if constexpr (sizeof(T) == 4 || sizeof(T) == 8) {
                MicroAPI::MaskUnPack(selMask, selMask);
                MicroAPI::MaskUnPack(selMask, selMask);
            }

            if constexpr (!isBcastSrc0) {
                MicroAPI::DataCopy(src0Reg, src0Ub + j * srcStride + i * repeatElm);
            }
            if constexpr (!isBcastSrc1) {
                MicroAPI::DataCopy(src1Reg, src1Ub + j * srcStride + i * repeatElm);
            }
            MicroAPI::Select(dstReg, src0Reg, src1Reg, selMask);
            MicroAPI::DataCopy(dstUb + j * dstStride + i * repeatElm, dstReg, maskReg);
        }
    }
}

template <bool isBcastSrc0 = false, bool isBcastSrc1 = false, uint8_t dim, typename T, typename T1, typename T2>
inline __aicore__  void WhereExtend(const LocalTensor<T> &dst, const LocalTensor<uint8_t> &mask,
    const LocalTensor<T1> &src0, const LocalTensor<T2> &src1, const uint16_t (&output_dims)[dim],
    const uint16_t (&output_stride)[dim], const uint16_t (&mask_stride)[dim], const uint16_t (&input_stride)[dim]) 
{
    static_assert((dim == 1) || (dim == 2),"Where only support dim=1 or dim=2");
    static_assert(Std::is_same<T, T1>::value,"Where:T must be the same as T1");
    static_assert(Std::is_same<T, T2>::value,"Where:T must be the same as T2");
    T1 scalar0 = 0;
    T2 scalar1 = 0;
    if (isBcastSrc0 == true) {
        scalar0 = src0.GetValue(0);
    }
    if (isBcastSrc1 == true) {
        scalar1 = src1.GetValue(0);
    }
    __local_mem__ T *dstLocal = (__local_mem__ T *)dst.GetPhyAddr();
    __local_mem__ uint8_t *maskLocal = (__local_mem__ uint8_t *)mask.GetPhyAddr();
    __local_mem__ T1 *src0Local = (__local_mem__ T1 *)src0.GetPhyAddr();
    __local_mem__ T2 *src1Local = (__local_mem__ T2 *)src1.GetPhyAddr();
    const uint16_t dstStride = dim == 1 ? 1 : output_stride[0];
    const uint16_t maskStride = dim == 1 ? 1 : mask_stride[0];
    const uint16_t srcStride = dim == 1 ? 1 : input_stride[0];
    uint16_t counterFirst = dim == 1 ? 1 : output_dims[0];
    uint32_t counterLast = dim == 1 ? output_dims[0] : output_dims[1];

    if constexpr(isBcastSrc0 && isBcastSrc1) {
        if constexpr (sizeof(T) != 8) {
            uint16_t repeat = static_cast<uint16_t>(CeilDivision(counterLast, GetVecLen() / sizeof(T)));
            WhereNormal2DVecImpl<true, true, T>(dstLocal, 
                maskLocal, 
                src0Local, 
                src1Local, 
                dstStride, 
                maskStride,
                srcStride, 
                repeat, 
                counterFirst, 
                counterLast, 
                scalar0,
                scalar1);
        } else {
            uint16_t repeat = static_cast<uint16_t>(CeilDivision(counterLast, 2 * GetVecLen() / sizeof(T)));
            WhereNormal2DVecImpl<true, true, T, MicroAPI::RegTraitNumTwo>(dstLocal, 
                maskLocal, 
                src0Local, 
                src1Local, 
                dstStride, 
                maskStride,
                srcStride, 
                repeat, 
                counterFirst, 
                counterLast, 
                scalar0,
                scalar1);
        }
    } else if constexpr(isBcastSrc0 && !isBcastSrc1) {
        if constexpr (sizeof(T) != 8) {
            uint16_t repeat = static_cast<uint16_t>(CeilDivision(counterLast, GetVecLen() / sizeof(T)));
            WhereNormal2DVecImpl<true, false, T>(dstLocal, 
                maskLocal, 
                src0Local, 
                src1Local, 
                dstStride, 
                maskStride,
                srcStride, 
                repeat, 
                counterFirst, 
                counterLast, 
                scalar0,
                scalar1);
        } else {
            uint16_t repeat = static_cast<uint16_t>(CeilDivision(counterLast, 2 * GetVecLen() / sizeof(T)));
            WhereNormal2DVecImpl<true, false, T, MicroAPI::RegTraitNumTwo>(dstLocal, 
                maskLocal, 
                src0Local, 
                src1Local, 
                dstStride, 
                maskStride,
                srcStride, 
                repeat, 
                counterFirst, 
                counterLast, 
                scalar0,
                scalar1);
        }
    } else if constexpr(!isBcastSrc0 && isBcastSrc1) {
        if constexpr (sizeof(T) != 8) {
            uint16_t repeat = static_cast<uint16_t>(CeilDivision(counterLast, GetVecLen() / sizeof(T)));
            WhereNormal2DVecImpl<false, true, T>(dstLocal, 
                maskLocal, 
                src0Local, 
                src1Local, 
                dstStride, 
                maskStride,
                srcStride, 
                repeat, 
                counterFirst, 
                counterLast, 
                scalar0,
                scalar1);
        } else {
            uint16_t repeat = static_cast<uint16_t>(CeilDivision(counterLast, 2 * GetVecLen() / sizeof(T)));
            WhereNormal2DVecImpl<false, true, T, MicroAPI::RegTraitNumTwo>(dstLocal, 
                maskLocal, 
                src0Local, 
                src1Local, 
                dstStride, 
                maskStride,
                srcStride, 
                repeat, 
                counterFirst, 
                counterLast, 
                scalar0,
                scalar1);
        }
    } else {
        if constexpr (sizeof(T) != 8) {
            uint16_t repeat = static_cast<uint16_t>(CeilDivision(counterLast, GetVecLen() / sizeof(T)));
            WhereNormal2DVecImpl<false, false, T>(dstLocal, 
                maskLocal, 
                src0Local, 
                src1Local, 
                dstStride, 
                maskStride,
                srcStride, 
                repeat, 
                counterFirst, 
                counterLast, 
                scalar0,
                scalar1);
        } else {
            uint16_t repeat = static_cast<uint16_t>(CeilDivision(counterLast, 2 * GetVecLen() / sizeof(T)));
            WhereNormal2DVecImpl<false, false, T, MicroAPI::RegTraitNumTwo>(dstLocal, 
                maskLocal, 
                src0Local, 
                src1Local, 
                dstStride, 
                maskStride,
                srcStride, 
                repeat, 
                counterFirst, 
                counterLast, 
                scalar0,
                scalar1);
        }
    }
}
#endif  //__ASCENDC_API_REGBASE_WHERE_V2_H__
