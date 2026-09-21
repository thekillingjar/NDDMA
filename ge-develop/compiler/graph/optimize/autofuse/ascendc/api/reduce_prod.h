/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCENDC_API_REDUCEPROD_H__
#define __ASCENDC_API_REDUCEPROD_H__

namespace AscendC {
template <class T, bool isAr, bool isReuseSource = false>
__aicore__ inline void ReduceProdExtend(const LocalTensor<T> &dst, const LocalTensor<T> &src,
                                        const LocalTensor<uint8_t> &tmp, const uint32_t srcShape[],
                                        bool srcInnerPad = false) {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, int32_t>, "Only support int32_t and float");

  ASCENDC_ASSERT((srcInnerPad), { KERNEL_LOG(KERNEL_ERROR, "Current srcInnerPad must be set to true!"); });
  uint32_t last = srcShape[1];
  uint32_t first = srcShape[0];
  constexpr uint32_t bytePerBlk = 32U;
  constexpr uint32_t bytePerRep = 256U;
  constexpr uint32_t elePerBlk = bytePerBlk / sizeof(T);
  constexpr uint32_t elePerRep = bytePerRep / sizeof(T);
  uint32_t padLast = (last + elePerBlk - 1) / elePerBlk * elePerBlk;

  BinaryRepeatParams defaultParam;
  UnaryRepeatParams defaultUnaryParam;

  LocalTensor<T> tmpDst = tmp.ReinterpretCast<T>();

  SetMaskCount();
  if constexpr (isAr) {
    uint32_t k = 0, lastCopy = last;
    while (lastCopy > 0) {
      k++;
      lastCopy >>= 1;
    }
    auto eventIdVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    auto eventIdSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    for (int j = 0; j < first; j++) {
      uint32_t splitK = 1 << (k - 1);
      uint32_t remain = last - splitK;

      SetVectorMask<T, MaskMode::COUNTER>(0, last);

      Adds<T, false>(tmpDst, src[j * padLast], static_cast<T>(0), MASK_PLACEHOLDER, 1, defaultUnaryParam);
      AscendC::PipeBarrier<PIPE_V>();
      if (last >= elePerBlk && remain != 0) {
        SetVectorMask<T, MaskMode::COUNTER>(0, remain);
        Mul<T, false>(tmpDst, tmpDst, src[j * padLast + splitK], MASK_PLACEHOLDER, 1, defaultParam);
        AscendC::PipeBarrier<PIPE_V>();
      }
      while (splitK > 8) {
        splitK >>= 1;
        SetVectorMask<T, MaskMode::COUNTER>(0, splitK);

        Mul<T, false>(tmpDst, tmpDst, tmpDst[splitK], MASK_PLACEHOLDER, 1, defaultParam);
        AscendC::PipeBarrier<PIPE_V>();
      }

      SetFlag<HardEvent::V_S>(eventIdVS);
      WaitFlag<HardEvent::V_S>(eventIdVS);

      T ret = T(1);
      for (int i = 0; i < last && i < elePerBlk; i++) {
        ret *= tmpDst.GetValue(i);
      }
      dst.SetValue(j, ret);

      SetFlag<HardEvent::S_V>(eventIdSV);
      WaitFlag<HardEvent::S_V>(eventIdSV);
    }
  } else {
    uint32_t k = 0, firstCopy = first;
    while (firstCopy > 0) {
      k++;
      firstCopy >>= 1;
    }
    if (padLast > elePerBlk * 255) {
      uint32_t splitK = 1 << (k - 1);
      uint32_t remain = first - splitK;

      SetVectorMask<T, MaskMode::COUNTER>(0, padLast * splitK);
      Duplicate<T, false>(tmpDst, static_cast<T>(1), MASK_PLACEHOLDER, 1, 1, 8);
      if (remain != 0) {
        SetVectorMask<T, MaskMode::COUNTER>(0, padLast * remain);
        Mul<T, false>(tmpDst, tmpDst, src[splitK * padLast], MASK_PLACEHOLDER, 1, defaultParam);
        AscendC::PipeBarrier<PIPE_V>();
      }

      SetVectorMask<T, MaskMode::COUNTER>(0, splitK * padLast);
      Mul<T, false>(tmpDst, tmpDst, src, MASK_PLACEHOLDER, 1, defaultParam);
      AscendC::PipeBarrier<PIPE_V>();
      while (splitK > 1) {
        splitK >>= 1;
        SetVectorMask<T, MaskMode::COUNTER>(0, splitK * padLast);
        Mul<T, false>(tmpDst, tmpDst, tmpDst[splitK * padLast], MASK_PLACEHOLDER, 1, defaultParam);
        AscendC::PipeBarrier<PIPE_V>();
      }

      SetVectorMask<T, MaskMode::COUNTER>(0, last);
      Adds<T, false>(dst, tmpDst, static_cast<T>(0), MASK_PLACEHOLDER, 1, defaultUnaryParam);
      AscendC::PipeBarrier<PIPE_V>();
    } else {
      BinaryRepeatParams binaryStridePerLast = {1, 1, 1, 8, 8, static_cast<uint8_t>(padLast / elePerBlk)};
      UnaryRepeatParams unaryStridePerLast = {1, 1, 8, static_cast<uint8_t>(padLast / elePerBlk)};
      for (int j = 0; j < last; j += elePerRep) {
        uint32_t splitK = 1 << (k - 1);
        uint32_t remain = first - splitK;
        uint32_t line = j + elePerRep < last ? elePerRep : last - j;

        SetVectorMask<T, MaskMode::COUNTER>(0, splitK * elePerRep);
        Adds<T, false>(tmpDst, src[j], static_cast<T>(0), MASK_PLACEHOLDER, 1, unaryStridePerLast);
        AscendC::PipeBarrier<PIPE_V>();

        if (remain != 0) {
          SetVectorMask<T, MaskMode::COUNTER>(0, remain * elePerRep);
          Mul<T, false>(tmpDst, tmpDst, src[j + splitK * padLast], MASK_PLACEHOLDER, 1, binaryStridePerLast);
          AscendC::PipeBarrier<PIPE_V>();
        }

        while (splitK > 1) {
          splitK >>= 1;
          SetVectorMask<T, MaskMode::COUNTER>(0, splitK * elePerRep);

          Mul<T, false>(tmpDst, tmpDst, tmpDst[splitK * elePerRep], MASK_PLACEHOLDER, 1, defaultParam);
          AscendC::PipeBarrier<PIPE_V>();
        }

        SetVectorMask<T, MaskMode::COUNTER>(0, line);
        Adds<T, false>(dst[j], tmpDst, static_cast<T>(0), MASK_PLACEHOLDER, 1, unaryStridePerLast);
        AscendC::PipeBarrier<PIPE_V>();
      }
    }
  }
  SetMaskNorm();
  ResetMask();
}
}  // namespace AscendC

#endif  // __ASCENDC_API_REDUCEPROD_H__