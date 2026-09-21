/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCENDC_API_TRANSPOSE_H__
#define __ASCENDC_API_TRANSPOSE_H__

namespace codegen {

struct ConfusionTranspose3DTiling {
  uint32_t height;
  uint32_t width;
  uint32_t channel;
  uint32_t highBlock;
  uint32_t stride;
  uint32_t blockSize;
  uint32_t repeat;
  uint32_t firstAxisAlign;
  uint32_t firstAxisRem;
  uint32_t secondAxisAlign;
  uint32_t secondAxisRem;
};

struct ConfusionTranspose4DTiling {
  uint32_t height;
  uint32_t width;
  uint32_t channel;
  uint32_t batch;
  uint32_t highBlock;
  uint32_t stride;
  uint32_t blockSize;
  uint32_t repeat;
  uint32_t firstAxisAlign;
  uint32_t firstAxisRem;
  uint32_t secondAxisAlign;
  uint32_t secondAxisRem;
};

struct ConfusionTransposeLastTiling {
  uint32_t height;
  uint32_t width;
  uint32_t highBlock;
  uint32_t stride;
  uint32_t blockSize;
  uint32_t repeat;
  uint32_t firstAxisAlign;
  uint32_t firstAxisRem;
  uint32_t secondAxisAlign;
  uint32_t secondAxisRem;
};

struct ConfusionTransposeNLast3DTiling {
  uint32_t blockLen;
  uint32_t blockCount;
  uint32_t srcStride;
  uint32_t dstStride;
  uint32_t thirdDimSrcStride;
  uint32_t thirdDimDstStride;
  uint32_t thirdDimCnt;
};

struct ConfusionTransposeNLast4DTiling {
  uint32_t blockLen;
  uint32_t blockCount;
  uint32_t srcStride;
  uint32_t dstStride;
  uint32_t thirdDimSrcStride;
  uint32_t thirdDimDstStride;
  uint32_t thirdDimCnt;
  uint32_t fourthDimSrcStride;
  uint32_t fourthDimDstStride;
  uint32_t fourthDimCnt;
};

template <typename T>
__aicore__ inline void Transpose10ConfigMatrixA(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor, const ConfusionTransposeLastTiling &tiling) { 
  uint64_t dstLocalList[NCHW_CONV_ADDR_LIST_SIZE];
  uint64_t srcLocalList[NCHW_CONV_ADDR_LIST_SIZE];
  AscendC::TransDataTo5HDParams transParams;
  uint32_t loopIdx, i;
  transParams.repeatTimes = tiling.repeat;
  transParams.srcRepStride = tiling.repeat > 1 ? 1 : 0;
  transParams.dstRepStride = tiling.repeat > 1 ? tiling.stride : 0;
  for (loopIdx = 0; loopIdx < tiling.highBlock; loopIdx++) {
    if constexpr (sizeof(T) == sizeof(half)) {
      for (i = 0; i < NCHW_CONV_ADDR_LIST_SIZE; i++) {
        dstLocalList[i] = reinterpret_cast<uint64_t>(dstTensor[loopIdx * BLOCK_CUBE + tiling.firstAxisAlign * i].GetPhyAddr());
        srcLocalList[i] = reinterpret_cast<uint64_t>(srcTensor[loopIdx * BLOCK_CUBE * tiling.secondAxisAlign + tiling.secondAxisAlign * i].GetPhyAddr());
      }
      TransDataTo5HD<T>(dstLocalList, srcLocalList, transParams);
    } else if constexpr (sizeof(T) == sizeof(float)) {
      for (i = 0; i < NCHW_CONV_ADDR_LIST_SIZE; i = i + 2) {
        dstLocalList[i] = reinterpret_cast<uint64_t>(dstTensor[loopIdx * BLOCK_CUBE + tiling.firstAxisAlign * (i / 2)].GetPhyAddr());
        dstLocalList[i + 1] = reinterpret_cast<uint64_t>(dstTensor[loopIdx * BLOCK_CUBE + tiling.firstAxisAlign * (i / 2) + tiling.blockSize].GetPhyAddr());
      }
      for (i = 0; i < NCHW_CONV_ADDR_LIST_SIZE; i++) {
        srcLocalList[i] = reinterpret_cast<uint64_t>(srcTensor[loopIdx * BLOCK_CUBE * tiling.secondAxisAlign + tiling.secondAxisAlign * i].GetPhyAddr());
      }
      TransDataTo5HD<T>(dstLocalList, srcLocalList, transParams);
    }
  }
}

template <typename T>
__aicore__ inline void Transpose10ConfigMatrixB(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor, const ConfusionTransposeLastTiling &tiling) { 
  uint64_t dstLocalList[NCHW_CONV_ADDR_LIST_SIZE];
  uint64_t srcLocalList[NCHW_CONV_ADDR_LIST_SIZE];
  AscendC::TransDataTo5HDParams transParams;
  uint32_t i;
  transParams.repeatTimes = tiling.repeat;
  transParams.srcRepStride = tiling.repeat > 1 ? 1 : 0;
  transParams.dstRepStride = tiling.repeat > 1 ? tiling.stride : 0;
  if constexpr (sizeof(T) == sizeof(half)) {
    for (i = 0; i < tiling.firstAxisRem; i++) {
      dstLocalList[i] = reinterpret_cast<uint64_t>(dstTensor[tiling.highBlock * BLOCK_CUBE + tiling.firstAxisAlign * i].GetPhyAddr());
      srcLocalList[i] = reinterpret_cast<uint64_t>(srcTensor[tiling.highBlock * BLOCK_CUBE * tiling.secondAxisAlign + tiling.secondAxisAlign * i].GetPhyAddr());
    }
    for (; i < NCHW_CONV_ADDR_LIST_SIZE; i++) {
      dstLocalList[i] = reinterpret_cast<uint64_t>(dstTensor[tiling.highBlock * BLOCK_CUBE + tiling.firstAxisAlign * i].GetPhyAddr());
      srcLocalList[i] = reinterpret_cast<uint64_t>(srcTensor[tiling.highBlock * BLOCK_CUBE * tiling.secondAxisAlign + tiling.secondAxisAlign * (tiling.firstAxisRem - 1)].GetPhyAddr());
    }
    TransDataTo5HD<T>(dstLocalList, srcLocalList, transParams);
  } else if constexpr (sizeof(T) == sizeof(float)) {
    for (i = 0; i < NCHW_CONV_ADDR_LIST_SIZE; i = i + 2) {
      dstLocalList[i] = reinterpret_cast<uint64_t>(dstTensor[tiling.highBlock * BLOCK_CUBE + tiling.firstAxisAlign * (i / 2)].GetPhyAddr());
      dstLocalList[i + 1] = reinterpret_cast<uint64_t>(dstTensor[tiling.highBlock * BLOCK_CUBE + tiling.firstAxisAlign * (i / 2) + tiling.blockSize].GetPhyAddr());
    }
    for (i = 0; i < tiling.firstAxisRem; i++) {
      srcLocalList[i] = reinterpret_cast<uint64_t>(srcTensor[tiling.highBlock * BLOCK_CUBE * tiling.secondAxisAlign + tiling.secondAxisAlign * i].GetPhyAddr());
    }
    for (; i < NCHW_CONV_ADDR_LIST_SIZE; i++) {
      srcLocalList[i] = reinterpret_cast<uint64_t>(srcTensor[tiling.highBlock * BLOCK_CUBE * tiling.secondAxisAlign + tiling.secondAxisAlign * (tiling.firstAxisRem - 1)].GetPhyAddr());
    }
    TransDataTo5HD<T>(dstLocalList, srcLocalList, transParams);
  }
}

template <typename T>
__aicore__ inline void Transpose10ConfigMatrixC(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor, const ConfusionTransposeLastTiling &tiling, const LocalTensor<uint8_t> &tmpbuf) {
  uint64_t dstLocalList[NCHW_CONV_ADDR_LIST_SIZE];
  uint64_t srcLocalList[NCHW_CONV_ADDR_LIST_SIZE];
  AscendC::TransDataTo5HDParams transParams;
  const LocalTensor<T> tmpTensor = tmpbuf.ReinterpretCast<T>();
  const uint64_t dstAddrOffset = tiling.repeat * tiling.blockSize * tiling.firstAxisAlign; // 矩阵 C 目的地址相对于起始地址的偏移
  const uint64_t srcAddrOffset = tiling.repeat * tiling.blockSize; // 矩阵 C 源地址相对于起始地址的偏移
  uint32_t loopIdx, i;
  transParams.repeatTimes = 1;
  transParams.srcRepStride = 0;
  transParams.dstRepStride = 0;
  for (loopIdx = 0; loopIdx < tiling.highBlock; loopIdx++) {
    if constexpr (sizeof(T) == sizeof(half)) {
      for (i = 0; i < tiling.secondAxisRem; i++) {
        dstLocalList[i] = reinterpret_cast<uint64_t>(dstTensor[dstAddrOffset + loopIdx * BLOCK_CUBE + tiling.firstAxisAlign * i].GetPhyAddr());
        srcLocalList[i] = reinterpret_cast<uint64_t>(srcTensor[srcAddrOffset + loopIdx * BLOCK_CUBE * tiling.secondAxisAlign + tiling.secondAxisAlign * i].GetPhyAddr());
      }
      for (; i < NCHW_CONV_ADDR_LIST_SIZE; i++) {
        dstLocalList[i] = reinterpret_cast<uint64_t>(tmpTensor[(i - tiling.secondAxisRem) * tiling.blockSize].GetPhyAddr());
        srcLocalList[i] = reinterpret_cast<uint64_t>(srcTensor[srcAddrOffset + loopIdx * BLOCK_CUBE * tiling.secondAxisAlign + tiling.secondAxisAlign * i].GetPhyAddr());
      }
      TransDataTo5HD<T>(dstLocalList, srcLocalList, transParams);
    } else if constexpr (sizeof(T) == sizeof(float)) { 
      for (i = 0; i < tiling.secondAxisRem * 2; i = i + 2) {
        dstLocalList[i] = reinterpret_cast<uint64_t>(dstTensor[dstAddrOffset + loopIdx * BLOCK_CUBE + tiling.firstAxisAlign * (i / 2)].GetPhyAddr());
        dstLocalList[i + 1] = reinterpret_cast<uint64_t>(dstTensor[dstAddrOffset + loopIdx * BLOCK_CUBE + tiling.firstAxisAlign * (i / 2) + tiling.blockSize].GetPhyAddr());
      }
      for (; i < NCHW_CONV_ADDR_LIST_SIZE; i = i + 2) {
        dstLocalList[i] = reinterpret_cast<uint64_t>(tmpTensor[(i - tiling.secondAxisRem * 2) * tiling.blockSize].GetPhyAddr());
        dstLocalList[i + 1] = reinterpret_cast<uint64_t>(tmpTensor[(i - tiling.secondAxisRem * 2 + 1) * tiling.blockSize].GetPhyAddr());
      }
      for (i = 0; i < NCHW_CONV_ADDR_LIST_SIZE; i++) {
        srcLocalList[i] = reinterpret_cast<uint64_t>(srcTensor[srcAddrOffset + loopIdx * BLOCK_CUBE * tiling.secondAxisAlign + tiling.secondAxisAlign * i].GetPhyAddr());
      }
      TransDataTo5HD<T>(dstLocalList, srcLocalList, transParams);
    }
  }
}

template <typename T>
__aicore__ inline void Transpose10ConfigMatrixD(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor, const ConfusionTransposeLastTiling &tiling, const LocalTensor<uint8_t> &tmpbuf) {
  uint64_t dstLocalList[NCHW_CONV_ADDR_LIST_SIZE];
  uint64_t srcLocalList[NCHW_CONV_ADDR_LIST_SIZE];
  AscendC::TransDataTo5HDParams transParams;
  const LocalTensor<T> tmpTensor = tmpbuf.ReinterpretCast<T>();
  const uint64_t dstAddrOffset = tiling.repeat * tiling.blockSize * tiling.firstAxisAlign; // 矩阵 C 目的地址相对于起始地址的偏移
  const uint64_t srcAddrOffset = tiling.repeat * tiling.blockSize; // 矩阵 C 源地址相对于起始地址的偏移
  uint32_t i;
  transParams.repeatTimes = 1;
  transParams.srcRepStride = 0;
  transParams.dstRepStride = 0;
  if constexpr (sizeof(T) == sizeof(half)) {
    for (i = 0; i < tiling.secondAxisRem; i++) {
      dstLocalList[i] = reinterpret_cast<uint64_t>(dstTensor[dstAddrOffset + tiling.highBlock * BLOCK_CUBE + tiling.firstAxisAlign * i].GetPhyAddr());
    }
    for (; i < NCHW_CONV_ADDR_LIST_SIZE; i++) {
      dstLocalList[i] = reinterpret_cast<uint64_t>(tmpTensor[(i - tiling.secondAxisRem) * tiling.blockSize].GetPhyAddr());
    }
    for (i = 0; i < tiling.firstAxisRem; i++) {
      srcLocalList[i] = reinterpret_cast<uint64_t>(srcTensor[srcAddrOffset + tiling.highBlock * BLOCK_CUBE * tiling.secondAxisAlign + tiling.secondAxisAlign * i].GetPhyAddr());
    }
    for (; i < NCHW_CONV_ADDR_LIST_SIZE; i++) {
      srcLocalList[i] = reinterpret_cast<uint64_t>(srcTensor[srcAddrOffset + tiling.highBlock * BLOCK_CUBE * tiling.secondAxisAlign + tiling.secondAxisAlign * (tiling.firstAxisRem - 1)].GetPhyAddr());
    }
    TransDataTo5HD<T>(dstLocalList, srcLocalList, transParams);
  } else if constexpr (sizeof(T) == sizeof(float)) { 
    for (i = 0; i < tiling.secondAxisRem * 2; i = i + 2) {
      dstLocalList[i] = reinterpret_cast<uint64_t>(dstTensor[dstAddrOffset + tiling.highBlock * BLOCK_CUBE + tiling.firstAxisAlign * (i / 2)].GetPhyAddr());
      dstLocalList[i + 1] = reinterpret_cast<uint64_t>(dstTensor[dstAddrOffset + tiling.highBlock * BLOCK_CUBE + tiling.firstAxisAlign * (i / 2) + tiling.blockSize].GetPhyAddr());
    }
    for (; i < NCHW_CONV_ADDR_LIST_SIZE; i = i + 2) {
      dstLocalList[i] = reinterpret_cast<uint64_t>(tmpTensor[(i - tiling.secondAxisRem * 2) * tiling.blockSize].GetPhyAddr());
      dstLocalList[i + 1] = reinterpret_cast<uint64_t>(tmpTensor[(i - tiling.secondAxisRem * 2 + 1) * tiling.blockSize].GetPhyAddr());
    }
    for (i = 0; i < tiling.firstAxisRem; i++) {
      srcLocalList[i] = reinterpret_cast<uint64_t>(srcTensor[srcAddrOffset + tiling.highBlock * BLOCK_CUBE * tiling.secondAxisAlign + tiling.secondAxisAlign * i].GetPhyAddr());
    }
    for (; i < NCHW_CONV_ADDR_LIST_SIZE; i++) {
      srcLocalList[i] = reinterpret_cast<uint64_t>(srcTensor[srcAddrOffset + tiling.highBlock * BLOCK_CUBE * tiling.secondAxisAlign + tiling.secondAxisAlign * (tiling.firstAxisRem - 1)].GetPhyAddr());
    }
    TransDataTo5HD<T>(dstLocalList, srcLocalList, transParams);
  }
}

/*
FP16 尾轴非对齐场景，切分为如下ABCD四个矩阵分别进行转置
____________           ______________
|        | |           |        |   |
|    A   |C| transpose |    A'  | B'|
|        | |  ------>  |        |   |
|________|_|           |________|___|
|    B   |D|           |____C'__|_D'|
|________|_|        

以转置前矩阵[37,35]为例，切分为A[32,32], B[5,32], C[32,3]， D[5,3]分别进行转置
*/
template <typename T>
__aicore__ inline void ConfusionTranspose10Compute(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor, const LocalTensor<uint8_t> &tmpbuf,
                                                   const ConfusionTransposeLastTiling &tiling) { 
  if (tiling.highBlock > 0) {
    if (tiling.repeat > 0) {
      Transpose10ConfigMatrixA(dstTensor, srcTensor, tiling);
    }
    if (tiling.secondAxisRem > 0) {
      Transpose10ConfigMatrixC(dstTensor, srcTensor, tiling, tmpbuf);
    }
  }

  if (tiling.firstAxisRem > 0) {
    if (tiling.repeat > 0) {
      Transpose10ConfigMatrixB(dstTensor, srcTensor, tiling);
    }
    if (tiling.secondAxisRem > 0) {
      Transpose10ConfigMatrixD(dstTensor, srcTensor, tiling, tmpbuf);
    }
  }
}

/* 3维非尾轴转置场景，尾轴较小时，需要进行UB重排，UB内使用DataCopy重排 */
template <typename T>
__aicore__ inline void ConfusionTransposeNLast3DCompute(const LocalTensor<T> &dstTensor,
                                                        const LocalTensor<T> &srcTensor,
                                                        const ConfusionTransposeNLast3DTiling &tiling) {
  AscendC::DataCopyParams dataCopyParams;

  dataCopyParams.blockCount = tiling.blockCount;
  dataCopyParams.blockLen = tiling.blockLen;
  dataCopyParams.srcStride = tiling.srcStride;
  dataCopyParams.dstStride = tiling.dstStride;

  for (int32_t i = 0; i < tiling.thirdDimCnt; i++) {
    AscendC::DataCopy(dstTensor[i * tiling.thirdDimDstStride], srcTensor[i * tiling.thirdDimSrcStride], dataCopyParams);
  }
}

/* 4维非尾轴转置场景，尾轴较小时，需要进行UB重拍，UB内使用DataCopy重排*/
template <typename T>
__aicore__ inline void ConfusionTransposeNLast4DCompute(const LocalTensor<T> &dstTensor,
                                                        const LocalTensor<T> &srcTensor,
                                                        const ConfusionTransposeNLast4DTiling &tiling) {
  AscendC::DataCopyParams dataCopyParams;

  dataCopyParams.blockCount = tiling.blockCount;
  dataCopyParams.blockLen = tiling.blockLen;
  dataCopyParams.srcStride = tiling.srcStride;
  dataCopyParams.dstStride = tiling.dstStride;

  for (int32_t i = 0; i < tiling.fourthDimCnt; i++) {
    for (int32_t j = 0; j < tiling.thirdDimCnt; j++) {
      AscendC::DataCopy(dstTensor[i * tiling.fourthDimDstStride + j * tiling.thirdDimDstStride],
                        srcTensor[i * tiling.fourthDimSrcStride + j * tiling.thirdDimSrcStride], dataCopyParams);
    }
  }
}

template <typename T>
__aicore__ inline void Transpose021ConfigMatrixA(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor, const ConfusionTranspose3DTiling &tiling) { 
  uint64_t dstLocalList[NCHW_CONV_ADDR_LIST_SIZE];
  uint64_t srcLocalList[NCHW_CONV_ADDR_LIST_SIZE];
  AscendC::TransDataTo5HDParams transParams;
  uint32_t outerMostLoopIdx, loopIdx, i;
  uint64_t srcMostOuterOffset, dstMostOuterOffset;
  transParams.repeatTimes = tiling.repeat;
  transParams.srcRepStride = tiling.repeat > 1 ? 1 : 0;
  transParams.dstRepStride = tiling.repeat > 1 ? tiling.stride : 0;
  for (outerMostLoopIdx = 0; outerMostLoopIdx < tiling.channel; outerMostLoopIdx++) {
    srcMostOuterOffset = outerMostLoopIdx * tiling.height * tiling.secondAxisAlign;
    dstMostOuterOffset = outerMostLoopIdx  * tiling.width * tiling.firstAxisAlign;
    for (loopIdx = 0; loopIdx < tiling.highBlock; loopIdx++) {
      if constexpr (sizeof(T) == sizeof(half)) {
        for (i = 0; i < NCHW_CONV_ADDR_LIST_SIZE; i++) {
          dstLocalList[i] = reinterpret_cast<uint64_t>(dstTensor[dstMostOuterOffset + loopIdx * BLOCK_CUBE + tiling.firstAxisAlign * i].GetPhyAddr());
          srcLocalList[i] = reinterpret_cast<uint64_t>(srcTensor[srcMostOuterOffset + loopIdx * BLOCK_CUBE * tiling.secondAxisAlign + tiling.secondAxisAlign * i].GetPhyAddr());
        }
        TransDataTo5HD<T>(dstLocalList, srcLocalList, transParams);
      } else if constexpr (sizeof(T) == sizeof(float)) {
        for (i = 0; i < NCHW_CONV_ADDR_LIST_SIZE; i = i + 2) {
          dstLocalList[i] = reinterpret_cast<uint64_t>(dstTensor[dstMostOuterOffset + loopIdx * BLOCK_CUBE + tiling.firstAxisAlign * (i / 2)].GetPhyAddr());
          dstLocalList[i + 1] = reinterpret_cast<uint64_t>(dstTensor[dstMostOuterOffset + loopIdx * BLOCK_CUBE + tiling.firstAxisAlign * (i / 2) + tiling.blockSize].GetPhyAddr());
        }
        for (i = 0; i < NCHW_CONV_ADDR_LIST_SIZE; i++) {
          srcLocalList[i] = reinterpret_cast<uint64_t>(srcTensor[srcMostOuterOffset + loopIdx * BLOCK_CUBE * tiling.secondAxisAlign + tiling.secondAxisAlign * i].GetPhyAddr());
        }
        TransDataTo5HD<T>(dstLocalList, srcLocalList, transParams);
      }
    }
  }
}

template <typename T>
__aicore__ inline void Transpose021ConfigMatrixB(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor, const ConfusionTranspose3DTiling &tiling) { 
  uint64_t dstLocalList[NCHW_CONV_ADDR_LIST_SIZE];
  uint64_t srcLocalList[NCHW_CONV_ADDR_LIST_SIZE];
  AscendC::TransDataTo5HDParams transParams;
  uint32_t outerMostLoopIdx, i;
  uint64_t srcMostOuterOffset, dstMostOuterOffset;
  transParams.repeatTimes = tiling.repeat;
  transParams.srcRepStride = tiling.repeat > 1 ? 1 : 0;
  transParams.dstRepStride = tiling.repeat > 1 ? tiling.stride : 0;
  for (outerMostLoopIdx = 0; outerMostLoopIdx < tiling.channel; outerMostLoopIdx++) {
    srcMostOuterOffset = outerMostLoopIdx * tiling.height * tiling.secondAxisAlign;
    dstMostOuterOffset = outerMostLoopIdx  * tiling.width * tiling.firstAxisAlign;
    if constexpr (sizeof(T) == sizeof(half)) {
      for (i = 0; i < tiling.firstAxisRem; i++) {
        dstLocalList[i] = reinterpret_cast<uint64_t>(dstTensor[dstMostOuterOffset + tiling.highBlock * BLOCK_CUBE + tiling.firstAxisAlign * i].GetPhyAddr());
        srcLocalList[i] = reinterpret_cast<uint64_t>(srcTensor[srcMostOuterOffset + tiling.highBlock * BLOCK_CUBE * tiling.secondAxisAlign + tiling.secondAxisAlign * i].GetPhyAddr());
      }
      for (; i < NCHW_CONV_ADDR_LIST_SIZE; i++) {
        dstLocalList[i] = reinterpret_cast<uint64_t>(dstTensor[dstMostOuterOffset + tiling.highBlock * BLOCK_CUBE + tiling.firstAxisAlign * i].GetPhyAddr());
        srcLocalList[i] = reinterpret_cast<uint64_t>(srcTensor[srcMostOuterOffset + tiling.highBlock * BLOCK_CUBE * tiling.secondAxisAlign + tiling.secondAxisAlign * (tiling.firstAxisRem - 1)].GetPhyAddr());
      }
      TransDataTo5HD<T>(dstLocalList, srcLocalList, transParams);
    } else if constexpr (sizeof(T) == sizeof(float)) {
      for (i = 0; i < NCHW_CONV_ADDR_LIST_SIZE; i = i + 2) {
        dstLocalList[i] = reinterpret_cast<uint64_t>(dstTensor[dstMostOuterOffset + tiling.highBlock * BLOCK_CUBE + tiling.firstAxisAlign * (i / 2)].GetPhyAddr());
        dstLocalList[i + 1] = reinterpret_cast<uint64_t>(dstTensor[dstMostOuterOffset + tiling.highBlock * BLOCK_CUBE + tiling.firstAxisAlign * (i / 2) + tiling.blockSize].GetPhyAddr());
      }
      for (i = 0; i < tiling.firstAxisRem; i++) {
        srcLocalList[i] = reinterpret_cast<uint64_t>(srcTensor[srcMostOuterOffset + tiling.highBlock * BLOCK_CUBE * tiling.secondAxisAlign + tiling.secondAxisAlign * i].GetPhyAddr());
      }
      for (; i < NCHW_CONV_ADDR_LIST_SIZE; i++) {
        srcLocalList[i] = reinterpret_cast<uint64_t>(srcTensor[srcMostOuterOffset + tiling.highBlock * BLOCK_CUBE * tiling.secondAxisAlign + tiling.secondAxisAlign * (tiling.firstAxisRem - 1)].GetPhyAddr());
      }
      TransDataTo5HD<T>(dstLocalList, srcLocalList, transParams);
    }
  }
}

template <typename T>
__aicore__ inline void Transpose021ConfigMatrixC(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor, const ConfusionTranspose3DTiling &tiling, const LocalTensor<uint8_t> &tmpbuf) {
  uint64_t dstLocalList[NCHW_CONV_ADDR_LIST_SIZE];
  uint64_t srcLocalList[NCHW_CONV_ADDR_LIST_SIZE];
  AscendC::TransDataTo5HDParams transParams;
  const LocalTensor<T> tmpTensor = tmpbuf.ReinterpretCast<T>();
  const uint64_t dstAddrOffset = tiling.repeat * tiling.blockSize * tiling.firstAxisAlign; // 矩阵 C 目的地址相对于起始地址的偏移
  const uint64_t srcAddrOffset = tiling.repeat * tiling.blockSize; // 矩阵 C 源地址相对于起始地址的偏移
  uint32_t outerMostLoopIdx, loopIdx, i;
  uint64_t srcMostOuterOffset, dstMostOuterOffset;
  transParams.repeatTimes = 1;
  transParams.srcRepStride = 0;
  transParams.dstRepStride = 0;
  for (outerMostLoopIdx = 0; outerMostLoopIdx < tiling.channel; outerMostLoopIdx++) {
    srcMostOuterOffset = outerMostLoopIdx * tiling.height * tiling.secondAxisAlign;
    dstMostOuterOffset = outerMostLoopIdx  * tiling.width * tiling.firstAxisAlign;
    for (loopIdx = 0; loopIdx < tiling.highBlock; loopIdx++) {
      if constexpr (sizeof(T) == sizeof(half)) {
        for (i = 0; i < tiling.secondAxisRem; i++) {
          dstLocalList[i] = reinterpret_cast<uint64_t>(dstTensor[dstMostOuterOffset + dstAddrOffset + loopIdx * BLOCK_CUBE + tiling.firstAxisAlign * i].GetPhyAddr());
          srcLocalList[i] = reinterpret_cast<uint64_t>(srcTensor[srcMostOuterOffset + srcAddrOffset + loopIdx * BLOCK_CUBE * tiling.secondAxisAlign + tiling.secondAxisAlign * i].GetPhyAddr());
        }
        for (; i < NCHW_CONV_ADDR_LIST_SIZE; i++) {
          dstLocalList[i] = reinterpret_cast<uint64_t>(tmpTensor[(i - tiling.secondAxisRem) * tiling.blockSize].GetPhyAddr());
          srcLocalList[i] = reinterpret_cast<uint64_t>(srcTensor[srcMostOuterOffset + srcAddrOffset + loopIdx * BLOCK_CUBE * tiling.secondAxisAlign + tiling.secondAxisAlign * i].GetPhyAddr());
        }
        TransDataTo5HD<T>(dstLocalList, srcLocalList, transParams);
      } else if constexpr (sizeof(T) == sizeof(float)) {
        for (i = 0; i < tiling.secondAxisRem * 2; i = i + 2) {
          dstLocalList[i] = reinterpret_cast<uint64_t>(dstTensor[dstMostOuterOffset + dstAddrOffset + loopIdx * BLOCK_CUBE + tiling.firstAxisAlign * (i / 2)].GetPhyAddr());
          dstLocalList[i + 1] = reinterpret_cast<uint64_t>(dstTensor[dstMostOuterOffset + dstAddrOffset + loopIdx * BLOCK_CUBE + tiling.firstAxisAlign * (i / 2) + tiling.blockSize].GetPhyAddr());
        }
        for (; i < NCHW_CONV_ADDR_LIST_SIZE; i = i + 2) {
          dstLocalList[i] = reinterpret_cast<uint64_t>(tmpTensor[(i - tiling.secondAxisRem * 2) * tiling.blockSize].GetPhyAddr());
          dstLocalList[i + 1] = reinterpret_cast<uint64_t>(tmpTensor[(i - tiling.secondAxisRem * 2 + 1) * tiling.blockSize].GetPhyAddr());
        }
        for (i = 0; i < NCHW_CONV_ADDR_LIST_SIZE; i++) {
          srcLocalList[i] = reinterpret_cast<uint64_t>(srcTensor[srcMostOuterOffset + srcAddrOffset + loopIdx * BLOCK_CUBE * tiling.secondAxisAlign + tiling.secondAxisAlign * i].GetPhyAddr());
        }
        TransDataTo5HD<T>(dstLocalList, srcLocalList, transParams);
      }
    }
  }
}

template <typename T>
__aicore__ inline void Transpose021ConfigMatrixD(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor, const ConfusionTranspose3DTiling &tiling, const LocalTensor<uint8_t> &tmpbuf) {
  uint64_t dstLocalList[NCHW_CONV_ADDR_LIST_SIZE];
  uint64_t srcLocalList[NCHW_CONV_ADDR_LIST_SIZE];
  AscendC::TransDataTo5HDParams transParams;
  const LocalTensor<T> tmpTensor = tmpbuf.ReinterpretCast<T>();
  const uint64_t dstAddrOffset = tiling.repeat * tiling.blockSize * tiling.firstAxisAlign; // 矩阵 C 目的地址相对于起始地址的偏移
  const uint64_t srcAddrOffset = tiling.repeat * tiling.blockSize; // 矩阵 C 源地址相对于起始地址的偏移
  uint32_t outerMostLoopIdx, i;
  uint64_t srcMostOuterOffset, dstMostOuterOffset;
  transParams.repeatTimes = 1;
  transParams.srcRepStride = 0;
  transParams.dstRepStride = 0;
  for (outerMostLoopIdx = 0; outerMostLoopIdx < tiling.channel; outerMostLoopIdx++) {
    srcMostOuterOffset = outerMostLoopIdx * tiling.height * tiling.secondAxisAlign;
    dstMostOuterOffset = outerMostLoopIdx  * tiling.width * tiling.firstAxisAlign;
    if constexpr (sizeof(T) == sizeof(half)) {
      for (i = 0; i < tiling.secondAxisRem; i++) {
        dstLocalList[i] = reinterpret_cast<uint64_t>(dstTensor[dstMostOuterOffset + dstAddrOffset + tiling.highBlock * BLOCK_CUBE + tiling.firstAxisAlign * i].GetPhyAddr());
      }
      for (; i < NCHW_CONV_ADDR_LIST_SIZE; i++) {
        dstLocalList[i] = reinterpret_cast<uint64_t>(tmpTensor[(i - tiling.secondAxisRem) * tiling.blockSize].GetPhyAddr());
      }
      for (i = 0; i < tiling.firstAxisRem; i++) {
        srcLocalList[i] = reinterpret_cast<uint64_t>(srcTensor[srcMostOuterOffset + srcAddrOffset + tiling.highBlock * BLOCK_CUBE * tiling.secondAxisAlign + tiling.secondAxisAlign * i].GetPhyAddr());
      }
      for (; i < NCHW_CONV_ADDR_LIST_SIZE; i++) {
        srcLocalList[i] = reinterpret_cast<uint64_t>(srcTensor[srcMostOuterOffset + srcAddrOffset + tiling.highBlock * BLOCK_CUBE * tiling.secondAxisAlign + tiling.secondAxisAlign * (tiling.firstAxisRem - 1)].GetPhyAddr());
      }
      TransDataTo5HD<T>(dstLocalList, srcLocalList, transParams);
    } else if constexpr (sizeof(T) == sizeof(float)) { 
      for (i = 0; i < tiling.secondAxisRem * 2; i = i + 2) {
        dstLocalList[i] = reinterpret_cast<uint64_t>(dstTensor[dstMostOuterOffset + dstAddrOffset + tiling.highBlock * BLOCK_CUBE + tiling.firstAxisAlign * (i / 2)].GetPhyAddr());
        dstLocalList[i + 1] = reinterpret_cast<uint64_t>(dstTensor[dstMostOuterOffset + dstAddrOffset + tiling.highBlock * BLOCK_CUBE + tiling.firstAxisAlign * (i / 2) + tiling.blockSize].GetPhyAddr());
      }
      for (; i < NCHW_CONV_ADDR_LIST_SIZE; i = i + 2) {
        dstLocalList[i] = reinterpret_cast<uint64_t>(tmpTensor[(i - tiling.secondAxisRem * 2) * tiling.blockSize].GetPhyAddr());
        dstLocalList[i + 1] = reinterpret_cast<uint64_t>(tmpTensor[(i - tiling.secondAxisRem * 2 + 1) * tiling.blockSize].GetPhyAddr());
      }
      for (i = 0; i < tiling.firstAxisRem; i++) {
        srcLocalList[i] = reinterpret_cast<uint64_t>(srcTensor[srcMostOuterOffset + srcAddrOffset + tiling.highBlock * BLOCK_CUBE * tiling.secondAxisAlign + tiling.secondAxisAlign * i].GetPhyAddr());
      }
      for (; i < NCHW_CONV_ADDR_LIST_SIZE; i++) {
        srcLocalList[i] = reinterpret_cast<uint64_t>(srcTensor[srcMostOuterOffset + srcAddrOffset + tiling.highBlock * BLOCK_CUBE * tiling.secondAxisAlign + tiling.secondAxisAlign * (tiling.firstAxisRem - 1)].GetPhyAddr());
      }
      TransDataTo5HD<T>(dstLocalList, srcLocalList, transParams);
    }
  }
}

/* 3维尾轴转置场景，需要进行UB重拍，UB内使用TransDataTo5HD重排   021 */
template <typename T>
__aicore__ inline void ConfusionTranspose021Compute(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor, const LocalTensor<uint8_t> &tmpbuf,
                                                   const ConfusionTranspose3DTiling &tiling) { 
  if (tiling.highBlock > 0) {
    if (tiling.repeat > 0) {
      Transpose021ConfigMatrixA(dstTensor, srcTensor, tiling);
    }
    if (tiling.secondAxisRem > 0) {
      Transpose021ConfigMatrixC(dstTensor, srcTensor, tiling, tmpbuf);
    }
  }

  if (tiling.firstAxisRem > 0) {
    if (tiling.repeat > 0) {
      Transpose021ConfigMatrixB(dstTensor, srcTensor, tiling);
    }
    if (tiling.secondAxisRem > 0) {
      Transpose021ConfigMatrixD(dstTensor, srcTensor, tiling, tmpbuf);
    }
  }
}

template <typename T>
__aicore__ inline void Transpose210ConfigMatrixA(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor, const ConfusionTranspose3DTiling &tiling) { 
  uint64_t dstLocalList[NCHW_CONV_ADDR_LIST_SIZE];
  uint64_t srcLocalList[NCHW_CONV_ADDR_LIST_SIZE];
  AscendC::TransDataTo5HDParams transParams;
  uint32_t outerLoopIdx, loopIdx, i;
  uint64_t srcOuterOffset, dstOuterOffset;
  transParams.repeatTimes = tiling.repeat;
  transParams.srcRepStride = tiling.repeat > 1 ? 1 : 0;
  transParams.dstRepStride = tiling.repeat > 1 ? tiling.stride : 0;
  for (outerLoopIdx = 0; outerLoopIdx < tiling.height; outerLoopIdx++) {
    srcOuterOffset = outerLoopIdx * tiling.secondAxisAlign;
    dstOuterOffset = outerLoopIdx * tiling.firstAxisAlign;
    for (loopIdx = 0; loopIdx < tiling.highBlock; loopIdx++) {
      if constexpr (sizeof(T) == sizeof(half)) {
        for (i = 0; i < NCHW_CONV_ADDR_LIST_SIZE; i++) {
          dstLocalList[i] = reinterpret_cast<uint64_t>(dstTensor[dstOuterOffset + loopIdx * BLOCK_CUBE + tiling.firstAxisAlign * tiling.height * i].GetPhyAddr());
          srcLocalList[i] = reinterpret_cast<uint64_t>(srcTensor[srcOuterOffset + loopIdx * BLOCK_CUBE * tiling.secondAxisAlign * tiling.height + tiling.secondAxisAlign * tiling.height * i].GetPhyAddr());
        }
        TransDataTo5HD<T>(dstLocalList, srcLocalList, transParams);
      } else if constexpr (sizeof(T) == sizeof(float)) {
        for (i = 0; i < NCHW_CONV_ADDR_LIST_SIZE; i = i + 2) {
          dstLocalList[i] = reinterpret_cast<uint64_t>(dstTensor[dstOuterOffset + loopIdx * BLOCK_CUBE + tiling.firstAxisAlign * tiling.height * (i / 2)].GetPhyAddr());
          dstLocalList[i + 1] = reinterpret_cast<uint64_t>(dstTensor[dstOuterOffset + loopIdx * BLOCK_CUBE + tiling.firstAxisAlign * tiling.height * (i / 2) + tiling.blockSize].GetPhyAddr());
        }
        for (i = 0; i < NCHW_CONV_ADDR_LIST_SIZE; i++) {
          srcLocalList[i] = reinterpret_cast<uint64_t>(srcTensor[srcOuterOffset + loopIdx * BLOCK_CUBE * tiling.secondAxisAlign * tiling.height + tiling.secondAxisAlign * tiling.height * i].GetPhyAddr());
        }
        TransDataTo5HD<T>(dstLocalList, srcLocalList, transParams);
      }
    }
  }
}

template <typename T>
__aicore__ inline void Transpose210ConfigMatrixB(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor, const ConfusionTranspose3DTiling &tiling) { 
  uint64_t dstLocalList[NCHW_CONV_ADDR_LIST_SIZE];
  uint64_t srcLocalList[NCHW_CONV_ADDR_LIST_SIZE];
  AscendC::TransDataTo5HDParams transParams;
  uint32_t outerLoopIdx, i;
  uint64_t srcOuterOffset, dstOuterOffset;
  transParams.repeatTimes = tiling.repeat;
  transParams.srcRepStride = tiling.repeat > 1 ? 1 : 0;
  transParams.dstRepStride = tiling.repeat > 1 ? tiling.stride : 0;
  for (outerLoopIdx = 0; outerLoopIdx < tiling.height; outerLoopIdx++) {
    srcOuterOffset = outerLoopIdx * tiling.secondAxisAlign;
    dstOuterOffset = outerLoopIdx * tiling.firstAxisAlign;
    if constexpr (sizeof(T) == sizeof(half)) {
      for (i = 0; i < tiling.firstAxisRem; i++) {
        dstLocalList[i] = reinterpret_cast<uint64_t>(dstTensor[dstOuterOffset + tiling.highBlock * BLOCK_CUBE + tiling.firstAxisAlign * tiling.height * i].GetPhyAddr());
        srcLocalList[i] = reinterpret_cast<uint64_t>(srcTensor[srcOuterOffset + tiling.highBlock * BLOCK_CUBE * tiling.secondAxisAlign * tiling.height + tiling.secondAxisAlign * tiling.height * i].GetPhyAddr());
      }
      for (; i < NCHW_CONV_ADDR_LIST_SIZE; i++) {
        dstLocalList[i] = reinterpret_cast<uint64_t>(dstTensor[dstOuterOffset + tiling.highBlock * BLOCK_CUBE + tiling.firstAxisAlign * tiling.height * i].GetPhyAddr());
        srcLocalList[i] = reinterpret_cast<uint64_t>(srcTensor[srcOuterOffset + tiling.highBlock * BLOCK_CUBE * tiling.secondAxisAlign * tiling.height + tiling.secondAxisAlign * tiling.height * (tiling.firstAxisRem - 1)].GetPhyAddr());
      }
      TransDataTo5HD<T>(dstLocalList, srcLocalList, transParams);
    } else if constexpr (sizeof(T) == sizeof(float)) { 
      for (i = 0; i < NCHW_CONV_ADDR_LIST_SIZE; i = i + 2) {
        dstLocalList[i] = reinterpret_cast<uint64_t>(dstTensor[dstOuterOffset + tiling.highBlock * BLOCK_CUBE + tiling.firstAxisAlign * tiling.height * (i / 2)].GetPhyAddr());
        dstLocalList[i + 1] = reinterpret_cast<uint64_t>(dstTensor[dstOuterOffset + tiling.highBlock * BLOCK_CUBE + tiling.firstAxisAlign * tiling.height * (i / 2) + tiling.blockSize].GetPhyAddr());
      }
      for (i = 0; i < tiling.firstAxisRem; i++) {
        srcLocalList[i] = reinterpret_cast<uint64_t>(srcTensor[srcOuterOffset + tiling.highBlock * BLOCK_CUBE * tiling.secondAxisAlign * tiling.height + tiling.secondAxisAlign * tiling.height * i].GetPhyAddr());
      }
      for (; i < NCHW_CONV_ADDR_LIST_SIZE; i++) {
        srcLocalList[i] = reinterpret_cast<uint64_t>(srcTensor[srcOuterOffset + tiling.highBlock * BLOCK_CUBE * tiling.secondAxisAlign * tiling.height + tiling.secondAxisAlign * tiling.height * (tiling.firstAxisRem - 1)].GetPhyAddr());
      }
      TransDataTo5HD<T>(dstLocalList, srcLocalList, transParams);
    }
  }
}

template <typename T>
__aicore__ inline void Transpose210ConfigMatrixC(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor, const ConfusionTranspose3DTiling &tiling, const LocalTensor<uint8_t> &tmpbuf) {
  uint64_t dstLocalList[NCHW_CONV_ADDR_LIST_SIZE];
  uint64_t srcLocalList[NCHW_CONV_ADDR_LIST_SIZE];
  AscendC::TransDataTo5HDParams transParams;
  const LocalTensor<T> tmpTensor = tmpbuf.ReinterpretCast<T>();
  const uint64_t dstAddrOffset = tiling.repeat * tiling.blockSize * tiling.firstAxisAlign * tiling.height; // 矩阵 C 目的地址相对于起始地址的偏移
  const uint64_t srcAddrOffset = tiling.repeat * tiling.blockSize; // 矩阵 C 源地址相对于起始地址的偏移
  uint32_t outerLoopIdx, loopIdx, i;
  uint64_t srcOuterOffset, dstOuterOffset;
  transParams.repeatTimes = 1;
  transParams.srcRepStride = 0;
  transParams.dstRepStride = 0;
  for (outerLoopIdx = 0; outerLoopIdx < tiling.height; outerLoopIdx++) {
    srcOuterOffset = outerLoopIdx * tiling.secondAxisAlign;
    dstOuterOffset = outerLoopIdx * tiling.firstAxisAlign;
    for (loopIdx = 0; loopIdx < tiling.highBlock; loopIdx++) {
      if constexpr (sizeof(T) == sizeof(half)) {
        for (i = 0; i < tiling.secondAxisRem; i++) {
          dstLocalList[i] = reinterpret_cast<uint64_t>(dstTensor[dstOuterOffset + dstAddrOffset + loopIdx * BLOCK_CUBE + tiling.firstAxisAlign * tiling.height * i].GetPhyAddr());
          srcLocalList[i] = reinterpret_cast<uint64_t>(srcTensor[srcOuterOffset + srcAddrOffset + loopIdx * BLOCK_CUBE * tiling.secondAxisAlign * tiling.height + tiling.secondAxisAlign * tiling.height * i].GetPhyAddr());
        }
        for (; i < NCHW_CONV_ADDR_LIST_SIZE; i++) {
          dstLocalList[i] = reinterpret_cast<uint64_t>(tmpTensor[(i - tiling.secondAxisRem) * tiling.blockSize].GetPhyAddr());
          srcLocalList[i] = reinterpret_cast<uint64_t>(srcTensor[srcOuterOffset + srcAddrOffset + loopIdx * BLOCK_CUBE * tiling.secondAxisAlign * tiling.height + tiling.secondAxisAlign * tiling.height * i].GetPhyAddr());
        }
        TransDataTo5HD<T>(dstLocalList, srcLocalList, transParams);
      } else if constexpr (sizeof(T) == sizeof(float)) { 
        for (i = 0; i < tiling.secondAxisRem * 2; i = i + 2) {
          dstLocalList[i] = reinterpret_cast<uint64_t>(dstTensor[dstOuterOffset + dstAddrOffset + loopIdx * BLOCK_CUBE + tiling.firstAxisAlign * tiling.height * (i / 2)].GetPhyAddr());
          dstLocalList[i + 1] = reinterpret_cast<uint64_t>(dstTensor[dstOuterOffset + dstAddrOffset + loopIdx * BLOCK_CUBE + tiling.firstAxisAlign * tiling.height * (i / 2) + tiling.blockSize].GetPhyAddr());
        }
        for (; i < NCHW_CONV_ADDR_LIST_SIZE; i = i + 2) {
          dstLocalList[i] = reinterpret_cast<uint64_t>(tmpTensor[(i - tiling.secondAxisRem * 2) * tiling.blockSize].GetPhyAddr());
          dstLocalList[i + 1] = reinterpret_cast<uint64_t>(tmpTensor[(i - tiling.secondAxisRem * 2 + 1) * tiling.blockSize].GetPhyAddr());
        }
        for (i = 0; i < NCHW_CONV_ADDR_LIST_SIZE; i++) {
          srcLocalList[i] = reinterpret_cast<uint64_t>(srcTensor[srcOuterOffset + srcAddrOffset + loopIdx * BLOCK_CUBE * tiling.secondAxisAlign * tiling.height + tiling.secondAxisAlign * tiling.height * i].GetPhyAddr());
        }
        TransDataTo5HD<T>(dstLocalList, srcLocalList, transParams);
      }
    }
  }
}

template <typename T>
__aicore__ inline void Transpose210ConfigMatrixD(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor, const ConfusionTranspose3DTiling &tiling, const LocalTensor<uint8_t> &tmpbuf) {
  uint64_t dstLocalList[NCHW_CONV_ADDR_LIST_SIZE];
  uint64_t srcLocalList[NCHW_CONV_ADDR_LIST_SIZE];
  AscendC::TransDataTo5HDParams transParams;
  const LocalTensor<T> tmpTensor = tmpbuf.ReinterpretCast<T>();
  const uint64_t dstAddrOffset = tiling.repeat * tiling.blockSize * tiling.firstAxisAlign * tiling.height; // 矩阵 C 目的地址相对于起始地址的偏移
  const uint64_t srcAddrOffset = tiling.repeat * tiling.blockSize; // 矩阵 C 源地址相对于起始地址的偏移
  uint32_t outerLoopIdx, i;
  uint64_t srcOuterOffset, dstOuterOffset;
  transParams.repeatTimes = 1;
  transParams.srcRepStride = 0;
  transParams.dstRepStride = 0;
  for (outerLoopIdx = 0; outerLoopIdx < tiling.height; outerLoopIdx++) {
    srcOuterOffset = outerLoopIdx * tiling.secondAxisAlign;
    dstOuterOffset = outerLoopIdx * tiling.firstAxisAlign;
    if constexpr (sizeof(T) == sizeof(half)) {
      for (i = 0; i < tiling.secondAxisRem; i++) {
        dstLocalList[i] = reinterpret_cast<uint64_t>(dstTensor[dstOuterOffset + dstAddrOffset + tiling.highBlock * BLOCK_CUBE + tiling.firstAxisAlign * tiling.height * i].GetPhyAddr());
      }
      for (; i < NCHW_CONV_ADDR_LIST_SIZE; i++) {
        dstLocalList[i] = reinterpret_cast<uint64_t>(tmpTensor[(i - tiling.secondAxisRem) * tiling.blockSize].GetPhyAddr());
      }
      for (i = 0; i < tiling.firstAxisRem; i++) {
        srcLocalList[i] = reinterpret_cast<uint64_t>(srcTensor[srcOuterOffset + srcAddrOffset + tiling.highBlock * BLOCK_CUBE * tiling.secondAxisAlign * tiling.height + tiling.secondAxisAlign * tiling.height * i].GetPhyAddr());
      }
      for (; i < NCHW_CONV_ADDR_LIST_SIZE; i++) {
        srcLocalList[i] = reinterpret_cast<uint64_t>(srcTensor[srcOuterOffset + srcAddrOffset + tiling.highBlock * BLOCK_CUBE * tiling.secondAxisAlign * tiling.height + tiling.secondAxisAlign * tiling.height * (tiling.firstAxisRem - 1)].GetPhyAddr());
      }
      TransDataTo5HD<T>(dstLocalList, srcLocalList, transParams);
    } else if constexpr (sizeof(T) == sizeof(float)) { 
      for (i = 0; i < tiling.secondAxisRem * 2; i = i + 2) {
        dstLocalList[i] = reinterpret_cast<uint64_t>(dstTensor[dstOuterOffset + dstAddrOffset + tiling.highBlock * BLOCK_CUBE + tiling.firstAxisAlign * tiling.height * (i / 2)].GetPhyAddr());
        dstLocalList[i + 1] = reinterpret_cast<uint64_t>(dstTensor[dstOuterOffset + dstAddrOffset + tiling.highBlock * BLOCK_CUBE + tiling.firstAxisAlign * tiling.height * (i / 2) + tiling.blockSize].GetPhyAddr());
      }
      for (; i < NCHW_CONV_ADDR_LIST_SIZE; i = i + 2) {
        dstLocalList[i] = reinterpret_cast<uint64_t>(tmpTensor[(i - tiling.secondAxisRem * 2) * tiling.blockSize].GetPhyAddr());
        dstLocalList[i + 1] = reinterpret_cast<uint64_t>(tmpTensor[(i - tiling.secondAxisRem * 2 + 1) * tiling.blockSize].GetPhyAddr());
      }
      for (i = 0; i < tiling.firstAxisRem; i++) {
        srcLocalList[i] = reinterpret_cast<uint64_t>(srcTensor[srcOuterOffset + srcAddrOffset + tiling.highBlock * BLOCK_CUBE * tiling.secondAxisAlign * tiling.height + tiling.secondAxisAlign * tiling.height * i].GetPhyAddr());
      }
      for (; i < NCHW_CONV_ADDR_LIST_SIZE; i++) {
        srcLocalList[i] = reinterpret_cast<uint64_t>(srcTensor[srcOuterOffset + srcAddrOffset + tiling.highBlock * BLOCK_CUBE * tiling.secondAxisAlign * tiling.height + tiling.secondAxisAlign * tiling.height * (tiling.firstAxisRem - 1)].GetPhyAddr());
      }
      TransDataTo5HD<T>(dstLocalList, srcLocalList, transParams);
    }
  } 
}

/* 3维尾轴转置场景，需要进行UB重拍，UB内使用TransDataTo5HD重排   210 */
template <typename T>
__aicore__ inline void ConfusionTranspose210Compute(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor, const LocalTensor<uint8_t> &tmpbuf,
                                                   const ConfusionTranspose3DTiling &tiling) { 
  uint64_t dstLocalList[NCHW_CONV_ADDR_LIST_SIZE];
  uint64_t srcLocalList[NCHW_CONV_ADDR_LIST_SIZE];
  AscendC::TransDataTo5HDParams transParams;
  const LocalTensor<T> tmpTensor = tmpbuf.ReinterpretCast<T>();
  const uint64_t dstAddrOffset = tiling.repeat * tiling.blockSize * tiling.firstAxisAlign * tiling.height; // 矩阵 C 目的地址相对于起始地址的偏移
  const uint64_t srcAddrOffset = tiling.repeat * tiling.blockSize; // 矩阵 C 源地址相对于起始地址的偏移
  uint32_t outerLoopIdx, loopIdx, i;
  uint64_t srcOuterOffset, dstOuterOffset;

  if (tiling.highBlock > 0) {
    if (tiling.repeat > 0) {
      Transpose210ConfigMatrixA(dstTensor, srcTensor, tiling);
    }
    if (tiling.secondAxisRem > 0) {
      Transpose210ConfigMatrixC(dstTensor, srcTensor, tiling, tmpbuf);
    }
  }

  if (tiling.firstAxisRem > 0) {
    if (tiling.repeat > 0) {
      Transpose210ConfigMatrixB(dstTensor, srcTensor, tiling);
    }
    if (tiling.secondAxisRem > 0) {
      Transpose210ConfigMatrixD(dstTensor, srcTensor, tiling, tmpbuf);
    }
  }
}

template <typename T>
__aicore__ inline void Transpose0321ConfigMatrixAInnerProc(const LocalTensor<T> &dstTensor,
                                                           const LocalTensor<T> &srcTensor,
                                                           const ConfusionTranspose4DTiling &tiling,
                                                           const TransDataTo5HDParams &transParams,
                                                           uint64_t srcMostOuterOffset, uint64_t srcOuterOffset,
                                                           uint64_t dstMostOuterOffset, uint64_t dstOuterOffset) {
  uint64_t dstLocalList[NCHW_CONV_ADDR_LIST_SIZE];
  uint64_t srcLocalList[NCHW_CONV_ADDR_LIST_SIZE];
  uint32_t loopIdx, i;
  for (loopIdx = 0; loopIdx < tiling.highBlock; loopIdx++) {
    if constexpr (sizeof(T) == sizeof(half)) {
      for (i = 0; i < NCHW_CONV_ADDR_LIST_SIZE; i++) {
        dstLocalList[i] = reinterpret_cast<uint64_t>(dstTensor[dstMostOuterOffset + dstOuterOffset + loopIdx * BLOCK_CUBE + tiling.firstAxisAlign * tiling.height * i].GetPhyAddr());
        srcLocalList[i] = reinterpret_cast<uint64_t>(srcTensor[srcMostOuterOffset + srcOuterOffset + loopIdx * BLOCK_CUBE * tiling.secondAxisAlign * tiling.height + tiling.secondAxisAlign * tiling.height * i].GetPhyAddr());
      }
      TransDataTo5HD<T>(dstLocalList, srcLocalList, transParams);
    } else if constexpr (sizeof(T) == sizeof(float)) {
      for (i = 0; i < NCHW_CONV_ADDR_LIST_SIZE; i = i + 2) {
        dstLocalList[i] = reinterpret_cast<uint64_t>(dstTensor[dstMostOuterOffset + dstOuterOffset + loopIdx * BLOCK_CUBE + tiling.firstAxisAlign * tiling.height * (i / 2)].GetPhyAddr());
        dstLocalList[i + 1] = reinterpret_cast<uint64_t>(dstTensor[dstMostOuterOffset + dstOuterOffset + loopIdx * BLOCK_CUBE + tiling.firstAxisAlign * tiling.height * (i / 2) + tiling.blockSize].GetPhyAddr());
      }
      for (i = 0; i < NCHW_CONV_ADDR_LIST_SIZE; i++) {
        srcLocalList[i] = reinterpret_cast<uint64_t>(srcTensor[srcMostOuterOffset + srcOuterOffset + loopIdx * BLOCK_CUBE * tiling.secondAxisAlign * tiling.height + tiling.secondAxisAlign * tiling.height * i].GetPhyAddr());
      }
      TransDataTo5HD<T>(dstLocalList, srcLocalList, transParams);
    }
  }
}

template <typename T>
__aicore__ inline void Transpose0321ConfigMatrixA(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor, const ConfusionTranspose4DTiling &tiling) { 
  AscendC::TransDataTo5HDParams transParams;
  uint32_t outerMostLoopIdx, outerLoopIdx;
  uint64_t srcMostOuterOffset, srcOuterOffset, dstMostOuterOffset, dstOuterOffset;
  transParams.repeatTimes = tiling.repeat;
  transParams.srcRepStride = tiling.repeat > 1 ? 1 : 0;
  transParams.dstRepStride = tiling.repeat > 1 ? tiling.stride : 0;
  for (outerMostLoopIdx = 0; outerMostLoopIdx < tiling.batch; outerMostLoopIdx++) {
    srcMostOuterOffset = outerMostLoopIdx * tiling.channel * tiling.height * tiling.secondAxisAlign;
    dstMostOuterOffset = outerMostLoopIdx  * tiling.width * tiling.height * tiling.firstAxisAlign;
    for (outerLoopIdx = 0; outerLoopIdx < tiling.height; outerLoopIdx++) {
      srcOuterOffset = outerLoopIdx * tiling.secondAxisAlign;
      dstOuterOffset = outerLoopIdx * tiling.firstAxisAlign;
      Transpose0321ConfigMatrixAInnerProc(dstTensor, srcTensor, tiling, transParams, srcMostOuterOffset, srcOuterOffset, dstMostOuterOffset, dstOuterOffset);
    }
  }
}

template <typename T>
__aicore__ inline void Transpose0321ConfigMatrixB(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor, const ConfusionTranspose4DTiling &tiling) { 
  uint64_t dstLocalList[NCHW_CONV_ADDR_LIST_SIZE];
  uint64_t srcLocalList[NCHW_CONV_ADDR_LIST_SIZE];
  AscendC::TransDataTo5HDParams transParams;
  uint32_t outerMostLoopIdx, outerLoopIdx, i;
  uint64_t srcMostOuterOffset, srcOuterOffset, dstMostOuterOffset, dstOuterOffset;
  transParams.repeatTimes = tiling.repeat;
  transParams.srcRepStride = tiling.repeat > 1 ? 1 : 0;
  transParams.dstRepStride = tiling.repeat > 1 ? tiling.stride : 0;
  for (outerMostLoopIdx = 0; outerMostLoopIdx < tiling.batch; outerMostLoopIdx++) {
    srcMostOuterOffset = outerMostLoopIdx * tiling.channel * tiling.height * tiling.secondAxisAlign;
    dstMostOuterOffset = outerMostLoopIdx  * tiling.width * tiling.height * tiling.firstAxisAlign;
    for (outerLoopIdx = 0; outerLoopIdx < tiling.height; outerLoopIdx++) {
      srcOuterOffset = outerLoopIdx * tiling.secondAxisAlign;
      dstOuterOffset = outerLoopIdx * tiling.firstAxisAlign;
      if constexpr (sizeof(T) == sizeof(half)) {
        for (i = 0; i < tiling.firstAxisRem; i++) {
          dstLocalList[i] = reinterpret_cast<uint64_t>(dstTensor[dstMostOuterOffset + dstOuterOffset + tiling.highBlock * BLOCK_CUBE + tiling.firstAxisAlign * tiling.height * i].GetPhyAddr());
          srcLocalList[i] = reinterpret_cast<uint64_t>(srcTensor[srcMostOuterOffset + srcOuterOffset + tiling.highBlock * BLOCK_CUBE * tiling.secondAxisAlign * tiling.height + tiling.secondAxisAlign * tiling.height * i].GetPhyAddr());
        }
        for (; i < NCHW_CONV_ADDR_LIST_SIZE; i++) {
          dstLocalList[i] = reinterpret_cast<uint64_t>(dstTensor[dstMostOuterOffset + dstOuterOffset + tiling.highBlock * BLOCK_CUBE + tiling.firstAxisAlign * tiling.height * i].GetPhyAddr());
          srcLocalList[i] = reinterpret_cast<uint64_t>(srcTensor[srcMostOuterOffset + srcOuterOffset + tiling.highBlock * BLOCK_CUBE * tiling.secondAxisAlign * tiling.height + tiling.secondAxisAlign * tiling.height * (tiling.firstAxisRem - 1)].GetPhyAddr());
        }
        TransDataTo5HD<T>(dstLocalList, srcLocalList, transParams);
      } else if constexpr (sizeof(T) == sizeof(float)) {
        for (i = 0; i < NCHW_CONV_ADDR_LIST_SIZE; i = i + 2) {
          dstLocalList[i] = reinterpret_cast<uint64_t>(dstTensor[dstMostOuterOffset + dstOuterOffset + tiling.highBlock * BLOCK_CUBE + tiling.firstAxisAlign * tiling.height * (i / 2)].GetPhyAddr());
          dstLocalList[i + 1] = reinterpret_cast<uint64_t>(dstTensor[dstMostOuterOffset + dstOuterOffset + tiling.highBlock * BLOCK_CUBE + tiling.firstAxisAlign * tiling.height * (i / 2) + tiling.blockSize].GetPhyAddr());
        }
        for (i = 0; i < tiling.firstAxisRem; i++) {
          srcLocalList[i] = reinterpret_cast<uint64_t>(srcTensor[srcMostOuterOffset + srcOuterOffset + tiling.highBlock * BLOCK_CUBE * tiling.secondAxisAlign * tiling.height + tiling.secondAxisAlign * tiling.height * i].GetPhyAddr());
        }
        for (; i < NCHW_CONV_ADDR_LIST_SIZE; i++) {
          srcLocalList[i] = reinterpret_cast<uint64_t>(srcTensor[srcMostOuterOffset + srcOuterOffset + tiling.highBlock * BLOCK_CUBE * tiling.secondAxisAlign * tiling.height + tiling.secondAxisAlign * tiling.height * (tiling.firstAxisRem - 1)].GetPhyAddr());
        }
        TransDataTo5HD<T>(dstLocalList, srcLocalList, transParams);
      }
    }
  }
}

template <typename T>
__aicore__ inline void Transpose0321ConfigMatrixCInnerProc(const LocalTensor<T> &dstTensor,
                                                           const LocalTensor<T> &srcTensor,
                                                           const ConfusionTranspose4DTiling &tiling,
                                                           const LocalTensor<T> tmpTensor,
                                                           const TransDataTo5HDParams &transParams,
                                                           uint64_t srcMostOuterOffset, uint64_t srcOuterOffset,
                                                           uint64_t dstMostOuterOffset, uint64_t dstOuterOffset) {
  uint64_t dstLocalList[NCHW_CONV_ADDR_LIST_SIZE];
  uint64_t srcLocalList[NCHW_CONV_ADDR_LIST_SIZE];
  const uint64_t dstAddrOffset = tiling.repeat * tiling.blockSize * tiling.firstAxisAlign * tiling.height; // 矩阵 C 目的地址相对于起始地址的偏移
  const uint64_t srcAddrOffset = tiling.repeat * tiling.blockSize; // 矩阵 C 源地址相对于起始地址的偏移
  uint32_t loopIdx, i;
  for (loopIdx = 0; loopIdx < tiling.highBlock; loopIdx++) {
    if constexpr (sizeof(T) == sizeof(half)) {
      for (i = 0; i < tiling.secondAxisRem; i++) {
        dstLocalList[i] = reinterpret_cast<uint64_t>(dstTensor[dstMostOuterOffset + dstOuterOffset + dstAddrOffset + loopIdx * BLOCK_CUBE + tiling.firstAxisAlign * tiling.height * i].GetPhyAddr());
        srcLocalList[i] = reinterpret_cast<uint64_t>(srcTensor[srcMostOuterOffset + srcOuterOffset + srcAddrOffset + loopIdx * BLOCK_CUBE * tiling.secondAxisAlign * tiling.height + tiling.secondAxisAlign * tiling.height * i].GetPhyAddr());
      }
      for (; i < NCHW_CONV_ADDR_LIST_SIZE; i++) {
        dstLocalList[i] = reinterpret_cast<uint64_t>(tmpTensor[(i - tiling.secondAxisRem) * tiling.blockSize].GetPhyAddr());
        srcLocalList[i] = reinterpret_cast<uint64_t>(srcTensor[srcMostOuterOffset + srcOuterOffset + srcAddrOffset + loopIdx * BLOCK_CUBE * tiling.secondAxisAlign * tiling.height + tiling.secondAxisAlign * tiling.height * i].GetPhyAddr());
      }
      TransDataTo5HD<T>(dstLocalList, srcLocalList, transParams);
    } else if constexpr (sizeof(T) == sizeof(float)) {
      for (i = 0; i < tiling.secondAxisRem * 2; i = i + 2) {
        dstLocalList[i] = reinterpret_cast<uint64_t>(dstTensor[dstMostOuterOffset + dstOuterOffset + dstAddrOffset + loopIdx * BLOCK_CUBE + tiling.firstAxisAlign * tiling.height * (i / 2)].GetPhyAddr());
        dstLocalList[i + 1] = reinterpret_cast<uint64_t>(dstTensor[dstMostOuterOffset + dstOuterOffset + dstAddrOffset + loopIdx * BLOCK_CUBE + tiling.firstAxisAlign * tiling.height * (i / 2) + tiling.blockSize].GetPhyAddr());
      }
      for (; i < NCHW_CONV_ADDR_LIST_SIZE; i = i + 2) {
        dstLocalList[i] = reinterpret_cast<uint64_t>(tmpTensor[(i - tiling.secondAxisRem * 2) * tiling.blockSize].GetPhyAddr());
        dstLocalList[i + 1] = reinterpret_cast<uint64_t>(tmpTensor[(i - tiling.secondAxisRem * 2 + 1) * tiling.blockSize].GetPhyAddr());
      }
      for (i = 0; i < NCHW_CONV_ADDR_LIST_SIZE; i++) {
        srcLocalList[i] = reinterpret_cast<uint64_t>(srcTensor[srcMostOuterOffset + srcOuterOffset + srcAddrOffset + loopIdx * BLOCK_CUBE * tiling.secondAxisAlign * tiling.height + tiling.secondAxisAlign * tiling.height * i].GetPhyAddr());
      }
      TransDataTo5HD<T>(dstLocalList, srcLocalList, transParams);
    }
  }
}

template <typename T>
__aicore__ inline void Transpose0321ConfigMatrixC(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor, const ConfusionTranspose4DTiling &tiling, const LocalTensor<uint8_t> &tmpbuf) {
  AscendC::TransDataTo5HDParams transParams;
  const LocalTensor<T> tmpTensor = tmpbuf.ReinterpretCast<T>();
  uint32_t outerMostLoopIdx, outerLoopIdx;
  uint64_t srcMostOuterOffset, srcOuterOffset, dstMostOuterOffset, dstOuterOffset;
  transParams.repeatTimes = 1;
  transParams.srcRepStride = 0;
  transParams.dstRepStride = 0;
  for (outerMostLoopIdx = 0; outerMostLoopIdx < tiling.batch; outerMostLoopIdx++) {
    srcMostOuterOffset = outerMostLoopIdx * tiling.channel * tiling.height * tiling.secondAxisAlign;
    dstMostOuterOffset = outerMostLoopIdx  * tiling.width * tiling.height * tiling.firstAxisAlign;
    for (outerLoopIdx = 0; outerLoopIdx < tiling.height; outerLoopIdx++) {
      srcOuterOffset = outerLoopIdx * tiling.secondAxisAlign;
      dstOuterOffset = outerLoopIdx * tiling.firstAxisAlign;
      Transpose0321ConfigMatrixCInnerProc(dstTensor, srcTensor, tiling, tmpTensor, transParams, srcMostOuterOffset, srcOuterOffset, dstMostOuterOffset, dstOuterOffset);
    }
  }
}

template <typename T>
__aicore__ inline void Transpose0321ConfigMatrixD(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor, const ConfusionTranspose4DTiling &tiling, const LocalTensor<uint8_t> &tmpbuf) {
  uint64_t dstLocalList[NCHW_CONV_ADDR_LIST_SIZE], srcLocalList[NCHW_CONV_ADDR_LIST_SIZE], srcMostOuterOffset, srcOuterOffset, dstMostOuterOffset, dstOuterOffset;
  AscendC::TransDataTo5HDParams transParams;
  const LocalTensor<T> tmpTensor = tmpbuf.ReinterpretCast<T>();
  const uint64_t dstAddrOffset = tiling.repeat * tiling.blockSize * tiling.firstAxisAlign * tiling.height; // 矩阵 C 目的地址相对于起始地址的偏移
  const uint64_t srcAddrOffset = tiling.repeat * tiling.blockSize; // 矩阵 C 源地址相对于起始地址的偏移
  uint32_t outerMostLoopIdx, outerLoopIdx, i;
  transParams.repeatTimes = 1;
  transParams.srcRepStride = 0;
  transParams.dstRepStride = 0;
  for (outerMostLoopIdx = 0; outerMostLoopIdx < tiling.batch; outerMostLoopIdx++) {
    srcMostOuterOffset = outerMostLoopIdx * tiling.channel * tiling.height * tiling.secondAxisAlign;
    dstMostOuterOffset = outerMostLoopIdx  * tiling.width * tiling.height * tiling.firstAxisAlign;
    for (outerLoopIdx = 0; outerLoopIdx < tiling.height; outerLoopIdx++) {
      srcOuterOffset = outerLoopIdx * tiling.secondAxisAlign;
      dstOuterOffset = outerLoopIdx * tiling.firstAxisAlign;
      if constexpr (sizeof(T) == sizeof(half)) {
        for (i = 0; i < tiling.secondAxisRem; i++) {
          dstLocalList[i] = reinterpret_cast<uint64_t>(dstTensor[dstMostOuterOffset + dstOuterOffset + dstAddrOffset + tiling.highBlock * BLOCK_CUBE + tiling.firstAxisAlign * tiling.height * i].GetPhyAddr());
        }
        for (; i < NCHW_CONV_ADDR_LIST_SIZE; i++) {
          dstLocalList[i] = reinterpret_cast<uint64_t>(tmpTensor[(i - tiling.secondAxisRem) * tiling.blockSize].GetPhyAddr());
        }
        for (i = 0; i < tiling.firstAxisRem; i++) {
          srcLocalList[i] = reinterpret_cast<uint64_t>(srcTensor[srcMostOuterOffset + srcOuterOffset + srcAddrOffset + tiling.highBlock * BLOCK_CUBE * tiling.secondAxisAlign * tiling.height + tiling.secondAxisAlign * tiling.height * i].GetPhyAddr());
        }
        for (; i < NCHW_CONV_ADDR_LIST_SIZE; i++) {
          srcLocalList[i] = reinterpret_cast<uint64_t>(srcTensor[srcMostOuterOffset + srcOuterOffset + srcAddrOffset + tiling.highBlock * BLOCK_CUBE * tiling.secondAxisAlign * tiling.height + tiling.secondAxisAlign * tiling.height * (tiling.firstAxisRem - 1)].GetPhyAddr());
        }
        TransDataTo5HD<T>(dstLocalList, srcLocalList, transParams);
      } else if constexpr (sizeof(T) == sizeof(float)) {
        for (i = 0; i < tiling.secondAxisRem * 2; i = i + 2) {
          dstLocalList[i] = reinterpret_cast<uint64_t>(dstTensor[dstMostOuterOffset + dstOuterOffset + dstAddrOffset + tiling.highBlock * BLOCK_CUBE + tiling.firstAxisAlign * tiling.height * (i / 2)].GetPhyAddr());
          dstLocalList[i + 1] = reinterpret_cast<uint64_t>(dstTensor[dstMostOuterOffset + dstOuterOffset + dstAddrOffset + tiling.highBlock * BLOCK_CUBE + tiling.firstAxisAlign * tiling.height * (i / 2) + tiling.blockSize].GetPhyAddr());
        }
        for (; i < NCHW_CONV_ADDR_LIST_SIZE; i = i + 2) {
          dstLocalList[i] = reinterpret_cast<uint64_t>(tmpTensor[(i - tiling.secondAxisRem * 2) * tiling.blockSize].GetPhyAddr());
          dstLocalList[i + 1] = reinterpret_cast<uint64_t>(tmpTensor[(i - tiling.secondAxisRem * 2 + 1) * tiling.blockSize].GetPhyAddr());
        }
        for (i = 0; i < tiling.firstAxisRem; i++) {
          srcLocalList[i] = reinterpret_cast<uint64_t>(srcTensor[srcMostOuterOffset + srcOuterOffset + srcAddrOffset + tiling.highBlock * BLOCK_CUBE * tiling.secondAxisAlign * tiling.height + tiling.secondAxisAlign * tiling.height * i].GetPhyAddr());
        }
        for (; i < NCHW_CONV_ADDR_LIST_SIZE; i++) {
          srcLocalList[i] = reinterpret_cast<uint64_t>(srcTensor[srcMostOuterOffset + srcOuterOffset + srcAddrOffset + tiling.highBlock * BLOCK_CUBE * tiling.secondAxisAlign * tiling.height + tiling.secondAxisAlign * tiling.height * (tiling.firstAxisRem - 1)].GetPhyAddr());
        }
        TransDataTo5HD<T>(dstLocalList, srcLocalList, transParams);
      }
    }
  }
}

/* 4维尾轴转置场景，需要进行UB重拍，UB内使用TransDataTo5HD重排   0321 */
template <typename T>
__aicore__ inline void ConfusionTranspose0321Compute(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor, const LocalTensor<uint8_t> &tmpbuf,
                                                   const ConfusionTranspose4DTiling &tiling) { 
  if (tiling.highBlock > 0) {
    if (tiling.repeat > 0) {
      Transpose0321ConfigMatrixA(dstTensor, srcTensor, tiling);
    }
    if (tiling.secondAxisRem > 0) {
      Transpose0321ConfigMatrixC(dstTensor, srcTensor, tiling, tmpbuf);
    }
  }

  if (tiling.firstAxisRem > 0) {
    if (tiling.repeat > 0) {
      Transpose0321ConfigMatrixB(dstTensor, srcTensor, tiling);
    }
    if (tiling.secondAxisRem > 0) {
      Transpose0321ConfigMatrixD(dstTensor, srcTensor, tiling, tmpbuf);
    }
  }
}

/* scene7：{ shape:[s0,s1], format:"ND"} -->{ shape:[s1, s0], format:"ND"} */
template <typename T>
__aicore__ inline void ConfusionTransposeNd2Nd10(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor, const LocalTensor<uint8_t> &tmpBuf,
                                                 ConfusionTransposeLastTiling &tiling) {
  ConfusionTranspose10Compute(dstTensor, srcTensor, tmpBuf, tiling);
}

/* scene8：{ shape:[s0,s1,s2], format:"ND"} -->{ shape:[s1, s0, s2], format:"ND"} */
template <typename T>
__aicore__ inline void ConfusionTransposeNd2Nd102(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor,
                                                  const ConfusionTransposeNLast3DTiling &tiling) {
  ConfusionTransposeNLast3DCompute(dstTensor, srcTensor, tiling);
}

/* scene9：{ shape:[s0,s1,s2,s3], format:"ND"} -->{ shape:[s0,s2,s1,s3], format:"ND"}*/
template <typename T>
__aicore__ inline void ConfusionTransposeNd2Nd0213(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor,
                                                   const ConfusionTransposeNLast4DTiling &tiling) {
  ConfusionTransposeNLast4DCompute(dstTensor, srcTensor, tiling);
}

/*scene10：{ shape:[s0,s1,s2,s3], format:"ND"} -->{ shape:[s2,s1,s0,s3], format:"ND"}*/
template <typename T>
__aicore__ inline void ConfusionTransposeNd2Nd2103(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor,
                                                   const ConfusionTransposeNLast4DTiling &tiling) {
  ConfusionTransposeNLast4DCompute(dstTensor, srcTensor, tiling);
}

/*scene11：{ shape:[s0,s1,s2], format:"ND"} -->{ shape:[s0, s2, s1], format:"ND"}*/
template <typename T>
__aicore__ inline void ConfusionTransposeNd2Nd021(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor, const LocalTensor<uint8_t> &tmpBuf,
                                                  const ConfusionTranspose3DTiling &tiling) {
  ConfusionTranspose021Compute(dstTensor, srcTensor, tmpBuf, tiling);
}

/*scene12：{ shape:[s0,s1,s2], format:"ND"} -->{ shape:[s2, s1, s0], format:"ND"}*/
template <typename T>
__aicore__ inline void ConfusionTransposeNd2Nd210(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor, const LocalTensor<uint8_t> &tmpBuf,
                                                  const ConfusionTranspose3DTiling &tiling) {
  ConfusionTranspose210Compute(dstTensor, srcTensor, tmpBuf, tiling);
}

/*scene13：{ shape:[s0,s1,s2,s3], format:"ND"} -->{ shape:[s0,s3,s2,s1], format:"ND"}*/
template <typename T>
__aicore__ inline void ConfusionTransposeNd2Nd0321(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor, const LocalTensor<uint8_t> &tmpBuf,
                                                   const ConfusionTranspose4DTiling &tiling) {
  ConfusionTranspose0321Compute(dstTensor, srcTensor, tmpBuf, tiling);
}

template <typename T>
__aicore__ inline void ConfusionTransposeImpl(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor,
                                              const LocalTensor<uint8_t> &sharedTmpBuffer,
                                              AutoFuseTransposeType transposeType,
                                              ConfusionTransposeTiling &tiling) {
  if (transposeType == AutoFuseTransposeType::TRANSPOSE_ND2ND_ONLY) {
    ConfusionTransposeNd2Nd10(dstTensor, srcTensor, sharedTmpBuffer, reinterpret_cast<ConfusionTransposeLastTiling &>(tiling));
  } else if (transposeType == AutoFuseTransposeType::TRANSPOSE_ND2ND_102) {
    ConfusionTransposeNd2Nd102(dstTensor, srcTensor, reinterpret_cast<ConfusionTransposeNLast3DTiling &>(tiling));
  } else if (transposeType == AutoFuseTransposeType::TRANSPOSE_ND2ND_0213) {
    ConfusionTransposeNd2Nd0213(dstTensor, srcTensor, reinterpret_cast<ConfusionTransposeNLast4DTiling &>(tiling));
  } else if (transposeType == AutoFuseTransposeType::TRANSPOSE_ND2ND_2103) {
    ConfusionTransposeNd2Nd2103(dstTensor, srcTensor, reinterpret_cast<ConfusionTransposeNLast4DTiling &>(tiling));
  } else if (transposeType == AutoFuseTransposeType::TRANSPOSE_ND2ND_021) {
    ConfusionTransposeNd2Nd021(dstTensor, srcTensor, sharedTmpBuffer, reinterpret_cast<ConfusionTranspose3DTiling &>(tiling));
  } else if (transposeType == AutoFuseTransposeType::TRANSPOSE_ND2ND_210) {
    ConfusionTransposeNd2Nd210(dstTensor, srcTensor, sharedTmpBuffer, reinterpret_cast<ConfusionTranspose3DTiling &>(tiling));
  } else if (transposeType == AutoFuseTransposeType::TRANSPOSE_ND2ND_0321) {
    ConfusionTransposeNd2Nd0321(dstTensor, srcTensor, sharedTmpBuffer, reinterpret_cast<ConfusionTranspose4DTiling &>(tiling));
  }
}

template <typename T>
__aicore__ inline void ConfusionTranspose(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor,
                                          const LocalTensor<uint8_t> &sharedTmpBuffer,
                                          AutoFuseTransposeType transposeType, ConfusionTransposeTiling &tiling) {
  ConfusionTransposeImpl<T>(dstTensor, srcTensor, sharedTmpBuffer, transposeType, tiling);
}

template <typename T>
__aicore__ inline void ConfusionTranspose(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor,
                                          AutoFuseTransposeType transposeType, ConfusionTransposeTiling &tiling) {
  LocalTensor<uint8_t> tmpBuffer;
  bool res = PopStackBuffer<uint8_t, TPosition::LCM>(tmpBuffer);
  ASCENDC_ASSERT(res, { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });

  ConfusionTransposeImpl<T>(dstTensor, srcTensor, tmpBuffer, transposeType, tiling);
}

}  // namespace codegen

#endif  // __ASCENDC_API_TRANSPOSE_H__