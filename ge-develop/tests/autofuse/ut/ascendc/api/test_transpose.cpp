/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "kernel_operator.h"
using namespace AscendC;
#include "utils.h"
#include "test_api_utils.h"
#include "transpose_base_type.h"
#include "transpose.h"

using namespace codegen;

inline uint32_t ALIGN_UP(uint32_t origin_size, uint32_t align_num) {
  return (0 == (origin_size & (align_num - 1))) ? origin_size  :  (origin_size + align_num - (origin_size & (align_num - 1)));
}


// scene7：srcShape[s0,s1] -> dstShape[s1,s0]
void GetConfusionTranspose10TilingInfo(const ShapeInfo srcShape, const uint32_t stackBufferSize,
                                       const uint32_t typeSize, ConfusionTransposeTiling &tiling) {
  uint32_t height = srcShape.originalShape[0];
  uint32_t width = srcShape.originalShape[1];

  uint32_t blockSize = ONE_BLK_SIZE / typeSize;
  uint32_t highBlock = height / BLOCK_CUBE;
  uint32_t repeat = width / blockSize;
  uint32_t firstAxisAlign = ALIGN_UP(height, BLOCK_CUBE);
  uint32_t firstAxisRem = height % BLOCK_CUBE;
  uint32_t secondAxisAlign = ALIGN_UP(width, 16); // float32和float16的尾轴都对齐到16
  uint32_t secondAxisRem = width % blockSize;
  uint32_t stride = firstAxisAlign;
  
  tiling.param0 = height;
  tiling.param1 = width;
  tiling.param2 = highBlock;
  tiling.param3 = stride;
  tiling.param4 = blockSize;
  tiling.param5 = repeat;
  tiling.param6 = firstAxisAlign;
  tiling.param7 = firstAxisRem;
  tiling.param8 = secondAxisAlign;
  tiling.param9 = secondAxisRem;
}

// scene8：srcShape[s0,s1,s2] -> dstShape[s1,s0,s2]
void GetConfusionTranspose102TilingInfo(const ShapeInfo srcShape, const uint32_t stackBufferSize,
                                        const uint32_t typeSize, ConfusionTransposeTiling &tiling) {
    /* 尾轴要求block对齐 */
    uint32_t blockSize = ONE_BLK_SIZE / typeSize;
    uint32_t lastDim = srcShape.originalShape[2];
    uint32_t lastDimAlign = ALIGN_UP(lastDim, 16);

  /* s2为单位搬移数据快大小，s1为blockCount，单位为block */
  uint32_t blockLen = lastDimAlign / blockSize;
  uint32_t blockCount = srcShape.originalShape[1];

  /* 输入连续，输出不连续，单位为block */
  uint32_t srcStride = 0;
  uint32_t dstStride = (srcShape.originalShape[0] * lastDimAlign - lastDimAlign) / blockSize;

  /* 输入连续，输出不连续，单位为数据个数 */
  uint32_t thirdDimSrcStride = srcShape.originalShape[1] * lastDimAlign;
  uint32_t thirdDimDstStride = lastDimAlign;
  uint32_t thirdDimCnt = srcShape.originalShape[0];

  tiling.param0 = blockLen;
  tiling.param1 = blockCount;
  tiling.param2 = srcStride;
  tiling.param3 = dstStride;
  tiling.param4 = thirdDimSrcStride;
  tiling.param5 = thirdDimDstStride;
  tiling.param6 = thirdDimCnt;
}

// scene9：srcShape[s0,s1,s2,s3] -> dstShape[s0,s2,s1,s3]
void GetConfusionTransposeND2ND0213TilingInfo(const ShapeInfo srcShape, const uint32_t stackBufferSize,
                                              const uint32_t typeSize, ConfusionTransposeTiling &tiling) {
  /* 尾轴要求block对齐 */
  uint32_t blockSize = ONE_BLK_SIZE / typeSize;
  uint32_t lastDim = srcShape.originalShape[3];
  uint32_t lastDimAlign = ALIGN_UP(lastDim, 16);

  /* s3为单位搬移数据快大小，s1为blockCount，单位为block */
  uint32_t blockLen = lastDimAlign / blockSize;
  uint32_t blockCount = srcShape.originalShape[2];

  /* 输入连续，输出不连续，单位为block */
  uint32_t srcStride = 0;
  uint32_t dstStride = (srcShape.originalShape[1] * lastDimAlign - lastDimAlign) / blockSize;

  /* 输入连续，输出不连续，单位为数据个数,内层循环 */
  uint32_t thirdDimSrcStride = srcShape.originalShape[2] * lastDimAlign;
  uint32_t thirdDimDstStride = lastDimAlign;
  uint32_t thirdDimCnt = srcShape.originalShape[1];

  /* 输入连续，输出不连续，单位为数据个数,外层循环 */
  uint32_t fourthDimSrcStride = srcShape.originalShape[1] * srcShape.originalShape[2] * lastDimAlign;
  uint32_t fourthDimDstStride = srcShape.originalShape[1] * srcShape.originalShape[2] * lastDimAlign;
  uint32_t fourthDimCnt = srcShape.originalShape[0];

  tiling.param0 = blockLen;
  tiling.param1 = blockCount;
  tiling.param2 = srcStride;
  tiling.param3 = dstStride;
  tiling.param4 = thirdDimSrcStride;
  tiling.param5 = thirdDimDstStride;
  tiling.param6 = thirdDimCnt;
  tiling.param7 = fourthDimSrcStride;
  tiling.param8 = fourthDimDstStride;
  tiling.param9 = fourthDimCnt;
}

// scene10：srcShape[s0,s1,s2,s3] -> dstShape[s2,s1,s0,s3]
void GetConfusionTranspose2103TilingInfo(const ShapeInfo srcShape, const uint32_t stackBufferSize,
                                         const uint32_t typeSize, ConfusionTransposeTiling &tiling) {
  /* 尾轴要求block对齐 */
  uint32_t blockSize = ONE_BLK_SIZE / typeSize;
  uint32_t lastDim = srcShape.originalShape[3];
  uint32_t lastDimAlign = ALIGN_UP(lastDim, 16);

  /* s3为单位搬移数据快大小，s1为blockCount，单位为block */
  uint32_t blockLen = lastDimAlign / blockSize;
  uint32_t blockCount = srcShape.originalShape[2];

  /* 输入连续，输出不连续，单位为block */
  uint32_t srcStride = 0;
  uint32_t dstStride =
      (srcShape.originalShape[0] * srcShape.originalShape[1] * lastDimAlign - lastDimAlign) /
      blockSize;

  /* 输入连续，输出不连续，单位为数据个数,内层循环 */
  uint32_t thirdDimSrcStride = srcShape.originalShape[2] * lastDimAlign;
  uint32_t thirdDimDstStride = srcShape.originalShape[0] * lastDimAlign;
  uint32_t thirdDimCnt = srcShape.originalShape[1];

  /* 输入连续，输出不连续，单位为数据个数,外层循环 */
  uint32_t fourthDimSrcStride = srcShape.originalShape[1] * srcShape.originalShape[2] * lastDimAlign;
  uint32_t fourthDimDstStride = lastDimAlign;
  uint32_t fourthDimCnt = srcShape.originalShape[0];

  tiling.param0 = blockLen;
  tiling.param1 = blockCount;
  tiling.param2 = srcStride;
  tiling.param3 = dstStride;
  tiling.param4 = thirdDimSrcStride;
  tiling.param5 = thirdDimDstStride;
  tiling.param6 = thirdDimCnt;
  tiling.param7 = fourthDimSrcStride;
  tiling.param8 = fourthDimDstStride;
  tiling.param9 = fourthDimCnt;
}

// scene11 shape[s0,s1,s2]->shape[s0,s2,s1]
void GetConfusionTranspose021TilingInfo(const ShapeInfo srcShape, const uint32_t stackBufferSize,
                                        const uint32_t typeSize, ConfusionTransposeTiling &tiling) {
  uint32_t channel = srcShape.originalShape[0];
  uint32_t height = srcShape.originalShape[1];
  uint32_t width = srcShape.originalShape[2];

  uint32_t blockSize = ONE_BLK_SIZE / typeSize;
  uint32_t highBlock = height / BLOCK_CUBE;
  uint32_t firstAxisAlign = ALIGN_UP(height, BLOCK_CUBE);
  uint32_t firstAxisRem = height % BLOCK_CUBE;
  uint32_t secondAxisAlign = ALIGN_UP(width, 16); // float32和float16的尾轴都对齐到16
  uint32_t secondAxisRem = width % blockSize;
  uint32_t stride = firstAxisAlign;
  uint32_t repeat = width / blockSize;

  tiling.param0 = height;
  tiling.param1 = width;
  tiling.param2 = channel;
  tiling.param3 = highBlock;
  tiling.param4 = stride;
  tiling.param5 = blockSize;
  tiling.param6 = repeat;
  tiling.param7 = firstAxisAlign;
  tiling.param8 = firstAxisRem;
  tiling.param9 = secondAxisAlign;
  tiling.param10 = secondAxisRem;
}

// scene12 shape[s0,s1,s2]->shape[s2,s1,s0]
void GetConfusionTranspose210TilingInfo(const ShapeInfo srcShape, const uint32_t stackBufferSize,
                                        const uint32_t typeSize, ConfusionTransposeTiling &tiling) {
  uint32_t channel = srcShape.originalShape[0];
  uint32_t height = srcShape.originalShape[1];
  uint32_t width = srcShape.originalShape[2];

  uint32_t blockSize = ONE_BLK_SIZE / typeSize;
  uint32_t highBlock = channel / BLOCK_CUBE;
  uint32_t firstAxisAlign = ALIGN_UP(channel, BLOCK_CUBE);
  uint32_t firstAxisRem = channel % BLOCK_CUBE;
  uint32_t secondAxisAlign = ALIGN_UP(width, 16); // float32和float16的尾轴都对齐到16
  uint32_t secondAxisRem = width % blockSize;
  uint32_t stride = firstAxisAlign * height;
  uint32_t repeat = width / blockSize;

  tiling.param0 = height;
  tiling.param1 = width;
  tiling.param2 = channel;
  tiling.param3 = highBlock;
  tiling.param4 = stride;
  tiling.param5 = blockSize;
  tiling.param6 = repeat;
  tiling.param7 = firstAxisAlign;
  tiling.param8 = firstAxisRem;
  tiling.param9 = secondAxisAlign;
  tiling.param10 = secondAxisRem;
}

// scene13 shape[s0, s1, s2, s3]->shape[s0, s3, s2, s1]
void GetConfusionTranspose0321TilingInfo(const ShapeInfo srcShape, const uint32_t stackBufferSize,
                                         const uint32_t typeSize, ConfusionTransposeTiling &tiling) {
  uint32_t batch = srcShape.originalShape[0];
  uint32_t channel = srcShape.originalShape[1];
  uint32_t height = srcShape.originalShape[2];
  uint32_t width = srcShape.originalShape[3];

  uint32_t blockSize = ONE_BLK_SIZE / typeSize;
  uint32_t highBlock = channel / BLOCK_CUBE;
  uint32_t firstAxisAlign = ALIGN_UP(channel, BLOCK_CUBE);
  uint32_t firstAxisRem = channel % BLOCK_CUBE;
  uint32_t secondAxisAlign = ALIGN_UP(width, 16); // float32和float16的尾轴都对齐到16
  uint32_t secondAxisRem = width % blockSize;
  uint32_t stride = firstAxisAlign * height;
  uint32_t repeat = width / blockSize;

  tiling.param0 = height;
  tiling.param1 = width;
  tiling.param2 = channel;
  tiling.param3 = batch;
  tiling.param4 = highBlock;
  tiling.param5 = stride;
  tiling.param6 = blockSize;
  tiling.param7 = repeat;
  tiling.param8 = firstAxisAlign;
  tiling.param9 = firstAxisRem;
  tiling.param10 = secondAxisAlign;
  tiling.param11 = secondAxisRem;
}

__aicore__ inline void GetConfusionTransposeTilingInfo(ShapeInfo srcShape, uint32_t stackBufferSize,
                                                       uint32_t typeSize, const uint32_t transposeTypeIn,
                                                       ConfusionTransposeTiling &tiling) {
  if (static_cast<AutoFuseTransposeType>(transposeTypeIn) == AutoFuseTransposeType::TRANSPOSE_ND2ND_ONLY) {
    GetConfusionTranspose10TilingInfo(srcShape, stackBufferSize, typeSize, tiling);
  } else if (static_cast<AutoFuseTransposeType>(transposeTypeIn) == AutoFuseTransposeType::TRANSPOSE_ND2ND_102) {
    GetConfusionTranspose102TilingInfo(srcShape, stackBufferSize, typeSize, tiling);
  } else if (static_cast<AutoFuseTransposeType>(transposeTypeIn) == AutoFuseTransposeType::TRANSPOSE_ND2ND_0213) {
    GetConfusionTransposeND2ND0213TilingInfo(srcShape, stackBufferSize, typeSize, tiling);
  } else if (static_cast<AutoFuseTransposeType>(transposeTypeIn) == AutoFuseTransposeType::TRANSPOSE_ND2ND_2103) {
    GetConfusionTranspose2103TilingInfo(srcShape, stackBufferSize, typeSize, tiling);
  } else if (static_cast<AutoFuseTransposeType>(transposeTypeIn) == AutoFuseTransposeType::TRANSPOSE_ND2ND_021) {
    GetConfusionTranspose021TilingInfo(srcShape, stackBufferSize, typeSize, tiling);
  } else if (static_cast<AutoFuseTransposeType>(transposeTypeIn) == AutoFuseTransposeType::TRANSPOSE_ND2ND_210) {
    GetConfusionTranspose210TilingInfo(srcShape, stackBufferSize, typeSize, tiling);
  } else if (static_cast<AutoFuseTransposeType>(transposeTypeIn) == AutoFuseTransposeType::TRANSPOSE_ND2ND_0321) {
    GetConfusionTranspose0321TilingInfo(srcShape, stackBufferSize, typeSize, tiling);
  }
}

template <typename T>
static auto CreateTransposeKernel() {
  return [](T *x, T *y, const ShapeInfo &srcShape, AutoFuseTransposeType transposetypeIn) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> inputLocal, outPutLocal, tmp_tbuf;

    uint32_t shapeDim = srcShape.shapeDim;
    uint32_t srcShapeSize = 1;
    uint32_t dstShapeSize = 1;
    if (static_cast<AutoFuseTransposeType>(transposetypeIn) == AutoFuseTransposeType::TRANSPOSE_ND2ND_ONLY) {
      srcShapeSize = srcShape.shape[0] * ALIGN_UP(srcShape.shape[1], 16);
      dstShapeSize = srcShape.shape[1] * ALIGN_UP(srcShape.shape[0], BLOCK_CUBE);
    } else if (static_cast<AutoFuseTransposeType>(transposetypeIn) == AutoFuseTransposeType::TRANSPOSE_ND2ND_021) {
      srcShapeSize = srcShape.shape[0] * srcShape.shape[1] * ALIGN_UP(srcShape.shape[2], 16);
      dstShapeSize = srcShape.shape[0] * srcShape.shape[2] * ALIGN_UP(srcShape.shape[1], BLOCK_CUBE);
    } else if (static_cast<AutoFuseTransposeType>(transposetypeIn) == AutoFuseTransposeType::TRANSPOSE_ND2ND_210) {
      srcShapeSize = srcShape.shape[0] * srcShape.shape[1] * ALIGN_UP(srcShape.shape[2], 16);
      dstShapeSize = srcShape.shape[2] * srcShape.shape[1] * ALIGN_UP(srcShape.shape[0], BLOCK_CUBE);
    } else if (static_cast<AutoFuseTransposeType>(transposetypeIn) == AutoFuseTransposeType::TRANSPOSE_ND2ND_0321) {
      srcShapeSize = srcShape.shape[0] * srcShape.shape[1] * srcShape.shape[2] * ALIGN_UP(srcShape.shape[3], 16);
      dstShapeSize = srcShape.shape[0] * srcShape.shape[3] * srcShape.shape[2] * ALIGN_UP(srcShape.shape[1], BLOCK_CUBE);
    } else {
      for (uint32_t dimIdx = 0; dimIdx < shapeDim; dimIdx++) {
        srcShapeSize *= srcShape.shape[dimIdx];
        dstShapeSize *= srcShape.shape[dimIdx];
      }
    }

    tpipe.InitBuffer(inputLocal, sizeof(T) * srcShapeSize);
    tpipe.InitBuffer(outPutLocal, sizeof(T) * dstShapeSize);
    tpipe.InitBuffer(tmp_tbuf, 8192);
    LocalTensor<uint8_t> tmp_buf = tmp_tbuf.Get<uint8_t>();

    auto inPutTensor = inputLocal.Get<T>();
    auto outPutTensor = outPutLocal.Get<T>();

    const uint32_t stackBufferSize = 0;
    ConfusionTransposeTiling tiling;

    if (static_cast<AutoFuseTransposeType>(transposetypeIn) == AutoFuseTransposeType::TRANSPOSE_ND2ND_ONLY) {
      GmToUbGeneral(inPutTensor, x, 1, 1, srcShape.shape[0], srcShape.shape[1], ALIGN_UP(srcShape.shape[1], 16));
    } else if (static_cast<AutoFuseTransposeType>(transposetypeIn) == AutoFuseTransposeType::TRANSPOSE_ND2ND_021) {
      GmToUbGeneral(inPutTensor, x, 1, srcShape.shape[0], srcShape.shape[1], srcShape.shape[2], ALIGN_UP(srcShape.shape[2], 16));
    } else if (static_cast<AutoFuseTransposeType>(transposetypeIn) == AutoFuseTransposeType::TRANSPOSE_ND2ND_210) {
      GmToUbGeneral(inPutTensor, x, 1, srcShape.shape[0], srcShape.shape[1], srcShape.shape[2], ALIGN_UP(srcShape.shape[2], 16));
    } else if (static_cast<AutoFuseTransposeType>(transposetypeIn) == AutoFuseTransposeType::TRANSPOSE_ND2ND_0321) {
      GmToUbGeneral(inPutTensor, x, srcShape.shape[0], srcShape.shape[1], srcShape.shape[2], srcShape.shape[3], ALIGN_UP(srcShape.shape[3], 16));
    } else {
      GmToUb(inPutTensor, x, srcShapeSize);
    }

    GetConfusionTransposeTilingInfo(srcShape, stackBufferSize, sizeof(T), (uint32_t)transposetypeIn, tiling);
    if constexpr (std::is_same_v<T, half>) {
        codegen::ConfusionTranspose<half>(outPutTensor, inPutTensor, tmp_buf, transposetypeIn, tiling);
    } else if constexpr (std::is_same_v<T, float>) {
        codegen::ConfusionTranspose<float>(outPutTensor, inPutTensor, tmp_buf, transposetypeIn, tiling);
    }

    if (static_cast<AutoFuseTransposeType>(transposetypeIn) == AutoFuseTransposeType::TRANSPOSE_ND2ND_ONLY) {
      UbToGmGeneral(y, outPutTensor, 1, 1, srcShape.shape[1], srcShape.shape[0], ALIGN_UP(srcShape.shape[0], BLOCK_CUBE));
    } else if (static_cast<AutoFuseTransposeType>(transposetypeIn) == AutoFuseTransposeType::TRANSPOSE_ND2ND_021) {
      UbToGmGeneral(y, outPutTensor, 1, srcShape.shape[0], srcShape.shape[2], srcShape.shape[1], ALIGN_UP(srcShape.shape[1], BLOCK_CUBE));
    } else if (static_cast<AutoFuseTransposeType>(transposetypeIn) == AutoFuseTransposeType::TRANSPOSE_ND2ND_210) {
      UbToGmGeneral(y, outPutTensor, 1, srcShape.shape[2], srcShape.shape[1], srcShape.shape[0], ALIGN_UP(srcShape.shape[0], BLOCK_CUBE));
    } else if (static_cast<AutoFuseTransposeType>(transposetypeIn) == AutoFuseTransposeType::TRANSPOSE_ND2ND_0321) {
      UbToGmGeneral(y, outPutTensor, srcShape.shape[0], srcShape.shape[3], srcShape.shape[2], srcShape.shape[1], ALIGN_UP(srcShape.shape[1], BLOCK_CUBE));
    } else {
      UbToGm(y, outPutTensor, dstShapeSize);
    }
  };
}
static auto kernel_half = CreateTransposeKernel<half>();
static auto kernel_float = CreateTransposeKernel<float>();

// 10
template <typename T>
class TestApiTranspose10Base;

template <typename T>
class TestApiTranspose10Base : public testing::Test {
 protected:
  void RunTest(const std::vector<int> &param) {
    uint32_t dimNum = param[0];
    uint32_t srcZ0 = param[1];
    uint32_t srcZ1 = param[2];
    AutoFuseTransposeType transposetypeIn = (AutoFuseTransposeType)param[3];

    uint32_t srcShapeSize = srcZ0 * srcZ1;
    uint32_t srcOriShape[] = {srcZ0, srcZ1};
    uint32_t srcShape[] = {srcZ0, srcZ1};

    ShapeInfo srcShapeInfo(2, srcShape, 2, srcOriShape, DataFormat::ND);

    auto *input = (T *)AscendC::GmAlloc(sizeof(T) * srcShapeSize);
    auto *output = (T *)AscendC::GmAlloc(sizeof(T) * srcShapeSize);
    auto *expectOutput = (T *)AscendC::GmAlloc(sizeof(T) * srcShapeSize);

    for (uint32_t i = 0; i < srcZ0; i++) {
      for (uint32_t j = 0; j < srcZ1; j++) {
      input[i * srcZ1 + j] = static_cast<T>(i * srcZ1 + j);
      expectOutput[j * srcZ0 + i] = static_cast<T>(i * srcZ1 + j);
      }
    }

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    if constexpr (std::is_same_v<T, half>) {
      ICPU_RUN_KF(kernel_half, 1, input, output, srcShapeInfo, AutoFuseTransposeType::TRANSPOSE_ND2ND_ONLY);
    } else {
      ICPU_RUN_KF(kernel_float, 1, input, output, srcShapeInfo, AutoFuseTransposeType::TRANSPOSE_ND2ND_ONLY);
    }

    int diff_count = 0;
    for (int i = 0; i < srcShapeSize; i++) {
      auto diff = static_cast<double>(output[i] - expectOutput[i]);
      if (diff < -1e-5 || diff > 1e-5) {
        diff_count++;
      }
    }

    EXPECT_EQ(diff_count, 0);

    AscendC::GmFree(input);
    AscendC::GmFree(output);
    AscendC::GmFree(expectOutput);
  }
};

// half类型
class TestApiTranspose10Half : public TestApiTranspose10Base<half>,
                                public testing::WithParamInterface<std::vector<int>> {};

TEST_P(TestApiTranspose10Half, TestTranspose) {
  RunTest(GetParam());
}

// float类型
class TestApiTranspose10Float : public TestApiTranspose10Base<float>,
                                 public testing::WithParamInterface<std::vector<int>> {};

TEST_P(TestApiTranspose10Float, TestTranspose) {
  RunTest(GetParam());
}

// 实例化测试用例
INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShapeHalf, TestApiTranspose10Half,
                         ::testing::Values(std::vector<int>{2, 37, 61,
                                                            (int)AutoFuseTransposeType::TRANSPOSE_ND2ND_ONLY}));

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShapeFloat, TestApiTranspose10Float,
                         ::testing::Values(std::vector<int>{2, 37, 61,
                                                            (int)AutoFuseTransposeType::TRANSPOSE_ND2ND_ONLY}));

template <typename T>
static auto CreateTransposeKernelPerm102() {
    return [](T *output_gm, T *input_gm, const ShapeInfo &srcShape, AutoFuseTransposeType transposetypeIn) {

        TPipe tpipe;
        TBuf<TPosition::VECCALC> local_input;
        TBuf<TPosition::VECCALC> local_output;

        uint32_t srcZ0 = srcShape.originalShape[0];
        uint32_t srcZ1 = srcShape.originalShape[1];
        uint32_t srcZ2 = srcShape.originalShape[2];
        uint32_t srcShapeSize = srcZ0 * srcZ1 * srcZ2;

        /* 尾轴要求block对齐 */
        uint32_t blockSize = ONE_BLK_SIZE / sizeof(T);
        uint32_t lastDim = srcZ2;
        uint32_t lastDimAlign = ALIGN_UP(lastDim, 16);

        tpipe.InitBuffer(local_input, sizeof(T) * lastDimAlign * srcZ1 * srcZ0);
        tpipe.InitBuffer(local_output, sizeof(T) * lastDimAlign * srcZ1 * srcZ0);

        auto l_input = local_input.Get<T>();
        auto l_output = local_output.Get<T>();

        for (uint32_t i = 0; i < srcZ0; i++) {
            for (uint32_t j = 0; j < srcZ1; j++) {
                auto input_tensor = l_input[i * srcZ1 * lastDimAlign + j * lastDimAlign];
                GmToUb(input_tensor, &input_gm[i * srcZ1 * srcZ2 + j * srcZ2], srcZ2);
            }
        }

        uint32_t stackBufferSize = 0;
        ConfusionTransposeTiling tiling;
        GetConfusionTransposeTilingInfo(srcShape, stackBufferSize, sizeof(T), (uint32_t)transposetypeIn, tiling);
        codegen::ConfusionTranspose<T>(l_output, l_input, transposetypeIn, tiling);

        for (uint32_t i = 0; i < srcZ0; i++) {
            for (uint32_t j = 0; j < srcZ1; j++) {
                auto output_tensor = l_output[i * lastDimAlign +  j * srcZ0 * lastDimAlign];
                UbToGm(&output_gm[i * srcZ2 + j * srcZ0 * srcZ2], output_tensor, srcZ2);
            }
        }
    };
}
static auto kernel_perm102_half = CreateTransposeKernelPerm102<half>();
static auto kernel_perm102_float = CreateTransposeKernelPerm102<float>();
// 102
template <typename T>
class TestApiTranspose102Base;

template <typename T>
class TestApiTranspose102Base : public testing::Test {
protected:
    void RunTest(const std::vector<int> &param) {
        // 构造测试输入和预期结果
        uint32_t dimNum = param[0];
        uint32_t srcZ0 = param[1];
        uint32_t srcZ1 = param[2];
        uint32_t srcZ2 = param[3];
        uint32_t srcZ3 = param[4];
        uint32_t srcShapeSize = srcZ0 * srcZ1 * srcZ2;
        AutoFuseTransposeType transposetypeIn = (AutoFuseTransposeType) param[5];

        auto *input_gm = (T *) AscendC::GmAlloc(sizeof(T) * srcShapeSize);
        auto *output_gm = (T *) AscendC::GmAlloc(sizeof(T) * srcShapeSize);
        auto *expectOutput = (T *) AscendC::GmAlloc(sizeof(T) * srcShapeSize);


        for (uint32_t i = 0; i < srcZ0; i++) {
            for (uint32_t j = 0; j < srcZ1; j++) {
                for (uint32_t k = 0; k < srcZ2; k++) {
                    input_gm[i * srcZ1 * srcZ2 + j * srcZ2 + k] = static_cast<T>(i * srcZ1 * srcZ2 + j * srcZ2 + k);
                    expectOutput[j * srcZ0 * srcZ2 + i * srcZ2 + k] = static_cast<T>(i * srcZ1 * srcZ2 + j * srcZ2 + k);
                }
            }
        }

        uint32_t srcOriShape[] = {srcZ0, srcZ1, srcZ2};
        uint32_t srcShape[] = {srcZ0, srcZ1, srcZ2};
        ShapeInfo srcShapeInfo(3, srcShape, 3, srcOriShape, DataFormat::ND);

        AscendC::SetKernelMode(KernelMode::AIV_MODE);
        if constexpr (std::is_same_v<T, half>) {
            ICPU_RUN_KF(kernel_perm102_half, 1, output_gm, input_gm, srcShapeInfo,
                        AutoFuseTransposeType::TRANSPOSE_ND2ND_102);
        } else {
            ICPU_RUN_KF(kernel_perm102_float, 1, output_gm, input_gm, srcShapeInfo,
                        AutoFuseTransposeType::TRANSPOSE_ND2ND_102);
        }

        // 验证结果
        int diff_count = 0;
        for (int i = 0; i < srcShapeSize; i++) {
            auto diff = (double) (output_gm[i] - expectOutput[i]);
            if (diff < -1e-5 || diff > 1e-5) {
                diff_count++;
            }
        }

        EXPECT_EQ(diff_count, 0);

        AscendC::GmFree(input_gm);
        AscendC::GmFree(output_gm);
        AscendC::GmFree(expectOutput);
    }
};

// half类型
class TestApiTranspose102Half : public TestApiTranspose102Base<half>,
                               public testing::WithParamInterface<std::vector<int>> {};

TEST_P(TestApiTranspose102Half, TestTranspose) {
    RunTest(GetParam());
}

// float类型
class TestApiTranspose102Float : public TestApiTranspose102Base<float>,
                                public testing::WithParamInterface<std::vector<int>> {};

TEST_P(TestApiTranspose102Float, TestTranspose) {
    RunTest(GetParam());
}

// 实例化测试用例
INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShapeHalf, TestApiTranspose102Half,
                         ::testing::Values(std::vector<int>{3, 2, 4, 16, 1, (int)AutoFuseTransposeType::TRANSPOSE_ND2ND_102},
                                           std::vector<int>{3, 2, 16, 512, 1, (int)AutoFuseTransposeType::TRANSPOSE_ND2ND_102},
                                           std::vector<int>{3, 2, 16, 15, 1, (int)AutoFuseTransposeType::TRANSPOSE_ND2ND_102}));

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShapeHalf, TestApiTranspose102Float,
                         ::testing::Values(std::vector<int>{3, 2, 4, 15, 1, (int)AutoFuseTransposeType::TRANSPOSE_ND2ND_102},
                                           std::vector<int>{3, 2, 16, 16, 1, (int)AutoFuseTransposeType::TRANSPOSE_ND2ND_102},
                                           std::vector<int>{3, 2, 16, 17, 1, (int)AutoFuseTransposeType::TRANSPOSE_ND2ND_102}));

template <typename T>
static auto CreateTransposeKernelPerm0213() {
    return [](T *output_gm, T *input_gm, const ShapeInfo &srcShape, AutoFuseTransposeType transposetypeIn) {

        /* 新增UT适配 */
        TPipe tpipe;
        TBuf<TPosition::VECCALC> local_input;
        TBuf<TPosition::VECCALC> local_output;

        uint32_t srcZ0 = srcShape.originalShape[0];
        uint32_t srcZ1 = srcShape.originalShape[1];
        uint32_t srcZ2 = srcShape.originalShape[2];
        uint32_t srcZ3 = srcShape.originalShape[3];
        uint32_t srcShapeSize = srcZ0 * srcZ1 * srcZ2 * srcZ3;

        /* 尾轴要求block对齐 */
        uint32_t blockSize = ONE_BLK_SIZE / sizeof(T);
        uint32_t lastDim = srcZ3;
        uint32_t lastDimAlign = ALIGN_UP(lastDim, 16);

        tpipe.InitBuffer(local_input, sizeof(T) * srcZ0 * srcZ1 * srcZ2 * lastDimAlign);
        tpipe.InitBuffer(local_output, sizeof(T) * srcZ0 * srcZ1 * srcZ2 * lastDimAlign);

        auto l_input = local_input.Get<T>();
        auto l_output = local_output.Get<T>();

        for (uint32_t i = 0; i < srcZ0; i++) {
            for (uint32_t j = 0; j < srcZ1; j++) {
                for (uint32_t k = 0; k < srcZ2; k++) {
                    auto input_tensor = l_input[i * srcZ1 * srcZ2 * lastDimAlign + j * srcZ2 * lastDimAlign + k * lastDimAlign];
                    GmToUb(input_tensor, &input_gm[i * srcZ1 * srcZ2 * srcZ3 + j * srcZ2 * srcZ3 + k * srcZ3], srcZ3);
                }
            }
        }

        uint32_t stackBufferSize = 0;
        ConfusionTransposeTiling tiling;
        GetConfusionTransposeTilingInfo(srcShape, stackBufferSize, sizeof(T), (uint32_t)transposetypeIn, tiling);
        codegen::ConfusionTranspose<T>(l_output, l_input, transposetypeIn, tiling);

        for (uint32_t i = 0; i < srcZ0; i++) {
            for (uint32_t j = 0; j < srcZ1; j++) {
                for (uint32_t k = 0; k < srcZ2; k++) {
                    auto output_tensor = l_output[i * srcZ1 * srcZ2 * lastDimAlign +  j * lastDimAlign + k * srcZ1 * lastDimAlign];
                    UbToGm(&output_gm[i * srcZ1 * srcZ2 * srcZ3 +  j * srcZ3 + k * srcZ1 * srcZ3], output_tensor, srcZ3);
                }
            }
        }
    };
}

static auto kernel_perm0213_half = CreateTransposeKernelPerm0213<half>();
static auto kernel_perm0213_float = CreateTransposeKernelPerm0213<float>();

// 0213
template <typename T>
class TestApiTranspose0213Base;

template <typename T>
class TestApiTranspose0213Base : public testing::Test {
protected:
    void RunTest(const std::vector<int> &param) {
        // 构造测试输入和预期结果
        uint32_t dimNum = param[0];
        uint32_t srcZ0 = param[1];
        uint32_t srcZ1 = param[2];
        uint32_t srcZ2 = param[3];
        uint32_t srcZ3 = param[4];
        AutoFuseTransposeType transposetypeIn = (AutoFuseTransposeType) param[5];

        uint32_t srcShapeSize = srcZ0 * srcZ1 * srcZ2 * srcZ3;
        uint32_t srcOriShape[] = {srcZ0, srcZ1, srcZ2, srcZ3};
        uint32_t srcShape[] = {srcZ0, srcZ1, srcZ2, srcZ3};
        ShapeInfo srcShapeInfo(4, srcShape, 4, srcOriShape, DataFormat::ND);

        auto *input_gm = (T *) AscendC::GmAlloc(sizeof(T) * srcShapeSize);
        auto *output_gm = (T *) AscendC::GmAlloc(sizeof(T) * srcShapeSize);
        auto *expectOutput = (T *) AscendC::GmAlloc(sizeof(T) * srcShapeSize);

        for (uint32_t i = 0; i < srcZ0; i++) {
            for (uint32_t j = 0; j < srcZ1; j++) {
                for (uint32_t k = 0; k < srcZ2; k++) {
                    for (uint32_t l = 0; l < srcZ3; l++) {
                        input_gm[i * srcZ1 * srcZ2 * srcZ3 + j * srcZ2 * srcZ3 + k * srcZ3 + l] =
                                static_cast<T>(i * srcZ1 * srcZ2 * srcZ3 + j * srcZ2 * srcZ3 + k * srcZ3 + l);
                        expectOutput[i * srcZ1 * srcZ2 * srcZ3 + k * srcZ1 * srcZ3 + j * srcZ3 + l] =
                                static_cast<T>(i * srcZ1 * srcZ2 * srcZ3 + j * srcZ2 * srcZ3 + k * srcZ3 + l);
                    }
                }
            }
        }

        AscendC::SetKernelMode(KernelMode::AIV_MODE);
        if constexpr (std::is_same_v<T, half>) {
            ICPU_RUN_KF(kernel_perm0213_half, 1, output_gm, input_gm, srcShapeInfo,
                        AutoFuseTransposeType::TRANSPOSE_ND2ND_0213);
        } else {
            ICPU_RUN_KF(kernel_perm0213_float, 1, output_gm, input_gm, srcShapeInfo,
                        AutoFuseTransposeType::TRANSPOSE_ND2ND_0213);
        }


        // 验证结果
        int diff_count = 0;
        for (int i = 0; i < srcShapeSize; i++) {
            auto diff = (double) (output_gm[i] - expectOutput[i]);
            if (diff < -1e-5 || diff > 1e-5) {
                diff_count++;
            }
        }

        EXPECT_EQ(diff_count, 0);

        AscendC::GmFree(input_gm);
        AscendC::GmFree(output_gm);
        AscendC::GmFree(expectOutput);
    }
};

// half类型
class TestApiTranspose0213Half : public TestApiTranspose0213Base<half>,
                               public testing::WithParamInterface<std::vector<int>> {};

TEST_P(TestApiTranspose0213Half, TestTranspose) {
    RunTest(GetParam());
}

// float类型
class TestApiTranspose0213Float : public TestApiTranspose0213Base<float>,
                                public testing::WithParamInterface<std::vector<int>> {};

TEST_P(TestApiTranspose0213Float, TestTranspose) {
    RunTest(GetParam());
}

// 实例化测试用例
INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, TestApiTranspose0213Half,
                         ::testing::Values(std::vector<int>{4, 2, 4, 8, 16, (int)AutoFuseTransposeType::TRANSPOSE_ND2ND_0213},
                                            std::vector<int>{4, 2, 4, 8, 17, (int)AutoFuseTransposeType::TRANSPOSE_ND2ND_0213},
                                            std::vector<int>{4, 2, 4, 8, 7, (int)AutoFuseTransposeType::TRANSPOSE_ND2ND_0213}));
INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, TestApiTranspose0213Float,
                         ::testing::Values(std::vector<int>{4, 2, 4, 8, 15, (int)AutoFuseTransposeType::TRANSPOSE_ND2ND_0213},
                                           std::vector<int>{4, 2, 4, 8, 16, (int)AutoFuseTransposeType::TRANSPOSE_ND2ND_0213},
                                           std::vector<int>{4, 2, 4, 8, 17, (int)AutoFuseTransposeType::TRANSPOSE_ND2ND_0213}));

template <typename T>
static auto CreateTransposeKernelPerm2103() {
    return [](T *output_gm, T *input_gm, const ShapeInfo &srcShape, AutoFuseTransposeType transposetypeIn) {

        /* 新增UT适配 */
        TPipe tpipe;
        TBuf<TPosition::VECCALC> local_input;
        TBuf<TPosition::VECCALC> local_output;

        uint32_t srcZ0 = srcShape.originalShape[0];
        uint32_t srcZ1 = srcShape.originalShape[1];
        uint32_t srcZ2 = srcShape.originalShape[2];
        uint32_t srcZ3 = srcShape.originalShape[3];
        uint32_t srcShapeSize = srcZ0 * srcZ1 * srcZ2 * srcZ3;

        /* 尾轴要求block对齐 */
        uint32_t blockSize = ONE_BLK_SIZE / sizeof(T);
        uint32_t lastDim = srcZ3;
        uint32_t lastDimAlign = ALIGN_UP(lastDim, 16);

        tpipe.InitBuffer(local_input, sizeof(T) * srcZ0 * srcZ1 * srcZ2 * lastDimAlign);
        tpipe.InitBuffer(local_output, sizeof(T) * srcZ0 * srcZ1 * srcZ2 * lastDimAlign);

        auto l_input = local_input.Get<T>();
        auto l_output = local_output.Get<T>();

        for (uint32_t i = 0; i < srcZ0; i++) {
            for (uint32_t j = 0; j < srcZ1; j++) {
                for (uint32_t k = 0; k < srcZ2; k++) {
                    auto input_tensor = l_input[i * srcZ1 * srcZ2 * lastDimAlign + j * srcZ2 * lastDimAlign + k * lastDimAlign];
                    GmToUb(input_tensor, &input_gm[i * srcZ1 * srcZ2 * srcZ3 + j * srcZ2 * srcZ3 + k * srcZ3], srcZ3);
                }
            }
        }

        uint32_t stackBufferSize = 0;
        ConfusionTransposeTiling tiling;
        GetConfusionTransposeTilingInfo(srcShape, stackBufferSize, sizeof(T), (uint32_t)transposetypeIn, tiling);
        codegen::ConfusionTranspose<T>(l_output, l_input, transposetypeIn, tiling);

        for (uint32_t i = 0; i < srcZ0; i++) {
            for (uint32_t j = 0; j < srcZ1; j++) {
                for (uint32_t k = 0; k < srcZ2; k++) {
                    auto output_tensor = l_output[i * lastDimAlign +  j * srcZ0 * lastDimAlign + k * srcZ0 * srcZ1 * lastDimAlign];
                    UbToGm(&output_gm[i * srcZ3 +  j * srcZ0 * srcZ3 + k * srcZ0 * srcZ1 * srcZ3], output_tensor, srcZ3);
                }
            }
        }
    };
}

static auto kernel_perm2103_half = CreateTransposeKernelPerm2103<half>();
static auto kernel_perm2103_float = CreateTransposeKernelPerm2103<float>();

template <typename T>
class TestApiTranspose2103Base;

template <typename T>
class TestApiTranspose2103Base : public testing::Test {
  protected:
    void RunTest(const std::vector<int> &param) {
    // 构造测试输入和预期结果
    uint32_t dimNum = param[0];
    uint32_t srcZ0 = param[1];
    uint32_t srcZ1 = param[2];
    uint32_t srcZ2 = param[3];
    uint32_t srcZ3 = param[4];
    AutoFuseTransposeType transposetypeIn = (AutoFuseTransposeType)param[5];

    uint32_t srcShapeSize = srcZ0 * srcZ1 * srcZ2 * srcZ3;
    uint32_t srcOriShape[] = {srcZ0, srcZ1, srcZ2, srcZ3};
    uint32_t srcShape[] = {srcZ0, srcZ1, srcZ2, srcZ3};

    ShapeInfo srcShapeInfo(4, srcShape, 4, srcOriShape, DataFormat::ND);

    auto *input_gm = (T *)AscendC::GmAlloc(sizeof(T) * srcShapeSize);
    auto *output_gm = (T *)AscendC::GmAlloc(sizeof(T) * srcShapeSize);
    auto *expectOutput = (T *)AscendC::GmAlloc(sizeof(T) * srcShapeSize);

    for (uint32_t i = 0; i < srcZ0; i++) {
      for (uint32_t j = 0; j < srcZ1; j++) {
        for (uint32_t k = 0; k < srcZ2; k++) {
          for (uint32_t l = 0; l < srcZ3; l++) {
              input_gm[i * srcZ1 * srcZ2 * srcZ3 + j * srcZ2 * srcZ3 + k * srcZ3 + l] =
                      static_cast<T>(i * srcZ1 * srcZ2 * srcZ3 + j * srcZ2 * srcZ3 + k * srcZ3 + l);
              expectOutput[k * srcZ0 * srcZ1 * srcZ3 + j * srcZ0 * srcZ3 + i * srcZ3 + l] =
                      static_cast<T>(i * srcZ1 * srcZ2 * srcZ3 + j * srcZ2 * srcZ3 + k * srcZ3 + l);
          }
        }
      }
    }

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    if constexpr (std::is_same_v<T, half>) {
        ICPU_RUN_KF(kernel_perm2103_half, 1, output_gm, input_gm, srcShapeInfo, AutoFuseTransposeType::TRANSPOSE_ND2ND_2103);
    } else {
        ICPU_RUN_KF(kernel_perm2103_float, 1, output_gm, input_gm, srcShapeInfo, AutoFuseTransposeType::TRANSPOSE_ND2ND_2103);
    }

    // 验证结果
    int diff_count = 0;
    for (int i = 0; i < srcShapeSize; i++) {
      auto diff = (double)(output_gm[i] - expectOutput[i]);
      if (diff < -1e-5 || diff > 1e-5) {
        diff_count++;
      }
    }

    EXPECT_EQ(diff_count, 0);
    AscendC::GmFree(input_gm);
    AscendC::GmFree(output_gm);
    AscendC::GmFree(expectOutput);
  }
};

// half类型
class TestApiTranspose2103Half : public TestApiTranspose2103Base<half>,
                               public testing::WithParamInterface<std::vector<int>> {};

TEST_P(TestApiTranspose2103Half, TestTranspose) {
    RunTest(GetParam());
}

// float类型
class TestApiTranspose2103Float : public TestApiTranspose2103Base<float>,
                                public testing::WithParamInterface<std::vector<int>> {};

TEST_P(TestApiTranspose2103Float, TestTranspose) {
    RunTest(GetParam());
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, TestApiTranspose2103Half,
                         ::testing::Values(std::vector<int>{4, 2, 4, 8, 16, (int)AutoFuseTransposeType::TRANSPOSE_ND2ND_2103},
                                           std::vector<int>{4, 2, 4, 8, 7, (int)AutoFuseTransposeType::TRANSPOSE_ND2ND_2103},
                                           std::vector<int>{4, 2, 4, 8, 17, (int)AutoFuseTransposeType::TRANSPOSE_ND2ND_2103}));
INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, TestApiTranspose2103Float,
                         ::testing::Values(std::vector<int>{4, 2, 4, 8, 15, (int)AutoFuseTransposeType::TRANSPOSE_ND2ND_2103},
                                           std::vector<int>{4, 2, 4, 8, 16, (int)AutoFuseTransposeType::TRANSPOSE_ND2ND_2103},
                                           std::vector<int>{4, 2, 4, 8, 17, (int)AutoFuseTransposeType::TRANSPOSE_ND2ND_2103}));

// 021
template <typename T>
class TestApiTranspose021Base;

template <typename T>
class TestApiTranspose021Base : public testing::Test {
 protected:
  void RunTest(const std::vector<int> &param) {
    uint32_t dimNum = param[0];
    uint32_t srcZ0 = param[1];
    uint32_t srcZ1 = param[2];
    uint32_t srcZ2 = param[3];
    AutoFuseTransposeType transposetypeIn = (AutoFuseTransposeType)param[4];

    uint32_t srcShapeSize = srcZ0 * srcZ1 * srcZ2;
    uint32_t srcOriShape[] = {srcZ0, srcZ1, srcZ2};
    uint32_t srcShape[] = {srcZ0, srcZ1, srcZ2};

    ShapeInfo srcShapeInfo(3, srcShape, 3, srcOriShape, DataFormat::ND);

    auto *input = (T *)AscendC::GmAlloc(sizeof(T) * srcShapeSize);
    auto *output = (T *)AscendC::GmAlloc(sizeof(T) * srcShapeSize);
    auto *expectOutput = (T *)AscendC::GmAlloc(sizeof(T) * srcShapeSize);

    for (uint32_t i = 0; i < srcZ0; i++) {
      for (uint32_t j = 0; j < srcZ1; j++) {
        for (uint32_t k = 0; k < srcZ2; k++) {
          input[i * srcZ1 * srcZ2 + j * srcZ2 + k] = static_cast<T>(i * srcZ1 * srcZ2 + j * srcZ2 + k);
          expectOutput[i * srcZ1 * srcZ2 + k * srcZ1 + j] = static_cast<T>(i * srcZ1 * srcZ2 + j * srcZ2 + k);
        }
      }
    }

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    if constexpr (std::is_same_v<T, half>) {
      ICPU_RUN_KF(kernel_half, 1, input, output, srcShapeInfo, AutoFuseTransposeType::TRANSPOSE_ND2ND_021);
    } else {
      ICPU_RUN_KF(kernel_float, 1, input, output, srcShapeInfo, AutoFuseTransposeType::TRANSPOSE_ND2ND_021);
    }

    int diff_count = 0;
    for (int i = 0; i < srcShapeSize; i++) {
      auto diff = static_cast<double>(output[i] - expectOutput[i]);
      if (diff < -1e-5 || diff > 1e-5) {
        diff_count++;
      }
    }

    if constexpr (std::is_same_v<T, half>) {
      EXPECT_EQ(diff_count, 0);
    } else {
      EXPECT_EQ(diff_count, 0); // 先上尾轴非对齐转置fp16，fp32后续适配修改
    }

    AscendC::GmFree(input);
    AscendC::GmFree(output);
    AscendC::GmFree(expectOutput);
  }
};

// half类型
class TestApiTranspose021Half : public TestApiTranspose021Base<half>,
                                public testing::WithParamInterface<std::vector<int>> {};

TEST_P(TestApiTranspose021Half, TestTranspose) {
  RunTest(GetParam());
}

// float类型
class TestApiTranspose021Float : public TestApiTranspose021Base<float>,
                                 public testing::WithParamInterface<std::vector<int>> {};

TEST_P(TestApiTranspose021Float, TestTranspose) {
  RunTest(GetParam());
}

// 实例化测试用例
INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShapeHalf, TestApiTranspose021Half,
                         ::testing::Values(std::vector<int>{3, 3, 37, 35,
                                                            (int)AutoFuseTransposeType::TRANSPOSE_ND2ND_021}));

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShapeFloat, TestApiTranspose021Float,
                         ::testing::Values(std::vector<int>{3, 4, 37, 71,
                                                            (int)AutoFuseTransposeType::TRANSPOSE_ND2ND_021}));

// 210
template <typename T>
class TestApiTranspose210Base;

template <typename T>
class TestApiTranspose210Base : public testing::Test {
 protected:
  void RunTest(const std::vector<int> &param) {
    uint32_t dimNum = param[0];
    uint32_t srcZ0 = param[1];
    uint32_t srcZ1 = param[2];
    uint32_t srcZ2 = param[3];
    AutoFuseTransposeType transposetypeIn = (AutoFuseTransposeType)param[4];

    uint32_t srcShapeSize = srcZ0 * srcZ1 * srcZ2;
    uint32_t srcOriShape[] = {srcZ0, srcZ1, srcZ2};
    uint32_t srcShape[] = {srcZ0, srcZ1, srcZ2};

    ShapeInfo srcShapeInfo(3, srcShape, 3, srcOriShape, DataFormat::ND);

    auto *input = (T *)AscendC::GmAlloc(sizeof(T) * srcShapeSize);
    auto *output = (T *)AscendC::GmAlloc(sizeof(T) * srcShapeSize);
    auto *expectOutput = (T *)AscendC::GmAlloc(sizeof(T) * srcShapeSize);

    for (uint32_t i = 0; i < srcZ0; i++) {
      for (uint32_t j = 0; j < srcZ1; j++) {
        for (uint32_t k = 0; k < srcZ2; k++) {
          input[i * srcZ1 * srcZ2 + j * srcZ2 + k] = static_cast<T>(i * srcZ1 * srcZ2 + j * srcZ2 + k);
          expectOutput[k * srcZ0 * srcZ1 + j * srcZ0 + i] = static_cast<T>(i * srcZ1 * srcZ2 + j * srcZ2 + k);
        }
      }
    }

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    if constexpr (std::is_same_v<T, half>) {
      ICPU_RUN_KF(kernel_half, 1, input, output, srcShapeInfo, AutoFuseTransposeType::TRANSPOSE_ND2ND_210);
    } else {
      ICPU_RUN_KF(kernel_float, 1, input, output, srcShapeInfo, AutoFuseTransposeType::TRANSPOSE_ND2ND_210);
    }

    int diff_count = 0;
    for (int i = 0; i < srcShapeSize; i++) {
      auto diff = static_cast<double>(output[i] - expectOutput[i]);
      if (diff < -1e-5 || diff > 1e-5) {
        diff_count++;
      }
    }

    EXPECT_EQ(diff_count, 0);

    AscendC::GmFree(input);
    AscendC::GmFree(output);
    AscendC::GmFree(expectOutput);
  }
};

// half类型
class TestApiTranspose210Half : public TestApiTranspose210Base<half>,
                                public testing::WithParamInterface<std::vector<int>> {};

TEST_P(TestApiTranspose210Half, TestTranspose) {
  RunTest(GetParam());
}

// float类型
class TestApiTranspose210Float : public TestApiTranspose210Base<float>,
                                 public testing::WithParamInterface<std::vector<int>> {};

TEST_P(TestApiTranspose210Float, TestTranspose) {
  RunTest(GetParam());
}

// 实例化测试用例
INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShapeHalf, TestApiTranspose210Half,
                         ::testing::Values(std::vector<int>{3, 37, 10, 35,
                                                            (int)AutoFuseTransposeType::TRANSPOSE_ND2ND_210}));

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShapeFloat, TestApiTranspose210Float,
                         ::testing::Values(std::vector<int>{3, 39, 3, 17,
                                                            (int)AutoFuseTransposeType::TRANSPOSE_ND2ND_210}));

// 0321
template <typename T>
class TestApiTranspose0321Base;

template <typename T>
class TestApiTranspose0321Base : public testing::Test {
 protected:
  void RunTest(const std::vector<int> &param) {
    uint32_t dimNum = param[0];
    uint32_t srcZ0 = param[1];
    uint32_t srcZ1 = param[2];
    uint32_t srcZ2 = param[3];
    uint32_t srcZ3 = param[4];
    AutoFuseTransposeType transposetypeIn = (AutoFuseTransposeType)param[5];

    uint32_t srcShapeSize = srcZ0 * srcZ1 * srcZ2 * srcZ3;
    uint32_t srcOriShape[] = {srcZ0, srcZ1, srcZ2, srcZ3};
    uint32_t srcShape[] = {srcZ0, srcZ1, srcZ2, srcZ3};

    ShapeInfo srcShapeInfo(4, srcShape, 4, srcOriShape, DataFormat::ND);

    auto *input = (T *)AscendC::GmAlloc(sizeof(T) * srcShapeSize);
    auto *output = (T *)AscendC::GmAlloc(sizeof(T) * srcShapeSize);
    auto *expectOutput = (T *)AscendC::GmAlloc(sizeof(T) * srcShapeSize);

    for (uint32_t i = 0; i < srcZ0; i++) {
      for (uint32_t j = 0; j < srcZ1; j++) {
        for (uint32_t k = 0; k < srcZ2; k++) {
          for (uint32_t l = 0; l < srcZ3; l++) {
            input[i * srcZ1 * srcZ2 * srcZ3 + j * srcZ2 * srcZ3 + k * srcZ3 + l] =
                static_cast<T>(i * srcZ1 * srcZ2 * srcZ3 + j * srcZ2 * srcZ3 + k * srcZ3 + l);
            expectOutput[i * srcZ1 * srcZ2 * srcZ3 + l * srcZ1 * srcZ2 + k * srcZ1 + j] =
                static_cast<T>(i * srcZ1 * srcZ2 * srcZ3 + j * srcZ2 * srcZ3 + k * srcZ3 + l);
          }
        }
      }
    }

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    if constexpr (std::is_same_v<T, half>) {
      ICPU_RUN_KF(kernel_half, 1, input, output, srcShapeInfo, AutoFuseTransposeType::TRANSPOSE_ND2ND_0321);
    } else {
      ICPU_RUN_KF(kernel_float, 1, input, output, srcShapeInfo, AutoFuseTransposeType::TRANSPOSE_ND2ND_0321);
    }

    int diff_count = 0;
    for (int i = 0; i < srcShapeSize; i++) {
      auto diff = static_cast<double>(output[i] - expectOutput[i]);
      if (diff < -1e-5 || diff > 1e-5) {
        diff_count++;
      }
    }

    EXPECT_EQ(diff_count, 0);

    AscendC::GmFree(input);
    AscendC::GmFree(output);
    AscendC::GmFree(expectOutput);
  }
};

// half类型
class TestApiTranspose0321Half : public TestApiTranspose0321Base<half>,
                                 public testing::WithParamInterface<std::vector<int>> {};

TEST_P(TestApiTranspose0321Half, TestTranspose) {
  RunTest(GetParam());
}

// float类型
class TestApiTranspose0321Float : public TestApiTranspose0321Base<float>,
                                  public testing::WithParamInterface<std::vector<int>> {};

TEST_P(TestApiTranspose0321Float, TestTranspose) {
  RunTest(GetParam());
}

// 实例化测试用例
INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShapeHalf, TestApiTranspose0321Half,
                         ::testing::Values(std::vector<int>{4, 3, 17, 2, 17,
                                                            (int)AutoFuseTransposeType::TRANSPOSE_ND2ND_0321}));

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShapeFloat, TestApiTranspose0321Float,
                         ::testing::Values(std::vector<int>{4, 4, 19, 2, 19,
                                                            (int)AutoFuseTransposeType::TRANSPOSE_ND2ND_0321}));