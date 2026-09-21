/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "tikicpulib.h"
#include "kernel_operator.h"
using namespace AscendC;
#include "autofuse_tiling_data.h"

inline uint32_t ALIGN_UP(uint32_t origin_size, uint32_t align_num) {
  return (0 == (origin_size & (align_num - 1))) ? origin_size  :  (origin_size + align_num - (origin_size & (align_num - 1)));
}

enum class AutoFuseTransposeType : uint8_t {
    // scene 7 shape[s0,s1] -> shape[s1,s0]
    TRANSPOSE_ND2ND_ONLY,

    // scene 8 shape[s0,s1,s2] -> shape[s1,s0,s2]
    TRANSPOSE_ND2ND_102,

    // scene9 shape[s0,s1,s2,s3]-> shape[s0,s2,s1,s3]
    TRANSPOSE_ND2ND_0213,

    // scene10 shape[s0,s1,s2,s3]->shape[s2,s1,s0,s3]
    TRANSPOSE_ND2ND_2103,

    // scene11 shape[s0,s1,s2]->shape[s0,s2,s1]
    TRANSPOSE_ND2ND_021,

    // scene12 shape[s0,s1,s2]->shape[s2,s1,s0]
    TRANSPOSE_ND2ND_210,

    // scene13 shape[s0, s1, s2, s3]->shape[s0, s3, s2, s1]
    TRANSPOSE_ND2ND_0321,

    TRANSPOSE_INVALID
};


// scene7：srcShape[s0,s1] -> dstShape[s1,s0]
void GetConfusionTranspose10TilingInfo(const ShapeInfo srcShape, const uint32_t stackBufferSize,
                                       const uint32_t typeSize, ConfusionTransposeTiling& tiling)
{
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
                                        const uint32_t typeSize, ConfusionTransposeTiling& tiling)
{
  uint32_t blockSize = ONE_BLK_SIZE / typeSize;

  /* s2为单位搬移数据快大小，s1为blockCount，单位为block */
  uint32_t blockLen = srcShape.originalShape[2] / blockSize;
  uint32_t blockCount = srcShape.originalShape[1];

  /* 输入连续，输出不连续，单位为block */
  uint32_t srcStride = 0;
  uint32_t dstStride = (srcShape.originalShape[0] * srcShape.originalShape[2] - srcShape.originalShape[2]) / blockSize;

  /* 输入连续，输出不连续，单位为数据个数 */
  uint32_t thirdDimSrcStride = srcShape.originalShape[1] * srcShape.originalShape[2];
  uint32_t thirdDimDstStride = srcShape.originalShape[2];
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
                                              const uint32_t typeSize, ConfusionTransposeTiling& tiling)
{
  uint32_t blockSize = ONE_BLK_SIZE / typeSize;

  /* s3为单位搬移数据快大小，s1为blockCount，单位为block */
  uint32_t blockLen = srcShape.originalShape[3] / blockSize;
  uint32_t blockCount = srcShape.originalShape[2];

  /* 输入连续，输出不连续，单位为block */
  uint32_t srcStride = 0;
  uint32_t dstStride = (srcShape.originalShape[1] * srcShape.originalShape[3] - srcShape.originalShape[3]) / blockSize;

  /* 输入连续，输出不连续，单位为数据个数,内层循环 */
  uint32_t thirdDimSrcStride = srcShape.originalShape[2] * srcShape.originalShape[3];
  uint32_t thirdDimDstStride = srcShape.originalShape[3];
  uint32_t thirdDimCnt = srcShape.originalShape[1];

  /* 输入连续，输出不连续，单位为数据个数,外层循环 */
  uint32_t fourthDimSrcStride = srcShape.originalShape[1] * srcShape.originalShape[2] * srcShape.originalShape[3];
  uint32_t fourthDimDstStride = srcShape.originalShape[1] * srcShape.originalShape[2] * srcShape.originalShape[3];
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
                                         const uint32_t typeSize, ConfusionTransposeTiling& tiling)
{
  uint32_t blockSize = ONE_BLK_SIZE / typeSize;

  /* s3为单位搬移数据快大小，s1为blockCount，单位为block */
  uint32_t blockLen = srcShape.originalShape[3] / blockSize;
  uint32_t blockCount = srcShape.originalShape[2];

  /* 输入连续，输出不连续，单位为block */
  uint32_t srcStride = 0;
  uint32_t dstStride = (srcShape.originalShape[0] * srcShape.originalShape[1] * srcShape.originalShape[3] - srcShape.originalShape[3]) / blockSize;

  /* 输入连续，输出不连续，单位为数据个数,内层循环 */
  uint32_t thirdDimSrcStride = srcShape.originalShape[2] * srcShape.originalShape[3];
  uint32_t thirdDimDstStride = srcShape.originalShape[0] * srcShape.originalShape[3];
  uint32_t thirdDimCnt = srcShape.originalShape[1];

  /* 输入连续，输出不连续，单位为数据个数,外层循环 */
  uint32_t fourthDimSrcStride = srcShape.originalShape[1] * srcShape.originalShape[2] * srcShape.originalShape[3];
  uint32_t fourthDimDstStride = srcShape.originalShape[3];
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
                                        const uint32_t typeSize, ConfusionTransposeTiling& tiling)
{
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
                                        const uint32_t typeSize, ConfusionTransposeTiling& tiling)
{
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
                                         const uint32_t typeSize, ConfusionTransposeTiling& tiling)
{
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

__aicore__ inline void GetConfusionTransposeTilingInfo(const ShapeInfo srcShape, const uint32_t stackBufferSize,
                                                       const uint32_t typeSize, const uint32_t transposeTypeIn, ConfusionTransposeTiling& tiling)
{
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


extern "C" __global__ __aicore__ void load_transpose_store(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);
extern "C" void GetTiling(AutofuseTilingData & tiling_data);


class E2E_LoadTransposeStore_Code : public testing::Test, public testing::WithParamInterface<std::pair<std::vector<int>, std::vector<int>>> {
};

TEST_P(E2E_LoadTransposeStore_Code, CalculateCorrect) {
  auto [test_shape, test_tiling] = GetParam();

  uint32_t block_dim = 1;
  int test_size = test_shape[0] * test_shape[1] * test_shape[2];
  int test_size_dst = test_shape[0] * test_shape[2] * test_shape[1];

  AutofuseTilingData  tiling_data;
  uint32_t *x = (uint32_t *)AscendC::GmAlloc(test_size * sizeof(uint32_t) + 32);
  uint32_t *y = (uint32_t *)AscendC::GmAlloc(test_size_dst * sizeof(uint32_t) + 32);
  uint32_t *expect = (uint32_t *)AscendC::GmAlloc(test_size_dst * sizeof(uint32_t) + 32);

  // Prepare test and expect data
  for (int i = 0; i < test_shape[0]; i++) {
    for (int j = 0; j < test_shape[1]; j++) {
      for (int k = 0; k < test_shape[2]; k++) {
        x[i * test_shape[1] * test_shape[2] + j * test_shape[2] + k]      = i * test_shape[1] * test_shape[2] + j * test_shape[2] + k;
        expect[i * test_shape[1] * test_shape[2] + k * test_shape[1] + j] = i * test_shape[1] * test_shape[2] + j * test_shape[2] + k;
      }
    }
  }

  // Launch
  tiling_data.block_dim = block_dim;
  tiling_data.s0 = test_shape[0];
  tiling_data.s1 = test_shape[1];
  tiling_data.s2 = test_shape[2];

  uint32_t srcShapeSize = tiling_data.s0 * tiling_data.s1 * tiling_data.s2;
  uint32_t srcOriShape[] = {tiling_data.s0, tiling_data.s1, tiling_data.s2};
  uint32_t srcShape[] = {tiling_data.s0, tiling_data.s1, tiling_data.s2};
  ShapeInfo srcShapeInfo(3, srcShape, 3, srcOriShape, DataFormat::ND);

  AutoFuseTransposeType transposeType = AutoFuseTransposeType::TRANSPOSE_ND2ND_021;
  const uint32_t stackBufferSize = 0;

  GetConfusionTransposeTilingInfo(srcShapeInfo, stackBufferSize, sizeof(uint32_t), (uint32_t)transposeType, tiling_data.transpose_tilingData_0);
  if (test_tiling.size() == 0U) { // tiling data 来源于tiling函数GetTiling
    GetTiling(tiling_data);
  } else { // tiling信息来源于测试用例入参
    tiling_data.block_dim = test_tiling[0];
    tiling_data.z0t_size = test_tiling[1];
    tiling_data.z0Tb_size = test_tiling[2];
  }
  tiling_data.tiling_key = 0;
  tiling_data.z0t_size = 4;

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(load_transpose_store, tiling_data.block_dim, (uint8_t *)x, (uint8_t *)y, nullptr, (uint8_t*)&tiling_data);

  // Count difference
  uint32_t diff_count = 0;
  for (int i = 0; i < test_size; i++) {
    half diff = y[i] - expect[i];
    if (diff > (half)0.0001 || diff < (half)-0.0001) {
      diff_count++;
    }
  }

  EXPECT_EQ(diff_count, 0) << " of " << test_size;
  AscendC::GmFree(x);
  AscendC::GmFree(y);
  AscendC::GmFree(expect);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2E_LoadTransposeStore_Code,
    ::testing::Values(std::pair<std::vector<int>, std::vector<int>>{{2, 37, 23}, {}}
                      ));
