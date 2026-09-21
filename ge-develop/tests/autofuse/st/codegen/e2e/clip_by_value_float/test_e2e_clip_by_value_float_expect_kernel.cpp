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

#include "autofuse_tiling_data.h"
extern "C" __global__ __aicore__ void clip_by_value_float(GM_ADDR x1, GM_ADDR x2, GM_ADDR x3, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);
extern "C" void GetTiling(AutofuseTilingData& tiling_data);

class E2E_ClipByValueFloat_Code : public testing::Test, public testing::WithParamInterface<std::vector<int>> {
};

TEST_P(E2E_ClipByValueFloat_Code, CalculateCorrect) {
  auto test_shape = GetParam();

  uint32_t block_dim = 48;
  int test_size = test_shape[0];

  AutofuseTilingData tiling_data;
  float *input = (float *)AscendC::GmAlloc(test_size * sizeof(float) + 32);
  float *srcMin = (float *)AscendC::GmAlloc(test_size * sizeof(float) + 32);
  float *srcMax = (float *)AscendC::GmAlloc(test_size * sizeof(float) + 32);
  float *expect = (float *)AscendC::GmAlloc(test_size * sizeof(float) + 32);
  float *y = (float *)AscendC::GmAlloc(test_size * sizeof(float) + 32);

  // Prepare test and expect data
  srand(1);
  for (int i = 0; i < test_size; i++) {
    input[i] = rand() / (double)RAND_MAX;
    auto src1 = rand() / (double)RAND_MAX;
    auto src2 = rand() / (double)RAND_MAX;
    if (src1 < src2) {
      srcMin[i] = src1;
      srcMax[i] = src2;
    } else {
      srcMin[i] = src2;
      srcMax[i] = src1;
    }

    if (input[i] < srcMin[i]) {
      expect[i] = srcMin[i];
    } else if (input[i] > srcMax[i]){
      expect[i] = srcMax[i];
    } else {
      expect[i] = input[i];
    }

  }

  // Launch
  tiling_data.block_dim = block_dim;
  tiling_data.s0 = test_shape[0];
  tiling_data.tiling_key = 0;
  GetTiling(tiling_data);

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(clip_by_value_float, tiling_data.block_dim, (uint8_t *)input, (uint8_t *)srcMin, (uint8_t *)srcMax, (uint8_t *)y, nullptr, (uint8_t*)&tiling_data);

  // Count difference
  uint32_t diff_count = 0;
  for (int i = 0; i < test_size; i++) {
    if (y[i] != expect[i]) {
      diff_count++;
    }
  }

  EXPECT_EQ(diff_count, 0) << " of " << test_size;

  AscendC::GmFree(input);
  AscendC::GmFree(srcMin);
  AscendC::GmFree(srcMax);
  AscendC::GmFree(y);
  AscendC::GmFree(expect);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2E_ClipByValueFloat_Code,
    ::testing::Values(std::vector<int>{2 * 8 * 8},
                      std::vector<int>{8 * 16 * 16},
                      std::vector<int>{48 * 16 * 16}
                      ));
