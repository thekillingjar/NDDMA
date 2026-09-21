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
extern "C" __global__ __aicore__ void load_scalar_clip_store(GM_ADDR x1, GM_ADDR x2, GM_ADDR x3, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);
extern "C" void GetTiling(AutofuseTilingData& tiling_data);

class E2E_LoadScalarClipStore_Code : public testing::Test, public testing::WithParamInterface<std::vector<int>> {
};

TEST_P(E2E_LoadScalarClipStore_Code, CalculateCorrect) {
  auto test_shape = GetParam();

  uint32_t block_dim = 1;
  int test_size = test_shape[0] * test_shape[1];

  AutofuseTilingData tiling_data;
  float *x1 = (float *)AscendC::GmAlloc(sizeof(float) + 32);
  float *x2 = (float *)AscendC::GmAlloc(sizeof(float) + 32);
  float *x3 = (float *)AscendC::GmAlloc(sizeof(float) + 32);
  float *y = (float *)AscendC::GmAlloc(test_size * sizeof(float) + 32);
  float *expect = (float *)AscendC::GmAlloc(test_size * sizeof(float) + 32);

  // Prepare test and expect data
  srand(1);
  x1[0] = rand() / (double)RAND_MAX;
  x2[0] = rand() / (double)RAND_MAX;
  x3[0] = rand() / (double)RAND_MAX;
  if (x2[0] > x3[0]) {
    auto tmp = x3[0];
    x3[0] = x2[0];
    x2[0] = tmp;
  }
  if (x1[0] < x2[0]) {
    expect[0] = x2[0];
  } else if (x1[0] > x3[0]){
    expect[0] = x3[0];
  } else {
    expect[0] = x1[0];
  }
  for (int i = 0; i < test_size; i++) {
    expect[i] = expect[0];
  }

  // Launch
  tiling_data.block_dim = block_dim;
  tiling_data.s0 = test_shape[0];
  tiling_data.s1 = test_shape[1];
  tiling_data.tiling_key = 0;

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(load_scalar_clip_store, tiling_data.block_dim, (uint8_t*)x1, (uint8_t*)x2, (uint8_t*)x3, (uint8_t*)y, nullptr, (uint8_t*)&tiling_data);

  // Count difference
  uint32_t diff_count = 0;
  for (int i = 0; i < test_size; i++) {
    if (fabs(y[i] - expect[i]) > 1e-6) {
      diff_count++;
    }
  }
  EXPECT_EQ(diff_count, 0) << " of " << test_size;

  AscendC::GmFree(x1);
  AscendC::GmFree(x2);
  AscendC::GmFree(x3);
  AscendC::GmFree(y);
  AscendC::GmFree(expect);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2E_LoadScalarClipStore_Code,
    ::testing::Values(std::vector<int>{4, 4}));
