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
#include <cmath>
#include "tikicpulib.h"

#include "autofuse_tiling_data.h"
extern "C" __global__ __aicore__ void binary_api_scalar_test(GM_ADDR x0, GM_ADDR x1, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);
extern "C" int64_t AutofuseTiling(int s0, int s1, AutofuseTilingData* tiling, uint32_t* workspaceSize, uint64_t *blockDim, uint32_t aiv_num, uint32_t ub_size);

class E2E_BackendAddAbsHalfScalar_Code : public testing::Test, public testing::WithParamInterface<std::vector<int>> {
};

TEST_P(E2E_BackendAddAbsHalfScalar_Code, CalculateCorrect) {
  auto test_shape = GetParam();
  uint64_t block_dim = 48;
  int test_size = test_shape[0] * test_shape[1];

  AutofuseTilingData tiling_data;
  half* x0 = (half *)AscendC::GmAlloc(test_size * sizeof(half) + 32);
  half* x1 = (half *)AscendC::GmAlloc(test_size * sizeof(half) + 32);
  half* y = (half *)AscendC::GmAlloc(test_size * sizeof(half) + 32);
  half *expect = (half *)AscendC::GmAlloc(test_size * sizeof(half) + 32);

  // Prepare test and expect data
  srand(1);
  for (int i = 0; i < test_size; i++) {
    x0[i] = (half)(0.5);
    x1[i] = (half)(0.7);
    expect[i] = (0.5 + 0.7) + (0.5 * 0.7) + (0.5 > 0.7 ? 0.5 : 0.7) + (0.5 < 0.7 ? 0.5 : 0.7) + (0.5 && 0.7) + (0.5 || 0.7);
  }

  // Launch
  uint32_t ws_size = 0;
  AutofuseTiling(test_shape[0], test_shape[1], &tiling_data, &ws_size, &block_dim, 48, 192*1024);
  printf("tiling key: %d, core_num: %d\n", tiling_data.tiling_key, tiling_data.block_dim);

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(binary_api_scalar_test, tiling_data.block_dim, (uint8_t *)x0, (uint8_t *)x1, (uint8_t *)y, nullptr, (uint8_t*)&tiling_data);

  // Count difference
  uint32_t diff_count = 0;
  for (int i = 0; i < test_size; i++) {
    printf("y[%d]: %f, expect[%d]: %f\n", i, static_cast<float>(y[i]), i, static_cast<float>(expect[i]));
    if (y[i] != expect[i]) {
      diff_count++;
    }
  }

  EXPECT_EQ(diff_count, 0) << " of " << test_size;

  AscendC::GmFree(x0);
  AscendC::GmFree(x1);
  AscendC::GmFree(y);
  AscendC::GmFree(expect);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2E_BackendAddAbsHalfScalar_Code,
    ::testing::Values(std::vector<int>{1, 1}));
