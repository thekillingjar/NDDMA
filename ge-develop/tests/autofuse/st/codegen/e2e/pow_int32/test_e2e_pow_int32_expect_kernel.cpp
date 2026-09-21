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
#include <cmath>
#include <random>

#include "autofuse_tiling_data.h"
extern "C" __global__ __aicore__ void pow_int32(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);
extern "C" void GetTiling(AutofuseTilingData& tiling_data);

class E2E_PowInt32_Code : public testing::Test, public testing::WithParamInterface<std::vector<int>> {
};

TEST_P(E2E_PowInt32_Code, CalculateCorrect) {
  auto test_shape = GetParam();

  uint32_t block_dim = 48;
  int test_size = test_shape[0];

  AutofuseTilingData tiling_data;
  int32_t *input0 = (int32_t *)AscendC::GmAlloc(test_size * sizeof(int32_t) + 32);
  int32_t *input1 = (int32_t *)AscendC::GmAlloc(test_size * sizeof(int32_t) + 32);
  int32_t *expect = (int32_t *)AscendC::GmAlloc(test_size * sizeof(int32_t) + 32);
  int32_t *y = (int32_t *)AscendC::GmAlloc(test_size * sizeof(int32_t) + 32);

  // Prepare test and expect data
  int input_range = 3;
  std::mt19937 eng(1);                                         // Seed the generator
  std::uniform_int_distribution distr(0, input_range);  // Define the range
  for (int i = 0; i < test_size; i++) {
    auto src1 = distr(eng);  // Use the secure random number generator
    auto src2 = distr(eng);  // Use the secure random number generator
    input0[i] = src1;
    input1[i] = src2;
    expect[i] = pow(src1, src2);
  }

  // Launch
  tiling_data.block_dim = block_dim;
  tiling_data.s0 = test_shape[0];
  tiling_data.tiling_key = 0;
  GetTiling(tiling_data);

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(pow_int32, tiling_data.block_dim, (uint8_t *)input0, (uint8_t *)input1, (uint8_t *)y, nullptr, (uint8_t*)&tiling_data);

  // Count difference
  uint32_t diff_count = 0;
  for (int i = 0; i < test_size; i++) {
    if (y[i] != expect[i]) {
      diff_count++;
    }
  }

  EXPECT_EQ(diff_count, 0) << " of " << test_size;

  AscendC::GmFree(input0);
  AscendC::GmFree(input1);
  AscendC::GmFree(y);
  AscendC::GmFree(expect);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2E_PowInt32_Code,
    ::testing::Values(std::vector<int>{2 * 8 * 8},
                      std::vector<int>{8 * 16 * 16},
                      std::vector<int>{48 * 16 * 16}
                      ));
