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
extern "C" __global__ __aicore__ void load_compare_int64_tensor_ne_store(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);
extern "C" void GetTiling(AutofuseTilingData& tiling_data);

class E2E_LoadCompareInt64TensorNeStore_Code : public testing::Test, public testing::WithParamInterface<std::vector<int>> {
};

TEST_P(E2E_LoadCompareInt64TensorNeStore_Code, CalculateCorrect) {
  auto test_shape = GetParam();

  uint32_t block_dim = 48;
  int test_size = test_shape[0];

  AutofuseTilingData tiling_data;
  int64_t *x1 = (int64_t *)AscendC::GmAlloc(test_size * sizeof(int64_t) + 32);
  int64_t *x2 = (int64_t *)AscendC::GmAlloc(test_size * sizeof(int64_t) + 32);
  uint8_t *expect = (uint8_t *)AscendC::GmAlloc(test_size * sizeof(uint8_t) + 32);
  uint8_t *y = (uint8_t *)AscendC::GmAlloc(test_size * sizeof(uint8_t) + 32);

  // Prepare test and expect data
  srand(0);
  for (int i = 0; i < test_size; i++) {
    x1[i] = rand() / (double)RAND_MAX;
    x2[i] = x1[i];
    if (i % 2) {
      x2[i] += 0.1;
    }
    expect[i] = x1[i] != x2[i];
  }

  // Launch
  tiling_data.block_dim = block_dim;
  tiling_data.s0 = test_shape[0];
  tiling_data.tiling_key = 0;
  GetTiling(tiling_data);

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(load_compare_int64_tensor_ne_store, tiling_data.block_dim, (uint8_t *)x1, (uint8_t *)x2, (uint8_t *)y, nullptr, (uint8_t*)&tiling_data);

  // Count difference
  uint32_t diff_count = 0;
  for (int i = 0; i < test_size; i++) {
    if (y[i] != expect[i]) {
      diff_count++;
    }
  }

  EXPECT_EQ(diff_count, 0) << " of " << test_size;

  AscendC::GmFree(x1);
  AscendC::GmFree(x2);
  AscendC::GmFree(y);
  AscendC::GmFree(expect);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2E_LoadCompareInt64TensorNeStore_Code,
    ::testing::Values(std::vector<int>{2 * 8 * 8},
                      std::vector<int>{8 * 16 * 16},
                      std::vector<int>{48 * 16 * 16}
                      ));
