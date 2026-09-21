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
extern "C" __global__ __aicore__ void load_store(GM_ADDR x, GM_ADDR y, GM_ADDR y2, GM_ADDR workspace, GM_ADDR tiling);
extern "C" void GetTiling(AutofuseTilingData& tiling_data);

class E2E_LoadBroadcastShapeOneStoreTest_Code : public testing::Test, public testing::WithParamInterface<std::vector<int>> {
};

TEST_P(E2E_LoadBroadcastShapeOneStoreTest_Code, CalculateCorrect) {

  uint32_t block_dim = 48;
  uint32_t s0 = GetParam()[0];
  uint32_t s1 = GetParam()[1];
  uint32_t out_size = s0 * s1;

  AutofuseTilingData tiling_data;
  half *x = (half *)AscendC::GmAlloc(s0 * sizeof(half) + 32);
  half *expect1 = (half *)AscendC::GmAlloc(s0 * sizeof(half) + 32);
  half *y1 = (half *)AscendC::GmAlloc(s0 * sizeof(half) + 32);
  half *expect2 = (half *)AscendC::GmAlloc(out_size * sizeof(half) + 32);
  half *y2 = (half *)AscendC::GmAlloc(out_size * sizeof(half) + 32);

  // Prepare test and expect data
  srand(0);
  for (int i = 0; i < s0; i++) {
    x[i] = rand() / (double)RAND_MAX;
    expect1[i] = x[i];
  }
  

  for (int i = 0; i < s0; i++) {
    for (int j = 0; j < s1; j++) {
      expect2[i * s1 + j] = x[i];
    }
  }

  // Launch
  tiling_data.block_dim = block_dim;
  tiling_data.s0 = s0;
  tiling_data.s1 = s1;
  tiling_data.tiling_key = 0;

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(load_store, tiling_data.block_dim, (uint8_t *)x, (uint8_t *)y1, (uint8_t *)y2, nullptr, (uint8_t*)&tiling_data);

  // Count difference
  uint32_t diff_count = 0;
  for (int i = 0; i < out_size; i++) {
    half diff = y2[i] - expect2[i];
    if (diff > (half)0.0001 || diff < (half)-0.0001) {
        diff_count++;
    }
  }

  for (int i = 0; i < s0; i++) {
    half diff = y1[i] - expect1[i];
    if (diff > (half)0.0001 || diff < (half)-0.0001) {
        diff_count++;
    }
  }

  EXPECT_EQ(diff_count, 0) << " of " << out_size << " and " << s0;

  AscendC::GmFree(x);
  AscendC::GmFree(y1);
  AscendC::GmFree(expect1);
  AscendC::GmFree(y2);
  AscendC::GmFree(expect2);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2E_LoadBroadcastShapeOneStoreTest_Code,
    ::testing::Values(std::vector<int>{8, 128},
                      std::vector<int>{15 , 128},
                      std::vector<int>{15 , 63}));
