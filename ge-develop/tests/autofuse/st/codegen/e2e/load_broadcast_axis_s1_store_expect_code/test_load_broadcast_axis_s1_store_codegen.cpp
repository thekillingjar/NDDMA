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

#include "autofuse_tiling_data.h"

extern "C" __global__ __aicore__ void load_broadcast_axis_s1_store(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR t);
extern "C" void GetTiling(AutofuseTilingData& tiling_data);

class E2E_Load_Broadcast_Axis_S1_Store : public testing::Test,
                               public testing::WithParamInterface<std::vector<int>> {};

TEST_P(E2E_Load_Broadcast_Axis_S1_Store, BroadCast_Axis_s1_fp32) {
  uint32_t block_dim = 16;
  uint32_t s0 = GetParam()[0];
  uint32_t s1 = GetParam()[1];

  AutofuseTilingData tiling_data;
  float* x = (float*)AscendC::GmAlloc(s1 * sizeof(float));
  float* y = (float*)AscendC::GmAlloc(s0*s1 * sizeof(float));
  float* expect = (float*)AscendC::GmAlloc(s0*s1 * sizeof(float));

  // Prepare test and expect data
  for (int i = 0; i < s1; i++) {
    x[i] = i;
    for (int j = 0; j < s0; j++) {
      expect[j * s1 + i] = i;
    }
  }

  // Launch
  tiling_data.block_dim = block_dim;
  tiling_data.s0 = s0;
  tiling_data.s1 = s1;
  tiling_data.tiling_key = 0;
  GetTiling(tiling_data);

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(load_broadcast_axis_s1_store, tiling_data.block_dim, (uint8_t*)x, (uint8_t*)y, nullptr, (uint8_t*)&tiling_data);

  // Count difference
  uint32_t diff_count = 0;
  for (int i = 0; i < s0; i++) {
    for (int j = 0; j < s1; j++) {
      if (y[i * s1 + j] != expect[i * s1 + j]) {
        diff_count++;
      }
    }
  }

  EXPECT_EQ(diff_count, 0) << " of " << std::to_string(s0 * s1);

  AscendC::GmFree(x);
  AscendC::GmFree(y);
  AscendC::GmFree(expect);
}

INSTANTIATE_TEST_SUITE_P(M32_K_BlockAlign, E2E_Load_Broadcast_Axis_S1_Store,
   ::testing::Values(
       std::vector<int>{16*4*8, 16},
       std::vector<int>{16*4*16, 16},
       std::vector<int>{16*4*32, 16},
       std::vector<int>{16*4*8, 64},
       std::vector<int>{16*4*8, 16}
   ));
