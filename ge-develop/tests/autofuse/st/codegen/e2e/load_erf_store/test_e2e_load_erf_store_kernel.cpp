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
extern "C" __global__ __aicore__ void load_erf_store(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);
extern "C" void GetTiling(AutofuseTilingData& tiling_data);

class E2E_LoadErfStore_Code : public testing::Test, public testing::WithParamInterface<std::pair<std::vector<int>, std::vector<int>>> {
};

TEST_P(E2E_LoadErfStore_Code, CalculateCorrect) {
  auto [test_shape, test_tiling] = GetParam();

  uint32_t block_dim = 48;
  int test_size = test_shape[0] * test_shape[1] * test_shape[2];

  AutofuseTilingData tiling_data;
  half *x = (half *)AscendC::GmAlloc(test_size * sizeof(half) + 32);
  half *y = (half *)AscendC::GmAlloc(test_size * sizeof(half) + 32);
  half *expect = (half *)AscendC::GmAlloc(test_size * sizeof(half) + 32);

  // Prepare test and expect data
  for (int i = 0; i < test_size; i++) {
    x[i] = -1;
    expect[i] = erf(-1);
  }

  // Launch
  tiling_data.block_dim = block_dim;
  tiling_data.graph0_result0_g0_tiling_data.block_dim = block_dim;
  tiling_data.graph0_result0_g0_tiling_data.s0 = test_shape[0];
  tiling_data.graph0_result0_g0_tiling_data.s1 = test_shape[1];
  tiling_data.graph0_result0_g0_tiling_data.s2 = test_shape[2];
  tiling_data.graph0_result1_g0_tiling_data.block_dim = block_dim;
  tiling_data.graph0_result1_g0_tiling_data.s0 = test_shape[0];
  tiling_data.graph0_result1_g0_tiling_data.s1 = test_shape[1];
  tiling_data.graph0_result1_g0_tiling_data.s2 = test_shape[2];
  if (test_tiling.size() == 0U) { // tiling data 来源于tiling函数GetTiling
    GetTiling(tiling_data);
  } else { // tiling信息来源于测试用例入参
    tiling_data.block_dim = test_tiling[0];
    tiling_data.graph0_result0_g0_tiling_data.block_dim = test_tiling[0];
    tiling_data.graph0_result0_g0_tiling_data.z0z1Tb_size = test_tiling[1];
    tiling_data.graph0_result0_g0_tiling_data.z1t_size = test_tiling[2];
    tiling_data.graph0_result1_g0_tiling_data.block_dim = test_tiling[0];
    tiling_data.graph0_result1_g0_tiling_data.z0z1Tb_size = test_tiling[1];
    tiling_data.graph0_result1_g0_tiling_data.z1t_size = test_tiling[2];
  }
  tiling_data.graph0_tiling_key = 0;
  tiling_data.graph0_result0_g0_tiling_data.tiling_key = 0;
  tiling_data.graph0_result1_g0_tiling_data.tiling_key = 0;

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(load_erf_store, tiling_data.block_dim, (uint8_t *)x, (uint8_t *)y, nullptr, (uint8_t*)&tiling_data);

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

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2E_LoadErfStore_Code,
    ::testing::Values(std::pair<std::vector<int>, std::vector<int>>{{96, 16, 16}, {}},
                      // block tail
                      std::pair<std::vector<int>, std::vector<int>>{{95, 16, 16}, {}},
                      std::pair<std::vector<int>, std::vector<int>>{{97, 16, 16}, {}},
                      // tile tail
                      std::pair<std::vector<int>, std::vector<int>>{{4, 10, 16}, {2, 6, 4}},
                      // block tail and tile tail
                      std::pair<std::vector<int>, std::vector<int>>{{4, 10, 16}, {2, 7, 4}}
                      ));
