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
#include <iostream>
#include "tikicpulib.h"

#include "autofuse_tiling_data.h"
extern "C" __global__ __aicore__ void gather_abs_store(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);
extern "C" void GetTiling(AutofuseTilingData& tiling_data);

class E2E_GatherAbsStore_Code : public testing::Test, public testing::WithParamInterface<std::pair<std::vector<int>, std::vector<int>>> {
};

TEST_P(E2E_GatherAbsStore_Code, CalculateCorrect) {
  auto [test_shape, test_tiling] = GetParam();

  uint32_t block_dim = 20;
  int test_size = test_shape[1] * test_shape[2];

  AutofuseTilingData tiling_data;
  float *x1 = (float *)AscendC::GmAlloc(test_shape[0] * sizeof(float));
  int64_t *x2 = (int64_t *)AscendC::GmAlloc(test_size * sizeof(int64_t));
  float *y = (float *)AscendC::GmAlloc(test_size * sizeof(float));
  float *expect = (float *)AscendC::GmAlloc(test_size * sizeof(float));

  // Prepare test and expect data
  for (int i = 0; i < test_size; i++) {
    x2[i] = (int64_t)1;
    expect[i] = (float)1;
  }

  for (int i = 0; i < test_shape[0]; ++i) {
    x1[i] = (float)-1;
  }

  // Launch
  tiling_data.block_dim = block_dim;
  tiling_data.s0 = test_shape[0];
  tiling_data.s1 = test_shape[1];
  tiling_data.s2 = test_shape[2];
  if (test_tiling.size() == 0U) { // tiling data 来源于tiling函数GetTiling
    GetTiling(tiling_data);
  } else { // tiling信息来源于测试用例入参
    tiling_data.block_dim = test_tiling[0];
    tiling_data.z1z2Tb_size = test_tiling[1];
    tiling_data.z1z2t_size = test_tiling[2];
  }
  tiling_data.tiling_key = 0;

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(gather_abs_store, tiling_data.block_dim, (uint8_t *)x1, (uint8_t *)x2, (uint8_t *)y, nullptr, (uint8_t*)&tiling_data);

  // Count difference
  uint32_t diff_count = 0;
  for (int i = 0; i < test_size; i++) {
    half diff = y[i] - expect[i];
    if (diff > (half)0.0001 || diff < (half)-0.0001) {
      diff_count++;
    }
  }
  EXPECT_EQ(diff_count, 0) << " of " << test_size;

  AscendC::GmFree(x1);
  AscendC::GmFree(x2);
  AscendC::GmFree(y);
  AscendC::GmFree(expect);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2E_GatherAbsStore_Code,
                         ::testing::Values(std::pair<std::vector<int>, std::vector<int>>{{4000, 100, 100}, {}},
                                           // tile tail
                                           std::pair<std::vector<int>, std::vector<int>>{{4000, 100, 100}, {20, 100, 100}}
                                           ));