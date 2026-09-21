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

extern "C" __global__ __aicore__ void broadcast_force_merge(GM_ADDR x0, GM_ADDR x1, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling_data);
extern "C" void GetTiling(AutofuseTilingData &tiling_data);

class E2E_Broadcast_Force_Merge : public testing::Test, public testing::WithParamInterface<std::pair<std::vector<int>, std::vector<int>>> {};

TEST_P(E2E_Broadcast_Force_Merge, broadcast_force_merge_0) {
  auto [test_shape, test_tiling] = GetParam();

  uint32_t block_dim = 48;
  uint32_t s0 = test_shape[0];
  uint32_t s1 = test_shape[1];

  AutofuseTilingData tiling_data;
  float *x0 = (float *)AscendC::GmAlloc(s0 * sizeof(float));
  float *x1 = (float *)AscendC::GmAlloc(s0 * s1 * sizeof(float));
  float *y = (float *)AscendC::GmAlloc(s0 * s1 * sizeof(float));
  float *expect = (float *)AscendC::GmAlloc(s0 * s1 * sizeof(float));

  for (int i = 0; i < s0; i++) {
    x0[i] = (float)i;
  }

  for (int i = 0; i < s0; i++) {
    for (int j = 0; j < s1; j++) {
      x1[i * s1 + j] = float(i * s1 + j);
      expect[i * s1 + j] = x1[i * s1 + j] * x0[i];
    }
  }

  tiling_data.block_dim = block_dim;
  tiling_data.s0 = 8;
  tiling_data.s1 = 8;
  // tiling信息由用例输入参数确定
  if (test_tiling.size() == 0U) {  // tiling data 来源于tiling函数GetTiling
    GetTiling(tiling_data);
  } else {  // tiling信息来源于测试用例入参
    tiling_data.block_dim = test_tiling[0];
    tiling_data.z0z1Tb_size = test_tiling[1];
    tiling_data.z1t_size = test_tiling[2];
  }
  tiling_data.tiling_key = 0;

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(broadcast_force_merge, tiling_data.block_dim, (uint8_t *)x0, (uint8_t *)x1, (uint8_t *)y, nullptr,
              (uint8_t *)&tiling_data);
  // Count difference
  uint32_t diff_count = 0;
  for (int i = 0; i < s0 * s1; i++) {
    half diff = y[i] - expect[i];
    if (diff > (half)0.0001 || diff < (half)-0.0001) {
      diff_count++;
    }
  }

  EXPECT_EQ(diff_count, 0) << " of " << (s0 * s1);

  AscendC::GmFree(x0);
  AscendC::GmFree(x1);
  AscendC::GmFree(y);
  AscendC::GmFree(expect);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2E_Broadcast_Force_Merge,
                         ::testing::Values(std::pair<std::vector<int>, std::vector<int>>{{8, 8}, {32, 1, 2}},
                                           std::pair<std::vector<int>, std::vector<int>>{{8, 8}, {16, 1, 4}},
                                           std::pair<std::vector<int>, std::vector<int>>{{8, 8}, {8, 3, 3}}
                                          ));