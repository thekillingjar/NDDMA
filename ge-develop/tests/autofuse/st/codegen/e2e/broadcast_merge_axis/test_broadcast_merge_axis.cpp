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
extern "C" __global__ __aicore__ void broadcast_merge_axis(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

class E2E_BroadcastMargeAxis_Code : public testing::Test, public testing::WithParamInterface<std::vector<int>>  {
};


TEST_P(E2E_BroadcastMargeAxis_Code, CalculateCorrect) {
  std::vector<int> test_shape = GetParam();

  uint32_t block_dim = 48;
  int test_size = test_shape[0] * test_shape[1] * test_shape[2];

  AutofuseTilingData tiling_data;
  half *x = (half *)AscendC::GmAlloc(test_shape[1] * test_shape[2] * sizeof(half) + 32);
  half *y = (half *)AscendC::GmAlloc(test_shape[0] * test_shape[1] * test_shape[2] * sizeof(half) + 32);
  half *expect = (half *)AscendC::GmAlloc(test_shape[0] * test_shape[1] * test_shape[2] * sizeof(half) + 32);

  // Prepare test and expect data
  for (int i = 0; i < test_shape[0]; i++) {
    for (int j = 0; j < test_shape[1]; j++) {
      for (int k = 0; k < test_shape[2]; k++) {
        x[j * test_shape[2] + k] = (half)(j*test_shape[2] + k + 1);
        expect[i * test_shape[1] * test_shape[2] + j * test_shape[2] + k] = (half)(j*test_shape[2] + k + 1);
        y[i * test_shape[1] * test_shape[2] + j * test_shape[2] + k] = (half)(j*test_shape[2] + k + 2);
      }
    }
  }

  // Launch
  tiling_data.block_dim = block_dim;
  tiling_data.s0 = test_shape[0];
  tiling_data.s1 = test_shape[1];
  tiling_data.s2 = test_shape[2];
  tiling_data.set_z0z1b_size(test_shape[0] * test_shape[1] / 48);
  tiling_data.tiling_key = 0;

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(broadcast_merge_axis, tiling_data.block_dim, (uint8_t *)x, (uint8_t *)y, nullptr, (uint8_t*)&tiling_data);

  // Count difference
  uint32_t diff_count = 0;
  for (int i = 0; i < test_size; i++) {
    half diff = y[i] - expect[i];
    if (diff > (half)0.0001 || diff < (half)-0.0001) {
      diff_count++;
    }
  }

  for (int i = 0; i < test_shape[0]; i++) {
    for (int j = 0; j < test_shape[1]; j++) {
      printf("%.2f ", (double)y[i * test_shape[1] * test_shape[2] + j]);
    }
    printf("\n");
  }

  EXPECT_EQ(diff_count, 0) << " of " << test_size;

  AscendC::GmFree(x);
  AscendC::GmFree(y);
  AscendC::GmFree(expect);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2E_BroadcastMargeAxis_Code,
    ::testing::Values(std::vector<int>{48*2, 16, 16},
                      std::vector<int>{96*2, 32, 32},
                      std::vector<int>{24*2, 8, 8}
                    ));