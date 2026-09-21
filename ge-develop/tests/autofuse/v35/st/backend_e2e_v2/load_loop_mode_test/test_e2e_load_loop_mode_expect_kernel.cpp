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

extern "C" __global__ __aicore__ void load_loop_mode_test(GM_ADDR data0, GM_ADDR data1, GM_ADDR output,
                                                          GM_ADDR workspace, GM_ADDR gm_tiling_data);
extern "C" int64_t AutofuseTiling(AutofuseTilingData *tiling, uint32_t *workspaceSize, uint64_t *blockDim,
                                  uint32_t aiv_num, uint32_t ub_size);

namespace {
class E2E_LoadLoopMode_Code : public testing::Test, public testing::WithParamInterface<std::vector<int>> {};

TEST_P(E2E_LoadLoopMode_Code, CalculateCorrect) {
  auto test_shape = GetParam();

  uint64_t block_dim = 48;

  int input0_size = test_shape[0] * test_shape[1] * test_shape[2] * 3;
  int input1_size = test_shape[0] * test_shape[1] * test_shape[2];
  int output_size = test_shape[0] * test_shape[1] * test_shape[2];

  AutofuseTilingData tiling_data;
  float *input0 = (float *)AscendC::GmAlloc(input0_size * sizeof(float) + 32);
  float *input1 = (float *)AscendC::GmAlloc(input1_size * sizeof(float) + 32);
  float *y = (float *)AscendC::GmAlloc(output_size * sizeof(float) + 32);
  float *expect = (float *)AscendC::GmAlloc(output_size * sizeof(float) + 32);

  // Prepare test and expect data
  srand(1);
  for (int i = 0; i < input0_size; i++) {
    // input0[i] = rand() / (double)RAND_MAX;
    input0[i] = 1.0f;
  }

  for (int i = 0; i < input1_size; i++) {
    // input1[i] = rand() / (double)RAND_MAX;
    input1[i] = 2.0f;
  }

  for (int i0 = 0; i0 < test_shape[0]; ++i0) {
    for (int i1 = 0; i1 < test_shape[1]; ++i1) {
      for (int i2 = 0; i2 < test_shape[2]; ++i2) {
        expect[i0 * test_shape[1] * test_shape[2] + i1 * test_shape[2] + i2] =
            input0[i0 * test_shape[1] * test_shape[2] * 3 + i1 * test_shape[2] * 2 + i2] +
            input1[i0 * test_shape[1] * test_shape[2] + i1 * test_shape[2] + i2];
      }
    }
  }

  // Launch
  uint32_t ws_size = 0;
  AutofuseTiling(&tiling_data, &ws_size, &block_dim, 48, 192 * 1024);
  printf("tiling key: %d, core_num: %d\n", tiling_data.tiling_key, tiling_data.block_dim);

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(load_loop_mode_test, tiling_data.block_dim, (uint8_t *)input0, (uint8_t *)input1, (uint8_t *)y, nullptr,
              (uint8_t *)&tiling_data);

  // Count difference
  uint32_t diff_count = 0;
  for (int i0 = 0; i0 < test_shape[0]; ++i0) {
    for (int i1 = 0; i1 < test_shape[1]; ++i1) {
      for (int i2 = 0; i2 < test_shape[2]; ++i2) {
        auto diff = (double)(expect[i0 * test_shape[1] * test_shape[2] + i1 * test_shape[2] + i2] -
                             y[i0 * test_shape[1] * test_shape[2] + i1 * test_shape[2] + i2]);
        if (diff < -1e-5 || diff > 1e-5) {
          // printf("expect:%f, y:%f\n", expect[i0 * test_shape[1] * test_shape[2] + i1 * test_shape[2] + i2],
          //        y[i0 * test_shape[1] * test_shape[2] + i1 * test_shape[2] + i2]);
          diff_count++;
        }
      }
    }
  }

  // EXPECT_EQ(diff_count, 0) << " of " << output_size;

  AscendC::GmFree(input0);
  AscendC::GmFree(input1);
  AscendC::GmFree(y);
  AscendC::GmFree(expect);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2E_LoadLoopMode_Code, ::testing::Values(std::vector<int>{4, 8, 16}));

}  // namespace
