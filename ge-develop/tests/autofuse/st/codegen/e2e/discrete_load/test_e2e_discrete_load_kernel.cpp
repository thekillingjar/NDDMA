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
extern "C" __global__ __aicore__ void discrete_load(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);
extern "C" void GetTiling(AutofuseTilingData& tiling_data);

class E2E_DiscreteLoad_Code : public testing::Test, public testing::WithParamInterface<std::pair<std::vector<int>, std::vector<int>>> {
};

TEST_P(E2E_DiscreteLoad_Code, CalculateCorrect) {
  auto [test_shape, test_tiling] = GetParam();

  uint32_t block_dim = 1;
  int test_size_x = test_shape[0] * test_shape[1] * test_shape[2] * 4;
  int test_size_y = test_shape[0] * test_shape[1] * test_shape[2];

  AutofuseTilingData tiling_data;
  float *x = (float *)AscendC::GmAlloc(test_size_x * sizeof(float));
  float *y = (float *)AscendC::GmAlloc(test_size_y * sizeof(float));
  float *expect = (float *)AscendC::GmAlloc(test_size_y * sizeof(float));

  // Prepare test and expect data
  for (int i = 0; i < test_shape[0]; i++) {
    for (int j = 0; j < test_shape[1] * 2; j++) {
      for (int k = 0; k < test_shape[2] * 2; k++) {
        x[i * (test_shape[1] * test_shape[2] * 4) + j * test_shape[2] * 2 + k] = k;
      }
    }
  }

  for (int i = 0; i < test_shape[0]; i++) {
    for (int j = 0; j < test_shape[1]; j++) {
      for (int k = 0; k < test_shape[2]; k++) {
        if (k < test_shape[2]) {
          expect[i * (test_shape[1] * test_shape[2]) + j * test_shape[2] + k] = k;
        }
      }
    }
  }

  // Launch
  tiling_data.block_dim = block_dim;
  tiling_data.s0 = test_shape[0];
  tiling_data.s1 = test_shape[1];
  tiling_data.s2 = test_shape[2];
  // tiling信息由用例输入参数确定
  if (test_tiling.size() == 0U) { // tiling data 来源于tiling函数GetTiling
    GetTiling(tiling_data);
  } else {                        // tiling信息来源于测试用例入参
    tiling_data.block_dim = test_tiling[0];
    tiling_data.z0Tb_size = test_tiling[1];
    tiling_data.z0t_size = test_tiling[2];
  }
  tiling_data.tiling_key = 0;

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(discrete_load, tiling_data.block_dim, (uint8_t *)x, (uint8_t *)y, nullptr, (uint8_t*)&tiling_data);

  // Count difference
  uint32_t diff_count = 0;
  for (int i = 0; i < test_shape[0]; i++) {
    for (int j = 0; j < test_shape[1]; j++) {
      for (int k = 0; k < test_shape[2]; k++) {
        auto index = i * (test_shape[1] * test_shape[2]) + j * test_shape[2] + k;
        float diff = y[index] - expect[index];
        if (diff > (float)0.0001 || diff < (float)-0.0001) {
            diff_count++;
        }
      }
    }
  }

  EXPECT_EQ(diff_count, 0) << " of " << test_size_y;

  AscendC::GmFree(x);
  AscendC::GmFree(y);
  AscendC::GmFree(expect);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2E_DiscreteLoad_Code,
    ::testing::Values(
                      std::pair<std::vector<int>, std::vector<int>>{{2, 4, 8}, {1, 1, 2}} // block尾块 尾轴32对齐padding为size32
                      ));


