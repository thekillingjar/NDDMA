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
extern "C" __global__ __aicore__ void discrete_store_merge_axis(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);
extern "C" void GetTiling(AutofuseTilingData& tiling_data);

class E2E_DiscreteStoreMergeAxis_Code : public testing::Test, public testing::WithParamInterface<std::pair<std::vector<int>, std::vector<int>>> {
};

TEST_P(E2E_DiscreteStoreMergeAxis_Code, CalculateCorrect) {
  auto [test_shape, test_tiling] = GetParam();

  uint32_t block_dim = 48;
  int test_size = test_shape[0] * test_shape[1] * test_shape[2] * test_shape[3] * test_shape[4];

  AutofuseTilingData tiling_data;
  half *x = (half *)AscendC::GmAlloc(test_size * sizeof(half));
  half *y = (half *)AscendC::GmAlloc(2*test_size * sizeof(half));
  half *expect = (half *)AscendC::GmAlloc(2*test_size * sizeof(half));

  // Prepare test and expect data
  for (int i = 0; i < test_shape[0]; i++) {
    for (int j = 0; j < test_shape[1]; j++) {
      for (int k = 0; k < test_shape[2]; k++) {
        for (int m = 0; m < test_shape[3]; m++) {
          for (int n = 0; n < test_shape[4]; n++) {
            auto index = i * (test_shape[1] * test_shape[2] * test_shape[3] * test_shape[4]) +
              j * (test_shape[2] * test_shape[3] * test_shape[4]) +
              k * (test_shape[3] * test_shape[4]) + m * (test_shape[4]) + n;
            x[index] = n;
          }
        }
      }
    }
  }

  for (int i = 0; i < test_shape[0]; i++) {
    for (int j = 0; j < test_shape[1]; j++) {
      for (int k = 0; k < test_shape[2]; k++) {
        for (int m = 0; m < 2 * test_shape[3]; m++) {
          for (int n = 0; n < test_shape[4]; n++) {
            auto index = i * (2*test_shape[1] * test_shape[2] * test_shape[3] * test_shape[4]) +
              j * (2 * test_shape[2] * test_shape[3] * test_shape[4]) +
              k * (2 * test_shape[3] * test_shape[4]) + m * (test_shape[4]) + n;
            if (m < test_shape[3]) {
              expect[index] = n;
            } else {
              expect[index] = 0xEF;
            }
          }
        }
      }
    }
  }

  // Launch
  tiling_data.block_dim = block_dim;
  tiling_data.s0 = test_shape[0];
  tiling_data.s1 = test_shape[1];
  tiling_data.s2 = test_shape[2];
  tiling_data.s3 = test_shape[3];
  tiling_data.s4 = test_shape[4];
  // tiling信息由用例输入参数确定
  if (test_tiling.size() == 0U) { // tiling data 来源于tiling函数GetTiling
    GetTiling(tiling_data);
  } else {                        // tiling信息来源于测试用例入参
    tiling_data.block_dim = test_tiling[0];
    tiling_data.z0b_size = test_tiling[1];
  }
  tiling_data.tiling_key = 0;

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(discrete_store_merge_axis, tiling_data.block_dim, (uint8_t *)x, (uint8_t *)y, nullptr, (uint8_t*)&tiling_data);

  // Count difference
  uint32_t diff_count = 0;
  for (int i = 0; i < test_shape[0]; i++) {
    for (int j = 0; j < test_shape[1]; j++) {
      for (int k = 0; k < test_shape[2]; k++) {
        for (int m = 0; m < 2 * test_shape[3]; m++) {
          for (int n = 0; n < test_shape[4]; n++) {
            auto index = i * (2 * test_shape[1] * test_shape[2] * test_shape[3] * test_shape[4]) +
              j * (2 * test_shape[2] * test_shape[3] * test_shape[4]) +
              k * (2 * test_shape[3] * test_shape[4]) + m * (test_shape[4]) + n;
            if (m < test_shape[3]) {
              half diff = y[index] - expect[index];
              if (diff > (half)0.0001 || diff < (half)-0.0001) {
                diff_count++;
              }
            }
          }
        }
      }
    }
  }

  EXPECT_EQ(diff_count, 0) << " of " << test_size;

  AscendC::GmFree(x);
  AscendC::GmFree(y);
  AscendC::GmFree(expect);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2E_DiscreteStoreMergeAxis_Code,
    ::testing::Values(
                      std::pair<std::vector<int>, std::vector<int>>{{32*5, 2, 2, 4, 9}, {32, 5}},
                      std::pair<std::vector<int>, std::vector<int>>{{32*16, 2, 5, 10, 100}, {32, 16}},
                      std::pair<std::vector<int>, std::vector<int>>{{32*15, 2, 3, 100, 5}, {32, 15}}
                      ));


