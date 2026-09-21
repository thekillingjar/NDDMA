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

extern "C" __global__ __aicore__ void concat_same_tail_dim(GM_ADDR x1,
                                                           GM_ADDR x2,
                                                           GM_ADDR y,
                                                           GM_ADDR workspace,
                                                           GM_ADDR tiling);
extern "C" void GetTiling(AutofuseTilingData &tiling_data);

class E2E_ConcatSameTailDim_Code
    : public testing::Test, public testing::WithParamInterface<std::pair<std::vector<int>, std::vector<int>>> {
 protected:
  using DType = float;
};

TEST_P(E2E_ConcatSameTailDim_Code, ConcatSameTailDim) {
  auto [test_shape, test_tiling] = GetParam();
  uint32_t block_dim = 1;
  int test_size_x1 = test_shape[0] * test_shape[1];
  int test_size_x2 = test_shape[2] * test_shape[3];
  int test_size_y = test_shape[0] * (test_shape[1] + test_shape[3]);

  AutofuseTilingData tiling_data;
  DType *x1 = (DType *) AscendC::GmAlloc(test_size_x1 * sizeof(DType));
  DType *x2 = (DType *) AscendC::GmAlloc(test_size_x2 * sizeof(DType));
  DType *y = (DType *) AscendC::GmAlloc(test_size_y * sizeof(DType));
  DType *expect = (DType *) AscendC::GmAlloc(test_size_y * sizeof(DType));
  memset_s(y, test_size_y * sizeof(DType), 1, test_size_y * sizeof(DType));

  // Prepare test and expect data
  for (int i = 0; i < test_shape[0]; i++) {
    for (int j = 0; j < test_shape[1]; j++) {
      x1[i * test_shape[1] + j] = j;
    }
  }
  for (int i = 0; i < test_shape[2]; i++) {
    for (int j = 0; j < test_shape[3]; j++) {
      x2[i * test_shape[3] + j] = j + test_shape[1];
    }
  }

  for (int i = 0; i < test_shape[0]; i++) {
    for (int j = 0; j < test_shape[1] + test_shape[3]; j++) {
      expect[i * (test_shape[1] + test_shape[3]) + j] = j;
    }
  }

  // Launch
  tiling_data.block_dim = block_dim;
  tiling_data.s0 = test_shape[0];
  tiling_data.tiling_key = 0;
  // tiling信息由用例输入参数确定
  if (test_tiling.empty()) { // tiling data 来源于tiling函数GetTiling
    GetTiling(tiling_data);
  } else {                        // tiling信息来源于测试用例入参
    tiling_data.block_dim = test_tiling[0];
    tiling_data.z0Tb_size = test_tiling[1];
    tiling_data.z0t_size = test_tiling[2];
  }

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(concat_same_tail_dim,
              tiling_data.block_dim,
              (uint8_t *) x1,
              (uint8_t *) x2,
              (uint8_t *) y,
              nullptr,
              (uint8_t *) &tiling_data);

  // Count difference
  uint32_t diff_count = 0;
  for (int i = 0; i < test_size_y; i++) {
    half diff = y[i] - expect[i];
    if (diff > (half) 0.0001 || diff < (half) -0.0001) {
      diff_count++;
    }
  }

  EXPECT_EQ(diff_count, 0) << " of " << test_size_y;

  AscendC::GmFree(x1);
  AscendC::GmFree(x2);
  AscendC::GmFree(y);
  AscendC::GmFree(expect);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2E_ConcatSameTailDim_Code,
                         ::testing::Values(
                             std::pair<std::vector<int>, std::vector<int>>{{65, 1, 65, 1}, {3, 1, 31}}));