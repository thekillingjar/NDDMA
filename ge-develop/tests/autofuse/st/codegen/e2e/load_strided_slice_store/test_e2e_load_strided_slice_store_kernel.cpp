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
extern "C" __global__ __aicore__ void load_strided_slice_store(GM_ADDR x1, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);
extern "C" void GetTiling(AutofuseTilingData& tiling_data);

class E2E_LoadStridedSliceStore_Code : public testing::Test, public testing::WithParamInterface<std::vector<int>> {
};

TEST_P(E2E_LoadStridedSliceStore_Code, CalculateCorrect) {
  auto test_shape = GetParam();
  uint32_t block_dim = 1;
  auto x1_shape = test_shape[0] * (test_shape[1] + test_shape[2]);
  auto y_shape = test_shape[0] * test_shape[2];
  AutofuseTilingData tiling_data;
  float *x1 = (float *)AscendC::GmAlloc(x1_shape * sizeof(float));
  float *expect = (float *)AscendC::GmAlloc(y_shape * sizeof(float));
  float *y = (float *)AscendC::GmAlloc(y_shape * sizeof(float));

  // Prepare test and expect data
  for (int i = 0; i < test_shape[0]; i++) {
    for (int j = 0; j < (test_shape[1] + test_shape[2]); j++) {
      x1[i * (test_shape[1] + test_shape[2]) + j] = j;
    }
  }
  for (int i = 0; i < test_shape[0]; i++) {
    for (int j = 0; j < test_shape[2]; j++) {
      expect[i * test_shape[2] + j] = x1[i * (test_shape[1] + test_shape[2]) + j + test_shape[1]];
    }
  }

  // Launch
  tiling_data.block_dim = block_dim;
  tiling_data.s0 = test_shape[0];
  tiling_data.s1 = test_shape[1];
  tiling_data.s2 = test_shape[2];
  tiling_data.tiling_key = 0;

  GetTiling(tiling_data);

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(load_strided_slice_store, tiling_data.block_dim, (uint8_t *)x1, (uint8_t *)y, nullptr, (uint8_t*)&tiling_data);

  // Count difference
  uint32_t diff_count = 0;
  for (int i = 0; i < y_shape; i++) {
    half diff = y[i] - expect[i];
    if (diff > (half)0.0001 || diff < (half)-0.0001) {
      diff_count++;
    }
  }

  EXPECT_EQ(diff_count, 0) << " of " << y_shape;

  AscendC::GmFree(x1);
  AscendC::GmFree(y);
  AscendC::GmFree(expect);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2E_LoadStridedSliceStore_Code,
    ::testing::Values(std::vector<int>{2, 2, 2}));