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
extern "C" __global__ __aicore__ void load_rsum_store(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);
extern "C" void GetTiling(AutofuseTilingData& tiling_data);
extern "C" void GetTilingInt32Blk(AutofuseTilingData& tiling_data);

class E2E_LoadRsumStore_Code : public testing::Test,
    public testing::WithParamInterface<std::vector<int>> {};

TEST_P(E2E_LoadRsumStore_Code, CalculateCorrect) {
  auto test_shape = GetParam();

  uint32_t block_dim = 48;
  int test_size = test_shape[0] * test_shape[1];

  AutofuseTilingData tiling_data;
  float *x = (float *)AscendC::GmAlloc(test_size * sizeof(float) + 32);
  float *y = (float *)AscendC::GmAlloc(test_shape[0] * sizeof(float) + 32);
  float *expect = (float *)AscendC::GmAlloc(test_shape[0] * sizeof(float) + 32);

  // Prepare test and expect data
  for (int i = 0; i < test_shape[0] * test_shape[1]; i++) {
    x[i] = (double)1;
  }
  x[5] = (double)2;
  x[6] = (double)(-2);
  x[20] = (double)2;
  x[52] = (double)3;

  for (int i = 0; i < test_shape[0]; i++) {
    expect[i] = 0;
    for (int j = 0; j < test_shape[1]; j++) {
        int idx = i * test_shape[1] + j;
        expect[i] += x[idx];
    }
  }

  // Launch
  tiling_data.block_dim = block_dim;
  tiling_data.s0 = test_shape[0];
  tiling_data.s1 = test_shape[1];
  tiling_data.tiling_key = 0;
  GetTiling(tiling_data);
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(load_rsum_store, tiling_data.block_dim, (uint8_t *)x, (uint8_t *)y, nullptr, (uint8_t *)&tiling_data);

  // Count difference
  uint32_t diff_count = 0;
  for (int i = 0; i < test_shape[0]; i++) {
    double diff = (double)(y[i] - expect[i]);
    if (diff > (double)0.5 || diff < (double)-0.5) {
      diff_count++;
    }
  }

  EXPECT_EQ(diff_count, 0) << " of " << test_size;

  AscendC::GmFree(x);
  AscendC::GmFree(y);
  AscendC::GmFree(expect);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2E_LoadRsumStore_Code,
    ::testing::Values(
        std::vector<int>{48*4*8, 64},
        std::vector<int>{48*4, 128},
        std::vector<int>{48*4, 256},
        std::vector<int>{48*4, 512},
        std::vector<int>{48*4, 976},

        std::vector<int>{48*4*8, 16},
        std::vector<int>{48*4*16, 16},
        std::vector<int>{48*4*32, 16},
        std::vector<int>{48*4*64, 16},
        std::vector<int>{48*4*128, 16},
        std::vector<int>{48*4*256, 16},

        std::vector<int>{96*2, 32},
        std::vector<int>{96*2, 128},
        std::vector<int>{96*2, 256}
        ));

extern "C" __global__ __aicore__ void load_rsum_store_int32(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

class E2E_LoadRsumStoreInt32_Code : public testing::Test,
    public testing::WithParamInterface<std::vector<int>> {};

TEST_P(E2E_LoadRsumStoreInt32_Code, CalculateCorrect) {
  auto test_shape = GetParam();

  uint32_t block_dim = 48;
  int test_size = test_shape[0] * test_shape[1];

  AutofuseTilingData tiling_data;
  int32_t *x = (int32_t *)AscendC::GmAlloc(test_size * sizeof(int32_t) + 32);
  int32_t *y = (int32_t *)AscendC::GmAlloc(test_shape[0] * sizeof(int32_t) + 32);
  int32_t *expect = (int32_t *)AscendC::GmAlloc(test_shape[0] * sizeof(int32_t) + 32);

  // Prepare test and expect data
  for (int i = 0; i < test_shape[0] * test_shape[1]; i++) {
    x[i] = (int32_t)1;
  }
  x[5] = (int32_t)2;
  x[6] = (int32_t)(-2);
  x[20] = (int32_t)2;
  x[52] = (int32_t)3;

  for (int i = 0; i < test_shape[0]; i++) {
    expect[i] = 0;
    for (int j = 0; j < test_shape[1]; j++) {
        int idx = i * test_shape[1] + j;
        expect[i] += x[idx];
    }
  }

  // Launch
  tiling_data.block_dim = block_dim;
  tiling_data.s0 = test_shape[0];
  tiling_data.s1 = test_shape[1];
  tiling_data.tiling_key = 0;
  GetTiling(tiling_data);
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(load_rsum_store, tiling_data.block_dim, (uint8_t *)x, (uint8_t *)y, nullptr, (uint8_t *)&tiling_data);

  // Count difference
  uint32_t diff_count = 0;
  for (int i = 0; i < test_shape[0]; i++) {
    double diff = (double)(y[i] - expect[i]);
    if (diff > (double)0.5 || diff < (double)-0.5) {
      diff_count++;
    }
  }

  EXPECT_EQ(diff_count, 1) << " of " << test_size;

  AscendC::GmFree(x);
  AscendC::GmFree(y);
  AscendC::GmFree(expect);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2E_LoadRsumStoreInt32_Code,
    ::testing::Values(
        std::vector<int>{48*4*8, 64},
        std::vector<int>{48*4, 128},
        std::vector<int>{48*4, 256},
        std::vector<int>{48*4, 512},
        std::vector<int>{48*4, 976},

        std::vector<int>{48*4*8, 16},
        std::vector<int>{48*4*16, 16},
        std::vector<int>{48*4*32, 16},
        std::vector<int>{48*4*64, 16},
        std::vector<int>{48*4*128, 16},
        std::vector<int>{48*4*256, 16},

        std::vector<int>{96*2, 32},
        std::vector<int>{96*2, 128},
        std::vector<int>{96*2, 256}
        ));