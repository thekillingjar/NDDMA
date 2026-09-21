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

extern "C" __global__ __aicore__ void load_broadcast_store(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);
extern "C" void GetTiling(AutofuseTilingData& tiling_data);

class E2E_Load_Broadcast_Store : public testing::Test,
                               public testing::WithParamInterface<std::vector<int>> {};

TEST_P(E2E_Load_Broadcast_Store, BroadCastLast111) {
  uint32_t block_dim = 48;
  uint32_t s0 = GetParam()[0];
  uint32_t s1 = GetParam()[1];

  AutofuseTilingData tiling_data;
  float* x = (float*)AscendC::GmAlloc(s0 * sizeof(float));
  float* y = (float*)AscendC::GmAlloc(s0*s1 * sizeof(float));
  float* expect = (float*)AscendC::GmAlloc(s0*s1 * sizeof(float));

  // Prepare test and expect data
  for (int i = 0; i < s0; i++) {
    x[i] = i;
    for (int j = 0; j < s1; j++) {
      expect[i * s1 + j] = i;
    }
  }

  // Launch
  tiling_data.block_dim = block_dim;
  tiling_data.s0 = s0;
  tiling_data.s1 = s1;
  tiling_data.tiling_key = 0;
  GetTiling(tiling_data);

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(load_broadcast_store, tiling_data.block_dim, (uint8_t*)x, (uint8_t*)y, nullptr, (uint8_t*)&tiling_data);

  // Count difference
  uint32_t diff_count = 0;
  for (int i = 0; i < s0; i++) {
    for (int j = 0; j < s1; j++) {
      if (y[i * s1 + j] != expect[i * s1 + j]) {
        diff_count++;
      }
    }
  }

  EXPECT_EQ(diff_count, 0) << " of " << std::to_string(s0 * s1);

  AscendC::GmFree(x);
  AscendC::GmFree(y);
  AscendC::GmFree(expect);
}

INSTANTIATE_TEST_SUITE_P(M32_K_BlockAlign, E2E_Load_Broadcast_Store,
   ::testing::Values(
       std::vector<int>{48*4*8, 16},
       std::vector<int>{48*4*16, 16},
       std::vector<int>{48*4*32, 16},
       std::vector<int>{48*4*64, 16},
       std::vector<int>{48*4*128, 16},
       std::vector<int>{48*4*8, 512},
       std::vector<int>{48*4*8, 256},
       std::vector<int>{48*4*8, 128},
       std::vector<int>{48*4*8, 64},
       std::vector<int>{48*4*8, 32}
   ));

extern "C" __global__ __aicore__ void load_broadcast_store_int64(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

class E2E_Load_Broadcast_Store_Int64 : public testing::Test,
                               public testing::WithParamInterface<std::vector<int>> {};

TEST_P(E2E_Load_Broadcast_Store_Int64, BroadCastLast111_int64) {
  uint32_t block_dim = 48;
  uint32_t s0 = GetParam()[0];
  uint32_t s1 = GetParam()[1];

  AutofuseTilingData tiling_data;
  int64_t* x = (int64_t*)AscendC::GmAlloc(s0 * sizeof(int64_t));
  int64_t* y = (int64_t*)AscendC::GmAlloc(s0*s1 * sizeof(int64_t));
  int64_t* expect = (int64_t*)AscendC::GmAlloc(s0*s1 * sizeof(int64_t));

  // Prepare test and expect data
  for (int i = 0; i < s0; i++) {
    x[i] = i;
    for (int j = 0; j < s1; j++) {
      expect[i * s1 + j] = i;
    }
  }

  // Launch
  tiling_data.block_dim = block_dim;
  tiling_data.s0 = s0;
  tiling_data.s1 = s1;
  tiling_data.tiling_key = 0;
  GetTiling(tiling_data);

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(load_broadcast_store_int64, tiling_data.block_dim, (uint8_t*)x, (uint8_t*)y, nullptr, (uint8_t*)&tiling_data);

  // Count difference
  uint32_t diff_count = 0;
  for (int i = 0; i < s0; i++) {
    for (int j = 0; j < s1; j++) {
      if (y[i * s1 + j] != expect[i * s1 + j]) {
        diff_count++;
      }
    }
  }

  EXPECT_EQ(diff_count, 0) << " of " << std::to_string(s0 * s1);

  AscendC::GmFree(x);
  AscendC::GmFree(y);
  AscendC::GmFree(expect);
}

INSTANTIATE_TEST_SUITE_P(M32_K_BlockAlign, E2E_Load_Broadcast_Store_Int64,
   ::testing::Values(
       std::vector<int>{48*4*8, 16},
       std::vector<int>{48*4*8, 256},
       std::vector<int>{48*4*8, 128},
       std::vector<int>{48*4*8, 64},
       std::vector<int>{48*4*8, 32}
   ));

extern "C" __global__ __aicore__ void load_broadcast_store_uint8(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

class E2E_Load_Broadcast_Store_uint8 : public testing::Test,
                               public testing::WithParamInterface<std::vector<int>> {};

TEST_P(E2E_Load_Broadcast_Store_uint8, BroadCastLast111_uint8) {
  uint32_t block_dim = 48;
  uint32_t s0 = GetParam()[0];
  uint32_t s1 = GetParam()[1];

  AutofuseTilingData tiling_data;
  uint8_t* x = (uint8_t*)AscendC::GmAlloc(s0 * sizeof(uint8_t));
  uint8_t* y = (uint8_t*)AscendC::GmAlloc(s0*s1 * sizeof(uint8_t));
  uint8_t* expect = (uint8_t*)AscendC::GmAlloc(s0*s1 * sizeof(uint8_t));

  // Prepare test and expect data
  for (int i = 0; i < s0; i++) {
    x[i] = i;
    for (int j = 0; j < s1; j++) {
      expect[i * s1 + j] = i;
    }
  }

  // Launch
  tiling_data.block_dim = block_dim;
  tiling_data.s0 = s0;
  tiling_data.s1 = s1;
  tiling_data.tiling_key = 0;
  GetTiling(tiling_data);

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(load_broadcast_store_uint8, tiling_data.block_dim, (uint8_t*)x, (uint8_t*)y, nullptr, (uint8_t*)&tiling_data);

  // Count difference
  uint32_t diff_count = 0;
  for (int i = 0; i < s0; i++) {
    for (int j = 0; j < s1; j++) {
      if (y[i * s1 + j] != expect[i * s1 + j]) {
        diff_count++;
      }
    }
  }

  EXPECT_EQ(diff_count, 0) << " of " << std::to_string(s0 * s1);

  AscendC::GmFree(x);
  AscendC::GmFree(y);
  AscendC::GmFree(expect);
}

INSTANTIATE_TEST_SUITE_P(M32_K_BlockAlign, E2E_Load_Broadcast_Store_uint8,
   ::testing::Values(
       std::vector<int>{48*4*7, 50},
       std::vector<int>{48*4*15, 23},
       std::vector<int>{48*4*23, 31},
       std::vector<int>{48*4*50, 15},
       std::vector<int>{48*4*8, 200},
       std::vector<int>{48*4*8, 100},
       std::vector<int>{48*4*8, 64},
       std::vector<int>{48*4*8, 32}
   ));
