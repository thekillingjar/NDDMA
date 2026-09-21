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
extern "C" __global__ __aicore__ void load_broadcast_multi_axis_store(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);
class E2E_Load_Broadcast_Multi_Axis_Store : public testing::Test,
                               public testing::WithParamInterface<std::vector<int>> {};

TEST_P(E2E_Load_Broadcast_Multi_Axis_Store, BroadCast_A11toABC) {
  uint32_t s0 = GetParam()[0];
  uint32_t s1 = GetParam()[1];
  uint32_t s2 = GetParam()[2];
  uint32_t s3 = GetParam()[3];
  uint32_t s4 = GetParam()[4];
  uint32_t dst_size = s0*s1*s2*s3*s4;
  uint32_t src_size = s0*s1*s2;

  AutofuseTilingData tiling_data;
  float* x = (float*)AscendC::GmAlloc(src_size * sizeof(float));
  float* y = (float*)AscendC::GmAlloc(dst_size * sizeof(float));
  float* expect = (float*)AscendC::GmAlloc(dst_size * sizeof(float));

  // Prepare test and expect data
  for (int i = 0; i < src_size; i++) {
    x[i] = i;
    for (int j = 0; j < s3*s4; j++) {
      expect[i*s3*s4 + j] = i;
    }
  }
  // Launch
  tiling_data.block_dim = 1;
  tiling_data.s0 = s0;
  tiling_data.s1 = s1;
  tiling_data.s2 = s2;
  tiling_data.s3 = s3;
  tiling_data.s4 = s4;
  tiling_data.tiling_key = 0;

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(load_broadcast_multi_axis_store, tiling_data.block_dim, (uint8_t*)x, (uint8_t*)y, nullptr, (uint8_t*)&tiling_data);

  // Count difference
  uint32_t diff_count = 0;
  for (int i = 0; i < dst_size; i++) {
    if (y[i] != expect[i]) {
        diff_count++;
      }
  }

  EXPECT_EQ(diff_count, 0) << " of " << std::to_string(dst_size);

  AscendC::GmFree(x);
  AscendC::GmFree(y);
  AscendC::GmFree(expect);
}


extern "C" __global__ __aicore__ void load_broadcast_multi_axis_store_2(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

TEST_P(E2E_Load_Broadcast_Multi_Axis_Store, BroadCast_11CtoABC) {
  uint32_t s0 = GetParam()[0];
  uint32_t s1 = GetParam()[1];
  uint32_t s2 = GetParam()[2];
  uint32_t s3 = GetParam()[3];
  uint32_t s4 = GetParam()[4];
  uint32_t dst_size = s0*s1*s2*s3*s4;
  //uint32_t src_size = s0*s1*s2*s4;
  uint32_t src_size = s0*s3*s4;

  AutofuseTilingData tiling_data;
  float* x = (float*)AscendC::GmAlloc(src_size * sizeof(float));
  float* y = (float*)AscendC::GmAlloc(dst_size * sizeof(float));
  float* expect = (float*)AscendC::GmAlloc(dst_size * sizeof(float));

  // Prepare test and expect data
  uint32_t m = s0;
  uint32_t n = s1*s2;
  uint32_t p = s3*s4;
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < p; k++) {
        int index = i * n * p + j * p + k;
        x[i * p + k] = (float)(i * p + k);
        expect[i * n * p + j * p + k] = (float)(i * p + k);
      }
    }
  }
  // Launch
  tiling_data.block_dim = 1;
  tiling_data.s0 = s0;
  tiling_data.s1 = s1;
  tiling_data.s2 = s2;
  tiling_data.s3 = s3;
  tiling_data.s4 = s4;
  tiling_data.tiling_key = 0;

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(load_broadcast_multi_axis_store_2, tiling_data.block_dim, (uint8_t*)x, (uint8_t*)y, nullptr, (uint8_t*)&tiling_data);

  // Count difference
  uint32_t diff_count = 0;
  for (int i = 0; i < dst_size; i++) {
    if (y[i] != expect[i]) {
        diff_count++;
      }
  }

  EXPECT_EQ(diff_count, 0) << " of " << std::to_string(dst_size);

  AscendC::GmFree(x);
  AscendC::GmFree(y);
  AscendC::GmFree(expect);
}


extern "C" __global__ __aicore__ void load_broadcast_multi_axis_store_3(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

TEST_P(E2E_Load_Broadcast_Multi_Axis_Store, BroadCast_A1CtoABC) {
  uint32_t s0 = GetParam()[0];
  uint32_t s1 = GetParam()[1];
  uint32_t s2 = GetParam()[2];
  uint32_t s3 = GetParam()[3];
  uint32_t s4 = GetParam()[4];
  uint32_t dst_size = s0*s1*s2*s3*s4;
  uint32_t src_size = s0*s1*s2*s4;

  AutofuseTilingData tiling_data;
  float* x = (float*)AscendC::GmAlloc(src_size * sizeof(float));
  float* y = (float*)AscendC::GmAlloc(dst_size * sizeof(float));
  float* expect = (float*)AscendC::GmAlloc(dst_size * sizeof(float));

  // Prepare test and expect data
  uint32_t m = s0*s1*s2;
  uint32_t n = s3;
  uint32_t p = s4;
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < p; k++) {
        int index = i * n * p + j * p + k;
        x[i * p + k] = (float)(i * p + k);
        expect[i * n * p + j * p + k] = (float)(i * p + k);
      }
    }
  }
  // Launch
  tiling_data.block_dim = 1;
  tiling_data.s0 = s0;
  tiling_data.s1 = s1;
  tiling_data.s2 = s2;
  tiling_data.s3 = s3;
  tiling_data.s4 = s4;
  tiling_data.tiling_key = 0;

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(load_broadcast_multi_axis_store_3, tiling_data.block_dim, (uint8_t*)x, (uint8_t*)y, nullptr, (uint8_t*)&tiling_data);

  // Count difference
  uint32_t diff_count = 0;
  for (int i = 0; i < dst_size; i++) {
    if (y[i] != expect[i]) {
        diff_count++;
      }
  }

  EXPECT_EQ(diff_count, 0) << " of " << std::to_string(dst_size);

  AscendC::GmFree(x);
  AscendC::GmFree(y);
  AscendC::GmFree(expect);
}

INSTANTIATE_TEST_SUITE_P(M32_K_BlockAlign, E2E_Load_Broadcast_Multi_Axis_Store,
   ::testing::Values(
       // 2,4,4,32 to 2,4,4,8,32
       std::vector<int>{2, 4, 4, 8, 32}
   ));

INSTANTIATE_TEST_SUITE_P(M32_K_BlockNotAlign, E2E_Load_Broadcast_Multi_Axis_Store,
::testing::Values(
    std::vector<int>{2, 4, 5, 5, 10}
    //会跑上面三个用例, 其中对第一个用例的场景A 1 1 to A B C 是 2, 4, 5, 1, 1 to 2, 4, 5, 5, 10
));

