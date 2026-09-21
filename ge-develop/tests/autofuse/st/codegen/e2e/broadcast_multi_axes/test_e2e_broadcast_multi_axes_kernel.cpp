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

extern "C" __global__ __aicore__ void broadcast_multi_axes(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling_data);
extern "C" void GetTiling(AutofuseTilingData &tiling_data);
 
class E2E_Broadcast_Multi_Axes : public testing::Test, public testing::WithParamInterface<std::vector<int>> {};

TEST_P(E2E_Broadcast_Multi_Axes, broadcast_multi_axes_a1a1) {

  uint32_t block_dim = 48;
  uint32_t s0 = GetParam()[0];
  uint32_t s1 = GetParam()[1];
  uint32_t s2 = GetParam()[2];
  uint32_t s3 = GetParam()[3];
  uint32_t s4 = GetParam()[4];
  uint32_t dst_size = s0*s1*s2*s3*s4;
  uint32_t src_size = s0*s1*s3;

  AutofuseTilingData tiling_data;
  half *x0 = (half *)AscendC::GmAlloc(src_size * sizeof(half));
  half *y = (half *)AscendC::GmAlloc(dst_size * sizeof(half));
  half *expect = (half *)AscendC::GmAlloc(dst_size * sizeof(half));

  for (int32_t j = 0; j < s0; j++) {
    for (int32_t i = 0; i < s1; i++) {
      for (int32_t k = 0; k < s1; k++) {
        x0[i * s1 + k] = (half)(i * s1 + k);
      }
    }
  }

  for (uint32_t i = 0; i < s1; i++) {
    for (uint32_t k = 0; k < s2; k++) {
      for (uint32_t j = 0; j < s3; j++) {
        for (uint32_t z = 0; z < s4; z++) {
          expect[i* s2 * s3 * s4 + k * s3 * s4 + j * s4 + z] = (half)(i * s1 + j);
        }
      }
    }
  }

  tiling_data.block_dim = block_dim;
  tiling_data.s0 = s0;
  tiling_data.s1 = s1;
  tiling_data.s2 = s2;
  tiling_data.s3 = s3;
  tiling_data.s4 = s4;
  tiling_data.tiling_key = 0;

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(broadcast_multi_axes, tiling_data.block_dim, (uint8_t *)x0, (uint8_t *)y, nullptr, (uint8_t *)&tiling_data);
  // Count difference
  uint32_t diff_count = 0;
  for (int i = 0; i < s0 * s1 * s2 * s3 * s4; i++) {
    half diff = y[i] - expect[i];
    if (diff > (half)0.0001 || diff < (half)-0.0001) {
      diff_count++;
    }
  }

  EXPECT_EQ(diff_count, 0) << " of " << (s0 * s1 * s2 * s3 * s4);

  AscendC::GmFree(x0);
  AscendC::GmFree(y);
  AscendC::GmFree(expect);
}

INSTANTIATE_TEST_SUITE_P(CalcBroadCastMultiAxes, E2E_Broadcast_Multi_Axes,
   ::testing::Values(
       std::vector<int>{1, 2, 8, 2, 8}
   ));

extern "C" __global__ __aicore__ void broadcast_multi_axes_bab(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling_data);
 
class E2E_Broadcast_Multi_Axes_BAB : public testing::Test, public testing::WithParamInterface<std::vector<int>> {};

TEST_P(E2E_Broadcast_Multi_Axes_BAB, broadcast_multi_axes_11a1) {

  uint32_t block_dim = 48;
  uint32_t s0 = GetParam()[0];
  uint32_t s1 = GetParam()[1];
  uint32_t s2 = GetParam()[2];
  uint32_t s3 = GetParam()[3];
  uint32_t s4 = GetParam()[4];
  uint32_t dst_size = s0*s1*s2*s3*s4;
  uint32_t src_size = s0*s1*s3;

  AutofuseTilingData tiling_data;
  half *x0 = (half *)AscendC::GmAlloc(src_size * sizeof(half));
  half *y = (half *)AscendC::GmAlloc(dst_size * sizeof(half));
  half *expect = (half *)AscendC::GmAlloc(dst_size * sizeof(half));

  for (int32_t j = 0; j < s0; j++) {
    for (int32_t i = 0; i < s1; i++) {
      for (int32_t k = 0; k < s1; k++) {
        x0[i * s1 + k] = (half)(i * s1 + k);
      }
    }
  }

  for (uint32_t i = 0; i < s1; i++) {
    for (uint32_t k = 0; k < s2; k++) {
      for (uint32_t j = 0; j < s3; j++) {
        for (uint32_t z = 0; z < s4; z++) {
          expect[i* s2 * s3 * s4 + k * s3 * s4 + j * s4 + z] = (half)(i * s1 + j);
        }
      }
    }
  }

  tiling_data.block_dim = block_dim;
  tiling_data.s0 = s0;
  tiling_data.s1 = s1;
  tiling_data.s2 = s2;
  tiling_data.s3 = s3;
  tiling_data.s4 = s4;
  tiling_data.tiling_key = 0;

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(broadcast_multi_axes_bab, tiling_data.block_dim, (uint8_t *)x0, (uint8_t *)y, nullptr, (uint8_t *)&tiling_data);
  // Count difference
  uint32_t diff_count = 0;
  for (int i = 0; i < s0 * s1 * s2 * s3 * s4; i++) {
    half diff = y[i] - expect[i];
    if (diff > (half)0.0001 || diff < (half)-0.0001) {
      diff_count++;
    }
  }

  EXPECT_EQ(diff_count, 0) << " of " << (s0 * s1 * s2 * s3 * s4);

  AscendC::GmFree(x0);
  AscendC::GmFree(y);
  AscendC::GmFree(expect);
}

INSTANTIATE_TEST_SUITE_P(CalcBroadCastMultiAxesBAB, E2E_Broadcast_Multi_Axes_BAB,
   ::testing::Values(
       std::vector<int>{1, 2, 8, 2, 8}
   ));