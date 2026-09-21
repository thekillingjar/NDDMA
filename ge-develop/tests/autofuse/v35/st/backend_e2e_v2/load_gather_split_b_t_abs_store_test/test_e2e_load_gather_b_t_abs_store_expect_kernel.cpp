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
extern "C" __global__ __aicore__ void load_gather_abs_store_store_test(GM_ADDR param, GM_ADDR indices, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);
extern "C" int64_t AutofuseTiling(uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3, uint32_t s4, uint32_t s5, uint32_t s6, AutofuseTilingData* tiling, uint32_t* workspaceSize, uint32_t *blockDim, uint32_t aiv_num, uint32_t ub_size);
class E2E_Load_Gather_B_T_Abs_Store : public testing::Test,
                               public testing::WithParamInterface<std::pair<std::vector<int>, std::vector<int>>> {};

TEST_P(E2E_Load_Gather_B_T_Abs_Store, Gather_B_T_AbsTest) {
  auto [param_shape, indices_shape] = GetParam();
  int s0 = param_shape[0];
  int s1 = param_shape[1];
  int s2 = param_shape[2];
  int s3 = param_shape[3];
  int s4 = param_shape[4];

  int s5 = indices_shape[0];
  int s6 = indices_shape[1];

  int param_size = s0 * s1 * s2 * s3 * s4;
  int indices_size = s5 * s6;
  int output_size = s0 * s1 * s5 * s6 * s3 * s4;

  //AutofuseTilingData tiling_data;
  float* param = (float*)AscendC::GmAlloc(param_size * sizeof(float));
  int32_t* indices = (int32_t*)AscendC::GmAlloc(indices_size * sizeof(int32_t));
  float* output = (float*)AscendC::GmAlloc(output_size * sizeof(float));
  float* expect = (float*)AscendC::GmAlloc(output_size * sizeof(float));

  // Prepare test and expect data
  for (int i = 0; i < param_size; i++) {
    param[i] = i;
  }
  for (int i = 0; i < indices_size; i++) {
    indices[i] = i % s2;
  }
  for (int i = 0; i < s0; i++) {
    for (int j = 0; j < s1; j++) {
      for (int m = 0; m < s5; m++) {
        for (int n = 0; n < s6; n++) {
          int indices_value = indices[m * s6 + n];
          int param_offset = i * s1 * s2 * s3 * s4 + j * s2 * s3 * s4 + indices_value * s3 * s4;
          int output_offset = i * s1 * s5 * s6 * s3 * s4 + j * s5 * s6 * s3 * s4 + m * s6 * s3 * s4 + n * s3 * s4;
          for (int k = 0; k < s3 * s4; k++) {
            // printf("output_offset + k is %d, param_offset is %d\n", output_offset + k, param_offset);
            expect[output_offset + k] = param[param_offset + k];
          }
        }
      }
    }
  }
  // Launch
  AutofuseTilingData tiling_data;
  tiling_data.tiling_key = 0;
  uint32_t ws_size = 0;
  uint32_t block_dim = 48;
  //AutofuseTiling(uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3, uint32_t s4, uint32_t s5, uint32_t s6, AutofuseTilingData* tiling, uint32_t* workspaceSize, uint32_t *blockDim, uint32_t aiv_num, uint32_t ub_size);
  AutofuseTiling(s0, s1, s2, s3, s4, s5, s6, &tiling_data, &ws_size, &block_dim, 48, 192 * 1024);
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(load_gather_abs_store_store_test, tiling_data.block_dim, (uint8_t*)param, (uint8_t*)indices, (uint8_t*)output, nullptr, (uint8_t*)&tiling_data);
  // Count difference

  uint32_t diff_count = 0;
  for (int i = 0; i < output_size; i++) {
    if (output[i] != expect[i]) {
      diff_count++;
    }
  }

  EXPECT_EQ(diff_count, 0) << " of " << std::to_string(output_size);

  AscendC::GmFree(param);
  AscendC::GmFree(indices);
  AscendC::GmFree(output);
  AscendC::GmFree(expect);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2E_Load_Gather_B_T_Abs_Store,
   ::testing::Values(
       std::pair<std::vector<int>, std::vector<int>>{{2, 2, 4, 2, 7}, {2, 4}}
   ));