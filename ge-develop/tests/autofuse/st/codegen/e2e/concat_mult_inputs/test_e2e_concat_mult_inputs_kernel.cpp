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
extern "C" __global__ __aicore__ void concat_mult_inputs(GM_ADDR x1, GM_ADDR x2, GM_ADDR x3, GM_ADDR x4, GM_ADDR x5, GM_ADDR x6, GM_ADDR x7, GM_ADDR y, GM_ADDR workspace, GM_ADDR gm_tiling_data);
extern "C" void GetTiling(AutofuseTilingData& tiling_data);

class E2E_ConcatMultInputs_Code : public testing::Test, public testing::WithParamInterface<std::pair<std::vector<int>, std::vector<int>>> {
};

TEST_P(E2E_ConcatMultInputs_Code, ConcatMultInputs) {
  auto [test_shape, test_tiling] = GetParam();
  uint32_t block_dim = 1;
  int test_size_x1 = test_shape[0] * test_shape[1];
  int test_size_x2 = test_shape[0] * test_shape[2];
  int test_size_x3 = test_shape[0] * test_shape[3];
  int test_size_x4 = test_shape[0] * test_shape[4];
  int test_size_x5 = test_shape[0] * test_shape[5];
  int test_size_x6 = test_shape[0] * test_shape[6];
  int test_size_x7 = test_shape[0] * test_shape[7];
  int test_size_y = test_shape[0] * (test_shape[1] + test_shape[2]
    + test_shape[3]+ test_shape[4]+ test_shape[5]+ test_shape[6]
    + test_shape[7]);

  AutofuseTilingData tiling_data;
  int64_t* x1 = (int64_t*)AscendC::GmAlloc(test_size_x1 * sizeof(int64_t));
  int64_t* x2 = (int64_t*)AscendC::GmAlloc(test_size_x2 * sizeof(int64_t));
  int64_t* x3 = (int64_t*)AscendC::GmAlloc(test_size_x3 * sizeof(int64_t));
  int64_t* x4 = (int64_t*)AscendC::GmAlloc(test_size_x4 * sizeof(int64_t));
  int64_t* x5 = (int64_t*)AscendC::GmAlloc(test_size_x5 * sizeof(int64_t));
  int64_t* x6 = (int64_t*)AscendC::GmAlloc(test_size_x6 * sizeof(int64_t));
  int64_t* x7 = (int64_t*)AscendC::GmAlloc(test_size_x7 * sizeof(int64_t));
  int64_t* y = (int64_t*)AscendC::GmAlloc(test_size_y * sizeof(int64_t));
  int64_t* expect = (int64_t*)AscendC::GmAlloc(test_size_y * sizeof(int64_t));

  // Prepare test and expect data
  for (int i = 0; i < test_shape[0]; i++) {
      for (int j = 0; j < test_shape[1]; j++){
        x1[i * test_shape[1] + j] = j;
      }
  }
  for (int i = 0; i < test_shape[0]; i++) {
      for (int j = 0; j < test_shape[2]; j++){
        x2[i * test_shape[2] + j] = j + test_shape[1];
      }
  }
  for (int i = 0; i < test_shape[0]; i++) {
    for (int j = 0; j < test_shape[3]; j++){
      x3[i * test_shape[3] + j] = j + test_shape[1] + test_shape[2];
    }
  }
  for (int i = 0; i < test_shape[0]; i++) {
    for (int j = 0; j < test_shape[4]; j++){
      x4[i * test_shape[4] + j] = j + test_shape[1] + test_shape[2]
      + test_shape[3];
    }
  }
  for (int i = 0; i < test_shape[0]; i++) {
    for (int j = 0; j < test_shape[5]; j++){
      x5[i * test_shape[5] + j] = j + test_shape[1] + test_shape[2]
      + test_shape[3] + test_shape[4];
    }
  }
  for (int i = 0; i < test_shape[0]; i++) {
    for (int j = 0; j < test_shape[6]; j++){
      x6[i * test_shape[6] + j] = j + test_shape[1] + test_shape[2]
      + test_shape[3] + test_shape[4] + test_shape[5];
    }
  }
  for (int i = 0; i < test_shape[0]; i++) {
    for (int j = 0; j < test_shape[7]; j++){
      x7[i * test_shape[7] + j] = j + test_shape[1] + test_shape[2]
      + test_shape[3]+ test_shape[4]+ test_shape[5]+ test_shape[6];
    }
  }

  auto y_shape1 = test_shape[1] + test_shape[2]
    + test_shape[3]+ test_shape[4]+ test_shape[5]+ test_shape[6]
    + test_shape[7];
  for (int i = 0; i < test_shape[0]; i++) {
      for (int j = 0; j < y_shape1; j++){
        expect[i * y_shape1 + j] = j;
      }
  }

  // Launch
  tiling_data.block_dim = block_dim;
  tiling_data.s0 = test_shape[0];
  tiling_data.s1 = test_shape[1];
  tiling_data.s2 = test_shape[2];
  tiling_data.s3 = test_shape[3];
  tiling_data.s4 = test_shape[4];
  tiling_data.s5 = test_shape[5];
  tiling_data.s6 = test_shape[6];
  tiling_data.s7 = test_shape[7];
  tiling_data.tiling_key = 0;
  // tiling信息由用例输入参数确定
  if (test_tiling.size() == 0U) { // tiling data 来源于tiling函数GetTiling
    GetTiling(tiling_data);
  } else {                        // tiling信息来源于测试用例入参
    tiling_data.block_dim = test_tiling[0];
    tiling_data.z0Tb_size = test_tiling[1];
    tiling_data.z0t_size = test_tiling[2];
  }

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(concat_mult_inputs, tiling_data.block_dim, (uint8_t*)x1, (uint8_t*)x2,
  (uint8_t*)x3, (uint8_t*)x4, (uint8_t*)x5, (uint8_t*)x6, (uint8_t*)x7,
  (uint8_t*)y, nullptr, (uint8_t*)&tiling_data);

  // Count difference
  uint32_t diff_count = 0;
  for (int i = 0; i < test_shape[0]; i++) {
    for (int j = 0; j < y_shape1; j++) {
      double diff = y[i * y_shape1 + j] - expect[i * y_shape1 + j];
      if (diff > (double)0.0001 || diff < (double)-0.0001) {
          diff_count++;
      }
    }
  }

  EXPECT_EQ(diff_count, 0) << " of " << test_size_y;

  AscendC::GmFree(x1);
  AscendC::GmFree(x2);
  AscendC::GmFree(x3);
  AscendC::GmFree(x4);
  AscendC::GmFree(x5);
  AscendC::GmFree(x6);
  AscendC::GmFree(x7);
  AscendC::GmFree(y);
  AscendC::GmFree(expect);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2E_ConcatMultInputs_Code,
    ::testing::Values(
      std::pair<std::vector<int>, std::vector<int>>{{16, 17, 31, 6, 2, 2, 1, 7}, {1 , 1, 16}},
      std::pair<std::vector<int>, std::vector<int>>{{31, 50, 3, 6, 2, 100, 1, 7}, {1 , 1, 31}},
      std::pair<std::vector<int>, std::vector<int>>{{12*1*31, 1, 3, 60, 2, 100, 1, 7}, {12 , 1, 31}},
      std::pair<std::vector<int>, std::vector<int>>{{12*1*31, 3, 48, 8, 2, 10, 1, 70}, {12 , 1, 31}},
      std::pair<std::vector<int>, std::vector<int>>{{12*1*31, 1, 3, 65, 77, 10, 1, 2}, {12 , 1, 31}}));
    


