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
extern "C" __global__ __aicore__ void slice_concat_v2_test(GM_ADDR data0,
                                                           GM_ADDR data1,
                                                           GM_ADDR output,
                                                           GM_ADDR workspace,
                                                           GM_ADDR gm_tiling_data);
extern "C" int64_t AutofuseTiling(uint32_t s0,
                                  uint32_t s1,
                                  AutofuseTilingData *tiling,
                                  uint32_t *workspaceSize,
                                  uint64_t *blockDim,
                                  uint32_t aiv_num,
                                  uint32_t ub_size);

class E2E_BackendConcat_Code : public testing::Test, public testing::WithParamInterface<std::vector<int>> {
};

TEST_P(E2E_BackendConcat_Code, CalculateCorrect) {
  auto test_shape = GetParam();

  uint64_t block_dim = 48;

  auto input_col_size = test_shape[1] * 2; // 构图时slice一半数据, 所以输入要*2
  int input_size = test_shape[0] * input_col_size;
  int output_size = test_shape[0] * test_shape[1] * 2;

  AutofuseTilingData tiling_data;
  int32_t *input1 = (int32_t *) AscendC::GmAlloc(input_size * sizeof(int32_t) + 32);
  int32_t *input2 = (int32_t *) AscendC::GmAlloc(input_size * sizeof(int32_t) + 32);
  int32_t *y = (int32_t *) AscendC::GmAlloc(output_size * sizeof(int32_t) + 32);
  int32_t *expect = (int32_t *) AscendC::GmAlloc(output_size * sizeof(int32_t) + 32);

  // Prepare test and expect data
  int32_t value = 1;
  for (int i = 0; i < test_shape[0]; ++i) {
    for (int j = 0; j < test_shape[1]; ++j) {
      input1[i * input_col_size + j] = value;
      input2[i * input_col_size + j] = 100000 + value;
      expect[i * 2 * test_shape[1] + j] = -1 * input1[i * input_col_size + j];
      expect[i * 2 * test_shape[1] + test_shape[1] + j] = -1 * input2[i * input_col_size + j];
      value++;
    }
  }

  // Launch
  uint32_t ws_size = 0;
  AutofuseTiling(test_shape[0], test_shape[1], &tiling_data, &ws_size, &block_dim, 1, 192 * 1024);

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(slice_concat_v2_test,
              tiling_data.block_dim,
              (uint8_t *) input1,
              (uint8_t *) input2,
              (uint8_t *) y,
              nullptr,
              (uint8_t *) &tiling_data);

  // Count difference
  uint32_t diff_count = 0;
  for (int i = 0; i < output_size; i++) {
    if (y[i] != expect[i]) {
      if (i < 32) {
        std::cout << "index = " << i << ", expect = " << expect[i] << ", actual = " << y[i] << std::endl;
      }
      diff_count++;
    }
  }

  EXPECT_EQ(diff_count, 0) << " of " << output_size;

  AscendC::GmFree(input1);
  AscendC::GmFree(input2);
  AscendC::GmFree(y);
  AscendC::GmFree(expect);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2E_BackendConcat_Code,
                         ::testing::Values(
                             std::vector<int>{15, 7},
                             std::vector<int>{16, 7},
                             std::vector<int>{17, 7},
                             // std::vector<int>{15, 9}, cpu nddma bug，临时注释
                             // std::vector<int>{17, 9}, cpu nddma bug，临时注释
                             std::vector<int>{31, 15},
                             std::vector<int>{31, 17},
                             // std::vector<int>{31, 31}, cpu nddma bug，临时注释
                             std::vector<int>{31, 33},
                             std::vector<int>{16, 63},
                             std::vector<int>{31, 65}
                         ));