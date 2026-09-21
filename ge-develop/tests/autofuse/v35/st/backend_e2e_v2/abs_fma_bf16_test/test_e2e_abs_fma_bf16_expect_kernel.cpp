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
extern "C" __global__ __aicore__ void abs_fma_bf16_test(GM_ADDR x1, GM_ADDR x2, GM_ADDR x3, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);
extern "C" int64_t AutofuseTiling(uint32_t s0, AutofuseTilingData* tiling, uint32_t* workspaceSize, uint64_t *blockDim, uint32_t aiv_num, uint32_t ub_size);

class E2E_BackendAbsFmaBf16_Code : public testing::Test, public testing::WithParamInterface<std::vector<int>> {
};

TEST_P(E2E_BackendAbsFmaBf16_Code, CalculateCorrect) {
  auto test_shape = GetParam();

  uint64_t block_dim = 1;

  AutofuseTilingData tiling_data;
  bfloat16_t* input1 = (bfloat16_t *)AscendC::GmAlloc(test_shape[0] * sizeof(bfloat16_t) + 32);
  bfloat16_t* input2 = (bfloat16_t *)AscendC::GmAlloc(test_shape[0] * sizeof(bfloat16_t) + 32);
  bfloat16_t* input3 = (bfloat16_t *)AscendC::GmAlloc(test_shape[0] * sizeof(bfloat16_t) + 32);
  bfloat16_t* y = (bfloat16_t *)AscendC::GmAlloc(test_shape[0] * sizeof(bfloat16_t) + 32);
  bfloat16_t *expect = (bfloat16_t *)AscendC::GmAlloc(test_shape[0] * sizeof(bfloat16_t) + 32);

  // Prepare test and expect data
  srand(1);
  for (int i = 0; i < test_shape[0]; i++) {
    input1[i] = rand() / (double)RAND_MAX;
    input2[i] = rand() / (double)RAND_MAX;
    input3[i] = rand() / (double)RAND_MAX;
    expect[i] = input1[i] * input2[i] + input3[i];
  }

  // Launch
  uint32_t ws_size = 0;
  AutofuseTiling(test_shape[0], &tiling_data, &ws_size, &block_dim, 48, 192*1024);

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(abs_fma_bf16_test, tiling_data.block_dim, (uint8_t *)input1, (uint8_t *)input2, (uint8_t *)input3, (uint8_t *)y, nullptr, (uint8_t*)&tiling_data);

  // Count difference
  uint32_t diff_count = 0;
  for (int i = 0; i < test_shape[0]; i++) {
    if (fabs(y[i] - expect[i]) > 1e-6) {
      std::cout << "y[" << i << "]" << y[i] << std::endl;
      std::cout << "expect[" << i << "]" << expect[i] << std::endl;
      diff_count++;
    }
  }

  EXPECT_EQ(diff_count, 0) << " of " << test_shape[0];

  AscendC::GmFree(input1);
  AscendC::GmFree(input2);
  AscendC::GmFree(input3);
  AscendC::GmFree(y);
  AscendC::GmFree(expect);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2E_BackendAbsFmaBf16_Code,
    ::testing::Values(std::vector<int>{32}  // 用例输入的维度需要与构图接口的dims_size匹配
                      ));
