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
extern "C" __global__ __aicore__
void dynamic_inputs_and_outputs(GM_ADDR x0, GM_ADDR x1, GM_ADDR x2, GM_ADDR x3, GM_ADDR x4,
                                GM_ADDR x5, GM_ADDR x6, GM_ADDR x7, GM_ADDR x8, GM_ADDR x9,
                                GM_ADDR y0, GM_ADDR y1, GM_ADDR y2, GM_ADDR y3, GM_ADDR y4,
                                GM_ADDR y5, GM_ADDR y6, GM_ADDR y7, GM_ADDR y8, GM_ADDR y9,
                                GM_ADDR workspace, GM_ADDR tiling);
extern "C" void GetTiling(AutofuseTilingData &tiling_data);

class E2E_MultiLoadAbsStore_Code
    : public testing::Test, public testing::WithParamInterface<std::pair<std::vector<int>, std::vector<int>>> {
};

template<typename T>
void SetParams(T &t, const std::vector<int> &test_shape) {
  t.s0 = test_shape[0];
  t.s1 = test_shape[1];
  t.s2 = test_shape[2];
}

TEST_P(E2E_MultiLoadAbsStore_Code, packing_func_with_10_io) {
  auto [test_shape, test_tiling] = GetParam();

  uint32_t block_dim = 48;
  int test_size = test_shape[0] * test_shape[1] * test_shape[2];

  AutofuseTilingData tiling_data;
  std::vector<half *> inputs;
  std::vector<half *> outputs;
  std::vector<half *> expects;
  for (int32_t i = 0; i < 10; ++i) {
    half *x = (half *) AscendC::GmAlloc(test_size * sizeof(half) + 32);
    half *y = (half *) AscendC::GmAlloc(test_size * sizeof(half) + 32);
    half *expect = (half *) AscendC::GmAlloc(test_size * sizeof(half) + 32);
    inputs.emplace_back(x);
    outputs.emplace_back(y);
    expects.emplace_back(expect);

    // Prepare test and expect data
    for (int j = 0; j < test_size; j++) {
      x[j] = -1;
      expect[j] = 1;
    }
  }

  // Launch
  tiling_data.block_dim = block_dim;
  if (test_tiling.empty()) { // tiling data 来源于tiling函数GetTiling
    GetTiling(tiling_data);
  }
  tiling_data.graph0_tiling_key = 0;
  SetParams(tiling_data.graph0_result0_g0_tiling_data, test_shape);
  SetParams(tiling_data.graph0_result0_g1_tiling_data, test_shape);
  SetParams(tiling_data.graph0_result0_g2_tiling_data, test_shape);
  SetParams(tiling_data.graph0_result0_g3_tiling_data, test_shape);
  SetParams(tiling_data.graph0_result0_g4_tiling_data, test_shape);
  SetParams(tiling_data.graph0_result0_g5_tiling_data, test_shape);
  SetParams(tiling_data.graph0_result0_g6_tiling_data, test_shape);
  SetParams(tiling_data.graph0_result0_g7_tiling_data, test_shape);
  SetParams(tiling_data.graph0_result0_g8_tiling_data, test_shape);
  SetParams(tiling_data.graph0_result0_g9_tiling_data, test_shape);
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(dynamic_inputs_and_outputs,
              tiling_data.block_dim,
              (uint8_t *) inputs[0], (uint8_t *) inputs[1],(uint8_t *) inputs[2],(uint8_t *) inputs[3],(uint8_t *) inputs[4],
              (uint8_t *) inputs[5], (uint8_t *) inputs[6],(uint8_t *) inputs[7],(uint8_t *) inputs[8],(uint8_t *) inputs[9],
              (uint8_t *) outputs[0], (uint8_t *) outputs[1],(uint8_t *) outputs[2],(uint8_t *) outputs[3],(uint8_t *) outputs[4],
              (uint8_t *) outputs[5], (uint8_t *) outputs[6],(uint8_t *) outputs[7],(uint8_t *) outputs[8],(uint8_t *) outputs[9],
              nullptr,
              (uint8_t *) &tiling_data);

  // Count difference
  uint32_t diff_count = 0;
  for (int k = 0; k < 10; ++k) {
    auto y = outputs[k];
    auto expect = expects[k];
    for (int i = 0; i < test_size; i++) {
      half diff = y[i] - expect[i];
      if (diff > (half) 0.0001 || diff < (half) - 0.0001) {
        diff_count++;
        std::cout << "group = " << k << "index = " << i << ", val = " << y[i].ToFloat() << std::endl;
      }
    }
  }


  EXPECT_EQ(diff_count, 0) << " of " << test_size;
  for (auto x : inputs) {
    AscendC::GmFree(x);
  }
  for (auto y : outputs) {
    AscendC::GmFree(y);
  }
  for (auto expect : expects) {
    AscendC::GmFree(expect);
  }
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2E_MultiLoadAbsStore_Code,
                         ::testing::Values(std::pair<std::vector<int>, std::vector<int>>{{4, 16, 16}, {}}
                         ));
