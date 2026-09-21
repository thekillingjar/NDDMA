/**
* Copyright (c) Huawei Technologies Co., Ltd. 2025 All rights reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
#include <gtest/gtest.h>
#include <cmath>
#include "tikicpulib.h"

#include "autofuse_tiling_data.h"
extern "C" __global__ __aicore__ void load_bitwise_or_store_test(GM_ADDR x0, GM_ADDR x1, GM_ADDR y1, GM_ADDR workspace, GM_ADDR tiling);
extern "C" int64_t AutofuseTiling(uint32_t s0, uint32_t s1, uint32_t s2, AutofuseTilingData* tiling, uint32_t* workspaceSize, uint64_t *blockDim, uint32_t aiv_num, uint32_t ub_size);

class E2E_BackendLoadBitwiseOrStore_Code : public testing::Test, public testing::WithParamInterface<std::vector<int>> {
};

TEST_P(E2E_BackendLoadBitwiseOrStore_Code, CalculateCorrect) {
 auto test_shape = GetParam();

 uint64_t block_dim = 48;

 int test_size = test_shape[0] * test_shape[1] * test_shape[2];

 AutofuseTilingData tiling_data;
 uint8_t* x0 = (uint8_t *)AscendC::GmAlloc(test_size * sizeof(uint8_t) + 32);
 uint8_t* x1 = (uint8_t *)AscendC::GmAlloc(test_size * sizeof(uint8_t) + 32);
 uint8_t* y = (uint8_t *)AscendC::GmAlloc(test_size * sizeof(uint8_t) + 32);
 uint8_t* expect = (uint8_t *)AscendC::GmAlloc(test_size * sizeof(uint8_t) + 32);

 // Prepare test and expect data
 srand(1);
 for (int i = 0; i < test_size; i++) {
   // 生成 0和1 的随机数
   x0[i] = rand() % 2;
   x1[i] = rand() % 2;
   expect[i] = x0[i] | x1[i];
 }

 // Launch
 uint32_t ws_size = 0;
 AutofuseTiling(test_shape[0], test_shape[1], test_shape[2], &tiling_data, &ws_size, &block_dim, 48, 192*1024);
 printf("tiling key: %d, core_num: %d\n", tiling_data.tiling_key, tiling_data.block_dim);

 AscendC::SetKernelMode(KernelMode::AIV_MODE);
 ICPU_RUN_KF(load_bitwise_or_store_test, tiling_data.block_dim, (uint8_t *)x0, (uint8_t *)x1, (uint8_t *)y, nullptr, (uint8_t*)&tiling_data);

 // Count difference
 uint32_t diff_count = 0;
 for (int i = 0; i < test_size; i++) {
   if (y[i] != expect[i]) {
     diff_count++;
   }
 }

 EXPECT_EQ(diff_count, 0) << " of " << test_size;

 AscendC::GmFree(x0);
 AscendC::GmFree(x1);
 AscendC::GmFree(y);
 AscendC::GmFree(expect);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2E_BackendLoadBitwiseOrStore_Code,
                        ::testing::Values(std::vector<int>{2, 2, 2},  // 用例输入的维度需要与构图接口的dims_size匹配
                                          std::vector<int>{32, 16, 18},
                                          std::vector<int>{32, 512, 15}
                                          ));
