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

class E2E_BackendMatmul_Code : public testing::Test, public testing::WithParamInterface<std::pair<std::vector<int>, std::vector<int>>> {
};

TEST_P(E2E_BackendMatmul_Code, CalculateCorrect) {
  uint8_t *x1 = (uint8_t *)AscendC::GmAlloc(8 * sizeof(uint8_t));
  AscendC::GmFree(x1);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2E_BackendMatmul_Code,
    ::testing::Values(std::pair<std::vector<int>, std::vector<int>>{{2, 2, 2}, {3, 3, 3}}));