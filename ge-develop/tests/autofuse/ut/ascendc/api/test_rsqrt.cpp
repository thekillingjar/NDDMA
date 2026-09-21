/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include "gtest/gtest.h"
#include "tikicpulib.h"

using namespace AscendC;

#include "test_api_utils.h"
#include "rsqrt.h"

class TestApiRsqrt32 : public testing::Test, public testing::WithParamInterface<size_t> {};
TEST_P(TestApiRsqrt32, Calc) {
  int size = this->GetParam();
  UnaryTest<float, float>(size, RsqrtExtend<float>,
          [](double x){ return 1.0 / sqrt(x);});
}

INSTANTIATE_TEST_SUITE_P(DiffLength, TestApiRsqrt32,
         ::testing::Values(
             /* 1 block */ ONE_BLK_SIZE / sizeof(float),
             /* 1 repeat */ ONE_REPEAT_BYTE_SIZE / sizeof(float),
             /* max repeat */ MAX_REPEAT_NUM *ONE_REPEAT_BYTE_SIZE / sizeof(float),
             /* less than 1 block */ (ONE_BLK_SIZE - sizeof(float)) / sizeof(float),
             /* less than 1 repeat */ (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(float),
             /* less than max repeat */ (MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(float),
             /* mix block, repeat, max repeat*/
                 ((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                  (ONE_BLK_SIZE - sizeof(float))) /
                 sizeof(float)));

class TestApiRsqrt16 : public testing::Test, public testing::WithParamInterface<size_t> {};
TEST_P(TestApiRsqrt16, Calc) {
  int size = this->GetParam();
  UnaryTest<half, float>(size, RsqrtExtend<half>,
          [](double x){ return 1.0 / sqrt(x);});
}

INSTANTIATE_TEST_SUITE_P(DiffLength, TestApiRsqrt16,
        ::testing::Values(
            /* 1 block */ ONE_BLK_SIZE / sizeof(half),
            /* 1 repeat */ ONE_REPEAT_BYTE_SIZE / sizeof(half),
            /* max repeat */ MAX_REPEAT_NUM *ONE_REPEAT_BYTE_SIZE / sizeof(half),
            /* less than 1 block */ (ONE_BLK_SIZE - sizeof(half)) / sizeof(half),
            /* less than 1 repeat */ (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(half),
            /* less than max repeat */ (MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(half),
            /* mix block, repeat, max repeat*/
                ((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                 (ONE_BLK_SIZE - sizeof(half))) /
                sizeof(half)));
