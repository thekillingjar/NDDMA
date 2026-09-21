/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
* \file test_isnan.cpp
* \brief
*/

#include <cmath>
#include "gtest/gtest.h"
#include "tikicpulib.h"

#include "test_api_utils.h"
#include "isnan.h"

using namespace AscendC;


constexpr int gen_index_two = 2;
constexpr int gen_index_three = 3;
constexpr int gen_index_div = 1000;
constexpr float gen_float_suffix = 0.12;

constexpr float IsnanGen(const int index) {
  if (index % gen_index_two == 0) {
    return index / gen_index_div + gen_float_suffix;
  }
  if (index % gen_index_three == 0) {
    return (index / gen_index_div + gen_float_suffix) * -1;
  }
  return 0.0 / 0.0;
}

half IsnanGenHalf(const int index) {
  if (index % gen_index_two == 0) {
    return index / gen_index_div + gen_float_suffix;
  }
  if (index % gen_index_three == 0) {
    return (index / gen_index_div + gen_float_suffix) * -1;
  }
  return GetScalarBitcodeValue<uint16_t, half>(0x7E00);
}

constexpr bool IsnanCompare(const uint8_t a, const uint8_t b) {
  return a == b;
}

constexpr uint8_t ExpectGen(const int index, const float x) {
  return index % gen_index_two == 0 ? 0 : index % gen_index_three == 0 ? 0 : 1;
}

class TestApiIsnan32 : public ::testing::Test, public testing::WithParamInterface<size_t> {
};

TEST_P(TestApiIsnan32, Calc) {
  const int size = this->GetParam();
  UnaryTest<float, uint8_t>(size, IsnanExtend<float>, ExpectGen, IsnanGen, IsnanCompare);
}

INSTANTIATE_TEST_SUITE_P(DiffLength, TestApiIsnan32,
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


class TestApiIsnan16 : public testing::Test, public testing::WithParamInterface<size_t> {
};

TEST_P(TestApiIsnan16, Calc) {
  int size = this->GetParam();
  UnaryTest<half, uint8_t>(size, IsnanExtend<half>, ExpectGen, IsnanGenHalf, IsnanCompare);
}

INSTANTIATE_TEST_SUITE_P(DiffLength, TestApiIsnan16,
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