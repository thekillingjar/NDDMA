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
 * \file test_sign.cpp
 * \brief
 */

#include <cmath>
#include "gtest/gtest.h"
#include "tikicpulib.h"

using namespace AscendC;

#include "test_api_utils.h"
#include "utils.h"
#include "cast.h"
#include "sign.h"

constexpr int gen_index_two = 2;
constexpr int gen_index_three = 3;

constexpr float SignSrcGen(const int index) {
  if (index % gen_index_two == 0) {
    return index + 1;
  }
  if (index % gen_index_three == 0) {
    return index * -1 - 1;
  }
  return 0;
}

inline float SignExpectGen(const int index, const float x) {
  if (index % gen_index_two == 0) {
    return 1.0;
  }
  if (index % gen_index_three == 0) {
    return -1.0;
  }
  return 0.0;
}

inline bool SignCompareGen(const float expect, const float actual) {
  return static_cast<int>(expect) == static_cast<int>(actual);
}

class TestApiSignFloat : public ::testing::Test, public testing::WithParamInterface<size_t> {};

TEST_P(TestApiSignFloat, Calc) {
  const int size = this->GetParam();
  UnaryTest<float, float>(size, SignExtend<float>, SignExpectGen, SignSrcGen, SignCompareGen);
}

INSTANTIATE_TEST_SUITE_P(DiffLength, TestApiSignFloat,
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

constexpr int64_t SignSrcGenInt64(const int index) {
  if (index % gen_index_two == 0) {
    return index + 1;
  }
  if (index % gen_index_three == 0) {
    return index * -1 - 1;
  }
  return 0;
}

inline int64_t SignExpectGenInt64(const int index, const int64_t x) {
  if (index % gen_index_two == 0) {
    return 1;
  }
  if (index % gen_index_three == 0) {
    return -1;
  }
  return 0;
}

inline bool SignCompareGenInt64(const int64_t expect, const int64_t actual) {
  return expect == actual;
}

class TestApiSignInt64 : public testing::Test, public testing::WithParamInterface<size_t> {};

TEST_P(TestApiSignInt64, Calc) {
  int size = this->GetParam();
  UnaryTest<int64_t, int64_t>(size, SignExtend<int64_t>, SignExpectGenInt64, SignSrcGenInt64, SignCompareGenInt64);
}

INSTANTIATE_TEST_SUITE_P(DiffLength, TestApiSignInt64,
                         ::testing::Values(
                             /* 1 block */ ONE_BLK_SIZE / sizeof(int64_t),
                             /* 1 repeat */ ONE_REPEAT_BYTE_SIZE / sizeof(int64_t),
                             /* max repeat */ MAX_REPEAT_NUM *ONE_REPEAT_BYTE_SIZE / sizeof(int64_t),
                             /* less than 1 block */ (ONE_BLK_SIZE - sizeof(int64_t)) / sizeof(int64_t),
                             /* less than 1 repeat */ (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int64_t),
                             /* less than max repeat */ (MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(int64_t),
                             /* mix block, repeat, max repeat*/
                             ((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                              (ONE_BLK_SIZE - sizeof(int64_t))) /
                                 sizeof(int64_t)));