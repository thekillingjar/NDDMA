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
#include <random>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_api_utils.h"
#include "logical_not.h"

using namespace AscendC;

inline uint8_t LogicalNotSrcGenUInt8(const int index) {
  // 创建随机数生成器
  std::random_device rd;   // 随机数设备，用于获取随机种子
  std::mt19937 gen(rd());  // 以随机设备作为种子的Mersenne Twister生成器

  // 产生一个0或1的均匀分布的随机数
  std::uniform_int_distribution<> dis(0, 1);  // 0和1之间的均匀分布
  int randNum = dis(gen);                     // 生成随机数
  return randNum != 0;                        // 如果随机数不为0，则返回true，否则返回false
}

inline uint8_t LogicalNotExpectGenUInt8(const uint8_t x) {
  return !x;
}

inline bool LogicalNotCompareGenUInt8(const uint8_t expect, const uint8_t actual) {
  return expect == actual;
}

class TestApiLogicalNot8 : public testing::Test, public testing::WithParamInterface<size_t> {};

TEST_P(TestApiLogicalNot8, Calc) {
  int size = this->GetParam();
  UnaryTest<uint8_t, half>(size, LogicalNot<uint8_t>, LogicalNotExpectGenUInt8, LogicalNotSrcGenUInt8,
                     LogicalNotCompareGenUInt8);
}

INSTANTIATE_TEST_SUITE_P(DiffLength, TestApiLogicalNot8,
                         ::testing::Values(
                             /* 1 block */ ONE_BLK_SIZE / sizeof(uint8_t),
                             /* 1 repeat */ ONE_REPEAT_BYTE_SIZE / sizeof(uint8_t),
                             /* max repeat */ MAX_REPEAT_NUM *ONE_REPEAT_BYTE_SIZE / sizeof(uint8_t),
                             /* less than 1 block */ (ONE_BLK_SIZE - sizeof(uint8_t)) / sizeof(uint8_t),
                             /* less than 1 repeat */ (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(uint8_t),
                             /* less than max repeat */
                             (MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(uint8_t),
                             /* mix block, repeat, max repeat*/
                             ((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                              (ONE_BLK_SIZE - sizeof(uint8_t))) /
                                 sizeof(uint8_t)));

inline int64_t LogicalNotSrcGenInt64(const int index) {
  // 创建随机数生成器
  std::random_device rd;   // 随机数设备，用于获取随机种子
  std::mt19937 gen(rd());  // 以随机设备作为种子的Mersenne Twister生成器

  // 产生一个0或1的均匀分布的随机数
  std::uniform_int_distribution<> dis(0, 1);  // 0和1之间的均匀分布
  int randNum = dis(gen);                     // 生成随机数
  return randNum;
}

inline int64_t LogicalNotExpectGenInt64(const int64_t x) {
  return !x;
}

inline bool LogicalNotCompareGenInt64(const int64_t expect, const int64_t actual) {
  return expect == actual;
}

class TestApiLogicalNotInt64 : public testing::Test, public testing::WithParamInterface<size_t> {};

TEST_P(TestApiLogicalNotInt64, Calc) {
  int size = this->GetParam();
  UnaryTest<int64_t, half>(size, LogicalNot<int64_t>, LogicalNotExpectGenInt64, LogicalNotSrcGenInt64,
                     LogicalNotCompareGenInt64);
}

INSTANTIATE_TEST_SUITE_P(DiffLength, TestApiLogicalNotInt64,
                         ::testing::Values(
                             /* 1 block */ ONE_BLK_SIZE / sizeof(int64_t),
                             /* 1 repeat */ ONE_REPEAT_BYTE_SIZE / sizeof(int64_t),
                             /* max repeat */ MAX_REPEAT_NUM *ONE_REPEAT_BYTE_SIZE / sizeof(int64_t),
                             /* less than 1 block */ (ONE_BLK_SIZE - sizeof(int64_t)) / sizeof(int64_t),
                             /* less than 1 repeat */ (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int64_t),
                             /* less than max repeat */
                             (MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(int64_t),
                             /* mix block, repeat, max repeat*/
                           ((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                             (ONE_BLK_SIZE - sizeof(int64_t))) /sizeof(int64_t)));