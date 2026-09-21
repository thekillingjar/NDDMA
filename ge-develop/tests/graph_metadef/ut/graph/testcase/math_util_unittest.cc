/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/utils/math_util.h"
#include <gtest/gtest.h>
namespace ge {
class MathUtilUT : public testing::Test {};
TEST_F(MathUtilUT, AddOverflow_NotOverflow) {
  size_t i = 0;
  size_t j = 0;
  size_t ret;
  EXPECT_FALSE(AddOverflow(i, j, ret));
  EXPECT_EQ(ret, 0);

  i = 100;
  j = 200;
  EXPECT_FALSE(AddOverflow(i, j, ret));
  EXPECT_EQ(ret, 300);

  i = 0xFFFFFFFFFFFFFFFF;
  j = 0;
  EXPECT_FALSE(AddOverflow(i, j, ret));
  EXPECT_EQ(ret, 0xFFFFFFFFFFFFFFFF);

  i = 0x7FFFFFFFFFFFFFFF;
  j = 0x8000000000000000;
  EXPECT_FALSE(AddOverflow(i, j, ret));
  EXPECT_EQ(ret, 0xFFFFFFFFFFFFFFFF);
}
TEST_F(MathUtilUT, AddOverflow_Overflow) {

  size_t i = 0xFFFFFFFFFFFFFFFF;
  size_t j = 1;
  size_t ret;
  EXPECT_TRUE(AddOverflow(i, j, ret));

  i = 0x7FFFFFFFFFFFFFFF;
  j = 0x8000000000000001;
  EXPECT_TRUE(AddOverflow(i, j, ret));
}
TEST_F(MathUtilUT, AddOverflow_OverflowUint8) {
  uint8_t i = 255;
  uint8_t j = 0;
  uint8_t ret;
  EXPECT_FALSE(AddOverflow(i, j, ret));

  i = 255;
  j = 1;
  EXPECT_TRUE(AddOverflow(i, j, ret));

  i = 2;
  j = 254;
  EXPECT_TRUE(AddOverflow(i, j, ret));
}

TEST_F(MathUtilUT, AddOverflow_OverflowDiffType) {
  uint16_t i = 255;
  uint8_t j = 0;
  uint8_t ret;
  EXPECT_FALSE(AddOverflow(i, j, ret));
  EXPECT_FALSE(AddOverflow(j, i, ret));

  i = 256;
  j = 0;
  EXPECT_TRUE(AddOverflow(i, j, ret));
  EXPECT_TRUE(AddOverflow(j, i, ret));

  i = 100;
  j = 156;
  EXPECT_TRUE(AddOverflow(i, j, ret));
  EXPECT_TRUE(AddOverflow(j, i, ret));
}

TEST_F(MathUtilUT, AddOverflow_IntUnderflow) {
  int8_t i = -128;
  int8_t j = 0;
  int8_t ret;
  EXPECT_FALSE(AddOverflow(i, j, ret));
  EXPECT_FALSE(AddOverflow(j, i, ret));

  i = -128;
  j = -1;
  EXPECT_TRUE(AddOverflow(i, j, ret));
  EXPECT_TRUE(AddOverflow(j, i, ret));
}

TEST_F(MathUtilUT, AddOverflow_IntDiffTypeUnderflow) {
  int16_t i = -128;
  int8_t j = 0;
  int8_t ret;
  EXPECT_FALSE(AddOverflow(i, j, ret));
  EXPECT_FALSE(AddOverflow(j, i, ret));

  i = -129;
  j = 0;
  EXPECT_TRUE(AddOverflow(i, j, ret));
  EXPECT_TRUE(AddOverflow(j, i, ret));

  i = -128;
  j = -1;
  EXPECT_TRUE(AddOverflow(i, j, ret));
  EXPECT_TRUE(AddOverflow(j, i, ret));
}

TEST_F(MathUtilUT, RoundUp) {
  EXPECT_EQ(RoundUp(10, 8), 16);
  EXPECT_EQ(RoundUp(10, 3), 12);
  EXPECT_EQ(RoundUp(10, 2), 10);
  EXPECT_EQ(RoundUp(10, 1), 10);
  // fail
  EXPECT_EQ(RoundUp(std::numeric_limits<uint64_t>::max(), 10), 0);
}

TEST_F(MathUtilUT, CeilDiv16) {
  EXPECT_EQ(CeilDiv16(0), 0);
  EXPECT_EQ(CeilDiv16(1), 1);
  EXPECT_EQ(CeilDiv16(15), 1);
  EXPECT_EQ(CeilDiv16(16), 1);
  EXPECT_EQ(CeilDiv16(17), 2);
  EXPECT_EQ(CeilDiv16(32), 2);
  EXPECT_EQ(CeilDiv16(33), 3);
}

TEST_F(MathUtilUT, CeilDiv32) {
  EXPECT_EQ(CeilDiv32(0), 0);
  EXPECT_EQ(CeilDiv32(1), 1);
  EXPECT_EQ(CeilDiv32(31), 1);
  EXPECT_EQ(CeilDiv32(32), 1);
  EXPECT_EQ(CeilDiv32(33), 2);
  EXPECT_EQ(CeilDiv32(63), 2);
  EXPECT_EQ(CeilDiv32(64), 2);
  EXPECT_EQ(CeilDiv32(65), 3);
}

TEST_F(MathUtilUT, MulOverflow_NotOverflow) {
  int32_t i;
  EXPECT_FALSE(MulOverflow(10, 20, i));
  EXPECT_EQ(i, 200);

  EXPECT_FALSE(MulOverflow(-10, -20, i));
  EXPECT_EQ(i, 200);

  EXPECT_FALSE(MulOverflow(-10, 20, i));
  EXPECT_EQ(i, -200);

  EXPECT_FALSE(MulOverflow(0, 0, i));
  EXPECT_EQ(i, 0);
}

TEST_F(MathUtilUT, MulOverflow_Overflow) {
  int32_t i;
  EXPECT_TRUE(MulOverflow(std::numeric_limits<int32_t>::max(), 2, i));
  EXPECT_TRUE(MulOverflow(std::numeric_limits<int32_t>::min(), 2, i));
  EXPECT_TRUE(MulOverflow(std::numeric_limits<int32_t>::min(), -1, i));
  EXPECT_TRUE(MulOverflow(2, std::numeric_limits<int32_t>::max(), i));
  EXPECT_TRUE(MulOverflow(2, std::numeric_limits<int32_t>::min(), i));
  EXPECT_TRUE(MulOverflow(-1, std::numeric_limits<int32_t>::min(), i));
  EXPECT_TRUE(MulOverflow(std::numeric_limits<int32_t>::max() / 2 + 1, std::numeric_limits<int32_t>::max() / 2 + 1, i));
  EXPECT_TRUE(MulOverflow(std::numeric_limits<int32_t>::min() / 2 - 1, std::numeric_limits<int32_t>::min() / 2 - 1, i));
}

TEST_F(MathUtilUT, MulOverflow_OverflowUint8) {
  uint8_t i;
  EXPECT_TRUE(MulOverflow(static_cast<uint8_t>(255), static_cast<uint8_t>(2), i));
  EXPECT_TRUE(MulOverflow(static_cast<uint8_t>(2), static_cast<uint8_t>(255), i));
}

TEST_F(MathUtilUT, MulOverflow_OverflowDiffType) {
  uint8_t i;
  EXPECT_TRUE(MulOverflow(300, 1, i));
  EXPECT_TRUE(MulOverflow(1, 300, i));
}

TEST_F(MathUtilUT, RoundUpOverflow_Overflow_Int8) {
  int8_t value = 127;
  int8_t v1;
  EXPECT_TRUE(RoundUpOverflow(value, static_cast<int8_t>(4), v1));
}
TEST_F(MathUtilUT, RoundUpOverflow_Overflow_Int64) {
  int64_t value = std::numeric_limits<int64_t>::max() - 2;
  int64_t v1;
  EXPECT_TRUE(RoundUpOverflow(value, static_cast<int64_t>(4), v1));
}
TEST_F(MathUtilUT, RoundUpOverflow_Overflow_RetValueSmall) {
  int32_t value = 1024;
  int8_t v1;
  EXPECT_TRUE(RoundUpOverflow(value, static_cast<int32_t>(4), v1));
}
TEST_F(MathUtilUT, RoundUpOverflow_Overflow_MaxUint32) {
  for (uint32_t i = 0U; i < 7; ++i) {
    uint32_t value = std::numeric_limits<uint32_t>::max() - i;
    uint32_t v1;
    EXPECT_TRUE(RoundUpOverflow(value, static_cast<uint32_t>(8), v1));
  }
}
TEST_F(MathUtilUT, RoundUpOverflow_Overflow_Inplace) {
  int64_t value = std::numeric_limits<int64_t>::max() - 2;
  EXPECT_TRUE(RoundUpOverflow(value, static_cast<int64_t>(4), value));
}
TEST_F(MathUtilUT, RoundUpOverflow_NotOverflow_MaxUint32) {
  uint32_t value = std::numeric_limits<uint32_t>::max() - 7U;
  uint32_t v1;
  EXPECT_FALSE(RoundUpOverflow(value, static_cast<uint32_t>(8), v1));
  EXPECT_EQ(v1, std::numeric_limits<uint32_t>::max() - 7U);
}
TEST_F(MathUtilUT, RoundUpOverflow_NotOverflow_EvenlyDivInt8) {
  int8_t value = 64;
  int8_t v1;
  EXPECT_FALSE(RoundUpOverflow(value, static_cast<int8_t>(4), v1));
  EXPECT_EQ(v1, 64);
}
TEST_F(MathUtilUT, RoundUpOverflow_NotOverflow_EvenlyDivInt32) {
  int32_t value = 2048;
  int32_t v1;
  EXPECT_FALSE(RoundUpOverflow(value, static_cast<int32_t>(32), v1));
  EXPECT_EQ(v1, 2048);
}
TEST_F(MathUtilUT, RoundUpOverflow_NotOverflow_NotEvenlyDivInt32) {
  for (int32_t i = 0; i < 4; ++i) {
    int32_t value = 2047 - i;
    int32_t v1;
    EXPECT_FALSE(RoundUpOverflow(value, static_cast<int32_t>(32), v1));
    EXPECT_EQ(v1, 2048);
  }
}
TEST_F(MathUtilUT, RoundUpOverflow_NotOverflow_Inplace) {
  int32_t value = 2048;
  EXPECT_FALSE(RoundUpOverflow(value, static_cast<int32_t>(32), value));
  EXPECT_TRUE(value == 2048);

  value = 2040;
  EXPECT_FALSE(RoundUpOverflow(value, static_cast<int32_t>(32), value));
  EXPECT_TRUE(value == 2048);

  value = 32;
  EXPECT_FALSE(RoundUpOverflow(value,value, value));
  EXPECT_TRUE(value == 32);
}
TEST_F(MathUtilUT, RoundUpOverflow_Failed_MultipleOfZero) {
  int8_t value = 10;
  int8_t v1;
  EXPECT_TRUE(RoundUpOverflow(value, static_cast<int8_t>(0), v1));
}
}  // namespace ge
