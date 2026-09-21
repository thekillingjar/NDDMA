/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "common/data_utils.h"

namespace FlowFunc {
class DataUtilUTest : public testing::Test {
 protected:
  virtual void SetUp() {}

  virtual void TearDown() {}
};

TEST_F(DataUtilUTest, CalcDataSize_not_support_type) {
  int64_t ret = CalcDataSize({}, TensorDataType::DT_UNDEFINED);
  EXPECT_EQ(ret, -1);
}

TEST_F(DataUtilUTest, CalcDataSize_dim_navigate) {
  int64_t ret = CalcDataSize({2, -1, 3}, TensorDataType::DT_FLOAT);
  EXPECT_EQ(ret, -1);
}

TEST_F(DataUtilUTest, CalcDataSize_dim_overflow) {
  int64_t ret = CalcDataSize({UINT32_MAX, UINT32_MAX, UINT32_MAX}, TensorDataType::DT_FLOAT);
  EXPECT_EQ(ret, -1);
}

TEST_F(DataUtilUTest, CalcDataSize_overflow_with_type) {
  int64_t ret = CalcDataSize({UINT32_MAX, UINT32_MAX / 2}, TensorDataType::DT_FLOAT);
  EXPECT_EQ(ret, -1);
}

TEST_F(DataUtilUTest, CalcDataSize_normal) {
  int64_t ret = CalcDataSize({100, 200}, TensorDataType::DT_FLOAT);
  EXPECT_EQ(ret, 100 * 200 * sizeof(float));
}

TEST_F(DataUtilUTest, CalcDataSize_bit_count) {
  int64_t ret = CalcDataSize({100, 200}, TensorDataType::DT_UINT2);
  EXPECT_EQ(ret, 100 * 200 * 2 / 8);
}

TEST_F(DataUtilUTest, CalcDataSize_bit_padding) {
  int64_t ret = CalcDataSize({2}, TensorDataType::DT_UINT1);
  EXPECT_EQ(ret, 1);
}

TEST_F(DataUtilUTest, ConvertToInt64) {
  int64_t val = 0;
  EXPECT_FALSE(ConvertToInt64("xxxx", val));
  EXPECT_FALSE(ConvertToInt64("9999999999999999999999999999999999999999999999", val));
  EXPECT_TRUE(ConvertToInt64(" 987654 kB", val));
  EXPECT_EQ(val, 987654);
}
}  // namespace FlowFunc