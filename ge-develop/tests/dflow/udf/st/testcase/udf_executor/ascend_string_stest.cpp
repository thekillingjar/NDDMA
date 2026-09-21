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
#include "flow_func/ascend_string.h"

namespace FlowFunc {
class AscendStringSTest : public testing::Test {
 protected:
  virtual void SetUp() {}

  virtual void TearDown() {}
};

TEST_F(AscendStringSTest, string_equal) {
  AscendString src_str("abc");
  AscendString target_str("abc");
  AscendString empty_str1;
  AscendString empty_str2;
  EXPECT_EQ(empty_str1 == empty_str2, true);
  EXPECT_EQ(src_str == empty_str2, false);
  EXPECT_EQ(empty_str2 == src_str, false);
  EXPECT_EQ(src_str == target_str, true);
}

TEST_F(AscendStringSTest, string_not_equal) {
  AscendString src_str("abc");
  AscendString target_str("def");
  AscendString empty_str1;
  AscendString empty_str2;
  EXPECT_NE(empty_str1, src_str);
  EXPECT_NE(src_str, empty_str1);
  EXPECT_NE(src_str, target_str);
  EXPECT_EQ(empty_str1 != src_str, true);
  EXPECT_EQ(src_str != empty_str1, true);
  EXPECT_EQ(target_str != src_str, true);
  EXPECT_EQ(empty_str1 != empty_str2, false);
}

TEST_F(AscendStringSTest, string_large_than) {
  AscendString src_str("abc");
  AscendString target_str("aaa");
  AscendString empty_str1;
  AscendString empty_str2;
  EXPECT_EQ(empty_str1 > empty_str2, false);
  EXPECT_EQ(empty_str1 < empty_str2, false);
  EXPECT_EQ(src_str > empty_str1, true);
  EXPECT_EQ(empty_str1 > src_str, false);
  EXPECT_EQ(empty_str1 < src_str, true);
  EXPECT_EQ(src_str < empty_str1, false);
  EXPECT_EQ(src_str < target_str, false);
}

TEST_F(AscendStringSTest, string_large_equal_than) {
  AscendString src_str("abc");
  AscendString target_str("abc");
  AscendString empty_str1;
  AscendString empty_str2;
  EXPECT_EQ(empty_str1 >= empty_str2, true);
  EXPECT_EQ(empty_str1 <= empty_str2, true);
  EXPECT_EQ(src_str >= empty_str1, true);
  EXPECT_EQ(empty_str1 >= src_str, false);
  EXPECT_EQ(empty_str1 <= src_str, true);
  EXPECT_EQ(src_str <= empty_str1, false);
  EXPECT_EQ(src_str <= target_str, true);
  EXPECT_EQ(src_str >= target_str, true);
}

TEST_F(AscendStringSTest, with_length) {
  size_t trunk_size = strlen("strcmp");
  AscendString strcmp1("strcmp1", trunk_size);
  AscendString strcmp2("strcmp2", trunk_size);
  AscendString strcmp3("strcmp1");
  ASSERT_EQ(strcmp1.GetLength(), trunk_size);
  ASSERT_EQ(strcmp2.GetLength(), trunk_size);
  ASSERT_GT(strcmp3.GetLength(), trunk_size);
  ASSERT_TRUE(strcmp1 == strcmp2);
  ASSERT_FALSE(strcmp1 == strcmp3);
}

TEST_F(AscendStringSTest, null_size) {
  AscendString strcmp1(nullptr, 1);
  ASSERT_EQ(strcmp1.GetLength(), 0);
}

TEST_F(AscendStringSTest, with_terminal) {
  std::string with_terminal_str("abc\0def", 7);
  AscendString with_terminal("abc\0def", 7);
  AscendString without_terminal("abc\0def");
  ASSERT_EQ(with_terminal.GetLength(), with_terminal_str.length());
  ASSERT_GT(with_terminal.GetLength(), without_terminal.GetLength());
  std::string re_build_str(with_terminal.GetString(), with_terminal.GetLength());
  ASSERT_EQ(re_build_str, with_terminal_str);
}
}  // namespace FlowFunc