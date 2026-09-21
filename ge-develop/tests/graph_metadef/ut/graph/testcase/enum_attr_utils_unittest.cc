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
#include "graph/utils/enum_attr_utils.h"

namespace ge {
class UtestEnumAttrUtils : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(UtestEnumAttrUtils, TestGetEnumAttrName) {
  // 验证一个属性
  vector<string> enum_attr_names = {};
  string attr_name = "name0";
  string enum_attr_name = "";
  bool is_new_attr = false;
  EnumAttrUtils::GetEnumAttrName(enum_attr_names, attr_name, enum_attr_name, is_new_attr);
  ASSERT_EQ(is_new_attr, true);
  ASSERT_EQ(enum_attr_name.size(), 2U);
  ASSERT_EQ(enum_attr_name.at(1), 1);
  ASSERT_EQ(enum_attr_names.size(), 1U);
  ASSERT_EQ(enum_attr_names[0], attr_name);

  // 验证两个不同的属性
  attr_name = "name1";
  enum_attr_name = "";
  is_new_attr = false;
  EnumAttrUtils::GetEnumAttrName(enum_attr_names, attr_name, enum_attr_name, is_new_attr);
  ASSERT_EQ(is_new_attr, true);
  ASSERT_EQ(enum_attr_name.size(), 2U);
  ASSERT_EQ(enum_attr_name.at(1), 2);
  ASSERT_EQ(enum_attr_names.size(), 2U);
  ASSERT_EQ(enum_attr_names[1], attr_name);

  // 验证kMaxValueOfEachDigit个不同属性， 边界值为kMaxValueOfEachDigit，一位数最大值
  for (uint32_t i = 2U; i < kMaxValueOfEachDigit; i++) {
    attr_name = "name" + to_string(i);
    enum_attr_name = "";
    is_new_attr = false;
    EnumAttrUtils::GetEnumAttrName(enum_attr_names, attr_name, enum_attr_name, is_new_attr);
  }
  ASSERT_EQ(is_new_attr, true);
  ASSERT_EQ(enum_attr_name.size(), 2U);
  ASSERT_EQ(enum_attr_name.at(1), kMaxValueOfEachDigit);
  ASSERT_EQ(enum_attr_names.size(), kMaxValueOfEachDigit);
  ASSERT_EQ(enum_attr_names[kMaxValueOfEachDigit - 1U], attr_name);

  // 验证kMaxValueOfEachDigit + 1个不同属性， 两位数初始值
  attr_name = "name" + to_string(kMaxValueOfEachDigit);
  enum_attr_name = "";
  is_new_attr = false;
  EnumAttrUtils::GetEnumAttrName(enum_attr_names, attr_name, enum_attr_name, is_new_attr);
  ASSERT_EQ(is_new_attr, true);
  ASSERT_EQ(enum_attr_name.size(), 3U);
  ASSERT_EQ(enum_attr_name.at(1), 1U);
  ASSERT_EQ(enum_attr_name.at(2), 2U);
  ASSERT_EQ(enum_attr_names.size(), kMaxValueOfEachDigit + 1U);
  ASSERT_EQ(enum_attr_names[kMaxValueOfEachDigit], attr_name);

  // 验证kMaxValueOfEachDigit * kMaxValueOfEachDigit个不同属性， 两位数初始值的最大值
  for (uint32_t i = kMaxValueOfEachDigit + 1U; i < kMaxValueOfEachDigit * kMaxValueOfEachDigit; i++) {
    attr_name = "name" + to_string(i);
    enum_attr_name = "";
    is_new_attr = false;
    EnumAttrUtils::GetEnumAttrName(enum_attr_names, attr_name, enum_attr_name, is_new_attr);
  }
  ASSERT_EQ(is_new_attr, true);
  ASSERT_EQ(enum_attr_name.size(), 3U);
  ASSERT_EQ(enum_attr_name.at(1), kMaxValueOfEachDigit);
  ASSERT_EQ(enum_attr_name.at(2), kMaxValueOfEachDigit);
  ASSERT_EQ(enum_attr_names.size(), kMaxValueOfEachDigit * kMaxValueOfEachDigit);
  ASSERT_EQ(enum_attr_names[kMaxValueOfEachDigit * kMaxValueOfEachDigit - 1U], attr_name);

  // 验证kMaxValueOfEachDigit * kMaxValueOfEachDigit + 1个不同属性， 三位数初始值
  attr_name = "name" + to_string(kMaxValueOfEachDigit * kMaxValueOfEachDigit);
  enum_attr_name = "";
  is_new_attr = false;
  EnumAttrUtils::GetEnumAttrName(enum_attr_names, attr_name, enum_attr_name, is_new_attr);
  ASSERT_EQ(is_new_attr, true);
  ASSERT_EQ(enum_attr_name.size(), 4U);
  ASSERT_EQ(enum_attr_name.at(1), 1U);
  ASSERT_EQ(enum_attr_name.at(2), 1U);
  ASSERT_EQ(enum_attr_name.at(3), 2U);
  ASSERT_EQ(enum_attr_names.size(), kMaxValueOfEachDigit * kMaxValueOfEachDigit + 1U);
  ASSERT_EQ(enum_attr_names[kMaxValueOfEachDigit * kMaxValueOfEachDigit], attr_name);

  // 验证属性名重复场景1
  attr_name = "name0";
  enum_attr_name = "";
  is_new_attr = true;
  EnumAttrUtils::GetEnumAttrName(enum_attr_names, attr_name, enum_attr_name, is_new_attr);
  ASSERT_EQ(is_new_attr, false);
  ASSERT_EQ(enum_attr_name.size(), 2U);
  ASSERT_EQ(enum_attr_name.at(1), 1);
  ASSERT_EQ(enum_attr_names[0], attr_name);

  // 验证属性名重复场景2
  attr_name = "name" + to_string(kMaxValueOfEachDigit * kMaxValueOfEachDigit);
  enum_attr_name = "";
  is_new_attr = true;
  EnumAttrUtils::GetEnumAttrName(enum_attr_names, attr_name, enum_attr_name, is_new_attr);
  ASSERT_EQ(is_new_attr, false);
  ASSERT_EQ(enum_attr_name.size(), 4U);
  ASSERT_EQ(enum_attr_name.at(1), 1U);
  ASSERT_EQ(enum_attr_name.at(2), 1U);
  ASSERT_EQ(enum_attr_name.at(3), 2U);
  ASSERT_EQ(enum_attr_names.size(), kMaxValueOfEachDigit * kMaxValueOfEachDigit + 1U);
  ASSERT_EQ(enum_attr_names[kMaxValueOfEachDigit * kMaxValueOfEachDigit], attr_name);
}

TEST_F(UtestEnumAttrUtils, TestGetEnumAttrValue) {
  // 验证一个属性值
  vector<string> enum_attr_values = {};
  string attr_value = "value0";
  int64_t enum_attr_value = 0;
  EnumAttrUtils::GetEnumAttrValue(enum_attr_values, attr_value, enum_attr_value);
  ASSERT_EQ(enum_attr_value, 0);
  ASSERT_EQ(enum_attr_values.size(), 1U);
  ASSERT_EQ(enum_attr_values[0], attr_value);

  // 验证两个属性值
  attr_value = "value1";
  enum_attr_value = 0;
  EnumAttrUtils::GetEnumAttrValue(enum_attr_values, attr_value, enum_attr_value);
  ASSERT_EQ(enum_attr_value, 1);
  ASSERT_EQ(enum_attr_values.size(), 2U);
  ASSERT_EQ(enum_attr_values[1], attr_value);

  // 验证重复属性场景
  attr_value = "value1";
  enum_attr_value = 0;
  EnumAttrUtils::GetEnumAttrValue(enum_attr_values, attr_value, enum_attr_value);
  ASSERT_EQ(enum_attr_value, 1);
  ASSERT_EQ(enum_attr_values.size(), 2U);
  ASSERT_EQ(enum_attr_values[1], attr_value);

  // 验证重复属性场景
  attr_value = "value0";
  enum_attr_value = 0;
  EnumAttrUtils::GetEnumAttrValue(enum_attr_values, attr_value, enum_attr_value);
  ASSERT_EQ(enum_attr_value, 0);
  ASSERT_EQ(enum_attr_values.size(), 2U);
  ASSERT_EQ(enum_attr_values[0], attr_value);

  // 验证三个属性值
  attr_value = "value2";
  enum_attr_value = 0;
  EnumAttrUtils::GetEnumAttrValue(enum_attr_values, attr_value, enum_attr_value);
  ASSERT_EQ(enum_attr_value, 2);
  ASSERT_EQ(enum_attr_values.size(), 3U);
  ASSERT_EQ(enum_attr_values[2], attr_value);
}

TEST_F(UtestEnumAttrUtils, TestGetEnumAttrValues) {
  // 验证三个不同属性
  vector<string> enum_attr_values = {};
  vector<string> attr_values;
  attr_values.emplace_back("value0");
  attr_values.emplace_back("value1");
  attr_values.emplace_back("value2");
  vector<int64_t> enum_values = {};
  EnumAttrUtils::GetEnumAttrValues(enum_attr_values, attr_values, enum_values);
  for (size_t i = 0U; i < attr_values.size(); i++) {
    ASSERT_EQ(enum_values[i], i);
    ASSERT_EQ(enum_attr_values[i], "value" + to_string(i));
  }

  // 验证包含两个相同属性
  vector<string> attr_values1;
  attr_values1.emplace_back("value2");
  attr_values1.emplace_back("value0");
  vector<int64_t> enum_values1 = {};
  EnumAttrUtils::GetEnumAttrValues(enum_attr_values, attr_values1, enum_values1);
  ASSERT_EQ(enum_values1[0], 2);
  ASSERT_EQ(enum_values1[1], 0);
  ASSERT_EQ(enum_attr_values.size(), 3U);

  // 验证包含一个相同属性， 一个不同属性
  vector<string> attr_values2;
  attr_values2.emplace_back("value3");
  attr_values2.emplace_back("value1");
  vector<int64_t> enum_values2 = {};
  EnumAttrUtils::GetEnumAttrValues(enum_attr_values, attr_values2, enum_values2);
  ASSERT_EQ(enum_values2[0], 3);
  ASSERT_EQ(enum_values2[1], 1);
  ASSERT_EQ(enum_attr_values.size(), 4U);
}

TEST_F(UtestEnumAttrUtils, TestGetAttrName) {
  // enum_attr_name为空校验
  vector<string> enum_attr_names = {};
  vector<bool> name_use_string_values = {};
  string enum_attr_name = "";
  string attr_name = "";
  bool is_value_string = false;
  auto ret = EnumAttrUtils::GetAttrName(enum_attr_names, name_use_string_values,
                                        enum_attr_name, attr_name, is_value_string);
  ASSERT_EQ(ret, GRAPH_FAILED);

  // enum_attr_names为空校验
  char_t a1 = 1;
  enum_attr_name.append(kAppendNum, prefix);
  enum_attr_name.append(kAppendNum, a1);
  ret = EnumAttrUtils::GetAttrName(enum_attr_names, name_use_string_values,
                                   enum_attr_name, attr_name, is_value_string);
  ASSERT_EQ(ret, GRAPH_FAILED);

  // name_use_string_values为空校验
  enum_attr_names.emplace_back("name1");
  ret = EnumAttrUtils::GetAttrName(enum_attr_names, name_use_string_values,
                                   enum_attr_name, attr_name, is_value_string);
  ASSERT_EQ(ret, GRAPH_FAILED);

  // 一个成员的正常的流程 enum化的属性名
  name_use_string_values.emplace_back(true);
  ret = EnumAttrUtils::GetAttrName(enum_attr_names, name_use_string_values,
                                   enum_attr_name, attr_name, is_value_string);
  ASSERT_EQ(ret, GRAPH_SUCCESS);
  ASSERT_EQ(attr_name, "name1");
  ASSERT_EQ(is_value_string, true);

  // 两个成员的正常的流程 enum化的属性名
  enum_attr_name = "";
  char_t a2 = 2;
  enum_attr_name.append(kAppendNum, prefix);
  enum_attr_name.append(kAppendNum, a2);
  enum_attr_names.emplace_back("name2");
  name_use_string_values.emplace_back(false);
  ret = EnumAttrUtils::GetAttrName(enum_attr_names, name_use_string_values,
                                   enum_attr_name, attr_name, is_value_string);
  ASSERT_EQ(ret, GRAPH_SUCCESS);
  ASSERT_EQ(attr_name, "name2");
  ASSERT_EQ(is_value_string, false);

  // 127个成员的正常的流程 enum化的属性名, 一位数最大值
  enum_attr_name = "";
  char_t a127 = 127;
  enum_attr_name.append(kAppendNum, prefix);
  enum_attr_name.append(kAppendNum, a127);
  for (int i = 3; i <= 127; i++) {
    enum_attr_names.emplace_back("name" + to_string(i));
    name_use_string_values.emplace_back(true);
  }
  ret = EnumAttrUtils::GetAttrName(enum_attr_names, name_use_string_values,
                                   enum_attr_name, attr_name, is_value_string);
  ASSERT_EQ(ret, GRAPH_SUCCESS);
  ASSERT_EQ(attr_name, "name127");
  ASSERT_EQ(is_value_string, true);

  // 128个成员的正常的流程 enum化的属性名，两位数的初始值
  enum_attr_name = "";
  char_t a128_1 = 1;
  char_t a128_2 = 2;
  enum_attr_name.append(kAppendNum, prefix);
  enum_attr_name.append(kAppendNum, a128_1);
  enum_attr_name.append(kAppendNum, a128_2);
  enum_attr_names.emplace_back("name128");
  name_use_string_values.emplace_back(true);
  ret = EnumAttrUtils::GetAttrName(enum_attr_names, name_use_string_values,
                                   enum_attr_name, attr_name, is_value_string);
  ASSERT_EQ(ret, GRAPH_SUCCESS);
  ASSERT_EQ(attr_name, "name128");
  ASSERT_EQ(is_value_string, true);

  // 正常的流程 非enum化的属性名
  enum_attr_name = "name1";
  ret = EnumAttrUtils::GetAttrName(enum_attr_names, name_use_string_values,
                                   enum_attr_name, attr_name, is_value_string);
  ASSERT_EQ(ret, GRAPH_SUCCESS);
  ASSERT_EQ(attr_name, enum_attr_name);
  ASSERT_EQ(is_value_string, false);
}

TEST_F(UtestEnumAttrUtils, TestGetAttrValue) {
  // 异常场景测试
  vector<string> enum_attr_values = {};
  int64_t enum_attr_value = 0;
  string attr_value = "";
  auto ret = EnumAttrUtils::GetAttrValue(enum_attr_values, enum_attr_value, attr_value);
  ASSERT_EQ(ret, GRAPH_FAILED);

  // 正常场景测试
  enum_attr_values.emplace_back("value1");
  ret = EnumAttrUtils::GetAttrValue(enum_attr_values, enum_attr_value, attr_value);
  ASSERT_EQ(ret, GRAPH_SUCCESS);
  ASSERT_EQ(attr_value, "value1");
}

TEST_F(UtestEnumAttrUtils, TestGetAttrValues) {
  // 异常场景测试
  vector<string> enum_attr_values = {};
  vector<int64_t> enum_values = {};
  vector<string> attr_values = {};
  enum_values.emplace_back(1);
  auto ret = EnumAttrUtils::GetAttrValues(enum_attr_values, enum_values, attr_values);
  ASSERT_EQ(ret, GRAPH_FAILED);

  // 正常场景测试
  enum_attr_values.emplace_back("value1");
  enum_attr_values.emplace_back("value2");
  ret = EnumAttrUtils::GetAttrValues(enum_attr_values, enum_values, attr_values);
  ASSERT_EQ(ret, GRAPH_SUCCESS);
  ASSERT_EQ(attr_values.size(), 1U);
  ASSERT_EQ(attr_values[0], "value2");
}
}
