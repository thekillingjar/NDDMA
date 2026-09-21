/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_TEST_UTIL_H
#define AUTOFUSE_TEST_UTIL_H

template <typename T>
void AttrEq(T &holder, const std::string attr_name, const int64_t &expect) {
  int64_t value = -1;
  ge::AttrUtils::GetInt(holder, attr_name, value);
  EXPECT_EQ(value, expect);
}

template <typename T>
void AttrEq(T &holder, const std::string attr_name, const ge::DataType &expect) {
  ge::DataType value = ge::DT_UNDEFINED;
  ge::AttrUtils::GetDataType(holder, attr_name, value);
  EXPECT_EQ(value, expect);
}

template <typename T>
void AttrEq(T &holder, const std::string attr_name, const vector<int64_t> &expect) {
  vector<int64_t> value;
  ge::AttrUtils::GetListInt(holder, attr_name, value);
  EXPECT_EQ(value, expect);
}

template <typename T>
void AttrEq(T &holder, const std::string attr_name, const vector<vector<int64_t>> &expect) {
  vector<vector<int64_t>> value;
  ge::AttrUtils::GetListListInt(holder, attr_name, value);
  EXPECT_EQ(value, expect);
}

#endif  // AUTOFUSE_TEST_UTIL_H
