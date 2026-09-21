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
#include "model/attr_value_impl.h"

namespace FlowFunc {
class AttrValueImplSTest : public testing::Test {
 protected:
  virtual void SetUp() {}

  virtual void TearDown() {}
};

TEST_F(AttrValueImplSTest, get_float_success) {
  ff::udf::AttrValue proto_attr;
  proto_attr.set_f(1.0f);
  AttrValueImpl impl(proto_attr);
  float float_val;
  auto ret = impl.GetVal(float_val);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  EXPECT_FLOAT_EQ(float_val, 1.0f);
}

TEST_F(AttrValueImplSTest, get_float_failed) {
  ff::udf::AttrValue proto_attr;
  proto_attr.set_i(1);
  AttrValueImpl impl(proto_attr);
  float float_val;
  auto ret = impl.GetVal(float_val);
  EXPECT_EQ(ret, FLOW_FUNC_ERR_ATTR_TYPE_MISMATCH);
}

TEST_F(AttrValueImplSTest, get_int_success) {
  ff::udf::AttrValue proto_attr;
  proto_attr.set_i(1024);
  AttrValueImpl impl(proto_attr);
  int64_t int_val;
  auto ret = impl.GetVal(int_val);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  EXPECT_EQ(int_val, 1024);
}

TEST_F(AttrValueImplSTest, get_int_failed) {
  ff::udf::AttrValue proto_attr;
  proto_attr.set_f(1.0f);
  AttrValueImpl impl(proto_attr);
  int64_t int_val;
  auto ret = impl.GetVal(int_val);
  EXPECT_EQ(ret, FLOW_FUNC_ERR_ATTR_TYPE_MISMATCH);
}

TEST_F(AttrValueImplSTest, get_bool_success) {
  ff::udf::AttrValue proto_attr;
  proto_attr.set_b(true);
  AttrValueImpl impl(proto_attr);
  bool bool_val;
  auto ret = impl.GetVal(bool_val);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  EXPECT_TRUE(bool_val);
}

TEST_F(AttrValueImplSTest, get_bool_failed) {
  ff::udf::AttrValue proto_attr;
  proto_attr.set_f(1.0f);
  AttrValueImpl impl(proto_attr);
  bool bool_val;
  auto ret = impl.GetVal(bool_val);
  EXPECT_EQ(ret, FLOW_FUNC_ERR_ATTR_TYPE_MISMATCH);
}

TEST_F(AttrValueImplSTest, get_string_success) {
  ff::udf::AttrValue proto_attr;
  AscendString val("ssssssssss");
  proto_attr.set_s(val.GetString());
  AttrValueImpl impl(proto_attr);
  AscendString str_val;
  auto ret = impl.GetVal(str_val);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  EXPECT_EQ(str_val, val);
}

TEST_F(AttrValueImplSTest, get_string_failed) {
  ff::udf::AttrValue proto_attr;
  proto_attr.set_f(1.0f);
  AttrValueImpl impl(proto_attr);
  AscendString str_val;
  auto ret = impl.GetVal(str_val);
  EXPECT_EQ(ret, FLOW_FUNC_ERR_ATTR_TYPE_MISMATCH);
}

TEST_F(AttrValueImplSTest, get_type_success) {
  ff::udf::AttrValue proto_attr;
  TensorDataType type = TensorDataType::DT_DOUBLE;
  proto_attr.set_type(static_cast<int32_t>(type));
  AttrValueImpl impl(proto_attr);
  TensorDataType type_val;
  auto ret = impl.GetVal(type_val);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  EXPECT_EQ(type_val, type);
}

TEST_F(AttrValueImplSTest, get_type_failed) {
  ff::udf::AttrValue proto_attr;
  proto_attr.set_f(1.0f);
  AttrValueImpl impl(proto_attr);
  TensorDataType type_val;
  auto ret = impl.GetVal(type_val);
  EXPECT_EQ(ret, FLOW_FUNC_ERR_ATTR_TYPE_MISMATCH);
}

TEST_F(AttrValueImplSTest, get_type_list_success) {
  ff::udf::AttrValue proto_attr;
  auto array = proto_attr.mutable_array();
  array->add_type(static_cast<int32_t>(TensorDataType::DT_DOUBLE));
  array->add_type(static_cast<int32_t>(TensorDataType::DT_INT64));
  AttrValueImpl impl(proto_attr);
  std::vector<TensorDataType> list_type_val;
  auto ret = impl.GetVal(list_type_val);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  std::vector<TensorDataType> expect_list = {TensorDataType::DT_DOUBLE, TensorDataType::DT_INT64};
  EXPECT_EQ(list_type_val, expect_list);
}

TEST_F(AttrValueImplSTest, get_type_list_failed) {
  ff::udf::AttrValue proto_attr;
  AttrValueImpl impl(proto_attr);
  std::vector<TensorDataType> list_type_val;
  auto ret = impl.GetVal(list_type_val);
  EXPECT_EQ(ret, FLOW_FUNC_ERR_ATTR_TYPE_MISMATCH);
}

TEST_F(AttrValueImplSTest, get_string_list_success) {
  std::vector<AscendString> expect_str = {"111", "222", "333"};
  ff::udf::AttrValue proto_attr;
  auto array = proto_attr.mutable_array();
  for (auto &s : expect_str) {
    array->add_s(s.GetString());
  }
  AttrValueImpl impl(proto_attr);
  std::vector<AscendString> list_str_val;
  auto ret = impl.GetVal(list_str_val);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  EXPECT_EQ(list_str_val, expect_str);
}

TEST_F(AttrValueImplSTest, get_string_list_failed) {
  ff::udf::AttrValue proto_attr;
  AttrValueImpl impl(proto_attr);
  std::vector<AscendString> list_str_val;
  auto ret = impl.GetVal(list_str_val);
  EXPECT_EQ(ret, FLOW_FUNC_ERR_ATTR_TYPE_MISMATCH);
}

TEST_F(AttrValueImplSTest, get_float_list_success) {
  std::vector<float> expect_float_list = {1.0f, 2.0f, 3.0f};
  ff::udf::AttrValue proto_attr;
  auto array = proto_attr.mutable_array();
  for (auto &f : expect_float_list) {
    array->add_f(f);
  }
  AttrValueImpl impl(proto_attr);
  std::vector<float> list_float_val;
  auto ret = impl.GetVal(list_float_val);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  EXPECT_EQ(list_float_val, expect_float_list);
}

TEST_F(AttrValueImplSTest, get_float_list_failed) {
  ff::udf::AttrValue proto_attr;
  AttrValueImpl impl(proto_attr);
  std::vector<float> list_float_val;
  auto ret = impl.GetVal(list_float_val);
  EXPECT_EQ(ret, FLOW_FUNC_ERR_ATTR_TYPE_MISMATCH);
}

TEST_F(AttrValueImplSTest, get_int_list_success) {
  std::vector<int64_t> expect_int_list = {1, 2, 3};
  ff::udf::AttrValue proto_attr;
  auto array = proto_attr.mutable_array();
  for (auto &i : expect_int_list) {
    array->add_i(i);
  }
  AttrValueImpl impl(proto_attr);
  std::vector<int64_t> list_int_val;
  auto ret = impl.GetVal(list_int_val);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  EXPECT_EQ(list_int_val, expect_int_list);
}

TEST_F(AttrValueImplSTest, get_int_list_failed) {
  ff::udf::AttrValue proto_attr;
  AttrValueImpl impl(proto_attr);
  std::vector<int64_t> list_int_val;
  auto ret = impl.GetVal(list_int_val);
  EXPECT_EQ(ret, FLOW_FUNC_ERR_ATTR_TYPE_MISMATCH);
}

TEST_F(AttrValueImplSTest, get_bool_list_success) {
  std::vector<bool> expect_bool_list = {true, false, true};
  ff::udf::AttrValue proto_attr;
  auto array = proto_attr.mutable_array();
  for (auto b : expect_bool_list) {
    array->add_b(b);
  }
  AttrValueImpl impl(proto_attr);
  std::vector<bool> list_bool_val;
  auto ret = impl.GetVal(list_bool_val);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  EXPECT_EQ(list_bool_val, expect_bool_list);
}

TEST_F(AttrValueImplSTest, get_bool_list_failed) {
  ff::udf::AttrValue proto_attr;
  AttrValueImpl impl(proto_attr);
  std::vector<bool> list_bool_val;
  auto ret = impl.GetVal(list_bool_val);
  EXPECT_EQ(ret, FLOW_FUNC_ERR_ATTR_TYPE_MISMATCH);
}

TEST_F(AttrValueImplSTest, get_list_list_int_success) {
  std::vector<std::vector<int64_t>> expect_list = {{1, 2, 3}, {4, 5, 6}};
  ff::udf::AttrValue proto_attr;
  auto list_list = proto_attr.mutable_list_list_i();
  for (auto &list_i : expect_list) {
    auto proto_list_i = list_list->add_list_i();
    for (auto i : list_i) {
      proto_list_i->add_i(i);
    }
  }
  AttrValueImpl impl(proto_attr);
  std::vector<std::vector<int64_t>> list_list_val;
  auto ret = impl.GetVal(list_list_val);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  EXPECT_EQ(list_list_val, expect_list);
}

TEST_F(AttrValueImplSTest, get_list_list_int_failed) {
  ff::udf::AttrValue proto_attr;
  AttrValueImpl impl(proto_attr);
  std::vector<std::vector<int64_t>> list_list_val;
  auto ret = impl.GetVal(list_list_val);
  EXPECT_EQ(ret, FLOW_FUNC_ERR_ATTR_TYPE_MISMATCH);
}
}  // namespace FlowFunc