/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <fstream>
#include "gtest/gtest.h"
#include "proto/dflow.pb.h"
#include "dflow/inc/data_flow/flow_graph/inner_pp.h"
#include "dflow/flow_graph/data_flow_attr_define.h"

using namespace testing;
namespace ge {
namespace {
class InnerPpStub : public dflow::InnerPp {
 public:
  InnerPpStub(const char_t *pp_name, const char_t *inner_type) : dflow::InnerPp(pp_name, inner_type) {}
  ~InnerPpStub() override = default;

 protected:
  void InnerSerialize(std::map<ge::AscendString, ge::AscendString> &serialize_map) const override {
    serialize_map["TestAttr"] = "TestAttrValue";
  }
};
}  // namespace

class InnerPpTest : public Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(InnerPpTest, Serialize_success) {
  InnerPpStub pp_stub("stub_name", "test_type");
  AscendString str;
  pp_stub.Serialize(str);
  dataflow::ProcessPoint process_point;
  ASSERT_TRUE(process_point.ParseFromArray(str.GetString(), str.GetLength()));
  ASSERT_EQ(process_point.name(), "stub_name");
  const auto &extend_attrs = process_point.pp_extend_attrs();
  const auto &find_ret = extend_attrs.find("TestAttr");
  ASSERT_FALSE(find_ret == extend_attrs.cend());
  EXPECT_EQ(find_ret->second, "TestAttrValue");
}

TEST_F(InnerPpTest, param_invalid) {
  InnerPpStub pp_stub("stub_name", nullptr);
  AscendString str;
  pp_stub.Serialize(str);
  ASSERT_EQ(str.GetLength(), 0);
}
}  // namespace ge
