/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/convert/message2operator.h"

#include <fstream>
#include <gtest/gtest.h>

#include "proto/onnx/ge_onnx.pb.h"
#include "parser/common/convert/pb2json.h"

namespace ge {
class UtestMessage2Operator : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(UtestMessage2Operator, message_to_operator_success) {
  ge::onnx::NodeProto input_node;
  ge::onnx::AttributeProto *attribute = input_node.add_attribute();
  attribute->set_name("attribute");
  attribute->set_type(onnx::AttributeProto::AttributeType(1));
  attribute->set_f(1.0);
  ge::onnx::TensorProto *attribute_tensor = attribute->mutable_t();
  attribute_tensor->set_data_type(1);
  attribute_tensor->add_dims(4);
  ge::Operator op_src("add", "Add");
  auto ret = Message2Operator::ParseOperatorAttrs(attribute, 1, op_src);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestMessage2Operator, message_to_operator_fail) {
  ge::onnx::NodeProto input_node;
  ge::onnx::AttributeProto *attribute = input_node.add_attribute();
  ge::onnx::TensorProto *attribute_tensor = attribute->mutable_t();
  attribute_tensor->add_double_data(1.00);

  ge::Operator op_src("add", "Add");
  auto ret = Message2Operator::ParseOperatorAttrs(attribute, 6, op_src);
  EXPECT_EQ(ret, FAILED);

  ret = Message2Operator::ParseOperatorAttrs(attribute, 1, op_src);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(UtestMessage2Operator, pb2json_one_field_json) {
  ge::onnx::NodeProto input_node;
  ge::onnx::AttributeProto *attribute = input_node.add_attribute();
  attribute->set_name("attribute");
  attribute->set_type(onnx::AttributeProto::AttributeType(1));
  ge::onnx::TensorProto *attribute_tensor = attribute->mutable_t();
  attribute_tensor->set_data_type(1);
  attribute_tensor->add_dims(4);
  attribute_tensor->set_raw_data("\007");
  Json json;
  ge::Pb2Json::Message2Json(input_node, std::set<std::string>{}, json, true);
  EXPECT_NE(json.size(), 0U);
}

TEST_F(UtestMessage2Operator, pb2json_one_field_json_depth_max) {
  ge::onnx::NodeProto input_node;
  ge::onnx::AttributeProto *attribute = input_node.add_attribute();
  attribute->set_name("attribute");
  attribute->set_type(onnx::AttributeProto::AttributeType(1));
  ge::onnx::TensorProto *attribute_tensor = attribute->mutable_t();
  attribute_tensor->set_data_type(1);
  attribute_tensor->add_dims(4);
  attribute_tensor->set_raw_data("\007");
  Json json;
  ge::Pb2Json::Message2Json(input_node, std::set<std::string>{}, json, true, 21);
  EXPECT_EQ(json.size(), 0U);
}

TEST_F(UtestMessage2Operator, pb2json_one_field_json_type) {
  ge::onnx::NodeProto input_node;
  ge::onnx::AttributeProto *attribute = input_node.add_attribute();
  attribute->set_name("attribute");
  attribute->set_type(onnx::AttributeProto::AttributeType(1));
  ge::onnx::TensorProto *attribute_tensor = attribute->mutable_t();
  attribute_tensor->set_data_type(3);
  attribute_tensor->add_dims(4);
  attribute_tensor->set_raw_data("\007");
  Json json;
  ge::Pb2Json::Message2Json(input_node, std::set<std::string>{}, json, true);
  EXPECT_NE(json.size(), 0U);
}

TEST_F(UtestMessage2Operator, enum_to_json_success) {
  nlohmann::json json = { {"attr1", "attr1"}};
  ge::Pb2Json::EnumJson2Json(json);
  EXPECT_NE(json.size(), 0U);
}
}  // namespace ge
