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
#include <iostream>
#include "parser/common/op_parser_factory.h"
#include "parser/tensorflow/tensorflow_auto_mapping_parser_adapter.h"
#include "framework/omg/parser/parser_factory.h"
#include "graph/operator_reg.h"
#include "graph/types.h"
#include "register/op_registry.h"
#include "parser/common/op_registration_tbe.h"


namespace ge {
class UtestTensorflowAutoMappingParserAdapter : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}

};


TEST_F(UtestTensorflowAutoMappingParserAdapter, success) {
  auto parser = TensorFlowAutoMappingParserAdapter();

  domi::tensorflow::NodeDef arg_node;
  arg_node.set_name("size");
  arg_node.set_op("Size");
  auto attr = arg_node.mutable_attr();
  domi::tensorflow::AttrValue value;
  value.set_type(domi::tensorflow::DataType::DT_HALF);
  (*attr)["out_type"] = value;

  auto op_desc = ge::parser::MakeShared<ge::OpDesc>("size", "Size");
  auto ret = parser.ParseParams(reinterpret_cast<Message *>(&arg_node), op_desc);
  EXPECT_EQ(ret, ge::SUCCESS);


  auto ret2 = ge::AttrUtils::SetBool(op_desc, "test_fail", true);
  EXPECT_EQ(ret2, true);
  EXPECT_EQ(ge::AttrUtils::HasAttr(op_desc, "test_fail"), true);

  ret = parser.ParseParams(reinterpret_cast<Message *>(&arg_node), op_desc);
  EXPECT_EQ(ret, ge::FAILED);
}


} // namespace ge
