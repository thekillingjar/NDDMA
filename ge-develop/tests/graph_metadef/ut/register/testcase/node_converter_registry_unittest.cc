/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/node_converter_registry.h"
#include <gtest/gtest.h>

class NodeConverterRegistryUnittest : public testing::Test {};

namespace TestNodeConverterRegistry {
gert::LowerResult TestFunc(const ge::NodePtr &node, const gert::LowerInput &lower_input) {
  return {};
}
gert::LowerResult TestFunc2(const ge::NodePtr &node, const gert::LowerInput &lower_input) {
  return {};
}

TEST_F(NodeConverterRegistryUnittest, RegisterSuccess_DefaultPlacement) {
  EXPECT_EQ(gert::NodeConverterRegistry::GetInstance().FindNodeConverter("RegisterSuccess1"), nullptr);
  REGISTER_NODE_CONVERTER("RegisterSuccess1", TestFunc);
  EXPECT_EQ(gert::NodeConverterRegistry::GetInstance().FindNodeConverter("RegisterSuccess1"), TestFunc);
  auto reg_data1 = gert::NodeConverterRegistry::GetInstance().FindRegisterData("RegisterSuccess1");
  ASSERT_NE(reg_data1, nullptr);
  EXPECT_EQ(reg_data1->converter, TestFunc);
  EXPECT_EQ(reg_data1->require_placement, -1);
}

TEST_F(NodeConverterRegistryUnittest, RegisterSuccess_WithPlacement) {
  EXPECT_EQ(gert::NodeConverterRegistry::GetInstance().FindNodeConverter("RegisterSuccess2"), nullptr);
  REGISTER_NODE_CONVERTER_PLACEMENT("RegisterSuccess2", 10, TestFunc2);
  EXPECT_EQ(gert::NodeConverterRegistry::GetInstance().FindNodeConverter("RegisterSuccess2"), TestFunc2);
  auto reg_data1 = gert::NodeConverterRegistry::GetInstance().FindRegisterData("RegisterSuccess2");
  ASSERT_NE(reg_data1, nullptr);
  EXPECT_EQ(reg_data1->converter, TestFunc2);
  EXPECT_EQ(reg_data1->require_placement, 10);
}
}
