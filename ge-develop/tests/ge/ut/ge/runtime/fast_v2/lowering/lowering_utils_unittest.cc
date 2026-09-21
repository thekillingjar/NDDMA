/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "lowering/lowering_utils.h"

#include <gtest/gtest.h>
namespace gert {
class LoweringUtilsUT : public testing::Test {};

TEST_F(LoweringUtilsUT, CreateAndCheckEngineTaskNode_Ok) {
  auto graph = std::make_shared<ge::ComputeGraph>("test_graph");
  auto op_desc = std::make_shared<ge::OpDesc>("data", "Data");
  int64_t id = 7;
  auto origin_node = graph->AddNode(op_desc, id);
  auto node = LoweringUtils::CreateEngineTaskNode(graph, op_desc, origin_node);
  ASSERT_NE(node, nullptr);
  ASSERT_NE(node->GetOpDesc(), nullptr);
  EXPECT_EQ(node->GetOpDesc()->GetId(), id);
  EXPECT_EQ(LoweringUtils::IsEngineTaskNode(node), true);
}

TEST_F(LoweringUtilsUT, CreateAndCheckEngineTaskNode_Ok_NoExtAttr) {
  auto graph = std::make_shared<ge::ComputeGraph>("test_graph");
  auto op_desc = std::make_shared<ge::OpDesc>("data", "Data");
  auto node = graph->AddNode(op_desc);
  ASSERT_NE(node, nullptr);
  EXPECT_EQ(LoweringUtils::IsEngineTaskNode(node), false);
}

TEST_F(LoweringUtilsUT, CreateAndCheckEngineTaskNode_Failed_NodeIsNull) {
  auto graph = std::make_shared<ge::ComputeGraph>("test_graph");
  ge::OpDescPtr op_desc = nullptr;
  int64_t id = 7;
  auto op_desc_7 = std::make_shared<ge::OpDesc>("data", "Data");
  auto origin_node = graph->AddNode(op_desc_7, id);
  auto node = LoweringUtils::CreateEngineTaskNode(graph, op_desc, origin_node);
  EXPECT_EQ(node, nullptr);
}
}  // namespace gert