/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/standard_optimize/constant_folding/potential_const_taken_effect_pass.h"

#include <string>
#include <vector>
#include <gtest/gtest.h>

#include "common/types.h"
#include "common/plugin/ge_make_unique_util.h"
#include "graph/passes/base_pass.h"
#include "graph_builder_utils.h"
#include "graph/utils/constant_utils.h"

namespace ge {
class UtestPotentialConstTakenEffectPass : public testing::Test {
 protected:
  UtestPotentialConstTakenEffectPass() = default;
};

namespace {
void SetWeightForConstNode(NodePtr &const_node) {
  // new a tensor
  ge::GeTensorPtr tensor = std::make_shared<GeTensor>();
  std::vector<uint8_t> value{1, 2, 3};
  std::vector<int64_t> shape{3};
  tensor->MutableTensorDesc().SetShape(GeShape(shape));
  tensor->SetData(value);
  tensor->MutableTensorDesc().SetDataType(DT_UINT8);
  ConstantUtils::SetWeight(const_node->GetOpDesc(), 0, tensor);
}

/**
 *     netoutput1
 *        |
 *      shapeNo1
 *       |
 *     addnYes1
 *    /    \.
 *  /       \.
 * const1   const2
 */
ComputeGraphPtr BuildGraph1() {
  auto builder = ut::GraphBuilder("test");
  auto const1 = builder.AddNode("const1", CONSTANT, 0, 1);
  auto const2 = builder.AddNode("const2", CONSTANT, 0, 1);
  SetWeightForConstNode(const1);
  SetWeightForConstNode(const2);
  auto addn1 = builder.AddNode("addn1", ADD, 2, 1);
  auto shape1 = builder.AddNode("shape1", SHAPE, 1, 1);
  auto netoutput1 = builder.AddNode("netoutput", NETOUTPUT, 1, 0);

  builder.AddDataEdge(const1, 0, addn1, 0);
  builder.AddDataEdge(const2, 0, addn1, 1);
  builder.AddDataEdge(addn1, 0, shape1, 0);
  builder.AddDataEdge(shape1, 0, netoutput1, 0);

  return builder.GetGraph();
}

/**
 *  shape1(pc)  shape2(abnormal potential_const)
 *          \   /
 *           add
 *            |
 *         netoutput
 * 
 */
ComputeGraphPtr BuildAbnormalPotentialConstGraph() {
  // new a tensor
  ge::GeTensorPtr tensor = std::make_shared<GeTensor>();
  std::vector<uint8_t> value{1, 2, 3};
  std::vector<int64_t> shape{3};
  tensor->MutableTensorDesc().SetShape(GeShape(shape));
  tensor->SetData(value);
  tensor->MutableTensorDesc().SetDataType(DT_UINT8);

  auto builder = ut::GraphBuilder("g5");
  auto shape1 = builder.AddNode("shape1", SHAPE, 1, 1, FORMAT_NCHW, DT_FLOAT, {1, 2, 3});
  ConstantUtils::MarkPotentialConst(shape1->GetOpDesc(), {0}, {tensor});
  auto shape2 = builder.AddNode("shape2", SHAPE, 1, 1, FORMAT_NCHW, DT_FLOAT, {1, 2, 3});
  AttrUtils::SetBool(shape2->GetOpDesc(), ATTR_NAME_POTENTIAL_CONST, true);
  auto add = builder.AddNode("add", ADD, 2, 1);
  auto netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 0);

  builder.AddDataEdge(shape1, 0, add, 0);
  builder.AddDataEdge(shape2, 0, add, 1);
  builder.AddDataEdge(add, 0,netoutput, 0);
  return builder.GetGraph();
}
}// namespace

TEST_F(UtestPotentialConstTakenEffectPass, TestNeedIgnorePass) {
  auto graph = BuildGraph1();
  vector<NodePtr> repass_node;
  PotentialConstTakenEffectPass convert_pass;
  EXPECT_EQ(convert_pass.OnFinishGraph(graph, repass_node), SUCCESS);
  EXPECT_EQ(graph->GetAllNodesSize(), 5);
}

TEST_F(UtestPotentialConstTakenEffectPass, TestAbnormalPotentialConst) {
  auto graph = BuildAbnormalPotentialConstGraph();
  PotentialConstTakenEffectPass convert_pass;

  vector<NodePtr> repass_node;
  EXPECT_EQ(convert_pass.OnFinishGraph(graph, repass_node), SUCCESS);
  EXPECT_EQ(graph->GetAllNodesSize(), 4);
  EXPECT_EQ(repass_node.size(), 3);
  // shape1 fold
  auto add_node = graph->FindNode("add");
  EXPECT_NE(add_node, nullptr);
  auto first_input_node_of_add = add_node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();
  EXPECT_EQ(first_input_node_of_add->GetType(), CONSTANT);

  // shape2 unmark
  auto shape2_node = graph->FindNode("shape2");
  bool has_attr = AttrUtils::HasAttr(shape2_node->GetOpDesc(), ATTR_NAME_POTENTIAL_CONST);
  EXPECT_FALSE(has_attr);
}

TEST_F(UtestPotentialConstTakenEffectPass, TestPotentialConstWhichOwnerGraphIsNull) {
  auto graph = BuildAbnormalPotentialConstGraph();
  PotentialConstTakenEffectPass convert_pass;
  auto shape1 = graph->FindNode("shape1");
  shape1->ClearOwnerGraph(nullptr);

  vector<NodePtr> repass_node;
  EXPECT_EQ(convert_pass.OnFinishGraph(graph, repass_node), SUCCESS);
  EXPECT_EQ(graph->GetAllNodesSize(), 4);
  EXPECT_EQ(repass_node.size(), 0);
  // shape1 not fold because its owner graph is null
  auto add_node = graph->FindNode("add");
  EXPECT_NE(add_node, nullptr);
  auto first_input_node_of_add = add_node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();
  EXPECT_EQ(first_input_node_of_add->GetType(), SHAPE);
}
}  // namespace ge
