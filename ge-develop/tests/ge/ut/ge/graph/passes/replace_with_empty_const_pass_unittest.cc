/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/standard_optimize/constant_folding/replace_with_empty_const_pass.h"

#include <gtest/gtest.h>
#include <string>

#include "graph_builder_utils.h"
#include "graph/utils/constant_utils.h"

namespace ge {
class UtestReplaceWithEmptyConstPass : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

namespace {
/**
 * data1    const1
 *      \ /
 *     add1
 *      |
 *    cast1(empty)
 *      |
 *    conv2d
 */
ut::GraphBuilder Graph1Builder() {
  ut::GraphBuilder builder = ut::GraphBuilder("g1");
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  auto const1 = builder.AddNode("const1", "Const", 0, 1);
  auto add1 = builder.AddNode("add1", "Add", 2, 1);
  auto cast1 = builder.AddNode("cast1", "Cast", 1, 1);
  auto conv2d = builder.AddNode("conv2d", "Conv2D", 1, 0);

  add1->GetOpDesc()->AddInputDesc(GeTensorDesc(GeShape({1,1,8,8}),FORMAT_NCHW));
  add1->GetOpDesc()->AddInputDesc(GeTensorDesc(GeShape({1,1,8,8}),FORMAT_NCHW));
  add1->GetOpDesc()->AddOutputDesc(GeTensorDesc(GeShape({1,1,8,8}),FORMAT_NCHW));
  GeTensorDesc empty_tensor(GeShape({1,0,8,8}),FORMAT_NCHW);
  cast1->GetOpDesc()->UpdateOutputDesc(0,empty_tensor);

  builder.AddDataEdge(data1, 0, add1, 0);
  builder.AddDataEdge(const1, 0, add1, 1);
  builder.AddDataEdge(add1, 0, cast1, 0);
  builder.AddDataEdge(cast1, 0, conv2d, 0);
  return builder;
}

/**
 *           data1    const1
 *                \ /
 *               add1
 *                |
 *   data2  -> switch1 (empty)
 *                |
 *              conv2d
 */
ut::GraphBuilder Graph2Builder() {
  ut::GraphBuilder builder = ut::GraphBuilder("graph2");
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  auto data2 = builder.AddNode("data2", "Data", 0, 1);
  auto const1 = builder.AddNode("const1", "Const", 0, 1);
  auto add1 = builder.AddNode("add1", "Add", 2, 1);
  auto switch1 = builder.AddNode("switch1", "Switch", 2, 1);
  auto conv2d = builder.AddNode("conv2d", "Conv2D", 1, 0);

  add1->GetOpDesc()->AddInputDesc(GeTensorDesc(GeShape({1, 1, 8, 8}),FORMAT_NCHW));
  add1->GetOpDesc()->AddInputDesc(GeTensorDesc(GeShape({1, 1, 8, 8}),FORMAT_NCHW));
  add1->GetOpDesc()->AddOutputDesc(GeTensorDesc(GeShape({1, 1, 8, 8}),FORMAT_NCHW));
  GeTensorDesc empty_tensor(GeShape({1, 0, 8, 8}),FORMAT_NCHW);
  switch1->GetOpDesc()->UpdateOutputDesc(0, empty_tensor);

  builder.AddDataEdge(data1, 0, add1, 0);
  builder.AddDataEdge(const1, 0, add1, 1);
  builder.AddDataEdge(add1, 0, switch1, 0);
  builder.AddDataEdge(data2, 0, switch1, 1);
  builder.AddDataEdge(switch1, 0, conv2d, 0);
  return builder;
}

/**
 *             switch1 (empty)
 *                |
 *              identity(empty)
 *                |.
 *               const
 */
ut::GraphBuilder Graph3Builder() {
  ut::GraphBuilder builder = ut::GraphBuilder("graph2");
  auto switch1 = builder.AddNode("switch1", "Switch", 2, 1);
  auto identity = builder.AddNode("identity", "Identity", 1, 1);
  (void)AttrUtils::SetBool(identity->GetOpDesc(), ATTR_NAME_IS_INSERTED_BY_GE, true);
  auto const_node = builder.AddNode("const", "Const", 0, 1);

  GeTensorDesc empty_tensor(GeShape({1, 0, 8, 8}),FORMAT_NCHW);
  switch1->GetOpDesc()->UpdateOutputDesc(0, empty_tensor);
  identity->GetOpDesc()->UpdateOutputDesc(0, empty_tensor);
  (void)AttrUtils::SetBool(identity->GetOpDesc(), ATTR_NO_NEED_CONSTANT_FOLDING, true);

  builder.AddDataEdge(switch1, 0, identity, 0);
  builder.AddControlEdge(identity, const_node);
  return builder;
}

/**
 * data1    const1
 *      \ /
 *     add1
 *      |
 *    cast1(potential empty)
 *      |
 *    conv2d
 */
ut::GraphBuilder Graph4Builder() {
  ut::GraphBuilder builder = ut::GraphBuilder("g1");
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  auto const1 = builder.AddNode("const1", "Const", 0, 1);
  auto add1 = builder.AddNode("add1", "Add", 2, 1);
  auto cast1 = builder.AddNode("cast1", "Cast", 1, 1);
  auto conv2d = builder.AddNode("conv2d", "Conv2D", 1, 0);

  add1->GetOpDesc()->AddInputDesc(GeTensorDesc(GeShape({1,1,8,8}),FORMAT_NCHW));
  add1->GetOpDesc()->AddInputDesc(GeTensorDesc(GeShape({1,1,8,8}),FORMAT_NCHW));
  add1->GetOpDesc()->AddOutputDesc(GeTensorDesc(GeShape({1,1,8,8}),FORMAT_NCHW));
  cast1->GetOpDesc()->UpdateOutputDesc(0,GeTensorDesc(GeShape({1,1,8,8}),FORMAT_NCHW));
  GeTensorDesc empty_tensor_desc(GeShape({1,0,8,8}),FORMAT_NCHW);
  GeTensorPtr empty_tensor = std::make_shared<GeTensor>(empty_tensor_desc);
  ConstantUtils::MarkPotentialConst(cast1->GetOpDesc(), {0}, {empty_tensor});
  AttrUtils::SetStr(cast1->GetOpDesc(), "_source_pass_of_potential_const", "ReplaceWithEmptyConstPass");

  builder.AddDataEdge(data1, 0, add1, 0);
  builder.AddDataEdge(const1, 0, add1, 1);
  builder.AddDataEdge(add1, 0, cast1, 0);
  builder.AddDataEdge(cast1, 0, conv2d, 0);
  return builder;
}

/**
 *              switch 
 *                |
 *              identity(empty)
 *                |.
 *               cast
 */
ut::GraphBuilder Graph5Builder() {
  ut::GraphBuilder builder = ut::GraphBuilder("graph2");
  auto switch_node = builder.AddNode("switch", "Switch", 2, 1);
  auto identity = builder.AddNode("identity", "Identity", 1, 1);
  auto cast_node = builder.AddNode("cast", "Cast", 1, 1);

  GeTensorDesc empty_tensor(GeShape({1, 0, 8, 8}),FORMAT_NCHW);
  switch_node->GetOpDesc()->UpdateOutputDesc(0, empty_tensor);
  identity->GetOpDesc()->UpdateOutputDesc(0, empty_tensor);

  builder.AddDataEdge(switch_node, 0, identity, 0);
  builder.AddControlEdge(identity, cast_node);
  return builder;
}
}  // namespace

TEST_F(UtestReplaceWithEmptyConstPass, replace_whith_empty_const_success) {
  auto builder = Graph1Builder();
  auto graph = builder.GetGraph();
  graph->SetSessionID(0);
  ReplaceWithEmptyConstPass replace_with_empty_const_pass;

  EXPECT_EQ(graph->GetDirectNodesSize(),5);
  // run pass on add1, graph still has 5 nodes
  auto add1 = graph->FindNode("add1");
  Status ret = replace_with_empty_const_pass.Run(add1);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(),5);

  // run pass on const1, graph still has 5 nodes
  auto const1 = graph->FindNode("const1");
  ret = replace_with_empty_const_pass.Run(const1);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(),5);

  auto cast1 = graph->FindNode("cast1");
  ret = replace_with_empty_const_pass.Run(cast1);
  EXPECT_EQ(cast1->GetOutAllNodes().size(),0);
  auto conv2d = graph->FindNode("conv2d");
  EXPECT_EQ(conv2d->GetInDataNodes().at(0)->GetType(),"Const");
}

TEST_F(UtestReplaceWithEmptyConstPass, replace_whith_empty_switch_skip) {
  auto builder = Graph2Builder();
  auto graph = builder.GetGraph();
  graph->SetSessionID(0);
  ReplaceWithEmptyConstPass replace_with_empty_const_pass;

  EXPECT_EQ(graph->GetDirectNodesSize(), 6);
  // run pass on switch1, graph still has 6 nodes
  auto switch1 = graph->FindNode("switch1");
  EXPECT_NE(switch1, nullptr);
  Status ret = replace_with_empty_const_pass.Run(switch1);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(), 6);
}

TEST_F(UtestReplaceWithEmptyConstPass, skip_node_ctrl_identity) {
  auto builder = Graph3Builder();
  auto graph = builder.GetGraph();
  graph->SetSessionID(0);
  ReplaceWithEmptyConstPass replace_with_empty_const_pass;

  EXPECT_EQ(graph->GetDirectNodesSize(), 3);
  // run pass on identity, graph still has 3 nodes
  auto identity = graph->FindNode("identity");
  EXPECT_NE(identity, nullptr);
  Status ret = replace_with_empty_const_pass.Run(identity);
  EXPECT_EQ(ret, SUCCESS);
  auto switch_node = graph->FindNode("switch1");
  EXPECT_EQ(graph->GetDirectNodesSize(), 3);
  EXPECT_EQ(switch_node->GetOutNodes().at(0)->GetName(), "identity");
}

TEST_F(UtestReplaceWithEmptyConstPass, replace_node_without_data_output) {
  auto builder = Graph5Builder();
  auto graph = builder.GetGraph();
  graph->SetSessionID(0);
  ReplaceWithEmptyConstPass replace_with_empty_const_pass;

  EXPECT_EQ(graph->GetDirectNodesSize(), 3);
  // run pass on identity, graph will has 3 nodes
  auto identity = graph->FindNode("identity");
  EXPECT_NE(identity, nullptr);
  Status ret = replace_with_empty_const_pass.Run(identity);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(), 3);
  auto switch_node = graph->FindNode("switch");
  EXPECT_EQ(switch_node->GetOutNodes().at(0)->GetName(), "identity_ctrl_identity_0");
}

TEST_F(UtestReplaceWithEmptyConstPass, replace_whith_empty_mark_potential_const) {
  auto builder = Graph1Builder();
  auto graph = builder.GetGraph();
  graph->SetSessionID(0);
  ReplaceWithEmptyConstPass replace_with_empty_const_pass(false);

  EXPECT_EQ(graph->GetDirectNodesSize(), 5);
  // run pass on cast1, graph still has 5 nodes, and cast1 is potential const
  auto cast1 = graph->FindNode("cast1");
  EXPECT_NE(cast1, nullptr);
  Status ret = replace_with_empty_const_pass.Run(cast1);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(), 5);
  EXPECT_TRUE(ConstantUtils::IsPotentialConst(cast1->GetOpDesc()));
}
TEST_F(UtestReplaceWithEmptyConstPass, replace_whith_empty_unmark_potential_const) {
  auto builder = Graph4Builder();
  auto graph = builder.GetGraph();
  graph->SetSessionID(0);
  ReplaceWithEmptyConstPass replace_with_empty_const_pass(false);

  EXPECT_EQ(graph->GetDirectNodesSize(), 5);
  // run pass on cast1, graph still has 5 nodes, and cast1 is potential const
  auto cast1 = graph->FindNode("cast1");
  EXPECT_NE(cast1, nullptr);
  Status ret = replace_with_empty_const_pass.Run(cast1);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(), 5);
  EXPECT_FALSE(ConstantUtils::IsPotentialConst(cast1->GetOpDesc()));
}

TEST_F(UtestReplaceWithEmptyConstPass, Don_Not_Constat_Folding) {
  auto builder = Graph1Builder();
  auto graph = builder.GetGraph();
  ReplaceWithEmptyConstPass replace_with_empty_const_pass;

  auto cast1 = graph->FindNode("cast1");
  (void)AttrUtils::SetBool(cast1->GetOpDesc(), ATTR_NAME_DO_NOT_CONSTANT_FOLDING, true);
  EXPECT_EQ(replace_with_empty_const_pass.Run(cast1), SUCCESS);
  EXPECT_NE(graph->FindNode("cast1"), nullptr);
}
}  // namespace ge
