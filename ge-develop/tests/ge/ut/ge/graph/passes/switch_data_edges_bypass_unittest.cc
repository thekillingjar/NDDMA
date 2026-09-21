/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// To test the SwitchDataEdgesBypass

#include <gtest/gtest.h>
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/utils/node_utils.h"
#include "graph/passes/control_flow_and_stream/switch_data_edges_bypass.h"
#include "graph_builder_utils.h"

namespace ge {
class UtestSwitchDataEdgesBypass : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

namespace {
/*
 * netoutput1
 *    |
 *  merge1
 *    |
 *   addn1
 *    |
 * switch1
 *   |   \
 * data1 const1
 */
ComputeGraphPtr BuildGraph1() {
  ut::GraphBuilder builder("g1");
  auto data1 = builder.AddNode("data1", "Data", 1, 1);
  auto const1 = builder.AddNode("const1", "Const", 0, 1);
  auto switch1 = builder.AddNode("switch1", "Switch", 2, 2);
  auto addn1 = builder.AddNode("addn1", "AddN", 1, 1);
  auto merge1 = builder.AddNode("merge1", "Merge", 1, 2);
  auto netoutput1 = builder.AddNode("netoutput1", "NetOutput", 1, 1);

  builder.AddDataEdge(data1, 0, switch1, 0);
  builder.AddDataEdge(const1, 0, switch1, 1);
  builder.AddDataEdge(switch1, 0, addn1, 0);
  builder.AddDataEdge(addn1, 0, merge1, 0);
  builder.AddDataEdge(merge1, 0, netoutput1, 0);

  return builder.GetGraph();
}

/*
 *      netoutput1
 *         |
 *       merge1
 *       /   \
 *   addn1   addn2
 *      \     /
 *      switch1
 *      /   \
 *   data1 const1
 */
ComputeGraphPtr BuildGraph2() {
  ut::GraphBuilder builder("g1");
  auto data1 = builder.AddNode("data1", "Data", 1, 1);
  auto const1 = builder.AddNode("const1", "Const", 0, 1);
  auto switch1 = builder.AddNode("switch1", "Switch", 2, 2);
  auto addn1 = builder.AddNode("addn1", "AddN", 1, 1);
  auto addn2 = builder.AddNode("addn2", "AddN", 1, 1);
  auto merge1 = builder.AddNode("merge1", "Merge", 1, 2);
  auto netoutput1 = builder.AddNode("netoutput1", "NetOutput", 1, 1);

  builder.AddDataEdge(data1, 0, switch1, 0);
  builder.AddDataEdge(const1, 0, switch1, 1);
  builder.AddDataEdge(switch1, 0, addn1, 0);
  builder.AddDataEdge(switch1, 1, addn2, 0);
  builder.AddDataEdge(addn1, 0, merge1, 0);
  builder.AddDataEdge(addn2, 0, merge1, 1);
  builder.AddDataEdge(merge1, 0, netoutput1, 0);

  return builder.GetGraph();
}

/*
 * netoutput1
 *    |
 *  merge1
 *    |
 *   addn1
 *    |
 * switch1
 *   |
 * data1
 */
ComputeGraphPtr BuildGraph3() {
  ut::GraphBuilder builder("g3");
  auto data1 = builder.AddNode("data1", "Data", 1, 1);
  auto switch1 = builder.AddNode("switch1", "Switch", 2, 2);
  auto addn1 = builder.AddNode("addn1", "AddN", 1, 1);
  auto merge1 = builder.AddNode("merge1", "Merge", 1, 2);
  auto netoutput1 = builder.AddNode("netoutput1", "NetOutput", 1, 1);

  builder.AddDataEdge(data1, 0, switch1, 0);
  builder.AddDataEdge(switch1, 0, addn1, 0);
  builder.AddDataEdge(addn1, 0, merge1, 0);
  builder.AddDataEdge(merge1, 0, netoutput1, 0);

  return builder.GetGraph();
}

/*
 *      netoutput1
 *         |
 *       merge1
 *         |
 *       addn3
 *      /    \
 *   addn1    addn2
 *       \T   /T
 *       switch1
 *         |   \
 *       data1 const1
 */
ComputeGraphPtr BuildGraph5() {
  ut::GraphBuilder builder("g1");
  auto data1 = builder.AddNode("data1", "Data", 1, 1);
  auto const1 = builder.AddNode("const1", "Const", 0, 1);
  auto switch1 = builder.AddNode("switch1", "Switch", 2, 2);
  auto addn1 = builder.AddNode("addn1", "AddN", 1, 1);
  auto addn2 = builder.AddNode("addn2", "AddN", 1, 1);
  auto addn3 = builder.AddNode("addn3", "AddN", 2, 1);
  auto merge1 = builder.AddNode("merge1", "Merge", 1, 2);
  auto netoutput1 = builder.AddNode("netoutput1", "NetOutput", 1, 1);

  builder.AddDataEdge(data1, 0, switch1, 0);
  builder.AddDataEdge(const1, 0, switch1, 1);
  builder.AddDataEdge(switch1, 1, addn1, 0);
  builder.AddDataEdge(switch1, 1, addn2, 0);
  builder.AddDataEdge(addn1, 0, addn3, 0);
  builder.AddDataEdge(addn2, 0, addn3, 1);
  builder.AddDataEdge(addn3, 0, merge1, 0);
  builder.AddDataEdge(merge1, 0, netoutput1, 0);

  return builder.GetGraph();
}


/*
 *      netoutput1
 *         |
 *       merge1
 *         |
 *        addn3
 *        |T |T
 *       switch1
 *         |   \
 *       data1 const1
 */
ComputeGraphPtr BuildGraph6() {
  ut::GraphBuilder builder("g1");
  auto data1 = builder.AddNode("data1", "Data", 1, 1);
  auto const1 = builder.AddNode("const1", "Const", 0, 1);
  auto switch1 = builder.AddNode("switch1", "Switch", 2, 1);
  auto addn3 = builder.AddNode("addn3", "AddN", 2, 1);
  auto merge1 = builder.AddNode("merge1", "Merge", 1, 2);
  auto netoutput1 = builder.AddNode("netoutput1", "NetOutput", 1, 1);

  builder.AddDataEdge(data1, 0, switch1, 0);
  builder.AddDataEdge(const1, 0, switch1, 1);
  builder.AddDataEdge(switch1, 0, addn3, 0);
  builder.AddDataEdge(switch1, 0, addn3, 1);
  builder.AddDataEdge(addn3, 0, merge1, 0);
  builder.AddDataEdge(merge1, 0, netoutput1, 0);

  return builder.GetGraph();
}

/*
 *       netoutput1
 *           \F
 *           switch1 ------> addn1 --> nextiter1
 *          /     \                     /
 *   loopcond1  merge1 <---------------
 *             /
 *        enter1
 *         |
 *       data1
 */
ComputeGraphPtr BuildGraph7() {
  ut::GraphBuilder builder("g1");
  auto data1 = builder.AddNode("data1", "Data", 1, 1);
  auto enter1 = builder.AddNode("enter1", "Enter", 1, 1);
  auto loopcond1 = builder.AddNode("loopcond1", "LoopCond", 1, 1);
  auto merge1 = builder.AddNode("merge1", "Merge", 2, 2);
  auto switch1 = builder.AddNode("switch1", "Switch", 2, 2);
  auto addn1 = builder.AddNode("addn1", "AddN", 1, 1);
  auto nextiter1 = builder.AddNode("nextiter1", "NextIteration", 1, 1);
  auto netoutput1 = builder.AddNode("netoutput1", "NetOutput", 1, 1);

  builder.AddDataEdge(data1, 0, switch1, 0);
  builder.AddDataEdge(enter1, 0, merge1, 0);
  builder.AddDataEdge(loopcond1, 0, switch1, 1);
  builder.AddDataEdge(merge1, 0, switch1, 0);
  builder.AddDataEdge(switch1, 0, netoutput1, 0);
  builder.AddDataEdge(switch1, 1, addn1, 0);
  builder.AddDataEdge(addn1, 0, nextiter1, 0);
  builder.AddDataEdge(nextiter1, 0, merge1, 1);

  return builder.GetGraph();
}

/*
 *                       netoutput1
 *                           |
 * netoutput1             memcpy
 *    |                      |       c
 *  memcpy                memcpy1 <---  identity1
 *    |           ==>       |            /
 * switch1                  |    switch1
 *   |   \                   \    /   \
 * data1 const1               data1 const1
 */
ComputeGraphPtr BuildGraph8() {
  ut::GraphBuilder builder("g8");
  auto data1 = builder.AddNode("data1", "Data", 1, 1);
  auto const1 = builder.AddNode("const1", "Const", 0, 1);
  auto switch1 = builder.AddNode("switch1", "Switch", 2, 2);
  auto memcpy = builder.AddNode("memcpy", "MemcpyAsync", 1, 1);
  auto netoutput1 = builder.AddNode("netoutput1", "NetOutput", 1, 1);

  builder.AddDataEdge(data1, 0, switch1, 0);
  builder.AddDataEdge(const1, 0, switch1, 1);
  builder.AddDataEdge(switch1, 0, memcpy, 0);
  builder.AddDataEdge(memcpy, 0, netoutput1, 0);

  return builder.GetGraph();
}
/*
 * netoutput1
 *    |
 *  merge1
 *    |
 *   addn1
 *    |
 * switch1
 *   |   \
 *      const1
 */
ComputeGraphPtr BuildGraph9() {
  ut::GraphBuilder builder("g1");
  auto const1 = builder.AddNode("const1", "Const", 0, 1);
  auto switch1 = builder.AddNode("switch1", "Switch", 2, 2);
  auto addn1 = builder.AddNode("addn1", "AddN", 1, 1);
  auto merge1 = builder.AddNode("merge1", "Merge", 1, 2);
  auto netoutput1 = builder.AddNode("netoutput1", "NetOutput", 1, 1);

  builder.AddDataEdge(const1, 0, switch1, 1);
  builder.AddDataEdge(switch1, 0, addn1, 0);
  builder.AddDataEdge(addn1, 0, merge1, 0);
  builder.AddDataEdge(merge1, 0, netoutput1, 0);

  return builder.GetGraph();
}

std::set<std::string> GetNames(const Node::Vistor<NodePtr> &node_list) {
  std::set<std::string> name_set;
  for (const auto &node : node_list) {
    name_set.insert(node->GetName());
  }
  return  name_set;
}

std::set<std::string> GetNames(const ComputeGraph::Vistor<NodePtr> &node_list) {
  std::set<std::string> name_set;
  for (const auto &node : node_list) {
    name_set.insert(node->GetName());
  }
  return  name_set;
}
}// namespace

TEST_F(UtestSwitchDataEdgesBypass, OneOut) {
  auto graph = BuildGraph1();
  SwitchDataEdgesBypass pass;
  EXPECT_EQ(pass.Run(graph), SUCCESS);
  auto switch1 = graph->FindNode("switch1");
  EXPECT_EQ(switch1->GetOutDataAnchor(1)->GetPeerInDataNodesSize(), 0);
  EXPECT_EQ(switch1->GetOutDataAnchor(0)->GetPeerInDataNodesSize(), 1);
  auto identity_name = switch1->GetOutDataNodes().at(0)->GetName();
  EXPECT_EQ(identity_name.find("SwitchDataEdgesByPass_Identity_"), 0);
  auto identity0 = graph->FindNode(identity_name);
  EXPECT_EQ(identity0->GetOutDataNodesSize(), 0);
  EXPECT_EQ(GetNames(identity0->GetOutControlNodes()),
            std::set<std::string>({"addn1"}));
  auto data1 = graph->FindNode("data1");
  EXPECT_EQ(GetNames(data1->GetOutDataNodes()), std::set<std::string>({"addn1", "switch1"}));
}

TEST_F(UtestSwitchDataEdgesBypass, TwoOut) {
  auto graph = BuildGraph2();
  SwitchDataEdgesBypass pass;
  EXPECT_EQ(pass.Run(graph), SUCCESS);
  auto switch1 = graph->FindNode("switch1");
  EXPECT_EQ(switch1->GetOutDataAnchor(1)->GetPeerInDataNodesSize(), 1);
  EXPECT_EQ(switch1->GetOutDataAnchor(0)->GetPeerInDataNodesSize(), 1);

  auto identity0_name = switch1->GetOutDataNodes().at(0)->GetName();
  EXPECT_EQ(identity0_name.find("SwitchDataEdgesByPass_Identity_"), 0);
  auto identity0 = graph->FindNode(identity0_name);
  EXPECT_EQ(identity0->GetOutDataNodesSize(), 0);
  EXPECT_EQ(GetNames(identity0->GetOutControlNodes()),std::set<std::string>({"addn1"}));

  auto identity1_name = switch1->GetOutDataNodes().at(1)->GetName();
  EXPECT_EQ(identity1_name.find("SwitchDataEdgesByPass_Identity_"), 0);
  auto identity1 = graph->FindNode(identity1_name);
  EXPECT_EQ(identity1->GetOutDataNodesSize(), 0);
  EXPECT_EQ(GetNames(identity1->GetOutControlNodes()),std::set<std::string>({"addn2"}));

  EXPECT_NE(identity0_name, identity1_name);
  auto data1 = graph->FindNode("data1");
  EXPECT_EQ(GetNames(data1->GetOutDataNodes()), std::set<std::string>({"addn1", "addn2", "switch1"}));
}

TEST_F(UtestSwitchDataEdgesBypass, SwitchWithoutPred) {
  auto graph = BuildGraph3();
  SwitchDataEdgesBypass pass;
  EXPECT_EQ(pass.Run(graph), SUCCESS);
  EXPECT_EQ(GetNames(graph->GetDirectNode()),
            std::set<std::string>({"data1", "switch1", "addn1", "merge1", "netoutput1"}));
}

TEST_F(UtestSwitchDataEdgesBypass, SwitchWithoutPredAnchor) {
  auto graph = BuildGraph3();
  SwitchDataEdgesBypass pass;
  EXPECT_EQ(pass.Run(graph), SUCCESS);
  EXPECT_EQ(GetNames(graph->GetDirectNode()),
            std::set<std::string>({"data1", "switch1", "addn1", "merge1", "netoutput1"}));
}


TEST_F(UtestSwitchDataEdgesBypass, SwitchWithoutNoDataInput) {
  auto graph = BuildGraph9();
  SwitchDataEdgesBypass pass;
  EXPECT_EQ(pass.Run(graph), SUCCESS);
  EXPECT_EQ(GetNames(graph->GetDirectNode()),
            std::set<std::string>({"const1", "switch1", "addn1", "merge1", "netoutput1"}));
}

TEST_F(UtestSwitchDataEdgesBypass, MultiPeerNodeInOneBranch) {
  auto graph = BuildGraph5();
  SwitchDataEdgesBypass pass;
  EXPECT_EQ(pass.Run(graph), SUCCESS);
  auto switch1 = graph->FindNode("switch1");
  EXPECT_EQ(switch1->GetOutDataAnchor(0)->GetPeerInDataNodesSize(), 0);
  EXPECT_EQ(switch1->GetOutDataAnchor(1)->GetPeerInDataNodesSize(), 1);

  auto identity_name = switch1->GetOutDataNodes().at(0)->GetName();
  EXPECT_EQ(identity_name.find("SwitchDataEdgesByPass_Identity_"), 0);
  auto identity = graph->FindNode(identity_name);

  EXPECT_EQ(identity->GetOutDataNodesSize(), 0);
  EXPECT_EQ(GetNames(identity->GetOutControlNodes()),
            std::set<std::string>({"addn1", "addn2"}));
  auto data1 = graph->FindNode("data1");
  EXPECT_EQ(GetNames(data1->GetOutDataNodes()), std::set<std::string>({"addn1", "addn2", "switch1"}));
}

TEST_F(UtestSwitchDataEdgesBypass, MultiEdgesInOneBranch) {
  auto graph = BuildGraph6();
  SwitchDataEdgesBypass pass;
  EXPECT_EQ(pass.Run(graph), SUCCESS);
  auto switch1 = graph->FindNode("switch1");
  EXPECT_EQ(switch1->GetOutDataAnchor(0)->GetPeerInDataNodesSize(), 1);

  auto identity_name = switch1->GetOutDataNodes().at(0)->GetName();
  EXPECT_EQ(identity_name.find("SwitchDataEdgesByPass_Identity_"), 0);
  auto identity = graph->FindNode(identity_name);

  EXPECT_EQ(identity->GetOutDataNodesSize(), 0);
  EXPECT_EQ(identity->GetOutControlNodes().size(), 1);
  EXPECT_EQ(GetNames(identity->GetOutControlNodes()),
            std::set<std::string>({"addn3"}));
  auto data1 = graph->FindNode("data1");
  EXPECT_EQ(data1->GetOutDataNodesSize(), 3);
  EXPECT_EQ(GetNames(data1->GetOutDataNodes()), std::set<std::string>({"addn3", "switch1"}));
}

TEST_F(UtestSwitchDataEdgesBypass, IgnoreSwitchInWhile) {
  auto graph = BuildGraph7();
  SwitchDataEdgesBypass pass;
  EXPECT_EQ(pass.Run(graph), SUCCESS);
  EXPECT_EQ(GetNames(graph->GetDirectNode()),
            std::set<std::string>({"data1", "enter1", "merge1", "loopcond1",
                                   "switch1", "addn1", "nextiter1", "netoutput1"}));
}

TEST_F(UtestSwitchDataEdgesBypass, SwitchConnectToMerge) {
  auto graph = BuildGraph8();
  SwitchDataEdgesBypass pass;
  EXPECT_EQ(pass.Run(graph), SUCCESS);
  auto switch1 = graph->FindNode("switch1");
  EXPECT_EQ(switch1->GetOutDataAnchor(1)->GetPeerInDataNodesSize(), 0);
  EXPECT_EQ(switch1->GetOutDataAnchor(0)->GetPeerInDataNodesSize(), 1);

  auto identity1_name = switch1->GetOutDataNodes().at(0)->GetName();
  EXPECT_EQ(identity1_name.find("SwitchDataEdgesByPass_Identity_"), 0);
  auto identity1 = graph->FindNode(identity1_name);
  EXPECT_EQ(identity1->GetOutDataNodesSize(), 0);
  EXPECT_EQ(identity1->GetOutControlNodes().size(), 1);
  auto memcpy1 = identity1->GetOutControlNodes().at(0);
  EXPECT_EQ(memcpy1->GetType(), "MemcpyAsync");
  EXPECT_EQ(memcpy1->GetInControlNodes().size(), 1);
  EXPECT_EQ(memcpy1->GetInDataNodes().size(), 1);
  EXPECT_EQ(GetNames(memcpy1->GetInDataNodes()), std::set<std::string>({"data1"}));
  EXPECT_EQ(memcpy1->GetOutDataNodes().size(), 1);
  EXPECT_EQ(GetNames(memcpy1->GetOutDataNodes()), std::set<std::string>({"memcpy"}));
  EXPECT_EQ(memcpy1->GetOutControlNodes().size(), 0);
  auto merge1 = graph->FindNode("memcpy");
  EXPECT_EQ(merge1->GetInNodes().size(), 1);
  EXPECT_EQ(merge1->GetInDataNodes().size(), 1);
  EXPECT_EQ(merge1->GetInDataNodes().at(0)->GetName(), memcpy1->GetName());
}
}  // namespace ge
