/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <string>
#include <gtest/gtest.h>

#include "macro_utils/dt_public_scope.h"
#include "common/ge_inner_error_codes.h"
#include "graph/passes/control_flow_and_stream/switch_dead_branch_elimination.h"
#include "graph_builder_utils.h"
#include "macro_utils/dt_public_unscope.h"

namespace ge {
class UtestSwitchDeadBranchElimination : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

namespace {
/*
 *   data1  const1
 *     \     /
 *      case1
 *        |
 *      relu1
 *        |
 *    netoutput
 */
ut::GraphBuilder ParentGraphBuilder() {
  ut::GraphBuilder builder = ut::GraphBuilder("g1");
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  auto const1 = builder.AddNode("const1", "Const", 0, 1);
  auto case1 = builder.AddNode("case1", CASE, 2, 1);
  auto relu1 = builder.AddNode("relu1", "Relu", 1, 1);
  auto netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 0);

  int32_t weight[1] = {1};
  GeTensorDesc weight_desc(GeShape({1}), FORMAT_NHWC, DT_INT32);
  GeTensorPtr tensor = std::make_shared<GeTensor>(weight_desc, (uint8_t *)weight, sizeof(weight));
  OpDescUtils::SetWeights(const1, {tensor});

  builder.AddDataEdge(data1, 0, case1, 0);
  builder.AddDataEdge(const1, 0, case1, 1);
  builder.AddDataEdge(case1, 0, relu1, 0);
  builder.AddDataEdge(relu1, 0, netoutput, 0);
  return builder;
}

ut::GraphBuilder ParentGraphBuilder2() {
  ut::GraphBuilder builder = ut::GraphBuilder("g1");
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  auto const1 = builder.AddNode("const1", "Const", 0, 1);
  auto case1 = builder.AddNode("case1", CASE, 2, 1);
  auto relu1 = builder.AddNode("relu1", "Relu", 1, 1);
  auto netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 0);

  int32_t weight[1] = {1};
  GeTensorDesc weight_desc(GeShape({1}), FORMAT_NHWC, DT_FLOAT16);
  GeTensorPtr tensor = std::make_shared<GeTensor>(weight_desc, (uint8_t *)weight, sizeof(weight));
  OpDescUtils::SetWeights(const1, {tensor});

  builder.AddDataEdge(data1, 0, case1, 0);
  builder.AddDataEdge(const1, 0, case1, 1);
  builder.AddDataEdge(case1, 0, relu1, 0);
  builder.AddDataEdge(relu1, 0, netoutput, 0);
  return builder;
}

ut::GraphBuilder ParentGraphBuilder3() {
  ut::GraphBuilder builder = ut::GraphBuilder("g1");
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  auto const1 = builder.AddNode("const1", "Const", 0, 1);
  auto case1 = builder.AddNode("case1", CASE, 2, 1);
  auto relu1 = builder.AddNode("relu1", "Relu", 1, 1);
  auto netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 0);

  int32_t weight[1] = {1};
  GeTensorDesc weight_desc(GeShape({1}), FORMAT_NHWC, DT_DOUBLE);
  GeTensorPtr tensor = std::make_shared<GeTensor>(weight_desc, (uint8_t *)weight, sizeof(weight));
  OpDescUtils::SetWeights(const1, {tensor});

  builder.AddDataEdge(data1, 0, case1, 0);
  builder.AddDataEdge(const1, 0, case1, 1);
  builder.AddDataEdge(case1, 0, relu1, 0);
  builder.AddDataEdge(relu1, 0, netoutput, 0);
  return builder;
}

ut::GraphBuilder ParentGraphBuilder4() {
  ut::GraphBuilder builder = ut::GraphBuilder("g1");
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  auto const1 = builder.AddNode("const1", "Const", 0, 1);
  auto case1 = builder.AddNode("case1", CASE, 2, 1);
  auto relu1 = builder.AddNode("relu1", "Relu", 1, 1);
  auto netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 0);

  int32_t weight[1] = {1};
  GeTensorDesc weight_desc(GeShape({1}), FORMAT_NHWC, DT_INT8);
  GeTensorPtr tensor = std::make_shared<GeTensor>(weight_desc, (uint8_t *)weight, sizeof(weight));
  OpDescUtils::SetWeights(const1, {tensor});

  builder.AddDataEdge(data1, 0, case1, 0);
  builder.AddDataEdge(const1, 0, case1, 1);
  builder.AddDataEdge(case1, 0, relu1, 0);
  builder.AddDataEdge(relu1, 0, netoutput, 0);
  return builder;
}

ut::GraphBuilder ParentGraphBuilder5() {
  ut::GraphBuilder builder = ut::GraphBuilder("g1");
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  auto const1 = builder.AddNode("const1", "Const", 0, 1);
  auto case1 = builder.AddNode("case1", CASE, 2, 1);
  auto relu1 = builder.AddNode("relu1", "Relu", 1, 1);
  auto netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 0);

  int32_t weight[1] = {1};
  GeTensorDesc weight_desc(GeShape({1}), FORMAT_NHWC, DT_INT64);
  GeTensorPtr tensor = std::make_shared<GeTensor>(weight_desc, (uint8_t *)weight, sizeof(weight));
  OpDescUtils::SetWeights(const1, {tensor});

  builder.AddDataEdge(data1, 0, case1, 0);
  builder.AddDataEdge(const1, 0, case1, 1);
  builder.AddDataEdge(case1, 0, relu1, 0);
  builder.AddDataEdge(relu1, 0, netoutput, 0);
  return builder;
}

ut::GraphBuilder ParentGraphBuilder6() {
  ut::GraphBuilder builder = ut::GraphBuilder("g1");
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  auto const1 = builder.AddNode("const1", "Const", 0, 1);
  auto case1 = builder.AddNode("case1", CASE, 2, 1);
  auto relu1 = builder.AddNode("relu1", "Relu", 1, 1);
  auto netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 0);

  int32_t weight[1] = {1};
  GeTensorDesc weight_desc(GeShape({1}), FORMAT_NHWC, DT_FLOAT);
  GeTensorPtr tensor = std::make_shared<GeTensor>(weight_desc, (uint8_t *)weight, sizeof(weight));
  OpDescUtils::SetWeights(const1, {tensor});

  builder.AddDataEdge(data1, 0, case1, 0);
  builder.AddDataEdge(const1, 0, case1, 1);
  builder.AddDataEdge(case1, 0, relu1, 0);
  builder.AddDataEdge(relu1, 0, netoutput, 0);
  return builder;
}

/*   
 *   data1   data2
 *     \      / 
 *      switch
 *     /      \
 *   relu1   relu2
 *     \      /
 *       merge
 *         |
 *     netoutput
 */
ut::GraphBuilder SwitchSubgraphBuilder(string graph_name, uint32_t num) {
  ut::GraphBuilder builder = ut::GraphBuilder(graph_name);

  string data1_name = "data1_" + std::to_string(num);
  auto data1 = builder.AddNode(data1_name, "Data", 0, 1);
  auto data1_desc = data1->GetOpDesc();
  EXPECT_NE(data1_desc, nullptr);
  AttrUtils::SetInt(data1_desc, "_parent_node_index", 0);

  string data2_name = "data2_" + std::to_string(num);
  auto data2 = builder.AddNode(data2_name, "Data", 0, 1);
  auto data2_desc = data2->GetOpDesc();
  EXPECT_NE(data2_desc, nullptr);
  AttrUtils::SetInt(data2_desc, "_parent_node_index", 1);

  string switch_name = "switch_" + std::to_string(num);
  auto switch1 = builder.AddNode(switch_name, "Switch", 2, 2);

  string relu1_name = "relu1_" + std::to_string(num);
  auto relu1 = builder.AddNode(relu1_name, "Relu", 1, 1);

  string relu2_name = "relu2_" + std::to_string(num);
  auto relu2 = builder.AddNode(relu2_name, "Relu", 1, 1);

  string merge_name = "merge_" + std::to_string(num);
  auto merge = builder.AddNode(merge_name, "Merge", 2, 1);

  string output_name = "output_" + std::to_string(num);
  auto netoutput = builder.AddNode(output_name, NETOUTPUT, 1, 0);

  builder.AddDataEdge(data1, 0, switch1, 0);
  builder.AddDataEdge(data2, 0, switch1, 1);
  builder.AddDataEdge(switch1, 0, relu1, 0);
  builder.AddDataEdge(switch1, 1, relu2, 0);
  builder.AddDataEdge(relu1, 0, merge, 0);
  builder.AddDataEdge(relu2, 0, merge, 1);
  builder.AddDataEdge(merge, 0, netoutput, 0);

  return builder;
}

void AddCaseSubgraph(ComputeGraphPtr &parent_graph, uint32_t branch_num) {
  auto case_node = parent_graph->FindNode("case1");
  EXPECT_NE(case_node, nullptr);

  for (uint32_t i = 0; i < branch_num; ++i) {
  	string name = "Branch_Graph_" + std::to_string(i);

  	auto builder_subgraph = SwitchSubgraphBuilder(name, i);
  	auto switch_subgraph = builder_subgraph.GetGraph();

  	case_node->GetOpDesc()->AddSubgraphName(switch_subgraph->GetName());
  	case_node->GetOpDesc()->SetSubgraphInstanceName(i, switch_subgraph->GetName());

  	switch_subgraph->SetParentNode(case_node);
  	switch_subgraph->SetParentGraph(parent_graph);
  	EXPECT_EQ(parent_graph->AddSubgraph(switch_subgraph->GetName(), switch_subgraph), GRAPH_SUCCESS);
  }
}
}  // namespace


TEST_F(UtestSwitchDeadBranchElimination, switch_dead_branch_elimination_across_case_success) {
  auto builder = ParentGraphBuilder();
  auto parent_graph = builder.GetGraph();

  AddCaseSubgraph(parent_graph, 2);
  auto subgraphs = parent_graph->GetAllSubgraphs();
  EXPECT_EQ(subgraphs.size(), 2);

  SwitchDeadBranchElimination switch_pass;
  for (auto &subgraph : subgraphs) {
    auto switch_node = subgraph->FindFirstNodeMatchType("Switch");
    if (switch_node != nullptr) {
      EXPECT_EQ(switch_pass.Run(switch_node), SUCCESS);
    }
  }
  
  auto all_nodes = parent_graph->GetAllNodes();
  EXPECT_EQ(all_nodes.size(), 15);

  for (auto &subgraph : subgraphs) {
    EXPECT_EQ(subgraph->GetDirectNode().size(), 5);
    EXPECT_EQ(subgraph->FindFirstNodeMatchType("Switch"), nullptr);
    auto merge_node = subgraph->FindFirstNodeMatchType("Merge");
    EXPECT_NE(merge_node, nullptr);
    auto merge_innode = merge_node->GetInDataNodes();
    EXPECT_EQ(merge_innode.size(), 1);
  }
}

TEST_F(UtestSwitchDeadBranchElimination, DeleteSwitchNode_Test)
{
  SwitchDeadBranchElimination switch_pass;
  NodePtr node = nullptr;
  NodePtr pred_node = nullptr;
  OutDataAnchorPtr active_out_data_anchor = nullptr;

  Status ret = switch_pass.DeleteSwitchNode(node, pred_node, active_out_data_anchor);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(UtestSwitchDeadBranchElimination, Run_Test)
{
  SwitchDeadBranchElimination switch_pass;
  NodePtr node = nullptr;

  Status ret = switch_pass.Run(node);
  EXPECT_EQ(ret, PARAM_INVALID);
}

TEST_F(UtestSwitchDeadBranchElimination, ParsePred_DT_FLOAT) {
  auto builder = ParentGraphBuilder2();
  auto parent_graph = builder.GetGraph();

  AddCaseSubgraph(parent_graph, 2);
  auto subgraphs = parent_graph->GetAllSubgraphs();
  EXPECT_EQ(subgraphs.size(), 2);

  SwitchDeadBranchElimination switch_pass;
  for (auto &subgraph : subgraphs) {
    auto switch_node = subgraph->FindFirstNodeMatchType("Switch");
    if (switch_node != nullptr) {
      EXPECT_EQ(switch_pass.Run(switch_node), SUCCESS);
    }
  }
  
  auto all_nodes = parent_graph->GetAllNodes();
  EXPECT_EQ(all_nodes.size(), 15);

  for (auto &subgraph : subgraphs) {
    EXPECT_EQ(subgraph->GetDirectNode().size(), 5);
    EXPECT_EQ(subgraph->FindFirstNodeMatchType("Switch"), nullptr);
    auto merge_node = subgraph->FindFirstNodeMatchType("Merge");
    EXPECT_NE(merge_node, nullptr);
    auto merge_innode = merge_node->GetInDataNodes();
    EXPECT_EQ(merge_innode.size(), 1);
  }
}

TEST_F(UtestSwitchDeadBranchElimination, ParsePred_DT_FLOAT2) {
  auto builder = ParentGraphBuilder3();
  auto parent_graph = builder.GetGraph();

  AddCaseSubgraph(parent_graph, 2);
  auto subgraphs = parent_graph->GetAllSubgraphs();
  EXPECT_EQ(subgraphs.size(), 2);

  SwitchDeadBranchElimination switch_pass;
  for (auto &subgraph : subgraphs) {
    auto switch_node = subgraph->FindFirstNodeMatchType("Switch");
    if (switch_node != nullptr) {
      EXPECT_EQ(switch_pass.Run(switch_node), SUCCESS);
    }
  }
  
  auto all_nodes = parent_graph->GetAllNodes();
  EXPECT_EQ(all_nodes.size(), 15);

  for (auto &subgraph : subgraphs) {
    EXPECT_EQ(subgraph->GetDirectNode().size(), 5);
    EXPECT_EQ(subgraph->FindFirstNodeMatchType("Switch"), nullptr);
    auto merge_node = subgraph->FindFirstNodeMatchType("Merge");
    EXPECT_NE(merge_node, nullptr);
    auto merge_innode = merge_node->GetInDataNodes();
    EXPECT_EQ(merge_innode.size(), 1);
  }
}

TEST_F(UtestSwitchDeadBranchElimination, ParsePred_DT_INT8) {
  auto builder = ParentGraphBuilder4();
  auto parent_graph = builder.GetGraph();

  AddCaseSubgraph(parent_graph, 2);
  auto subgraphs = parent_graph->GetAllSubgraphs();
  EXPECT_EQ(subgraphs.size(), 2);

  SwitchDeadBranchElimination switch_pass;
  for (auto &subgraph : subgraphs) {
    auto switch_node = subgraph->FindFirstNodeMatchType("Switch");
    if (switch_node != nullptr) {
      EXPECT_EQ(switch_pass.Run(switch_node), SUCCESS);
    }
  }
  
  auto all_nodes = parent_graph->GetAllNodes();
  EXPECT_EQ(all_nodes.size(), 15);

  for (auto &subgraph : subgraphs) {
    EXPECT_EQ(subgraph->GetDirectNode().size(), 5);
    EXPECT_EQ(subgraph->FindFirstNodeMatchType("Switch"), nullptr);
    auto merge_node = subgraph->FindFirstNodeMatchType("Merge");
    EXPECT_NE(merge_node, nullptr);
    auto merge_innode = merge_node->GetInDataNodes();
    EXPECT_EQ(merge_innode.size(), 1);
  }
}

TEST_F(UtestSwitchDeadBranchElimination, ParsePred_DT_INT64) {
  auto builder = ParentGraphBuilder5();
  auto parent_graph = builder.GetGraph();

  AddCaseSubgraph(parent_graph, 2);
  auto subgraphs = parent_graph->GetAllSubgraphs();
  EXPECT_EQ(subgraphs.size(), 2);

  SwitchDeadBranchElimination switch_pass;
  for (auto &subgraph : subgraphs) {
    auto switch_node = subgraph->FindFirstNodeMatchType("Switch");
    if (switch_node != nullptr) {
      EXPECT_EQ(switch_pass.Run(switch_node), SUCCESS);
    }
  }
  
  auto all_nodes = parent_graph->GetAllNodes();
  EXPECT_EQ(all_nodes.size(), 15);

  for (auto &subgraph : subgraphs) {
    EXPECT_EQ(subgraph->GetDirectNode().size(), 5);
    EXPECT_EQ(subgraph->FindFirstNodeMatchType("Switch"), nullptr);
    auto merge_node = subgraph->FindFirstNodeMatchType("Merge");
    EXPECT_NE(merge_node, nullptr);
    auto merge_innode = merge_node->GetInDataNodes();
    EXPECT_EQ(merge_innode.size(), 1);
  }
}

TEST_F(UtestSwitchDeadBranchElimination, ParsePred_DT_FLOAT3) {
  auto builder = ParentGraphBuilder6();
  auto parent_graph = builder.GetGraph();

  AddCaseSubgraph(parent_graph, 2);
  auto subgraphs = parent_graph->GetAllSubgraphs();
  EXPECT_EQ(subgraphs.size(), 2);

  SwitchDeadBranchElimination switch_pass;
  for (auto &subgraph : subgraphs) {
    auto switch_node = subgraph->FindFirstNodeMatchType("Switch");
    if (switch_node != nullptr) {
      EXPECT_EQ(switch_pass.Run(switch_node), SUCCESS);
    }
  }
  
  auto all_nodes = parent_graph->GetAllNodes();
  EXPECT_EQ(all_nodes.size(), 15);

  for (auto &subgraph : subgraphs) {
    EXPECT_EQ(subgraph->GetDirectNode().size(), 5);
    EXPECT_EQ(subgraph->FindFirstNodeMatchType("Switch"), nullptr);
    auto merge_node = subgraph->FindFirstNodeMatchType("Merge");
    EXPECT_NE(merge_node, nullptr);
    auto merge_innode = merge_node->GetInDataNodes();
    EXPECT_EQ(merge_innode.size(), 1);
  }
}

}  // namespace ge
