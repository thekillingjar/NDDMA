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

#include "jit_execution/utils/partitioner/binary_graph_builder.h"
#include "graph/passes/graph_builder_utils.h"
#include "graph/debug/ge_op_types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_dump_utils.h"

using namespace testing;
using namespace ge;

class BinaryGraphBuilderUT : public testing::Test {
 protected:
  void SetUp() override {
    setenv("DUMP_GE_GRAPH", "2", 1);
    setenv("DUMP_GRAPH_LEVEL", "1", 1);
  }
  void TearDown() override {
    unsetenv("DUMP_GE_GRAPH");
    unsetenv("DUMP_GRAPH_LEVEL");
  }
  BinaryGraphBuilder graph_builder_;
};

bool CheckNodesIsExistInGraph(const ComputeGraphPtr &graph, const std::vector<NodePtr> &nodes) {
  for (const auto &node : nodes) {
    auto it = graph->FindNode(node->GetName());
    if (it == nullptr) {
      return false;
    }
  }
  return true;
}

bool CheckNodesIsSameLinkInGraph(const ComputeGraphPtr &graph, const std::vector<NodePtr> &nodes) {
  for (const auto &node : nodes) {
    auto it = graph->FindNode(node->GetName());
    if (it == nullptr) {
      return false;
    }
    if (!(*it == *node)) {
      std::cout << it->GetName() << ", " << node->GetName() << std::endl;
      return false;
    }
  }
  return true;
}

/*
 *      data1  data2
 *         \    /
 *          add1     var   data3
 *            \     /   \   /
 *              add2     mul
 *                \      /
 *                netoutput
 */
void BuildGraphForTestInOrOutNodeGenerate(ComputeGraphPtr &graph, std::vector<NodePtr> &sliced_nodes, std::vector<NodePtr> &remaining_nodes) {
  auto builder = ut::GraphBuilder("Origin_Graph");
  const auto &data1 = builder.AddNode("data1", DATA, 1, 1);
  const auto &data2 = builder.AddNode("data2", DATA, 1, 1);
  const auto &add1 = builder.AddNode("add1", "Add", 2, 1);
  const auto &var = builder.AddNode("var", VARIABLE, 0, 1);
  const auto &data3 = builder.AddNode("data3", DATA, 1, 1);
  const auto &add2 = builder.AddNode("add2", "Add", 2, 1);
  const auto &mul = builder.AddNode("mul", "Mul", 2, 1);
  const auto &netoutput = builder.AddNode("netoutput", NETOUTPUT, 2, 0);

  builder.AddDataEdge(data1, 0, add1, 0);
  builder.AddDataEdge(data2, 0, add1, 1);
  builder.AddDataEdge(add1, 0, add2, 0);
  builder.AddDataEdge(var, 0, add2, 1);
  builder.AddDataEdge(var, 0, mul, 0);
  builder.AddDataEdge(data3, 0, mul, 1);
  builder.AddDataEdge(add2, 0, netoutput, 0);
  builder.AddDataEdge(mul, 0, netoutput, 1);

  graph = builder.GetGraph();

  sliced_nodes.push_back(data1);
  sliced_nodes.push_back(data2);
  sliced_nodes.push_back(var);
  sliced_nodes.push_back(data3);

  remaining_nodes.push_back(add1);
  remaining_nodes.push_back(add2);
  remaining_nodes.push_back(mul);
  remaining_nodes.push_back(netoutput);
}

TEST_F(BinaryGraphBuilderUT, return_nullptr_when_input_null_nodes) {
  std::vector<NodePtr> infered_nodes;
  auto graph = graph_builder_.BuildGraph(infered_nodes, "Sliced_Graph");
  EXPECT_EQ(graph, nullptr);
}

TEST_F(BinaryGraphBuilderUT, return_graph_with_same_node_link_when_give_all_nodes_in_origin_graph) {
  ComputeGraphPtr graph;
  std::vector<NodePtr> sliced_nodes;
  std::vector<NodePtr> remaining_nodes;
  BuildGraphForTestInOrOutNodeGenerate(graph, sliced_nodes, remaining_nodes);
  std::vector<NodePtr> all_nodes;
  all_nodes.insert(all_nodes.end(), sliced_nodes.begin(), sliced_nodes.end());
  all_nodes.insert(all_nodes.end(), remaining_nodes.begin(), remaining_nodes.end());
  auto origin_graph = graph_builder_.BuildGraph(all_nodes, "Origin_Graph");
  EXPECT_TRUE(CheckNodesIsExistInGraph(origin_graph, all_nodes));

  EXPECT_EQ(origin_graph->GetAllNodesSize(), all_nodes.size());
  EXPECT_TRUE(CheckNodesIsSameLinkInGraph(origin_graph, all_nodes));
}

TEST_F(BinaryGraphBuilderUT, return_graph_with_add_input_data_node_and_link_when_give_nodes_without_input_node) {
  ComputeGraphPtr graph;
  std::vector<NodePtr> sliced_nodes;
  std::vector<NodePtr> remaining_nodes;
  BuildGraphForTestInOrOutNodeGenerate(graph, sliced_nodes, remaining_nodes);
  auto remaining_graph = graph_builder_.BuildGraph(remaining_nodes, "Remaining_Graph");
  EXPECT_TRUE(CheckNodesIsExistInGraph(remaining_graph, remaining_nodes));

  EXPECT_EQ(remaining_graph->GetAllNodesSize(), remaining_nodes.size() + 5);
  EXPECT_EQ(remaining_graph->GetInputNodes().size(), 5);
}

TEST_F(BinaryGraphBuilderUT, return_graph_with_add_netout_node_and_link_when_give_nodes_without_output_node) {
  ComputeGraphPtr graph;
  std::vector<NodePtr> sliced_nodes;
  std::vector<NodePtr> remaining_nodes;
  BuildGraphForTestInOrOutNodeGenerate(graph, sliced_nodes, remaining_nodes);
  auto sliced_graph = graph_builder_.BuildGraph(sliced_nodes, "Sliced_Graph");
  EXPECT_TRUE(CheckNodesIsExistInGraph(sliced_graph, sliced_nodes));

  EXPECT_EQ(sliced_graph->GetAllNodesSize(), sliced_nodes.size() + 1);
  EXPECT_NE(sliced_graph->GetOrUpdateNetOutputNode(), nullptr);
  EXPECT_EQ(sliced_graph->GetOrUpdateNetOutputNode()->GetAllInDataAnchors().size(), 4);
}

TEST_F(BinaryGraphBuilderUT, return_two_graph_with_io_link_when_give_two_nodes_from_one_graph) {
  ComputeGraphPtr graph;
  std::vector<NodePtr> sliced_nodes;
  std::vector<NodePtr> remaining_nodes;
  BuildGraphForTestInOrOutNodeGenerate(graph, sliced_nodes, remaining_nodes);

  auto sliced_graph = graph_builder_.BuildGraph(sliced_nodes, "Sliced_Graph");
  auto remaining_graph = graph_builder_.BuildGraph(remaining_nodes, "Remaining_Graph");

  BinaryGraphIOLinkage io_link;
  io_link.sliced_graph = sliced_graph;
  io_link.remaining_graph = remaining_graph;
  io_link.infered_nodes = sliced_nodes;
  io_link.uninfer_nodes = remaining_nodes;

  auto ret = graph_builder_.GetIOMapping(io_link);
  EXPECT_EQ(ret, GRAPH_SUCCESS);

  GraphMapping target_mapping = {{"data1", {{0, {{"add1", 0}}}}},
                                {"data2", {{0, {{"add1", 1}}}}},
                                {"var", {{0, {{"add2", 1}, {"mul", 0}}}}},
                                {"data3", {{0, {{"mul", 1}}}}}};
  EXPECT_EQ(io_link.binary_graph_mapping, target_mapping);

  auto netout_node = sliced_graph->GetOrUpdateNetOutputNode();
  auto in_data_nodes = remaining_graph->GetInputNodes();
  for (const auto &io_idx_pair : io_link.out_idx_2_in_idxs) {
    auto out_name = netout_node->GetInDataAnchor(io_idx_pair.first)->GetPeerOutAnchor()->GetOwnerNode()->GetName();
    auto out_idx = netout_node->GetInDataAnchor(io_idx_pair.first)->GetPeerOutAnchor()->GetIdx();
    auto in_name = in_data_nodes.at(io_idx_pair.second)->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode()->GetName();
    auto in_idx = in_data_nodes.at(io_idx_pair.second)->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetIdx();
    int64_t data_index;
    ASSERT_TRUE(AttrUtils::GetInt(in_data_nodes.at(io_idx_pair.second)->GetOpDesc(), ATTR_NAME_INDEX, data_index));
    ASSERT_EQ(data_index, io_idx_pair.first);
    auto out_to_in_idx = target_mapping.find(out_name);
    ASSERT_TRUE(out_to_in_idx != target_mapping.end());
    auto peer_datas = out_to_in_idx->second.find(out_idx);
    std::pair<std::string, uint32_t> peer_data{in_name, in_idx};
    auto it = std::find(peer_datas->second.begin(), peer_datas->second.end(), peer_data);
    ASSERT_TRUE(it != peer_datas->second.end());
  }
  GE_DUMP(sliced_graph, "sliced_graph");
  GE_DUMP(remaining_graph, "remaining_graph");
}

TEST_F(BinaryGraphBuilderUT, return_remaining_graph_input_with_var_node_when_sliced_graph_out_is_var) {
  ComputeGraphPtr graph;
  std::vector<NodePtr> sliced_nodes;
  std::vector<NodePtr> remaining_nodes;
  BuildGraphForTestInOrOutNodeGenerate(graph, sliced_nodes, remaining_nodes);

  auto sliced_graph = graph_builder_.BuildGraph(sliced_nodes, "Sliced_Graph");
  auto remaining_graph = graph_builder_.BuildGraph(remaining_nodes, "Remaining_Graph");

  BinaryGraphIOLinkage io_link;
  io_link.sliced_graph = sliced_graph;
  io_link.remaining_graph = remaining_graph;
  io_link.infered_nodes = sliced_nodes;
  io_link.uninfer_nodes = remaining_nodes;

  auto ret = graph_builder_.GetIOMapping(io_link);
  EXPECT_EQ(ret, GRAPH_SUCCESS);

  auto mul_node = remaining_graph->FindNode("mul");
  auto node_type = mul_node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetType();
  EXPECT_EQ(node_type, DATA);

  auto add2_node = remaining_graph->FindNode("add2");
  node_type = add2_node->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetType();
  EXPECT_EQ(node_type, DATA);

  EXPECT_EQ(remaining_graph->GetAllNodesSize(), 9);
  EXPECT_EQ(remaining_graph->GetInputNodes().size(), 5);

  ret = graph_builder_.ReplaceInputNode(io_link);
  EXPECT_EQ(ret, GRAPH_SUCCESS);

  node_type = mul_node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetType();
  EXPECT_EQ(node_type, VARIABLE);

  node_type = add2_node->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetType();
  EXPECT_EQ(node_type, VARIABLE);

  EXPECT_EQ(remaining_graph->GetAllNodesSize(), 8);
  EXPECT_EQ(remaining_graph->GetInputNodes().size(), 3);
  EXPECT_EQ(io_link.out_idx_2_in_idxs.size(), 3);

  auto var_node = remaining_graph->FindNode("var");
  EXPECT_EQ(mul_node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode(),
            var_node);
  EXPECT_EQ(mul_node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode(),
            add2_node->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode());
}

TEST_F(BinaryGraphBuilderUT, return_sliced_graph_output_unlink_var_node_when_sliced_graph_out_is_var) {
  ComputeGraphPtr graph;
  std::vector<NodePtr> sliced_nodes;
  std::vector<NodePtr> remaining_nodes;
  BuildGraphForTestInOrOutNodeGenerate(graph, sliced_nodes, remaining_nodes);

  auto sliced_graph = graph_builder_.BuildGraph(sliced_nodes, "Sliced_Graph");
  auto remaining_graph = graph_builder_.BuildGraph(remaining_nodes, "Remaining_Graph");

  BinaryGraphIOLinkage io_link;
  io_link.sliced_graph = sliced_graph;
  io_link.remaining_graph = remaining_graph;
  io_link.infered_nodes = sliced_nodes;
  io_link.uninfer_nodes = remaining_nodes;

  auto ret = graph_builder_.GetIOMapping(io_link);
  EXPECT_EQ(ret, GRAPH_SUCCESS);

  auto netout = sliced_graph->FindFirstNodeMatchType(NETOUTPUT);
  EXPECT_NE(netout, nullptr);
  EXPECT_EQ(netout->GetAllInDataAnchorsSize(), 4);
  EXPECT_EQ(netout->GetOpDesc()->GetAllInputsDesc().size(), 4);
  EXPECT_EQ(io_link.out_idx_2_in_idxs.size(), 5);
  EXPECT_EQ(io_link.sliced_graph->GetOutputSize(), 4);
  auto data1 = sliced_graph->FindNode("data1");
  auto data2 = sliced_graph->FindNode("data2");
  auto data3 = sliced_graph->FindNode("data3");
  auto var = sliced_graph->FindNode("var");

  EXPECT_EQ(data1->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode(),
            data2->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode());
  EXPECT_EQ(data2->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode(),
            data3->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode());
  EXPECT_EQ(data3->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode(),
            var->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode());

  ret = graph_builder_.ReplaceInputNode(io_link);
  EXPECT_EQ(ret, GRAPH_SUCCESS);

  auto old_netout = netout;
  netout = sliced_graph->FindFirstNodeMatchType(NETOUTPUT);
  EXPECT_NE(old_netout, netout);
  EXPECT_EQ(netout->GetAllInDataAnchorsSize(), 3);
  EXPECT_EQ(netout->GetOpDesc()->GetAllInputsDesc().size(), 3);
  EXPECT_EQ(io_link.out_idx_2_in_idxs.size(), 3);
  EXPECT_EQ(io_link.sliced_graph->GetOutputSize(), 3);

  for (uint32_t index = 0; index < netout->GetAllInDataAnchorsSize(); ++index) {
    auto in_data_anchor = netout->GetInDataAnchor(index);
    EXPECT_EQ(index, in_data_anchor->GetIdx());
    auto peer_node = in_data_anchor->GetPeerOutAnchor()->GetOwnerNode();
    auto peer_idex = in_data_anchor->GetPeerOutAnchor()->GetIdx();
    EXPECT_NE(peer_node, nullptr);
    EXPECT_EQ(peer_node->GetOpDesc()->GetOutputDesc(peer_idex), netout->GetOpDesc()->GetInputDesc(index));
  }

  EXPECT_EQ(data1->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode(),
            data2->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode());
  EXPECT_EQ(data2->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode(),
            data3->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode());
  EXPECT_EQ(var->GetOutDataAnchor(0)->GetPeerInDataAnchors().size(), 0);
  EXPECT_EQ(data3->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode(),
            var->GetOutControlAnchor()->GetPeerInControlAnchors().at(0)->GetOwnerNode());
}

TEST_F(BinaryGraphBuilderUT, return_remaining_graph_input_merge_node_when_sliced_graph_out_is_same_one_node) {
  ComputeGraphPtr graph;
  std::vector<NodePtr> sliced_nodes;
  std::vector<NodePtr> remaining_nodes;
  BuildGraphForTestInOrOutNodeGenerate(graph, sliced_nodes, remaining_nodes);

  auto sliced_graph = graph_builder_.BuildGraph(sliced_nodes, "Sliced_Graph");
  auto remaining_graph = graph_builder_.BuildGraph(remaining_nodes, "Remaining_Graph");

  BinaryGraphIOLinkage io_link;
  io_link.sliced_graph = sliced_graph;
  io_link.remaining_graph = remaining_graph;
  io_link.infered_nodes = sliced_nodes;
  io_link.uninfer_nodes = remaining_nodes;

  auto ret = graph_builder_.GetIOMapping(io_link);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(remaining_graph->GetInputNodes().size(), 5);
  ret = graph_builder_.MergeSameInputNode(io_link);
  EXPECT_EQ(remaining_graph->GetInputNodes().size(), 4);  
  auto mul_node = remaining_graph->FindNode("mul");
  auto add2_node = remaining_graph->FindNode("add2");
  EXPECT_EQ(mul_node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode(),
            add2_node->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode());
}

TEST_F(BinaryGraphBuilderUT, return_remaining_graph_input_node_with_desc_when_origin_graph_has_desc) {
  ComputeGraphPtr graph;
  std::vector<NodePtr> sliced_nodes;
  std::vector<NodePtr> remaining_nodes;
  BuildGraphForTestInOrOutNodeGenerate(graph, sliced_nodes, remaining_nodes);

  auto sliced_graph = graph_builder_.BuildGraph(sliced_nodes, "Sliced_Graph");
  auto remaining_graph = graph_builder_.BuildGraph(remaining_nodes, "Remaining_Graph");

  BinaryGraphIOLinkage io_link;
  io_link.sliced_graph = sliced_graph;
  io_link.remaining_graph = remaining_graph;
  io_link.infered_nodes = sliced_nodes;
  io_link.uninfer_nodes = remaining_nodes;

  auto ret = graph_builder_.GetIOMapping(io_link);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  ret = graph_builder_.SetInputNodeDesc(io_link);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  auto in_data_nodes = remaining_graph->GetInputNodes();
  for (const auto & in_node : in_data_nodes) {
      EXPECT_EQ(in_node->GetOpDesc()->GetOutputDescPtr(0)->GetShape(), GeShape({1, 1, 224, 224}));
      EXPECT_EQ(in_node->GetOpDesc()->GetOutputDescPtr(0)->GetFormat(), FORMAT_NCHW);
      EXPECT_EQ(in_node->GetOpDesc()->GetOutputDescPtr(0)->GetDataType(), DT_FLOAT);
  }
}

/*
 *   data1  data2
 *      \    /
 *       sub
 *        |
 *     netoutput
 */
ut::GraphBuilder TestSubgraphBuilder() {
  ut::GraphBuilder graph_builder = ut::GraphBuilder("branch_graph");
  std::vector<int64_t> shape1 = {1,1};
  auto data1 = graph_builder.AddNode("data1_1", "Data", 1, 1, FORMAT_NCHW, DT_INT32, shape1);
  auto data1_desc = data1->GetOpDesc();
  EXPECT_NE(data1_desc, nullptr);
  AttrUtils::SetInt(data1_desc, "_parent_node_index", 0);
  std::vector<int64_t> shape2 = {2,2};
  auto data2 = graph_builder.AddNode("data2_1", "Data", 1, 1, FORMAT_NCHW, DT_INT32, shape2);
  auto data2_desc = data2->GetOpDesc();
  EXPECT_NE(data2_desc, nullptr);
  AttrUtils::SetInt(data2_desc, "_parent_node_index", 1);

  auto sub = graph_builder.AddNode("Sub", "Sub", 2, 1);
  std::vector<int64_t> shape3 = {8,8};
  auto netoutput = graph_builder.AddNode("output", NETOUTPUT, 1, 0, FORMAT_NCHW, DT_INT32, shape3);
  auto input0_desc = netoutput->GetOpDesc()->MutableInputDesc(0);
  EXPECT_NE(input0_desc, nullptr);
  AttrUtils::SetInt(input0_desc, "_parent_node_index", 0);

  graph_builder.AddDataEdge(data1, 0, sub, 0);
  graph_builder.AddDataEdge(data2, 0, sub, 1);
  graph_builder.AddDataEdge(sub, 0, netoutput, 0);
  return graph_builder;
}

/*
 *   data1  data2
 *     \     /
 *      case
 *        |
 *    netoutput
 */
ut::GraphBuilder GraphBuilder() {
  ut::GraphBuilder graph_builder = ut::GraphBuilder("root_graph");
  auto data1 = graph_builder.AddNode("data1", "Data", 0, 1);
  auto data2 = graph_builder.AddNode("data2", "Data", 0, 1);
  auto case_node = graph_builder.AddNode("case", CASE, 2, 1);
  auto netoutput = graph_builder.AddNode("netoutput", NETOUTPUT, 1, 0);
  graph_builder.AddDataEdge(data1, 0, case_node, 0);
  graph_builder.AddDataEdge(data2, 0, case_node, 1);
  graph_builder.AddDataEdge(case_node, 0, netoutput, 0);

  auto parent_graph = graph_builder.GetGraph();
  auto subgraph_builder = TestSubgraphBuilder();
  auto subgraph = subgraph_builder.GetGraph();
  case_node->GetOpDesc()->AddSubgraphName(subgraph->GetName());
  case_node->GetOpDesc()->SetSubgraphInstanceName(0, subgraph->GetName());
  subgraph->SetParentNode(case_node);
  subgraph->SetParentGraph(parent_graph);
  EXPECT_EQ(parent_graph->AddSubgraph(subgraph->GetName(), subgraph), GRAPH_SUCCESS);
  return graph_builder;
}

TEST_F(BinaryGraphBuilderUT, return_sliced_graph_with_subgrapg_when_infered_node_contain_subgraph) {
  auto graph_builder = GraphBuilder();
  auto parent_graph = graph_builder.GetGraph();
  auto subgraphs = parent_graph->GetAllSubgraphs();
  EXPECT_EQ(subgraphs.size(), 1);

  auto data1_node = parent_graph->FindNode("data1");
  auto data2_node = parent_graph->FindNode("data2");
  auto case_node = parent_graph->FindNode("case");
  std::vector<NodePtr> sliced_nodes{data1_node, data2_node, case_node};

  auto sliced_graph = graph_builder_.BuildGraph(sliced_nodes, "Sliced_Graph");
  EXPECT_NE(sliced_graph, nullptr);
  auto new_subgraphs = sliced_graph->GetAllSubgraphs();
  EXPECT_EQ(new_subgraphs.size(), 1);
  auto new_case_node = sliced_graph->FindNode("case");
  const auto &subgraph_names = new_case_node->GetOpDesc()->GetSubgraphInstanceNames();
  EXPECT_EQ(subgraph_names.size(), 1);
  auto subgraph = sliced_graph->GetSubgraph(subgraph_names[0]);
  EXPECT_EQ(subgraph->GetAllNodesSize(), 4);
}