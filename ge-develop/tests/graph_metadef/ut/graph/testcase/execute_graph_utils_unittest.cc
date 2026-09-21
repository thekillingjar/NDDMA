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
#include <vector>
#include "graph/utils/execute_graph_utils.h"
#include "graph/utils/fast_node_utils.h"
#include "graph/debug/ge_op_types.h"
#include "graph/debug/ge_attr_define.h"
#include "framework/common/debug/ge_log.h"
#include "graph_builder_utils.h"
#include "share_graph.h"

namespace ge {

using ExecuteSharedGraph = SharedGraph<ExecuteGraphPtr, ut::ExecuteGraphBuilder>;

class UtestExecuteGraphUtils : public testing::Test {
 public:
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(UtestExecuteGraphUtils, FindNodeFromAllNodes_Ok_GraphIsNull) {
  const auto node = ExecuteGraphUtils::FindNodeFromAllNodes(nullptr, "graph_name");
  EXPECT_EQ(node, nullptr);
}

TEST_F(UtestExecuteGraphUtils, FindNodeFromAllNodes_Ok_NameIsNull) {
  auto builder = ut::ExecuteGraphBuilder("Test1");
  auto graph = builder.GetGraph();
  auto node = ExecuteGraphUtils::FindNodeFromAllNodes(graph.get(), nullptr);
  EXPECT_EQ(node, nullptr);
}

TEST_F(UtestExecuteGraphUtils, FindNodeFromAllNodes_Ok_TryFindNodeInSubGraph) {
  const auto graph = ExecuteSharedGraph::BuildGraphWithSubGraph();
  const auto node = ExecuteGraphUtils::FindNodeFromAllNodes(graph.get(), "data1");
  ASSERT_NE(node, nullptr);
  EXPECT_EQ(node->GetName(), "data1");
}

TEST_F(UtestExecuteGraphUtils, FindNodesByTypeFromAllNodes_Ok_FindAllDataNodes) {
  const auto graph = ExecuteSharedGraph::BuildGraphWithSubGraph();
  const auto nodes = ExecuteGraphUtils::FindNodesByTypeFromAllNodes(graph.get(), "Data");
  EXPECT_EQ(nodes.size(), 3);
}

TEST_F(UtestExecuteGraphUtils, FindFirstNodeMatchType_OK) {
  const auto root_graph = ExecuteSharedGraph::BuildGraphWithSubGraph();
  auto node = ExecuteGraphUtils::FindFirstNodeMatchType(root_graph.get(), "Data");
  ASSERT_NE(node, nullptr);
  EXPECT_EQ(node->GetName(), "data0");
  node = ExecuteGraphUtils::FindFirstNodeMatchType(root_graph.get(), "Case");
  ASSERT_NE(node, nullptr);
  EXPECT_EQ(node->GetName(), "case0");
  node = ExecuteGraphUtils::FindFirstNodeMatchType(root_graph.get(), "Relu");
  ASSERT_NE(node, nullptr);
  EXPECT_EQ(node->GetName(), "relu0");
}

TEST_F(UtestExecuteGraphUtils, InsertNodeAfter_Fail_NodesInDifferentGraph) {
  auto graph_builder0 = ut::ExecuteGraphBuilder("test_graph0");
  const auto &node0 = graph_builder0.AddNode("data0", DATA, 1, 1);
  const auto &graph0 = graph_builder0.GetGraph();

  auto graph_builder1 = ut::ExecuteGraphBuilder("test_graph1");
  const auto &node1 = graph_builder1.AddNode("data1", DATA, 1, 1);
  const auto &graph1 = graph_builder1.GetGraph();
  ASSERT_EQ(ExecuteGraphUtils::InsertNodeAfter({node0, 0}, {}, node1, 0, 0), PARAM_INVALID);
}

TEST_F(UtestExecuteGraphUtils, InsertNodeAfter_Fail_AddEdgeFail) {
  auto graph_builder0 = ut::ExecuteGraphBuilder("test_graph0");
  const auto &node0 = graph_builder0.AddNode("data0", DATA, 1, 1);
  const auto &graph0 = graph_builder0.GetGraph();
  EdgeSrcEndpoint src = {node0, 0};
  std::vector<EdgeDstEndpoint> dsts;
  dsts.emplace_back(node0, 0);
  // case1: input_index exceeds the size of in edges
  int ret = ExecuteGraphUtils::InsertNodeAfter(src, dsts, node0, 1, 0);
  EXPECT_NE(ret, GRAPH_SUCCESS);

  // case2: output_index exceeds the size of out edges
  int ret2 = ExecuteGraphUtils::InsertNodeAfter(src, dsts, node0, 0, 1);
  EXPECT_NE(ret2, GRAPH_SUCCESS);
}

TEST_F(UtestExecuteGraphUtils, InsertNodeAfter_Ok_NodeTypeIsSwitch) {
  auto graph_builder0 = ut::ExecuteGraphBuilder("test_graph0");
  const auto &node0 = graph_builder0.AddNode("data0", SWITCH, 1, 1);
  const auto &graph0 = graph_builder0.GetGraph();
  EdgeSrcEndpoint src = {node0, 0};
  std::vector<EdgeDstEndpoint> dsts;
  dsts.emplace_back(node0, 0);
  int ret = ExecuteGraphUtils::InsertNodeAfter(src, dsts, node0, 0, 0);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestExecuteGraphUtils, InsertNodeAfter_Fail_SrcOwnerGraphNotEqualDstOwnerGraph) {
  auto graph_builder0 = ut::ExecuteGraphBuilder("test_graph0");
  const auto &node0 = graph_builder0.AddNode("data0", DATA, 1, 1);
  const auto &graph0 = graph_builder0.GetGraph();

  auto graph_builder1 = ut::ExecuteGraphBuilder("test_graph1");
  const auto &node1 = graph_builder1.AddNode("data1", DATA, 1, 1);
  const auto &graph1 = graph_builder1.GetGraph();
  EdgeSrcEndpoint src = {node0, 0};
  std::vector<EdgeDstEndpoint> dsts;
  dsts.emplace_back(node1, 0);
  int ret = ExecuteGraphUtils::InsertNodeAfter(src, dsts, node0, 0, 0);
  EXPECT_NE(ret, GRAPH_SUCCESS);
}

TEST_F(UtestExecuteGraphUtils, InsertNodeAfter_Ok_GraphWithControlEdge) {
  auto graph = ExecuteSharedGraph::BuildGraphWithControlEdge();
  auto src_node = ExecuteGraphUtils::FindNodeFromAllNodes(graph.get(), "n1");
  EdgeSrcEndpoint src = {src_node, 0};
  std::vector<EdgeDstEndpoint> dsts;
  for (const auto out_edge : src_node->GetOutEdgesByIndex(0)) {
    dsts.emplace_back(FastNodeUtils::GetDstEndpoint(out_edge));
  }
  auto builder = ut::ExecuteGraphBuilder("test_graph");
  auto node_to_insert = builder.AddNode("inserted_node", "Op", 1, 3);
  (void) node_to_insert->GetExtendInfo()->SetOwnerGraph(graph.get(), node_to_insert);
  graph->AddNode(node_to_insert);
  auto original_edge_size = graph->GetAllEdges().size();
  EXPECT_EQ(ExecuteGraphUtils::InsertNodeAfter(src, dsts, node_to_insert, 0, 0), GRAPH_SUCCESS);
  EXPECT_EQ(graph->GetAllEdges().size(), original_edge_size + 1);
}

TEST_F(UtestExecuteGraphUtils, InsertNodeAfter_Ok_will_null_out_control_edge) {
  auto builder = ut::ExecuteGraphBuilder("test");
  const auto &data = builder.AddNode("data", "Data", 0, 1);
  const auto &n1 = builder.AddNode("n1", "Op", 1, 1);
  const auto &n2 = builder.AddNode("n2", "Op", 1, 1);
  const auto &insert_node = builder.AddNode("insert_node", "Op", 1, 1);

  builder.AddDataEdge(data, 0, n1, 0);
  builder.AddDataEdge(n1, 0, n2, 0);
  builder.AddControlEdge(n1, n2);
  auto graph = builder.GetGraph();
  for (const auto edge : n2->GetAllInControlEdgesRef()) {
    if (edge != nullptr) {
      graph->RemoveEdge(edge);
    }
  }
  EXPECT_EQ(ExecuteGraphUtils::InsertNodeAfter({n1, 0}, {{n2, 0}}, insert_node, 0, 0), GRAPH_SUCCESS);
  EXPECT_EQ(n2->GetInDataNodes().at(0)->GetName(), "insert_node");
  EXPECT_EQ(n2->GetInControlNodes().size(), 0);
}

TEST_F(UtestExecuteGraphUtils, InsertNodeBefore_Fail_GetOwnerComputeGraphFail) {
  auto graph_builder0 = ut::ExecuteGraphBuilder("test_graph0");
  const auto &node0 = graph_builder0.AddNode("data0", DATA, 1, 1);
  const auto &graph0 = graph_builder0.GetGraph();

  auto graph_builder1 = ut::ExecuteGraphBuilder("test_graph1");
  const auto &node1 = graph_builder1.AddNode("data1", DATA, 1, 1);
  const auto &graph1 = graph_builder1.GetGraph();

  int ret = ExecuteGraphUtils::InsertNodeBefore({node0, 0}, node1, 0, 0);
  EXPECT_NE(ret, GRAPH_SUCCESS);
}

TEST_F(UtestExecuteGraphUtils, InsertNodeBefore_Fail_OutputIndexOutOfBounds) {
  auto builder = ut::ExecuteGraphBuilder("test");
  const auto &var = builder.AddNode("var", VARIABLE, 0, 1);
  const auto &assign = builder.AddNode("assign", "Assign", 1, 1);
  const auto &allreduce = builder.AddNode("allreduce", "HcomAllReduce", 1, 1);
  const auto &atomic_clean = builder.AddNode("atomic_clean", ATOMICADDRCLEAN, 0, 0);
  const auto &netoutput1 = builder.AddNode("netoutput", NETOUTPUT, 1, 0);
  const auto &identity = builder.AddNode("identity", "Identity", 1, 1);

  builder.AddDataEdge(var, 0, assign, 0);
  builder.AddDataEdge(var, 0, allreduce, 0);
  builder.AddDataEdge(allreduce, 0, netoutput1, 0);
  builder.AddControlEdge(assign, allreduce);
  builder.AddControlEdge(atomic_clean, allreduce);
  auto graph = builder.GetGraph();

  int ret = ExecuteGraphUtils::InsertNodeBefore({allreduce, 0}, identity, 0, 5);
  EXPECT_NE(ret, GRAPH_SUCCESS);
}

/*
*               var                              var
*  atomicclean  |  \                             |  \
*         c\    |   assign                       |   assign
*           \   |   c/         =======>          |   c/
*           allreduce                         identity  atomicclean
*             |                                 |     c/
*            netoutput                        allreduce
*                                               |
*                                           netoutput
 */
TEST_F(UtestExecuteGraphUtils, InsertNodeBefore_Ok_DoNotMoveCtrlEdgeFromAtomicClean) {
  auto builder = ut::ExecuteGraphBuilder("test");
  const auto &var = builder.AddNode("var", VARIABLE, 0, 1);
  const auto &assign = builder.AddNode("assign", "Assign", 1, 1);
  const auto &allreduce = builder.AddNode("allreduce", "HcomAllReduce", 1, 1);
  const auto &atomic_clean = builder.AddNode("atomic_clean", ATOMICADDRCLEAN, 0, 0);
  const auto &netoutput1 = builder.AddNode("netoutput", NETOUTPUT, 1, 0);
  const auto &identity = builder.AddNode("identity", "Identity", 1, 1);

  builder.AddDataEdge(var, 0, assign, 0);
  builder.AddDataEdge(var, 0, allreduce, 0);
  builder.AddDataEdge(allreduce, 0, netoutput1, 0);
  builder.AddControlEdge(assign, allreduce);
  builder.AddControlEdge(atomic_clean, allreduce);
  auto graph = builder.GetGraph();

  EXPECT_EQ(ExecuteGraphUtils::InsertNodeBefore({allreduce, 0}, identity, 0, 0), GRAPH_SUCCESS);
  EXPECT_EQ(identity->GetInControlNodes().at(0)->GetName(), "assign");
}

TEST_F(UtestExecuteGraphUtils, InsertNodeBefore_Ok_will_null_in_control_edge) {
  auto builder = ut::ExecuteGraphBuilder("test");
  const auto &data = builder.AddNode("data", "Data", 0, 1);
  const auto &n1 = builder.AddNode("n1", "Op", 1, 1);
  const auto &n2 = builder.AddNode("n2", "Op", 1, 1);
  const auto &insert_node = builder.AddNode("insert_node", "Op", 1, 1);

  builder.AddDataEdge(data, 0, n1, 0);
  builder.AddDataEdge(n1, 0, n2, 0);
  builder.AddControlEdge(n1, n2);
  auto graph = builder.GetGraph();
  for (const auto edge : n2->GetAllInControlEdgesRef()) {
    if (edge != nullptr) {
      graph->RemoveEdge(edge);
    }
  }
  EXPECT_EQ(ExecuteGraphUtils::InsertNodeBefore({n2, 0}, insert_node, 0, 0), GRAPH_SUCCESS);
  EXPECT_EQ(n2->GetInDataNodes().at(0)->GetName(), "insert_node");
  EXPECT_EQ(n2->GetInControlNodes().size(), 0);
}

TEST_F(UtestExecuteGraphUtils, CopyInCtrlEdges_Fail_NodeIsNull) {
  auto builder = ut::ExecuteGraphBuilder("test");
  const auto &src_node = builder.AddNode("src_node", "node", 1, 1);
  int ret = ExecuteGraphUtils::MoveInCtrlEdges(src_node, nullptr);
  EXPECT_EQ(ret, PARAM_INVALID);
}

TEST_F(UtestExecuteGraphUtils, CopyInCtrlEdges_Ok_SrcCtrlInNodesIsEmpty) {
  auto builder = ut::ExecuteGraphBuilder("test_graph0");
  const auto &src_node = builder.AddNode("data0", "data", 1, 1);
  auto dst_node = builder.AddNode("data1", "data", 1, 1);
  int ret = ExecuteGraphUtils::MoveInCtrlEdges(src_node, dst_node);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestExecuteGraphUtils, CopyInCtrlEdges_Ok) {
  auto builder = ut::ExecuteGraphBuilder("test");
  const auto &in_ctrl_node1 = builder.AddNode("in_ctrl_node1", "node", 1, 1);
  const auto &in_ctrl_node2 = builder.AddNode("in_ctrl_node2", "node", 1, 1);
  const auto &src_node = builder.AddNode("src_node", "node", 1, 1);
  auto dst_node = builder.AddNode("dst_node", "node", 1, 1);
  builder.AddDataEdge(src_node, 0, dst_node, 0);
  builder.AddControlEdge(src_node, dst_node);
  builder.AddControlEdge(in_ctrl_node1, src_node);
  builder.AddControlEdge(in_ctrl_node2, dst_node);
  const auto graph = builder.GetGraph();
  ASSERT_EQ(ExecuteGraphUtils::MoveInCtrlEdges(src_node, dst_node), GRAPH_SUCCESS);
  EXPECT_EQ(graph->GetAllEdges().size(), 4);
  EXPECT_EQ(dst_node->GetAllInControlEdgesSize(), 3);
  EXPECT_EQ(dst_node->GetInControlNodes().back()->GetName(), "in_ctrl_node1");
}

TEST_F(UtestExecuteGraphUtils, MoveInCtrlEdges_Fail_NodeIsNull) {
  int ret = ExecuteGraphUtils::MoveInCtrlEdges(nullptr, nullptr);
  EXPECT_EQ(ret, PARAM_INVALID);
}

TEST_F(UtestExecuteGraphUtils, MoveInCtrlEdges_Ok) {
  auto builder = ut::ExecuteGraphBuilder("test");
  const auto &in_ctrl_node1 = builder.AddNode("in_ctrl_node1", "node", 1, 1);
  const auto &in_ctrl_node2 = builder.AddNode("in_ctrl_node2", "node", 1, 1);
  const auto &src_node = builder.AddNode("src_node", "node", 1, 1);
  auto dst_node = builder.AddNode("dst_node", "node", 1, 1);
  builder.AddDataEdge(src_node, 0, dst_node, 0);
  builder.AddControlEdge(src_node, dst_node);
  builder.AddControlEdge(in_ctrl_node1, src_node);
  builder.AddControlEdge(in_ctrl_node2, src_node);
  const auto graph = builder.GetGraph();
  const auto original_edge_size = graph->GetAllEdges().size();
  ASSERT_EQ(ExecuteGraphUtils::MoveInCtrlEdges(src_node, dst_node), GRAPH_SUCCESS);
  // move control edge does not change edge size of graph
  EXPECT_TRUE(graph->GetAllEdges().size() == original_edge_size);
  EXPECT_EQ(dst_node->GetAllInControlEdgesSize(), 3);
  EXPECT_EQ(dst_node->GetInControlNodes()[1]->GetName(), "in_ctrl_node1");
  EXPECT_EQ(dst_node->GetInControlNodes()[2]->GetName(), "in_ctrl_node2");
}

TEST_F(UtestExecuteGraphUtils, CopyOutCtrlEdges_Fail_NodeIsNull) {
  int ret = ExecuteGraphUtils::MoveOutCtrlEdges(nullptr, nullptr);
  EXPECT_EQ(ret, PARAM_INVALID);
}

TEST_F(UtestExecuteGraphUtils, CopyOutCtrlEdges_Fail_OutCtrlNodesIsEmpty) {
  auto builder = ut::ExecuteGraphBuilder("test_graph0");
  const auto &src_node = builder.AddNode("data0", "data", 1, 1);
  auto dst_node = builder.AddNode("data1", "data", 1, 1);
  int ret = ExecuteGraphUtils::MoveOutCtrlEdges(src_node, dst_node);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestExecuteGraphUtils, CopyOutCtrlEdges_Ok) {
  auto builder = ut::ExecuteGraphBuilder("test_graph0");
  const auto &src_node = builder.AddNode("src_node", NETOUTPUT, 1, 1);
  const auto &ctrl_node = builder.AddNode("ctrl_node", CONSTANT, 0, 0);
  const auto &ctrl_node2 = builder.AddNode("ctrl_node2", CONSTANT, 0, 0);
  auto dst_node = builder.AddNode("dst_node", NETOUTPUT, 1, 1);
  auto graph = builder.GetGraph();
  // 疑问: 是否会出现自依赖的情况?
  // builder.AddControlEdge(src_node, dst_node);
  builder.AddControlEdge(src_node, ctrl_node);
  builder.AddControlEdge(src_node, ctrl_node2);

  EXPECT_EQ(ExecuteGraphUtils::MoveOutCtrlEdges(src_node, dst_node), GRAPH_SUCCESS);
  // EXPECT_EQ(dst_node->GetOutControlNodes().size(), src_node->GetOutControlNodes().size());
  EXPECT_EQ(dst_node->GetOutControlNodes().at(1U), ctrl_node2);
}

TEST_F(UtestExecuteGraphUtils, MoveOutCtrlEdges_Fail_NodeIsNull) {
  int ret = ExecuteGraphUtils::MoveOutCtrlEdges(nullptr, nullptr);
  EXPECT_EQ(ret, PARAM_INVALID);
}

TEST_F(UtestExecuteGraphUtils, MoveOutCtrlEdges_Ok) {
  auto graph = ExecuteSharedGraph::BuildGraphWithControlEdge();
  auto src_node = ExecuteGraphUtils::FindNodeFromAllNodes(graph.get(), "n1");
  auto dst_node = ExecuteGraphUtils::FindNodeFromAllNodes(graph.get(), "n4");
  int ret = ExecuteGraphUtils::MoveOutCtrlEdges(src_node, dst_node);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(src_node->GetAllOutControlEdgesSize(), 0);
  EXPECT_EQ(dst_node->GetAllOutControlEdgesSize(), 2);
}

TEST_F(UtestExecuteGraphUtils, MoveNodeToGraph_Ok_MoveNodeWithSubGraph) {
  auto src_graph = ExecuteSharedGraph::BuildGraphWithSubGraph();
  auto dst_graph = ExecuteSharedGraph::BuildGraphWithControlEdge();
  // find a node with subgraph
  auto node_to_move = ExecuteGraphUtils::FindNodeFromAllNodes(src_graph.get(), "case0");
  auto src_direct_node_size = src_graph->GetDirectNodesSize();
  auto dst_direct_node_size = dst_graph->GetDirectNodesSize();
  EXPECT_EQ(ExecuteGraphUtils::MoveNodeToGraph(node_to_move, dst_graph.get()), GRAPH_SUCCESS);
  EXPECT_EQ(dst_graph->GetDirectNodesSize(), dst_direct_node_size + 1);
  EXPECT_EQ(src_graph->GetDirectNodesSize(), src_direct_node_size - 1);
  EXPECT_EQ(src_graph->GetAllSubgraphs().size(), 0);
}

TEST_F(UtestExecuteGraphUtils, ReplaceEdgeSrc_Ok) {
  auto graph = ExecuteSharedGraph::BuildGraphWithControlEdge();
  auto n2 = ExecuteGraphUtils::FindNodeFromAllNodes(graph.get(), "n2");
  EdgeSrcEndpoint src = {n2, 0};
  auto n3 = ExecuteGraphUtils::FindNodeFromAllNodes(graph.get(), "n3");
  auto old_edge = n3->GetInDataEdgeByIndex(0);
  EXPECT_EQ(ExecuteGraphUtils::ReplaceEdgeSrc(old_edge, src), GRAPH_SUCCESS);
  auto new_edges = n3->GetInDataEdgeByIndex(0);
  EXPECT_NE(new_edges, nullptr);
  auto src_node = new_edges->src;
  EXPECT_EQ(src_node, n2);
  auto dst_node = new_edges->dst;
  EXPECT_EQ(dst_node, n3);
}

TEST_F(UtestExecuteGraphUtils, ReplaceEdgeSrc_Fail_null_node) {
  auto graph = ExecuteSharedGraph::BuildGraphWithControlEdge();
  EdgeSrcEndpoint src = {nullptr, 0};
  auto n3 = ExecuteGraphUtils::FindNodeFromAllNodes(graph.get(), "n3");
  auto old_edge = n3->GetInDataEdgeByIndex(0);
  EXPECT_EQ(ExecuteGraphUtils::ReplaceEdgeSrc(old_edge, src), GRAPH_FAILED);
  EXPECT_EQ(ExecuteGraphUtils::ReplaceEdgeSrc(nullptr, src), PARAM_INVALID);
}

TEST_F(UtestExecuteGraphUtils, FindRootGraph_Ok_subgraph) {
  auto graph = ExecuteSharedGraph::BuildGraphWithSubGraph();
  auto sub_graph = graph->GetSubGraph("sub1");
  auto root_graph = ExecuteGraphUtils::FindRootGraph(sub_graph);
  EXPECT_EQ(root_graph, graph.get());
}

TEST_F(UtestExecuteGraphUtils, FindRootGraph_Ok_root_graph) {
  auto graph = ExecuteSharedGraph::BuildGraphWithSubGraph();
  auto root_graph = ExecuteGraphUtils::FindRootGraph(graph.get());
  EXPECT_EQ(root_graph, graph.get());
}

TEST_F(UtestExecuteGraphUtils, FindRootGraph_Ok_null_param) {
  auto root_graph = ExecuteGraphUtils::FindRootGraph(nullptr);
  EXPECT_EQ(root_graph, nullptr);
}

TEST_F(UtestExecuteGraphUtils, ReplaceNodeEdges_Ok_replace_data_control_edge) {
  auto graph = ExecuteSharedGraph::BuildGraphWithControlEdge();
  auto n4 = ExecuteGraphUtils::FindNodeFromAllNodes(graph.get(), "n4");
  auto builder = ut::ExecuteGraphBuilder("test_graph");
  auto new_node = builder.AddNode("inserted_node", "Op", 1, 1);
  (void) new_node->GetExtendInfo()->SetOwnerGraph(graph.get(), new_node);
  graph->AddNode(new_node);
  EXPECT_EQ(ExecuteGraphUtils::ReplaceNodeEdges(new_node, n4, {0}, {0}), GRAPH_SUCCESS);
  EXPECT_EQ(n4->GetAllInDataEdgesSize(), 0U);
  EXPECT_EQ(n4->GetAllOutDataEdgesSize(), 0U);
  EXPECT_EQ(new_node->GetAllInDataEdgesSize(), 1U);
  EXPECT_EQ(new_node->GetAllInControlEdgesSize(), 1U);
  EXPECT_EQ(new_node->GetAllOutDataEdgesSize(), 1U);
  EXPECT_EQ(new_node->GetAllOutControlEdgesSize(), 1U);
}

TEST_F(UtestExecuteGraphUtils, IsolateNode_Ok_relink_data_control_edge_with_iomap) {
  auto graph = ExecuteSharedGraph::BuildGraphWithControlEdge();
  auto n4 = ExecuteGraphUtils::FindNodeFromAllNodes(graph.get(), "n4");
  EXPECT_EQ(ExecuteGraphUtils::IsolateNode(n4, {0}), GRAPH_SUCCESS);
  EXPECT_EQ(n4->GetAllInDataEdgesSize(), 0U);
  EXPECT_EQ(n4->GetAllInControlEdgesSize(), 0U);
  EXPECT_EQ(n4->GetAllOutDataEdgesSize(), 0U);
  EXPECT_EQ(n4->GetAllOutControlEdgesSize(), 0U);
  auto n1 = ExecuteGraphUtils::FindNodeFromAllNodes(graph.get(), "n1");
  auto n5 = ExecuteGraphUtils::FindNodeFromAllNodes(graph.get(), "n5");
  auto edge = n5->GetInDataEdgeByIndex(2);
  EXPECT_EQ(edge->src, n1);
  EXPECT_EQ(edge->src_output, 0);
}

TEST_F(UtestExecuteGraphUtils, IsolateNode_Ok_relink_data_control_edge_without_iomap) {
  auto graph = ExecuteSharedGraph::BuildGraphWithControlEdge();
  auto n4 = ExecuteGraphUtils::FindNodeFromAllNodes(graph.get(), "n4");
  EXPECT_EQ(ExecuteGraphUtils::IsolateNode(n4, {}), GRAPH_SUCCESS);
  EXPECT_EQ(n4->GetAllInDataEdgesSize(), 0U);
  EXPECT_EQ(n4->GetAllInControlEdgesSize(), 0U);
  EXPECT_EQ(n4->GetAllOutDataEdgesSize(), 0U);
  EXPECT_EQ(n4->GetAllOutControlEdgesSize(), 0U);
  auto n5 = ExecuteGraphUtils::FindNodeFromAllNodes(graph.get(), "n5");
  EXPECT_EQ(n5->GetInDataEdgeByIndex(2), nullptr);
}

TEST_F(UtestExecuteGraphUtils, IsolateNode_Ok_connected_data_out) {
  auto builder = ut::ExecuteGraphBuilder("test_graph0");
  auto n1 = builder.AddNode("n1", "Data", 1, 1);
  auto n2 = builder.AddNode("n2", "Op", 1, 1);
  auto n3 = builder.AddNode("n3", "Op", 1, 1);
  builder.AddDataEdge(n1, 0, n2, 0);
  builder.AddDataEdge(n2, 0, n3, 0);
  builder.AddControlEdge(n1, n3);
  EXPECT_EQ(ExecuteGraphUtils::IsolateNode(n2, {}), GRAPH_SUCCESS);
  //EXPECT_EQ(n3->GetAllInDataEdgesSize(), 1);
  EXPECT_EQ(n3->GetAllInControlEdgesSize(), 1);
}

TEST_F(UtestExecuteGraphUtils, RemoveSubgraphRecursively_Fail_NodeToRemoveIsInvalid) {
  auto graph = ExecuteSharedGraph::BuildGraphWithConst();
  auto builder = ut::ExecuteGraphBuilder("wild_graph");
  // case1: node has null owner graph
  auto invalid_node = builder.AddNode("invalid", "Data", 1, 1);
  ExecuteGraph *null_graph = nullptr;
  invalid_node->GetExtendInfo()->SetOwnerGraph(null_graph, invalid_node);
  EXPECT_EQ(ExecuteGraphUtils::RemoveSubgraphRecursively(graph.get(), invalid_node), GRAPH_SUCCESS);

  // case2: node is not in graph
  auto wild_node = builder.AddNode("wild", "Data", 1, 1);
  EXPECT_EQ(ExecuteGraphUtils::RemoveSubgraphRecursively(graph.get(), wild_node), GRAPH_SUCCESS);

  // case3: node without subgraph
  auto node_without_subg = graph->GetDirectNode().front();
  EXPECT_EQ(ExecuteGraphUtils::RemoveSubgraphRecursively(graph.get(), node_without_subg), GRAPH_SUCCESS);
}

TEST_F(UtestExecuteGraphUtils, ReplaceNodeDataEdges_Fail_GraphIsNull) {
  auto builder = ut::ExecuteGraphBuilder("demo_graph");
  auto node1 = builder.AddNode("data", "Data", 1, 1);
  auto node2 = builder.AddNode("data2", "Data", 1, 1);
  EXPECT_EQ(ExecuteGraphUtils::ReplaceNodeDataEdges(node1, node2, {}, {}, nullptr), GRAPH_SUCCESS);
}

TEST_F(UtestExecuteGraphUtils, GetNodeMapFromAllNodes_ok) {
  const auto root_graph = ExecuteSharedGraph::BuildGraphWithSubGraph();
  auto node_name_to_nodes = ExecuteGraphUtils::GetNodeMapFromAllNodes(root_graph.get());
  EXPECT_EQ(node_name_to_nodes.size(), 9U);
}

}  // namespace ge
