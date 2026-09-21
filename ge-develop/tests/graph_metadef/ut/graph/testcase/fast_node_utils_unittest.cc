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
#include <unordered_map>
#include "graph/fast_graph/execute_graph.h"
#include "graph/utils/fast_node_utils.h"
#include "graph/utils/execute_graph_utils.h"
#include "graph/fast_graph/edge.h"
#include "graph_builder_utils.h"
#include "graph/debug/ge_op_types.h"
#include "graph/debug/ge_attr_define.h"
#include "framework/common/debug/ge_log.h"

namespace ge {
namespace {
/*
 *                          +-----------+  +-----------+
 *                          | Graph 0   |  | Graph 1   |
 *                          |           |  |           |
 *                          | NetOutput |  | NetOutput |
 *       NetOutput          |   |       |  |   |       |
 *           |              |  Shape    |  |  Rank     |
 *         Case <---------> |   |       |  |   |       |
 *        /    \            | Data(1)   |  | Data(1)   |
 * pred(Data)  input(Data)  +-----------+  +-----------+
 */
std::shared_ptr<ExecuteGraph> BuildSimpleCaseGraph() {
  // main case graph
  auto builder = ut::ExecuteGraphBuilder("main_case_graph");
  auto main_data1 = builder.AddNode("data1", DATA, 1, 1);
  auto main_data2 = builder.AddNode("data2", DATA, 1, 1);
  auto case_node = builder.AddNode("case", CASE, 2, 1);
  auto main_output = builder.AddNode("output1", NETOUTPUT, 1, 1);
  builder.AddDataEdge(main_data1, 0, case_node, 0);
  builder.AddDataEdge(main_data2, 0, case_node, 1);
  builder.AddDataEdge(case_node, 0, main_output, 0);
  auto main_graph = builder.GetGraph();

  // case-1 subgraph
  auto sub_builder1 = ut::ExecuteGraphBuilder("case1_graph");
  auto sub_data1 = sub_builder1.AddNode("sub_data1", DATA, 1, 1);
  auto shape_node = sub_builder1.AddNode("shape", "Shape", 1, 1);
  auto sub_out1 = sub_builder1.AddNode("sub_output1", NETOUTPUT, 1, 1);
  sub_builder1.AddDataEdge(sub_data1, 0, shape_node, 0);
  sub_builder1.AddDataEdge(shape_node, 0, sub_out1, 0);
  AttrUtils::SetInt(sub_data1->GetOpDescBarePtr(), ATTR_NAME_PARENT_NODE_INDEX, 1);
  auto sub_graph1 = sub_builder1.GetGraph();

  // case-2 subgraph
  auto sub_builder2 = ut::ExecuteGraphBuilder("case2_graph");
  auto sub_data2 = sub_builder1.AddNode("sub_data2", DATA, 1, 1);
  auto rank_node = sub_builder1.AddNode("rank", "RANK", 1, 1);
  auto sub_out2 = sub_builder1.AddNode("sub_output2", NETOUTPUT, 1, 1);
  sub_builder1.AddDataEdge(sub_data2, 0, rank_node, 0);
  sub_builder1.AddDataEdge(rank_node, 0, sub_out2, 0);
  AttrUtils::SetInt(sub_data2->GetOpDescBarePtr(), ATTR_NAME_PARENT_NODE_INDEX, 1);
  auto sub_graph2 = sub_builder2.GetGraph();

  // add subgraph to case_node
  sub_graph1->SetParentGraph(main_graph.get());
  sub_graph1->SetParentNode(case_node);
  sub_graph2->SetParentGraph(main_graph.get());
  sub_graph2->SetParentNode(case_node);
  auto g_name1 = sub_graph1->GetName();
  case_node->GetOpDescBarePtr()->AddSubgraphName(g_name1);
  case_node->GetOpDescBarePtr()->SetSubgraphInstanceName(0, g_name1);
  main_graph->AddSubGraph(sub_graph1, g_name1);
  auto g_name2 = sub_graph2->GetName();
  case_node->GetOpDescBarePtr()->AddSubgraphName(g_name2);
  case_node->GetOpDescBarePtr()->SetSubgraphInstanceName(1, g_name2);
  main_graph->AddSubGraph(sub_graph2, g_name2);

  return main_graph;
}

std::shared_ptr<ExecuteGraph> BuildSimpleLineGraph(const std::string &graph_name, const int node_num = 3,
                                                   const int io_num = 1) {
  auto exe_graph = std::make_shared<ge::ExecuteGraph>(graph_name);
  std::vector<FastNode *> node(node_num, nullptr);
  for (int i = 0; i < node_num; ++i) {
    std::string op_name = "op_" + std::to_string(i);
    std::string op_type = "op_type_" + std::to_string(i);
    auto op_desc = std::make_shared<OpDesc>(op_name, op_type);
    auto td = GeTensorDesc();
    for (int j = 0; j < io_num; ++j) {
      op_desc->AddInputDesc(td);
      op_desc->AddOutputDesc(td);
    }
    node[i] = exe_graph->AddNode(op_desc);
  }

  std::vector<FastEdge *> edge(node_num, nullptr);
  for (int i = 1; i < node_num; ++i) {
    edge[i] = exe_graph->AddEdge(node[i - 1], 0, node[i], 0);
  }
  return exe_graph;
}
}  // namespace

class UtestFastNodeUtils : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(UtestFastNodeUtils, GetConstOpType_CONST) {
  ut::ExecuteGraphBuilder builder = ut::ExecuteGraphBuilder("graph");
  auto data = builder.AddNode("const1", CONSTANT, 0, 1);
  std::cout << data->GetType() << std::endl;
  auto ret = FastNodeUtils::GetConstOpType(data);
  EXPECT_EQ(ret, true);
  // case: null input
  ret = FastNodeUtils::GetConstOpType(nullptr);
  EXPECT_EQ(ret, false);
}

TEST_F(UtestFastNodeUtils, GetConstOpType_DATA) {
  ut::ExecuteGraphBuilder builder = ut::ExecuteGraphBuilder("graph");
  auto data = builder.AddNode("Data", "Data", 0, 1);
  std::cout << data->GetType() << std::endl;
  std::string op_type;
  auto ret = FastNodeUtils::GetConstOpType(data);
  ASSERT_EQ(ret, false);
}

TEST_F(UtestFastNodeUtils, GetConstOpType) {
  ut::ExecuteGraphBuilder builder = ut::ExecuteGraphBuilder("graph");
  auto data = builder.AddNode("nt", NETOUTPUT, 0, 1);
  EXPECT_EQ(FastNodeUtils::GetConstOpType(data), false);
}

TEST_F(UtestFastNodeUtils, GetParentInput_invalid) {
  auto builder = ut::ExecuteGraphBuilder("test_graph0");
  const auto &data_node = builder.AddNode("data", DATA, 0, 0);
  auto graph = builder.GetGraph();
  AttrUtils::SetInt(data_node->GetOpDescPtr(), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);
  EXPECT_EQ(FastNodeUtils::GetParentInput(data_node), nullptr);
}

TEST_F(UtestFastNodeUtils, GetParentInput) {
  const size_t node_num = 10;
  const size_t io_num = 10;
  const size_t subgraph_num = 1;
  const size_t subgraph_node_num = 10;
  std::shared_ptr<OpDesc> op_desc[node_num] = {nullptr};
  for (size_t j = 0; j < node_num; j++) {
    if (j == 1) {
      op_desc[j] = std::make_shared<OpDesc>("op", DATA);
    } else {
      op_desc[j] = std::make_shared<OpDesc>("op", "op");
    }

    auto td = GeTensorDesc();

    for (size_t i = 0; i < io_num; ++i) {
      op_desc[j]->AddInputDesc(td);
    }
    for (size_t i = 0; i < io_num; ++i) {
      op_desc[j]->AddOutputDesc(td);
    }
  }

  std::shared_ptr<ExecuteGraph> sub_graph[subgraph_num] = {nullptr};
  FastNode *node[node_num] = {};
  FastEdge *edge[node_num] = {};
  ExecuteGraph *quick_graph[subgraph_num] = {nullptr};

  auto root_graph = std::make_shared<ExecuteGraph>("root_graph");
  for (size_t i = 0; i < node_num; i++) {
    node[i] = root_graph->AddNode(op_desc[i]);
    ASSERT_NE(node[i], nullptr);
  }

  for (size_t i = 1; i < node_num; i++) {
    edge[i] = root_graph->AddEdge(node[i], 1, node[i - 1], 0);
    ASSERT_NE(edge[i], nullptr);
  }

  FastNode *sub_graph_node[subgraph_node_num] = {};
  std::string name = "subgraph_" + std::to_string(0);
  sub_graph[0] = std::make_shared<ExecuteGraph>(name);

  for (size_t j = 0; j < subgraph_node_num; j++) {
    sub_graph_node[j] = sub_graph[0]->AddNode(op_desc[j]);
    AttrUtils::SetInt(sub_graph_node[j]->GetOpDescPtr(), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
    ASSERT_NE(sub_graph_node[j], nullptr);
  }

  sub_graph[0]->SetParentGraph(root_graph.get());
  sub_graph[0]->SetParentNode(node[0]);
  quick_graph[0] = root_graph->AddSubGraph(sub_graph[0], name);
  ASSERT_NE(quick_graph[0], nullptr);

  EXPECT_EQ(FastNodeUtils::GetParentInput(sub_graph_node[1]), nullptr);
}

TEST_F(UtestFastNodeUtils, GetInDataNodeByIndex_Ok_NodeWithTwoInputNodes) {
  auto exe_graph = BuildSimpleCaseGraph();
  const auto nodes = exe_graph->GetAllNodes();
  auto case_node = nodes[2];
  // case1: find second input of case_node
  auto expect_node = nodes[1];
  auto ret_node = FastNodeUtils::GetInDataNodeByIndex(case_node, 1);
  EXPECT_EQ(ret_node, expect_node);

  // case2: empty
  ret_node = FastNodeUtils::GetInDataNodeByIndex(case_node, 2);
  EXPECT_EQ(ret_node, nullptr);

  // case3: null input
  ret_node = FastNodeUtils::GetInDataNodeByIndex(nullptr, 2);
  EXPECT_EQ(ret_node, nullptr);
}

TEST_F(UtestFastNodeUtils, AddSubgraph_Ok_MultiScenarios) {
  auto exe_graph = BuildSimpleLineGraph("main_graph", 3, 1);
  auto sug_graph1 = BuildSimpleLineGraph("sub_graph1", 2, 1);
  auto nodes = exe_graph->GetDirectNode();
  ASSERT_EQ(nodes.size(), 3);
  ASSERT_NE(nodes[1], nullptr);
  // case1: add subgraph once
  auto ret = FastNodeUtils::AppendSubgraphToNode(nodes[1], "sub_g1", sug_graph1);
  EXPECT_EQ(ret, GRAPH_SUCCESS);

  // case2: add graph with same name
  ret = FastNodeUtils::AppendSubgraphToNode(nodes[1], "sub_g1", sug_graph1);
  EXPECT_NE(ret, GRAPH_SUCCESS);

  // case3: null input
  ret = FastNodeUtils::AppendSubgraphToNode(nullptr, "sub_g1", sug_graph1);
  EXPECT_EQ(ret, PARAM_INVALID);
  ret = FastNodeUtils::AppendSubgraphToNode(nodes[1], "sub_g1", nullptr);
  EXPECT_EQ(ret, PARAM_INVALID);
}

TEST_F(UtestFastNodeUtils, GetSubgraph_Ok_GetCaseBranchGraph) {
  auto exe_graph = BuildSimpleCaseGraph();
  auto case_node = exe_graph->GetDirectNode()[2];
  auto sub_graph1 = FastNodeUtils::GetSubgraphFromNode(case_node, 0);
  ASSERT_NE(sub_graph1, nullptr);
  EXPECT_EQ(sub_graph1->GetName(), "case1_graph");
  EXPECT_EQ(FastNodeUtils::GetSubgraphFromNode(nullptr, 0), nullptr);
}

TEST_F(UtestFastNodeUtils, SetSubgraph_Ok_MultiScenarios) {
  auto main_graph = BuildSimpleLineGraph("main_graph", 1, 1);
  auto sub_graph = BuildSimpleLineGraph("sub_graph", 2, 1);
  auto par_node = main_graph->GetDirectNode()[0];
  // case1: subgraph is nullptr
  auto ret = FastNodeUtils::MountSubgraphToNode(par_node, 0, nullptr);
  EXPECT_EQ(ret, PARAM_INVALID);

  // case2: subgraph instance does not exist
  ret = FastNodeUtils::MountSubgraphToNode(par_node, 0, sub_graph);
  EXPECT_NE(ret, GRAPH_SUCCESS);

  // case3: add a subgraph name to op_desc, and then set subgraph
  EXPECT_EQ(par_node->GetOpDescBarePtr()->AddSubgraphName(sub_graph->GetName()), GRAPH_SUCCESS);
  ret = FastNodeUtils::MountSubgraphToNode(par_node, 0, sub_graph);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(main_graph->GetAllNodes().size(), 3);
}

TEST_F(UtestFastNodeUtils, SetSubgraph_Fail_NullRootGraph) {
  auto sub_graph = BuildSimpleLineGraph("sub_graph", 2, 1);
  auto par_node = std::make_shared<FastNode>();
  auto op_desc = std::make_shared<OpDesc>("op", DATA);
  EXPECT_EQ(par_node->Init(op_desc), GRAPH_SUCCESS);
  // case4: parent node's op has no OwnerGraph (null root graph)
  auto ret = FastNodeUtils::MountSubgraphToNode(par_node.get(), 0, sub_graph);
  EXPECT_NE(ret, GRAPH_SUCCESS);
}

TEST_F(UtestFastNodeUtils, AppendInputEdgeInfo_Ok) {
  // case1: null input
  EXPECT_EQ(FastNodeUtils::AppendInputEdgeInfo(nullptr, 0), PARAM_INVALID);

  // case2: append extra input edge info
  ut::ExecuteGraphBuilder builder = ut::ExecuteGraphBuilder("graph");
  auto data = builder.AddNode("Data", "Data", 2, 1);
  EXPECT_EQ(data->GetInEdgeSize(), 0);
  EXPECT_EQ(data->GetDataInNum(), 2);
  EXPECT_EQ(FastNodeUtils::AppendInputEdgeInfo(data, 11), GRAPH_SUCCESS);
  EXPECT_EQ(data->GetInEdgeSize(), 0);
  EXPECT_EQ(data->GetDataInNum(), 11);
}

TEST_F(UtestFastNodeUtils, AppendOutputEdgeInfo_Ok) {
  // case1: null input
  EXPECT_EQ(FastNodeUtils::AppendOutputEdgeInfo(nullptr, 0), PARAM_INVALID);

  // case2: append extra output edge info
  ut::ExecuteGraphBuilder builder = ut::ExecuteGraphBuilder("graph");
  auto data = builder.AddNode("Data", DATA, 2, 1);
  auto net_out = builder.AddNode("out", NETOUTPUT, 1, 1);
  builder.AddDataEdge(data, 0, net_out, 0);
  EXPECT_EQ(data->GetOutEdgesSizeByIndex(0), 1);
  EXPECT_EQ(data->GetDataInNum(), 2);
  EXPECT_EQ(FastNodeUtils::AppendOutputEdgeInfo(data, 11), GRAPH_SUCCESS);
  EXPECT_EQ(data->GetOutEdgesSizeByIndex(0), 1);
  EXPECT_EQ(data->GetDataOutNum(), 11);
}

TEST_F(UtestFastNodeUtils, ClearInputDesc_Fail_InvalidInput) {
  EXPECT_FALSE(FastNodeUtils::ClearInputDesc(nullptr, 0));
  auto op_desc = std::make_shared<OpDesc>();
  EXPECT_FALSE(FastNodeUtils::ClearInputDesc(op_desc.get(), 3));
}

TEST_F(UtestFastNodeUtils, RemoveInputEdgeInfo_Ok) {
  // case1: null input
  EXPECT_EQ(FastNodeUtils::RemoveInputEdgeInfo(nullptr, 0), PARAM_INVALID);

  // case2: remove input edge info until size is num
  ut::ExecuteGraphBuilder builder = ut::ExecuteGraphBuilder("graph");
  auto data = builder.AddNode("Data", "Data", 1, 1);
  EXPECT_EQ(data->GetOpDescBarePtr()->GetInputsSize(), 1);
  EXPECT_EQ(data->GetOpDescBarePtr()->AddInputDesc(GeTensorDesc()), GRAPH_SUCCESS);
  EXPECT_EQ(data->GetOpDescBarePtr()->GetInputsSize(), 2);
  EXPECT_EQ(FastNodeUtils::RemoveInputEdgeInfo(data, 0), GRAPH_SUCCESS);
  EXPECT_EQ(data->GetDataInNum(), 0);
}

TEST_F(UtestFastNodeUtils, UnlinkAll_Ok_UnlinkAllAndCheckEdgeNum) {
  // case1: null input, no return value
  FastNodeUtils::UnlinkAll(nullptr);

  // case2: remove all the edges connecting to the second node, 2 data egde and 1 control edge
  auto exe_graph = BuildSimpleLineGraph("simple_graph", 3, 1);
  auto node1 = exe_graph->GetDirectNode()[0];
  auto node2 = exe_graph->GetDirectNode()[1];
  (void) exe_graph->AddEdge(node1, kControlEdgeIndex, node2, kControlEdgeIndex);
  EXPECT_EQ(exe_graph->GetAllEdges().size(), 3);
  FastNodeUtils::UnlinkAll(node2);
  EXPECT_EQ(exe_graph->GetAllEdges().size(), 0);
}

TEST_F(UtestFastNodeUtils, GetInEndpoint_Ok) {
  auto exe_graph = BuildSimpleLineGraph("simple_graph", 2, 1);
  auto in_node = exe_graph->GetDirectNode()[1];
  auto edge = exe_graph->GetAllEdges().front();
  EXPECT_EQ(FastNodeUtils::GetDstEndpoint(edge).node, in_node);
  EXPECT_EQ(FastNodeUtils::GetDstEndpoint(edge).index, 0);
}

TEST_F(UtestFastNodeUtils, GetOutEndpoint_Ok) {
  auto exe_graph = BuildSimpleLineGraph("simple_graph", 2, 1);
  auto out_node = exe_graph->GetDirectNode()[0];
  auto edge = exe_graph->GetAllEdges().front();
  EXPECT_EQ(FastNodeUtils::GetSrcEndpoint(edge).node, out_node);
  EXPECT_EQ(FastNodeUtils::GetSrcEndpoint(edge).index, 0);
}
}  // namespace ge
