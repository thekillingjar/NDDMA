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
#include "common/mem_conflict_share_graph.h"
#include "result_checker.h"
#include "graph/compute_graph.h"
#include "compiler/graph/optimize/mem_layout_conflict_optimize/mem_layout_conflict_optimizer.h"
#include "graph/passes/graph_builder_utils.h"

namespace ge {
class UtestMemLayoutConflictWhileBody : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

namespace {
void AddSubgraphInstance(ge::ComputeGraphPtr parent_graph, ge::ComputeGraphPtr sub_graph,
                         const int parent_node_index, const std::string &parent_node_name) {
  auto sub_graph_name = sub_graph->GetName();
  sub_graph->SetParentGraph(parent_graph);
  sub_graph->SetParentNode(parent_graph->FindNode(parent_node_name));
  parent_graph->FindNode(parent_node_name)->GetOpDesc()->AddSubgraphName(sub_graph_name);
  parent_graph->FindNode(parent_node_name)->GetOpDesc()->SetSubgraphInstanceName(parent_node_index, sub_graph_name);
  parent_graph->AddSubgraph(sub_graph_name, sub_graph);
}

void AddWhileDummyCond(ge::ComputeGraphPtr parent_graph, const std::string &parent_node_name) {
  ge::ut::GraphBuilder builder("cond");
  auto data = builder.AddNode("cond_data", "Data", 1, 1);
  auto netoutput = builder.AddNode("cond_netoutput", "NetOutput", 1, 1);

  ge::AttrUtils::SetInt(data->GetOpDesc(), "_parent_node_index", static_cast<int>(0));
  ge::AttrUtils::SetInt(netoutput->GetOpDesc()->MutableInputDesc(0), "_parent_node_index", static_cast<int>(0));

  builder.AddDataEdge(data, 0, netoutput, 0);
  auto sub_graph = builder.GetGraph();

  AddSubgraphInstance(parent_graph, sub_graph, 0, parent_node_name);
}

void AddWhileBodyWithNNode(ge::ComputeGraphPtr parent_graph, const std::string &parent_node_name,
                           const std::string &body_name, int n) {
  ge::ut::GraphBuilder builder(body_name);
  auto data = builder.AddNode(body_name + "_data", "Data", 1, 1);
  auto netoutput = builder.AddNode(body_name + "_netoutput", "NetOutput", 1, 1);

  ge::AttrUtils::SetInt(data->GetOpDesc(), "_parent_node_index", static_cast<int>(0));
  ge::AttrUtils::SetInt(netoutput->GetOpDesc()->MutableInputDesc(0), "_parent_node_index", static_cast<int>(0));

  int cnt = 1;
  auto last_node = data;

  while (cnt < n) {
    auto cast = builder.AddNode(body_name + "_cast" + std::to_string(cnt - 1), "Cast", 1, 1);
    builder.AddDataEdge(last_node, 0, cast, 0);
    last_node = cast;
    cnt++;
  }

  builder.AddDataEdge(last_node, 0, netoutput, 0);
  auto sub_graph = builder.GetGraph();

  AddSubgraphInstance(parent_graph, sub_graph, 1, parent_node_name);
}

}

/*
 *        data
 *          |
 *        while1
 *          |
 *         op
 *          |
 *       netoutput0
 *
 * subgraph cond             subgraph body
 * +-------------------+     +-------------------+
 * | data              |     | data--netoutput   |
 * +-------------------+     +-------------------+
 *
 */
TEST_F(UtestMemLayoutConflictWhileBody, WhileBodyDataToNetoutputWithNoExchange_NotInsertIdentityN_Success) {
  auto builder = ut::GraphBuilder("test");
  auto data = builder.AddNode("data", DATA, 0, 1);
  auto while1 = builder.AddNode("while1", WHILE, 1, 1);
  auto op = builder.AddNode("op", ADD, 1, 1);
  auto net_output = builder.AddNode("net_output", NETOUTPUT, 1, 0);
  builder.AddDataEdge(data, 0, while1, 0);
  builder.AddDataEdge(while1, 0, op, 0);
  builder.AddDataEdge(op, 0, net_output, 0);
  ge::ComputeGraphPtr graph = builder.GetGraph();

  AddWhileDummyCond(graph, "while1");
  AddWhileBodyWithNNode(graph, "while1", "body", 1);

  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);

  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 0U), GRAPH_SUCCESS);
}

/*
 *        data
 *          |
 *        while1
 *          |
 *         op
 *          |
 *       netoutput0
 *
 * subgraph cond             subgraph body
 * +-------------------+     +-----------------------+
 * | data              |     | data--cast--netoutput |
 * +-------------------+     +-----------------------+
 *
 */
TEST_F(UtestMemLayoutConflictWhileBody, WhileBodyOneNodeInside_InsertIdentityN_Success) {
  auto builder = ut::GraphBuilder("test");
  auto data = builder.AddNode("data", DATA, 0, 1);
  auto while1 = builder.AddNode("while1", WHILE, 1, 1);
  auto op = builder.AddNode("op", ADD, 1, 1);
  auto net_output = builder.AddNode("net_output", NETOUTPUT, 1, 0);
  builder.AddDataEdge(data, 0, while1, 0);
  builder.AddDataEdge(while1, 0, op, 0);
  builder.AddDataEdge(op, 0, net_output, 0);
  ge::ComputeGraphPtr graph = builder.GetGraph();

  AddWhileDummyCond(graph, "while1");
  AddWhileBodyWithNNode(graph, "while1", "body", 2);

  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);

  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 1U), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"body_netoutput"}}), GRAPH_SUCCESS);

  MemLayoutConflictOptimizer redundant_pass;
  ASSERT_EQ(redundant_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 1U), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"body_netoutput"}}), GRAPH_SUCCESS);
}

/*
 *        data
 *          |
 *        while1
 *          |
 *         op
 *          |
 *       netoutput0
 *
 * subgraph cond             subgraph body
 * +-------------------+     +-----------------------+
 * | data              |     | data--cast--netoutput |
 * +-------------------+     +-----------------------+
 *
 */
TEST_F(UtestMemLayoutConflictWhileBody, WhileBodyOneNodeInside_InsertIdentityN_CopyAttr_Success) {
  auto builder = ut::GraphBuilder("test");
  auto data = builder.AddNode("data", DATA, 0, 1);
  auto while1 = builder.AddNode("while1", WHILE, 1, 1);
  auto op = builder.AddNode("op", ADD, 1, 1);
  auto net_output = builder.AddNode("net_output", NETOUTPUT, 1, 0);
  builder.AddDataEdge(data, 0, while1, 0);
  builder.AddDataEdge(while1, 0, op, 0);
  builder.AddDataEdge(op, 0, net_output, 0);
  ge::ComputeGraphPtr graph = builder.GetGraph();

  AddWhileDummyCond(graph, "while1");
  AddWhileBodyWithNNode(graph, "while1", "body", 2);

  NodePtr cast_node = nullptr;
  for (const auto &node : graph->GetAllNodes()) {
    if (node->GetType() == CAST) {
      cast_node = node;
    }
  }

  AttrUtils::SetStr(cast_node->GetOpDesc(), ATTR_NAME_BATCH_LABEL, "batch_label");

  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);

  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 1U), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"body_netoutput"}}), GRAPH_SUCCESS);

  NodePtr identityn_node = nullptr;
  for (const auto &node : graph->GetAllNodes()) {
    if (node->GetType() == IDENTITY) {
      identityn_node = node;
    }
  }
  std::string batch_label;
  EXPECT_EQ(AttrUtils::GetStr(identityn_node->GetOpDesc(), ATTR_NAME_BATCH_LABEL, batch_label), true);
  EXPECT_EQ(batch_label, "batch_label"); 
}

void AddWhileBodyTwoInputSwap(ge::ComputeGraphPtr parent_graph, const std::string &parent_node_name,
                              const std::string &body_name) {
  ge::ut::GraphBuilder builder(body_name);
  auto data1 = builder.AddNode(body_name + "_data1", "Data", 1, 1);
  auto data2 = builder.AddNode(body_name + "_data2", "Data", 1, 1);
  auto netoutput = builder.AddNode(body_name + "_netoutput", "NetOutput", 2, 1);

  ge::AttrUtils::SetInt(data1->GetOpDesc(), "_parent_node_index", static_cast<int>(0));
  ge::AttrUtils::SetInt(data2->GetOpDesc(), "_parent_node_index", static_cast<int>(1));
  ge::AttrUtils::SetInt(netoutput->GetOpDesc()->MutableInputDesc(0), "_parent_node_index", static_cast<int>(0));
  ge::AttrUtils::SetInt(netoutput->GetOpDesc()->MutableInputDesc(1), "_parent_node_index", static_cast<int>(1));

  builder.AddDataEdge(data1, 0, netoutput, 1);
  builder.AddDataEdge(data2, 0, netoutput, 0);
  auto sub_graph = builder.GetGraph();

  AddSubgraphInstance(parent_graph, sub_graph, 1, parent_node_name);
}

/*
 *        data
 *          |
 *        cast
 *          |
 *        while1
 *          |
 *         op
 *          |
 *       netoutput0
 *
 * subgraph cond             subgraph body
 * +-------------------+     +--------------------------+
 * | data              |     | data--refnode--netoutput |
 * +-------------------+     +--------------------------+
 *
 */
TEST_F(UtestMemLayoutConflictWhileBody, WhileBodyOneRefInside_NotInsertIdentityN_Success) {
  auto graph = MemConflictShareGraph::BuildWhileDataRefNetoutputGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 0U), GRAPH_SUCCESS);
}

/*
 *        input
 *          |
 *        cast0
 *          |
 *        while1
 *          |
 *        cast1
 *          |
 *       netoutput
 *
 * subgraph cond       subgraph body
 * +-------------+     +------------------------------------+
 * |   data0     |     |     data1                          |
 * |     |       |     |       |                            |
 * | netoutput0  |     |  partitioned_call   +------------+ |
 * +-------------+     |       |             |   data2    | |
 *                     |   netoutput1        |     |      | |
 *                     |                     | netoutput2 | |
 *                     |                     +------------+ |
 *                     +------------------------------------+                      
 *
 */
TEST_F(UtestMemLayoutConflictWhileBody, WhileBodyOnePartitionedCallInside_NotInsertIdentityN_Success) {
  auto graph = MemConflictShareGraph::BuildWhileDataPartitionedCallNetoutputGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 0U), GRAPH_SUCCESS);
}

/*
 *    data1  data2
 *       \    /
 *        while
 *          |
 *         op
 *          |
 *       netoutput0
 *
 * subgraph cond             subgraph body
 * +-------------------+     +-------------------------+
 * | data3             |     | data4--\ /---netoutput1 |
 * +-------------------+     | data5--/ \--+           |  
 *                           +-------------------------+
 *
 */
TEST_F(UtestMemLayoutConflictWhileBody, WhileBodyTwoDataToNetoutputWithExchange_InsertIdentityN_Success) {
  auto builder = ut::GraphBuilder("test");
  auto data1 = builder.AddNode("data1", DATA, 0, 1);
  auto data2 = builder.AddNode("data2", DATA, 0, 1);
  auto while1 = builder.AddNode("while1", WHILE, 2, 2);
  auto op = builder.AddNode("op", ADD, 2, 1);
  auto net_output = builder.AddNode("net_output", NETOUTPUT, 1, 0);
  builder.AddDataEdge(data1, 0, while1, 0);
  builder.AddDataEdge(data2, 0, while1, 1);
  builder.AddDataEdge(while1, 0, op, 0);
  builder.AddDataEdge(while1, 1, op, 1);
  builder.AddDataEdge(op, 0, net_output, 0);

  ge::ComputeGraphPtr graph = builder.GetGraph();

  AddWhileDummyCond(graph, "while1");
  AddWhileBodyTwoInputSwap(graph, "while1", "body");

  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 2U), GRAPH_SUCCESS);

  MemLayoutConflictOptimizer redundant_pass;
  ASSERT_EQ(redundant_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 2U), GRAPH_SUCCESS);
}

/*
 *    input1 input2
 *       |      |
 *     cast1  cast2
 *        \    /
 *        while1
 *         \ /
 *         add
 *          |
 *       netoutput
 *
 * subgraph cond             subgraph body
 * +-------------------+     +----------------------+
 * | data0--netoutput0 |     | data1-----netoutput1 |
 * +-------------------+     | data2----+           |
 *                           +----------------------+
 */
TEST_F(UtestMemLayoutConflictWhileBody, WhileBodyTwoDataToNetoutputWithNoExchange_NotInsert_Success) {
  auto graph = MemConflictShareGraph::BuildWhileTwoDataToNetoutputNoExchangeGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 0U), GRAPH_SUCCESS);
}

}
