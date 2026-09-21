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

#include "jit_execution/utils/partitioner/binary_partitioner.h"
#include "graph/passes/graph_builder_utils.h"
#include "graph/debug/ge_op_types.h"
#include "graph/utils/graph_utils.h"

using namespace testing;
using namespace ge;

class BinaryPartitionerUT : public testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

bool CheckNodesIsExist(const ComputeGraphPtr &graph, const std::vector<NodePtr> &nodes) {
  for (const auto &node : nodes) {
    auto it = graph->FindNode(node->GetName());
    if (it == nullptr) {
      return false;
    }
  }
  return true;
}

/*
 *             data1  const1
 *                \    /
 *                 add1  const2
 *                   \     /
 *          data2      mul     var
 *             \     /     \    /
 *              gather   reshape
 *                   \     /
 *                    add2
 *                      |
 *                  netoutput
 */
void BuildDynamicGraphAndNodes(ComputeGraphPtr &graph, std::vector<NodePtr> &infered_nodes, std::vector<NodePtr> &uninfer_nodes) {
  auto builder = ut::GraphBuilder("Origin_Graph");
  const auto &data1 = builder.AddNode("data1", DATA, 1, 1);
  const auto &const1 = builder.AddNode("const1", CONSTANTOP, 0, 1);
  const auto &add1 = builder.AddNode("add1", "Add", 2, 1);
  const auto &const2 = builder.AddNode("const2", CONSTANTOP, 0, 1);
  const auto &data2 = builder.AddNode("data2", DATA, 1, 1);
  const auto &mul = builder.AddNode("mul", "Mul", 2, 2);
  const auto &var = builder.AddNode("var", VARIABLE, 0, 1);
  const auto &gather = builder.AddNode("gather", "Gather", 2, 1);
  const auto &reshape = builder.AddNode("reshape", RESHAPE, 2, 1);
  const auto &add2 = builder.AddNode("add2", "Add", 2, 1);
  const auto &netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 0);

  builder.AddDataEdge(data1, 0, add1, 0);
  builder.AddDataEdge(const1, 0, add1, 1);
  builder.AddDataEdge(add1, 0, mul, 0);
  builder.AddDataEdge(const2, 0, mul, 1);
  builder.AddDataEdge(data2, 0, gather, 0);
  builder.AddDataEdge(mul, 0, gather, 1);
  builder.AddDataEdge(mul, 1, reshape, 0);
  builder.AddDataEdge(var, 0, reshape, 1);
  builder.AddDataEdge(gather, 0, add2, 0);
  builder.AddDataEdge(reshape, 0, add2, 1);
  builder.AddDataEdge(add2, 0, netoutput, 0);

  graph = builder.GetGraph();

  infered_nodes.push_back(data1);
  infered_nodes.push_back(const1);
  infered_nodes.push_back(add1);
  infered_nodes.push_back(const2);
  infered_nodes.push_back(data2);
  infered_nodes.push_back(mul);
  infered_nodes.push_back(var);
  infered_nodes.push_back(gather);

  uninfer_nodes.push_back(reshape);
  uninfer_nodes.push_back(add2);
  uninfer_nodes.push_back(netoutput);
}

TEST_F(BinaryPartitionerUT, return_remaining_nullptr_when_input_all_nodes) {
  ComputeGraphPtr graph;
  std::vector<NodePtr> infered_nodes;
  std::vector<NodePtr> uninfer_nodes;
  BuildDynamicGraphAndNodes(graph, infered_nodes, uninfer_nodes);
  std::vector<NodePtr> all_nodes;
  all_nodes.insert(all_nodes.end(), infered_nodes.begin(), infered_nodes.end());
  all_nodes.insert(all_nodes.end(), uninfer_nodes.begin(), uninfer_nodes.end());
  PartionResult partition_result;
  auto ret = BinaryPartitioner::Partition(graph, all_nodes, partition_result);
  ASSERT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(partition_result.sliced_graph->GetAllNodesSize(), graph->GetAllNodesSize());
  EXPECT_EQ(partition_result.remaining_graph, nullptr);
}

TEST_F(BinaryPartitionerUT, return_failed_when_input_nodes_contain_cycle) {
  ComputeGraphPtr graph;
  std::vector<NodePtr> infered_nodes;
  std::vector<NodePtr> uninfer_nodes;
  BuildDynamicGraphAndNodes(graph, infered_nodes, uninfer_nodes);

  // build cycle
  auto add1 = graph->FindNode("add1");
  auto it = std::find(infered_nodes.begin(), infered_nodes.end(), add1);
  infered_nodes.erase(it); // remove node add1

  PartionResult partition_result;
  auto ret = BinaryPartitioner::Partition(graph, infered_nodes, partition_result);
  ASSERT_EQ(ret, GRAPH_PARAM_INVALID);
}

TEST_F(BinaryPartitionerUT, return_failed_when_input_nodes_contain_control_cycle) {
  ComputeGraphPtr graph;
  std::vector<NodePtr> infered_nodes;
  std::vector<NodePtr> uninfer_nodes;
  BuildDynamicGraphAndNodes(graph, infered_nodes, uninfer_nodes);

  // build cycle
  auto add2 = graph->FindNode("add2");
  auto const2 = graph->FindNode("const2");
  ASSERT_EQ(GraphUtils::AddEdge(add2->GetOutControlAnchor(), const2->GetInControlAnchor()), GRAPH_SUCCESS);

  PartionResult partition_result;
  auto ret = BinaryPartitioner::Partition(graph, infered_nodes, partition_result);
  ASSERT_EQ(ret, GRAPH_PARAM_INVALID);
}

TEST_F(BinaryPartitionerUT, return_sliced_graph_and_remaining_graph_when_input_infered_nodes) {
  ComputeGraphPtr graph;
  std::vector<NodePtr> infered_nodes;
  std::vector<NodePtr> uninfer_nodes;
  BuildDynamicGraphAndNodes(graph, infered_nodes, uninfer_nodes);

  PartionResult partition_result;
  auto ret = BinaryPartitioner::Partition(graph, infered_nodes, partition_result);
  ASSERT_EQ(ret, GRAPH_SUCCESS);

  EXPECT_NE(partition_result.sliced_graph, nullptr);
  EXPECT_NE(partition_result.remaining_graph, nullptr);
  EXPECT_TRUE(CheckNodesIsExist(partition_result.sliced_graph, infered_nodes));
  EXPECT_TRUE(CheckNodesIsExist(partition_result.remaining_graph, uninfer_nodes));
  EXPECT_EQ(partition_result.sliced_graph->GetAllNodesSize(), 9);
  EXPECT_EQ(partition_result.remaining_graph->GetAllNodesSize(), 6);
  EXPECT_EQ(partition_result.out_idx_2_in_idxs.size(), 2);
}

TEST_F(BinaryPartitionerUT, return_failed_when_input_nodes_empty) {
  ComputeGraphPtr graph;
  std::vector<NodePtr> infered_nodes;
  std::vector<NodePtr> uninfer_nodes;
  BuildDynamicGraphAndNodes(graph, infered_nodes, uninfer_nodes);

  // build empty
  infered_nodes.clear();

  PartionResult partition_result;
  auto ret = BinaryPartitioner::Partition(graph, infered_nodes, partition_result);
  ASSERT_EQ(ret, GRAPH_FAILED);
}