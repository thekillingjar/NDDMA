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
#include "graph_builder_utils.h"
#include "fast_graph/fast_graph_impl.h"

namespace ge {
class UtestFastGraphImpl : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(UtestFastGraphImpl, testnodes) {
  auto root_graph = std::make_shared<ExecuteGraph>("root_graph");
  auto compute_graph = std::make_shared<FastGraphImpl<FastNode, ExecuteGraph>>("Hello World.");
  int node_num = 10;
  int edge_num = 5;
  FastNode *node[node_num] = {};
  for (int i = 0; i < node_num; ++i) {
    auto op_desc = std::make_shared<OpDesc>("op", "op");
    auto td = GeTensorDesc();
    for (int j = 0; j < edge_num; ++j) {
      op_desc->AddInputDesc(td);
      op_desc->AddOutputDesc(td);
    }
    node[i] = compute_graph->AddNode(op_desc);
    ASSERT_NE(node[i], nullptr);
  }

  ASSERT_EQ(compute_graph->GetAllNodeInfo().size(), node_num);
  ASSERT_EQ(compute_graph->GetAllNodeInfoForModify().size(), node_num);
  ASSERT_EQ(compute_graph->GetRawDirectNode().size(), node_num);
  ASSERT_EQ(compute_graph->GetRawDirectNode().size(), node_num);
}

TEST_F(UtestFastGraphImpl, SetNodes) {
  auto root_graph = std::make_shared<ExecuteGraph>("root_graph");
  auto compute_graph = std::make_shared<FastGraphImpl<FastNode, ExecuteGraph>>("Hello World.");
  int node_num = 10;
  int edge_num = 5;
  FastNode *node[node_num] = {};
  for (int i = 0; i < node_num; ++i) {
    auto op_desc = std::make_shared<OpDesc>("op", "op");
    auto td = GeTensorDesc();
    for (int j = 0; j < edge_num; ++j) {
      op_desc->AddInputDesc(td);
      op_desc->AddOutputDesc(td);
    }
    node[i] = compute_graph->AddNode(op_desc);
    auto quick_node = FastGraphUtils::GetListElementAddr(node[i]);
    compute_graph->RemoveJustNode(quick_node);
    ASSERT_NE(node[i], nullptr);
  }

  std::vector<FastNode *> nodes;
  for (int i = 0; i < node_num; ++i) {
    auto quick_node = FastGraphUtils::GetListElementAddr(node[i]);
    FastGraphUtils::GetOwner(quick_node)->erase(quick_node);
    nodes.push_back(node[i]);
  }
  auto ret = compute_graph->SetNodes(nodes);
  ASSERT_EQ(ret, GRAPH_SUCCESS);

  nodes.push_back(nullptr);
  ret = compute_graph->SetNodes(nodes);
  ASSERT_EQ(ret, PARAM_INVALID);
}

TEST_F(UtestFastGraphImpl, FastGraphImpl) {
  auto compute_graph = std::make_shared<FastGraphImpl<FastNode, ExecuteGraph>>("Hello World.");
  auto root_graph = std::make_shared<ge::ExecuteGraph>("graph");
  auto sub_graph = std::make_shared<ge::ExecuteGraph>("graph");
  ExecuteGraph *quick_graph = root_graph->AddSubGraph(sub_graph);
  ASSERT_NE(quick_graph, nullptr);

  auto ret = root_graph->RemoveSubGraph(quick_graph);
  ASSERT_EQ(ret, GRAPH_SUCCESS);

  std::vector<ExecuteGraph *> sub_graphs{quick_graph};
  compute_graph->SetSubGraph(sub_graphs);
  auto size = compute_graph->GetAllSubGraphSize();
  ASSERT_EQ(size, 1);
}

TEST_F(UtestFastGraphImpl, TestIO) {
  auto root_graph = std::make_shared<ge::ExecuteGraph>("graph");
  auto compute_graph = std::make_shared<FastGraphImpl<FastNode, ExecuteGraph>>("Hello World.");
  int node_num = 10;
  int edge_num = 5;
  FastNode *node[node_num] = {};
  std::vector<std::pair<FastNode *, int32_t>> out_nodes_info;
  for (int i = 0; i < node_num; ++i) {
    auto op_desc = std::make_shared<OpDesc>("op", "op");
    auto td = GeTensorDesc();
    for (int j = 0; j < edge_num; ++j) {
      op_desc->AddInputDesc(td);
      op_desc->AddOutputDesc(td);
    }
    node[i] = compute_graph->AddNode(op_desc);
    ASSERT_NE(node[i], nullptr);
  }
  auto input = compute_graph->AddInputNode(node[0]);
  ASSERT_NE(input, nullptr);
  auto output = compute_graph->AddOutputNodeByIndex(node[node_num - 1], 0);
  ASSERT_NE(output, nullptr);

  auto inputs = compute_graph->GetAllInputNodeInfo();
  ASSERT_EQ(inputs.size(), 1);
  ASSERT_EQ(inputs[0], input);

  auto outputs = compute_graph->GetAllOutNodeInfo();
  ASSERT_EQ(outputs.size(), 1);
  ASSERT_EQ(outputs[0].first, output);

  out_nodes_info.push_back(std::make_pair(node[node_num - 2], 0));
  compute_graph->SetGraphOutNodesInfo(out_nodes_info);

  output = compute_graph->AddOutputNodeByIndex(node[node_num - 1], 0);
  ASSERT_NE(output, nullptr);

  outputs = compute_graph->GetAllOutNodeInfo();
  ASSERT_EQ(outputs.size(), 2);
  ASSERT_EQ(outputs[0].first, node[node_num - 2]);

  inputs = compute_graph->GetInputNodes();
  ASSERT_EQ(inputs.size(), 1);

  outputs = compute_graph->GetAllOutNodes();
  ASSERT_EQ(outputs.size(), 2);

  auto size = compute_graph->GetDirectNodesSize();
  ASSERT_EQ(size, node_num);

  bool flag = compute_graph->CheckNodeIsInGraph(node[0]);
  ASSERT_EQ(flag, true);

  compute_graph->InValid();
  flag = compute_graph->IsValid();
  ASSERT_EQ(flag, false);

  auto ret = compute_graph->ClearNode([](QuickNode *quick_node) { return GRAPH_SUCCESS; });
  ASSERT_EQ(ret, GRAPH_SUCCESS);

  auto compute_graph1 = new FastGraphImpl<FastNode, ExecuteGraph>("Hello World 1.");
  compute_graph->Swap(*compute_graph1);
  ASSERT_EQ(compute_graph->GetName(), "Hello World 1.");

  delete compute_graph1;
}

TEST_F(UtestFastGraphImpl, Invalid) {
  auto root_graph = std::make_shared<ExecuteGraph>("root_graph");
  auto compute_graph = std::make_shared<FastGraphImpl<FastNode, ExecuteGraph>>("Hello World.");
  auto ret = compute_graph->RemoveInputNode(nullptr);
  ASSERT_EQ(ret, GRAPH_FAILED);

  auto pointer = compute_graph->AddOutputNodeByIndex(nullptr, 0);
  ASSERT_EQ(pointer, nullptr);

  ret = compute_graph->RemoveOutputNode(nullptr);
  ASSERT_EQ(ret, GRAPH_FAILED);

  pointer = compute_graph->AddNode(nullptr);
  ASSERT_EQ(pointer, nullptr);

  pointer = compute_graph->AddNodeFront(nullptr);
  ASSERT_EQ(pointer, nullptr);

  ret = compute_graph->RemoveJustNode(nullptr);
  ASSERT_EQ(ret, GRAPH_FAILED);

  ret = compute_graph->RecycleQuickNode(nullptr);
  ASSERT_EQ(ret, GRAPH_FAILED);

  ret = compute_graph->RecycleQuickEdge(nullptr);
  ASSERT_EQ(ret, GRAPH_FAILED);

  auto edge = compute_graph->AddEdge(nullptr, 0, nullptr, 1);
  ASSERT_EQ(edge, nullptr);

  auto node1 = new FastNode();
  auto node2 = new FastNode();
  edge = compute_graph->AddEdge(node1, -1, node2, 1);
  delete node1;
  delete node2;
  ASSERT_EQ(edge, nullptr);

  ret = compute_graph->RemoveEdge(nullptr);
  ASSERT_EQ(ret, GRAPH_FAILED);
}
}  // namespace ge
