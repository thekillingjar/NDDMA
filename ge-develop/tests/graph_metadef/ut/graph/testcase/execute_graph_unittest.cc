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
#include "common/share_graph.h"
#include "graph/ge_local_context.h"
#include "graph_builder_utils.h"
#include "graph/fast_graph/fast_node.h"
#include "graph/debug/ge_op_types.h"
#include "fast_graph/fast_graph_impl.h"

namespace {
using ExecuteSharedGraph = ge::SharedGraph<ge::ExecuteGraphPtr, ge::ut::ExecuteGraphBuilder>;
std::shared_ptr<ge::ExecuteGraph> BuildDelayTopoGraph(const std::string &name) {
  auto builder = ge::ut::ExecuteGraphBuilder(name);
  const auto &variable = builder.AddNode("variable", ge::VARIABLE, 0, 2);
  const auto &node1 = builder.AddNode("node1", "node1", 1, 1);
  const auto &node2 = builder.AddNode("node2", "node2", 1, 1);
  const auto &node3 = builder.AddNode("node3", "node3", 1, 1);
  const auto &node4 = builder.AddNode("node4", "node4", 1, 1);
  const auto &node5 = builder.AddNode("node5", "node5", 3, 0);
  const auto &data = builder.AddNode("data", "DATA", 0, 1);

  builder.AddDataEdge(variable, 0, node1, 0);
  builder.AddDataEdge(variable, 1, node2, 0);
  builder.AddDataEdge(node1, 0, node5, 0);
  builder.AddDataEdge(node2, 0, node5, 1);
  builder.AddDataEdge(data, 0, node3, 0);
  builder.AddDataEdge(node3, 0, node4, 0);

  int dst_idx = 2;
  builder.AddDataEdge(node4, 0, node5, dst_idx);

  builder.AddControlEdge(node2, node3);
  return builder.GetGraph();
}
}  // namespace

namespace ge {
class UtestExecuteGraph : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(UtestExecuteGraph, ExecuteGraph) {
  auto compute_graph = std::make_shared<ge::ExecuteGraph>("graph");
  size_t graph_id = 10;
  compute_graph->SetGraphId(graph_id);
  auto ret = compute_graph->GetGraphId();
  ASSERT_EQ(ret, graph_id);
}

TEST_F(UtestExecuteGraph, AddNodeToExecuteGraph) {
  auto compute_graph = std::make_shared<ge::ExecuteGraph>("graph");
  {
    auto op_desc = std::make_shared<OpDesc>("op", "op");
    std::vector<std::string> inputs_order;
    int num = 10;
    for (int i = 0; i < num; ++i) {
      inputs_order.push_back("test" + std::to_string(i));
    }
    compute_graph->SetInputsOrder(inputs_order);
    auto td = GeTensorDesc();
    op_desc->AddInputDesc(td);
    op_desc->AddOutputDesc(td);
    auto node = compute_graph->AddNode(op_desc);
    ASSERT_NE(node, nullptr);
    auto ret = compute_graph->RemoveJustNode(node);
    ASSERT_EQ(ret, GRAPH_SUCCESS);

    node = compute_graph->AddNode(node);
    ASSERT_NE(node, nullptr);

    auto node_with_idx = compute_graph->AddNode(op_desc, 0);
    ASSERT_NE(node_with_idx, nullptr);

    ret = compute_graph->RemoveJustNode(node);
    ASSERT_EQ(ret, GRAPH_SUCCESS);
  }

  {
    auto op_desc = std::make_shared<OpDesc>("op", "op");
    auto td = GeTensorDesc();
    op_desc->AddInputDesc(td);
    op_desc->AddOutputDesc(td);
    auto node = compute_graph->AddNodeFront(op_desc);
    ASSERT_NE(node, nullptr);
    auto ret = compute_graph->RemoveJustNode(node);
    ASSERT_EQ(ret, GRAPH_SUCCESS);

    node = compute_graph->AddNodeFront(node);
    ASSERT_NE(node, nullptr);
  }
}

TEST_F(UtestExecuteGraph, RemoveJustNode) {
  auto compute_graph = std::make_shared<ge::ExecuteGraph>("graph");
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

  FastEdge *edge[edge_num] = {};
  for (int i = 0; i < edge_num; ++i) {
    edge[i] = compute_graph->AddEdge(node[0], i, node[1], i);
    ASSERT_NE(edge[i], nullptr);
  }

  for (int i = 0; i < node_num; ++i) {
    auto ret = compute_graph->RemoveJustNode(node[i]);
    ASSERT_EQ(ret, GRAPH_SUCCESS);
  }

  for (int i = 0; i < edge_num; ++i) {
    auto ret = compute_graph->RemoveEdge(edge[i]);
    ASSERT_EQ(ret, GRAPH_SUCCESS);
  }
}

TEST_F(UtestExecuteGraph, AddEdgeToExecuteGraph) {
  auto compute_graph = std::make_shared<ge::ExecuteGraph>("graph");
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

  FastEdge *edge[edge_num] = {};
  for (int i = 0; i < edge_num; ++i) {
    edge[i] = compute_graph->AddEdge(node[0], i, node[1], i);
    ASSERT_NE(edge[i], nullptr);
  }

  ASSERT_EQ(compute_graph->GetAllEdges().size(), edge_num);

  for (int i = 0; i < edge_num; ++i) {
    auto ret = compute_graph->RemoveEdge(edge[i]);
    ASSERT_EQ(ret, GRAPH_SUCCESS);
  }
}

TEST_F(UtestExecuteGraph, AddSubGraphToExecuteGraph) {
  auto root_graph = std::make_shared<ExecuteGraph>("root_graph");
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
    node[i] = root_graph->AddNode(op_desc);
    ASSERT_NE(node[i], nullptr);
  }

  int subgraph_num = 5;
  std::shared_ptr<ExecuteGraph> sub_graph[subgraph_num] = {nullptr};
  for (int i = 0; i < subgraph_num; i++) {
    sub_graph[i] = std::make_shared<ExecuteGraph>("subgraph_" + std::to_string(i));
    sub_graph[i]->SetParentGraph(root_graph.get());
    sub_graph[i]->SetParentNode(node[i]);
  }

  {
    /* Test Error */
    std::string name = "subgraph_" + std::to_string(0);
    std::shared_ptr<ExecuteGraph> invalid_sub_graph = nullptr;
    auto fast_graph = root_graph->AddSubGraph(invalid_sub_graph, name);
    ASSERT_EQ(fast_graph, nullptr);

    sub_graph[0]->SetParentGraph(nullptr);
    fast_graph = root_graph->AddSubGraph(sub_graph[0], name);
    ASSERT_EQ(fast_graph, nullptr);

    sub_graph[0]->SetParentGraph(root_graph.get());
    sub_graph[0]->SetParentNode(nullptr);
    fast_graph = root_graph->AddSubGraph(sub_graph[0], name);
    ASSERT_EQ(fast_graph, nullptr);

    sub_graph[0]->SetParentGraph(root_graph.get());
    sub_graph[0]->SetParentNode(node[0]);

    auto ok_fast_graph = root_graph->AddSubGraph(sub_graph[0], name);
    ASSERT_NE(ok_fast_graph, nullptr);

    auto bad_graph = root_graph->AddSubGraph(sub_graph[0], name);
    ASSERT_EQ(bad_graph, nullptr);

    auto ret = root_graph->RemoveSubGraph(name);
    ASSERT_EQ(ret, GRAPH_SUCCESS);

    auto root_graph2 = std::make_shared<ExecuteGraph>("root_graph2");
    sub_graph[1]->SetParentGraph(root_graph2.get());
    sub_graph[1]->SetParentNode(node[1]);
    bad_graph = root_graph->AddSubGraph(sub_graph[1], name);
    ASSERT_EQ(bad_graph, nullptr);

    sub_graph[1]->SetParentGraph(root_graph.get());
    sub_graph[1]->SetParentNode(node[1]);
  }

  {
    std::string name = "root_graph2";
    auto root_graph2 = std::make_shared<ExecuteGraph>(name);
    root_graph2->SetParentGraph(root_graph.get());
    root_graph2->SetParentNode(node[0]);

    auto ok_fast_graph = root_graph->AddSubGraph(root_graph2, name);
    ASSERT_NE(ok_fast_graph, nullptr);

    auto find_graph = root_graph2->GetSubGraph(name);
    ASSERT_NE(find_graph, nullptr);

    auto ret = root_graph->RemoveSubGraph(name);
    ASSERT_EQ(ret, GRAPH_SUCCESS);
  }

  for (int i = 0; i < subgraph_num; i++) {
    std::string name = "subgraph_" + std::to_string(i);
    auto fast_graph = root_graph->AddSubGraph(sub_graph[i], name);
    ASSERT_NE(fast_graph, nullptr);

    auto find_graph = root_graph->GetSubGraph(name);
    ASSERT_NE(find_graph, nullptr);

    auto ret = root_graph->RemoveSubGraph(name);
    ASSERT_EQ(ret, GRAPH_SUCCESS);
  }

  for (int i = 0; i < subgraph_num; i++) {
    std::string name = "subgraph_" + std::to_string(i);
    auto fast_graph = root_graph->AddSubGraph(sub_graph[i], name);
    ASSERT_NE(fast_graph, nullptr);
  }

  auto find_graph = root_graph->GetSubGraph("subgraph_1");
  ASSERT_NE(find_graph, nullptr);

  auto subgraphs = root_graph->GetAllSubgraphs();
  ASSERT_EQ(subgraphs.size(), subgraph_num);
  root_graph->ClearAllSubGraph();
}

TEST_F(UtestExecuteGraph, AddOKSubGraphToExecuteGraph) {
  auto root_graph = std::make_shared<ExecuteGraph>("root_graph");
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
    node[i] = root_graph->AddNode(op_desc);
    ASSERT_NE(node[i], nullptr);
  }

  int subgraph_num = 5;
  std::shared_ptr<ExecuteGraph> sub_graph[subgraph_num] = {nullptr};
  for (int i = 0; i < subgraph_num; i++) {
    sub_graph[i] = std::make_shared<ExecuteGraph>("subgraph_" + std::to_string(i));
    sub_graph[i]->SetParentGraph(root_graph.get());
    sub_graph[i]->SetParentNode(node[i]);
  }

  {
    std::shared_ptr<ExecuteGraph> invalid_fast_graph = nullptr;
    auto fast_graph = root_graph->AddSubGraph(invalid_fast_graph);
    ASSERT_EQ(fast_graph, nullptr);

    auto ret = root_graph->RemoveSubGraph(nullptr);
    ASSERT_EQ(ret, GRAPH_PARAM_INVALID);
  }

  for (int i = 0; i < subgraph_num; i++) {
    std::string name = "subgraph_" + std::to_string(i);
    auto fast_graph = root_graph->AddSubGraph(sub_graph[i]);
    ASSERT_NE(fast_graph, nullptr);

    auto ret = root_graph->RemoveSubGraph(fast_graph);
    ASSERT_EQ(ret, GRAPH_SUCCESS);
  }
}

TEST_F(UtestExecuteGraph, TestExecuteGraphAssign) {
  auto root_graph = std::make_shared<ExecuteGraph>("root_graph");

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
    node[i] = root_graph->AddNode(op_desc);
    ASSERT_NE(node[i], nullptr);
  }

  FastEdge *edge[edge_num] = {};
  for (int i = 0; i < edge_num; ++i) {
    edge[i] = root_graph->AddEdge(node[0], i, node[1], i);
    ASSERT_NE(edge[i], nullptr);
  }

  int subgraph_num = 5;
  std::shared_ptr<ExecuteGraph> sub_graph[subgraph_num] = {nullptr};
  for (int i = 0; i < subgraph_num; i++) {
    sub_graph[i] = std::make_shared<ExecuteGraph>("subgraph_" + std::to_string(i));
    ASSERT_NE(sub_graph[i], nullptr);
    sub_graph[i]->SetParentGraph(root_graph.get());
    sub_graph[i]->SetParentNode(node[i]);
  }

  for (int i = 0; i < subgraph_num; i++) {
    std::string name = "subgraph_" + std::to_string(i);
    auto fast_graph = root_graph->AddSubGraph(sub_graph[i], name);
    ASSERT_NE(fast_graph, nullptr);

    auto ret = root_graph->RemoveSubGraph(name);
    ASSERT_EQ(ret, GRAPH_SUCCESS);
  }

  auto Assign_graph = std::make_shared<ExecuteGraph>("root_graph");
  *Assign_graph = *root_graph;
}

TEST_F(UtestExecuteGraph, TestRecycle) {
  auto compute_graph = std::make_shared<ge::ExecuteGraph>("graph");
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

  FastEdge *edge[edge_num] = {};
  for (int i = 0; i < edge_num; ++i) {
    edge[i] = compute_graph->AddEdge(node[0], i, node[1], i);
    ASSERT_NE(edge[i], nullptr);
  }

  FastGraphUtils::GetListElementAddr(node[0])->owner->erase(FastGraphUtils::GetListElementAddr(node[0]));
  auto ret = compute_graph->RecycleQuickNode(node[0]);
  ASSERT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestExecuteGraph, TestNodes) {
  auto compute_graph = std::make_shared<ge::ExecuteGraph>("graph");
  int node_num = 10;
  int edge_num = 10;
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

  ASSERT_EQ(compute_graph->GetDirectNodesSize(), node_num);
  ASSERT_EQ(compute_graph->GetDirectNode().size(), node_num);

  {
    auto ret = compute_graph->RemoveJustNode(node[0]);
    ASSERT_EQ(ret, GRAPH_SUCCESS);

    auto op_desc = std::make_shared<OpDesc>("op", "op");
    auto td = GeTensorDesc();
    for (int j = 0; j < edge_num; ++j) {
      op_desc->AddInputDesc(td);
      op_desc->AddOutputDesc(td);
    }
    auto node = compute_graph->AddNode(op_desc);
    ASSERT_NE(node, nullptr);
  }
}

TEST_F(UtestExecuteGraph, TestIONodes) {
  auto compute_graph = std::make_shared<ge::ExecuteGraph>("graph");
  int node_num = 10;
  FastNode *quick_node[node_num] = {};

  for (int i = 0; i < node_num; ++i) {
    auto op_desc = std::make_shared<OpDesc>("op", "op");
    auto td = GeTensorDesc();
    op_desc->AddInputDesc(td);
    op_desc->AddOutputDesc(td);
    quick_node[i] = compute_graph->AddNode(op_desc);
    ASSERT_NE(quick_node[i], nullptr);
  }

  {
    auto add_node = compute_graph->AddNode(nullptr);
    ASSERT_EQ(add_node, nullptr);

    add_node = compute_graph->AddNode(nullptr, 1);
    ASSERT_EQ(add_node, nullptr);

    auto input_node = compute_graph->AddInputNode(nullptr);
    ASSERT_EQ(input_node, nullptr);

    auto ret = compute_graph->RemoveInputNode(nullptr);
    ASSERT_NE(ret, GRAPH_SUCCESS);

    auto node = compute_graph->AddOutputNodeByIndex(nullptr, 0);
    ASSERT_EQ(node, nullptr);

    ret = compute_graph->RemoveOutputNode(nullptr);
    ASSERT_NE(ret, GRAPH_SUCCESS);
  }

  auto input_node = compute_graph->AddInputNode(quick_node[0]);
  ASSERT_NE(input_node, nullptr);

  auto output_node = compute_graph->AddOutputNodeByIndex(quick_node[node_num - 1], 0);
  ASSERT_NE(output_node, nullptr);

  auto ret = compute_graph->RemoveInputNode(quick_node[0]);
  ASSERT_EQ(ret, GRAPH_SUCCESS);
  ret = compute_graph->RemoveOutputNode(quick_node[node_num - 1]);
  ASSERT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestExecuteGraph, TestSortsNodes) {
  auto root_graph = std::make_shared<ExecuteGraph>("root_graph");
  int node_num = 10;
  auto subgraph_num = 10;
  auto subgraph_node_num = 5;
  auto io_num = 5;

  std::shared_ptr<OpDesc> op_desc[node_num] = {nullptr};
  for (int j = 0; j < node_num; j++) {
    op_desc[j] = std::make_shared<OpDesc>("op", "op");
    auto td = GeTensorDesc();

    for (int64_t i = 0; i < io_num; ++i) {
      op_desc[j]->AddInputDesc(td);
    }
    for (int64_t i = 0; i < io_num; ++i) {
      op_desc[j]->AddOutputDesc(td);
    }

    op_desc[j]->AddSubgraphName("subgraph_" + std::to_string(j));
    op_desc[j]->SetSubgraphInstanceName(j, "subgraph_" + std::to_string(j));
  }

  FastNode *quick_node[node_num] = {};
  for (int i = 0; i < node_num; i++) {
    quick_node[i] = root_graph->AddNode(op_desc[i]);
    ASSERT_NE(quick_node[i], nullptr);
  }

  for (int j = 1; j < node_num; j++) {
    root_graph->AddEdge(quick_node[j], 1, quick_node[j - 1], 0);
  }

  std::shared_ptr<OpDesc> sub_op_desc[subgraph_num][subgraph_node_num] = {};
  for (int i = 0; i < subgraph_num; i++) {
    for (int j = 0; j < subgraph_node_num; j++) {
      sub_op_desc[i][j] = std::make_shared<OpDesc>("op", "op");
      auto td = GeTensorDesc();

      for (int64_t x = 0; x < io_num; ++x) {
        sub_op_desc[i][j]->AddInputDesc(td);
      }
      for (int64_t x = 0; x < io_num; ++x) {
        sub_op_desc[i][j]->AddOutputDesc(td);
      }
    }
  }

  std::shared_ptr<ExecuteGraph> sub_graph[subgraph_num]{};
  for (int i = 0; i < subgraph_num; i++) {
    sub_graph[i] = std::make_shared<ExecuteGraph>("subgraph_" + std::to_string(i));
    FastNode *sub_graph_node[subgraph_node_num] = {};
    for (int j = 0; j < subgraph_node_num; j++) {
      sub_graph_node[j] = sub_graph[i]->AddNode(sub_op_desc[i][j]);
    }
    for (int j = 1; j < subgraph_node_num; j++) {
      sub_graph[i]->AddEdge(sub_graph_node[j], 1, sub_graph_node[j - 1], 0);
      sub_graph[i]->SetParentGraph(root_graph.get());
      sub_graph[i]->SetParentNode(quick_node[i]);
    }
  }

  for (int64_t i = 0; i < subgraph_num; ++i) {
    std::string name = "subgraph_" + std::to_string(i);
    auto ret = root_graph->AddSubGraph(sub_graph[i], name);
    ASSERT_NE(ret, nullptr);
  }

  /* bfs   reverse      no memory priority */
  {
    static const std::string kTopoSortingBfs = "0";
    auto ori_graph_options = GetThreadLocalContext().GetAllGraphOptions();
    auto graph_options = ori_graph_options;
    graph_options[OPTION_TOPOSORTING_MODE] = kTopoSortingBfs;
    graph_options[MEMORY_OPTIMIZATION_POLICY] = "xxx";
    graph_options[ENABLE_SINGLE_STREAM] = "true";
    GetThreadLocalContext().SetGraphOption(graph_options);
    auto ret = root_graph->TopologicalSorting();
    ASSERT_EQ(ret, GRAPH_SUCCESS);
  }

  /* dfs   reverse      no memory priority */
  {
    static const std::string kTopoSortingDfs = "1";
    auto ori_graph_options = GetThreadLocalContext().GetAllGraphOptions();
    auto graph_options = ori_graph_options;
    graph_options[OPTION_TOPOSORTING_MODE] = kTopoSortingDfs;
    graph_options[MEMORY_OPTIMIZATION_POLICY] = "xxx";
    graph_options[ENABLE_SINGLE_STREAM] = "true";
    GetThreadLocalContext().SetGraphOption(graph_options);
    auto ret = root_graph->TopologicalSorting();
    ASSERT_EQ(ret, GRAPH_SUCCESS);
  }

  /* DfsPostOrder   reverse      no memory priority */
  {
    static const std::string kTopoSortingDfsPostOrder = "2";
    auto ori_graph_options = GetThreadLocalContext().GetAllGraphOptions();
    auto graph_options = ori_graph_options;
    graph_options[OPTION_TOPOSORTING_MODE] = kTopoSortingDfsPostOrder;
    graph_options[MEMORY_OPTIMIZATION_POLICY] = "xxx";
    graph_options[ENABLE_SINGLE_STREAM] = "true";
    GetThreadLocalContext().SetGraphOption(graph_options);
    auto ret = root_graph->TopologicalSorting();
    ASSERT_EQ(ret, GRAPH_SUCCESS);
  }

  /* bfs   no reverse      no memory priority */
  {
    static const std::string kTopoSortingBfs = "0";
    auto ori_graph_options = GetThreadLocalContext().GetAllGraphOptions();
    auto graph_options = ori_graph_options;
    graph_options[OPTION_TOPOSORTING_MODE] = kTopoSortingBfs;
    graph_options[MEMORY_OPTIMIZATION_POLICY] = "xxxx";
    graph_options[ENABLE_SINGLE_STREAM] = "false";
    GetThreadLocalContext().SetGraphOption(graph_options);
    auto ret = root_graph->TopologicalSorting();
    ASSERT_EQ(ret, GRAPH_SUCCESS);
  }

  /* dfs   no reverse      no memory priority */
  {
    static const std::string kTopoSortingDfs = "1";
    auto ori_graph_options = GetThreadLocalContext().GetAllGraphOptions();
    auto graph_options = ori_graph_options;
    graph_options[OPTION_TOPOSORTING_MODE] = kTopoSortingDfs;
    graph_options[MEMORY_OPTIMIZATION_POLICY] = "xxx";
    graph_options[ENABLE_SINGLE_STREAM] = "false";
    GetThreadLocalContext().SetGraphOption(graph_options);
    auto ret = root_graph->TopologicalSorting();
    ASSERT_EQ(ret, GRAPH_SUCCESS);
  }

  /* DfsPostOrder   no reverse      no memory priority */
  {
    static const std::string kTopoSortingDfsPostOrder = "2";
    auto ori_graph_options = GetThreadLocalContext().GetAllGraphOptions();
    auto graph_options = ori_graph_options;
    graph_options[OPTION_TOPOSORTING_MODE] = kTopoSortingDfsPostOrder;
    graph_options[MEMORY_OPTIMIZATION_POLICY] = "xx";
    graph_options[ENABLE_SINGLE_STREAM] = "false";
    GetThreadLocalContext().SetGraphOption(graph_options);
    auto ret = root_graph->TopologicalSorting();
    ASSERT_EQ(ret, GRAPH_SUCCESS);
  }

  /* bfs   reverse       with memory priority */
  {
    static const std::string kTopoSortingBfs = "0";
    auto ori_graph_options = GetThreadLocalContext().GetAllGraphOptions();
    auto graph_options = ori_graph_options;
    graph_options[OPTION_TOPOSORTING_MODE] = kTopoSortingBfs;
    graph_options[MEMORY_OPTIMIZATION_POLICY] = "MemoryPriority";
    graph_options[ENABLE_SINGLE_STREAM] = "true";
    GetThreadLocalContext().SetGraphOption(graph_options);
    auto ret = root_graph->TopologicalSorting();
    ASSERT_EQ(ret, GRAPH_SUCCESS);
  }

  /* dfs   reverse      with memory priority */
  {
    static const std::string kTopoSortingDfs = "1";
    auto ori_graph_options = GetThreadLocalContext().GetAllGraphOptions();
    auto graph_options = ori_graph_options;
    graph_options[OPTION_TOPOSORTING_MODE] = kTopoSortingDfs;
    graph_options[MEMORY_OPTIMIZATION_POLICY] = "MemoryPriority";
    graph_options[ENABLE_SINGLE_STREAM] = "true";
    GetThreadLocalContext().SetGraphOption(graph_options);
    auto ret = root_graph->TopologicalSorting();
    ASSERT_EQ(ret, GRAPH_SUCCESS);
  }

  /* DfsPostOrder   reverse      with memory priority */
  {
    static const std::string kTopoSortingDfsPostOrder = "2";
    auto ori_graph_options = GetThreadLocalContext().GetAllGraphOptions();
    auto graph_options = ori_graph_options;
    graph_options[OPTION_TOPOSORTING_MODE] = kTopoSortingDfsPostOrder;
    graph_options[MEMORY_OPTIMIZATION_POLICY] = "MemoryPriority";
    graph_options[ENABLE_SINGLE_STREAM] = "true";
    GetThreadLocalContext().SetGraphOption(graph_options);
    auto ret = root_graph->TopologicalSorting();
    ASSERT_EQ(ret, GRAPH_SUCCESS);
  }

  /* bfs   no reverse       with memory priority */
  {
    static const std::string kTopoSortingBfs = "0";
    auto ori_graph_options = GetThreadLocalContext().GetAllGraphOptions();
    auto graph_options = ori_graph_options;
    graph_options[OPTION_TOPOSORTING_MODE] = kTopoSortingBfs;
    graph_options[MEMORY_OPTIMIZATION_POLICY] = "MemoryPriority";
    graph_options[ENABLE_SINGLE_STREAM] = "false";
    GetThreadLocalContext().SetGraphOption(graph_options);
    auto ret = root_graph->TopologicalSorting();
    ASSERT_EQ(ret, GRAPH_SUCCESS);
  }

  /* dfs   no reverse      with memory priority */
  {
    static const std::string kTopoSortingDfs = "1";
    auto ori_graph_options = GetThreadLocalContext().GetAllGraphOptions();
    auto graph_options = ori_graph_options;
    graph_options[OPTION_TOPOSORTING_MODE] = kTopoSortingDfs;
    graph_options[MEMORY_OPTIMIZATION_POLICY] = "MemoryPriority";
    graph_options[ENABLE_SINGLE_STREAM] = "false";
    GetThreadLocalContext().SetGraphOption(graph_options);
    auto ret = root_graph->TopologicalSorting();
    ASSERT_EQ(ret, GRAPH_SUCCESS);
  }
}

TEST_F(UtestExecuteGraph, TestSortsNodesError) {
  auto root_graph = std::make_shared<ExecuteGraph>("root_graph");
  int node_num = 10;
  auto subgraph_num = 10;
  auto subgraph_node_num = 5;
  auto io_num = 5;

  std::shared_ptr<OpDesc> op_desc[node_num] = {nullptr};
  for (int j = 0; j < node_num; j++) {
    op_desc[j] = std::make_shared<OpDesc>("op", "op");
    auto td = GeTensorDesc();

    for (int64_t i = 0; i < io_num; ++i) {
      op_desc[j]->AddInputDesc(td);
    }
    for (int64_t i = 0; i < io_num; ++i) {
      op_desc[j]->AddOutputDesc(td);
    }
  }

  FastNode *quick_node[node_num] = {};
  for (int i = 0; i < node_num; i++) {
    quick_node[i] = root_graph->AddNode(op_desc[i]);
    ASSERT_NE(quick_node[i], nullptr);
  }

  FastEdge *edge[node_num] = {};
  for (int j = 1; j < node_num; j++) {
    edge[j] = root_graph->AddEdge(quick_node[j], 1, quick_node[j - 1], 0);
  }

  std::shared_ptr<OpDesc> sub_op_desc[subgraph_num][subgraph_node_num] = {};
  for (int i = 0; i < subgraph_num; i++) {
    for (int j = 0; j < subgraph_node_num; j++) {
      sub_op_desc[i][j] = std::make_shared<OpDesc>("op", "op");
      auto td = GeTensorDesc();

      for (int64_t x = 0; x < io_num; ++x) {
        sub_op_desc[i][j]->AddInputDesc(td);
      }
      for (int64_t x = 0; x < io_num; ++x) {
        sub_op_desc[i][j]->AddOutputDesc(td);
      }
    }
  }

  std::shared_ptr<ExecuteGraph> sub_graph[subgraph_num]{};
  for (int i = 0; i < subgraph_num; i++) {
    sub_graph[i] = std::make_shared<ExecuteGraph>("subgraph_" + std::to_string(i));
    FastNode *sub_graph_node[subgraph_node_num] = {};
    for (int j = 0; j < subgraph_node_num; j++) {
      sub_graph_node[j] = sub_graph[i]->AddNode(sub_op_desc[i][j]);
      sub_graph[i]->SetParentGraph(root_graph.get());
      sub_graph[i]->SetParentNode(quick_node[i]);
    }
  }

  for (int64_t i = 0; i < subgraph_num; ++i) {
    std::string name = "subgraph_" + std::to_string(i);
    auto ret = root_graph->AddSubGraph(sub_graph[i], name);
    ASSERT_NE(ret, nullptr);
  }

  /* ERROR */
  {
    static const std::string kTopoSortingBfs = "1";
    auto ori_graph_options = GetThreadLocalContext().GetAllGraphOptions();
    auto graph_options = ori_graph_options;
    graph_options[OPTION_TOPOSORTING_MODE] = kTopoSortingBfs;
    graph_options[MEMORY_OPTIMIZATION_POLICY] = "MemoryPriority";
    graph_options[ENABLE_SINGLE_STREAM] = "false";
    GetThreadLocalContext().SetGraphOption(graph_options);
    auto ret = root_graph->TopologicalSorting();
    ASSERT_EQ(ret, GRAPH_FAILED);
  }

  for (int j = 1; j < node_num; j++) {
    root_graph->RemoveEdge(edge[j]);
  }

  /* ERROR */
  {
    static const std::string kTopoSortingBfs = "1";
    auto ori_graph_options = GetThreadLocalContext().GetAllGraphOptions();
    auto graph_options = ori_graph_options;
    graph_options[OPTION_TOPOSORTING_MODE] = kTopoSortingBfs;
    graph_options[MEMORY_OPTIMIZATION_POLICY] = "MemoryPriority";
    graph_options[ENABLE_SINGLE_STREAM] = "false";
    GetThreadLocalContext().SetGraphOption(graph_options);
    auto ret = root_graph->TopologicalSorting();
    ASSERT_EQ(ret, GRAPH_FAILED);
  }
}

TEST_F(UtestExecuteGraph, TestGraphName) {
  auto root_graph = std::make_shared<ExecuteGraph>("root_graph");
  std::string str = "changetohelloworld";
  root_graph->SetName(str);
  auto ret = root_graph->GetName();
  ASSERT_EQ(ret, str);
}

TEST_F(UtestExecuteGraph, TestGraphParent) {
  auto root_graph = std::make_shared<ExecuteGraph>("root_graph");
  auto sub_graph = std::make_shared<ExecuteGraph>("sub_graph");
  sub_graph->SetParentGraph(root_graph.get());
  auto ret = sub_graph->GetParentGraphBarePtr();
  ASSERT_EQ(ret, root_graph.get());
}

TEST_F(UtestExecuteGraph, TestRecycleNode) {
  auto root_graph = std::make_shared<ComputeGraph>("root_graph");
  int node_num = 10;
  int io_num = 5;
  std::shared_ptr<OpDesc> op_desc[node_num] = {nullptr};
  NodePtr node[node_num] = {};
  for (int j = 0; j < 1; j++) {
    op_desc[j] = std::make_shared<OpDesc>("op", "op");
    auto td = GeTensorDesc();

    for (int64_t i = 0; i < io_num; ++i) {
      op_desc[j]->AddInputDesc(td);
    }
    for (int64_t i = 0; i < io_num; ++i) {
      op_desc[j]->AddOutputDesc(td);
    }
  }

  for (int i = 0; i < 1; i++) {
    node[i] = root_graph->AddNode(op_desc[i]);
  }

  auto compute_graph = std::make_shared<ge::ExecuteGraph>("graph");
  {
    auto op_desc = std::make_shared<OpDesc>("op", "op");
    auto td = GeTensorDesc();
    op_desc->AddInputDesc(td);
    op_desc->AddOutputDesc(td);
    auto quick_node = compute_graph->AddNode(op_desc);
    ASSERT_NE(quick_node, nullptr);
    quick_node->SetNodePtr(node[0]);
    auto ret = compute_graph->RemoveJustNode(quick_node);
    ASSERT_EQ(ret, GRAPH_SUCCESS);
    ret = compute_graph->RecycleQuickNode(quick_node);
    ASSERT_EQ(ret, GRAPH_SUCCESS);
  }
  {
    auto ret = compute_graph->RecycleQuickNode(nullptr);
    ASSERT_NE(ret, GRAPH_SUCCESS);

    ret = compute_graph->RecycleQuickEdge(nullptr);
    ASSERT_NE(ret, GRAPH_SUCCESS);
  }
}

TEST_F(UtestExecuteGraph, ClearEdge) {
  auto compute_graph = std::make_shared<ge::ExecuteGraph>("graph");
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

  FastEdge *edge[edge_num] = {};
  for (int i = 0; i < edge_num; ++i) {
    edge[i] = compute_graph->AddEdge(node[0], i, node[1], i);
    ASSERT_NE(edge[i], nullptr);
  }

  for (int i = 0; i < edge_num; ++i) {
    auto ret = compute_graph->RemoveEdge(edge[i]);
    ASSERT_EQ(ret, GRAPH_SUCCESS);
  }
}

TEST_F(UtestExecuteGraph, DelayTopologicalSorting) {
  auto graph = BuildDelayTopoGraph("test_delay_topo_graph");
  std::map<std::string, std::string> options_map;
  options_map["ge.topoSortingMode"] = "2";
  options_map["ge.exec.memoryOptimizationPolicy"] = "MemoryPriority";
  GetThreadLocalContext().SetGraphOption(options_map);
  EXPECT_EQ(graph->TopologicalSorting(), GRAPH_SUCCESS);
  std::vector<std::string> expected_dfs_names = {"variable", "data", "node2", "node3", "node4", "node1", "node5"};
  std::vector<std::string> dfs_names;
  const auto &graph_dfs_topo = graph->GetAllNodes();
  for (auto &node : graph_dfs_topo) {
    dfs_names.push_back(node->GetName());
  }

  EXPECT_EQ(dfs_names, expected_dfs_names);
}

TEST_F(UtestExecuteGraph, NoDelayTopologicalSorting) {
  auto graph = BuildDelayTopoGraph("test_delay_topo_graph");
  std::map<std::string, std::string> options_map;
  options_map["ge.topoSortingMode"] = "1";
  GetThreadLocalContext().SetGraphOption(options_map);
  EXPECT_EQ(graph->TopologicalSorting(), GRAPH_SUCCESS);
  std::vector<std::string> expected_dfs_names = {"variable", "node2", "node1", "data", "node3", "node4", "node5"};
  std::vector<std::string> dfs_names;
  const auto &graph_dfs_topo = graph->GetAllNodes();
  for (auto &node : graph_dfs_topo) {
    dfs_names.push_back(node->GetName());
  }
  EXPECT_EQ(dfs_names, expected_dfs_names);

  {
    /* recovery environment */
    static const std::string kTopoSortingDfs = "1";
    auto ori_graph_options = GetThreadLocalContext().GetAllGraphOptions();
    auto graph_options = ori_graph_options;
    graph_options[OPTION_TOPOSORTING_MODE] = kTopoSortingDfs;
    graph_options[MEMORY_OPTIMIZATION_POLICY] = "xxx";
    graph_options[ENABLE_SINGLE_STREAM] = "false";
    GetThreadLocalContext().SetGraphOption(graph_options);
  }
}

TEST_F(UtestExecuteGraph, TestNodeAttr) {
  auto compute_graph = std::make_shared<ge::ExecuteGraph>("graph");
  auto op_desc = std::make_shared<OpDesc>("op", "op");
  auto td = GeTensorDesc();
  op_desc->AddInputDesc(td);
  op_desc->AddOutputDesc(td);
  auto node = compute_graph->AddNode(op_desc);
  ASSERT_NE(node, nullptr);
  ASSERT_NE(node->GetExtendInfo(), nullptr);
  auto name_ptr = node->GetNamePtr();
  ASSERT_NE(name_ptr, nullptr);
  auto name = node->GetName();
  ASSERT_EQ(name, "op");
  auto type = node->GetType();
  ASSERT_EQ(name, "op");
  auto type_ptr = node->GetTypePtr();
  ASSERT_NE(type_ptr, nullptr);
  auto graph = node->GetExtendInfo()->GetOwnerGraphBarePtr();
  ASSERT_EQ(graph, compute_graph.get());

  auto ret = compute_graph->RemoveJustNode(node);
  ASSERT_EQ(ret, GRAPH_SUCCESS);

  auto node1 = compute_graph->AddNode(node);
  ASSERT_EQ(node1, node);

  {
    auto op_desc1 = std::make_shared<OpDesc>("op", "op");
    auto td = GeTensorDesc();
    op_desc1->AddInputDesc(td);
    op_desc1->AddOutputDesc(td);
    auto new_node_1 = compute_graph->AddNode(op_desc);
    new_node_1->Init(op_desc);
    ASSERT_NE(*new_node_1 == *node1, true);
  }
}

TEST_F(UtestExecuteGraph, TestNodeEqual) {
  auto compute_graph = std::make_shared<ge::ExecuteGraph>("graph");
  auto op_desc = std::make_shared<OpDesc>("op", "op");
  auto td = GeTensorDesc();
  op_desc->AddInputDesc(td);
  op_desc->AddOutputDesc(td);
  auto node = compute_graph->AddNode(op_desc);

  auto ret = compute_graph->RemoveJustNode(node);
  ASSERT_EQ(ret, GRAPH_SUCCESS);

  auto node1 = compute_graph->AddNode(node);
  ASSERT_EQ(*node1 == *node, true);
}

TEST_F(UtestExecuteGraph, TestEdgeError) {
  auto compute_graph = std::make_shared<ge::ExecuteGraph>("graph");
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

  int32_t invalid_index = -2;
  {
    QuickEdge *edge = new QuickEdge;
    FastGraphUtils::GetEdge(edge).dst_input = invalid_index;
    graphStatus ret = node[0]->RecordEdge(&FastGraphUtils::GetEdge(edge), DirectionType::kDirectionInType);
    ASSERT_NE(ret, GRAPH_SUCCESS);
    delete edge;
  }

  {
    QuickEdge *edge = new QuickEdge;
    FastGraphUtils::GetEdgeSrcOutput(edge) = invalid_index;
    graphStatus ret = node[0]->RecordEdge(&FastGraphUtils::GetEdge(edge), DirectionType::kDirectionOutType);
    ASSERT_NE(ret, GRAPH_SUCCESS);
    delete edge;
  }

  {
    QuickEdge *edge = new QuickEdge;
    FastGraphUtils::GetEdgeSrcOutput(edge) = invalid_index;
    graphStatus ret = node[0]->EraseEdge(&FastGraphUtils::GetEdge(edge), DirectionType::kDirectionOutType);
    ASSERT_NE(ret, GRAPH_SUCCESS);
    delete edge;
  }

  {
    QuickEdge *edge = new QuickEdge;
    FastGraphUtils::GetEdgeDstInput(edge) = invalid_index;
    graphStatus ret = node[0]->EraseEdge(&FastGraphUtils::GetEdge(edge), DirectionType::kDirectionInType);
    ASSERT_NE(ret, GRAPH_SUCCESS);
    delete edge;
  }
}

TEST_F(UtestExecuteGraph, TestNodeEdgeError) {
  auto compute_graph = std::make_shared<ge::ExecuteGraph>("graph");
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
  {
    auto ret = node[0]->GetOutDataNodesByIndex(-2);
    ASSERT_EQ(ret.size(), 0);
  }

  {
    auto ret = node[1]->GetOutControlNodes();
    ASSERT_EQ(ret.size(), 0);
  }

  {
    auto ret = node[2]->GetInDataEdgeByIndex(-2);
    ASSERT_EQ(ret, nullptr);
  }
}

TEST_F(UtestExecuteGraph, TestMoveEdge) {
  auto compute_graph = std::make_shared<ge::ExecuteGraph>("graph");
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

  FastEdge *edge[edge_num] = {};
  for (int i = 0; i < edge_num; ++i) {
    edge[i] = compute_graph->AddEdge(node[0], i, node[1], i);
    ASSERT_NE(edge[i], nullptr);
  }

  for (int i = 0; i < edge_num; ++i) {
    auto ret = compute_graph->RemoveEdge(edge[i]);
    ASSERT_EQ(ret, GRAPH_SUCCESS);
  }

  edge[0] = compute_graph->AddEdge(node[0], 0, node[1], 0);

  auto size = node[0]->GetOutEdgesSizeByIndex(0);
  ASSERT_EQ(size, 1);

  auto edges = node[0]->GetOutEdgesByIndex(0);
  ASSERT_EQ(edges.empty(), false);
  ASSERT_EQ(edges[0], edge[0]);

  size = node[1]->GetInEdgesSizeByIndex(0);
  ASSERT_EQ(size, 1);

  size = node[1]->GetInEdgesSizeByIndex(-1);
  ASSERT_EQ(size, 0);

  auto data_edge = node[1]->GetInDataEdgeByIndex(0);
  ASSERT_EQ(data_edge, edge[0]);

  auto ret = node[0]->MoveEdge(DirectionType::kDirectionOutType, 0, 1, 0);
  ASSERT_EQ(ret, GRAPH_SUCCESS);

  size = node[0]->GetOutEdgesSizeByIndex(0);
  ASSERT_EQ(size, 1);

  size = node[0]->GetOutEdgesSizeByIndex(-1);
  ASSERT_EQ(size, 0);

  edges = node[0]->GetOutEdgesByIndex(0);
  ASSERT_EQ(edges[0], edge[0]);

  ret = node[1]->MoveEdge(DirectionType::kDirectionInType, 0, 0, 0);
  ASSERT_EQ(ret, GRAPH_SUCCESS);

  size = node[1]->GetInEdgesSizeByIndex(0);
  ASSERT_EQ(size, 1);

  data_edge = node[1]->GetInDataEdgeByIndex(0);
  ASSERT_EQ(data_edge, edge[0]);
}

TEST_F(UtestExecuteGraph, GetDataEdgeInfo) {
  auto compute_graph = std::make_shared<ge::ExecuteGraph>("graph");
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

  FastEdge *edge[edge_num] = {};
  for (int i = 0; i < edge_num; ++i) {
    edge[i] = compute_graph->AddEdge(node[0], i, node[1], i);
    ASSERT_NE(edge[i], nullptr);
  }

  auto out_size = node[0]->GetAllOutEdgesSize();
  ASSERT_EQ(out_size, edge_num);

  auto nodes = node[0]->GetOutDataNodesByIndex(0);
  ASSERT_EQ(nodes.size(), 1);

  auto size = node[1]->GetAllInEdgeSize();
  ASSERT_EQ(size, edge_num);

  auto vec_size = node[1]->GetAllInDataEdgesRef();
  ASSERT_EQ(vec_size.size(), edge_num);

  vec_size = node[1]->MutableAllInDataEdges();
  ASSERT_EQ(vec_size.size(), edge_num);

  auto node_vec = node[0]->GetOutDataNodes();
  ASSERT_EQ(node_vec.size(), edge_num);

  node_vec = node[0]->GetAllInNodes();
  ASSERT_EQ(node_vec.size(), 0);

  node_vec = node[1]->GetInDataNodes();
  ASSERT_EQ(node_vec.size(), edge_num);
}

TEST_F(UtestExecuteGraph, GetAllEdgeInfo) {
  auto compute_graph = std::make_shared<ge::ExecuteGraph>("graph");
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

  FastEdge *edge[edge_num] = {};
  FastEdge *control_edge = {};
  for (int i = 0; i < edge_num; ++i) {
    edge[i] = compute_graph->AddEdge(node[0], i, node[1], i);
    ASSERT_NE(edge[i], nullptr);
  }
  control_edge = compute_graph->AddEdge(node[0], kControlEdgeIndex, node[1], kControlEdgeIndex);
  ASSERT_NE(control_edge, nullptr);

  auto out_size = node[0]->GetAllOutEdgesSize();
  ASSERT_EQ(out_size, edge_num + 1);

  auto nodes = node[0]->GetOutDataNodesByIndex(0);
  ASSERT_EQ(nodes.size(), 1);

  auto size = node[1]->GetAllInEdgeSize();
  ASSERT_EQ(size, edge_num + 1);

  auto vec_size = node[1]->GetAllInDataEdgesRef();
  ASSERT_EQ(vec_size.size(), edge_num);

  vec_size = node[1]->MutableAllInDataEdges();
  ASSERT_EQ(vec_size.size(), edge_num);

  auto node_vec = node[0]->GetOutDataNodes();
  ASSERT_EQ(node_vec.size(), edge_num);

  node_vec = node[1]->GetAllInNodes();
  ASSERT_EQ(node_vec.size(), edge_num + 1);

  auto flag = node[1]->IsDirectlyControlledByNode(node[0]);
  ASSERT_EQ(flag, true);

  node_vec = node[1]->GetInDataNodes();
  ASSERT_EQ(node_vec.size(), edge_num);
}

TEST_F(UtestExecuteGraph, GetControlEdgeInfo) {
  auto compute_graph = std::make_shared<ge::ExecuteGraph>("graph");
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

  FastEdge *edge[node_num] = {};
  for (int i = 1; i < node_num; ++i) {
    edge[i] = compute_graph->AddEdge(node[0], kControlEdgeIndex, node[i], kControlEdgeIndex);
    ASSERT_NE(edge[i], nullptr);
  }

  auto out_edges = node[0]->GetOutEdgesRefByIndex(kControlEdgeIndex);
  ASSERT_EQ(out_edges.size(), node_num - 1);

  auto out_control_nodes = node[0]->GetOutControlNodes();
  ASSERT_EQ(out_control_nodes.size(), node_num - 1);

  auto nodes = node[0]->GetOutControlNodes();
  ASSERT_EQ(nodes.size(), node_num - 1);

  auto in_control_nodes = node[1]->GetInControlNodes();
  ASSERT_EQ(in_control_nodes.size(), 1);
}

TEST_F(UtestExecuteGraph, deepcopy) {
  auto node_num = 10;
  auto io_num = 10;
  auto subgraph_num = 10;
  auto subgraph_node_num = 10;
  std::shared_ptr<OpDesc> op_desc[node_num] = {nullptr};
  for (int j = 0; j < node_num; j++) {
    op_desc[j] = std::make_shared<OpDesc>("op", "op");
    auto td = GeTensorDesc();

    for (int64_t i = 0; i < io_num; ++i) {
      op_desc[j]->AddInputDesc(td);
    }
    for (int64_t i = 0; i < io_num; ++i) {
      op_desc[j]->AddOutputDesc(td);
    }
  }

  std::shared_ptr<ExecuteGraph> sub_graph[subgraph_num] = {nullptr};
  FastNode *node[node_num] = {};
  FastEdge *edge[node_num] = {};
  ExecuteGraph *quick_graph[subgraph_num] = {nullptr};
  std::shared_ptr<OpDesc> sub_op_desc[subgraph_num][subgraph_node_num] = {};
  for (int i = 0; i < subgraph_num; i++) {
    for (int j = 0; j < subgraph_node_num; j++) {
      sub_op_desc[i][j] = std::make_shared<OpDesc>("op", "op");
      auto td = GeTensorDesc();
      for (int64_t x = 0; x < io_num; ++x) {
        sub_op_desc[i][j]->AddInputDesc(td);
      }
      for (int64_t x = 0; x < io_num; ++x) {
        sub_op_desc[i][j]->AddOutputDesc(td);
      }
    }
  }

  auto root_graph = std::make_shared<ExecuteGraph>("root_graph");
  for (int i = 0; i < node_num; i++) {
    node[i] = root_graph->AddNode(op_desc[i]);
    ASSERT_NE(node[i], nullptr);
  }
  auto input_node = root_graph->AddInputNode(node[0]);
  ASSERT_NE(input_node, nullptr);

  for (int i = 1; i < node_num; i++) {
    edge[i] = root_graph->AddEdge(node[i], 1, node[i - 1], 0);
    ASSERT_NE(edge[i], nullptr);
  }

  for (int i = 0; i < subgraph_num; i++) {
    std::string name = "subgraph_" + std::to_string(i);
    sub_graph[i] = std::make_shared<ExecuteGraph>(name);
    FastNode *sub_graph_node[subgraph_node_num] = {};
    for (int j = 0; j < subgraph_node_num; j++) {
      sub_graph_node[j] = sub_graph[i]->AddNode(sub_op_desc[i][j]);
      ASSERT_NE(sub_graph_node[j], nullptr);
    }
    for (int j = 1; j < subgraph_node_num; j++) {
      auto ret = sub_graph[i]->AddEdge(sub_graph_node[j], 1, sub_graph_node[j - 1], 0);
      ASSERT_NE(ret, nullptr);
    }
  }

  for (int i = 0; i < subgraph_num; ++i) {
    sub_graph[i]->SetParentGraph(root_graph.get());
    sub_graph[i]->SetParentNode(node[i]);
    std::string name = "subgraph_" + std::to_string(i);
    quick_graph[i] = root_graph->AddSubGraph(sub_graph[i], name);
    ASSERT_NE(quick_graph[i], nullptr);
  }

  std::string name = "root_graph";
  auto test1_graph = std::make_shared<ExecuteGraph>(name);
  test1_graph->CompleteCopy(*(root_graph.get()));
}

TEST_F(UtestExecuteGraph, TopologicalSorting_ok_with_subgraph) {
  auto graph = ExecuteSharedGraph::BuildGraphWithSubGraph();
  auto sub_graph1 = graph->GetSubGraph("sub1");
  EXPECT_NE(sub_graph1, nullptr);
  EXPECT_EQ(sub_graph1->GetName(), "sub1");
  auto sub_graph2 = graph->GetSubGraph("sub2");
  EXPECT_NE(sub_graph2, nullptr);
  EXPECT_EQ(sub_graph2->GetName(), "sub2");
}

TEST_F(UtestExecuteGraph, AddEdge_check) {
  const auto create_op_func = []() -> OpDescPtr {
    auto td = GeTensorDesc();
    auto op_desc = std::make_shared<OpDesc>("op", "op");
    op_desc->AddInputDesc(td);
    op_desc->AddOutputDesc(td);
    return op_desc;
  };
  auto exe_graph0 = std::make_shared<ge::ExecuteGraph>("exe_graph0");
  auto op_desc0 = create_op_func();
  auto node0 = exe_graph0->AddNode(op_desc0);
  auto exe_graph1 = std::make_shared<ge::ExecuteGraph>("exe_graph1");
  auto op_desc1 = create_op_func();
  auto node1 = exe_graph1->AddNode(op_desc1);
  EXPECT_EQ(exe_graph0->AddEdge(node0, 0, node1, 1), nullptr);

  auto op_desc2 = create_op_func();
  auto node2 = exe_graph0->AddNode(op_desc2);
  EXPECT_NE(exe_graph0->AddEdge(node0, 0, node2, 0), nullptr);
  EdgeEndpointWithDirection eep0(node0, 0);
  EdgeEndpointWithDirection eep1(node0, 0);
  EdgeEndpointWithDirection eep2(node2, 0);
  EXPECT_EQ(eep0 < eep0, false);
  EXPECT_EQ(eep0 == eep2, false);
  EXPECT_EQ(eep0 == eep1, true);
}

TEST_F(UtestExecuteGraph, EdgeAndNodeOwner) {
  auto compute_graph = std::make_shared<ge::ExecuteGraph>("graph");
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

  FastEdge *edge[node_num] = {};
  FastEdge *ctrl_edge[node_num] = {};
  for (int i = 1; i < node_num; ++i) {
    edge[i] = compute_graph->AddEdge(node[i - 1], 0, node[i], 0);
    ASSERT_NE(edge[i], nullptr);
    auto flag = compute_graph->CheckEdgeIsInGraph(edge[i]);
    ASSERT_EQ(flag, true);

    ctrl_edge[i] = compute_graph->AddEdge(node[i - 1], kControlEdgeIndex, node[i], kControlEdgeIndex);
    ASSERT_NE(ctrl_edge[i], nullptr);
    flag = compute_graph->CheckEdgeIsInGraph(ctrl_edge[i]);
    ASSERT_EQ(flag, true);
  }

  auto graph1 = std::make_shared<ge::ExecuteGraph>("graph1");
  for (int i = 0; i < node_num; ++i) {
    auto ret = graph1->AddNode(node[i]);
    ASSERT_NE(ret, nullptr);
    ASSERT_NE(ret->GetExtendInfo(), nullptr);
    ASSERT_EQ(ret->GetExtendInfo()->GetOwnerGraphBarePtr(), graph1.get());
    auto flag = graph1->CheckNodeIsInGraph(ret);
    ASSERT_EQ(flag, true);
    flag = compute_graph->CheckNodeIsInGraph(ret);
    ASSERT_EQ(flag, false);
  }

  for (int i = 0; i < node_num; ++i) {
    auto &edges = node[i]->GetAllInDataEdgesRef();
    for (auto edge : edges) {
      if (edge != nullptr) {
        auto flag = compute_graph->CheckEdgeIsInGraph(edge);
        ASSERT_EQ(flag, true);
      }
    }

    auto &ctl_edges = node[i]->GetAllInControlEdgesRef();
    for (auto edge : ctl_edges) {
      if (edge != nullptr) {
        auto flag = compute_graph->CheckEdgeIsInGraph(edge);
        ASSERT_EQ(flag, true);
      }
    }

    auto &out_data_edges = node[i]->GetAllOutDataEdgesRef();
    for (auto edges : out_data_edges) {
      for (auto edge : edges) {
        if (edge != nullptr) {
          auto flag = compute_graph->CheckEdgeIsInGraph(edge);
          ASSERT_EQ(flag, true);
        }
      }
    }

    auto &out_ctl_edges = node[i]->GetAllOutControlEdgesRef();
    for (auto edge : out_ctl_edges) {
      if (edge != nullptr) {
        auto flag = compute_graph->CheckEdgeIsInGraph(edge);
        ASSERT_EQ(flag, true);
      }
    }
  }


  for (int i = 0; i < node_num; ++i) {
    auto &edges = node[i]->GetAllInDataEdgesRef();
    for (auto edge : edges) {
      if (edge != nullptr) {
        auto ret = graph1->MoveEdgeToGraph(edge);
        ASSERT_EQ(ret, GRAPH_SUCCESS);
        auto flag = compute_graph->CheckEdgeIsInGraph(edge);
        ASSERT_EQ(flag, false);
        flag = graph1->CheckEdgeIsInGraph(edge);
        ASSERT_EQ(flag, true);
      }
    }

    auto &ctl_edges = node[i]->GetAllInControlEdgesRef();
    for (auto edge : ctl_edges) {
      if (edge != nullptr) {
        auto ret = graph1->MoveEdgeToGraph(edge);
        ASSERT_EQ(ret, GRAPH_SUCCESS);
        auto flag = compute_graph->CheckEdgeIsInGraph(edge);
        ASSERT_EQ(flag, false);
        flag = graph1->CheckEdgeIsInGraph(edge);
        ASSERT_EQ(flag, true);
      }
    }

    auto &out_data_edges = node[i]->GetAllOutDataEdgesRef();
    for (auto edges : out_data_edges) {
      for (auto edge : edges) {
        if (edge != nullptr) {
          auto ret = graph1->MoveEdgeToGraph(edge);
          ASSERT_EQ(ret, GRAPH_SUCCESS);
          auto flag = compute_graph->CheckEdgeIsInGraph(edge);
          ASSERT_EQ(flag, false);
          flag = graph1->CheckEdgeIsInGraph(edge);
          ASSERT_EQ(flag, true);
        }
      }
    }

    auto &out_ctl_edges = node[i]->GetAllOutControlEdgesRef();
    for (auto edge : out_ctl_edges) {
      if (edge != nullptr) {
        auto ret = graph1->MoveEdgeToGraph(edge);
        ASSERT_EQ(ret, GRAPH_SUCCESS);
        auto flag = compute_graph->CheckEdgeIsInGraph(edge);
        ASSERT_EQ(flag, false);
        flag = graph1->CheckEdgeIsInGraph(edge);
        ASSERT_EQ(flag, true);
      }
    }
  }
}

TEST_F(UtestExecuteGraph, SetEdgeOwnerFail) {
  auto compute_graph = std::make_shared<ge::ExecuteGraph>("graph");
  auto ret = compute_graph->MoveEdgeToGraph(nullptr);
  ASSERT_NE(ret, GRAPH_SUCCESS);
}

TEST_F(UtestExecuteGraph, GetAllNodes_ok_with_filter) {
  auto graph = ExecuteSharedGraph::BuildGraphWithSubGraph();
  const auto &nodes = graph->GetAllNodes();
  EXPECT_EQ(nodes.size(), 9U);
  const auto &filter_func = [](const FastNode *node) ->bool {
    return node->GetType() == "Data";
  };
  const auto &filted_nodes = graph->GetAllNodes(filter_func);
  EXPECT_EQ(filted_nodes.size(), 3U);
}

TEST_F(UtestExecuteGraph, SortNodes_Order_ok) {
  auto builder = ge::ut::ExecuteGraphBuilder("test");
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  auto node1 = builder.AddNode("node1", "Node1", 0, 1);
  auto node2 = builder.AddNode("node2", "Node2", 0, 1);
  auto data2 = builder.AddNode("data2", "Data", 0, 1);
  auto data3 = builder.AddNode("data3", "Data", 0, 1);
  auto node3 = builder.AddNode("node3", "Node3", 0, 1);
  auto data4 = builder.AddNode("data4", "Data", 0, 1);
  auto node4 = builder.AddNode("node4", "Node4", 4, 1);
  builder.AddDataEdge(data1, 0, node4, 0);
  builder.AddDataEdge(data2, 0, node4, 1);
  builder.AddDataEdge(data3, 0, node4, 2);
  builder.AddDataEdge(data4, 0, node4, 3);
  builder.AddControlEdge(node1, node4);
  builder.AddControlEdge(node2, node4);
  builder.AddControlEdge(node3, node4);

  auto graph = builder.GetGraphBeforeTopo();
  ASSERT_NE(graph, nullptr);

  std::vector<FastNode *> stack;
  std::map<FastNode *, uint32_t> map_in_edge_num;
  auto ret = graph->SortNodes(stack, map_in_edge_num);
  ASSERT_EQ(ret, GRAPH_SUCCESS);
  vector<std::string> expect_node_type = {"node3", "node2", "node1", "data4", "data3", "data2", "data1"};
  for (size_t i = 0UL; i < stack.size(); ++i) {
    EXPECT_EQ(stack[i]->GetName(), expect_node_type[i]);
    auto it = map_in_edge_num.find(stack[i]);
    EXPECT_NE(it, map_in_edge_num.end());
    EXPECT_EQ(it->second, 0);
  }
  auto it1 = map_in_edge_num.find(node4);
  EXPECT_NE(it1, map_in_edge_num.end());
  EXPECT_EQ(it1->second, 7); // 4 data edge + 3 control edge
}
}  // namespace ge
