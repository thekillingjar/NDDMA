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
#include "graph/fast_graph/execute_graph.h"
#include "graph/ge_local_context.h"
#include "graph_builder_utils.h"
#include "fast_graph/fast_graph_impl.h"
#include "graph/fast_graph/fast_node.h"
#include "fast_graph/fast_graph_utils.h"

namespace {
std::shared_ptr<ge::ExecuteGraph> BuildDelayTopoGraphMultiInput(
    const std::string &name, std::unordered_map<std::string, ge::FastNode *> &name_to_nodes,
    bool all_is_log_life = true) {
  auto builder = ge::ut::ExecuteGraphBuilder(name);
  const auto &constant = builder.AddNode("const", ge::CONSTANT, 0, 1);
  auto type = ge::CONSTANTOP;
  if (!all_is_log_life) {
    type = "test";
  }
  const auto &constantop = builder.AddNode("constant", type, 0, 1);
  const auto &variable = builder.AddNode("variable", ge::VARIABLE, 0, 2);
  const auto &node1 = builder.AddNode("node1", "node1", 3, 1);
  const auto &node2 = builder.AddNode("node2", "node2", 1, 1);
  const auto &node3 = builder.AddNode("node3", "node3", 1, 1);
  const auto &node4 = builder.AddNode("node4", "node4", 1, 1);
  const auto &node5 = builder.AddNode("node5", "node5", 3, 0);
  const auto &data = builder.AddNode("data", "DATA", 0, 1);

  name_to_nodes.insert(std::make_pair("const", constant));
  name_to_nodes.insert(std::make_pair("constant", constantop));
  name_to_nodes.insert(std::make_pair("variable", variable));
  name_to_nodes.insert(std::make_pair("node1", node1));
  name_to_nodes.insert(std::make_pair("node2", node2));
  name_to_nodes.insert(std::make_pair("node3", node3));
  name_to_nodes.insert(std::make_pair("node4", node4));
  name_to_nodes.insert(std::make_pair("node5", node5));
  name_to_nodes.insert(std::make_pair("data", data));

  int32_t dst_idx = 2;
  builder.AddDataEdge(constant, 0, node1, 0);
  builder.AddDataEdge(constantop, 0, node1, 1);
  builder.AddDataEdge(variable, 0, node1, dst_idx);
  builder.AddDataEdge(variable, 1, node2, 0);
  builder.AddDataEdge(node1, 0, node5, 0);
  builder.AddDataEdge(node2, 0, node5, 1);
  builder.AddDataEdge(data, 0, node3, 0);
  builder.AddDataEdge(node3, 0, node4, 0);
  builder.AddDataEdge(node4, 0, node5, dst_idx);

  builder.AddControlEdge(node2, node3);
  return builder.GetGraph();
}
}  // namespace

namespace ge {
class UtestFastNode : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(UtestFastNode, NodeToken) {
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

  auto token = node[0]->GetNodeToken();
  ASSERT_NE(token, 0);

  auto quick_node = compute_graph->FindNode(token);
  ASSERT_NE(quick_node, nullptr);
}

TEST_F(UtestFastNode, NodeIoOper) {
  auto compute_graph = std::make_shared<ge::ExecuteGraph>("graph");
  int node_num = 1;
  int edge_num = 1;
  OpDescPtr op_desc;
  for (int i = 0; i < node_num; ++i) {
    op_desc = std::make_shared<OpDesc>("op", "op");
    auto td = GeTensorDesc();
    for (int j = 0; j < edge_num; ++j) {
      op_desc->AddInputDesc(td);
      op_desc->AddOutputDesc(td);
    }
    ASSERT_NE(op_desc, nullptr);
  }

  auto node = new FastNode();
  auto res = node->Init(op_desc);
  ASSERT_EQ(res, GRAPH_SUCCESS);
  node->GetExtendInfo()->SetOwnerGraph(compute_graph.get(), node);

  int invalid_num = 10;
  auto ret = node->GetOutEdgesByIndex(invalid_num);
  ASSERT_EQ(ret.size(), 0);

  ret = node->GetAllInControlEdgesRef();
  ASSERT_EQ(ret.size(), 0);

  ret = node->GetAllInControlEdges();
  ASSERT_EQ(ret.size(), 0);

  ret = node->GetOutEdgesRefByIndex(invalid_num);
  ASSERT_EQ(ret.size(), 0);

  auto size = node->GetDataOutNum();
  ASSERT_EQ(size, 1);

  size = node->GetDataInNum();
  ASSERT_EQ(size, 1);

  int new_num = 2;
  node->UpdateDataInNum(new_num);
  size = node->GetDataInNum();
  ASSERT_EQ(size, new_num);

  node->UpdateDataOutNum(new_num);
  size = node->GetDataOutNum();
  ASSERT_EQ(size, new_num);

  delete node;
}

TEST_F(UtestFastNode, GetEdgesOfNode) {
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
    edge[i] = compute_graph->AddEdge(node[i - 1], 0, node[i], 0);
    ASSERT_NE(edge[i], nullptr);
    auto edge = compute_graph->AddEdge(node[i - 1], -1, node[i], -1);
    ASSERT_NE(edge, nullptr);
  }

  auto edges = node[1]->GetAllInDataEdges();
  auto size = node[1]->GetAllInDataEdgesSize();
  ASSERT_EQ(edges.size(), size);

  edges = node[1]->GetAllInControlEdges();
  size = node[1]->GetAllInControlEdgesSize();
  ASSERT_EQ(edges.size(), size);

  edges = node[1]->GetAllOutControlEdges();
  size = node[1]->GetAllOutControlEdgesSize();
  ASSERT_EQ(edges.size(), size);

  edges = node[1]->GetAllOutDataEdges();
  size = node[1]->GetAllOutDataEdgesSize();
  ASSERT_EQ(edges.size(), size);
}

TEST_F(UtestFastNode, NodePtr) {
  auto graph = std::make_shared<ComputeGraph>("graph");
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetDataType(DT_FLOAT);
  tensor_desc->SetFormat(FORMAT_CHWN);
  auto op_desc1 = std::make_shared<OpDesc>("add_front", "add_front");
  op_desc1->AddInputDesc(tensor_desc->Clone());
  auto nodeptr = graph->AddNodeFront(op_desc1);
  ASSERT_NE(nodeptr, nullptr);

  auto compute_graph = std::make_shared<ge::ExecuteGraph>("graph");
  FastNode *fast_node = nullptr;
  fast_node = compute_graph->AddNode(op_desc1);
  ASSERT_NE(fast_node, nullptr);

  fast_node->SetNodePtr(nodeptr);
  auto node = fast_node->GetNodePtr();
  ASSERT_EQ(node, nodeptr);

  auto node_ref = fast_node->GetNodePtr();
  ASSERT_EQ(node_ref, nodeptr);

  fast_node->ClearNodeBarePtr();
  auto clear_node = fast_node->GetNodeBarePtr();
  ASSERT_EQ(clear_node, nullptr);

  fast_node->ClearNodePtr();
  auto clear_node_ptr = fast_node->GetNodePtr();
  ASSERT_EQ(clear_node_ptr, nullptr);
}

TEST_F(UtestFastNode, RemoveEdgeFunc) {
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
    edge[i] = compute_graph->AddEdge(node[i - 1], 0, node[i], 0);
    ASSERT_NE(edge[i], nullptr);
  }

  FastGraphUtils::GetListElementAddr(node[1])->owner->erase(FastGraphUtils::GetListElementAddr(node[1]));
  node[1]->RemoveAllEdge([&compute_graph](FastEdge *e) {
    if (e->src != nullptr) {
      e->src->EraseEdge(e, DirectionType::kDirectionOutType);
      e->src = nullptr;
    }

    if (e->dst != nullptr) {
      e->dst->EraseEdge(e, DirectionType::kDirectionInType);
      e->dst = nullptr;
    }

    if (FastGraphUtils::GetListElementAddr(e)->owner != nullptr) {
      FastGraphUtils::GetListElementAddr(e)->owner->erase(FastGraphUtils::GetListElementAddr(e));
    }
    auto ret = compute_graph->RecycleQuickEdge(e);
    if ((ret != GRAPH_SUCCESS) && (e != nullptr)) {
      delete e;
    }
  });
  auto ret = compute_graph->RecycleQuickNode(node[1]);
  if ((ret != GRAPH_SUCCESS) && (node[1] != nullptr)) {
    delete node[1];
  }
  ASSERT_EQ(compute_graph->GetDirectNodesSize(), node_num - 1);
}

TEST_F(UtestFastNode, TestNodeError) {
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

  FastEdge *edge = compute_graph->AddEdge(node[0], 0, node[1], 0);
  ASSERT_NE(edge, nullptr);

  FastEdge *fail_edge = compute_graph->AddEdge(node[0], 0, node[1], 0);
  ASSERT_EQ(fail_edge, nullptr);

  {
    FastEdge *edge = compute_graph->AddEdge(nullptr, 0, nullptr, 0);
    ASSERT_EQ(edge, nullptr);

    int invalid_num = 10;
    edge = compute_graph->AddEdge(node[0], 0, node[1], invalid_num);
    ASSERT_EQ(edge, nullptr);

    edge = compute_graph->AddEdge(node[0], invalid_num, node[1], 0);
    ASSERT_EQ(edge, nullptr);
  }

  {
    auto compute_graph2 = std::make_shared<ge::ExecuteGraph>("graph2");
    auto ret = compute_graph2->RemoveEdge(edge);
    ASSERT_NE(ret, GRAPH_SUCCESS);
  }

  {
    auto ret = compute_graph->RemoveEdge(nullptr);
    ASSERT_NE(ret, GRAPH_SUCCESS);
  }

  {
    int32_t invalid_num = 10;
    auto size = node[0]->GetInEdgesSizeByIndex(invalid_num);
    ASSERT_EQ(size, 0);
    size = node[0]->GetOutEdgesSizeByIndex(invalid_num);
    ASSERT_EQ(size, 0);
    auto ret = node[0]->GetInDataEdgeByIndex(invalid_num);
    ASSERT_EQ(ret, nullptr);
  }

  {
    auto graph = std::make_shared<ge::ExecuteGraph>("graph");
    int node_num = 2;
    int edge_num = 5;
    FastNode *node[node_num] = {};
    {
      auto op_desc = std::make_shared<OpDesc>("op", "op");
      auto td = GeTensorDesc();
      for (int j = 0; j < edge_num; ++j) {
        op_desc->AddOutputDesc(td);
      }
      node[0] = graph->AddNode(op_desc);
      ASSERT_NE(node[0], nullptr);
    }

    {
      auto op_desc = std::make_shared<OpDesc>("op", "op");
      auto td = GeTensorDesc();
      for (int j = 0; j < edge_num; ++j) {
        op_desc->AddInputDesc(td);
      }
      node[1] = graph->AddNode(op_desc);
      ASSERT_NE(node[1], nullptr);
    }

    FastEdge *edge = graph->AddEdge(node[0], 1, node[1], 1);
    ASSERT_NE(edge, nullptr);

    auto size = node[0]->GetOutEdgesSizeByIndex(1);
    ASSERT_EQ(size, 1);
  }

  {
    auto no_init_node = new FastNode();
    auto nodes = no_init_node->GetOutDataNodesByIndex(0);
    ASSERT_EQ(nodes.size(), 0);
    nodes = no_init_node->GetOutControlNodes();
    ASSERT_EQ(nodes.size(), 0);
    delete no_init_node;
  }

  {
    auto ret = compute_graph->AddNodeFront(nullptr);
    ASSERT_EQ(ret, nullptr);
  }
}

TEST_F(UtestFastNode, TestAddNodeWithNode) {
  int node_num = 10;
  int io_num = 5;
  FastNode *node[node_num] = {};
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

  {
    auto root_graph2 = std::make_shared<ExecuteGraph>("root_graph2");
    auto root_graph = std::make_shared<ExecuteGraph>("root_graph");

    node[0] = root_graph2->AddNode(op_desc[0]);
    ASSERT_NE(node[0], nullptr);

    auto ret = root_graph2->RemoveJustNode(node[0]);
    ASSERT_EQ(ret, GRAPH_SUCCESS);

    node[0] = root_graph->AddNode(node[0]);
    ASSERT_NE(node[0], nullptr);

    ret = root_graph->RemoveJustNode(node[0]);
    ASSERT_EQ(ret, GRAPH_SUCCESS);
  }

  {
    auto root_graph2 = std::make_shared<ExecuteGraph>("root_graph2");
    auto root_graph = std::make_shared<ExecuteGraph>("root_graph");

    node[1] = root_graph2->AddNode(op_desc[1]);
    ASSERT_NE(node[1], nullptr);
    ASSERT_NE(node[1]->GetExtendInfo(), nullptr);

    auto ret = root_graph2->RemoveJustNode(node[1]);
    ASSERT_EQ(ret, GRAPH_SUCCESS);

    node[1]->GetExtendInfo()->SetOwnerGraph(root_graph.get(), node[1]);

    node[1] = root_graph->AddNode(node[1]);
    ASSERT_NE(node[1], nullptr);
  }

  {
    auto root_graph2 = std::make_shared<ExecuteGraph>("root_graph2");
    auto root_graph = std::make_shared<ExecuteGraph>("root_graph");
    int32_t node_idx = 2;

    node[node_idx] = root_graph2->AddNode(op_desc[node_idx]);
    ASSERT_NE(node[node_idx], nullptr);
    ASSERT_NE(node[node_idx]->GetExtendInfo(), nullptr);

    node[node_idx] = root_graph->AddNodeFront(node[node_idx]);
    ASSERT_NE(node[node_idx], nullptr);

    auto ret = root_graph2->RemoveJustNode(node[node_idx]);
    ASSERT_EQ(ret, GRAPH_NOT_CHANGED);

    node[node_idx]->GetExtendInfo()->SetOwnerGraph(root_graph.get(), node[node_idx]);
  }

  {
    auto root_graph = std::make_shared<ExecuteGraph>("root_graph");
    auto ret = root_graph->RemoveJustNode(nullptr);
    ASSERT_NE(ret, GRAPH_SUCCESS);
  }
}

TEST_F(UtestFastNode, ReorderByNodeId) {
  std::unordered_map<std::string, ge::FastNode *> name_to_nodes;
  auto graph = BuildDelayTopoGraphMultiInput("test_delay_topo_graph", name_to_nodes);

  auto iter = name_to_nodes.find("const");
  auto constant = iter->second;

  iter = name_to_nodes.find("constant");
  auto constantop = iter->second;

  iter = name_to_nodes.find("variable");
  auto variable = iter->second;

  iter = name_to_nodes.find("node1");
  auto node1 = iter->second;

  iter = name_to_nodes.find("node2");
  auto node2 = iter->second;

  iter = name_to_nodes.find("node3");
  auto node3 = iter->second;

  iter = name_to_nodes.find("node4");
  auto node4 = iter->second;

  iter = name_to_nodes.find("node5");
  auto node5 = iter->second;

  iter = name_to_nodes.find("data");
  auto data = iter->second;

  int64_t seq_id = 0L;
  std::vector<ge::FastNode *> nodes{node5, node4, node3, node2, node1, variable, data, constantop, constant};
  for (auto &node : nodes) {
    node->GetOpDescBarePtr()->SetId(seq_id++);
  }
  graph->ReorderByNodeId();
  auto sorted_nodes = graph->GetDirectNode();
  ASSERT_TRUE(sorted_nodes.size() == nodes.size());
  int32_t id = 0;
  for (auto &node : nodes) {
    EXPECT_EQ(node, sorted_nodes.at(id++));
  }
}

TEST_F(UtestFastNode, TestNodeCheckAllInputParamter) {
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
    int32_t idx = -2;
    auto ret = node[0]->MoveEdge(DirectionType::kDirectionOutType, idx, 0, 0);
    ASSERT_NE(ret, GRAPH_SUCCESS);
  }

  {
    int32_t idx = 10;
    auto ret = node[0]->MoveEdge(DirectionType::kDirectionOutType, idx, 0, 0);
    ASSERT_NE(ret, GRAPH_SUCCESS);
  }

  {
    int32_t idx = 0;
    int invalid_curr_idx = 10;
    auto ret = node[0]->MoveEdge(DirectionType::kDirectionOutType, idx, invalid_curr_idx, 0);
    ASSERT_NE(ret, GRAPH_SUCCESS);
  }

  {
    int32_t idx = 0;
    int invalid_replace_idx = 10;
    auto ret = node[0]->MoveEdge(DirectionType::kDirectionOutType, idx, 0, invalid_replace_idx);
    ASSERT_NE(ret, GRAPH_SUCCESS);
  }

  {
    FastEdge *edge = compute_graph->AddEdge(node[0], -1, node[1], -1);
    ASSERT_NE(edge, nullptr);

    edge = compute_graph->AddEdge(node[0], -1, node[2], -1);
    ASSERT_NE(edge, nullptr);

    auto ret = compute_graph->RemoveEdge(edge);
    ASSERT_EQ(ret, GRAPH_SUCCESS);

    edge = compute_graph->AddEdge(node[1], -1, node[2], -1);
    ASSERT_NE(edge, nullptr);

    ret = node[0]->MoveEdge(DirectionType::kDirectionOutType, -1, 0, 1);
    ASSERT_EQ(ret, GRAPH_SUCCESS);

    ret = node[2]->MoveEdge(DirectionType::kDirectionInType, -1, 0, 1);
    ASSERT_EQ(ret, GRAPH_SUCCESS);
  }
}

TEST_F(UtestFastNode, other) {
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
    ASSERT_NE(node[i]->GetExtendInfo(), nullptr);
    ASSERT_NE(node[i]->GetExtendInfo()->GetHostNode(), true);
  }

  node[0]->UpdateOpDesc(nullptr);
  ASSERT_EQ(node[0]->GetOpDescPtr(), nullptr);

  auto op_desc_test = std::make_shared<OpDesc>("test", "test");
  node[0]->UpdateOpDesc(op_desc_test);
  ASSERT_EQ(node[0]->GetOpDescPtr(), op_desc_test);
}

TEST_F(UtestFastNode, TestSymbol) {
  auto exe_graph = std::make_shared<ge::ExecuteGraph>("graph");
  size_t data_num = 2;
  auto op_desc = std::make_shared<OpDesc>("op", "op");
  auto td = GeTensorDesc();
  for (size_t j = 0; j < data_num; ++j) {
    op_desc->AddInputDesc(td);
    op_desc->AddOutputDesc(td);
  }
  const auto node = exe_graph->AddNode(op_desc);
  ASSERT_NE(node, nullptr);

  ASSERT_EQ(node->GetExtendInfo()->SetInputSymbol(0, 0), GRAPH_SUCCESS);
  ASSERT_EQ(node->GetExtendInfo()->SetInputSymbol(1, 1), GRAPH_SUCCESS);
  ASSERT_EQ(node->GetExtendInfo()->SetInputSymbol(2, 2), GRAPH_FAILED);

  ASSERT_EQ(node->GetExtendInfo()->SetOutputSymbol(0, 0), GRAPH_SUCCESS);
  ASSERT_EQ(node->GetExtendInfo()->SetOutputSymbol(1, 1), GRAPH_SUCCESS);
  ASSERT_EQ(node->GetExtendInfo()->SetOutputSymbol(2, 2), GRAPH_FAILED);

  ASSERT_EQ(node->GetExtendInfo()->GetInputSymbol(0), 0);
  ASSERT_EQ(node->GetExtendInfo()->GetInputSymbol(1), 1);
  ASSERT_EQ(node->GetExtendInfo()->GetInputSymbol(2), kInvalidSymbol);

  ASSERT_EQ(node->GetExtendInfo()->GetOutputSymbol(0), 0);
  ASSERT_EQ(node->GetExtendInfo()->GetOutputSymbol(1), 1);
  ASSERT_EQ(node->GetExtendInfo()->GetOutputSymbol(2), kInvalidSymbol);

  node->UpdateDataInNum(3);
  node->UpdateDataOutNum(3);
  ASSERT_EQ(node->GetExtendInfo()->SetInputSymbol(2, 2), GRAPH_SUCCESS);
  ASSERT_EQ(node->GetExtendInfo()->SetOutputSymbol(2, 2), GRAPH_SUCCESS);
  ASSERT_EQ(node->GetExtendInfo()->GetInputSymbol(2), 2);
  ASSERT_EQ(node->GetExtendInfo()->GetOutputSymbol(2), 2);
}
}  // namespace ge
