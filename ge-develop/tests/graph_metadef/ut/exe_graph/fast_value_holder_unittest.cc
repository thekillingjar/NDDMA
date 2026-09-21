/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "exe_graph/lowering/value_holder.h"
#include "exe_graph/lowering/value_holder_utils.h"
#include "exe_graph/runtime/context_extend.h"
#include <gtest/gtest.h>
#include <cstdint>
#include <numeric>
#include "exe_graph/lowering/exe_graph_attrs.h"
#include "checker/bg_test.h"
#include "graph/utils/graph_utils.h"
#include "checker/topo_checker.h"
#include "checker/summary_checker.h"
#include "exe_graph/lowering/extend_exe_graph.h"
#include "graph/fast_graph/execute_graph.h"
#include "graph/utils/execute_graph_utils.h"

namespace gert {
namespace bg {
namespace {
ge::NodePtr FakeNode() {
  static size_t counter = 0;
  static ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph");
  auto op_desc = std::make_shared<ge::OpDesc>("FakeNode_" + std::to_string(counter++), "FakeNode");
  return graph->AddNode(op_desc);
}
size_t GetComputeNodeIndex(const ge::FastNode *node) {
  int64_t index;
  if (!ge::AttrUtils::GetInt(node->GetOpDescBarePtr(), kComputeNodeIndex, index)) {
    return std::numeric_limits<size_t>::max();
  }
  return static_cast<size_t>(index);
}
}
class FastValueHolderUt : public BgTest {
 public:
  ge::ExecuteGraph *FindFirstSubgraphForNodeType(const ge::ExecuteGraph *root_graph,
                                                 const std::string &node_type) {
    for (const auto &subgraph : root_graph->GetAllSubgraphs()) {
      auto parent_node = subgraph->GetParentNodeBarePtr();
      if (parent_node->GetType() == node_type) {
        return subgraph;
      }
    }
    return nullptr;
  }
  ge::FastNode *FindData(const ge::ExecuteGraph *graph, int32_t index) {
    for (const auto &node : graph->GetDirectNode()) {
      if (node->GetType() != "Data") {
        continue;
      }
      int32_t data_index;
      if (!ge::AttrUtils::GetInt(node->GetOpDescBarePtr(), "index", data_index)) {
        continue;
      }
      if (data_index != index) {
        continue;
      }
      return node;
    }
    return nullptr;
  }

  void ConnectFromInnerData(const ge::FastNode *node, const std::vector<int32_t> &indexes) {
    ASSERT_EQ(node->GetInDataNodes().size(), indexes.size());
    size_t i = 0;
    for (const auto &in_node : node->GetInDataNodes()) {
      ASSERT_EQ(in_node->GetType(), "InnerData");
      int32_t index;
      ASSERT_TRUE(ge::AttrUtils::GetInt(in_node->GetOpDescBarePtr(), "index", index));
      ASSERT_EQ(index, indexes[i++]);
    }
  }
  void ConnectFromOuter(ge::FastNode *node, int32_t dst_index, const ge::FastNode *outer, int32_t src_index) {
    while (true) {
      auto edge = node->GetInDataEdgeByIndex(dst_index);
      ASSERT_NE(edge, nullptr);
      auto src_node = edge->src;
      ASSERT_NE(src_node, nullptr);
      if (src_node == outer) {
        return;
      }
      if (src_node->GetType() == "InnerData" || src_node->GetType() == "Data") {
        int32_t parent_index;
        ASSERT_TRUE(ge::AttrUtils::GetInt(src_node->GetOpDescBarePtr(), "index", parent_index));
        auto parent_graph = src_node->GetExtendInfo()->GetOwnerGraphBarePtr();
        ASSERT_NE(parent_graph, nullptr);
        auto parent_node = parent_graph->GetParentNodeBarePtr();
        ASSERT_NE(parent_node, nullptr);
        node = const_cast<ge::FastNode *>(parent_node);
        dst_index = parent_index;
      }
    }
  }
  void StrictSubgraphs(const ge::FastNode *node, const std::vector<std::string> &names) {
    const auto &subgraph_names_to_index = node->GetOpDescBarePtr()->GetSubgraphNameIndexes();

    ASSERT_EQ(subgraph_names_to_index.size(), names.size());
    for (const auto &name : names) {
      auto iter = subgraph_names_to_index.find(name);
      ASSERT_NE(iter, subgraph_names_to_index.end());
      auto ins_name = node->GetOpDescBarePtr()->GetSubgraphInstanceName(iter->second);
      auto root_graph = ge::ExecuteGraphUtils::FindRootGraph(node->GetExtendInfo()->GetOwnerGraphBarePtr());
      ASSERT_NE(root_graph->GetSubGraph(ins_name), nullptr);
    }
  }
};

TEST_F(FastValueHolderUt, CreateConstOk) {
  ge::Format f1 = ge::FORMAT_NC1HWC0;
  auto c = ValueHolder::CreateConst(reinterpret_cast<const uint8_t *>(&f1), sizeof(f1));
  EXPECT_NE(c, nullptr);
  ASSERT_TRUE(c->IsOk());
  ASSERT_NE(c->GetFastNode(), nullptr);
  EXPECT_EQ(c->GetType(), ValueHolder::ValueHolderType::kConst);
  EXPECT_EQ(c->GetOutIndex(), 0);
  auto node = c->GetFastNode();
  EXPECT_EQ(node->GetType(), "Const");
  EXPECT_EQ(node->GetDataOutNum(), 1);
  EXPECT_EQ(node->GetDataInNum(), 0);
  ge::Buffer buffer;
  ASSERT_TRUE(ge::AttrUtils::GetZeroCopyBytes(node->GetOpDescBarePtr(), "value", buffer));
  ASSERT_EQ(buffer.GetSize(), sizeof(ge::FORMAT_NC1HWC0));
  EXPECT_EQ(*reinterpret_cast<ge::Format *>(buffer.GetData()), ge::FORMAT_NC1HWC0);
}

TEST_F(FastValueHolderUt, CreateInnerOk) {
  auto inner_data_holder = bg::ValueHolder::CreateSingleDataOutput("InnerData", {});
  EXPECT_NE(inner_data_holder, nullptr);
  ASSERT_TRUE(inner_data_holder->IsOk());
  ASSERT_NE(inner_data_holder->GetFastNode(), nullptr);
  EXPECT_EQ(inner_data_holder->GetType(), ValueHolder::ValueHolderType::kOutput);
  EXPECT_EQ(inner_data_holder->GetOutIndex(), 0);
  auto node = inner_data_holder->GetFastNode();
  EXPECT_EQ(node->GetType(), "InnerData");
  EXPECT_EQ(node->GetDataOutNum(), 1);
  EXPECT_EQ(node->GetDataInNum(), 0);
  EXPECT_EQ(inner_data_holder->AddInnerDataToKVMap(0).IsSuccess(), true);
  ge::FastNode *data = nullptr;
  bool ret = FindValFromMapExtAttr<int32_t, ge::FastNode *>(node->GetExtendInfo()->GetOwnerGraphBarePtr(),
                                                            "_inner_data_nodes", 0, data);
  EXPECT_EQ(ret, true);
  EXPECT_EQ(data, node);
}

TEST_F(FastValueHolderUt, CreateInnerFailed) {
  auto inner_data_holder = bg::ValueHolder::CreateSingleDataOutput("InnerData1", {});
  EXPECT_NE(inner_data_holder, nullptr);
  ASSERT_TRUE(inner_data_holder->IsOk());
  ASSERT_NE(inner_data_holder->GetFastNode(), nullptr);
  EXPECT_EQ(inner_data_holder->AddInnerDataToKVMap(0).IsSuccess(), false);
}

TEST_F(FastValueHolderUt, CreateVectorConstOk) {
  int64_t shape[5] = {32, 1, 224, 224, 16};
  auto c = ValueHolder::CreateConst(reinterpret_cast<const uint8_t *>(shape), sizeof(shape));
  EXPECT_NE(c, nullptr);
  ASSERT_TRUE(c->IsOk());
  ASSERT_NE(c->GetFastNode(), nullptr);
  EXPECT_EQ(c->GetType(), ValueHolder::ValueHolderType::kConst);
  EXPECT_EQ(c->GetOutIndex(), 0);
  auto node = c->GetFastNode();
  EXPECT_EQ(node->GetType(), "Const");
  ge::Buffer buffer;
  ASSERT_TRUE(ge::AttrUtils::GetZeroCopyBytes(node->GetOpDescBarePtr(), "value", buffer));
  ASSERT_EQ(buffer.GetSize(), sizeof(shape));
  EXPECT_EQ(memcmp(buffer.GetData(), shape, sizeof(shape)), 0);
}
TEST_F(FastValueHolderUt, CreateFeedOk) {
  auto c = ValueHolder::CreateFeed(1);
  EXPECT_NE(c, nullptr);
  ASSERT_TRUE(c->IsOk());
  ASSERT_NE(c->GetFastNode(), nullptr);
  EXPECT_EQ(c->GetType(), ValueHolder::ValueHolderType::kFeed);
  EXPECT_EQ(c->GetOutIndex(), 0);
  auto node = c->GetFastNode();
  EXPECT_EQ(node->GetType(), "Data");
  EXPECT_EQ(node->GetDataOutNum(), 1);
  EXPECT_EQ(node->GetDataInNum(), 0);
  int32_t index;
  ASSERT_TRUE(ge::AttrUtils::GetInt(node->GetOpDescBarePtr(), "index", index));
  EXPECT_EQ(index, 1);
}
TEST_F(FastValueHolderUt, CreateErrorOk) {
  auto holder = ValueHolder::CreateError("This is a test error information, int %d, %s", 10240, "Test msg");
  ASSERT_NE(holder, nullptr);
  EXPECT_FALSE(holder->IsOk());
}
TEST_F(FastValueHolderUt, CreateDataOutOk) {
  ge::Format f1 = ge::FORMAT_NC1HWC0;
  auto const1 = ValueHolder::CreateConst(reinterpret_cast<const uint8_t *>(&f1), sizeof(f1));
  auto data1 = ValueHolder::CreateFeed(0);
  ASSERT_NE(const1, nullptr);
  ASSERT_NE(data1, nullptr);

  std::vector<ValueHolderPtr> inputs = {data1, const1};
  auto holders = ValueHolder::CreateDataOutput("TestNode", inputs, 3);

  ASSERT_EQ(holders.size(), 3);
  ASSERT_TRUE(holders[0]->IsOk());
  ASSERT_TRUE(holders[1]->IsOk());
  ASSERT_TRUE(holders[2]->IsOk());
  EXPECT_EQ(holders[0]->GetType(), ValueHolder::ValueHolderType::kOutput);
  EXPECT_EQ(holders[1]->GetType(), ValueHolder::ValueHolderType::kOutput);
  EXPECT_EQ(holders[2]->GetType(), ValueHolder::ValueHolderType::kOutput);

  ASSERT_NE(const1->GetExecuteGraph(), nullptr);
  ASSERT_NE(const1->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr(), nullptr);
  ASSERT_NE(data1->GetExecuteGraph(), nullptr);
  ASSERT_NE(data1->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr(), nullptr);
  ASSERT_NE(holders[0]->GetFastNode(), nullptr);
  ASSERT_NE(holders[0]->GetExecuteGraph(), nullptr);
  ASSERT_NE(holders[0]->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr(), nullptr);
  ASSERT_NE(holders[1]->GetFastNode(), nullptr);
  ASSERT_NE(holders[1]->GetExecuteGraph(), nullptr);
  ASSERT_NE(holders[1]->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr(), nullptr);
  ASSERT_NE(holders[2]->GetFastNode(), nullptr);
  ASSERT_NE(holders[2]->GetExecuteGraph(), nullptr);
  ASSERT_NE(holders[2]->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr(), nullptr);

  // check node is ok
  auto node = holders[0]->GetFastNode();
  ASSERT_EQ(node->GetType(), "TestNode");
  ASSERT_EQ(node->GetDataInNum(), 2);
  ASSERT_EQ(node->GetDataOutNum(), 3);

  // all holders point to the same node
  ASSERT_EQ(holders[0]->GetFastNode(), holders[1]->GetFastNode());
  ASSERT_EQ(holders[0]->GetFastNode(), holders[2]->GetFastNode());

  // all holders have correct out-index
  EXPECT_EQ(holders[0]->GetOutIndex(), 0);
  EXPECT_EQ(holders[1]->GetOutIndex(), 1);
  EXPECT_EQ(holders[2]->GetOutIndex(), 2);

  // all nodes(contains data and const) in the same graph
  EXPECT_EQ(holders[0]->GetExecuteGraph(), const1->GetExecuteGraph());
  EXPECT_EQ(holders[0]->GetExecuteGraph(), data1->GetExecuteGraph());

  // all holders holds the same graph
  EXPECT_EQ(holders[0]->GetExecuteGraph(), holders[1]->GetExecuteGraph());
  EXPECT_EQ(holders[0]->GetExecuteGraph(), holders[2]->GetExecuteGraph());
  EXPECT_EQ(holders[0]->GetExecuteGraph(), const1->GetExecuteGraph());
  EXPECT_EQ(holders[0]->GetExecuteGraph(), data1->GetExecuteGraph());

  // check graph is ok
  auto graph = holders[0]->GetExecuteGraph();
  ASSERT_EQ(graph->GetAllNodes().size(), 3);
  CheckGraphGenerally(graph);
  auto const1_g = ge::ExecuteGraphUtils::FindFirstNodeMatchType(graph, "Const");
  auto data1_g = ge::ExecuteGraphUtils::FindFirstNodeMatchType(graph, "Data");
  auto node_g = ge::ExecuteGraphUtils::FindFirstNodeMatchType(graph, "TestNode");
  ASSERT_NE(const1_g, nullptr);
  ASSERT_NE(data1_g, nullptr);
  ASSERT_NE(node_g, nullptr);

  EXPECT_EQ(node_g->GetInDataEdgeByIndex(0)->src, data1_g);
  EXPECT_EQ(node_g->GetInDataEdgeByIndex(1)->src, const1_g);
}

/*
 *            --------> node3
 *          /           /   \
 *     node1        node2   const3
 *     / \          /   \
 * data1 const1  data2 const2
 */
TEST_F(FastValueHolderUt, CreateDataOutOk2) {
  ge::Format fmt = ge::FORMAT_NC1HWC0;
  auto const1 = ValueHolder::CreateConst(&fmt, sizeof(fmt));
  auto data1 = ValueHolder::CreateFeed(0);
  auto node1 = ValueHolder::CreateSingleDataOutput("Node1", {const1, data1});

  auto const2 = ValueHolder::CreateConst(&fmt, sizeof(fmt));
  auto data2 = ValueHolder::CreateFeed(0);
  auto node2 = ValueHolder::CreateSingleDataOutput("Node1", {const2, data2});

  auto const3 = ValueHolder::CreateConst(&fmt, sizeof(fmt));
  auto n2_holder = ValueHolder::CreateVoid<ValueHolder>("Node2", {node1, node2, const3});

  for (const auto &holder : {const1, data1, node1, const2, data2, node2, const3, n2_holder}) {
    ASSERT_NE(holder, nullptr);
    ASSERT_TRUE(holder->IsOk());
    ASSERT_NE(holder->GetFastNode(), nullptr);
    ASSERT_NE(holder->GetExecuteGraph(), nullptr);
  }
  EXPECT_EQ(node1->GetFastNode()->GetType(), "Node1");
  EXPECT_EQ(node2->GetFastNode()->GetType(), "Node1");
  EXPECT_EQ(n2_holder->GetFastNode()->GetType(), "Node2");

  ASSERT_EQ(node1->GetFastNode()->GetDataOutNum(), 1);
  ASSERT_EQ(node1->GetFastNode()->GetDataInNum(), 2);
  EXPECT_EQ(node1->GetFastNode()->GetInDataEdgeByIndex(0)->src, const1->GetFastNode());
  EXPECT_EQ(node1->GetFastNode()->GetInDataEdgeByIndex(1)->src, data1->GetFastNode());

  ASSERT_EQ(n2_holder->GetFastNode()->GetDataOutNum(), 0);
  ASSERT_EQ(n2_holder->GetFastNode()->GetDataInNum(), 3);
  EXPECT_EQ(n2_holder->GetFastNode()->GetInDataEdgeByIndex(0)->src, node1->GetFastNode());
  EXPECT_EQ(n2_holder->GetFastNode()->GetInDataEdgeByIndex(1)->src, node2->GetFastNode());
  EXPECT_EQ(n2_holder->GetFastNode()->GetInDataEdgeByIndex(2)->src, const3->GetFastNode());
}
TEST_F(FastValueHolderUt, MergeIsolateNodeToGraphOk) {
  ge::Format f1 = ge::FORMAT_NC1HWC0;
  auto const1 = ValueHolder::CreateConst(reinterpret_cast<const uint8_t *>(&f1), sizeof(f1));
  auto data1 = ValueHolder::CreateFeed(0);
  auto node1 = ValueHolder::CreateDataOutput("Node1", {data1, const1}, 2);
  ASSERT_NE(const1, nullptr);
  ASSERT_NE(data1, nullptr);
  ASSERT_EQ(node1.size(), 2);
  ASSERT_NE(node1[0], nullptr);
  ASSERT_NE(node1[1], nullptr);

  ASSERT_NE(const1->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr(), nullptr);
  EXPECT_EQ(data1->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr(),
            const1->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr());
  EXPECT_EQ(node1[0]->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr(),
            const1->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr());
  EXPECT_EQ(node1[1]->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr(),
            const1->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr());

  ASSERT_NE(const1->GetExecuteGraph(), nullptr);
  EXPECT_EQ(const1->GetExecuteGraph(), const1->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr());
  EXPECT_EQ(data1->GetExecuteGraph(), const1->GetExecuteGraph());
  EXPECT_EQ(node1[0]->GetExecuteGraph(), const1->GetExecuteGraph());
  EXPECT_EQ(node1[1]->GetExecuteGraph(), const1->GetExecuteGraph());
}

/*
 *
 *           node3
 *          /     \   |
 *     node1       node2
 *     / \         /   \
 * data1 const1  data2 const2
 */
TEST_F(FastValueHolderUt, MergeTwoGraphOk1) {
  ge::Format f1 = ge::FORMAT_NC1HWC0;
  auto const1 = ValueHolder::CreateConst(reinterpret_cast<const uint8_t *>(&f1), sizeof(f1));
  auto data1 = ValueHolder::CreateFeed(0);
  auto node1 = ValueHolder::CreateDataOutput("Node1", {data1, const1}, 1);

  auto const2 = ValueHolder::CreateConst(reinterpret_cast<const uint8_t *>(&f1), sizeof(f1));
  auto data2 = ValueHolder::CreateFeed(0);
  auto node2 = ValueHolder::CreateDataOutput("Node2", {data2, const2}, 2);

  auto node3 = ValueHolder::CreateSingleDataOutput("Node3", {node1[0], node2[0]});
  ASSERT_NE(node3, nullptr);

  EXPECT_NE(data1->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr(), nullptr);
  EXPECT_NE(const1->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr(), nullptr);
  EXPECT_NE(node1[0]->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr(), nullptr);

  EXPECT_NE(data2->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr(), nullptr);
  EXPECT_NE(const2->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr(), nullptr);
  EXPECT_NE(node2[0]->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr(), nullptr);
  EXPECT_NE(node2[1]->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr(), nullptr);

  EXPECT_NE(node3->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr(), nullptr);

  ASSERT_NE(const1->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr(), nullptr);
  EXPECT_EQ(const1->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr()->GetAllNodes().size(), 7);

  EXPECT_EQ(const1->GetExecuteGraph(), data1->GetExecuteGraph());
  EXPECT_EQ(const1->GetExecuteGraph(), node1[0]->GetExecuteGraph());
  EXPECT_EQ(const1->GetExecuteGraph(), data2->GetExecuteGraph());
  EXPECT_EQ(const1->GetExecuteGraph(), const2->GetExecuteGraph());
  EXPECT_EQ(const1->GetExecuteGraph(), node2[0]->GetExecuteGraph());
  EXPECT_EQ(const1->GetExecuteGraph(), node2[1]->GetExecuteGraph());
  EXPECT_EQ(const1->GetExecuteGraph(), node3->GetExecuteGraph());
}
/*
 *
 *                node4
 *               /   \
 *           node3    \
 *          /     \   |
 *     node1       node2
 *     / \         /   \
 * data1 const1  data2 const2
 */
TEST_F(FastValueHolderUt, MergeTwoGraphOk) {
  ge::Format f1 = ge::FORMAT_NC1HWC0;
  auto const1 = ValueHolder::CreateConst(reinterpret_cast<const uint8_t *>(&f1), sizeof(f1));
  auto data1 = ValueHolder::CreateFeed(0);
  auto node1 = ValueHolder::CreateDataOutput("Node1", {data1, const1}, 1);
  ASSERT_NE(const1, nullptr);
  ASSERT_NE(data1, nullptr);
  ASSERT_EQ(node1.size(), 1);
  ASSERT_NE(node1[0], nullptr);
  ASSERT_NE(const1->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr(), nullptr);

  auto const2 = ValueHolder::CreateConst(reinterpret_cast<const uint8_t *>(&f1), sizeof(f1));
  auto data2 = ValueHolder::CreateFeed(0);
  auto node2 = ValueHolder::CreateDataOutput("Node2", {data2, const2}, 2);
  ASSERT_NE(const2, nullptr);
  ASSERT_NE(data2, nullptr);
  ASSERT_EQ(node2.size(), 2);
  ASSERT_NE(node2[0], nullptr);
  ASSERT_NE(node2[1], nullptr);

  EXPECT_NE(const2->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr(), nullptr);
  EXPECT_NE(node2[0]->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr(), nullptr);
  EXPECT_NE(node2[1]->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr(), nullptr);

  auto node3 = ValueHolder::CreateSingleDataOutput("Node3", {node1[0], node2[0]});
  ASSERT_NE(node3, nullptr);

  auto node4 = ValueHolder::CreateVoid<ValueHolder>("Node4", {node3, node2[1]});
  ASSERT_NE(node4, nullptr);

  EXPECT_NE(data1->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr(), nullptr);
  EXPECT_NE(node1[0]->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr(), nullptr);
  EXPECT_NE(data2->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr(), nullptr);
  EXPECT_NE(const2->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr(), nullptr);
  EXPECT_NE(node2[0]->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr(), nullptr);
  EXPECT_NE(node2[1]->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr(), nullptr);
  EXPECT_NE(node3->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr(), nullptr);
  EXPECT_NE(node4->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr(), nullptr);

  ASSERT_NE(const1->GetFastNode(), nullptr);
  ASSERT_NE(const1->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr(), nullptr);
  EXPECT_EQ(const1->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr()->GetAllNodes().size(), 8);

  EXPECT_EQ(const1->GetExecuteGraph(), data1->GetExecuteGraph());
  EXPECT_EQ(const1->GetExecuteGraph(), node1[0]->GetExecuteGraph());
  EXPECT_EQ(const1->GetExecuteGraph(), data2->GetExecuteGraph());
  EXPECT_EQ(const1->GetExecuteGraph(), const2->GetExecuteGraph());
  EXPECT_EQ(const1->GetExecuteGraph(), node2[0]->GetExecuteGraph());
  EXPECT_EQ(const1->GetExecuteGraph(), node2[1]->GetExecuteGraph());
  EXPECT_EQ(const1->GetExecuteGraph(), node3->GetExecuteGraph());
  EXPECT_EQ(const1->GetExecuteGraph(), node4->GetExecuteGraph());
}
TEST_F(FastValueHolderUt, CreateVoidOk) {
  ge::Format f1 = ge::FORMAT_NC1HWC0;
  auto const1 = ValueHolder::CreateConst(reinterpret_cast<const uint8_t *>(&f1), sizeof(f1));
  auto data1 = ValueHolder::CreateFeed(0);
  ASSERT_NE(const1, nullptr);
  ASSERT_NE(data1, nullptr);

  std::vector<ValueHolderPtr> inputs = {data1, const1};
  auto holder = ValueHolder::CreateVoid<ValueHolder>("TestNode", inputs);

  ASSERT_NE(holder, nullptr);

  ASSERT_NE(const1->GetExecuteGraph(), nullptr);
  ASSERT_NE(const1->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr(), nullptr);
  ASSERT_NE(data1->GetExecuteGraph(), nullptr);
  ASSERT_NE(data1->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr(), nullptr);
  ASSERT_NE(holder->GetFastNode(), nullptr);
  ASSERT_NE(holder->GetExecuteGraph(), nullptr);
  ASSERT_NE(holder->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr(), nullptr);

  // check graph is ok
  auto graph = holder->GetExecuteGraph();
  ASSERT_EQ(graph->GetAllNodes().size(), 3);
  CheckGraphGenerally(graph);

  auto const1_g = ge::ExecuteGraphUtils::FindFirstNodeMatchType(graph, "Const");
  auto data1_g = ge::ExecuteGraphUtils::FindFirstNodeMatchType(graph, "Data");
  auto node_g = ge::ExecuteGraphUtils::FindFirstNodeMatchType(graph, "TestNode");
  ASSERT_NE(const1_g, nullptr);
  ASSERT_NE(data1_g, nullptr);
  ASSERT_NE(node_g, nullptr);

  EXPECT_EQ(node_g->GetInDataEdgeByIndex(0)->src, data1_g);
  EXPECT_EQ(node_g->GetInDataEdgeByIndex(1)->src, const1_g);
}

TEST_F(FastValueHolderUt, AddDependencyOk) {
  auto data1 = ValueHolder::CreateFeed(0);
  auto data2 = ValueHolder::CreateFeed(1);
  ValueHolder::AddDependency(data1, data2);

  auto node1 = ValueHolder::CreateSingleDataOutput("Node1", {data1});
  auto node2 = ValueHolder::CreateSingleDataOutput("Node1", {data1});
  ValueHolder::AddDependency(node1, node2);

  ASSERT_NE(data1, nullptr);
  ASSERT_NE(data2, nullptr);
  ASSERT_NE(node1, nullptr);
  ASSERT_NE(node2, nullptr);

  ASSERT_NE(data1->GetFastNode(), nullptr);
  ASSERT_NE(data2->GetFastNode(), nullptr);
  ASSERT_NE(node1->GetFastNode(), nullptr);
  ASSERT_NE(node2->GetFastNode(), nullptr);

  ASSERT_EQ(data1->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr(),
            data2->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr());
  ASSERT_EQ(data1->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr(),
            node1->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr());
  ASSERT_EQ(data1->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr(),
            node2->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr());

  ASSERT_EQ(data1->GetFastNode()->GetOutControlNodes().size(), 1);
  ASSERT_EQ(data2->GetFastNode()->GetInControlNodes().size(), 1);
  EXPECT_EQ(data1->GetFastNode()->GetOutControlNodes()[0],
            data2->GetFastNode());

  ASSERT_EQ(node1->GetFastNode()->GetOutControlNodes().size(), 1);
  ASSERT_EQ(node2->GetFastNode()->GetInControlNodes().size(), 1);
  EXPECT_EQ(node1->GetFastNode()->GetOutControlNodes()[0],
            node2->GetFastNode());

  auto data3 = ValueHolder::CreateFeed(2);
  ASSERT_NE(data3, nullptr);
  ValueHolder::AddDependency(data3, data3);
  ASSERT_EQ(data3->GetFastNode()->GetOutControlNodes().size(), 0);
  ASSERT_EQ(data3->GetFastNode()->GetInControlNodes().size(), 0);
}

/*
 *           KernelLaunch
 *               |
 *             Tiling
 *             /    \
 *    InferShape   CompileInfo
 *     /   \          |
 * shape1  shape2   json
 */
TEST_F(FastValueHolderUt, CurrentNodeOk) {
  auto op_desc = std::make_shared<ge::OpDesc>("node", "node");
  ge::GeTensorDesc tensor_desc;
  tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  tensor_desc.SetFormat(ge::FORMAT_NC1HWC0);
  tensor_desc.SetDataType(ge::DT_FLOAT16);
  tensor_desc.SetOriginDataType(ge::DT_FLOAT);
  tensor_desc.SetShape(ge::GeShape({8, 1, 224, 224, 16}));
  tensor_desc.SetOriginShape(ge::GeShape({8, 3, 224, 224}));
  op_desc->AddInputDesc("x1", tensor_desc);
  op_desc->AppendIrInput("x1", ge::kIrInputRequired);
  op_desc->AppendIrInput("x2", ge::kIrInputOptional);

  auto graph = std::make_shared<ge::ComputeGraph>("graph");
  auto node = graph->AddNode(op_desc);

  auto shape1 = ValueHolder::CreateFeed(0);
  auto shape2 = ValueHolder::CreateFeed(1);
  auto json1 = ValueHolder::CreateConst("{}", 3);

  ValueHolder::SetCurrentComputeNode(node);
  auto frame = ValueHolder::GetCurrentFrame();
  ASSERT_NE(frame, nullptr);
  ASSERT_EQ(frame->GetCurrentComputeNode(), node);
  auto shape = ValueHolder::CreateSingleDataOutput("InferShape", {shape1, shape2});
  auto compile_info = ValueHolder::CreateSingleDataOutput("TilingParse", {json1});
  auto tiling_ret = ValueHolder::CreateSingleDataOutput("Tiling", {shape, compile_info});
  auto holder = ValueHolder::CreateVoid<ValueHolder>("KernelLaunch", {tiling_ret});

  ASSERT_NE(shape1, nullptr);
  ASSERT_NE(shape2, nullptr);
  ASSERT_NE(json1, nullptr);
  ASSERT_NE(shape, nullptr);
  ASSERT_NE(compile_info, nullptr);
  ASSERT_NE(tiling_ret, nullptr);
  ASSERT_NE(holder, nullptr);

  int64_t compute_node_index_none;
  ASSERT_FALSE(ge::AttrUtils::GetInt(shape1->GetFastNode()->GetOpDescBarePtr(), "ComputeNodeIndex",
                                     compute_node_index_none));
  ASSERT_FALSE(ge::AttrUtils::GetInt(shape2->GetFastNode()->GetOpDescBarePtr(), "ComputeNodeIndex",
                                     compute_node_index_none));
  ASSERT_FALSE(ge::AttrUtils::GetInt(json1->GetFastNode()->GetOpDescBarePtr(), "ComputeNodeIndex",
                                     compute_node_index_none));

  int64_t compute_node_index_shape, compute_node_index_compile_ifo, compute_node_index_tiling_ret,
      compute_node_index_holder;
  ASSERT_TRUE(ge::AttrUtils::GetInt(shape->GetFastNode()->GetOpDescBarePtr(), "ComputeNodeIndex",
                                    compute_node_index_shape));
  ASSERT_TRUE(
      ge::AttrUtils::GetInt(compile_info->GetFastNode()->GetOpDescBarePtr(), "ComputeNodeIndex",
                            compute_node_index_compile_ifo));
  ASSERT_TRUE(
      ge::AttrUtils::GetInt(tiling_ret->GetFastNode()->GetOpDescBarePtr(), "ComputeNodeIndex",
                            compute_node_index_tiling_ret));
  ASSERT_TRUE(ge::AttrUtils::GetInt(holder->GetFastNode()->GetOpDescBarePtr(), "ComputeNodeIndex",
                                    compute_node_index_holder));
  EXPECT_EQ(compute_node_index_shape, compute_node_index_compile_ifo);
  EXPECT_EQ(compute_node_index_shape, compute_node_index_tiling_ret);
  EXPECT_EQ(compute_node_index_shape, compute_node_index_holder);

  size_t frame_current_node_index;
  frame->GetCurrentNodeIndex(frame_current_node_index);
  EXPECT_EQ(compute_node_index_shape, frame_current_node_index);
}
/*
 *    hello
 *    /  \
 * data0 data1
 */
TEST_F(FastValueHolderUt, CreateExeGraphOk) {
  auto op_desc = std::make_shared<ge::OpDesc>("node", "node");
  ge::GeTensorDesc tensor_desc;
  tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  tensor_desc.SetFormat(ge::FORMAT_NC1HWC0);
  tensor_desc.SetDataType(ge::DT_FLOAT16);
  tensor_desc.SetOriginDataType(ge::DT_FLOAT);
  tensor_desc.SetShape(ge::GeShape({8, 1, 224, 224, 16}));
  tensor_desc.SetOriginShape(ge::GeShape({8, 3, 224, 224}));
  op_desc->AddInputDesc("x1", tensor_desc);
  op_desc->AppendIrInput("x1", ge::kIrInputRequired);
  op_desc->AppendIrInput("x2", ge::kIrInputOptional);

  auto graph = std::make_shared<ge::ComputeGraph>("graph");
  auto node = graph->AddNode(op_desc);

  auto data0 = ValueHolder::CreateFeed(0);
  auto data1 = ValueHolder::CreateFeed(1);

  ValueHolder::SetCurrentComputeNode(node);
  auto hello = ValueHolder::CreateSingleDataOutput("hello", {data0, data1});

  ValueHolder::AddRelevantInputNode(node);
  ASSERT_NE(graph, nullptr);
}
/*
 *    hello
 *    /  \
 * data0 data1
 */
TEST_F(FastValueHolderUt, CreateExeGraphWithTargetsOk) {
  auto op_desc = std::make_shared<ge::OpDesc>("node", "node");
  ge::GeTensorDesc tensor_desc;
  tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  tensor_desc.SetFormat(ge::FORMAT_NC1HWC0);
  tensor_desc.SetDataType(ge::DT_FLOAT16);
  tensor_desc.SetOriginDataType(ge::DT_FLOAT);
  tensor_desc.SetShape(ge::GeShape({8, 1, 224, 224, 16}));
  tensor_desc.SetOriginShape(ge::GeShape({8, 3, 224, 224}));
  op_desc->AddInputDesc("x1", tensor_desc);
  op_desc->AppendIrInput("x1", ge::kIrInputRequired);
  op_desc->AppendIrInput("x2", ge::kIrInputOptional);

  auto graph = std::make_shared<ge::ComputeGraph>("graph");
  auto node = graph->AddNode(op_desc);

  auto data0 = ValueHolder::CreateFeed(0);
  auto data1 = ValueHolder::CreateFeed(1);

  ValueHolder::SetCurrentComputeNode(node);
  auto hello = ValueHolder::CreateVoid<ValueHolder>("hello", {data0, data1});
  ASSERT_NE(graph, nullptr);
}
/*
 *                      c
 * Atomic-LaunchKernel ----> LaunchKernel
 *          |                 /
 *    Atomic-tiling      Tiling
 *        /    \        /    \
 * TilingParse InferShape   TilingParse
 *    |         /   \          |
 *   json1   shape1  shape2   json2
 */
TEST_F(FastValueHolderUt, ScopedCurrentNodeOk) {
  auto graph = std::make_shared<ge::ComputeGraph>("graph");

  auto op_desc = std::make_shared<ge::OpDesc>("node", "node");
  ge::GeTensorDesc tensor_desc;
  tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  tensor_desc.SetFormat(ge::FORMAT_NC1HWC0);
  tensor_desc.SetDataType(ge::DT_FLOAT16);
  tensor_desc.SetOriginDataType(ge::DT_FLOAT);
  tensor_desc.SetShape(ge::GeShape({8, 1, 224, 224, 16}));
  tensor_desc.SetOriginShape(ge::GeShape({8, 3, 224, 224}));
  op_desc->AddInputDesc("x1", tensor_desc);
  op_desc->AppendIrInput("x1", ge::kIrInputRequired);
  op_desc->AppendIrInput("x2", ge::kIrInputOptional);
  auto node = graph->AddNode(op_desc);

  auto clean_op_desc = std::make_shared<ge::OpDesc>("node-AtomicClean", "DynamicAtomicAddrClean");
  clean_op_desc->AddInputDesc("workspace", tensor_desc);
  clean_op_desc->AddInputDesc("clean1", tensor_desc);
  clean_op_desc->AddInputDesc("clean2", tensor_desc);
  clean_op_desc->AppendIrInput("workspace", ge::kIrInputRequired);
  clean_op_desc->AppendIrInput("clean", ge::kIrInputDynamic);
  auto clean_node = graph->AddNode(clean_op_desc);

  auto shape1 = ValueHolder::CreateFeed(0);
  auto shape2 = ValueHolder::CreateFeed(1);
  auto json1 = ValueHolder::CreateConst("{}", 2);
  auto json2 = ValueHolder::CreateConst("{}", 3);

  ValueHolder::SetCurrentComputeNode(node);
  auto frame = ValueHolder::GetCurrentFrame();
  ASSERT_NE(frame, nullptr);
  ASSERT_EQ(frame->GetCurrentComputeNode(), node);
  auto shape = ValueHolder::CreateSingleDataOutput("InferShape", {shape1, shape2});

  size_t node1_index;
  ValueHolderPtr compile_info1, tiling_ret1, holder1;
  {
    auto guarder = ValueHolder::SetScopedCurrentComputeNode(clean_node);
    compile_info1 = ValueHolder::CreateSingleDataOutput("TilingParse", {json1});
    tiling_ret1 = ValueHolder::CreateSingleDataOutput("Tiling", {shape, compile_info1});
    holder1 = ValueHolder::CreateVoid<ValueHolder>("AtomicKernelLaunch", {tiling_ret1});
    EXPECT_TRUE(frame->GetCurrentNodeIndex(node1_index));
  }

  auto compile_info2 = ValueHolder::CreateSingleDataOutput("TilingParse", {json2});
  auto tiling_ret2 = ValueHolder::CreateSingleDataOutput("Tiling", {shape, compile_info2});
  auto holder2 = ValueHolder::CreateVoid<ValueHolder>("KernelLaunch", {tiling_ret2});

  ValueHolder::AddDependency(holder1, holder2);

  ASSERT_NE(shape1, nullptr);
  ASSERT_NE(shape2, nullptr);
  ASSERT_NE(json1, nullptr);
  ASSERT_NE(shape, nullptr);
  ASSERT_NE(compile_info1, nullptr);
  ASSERT_NE(tiling_ret1, nullptr);
  ASSERT_NE(holder1, nullptr);
  ASSERT_NE(compile_info2, nullptr);
  ASSERT_NE(tiling_ret2, nullptr);
  ASSERT_NE(holder2, nullptr);

  int64_t compute_node_index_none;
  ASSERT_FALSE(ge::AttrUtils::GetInt(shape1->GetFastNode()->GetOpDescBarePtr(), "ComputeNodeIndex",
                                     compute_node_index_none));
  ASSERT_FALSE(ge::AttrUtils::GetInt(shape2->GetFastNode()->GetOpDescBarePtr(), "ComputeNodeIndex",
                                     compute_node_index_none));
  ASSERT_FALSE(ge::AttrUtils::GetInt(json1->GetFastNode()->GetOpDescBarePtr(), "ComputeNodeIndex",
                                     compute_node_index_none));

  int64_t shape_index, compile_info1_index, tiling_ret1_index, holder1_index, compile_info2_index, tiling_ret2_index,
      holder2_index;
  ASSERT_TRUE(ge::AttrUtils::GetInt(shape->GetFastNode()->GetOpDescBarePtr(), "ComputeNodeIndex", shape_index));

  ASSERT_TRUE(ge::AttrUtils::GetInt(compile_info1->GetFastNode()->GetOpDescBarePtr(), "ComputeNodeIndex",
                                    compile_info1_index));
  ASSERT_TRUE(ge::AttrUtils::GetInt(tiling_ret1->GetFastNode()->GetOpDescBarePtr(), "ComputeNodeIndex",
                                    tiling_ret1_index));
  ASSERT_TRUE(ge::AttrUtils::GetInt(holder1->GetFastNode()->GetOpDescBarePtr(), "ComputeNodeIndex",
                                    holder1_index));

  ASSERT_TRUE(ge::AttrUtils::GetInt(compile_info2->GetFastNode()->GetOpDescBarePtr(), "ComputeNodeIndex",
                                    compile_info2_index));
  ASSERT_TRUE(ge::AttrUtils::GetInt(tiling_ret2->GetFastNode()->GetOpDescBarePtr(), "ComputeNodeIndex",
                                    tiling_ret2_index));
  ASSERT_TRUE(ge::AttrUtils::GetInt(holder2->GetFastNode()->GetOpDescBarePtr(), "ComputeNodeIndex",
                                    holder2_index));

  EXPECT_EQ(shape_index, compile_info2_index);
  EXPECT_EQ(shape_index, tiling_ret2_index);
  EXPECT_EQ(shape_index, holder2_index);

  EXPECT_NE(shape_index, compile_info1_index);
  EXPECT_EQ(compile_info1_index, tiling_ret1_index);
  EXPECT_EQ(compile_info1_index, holder1_index);
}

TEST_F(FastValueHolderUt, CreateExeGraphNoOutpus) {
  auto op_desc = std::make_shared<ge::OpDesc>("node", "node");
  ge::GeTensorDesc tensor_desc;
  tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  tensor_desc.SetFormat(ge::FORMAT_NC1HWC0);
  tensor_desc.SetDataType(ge::DT_FLOAT16);
  tensor_desc.SetOriginDataType(ge::DT_FLOAT);
  tensor_desc.SetShape(ge::GeShape({8, 1, 224, 224, 16}));
  tensor_desc.SetOriginShape(ge::GeShape({8, 3, 224, 224}));
  op_desc->AddInputDesc("x1", tensor_desc);
  op_desc->AppendIrInput("x1", ge::kIrInputRequired);
  op_desc->AppendIrInput("x2", ge::kIrInputOptional);

  auto graph = std::make_shared<ge::ComputeGraph>("graph");
  auto node = graph->AddNode(op_desc);

  auto data0 = ValueHolder::CreateFeed(0);
  auto data1 = ValueHolder::CreateFeed(1);

  ValueHolder::SetCurrentComputeNode(node);
  auto hello = ValueHolder::CreateVoid<ValueHolder>("hello", {data0, data1});
  ASSERT_NE(hello, nullptr);
  EXPECT_NE(hello->GetCurrentExecuteGraph(), nullptr);
}

TEST_F(FastValueHolderUt, CreateExeGraphNoFrame) {
  auto op_desc = std::make_shared<ge::OpDesc>("node", "node");
  ge::GeTensorDesc tensor_desc;
  tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  tensor_desc.SetFormat(ge::FORMAT_NC1HWC0);
  tensor_desc.SetDataType(ge::DT_FLOAT16);
  tensor_desc.SetOriginDataType(ge::DT_FLOAT);
  tensor_desc.SetShape(ge::GeShape({8, 1, 224, 224, 16}));
  tensor_desc.SetOriginShape(ge::GeShape({8, 3, 224, 224}));
  op_desc->AddInputDesc("x1", tensor_desc);
  op_desc->AppendIrInput("x1", ge::kIrInputRequired);
  op_desc->AppendIrInput("x2", ge::kIrInputOptional);

  auto graph = std::make_shared<ge::ComputeGraph>("graph");
  auto node = graph->AddNode(op_desc);

  auto data0 = ValueHolder::CreateFeed(0);
  auto data1 = ValueHolder::CreateFeed(1);

  ValueHolder::SetCurrentComputeNode(node);
  auto hello = ValueHolder::CreateVoid<ValueHolder>("hello", {data0, data1});
  ASSERT_NE(hello, nullptr);
  EXPECT_NE(hello->GetCurrentExecuteGraph(), nullptr);
}
/*
 *    hello
 *    /  \
 * data0 data1
 */
TEST_F(FastValueHolderUt, GetCurrentGraphOk) {
  auto op_desc = std::make_shared<ge::OpDesc>("node", "node");
  ge::GeTensorDesc tensor_desc;
  tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  tensor_desc.SetFormat(ge::FORMAT_NC1HWC0);
  tensor_desc.SetDataType(ge::DT_FLOAT16);
  tensor_desc.SetOriginDataType(ge::DT_FLOAT);
  tensor_desc.SetShape(ge::GeShape({8, 1, 224, 224, 16}));
  tensor_desc.SetOriginShape(ge::GeShape({8, 3, 224, 224}));
  op_desc->AddInputDesc("x1", tensor_desc);
  op_desc->AppendIrInput("x1", ge::kIrInputRequired);
  op_desc->AppendIrInput("x2", ge::kIrInputOptional);

  auto graph = std::make_shared<ge::ComputeGraph>("graph");
  auto node = graph->AddNode(op_desc);

  auto data0 = ValueHolder::CreateFeed(0);
  auto data1 = ValueHolder::CreateFeed(1);

  ValueHolder::SetCurrentComputeNode(node);
  auto hello = ValueHolder::CreateVoid<ValueHolder>("hello", {data0, data1});

  EXPECT_NE(hello->GetCurrentFrame(), nullptr);
  EXPECT_NE(hello->GetCurrentExecuteGraph(), nullptr);
}
/*
 *       ref
 *     +------+
 *     |      |
 *   launch   |
 *     |      |
 *   tiling   |
 *     |      |
 *    alloc----
 *    /  \
 * data0 data1
 */
TEST_F(FastValueHolderUt, RefFromOk) {
  auto op_desc = std::make_shared<ge::OpDesc>("node", "node");
  ge::GeTensorDesc tensor_desc;
  tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  tensor_desc.SetFormat(ge::FORMAT_NC1HWC0);
  tensor_desc.SetDataType(ge::DT_FLOAT16);
  tensor_desc.SetOriginDataType(ge::DT_FLOAT);
  tensor_desc.SetShape(ge::GeShape({8, 1, 224, 224, 16}));
  tensor_desc.SetOriginShape(ge::GeShape({8, 3, 224, 224}));
  op_desc->AddInputDesc("x1", tensor_desc);
  op_desc->AppendIrInput("x1", ge::kIrInputRequired);
  op_desc->AppendIrInput("x2", ge::kIrInputOptional);

  auto graph = std::make_shared<ge::ComputeGraph>("graph");
  auto node = graph->AddNode(op_desc);

  auto data0 = ValueHolder::CreateFeed(0);
  auto data1 = ValueHolder::CreateFeed(1);

  ValueHolder::SetCurrentComputeNode(node);
  auto alloc_outs = ValueHolder::CreateDataOutput("alloc", {data0, data1}, 3);
  auto tiling_outs = ValueHolder::CreateDataOutput("tiling", {data0, data1}, 2);
  tiling_outs[1]->RefFrom(alloc_outs[1]);

  auto launch = ValueHolder::CreateSingleDataOutput("launch", {tiling_outs[0], tiling_outs[1]});
  ASSERT_NE(launch, nullptr);
  launch->RefFrom(alloc_outs[2]);
  EXPECT_NE(launch->GetCurrentExecuteGraph(), nullptr);
}
/*
 *    hello
 *    /  \
 * data0 data1
 */
TEST_F(FastValueHolderUt, AddNullOutputs) {
  auto op_desc = std::make_shared<ge::OpDesc>("node", "node");
  ge::GeTensorDesc tensor_desc;
  tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  tensor_desc.SetFormat(ge::FORMAT_NC1HWC0);
  tensor_desc.SetDataType(ge::DT_FLOAT16);
  tensor_desc.SetOriginDataType(ge::DT_FLOAT);
  tensor_desc.SetShape(ge::GeShape({8, 1, 224, 224, 16}));
  tensor_desc.SetOriginShape(ge::GeShape({8, 3, 224, 224}));
  op_desc->AddInputDesc("x1", tensor_desc);
  op_desc->AppendIrInput("x1", ge::kIrInputRequired);
  op_desc->AppendIrInput("x2", ge::kIrInputOptional);

  auto graph = std::make_shared<ge::ComputeGraph>("graph");
  auto node = graph->AddNode(op_desc);

  auto data0 = ValueHolder::CreateFeed(0);
  auto data1 = ValueHolder::CreateFeed(1);

  ValueHolder::SetCurrentComputeNode(node);
  auto hello = ValueHolder::CreateSingleDataOutput("hello", {data0, data1});

  EXPECT_NE(hello->GetCurrentFrame(), nullptr);
  EXPECT_NE(hello->GetCurrentExecuteGraph(), nullptr);
}
/*
 *    hello
 *    /  \
 * data0 data1
 */
TEST_F(FastValueHolderUt, AddNullTargets) {
  auto op_desc = std::make_shared<ge::OpDesc>("node", "node");
  ge::GeTensorDesc tensor_desc;
  tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  tensor_desc.SetFormat(ge::FORMAT_NC1HWC0);
  tensor_desc.SetDataType(ge::DT_FLOAT16);
  tensor_desc.SetOriginDataType(ge::DT_FLOAT);
  tensor_desc.SetShape(ge::GeShape({8, 1, 224, 224, 16}));
  tensor_desc.SetOriginShape(ge::GeShape({8, 3, 224, 224}));
  op_desc->AddInputDesc("x1", tensor_desc);
  op_desc->AppendIrInput("x1", ge::kIrInputRequired);
  op_desc->AppendIrInput("x2", ge::kIrInputOptional);

  auto graph = std::make_shared<ge::ComputeGraph>("graph");
  auto node = graph->AddNode(op_desc);

  auto data0 = ValueHolder::CreateFeed(0);
  auto data1 = ValueHolder::CreateFeed(1);

  ValueHolder::SetCurrentComputeNode(node);
  auto hello = ValueHolder::CreateSingleDataOutput("hello", {data0, data1});

  EXPECT_NE(hello->GetCurrentFrame(), nullptr);
  EXPECT_NE(hello->GetCurrentExecuteGraph(), nullptr);
}
TEST_F(FastValueHolderUt, Guard_AddFlagToNode) {
  auto data0 = ValueHolder::CreateFeed(0);
  auto allocator0 = ValueHolder::CreateSingleDataOutput("CreateAllocator", {data0});
  auto allocator_destroyer = ValueHolder::CreateVoidGuarder("DestroyAllocator", allocator0, {});
  ASSERT_NE(allocator_destroyer, nullptr);
  auto tmp_frame = ValueHolder::PopGraphFrame();
  auto graph = tmp_frame->GetExecuteGraph().get();

  auto node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(graph, "DestroyAllocator");
  ASSERT_NE(node, nullptr);
  int64_t index;
  EXPECT_TRUE(ge::AttrUtils::GetInt(node->GetOpDescBarePtr(), kReleaseResourceIndex, index));
  EXPECT_EQ(index, 0);

  const auto &allocator_node = allocator0->GetFastNode();
  ASSERT_NE(allocator_node, nullptr);
  const auto &allocator_desc = allocator_node->GetOpDescBarePtr();
  ASSERT_NE(allocator_desc, nullptr);
  string guarder_type;
  EXPECT_TRUE(ge::AttrUtils::GetStr(allocator_desc, kGuarderNodeType, guarder_type));
  EXPECT_EQ(guarder_type, "DestroyAllocator");
}
TEST_F(FastValueHolderUt, Guarder_AddDependencyAutomately_ConnectDataEdgeToResource) {
  auto data0 = ValueHolder::CreateFeed(0);
  auto allocator0 = ValueHolder::CreateSingleDataOutput("CreateAllocator", {data0});
  auto allocator_destroyer = ValueHolder::CreateVoidGuarder("DestroyAllocator", allocator0, {});
  ASSERT_NE(allocator_destroyer, nullptr);

  size_t alloc_size = 1024;
  auto size = ValueHolder::CreateConst(&alloc_size, sizeof(alloc_size));
  auto alloc_mem0 = ValueHolder::CreateSingleDataOutput("AllocMemory", {allocator0, size});
  auto alloc_mem1 = ValueHolder::CreateSingleDataOutput("AllocMemory", {allocator0, size});
  auto tmp_frame = ValueHolder::PopGraphFrame();
  auto graph = tmp_frame->GetExecuteGraph().get();

  CheckGraphGenerally(graph);

  ASSERT_NE(alloc_mem0, nullptr);
  ASSERT_NE(alloc_mem1, nullptr);
  HasControlEdge(graph, alloc_mem0->GetFastNode(), allocator_destroyer->GetFastNode());
  HasControlEdge(graph, alloc_mem1->GetFastNode(), allocator_destroyer->GetFastNode());

  const auto &allocator_node = allocator0->GetFastNode();
  ASSERT_NE(allocator_node, nullptr);
  const auto &allocator_desc = allocator_node->GetOpDescBarePtr();
  ASSERT_NE(allocator_desc, nullptr);
  string guarder_type;
  EXPECT_TRUE(ge::AttrUtils::GetStr(allocator_desc, kGuarderNodeType, guarder_type));
  EXPECT_EQ(guarder_type, "DestroyAllocator");
}
/*
*     NetOutput
*        |
*      Bar -c-> foo0_guarder
*      / \    /
* data1   foo0
*          |
*        data0
*/
TEST_F(FastValueHolderUt, Guarder_AddDependencyFromTheSameLevelNode_ConnectFromSrcToSubgraphNodes) {
  auto data0 = ValueHolder::CreateFeed(0);
  auto foo0 = ValueHolder::CreateSingleDataOutput("Foo", {data0});
  auto guarder = ValueHolder::CreateVoidGuarder("FooGuarder", foo0, {});
  ASSERT_NE(guarder, nullptr);
  auto data1 = ValueHolder::CreateFeed(1);
  auto bar1 = ValueHolder::CreateSingleDataOutput("Bar", {data1});

  ValueHolder::PushGraphFrame(bar1, "BarGraph");
  auto foo1 = ValueHolder::CreateSingleDataOutput("Foo", {foo0, data1});
  auto bar_frame = ValueHolder::PopGraphFrame({foo1}, {});

  auto frame = ValueHolder::PopGraphFrame({bar1}, {});
  ASSERT_NE(frame, nullptr);
  ASSERT_NE(frame->GetExecuteGraph(), nullptr);

  EXPECT_EQ(frame->GetExecuteGraph()->TopologicalSorting(), ge::GRAPH_SUCCESS);
  EXPECT_TRUE(FastNodeTopoChecker(bar1).OutChecker().CtrlToByType("FooGuarder").IsOk());
  EXPECT_EQ(FastNodeTopoChecker(bar1).StrictConnectFrom({{"Data"}, {"Foo"}}), "success");

  const auto &foo_node = foo0->GetFastNode();
  ASSERT_NE(foo_node, nullptr);
  const auto &foo_desc = foo_node->GetOpDescBarePtr();
  ASSERT_NE(foo_desc, nullptr);
  string guarder_type;
  EXPECT_TRUE(ge::AttrUtils::GetStr(foo_desc, kGuarderNodeType, guarder_type));
  EXPECT_EQ(guarder_type, "FooGuarder");
}
TEST_F(FastValueHolderUt, Guarder_DoNotAddDependency_ConnectDataEdgeToNetOutput) {
  auto data0 = ValueHolder::CreateFeed(0);
  auto foo0 = ValueHolder::CreateSingleDataOutput("Foo", {data0});
  auto guarder = ValueHolder::CreateVoidGuarder("FooGuarder", foo0, {});
  ASSERT_NE(guarder, nullptr);

  auto bar0 = ValueHolder::CreateSingleDataOutput("Bar", {foo0});

  auto frame = ValueHolder::PopGraphFrame({foo0}, {});
  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExecuteGraph().get();
  ASSERT_NE(graph, nullptr);

  EXPECT_EQ(FastNodeTopoChecker(foo0).StrictConnectTo(0, {{"NetOutput"}, {"FooGuarder"}, {"Bar"}}), "success");
  HasControlEdge(graph, bar0->GetFastNode(), guarder->GetFastNode());
  ASSERT_EQ(guarder->GetFastNode()->GetInControlNodes().size(), 1);

  const auto &foo_node = foo0->GetFastNode();
  ASSERT_NE(foo_node, nullptr);
  const auto &foo_desc = foo_node->GetOpDescBarePtr();
  ASSERT_NE(foo_desc, nullptr);
  string guarder_type;
  EXPECT_TRUE(ge::AttrUtils::GetStr(foo_desc, kGuarderNodeType, guarder_type));
  EXPECT_EQ(guarder_type, "FooGuarder");
}
TEST_F(FastValueHolderUt, AddDependencyForGuard_RleaseBy) {
  auto data0 = ValueHolder::CreateFeed(0);
  auto allocator0 = ValueHolder::CreateSingleDataOutput("CreateAllocator", {data0});
  auto allocator_destroyer = ValueHolder::CreateVoidGuarder("DestroyAllocator", allocator0, {});
  ASSERT_NE(allocator_destroyer, nullptr);

  size_t alloc_size = 1024;
  auto size = ValueHolder::CreateConst(&alloc_size, sizeof(alloc_size));
  auto alloc_mem0 = ValueHolder::CreateSingleDataOutput("AllocMemory", {allocator0, size});
  auto free_mem0 = ValueHolder::CreateVoidGuarder("FreeMemory", {alloc_mem0}, {});
  auto tmp_graph = ValueHolder::PopGraphFrame();
  auto graph = tmp_graph->GetExecuteGraph().get();
  CheckGraphGenerally(graph);

  ASSERT_NE(free_mem0, nullptr);
  ASSERT_NE(alloc_mem0, nullptr);
  HasControlEdge(graph, alloc_mem0->GetFastNode(), allocator_destroyer->GetFastNode());

  allocator0->ReleaseAfter(free_mem0);
  HasControlEdge(graph, free_mem0->GetFastNode(), allocator_destroyer->GetFastNode());

  const auto &allocator_node = allocator0->GetFastNode();
  ASSERT_NE(allocator_node, nullptr);
  const auto &allocator_desc = allocator_node->GetOpDescBarePtr();
  ASSERT_NE(allocator_desc, nullptr);
  string guarder_type;
  EXPECT_TRUE(ge::AttrUtils::GetStr(allocator_desc, kGuarderNodeType, guarder_type));
  EXPECT_EQ(guarder_type, "DestroyAllocator");

  const auto &alloc_node = alloc_mem0->GetFastNode();
  ASSERT_NE(alloc_node, nullptr);
  const auto &alloc_desc = alloc_node->GetOpDescBarePtr();
  ASSERT_NE(alloc_desc, nullptr);
  EXPECT_TRUE(ge::AttrUtils::GetStr(alloc_desc, kGuarderNodeType, guarder_type));
  EXPECT_EQ(guarder_type, "FreeMemory");
}
TEST_F(FastValueHolderUt, RleaseBy_NoGuarder) {
  auto data0 = ValueHolder::CreateFeed(0);
  auto allocator0 = ValueHolder::CreateSingleDataOutput("CreateAllocator", {data0});

  size_t alloc_size = 1024;
  auto size = ValueHolder::CreateConst(&alloc_size, sizeof(alloc_size));
  auto alloc_mem0 = ValueHolder::CreateSingleDataOutput("AllocMemory", {allocator0, size});
  auto tmp_graph = ValueHolder::PopGraphFrame();
  auto graph = tmp_graph->GetExecuteGraph().get();

  CheckGraphGenerally(graph);

  ASSERT_NE(alloc_mem0, nullptr);

  allocator0->ReleaseAfter(alloc_mem0);

  EXPECT_EQ(allocator0->GetFastNode()->GetAllOutNodes().size(), 1);
  EXPECT_EQ(allocator0->GetFastNode()->GetAllInNodes().size(), 1);

  EXPECT_EQ(alloc_mem0->GetFastNode()->GetAllOutNodes().size(), 0);
  EXPECT_EQ(alloc_mem0->GetFastNode()->GetAllInNodes().size(), 2);
}
TEST_F(FastValueHolderUt, PushFrame_ChildFrameIsNotRoot) {
  ValueHolder::PopGraphFrame();
  auto root_frame = ValueHolder::PushGraphFrame();
  EXPECT_TRUE(root_frame->IsRootFrame());
  auto feed0 = ValueHolder::CreateFeed(0);
  auto child_frame = ValueHolder::PushGraphFrame(feed0, "subgraph_name0");
  ASSERT_NE(child_frame, nullptr);
  EXPECT_FALSE(child_frame->IsRootFrame());
}
TEST_F(FastValueHolderUt, PushFrame_ComputeNodeIndexTheSame) {
  auto compute_node1 = FakeNode();
  auto compute_node2 = FakeNode();
  auto compute_node3 = FakeNode();

  ValueHolder::PopGraphFrame();
  auto root_frame = ValueHolder::PushGraphFrame();
  EXPECT_TRUE(root_frame->IsRootFrame());
  ValueHolder::SetCurrentComputeNode(compute_node1);
  auto feed0 = ValueHolder::CreateFeed(0);
  auto feed1 = ValueHolder::CreateFeed(1);
  auto foo0 = ValueHolder::CreateSingleDataOutput("Foo", {feed0, feed1});

  ValueHolder::SetCurrentComputeNode(compute_node2);
  auto bar0 = ValueHolder::CreateSingleDataOutput("Bar", {foo0});

  ValueHolder::PushGraphFrame(foo0, "subgraph_name0");
  auto sub_bar1 = ValueHolder::CreateSingleDataOutput("Bar", {feed0});

  EXPECT_EQ(GetComputeNodeIndex(sub_bar1->GetFastNode()), GetComputeNodeIndex(foo0->GetFastNode()));
  EXPECT_NE(GetComputeNodeIndex(bar0->GetFastNode()), GetComputeNodeIndex(foo0->GetFastNode()));

  ValueHolder::PopGraphFrame();
  ValueHolder::PopGraphFrame();
}

TEST_F(FastValueHolderUt, PlacementDefault0) {
  auto data0 = ValueHolder::CreateFeed(0);
  EXPECT_EQ(data0->GetPlacement(), 0);
}
TEST_F(FastValueHolderUt, SetGetPlacementOk) {
  auto data0 = ValueHolder::CreateFeed(0);
  data0->SetPlacement(1);
  EXPECT_EQ(data0->GetPlacement(), 1);
  data0->SetPlacement(2);
  EXPECT_EQ(data0->GetPlacement(), 2);
}
TEST_F(FastValueHolderUt, BuildGraphWithDataOutput) {
  auto data0 = ValueHolder::CreateFeed(0);
  auto data1 = ValueHolder::CreateFeed(1);
  auto foo1 = ValueHolder::CreateSingleDataOutput("Foo", {data0, data1});
  auto bar1 = ValueHolder::CreateSingleDataOutput("Bar", {data0, data1});
  auto frame = ValueHolder::PopGraphFrame({foo1, bar1}, {});
  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExecuteGraph().get();
  ASSERT_NE(graph, nullptr);

  CheckGraphGenerally(graph);

  EXPECT_EQ(graph->GetAllNodes().size(), 5);

  auto nodes = ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(graph, "NetOutput");
  ASSERT_EQ(nodes.size(), 1);
  auto netoutput = nodes[0];
  ASSERT_NE(netoutput, nullptr);
  EXPECT_EQ(netoutput->GetAllInNodes().size(), 2);
  ASSERT_EQ(netoutput->GetInDataNodes().size(), 2);
  EXPECT_EQ((*netoutput->GetInDataNodes().begin())->GetType(), "Foo");
  EXPECT_EQ((*(netoutput->GetInDataNodes().begin() + 1))->GetType(), "Bar");
}
TEST_F(FastValueHolderUt, BuildGraphWithCtrlOutput) {
  auto data0 = ValueHolder::CreateFeed(0);
  auto data1 = ValueHolder::CreateFeed(1);
  auto foo1 = ValueHolder::CreateSingleDataOutput("Foo", {data0, data1});
  auto bar1 = ValueHolder::CreateSingleDataOutput("Bar", {data0, data1});
  auto launch = ValueHolder::CreateVoid<ValueHolder>("Launch", {foo1, bar1});
  auto frame = ValueHolder::PopGraphFrame({}, {launch});
  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExecuteGraph().get();
  ASSERT_NE(graph, nullptr);

  CheckGraphGenerally(graph);

  EXPECT_EQ(graph->GetAllNodes().size(), 6);

  auto nodes = ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(graph, "NetOutput");
  ASSERT_EQ(nodes.size(), 1);
  auto netoutput = nodes[0];
  ASSERT_NE(netoutput, nullptr);
  EXPECT_EQ(netoutput->GetAllInNodes().size(), 1);
  ASSERT_EQ(netoutput->GetInControlNodes().size(), 1);
  EXPECT_EQ((*netoutput->GetInControlNodes().begin())->GetType(), "Launch");
}
TEST_F(FastValueHolderUt, AppendOutputOk) {
  auto foo = ValueHolder::CreateVoid<ValueHolder>("Foo", {});
  EXPECT_EQ(foo->GetFastNode()->GetDataOutNum(), 0);

  auto outputs = foo->AppendOutputs<ValueHolder>(5);
  EXPECT_EQ(outputs.size(), 5);
  EXPECT_EQ(foo->GetFastNode()->GetDataOutNum(), 5);

  auto bar = ValueHolder::CreateSingleDataOutput("Bar", outputs);
  EXPECT_NE(bar, nullptr);
  EXPECT_EQ(bar->GetFastNode()->GetDataInNum(), 5);
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(bar->GetFastNode()->GetInDataEdgeByIndex(i)->src_output, i);
    EXPECT_EQ(bar->GetFastNode()->GetInDataEdgeByIndex(i)->src,
              foo->GetFastNode());
  }
}
TEST_F(FastValueHolderUt, AppendOutputToNodeWithOutputs) {
  auto foo = ValueHolder::CreateDataOutput("Foo", {}, 3)[0];
  EXPECT_EQ(foo->GetFastNode()->GetDataOutNum(), 3);

  auto outputs = foo->AppendOutputs<ValueHolder>(5);
  ASSERT_EQ(outputs.size(), 5);
  EXPECT_EQ(foo->GetFastNode()->GetDataOutNum(), 8);

  auto bar = ValueHolder::CreateSingleDataOutput("Bar", outputs);
  EXPECT_NE(bar, nullptr);
  EXPECT_EQ(bar->GetFastNode()->GetDataInNum(), 5);
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(bar->GetFastNode()->GetInDataEdgeByIndex(i)->src_output, i + 3);
    EXPECT_EQ(bar->GetFastNode()->GetInDataEdgeByIndex(i)->src,
              foo->GetFastNode());
  }
}
TEST_F(FastValueHolderUt, AppendInputInOneGraphOk) {
  const auto &foo = ValueHolder::CreateVoid<ValueHolder>("Foo", {});
  const auto &input1 = ValueHolder::CreateConst("{}", 2);
  const auto &input2 = ValueHolder::CreateFeed(0);
  EXPECT_EQ(foo->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr(),
            input1->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr());
  EXPECT_EQ(foo->AppendInputs({input1, input2}), 0);
  const auto &input_nodes = foo->GetFastNode()->GetInDataNodes();
  EXPECT_EQ(input_nodes.size(), 2);
  EXPECT_EQ(input_nodes.at(0)->GetType(), "Const");
  EXPECT_EQ(input_nodes.at(1)->GetType(), "Data");
}
TEST_F(FastValueHolderUt, AppendInputInParentGraphOk) {
  const auto &input1 = ValueHolder::CreateConst("{}", 2);
  auto main_node = ValueHolder::CreateVoid<bg::ValueHolder>("Main", {});
  ValueHolder::PushGraphFrame(main_node, "Main");
  const auto &foo = ValueHolder::CreateVoid<ValueHolder>("Foo", {});
  EXPECT_NE(foo->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr(),
            input1->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr());
  EXPECT_EQ(foo->AppendInputs({input1}), 0);
  const auto &input_nodes = foo->GetFastNode()->GetInDataNodes();
  EXPECT_EQ(input_nodes.size(), 1);
  EXPECT_EQ(input_nodes.at(0)->GetType(), "InnerData");
}
TEST_F(FastValueHolderUt, AppendInputInChildGraph_noOk) {
  const auto &foo = ValueHolder::CreateConst("{}", 2);
  auto main_node = ValueHolder::CreateVoid<bg::ValueHolder>("Main", {});
  ValueHolder::PushGraphFrame(main_node, "Main");
  const auto &input1 = ValueHolder::CreateVoid<ValueHolder>("Foo", {});
  EXPECT_NE(foo->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr(),
            input1->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr());
  EXPECT_NE(foo->AppendInputs({input1}), 0);
  const auto &input_nodes = foo->GetFastNode()->GetInDataNodes();
  EXPECT_EQ(input_nodes.size(), 0);
  const uint32_t in_data_anchor_size = foo->GetFastNode()->GetDataInNum();
  EXPECT_EQ(in_data_anchor_size, 1);
}
TEST_F(FastValueHolderUt, AppendInputInDifferentGraph_noOk) {
  auto main_node = ValueHolder::CreateVoid<bg::ValueHolder>("Main", {});
  ValueHolder::PushGraphFrame(main_node, "Main");
  const auto &input1 = ValueHolder::CreateConst("{}", 2);
  const auto &main_frame = ValueHolder::PopGraphFrame();
  const auto &de_init_node = ValueHolder::CreateVoid<bg::ValueHolder>("DeInit", {});
  ValueHolder::PushGraphFrame(de_init_node, "DeInit");
  const auto &foo = ValueHolder::CreateVoid<ValueHolder>("Foo", {});
  EXPECT_NE(foo->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr(),
            input1->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr());
  EXPECT_NE(foo->AppendInputs({input1}), 0);
  const auto &input_nodes = foo->GetFastNode()->GetInDataNodes();
  EXPECT_EQ(input_nodes.size(), 0);
  const uint32_t in_data_anchor_size = foo->GetFastNode()->GetDataInNum();
  EXPECT_EQ(in_data_anchor_size, 1);
}
TEST_F(FastValueHolderUt, AppendInputWithMulitSubgraph_Ok) {
  ValueHolder::PopGraphFrame();
  auto root_frame = ValueHolder::PushGraphFrame();
  EXPECT_TRUE(root_frame->IsRootFrame());
  const auto root_graph = root_frame->GetExecuteGraph().get();
  ASSERT_NE(root_graph, nullptr);
  const auto &input1 = ValueHolder::CreateConst("{}", 2);

  const auto &main_node = ValueHolder::CreateVoid<bg::ValueHolder>("Main", {});
  ValueHolder::PushGraphFrame(main_node, "Main");
  const auto &if_holder = ValueHolder::CreateVoid<bg::ValueHolder>("If", {});
  ASSERT_NE(if_holder, nullptr);
  const auto &if_node =
    ge::ExecuteGraphUtils::FindFirstNodeMatchType(ValueHolder::GetCurrentFrame()->GetExecuteGraph().get(), "If");
  ASSERT_NE(if_node, nullptr);

  ValueHolder::PushGraphFrame(if_holder, "then");
  const auto &then_foo = ValueHolder::CreateVoid<ValueHolder>("Foo", {});
  EXPECT_EQ(then_foo->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr()->GetParentNodeBarePtr()->GetType(), "If");
  ValueHolder::PopGraphFrame();

  ValueHolder::PushGraphFrame(if_holder, "else");
  const auto &else_foo = ValueHolder::CreateVoid<ValueHolder>("Foo", {});
  EXPECT_EQ(else_foo->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr()->GetParentNodeBarePtr()->GetType(), "If");
  ValueHolder::PopGraphFrame();

  EXPECT_EQ(then_foo->AppendInputs({input1}), 0);
  const auto &then_inputs = then_foo->GetFastNode()->GetInDataNodes();
  EXPECT_EQ(then_inputs.size(), 1);
  const auto &if_inputs1 = if_holder->GetFastNode()->GetAllInNodes();
  EXPECT_EQ(if_inputs1.size(), 1);

  EXPECT_EQ(else_foo->AppendInputs({input1}), 0);
  const auto &else_inputs = else_foo->GetFastNode()->GetInDataNodes();
  EXPECT_EQ(else_inputs.size(), 1);
  const auto &if_inputs2 = if_holder->GetFastNode()->GetAllInNodes();
  EXPECT_EQ(if_inputs2.size(), 1);

  EXPECT_EQ(root_graph->GetAllSubgraphs().size(), 3);
  EXPECT_EQ(if_node->GetOpDescBarePtr()->GetSubgraphInstanceNames().size(), 2);
}
TEST_F(FastValueHolderUt, ConnectFromAncestor_CreateInnerData_ParentGraph) {
  auto data0 = ValueHolder::CreateFeed(0);
  auto data1 = ValueHolder::CreateFeed(1);
  auto data2 = ValueHolder::CreateFeed(2);
  auto foo = ValueHolder::CreateSingleDataOutput("Foo", {data0});
  ValueHolder::PushGraphFrame(foo, "Foo");
  auto sub_foo = ValueHolder::CreateSingleDataOutput("SubFoo", {data1, data2});
  ValueHolder::PopGraphFrame({sub_foo}, {});

  auto frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExecuteGraph().get();
  ASSERT_NE(graph, nullptr);
  CheckGraphGenerally(graph);

  EXPECT_EQ(ExeGraphSummaryChecker(graph).StrictDirectNodeTypes({{"Data", 3}, {"Foo", 1}}), "success");
  EXPECT_EQ(graph->GetAllSubgraphs().size(), 1);

  auto foo_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(graph, "Foo");
  ASSERT_NE(foo_node, nullptr);
  EXPECT_EQ(FastNodeTopoChecker(foo).StrictConnectFrom({data0, data1, data2}), "success");
  StrictSubgraphs(foo_node, {"Foo"});
  auto subgraph_name = foo_node->GetOpDescBarePtr()->GetSubgraphInstanceName(0);
  auto subgraph = graph->GetSubGraph(subgraph_name);
  ASSERT_NE(subgraph, nullptr);
  auto ret = gert::ExeGraphSummaryChecker(subgraph).StrictAllNodeTypes({{"InnerData", 2}, {"SubFoo", 1}, {"InnerNetOutput", 1}});
  EXPECT_EQ(ret, "success") << ret;

  auto sub_foo_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(subgraph, "SubFoo");
  ASSERT_NE(sub_foo_node, nullptr);
  ConnectFromInnerData(sub_foo_node, {1, 2});
  EXPECT_EQ(FastNodeTopoChecker(sub_foo_node).StrictConnectTo(0, {{"InnerNetOutput"}}), "success");
}

TEST_F(FastValueHolderUt, ConnectFromAncestor_InnerDataWithGuarderOutside) {
  auto data0 = ValueHolder::CreateFeed(0);
  auto data1 = ValueHolder::CreateFeed(1);
  auto data2 = ValueHolder::CreateFeed(2);
  auto foo = ValueHolder::CreateSingleDataOutput("Foo", {data0});
  auto data1_guarder = ValueHolder::CreateVoidGuarder("FreeMemory", data1, {});
  ValueHolder::PushGraphFrame(foo, "Foo");
  auto sub_foo = ValueHolder::CreateSingleDataOutput("SubFoo", {data1, data2});

  auto sub_frame = ValueHolder::PopGraphFrame({sub_foo}, {});
  auto subgraph = sub_frame->GetExecuteGraph().get();
  ASSERT_NE(subgraph, nullptr);
  auto innerdata_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(subgraph, "InnerData");
  std::string guarder_type_outside;
  (void) ge::AttrUtils::GetStr(innerdata_node->GetOpDescBarePtr(), kNodeWithGuarderOutside, guarder_type_outside);
  EXPECT_EQ(!guarder_type_outside.empty(), true);
  EXPECT_EQ(guarder_type_outside, "FreeMemory");

  const auto &data_node = data1->GetFastNode();
  ASSERT_NE(data_node, nullptr);
  const auto &data_desc = data_node->GetOpDescBarePtr();
  ASSERT_NE(data_desc, nullptr);
  string guarder_type;
  EXPECT_TRUE(ge::AttrUtils::GetStr(data_desc, kGuarderNodeType, guarder_type));
  EXPECT_EQ(guarder_type, "FreeMemory");
}

TEST_F(FastValueHolderUt, ConnectFromAncestor_InnerDataWithGuarderOutside_In_Subgraph_Nesting) {
  auto data0 = ValueHolder::CreateFeed(0);
  auto data1 = ValueHolder::CreateFeed(1);
  auto data2 = ValueHolder::CreateFeed(2);
  auto foo = ValueHolder::CreateSingleDataOutput("Foo", {data0});
  auto data2_guarder = ValueHolder::CreateVoidGuarder("FreeFftsMem", data2, {});

  ValueHolder::PushGraphFrame(foo, "Foo");
  auto sub_foo = ValueHolder::CreateSingleDataOutput("SubFoo", {data1});

  ValueHolder::PushGraphFrame(sub_foo, "SubFoo");
  auto sub_sub_foo = ValueHolder::CreateSingleDataOutput("SubFoo", {data2});

  auto sub_sub_frame = ValueHolder::PopGraphFrame({sub_sub_foo}, {});
  auto sub_sub_graph = sub_sub_frame->GetExecuteGraph().get();

  auto innerdata_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(sub_sub_graph, "InnerData");
  std::string guarder_type_outside;
  (void) ge::AttrUtils::GetStr(innerdata_node->GetOpDescBarePtr(), kNodeWithGuarderOutside, guarder_type_outside);
  EXPECT_EQ(!guarder_type_outside.empty(), true);
  EXPECT_EQ(guarder_type_outside, "FreeFftsMem");

  const auto &data_node = data2->GetFastNode();
  ASSERT_NE(data_node, nullptr);
  const auto &data_desc = data_node->GetOpDescBarePtr();
  ASSERT_NE(data_desc, nullptr);
  string guarder_type;
  EXPECT_TRUE(ge::AttrUtils::GetStr(data_desc, kGuarderNodeType, guarder_type));
  EXPECT_EQ(guarder_type, "FreeFftsMem");
}

/*
 * +-----------------------------+
 * |Foo                          |
 * |   +---------------------+   |
 * |   |SubFoo               |   |
 * |   |    NetOutput        |   |
 * |   |        |            |   |
 * |   |      Sub2Foo3       |   |
 * |   |       /    \        |   |
 * |   | Sub2Foo1  Sub2Foo2  |   |
 * |   |   |       /   |     |   |
 * |   +---+-----+-----+-----+   |
 * |       |     |     |         |
 * +-------+-----+-----+---------+
 *    /    |     |     |
 * data0 data1 data2 data3
 */
TEST_F(FastValueHolderUt, ConnectFromAncestor_CreateInnerDataRecursively_AncestorGraph) {
  auto data0 = ValueHolder::CreateFeed(0);
  auto data1 = ValueHolder::CreateFeed(1);
  auto data2 = ValueHolder::CreateFeed(2);
  auto data3 = ValueHolder::CreateFeed(3);

  auto foo = ValueHolder::CreateSingleDataOutput("Foo", {data0});
  ValueHolder::PushGraphFrame(foo, "Foo");

  auto sub_foo = ValueHolder::CreateSingleDataOutput("SubFoo", {data1});
  ValueHolder::PushGraphFrame(sub_foo, "Foo");

  auto sub2_foo1 = ValueHolder::CreateSingleDataOutput("Sub2Foo1", {data1});
  auto sub2_foo2 = ValueHolder::CreateSingleDataOutput("Sub2Foo2", {data3, data2});
  auto sub2_foo3 = ValueHolder::CreateSingleDataOutput("Sub2Foo3", {sub2_foo1, sub2_foo2});

  ValueHolder::PopGraphFrame({sub2_foo3}, {});
  ValueHolder::PopGraphFrame({sub_foo}, {});
  auto frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExecuteGraph().get();
  ASSERT_NE(graph, nullptr);
  CheckGraphGenerally(graph);

  // Check elements on root graph
  EXPECT_EQ(ExeGraphSummaryChecker(graph).StrictDirectNodeTypes({{"Data", 4}, {"Foo", 1}}), "success");
  EXPECT_EQ(graph->GetAllSubgraphs().size(), 2);
  auto foo_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(graph, "Foo");
  
  ASSERT_NE(foo_node, nullptr);
  EXPECT_EQ(FastNodeTopoChecker(foo_node).StrictConnectFrom({data0, data1, data3, data2}), "success");
  StrictSubgraphs(foo_node, {"Foo"});
  ASSERT_EQ(foo_node->GetOpDescBarePtr()->GetSubgraphInstanceNames().size(), 1);
  auto foo_graph = graph->GetSubGraph(foo_node->GetOpDescBarePtr()->GetSubgraphInstanceName(0));
  ASSERT_NE(foo_graph, nullptr);

  // Check elements on foo graph
  ASSERT_EQ(ExeGraphSummaryChecker(foo_graph).StrictDirectNodeTypes({{"InnerData", 3}, {"SubFoo", 1}, {"InnerNetOutput", 1}}),
            "success");
  auto sub_foo_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(foo_graph, "SubFoo");
  ASSERT_NE(sub_foo_node, nullptr);
  ASSERT_EQ(FastNodeTopoChecker(sub_foo_node).StrictConnectFrom({{"InnerData"}, {"InnerData"}, {"InnerData"}}),
                                                                      "success");
  ASSERT_EQ(FastNodeTopoChecker(sub_foo_node).StrictConnectTo(0, {{"InnerNetOutput"}}), "success");
  StrictSubgraphs(sub_foo_node, {"Foo"});
  auto subfoo_graph = graph->GetSubGraph(sub_foo_node->GetOpDescBarePtr()->GetSubgraphInstanceName(0));
  ASSERT_NE(subfoo_graph, nullptr);

  // Check elements on SubFoo graph
  auto ret = gert::ExeGraphSummaryChecker(subfoo_graph)
                 .StrictAllNodeTypes(
                     {{"InnerData", 3}, {"Sub2Foo1", 1}, {"Sub2Foo2", 1}, {"Sub2Foo3", 1}, {"InnerNetOutput", 1}});
  auto sub2_foo1_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(subfoo_graph, "Sub2Foo1");

  ASSERT_NE(sub2_foo1_node, nullptr);
  EXPECT_EQ(FastNodeTopoChecker(sub2_foo1_node).StrictConnectFrom({{"InnerData"}}), "success");
  ConnectFromOuter(sub2_foo1_node, 0, FindData(graph, 1), 0);

  auto sub2_foo2_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(subfoo_graph, "Sub2Foo2");
  ASSERT_NE(sub2_foo2_node, nullptr);
  EXPECT_EQ(FastNodeTopoChecker(sub2_foo2_node).StrictConnectFrom({{"InnerData"}, {"InnerData"}}), "success");
  ConnectFromOuter(sub2_foo2_node, 0, FindData(graph, 3), 0);
  ConnectFromOuter(sub2_foo2_node, 1, FindData(graph, 2), 0);

  auto sub2_foo3_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(subfoo_graph, "Sub2Foo3");
  ASSERT_NE(sub2_foo3_node, nullptr);
  ASSERT_EQ(FastNodeTopoChecker(sub2_foo3_node).StrictConnectFrom({sub2_foo1_node, sub2_foo2_node}), "success");
}

/*
 * +---------------------------------------+
 * |Foo                                    |
 * |   +-------------------------------+   |
 * |   |SubFoo                         |   |
 * |   |         NetOutput             |   |
 * |   |             |                 |   |
 * |   |          Sub2Foo3             |   |
 * |   |       /     |      \          |   |
 * |   | Sub2Foo1  Sub2Foo2   Sub2Foo4 |   |
 * |   |   |       /      \ /          |   |
 * |   +---+-----+---------+-----------+   |
 * |       |     |         |               |
 * +-------+-----+---------+---------------+
 *    /    |     |         |
 * data0 data1 data2     data3
 */
TEST_F(FastValueHolderUt, ConnectFromAncestor_DeDuplicate_SameSrc) {
  auto data0 = ValueHolder::CreateFeed(0);
  auto data1 = ValueHolder::CreateFeed(1);
  auto data2 = ValueHolder::CreateFeed(2);
  auto data3 = ValueHolder::CreateFeed(3);

  auto foo = ValueHolder::CreateSingleDataOutput("Foo", {data0});
  ValueHolder::PushGraphFrame(foo, "Foo");

  auto sub_foo = ValueHolder::CreateSingleDataOutput("SubFoo", {data1});
  ValueHolder::PushGraphFrame(sub_foo, "Foo");

  auto sub2_foo1 = ValueHolder::CreateSingleDataOutput("Sub2Foo1", {data1});
  auto sub2_foo2 = ValueHolder::CreateSingleDataOutput("Sub2Foo2", {data3, data2});
  auto sub2_foo4 = ValueHolder::CreateSingleDataOutput("Sub2Foo4", {data3});
  auto sub2_foo3 = ValueHolder::CreateSingleDataOutput("Sub2Foo3", {sub2_foo1, sub2_foo2, sub2_foo4});

  ValueHolder::PopGraphFrame({sub2_foo3}, {});
  ValueHolder::PopGraphFrame({sub_foo}, {});
  auto frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExecuteGraph().get();
  ASSERT_NE(graph, nullptr);
  CheckGraphGenerally(graph);

  auto foo_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(graph, "Foo");
  ASSERT_NE(foo_node, nullptr);
  ASSERT_EQ(FastNodeTopoChecker(foo_node).StrictConnectFrom({{"Data"}, {"Data"}, {"Data"}, {"Data"}}), "success");

  auto sub_foo_node = ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(graph, "SubFoo")[0];
  ASSERT_NE(sub_foo_node, nullptr);
  ASSERT_EQ(FastNodeTopoChecker(sub_foo_node).StrictConnectFrom({{"InnerData"}, {"InnerData"}, {"InnerData"}}),
                                                                      "success");
  auto sub_foo_graph = FindFirstSubgraphForNodeType(graph, "SubFoo");
  ASSERT_NE(sub_foo_graph, nullptr);
  auto sub2_foo2_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(sub_foo_graph, "Sub2Foo2");
  ASSERT_NE(sub2_foo2_node, nullptr);
  ASSERT_EQ(FastNodeTopoChecker(sub2_foo2_node).StrictConnectFrom({{"InnerData"}, {"InnerData"}}), "success");
  ConnectFromOuter(sub2_foo2_node, 0, FindData(graph, 3), 0);
  ConnectFromOuter(sub2_foo2_node, 1, FindData(graph, 2), 0);

  auto sub2_foo4_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(sub_foo_graph, "Sub2Foo4");
  ASSERT_EQ(FastNodeTopoChecker(sub2_foo4_node).StrictConnectFrom({{"InnerData"}}), "success");
  ConnectFromOuter(sub2_foo4_node, 0, FindData(graph, 3), 0);

  auto inner_data_from_3 = sub2_foo2_node->GetInDataEdgeByIndex(0)->src;
  ASSERT_EQ(FastNodeTopoChecker(inner_data_from_3).StrictConnectTo(0, {sub2_foo2_node, sub2_foo4_node}),
                                                                       "success");
}
TEST_F(FastValueHolderUt, PopFrame_CreateControlEdge_Targets) {
  auto data0 = ValueHolder::CreateFeed(0);
  auto data1 = ValueHolder::CreateFeed(1);
  auto foo = ValueHolder::CreateSingleDataOutput("Foo", {data0, data1});
  auto frame = ValueHolder::PopGraphFrame({}, {data0, foo});

  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExecuteGraph().get();
  ASSERT_NE(graph, nullptr);

  ASSERT_EQ(ExeGraphSummaryChecker(graph).StrictAllNodeTypes({{"Data", 2}, {"Foo", 1}, {"NetOutput", 1}}), "success");

  auto netoutput = ge::ExecuteGraphUtils::FindFirstNodeMatchType(graph, "NetOutput");
  ASSERT_NE(netoutput, nullptr);
  ASSERT_EQ(FastNodeTopoChecker(netoutput).StrictConnectFrom({{"Data", -1}, {"Foo", -1}}), "success");
  EXPECT_EQ(netoutput->GetInDataNodes().size(), 0);
  EXPECT_EQ(netoutput->GetInControlNodes().size(), 2);
}
TEST_F(FastValueHolderUt, PopFrame_CreateNetOuptut_PopRootGraph) {
  auto data0 = ValueHolder::CreateFeed(0);
  auto data1 = ValueHolder::CreateFeed(1);
  auto foo = ValueHolder::CreateSingleDataOutput("Foo", {data0, data1});
  auto frame = ValueHolder::PopGraphFrame({data0, foo}, {});

  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExecuteGraph().get();
  ASSERT_NE(graph, nullptr);

  ASSERT_EQ(ExeGraphSummaryChecker(graph).StrictAllNodeTypes({{"Data", 2}, {"Foo", 1}, {"NetOutput", 1}}), "success");

  auto netoutput = ge::ExecuteGraphUtils::FindFirstNodeMatchType(graph, "NetOutput");
  ASSERT_NE(netoutput, nullptr);
  EXPECT_EQ(netoutput->GetName(), "NetOutput");
  ASSERT_EQ(FastNodeTopoChecker(netoutput).StrictConnectFrom({data0, foo}), "success");

  auto foo_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(graph, "Foo");
  ASSERT_NE(foo_node, nullptr);
  ASSERT_EQ(FastNodeTopoChecker(foo_node).StrictConnectFrom({data0, data1}), "success");
  ASSERT_EQ(FastNodeTopoChecker(foo_node).StrictConnectTo(0, {netoutput}), "success");
}
TEST_F(FastValueHolderUt, PopFrame_CreateInnerNetOuptut_PopSubgraph) {
  auto data0 = ValueHolder::CreateFeed(0);
  auto data1 = ValueHolder::CreateFeed(1);
  auto foo = ValueHolder::CreateSingleDataOutput("Foo", {data0, data1});
  ValueHolder::PushGraphFrame(foo, "Foo");
  auto bar1 = ValueHolder::CreateSingleDataOutput("Bar1", {data0});
  auto bar2 = ValueHolder::CreateSingleDataOutput("Bar2", {data1});
  auto frame = ValueHolder::PopGraphFrame({bar1}, {bar2});

  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExecuteGraph().get();
  ASSERT_NE(graph, nullptr);
  ASSERT_EQ(ExeGraphSummaryChecker(graph).StrictAllNodeTypes(
    {{"InnerData", 2}, {"Bar1", 1}, {"Bar2", 1}, {"InnerNetOutput", 1}}), "success");
  auto netoutput = ge::ExecuteGraphUtils::FindFirstNodeMatchType(graph, "InnerNetOutput");

  ASSERT_NE(netoutput, nullptr);
  ASSERT_EQ(FastNodeTopoChecker(netoutput).StrictConnectFrom({{"Bar1"}, {"Bar2"}}), "success");
}
/*
 * +-----------------------------+
 * |Foo                          |
 * |   +---------------------+   |
 * |   |SubFoo               |   |
 * |   |    NetOutput        |   |
 * |   |  /   |      \       |   |
 * |   | | foo5      |       |   |
 * |   | |  |        |       |   |
 * |   +-+--+--------+-------+   |
 * |     |  |        |           |
 * |    /  Foo2    Foo3          |
 * |   |  /       /     \        |
 * +---+---------+--------+------+
 *     |         |        |
 *  data0      data1    data2
 */
TEST_F(FastValueHolderUt, PopFrame_CraeteInnerData_OutputsUseParentHolder) {
  auto data0 = ValueHolder::CreateFeed(0);
  auto data1 = ValueHolder::CreateFeed(1);
  auto data2 = ValueHolder::CreateFeed(2);

  auto foo1 = ValueHolder::CreateSingleDataOutput("Foo", {data0, data1, data2});
  ValueHolder::PushGraphFrame(foo1, "Foo");

  auto foo2 = ValueHolder::CreateSingleDataOutput("Foo2", {data0, data1});
  auto foo3 = ValueHolder::CreateDataOutput("Foo3", {data1, data2}, 3);

  auto foo4 = ValueHolder::CreateSingleDataOutput("Foo4", {foo2, foo3[0]});
  ValueHolder::PushGraphFrame(foo4, "Foo");
  auto foo5 = ValueHolder::CreateSingleDataOutput("Foo5", {foo2});

  ValueHolder::PopGraphFrame({data0, foo3[1], foo5}, {});
  ValueHolder::PopGraphFrame({}, {});

  auto frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExecuteGraph().get();
  ASSERT_NE(graph, nullptr);

  auto foo3_nodes = ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(graph, "Foo3");
  ASSERT_EQ(foo3_nodes.size(), 1);

  auto foo4_graph = FindFirstSubgraphForNodeType(graph, "Foo4");
  ASSERT_NE(foo4_graph, nullptr);
  ASSERT_EQ(ExeGraphSummaryChecker(foo4_graph).StrictAllNodeTypes({{"InnerData", 3}, {"Foo5", 1}, {"InnerNetOutput", 1}}),
            "success");
  auto netoutput = ge::ExecuteGraphUtils::FindFirstNodeMatchType(foo4_graph, "InnerNetOutput");
  ASSERT_EQ(FastNodeTopoChecker(netoutput).StrictConnectFrom({{"InnerData"}, {"InnerData"}, {"Foo5"}}), "success");
  ConnectFromOuter(netoutput, 0, FindData(graph, 0), 0);
  ConnectFromOuter(netoutput, 1, foo3_nodes[0], 1);
}

/*
 * +--------------------------------------------------------+
 * |Foo-Node                                                |
 * |   +---------------------+    +---------------------+   |
 * |   |Foo-Subgraph1        |    |Foo-Subgraph2        |   |
 * |   |   NetOutput         |    |          NetOutput  |   |
 * |   |      |              |    |     ERROR   |       |   |
 * |   |     Bar1            |    | Bar1 --->  Bar2     |   |
 * |   |   /    \            |    |          /    \     |   |
 * |   +--0------1-----------+    +---------0------1----+   |
 * +------0------1--------2---------------------------------+
 *        |      |        |
 *     data0   data1    data2
 */
TEST_F(FastValueHolderUt, PopFrame_Failed_OutpusUseGraphNotAncestor) {
  auto data0 = ValueHolder::CreateFeed(0);
  auto data1 = ValueHolder::CreateFeed(1);
  auto data2 = ValueHolder::CreateFeed(2);

  auto foo1 = ValueHolder::CreateSingleDataOutput("Foo", {data0, data1, data2});
  ValueHolder::PushGraphFrame(foo1, "Foo1");
  auto bar1 = ValueHolder::CreateDataOutput("Bar1", {data0, data1}, 3);
  ValueHolder::PopGraphFrame({bar1[0]}, {});

  ValueHolder::PushGraphFrame(foo1, "Foo2");
  auto bar2 = ValueHolder::CreateSingleDataOutput("Bar2", {bar1[0], data0, data1});
  ASSERT_EQ(bar2, nullptr);
  bar2 = ValueHolder::CreateSingleDataOutput("Bar2", {bar1[1], data0, data1});
  ASSERT_EQ(bar2, nullptr);
  ValueHolder::PopGraphFrame();
}

TEST_F(FastValueHolderUt, CreateConstDataOk) {
  auto const_data1 = ValueHolder::CreateConstData(0);
  auto data1 = ValueHolder::CreateFeed(0);
  ASSERT_NE(const_data1, nullptr);
  ASSERT_NE(data1, nullptr);

  std::vector<ValueHolderPtr> inputs = {data1, const_data1};
  auto holder = ValueHolder::CreateVoid<ValueHolder>("TestNode", inputs);

  ASSERT_NE(holder, nullptr);

  ASSERT_NE(const_data1->GetExecuteGraph(), nullptr);
  ASSERT_NE(const_data1->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr(), nullptr);
  ASSERT_NE(data1->GetExecuteGraph(), nullptr);
  ASSERT_NE(data1->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr(), nullptr);
  ASSERT_NE(holder->GetFastNode(), nullptr);
  ASSERT_NE(holder->GetExecuteGraph(), nullptr);
  ASSERT_NE(holder->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr(), nullptr);

  // check graph is ok
  auto graph = holder->GetExecuteGraph();
  ASSERT_EQ(graph->GetAllNodes().size(), 3);
  CheckGraphGenerally(graph);

  auto const1_g = ge::ExecuteGraphUtils::FindFirstNodeMatchType(graph, "ConstData");
  auto data1_g = ge::ExecuteGraphUtils::FindFirstNodeMatchType(graph, "Data");
  auto node_g = ge::ExecuteGraphUtils::FindFirstNodeMatchType(graph, "TestNode");
  ASSERT_NE(const1_g, nullptr);
  ASSERT_NE(data1_g, nullptr);
  ASSERT_NE(node_g, nullptr);

  EXPECT_EQ(node_g->GetInDataEdgeByIndex(0)->src, data1_g);
  EXPECT_EQ(node_g->GetInDataEdgeByIndex(1)->src, const1_g);
}

TEST_F(FastValueHolderUt, ValueHolderUtils_TEST_CheckNode_OK) {
  auto data1 = ValueHolder::CreateFeed(0);
  ASSERT_NE(data1, nullptr);
  ge::Format f1 = ge::FORMAT_NC1HWC0;
  auto const1 = ValueHolder::CreateConst(reinterpret_cast<const uint8_t *>(&f1), sizeof(f1));
  ASSERT_NE(const1, nullptr);
  auto foo1 = ValueHolder::CreateSingleDataOutput("Foo1", {data1, const1});
  auto foo2 = ValueHolder::CreateDataOutput("Foo2", {foo1, const1, data1}, 2);
  ASSERT_EQ(foo2.size(), 2);

  ASSERT_FALSE(ValueHolderUtils::IsNodeValid(nullptr));
  ASSERT_TRUE(ValueHolderUtils::IsNodeValid(foo1));
  ASSERT_TRUE(ValueHolderUtils::IsNodeValid(foo2[0]));
  ASSERT_TRUE(ValueHolderUtils::IsNodeValid(foo2[1]));

  ASSERT_TRUE(ValueHolderUtils::IsNodeEqual(nullptr, nullptr));
  ASSERT_FALSE(ValueHolderUtils::IsNodeEqual(foo1, nullptr));
  ASSERT_FALSE(ValueHolderUtils::IsNodeEqual(nullptr, foo1));
  ASSERT_TRUE(ValueHolderUtils::IsNodeEqual(foo2[0], foo2[1]));
}

TEST_F(FastValueHolderUt, ValueHolderUtils_Test_BaseInfo_OK) {
  ValueHolderPtr holder = nullptr;
  EXPECT_STREQ(ValueHolderUtils::GetNodeName(holder).c_str(), "");
  EXPECT_STREQ(ValueHolderUtils::GetNodeNameBarePtr(holder), "");
  EXPECT_STREQ(ValueHolderUtils::GetNodeType(holder).c_str(), "");
  EXPECT_STREQ(ValueHolderUtils::GetNodeTypeBarePtr(holder), "");
  auto desc = ValueHolderUtils::GetNodeOpDesc(holder);
  EXPECT_EQ(desc, nullptr);
  EXPECT_EQ(desc.get(), ValueHolderUtils::GetNodeOpDescBarePtr(holder));

  auto data1 = ValueHolder::CreateFeed(0);
  ASSERT_NE(data1, nullptr);
  EXPECT_STRNE(ValueHolderUtils::GetNodeName(data1).c_str(), "");
  EXPECT_STRNE(ValueHolderUtils::GetNodeNameBarePtr(data1), "");
  EXPECT_STREQ(ValueHolderUtils::GetNodeType(data1).c_str(), "Data");
  EXPECT_STREQ(ValueHolderUtils::GetNodeTypeBarePtr(data1), "Data");
  desc = ValueHolderUtils::GetNodeOpDesc(data1);
  EXPECT_NE(desc, nullptr);
  EXPECT_EQ(desc.get(), ValueHolderUtils::GetNodeOpDescBarePtr(data1));
  EXPECT_EQ(desc->GetAllInputsSize(), 0);
  EXPECT_EQ(desc->GetAllOutputsDescSize(), 1);

  ge::Format f1 = ge::FORMAT_NC1HWC0;
  auto const1 = ValueHolder::CreateConst(reinterpret_cast<const uint8_t *>(&f1), sizeof(f1));
  ASSERT_NE(const1, nullptr);
  EXPECT_STRNE(ValueHolderUtils::GetNodeName(const1).c_str(), "");
  EXPECT_STRNE(ValueHolderUtils::GetNodeNameBarePtr(const1), "");
  EXPECT_STREQ(ValueHolderUtils::GetNodeType(const1).c_str(), "Const");
  EXPECT_STREQ(ValueHolderUtils::GetNodeTypeBarePtr(const1), "Const");
  desc = ValueHolderUtils::GetNodeOpDesc(const1);
  EXPECT_NE(desc, nullptr);
  EXPECT_EQ(desc.get(), ValueHolderUtils::GetNodeOpDescBarePtr(const1));
  EXPECT_EQ(desc->GetAllInputsSize(), 0);
  EXPECT_EQ(desc->GetAllOutputsDescSize(), 1);

  auto foo1 = ValueHolder::CreateSingleDataOutput("Foo1", {data1, const1});
  ASSERT_NE(foo1, nullptr);
  EXPECT_STRNE(ValueHolderUtils::GetNodeName(foo1).c_str(), "");
  EXPECT_STRNE(ValueHolderUtils::GetNodeNameBarePtr(foo1), "");
  EXPECT_STREQ(ValueHolderUtils::GetNodeType(foo1).c_str(), "Foo1");
  EXPECT_STREQ(ValueHolderUtils::GetNodeTypeBarePtr(foo1), "Foo1");
  desc = ValueHolderUtils::GetNodeOpDesc(foo1);
  EXPECT_NE(desc, nullptr);
  EXPECT_EQ(desc.get(), ValueHolderUtils::GetNodeOpDescBarePtr(foo1));
  EXPECT_EQ(desc->GetAllInputsSize(), 2);
  EXPECT_EQ(desc->GetAllOutputsDescSize(), 1);

  auto foo2 = ValueHolder::CreateDataOutput("Foo2", {foo1, const1, data1}, 2);
  ASSERT_EQ(foo2.size(), 2);
  ASSERT_NE(foo2[0], nullptr);
  EXPECT_STRNE(ValueHolderUtils::GetNodeName(foo2[0]).c_str(), "");
  EXPECT_STRNE(ValueHolderUtils::GetNodeNameBarePtr(foo2[0]), "");
  EXPECT_STREQ(ValueHolderUtils::GetNodeType(foo2[0]).c_str(), "Foo2");
  EXPECT_STREQ(ValueHolderUtils::GetNodeTypeBarePtr(foo2[0]), "Foo2");
  desc = ValueHolderUtils::GetNodeOpDesc(foo2[0]);
  EXPECT_NE(desc, nullptr);
  EXPECT_EQ(desc.get(), ValueHolderUtils::GetNodeOpDescBarePtr(foo2[0]));
  EXPECT_EQ(desc->GetAllInputsSize(), 3);
  EXPECT_EQ(desc->GetAllOutputsDescSize(), 2);

  ASSERT_NE(foo2[1], nullptr);
  EXPECT_STRNE(ValueHolderUtils::GetNodeName(foo2[1]).c_str(), "");
  EXPECT_STRNE(ValueHolderUtils::GetNodeNameBarePtr(foo2[1]), "");
  EXPECT_STREQ(ValueHolderUtils::GetNodeType(foo2[1]).c_str(), "Foo2");
  EXPECT_STREQ(ValueHolderUtils::GetNodeTypeBarePtr(foo2[1]), "Foo2");
  desc = ValueHolderUtils::GetNodeOpDesc(foo2[1]);
  EXPECT_NE(desc, nullptr);
  EXPECT_EQ(desc.get(), ValueHolderUtils::GetNodeOpDescBarePtr(foo2[1]));
  EXPECT_EQ(desc->GetAllInputsSize(), 3);
  EXPECT_EQ(desc->GetAllOutputsDescSize(), 2);
}

/*
 *    hello
 *    /  \
 * data0 data1
 */
TEST_F(FastValueHolderUt, IsDirectlyControlled) {
  auto op_desc = std::make_shared<ge::OpDesc>("node", "node");
  ge::GeTensorDesc tensor_desc;
  tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  tensor_desc.SetFormat(ge::FORMAT_NC1HWC0);
  tensor_desc.SetDataType(ge::DT_FLOAT16);
  tensor_desc.SetOriginDataType(ge::DT_FLOAT);
  tensor_desc.SetShape(ge::GeShape({8, 1, 224, 224, 16}));
  tensor_desc.SetOriginShape(ge::GeShape({8, 3, 224, 224}));
  op_desc->AddInputDesc("x1", tensor_desc);
  op_desc->AppendIrInput("x1", ge::kIrInputRequired);
  op_desc->AppendIrInput("x2", ge::kIrInputOptional);

  auto graph = std::make_shared<ge::ComputeGraph>("graph");
  auto node = graph->AddNode(op_desc);

  auto data0 = ValueHolder::CreateFeed(0);
  auto data1 = ValueHolder::CreateFeed(1);

  ValueHolder::SetCurrentComputeNode(node);
  auto hello = ValueHolder::CreateSingleDataOutput("hello", {data0, data1});

  (void)bg::ValueHolder::AddDependency(hello, data0);
  EXPECT_EQ(ValueHolderUtils::IsDirectlyControlled(hello, data0), true);
  EXPECT_EQ(ValueHolderUtils::IsDirectlyControlled(data0, data1), false);
}

TEST_F(FastValueHolderUt, ClearGraphFrameSucc) {
  EXPECT_NE(ValueHolder::GetCurrentFrame(), nullptr);
  ValueHolder::ClearGraphFrameResource();
  EXPECT_EQ(ValueHolder::GetCurrentFrame(), nullptr);
  EXPECT_EQ(ValueHolder::PopGraphFrame(), nullptr);
}
}  // namespace bg
}  // namespace gert
