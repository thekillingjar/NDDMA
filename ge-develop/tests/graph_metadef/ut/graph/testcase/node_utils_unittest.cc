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

#include "graph/utils/node_utils.h"
#include "graph/utils/node_utils_ex.h"
#include "graph/normal_graph/node_impl.h"
#include "graph/normal_graph/op_desc_impl.h"
#include "graph_builder_utils.h"
#include "graph/debug/ge_op_types.h"
#include "graph/operator_factory_impl.h"
#include "graph/normal_graph/compute_graph_impl.h"
#include "graph/anchor.h"
#include "graph/debug/ge_attr_define.h"

#include "auto_mapping_util.h"
#include "graph/operator_reg.h"

#include <dlog_pub.h>

namespace ge {
class UtestNodeUtils : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

namespace {
REG_OP(FakeData)
.INPUT(x, TensorType::NumberType())
.OUTPUT(y, TensorType::NumberType())
.OP_END_FACTORY_REG(FakeData);

REG_OP(FakeOpNoOutput)
.INPUT(x, TensorType::NumberType())
.OP_END_FACTORY_REG(FakeOpNoOutput);

template<class T>
std::shared_ptr<T> MakeNullptr(){
  return nullptr;
}

/*                                  -------------------------
*                                  |  partitioncall_0_const1* |
*     partitioncall_0--------------|             |           |
*           |                      |          netoutput      |
*           |                      --------------------------
*           |                       ------------------         -------------
*           |                      |        data      |       |    data     |
*           |                      |          |       |       |     |       |
*     partitioncall_1--------------|        case -----|-------|   squeeze*  |
*                                  |          |       |       |     |       |
*                                  |      netoutput   |       |  netoutput  |
*                                   ------------------         -------------
*/
ComputeGraphPtr BuildGraphPartitionCall() {
  auto root_builder = ut::GraphBuilder("root");
  const auto &partitioncall_0 = root_builder.AddNode("partitioncall_0", PARTITIONEDCALL, 0, 1);
  const auto &partitioncall_1 = root_builder.AddNode("partitioncall_1", PARTITIONEDCALL, 1, 1);
  root_builder.AddDataEdge(partitioncall_0, 0, partitioncall_1, 0);
  const auto &root_graph = root_builder.GetGraph();

  // 1.build partitioncall_0 sub graph
  auto p1_sub_builder = ut::GraphBuilder("partitioncall_0_sub");
  const auto &partitioncall_0_const1 = p1_sub_builder.AddNode("partitioncall_0_const1", CONSTANT, 0, 1);
  const auto &partitioncall_0_netoutput = p1_sub_builder.AddNode("partitioncall_0_netoutput", NETOUTPUT, 1, 1);
  AttrUtils::SetInt(partitioncall_0_netoutput->GetOpDesc()->MutableInputDesc(0), "_parent_node_index", 0);
  p1_sub_builder.AddDataEdge(partitioncall_0_const1, 0, partitioncall_0_netoutput, 0);
  const auto &sub_graph = p1_sub_builder.GetGraph();
  sub_graph->SetParentNode(partitioncall_0);
  sub_graph->SetParentGraph(root_graph);
  partitioncall_0->GetOpDesc()->AddSubgraphName("f");
  partitioncall_0->GetOpDesc()->SetSubgraphInstanceName(0, "partitioncall_0_sub");

  // 2.build partitioncall_1 sub graph
  auto p2_sub_builder = ut::GraphBuilder("partitioncall_1_sub");
  const auto &partitioncall_1_data = p2_sub_builder.AddNode("partitioncall_1_data", DATA, 0, 1);
  AttrUtils::SetInt(partitioncall_1_data->GetOpDesc(), "_parent_node_index", 0);
  const auto &partitioncall_1_case = p2_sub_builder.AddNode("partitioncall_1_case", "Case", 1, 1);
  const auto &partitioncall_1_netoutput = p2_sub_builder.AddNode("partitioncall_1_netoutput", NETOUTPUT, 1, 1);
  p2_sub_builder.AddDataEdge(partitioncall_1_data, 0, partitioncall_1_case, 0);
  p2_sub_builder.AddDataEdge(partitioncall_1_case, 0, partitioncall_1_netoutput, 0);
  const auto &sub_graph2 = p2_sub_builder.GetGraph();
  sub_graph2->SetParentNode(partitioncall_1);
  sub_graph2->SetParentGraph(root_graph);
  partitioncall_1->GetOpDesc()->AddSubgraphName("f");
  partitioncall_1->GetOpDesc()->SetSubgraphInstanceName(0, "partitioncall_1_sub");

  // 2.1 build case sub graph
  auto case_sub_builder = ut::GraphBuilder("case_sub");
  const auto &case_data = case_sub_builder.AddNode("case_data", DATA, 0, 1);
  AttrUtils::SetInt(case_data->GetOpDesc(), "_parent_node_index", 0);
  const auto &case_squeeze = case_sub_builder.AddNode("case_squeeze", SQUEEZE, 1, 1);
  const auto &case_netoutput = case_sub_builder.AddNode("case_netoutput", NETOUTPUT, 1, 1);
  case_sub_builder.AddDataEdge(case_data, 0, case_squeeze, 0);
  case_sub_builder.AddDataEdge(case_squeeze, 0, case_netoutput, 0);
  const auto &case_sub_graph = case_sub_builder.GetGraph();
  case_sub_graph->SetParentNode(partitioncall_1_case);
  case_sub_graph->SetParentGraph(sub_graph2);
  partitioncall_1_case->GetOpDesc()->AddSubgraphName("branches");
  partitioncall_1_case->GetOpDesc()->SetSubgraphInstanceName(0, "case_sub");

  root_graph->AddSubgraph(case_sub_graph->GetName(), case_sub_graph);
  root_graph->AddSubgraph(sub_graph->GetName(), sub_graph);
  root_graph->AddSubgraph(sub_graph2->GetName(), sub_graph2);
  return root_graph;
}
/*                                  -------------------------
*                                  |           data0         |
*         data                     |             |           |
*           |                      |            cast         |
*     partitioncall_0--------------|             |           |
*           |                      |          netoutput      |
*           |                      --------------------------
*           |                       ------------------        
*           |                      |        data1     |    
*           |                      |          |       |    
*     partitioncall_1--------------|        squeeze   |
*                                  |          |       |
*                                  |      netoutput   |
*                                   ------------------ 
*/
ComputeGraphPtr BuildGraphPartitionCall2() {
  auto root_builder = ut::GraphBuilder("root");
  const auto &data = root_builder.AddNode("data", DATA, 1, 1);
  const auto &partitioncall_0 = root_builder.AddNode("partitioncall_0", PARTITIONEDCALL, 3, 3);
  const auto &partitioncall_1 = root_builder.AddNode("partitioncall_1", PARTITIONEDCALL, 1, 1);
  root_builder.AddDataEdge(data, 0, partitioncall_0, 1);
  root_builder.AddDataEdge(partitioncall_0, 1, partitioncall_1, 0);
  const auto &root_graph = root_builder.GetGraph();

  // 1.build partitioncall_0 sub graph
  auto p1_sub_builder = ut::GraphBuilder("partitioncall_0_sub");
  const auto &partitioncall_0_data = p1_sub_builder.AddNode("partitioncall_0_data", DATA, 0, 1);
  AttrUtils::SetInt(partitioncall_0_data->GetOpDesc(), "_parent_node_index", 1);
  const auto &partitioncall_0_cast = p1_sub_builder.AddNode("partitioncall_0_cast", "Cast", 1, 1);
  const auto &partitioncall_0_netoutput = p1_sub_builder.AddNode("partitioncall_0_netoutput", NETOUTPUT, 3, 3);
  AttrUtils::SetInt(partitioncall_0_netoutput->GetOpDesc()->MutableInputDesc(0), "_parent_node_index", 0);
  AttrUtils::SetInt(partitioncall_0_netoutput->GetOpDesc()->MutableInputDesc(1), "_parent_node_index", 1);
  AttrUtils::SetInt(partitioncall_0_netoutput->GetOpDesc()->MutableInputDesc(2), "_parent_node_index", 2);
  p1_sub_builder.AddDataEdge(partitioncall_0_data, 0, partitioncall_0_cast, 0);
  p1_sub_builder.AddDataEdge(partitioncall_0_cast, 0, partitioncall_0_netoutput, 1);
  const auto &sub_graph = p1_sub_builder.GetGraph();
  sub_graph->SetParentNode(partitioncall_0);
  sub_graph->SetParentGraph(root_graph);
  partitioncall_0->GetOpDesc()->AddSubgraphName("f");
  partitioncall_0->GetOpDesc()->SetSubgraphInstanceName(0, "partitioncall_0_sub");

  // 2.build partitioncall_1 sub graph
  auto p2_sub_builder = ut::GraphBuilder("partitioncall_1_sub");
  const auto &partitioncall_1_data = p2_sub_builder.AddNode("partitioncall_1_data", DATA, 0, 1);
  AttrUtils::SetInt(partitioncall_1_data->GetOpDesc(), "_parent_node_index", 0);
  const auto &partitioncall_1_squeeze = p2_sub_builder.AddNode("partitioncall_1_squeeze", SQUEEZE, 1, 1);
  const auto &partitioncall_1_netoutput = p2_sub_builder.AddNode("partitioncall_1_netoutput", NETOUTPUT, 1, 1);
  p2_sub_builder.AddDataEdge(partitioncall_1_data, 0, partitioncall_1_squeeze, 0);
  p2_sub_builder.AddDataEdge(partitioncall_1_squeeze, 0, partitioncall_1_netoutput, 0);
  const auto &sub_graph2 = p2_sub_builder.GetGraph();
  sub_graph2->SetParentNode(partitioncall_1);
  sub_graph2->SetParentGraph(root_graph);
  partitioncall_1->GetOpDesc()->AddSubgraphName("f");
  partitioncall_1->GetOpDesc()->SetSubgraphInstanceName(0, "partitioncall_1_sub");

  root_graph->AddSubgraph(sub_graph->GetName(), sub_graph);
  root_graph->AddSubgraph(sub_graph2->GetName(), sub_graph2);
  return root_graph;
}
/*                                  -------------------------                 -------------------
*                                  |      partitioncall_2---------------------|        Mul       |
*     partitioncall_0--------------|             |           |                |         |        |
*           |                      |         netoutput       |                |     netoutput    |
*           |                      --------------------------                  ------------------
*           |                       -------------
*           |                      |    data     |
*           |                      |     |       |
*     partitioncall_1--------------|   squeeze*  |
*                                  |     |       |
*                                  |  netoutput  |
*                                   -------------
*/
ComputeGraphPtr BuildGraphPartitionCall3() {
  auto root_builder = ut::GraphBuilder("root");
  const auto &partitioncall_0 = root_builder.AddNode("partitioncall_0", PARTITIONEDCALL, 1, 1);
  const auto &partitioncall_1 = root_builder.AddNode("partitioncall_1", PARTITIONEDCALL, 1, 1);
  root_builder.AddDataEdge(partitioncall_0, 0, partitioncall_1, 0);
  const auto &root_graph = root_builder.GetGraph();

  // 1.build partitioncall_0 sub graph
  auto p1_sub_builder = ut::GraphBuilder("partitioncall_0_sub");
  const auto &partitioncall_2 = p1_sub_builder.AddNode("partitioncall_2", PARTITIONEDCALL, 0, 1);
  const auto &partitioncall_0_netoutput = p1_sub_builder.AddNode("partitioncall_0_netoutput", NETOUTPUT, 1, 1);
  AttrUtils::SetInt(partitioncall_0_netoutput->GetOpDesc()->MutableInputDesc(0), "_parent_node_index", 0);
  p1_sub_builder.AddDataEdge(partitioncall_2, 0, partitioncall_0_netoutput, 0);
  const auto &sub_graph = p1_sub_builder.GetGraph();
  sub_graph->SetParentNode(partitioncall_0);
  sub_graph->SetParentGraph(root_graph);
  partitioncall_0->GetOpDesc()->AddSubgraphName("sub0");
  partitioncall_0->GetOpDesc()->SetSubgraphInstanceName(0, "partitioncall_0_sub");

  // 2.build partitioncall_1 sub graph
  auto p2_sub_builder = ut::GraphBuilder("partitioncall_1_sub");
  const auto &partitioncall_1_data = p2_sub_builder.AddNode("partitioncall_1_data", DATA, 0, 1);
  AttrUtils::SetInt(partitioncall_1_data->GetOpDesc(), "_parent_node_index", 0);
  const auto &partitioncall_1_squeeze = p2_sub_builder.AddNode("partitioncall_1_squeeze", SQUEEZE, 1, 1);
  const auto &partitioncall_1_netoutput = p2_sub_builder.AddNode("partitioncall_1_netoutput", NETOUTPUT, 1, 1);
  p2_sub_builder.AddDataEdge(partitioncall_1_data, 0, partitioncall_1_squeeze, 0);
  p2_sub_builder.AddDataEdge(partitioncall_1_squeeze, 0, partitioncall_1_netoutput, 0);
  const auto &sub_graph2 = p2_sub_builder.GetGraph();
  sub_graph2->SetParentNode(partitioncall_1);
  sub_graph2->SetParentGraph(root_graph);
  partitioncall_1->GetOpDesc()->AddSubgraphName("sub1");
  partitioncall_1->GetOpDesc()->SetSubgraphInstanceName(0, "partitioncall_1_sub");

  // 3 build partitioncall_2 sub graph
  auto p3_sub_builder = ut::GraphBuilder("partitioncall_2_sub");
  const auto &partitioncall_2_mul = p3_sub_builder.AddNode("partitioncall_2_mul", "Mul", 0, 1);
  const auto &partitioncall_2_netoutput = p3_sub_builder.AddNode("partitioncall_2_netoutput", NETOUTPUT, 1, 1);
  AttrUtils::SetInt(partitioncall_2_netoutput->GetOpDesc()->MutableInputDesc(0), "_parent_node_index", 0);
  p3_sub_builder.AddDataEdge(partitioncall_2_mul, 0, partitioncall_2_netoutput, 0);
  const auto &sub_graph3 = p3_sub_builder.GetGraph();
  sub_graph3->SetParentNode(partitioncall_2);
  sub_graph3->SetParentGraph(sub_graph);
  partitioncall_2->GetOpDesc()->AddSubgraphName("sub2");
  partitioncall_2->GetOpDesc()->SetSubgraphInstanceName(0, "partitioncall_2_sub");

  root_graph->AddSubgraph(sub_graph->GetName(), sub_graph);
  root_graph->AddSubgraph(sub_graph2->GetName(), sub_graph2);
  root_graph->AddSubgraph(sub_graph3->GetName(), sub_graph3);
  return root_graph;
}

/*
*                                       |          constant         |
*    data  partitioncall_0--------------|             |           |
*       \  /                            |          netoutput      |
*      concat
*/
ComputeGraphPtr BuildGraphPartitionCall4() {
  auto root_builder = ut::GraphBuilder("root");
  const auto &data = root_builder.AddNode("data", DATA, 1, 1);
  const auto &partitioncall_0 = root_builder.AddNode("partitioncall_0", PARTITIONEDCALL, 3, 3);
  const auto &concat = root_builder.AddNode("concat", "Concat", 2, 1);
  root_builder.AddDataEdge(data, 0, concat, 0);
  root_builder.AddDataEdge(partitioncall_0, 0, concat, 1);
  const auto &root_graph = root_builder.GetGraph();

  // 1.build partitioncall_0 sub graph
  auto p1_sub_builder = ut::GraphBuilder("partitioncall_0_sub");
  const auto &partitioncall_0_const = p1_sub_builder.AddNode("partitioncall_0_constant", CONSTANTOP, 0, 1);
  const auto &partitioncall_0_netoutput = p1_sub_builder.AddNode("partitioncall_0_netoutput", NETOUTPUT, 1, 1);
  AttrUtils::SetInt(partitioncall_0_netoutput->GetOpDesc()->MutableInputDesc(0), "_parent_node_index", 0);
  p1_sub_builder.AddDataEdge(partitioncall_0_const, 0, partitioncall_0_netoutput, 0);
  const auto &sub_graph = p1_sub_builder.GetGraph();
  sub_graph->SetParentNode(partitioncall_0);
  sub_graph->SetParentGraph(root_graph);
  partitioncall_0->GetOpDesc()->AddSubgraphName("f");
  partitioncall_0->GetOpDesc()->SetSubgraphInstanceName(0, "partitioncall_0_sub");
  root_graph->AddSubgraph(sub_graph->GetName(), sub_graph);
  return root_graph;
}

/**
 * var---->identity--->cast--->netoutput
 *    \       ||
 *       \    ||
 *         \  \/
 *  const->assgin
 */
ComputeGraphPtr BuildGraphWithUsefulIdentity() {
  auto builder = ut::GraphBuilder("test");
  // id1 is useful
  auto id1 = builder.AddNode("id1", IDENTITY, 1, 1);
  auto var0 = builder.AddNode("var0", VARIABLE, 1, 1);
  auto const0 = builder.AddNode("const0", CONSTANT, 1, 1);
  auto cast = builder.AddNode("cast", "CAST", 1, 1);
  auto ref_node = builder.AddNode("ref_node", ASSIGN, 2, 1);
  ref_node->GetOpDesc()->UpdateInputName({{"ref", 0}, {"value", 1}});
  ref_node->GetOpDesc()->UpdateOutputName({{"ref", 0}});
  auto netoutput_node = builder.AddNode("netoutput", NETOUTPUT, 1, 1);

  builder.AddDataEdge(var0, 0, id1, 0);
  builder.AddDataEdge(id1, 0, cast, 0);
  builder.AddDataEdge(cast, 0, netoutput_node, 0);
  builder.AddDataEdge(var0, 0, ref_node, 0);
  builder.AddDataEdge(const0, 0, ref_node, 1);
  builder.AddControlEdge(id1, ref_node);
  return builder.GetGraph();
}
}

TEST_F(UtestNodeUtils, UpdateOriginShapeAndShape) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data1 = builder.AddNode("Data1", "Data", 1, 1);
  auto data2 = builder.AddNode("Data2", "Data", 1, 1);

  vector<int64_t> dims = {1, 2};
  GeShape data_shape(dims);
  ASSERT_EQ(NodeUtils::UpdateInputOriginalShapeAndShape(*data1, 0, data_shape), GRAPH_SUCCESS);
  ASSERT_EQ(NodeUtils::UpdateOutputOriginalShapeAndShape(*data1, 0, data_shape), GRAPH_SUCCESS);
  ASSERT_EQ(NodeUtils::UpdateInputOriginalShapeAndShape(*data2, 0, data_shape), GRAPH_SUCCESS);
  ASSERT_EQ(NodeUtils::UpdateOutputOriginalShapeAndShape(*data2, 0, data_shape), GRAPH_SUCCESS);
  ASSERT_EQ(data1->GetOpDesc()->GetInputDesc(0).GetShape() == data1->GetOpDesc()->GetInputDesc(0).GetShape(), true);
  ASSERT_EQ(data1->GetOpDesc()->GetInputDesc(0).IsOriginShapeInitialized(), true);
}

TEST_F(UtestNodeUtils, GetSubgraphs) {
  auto root_builder = ut::GraphBuilder("root");
  const auto &case0 = root_builder.AddNode("case0", "Case", 0, 0);
  const auto &root_graph = root_builder.GetGraph();

  auto sub_builder1 = ut::GraphBuilder("sub1");
  const auto &case1 = sub_builder1.AddNode("case1", "Case", 0, 0);
  const auto &sub_graph1 = sub_builder1.GetGraph();
  root_graph->AddSubGraph(sub_graph1);
  sub_graph1->SetParentNode(case0);
  sub_graph1->SetParentGraph(root_graph);
  case0->GetOpDesc()->AddSubgraphName("branch1");
  case0->GetOpDesc()->SetSubgraphInstanceName(0, "sub1");

  std::vector<ComputeGraphPtr> subgraphs0;
  ASSERT_EQ(NodeUtils::GetDirectSubgraphs(case0, subgraphs0), GRAPH_SUCCESS);
  ASSERT_EQ(subgraphs0.size(), 1);
  std::vector<ComputeGraphPtr> subgraphs1;
  ASSERT_EQ(NodeUtils::GetDirectSubgraphs(case1, subgraphs1), GRAPH_SUCCESS);
  ASSERT_TRUE(subgraphs1.empty());
}

TEST_F(UtestNodeUtils, GetSubgraphs_nullptr_node) {
  std::vector<ComputeGraphPtr> subgraphs;
  ASSERT_NE(NodeUtils::GetDirectSubgraphs(nullptr, subgraphs), GRAPH_SUCCESS);
  ASSERT_TRUE(subgraphs.empty());
}

TEST_F(UtestNodeUtils, GetSubgraphs_nullptr_root_graph) {
  auto builder = ut::GraphBuilder("graph");
  const auto &node = builder.AddNode("node", "node", 0, 0);
  node->impl_->owner_graph_.reset();

  std::vector<ComputeGraphPtr> subgraphs;
  ASSERT_NE(NodeUtils::GetDirectSubgraphs(node, subgraphs), GRAPH_SUCCESS);
  ASSERT_TRUE(subgraphs.empty());
}

TEST_F(UtestNodeUtils, GetSubgraphs_nullptr_sub_graph) {
  auto root_builder = ut::GraphBuilder("root");
  const auto &node = root_builder.AddNode("node", "node", 0, 0);
  const auto &root_graph = root_builder.GetGraph();

  auto sub_builder = ut::GraphBuilder("sub");
  const auto &sub_graph = sub_builder.GetGraph();
  sub_graph->SetParentNode(node);
  sub_graph->SetParentGraph(root_graph);
  node->GetOpDesc()->AddSubgraphName("branch1");
  node->GetOpDesc()->SetSubgraphInstanceName(0, "sub");

  std::vector<ComputeGraphPtr> subgraphs;
  ASSERT_EQ(NodeUtils::GetDirectSubgraphs(node, subgraphs), GRAPH_SUCCESS);
  ASSERT_TRUE(subgraphs.empty());
}

TEST_F(UtestNodeUtils, GetNodeUnknownShapeStatus_success) {
  auto root_builder = ut::GraphBuilder("root");
  const auto &case0 = root_builder.AddNode("case0", "Case", 0, 0);
  const auto &root_graph = root_builder.GetGraph();
  auto sub_builder1 = ut::GraphBuilder("sub1");
  const auto &case1 = sub_builder1.AddNode("case1", "Case", 0, 0);
  const auto &sub_graph1 = sub_builder1.GetGraph();
  root_graph->AddSubGraph(sub_graph1);
  sub_graph1->SetParentNode(case0);
  sub_graph1->SetParentGraph(root_graph);
  case0->GetOpDesc()->AddSubgraphName("branch1");
  case0->GetOpDesc()->SetSubgraphInstanceName(0, "sub1");
  bool is_known = false;
  ASSERT_EQ(NodeUtils::GetNodeUnknownShapeStatus(*case0, is_known), GRAPH_SUCCESS);
}


TEST_F(UtestNodeUtils, GetInNodeCrossPartionedCallNode_cross_one_subgraph) {
  auto graph = BuildGraphPartitionCall4();
  NodePtr expect_peer_node;
  NodePtr concat_node;
  for (auto &node : graph->GetAllNodes()) {
    if (node->GetType() == "Concat") {
      concat_node = node;
    }
  }
  auto ret = NodeUtils::GetInNodeCrossPartionedCallNode(concat_node, 1 , expect_peer_node);
  ASSERT_EQ(ret, GRAPH_SUCCESS);
  ASSERT_NE(expect_peer_node, nullptr);
  ASSERT_EQ(expect_peer_node->GetName(), "partitioncall_0_constant");
}

TEST_F(UtestNodeUtils, GetInNodeCrossPartionedCallNode_subgraph_in_partitioncall) {
  auto graph = BuildGraphPartitionCall();
  NodePtr expect_peer_node;
  NodePtr squeeze_node;
  for (auto &node : graph->GetAllNodes()) {
    if (node->GetType() == SQUEEZE) {
      squeeze_node = node;
    }
  }
  auto ret = NodeUtils::GetInNodeCrossPartionedCallNode(squeeze_node, 0 , expect_peer_node);
  ASSERT_EQ(ret, GRAPH_SUCCESS);
  ASSERT_NE(expect_peer_node, nullptr);
  ASSERT_EQ(expect_peer_node->GetName(), "partitioncall_0_const1");
}

  ///       A(PartionedCall_0)->B(PartionedCall_1)
  ///          PartionedCall_0's subgraph: Data->A->Netoutput
  ///          PartionedCall_1's subgraph: Data1->B->Netoutput
  ///          If it is called like GetInNodeCrossPartionCallNode(B,0,peer_node)or(Data1,0,peer_node), peer_node is A
TEST_F(UtestNodeUtils, GetInNodeCrossPartionedCallNode_paritioncall_link_partitioncall) {
  auto graph = BuildGraphPartitionCall2();
  NodePtr expect_peer_node = nullptr;
  NodePtr squeeze_node;
  NodePtr data_in_root;
  NodePtr data_in_partition0;
  NodePtr data_in_partition1;
  NodePtr partitioncall_0;
  for (auto &node : graph->GetAllNodes()) {
    if (node->GetType() == SQUEEZE) {
      squeeze_node = node;
    }
    if (node->GetName() == "data") {
      data_in_root = node;
    }
    if (node->GetName() == "partitioncall_0_data") {
      data_in_partition0 = node;
    }
    if (node->GetName() == "partitioncall_1_data") {
      data_in_partition1 = node;
    }
    if (node->GetName() == "partitioncall_0") {
      partitioncall_0 = node;
    }
  }
  ASSERT_EQ(NodeUtils::GetInNodeCrossSubgraph(data_in_partition0), data_in_root);
  ASSERT_EQ(NodeUtils::GetInNodeCrossSubgraph(data_in_partition1), partitioncall_0);

  // test with src node
  auto ret = NodeUtils::GetInNodeCrossPartionedCallNode(squeeze_node, 0 , expect_peer_node);
  ASSERT_EQ(ret, GRAPH_SUCCESS);
  ASSERT_NE(expect_peer_node, nullptr);
  ASSERT_EQ(expect_peer_node->GetName(), "partitioncall_0_cast");

  // test subgraph_data node
  ret = NodeUtils::GetInNodeCrossPartionedCallNode(data_in_partition1, 0 , expect_peer_node);
  ASSERT_EQ(ret, GRAPH_SUCCESS);
  ASSERT_NE(expect_peer_node, nullptr);
  ASSERT_EQ(expect_peer_node->GetName(), "partitioncall_0_cast");

  // test peer_node is root_data node
  ret = NodeUtils::GetInNodeCrossPartionedCallNode(partitioncall_0, 1 , expect_peer_node);
  ASSERT_EQ(ret, GRAPH_SUCCESS);
  ASSERT_EQ(expect_peer_node, nullptr);
}

TEST_F(UtestNodeUtils, GetInNodeCrossPartionedCallNode_multi_partitioncall) {
  auto graph = BuildGraphPartitionCall3();
  NodePtr expect_peer_node;
  NodePtr squeeze_node;
  for (auto &node : graph->GetAllNodes()) {
    if (node->GetType() == SQUEEZE) {
      squeeze_node = node;
    }
  }
  auto ret = NodeUtils::GetInNodeCrossPartionedCallNode(squeeze_node, 0 , expect_peer_node);
  ASSERT_EQ(ret, GRAPH_SUCCESS);
  ASSERT_NE(expect_peer_node, nullptr);
  ASSERT_EQ(expect_peer_node->GetName(), "partitioncall_2_mul");
}

TEST_F(UtestNodeUtils, GetInNodeCrossPartionedCallNode_temp_test_return_success_when_peer_node_null) {
  auto graph = BuildGraphPartitionCall3();
  NodePtr expect_peer_node;
  NodePtr partition_node;
  for (auto &node : graph->GetAllNodes()) {
    if (node->GetName() == "partitioncall_0") {
      partition_node = node;
    }
  }
  auto ret = NodeUtils::GetInNodeCrossPartionedCallNode(partition_node, 0 , expect_peer_node);
  ASSERT_EQ(ret, GRAPH_SUCCESS);
  ASSERT_EQ(expect_peer_node, nullptr);
}

TEST_F(UtestNodeUtils, GetConstOpType_CONST) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data = builder.AddNode("const1", CONSTANT, 0, 1);
  std::cout << data->GetType() << std::endl;
  std::string op_type;
  auto ret = NodeUtils::GetConstOpType(data, op_type);
  ASSERT_EQ(ret, true);
  ASSERT_EQ(op_type, "Const");
}

TEST_F(UtestNodeUtils, GetConstOpType_DATA) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data = builder.AddNode("Data", "Data", 0, 1);
  std::cout << data->GetType() << std::endl;
  std::string op_type;
  auto ret = NodeUtils::GetConstOpType(data, op_type);
  ASSERT_EQ(ret, false);
}

TEST_F(UtestNodeUtils, GetNodeUnknownShapeStatus) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data = builder.AddNode("add", "Add", 2, 1, FORMAT_NHWC, DT_FLOAT, {16, 228, 228, 3});
  auto graph = builder.GetGraph();

  auto add_node = graph->FindNode("add");
  ASSERT_NE(add_node, nullptr);
  bool is_unknown = false;
  auto ret = NodeUtils::GetNodeUnknownShapeStatus(*add_node, is_unknown);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_FALSE(is_unknown);

  ASSERT_NE(add_node->GetOpDesc(), nullptr);
  auto out_desc = add_node->GetOpDesc()->MutableOutputDesc(0);
  ASSERT_NE(out_desc, nullptr);
  out_desc->SetShape(GeShape({-1, 228, 228, 3}));
  is_unknown = false;
  (void)NodeUtils::GetNodeUnknownShapeStatus(*add_node, is_unknown);
  EXPECT_EQ(is_unknown, true);
  out_desc->SetShape(GeShape({-2}));
  is_unknown = false;
  (void)NodeUtils::GetNodeUnknownShapeStatus(*add_node, is_unknown);
  EXPECT_EQ(is_unknown, true);

  auto in_desc = add_node->GetOpDesc()->MutableInputDesc(0);
  ASSERT_NE(in_desc, nullptr);
  in_desc->SetShape(GeShape({-1, 228, 228, 3}));
  is_unknown = false;
  (void)NodeUtils::GetNodeUnknownShapeStatus(*add_node, is_unknown);
  EXPECT_EQ(is_unknown, true);
  in_desc->SetShape(GeShape({-2}));
  is_unknown = false;
  (void)NodeUtils::GetNodeUnknownShapeStatus(*add_node, is_unknown);
  EXPECT_EQ(is_unknown, true);
}

TEST_F(UtestNodeUtils, ClearInDataAnchor) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data = builder.AddNode("Data", "Data", 1, 1);
  InDataAnchorPtr in_anch = std::make_shared<InDataAnchor>(data, 111);
  EXPECT_EQ(NodeUtils::ClearInDataAnchor(data, in_anch), GRAPH_FAILED);

  auto const1 = builder.AddNode("const1", "Const", 1, 1);
  auto const2 = builder.AddNode("const2", "Const", 1, 1);
  InDataAnchorPtr in_anch1 = std::make_shared<InDataAnchor>(const1, 111);
  InDataAnchorPtr in_anch2 = std::make_shared<InDataAnchor>(const2, 111);
  OutDataAnchorPtr out_anch = std::make_shared<OutDataAnchor>(const2, 222);
  EXPECT_EQ(const1->AddLinkFrom(const2), GRAPH_SUCCESS);
  EXPECT_EQ(const2->GetOutDataNodes().size(), 1);
  EXPECT_EQ(const1->impl_->in_data_anchors_.size(), 2);
  auto anch = const1->impl_->in_data_anchors_.at(0);
  EXPECT_EQ(NodeUtils::ClearInDataAnchor(const1, in_anch2), GRAPH_FAILED);
  EXPECT_EQ(NodeUtils::ClearInDataAnchor(const1, anch), GRAPH_SUCCESS);
}

TEST_F(UtestNodeUtils, SetStatus) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data = builder.AddNode("Data", "Data", 0, 1);
  EXPECT_EQ(NodeUtils::SetAllAnchorStatus(data), GRAPH_SUCCESS);
  EXPECT_EQ(NodeUtils::SetAllAnchorStatus(*data), GRAPH_SUCCESS);
  EXPECT_EQ(NodeUtils::IsAnchorStatusSet(data), true);
  EXPECT_EQ(NodeUtils::IsAnchorStatusSet(nullptr), false);
  EXPECT_EQ(NodeUtils::IsAnchorStatusSet(*data), true);
  data->impl_ = nullptr;
  EXPECT_EQ(NodeUtils::SetAllAnchorStatus(*data), GRAPH_FAILED);
  EXPECT_EQ(NodeUtils::IsAnchorStatusSet(*data), false);
}

TEST_F(UtestNodeUtils, MoveOutputEdges) {
  EXPECT_EQ(NodeUtils::MoveOutputEdges(nullptr, nullptr), GRAPH_FAILED);
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data = builder.AddNode("Data", "Data", 0, 1);
  auto dest = builder.AddNode("Dest", "Dest", 11, 22);
  auto attr = builder.AddNode("Attr", "Attr", 1, 1);
  auto node2 = builder.AddNode("Data2", "Data2", 2, 1);
  InDataAnchorPtr peer = std::make_shared<InDataAnchor>(node2, 22);
  EXPECT_EQ(NodeUtils::MoveOutputEdges(data, dest), GRAPH_FAILED);
  EXPECT_EQ(dest->GetAllOutDataAnchors().size(), 22);
  EXPECT_EQ(NodeUtils::MoveOutputEdges(data, attr), GRAPH_SUCCESS);

  auto const1 = builder.AddNode("const1", "Const", 1, 1);
  auto const2 = builder.AddNode("const2", "Const", 1, 1);
  EXPECT_EQ(const1->AddLinkFrom(const2), GRAPH_SUCCESS);
  EXPECT_EQ(NodeUtils::MoveOutputEdges(const2, const1), GRAPH_SUCCESS);
}

TEST_F(UtestNodeUtils, MoveOutputEdges_Link) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data0 = builder.AddNode("Data0", DATA, 1, 1);
  auto data1 = builder.AddNode("Data1", DATA, 1, 1);
  auto data2 = builder.AddNode("Data2", DATA, 1, 1);
  auto data3 = builder.AddNode("Data3", DATA, 1, 1);
  auto data4 = builder.AddNode("Data4", DATA, 1, 1);
  EXPECT_EQ(data0->GetAllOutDataAnchors().at(0)->LinkTo(data2->GetInControlAnchor()), GRAPH_SUCCESS);
  EXPECT_EQ(data0->GetOutControlAnchor()->LinkTo(data3->GetInControlAnchor()), GRAPH_SUCCESS);
  EXPECT_EQ(data0->GetOutControlAnchor()->LinkTo(data4->GetInDataAnchor(0)), GRAPH_SUCCESS);
  EXPECT_EQ(NodeUtils::MoveOutputEdges(data0, data1), GRAPH_SUCCESS);
}

TEST_F(UtestNodeUtils, UpdateIsInputConst_Normal) {
  EXPECT_NO_THROW(
    NodeUtils::UpdateIsInputConst(nullptr);
    ut::GraphBuilder builder = ut::GraphBuilder("graph");
    auto data = builder.AddNode("Data", "Data", 0, 1);
    NodeUtils::UpdateIsInputConst(data);
    NodeUtils::UpdateIsInputConst(*data);
    data->impl_->op_ = nullptr;
    NodeUtils::UpdateIsInputConst(data);
  );
}

TEST_F(UtestNodeUtils, UpdateIsInputConst_Nullptr) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data1 = builder.AddNode("Data1", DATA, 2, 2);
  auto data2 = builder.AddNode("Data2", DATA, 1, 1);
  data1->impl_->in_data_anchors_.at(1) = nullptr;
  EXPECT_EQ(data1->GetInDataAnchor(0)->LinkFrom(data2->GetOutDataAnchor(0)), GRAPH_SUCCESS);
  auto node = data2->GetOutDataAnchor(0)->GetOwnerNode();
  NodeUtils::UpdateIsInputConst(data1);
}


TEST_F(UtestNodeUtils, UpdateIsInputConst_OutDataAnchorNullptr) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data_node = builder.AddNode("Data", "Data", 1, 1);
  auto attr_node = builder.AddNode("Attr", "Attr", 2, 2);
  InDataAnchorPtr in_anch = std::make_shared<InDataAnchor>(data_node, 111);
  OutDataAnchorPtr out_anch = std::make_shared<OutDataAnchor>(data_node, 222);
  EXPECT_EQ(out_anch->LinkTo(in_anch), GRAPH_SUCCESS);
  EXPECT_EQ(attr_node->AddLinkFrom(data_node), GRAPH_SUCCESS);
  EXPECT_EQ(attr_node->GetAllInDataAnchors().size(), 3);
  NodeUtils::UpdateIsInputConst(attr_node);
}


TEST_F(UtestNodeUtils, UnlinkAll) {
  EXPECT_NO_THROW(
    ut::GraphBuilder builder = ut::GraphBuilder("graph");
    auto data = builder.AddNode("Data", "Data", 0, 1);
    NodeUtils::UnlinkAll(*data);
  );
}

TEST_F(UtestNodeUtils, AppendRemoveAnchor) {
  EXPECT_EQ(NodeUtils::AppendInputAnchor(nullptr, 0), GRAPH_FAILED);
  EXPECT_EQ(NodeUtils::RemoveInputAnchor(nullptr, 0), GRAPH_FAILED);
  EXPECT_EQ(NodeUtils::AppendOutputAnchor(nullptr, 0), GRAPH_FAILED);
  EXPECT_EQ(NodeUtils::RemoveOutputAnchor(nullptr, 0), GRAPH_FAILED);
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data = builder.AddNode("Data", "Data", 0, 1);
  EXPECT_EQ(NodeUtils::AppendInputAnchor(data, 11), GRAPH_SUCCESS);
  EXPECT_EQ(NodeUtils::RemoveInputAnchor(data, 11), GRAPH_SUCCESS);
  EXPECT_EQ(NodeUtils::AppendOutputAnchor(data, 22), GRAPH_SUCCESS);
  EXPECT_EQ(NodeUtils::RemoveOutputAnchor(data, 22), GRAPH_SUCCESS);
}

TEST_F(UtestNodeUtils, RemoveInputAnchor) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data = builder.AddNode("Data", "Data", 1, 1);
  EXPECT_EQ(data->GetOpDesc()->GetInputsSize(), 1);
  EXPECT_EQ(data->GetOpDesc()->AddInputDesc(GeTensorDesc()), GRAPH_SUCCESS);
  EXPECT_EQ(data->GetOpDesc()->GetInputsSize(), 2);
  EXPECT_EQ(NodeUtils::RemoveInputAnchor(data, 0), GRAPH_SUCCESS);
}

TEST_F(UtestNodeUtils, RemoveOutputAnchor) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data = builder.AddNode("Data", "Data", 1, 1);
  EXPECT_EQ(data->GetOpDesc()->GetOutputsSize(), 1);
  EXPECT_EQ(NodeUtils::RemoveOutputAnchor(data, 0), GRAPH_SUCCESS);
}

TEST_F(UtestNodeUtils, SetSubgraph) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data = builder.AddNode("Data", "Data", 0, 1);
  auto gragh = builder.GetGraph();
  EXPECT_EQ(NodeUtils::SetSubgraph(*data, 0, nullptr), GRAPH_PARAM_INVALID);
  EXPECT_EQ(NodeUtils::SetSubgraph(*data, 0, gragh), GRAPH_PARAM_INVALID);
  data->impl_->op_->AddSubgraphName("g1");
  data->impl_->op_->AddSubgraphName("g2");

  auto sub_graph1 = std::make_shared<ComputeGraph>("g1");
  auto sub_graph2 = std::make_shared<ComputeGraph>("g2");
  EXPECT_EQ(NodeUtils::SetSubgraph(*data, 0, sub_graph1), GRAPH_SUCCESS);
  EXPECT_EQ(NodeUtils::SetSubgraph(*data, 1, sub_graph2), GRAPH_SUCCESS);
}

TEST_F(UtestNodeUtils, IsSubgraphInput) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto node = builder.AddNode("Node", "Node", 11, 22);
  EXPECT_EQ(NodeUtils::IsSubgraphInput(node), false);
  auto root_builder = ut::GraphBuilder("root");
  const auto &partitioncall_0 = root_builder.AddNode("partitioncall_0", PARTITIONEDCALL, 0, 1);
  const auto &partitioncall_1 = root_builder.AddNode("partitioncall_1", PARTITIONEDCALL, 1, 1);
  root_builder.AddDataEdge(partitioncall_0, 0, partitioncall_1, 0);
  const auto &root_graph = root_builder.GetGraph();
  auto p1_sub_builder = ut::GraphBuilder("partitioncall_0_sub");
  const auto &partitioncall_0_const1 = p1_sub_builder.AddNode("partitioncall_0_const1", CONSTANT, 0, 1);
  const auto &partitioncall_0_netoutput = p1_sub_builder.AddNode("partitioncall_0_netoutput", NETOUTPUT, 1, 1);
  AttrUtils::SetInt(partitioncall_0_netoutput->GetOpDesc()->MutableInputDesc(0), "_parent_node_index", 0);
  p1_sub_builder.AddDataEdge(partitioncall_0_const1, 0, partitioncall_0_netoutput, 0);
  const auto &sub_graph = p1_sub_builder.GetGraph();
  sub_graph->SetParentNode(partitioncall_0);
  sub_graph->SetParentGraph(root_graph);
  partitioncall_0->GetOpDesc()->AddSubgraphName("f");
  partitioncall_0->GetOpDesc()->SetSubgraphInstanceName(0, "partitioncall_0_sub");
  EXPECT_EQ(NodeUtils::IsSubgraphInput(partitioncall_0_const1), false);
}

TEST_F(UtestNodeUtils, IsSubgraphInput_WithATTR_NAME_IS_UNKNOWN_SHAPE) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto root_builder = ut::GraphBuilder("root");
  const auto &while_node = root_builder.AddNode("while", WHILE, 1, 1);
  bool is_known = true;
  AttrUtils::SetBool(while_node->GetOpDesc(), ATTR_NAME_IS_UNKNOWN_SHAPE, is_known);
  const auto &root_graph = root_builder.GetGraph();

  auto p1_sub_builder = ut::GraphBuilder("sub");
  const auto &data_node = p1_sub_builder.AddNode("data", DATA, 0, 1);
  AttrUtils::SetInt(data_node->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);

  const auto &sub_graph = p1_sub_builder.GetGraph();
  sub_graph->SetParentNode(while_node);
  sub_graph->SetParentGraph(root_graph);
  while_node->GetOpDesc()->AddSubgraphName("sub");
  while_node->GetOpDesc()->SetSubgraphInstanceName(0, "sub");
  EXPECT_FALSE(NodeUtils::IsSubgraphInput(data_node));

  is_known = false;
  AttrUtils::SetBool(while_node->GetOpDesc(), ATTR_NAME_IS_UNKNOWN_SHAPE, is_known);
  EXPECT_TRUE(NodeUtils::IsSubgraphInput(data_node));
}

TEST_F(UtestNodeUtils, IsSubgraphOutput_WithATTR_NAME_IS_UNKNOWN_SHAPE) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto root_builder = ut::GraphBuilder("root");
  const auto &while_node = root_builder.AddNode("while", WHILE, 1, 1);
  bool is_known = true;
  AttrUtils::SetBool(while_node->GetOpDesc(), ATTR_NAME_IS_UNKNOWN_SHAPE, is_known);
  const auto &root_graph = root_builder.GetGraph();

  auto p1_sub_builder = ut::GraphBuilder("sub");
  const auto &partitioncall_0_const1 = p1_sub_builder.AddNode("partitioncall_0_const1", CONSTANT, 0, 1);
  const auto &partitioncall_0_netoutput = p1_sub_builder.AddNode("partitioncall_0_netoutput", NETOUTPUT, 1, 1);
  AttrUtils::SetInt(partitioncall_0_netoutput->GetOpDesc()->MutableInputDesc(0), "_parent_node_index", 0);
  p1_sub_builder.AddDataEdge(partitioncall_0_const1, 0, partitioncall_0_netoutput, 0);

  const auto &sub_graph = p1_sub_builder.GetGraph();
  sub_graph->SetParentNode(while_node);
  sub_graph->SetParentGraph(root_graph);
  while_node->GetOpDesc()->AddSubgraphName("sub");
  while_node->GetOpDesc()->SetSubgraphInstanceName(0, "sub");
  EXPECT_FALSE(NodeUtils::IsSubgraphOutput(partitioncall_0_netoutput));

  is_known = false;
  AttrUtils::SetBool(while_node->GetOpDesc(), ATTR_NAME_IS_UNKNOWN_SHAPE, is_known);
  EXPECT_TRUE(NodeUtils::IsSubgraphOutput(partitioncall_0_netoutput));
}

TEST_F(UtestNodeUtils, IsDynamicShape) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto node = builder.AddNode("Node", "Node", 11, 22);
  EXPECT_EQ(NodeUtils::IsDynamicShape(*node), false);
}

TEST_F(UtestNodeUtils, IsWhileVaryingInput) {
  auto root_builder = ut::GraphBuilder("root");
  const auto &partitioncall_0 = root_builder.AddNode("partitioncall_0", PARTITIONEDCALL, 0, 1);
  const auto &partitioncall_1 = root_builder.AddNode("partitioncall_1", PARTITIONEDCALL, 1, 1);
  root_builder.AddDataEdge(partitioncall_0, 0, partitioncall_1, 0);
  const auto &root_graph = root_builder.GetGraph();
  auto p1_sub_builder = ut::GraphBuilder("partitioncall_0_sub");
  const auto &partitioncall_0_const1 = p1_sub_builder.AddNode("partitioncall_0_const1", CONSTANT, 0, 1);
  const auto &partitioncall_0_netoutput = p1_sub_builder.AddNode("partitioncall_0_netoutput", NETOUTPUT, 1, 1);
  AttrUtils::SetInt(partitioncall_0_netoutput->GetOpDesc()->MutableInputDesc(0), "_parent_node_index", 0);
  p1_sub_builder.AddDataEdge(partitioncall_0_const1, 0, partitioncall_0_netoutput, 0);
  const auto &sub_graph = p1_sub_builder.GetGraph();
  sub_graph->SetParentNode(partitioncall_0);
  sub_graph->SetParentGraph(root_graph);
  partitioncall_0->GetOpDesc()->AddSubgraphName("f");
  partitioncall_0->GetOpDesc()->SetSubgraphInstanceName(0, "partitioncall_0_sub");
  auto sub2_builder = ut::GraphBuilder("partitioncall_0_sub2");
  const auto &data11 = root_builder.AddNode("data11", DATA, 0, 1);
  data11->SetOwnerComputeGraph(sub_graph);
  EXPECT_EQ(NodeUtils::IsWhileVaryingInput(data11), false);
}

TEST_F(UtestNodeUtils, IsWhileVaryingInput_While) {
  auto root_builder = ut::GraphBuilder("root");
  const auto &while1 = root_builder.AddNode("while1", "While", 1, 1);
  const auto &root_graph = root_builder.GetGraph();
  auto sub_builder = ut::GraphBuilder("sub");
  const auto &const1 = sub_builder.AddNode("const1", CONSTANT, 1, 1);
  const auto &netoutput = sub_builder.AddNode("netoutput", NETOUTPUT, 1, 1);
  const auto &data0 = sub_builder.AddNode("data0", DATA, 1, 1);
  const auto &sub_graph = sub_builder.GetGraph();
  sub_graph->SetParentNode(while1);
  sub_graph->SetParentGraph(root_graph);
  EXPECT_EQ(NodeUtils::IsWhileVaryingInput(data0), false);
  //EXPECT_EQ(NodeUtils::IsWhileVaryingInput(while1), false);
  EXPECT_EQ(AttrUtils::SetInt(data0->GetOpDesc(), "_parent_node_index", 1), true);
  EXPECT_EQ(NodeUtils::IsWhileVaryingInput(data0), true);
}

TEST_F(UtestNodeUtils, RemoveSubgraphsOnNode) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto node = builder.AddNode("Node", "Node", 11, 22);
  EXPECT_EQ(NodeUtils::RemoveSubgraphsOnNode(node), GRAPH_SUCCESS);
  node->impl_->op_->SetSubgraphInstanceName(0, "name");
  node->impl_->op_->SetSubgraphInstanceName(1, "name1");
  EXPECT_EQ(NodeUtils::RemoveSubgraphsOnNode(node), GRAPH_SUCCESS);
}

TEST_F(UtestNodeUtils, GetSubgraphDataNodesByIndex) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto node = builder.AddNode("Node", "Const", 1, 1);
  EXPECT_EQ(NodeUtils::GetSubgraphDataNodesByIndex(*node, 0).size(), 0);
}

TEST_F(UtestNodeUtils, GetSubgraphDataAndNetoutput) {
  auto root_builder = ut::GraphBuilder("root");
  const auto &data = root_builder.AddNode("data", DATA, 1, 1);
  const auto &partitioncall_0 = root_builder.AddNode("partitioncall_0", PARTITIONEDCALL, 3, 3);
  const auto &partitioncall_1 = root_builder.AddNode("partitioncall_1", PARTITIONEDCALL, 1, 1);
  root_builder.AddDataEdge(data, 0, partitioncall_0, 1);
  root_builder.AddDataEdge(partitioncall_0, 1, partitioncall_1, 0);
  const auto &root_graph = root_builder.GetGraph();

  int64_t index = 0;
  // 1.build partitioncall_0 sub graph
  auto p1_sub_builder = ut::GraphBuilder("partitioncall_0_sub");
  const auto &partitioncall_0_data = p1_sub_builder.AddNode("partitioncall_0_data", DATA, 0, 1);
  AttrUtils::SetInt(partitioncall_0_data->GetOpDesc(), "_parent_node_index", index);
  const auto &partitioncall_0_cast = p1_sub_builder.AddNode("partitioncall_0_cast", "Cast", 1, 1);
  const auto &partitioncall_0_netoutput = p1_sub_builder.AddNode("partitioncall_0_netoutput", NETOUTPUT, 3, 3);
  AttrUtils::SetInt(partitioncall_0_netoutput->GetOpDesc()->MutableInputDesc(0), "_parent_node_index", 0);
  AttrUtils::SetInt(partitioncall_0_netoutput->GetOpDesc()->MutableInputDesc(1), "_parent_node_index", 1);
  AttrUtils::SetInt(partitioncall_0_netoutput->GetOpDesc()->MutableInputDesc(2), "_parent_node_index", 2);
  p1_sub_builder.AddDataEdge(partitioncall_0_data, 0, partitioncall_0_cast, 0);
  p1_sub_builder.AddDataEdge(partitioncall_0_cast, 0, partitioncall_0_netoutput, 1);
  const auto &sub_graph = p1_sub_builder.GetGraph();
  sub_graph->SetParentNode(partitioncall_0);
  sub_graph->SetParentGraph(root_graph);
  root_graph->AddSubgraph("partitioncall_0_sub", sub_graph);
  partitioncall_0->GetOpDesc()->AddSubgraphName("f");
  partitioncall_0->GetOpDesc()->SetSubgraphInstanceName(0, "partitioncall_0_sub");
  auto compute_graph = partitioncall_0->GetOwnerComputeGraph();
  EXPECT_NE(compute_graph, nullptr);
  EXPECT_EQ(NodeUtils::GetSubgraphOutputNodes(*partitioncall_0).size(), 1);
  EXPECT_EQ(NodeUtils::GetSubgraphDataNodesByIndex(*partitioncall_0, index).size(), 1);
  EXPECT_NE(sub_graph->GetOrUpdateNetOutputNode(), nullptr);
  EXPECT_EQ(sub_graph->GetOrUpdateNetOutputNode()->GetType(), NETOUTPUT);
}

TEST_F(UtestNodeUtils, GetSubgraphOutputNodes) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto node = builder.AddNode("Node", "Node", 11, 22);
  auto op_desc = node->GetOpDesc();
  op_desc->impl_->subgraph_instance_names_.clear();
  auto subgraph_names = op_desc->GetSubgraphInstanceNames();
  EXPECT_EQ(NodeUtils::GetSubgraphOutputNodes(*node).size(), 0);
  auto compute_graph = node->GetOwnerComputeGraph();
  compute_graph->impl_->parent_graph_ = MakeNullptr<ComputeGraph>();
  EXPECT_EQ(NodeUtils::GetSubgraphOutputNodes(*node).size(), 0);
}

TEST_F(UtestNodeUtils, GetOutDataNodesWithAnchorByIndex) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto node = builder.AddNode("Node", "Node", 11, 22);
  EXPECT_EQ(NodeUtils::GetOutDataNodesWithAnchorByIndex(*node, 0).size(), 0);
  EXPECT_EQ(NodeUtils::GetOutDataNodesWithAnchorByIndex(*node, -1).size(), 0);
}

TEST_F(UtestNodeUtils, GetNodeFromOperator) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto node = builder.AddNode("Node", "Node", 11, 22);
  Operator op = OperatorFactoryImpl::CreateOperator("opname", "optp");
  EXPECT_EQ(NodeUtilsEx::GetNodeFromOperator(op), nullptr);
}

TEST_F(UtestNodeUtils, GetInConstNodeTypeCrossSubgraph) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto node = builder.AddNode("Node", "Node", 11, 22);
  EXPECT_EQ(NodeUtils::GetInConstNodeTypeCrossSubgraph(node), "Node");
  EXPECT_EQ(NodeUtils::GetInConstNodeTypeCrossSubgraph(nullptr), "");
}

TEST_F(UtestNodeUtils, CreatNodeWithoutGraph) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  OpDescPtr od = std::make_shared<OpDesc>("name", "type");
  EXPECT_NE(NodeUtils::CreatNodeWithoutGraph(od), nullptr);
  EXPECT_EQ(NodeUtils::CreatNodeWithoutGraph(nullptr), nullptr);
}

TEST_F(UtestNodeUtils, SetNodeParallelGroup) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto node = builder.AddNode("Node", "Node", 11, 22);
  EXPECT_EQ(NodeUtils::SetNodeParallelGroup(*node, nullptr), GRAPH_FAILED);
  EXPECT_NE(NodeUtils::SetNodeParallelGroup(*node, "node_group"), GRAPH_FAILED);
  auto amap = node->GetOpDesc()->GetAttrMap();
  amap.SetByName<std::string>("_parallel_group", "_parallel_group_value");
  node->impl_->op_->impl_->attrs_ = amap;
  EXPECT_EQ(NodeUtils::SetNodeParallelGroup(*node, "node_group"), GRAPH_FAILED);
  EXPECT_EQ(NodeUtils::SetNodeParallelGroup(*node, "_parallel_group_value"), GRAPH_SUCCESS);

}

TEST_F(UtestNodeUtils, GetSubgraph) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto node = builder.AddNode("Node", "Node", 11, 22);
  EXPECT_EQ(NodeUtils::GetSubgraph(*node, 0), nullptr);
  node->impl_->op_ = nullptr;
  EXPECT_EQ(NodeUtils::GetSubgraph(*node, 0), nullptr);
}

TEST_F(UtestNodeUtils, GetNodeType) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto node = builder.AddNode("Node", "Node", 11, 22);
  node->impl_->op_->impl_->meta_data_.type_ = "FrameworkOp";
  EXPECT_EQ(NodeUtils::GetNodeType(*node), "");
}

TEST_F(UtestNodeUtils, UpdateInOutputOriginalShapeAndShapeFailure) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto node = builder.AddNode("Node", "Node", 11, 22);
  auto desc = node->GetOpDesc();
  EXPECT_EQ(NodeUtils::UpdateInputOriginalShapeAndShape(*node, 100, GeShape()), GRAPH_PARAM_INVALID);
  EXPECT_EQ(NodeUtils::UpdateOutputOriginalShapeAndShape(*node, 100, GeShape()), GRAPH_PARAM_INVALID);
  node->impl_->op_ = nullptr;
  EXPECT_EQ(NodeUtils::UpdateInputOriginalShapeAndShape(*node, 100, GeShape()), GRAPH_PARAM_INVALID);
  EXPECT_EQ(NodeUtils::UpdateOutputOriginalShapeAndShape(*node, 100, GeShape()), GRAPH_PARAM_INVALID);
}

TEST_F(UtestNodeUtils, GetInNodeCrossPartionedCallNode){
  auto graph = BuildGraphPartitionCall();
  NodePtr expect_peer_node;
  NodePtr sq_node;
  NodePtr nt_node;

  for (auto &node : graph->GetAllNodes()) {
    if (node->GetType() == SQUEEZE) {
      sq_node = node;
    }
    if (node->GetType() == NETOUTPUT) {
      nt_node = node;
    }
  }
  EXPECT_NE(sq_node, nullptr);
  EXPECT_NE(sq_node, nullptr);
  EXPECT_NE(sq_node->GetType(), DATA);
  EXPECT_EQ(sq_node->GetType(), SQUEEZE);
  EXPECT_NE(sq_node->GetType(), PARTITIONEDCALL);
  EXPECT_EQ(sq_node->GetOpDesc()->GetSubgraphInstanceNames().empty(), true);
  EXPECT_EQ(NodeUtils::GetInNodeCrossPartionedCallNode(sq_node, 0 , expect_peer_node), GRAPH_SUCCESS);
  EXPECT_EQ(NodeUtils::GetInNodeCrossPartionedCallNode(sq_node, 1000 , expect_peer_node), GRAPH_FAILED);
  auto peer = NodeUtils::GetInDataNodeByIndex(*sq_node, 0);
  EXPECT_NE(NodeUtils::GetInDataNodeByIndex(*sq_node, 0), nullptr);
  EXPECT_NE(nt_node->GetType(), DATA);
  EXPECT_EQ(nt_node->GetType(), NETOUTPUT);
  EXPECT_NE(nt_node->GetType(), PARTITIONEDCALL);
  EXPECT_EQ(nt_node->GetOpDesc()->GetSubgraphInstanceNames().empty(), true);
  EXPECT_EQ(NodeUtils::GetInNodeCrossPartionedCallNode(nt_node, 0 , expect_peer_node), GRAPH_SUCCESS);
}


TEST_F(UtestNodeUtils, GetInNodeCrossSubgraph){
  auto graph = BuildGraphPartitionCall();
  NodePtr expect_peer_node;
  NodePtr dt_node;
  for (auto &node : graph->GetAllNodes()) {
    if (node->GetType() == DATA) {
      dt_node = node;
    }
  }
  EXPECT_NE(dt_node, nullptr);
  EXPECT_NE(NodeUtils::GetInNodeCrossSubgraph(dt_node), nullptr);
  auto owner_graph = dt_node->GetOwnerComputeGraph();
  owner_graph->impl_->parent_node_ = MakeNullptr<Node>();
  EXPECT_NE(NodeUtils::GetInNodeCrossSubgraph(dt_node), nullptr);
}

TEST_F(UtestNodeUtils, GetConstOpType) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data = builder.AddNode("nt", NETOUTPUT, 0, 1);
  std::string op_type;
  EXPECT_EQ(NodeUtils::GetConstOpType(data, op_type), false);
}

TEST_F(UtestNodeUtils, IsWhileVaryingInputFalse) {
  EXPECT_EQ(NodeUtils::IsWhileVaryingInput(nullptr), false);
}

TEST_F(UtestNodeUtils, IsSubgraphOutput) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto node = builder.AddNode("Node", "Node", 11, 22);
  EXPECT_EQ(NodeUtils::IsSubgraphOutput(node), false);
  auto root_builder = ut::GraphBuilder("root");
  const auto &partitioncall_0 = root_builder.AddNode("partitioncall_0", "_is_unknown_shape", 0, 1);
  const auto &partitioncall_1 = root_builder.AddNode("partitioncall_1", "_is_unknown_shape", 1, 1);
  root_builder.AddDataEdge(partitioncall_0, 0, partitioncall_1, 0);
  const auto &root_graph = root_builder.GetGraph();
  auto p1_sub_builder = ut::GraphBuilder("partitioncall_0_sub");
  const auto &partitioncall_0_const1 = p1_sub_builder.AddNode("partitioncall_0_const1", CONSTANT, 0, 1);
  const auto &partitioncall_0_netoutput = p1_sub_builder.AddNode("partitioncall_0_netoutput", NETOUTPUT, 1, 1);
  AttrUtils::SetInt(partitioncall_0_netoutput->GetOpDesc()->MutableInputDesc(0), "_parent_node_index", 0);
  p1_sub_builder.AddDataEdge(partitioncall_0_const1, 0, partitioncall_0_netoutput, 0);
  const auto &sub_graph = p1_sub_builder.GetGraph();
  sub_graph->SetParentNode(partitioncall_0);
  sub_graph->SetParentGraph(root_graph);
  partitioncall_0->GetOpDesc()->AddSubgraphName("f");
  partitioncall_0->GetOpDesc()->SetSubgraphInstanceName(0, "partitioncall_0_sub");
  EXPECT_EQ(NodeUtils::IsSubgraphOutput(partitioncall_0_const1), false);
}

TEST_F(UtestNodeUtils, IsDynamicShape_Null) {
  EXPECT_EQ(NodeUtils::IsDynamicShape(nullptr), false);
}

TEST_F(UtestNodeUtils, GetInDataNodeAndAnchorByIndex_InAnchorOutOfRange) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto node = builder.AddNode("Node", "Node", 1, 1);
  EXPECT_EQ(NodeUtils::GetInDataNodeAndAnchorByIndex(*node, 1).first, nullptr);
}
TEST_F(UtestNodeUtils, GetInDataNodeAndAnchorByIndex_NoPeerOutAnchor) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto node = builder.AddNode("Node", "Node", 1, 1);
  EXPECT_EQ(NodeUtils::GetInDataNodeAndAnchorByIndex(*node, 0).first, nullptr);
}
TEST_F(UtestNodeUtils, GetInDataNodeAndAnchorByIndex_Success) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto node1 = builder.AddNode("Node1", "Node1", 1, 1);
  auto node2 = builder.AddNode("Node2", "Node2", 1, 1);
  builder.AddDataEdge(node1, 0, node2, 0);
  EXPECT_EQ(NodeUtils::GetInDataNodeAndAnchorByIndex(*node2, 0).first, node1);
  EXPECT_EQ(NodeUtils::GetInDataNodeAndAnchorByIndex(*node2, 0).second, node1->GetOutDataAnchor(0));
}
TEST_F(UtestNodeUtils, IsDtResourceNode_Success) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto node1 = builder.AddNode("Node1", "Node1", 1, 1);
  auto in_desc1 = node1->GetOpDesc()->MutableInputDesc(0);
  in_desc1->SetDataType(DT_RESOURCE);
  EXPECT_EQ(NodeUtils::IsDtResourceNode(node1), true);
  auto node2 = builder.AddNode("Node2", "Node2", 1, 1);
  auto out_desc2 = node2->GetOpDesc()->MutableOutputDesc(0);
  out_desc2->SetDataType(DT_RESOURCE);
  EXPECT_EQ(NodeUtils::IsDtResourceNode(node2), true);
}
TEST_F(UtestNodeUtils, IsIdentityUsefulForRWControl) {
  ComputeGraphPtr graph = BuildGraphWithUsefulIdentity();
  auto node = graph->FindNode("id1");
  EXPECT_NE(node, nullptr);
  // id1 is useful, not remove
  EXPECT_EQ(NodeUtils::IsIdentityUsefulForRWControl(node), true);
}
TEST_F(UtestNodeUtils, FindRootGraph) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto node = builder.AddNode("Node", "Node", 1, 1);
  EXPECT_NE(node, nullptr);
  EXPECT_EQ(NodeUtils::FindRootGraph(*node), node->GetOwnerComputeGraph());
}
TEST_F(UtestNodeUtils, GetOutControlNodes) {
  auto builder = ut::GraphBuilder("test_graph0");
  const auto &src_node = builder.AddNode("src_node", DATA, 1, 1);
  const auto &ctrl_node = builder.AddNode("ctrl_node", CONSTANT, 0, 0);
  const auto &ctrl_node2 = builder.AddNode("ctrl_node2", CONSTANT, 0, 0);
  auto graph = builder.GetGraph();
  builder.AddControlEdge(src_node, ctrl_node);
  builder.AddControlEdge(src_node, ctrl_node2);
  EXPECT_EQ(NodeUtils::GetOutControlNodes(*src_node, nullptr).size(), 2U);
  NodeFilter node_filter = [&](const Node &node) { return node.GetName() == ctrl_node2->GetName(); };
  EXPECT_EQ(NodeUtils::GetOutControlNodes(*src_node, node_filter).size(), 1U);
  EXPECT_EQ(NodeUtils::GetOutControlNodes(*src_node, node_filter).front(), ctrl_node2);
}
TEST_F(UtestNodeUtils, GetInControlNodes) {
  auto builder = ut::GraphBuilder("test_graph0");
  const auto &ctrl_node = builder.AddNode("ctrl_node", CONSTANT, 0, 0);
  const auto &ctrl_node2 = builder.AddNode("ctrl_node2", CONSTANT, 0, 0);
  const auto &dst_node = builder.AddNode("dst_node", NETOUTPUT, 0, 0);

  auto graph = builder.GetGraph();
  builder.AddControlEdge(ctrl_node, dst_node);
  builder.AddControlEdge(ctrl_node2, dst_node);
  EXPECT_EQ(NodeUtils::GetInControlNodes(*dst_node, nullptr).size(), 2U);
  NodeFilter node_filter = [&](const Node &node) { return node.GetName() == ctrl_node2->GetName(); };
  EXPECT_EQ(NodeUtils::GetInControlNodes(*dst_node, node_filter).size(), 1U);
  EXPECT_EQ(NodeUtils::GetInControlNodes(*dst_node, node_filter).front(), ctrl_node2);
}

TEST_F(UtestNodeUtils, TryGetWeightByPlaceHolderNode_invalid) {
  auto node = std::make_shared<Node>();
  auto ge_tensor = std::make_shared<const GeTensor>();
  EXPECT_NE(NodeUtils::TryGetWeightByPlaceHolderNode(node, ge_tensor), GRAPH_SUCCESS);
}

TEST_F(UtestNodeUtils, TryGetWeightByDataNode_invalid) {
  auto node = std::make_shared<Node>();
  auto ge_tensor = std::make_shared<const GeTensor>();
  EXPECT_NE(NodeUtils::TryGetWeightByDataNode(node, ge_tensor), GRAPH_SUCCESS);
}

TEST_F(UtestNodeUtils, GetParentInput_invalid) {
  auto builder = ut::GraphBuilder("test_graph0");
  const auto &data_node = builder.AddNode("data", DATA, 0, 0);
  auto graph = builder.GetGraph();
  auto node = graph->FindNode("data");
  AttrUtils::SetInt(node->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX,1);
  EXPECT_EQ(NodeUtils::GetParentInput(node), nullptr);
}

TEST_F(UtestNodeUtils, TryGetWeightByPlaceHolderNode_fail) {
  auto builder = ut::GraphBuilder("test_graph0");
  const auto &pld = builder.AddNode("pld", PLACEHOLDER, 1, 1);
  ConstGeTensorPtr ge_tensor = nullptr;
  EXPECT_EQ(NodeUtils::TryGetWeightByPlaceHolderNode(pld, ge_tensor), GRAPH_SUCCESS);
  EXPECT_TRUE(ge_tensor == nullptr);
  const auto &parent_node = builder.AddNode("fake", "fake", 1, 1);
  EXPECT_TRUE(pld->GetOpDesc()->SetExtAttr("parentNode", parent_node));
  EXPECT_EQ(NodeUtils::TryGetWeightByPlaceHolderNode(pld, ge_tensor), GRAPH_SUCCESS);
  EXPECT_TRUE(ge_tensor == nullptr);
}

TEST_F(UtestNodeUtils, Verify_update_output_name_with_default_name) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data_node = builder.AddNode("Data", "FakeData", 0, 1);
  EXPECT_EQ(data_node->GetOpDesc()->GetAllOutputName().cbegin()->first, "__output0");
  EXPECT_EQ(NodeUtilsEx::Verify(data_node), GRAPH_SUCCESS);
  EXPECT_EQ(data_node->GetOpDesc()->GetAllOutputName().cbegin()->first, "y");
}
TEST_F(UtestNodeUtils, Verify_update_output_name_with_empty_output) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data_node = builder.AddNode("Data", "FakeData", 0, 1);
  std::map<std::string, uint32_t> output_name_idx;
  data_node->GetOpDesc()->UpdateOutputName(output_name_idx);
  int32_t event_level;
  int32_t old_level = dlog_getlevel(GE_MODULE_NAME, &event_level);
  dlog_setlevel(GE_MODULE_NAME, DLOG_DEBUG, event_level);
  EXPECT_EQ(NodeUtilsEx::Verify(data_node), GRAPH_SUCCESS);
  dlog_setlevel(GE_MODULE_NAME, old_level, event_level);
  EXPECT_EQ(data_node->GetOpDesc()->GetAllOutputName().cbegin()->first, "y");
}

TEST_F(UtestNodeUtils, Verify_no_need_update_output_name_already_has) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data_node = builder.AddNode("Data", "FakeData", 0, 1);
  data_node->GetOpDesc()->UpdateOutputName({{"y", 0}});
  EXPECT_EQ(NodeUtilsEx::Verify(data_node), GRAPH_SUCCESS);
  EXPECT_EQ(data_node->GetOpDesc()->GetAllOutputName().cbegin()->first, "y");
}

TEST_F(UtestNodeUtils, Verify_noneed_update_output_name) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data_node = builder.AddNode("Data", "FakeData", 1, 1);
  data_node->impl_->in_data_anchors_.at(0) = nullptr;
  data_node->GetOpDesc()->UpdateInputName({{"xx", 0}});
  data_node->GetOpDesc()->UpdateOutputName({{"yy", 0}});
  EXPECT_EQ(data_node->GetOpDesc()->GetAllOutputName().cbegin()->first, "yy");

  EXPECT_EQ(NodeUtilsEx::Verify(data_node), GRAPH_SUCCESS);

  EXPECT_EQ(data_node->GetOpDesc()->GetAllOutputName().cbegin()->first, "yy");
}
}  // namespace ge
