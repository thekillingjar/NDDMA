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

#define protected public
#define private public

#include "graph/gnode.h"
#include "graph/operator_reg.h"
#include "common/ge_common/ge_inner_error_codes.h"
#include "graph/normal_graph/node_impl.h"
#include "graph/utils/node_adapter.h"
#include "graph_builder_utils.h"
#include "node_utils.h"
#include "graph/attr_value.h"

#undef private
#undef protected

namespace ge {
class GNodeTest : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(GNodeTest, GetALLSubgraphs) {
  auto root_builder = ut::GraphBuilder("root");
  const auto &node = root_builder.AddNode("node", "node", 0, 0);
  const auto &root_graph = root_builder.GetGraph();

  auto sub_builder = ut::GraphBuilder("sub");
  const auto &sub_graph = sub_builder.GetGraph();
  root_graph->AddSubGraph(sub_graph);
  sub_graph->SetParentNode(node);
  sub_graph->SetParentGraph(root_graph);
  node->GetOpDesc()->AddSubgraphName("branch1");
  node->GetOpDesc()->SetSubgraphInstanceName(0, "sub");

  std::vector<GraphPtr> subgraphs;
  ASSERT_EQ(NodeAdapter::Node2GNode(node).GetALLSubgraphs(subgraphs), GRAPH_SUCCESS);
  ASSERT_EQ(subgraphs.size(), 1);
}

TEST_F(GNodeTest, GetALLSubgraphs_nullptr_root_graph) {
  auto builder = ut::GraphBuilder("graph");
  const auto &node = builder.AddNode("node", "node", 0, 0);
  node->impl_->owner_graph_.reset();

  std::vector<GraphPtr> subgraphs;
  ASSERT_NE(NodeAdapter::Node2GNode(node).GetALLSubgraphs(subgraphs), GRAPH_SUCCESS);
  ASSERT_TRUE(subgraphs.empty());
}

TEST_F(GNodeTest, GetInDataNodesAndPortIndexs_success) {
  auto builder = ut::GraphBuilder("graph");
  const auto node1 = builder.AddNode("node1", "node1", 0, 1);
  const auto node2 = builder.AddNode("node2", "node2", 2, 0);
  builder.AddDataEdge(node1, 0, node2, 0);
  GNode gnode;
  ASSERT_EQ(gnode.GetInDataNodesAndPortIndexs(0).first, nullptr);
  gnode.impl_ = nullptr;
  ASSERT_EQ(gnode.GetInDataNodesAndPortIndexs(0).first, nullptr);
  gnode = NodeAdapter::Node2GNode(node2);
  ASSERT_NE(gnode.GetInDataNodesAndPortIndexs(0).first, nullptr);
  ASSERT_EQ(gnode.GetInDataNodesAndPortIndexs(1).first, nullptr);
}

TEST_F(GNodeTest, GetoutDataNodesAndPortIndexs_success) {
  auto builder = ut::GraphBuilder("graph");
  const auto node1 = builder.AddNode("node1", "node1", 0, 1);
  const auto node2 = builder.AddNode("node2", "node2", 1, 0);
  builder.AddDataEdge(node1, 0, node2, 0);
  GNode gnode;
  ASSERT_EQ(gnode.GetOutDataNodesAndPortIndexs(0).size(), 0);
  gnode.impl_ = nullptr;
  ASSERT_EQ(gnode.GetOutDataNodesAndPortIndexs(0).size(), 0);
  gnode = NodeAdapter::Node2GNode(node1);
  ASSERT_EQ(gnode.GetOutDataNodesAndPortIndexs(0).size(), 1);
}

TEST_F(GNodeTest, GetInControlNodes_success) {
  auto builder = ut::GraphBuilder("graph");
  const auto node1 = builder.AddNode("node1", "node1", 1, 0);
  const auto node2 = builder.AddNode("node2", "node2", 0, 1);
  builder.AddControlEdge(node1, node2);
  GNode gnode;
  vector<GNodePtr> in_contorl_nodes = {};
  ASSERT_EQ(gnode.GetInControlNodes().size(), 0);
  gnode.impl_ = nullptr;
  ASSERT_EQ(gnode.GetInControlNodes().size(), 0);
  gnode = NodeAdapter::Node2GNode(node2);
  ASSERT_EQ(gnode.GetInControlNodes().size(), 1);
}

TEST_F(GNodeTest, GetOutControlNodes_success) {
  auto builder = ut::GraphBuilder("graph");
  const auto node1 = builder.AddNode("node1", "node1", 1, 0);
  const auto node2 = builder.AddNode("node2", "node2", 0, 1);
  builder.AddControlEdge(node1, node2);
  GNode gnode;
  vector<GNodePtr> in_contorl_nodes = {};
  ASSERT_EQ(gnode.GetOutControlNodes().size(), 0);
  gnode.impl_ = nullptr;
  ASSERT_EQ(gnode.GetOutControlNodes().size(), 0);
  gnode = NodeAdapter::Node2GNode(node1);
  ASSERT_EQ(gnode.GetOutControlNodes().size(), 1);
}

TEST_F(GNodeTest, Node2GNodePtr_success) {
  auto builder = ut::GraphBuilder("graph");
  NodePtr node = nullptr;
  ASSERT_EQ(NodeAdapter::Node2GNodePtr(node), nullptr);
  node = builder.AddNode("node", "node", 0, 0);
  ASSERT_NE(NodeAdapter::Node2GNodePtr(node), nullptr);
}

TEST_F(GNodeTest, Node2GNode2Node_success) {
  auto builder = ut::GraphBuilder("graph");
  NodePtr node = nullptr;
  ASSERT_EQ(NodeAdapter::GNode2Node(NodeAdapter::Node2GNode(node)), nullptr);
  node = builder.AddNode("node", "node", 0, 0);
  ASSERT_EQ(NodeAdapter::GNode2Node(NodeAdapter::Node2GNode(node)), node);
}

TEST_F(GNodeTest, GetName_Type_success) {
  auto builder = ut::GraphBuilder("graph");
  const auto node = builder.AddNode("name", "type", 0, 0);
  GNode gnode;
  AscendString name;
  AscendString type;
  ASSERT_EQ(gnode.GetName(name), GRAPH_FAILED);
  ASSERT_EQ(gnode.GetType(type), GRAPH_FAILED);
  gnode.impl_ = nullptr;
  ASSERT_EQ(gnode.GetName(name), GRAPH_FAILED);
  ASSERT_EQ(gnode.GetType(type), GRAPH_FAILED);
  gnode = NodeAdapter::Node2GNode(node);
  gnode.GetName(name);
  gnode.GetType(type);
  ASSERT_EQ(name, "name");
  ASSERT_EQ(type, "type");
}

TEST_F(GNodeTest, GetInputConstData_success) {
  auto sub_builder = ut::GraphBuilder("graph");
  const auto &sub_node = sub_builder.AddNode("sub_node", "Data", 3, 1);
  const auto &sub_in_data_node = sub_builder.AddNode("sub_in_data_node", "Data", 1, 1);
  const auto &sub_in_const_node = sub_builder.AddNode("sub_in_const_node", "Const", 1, 1);
  const auto &sub_in_other_node = sub_builder.AddNode("sub_in_other_node", "node_1", 0, 1);
  sub_builder.AddDataEdge(sub_in_const_node, 0, sub_node, 1);
  sub_builder.AddDataEdge(sub_in_other_node, 0, sub_node, 2);
  sub_builder.AddDataEdge(sub_in_data_node, 0, sub_node, 0);
  EXPECT_TRUE(AttrUtils::SetInt(sub_in_data_node->GetOpDesc(), "_parent_node_index", 0));
  auto root_builder = ut::GraphBuilder("graph1");
  auto root_graph = root_builder.GetGraph();
  auto sub_graph = sub_builder.GetGraph();
  const auto &root_in_node = sub_builder.AddNode("root_in_node", "Const", 0, 1);
  const auto &root_node = sub_builder.AddNode("root_node", "Data1", 1, 1);
  root_builder.AddDataEdge(root_in_node, 0, root_node, 0);
  sub_graph->SetParentGraph(root_graph);
  sub_graph->SetParentNode(root_node);
  GeTensor getensor;
  AttrUtils::SetTensor(root_in_node->GetOpDesc(), "value", getensor);
  AttrUtils::SetTensor(sub_in_const_node->GetOpDesc(), "value", getensor);
  root_node->GetOpDesc()->AddSubgraphName("sub_graph");
  root_node->GetOpDesc()->SetSubgraphInstanceName(0, "sub_graph");
  root_graph->AddSubgraph("sub_graph", sub_graph);
  Tensor data;
  GNode gnode;
  gnode.impl_ = nullptr;
  ASSERT_EQ(gnode.GetInputConstData(0, data), GRAPH_FAILED);
  gnode = NodeAdapter::Node2GNode(sub_node);
  ASSERT_EQ(gnode.GetInputConstData(0, data), GRAPH_SUCCESS);
  gnode = NodeAdapter::Node2GNode(sub_node);
  ASSERT_EQ(gnode.GetInputConstData(1, data), GRAPH_SUCCESS);
  gnode = NodeAdapter::Node2GNode(sub_node);
  ASSERT_EQ(gnode.GetInputConstData(2, data), GRAPH_NODE_WITHOUT_CONST_INPUT);
}

TEST_F(GNodeTest, GetInputIndexByName_success) {
  auto builder = ut::GraphBuilder("graph");
  const auto &node = builder.AddNode("node", "node", 1, 1);
  const auto &in_other_node = builder.AddNode("in_other_node", "node_1", 0, 1);
  builder.AddDataEdge(in_other_node, 0, node, 0);
  AscendString name = nullptr;
  GNode gnode;
  int input_index;
  ASSERT_EQ(gnode.GetInputIndexByName(name, input_index), GRAPH_PARAM_INVALID);
  ASSERT_EQ(gnode.GetInputIndexByName("in_other_node", input_index), GRAPH_FAILED);
  gnode.impl_ = nullptr;
  ASSERT_EQ(gnode.GetInputIndexByName("in_other_node", input_index), GRAPH_FAILED);
  gnode = NodeAdapter::Node2GNode(node);
  ASSERT_EQ(gnode.GetInputIndexByName("in_other_node", input_index), GRAPH_SUCCESS);
}

TEST_F(GNodeTest, GetDynamicInputIndexesByName_Failed) {
  auto builder = ut::GraphBuilder("graph");
  const auto &node = builder.AddNode("node", "node", 3, 1);
  const auto &in_node = builder.AddNode("in_node", "node_1", 0, 1);
  builder.AddDataEdge(in_node, 0, node, 0);
  AscendString name = nullptr;
  GNode gnode;
  std::vector<int> input_indexes;
  ASSERT_EQ(gnode.GetDynamicInputIndexesByName(name, input_indexes), GRAPH_PARAM_INVALID);
  ASSERT_EQ(gnode.GetDynamicInputIndexesByName("in_node", input_indexes), GRAPH_FAILED);
  gnode.impl_ = nullptr;
  ASSERT_EQ(gnode.GetDynamicInputIndexesByName("in_node", input_indexes), GRAPH_FAILED);
  gnode = NodeAdapter::Node2GNode(node);
  ASSERT_EQ(gnode.GetDynamicInputIndexesByName("in_node", input_indexes), GRAPH_FAILED);
}

TEST_F(GNodeTest, GetDynamicInputIndexesByName_Success) {
  auto builder = ut::GraphBuilder("graph");
  const auto &node = builder.AddNode("node", "node", 3, 1);
  const auto &in_node = builder.AddNode("in_node", "node_1", 0, 1);
  builder.AddDataEdge(in_node, 0, node, 0);
  GNode gnode;
  std::vector<int> input_indexes;
  auto op_desc = node->GetOpDesc();
  ge::GeTensorDesc tensor_desc;
  op_desc->AddInputDesc("in_node0", tensor_desc);
  gnode = NodeAdapter::Node2GNode(node);
  ASSERT_EQ(gnode.GetDynamicInputIndexesByName("in_node", input_indexes), GRAPH_SUCCESS);
  ASSERT_EQ(input_indexes.size(), 1U);
  input_indexes.clear();
  // 默认的名字是__input + index
  ASSERT_EQ(gnode.GetDynamicInputIndexesByName("__input", input_indexes), GRAPH_SUCCESS);
  ASSERT_EQ(input_indexes.size(), 3U);
}

TEST_F(GNodeTest, GetOutputIndexByName_success) {
  auto builder = ut::GraphBuilder("graph");
  const auto &node = builder.AddNode("node", "node", 0, 1);
  const auto &in_other_node = builder.AddNode("in_other_node", "node_1", 1, 0);
  builder.AddDataEdge(node, 0, in_other_node, 0);
  AscendString name = nullptr;
  GNode gnode;
  int input_index;
  ASSERT_EQ(gnode.GetOutputIndexByName(name, input_index), GRAPH_PARAM_INVALID);
  ASSERT_EQ(gnode.GetOutputIndexByName("in_other_node", input_index), GRAPH_FAILED);
  gnode.impl_ = nullptr;
  ASSERT_EQ(gnode.GetOutputIndexByName("in_other_node", input_index), GRAPH_FAILED);
  gnode = NodeAdapter::Node2GNode(node);
  ASSERT_EQ(gnode.GetOutputIndexByName("in_other_node", input_index), GRAPH_SUCCESS);
}

TEST_F(GNodeTest, GetDynamicOutputIndexesByName_Failed) {
  auto builder = ut::GraphBuilder("graph");
  const auto &node = builder.AddNode("node", "node", 0, 1);
  const auto &out_node = builder.AddNode("out_node", "node_1", 1, 0);
  builder.AddDataEdge(node, 0, out_node, 0);
  AscendString name = nullptr;
  GNode gnode;
  std::vector<int> output_indexes;
  ASSERT_EQ(gnode.GetDynamicOutputIndexesByName(name, output_indexes), GRAPH_PARAM_INVALID);
  ASSERT_EQ(gnode.GetDynamicOutputIndexesByName("out_node", output_indexes), GRAPH_FAILED);
  gnode.impl_ = nullptr;
  ASSERT_EQ(gnode.GetDynamicOutputIndexesByName("out_node", output_indexes), GRAPH_FAILED);
  gnode = NodeAdapter::Node2GNode(out_node);
  ASSERT_EQ(gnode.GetDynamicOutputIndexesByName("out_node", output_indexes), GRAPH_FAILED);
}

TEST_F(GNodeTest, GetDynamicOutputIndexesByName_Success) {
  auto builder = ut::GraphBuilder("graph");
  const auto &node = builder.AddNode("node", "node", 0, 1);
  const auto &out_node = builder.AddNode("out_node", "node_1", 1, 0);
  builder.AddDataEdge(node, 0, out_node, 0);
  AscendString name = nullptr;
  GNode gnode;
  std::vector<int> output_indexes;
  gnode = NodeAdapter::Node2GNode(node);
  ASSERT_EQ(gnode.GetDynamicOutputIndexesByName("__output", output_indexes), GRAPH_SUCCESS);
  ASSERT_EQ(output_indexes.size(), 1U);
}

TEST_F(GNodeTest, GetInputs_outputs_Size_success) {
  auto builder = ut::GraphBuilder("graph");
  const auto &node = builder.AddNode("node", "node", 1, 1);
  const auto &in_node = builder.AddNode("node_in", "node", 0, 1);
  const auto &out_node = builder.AddNode("node_out", "node", 1, 0);
  GNode gnode;
  ASSERT_EQ(gnode.GetInputsSize(), GRAPH_FAILED);
  ASSERT_EQ(gnode.GetOutputsSize(), GRAPH_FAILED);
  gnode.impl_ = nullptr;
  ASSERT_EQ(gnode.GetInputsSize(), GRAPH_FAILED);
  ASSERT_EQ(gnode.GetOutputsSize(), GRAPH_FAILED);
  gnode = NodeAdapter::Node2GNode(node);
  ASSERT_EQ(gnode.GetInputsSize(), 1);
  ASSERT_EQ(gnode.GetOutputsSize(), 1);
}

TEST_F(GNodeTest, GetInputDesc_success) {
  auto builder = ut::GraphBuilder("graph");
  const auto node = builder.AddNode("node", "node", 1, 1);
  auto opdesc = node->GetOpDesc();
  GNode gnode;
  TensorDesc tensordesc;
  ASSERT_EQ(gnode.GetInputDesc(1, tensordesc), GRAPH_FAILED);
  ASSERT_EQ(gnode.GetInputDesc(-1, tensordesc), GRAPH_PARAM_INVALID);
  gnode.impl_ = nullptr;
  ASSERT_EQ(gnode.GetInputDesc(0, tensordesc), GRAPH_FAILED);
  gnode = NodeAdapter::Node2GNode(node);
  ASSERT_EQ(gnode.GetInputDesc(0, tensordesc), GRAPH_SUCCESS);
}

TEST_F(GNodeTest, UpdateInputDesc_success) {
  auto builder = ut::GraphBuilder("graph");
  const auto node = builder.AddNode("node", "node", 1, 1);
  GNode gnode;
  const TensorDesc tensordesc;
  ASSERT_EQ(gnode.UpdateInputDesc(-1, tensordesc), GRAPH_PARAM_INVALID);
  ASSERT_EQ(gnode.UpdateInputDesc(1, tensordesc), GRAPH_FAILED);
  gnode.impl_ = nullptr;
  ASSERT_EQ(gnode.UpdateInputDesc(0, tensordesc), GRAPH_FAILED);
  gnode = NodeAdapter::Node2GNode(node);
  ASSERT_EQ(gnode.UpdateInputDesc(0, tensordesc), GRAPH_SUCCESS);
}

TEST_F(GNodeTest, GetOutputDesc_success) {
  auto builder = ut::GraphBuilder("graph");
  const auto node = builder.AddNode("node", "node", 1, 1);
  auto opdesc = node->GetOpDesc();
  GNode gnode;
  TensorDesc tensordesc;
  ASSERT_EQ(gnode.GetOutputDesc(1, tensordesc), GRAPH_FAILED);
  ASSERT_EQ(gnode.GetOutputDesc(-1, tensordesc), GRAPH_PARAM_INVALID);
  gnode.impl_ = nullptr;
  ASSERT_EQ(gnode.GetOutputDesc(0, tensordesc), GRAPH_FAILED);
  gnode = NodeAdapter::Node2GNode(node);
  ASSERT_EQ(gnode.GetOutputDesc(0, tensordesc), GRAPH_SUCCESS);
}

TEST_F(GNodeTest, UpdateOutputDesc_success) {
  auto builder = ut::GraphBuilder("graph");
  const auto node = builder.AddNode("node", "node", 1, 1);
  GNode gnode;
  const TensorDesc tensordesc;
  ASSERT_EQ(gnode.UpdateOutputDesc(-1, tensordesc), GRAPH_PARAM_INVALID);
  ASSERT_EQ(gnode.UpdateOutputDesc(1, tensordesc), GRAPH_FAILED);
  gnode.impl_ = nullptr;
  ASSERT_EQ(gnode.UpdateOutputDesc(0, tensordesc), GRAPH_FAILED);
  gnode = NodeAdapter::Node2GNode(node);
  ASSERT_EQ(gnode.UpdateOutputDesc(0, tensordesc), GRAPH_SUCCESS);
}

TEST_F(GNodeTest, SetAttr1_success) {
  auto builder = ut::GraphBuilder("graph");
  const auto &node = builder.AddNode("node", "node", 0, 0);
  GNode gnode;
  AscendString name = nullptr;
  vector<AscendString> attr_value;
  ASSERT_EQ(gnode.SetAttr(name, attr_value), GRAPH_PARAM_INVALID);
  attr_value.emplace_back(name);
  name = "node";
  ASSERT_EQ(gnode.SetAttr(name, attr_value), GRAPH_PARAM_INVALID);
  attr_value.clear();
  attr_value.emplace_back(name);
  ASSERT_EQ(gnode.SetAttr(name, attr_value), GRAPH_FAILED);
  gnode.impl_ = nullptr;
  ASSERT_EQ(gnode.SetAttr(name, attr_value), GRAPH_FAILED);
  gnode = NodeAdapter::Node2GNode(node);
  ASSERT_EQ(gnode.SetAttr(name, attr_value), GRAPH_SUCCESS);
}

TEST_F(GNodeTest, SetAttr2_success) {
  auto builder = ut::GraphBuilder("graph");
  const auto &node = builder.AddNode("node", "node", 0, 0);
  GNode gnode;
  AscendString name = nullptr;
  AscendString attr_value = nullptr;
  ASSERT_EQ(gnode.SetAttr(name, attr_value), GRAPH_PARAM_INVALID);
  name = "node";
  ASSERT_NE(gnode.SetAttr(name, attr_value), GRAPH_SUCCESS);
  attr_value = "value";
  ASSERT_EQ(gnode.SetAttr(name, attr_value), GRAPH_FAILED);
  gnode.impl_ = nullptr;
  ASSERT_EQ(gnode.SetAttr(name, attr_value), GRAPH_FAILED);
  gnode = NodeAdapter::Node2GNode(node);
  ASSERT_EQ(gnode.SetAttr(name, attr_value), GRAPH_SUCCESS);
}

TEST_F(GNodeTest, SetAttr3_success) {
  auto builder = ut::GraphBuilder("graph");
  const auto &node = builder.AddNode("node", "node", 0, 0);
  GNode gnode;
  AscendString name = nullptr;
  AttrValue attr_value;
  ASSERT_EQ(gnode.SetAttr(name, attr_value), GRAPH_PARAM_INVALID);
  name = "node";
  ASSERT_EQ(gnode.SetAttr(name, attr_value), GRAPH_FAILED);
  gnode.impl_ = nullptr;
  ASSERT_EQ(gnode.SetAttr(name, attr_value), GRAPH_FAILED);
  gnode = NodeAdapter::Node2GNode(node);
  ASSERT_EQ(gnode.SetAttr(name, attr_value), GRAPH_SUCCESS);
}

TEST_F(GNodeTest, GetAttr1_success) {
  auto builder = ut::GraphBuilder("graph");
  const auto &node = builder.AddNode("node", "node", 0, 0);
  GNode gnode;
  AscendString name = nullptr;
  AscendString attr_value = "value";
  ASSERT_EQ(gnode.GetAttr(name, attr_value), GRAPH_PARAM_INVALID);
  name = "node";
  ASSERT_EQ(gnode.GetAttr(name, attr_value), GRAPH_FAILED);
  gnode.impl_ = nullptr;
  ASSERT_EQ(gnode.GetAttr(name, attr_value), GRAPH_FAILED);
  gnode = NodeAdapter::Node2GNode(node);
  ASSERT_EQ(gnode.GetAttr("max_size", attr_value), GRAPH_FAILED);
  gnode.SetAttr(name, attr_value);
  ASSERT_EQ(gnode.GetAttr(name, attr_value), GRAPH_SUCCESS);
}

TEST_F(GNodeTest, GetAttr2_success) {
  auto builder = ut::GraphBuilder("graph");
  const auto &node = builder.AddNode("node", "node", 0, 0);
  GNode gnode;
  AscendString name = nullptr;
  vector<AscendString> attr_value = {"value"};
  ASSERT_EQ(gnode.GetAttr(name, attr_value), GRAPH_PARAM_INVALID);
  name = "node";
  ASSERT_EQ(gnode.GetAttr(name, attr_value), GRAPH_FAILED);
  gnode.impl_ = nullptr;
  ASSERT_EQ(gnode.GetAttr(name, attr_value), GRAPH_FAILED);
  gnode = NodeAdapter::Node2GNode(node);
  ASSERT_EQ(gnode.GetAttr("max_size", attr_value), GRAPH_FAILED);
  int32_t attr_info = 0;
  ASSERT_EQ(gnode.GetAttr("max_size", attr_info), GRAPH_FAILED);
  gnode.SetAttr(name, attr_value);
  attr_value.clear();
  ASSERT_EQ(gnode.GetAttr(name, attr_value), GRAPH_SUCCESS);
}

TEST_F(GNodeTest, HasAttr_Success) {
  auto builder = ut::GraphBuilder("graph");
  const auto node = builder.AddNode("node", "node", 0, 0);
  GNode gnode;
  AscendString name = nullptr;
  AscendString attr_value = "value";
  ASSERT_EQ(gnode.HasAttr(name), false);
  name = "node";
  ASSERT_EQ(gnode.HasAttr(name), false);
  gnode.impl_ = nullptr;
  ASSERT_EQ(gnode.HasAttr(name), false);
  gnode = NodeAdapter::Node2GNode(node);
  ASSERT_EQ(gnode.HasAttr("max_size"), false);
  gnode.SetAttr(name, attr_value);
  ASSERT_EQ(gnode.HasAttr(name), true);
}

TEST_F(GNodeTest, GetSubGraph_success) {
  auto sub_builder = ut::GraphBuilder("sub_graph");
  auto sub_graph = sub_builder.GetGraph();
  auto root_builder = ut::GraphBuilder("root_graph");
  const auto node = root_builder.AddNode("node", "node", 1, 1);
  auto root_graph = root_builder.GetGraph();
  sub_graph->SetParentGraph(root_graph);
  sub_graph->SetParentNode(node);
  node->GetOpDesc()->AddSubgraphName("sub_graph");
  node->GetOpDesc()->SetSubgraphInstanceName(0, "sub_graph");
  root_graph->AddSubgraph("sub_graph", sub_graph);
  GNode gnode;
  GraphPtr graph;
  ASSERT_EQ(gnode.GetSubgraph(0U, graph), GRAPH_FAILED);
  gnode.impl_ = nullptr;
  ASSERT_EQ(gnode.GetSubgraph(0U, graph), GRAPH_FAILED);
  gnode = NodeAdapter::Node2GNode(node);
  ASSERT_EQ(gnode.GetSubgraph(0U, graph), GRAPH_SUCCESS);
}
extern "C" {
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus aclCom_GNode_GetOutputAttr(void *node_ptr, const char *name,
                                                                                      int32_t output_index,
                                                                                      void *attr_value) {
  if (node_ptr == nullptr || name == nullptr || attr_value == nullptr) {
    return GRAPH_FAILED;
  }
  auto *node = static_cast<ge::GNode *>(node_ptr);
  auto *value = static_cast<ge::AttrValue *>(attr_value);
  return node->GetOutputAttr(ge::AscendString(name), output_index, *value);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus aclCom_GNode_SetOutputAttr(void *node_ptr, const char *name,
                                                                                      int32_t output_index,
                                                                                      const void *attr_value) {
  if (node_ptr == nullptr || name == nullptr || attr_value == nullptr) {
    return GRAPH_FAILED;
  }
  auto *node = static_cast<ge::GNode *>(node_ptr);
  auto *value = static_cast<const ge::AttrValue *>(attr_value);
  return node->SetOutputAttr(ge::AscendString(name), output_index, *value);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus aclCom_GNode_GetInputAttr(void *node_ptr, const char *name,
                                                                                     int32_t input_index,
                                                                                     void *attr_value) {
  if (node_ptr == nullptr || name == nullptr || attr_value == nullptr) {
    return GRAPH_FAILED;
  }
  auto *node = static_cast<ge::GNode *>(node_ptr);
  auto *value = static_cast<ge::AttrValue *>(attr_value);
  return node->GetInputAttr(ge::AscendString(name), input_index, *value);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus aclCom_GNode_SetInputAttr(void *node_ptr, const char *name,
                                                                                     int32_t input_index,
                                                                                     const void *attr_value) {
  if (node_ptr == nullptr || name == nullptr || attr_value == nullptr) {
    return GRAPH_FAILED;
  }
  auto *node = static_cast<ge::GNode *>(node_ptr);
  auto *value = static_cast<const ge::AttrValue *>(attr_value);
  return node->SetInputAttr(ge::AscendString(name), input_index, *value);
}
}
TEST_F(GNodeTest, ExternC_GNode_SetOutputAttr_Int64_Success) {
  auto builder = ut::GraphBuilder("graph");
  const auto &node = builder.AddNode("node", "node", 1, 1);
  GNode gnode = NodeAdapter::Node2GNode(node);

  AttrValue attr_value;
  int64_t int64_value = 12345;
  attr_value.SetAttrValue(int64_value);

  // 测试成功情况
  EXPECT_EQ(aclCom_GNode_SetOutputAttr(&gnode, "test_attr", 0, &attr_value), GRAPH_SUCCESS);

  // 验证设置的值
  AttrValue get_attr_value;
  EXPECT_EQ(gnode.GetOutputAttr("test_attr", 0, get_attr_value), GRAPH_SUCCESS);
  int64_t get_value = 0;
  EXPECT_EQ(get_attr_value.GetAttrValue(get_value), GRAPH_SUCCESS);
  EXPECT_EQ(get_value, int64_value);

  // 测试nullptr参数
  EXPECT_EQ(aclCom_GNode_SetOutputAttr(nullptr, "test_attr", 0, &attr_value), GRAPH_FAILED);
  EXPECT_EQ(aclCom_GNode_SetOutputAttr(&gnode, nullptr, 0, &attr_value), GRAPH_FAILED);
  EXPECT_EQ(aclCom_GNode_SetOutputAttr(&gnode, "test_attr", 0, nullptr), GRAPH_FAILED);
}

TEST_F(GNodeTest, ExternC_GNode_SetOutputAttr_String_Success) {
  auto builder = ut::GraphBuilder("graph");
  const auto &node = builder.AddNode("node", "node", 1, 1);
  GNode gnode = NodeAdapter::Node2GNode(node);

  AttrValue attr_value;
  const char *str_value = "test_string";
  attr_value.SetAttrValue(AscendString(str_value));

  // 测试成功情况
  EXPECT_EQ(aclCom_GNode_SetOutputAttr(&gnode, "test_attr", 0, &attr_value), GRAPH_SUCCESS);

  // 验证设置的值
  AttrValue get_attr_value;
  EXPECT_EQ(aclCom_GNode_GetOutputAttr(&gnode, "test_attr", 0, &get_attr_value), GRAPH_SUCCESS);
  AscendString get_value;
  EXPECT_EQ(get_attr_value.GetAttrValue(get_value), GRAPH_SUCCESS);
  EXPECT_STREQ(get_value.GetString(), str_value);

  // 测试nullptr参数
  EXPECT_EQ(aclCom_GNode_SetOutputAttr(nullptr, "test_attr", 0, &attr_value), GRAPH_FAILED);
  EXPECT_EQ(aclCom_GNode_SetOutputAttr(&gnode, nullptr, 0, &attr_value), GRAPH_FAILED);
  EXPECT_EQ(aclCom_GNode_SetOutputAttr(&gnode, "test_attr", 0, nullptr), GRAPH_FAILED);
}

TEST_F(GNodeTest, ExternC_GNode_SetOutputAttr_Bool_Success) {
  auto builder = ut::GraphBuilder("graph");
  const auto &node = builder.AddNode("node", "node", 1, 1);
  GNode gnode = NodeAdapter::Node2GNode(node);

  AttrValue attr_value;
  bool bool_value = true;
  attr_value.SetAttrValue(bool_value);

  // 测试成功情况
  EXPECT_EQ(aclCom_GNode_SetOutputAttr(&gnode, "test_attr", 0, &attr_value), GRAPH_SUCCESS);

  // 验证设置的值
  AttrValue get_attr_value;
  EXPECT_EQ(gnode.GetOutputAttr("test_attr", 0, get_attr_value), GRAPH_SUCCESS);
  bool get_value = false;
  EXPECT_EQ(get_attr_value.GetAttrValue(get_value), GRAPH_SUCCESS);
  EXPECT_EQ(get_value, bool_value);

  // 测试nullptr参数
  EXPECT_EQ(aclCom_GNode_SetOutputAttr(nullptr, "test_attr", 0, &attr_value), GRAPH_FAILED);
  EXPECT_EQ(aclCom_GNode_SetOutputAttr(&gnode, nullptr, 0, &attr_value), GRAPH_FAILED);
  EXPECT_EQ(aclCom_GNode_SetOutputAttr(&gnode, "test_attr", 0, nullptr), GRAPH_FAILED);
}

TEST_F(GNodeTest, ExternC_GNode_SetOutputAttr_InvalidIndex) {
  auto builder = ut::GraphBuilder("graph");
  const auto &node = builder.AddNode("node", "node", 1, 1);
  GNode gnode = NodeAdapter::Node2GNode(node);

  // 测试无效索引
  AttrValue attr_value;
  int64_t int64_value = 12345;
  attr_value.SetAttrValue(int64_value);
  EXPECT_NE(aclCom_GNode_SetOutputAttr(&gnode, "test_attr", 999, &attr_value), GRAPH_SUCCESS);
}

TEST_F(GNodeTest, ExternC_GNode_SetOutputAttr_EmptyGNode) {
  GNode empty_gnode;

  // 测试空GNode
  AttrValue attr_value;
  int64_t int64_value = 12345;
  attr_value.SetAttrValue(int64_value);
  EXPECT_NE(aclCom_GNode_SetOutputAttr(&empty_gnode, "test_attr", 0, &attr_value), GRAPH_SUCCESS);
}

TEST_F(GNodeTest, ExternC_GNode_SetInputAttr_Int64_Success) {
  auto builder = ut::GraphBuilder("graph");
  const auto &node = builder.AddNode("node", "node", 1, 1);
  GNode gnode = NodeAdapter::Node2GNode(node);

  AttrValue attr_value;
  int64_t int64_value = 12345;
  attr_value.SetAttrValue(int64_value);

  // 测试成功情况
  EXPECT_EQ(aclCom_GNode_SetInputAttr(&gnode, "test_input_attr", 0, &attr_value), GRAPH_SUCCESS);

  // 验证设置的值
  AttrValue get_attr_value;
  EXPECT_EQ(gnode.GetInputAttr("test_input_attr", 0, get_attr_value), GRAPH_SUCCESS);
  int64_t get_value = 0;
  EXPECT_EQ(get_attr_value.GetAttrValue(get_value), GRAPH_SUCCESS);
  EXPECT_EQ(get_value, int64_value);

  // 测试nullptr参数
  EXPECT_EQ(aclCom_GNode_SetInputAttr(nullptr, "test_input_attr", 0, &attr_value), GRAPH_FAILED);
  EXPECT_EQ(aclCom_GNode_SetInputAttr(&gnode, nullptr, 0, &attr_value), GRAPH_FAILED);
  EXPECT_EQ(aclCom_GNode_SetInputAttr(&gnode, "test_input_attr", 0, nullptr), GRAPH_FAILED);
}

TEST_F(GNodeTest, ExternC_GNode_SetInputAttr_String_Success) {
  auto builder = ut::GraphBuilder("graph");
  const auto &node = builder.AddNode("node", "node", 1, 1);
  GNode gnode = NodeAdapter::Node2GNode(node);

  AttrValue attr_value;
  const char *str_value = "test_input_string";
  attr_value.SetAttrValue(AscendString(str_value));

  // 测试成功情况
  EXPECT_EQ(aclCom_GNode_SetInputAttr(&gnode, "test_input_attr", 0, &attr_value), GRAPH_SUCCESS);

  // 验证设置的值
  AttrValue get_attr_value;
  EXPECT_EQ(aclCom_GNode_GetInputAttr(&gnode, "test_input_attr", 0, &get_attr_value), GRAPH_SUCCESS);
  AscendString get_value;
  EXPECT_EQ(get_attr_value.GetAttrValue(get_value), GRAPH_SUCCESS);
  EXPECT_STREQ(get_value.GetString(), str_value);

  // 测试nullptr参数
  EXPECT_EQ(aclCom_GNode_SetInputAttr(nullptr, "test_input_attr", 0, &attr_value), GRAPH_FAILED);
  EXPECT_EQ(aclCom_GNode_SetInputAttr(&gnode, nullptr, 0, &attr_value), GRAPH_FAILED);
  EXPECT_EQ(aclCom_GNode_SetInputAttr(&gnode, "test_input_attr", 0, nullptr), GRAPH_FAILED);
}

TEST_F(GNodeTest, ExternC_GNode_SetInputAttr_Bool_Success) {
  auto builder = ut::GraphBuilder("graph");
  const auto &node = builder.AddNode("node", "node", 1, 1);
  GNode gnode = NodeAdapter::Node2GNode(node);

  AttrValue attr_value;
  bool bool_value = true;
  attr_value.SetAttrValue(bool_value);

  // 测试成功情况
  EXPECT_EQ(aclCom_GNode_SetInputAttr(&gnode, "test_input_attr", 0, &attr_value), GRAPH_SUCCESS);

  // 验证设置的值
  AttrValue get_attr_value;
  EXPECT_EQ(gnode.GetInputAttr("test_input_attr", 0, get_attr_value), GRAPH_SUCCESS);
  bool get_value = false;
  EXPECT_EQ(get_attr_value.GetAttrValue(get_value), GRAPH_SUCCESS);
  EXPECT_EQ(get_value, bool_value);

  // 测试nullptr参数
  EXPECT_EQ(aclCom_GNode_SetInputAttr(nullptr, "test_input_attr", 0, &attr_value), GRAPH_FAILED);
  EXPECT_EQ(aclCom_GNode_SetInputAttr(&gnode, nullptr, 0, &attr_value), GRAPH_FAILED);
  EXPECT_EQ(aclCom_GNode_SetInputAttr(&gnode, "test_input_attr", 0, nullptr), GRAPH_FAILED);
}

TEST_F(GNodeTest, ExternC_GNode_SetInputAttr_VectorInt64_Success) {
  auto builder = ut::GraphBuilder("graph");
  const auto &node = builder.AddNode("node", "node", 1, 1);
  GNode gnode = NodeAdapter::Node2GNode(node);

  AttrValue attr_value;
  std::vector<int64_t> vector_value = {1, 2, 3, 4, 5};
  attr_value.SetAttrValue(vector_value);

  // 测试成功情况
  EXPECT_EQ(aclCom_GNode_SetInputAttr(&gnode, "test_input_attr", 0, &attr_value), GRAPH_SUCCESS);

  // 验证设置的值
  AttrValue get_attr_value;
  EXPECT_EQ(gnode.GetInputAttr("test_input_attr", 0, get_attr_value), GRAPH_SUCCESS);
  std::vector<int64_t> get_value;
  EXPECT_EQ(get_attr_value.GetAttrValue(get_value), GRAPH_SUCCESS);
  EXPECT_EQ(get_value, vector_value);

  // 测试nullptr参数
  EXPECT_EQ(aclCom_GNode_SetInputAttr(nullptr, "test_input_attr", 0, &attr_value), GRAPH_FAILED);
  EXPECT_EQ(aclCom_GNode_SetInputAttr(&gnode, nullptr, 0, &attr_value), GRAPH_FAILED);
  EXPECT_EQ(aclCom_GNode_SetInputAttr(&gnode, "test_input_attr", 0, nullptr), GRAPH_FAILED);
}

TEST_F(GNodeTest, ExternC_GNode_SetInputAttr_InvalidIndex) {
  auto builder = ut::GraphBuilder("graph");
  const auto &node = builder.AddNode("node", "node", 1, 1);
  GNode gnode = NodeAdapter::Node2GNode(node);

  // 测试无效索引
  AttrValue attr_value;
  int64_t int64_value = 12345;
  attr_value.SetAttrValue(int64_value);
  EXPECT_NE(aclCom_GNode_SetInputAttr(&gnode, "test_input_attr", 999, &attr_value), GRAPH_SUCCESS);
}

TEST_F(GNodeTest, ExternC_GNode_SetInputAttr_EmptyGNode) {
  GNode empty_gnode;

  // 测试空GNode
  AttrValue attr_value;
  int64_t int64_value = 12345;
  attr_value.SetAttrValue(int64_value);
  EXPECT_NE(aclCom_GNode_SetInputAttr(&empty_gnode, "test_input_attr", 0, &attr_value), GRAPH_SUCCESS);
}

REG_OP(subTest)
    .INPUT(inx, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .GRAPH(subgraph)
    .OP_END_FACTORY_REG(subTest);

REG_OP(dynamicSubTest)
    .INPUT(inx, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .DYNAMIC_GRAPH(subgraphs)
    .OP_END_FACTORY_REG(dynamicSubTest);

REG_OP(dataOp)
    .OUTPUT(y, TensorType::ALL())
    .ATTR(value, Int, 0)
    .OP_END_FACTORY_REG(dataOp);

TEST_F(GNodeTest, TestAddSubGraph_success) {
  auto op = op::subTest("subTest");
  Operator op_input1 = op::dataOp("op_input1");
  std::vector<Operator> inputs = {op_input1};
  AscendString name = "graph";
  GraphPtr graph = Graph::ConstructFromInputs(inputs, name);
  auto gnode = graph->AddNodeByOp(op);

  name = "subgraph1";
  Operator op_input2 = op::dataOp("op_input2");
  inputs = {op_input2};
  auto subgraph = Graph::ConstructFromInputs(inputs, name);
  ASSERT_EQ(GRAPH_SUCCESS, gnode.SetSubgraph("subgraph", *subgraph.get()));

  auto parent_graph = subgraph->GetParentGraph();
  AscendString parent_graph_name;
  parent_graph->GetName(parent_graph_name);
  AscendString graph_name;
  graph->GetName(graph_name);
  ASSERT_TRUE(parent_graph_name == graph_name);

  AscendString find_name;
  op.GetName(find_name);
  auto get_node = parent_graph->FindNodeByName(find_name);
  AscendString exp_name("subTest");
  ASSERT_TRUE(find_name == exp_name);

  std::vector<AscendString> subgraph_names{};
  ASSERT_EQ(GRAPH_SUCCESS, op.GetSubgraphNames(subgraph_names));
  ASSERT_EQ(1, subgraph_names.size());
  AscendString exp_subgraph_name("subgraph");
  ASSERT_TRUE(exp_subgraph_name == subgraph_names.at(0));
}

TEST_F(GNodeTest, TestAddSubGraph_failure_invalid_graph_and_node) {
  auto op = op::subTest("subTest");
  Operator op_input1 = op::dataOp("op_input1");
  std::vector<Operator> inputs1 = {op_input1};
  AscendString name = "graph";
  GraphPtr graph_valid = Graph::ConstructFromInputs(inputs1, name);
  auto node_valid = graph_valid->AddNodeByOp(op);

  Graph graph_invalid("graph_invalid");
  ASSERT_EQ(nullptr, graph_invalid.GetParentGraph());
  Graph subgraph_invalid("subgraph_invalid");
  ASSERT_EQ(ge::PARAM_INVALID, node_valid.SetSubgraph("subgraph", subgraph_invalid));

  Operator op_input2 = op::dataOp("op_input");
  std::vector<Operator> inputs2 = {op_input2};
  GraphPtr subgraph_valid = Graph::ConstructFromInputs(inputs2, name);
  ASSERT_EQ(GRAPH_SUCCESS, node_valid.SetSubgraph("subgraph", *subgraph_valid.get()));
}

TEST_F(GNodeTest, TestAddStaticSubGraph_success_different_name) {
  auto op = op::subTest("subTest");
  Operator op_input1 = op::dataOp("op_input1");
  std::vector<Operator> inputs1 = {op_input1};
  AscendString name = "graph";
  GraphPtr graph_valid = Graph::ConstructFromInputs(inputs1, name);
  auto node_valid = graph_valid->AddNodeByOp(op);

  Operator op_input2 = op::dataOp("op_input");
  std::vector<Operator> inputs2 = {op_input2};
  GraphPtr subgraph_valid = Graph::ConstructFromInputs(inputs2, name);
  ASSERT_EQ(GRAPH_SUCCESS, node_valid.SetSubgraph("subgraph_diff", *subgraph_valid.get()));
}

TEST_F(GNodeTest, TestAddDynamicSubGraph_success) {
  auto op = op::dynamicSubTest("dynamicSubTest");
  Operator op_input_parent = op::dataOp("op_input1");
  std::vector<Operator> inputs = {op_input_parent};
  AscendString name = "graph02";
  GraphPtr graph = Graph::ConstructFromInputs(inputs, name);
  auto gnode = graph->AddNodeByOp(op);

  name = "subgraph1";
  Operator op_input1 = op::dataOp("op_input1");
  inputs = {op_input1};
  GraphPtr subgraph1 = Graph::ConstructFromInputs(inputs, name);
  name = "subgraph2";
  Operator op_input2 = op::dataOp("op_input2");
  inputs = {op_input2};
  GraphPtr subgraph2 = Graph::ConstructFromInputs(inputs, name);
  name = "subgraph3";
  Operator op_input3 = op::dataOp("op_input3");
  inputs = {op_input3};
  GraphPtr subgraph3 = Graph::ConstructFromInputs(inputs, name);
  std::vector<Graph> dynamic_subgraphs{*subgraph1.get(), *subgraph2.get(), *subgraph3.get()};
  ASSERT_EQ(GRAPH_SUCCESS, gnode.SetSubgraphs("subgraphs", dynamic_subgraphs));

  auto parent_graph = subgraph1->GetParentGraph();
  AscendString parent_graph_name;
  parent_graph->GetName(parent_graph_name);
  AscendString graph_name;
  graph->GetName(graph_name);
  ASSERT_TRUE(parent_graph_name == graph_name);

  AscendString find_name;
  op.GetName(find_name);
  auto get_node = parent_graph->FindNodeByName(find_name);
  AscendString exp_name("dynamicSubTest");
  ASSERT_TRUE(find_name == exp_name);

  std::vector<AscendString> ir_subgraph_names(0);
  ASSERT_EQ(GRAPH_SUCCESS, op.GetSubgraphNames(ir_subgraph_names));
  ASSERT_EQ(1, ir_subgraph_names.size());
  AscendString exp_subgraph_name("subgraphs");
  ASSERT_TRUE(exp_subgraph_name == ir_subgraph_names.at(0));

  std::vector<GraphPtr> dynamic_subgraph_vec{};
  ASSERT_EQ(GRAPH_SUCCESS, gnode.GetALLSubgraphs(dynamic_subgraph_vec));
  ASSERT_EQ(dynamic_subgraphs.size(), dynamic_subgraph_vec.size());
  AscendString subgraph_name{};
  ASSERT_EQ(GRAPH_SUCCESS, dynamic_subgraph_vec.at(0)->GetName(subgraph_name));
  AscendString exp_instance_subgraphs_name("subgraph1");
  ASSERT_TRUE(exp_instance_subgraphs_name == subgraph_name);
  ASSERT_EQ(GRAPH_SUCCESS, dynamic_subgraph_vec.at(1)->GetName(subgraph_name));
  exp_instance_subgraphs_name = "subgraph2";
  ASSERT_TRUE(exp_instance_subgraphs_name == subgraph_name);
  ASSERT_EQ(GRAPH_SUCCESS, dynamic_subgraph_vec.at(2)->GetName(subgraph_name));
  exp_instance_subgraphs_name = "subgraph3";
  ASSERT_TRUE(exp_instance_subgraphs_name == subgraph_name);

  GNodePtr op_gnode = parent_graph.get()->FindNodeByName("dynamicSubTest");
  auto op_node = NodeAdapter::GNode2Node(*op_gnode.get()).get();
  auto op_desc = op_node->GetOpDesc();
  auto &subgraph = op_desc->GetSubgraphNameIndexes();
  ASSERT_EQ(dynamic_subgraphs.size(), subgraph.size());
  ASSERT_EQ(2, subgraph.find("subgraphs2")->second);
}

TEST_F(GNodeTest, TestAddDynamicSubGraph_failure_invalid_graph_and_node) {
  auto op = op::dynamicSubTest("dynamicSubTest");
  Operator op_input1 = op::dataOp("op_input1");
  std::vector<Operator> inputs1 = {op_input1};
  AscendString name = "graph";
  GraphPtr graph_valid = Graph::ConstructFromInputs(inputs1, name);
  auto node_valid = graph_valid->AddNodeByOp(op);

  Graph graph_invalid("graph_invalid");
  ASSERT_EQ(nullptr, graph_invalid.GetParentGraph());
  Graph subgraph_invalid("subgraph_invalid");
  ASSERT_EQ(ge::PARAM_INVALID, node_valid.SetSubgraph("subgraphs", subgraph_invalid));

  Operator op_input2 = op::dataOp("op_input");
  std::vector<Operator> inputs2 = {op_input2};
  GraphPtr subgraph_valid = Graph::ConstructFromInputs(inputs2, name);
  ASSERT_EQ(GRAPH_SUCCESS, node_valid.SetSubgraph("branch1", *subgraph_valid.get()));

  auto empty_node = GNode();
  ASSERT_EQ(PARAM_INVALID, empty_node.SetSubgraph("subgraphs", *subgraph_valid.get()));
  ASSERT_EQ(PARAM_INVALID, empty_node.SetSubgraphs("subgraphs", std::vector<ge::Graph>{*subgraph_valid.get()}));

  empty_node.impl_ = nullptr;
  ASSERT_EQ(PARAM_INVALID, empty_node.SetSubgraph("subgraphs", *subgraph_valid.get()));
  ASSERT_EQ(PARAM_INVALID, empty_node.SetSubgraphs("subgraphs", std::vector<ge::Graph>{*subgraph_valid.get()}));
}

TEST_F(GNodeTest, TestAddDynamicSubGraph_success_different_name) {
  auto op = op::dynamicSubTest("dynamicSubTest");
  Operator op_input1 = op::dataOp("op_input1");
  std::vector<Operator> inputs1 = {op_input1};
  AscendString name = "graph";
  GraphPtr graph_valid = Graph::ConstructFromInputs(inputs1, name);
  auto node_valid = graph_valid->AddNodeByOp(op);

  Operator op_input2 = op::dataOp("op_input");
  std::vector<Operator> inputs2 = {op_input2};
  GraphPtr subgraph_valid = Graph::ConstructFromInputs(inputs2, name);
  ASSERT_EQ(GRAPH_SUCCESS, node_valid.SetSubgraphs("branch1", std::vector<ge::Graph>{*subgraph_valid.get()}));
}
}  // namespace ge
