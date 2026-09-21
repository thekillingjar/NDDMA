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


#include "graph/node.h"
#include "graph/normal_graph/node_impl.h"
#include "graph/any_value.h"
#include "graph/anchor.h"
#include "graph/op_desc.h"
#include "graph/utils/graph_utils.h"
#include "graph/normal_graph/op_desc_impl.h"
#include "graph_builder_utils.h"
#include "graph/operator_factory_impl.h"
#include "graph/utils/node_utils_ex.h"

#include <ge_common/ge_inner_error_codes.h>

namespace ge {
class UtestNode : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

template<class T>
std::shared_ptr<T> MakeNullptr(){
  return nullptr;
}

Operator CreateOp(const AscendString& name){
  return Operator();
}


TEST_F(UtestNode, GetInDataAnchor) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data = builder.AddNode("Data", "Data", 1, 1);
  auto graph = builder.GetGraph();

  auto data_node = graph->FindNode("Data");
  auto in_data_anchor0 = data_node->GetInDataAnchor(0);
  EXPECT_NE(in_data_anchor0, nullptr);

  auto in_data_anchor1 = data_node->GetInDataAnchor(1);
  EXPECT_EQ(in_data_anchor1, nullptr);

  auto in_data_anchor2 = data_node->GetInDataAnchor(-1);
  EXPECT_EQ(in_data_anchor2, nullptr);
}
TEST_F(UtestNode, GetInAnchor) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data = builder.AddNode("Data", "Data", 1, 1);
  auto graph = builder.GetGraph();

  auto data_node = graph->FindNode("Data");
  auto in_anchor0 = data_node->GetInAnchor(-2);
  EXPECT_EQ(in_anchor0, nullptr);
}
TEST_F(UtestNode, GetOutAnchor) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data = builder.AddNode("Data", "Data", 1, 1);
  auto graph = builder.GetGraph();

  auto data_node = graph->FindNode("Data");
  auto out_anchor0 = data_node->GetOutAnchor(-2);
  EXPECT_EQ(out_anchor0, nullptr);
}

TEST_F(UtestNode, NodeInputAndOutCheck) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto attr_node = builder.AddNode("Attr", "Attr", 2, 2);
  {
    auto data_node = builder.AddNode("Data", "Data", 1, 1);
    auto graph = builder.GetGraph();
    EXPECT_TRUE(data_node->Init() == data_node->Init());
    EXPECT_EQ(data_node->SetOwnerComputeGraph(nullptr), GRAPH_PARAM_INVALID);
    EXPECT_EQ(data_node->ClearOwnerGraph(graph), GRAPH_SUCCESS);
    EXPECT_EQ(data_node->GetAllInDataAnchors().size(), 1);
    EXPECT_EQ(data_node->GetAllOutDataAnchors().size(), 1);
    EXPECT_EQ(data_node->NodeMembersAreEqual(*data_node), true);
    EXPECT_EQ(data_node->AddLinkFromForParse(attr_node), GRAPH_PARAM_INVALID);
    EXPECT_EQ(data_node->AddLinkFrom(attr_node), GRAPH_PARAM_INVALID);
    EXPECT_EQ(data_node->AddLinkFrom(2, attr_node), GRAPH_PARAM_INVALID);
    EXPECT_EQ(data_node->AddLinkFrom("Attr", attr_node), GRAPH_PARAM_INVALID);
    InDataAnchorPtr in_anch = std::make_shared<InDataAnchor>(data_node, 111);
    OutDataAnchorPtr out_anch = std::make_shared<OutDataAnchor>(data_node, 222);
    EXPECT_EQ(data_node->NodeAnchorIsEqual(nullptr, in_anch, 1), false);
    EXPECT_EQ(data_node->NodeAnchorIsEqual(in_anch, nullptr, 1), false);
    EXPECT_EQ(data_node->NodeAnchorIsEqual(in_anch, out_anch, 1), true);
    auto node3 = builder.AddNode("Data3", "Data3", 3, 3);
    InControlAnchorPtr inc_anch = std::make_shared<InControlAnchor>(node3, 33);
    EXPECT_EQ(out_anch->LinkTo(inc_anch), GRAPH_SUCCESS);
    EXPECT_EQ(data_node->NodeAnchorIsEqual(out_anch, inc_anch, 1), false);
    EXPECT_EQ(attr_node->AddLinkFrom(data_node), GRAPH_SUCCESS);
    EXPECT_EQ(attr_node->AddLinkFromForParse(data_node), GRAPH_SUCCESS);
    EXPECT_EQ(attr_node->AddLinkFrom(2, data_node), GRAPH_SUCCESS);
    EXPECT_EQ(attr_node->AddLinkFrom("Attr", data_node), GRAPH_SUCCESS);
    EXPECT_EQ(data_node->GetOutNodes().size(), 3U);
    EXPECT_EQ(data_node->GetOutNodesPtr().size(), 3U);
    EXPECT_EQ(data_node->GetOutDataNodes().size(), 3U);
    EXPECT_EQ(data_node->GetOutDataNodesSize(), 3U);
    EXPECT_EQ(attr_node->GetInNodes().size(), 3U);
    EXPECT_EQ(attr_node->GetInNodesPtr().size(), 3U);
    EXPECT_EQ(attr_node->GetInNodesSize(), 3U);
    EXPECT_EQ(attr_node->GetInDataNodesSize(), 3U);
    EXPECT_EQ(attr_node->GetInDataNodes().size(), 3U);
    EXPECT_EQ(attr_node->GetInControlNodesSize(), 0U);
    EXPECT_EQ(attr_node->GetInControlNodes().size(), 0U);
    builder.AddControlEdge(data_node, attr_node);
    EXPECT_EQ(attr_node->GetInNodes().size(), 4U);
    EXPECT_EQ(attr_node->GetInNodesPtr().size(), 4U);
    EXPECT_EQ(attr_node->GetInNodesSize(), 4U);
    EXPECT_EQ(attr_node->GetInDataNodesSize(), 3U);
    EXPECT_EQ(attr_node->GetInDataNodes().size(), 3U);
    EXPECT_EQ(attr_node->GetInControlNodesSize(), 1U);
    EXPECT_EQ(attr_node->GetInControlNodes().size(), 1U);
    EXPECT_EQ(data_node->GetOutNodes().size(), 4U);
    EXPECT_EQ(data_node->GetOutNodesPtr().size(), 4U);
    EXPECT_EQ(data_node->GetOutNodesSize(), 4U);
    EXPECT_EQ(data_node->GetOutControlNodesSize(), 1U);
    EXPECT_EQ(data_node->GetOutDataNodesSize(), 3U);
    data_node->impl_->out_data_anchors_.push_back(nullptr);
    EXPECT_EQ(data_node->GetOutNodesPtr().size(), 4U);
    EXPECT_EQ(data_node->GetOutNodes().size(), 4U);
    EXPECT_EQ(GraphUtils::RemoveNodeWithoutRelink(builder.GetGraph(), data_node), GRAPH_SUCCESS);
  }
  EXPECT_EQ(attr_node->GetInNodes().size(), 0U);
  EXPECT_EQ(attr_node->GetInNodesSize(), 0U);
  EXPECT_EQ(attr_node->GetInDataNodesSize(), 0U);
  EXPECT_EQ(attr_node->GetInDataNodes().size(), 0U);
  EXPECT_EQ(attr_node->GetInControlNodesSize(), 0U);
  EXPECT_EQ(attr_node->GetInControlNodes().size(), 0U);
}

TEST_F(UtestNode, NodeInputAndOutBarePtrCheck) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data_node = builder.AddNode("DataNode", "Data", 1, 1);
  auto attr_node = builder.AddNode("AttrNode", "Attr", 2, 2);
  data_node->GetOutControlAnchor()->LinkTo(attr_node->GetInControlAnchor());
  data_node->GetOutDataAnchor(0)->LinkTo(attr_node->GetInDataAnchor(0));
  auto graph = builder.GetGraph();

  EXPECT_EQ(data_node->GetInNodes().size(), 0U);
  EXPECT_EQ(data_node->GetInNodesSize(), 0U);
  EXPECT_EQ(data_node->GetAllInAnchors().size(), 1U);
  EXPECT_EQ(data_node->GetAllInAnchorsPtr().size(), 2U);
  EXPECT_EQ(data_node->GetOutNodes().size(), 2U);
  EXPECT_EQ(data_node->GetOutNodesSize(), 2U);
  EXPECT_EQ(data_node->GetAllOutAnchors().size(), 2U);
  EXPECT_EQ(data_node->GetAllOutAnchorsPtr().size(), 2U);
  EXPECT_EQ(data_node->GetOutDataNodes().size(), 1U);
  EXPECT_EQ(data_node->GetOutDataNodesPtr().size(), 1U);
  EXPECT_EQ(data_node->GetOutDataNodesSize(), 1U);

  EXPECT_EQ(attr_node->GetInNodes().size(), 2U);
  EXPECT_EQ(attr_node->GetInNodesSize(), 2U);
  EXPECT_EQ(attr_node->GetAllInAnchors().size(), 3U);
  EXPECT_EQ(attr_node->GetAllInAnchorsPtr().size(), 3U);
  EXPECT_EQ(attr_node->GetInDataNodes().size(), 1U);
  EXPECT_EQ(attr_node->GetInDataNodesSize(), 1U);
  EXPECT_EQ(attr_node->GetOutNodes().size(), 0U);
  EXPECT_EQ(attr_node->GetOutNodesSize(), 0U);
  EXPECT_EQ(attr_node->GetAllOutAnchors().size(), 2U);
  EXPECT_EQ(attr_node->GetAllOutAnchorsPtr().size(), 3U);
  EXPECT_EQ(attr_node->GetOutDataNodesPtr().size(), 0U);
}

TEST_F(UtestNode, GetCase) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data_node = builder.AddNode("DataNode", "Data", 1, 1);
  auto attr_node = builder.AddNode("AttrNode", "Attr", 2, 2);
  auto graph = builder.GetGraph();
  EXPECT_EQ(data_node->GetName(), "DataNode");
  EXPECT_EQ(data_node->GetType(), "Data");
  EXPECT_EQ(std::string(attr_node->GetNamePtr()), "AttrNode");
  EXPECT_EQ(std::string(attr_node->GetTypePtr()), "Attr");
  EXPECT_EQ(data_node->GetAllInAnchors().size(), 1);
  EXPECT_EQ(attr_node->GetAllOutAnchors().size(), 2);
  EXPECT_EQ(data_node->GetInNodes().size(), 0);
  EXPECT_EQ(data_node->GetInNodesPtr().size(), 0);
  EXPECT_EQ(attr_node->GetOutNodes().size(), 0);
  EXPECT_EQ(attr_node->GetOutDataNodes().size(), 0);
  EXPECT_EQ(NodeUtilsEx::InferShapeAndType(attr_node), GRAPH_PARAM_INVALID);
  EXPECT_EQ(attr_node->GetOutDataNodesAndAnchors().size(), 0);
  EXPECT_EQ(data_node->NodeInConnectsAreEqual(*attr_node), false);
  EXPECT_EQ(data_node->NodeOutConnectsAreEqual(*attr_node), false);
  EXPECT_EQ(attr_node->NodeInConnectsAreEqual(*data_node), false);
  EXPECT_EQ(attr_node->NodeOutConnectsAreEqual(*data_node), false);
  EXPECT_EQ((*data_node)==(*attr_node), false);
  std::unordered_set<Node *> us;
  us.insert(data_node.get());
  EXPECT_EQ(attr_node->IsAllInNodesSeen(us), true);
  data_node->AddSendEventId(10);
  data_node->AddRecvEventId(20);
  EXPECT_EQ(data_node->GetSendEventIdList().size(), 1);
  EXPECT_EQ(data_node->GetRecvEventIdList().size(), 1);
  kFusionDataFlowVec_t fusion_input_list;
  data_node->GetFusionInputFlowList(fusion_input_list);
  data_node->SetFusionInputFlowList(fusion_input_list);
  EXPECT_EQ(fusion_input_list.size(), 0);
  kFusionDataFlowVec_t fusion_output_list;
  data_node->GetFusionOutputFlowList(fusion_output_list);
  data_node->SetFusionOutputFlowList(fusion_output_list);
  EXPECT_EQ(fusion_output_list.size(), 0);
  EXPECT_EQ(data_node->GetHostNode(), false);
  data_node->SetOrigNode(attr_node);
  EXPECT_NE(data_node->GetOrigNode(), nullptr);
  OpDescPtr opd = std::make_shared<OpDesc>("Opdesc","OpdType");
  EXPECT_EQ(data_node->UpdateOpDesc(opd), GRAPH_PARAM_INVALID);
}

TEST_F(UtestNode, IsAllInNodesSeenSuccess) {
  auto builder = ut::GraphBuilder("graph");
  const auto &node1 = builder.AddNode("node1", "node1", 2, 2);
  const auto &node2 = builder.AddNode("node2", "node2", 1, 1);
  builder.AddDataEdge(node1, 0, node2, 0);
  builder.AddControlEdge(node1, node2);
  auto graph = builder.GetGraph();
  std::unordered_set<Node *> us;
  us.insert(node2.get());
  us.insert(node1.get());
  EXPECT_EQ(node2->IsAllInNodesSeen(us), true);
}

TEST_F(UtestNode, NodeInConnectsAreEqual) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data_node = builder.AddNode("Data", "Data", 1, 1);
  auto attr_node = builder.AddNode("Attr", "Attr", 2, 2);
  InDataAnchorPtr in_anch = std::make_shared<InDataAnchor>(data_node, 111);
  EXPECT_EQ(data_node->NodeAnchorIsEqual(nullptr, in_anch, 1), false);
  data_node->impl_->in_data_anchors_.push_back(in_anch);
  EXPECT_EQ(data_node->GetAllInDataAnchors().size(), 2);
  EXPECT_EQ(data_node->GetAllInDataAnchorsPtr().size(), 2);
  EXPECT_EQ(attr_node->GetAllInDataAnchors().size(), 2);
  EXPECT_EQ(data_node->NodeInConnectsAreEqual(*attr_node), true);
}

TEST_F(UtestNode, NodeOutConnectsAreEqual) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data_node = builder.AddNode("Data", "Data", 1, 1);
  auto attr_node = builder.AddNode("Attr", "Attr", 2, 2);
  OutDataAnchorPtr out_anch = std::make_shared<OutDataAnchor>(data_node, 111);
  EXPECT_EQ(data_node->NodeAnchorIsEqual(nullptr, out_anch, 1), false);
  data_node->impl_->out_data_anchors_.push_back(out_anch);
  EXPECT_EQ(data_node->GetAllOutDataAnchors().size(), 2);
  EXPECT_EQ(data_node->GetAllOutDataAnchorsPtr().size(), 2);
  EXPECT_EQ(attr_node->GetAllOutDataAnchors().size(), 2);
  EXPECT_EQ(attr_node->GetAllOutDataAnchorsPtr().size(), 2);
  EXPECT_EQ(data_node->NodeOutConnectsAreEqual(*attr_node), true);
}

TEST_F(UtestNode, NodeAnchorIsEqual) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data_node = builder.AddNode("Data", "Data", 1, 1);
  auto attr_node = builder.AddNode("Attr", "Attr", 2, 2);
  InDataAnchorPtr in_anch1 = std::make_shared<InDataAnchor>(data_node, 111);
  InDataAnchorPtr in_anch2 = std::make_shared<InDataAnchor>(attr_node, 222);
  OutDataAnchorPtr out_anch1 = std::make_shared<OutDataAnchor>(data_node, 333);
  EXPECT_EQ(in_anch1->LinkFrom(out_anch1), GRAPH_SUCCESS);
  EXPECT_EQ(data_node->NodeAnchorIsEqual(in_anch1, in_anch2, 2), false);
  OutDataAnchorPtr out_anch2 = std::make_shared<OutDataAnchor>(nullptr, 444);
  EXPECT_EQ(in_anch2->LinkFrom(out_anch2), GRAPH_SUCCESS);
  EXPECT_EQ(data_node->NodeAnchorIsEqual(in_anch1, in_anch2, 2), false);
}

TEST_F(UtestNode, AddLink) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data_node = builder.AddNode("Data", "Data", 1, 1);
  auto attr_node = builder.AddNode("Attr", "Attr", 2, 2);
  EXPECT_EQ(attr_node->AddLinkFrom(data_node), GRAPH_SUCCESS);
  data_node->impl_->op_->impl_->input_name_idx_["input_name"] = 10;
  data_node->impl_->op_->impl_->outputs_desc_.push_back(MakeNullptr<GeTensorDesc>());
  auto odesc = data_node->GetOpDesc()->GetOutputDesc(0);
  auto another_desc = data_node->GetOpDescBarePtr()->GetOutputDesc(0);
  EXPECT_EQ(odesc, another_desc);
  attr_node->impl_->op_->impl_->input_name_idx_["__input3"] = 20;
  EXPECT_NE(attr_node->impl_->op_->impl_->input_name_idx_.find("__input3"), attr_node->impl_->op_->impl_->input_name_idx_.end());
  EXPECT_EQ(attr_node->impl_->op_->impl_->inputs_desc_.size(), 3);
  EXPECT_EQ(attr_node->AddLinkFrom(data_node), GRAPH_FAILED);
}

TEST_F(UtestNode, AddLinkByIndex) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data_node = builder.AddNode("Data", "Data", 1, 1);
  auto attr_node = builder.AddNode("Attr", "Attr", 2, 2);
  InDataAnchorPtr in_anch = std::make_shared<InDataAnchor>(data_node, 111);
  OutDataAnchorPtr out_anch = std::make_shared<OutDataAnchor>(data_node, 222);
  EXPECT_EQ(data_node->NodeAnchorIsEqual(nullptr, in_anch, 1), false);
  EXPECT_EQ(data_node->NodeAnchorIsEqual(in_anch, nullptr, 1), false);
  EXPECT_EQ(data_node->NodeAnchorIsEqual(in_anch, out_anch, 1), true);
  auto node3 = builder.AddNode("Data3", "Data3", 3, 3);
  InControlAnchorPtr inc_anch = std::make_shared<InControlAnchor>(node3, 33);
  EXPECT_EQ(out_anch->LinkTo(inc_anch), GRAPH_SUCCESS);
  EXPECT_EQ(data_node->NodeAnchorIsEqual(out_anch, inc_anch, 1),false);
  EXPECT_EQ(attr_node->AddLinkFrom(data_node), GRAPH_SUCCESS);
  data_node->impl_->op_->impl_->input_name_idx_["input_name"] = 10;
  data_node->impl_->op_->impl_->outputs_desc_.push_back(MakeNullptr<GeTensorDesc>());
  auto odesc = data_node->GetOpDesc()->GetOutputDesc(0);
  attr_node->impl_->op_->impl_->input_name_idx_["__input3"] = 20;
  EXPECT_NE(attr_node->impl_->op_->impl_->input_name_idx_.find("__input3"), attr_node->impl_->op_->impl_->input_name_idx_.end());
  EXPECT_EQ(attr_node->impl_->op_->impl_->inputs_desc_.size(), 3);
  EXPECT_EQ(attr_node->AddLinkFrom(11, data_node), GRAPH_FAILED);
}

TEST_F(UtestNode, AddLinkByString) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data_node = builder.AddNode("Data", "Data", 1, 1);
  auto attr_node = builder.AddNode("Attr", "Attr", 2, 2);
  InDataAnchorPtr in_anch = std::make_shared<InDataAnchor>(data_node, 111);
  OutDataAnchorPtr out_anch = std::make_shared<OutDataAnchor>(data_node, 222);
  EXPECT_EQ(data_node->NodeAnchorIsEqual(nullptr, in_anch, 1), false);
  EXPECT_EQ(data_node->NodeAnchorIsEqual(in_anch, nullptr, 1), false);
  EXPECT_EQ(data_node->NodeAnchorIsEqual(in_anch, out_anch, 1), true);
  auto node3 = builder.AddNode("Data3", "Data3", 3, 3);
  InControlAnchorPtr inc_anch = std::make_shared<InControlAnchor>(node3, 33);
  EXPECT_EQ(out_anch->LinkTo(inc_anch), GRAPH_SUCCESS);
  EXPECT_EQ(data_node->NodeAnchorIsEqual(out_anch, inc_anch, 1),false);
  EXPECT_EQ(attr_node->AddLinkFrom(data_node), GRAPH_SUCCESS);
  data_node->impl_->op_->impl_->input_name_idx_["input_name"] = 10;
  data_node->impl_->op_->impl_->outputs_desc_.push_back(MakeNullptr<GeTensorDesc>());
  auto odesc = data_node->GetOpDesc()->GetOutputDesc(0);
  attr_node->impl_->op_->impl_->input_name_idx_["__input3"] = 20;
  EXPECT_NE(attr_node->impl_->op_->impl_->input_name_idx_.find("__input3"), attr_node->impl_->op_->impl_->input_name_idx_.end());
  EXPECT_EQ(attr_node->impl_->op_->impl_->inputs_desc_.size(), 3);
  EXPECT_EQ(attr_node->AddLinkFrom("__input3", data_node), GRAPH_FAILED);
  attr_node->impl_->op_->impl_->input_name_idx_["__input_succ"] = 5;
  EXPECT_EQ(attr_node->impl_->op_->impl_->inputs_desc_.size(), 3);
  EXPECT_NE(attr_node->impl_->op_->impl_->input_name_idx_.find("__input_succ"), attr_node->impl_->op_->impl_->input_name_idx_.end());
  EXPECT_EQ(attr_node->AddLinkFrom("__input_succ", data_node), GRAPH_FAILED);
}

TEST_F(UtestNode, AddLinkByStringInputDescFailure) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data_node = builder.AddNode("Data", "Data", 1, 1);
  auto attr_node = builder.AddNode("Attr", "Attr", 2, 2);
  InDataAnchorPtr in_anch = std::make_shared<InDataAnchor>(data_node, 111);
  OutDataAnchorPtr out_anch = std::make_shared<OutDataAnchor>(data_node, 222);
  EXPECT_EQ(data_node->NodeAnchorIsEqual(nullptr, in_anch, 1), false);
  EXPECT_EQ(data_node->NodeAnchorIsEqual(in_anch, nullptr, 1), false);
  EXPECT_EQ(data_node->NodeAnchorIsEqual(in_anch, out_anch, 1), true);
  auto node3 = builder.AddNode("Data3", "Data3", 3, 3);
  InControlAnchorPtr inc_anch = std::make_shared<InControlAnchor>(node3, 33);
  EXPECT_EQ(out_anch->LinkTo(inc_anch), GRAPH_SUCCESS);
  EXPECT_EQ(data_node->NodeAnchorIsEqual(out_anch, inc_anch, 1),false);
  EXPECT_EQ(attr_node->AddLinkFrom(data_node), GRAPH_SUCCESS);
  data_node->impl_->op_->impl_->input_name_idx_["input_name"] = 10;
  data_node->impl_->op_->impl_->outputs_desc_.push_back(nullptr);
  auto odesc = data_node->GetOpDesc()->GetOutputDesc(0);
  attr_node->impl_->op_->impl_->input_name_idx_["__input5"] = -1;
  auto it = attr_node->impl_->op_->impl_->input_name_idx_.find("__input5");
  EXPECT_NE(it, attr_node->impl_->op_->impl_->input_name_idx_.end());
  EXPECT_EQ(it->second, -1);
  EXPECT_EQ(attr_node->impl_->op_->impl_->inputs_desc_.size(), 3);
  EXPECT_EQ(attr_node->impl_->op_->impl_->AddInputDesc("__input5", odesc), GRAPH_FAILED);
  EXPECT_EQ(attr_node->AddLinkFrom("__input5", data_node), GRAPH_FAILED);
}

// 临时屏蔽
TEST_F(UtestNode, Verify) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data_node = builder.AddNode("Data", "Data", 1, 1);
  data_node->impl_->in_data_anchors_.push_back(nullptr);
  EXPECT_EQ(NodeUtilsEx::Verify(data_node), GRAPH_SUCCESS);
  auto node_op = ge::OperatorFactoryImpl::CreateOperator("node_op", data_node->impl_->op_->GetType());
  EXPECT_NE(OperatorFactoryImpl::operator_creators_v2_, nullptr);
  std::map<std::string, OpCreatorV2> mapv2;
  mapv2 = *OperatorFactoryImpl::operator_creators_v2_;
  mapv2["Data"] = CreateOp;
  EXPECT_EQ(data_node->impl_->op_->GetType(), "Data");
  // EXPECT_EQ(node_op.IsEmpty(), true);
  auto node_op2 = ge::OperatorFactoryImpl::CreateOperator("node_op", data_node->impl_->op_->GetType());
  // EXPECT_EQ(node_op2.IsEmpty(), true);
}

TEST_F(UtestNode, GetOutControlNodes) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data_node1 = builder.AddNode("Data1", "Data", 1, 1);
  auto data_node2 = builder.AddNode("Data2", "Data", 1, 1);
  EXPECT_EQ(data_node1->GetOutControlAnchor()->LinkTo(data_node2->GetInControlAnchor()), GRAPH_SUCCESS);
  EXPECT_EQ(data_node1->GetOutControlNodes().size(), 1);
  EXPECT_EQ(data_node1->GetOutControlNodesSize(), 1);
  EXPECT_EQ(data_node1->GetOutNodesSize(), 1);
  EXPECT_EQ(data_node1->GetOutDataAnchor(0)->LinkTo(data_node2->GetInDataAnchor(0)), GRAPH_SUCCESS);
  EXPECT_EQ(data_node1->GetOutDataNodesSize(), 1);
  EXPECT_EQ(data_node1->GetOutNodesSize(), 2);
}
}
