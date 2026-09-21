/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <string>
#include <gtest/gtest.h>

#include "common/ge_inner_error_codes.h"
#include "macro_utils/dt_public_scope.h"
#include "graph/passes/memory_conflict/hccl_continuous_memcpy_pass.h"
#include "macro_utils/dt_public_unscope.h"
#include "graph_builder_utils.h"

namespace ge {
class UtestGraphPassesHcclContinuousMemcpyPass : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};
namespace {

/*
 *         var                               var
 *         |  \                             |   \
 *         |   assign                       |   assign
 *         |   //         =======>          |   //
 *     allreduce                         identity
 *        |                                 |
 *       netoutput                        allreduce
 *                                          |
 *                                        netoutput
 */
ComputeGraphPtr BuildGraph_Allreduce_Read_Var_After_Assign(){
  auto builder = ut::GraphBuilder("test");
  auto var = builder.AddNode("var", VARIABLE, 0, 1);
  auto assign = builder.AddNode("assign", ASSIGN, 1, 1);
  auto allreduce = builder.AddNode("allreduce", HCOMALLREDUCE, 1, 1);
  auto netoutput1 = builder.AddNode("netoutput", NETOUTPUT, 1, 0);

  builder.AddDataEdge(var, 0, assign, 0);
  builder.AddDataEdge(var,0,allreduce,0);
  builder.AddControlEdge(assign, allreduce);
  return builder.GetGraph();
}

static ComputeGraphPtr BuildGraph1() {
  auto builder = ut::GraphBuilder("g1");
  auto var = builder.AddNode("var", VARIABLE, 1, 1, FORMAT_NC1HWC0, DT_FLOAT, std::vector<int64_t>({1, 1, 224, 224, 16}));
  var->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape({1, 1, 224, 224, 16}));
  auto cast1 = builder.AddNode("cast1", CAST, 1, 1, FORMAT_NC1HWC0, DT_FLOAT, std::vector<int64_t>({1, 1, 224, 224, 16}));
  cast1->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_FLOAT16);

  auto transdata1 = builder.AddNode("transdata1", "TransData", 1, 1, FORMAT_NC1HWC0, DT_FLOAT16, std::vector<int64_t>({1, 1, 224, 224, 16}));
  transdata1->GetOpDesc()->MutableOutputDesc(0)->SetFormat(FORMAT_NHWC);
  transdata1->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 224, 224, 3})));

  auto cast2 = builder.AddNode("cast2", CAST, 1, 1, FORMAT_NHWC, DT_FLOAT16, std::vector<int64_t>({1, 224, 224, 3}));
  cast2->GetOpDesc()->MutableInputDesc(0)->SetDataType(DT_FLOAT);

  auto transdata2 = builder.AddNode("transdata2", "TransData", 1, 1, FORMAT_NHWC, DT_FLOAT, std::vector<int64_t>({1, 224, 224, 3}));
  transdata2->GetOpDesc()->MutableOutputDesc(0)->SetFormat(FORMAT_NC1HWC0);
  transdata2->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 1, 224, 224, 16})));

  auto conv2d = builder.AddNode("conv2d", "Conv2D", 1, 1, FORMAT_NC1HWC0, DT_FLOAT, std::vector<int64_t>({1, 1, 224, 224, 16}));
  conv2d->GetOpDesc()->MutableInputDesc(0)->SetOriginShape(GeShape({1, 1, 224, 224, 16}));

  auto comBroadcast = builder.AddNode("HcomBroadcast", HCOMBROADCAST, 1, 1, FORMAT_NC1HWC0, DT_FLOAT, std::vector<int64_t>({1, 1, 224, 224, 16}));
  conv2d->GetOpDesc()->MutableInputDesc(0)->SetOriginShape(GeShape({1, 1, 224, 224, 16}));

  builder.AddDataEdge(var, 0, cast1, 0);
  builder.AddDataEdge(cast1, 0, transdata1, 0);
  builder.AddDataEdge(transdata1, 0, cast2, 0);
  builder.AddDataEdge(cast2, 0, transdata2, 0);
  builder.AddDataEdge(transdata2, 0, conv2d, 0);
  builder.AddDataEdge(conv2d, 0, comBroadcast, 0);

  return builder.GetGraph();
}
/*
 *          add 
 *         /   \
 *        /     \
 *  allreduce1  allreduce2  
 * 
 */
ComputeGraphPtr BuildGraphWithInputConflict() {
  auto builder = ut::GraphBuilder("g1");
  auto cast1 = builder.AddNode("cast1", CAST, 1, 1, FORMAT_NC1HWC0, DT_FLOAT, std::vector<int64_t>({1, 1, 224, 224, 16}));
  auto cast2 = builder.AddNode("cast2", CAST, 1, 1, FORMAT_NC1HWC0, DT_FLOAT, std::vector<int64_t>({1, 1, 224, 224, 16}));
  auto add = builder.AddNode("add", ADD, 1, 1, FORMAT_NC1HWC0, DT_FLOAT, std::vector<int64_t>({1, 1, 224, 224, 16}));
  auto allreduce1 = builder.AddNode("allreduce1", HCOMALLREDUCE, 2, 1, FORMAT_NC1HWC0, DT_FLOAT, std::vector<int64_t>({1, 1, 224, 224, 16}));
  auto allreduce2 = builder.AddNode("allreduce2", HCOMALLREDUCE, 2, 1, FORMAT_NC1HWC0, DT_FLOAT, std::vector<int64_t>({1, 1, 224, 224, 16}));
  (void)ge::AttrUtils::SetBool(allreduce1->GetOpDesc(), ge::ATTR_NAME_CONTINUOUS_INPUT, true);
  (void)ge::AttrUtils::SetBool(allreduce2->GetOpDesc(), ge::ATTR_NAME_CONTINUOUS_INPUT, true);

  builder.AddDataEdge(add, 0, allreduce1, 0);
  builder.AddDataEdge(cast1, 0, allreduce1, 1);
  builder.AddDataEdge(add, 0, allreduce2, 0);
  builder.AddDataEdge(cast2, 0, allreduce2, 1);
  return builder.GetGraph();
}

/*
 *              add 
 *         /     \        \
 *        /       \        \
 *  allreduce1  allreduce2  allreduce3
 * 
 */
ComputeGraphPtr BuildGraphWithInputConflict2() {
  auto builder = ut::GraphBuilder("g1");
  auto cast1 = builder.AddNode("cast1", CAST, 1, 1, FORMAT_NC1HWC0, DT_FLOAT, std::vector<int64_t>({1, 1, 224, 224, 16}));
  auto cast2 = builder.AddNode("cast2", CAST, 1, 1, FORMAT_NC1HWC0, DT_FLOAT, std::vector<int64_t>({1, 1, 224, 224, 16}));
  auto cast3 = builder.AddNode("cast3", CAST, 1, 1, FORMAT_NC1HWC0, DT_FLOAT, std::vector<int64_t>({1, 1, 224, 224, 16}));
  auto add = builder.AddNode("add", ADD, 1, 1, FORMAT_NC1HWC0, DT_FLOAT, std::vector<int64_t>({1, 1, 224, 224, 16}));
  auto allreduce1 = builder.AddNode("allreduce1", HCOMALLREDUCE, 2, 1, FORMAT_NC1HWC0, DT_FLOAT, std::vector<int64_t>({1, 1, 224, 224, 16}));
  auto allreduce2 = builder.AddNode("allreduce2", HCOMALLREDUCE, 2, 1, FORMAT_NC1HWC0, DT_FLOAT, std::vector<int64_t>({1, 1, 224, 224, 16}));
  auto allreduce3 = builder.AddNode("allreduce3", HCOMALLREDUCE, 2, 1, FORMAT_NC1HWC0, DT_FLOAT, std::vector<int64_t>({1, 1, 224, 224, 16}));
  (void)ge::AttrUtils::SetBool(allreduce1->GetOpDesc(), ge::ATTR_NAME_CONTINUOUS_INPUT, true);
  (void)ge::AttrUtils::SetBool(allreduce2->GetOpDesc(), ge::ATTR_NAME_CONTINUOUS_INPUT, true);
  (void)ge::AttrUtils::SetBool(allreduce3->GetOpDesc(), ge::ATTR_NAME_CONTINUOUS_INPUT, true);

  builder.AddDataEdge(add, 0, allreduce1, 0);
  builder.AddDataEdge(cast1, 0, allreduce1, 1);
  builder.AddDataEdge(add, 0, allreduce2, 0);
  builder.AddDataEdge(cast2, 0, allreduce2, 1);
  builder.AddDataEdge(add, 0, allreduce3, 0);
  builder.AddDataEdge(cast3, 0, allreduce3, 1);
  return builder.GetGraph();
}
}  // namespace

// const -> allreduce
// const -> Identity -> allreduce
TEST(UtestGraphPassesHcclContinuousMemcpyPass, testInsertIdentityBeforeHccl) {
  ComputeGraphPtr graph = BuildGraph_Allreduce_Read_Var_After_Assign();
  auto src_node = graph->FindNode("var");
  auto dst_node = graph->FindNode("allreduce");
  // test InsertIdentityBeforeHccl
  HcclContinuousMemcpyPass hccl_continuous_memcpy_pass;
  hccl_continuous_memcpy_pass.InsertIdentityBeforeHccl(src_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(0));

  // check
  dst_node = graph->FindNode("allreduce");
  auto in_node_before_dst_node = dst_node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();
  EXPECT_EQ(in_node_before_dst_node->GetType(), IDENTITY);
  EXPECT_EQ(in_node_before_dst_node->GetInControlNodes().size(), 1);
  EXPECT_EQ(in_node_before_dst_node->GetInControlNodes().at(0)->GetName(), "assign");
}

// const -> allreduce
// const -> Identity -> allreduce
TEST(UtestGraphPassesHcclContinuousMemcpyPass, testInsertIdentityBeforeHccl2) {
  ComputeGraphPtr graph = BuildGraph_Allreduce_Read_Var_After_Assign();
  auto src_node = graph->FindNode("var");
  auto dst_node = graph->FindNode("allreduce");
  // test InsertIdentityBeforeHccl
  HcclContinuousMemcpyPass hccl_continuous_memcpy_pass;
  hccl_continuous_memcpy_pass.InsertIdentityBeforeHccl(src_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(0));

  // check
  dst_node = graph->FindNode("allreduce");
  auto in_node_before_dst_node = dst_node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();
  EXPECT_EQ(in_node_before_dst_node->GetType(), IDENTITY);
  EXPECT_EQ(in_node_before_dst_node->GetInControlNodes().size(), 1);
  EXPECT_EQ(in_node_before_dst_node->GetInControlNodes().at(0)->GetName(), "assign");
}

TEST(UtestGraphPassesHcclContinuousMemcpyPass, testModifyEdgeConnection) {
  Status ret;
  HcclContinuousMemcpyPass hccl_continuous_memcpy_pass;
  ComputeGraphPtr graph = BuildGraph_Allreduce_Read_Var_After_Assign();
  auto src_out_anchor = graph->FindNode("var");
  auto hccl_in_anchor = graph->FindNode("allreduce");
  
  ret = hccl_continuous_memcpy_pass.ModifyEdgeConnection(graph, src_out_anchor->GetOutDataAnchor(0), hccl_in_anchor->GetInDataAnchor(0));
  EXPECT_EQ(ret, SUCCESS);
}

TEST(UtestGraphPassesHcclContinuousMemcpyPass, testCreateAssignNode) {
  NodePtr ret;
  HcclContinuousMemcpyPass hccl_continuous_memcpy_pass;
  ComputeGraphPtr graph = BuildGraph_Allreduce_Read_Var_After_Assign();
  auto src_out_anchor = graph->FindNode("var");
  ret = hccl_continuous_memcpy_pass.CreateAssignNode(graph, src_out_anchor->GetOutDataAnchor(0));
  EXPECT_NE(ret, nullptr);

  Status iRet;
  iRet = hccl_continuous_memcpy_pass.ClearStatus();
  EXPECT_EQ(iRet, SUCCESS);

  bool bRet;
  bRet = hccl_continuous_memcpy_pass.IsDataNode(CONSTANTOP);
  EXPECT_EQ(bRet, true);

  hccl_continuous_memcpy_pass.node_num_map_.insert(std::make_pair("ge", 1));
  std::string sRet= hccl_continuous_memcpy_pass.CheckDuplicateName("ge");
  EXPECT_EQ(sRet, "ge_1");
}

TEST(UtestGraphPassesHcclContinuousMemcpyPass, testInsertAssignAfterBroadcastIfNeed) {
  Status ret;
  HcclContinuousMemcpyPass hccl_continuous_memcpy_pass;
  ComputeGraphPtr graph = BuildGraph1();
  ge::OpDescPtr op = std::make_shared<ge::OpDesc>("HcomBroadcast", "HcomBroadcast");
  ge::NodePtr node = std::make_shared<ge::Node>(op, graph);
  auto src_out_anchor = graph->FindNode("HcomBroadcast");
  auto hccl_in_anchor = graph->FindNode("HcomBroadcast");

  ret = hccl_continuous_memcpy_pass.InsertAssignAfterBroadcastIfNeed(graph, src_out_anchor->GetOutDataAnchor(0), hccl_in_anchor->GetInDataAnchor(0));
  EXPECT_EQ(ret, SUCCESS);

  auto src_out_anchor_var = graph->FindNode("var");
  ret = hccl_continuous_memcpy_pass.InsertAssignAfterBroadcastIfNeed(graph, src_out_anchor_var->GetOutDataAnchor(0), hccl_in_anchor->GetInDataAnchor(0));
  EXPECT_EQ(ret, SUCCESS);
}

TEST(UtestGraphPassesHcclContinuousMemcpyPass, testContinuousInputProcess) {
  Status ret;
  HcclContinuousMemcpyPass hccl_continuous_memcpy_pass;
  ComputeGraphPtr graph = BuildGraph_Allreduce_Read_Var_After_Assign();
  NodePtr node = graph->FindNode("var");
  ret = hccl_continuous_memcpy_pass.ContinuousInputProcess(graph, node);
  EXPECT_EQ(ret, SUCCESS);
  
  ComputeGraphPtr graph1 = std::make_shared<ComputeGraph>("graph");
  OpDescPtr in_op_ptr_1 = std::make_shared<OpDesc>("in_op_1", "Variable");
  in_op_ptr_1->AddInputDesc("x", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  in_op_ptr_1->AddOutputDesc("y", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  NodePtr in_owner_node_1 = graph1->AddNode(in_op_ptr_1);
  InDataAnchorPtr in_data_anchor = in_owner_node_1->GetInDataAnchor(0);

  OpDescPtr in_op_ptr_2 = std::make_shared<OpDesc>("in_op_2", "Variable");
  in_op_ptr_2->AddInputDesc("x", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  in_op_ptr_2->AddInputDesc("y", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  in_op_ptr_2->AddOutputDesc("z", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  NodePtr in_owner_node_2 = graph1->AddNode(in_op_ptr_2);
  InDataAnchorPtr in_data_anchor_x = in_owner_node_2->GetInDataAnchor(0);
  InDataAnchorPtr in_data_anchor_y = in_owner_node_2->GetInDataAnchor(1);
  InControlAnchorPtr in_control_anchor = in_owner_node_2->GetInControlAnchor();

  OpDescPtr out_op_ptr_1 = std::make_shared<OpDesc>("out_op_1", "Variable");
  out_op_ptr_1->AddInputDesc("x", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  out_op_ptr_1->AddOutputDesc("y", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  NodePtr out_owner_node_1 = graph1->AddNode(out_op_ptr_1);
  OutDataAnchorPtr out_data_anchor_1 = out_owner_node_1->GetOutDataAnchor(0);

  OpDescPtr out_op_ptr_2 = std::make_shared<OpDesc>("out_op_2", "Variable");
  out_op_ptr_2->AddInputDesc("x", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  out_op_ptr_2->AddOutputDesc("y", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  NodePtr out_owner_node_2 = graph1->AddNode(out_op_ptr_2);
  OutDataAnchorPtr out_data_anchor_2 = out_owner_node_2->GetOutDataAnchor(0);

  EXPECT_EQ((in_data_anchor->LinkFrom(out_data_anchor_1)), GRAPH_SUCCESS);
  EXPECT_EQ(out_data_anchor_1->LinkTo(in_data_anchor_x), GRAPH_SUCCESS);
  EXPECT_EQ(in_data_anchor_y->LinkFrom(out_data_anchor_2), GRAPH_SUCCESS);
  EXPECT_EQ(out_data_anchor_2->LinkTo(in_control_anchor), GRAPH_SUCCESS);
  EXPECT_EQ(in_control_anchor->GetPeerOutDataAnchors().size(), 1);
  EXPECT_EQ(out_data_anchor_2->GetPeerAnchors().size(), 2);
  EXPECT_EQ(out_data_anchor_2->GetPeerInDataAnchors().size(), 1);
  EXPECT_EQ(out_data_anchor_2->GetPeerInControlAnchors().size(), 1);
  EXPECT_EQ(out_data_anchor_1->GetPeerAnchors().size(), 2);

  ge::AttrUtils::SetBool(in_owner_node_2->GetOpDesc(), ATTR_NAME_CONTINUOUS_INPUT, true);
  GeTensorDesc te_desc1(GeShape({1, 2, 3, 4}), FORMAT_NCHW, DT_FLOAT);
  in_owner_node_2->GetOpDesc()->AddInputDesc(te_desc1);
  GeTensorDesc te_desc2(GeShape({4, 5, 6, 7}), FORMAT_NCHW, DT_FLOAT);
  in_owner_node_2->GetOpDesc()->AddInputDesc("w", te_desc2);
  GeTensorDesc te_desc3(GeShape({8, 9, 10, 11}), FORMAT_NCHW, DT_FLOAT);
  in_owner_node_2->GetOpDesc()->AddInputDesc("w", te_desc3);

  ret = hccl_continuous_memcpy_pass.ContinuousInputProcess(graph1, in_owner_node_2);
  EXPECT_EQ(ret, SUCCESS);
}

TEST(UtestGraphPassesHcclContinuousMemcpyPass, testP2pmemInputProcess) {
  Status ret;
  HcclContinuousMemcpyPass hccl_continuous_memcpy_pass;
  ComputeGraphPtr graph = BuildGraph_Allreduce_Read_Var_After_Assign();
  NodePtr node = graph->FindNode("var");
  ret = hccl_continuous_memcpy_pass.P2pmemInputProcess(graph, node);
  EXPECT_EQ(ret, SUCCESS);

  const auto &root_graph = std::make_shared<ComputeGraph>("test_update_active_streams_for_subgraph_root_graph");
  EXPECT_NE(root_graph, nullptr);
  root_graph->SetGraphUnknownFlag(false);
  const auto &sub_graph1 = std::make_shared<ComputeGraph>("test_update_active_streams_for_subgraph_sub_graph1");
  EXPECT_NE(sub_graph1, nullptr);
  root_graph->AddSubGraph(sub_graph1);
  const auto &sub_graph2 = std::make_shared<ComputeGraph>("test_update_active_streams_for_subgraph_sub_graph2");
  EXPECT_NE(sub_graph2, nullptr);
  root_graph->AddSubGraph(sub_graph2);
  const auto &sub_graph3 = std::make_shared<ComputeGraph>("test_update_active_streams_for_subgraph_sub_graph3");
  EXPECT_NE(sub_graph3, nullptr);
  root_graph->AddSubGraph(sub_graph3);

  const auto &case_desc = std::make_shared<OpDesc>("case", CASE);
  EXPECT_NE(case_desc, nullptr);
  EXPECT_EQ(case_desc->AddInputDesc(GeTensorDesc()), GRAPH_SUCCESS);
  EXPECT_EQ(case_desc->AddOutputDesc(GeTensorDesc()), GRAPH_SUCCESS);
  case_desc->AddSubgraphName("branch1");
  case_desc->SetSubgraphInstanceName(0, "test_update_active_streams_for_subgraph_sub_graph1");
  case_desc->AddSubgraphName("branch2");
  case_desc->SetSubgraphInstanceName(1, "test_update_active_streams_for_subgraph_sub_graph2");
  case_desc->AddSubgraphName("branch3");
  case_desc->SetSubgraphInstanceName(2, "test_update_active_streams_for_subgraph_sub_graph3");
  const auto &case_node = root_graph->AddNode(case_desc);
  EXPECT_NE(case_node, nullptr);
  sub_graph1->SetParentNode(case_node);
  sub_graph2->SetParentNode(case_node);
  sub_graph3->SetParentNode(case_node);

  const auto &active_desc1 = std::make_shared<OpDesc>("active1", STREAMACTIVE);
  EXPECT_NE(active_desc1, nullptr);
  active_desc1->SetStreamId(2);
  EXPECT_TRUE(AttrUtils::SetListInt(active_desc1, ATTR_NAME_INPUT_MEM_TYPE_LIST, {0x11}));
  const auto &active_node1 = sub_graph1->AddNode(active_desc1);
  
  ret = hccl_continuous_memcpy_pass.P2pmemInputProcess(root_graph, active_node1);
  EXPECT_EQ(ret, SUCCESS);
}

TEST(UtestGraphPassesHcclContinuousMemcpyPass, testRun) {
  Status ret;
  HcclContinuousMemcpyPass hccl_continuous_memcpy_pass;
  ComputeGraphPtr graph = BuildGraph_Allreduce_Read_Var_After_Assign();
  ret = hccl_continuous_memcpy_pass.Run(graph);
  EXPECT_EQ(ret, SUCCESS);
}

TEST(UtestGraphPassesHcclContinuousMemcpyPass, testContinuousInputProcess1) {
  ComputeGraphPtr graph_ptr = std::make_shared<ComputeGraph>("graph");
  OpDescPtr in_op_ptr = std::make_shared<OpDesc>("in_op_1", "float");
  in_op_ptr->AddInputDesc("x", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  in_op_ptr->AddOutputDesc("y", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  NodePtr in_owner_node = graph_ptr->AddNode(in_op_ptr);
  InDataAnchorPtr in_data_anchor = in_owner_node->GetInDataAnchor(0);

  OpDescPtr out_op_ptr_1 = std::make_shared<OpDesc>("out_op_1", "float");
  out_op_ptr_1->AddInputDesc("x", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  out_op_ptr_1->AddOutputDesc("y", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  NodePtr out_owner_node_1 = graph_ptr->AddNode(out_op_ptr_1);
  OutDataAnchorPtr out_data_anchor_1 = out_owner_node_1->GetOutDataAnchor(0);

  OpDescPtr out_op_ptr_2 = std::make_shared<OpDesc>("out_op_2", "float");
  out_op_ptr_2->AddInputDesc("x", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  out_op_ptr_2->AddOutputDesc("y", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  NodePtr out_owner_node_2 = graph_ptr->AddNode(out_op_ptr_2);

  OutDataAnchorPtr out_data_anchor_2 = out_owner_node_2->GetOutDataAnchor(0);
  EXPECT_EQ(in_data_anchor->LinkFrom(nullptr), GRAPH_FAILED);
  EXPECT_EQ(out_data_anchor_2->LinkTo(InDataAnchorPtr(nullptr)), GRAPH_FAILED);
  EXPECT_EQ(out_data_anchor_2->LinkTo(InControlAnchorPtr(nullptr)), GRAPH_FAILED);
  EXPECT_EQ(in_data_anchor->Unlink(nullptr), GRAPH_FAILED);
  in_data_anchor->LinkFrom(out_data_anchor_1);
  EXPECT_EQ(out_data_anchor_2->LinkTo(in_data_anchor), GRAPH_FAILED);
  EXPECT_EQ(in_data_anchor->LinkFrom(out_data_anchor_2), GRAPH_FAILED);
  EXPECT_EQ(in_data_anchor->Unlink(out_data_anchor_2), GRAPH_FAILED);
  in_data_anchor->Unlink(out_data_anchor_1);
  
  ge::AttrUtils::SetBool(in_owner_node->GetOpDesc(), ATTR_NAME_CONTINUOUS_INPUT, true);
  GeTensorDesc te_desc1(GeShape({1, 2, 3, 4}), FORMAT_NCHW, DT_FLOAT);
  in_owner_node->GetOpDesc()->AddInputDesc(te_desc1);
  GeTensorDesc te_desc2(GeShape({4, 5, 6, 7}), FORMAT_NCHW, DT_FLOAT);
  in_owner_node->GetOpDesc()->AddInputDesc("w", te_desc2);
  GeTensorDesc te_desc3(GeShape({8, 9, 10, 11}), FORMAT_NCHW, DT_FLOAT);
  in_owner_node->GetOpDesc()->AddInputDesc("w", te_desc3);

  ge::AttrUtils::SetBool(in_owner_node->GetOpDesc(), ATTR_NAME_CONTINUOUS_INPUT, true);
  HcclContinuousMemcpyPass hccl_continuous_memcpy_pass;
  Status ret = hccl_continuous_memcpy_pass.ContinuousInputProcess(graph_ptr, in_owner_node);
  EXPECT_EQ(ret, INTERNAL_ERROR);
  
  ret = hccl_continuous_memcpy_pass.Run(graph_ptr);
}

TEST(UtestGraphPassesHcclContinuousMemcpyPass, testP2pmemInputProcess1) {
  HcclContinuousMemcpyPass hccl_continuous_memcpy_pass;
  // const auto &sub_graph1 = std::make_shared<ComputeGraph>("test_update_active_streams_for_subgraph_sub_graph1");
  const auto &sub_graph1 = BuildGraph1();
  const auto &op_desc = std::make_shared<OpDesc>("active1", STREAMACTIVE);
  EXPECT_NE(op_desc, nullptr);
  op_desc->SetStreamId(2);
  EXPECT_TRUE(AttrUtils::SetListInt(op_desc, ATTR_NAME_INPUT_MEM_TYPE_LIST, {0x11}));
  const auto &active_node1 = sub_graph1->AddNode(op_desc);

  string name = "Conv2d";
  string type = "Data";
  GeTensorDesc te_desc1(GeShape({1, 2, 3, 4}), FORMAT_NCHW, DT_FLOAT);
  EXPECT_EQ(GRAPH_SUCCESS, op_desc->AddInputDesc(te_desc1));
  GeTensorDesc te_desc2(GeShape({4, 5, 6, 7}), FORMAT_NCHW, DT_FLOAT);
  EXPECT_EQ(GRAPH_SUCCESS, op_desc->AddInputDesc("w", te_desc2));
  GeTensorDesc te_desc3(GeShape({8, 9, 10, 11}), FORMAT_NCHW, DT_FLOAT);
  EXPECT_EQ(GRAPH_SUCCESS, op_desc->AddInputDesc("w", te_desc3));
  EXPECT_EQ(GRAPH_SUCCESS, op_desc->AddInputDesc(1, te_desc3));
  EXPECT_EQ(GRAPH_SUCCESS, op_desc->AddInputDesc(2, te_desc3));

  GeTensorDesc te_desc4(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_FLOAT);
  EXPECT_EQ(op_desc->UpdateInputDesc(1, te_desc4), GRAPH_SUCCESS);
  EXPECT_EQ(op_desc->UpdateInputDesc(4, te_desc4), GRAPH_FAILED);
  EXPECT_EQ(op_desc->UpdateInputDesc("w", te_desc4), GRAPH_SUCCESS);
  EXPECT_EQ(op_desc->UpdateInputDesc("weight", te_desc4), GRAPH_FAILED);

  active_node1->GetOpDesc()->AddInputDesc(te_desc1);
  (void)hccl_continuous_memcpy_pass.P2pmemInputProcess(sub_graph1, active_node1);
}
/**
 *  若一个算子的同一个输出anchor，同时连给了两个要求输入连续的算子上
 *  这样就产生了冲突
 *  需要在其中一条边插入identity
 * 
 *
 *             add 
 * cast1      /   \     cast2
 *   \       /     \    /
 *  allreduce1  allreduce2  
 */
TEST(UtestGraphPassesHcclContinuousMemcpyPass, testInputContinuousConflict) {
  auto graph = BuildGraphWithInputConflict();
  EXPECT_EQ(graph->GetDirectNodesSize(), 5);

  HcclContinuousMemcpyPass hccl_continuous_memcpy_pass; 
  auto ret = hccl_continuous_memcpy_pass.Run(graph);
  EXPECT_EQ(ret, SUCCESS);
  // check has 1 identity between add and allreduce
  EXPECT_EQ(graph->GetDirectNodesSize(), 6);
  auto identity = graph->FindFirstNodeMatchType(IDENTITY);
  EXPECT_EQ(identity->GetInDataNodes().at(0)->GetType(), ADD);
  EXPECT_EQ(identity->GetOutDataNodes().at(0)->GetType(), HCOMALLREDUCE);
}

TEST(UtestGraphPassesHcclContinuousMemcpyPass, testInputContinuousConflict_multi_allreduce) {
  auto graph = BuildGraphWithInputConflict2();
  EXPECT_EQ(graph->GetDirectNodesSize(), 7);

  HcclContinuousMemcpyPass hccl_continuous_memcpy_pass; 
  auto ret = hccl_continuous_memcpy_pass.Run(graph);
  EXPECT_EQ(ret, SUCCESS);
  // check has 1 identity between add and allreduce
  EXPECT_EQ(graph->GetDirectNodesSize(), 9);
  std::vector<NodePtr> identity_nodes;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == IDENTITY) {
      identity_nodes.emplace_back(node);
    }
  }
  EXPECT_EQ(identity_nodes.size(), 2);
  for (const auto &identity : identity_nodes) {
    EXPECT_EQ(identity->GetInDataNodes().at(0)->GetType(), ADD);
    EXPECT_EQ(identity->GetOutDataNodes().at(0)->GetType(), HCOMALLREDUCE);
  }
}
}  // namespace ge
