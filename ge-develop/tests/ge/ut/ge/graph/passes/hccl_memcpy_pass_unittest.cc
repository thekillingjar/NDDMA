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
#include "graph/passes/memory_conflict/hccl_memcpy_pass.h"
#include "graph/passes/pass_manager.h"
#include "macro_utils/dt_public_unscope.h"
#include "graph_builder_utils.h"
#include "graph/optimize/graph_optimize.h"

namespace ge {
class UtestGraphPassesHcclMemcpyPass : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}

  /**
   *         input
   *      /       \
   *   allreduce   |
   *      |   \c   |
   *      |      relu
   *      \       /
   *         add
   */
  void make_graph(ComputeGraphPtr graph, bool match = true) {
    GeTensorDesc scalarTensorDesc(GeShape({7, 7, 3, 1}));

    auto input_op = std::make_shared<OpDesc>("input", DATA);
    input_op->AddOutputDesc(scalarTensorDesc);
    auto input_node = graph->AddNode(input_op);

    auto allreduce_op = std::make_shared<OpDesc>("allreduce_op", HCOMALLREDUCE);
    (void)ge::AttrUtils::SetBool(allreduce_op, "_input_mutable", true);
    allreduce_op->AddInputDesc(scalarTensorDesc);
    allreduce_op->AddOutputDesc(scalarTensorDesc);
    auto allreduce_node = graph->AddNode(allreduce_op);

    auto relu_op = std::make_shared<OpDesc>("relu_op", RELU);
    relu_op->AddInputDesc(scalarTensorDesc);
    relu_op->AddOutputDesc(scalarTensorDesc);
    auto relu_node = graph->AddNode(relu_op);

    auto add_op = std::make_shared<OpDesc>("add_op", ADD);
    add_op->AddInputDesc(scalarTensorDesc);
    add_op->AddInputDesc(scalarTensorDesc);
    add_op->AddOutputDesc(scalarTensorDesc);
    auto add_node = graph->AddNode(add_op);

    (void) GraphUtils::AddEdge(input_node->GetOutDataAnchor(0), allreduce_node->GetInDataAnchor(0));
    (void) GraphUtils::AddEdge(input_node->GetOutDataAnchor(0), relu_node->GetInDataAnchor(0));

    (void) GraphUtils::AddEdge(allreduce_node->GetOutDataAnchor(0), add_node->GetInDataAnchor(0));
    (void) GraphUtils::AddEdge(relu_node->GetOutDataAnchor(0), add_node->GetInDataAnchor(1));
    (void) GraphUtils::AddEdge(allreduce_node->GetOutControlAnchor(), relu_node->GetInControlAnchor());
  }


  void make_sub_graph(ComputeGraphPtr graph, bool match = true) {
    GeTensorDesc scalarTensorDesc(GeShape({7, 7, 3, 1}));

    auto input_op = std::make_shared<OpDesc>("input", CONSTANTOP);
    input_op->AddOutputDesc(scalarTensorDesc);
    auto input_node = graph->AddNode(input_op);

    auto allreduce_op = std::make_shared<OpDesc>("allreduce_op", HCOMALLREDUCE);
    (void)ge::AttrUtils::SetBool(allreduce_op, "_input_mutable", true);
    allreduce_op->AddInputDesc(scalarTensorDesc);
    allreduce_op->AddOutputDesc(scalarTensorDesc);
    auto allreduce_node = graph->AddNode(allreduce_op);

    auto relu_op = std::make_shared<OpDesc>("relu_op", RELU);
    relu_op->AddInputDesc(scalarTensorDesc);
    relu_op->AddOutputDesc(scalarTensorDesc);
    auto relu_node = graph->AddNode(relu_op);

    (void) GraphUtils::AddEdge(input_node->GetOutDataAnchor(0), allreduce_node->GetInDataAnchor(0));
    (void) GraphUtils::AddEdge(allreduce_node->GetOutDataAnchor(0), relu_node->GetInDataAnchor(0));
  }
};
namespace {

/*
 *        Data
 *        |  \
 *        |   \
 *        |    \
 *  allreduce1  allreduce2
 *        |  /
 *        |
 *       netoutput
 */
ComputeGraphPtr BuildGraph_Two_Allreduce_Read_OneOutput(){
  auto builder = ut::GraphBuilder("test");
  auto data = builder.AddNode("data", DATA, 0, 1);
  auto allreduce1 = builder.AddNode("allreduce1", HCOMALLREDUCE, 1, 1);
  auto allreduce2 = builder.AddNode("allreduce2", HCOMALLREDUCE, 1, 1);
  auto netoutput1 = builder.AddNode("netoutput", NETOUTPUT, 2, 0);

  (void)ge::AttrUtils::SetBool(allreduce1->GetOpDesc(), "_input_mutable", true);
  (void)ge::AttrUtils::SetBool(allreduce2->GetOpDesc(), "_input_mutable", true);

  builder.AddDataEdge(data, 0, allreduce1, 0);
  builder.AddDataEdge(data, 0, allreduce2, 0);
  builder.AddDataEdge(allreduce1, 0, netoutput1, 0);
  builder.AddDataEdge(allreduce2, 0, netoutput1, 1);
  return builder.GetGraph();
}

/*
 *        Data
 *        |  \
 *        |   relu
 *        |  c/ \
 *  allreduce1  |
 *        |     |
 *        |    /
 *       netoutput
 */
ComputeGraphPtr BuildGraph_Allreduce_Noneed_Insert(){
  auto builder = ut::GraphBuilder("test");
  auto data = builder.AddNode("data", DATA, 0, 1);
  auto allreduce1 = builder.AddNode("allreduce1", HCOMALLREDUCE, 1, 1);
  auto relu = builder.AddNode("relu", RELU, 1, 1);
  auto netoutput1 = builder.AddNode("netoutput", NETOUTPUT, 2, 0);

  (void)ge::AttrUtils::SetBool(allreduce1->GetOpDesc(), "_input_mutable", true);

  builder.AddDataEdge(data, 0, allreduce1, 0);
  builder.AddDataEdge(data, 0, relu, 0);
  builder.AddDataEdge(allreduce1, 0, netoutput1, 0);
  builder.AddDataEdge(relu, 0, netoutput1, 1);
  builder.AddControlEdge(relu, allreduce1);
  return builder.GetGraph();
}

/*
 *        Data
 *        |  \
 *        |   reshape
 *        |      |
 *  allreduce1  shape
 *        |      |
 *        |     /
 *       netoutput
 */
ComputeGraphPtr BuildGraph_Allreduce_Noneed_Insert_Shape_Sibling(){
  auto builder = ut::GraphBuilder("test");
  auto data = builder.AddNode("data", DATA, 0, 1);
  auto allreduce1 = builder.AddNode("allreduce1", HCOMALLREDUCE, 1, 1);
  auto reshape = builder.AddNode("reshape", RESHAPE, 1, 1);
  auto shape = builder.AddNode("shape", SHAPE, 1, 1);
  auto netoutput1 = builder.AddNode("netoutput", NETOUTPUT, 2, 0);

  (void)ge::AttrUtils::SetBool(allreduce1->GetOpDesc(), "_input_mutable", true);

  builder.AddDataEdge(data, 0, reshape, 0);
  builder.AddDataEdge(data, 0, allreduce1, 0);
  builder.AddDataEdge(allreduce1, 0, netoutput1, 0);
  builder.AddDataEdge(reshape, 0, shape, 0);
  builder.AddDataEdge(shape, 0, netoutput1, 1);
  return builder.GetGraph();
}

/*
 *       var                              var
 *        |  \                             |   \
 *        |   assign                       |   assign
 *        |   //         =======>          |   //
 *      allreduce                        IDENTITY
 *        |                                |
 *    netoutput                         allreduce
 *                                         |
 *                                      netoutput
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
}  // namespace

// const -> allreduce
// const -> Identity -> allreduce
TEST_F(UtestGraphPassesHcclMemcpyPass, testInsertIdentityBeforeHccl) {
  ComputeGraphPtr graph = BuildGraph_Allreduce_Read_Var_After_Assign();
  auto src_node = graph->FindNode("var");
  auto dst_node = graph->FindNode("allreduce");
  // test InsertIdentityBeforeHccl
  HcclMemcpyPass hccl_memcpy_pass;
  hccl_memcpy_pass.InsertIdentityBeforeHccl(src_node->GetOutDataAnchor(0),
                                            dst_node->GetInDataAnchor(0));

  // check
  dst_node = graph->FindNode("allreduce");
  auto in_node_before_dst_node = dst_node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();
  EXPECT_EQ(in_node_before_dst_node->GetType(), IDENTITY);
  EXPECT_EQ(in_node_before_dst_node->GetInControlNodes().size(), 1);
  EXPECT_EQ(in_node_before_dst_node->GetInControlNodes().at(0)->GetName(), "assign");
}

TEST_F(UtestGraphPassesHcclMemcpyPass, run_empty_graph_failed)
{
  ComputeGraphPtr graph = nullptr;

  HcclMemcpyPass hcclMemcpyPass;
  std::vector<std::pair<string, GraphPass*>> passes;
  passes.emplace_back("", &hcclMemcpyPass);
  EXPECT_EQ(hcclMemcpyPass.Run(graph), ge::PARAM_INVALID);
}

TEST_F(UtestGraphPassesHcclMemcpyPass, run_success)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  make_graph(graph);

  HcclMemcpyPass hcclMemcpyPass;
  std::vector<std::pair<string, GraphPass*>> passes;
  passes.emplace_back("", &hcclMemcpyPass);
  EXPECT_EQ(hcclMemcpyPass.Run(graph), ge::SUCCESS);

  bool has_memcpy_async = false;
  std::string node_type;
  for (ge::NodePtr node : graph->GetDirectNode()) {
    OpDescPtr tmp_desc = node->GetOpDesc();
    node_type = tmp_desc->GetType();
    if (node_type == IDENTITY) {
      has_memcpy_async = true;
      break;
    }
  }

  EXPECT_TRUE(has_memcpy_async == true);
}

TEST_F(UtestGraphPassesHcclMemcpyPass, run_sub_graph_success) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  make_sub_graph(graph);

  HcclMemcpyPass hcclMemcpyPass;
  std::vector<std::pair<string, GraphPass*>> passes;
  passes.emplace_back("", &hcclMemcpyPass);
  EXPECT_EQ(hcclMemcpyPass.Run(graph), ge::SUCCESS);

  bool has_memcpy_async = false;
  std::string node_type;
  for (const ge::NodePtr& node : graph->GetDirectNode()) {
    OpDescPtr tmp_desc = node->GetOpDesc();
    node_type = tmp_desc->GetType();
    if (node_type == IDENTITY) {
      has_memcpy_async = true;
      break;
    }
  }

  EXPECT_TRUE(has_memcpy_async == true);
}

TEST_F(UtestGraphPassesHcclMemcpyPass, create_memcpy_node_success)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  make_graph(graph);
  HcclMemcpyPass hcclMemcpyPass;
  std::vector<std::pair<string, GraphPass*>> passes;
  passes.emplace_back("", &hcclMemcpyPass);
  std::shared_ptr<ge::Node> null_Node;

  EXPECT_EQ(hcclMemcpyPass.Run(graph), SUCCESS);
}

TEST_F(UtestGraphPassesHcclMemcpyPass, ClearStatus)
{
  HcclMemcpyPass hcclMemcpyPass;
  Status ret = hcclMemcpyPass.ClearStatus();
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestGraphPassesHcclMemcpyPass, Run_test)
{
  HcclMemcpyPass hcclMemcpyPass;
  GeTensorDesc scalarTensorDesc(GeShape({7, 7, 3, 1}));
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  auto hcomBroadcast_op = std::make_shared<OpDesc>("HcomBroadcast", HCOMBROADCAST);
  (void)ge::AttrUtils::SetBool(hcomBroadcast_op, "_input_mutable", true);
  hcomBroadcast_op->AddInputDesc(scalarTensorDesc);
  hcomBroadcast_op->AddInputDesc(scalarTensorDesc);
  hcomBroadcast_op->AddOutputDesc(scalarTensorDesc);
  auto hcomBroadcast_node = graph->AddNode(hcomBroadcast_op);
  Status ret = hcclMemcpyPass.Run(graph);
  EXPECT_EQ(ret, SUCCESS);

  auto input_op = std::make_shared<OpDesc>("input", DATA);
  input_op->AddOutputDesc(scalarTensorDesc);
  auto input_node = graph->AddNode(input_op);

  auto add_op = std::make_shared<OpDesc>("add_op", ADD);
  add_op->AddInputDesc(scalarTensorDesc);
  add_op->AddOutputDesc(scalarTensorDesc);
  auto add_node = graph->AddNode(add_op);
  (void) GraphUtils::AddEdge(input_node->GetOutDataAnchor(0), hcomBroadcast_node->GetInDataAnchor(0));
  (void) GraphUtils::AddEdge(hcomBroadcast_node->GetOutDataAnchor(0), add_node->GetInDataAnchor(0));
  ret = hcclMemcpyPass.Run(graph);
  EXPECT_EQ(ret, PARAM_INVALID);

  auto variable_op = std::make_shared<OpDesc>("Variable", VARIABLE);
  variable_op->AddOutputDesc(scalarTensorDesc);
  variable_op->AddInputDesc(scalarTensorDesc);
  auto variable_node = graph->AddNode(variable_op);

  (void) GraphUtils::AddEdge(variable_node->GetOutDataAnchor(0), hcomBroadcast_node->GetInDataAnchor(1));
  ret = hcclMemcpyPass.Run(graph);
  EXPECT_EQ(ret, PARAM_INVALID);
}

TEST_F(UtestGraphPassesHcclMemcpyPass, Insert_assign_test)
{
  HcclMemcpyPass hcclMemcpyPass;
  GeTensorDesc scalarTensorDesc(GeShape({7, 7, 3, 1}));
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  auto hcomBroadcast_op = std::make_shared<OpDesc>("HcomBroadcast", HCOMBROADCAST);
  (void)ge::AttrUtils::SetBool(hcomBroadcast_op, "_input_mutable", true);
  hcomBroadcast_op->AddInputDesc(scalarTensorDesc);
  hcomBroadcast_op->AddInputDesc(scalarTensorDesc);
  hcomBroadcast_op->AddOutputDesc(scalarTensorDesc);
  auto hcomBroadcast_node = graph->AddNode(hcomBroadcast_op);

  auto variable_op = std::make_shared<OpDesc>("Variable", VARIABLE);
  variable_op->AddOutputDesc(scalarTensorDesc);
  variable_op->AddInputDesc(scalarTensorDesc);
  auto variable_node = graph->AddNode(variable_op);

  (void) GraphUtils::AddEdge(variable_node->GetOutDataAnchor(0), hcomBroadcast_node->GetInDataAnchor(0));
  auto ret = hcclMemcpyPass.InsertAssignAfterBroadcastIfNeed(graph, variable_node->GetOutDataAnchor(0), hcomBroadcast_node->GetInDataAnchor(0));
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestGraphPassesHcclMemcpyPass, Avoid_too_many_identity)
{
  HcclMemcpyPass hcclMemcpyPass;
  ComputeGraphPtr graph = BuildGraph_Two_Allreduce_Read_OneOutput();
  EXPECT_EQ(graph->GetDirectNodesSize(), 4);

  EXPECT_EQ(hcclMemcpyPass.Run(graph), SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(), 5);
}

TEST_F(UtestGraphPassesHcclMemcpyPass, Noneed_insert_identity)
{
  HcclMemcpyPass hcclMemcpyPass;
  ComputeGraphPtr graph = BuildGraph_Allreduce_Noneed_Insert();
  EXPECT_EQ(graph->GetDirectNodesSize(), 4);

  EXPECT_EQ(hcclMemcpyPass.Run(graph), SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(), 4);
}

TEST_F(UtestGraphPassesHcclMemcpyPass, Noneed_insert_identity_ShapeComputingSibling)
{
  HcclMemcpyPass hcclMemcpyPass;
  ComputeGraphPtr graph = BuildGraph_Allreduce_Noneed_Insert_Shape_Sibling();
  EXPECT_EQ(graph->GetDirectNodesSize(), 5);

  EXPECT_EQ(hcclMemcpyPass.Run(graph), SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(), 5);

  GraphOptimize graph_optimizer;
  auto ret = graph_optimizer.HandleMemoryRWConflict(graph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(), 5);
  GE_DUMP(graph, "mem_rw");
}
}  // namespace ge
