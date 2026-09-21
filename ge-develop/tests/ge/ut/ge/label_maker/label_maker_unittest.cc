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
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/debug/ge_attr_define.h"
#include "common/types.h"
#include "graph/passes/graph_builder_utils.h"
#include <vector>

#include "macro_utils/dt_public_scope.h"
#include "graph/label/label_maker.h"
#include "graph/build/label_allocator.h"
#include "macro_utils/dt_public_unscope.h"

using namespace std;
using namespace testing;
using namespace ge;

class UtestLabelMaker : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

namespace {
ComputeGraphPtr MakeFunctionGraph(const std::string &func_node_name, const std::string &func_node_type) {
  DEF_GRAPH(g1) {
    CHAIN(NODE("_arg_0", DATA)->NODE(func_node_name, func_node_type)->NODE("Node_Output", NETOUTPUT));
    CHAIN(NODE("_arg_1", DATA)->NODE(func_node_name));
    CHAIN(NODE("_arg_2", DATA)->NODE(func_node_name));
  };
  return ToComputeGraph(g1);
}
ComputeGraphPtr MakeSubGraph(const std::string &prefix) {
  DEF_GRAPH(g2, prefix.c_str()) {
    auto data_0 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0);
    auto data_1 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 1);
    auto data_2 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 2);
    auto conv_0 = OP_CFG(CONV2D).Attr(ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM));
    auto relu_0 = OP_CFG(RELU).Attr(ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::AI_CPU)).
                               Attr(ATTR_NAME_STREAM_LABEL, "stream_label");
    auto add_0 = OP_CFG(ADD).Attr(ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::AI_CPU));
    CHAIN(NODE(prefix + "_arg_0", data_0)
              ->EDGE(0, 0)
              ->NODE(prefix + "Conv2D", conv_0)
              ->EDGE(0, 0)
              ->NODE(prefix + "Relu", relu_0)
              ->EDGE(0, 0)
              ->NODE(prefix + "Add", add_0)
              ->EDGE(0, 0)
              ->NODE(prefix + "Node_Output", NETOUTPUT));
    CHAIN(NODE(prefix + "_arg_1", data_1)->EDGE(0, 1)->NODE(prefix + "Conv2D", conv_0));
    CHAIN(NODE(prefix + "_arg_2", data_2)->EDGE(0, 1)->NODE(prefix + "Add", add_0));
  };
  return ToComputeGraph(g2);
}

}  // namespace

TEST_F(UtestLabelMaker, partition_label_maker_run_failed) {
  std::string func_node_name = "PartitionedCall_0";
  const auto &root_graph = MakeFunctionGraph(func_node_name, PARTITIONEDCALL);
  auto func_node = root_graph->FindNode(func_node_name);
  EXPECT_NE(func_node, nullptr);
  // with no subgraph info
  auto maker = LabelMakerFactory::Instance().Create(func_node->GetType(), root_graph, func_node);
  auto index = 0U;
  auto ret = maker->Run(index);
  EXPECT_EQ(ret, FAILED);

  // with invalid subgraph
  const auto &sub_graph = std::make_shared<ComputeGraph>("invalid_sub_graph");
  ut::GraphBuilder::AddPartitionedCall(root_graph, func_node_name, sub_graph);
  ret = maker->Run(index);
  EXPECT_EQ(ret, FAILED);

  // with non_exist subgraph
  func_node->GetOpDesc()->SetSubgraphInstanceName(0, "non_exist_name");
  ret = maker->Run(index);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(UtestLabelMaker, case_label_maker_run_failed) {
  std::string func_node_name = "Case_0";
  const auto &root_graph = MakeFunctionGraph(func_node_name, CASE);
  auto func_node = root_graph->FindNode(func_node_name);
  EXPECT_NE(func_node, nullptr);
  // with no subgraph info
  auto maker = LabelMakerFactory::Instance().Create(func_node->GetType(), root_graph, func_node);
  auto index = 0U;
  auto ret = maker->Run(index);
  EXPECT_EQ(ret, FAILED);

  // with single subgraph
  const auto &sub_graph_0 = MakeSubGraph("sub_graph_0/");
  ut::GraphBuilder::AddPartitionedCall(root_graph, func_node_name, sub_graph_0);
  ret = maker->Run(index);
  EXPECT_EQ(ret, SUCCESS);

  // with invalid subgraph
  const auto &sub_graph = std::make_shared<ComputeGraph>("invalid_sub_graph");
  ut::GraphBuilder::AddPartitionedCall(root_graph, func_node_name, sub_graph);
  ret = maker->Run(index);
  EXPECT_EQ(ret, FAILED);

  // with non_exist subgraph
  func_node->GetOpDesc()->SetSubgraphInstanceName(0, "non_exist_name");
  ret = maker->Run(index);
  EXPECT_EQ(ret, PARAM_INVALID);
}

TEST_F(UtestLabelMaker, if_label_maker_run_failed_1) {
  std::string func_node_name = "If_0";
  const auto &root_graph = MakeFunctionGraph(func_node_name, IF);
  auto func_node = root_graph->FindNode(func_node_name);
  EXPECT_NE(func_node, nullptr);
  // with no subgraph info
  auto maker = LabelMakerFactory::Instance().Create(func_node->GetType(), root_graph, func_node);
  auto index = 0U;
  auto ret = maker->Run(index);
  EXPECT_EQ(ret, FAILED);

  // with invalid subgraph
  const auto &sub_graph = std::make_shared<ComputeGraph>("invalid_sub_graph");
  ut::GraphBuilder::AddPartitionedCall(root_graph, func_node_name, sub_graph);
  ret = maker->Run(index);
  EXPECT_EQ(ret, FAILED);
  const auto &sub_graph_1 = std::make_shared<ComputeGraph>("invalid_sub_graph_1");
  ut::GraphBuilder::AddPartitionedCall(root_graph, func_node_name, sub_graph_1);
  ret = maker->Run(index);
  EXPECT_EQ(ret, FAILED);

  // with non_exist subgraph
  func_node->GetOpDesc()->SetSubgraphInstanceName(0, "non_exist_name");
  ret = maker->Run(index);
  EXPECT_EQ(ret, PARAM_INVALID);
}

TEST_F(UtestLabelMaker, if_label_maker_run_failed_2) {
  std::string func_node_name = "If_0";
  const auto &root_graph = MakeFunctionGraph(func_node_name, IF);
  auto func_node = root_graph->FindNode(func_node_name);
  EXPECT_NE(func_node, nullptr);

  // with valid then subgraph + invalid else subgraph
  const auto &sub_graph_0 = MakeSubGraph("sub_graph_0/");
  ut::GraphBuilder::AddPartitionedCall(root_graph, func_node_name, sub_graph_0);
  const auto &sub_graph_1 = std::make_shared<ComputeGraph>("invalid_sub_graph_1");
  ut::GraphBuilder::AddPartitionedCall(root_graph, func_node_name, sub_graph_1);

  auto maker = LabelMakerFactory::Instance().Create(func_node->GetType(), root_graph, func_node);
  auto index = 0U;
  auto ret = maker->Run(index);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(UtestLabelMaker, while_label_maker_run_failed_1) {
  std::string func_node_name = "While_0";
  const auto &root_graph = MakeFunctionGraph(func_node_name, WHILE);
  auto func_node = root_graph->FindNode(func_node_name);
  EXPECT_NE(func_node, nullptr);
  // with no subgraph info
  auto maker = LabelMakerFactory::Instance().Create(func_node->GetType(), root_graph, func_node);
  auto index = 0U;
  auto ret = maker->Run(index);
  EXPECT_EQ(ret, FAILED);

  // with invalid subgraph
  const auto &sub_graph = std::make_shared<ComputeGraph>("invalid_sub_graph");
  ut::GraphBuilder::AddPartitionedCall(root_graph, func_node_name, sub_graph);
  ret = maker->Run(index);
  EXPECT_EQ(ret, FAILED);
  const auto &sub_graph_1 = std::make_shared<ComputeGraph>("invalid_sub_graph_1");
  ut::GraphBuilder::AddPartitionedCall(root_graph, func_node_name, sub_graph_1);
  ret = maker->Run(index);
  EXPECT_EQ(ret, FAILED);

  // with non_exist subgraph
  func_node->GetOpDesc()->SetSubgraphInstanceName(0, "non_exist_name");
  ret = maker->Run(index);
  EXPECT_EQ(ret, PARAM_INVALID);
}

TEST_F(UtestLabelMaker, while_label_maker_run_failed_2) {
  std::string func_node_name = "While_0";
  const auto &root_graph = MakeFunctionGraph(func_node_name, WHILE);
  auto func_node = root_graph->FindNode(func_node_name);
  EXPECT_NE(func_node, nullptr);

  // with valid cond subgraph + invalid body subgraph
  const auto &sub_graph_0 = MakeSubGraph("sub_graph_0/");
  ut::GraphBuilder::AddPartitionedCall(root_graph, func_node_name, sub_graph_0);
  const auto &sub_graph_1 = std::make_shared<ComputeGraph>("invalid_sub_graph_1");
  ut::GraphBuilder::AddPartitionedCall(root_graph, func_node_name, sub_graph_1);

  auto maker = LabelMakerFactory::Instance().Create(func_node->GetType(), root_graph, func_node);
  auto index = 0U;
  auto ret = maker->Run(index);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(UtestLabelMaker, while_label_maker_run_failed_3) {
  std::string func_node_name = "While_0";
  const auto &root_graph = MakeFunctionGraph(func_node_name, WHILE);
  auto func_node = root_graph->FindNode(func_node_name);
  EXPECT_NE(func_node, nullptr);

  // cond_graph has two output
  DEF_GRAPH(g2) {
    CHAIN(NODE("_arg_0", DATA)->EDGE(0, 0)->NODE("Relu", RELU)->EDGE(0, 0)->NODE("Node_Output", NETOUTPUT));
    CHAIN(NODE("_arg_0")->EDGE(0, 1)->NODE("Node_Output"));
  };
  const auto &cond_graph = ToComputeGraph(g2);
  ut::GraphBuilder::AddPartitionedCall(root_graph, func_node_name, cond_graph);
  const auto &body_graph = MakeSubGraph("sub_graph_1/");
  ut::GraphBuilder::AddPartitionedCall(root_graph, func_node_name, body_graph);

  auto maker = LabelMakerFactory::Instance().Create(func_node->GetType(), root_graph, func_node);
  auto index = 0U;
  auto ret = maker->Run(index);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(UtestLabelMaker, partition_label_maker_run_success) {
  std::string func_node_name = "PartitionedCall_0";
  const auto &root_graph = MakeFunctionGraph(func_node_name, PARTITIONEDCALL);
  const auto &sub_graph = MakeSubGraph("sub_graph_0/");
  ut::GraphBuilder::AddPartitionedCall(root_graph, func_node_name, sub_graph);
  LabelAllocator label_allocator(root_graph);
  auto ret = label_allocator.AssignFunctionalLabels();
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestLabelMaker, case_label_maker_run_success) {
  std::string func_node_name = "Case_0";
  const auto &root_graph = MakeFunctionGraph(func_node_name, CASE);
  const auto &sub_graph = MakeSubGraph("sub_graph_0/");
  ut::GraphBuilder::AddPartitionedCall(root_graph, func_node_name, sub_graph);
  const auto &sub_graph_1 = MakeSubGraph("sub_graph_1/");
  ut::GraphBuilder::AddPartitionedCall(root_graph, func_node_name, sub_graph_1);
  LabelAllocator label_allocator(root_graph);
  auto ret = label_allocator.AssignFunctionalLabels();
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestLabelMaker, if_label_maker_run_success) {
  std::string func_node_name = "If_0";
  const auto &root_graph = MakeFunctionGraph(func_node_name, IF);
  const auto &sub_graph = MakeSubGraph("sub_graph_0/");
  ut::GraphBuilder::AddPartitionedCall(root_graph, func_node_name, sub_graph);
  const auto &sub_graph_true = MakeSubGraph("sub_graph_true/");
  ut::GraphBuilder::AddPartitionedCall(root_graph, func_node_name, sub_graph_true);
  LabelAllocator label_allocator(root_graph);
  auto ret = label_allocator.AssignFunctionalLabels();
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestLabelMaker, while_label_maker_run_success) {
  std::string func_node_name = "While_0";
  const auto &root_graph = MakeFunctionGraph(func_node_name, WHILE);
  const auto &cond_graph = MakeSubGraph("sub_graph_0/");
  ut::GraphBuilder::AddPartitionedCall(root_graph, func_node_name, cond_graph);
  const auto &body_graph = MakeSubGraph("sub_graph_1/");
  ut::GraphBuilder::AddPartitionedCall(root_graph, func_node_name, body_graph);
  LabelAllocator label_allocator(root_graph);
  auto ret = label_allocator.AssignFunctionalLabels();
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestLabelMaker, other_lable_maker_func) {
  std::string func_node_name = "While_0";
  const auto &root_graph = MakeFunctionGraph(func_node_name, WHILE);
  auto func_node = root_graph->FindNode(func_node_name);
  EXPECT_NE(func_node, nullptr);
  auto maker = LabelMakerFactory::Instance().Create(func_node->GetType(), root_graph, func_node);
  EXPECT_NE(maker, nullptr);

  // with invalid subgraph
  const auto &sub_graph = std::make_shared<ComputeGraph>("invalid_sub_graph");
  auto index = 1U;
  auto ret = maker->AddLabelGotoEnter(sub_graph, "goto_enter", index);
  EXPECT_EQ(ret, nullptr);
  ret = maker->AddLabelSwitchEnter(sub_graph, "switch_enter", GeTensorDesc(), {});
  EXPECT_EQ(ret, nullptr);
  ret = maker->AddLabelSetEnter(sub_graph, "switch_enter", index, func_node);
  EXPECT_EQ(ret, nullptr);
  const auto &cond_graph = MakeSubGraph("sub_graph_0/");
  ret = maker->AddLabelGotoEnter(cond_graph, "goto_enter", index);
  EXPECT_NE(ret, nullptr);
}
