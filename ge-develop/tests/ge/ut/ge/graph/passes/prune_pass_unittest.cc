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
#include <vector>

#include "omg/omg_inner_types.h"

#include "macro_utils/dt_public_scope.h"
#include "graph/passes/standard_optimize/prune_pass.h"

#include "anchor.h"
#include "common/debug/log.h"
#include "common/debug/memory_dumper.h"
#include "common/types.h"
#include "framework/common/ge_inner_error_codes.h"
#include "graph/attr_value.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/op_desc_utils_ex.h"
#include "graph/passes/pass_manager.h"
#include "macro_utils/dt_public_unscope.h"
#include "graph_builder_utils.h"
#include "ge_graph_dsl/graph_dsl.h"

using namespace testing;
using namespace ge;
using namespace std;

class UtestGraphPassesPrunePass : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

// case1:no net_out_put_node
TEST_F(UtestGraphPassesPrunePass, no_net_out_put_node) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");

  ge::OpDescPtr reverse_op = std::make_shared<ge::OpDesc>();
  ge::OpDescUtilsEx::SetType(reverse_op, REVERSE);
  reverse_op->SetName("Reverse");
  reverse_op->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr reverse_node = graph->AddNode(reverse_op);

  ge::OpDescPtr floor_op = std::make_shared<ge::OpDesc>();
  ge::OpDescUtilsEx::SetType(floor_op, FLOOR);
  floor_op->SetName("Floor");
  floor_op->AddInputDesc(ge::GeTensorDesc());
  floor_op->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr floor_node = graph->AddNode(floor_op);

  ge::GraphUtils::AddEdge(reverse_node->GetOutDataAnchor(0), floor_node->GetInDataAnchor(0));

  uint64_t size_ori = graph->GetDirectNode().size();
  PrunePass prune_pass;
  std::vector<std::pair<string, GraphPass*>> passes = { {"PrunePass", &prune_pass} };
  Status status = PassManager::Run(graph, passes);

  EXPECT_EQ(ge::SUCCESS, status);

  uint64_t size = graph->GetDirectNode().size();
  EXPECT_TRUE(size == size_ori);
}
// case2: one net path with one bypass branch
TEST_F(UtestGraphPassesPrunePass, has_net_out_put_node_with_only_one_path) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");

  ge::OpDescPtr reverse_op = std::make_shared<ge::OpDesc>();
  ge::OpDescUtilsEx::SetType(reverse_op, REVERSE);
  reverse_op->SetName("Reverse");
  reverse_op->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr reverse_node = graph->AddNode(reverse_op);

  ge::OpDescPtr floor_op = std::make_shared<ge::OpDesc>();
  ge::OpDescUtilsEx::SetType(floor_op, FLOOR);
  floor_op->SetName("Floor");
  floor_op->AddInputDesc(ge::GeTensorDesc());
  floor_op->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr floor_node = graph->AddNode(floor_op);

  ge::OpDescPtr net_output_op = std::make_shared<ge::OpDesc>(NODE_NAME_NET_OUTPUT, NETOUTPUT);
  net_output_op->AddInputDesc(ge::GeTensorDesc());
  net_output_op->AddOutputDesc(ge::GeTensorDesc());
  ge::AttrUtils::SetBool(net_output_op, "identity_add_netoutput", true);
  ge::NodePtr netoutput_node = graph->AddNode(net_output_op);

  ge::OpDescPtr reverse_op1 = std::make_shared<ge::OpDesc>();
  ge::OpDescUtilsEx::SetType(reverse_op, REVERSE);
  reverse_op->SetName("Reverse1");
  reverse_op->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr reverse_node1 = graph->AddNode(reverse_op1);

  ge::GraphUtils::AddEdge(reverse_node->GetOutDataAnchor(0), floor_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(floor_node->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(0));

  uint64_t size_ori = graph->GetDirectNode().size();
  PrunePass prune_pass;
  std::vector<std::pair<string, GraphPass*>> passes = { {"PrunePass", &prune_pass} };
  Status status = PassManager::Run(graph, passes);

  uint64_t size = graph->GetDirectNode().size();
  int diff = size_ori - size;
  EXPECT_EQ(ge::SUCCESS, status);
  EXPECT_EQ(diff, 1);
}
// case3: one net path with one bypass branch
TEST_F(UtestGraphPassesPrunePass, has_net_out_put_node_with_one_valid_path_and_one_bypass_path) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");

  // valid path construct (reverse->floor->net_out)
  ge::OpDescPtr reverse_op = std::make_shared<ge::OpDesc>();
  ge::OpDescUtilsEx::SetType(reverse_op, REVERSE);
  reverse_op->SetName("Reverse");
  reverse_op->AddOutputDesc(ge::GeTensorDesc());
  reverse_op->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr reverse_node = graph->AddNode(reverse_op);

  ge::OpDescPtr floor_op = std::make_shared<ge::OpDesc>();
  ge::OpDescUtilsEx::SetType(floor_op, FLOOR);
  floor_op->SetName("Floor");
  floor_op->AddInputDesc(ge::GeTensorDesc());
  floor_op->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr floor_node = graph->AddNode(floor_op);

  ge::OpDescPtr net_output_op = std::make_shared<ge::OpDesc>(NODE_NAME_NET_OUTPUT, NETOUTPUT);
  net_output_op->AddInputDesc(ge::GeTensorDesc());
  net_output_op->AddOutputDesc(ge::GeTensorDesc());
  ge::AttrUtils::SetBool(net_output_op, "identity_add_netoutput", true);
  ge::NodePtr netoutput_node = graph->AddNode(net_output_op);

  ge::GraphUtils::AddEdge(reverse_node->GetOutDataAnchor(0), floor_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(floor_node->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(0));

  // incvalid path construct (reverse->floor1->floor2)
  ge::OpDescPtr floor_op1 = std::make_shared<ge::OpDesc>();
  ge::OpDescUtilsEx::SetType(floor_op1, FLOOR);
  floor_op1->SetName("Floor1");
  floor_op1->AddInputDesc(ge::GeTensorDesc());
  floor_op1->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr floor_node1 = graph->AddNode(floor_op1);

  ge::OpDescPtr floor_op2 = std::make_shared<ge::OpDesc>();
  ge::OpDescUtilsEx::SetType(floor_op2, FLOOR);
  floor_op2->SetName("Floor2");
  floor_op2->AddInputDesc(ge::GeTensorDesc());
  floor_op2->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr floor_node2 = graph->AddNode(floor_op2);
  // isolated node
  ge::OpDescPtr floor_op3 = std::make_shared<ge::OpDesc>();
  ge::OpDescUtilsEx::SetType(floor_op3, FLOOR);
  floor_op3->SetName("Floor3");
  floor_op3->AddInputDesc(ge::GeTensorDesc());
  floor_op3->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr floor_node3 = graph->AddNode(floor_op3);

  EXPECT_EQ(ge::GraphUtils::AddEdge(reverse_node->GetOutDataAnchor(1), floor_node1->GetInDataAnchor(0)), ge::GRAPH_SUCCESS);
  EXPECT_EQ(ge::GraphUtils::AddEdge(floor_node1->GetOutDataAnchor(0), floor_node2->GetInDataAnchor(0)), ge::GRAPH_SUCCESS);

  PrunePass prune_pass;
  vector<GraphPass *> passes = {&prune_pass};
}

// case 4: multi net path with one common netout(1:multi：1)
TEST_F(UtestGraphPassesPrunePass, has_net_out_put_node_with_multi_path) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");

  ge::OpDescPtr data_op = std::make_shared<ge::OpDesc>();
  ge::OpDescUtilsEx::SetType(data_op, DATA);
  data_op->SetName("data");
  data_op->AddOutputDesc(ge::GeTensorDesc());
  data_op->AddOutputDesc(ge::GeTensorDesc());
  data_op->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr data_node = graph->AddNode(data_op);

  ge::OpDescPtr reverse_op1 = std::make_shared<ge::OpDesc>();
  ge::OpDescUtilsEx::SetType(reverse_op1, REVERSE);
  reverse_op1->SetName("Reverse1");
  reverse_op1->AddInputDesc(ge::GeTensorDesc());
  reverse_op1->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr reverse_node1 = graph->AddNode(reverse_op1);

  ge::OpDescPtr floor_op1 = std::make_shared<ge::OpDesc>();
  ge::OpDescUtilsEx::SetType(floor_op1, FLOOR);
  floor_op1->SetName("Floor1");
  floor_op1->AddInputDesc(ge::GeTensorDesc());
  floor_op1->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr floor_node1 = graph->AddNode(floor_op1);

  ge::OpDescPtr reverse_op2 = std::make_shared<ge::OpDesc>();
  ge::OpDescUtilsEx::SetType(reverse_op2, REVERSE);
  reverse_op2->SetName("Reverse2");
  reverse_op2->AddInputDesc(ge::GeTensorDesc());
  reverse_op2->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr reverse_node2 = graph->AddNode(reverse_op2);

  ge::OpDescPtr floor_op2 = std::make_shared<ge::OpDesc>();
  ge::OpDescUtilsEx::SetType(floor_op2, FLOOR);
  floor_op2->SetName("Floor2");
  floor_op2->AddInputDesc(ge::GeTensorDesc());
  floor_op2->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr floor_node2 = graph->AddNode(floor_op2);

  ge::OpDescPtr reverse_op3 = std::make_shared<ge::OpDesc>();
  ge::OpDescUtilsEx::SetType(reverse_op3, REVERSE);
  reverse_op3->SetName("Reverse3");
  reverse_op3->AddInputDesc(ge::GeTensorDesc());
  reverse_op3->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr reverse_node3 = graph->AddNode(reverse_op3);

  ge::OpDescPtr floor_op3 = std::make_shared<ge::OpDesc>();
  ge::OpDescUtilsEx::SetType(floor_op3, FLOOR);
  floor_op3->SetName("Floor3");
  floor_op3->AddInputDesc(ge::GeTensorDesc());
  floor_op3->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr floor_node3 = graph->AddNode(floor_op3);

  ge::OpDescPtr net_output_op = std::make_shared<ge::OpDesc>(NODE_NAME_NET_OUTPUT, NETOUTPUT);
  net_output_op->AddInputDesc(ge::GeTensorDesc());
  net_output_op->AddInputDesc(ge::GeTensorDesc());
  net_output_op->AddInputDesc(ge::GeTensorDesc());
  net_output_op->AddOutputDesc(ge::GeTensorDesc());
  ge::AttrUtils::SetBool(net_output_op, "identity_add_netoutput", true);
  ge::NodePtr netoutput_node = graph->AddNode(net_output_op);

  ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), reverse_node1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(1), reverse_node2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(2), reverse_node3->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(reverse_node1->GetOutDataAnchor(0), floor_node1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(floor_node1->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(reverse_node2->GetOutDataAnchor(0), floor_node2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(floor_node2->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(reverse_node3->GetOutDataAnchor(0), floor_node3->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(floor_node3->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(2));

  uint64_t size_ori = graph->GetDirectNode().size();

  PrunePass prune_pass;
  std::vector<std::pair<string, GraphPass*>> passes = { {"PrunePass", &prune_pass} };
  (void)PassManager::Run(graph, passes);

  uint64_t size_after_proc = graph->GetDirectNode().size();
  EXPECT_TRUE(size_ori == size_after_proc);
}
// case 5: circle,diamand style
TEST_F(UtestGraphPassesPrunePass, multi_net_out_put_node_with_circle_net) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");

  ge::OpDescPtr data_op = std::make_shared<ge::OpDesc>();
  ge::OpDescUtilsEx::SetType(data_op, DATA);
  data_op->SetName("data");
  data_op->AddOutputDesc(ge::GeTensorDesc());
  data_op->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr data_node = graph->AddNode(data_op);

  ge::OpDescPtr op_1 = std::make_shared<ge::OpDesc>();
  ge::OpDescUtilsEx::SetType(op_1, REVERSE);
  op_1->SetName("Reverse1");
  op_1->AddInputDesc(ge::GeTensorDesc());
  op_1->AddOutputDesc(ge::GeTensorDesc());
  op_1->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr node_1 = graph->AddNode(op_1);

  ge::OpDescPtr op_2 = std::make_shared<ge::OpDesc>();
  ge::OpDescUtilsEx::SetType(op_2, REVERSE);
  op_2->SetName("Reverse2");
  op_2->AddInputDesc(ge::GeTensorDesc());
  op_2->AddInputDesc(ge::GeTensorDesc());
  op_2->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr node_2 = graph->AddNode(op_2);

  ge::OpDescPtr op_3 = std::make_shared<ge::OpDesc>();
  ge::OpDescUtilsEx::SetType(op_3, REVERSE);
  op_3->SetName("Reverse3");
  op_3->AddInputDesc(ge::GeTensorDesc());
  op_3->AddInputDesc(ge::GeTensorDesc());
  op_3->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr node_3 = graph->AddNode(op_3);

  ge::OpDescPtr op_4 = std::make_shared<ge::OpDesc>();
  ge::OpDescUtilsEx::SetType(op_4, REVERSE);
  op_4->SetName("Reverse4");
  op_4->AddInputDesc(ge::GeTensorDesc());
  op_4->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr node_4 = graph->AddNode(op_4);

  ge::OpDescPtr op_5 = std::make_shared<ge::OpDesc>();
  ge::OpDescUtilsEx::SetType(op_5, REVERSE);
  op_5->SetName("Reverse5");
  op_5->AddInputDesc(ge::GeTensorDesc());
  op_5->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr node_5 = graph->AddNode(op_5);

  ge::OpDescPtr net_output_op = std::make_shared<ge::OpDesc>(NODE_NAME_NET_OUTPUT, NETOUTPUT);
  net_output_op->AddInputDesc(ge::GeTensorDesc());
  net_output_op->AddOutputDesc(ge::GeTensorDesc());
  ge::AttrUtils::SetBool(net_output_op, "identity_add_netoutput", true);
  ge::NodePtr netoutput_node = graph->AddNode(net_output_op);

  ge::GraphUtils::AddEdge(node_1->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_2->GetOutDataAnchor(0), node_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_3->GetOutDataAnchor(0), node_2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_4->GetOutDataAnchor(0), node_3->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_1->GetOutDataAnchor(1), node_4->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), node_2->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(1), node_5->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_5->GetOutDataAnchor(0), node_3->GetInDataAnchor(1));

  uint64_t size_ori = graph->GetDirectNode().size();

  PrunePass prune_pass;
  std::vector<std::pair<string, GraphPass*>> passes = { {"PrunePass", &prune_pass} };
  Status status = PassManager::Run(graph, passes);
  EXPECT_EQ(ge::SUCCESS, status);
  uint64_t size_after_proc = graph->GetDirectNode().size();
  EXPECT_TRUE(size_ori == size_after_proc);
}

// case 6: two mix circle and multi path,diamand style
TEST_F(UtestGraphPassesPrunePass, mix_two_circle_net) {
  EXPECT_NO_THROW(
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");

    ge::OpDescPtr data_op = std::make_shared<ge::OpDesc>();
    ge::OpDescUtilsEx::SetType(data_op, DATA);
    data_op->SetName("data");
    data_op->AddOutputDesc(ge::GeTensorDesc());
    data_op->AddOutputDesc(ge::GeTensorDesc());
    ge::NodePtr data_node = graph->AddNode(data_op);

    ge::OpDescPtr op_1 = std::make_shared<ge::OpDesc>();
    ge::OpDescUtilsEx::SetType(op_1, REVERSE);
    op_1->SetName("Reverse1");
    op_1->AddInputDesc(ge::GeTensorDesc());
    op_1->AddInputDesc(ge::GeTensorDesc());
    op_1->AddOutputDesc(ge::GeTensorDesc());
    ge::NodePtr node_1 = graph->AddNode(op_1);

    ge::OpDescPtr op_2 = std::make_shared<ge::OpDesc>();
    ge::OpDescUtilsEx::SetType(op_2, REVERSE);
    op_2->SetName("Reverse2");
    op_2->AddInputDesc(ge::GeTensorDesc());
    op_2->AddOutputDesc(ge::GeTensorDesc());
    op_2->AddOutputDesc(ge::GeTensorDesc());
    ge::NodePtr node_2 = graph->AddNode(op_2);

    ge::OpDescPtr op_3 = std::make_shared<ge::OpDesc>();
    ge::OpDescUtilsEx::SetType(op_3, REVERSE);
    op_3->SetName("Reverse3");
    op_3->AddInputDesc(ge::GeTensorDesc());
    op_3->AddInputDesc(ge::GeTensorDesc());
    op_3->AddOutputDesc(ge::GeTensorDesc());
    ge::NodePtr node_3 = graph->AddNode(op_3);

    ge::OpDescPtr op_4 = std::make_shared<ge::OpDesc>();
    ge::OpDescUtilsEx::SetType(op_4, REVERSE);
    op_4->SetName("Reverse4");
    op_4->AddInputDesc(ge::GeTensorDesc());
    op_4->AddInputDesc(ge::GeTensorDesc());
    op_4->AddOutputDesc(ge::GeTensorDesc());
    ge::NodePtr node_4 = graph->AddNode(op_4);

    ge::OpDescPtr op_5 = std::make_shared<ge::OpDesc>();
    ge::OpDescUtilsEx::SetType(op_5, REVERSE);
    op_5->SetName("Reverse5");
    op_5->AddInputDesc(ge::GeTensorDesc());
    op_5->AddOutputDesc(ge::GeTensorDesc());
    op_5->AddOutputDesc(ge::GeTensorDesc());
    ge::NodePtr node_5 = graph->AddNode(op_5);

    ge::OpDescPtr net_output_op = std::make_shared<ge::OpDesc>(NODE_NAME_NET_OUTPUT, NETOUTPUT);
    net_output_op->AddInputDesc(ge::GeTensorDesc());
    net_output_op->AddOutputDesc(ge::GeTensorDesc());
    ge::AttrUtils::SetBool(net_output_op, "identity_add_netoutput", true);
    ge::NodePtr netoutput_node = graph->AddNode(net_output_op);

    ge::GraphUtils::AddEdge(node_1->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_2->GetOutDataAnchor(0), node_1->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_5->GetOutDataAnchor(0), node_1->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(node_4->GetOutDataAnchor(0), node_2->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_2->GetOutDataAnchor(1), node_3->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_5->GetOutDataAnchor(1), node_3->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(node_3->GetOutDataAnchor(0), node_4->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), node_4->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(node_4->GetOutDataAnchor(1), node_5->GetInDataAnchor(0));
    // construct two isolated node
    ge::OpDescPtr op_6 = std::make_shared<ge::OpDesc>();
    ge::OpDescUtilsEx::SetType(op_6, REVERSE);
    op_6->SetName("Reverse");
    op_6->AddInputDesc(ge::GeTensorDesc());
    op_6->AddOutputDesc(ge::GeTensorDesc());
    ge::NodePtr node_6 = graph->AddNode(op_6);

    ge::OpDescPtr op_7 = std::make_shared<ge::OpDesc>();
    ge::OpDescUtilsEx::SetType(op_7, REVERSE);
    op_7->SetName("Reverse");
    op_7->AddInputDesc(ge::GeTensorDesc());
    op_7->AddOutputDesc(ge::GeTensorDesc());
    ge::NodePtr node_7 = graph->AddNode(op_7);

    PrunePass prune_pass;
    vector<GraphPass *> passes = {&prune_pass};
  );
}
// case7: one net path with two DATA node
TEST_F(UtestGraphPassesPrunePass, has_net_out_put_node_with_two_isolate_data_node) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");

  ge::OpDescPtr reverse_op = std::make_shared<ge::OpDesc>();
  ge::OpDescUtilsEx::SetType(reverse_op, REVERSE);
  reverse_op->SetName("Reverse");
  reverse_op->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr reverse_node = graph->AddNode(reverse_op);

  ge::OpDescPtr floor_op = std::make_shared<ge::OpDesc>();
  ge::OpDescUtilsEx::SetType(floor_op, FLOOR);
  floor_op->SetName("Floor");
  floor_op->AddInputDesc(ge::GeTensorDesc());
  floor_op->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr floor_node = graph->AddNode(floor_op);

  ge::OpDescPtr net_output_op = std::make_shared<ge::OpDesc>(NODE_NAME_NET_OUTPUT, NETOUTPUT);
  net_output_op->AddInputDesc(ge::GeTensorDesc());
  net_output_op->AddOutputDesc(ge::GeTensorDesc());
  ge::AttrUtils::SetBool(net_output_op, "identity_add_netoutput", true);
  ge::NodePtr netoutput_node = graph->AddNode(net_output_op);
  // construct one isolated DATA node (to be deleted)
  ge::OpDescPtr reverse_op_1 = std::make_shared<ge::OpDesc>();
  ge::OpDescUtilsEx::SetType(reverse_op_1, REVERSE);
  reverse_op_1->SetName("Reverse1");
  reverse_op_1->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr reverse_node_1 = graph->AddNode(reverse_op_1);

  ge::GraphUtils::AddEdge(reverse_node->GetOutDataAnchor(0), floor_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(floor_node->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(0));
  // construct two isolated DATA nodes(to be not deleted)
  ge::OpDescPtr data_op_1 = std::make_shared<ge::OpDesc>();
  ge::OpDescUtilsEx::SetType(data_op_1, DATA);
  data_op_1->SetName("data");
  data_op_1->AddOutputDesc(ge::GeTensorDesc());
  data_op_1->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr data_node_1 = graph->AddNode(data_op_1);

  ge::OpDescPtr data_op_2 = std::make_shared<ge::OpDesc>();
  ge::OpDescUtilsEx::SetType(data_op_2, DATA);
  data_op_2->SetName("data1");
  data_op_2->AddOutputDesc(ge::GeTensorDesc());
  data_op_2->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr data_node = graph->AddNode(data_op_2);

  uint64_t size_ori = graph->GetDirectNode().size();
  PrunePass prune_pass;
  std::vector<std::pair<string, GraphPass*>> passes = { {"PrunePass", &prune_pass} };
  Status status = PassManager::Run(graph, passes);

  uint64_t size = graph->GetDirectNode().size();
  EXPECT_EQ(ge::SUCCESS, status);
  EXPECT_EQ(size_ori, (size + 1));

  // it should check net_out_put's input data node and input control node
  auto control_vec = netoutput_node->GetInControlNodes();
  EXPECT_EQ(control_vec.size(), 2);
  // check control_vec contains only data node
  for (auto node : control_vec) {
    bool result = (node->GetName() == "data" || node->GetName() == "data1") ? true : false;
    EXPECT_EQ(result, true);
  }

  auto data_vec = netoutput_node->GetInDataNodes();
  EXPECT_EQ(data_vec.size(), 1);
  // check data_vec contains only Floor node
  for (auto node : data_vec) {
    bool result = (node->GetName() == "Floor") ? true : false;
    EXPECT_EQ(result, true);
  }
}

TEST_F(UtestGraphPassesPrunePass, skip_do_prune_with_skip_attr) {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")->NODE("matmul1", "Matmul")->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("const1", "Constant")->EDGE(0, 1)->NODE("matmul1"));
    CHAIN(NODE("const1", "Constant")->EDGE(0, 0)->NODE("cmo1", "Cmo"));
    CHAIN(NODE("const2", "Constant")->EDGE(0, 0)->NODE("cmo2", "Cmo"));
    CHAIN(NODE("const3", "Constant")->EDGE(0, 0)->NODE("cmo3", "Cmo"));
  };
  auto graph = ToComputeGraph(g1);
  ASSERT_NE(graph, nullptr);
  auto cmo1 = graph->FindNode("cmo1");
  ASSERT_NE(cmo1, nullptr);
  AttrUtils::SetBool(cmo1->GetOpDescBarePtr(), ATTR_NAME_SKIP_PRUNE_OPTIMIZE, true);
  auto cmo2 = graph->FindNode("cmo2");
  AttrUtils::SetBool(cmo2->GetOpDescBarePtr(), ATTR_NAME_SKIP_PRUNE_OPTIMIZE, true);
  auto cmo3 = graph->FindNode("cmo3");

  PrunePass prune_pass;
  std::vector<std::pair<string, GraphPass *>> passes = {{"PrunePass", &prune_pass}};
  Status status = PassManager::Run(graph, passes);
  EXPECT_EQ(ge::SUCCESS, status);
  auto cmo1_res = graph->FindNode("cmo1");
  EXPECT_NE(cmo1_res, nullptr);
  auto cmo2_res = graph->FindNode("cmo2");
  
  EXPECT_NE(cmo2_res, nullptr);
  auto cmo3_res = graph->FindNode("cmo3");
  EXPECT_EQ(cmo3_res, nullptr);
  auto const1_res = graph->FindNode("const1");
  EXPECT_NE(const1_res, nullptr);
  auto const2_res = graph->FindNode("const2");
  EXPECT_NE(const2_res, nullptr);
  auto const3_res = graph->FindNode("const3");
  EXPECT_EQ(const3_res, nullptr);
}

TEST_F(UtestGraphPassesPrunePass, RefNodeWithoutOutput_Skip_prune) {
  // build sub graph
  auto graph_builder = ut::GraphBuilder("test");
  auto data_1 = graph_builder.AddNode("data_1", DATA, 0, 1);
  auto data_2 = graph_builder.AddNode("data_2", DATA, 0, 1);
  auto assign =  graph_builder.AddNode("assign", ASSIGN, 2, 1);
  AttrUtils::SetBool(assign->GetOpDesc(), ATTR_NAME_REFERENCE, true);
  auto out_0 =  graph_builder.AddNode("out", NETOUTPUT, 1, 1);
  graph_builder.AddDataEdge(data_1, 0, assign, 0);
  graph_builder.AddDataEdge(data_1, 0, assign, 1);

  auto graph = graph_builder.GetGraph();
  auto size_ori = graph->GetDirectNodesSize();
  PrunePass prune_pass;
  std::vector<std::pair<string, GraphPass*>> passes = { {"PrunePass", &prune_pass} };
  Status status = PassManager::Run(graph, passes);
  uint64_t size = graph->GetDirectNode().size();
  EXPECT_EQ(ge::SUCCESS, status);
  EXPECT_EQ(size_ori, size);
}
