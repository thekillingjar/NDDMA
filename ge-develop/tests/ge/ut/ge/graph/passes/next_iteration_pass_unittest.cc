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
#include <string>
#include <map>
#include "macro_utils/dt_public_scope.h"
#include "graph/passes/control_flow_and_stream/next_iteration_pass.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "api/gelib/gelib.h"
#include "graph/node.h"
#include "macro_utils/dt_public_unscope.h"

using namespace ge;

class UtestGraphPassesNextIterationPass : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

static ComputeGraphPtr BuildGraphNextIterationPass() {
  DEF_GRAPH(nextIterationGraph) {
    CHAIN(NODE("enter", ENTER)->EDGE(0U, 0U)->NODE("merge", MERGE));

    CHAIN(NODE("switch1", SWITCH)->EDGE(0U, 0U)->NODE("cast1", CAST));
    CHAIN(NODE("cast1", CAST)->EDGE(0U, 0U)->NODE("cast2", CAST));
    CHAIN(NODE("cast1", CAST)->EDGE(0U, 0U)->NODE("cast3", CAST));

    CHAIN(NODE("switch2", SWITCH)->EDGE(0U, 0U)->NODE("cast4", CAST));
    CHAIN(NODE("cast4", CAST)->EDGE(0U, 0U)->NODE("cast5", CAST));
  };

  const auto graph = ToGeGraph(nextIterationGraph);
  const auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  compute_graph->TopologicalSorting();

  return compute_graph;
}

TEST_F(UtestGraphPassesNextIterationPass, GroupEnterNodeFailed) {
  auto graph = BuildGraphNextIterationPass();
  ge::NextIterationPass pass_;

  Status status = pass_.Run(graph);
  EXPECT_EQ(INTERNAL_ERROR, status);

  // FindWhileGroups find NEXTITERATION failed
  auto enter_desc = graph->FindNode("enter")->GetOpDesc();
  std::string frame_name = "frame_name";
  std::string batch_label = "batch_label";
  ge::AttrUtils::SetStr(enter_desc, ENTER_ATTR_FRAME_NAME, frame_name);
  ge::AttrUtils::SetStr(enter_desc, ATTR_NAME_BATCH_LABEL, batch_label);
  status = pass_.Run(graph);
  EXPECT_EQ(INTERNAL_ERROR, status);

  // FindWhileGroups find SWITCH failed
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("next_iteration", NEXTITERATION);
  op_desc->AddOutputDesc(ge::GeTensorDesc());
  op_desc->AddInputDesc(ge::GeTensorDesc());
  ge::NodePtr next_iteration_node = graph->AddNode(op_desc);
  auto merge_node = graph->FindNode("merge");
  merge_node->AddLinkFrom(next_iteration_node);
  status = pass_.Run(graph);
  EXPECT_EQ(INTERNAL_ERROR, status);

  // FindWhileGroups find SWITCH failed
  op_desc = std::make_shared<ge::OpDesc>("switch", SWITCH);
  op_desc->AddOutputDesc(ge::GeTensorDesc());
  op_desc->AddInputDesc(ge::GeTensorDesc());
  ge::NodePtr switch_node = graph->AddNode(op_desc);
  switch_node->AddLinkFrom(merge_node);
  status = pass_.Run(graph);
  EXPECT_EQ(INTERNAL_ERROR, status);

  status = pass_.ClearStatus();
  EXPECT_EQ(SUCCESS, status);
}

TEST_F(UtestGraphPassesNextIterationPass, VerifyWhileGroupFailed) {
  auto graph = BuildGraphNextIterationPass();
  ge::NextIterationPass pass_;
  auto loop_group = MakeShared<LoopCondGroup>();
  pass_.loop_group_map_[std::string()] = loop_group;
  bool ret = pass_.VerifyWhileGroup();
  EXPECT_EQ(false, ret);
  (void)pass_.ClearStatus();

  pass_.loop_group_map_[std::string("abc")] = loop_group;
  ret = pass_.VerifyWhileGroup();
  EXPECT_EQ(false, ret);

  NodePtr op_node = graph->FindNode("merge");
  loop_group->loop_cond = op_node;
  loop_group->merge_next_pairs.push_back({nullptr, nullptr});
  ret = pass_.VerifyWhileGroup();
  EXPECT_EQ(false, ret);

  (void)pass_.ClearStatus();
}

TEST_F(UtestGraphPassesNextIterationPass, HandleSwitchExitNodesFailed) {
  auto graph = BuildGraphNextIterationPass();
  ge::NextIterationPass pass_;
  auto loop_group = MakeShared<LoopCondGroup>();

  auto node = graph->FindNode("switch1");
  loop_group->switch_nodes.push_back(node);
  EXPECT_NO_THROW(pass_.HandleSwitchExitNodes(*loop_group, 0));

  loop_group->switch_nodes.clear();
  node = graph->FindNode("switch2");
  loop_group->switch_nodes.push_back(node);
  EXPECT_NO_THROW(pass_.HandleSwitchExitNodes(*loop_group, 0));
}

TEST_F(UtestGraphPassesNextIterationPass, BreakNextIterationFailed) {
  ge::NextIterationPass pass_;
  NodePtr null_node = nullptr;
  auto ret = pass_.BreakNextIteration(null_node, null_node);
  EXPECT_EQ(PARAM_INVALID, ret);
}

TEST_F(UtestGraphPassesNextIterationPass, FindTargetNodeFailed) {
  ge::NextIterationPass pass_;
  NodePtr null_node = nullptr;
  auto ret = pass_.FindTargetNode(null_node, std::string(), false, null_node);
  EXPECT_EQ(PARAM_INVALID, ret);
}
