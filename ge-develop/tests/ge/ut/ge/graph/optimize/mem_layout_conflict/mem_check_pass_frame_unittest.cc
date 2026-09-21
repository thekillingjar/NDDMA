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
#include <cstdint>
#include <memory>
#include <string>
#include "result_checker.h"
#include "graph/passes/pass_manager.h"
#include "common/ge_inner_error_codes.h"
#include "graph/optimize/mem_layout_conflict_optimize/mem_layout_conflict_optimizer.h"
#include "graph/passes/graph_builder_utils.h"
#include "common/mem_conflict_share_graph.h"

using namespace testing;
namespace ge {

class TestMemCheckPassFrame : public Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};


TEST_F(TestMemCheckPassFrame, NullParam_Failed) {
  ge::ComputeGraphPtr graph = nullptr;
  MemLayoutConflictOptimizer pass;

  Status ret = pass.Run(graph);

  EXPECT_EQ(ret, PARAM_INVALID);
}

TEST_F(TestMemCheckPassFrame, DynamicGraph_SkipPass) {
  auto builder = ut::GraphBuilder("given");
  auto data = builder.AddNode("data", DATA, 0, 1);
  auto net_output = builder.AddNode("net_output", NETOUTPUT, 1, 0);
  builder.AddDataEdge(data, 0, net_output, 0);
  ge::ComputeGraphPtr graph = builder.GetGraph();
  graph->SetGraphUnknownFlag(true);
  MemLayoutConflictOptimizer pass;

  Status ret = pass.Run(graph);

  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(TestMemCheckPassFrame, SubGraph_SkipPass) {
  auto graph = MemConflictShareGraph::BuildIfGraph();
  auto sub_graph = graph->GetSubgraph("then");
  ASSERT_NE(sub_graph, nullptr);
  MemLayoutConflictOptimizer pass;
  EXPECT_EQ(pass.Run(sub_graph), GRAPH_SUCCESS);
}

TEST_F(TestMemCheckPassFrame, ShouldReturnGraphNoChange_WhenNoConflict) {
  // given
  auto builder = ut::GraphBuilder("given");
  auto data = builder.AddNode("data", DATA, 0, 1);
  auto add = builder.AddNode("add", ADD, 2, 1);
  auto net_output = builder.AddNode("net_output", NETOUTPUT, 1, 0);
  builder.AddDataEdge(data, 0, add, 0);
  builder.AddDataEdge(data, 0, add, 1);
  builder.AddDataEdge(add, 0, net_output, 0);
  ge::ComputeGraphPtr graph = builder.GetGraph();
  MemLayoutConflictOptimizer pass;

  auto except_builder = ut::GraphBuilder("except");
  auto except_data = except_builder.AddNode("data", DATA, 0, 1);
  auto except_add = except_builder.AddNode("add", ADD, 2, 1);
  auto except_net_output = except_builder.AddNode("net_output", NETOUTPUT, 1, 0);
  except_builder.AddDataEdge(except_data, 0, except_add, 0);
  except_builder.AddDataEdge(except_data, 0, except_add, 1);
  except_builder.AddDataEdge(except_add, 0, except_net_output, 0);
  ge::ComputeGraphPtr except_graph = except_builder.GetGraph();

  // when
  Status ret = pass.Run(graph);

  // then
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(mem_check::ResultChecker::CheckGraphEqual(graph, except_graph), true);
}


TEST_F(TestMemCheckPassFrame, ShouldReturnSuccess_WhenTwoInputsButOneSpecialMemory) {
  // given
  auto builder = ut::GraphBuilder("given");
  auto data = builder.AddNode("data", DATA, 0, 1);
  auto add = builder.AddNode("add", ADD, 2, 1);
  std::vector<int64_t> input_memory_types = {3};
  (void)ge::AttrUtils::SetListInt(add->GetOpDescBarePtr(), ATTR_NAME_INPUT_MEM_TYPE_LIST,
    input_memory_types);
  auto net_output = builder.AddNode("net_output", NETOUTPUT, 1, 0);
  builder.AddDataEdge(data, 0, add, 0);
  builder.AddDataEdge(data, 0, add, 1);
  builder.AddDataEdge(add, 0, net_output, 0);
  ge::ComputeGraphPtr graph = builder.GetGraph();
  MemLayoutConflictOptimizer pass;

  // when
  Status ret = pass.Run(graph);

  // then
  EXPECT_EQ(ret, 0);
}

TEST_F(TestMemCheckPassFrame, ShouldReturnSuccess_ControlSolveSuccess) {
  auto graph1 = MemConflictShareGraph::BuildUserInAndUserInGraph();
  Checker test_checker;
  
  auto node_out1 = graph1->FindNode("data1");
  auto node_out2 = graph1->FindNode("data2");
  auto node_out1_index = NodeIndexIO(node_out1, 0, kOut);
  auto node_out2_index = NodeIndexIO(node_out2, 0, kOut);
  auto anchor_out1 = node_out1->GetOutDataAnchor(0);
  auto anchor_out2 = node_out2->GetOutDataAnchor(0);
  auto type_out1 = SmallVector<AnchorAttribute, ATTR_BIT_MAX_LEN>{ANCHOR_ATTR_USER_MEMORY_INPUT};
  auto type_out2 = SmallVector<AnchorAttribute, ATTR_BIT_MAX_LEN>{ANCHOR_ATTR_USER_MEMORY_INPUT};
  AnchorSet result;
  NodeIndexIOVector all_nodes;
  GraphInfo info;

  CheckFuncContext context = {node_out1_index, node_out2_index, 0U, 0U, type_out1, type_out2, all_nodes, info, result};
  EXPECT_EQ(test_checker.CheckConditionConflict(type_out1[0], type_out2[0], context), SUCCESS);
}
}
