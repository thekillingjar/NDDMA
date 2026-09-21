/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/load/model_manager/task_node_map.h"
#include <gtest/gtest.h>
#include "common/share_graph.h"
namespace ge {
class TaskNodeMapUT : public testing::Test {};
TEST_F(TaskNodeMapUT, AddRelation_Failed_TaskIndexOutOfRange) {
  TaskNodeMap tnm;
  auto graph = gert::ShareGraph::BuildAiCoreRtsDsaToIdentityKnownShapeGraph();
  graph->TopologicalSorting();
  ASSERT_EQ(tnm.Init(graph, 10), SUCCESS);

  ASSERT_NE(tnm.AddRelation(10, 0), SUCCESS);
  ASSERT_NE(tnm.AddRelation(11, 0), SUCCESS);
  ASSERT_EQ(tnm.AddRelation(9, 0), SUCCESS);
}
TEST_F(TaskNodeMapUT, AddRelation_Failed_NodeIdDoesNotExists) {
  TaskNodeMap tnm;
  auto graph = gert::ShareGraph::BuildAiCoreRtsDsaToIdentityKnownShapeGraph();
  graph->TopologicalSorting();
  ASSERT_EQ(tnm.Init(graph, 10), SUCCESS);

  ASSERT_NE(tnm.AddRelation(1, 7), SUCCESS);
  ASSERT_NE(tnm.AddRelation(1, 8), SUCCESS);
  ASSERT_EQ(tnm.AddRelation(1, 6), SUCCESS);
}
TEST_F(TaskNodeMapUT, FindTaskByNodeId_Empty_NodeIdDoesNotAdded) {
  TaskNodeMap tnm;
  auto graph = gert::ShareGraph::BuildAiCoreRtsDsaToIdentityKnownShapeGraph();
  graph->TopologicalSorting();
  ASSERT_EQ(tnm.Init(graph, 10), SUCCESS);

  ASSERT_EQ(tnm.AddRelation(0, 0), SUCCESS);
  ASSERT_EQ(tnm.AddRelation(1, 1), SUCCESS);
  ASSERT_EQ(tnm.AddRelation(2, 2), SUCCESS);

  ASSERT_TRUE(tnm.FindTasksByNodeId(3).empty());
  ASSERT_TRUE(tnm.FindTasksByNodeId(4).empty());
  ASSERT_EQ(tnm.FindTasksByNodeId(2).size(), 1);
}
TEST_F(TaskNodeMapUT, FindTaskByNodeId_IdsCorrect) {
  TaskNodeMap tnm;
  auto graph = gert::ShareGraph::BuildAiCoreRtsDsaToIdentityKnownShapeGraph();
  graph->TopologicalSorting();
  ASSERT_EQ(tnm.Init(graph, 10), SUCCESS);

  ASSERT_EQ(tnm.AddRelation(0, 0), SUCCESS);
  ASSERT_EQ(tnm.AddRelation(1, 1), SUCCESS);
  ASSERT_EQ(tnm.AddRelation(2, 1), SUCCESS);

  ASSERT_EQ(tnm.FindTasksByNodeId(0), std::vector<size_t>({0}));
  ASSERT_EQ(tnm.FindTasksByNodeId(1), std::vector<size_t>({1, 2}));
}
}  // namespace ge
