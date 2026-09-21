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
#include "common/mem_conflict_share_graph.h"
#include "result_checker.h"
#include "graph/compute_graph.h"
#include "compiler/graph/optimize/mem_layout_conflict_optimize/mem_layout_conflict_optimizer.h"
namespace ge {
class UtestMemLayoutConflictImmutableOutput : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

/*
 *   const
 *     |
 *   hcomallreduce (p2p input)
 *
 *    ||
 *    \/
 *
 *   const
 *     |
 *   identity
 *     |
 *   hcomallreduce (p2p input)
 */
TEST_F(UtestMemLayoutConflictImmutableOutput, ImmutableOutAndRtsSpecailIn_InsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildImmutableOutAndRtsSpecailInGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"hcomallreduce"}}), GRAPH_SUCCESS);
}

/*
 *     data1
 *       |
 *       if
 *       |
 *      op3
 *       |
 *    netoutput
 *
 * then_subgraph          else_subgraph
 * +------------------+   +------------------+
 * | data2  netoutput1|   | data3  netoutput2|
 * |variable--+       |   |hcomallreduce--+  | hcomallreduce need p2p output
 * +------------------+   +------------------+
 *
 *   ||
 *   \/
 *
 *     data1
 *       |
 *       if
 *       |
 *      op3
 *       |
 *    netoutput
 *
 * then_subgraph                                else_subgraph
 * +-----------------------------+     +------------------------------------+
 * | data2                       |     | data3                              |
 * |variable--identity-netoutput1|     |hcomallreduce--identity-netoutput2  | hcomallreduce need p2p output
 * +-----------------------------+     +------------------------------------+
 *
 */
TEST_F(UtestMemLayoutConflictImmutableOutput, ImmutableOutAndRtsSpecailOut_InsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildImmutableOutAndRtsSpecailOutGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"netoutput1"}}), GRAPH_SUCCESS);
}

/*
 *  unknow root graph               know subgraph
 *   ref_data1   partitioned_call1  +-----------------+
 *         \     / /                | var   constant1 |
 *         partitioned_call2        |   \    /        |
 *         |    |    |              |   netoutput1    |
 *         op1 op2 op3              +-----------------+
 *         |   |    |               know subgraph
 *         netoutput                +---------------------------------+
 *                                  | ref_data2  data2       data3    |
 *                                  |   |           |          |      |
 *                                  |transdata1 transdata2 transdata3 |
 *                                  |         \      |      /         |
 *                                  |            netoutput2           |
 *                                  +---------------------------------+
 *             不插入任何identity
 */
TEST_F(UtestMemLayoutConflictImmutableOutput, RefDataInKnownSubGraph_NotInsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildRefDataGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 0U), GRAPH_SUCCESS);
}
/*
 *   const  variable       const  variable
 *      \     /       ==>     |      |
 *       concat             identity identity
 *                              \     /
 *                               concat
 */
TEST_F(UtestMemLayoutConflictImmutableOutput, ImmutableOutAndContinuousInput_InsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildImmutableOutAndContinuousInput();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"concat", 0}, {"concat", 1}}), GRAPH_SUCCESS);
}
/*
 *   const  variable       const  variable
 *      \     /       ==>     |      |
 *    phony_concat           identity identity
 *                              \     /
 *                             phony_concat
 */
TEST_F(UtestMemLayoutConflictImmutableOutput, ImmutableOutAndNoPaddingContinuousInput_InsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildImmutableOutAndNoPaddingContinuousInput();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"phony_concat", 0}, {"phony_concat", 1}}), GRAPH_SUCCESS);
}
/*
 *   const           const
 *     |       ==>     |
 *    split         identity
 *    /  \             |
 *   a    b          split
 *                    /  \
 *                   a    b
 */
TEST_F(UtestMemLayoutConflictImmutableOutput, ImmutableOutAndContinuousOutput_InsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildImmutableOutAndContinuousOutput();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"split", 0}}), GRAPH_SUCCESS);
}
/*
 *   const           const
 *     |       ==>     |
 *  phony_split      identity
 *    /  \             |
 *   a    b        phony_split
 *                    /  \
 *                   a    b
 */
TEST_F(UtestMemLayoutConflictImmutableOutput, ImmutableOutAndNoPaddingContinuousOutput_InsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildImmutableOutAndNoPaddingContinuousOutput();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"phony_split", 0}}), GRAPH_SUCCESS);
}
/*
 *     var                                                var
 *      +--ApplyMomentum-c                                 +--ApplyMomentum-c
 *      |                                                  |
 *   phony_split (输出引用输入，且NoPadding连续输出内存)      identity
 *     /  \                                                |
 *    a    b                                            phony_split
 *                                                        /  \
 *                                                       a    b
 * 关注点：不能在var后面，而是在phony_split前面。var-ApplyMomentum中间不能有identity
 */
TEST_F(UtestMemLayoutConflictImmutableOutput, BuildVarAndNoPaddingContinuousOutputWithMultiReference_InsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildVarAndNoPaddingContinuousOutputWithMultiReference();
  MemLayoutConflictOptimizer mem_check_pass;

  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"phony_split", 0}}), GRAPH_SUCCESS);
  auto node = graph->FindNode("apply_momentum");
  ASSERT_NE(node, nullptr);
  auto in_nodes = node->GetInDataNodes();
  ASSERT_FALSE(in_nodes.empty());
  EXPECT_EQ(in_nodes.at(0U)->GetName(), "var");
}

/*
 *    var  const        var  const
 *     \    /            \    /
 *     assign            assign
 *       |       ==>       |
 *   hcombroadcast       identity
 *       / \                |
 *      a   b          hcombroadcast (need continuous output)
 *      |   |              / \
 *     netoutput          a   b
 *                        |   |
 *                       netoutput
 */
TEST_F(UtestMemLayoutConflictImmutableOutput,
       ImmutableOutAndContinuousByAssign_InsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildImmutableOutAndContinuousByAssignOutput();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 1U), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"hcombroadcast", 0}}), GRAPH_SUCCESS);
}

/*
 *   var  const       var  const
 *    \    /           \    /
 *    assign           assign
 *      |       ==>      |
 *  hcombroadcast      identity
 *      / \              |
 *     a   b          hcombroadcast (need p2p in/out)
 *     |   |              / \
 *    netoutput          a   b
 *                       |   |
 *                      netoutput
 */
TEST_F(UtestMemLayoutConflictImmutableOutput,
       ImmutableOutAndRtsSpecailOutByAssign_InsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildImmutableOutAnRtsSpecailOutByAssignOutput();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 1U), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"hcombroadcast", 0}}), GRAPH_SUCCESS);
}

/*
 *    var  const       var  const
 *     \    /           \    /
 *     assign   a       assign
 *        \    /           \
 *          pc           identity a
 *           |     ==>       \   /
 *           b                 pc
 *           |                  |
 *       netoutput              b
 *                              |
 *                           netoutput
 */
TEST_F(UtestMemLayoutConflictImmutableOutput,
       ImmutableOutAndNoPaddingContinuousInByAssign_InsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildImmutableOutAndNoPaddingContinuousInByAssignOutput();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 1U), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"pc", 0}}), GRAPH_SUCCESS);
}


} // namespace ge
