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
#include "ge_context.h"

namespace ge {
class UtestMemLayoutConflictNotSupportRefreshOut : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

/*
 *  swtich
 *    |
 *  hcomallreduce p2p in
 *  ||
 *  \/
 *
 *  swtich
 *    |
 *  hcomallreduce p2p in
 */
TEST_F(UtestMemLayoutConflictNotSupportRefreshOut, NotSupportRefreshOutAndRtsSpecailIn_NotInsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildNotSupportRefreshOutAndRtsSpecialInGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 0U), GRAPH_SUCCESS);
}

/*
 * unknown graph                             unknown graph
 *      op1                                       op1
 *       |               known sub graph           |               known sub graph
 * partitioned_call      +-----------+       partitioned_call      +-----------------+
 *       |               |   switch  |  ==>        |               |   switch        |
 *      op2              |   |  |    |            op2              |   /     \       |
 *                       |   add     |                             |identity identity|
 *                       |    |      |                             |    |     /      |
 *                       |  netoutput|                             |  netoutput      |
 *                       +-----------+                             +-----------------+
 *  这个用例主要时验证在节点output index 1后面插入identity，功能正确不报错
 */
TEST_F(UtestMemLayoutConflictNotSupportRefreshOut, NotSupportRefreshOut_AnchorIndex1_UnknowGraph_InsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildNotSupportRefreshOutAnchorIndex1Graph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 2U), GRAPH_SUCCESS);
}

/*
 *  swtich
 *    |
 *  hcomallreduce p2p in
 *  ||
 *  \/
 *
 *  swtich
 *    |
 *  idenitty
 *    |
 *  hcomallreduce p2p in
 */
TEST_F(UtestMemLayoutConflictNotSupportRefreshOut,
       NotSupportRefreshOutAndRtsSpecailIn_InsertIdentityWhenFeatureMapRefreshable_Success) {
  auto graph = MemConflictShareGraph::BuildNotSupportRefreshOutAndRtsSpecialInGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  FeatureMapRefreshOptionGuarder guarder;
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
 * +--------------------+   +------------------------+
 * | data2              |   | data3                  |
 * |switch--netoutput1  |   |hcomallreduce-netoutput2|  hcomallreduce need p2p output
 * +--------------------+   +------------------------+
 *
 *  ||
 *  \/
 *
 *     data1
 *       |
 *       if
 *       |
 *      op3
 *       |
 *    netoutput
 *
 * then_subgraph          else_subgraph
 * +-------------------+   +------------------------+
 * | data2             |   | data3                  |
 * |switch-netoutput1  |   |hcomallreduce-netoutput2|  hcomallreduce need p2p output
 * +-------------------+   +------------------------+
 */
TEST_F(UtestMemLayoutConflictNotSupportRefreshOut, NotSupportRefreshOutAndRtsSpecailOut_NotInsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildNotSupportRefreshOutAndRtsSpecailOutGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 0U), GRAPH_SUCCESS);
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
 * +--------------------+   +------------------------+
 * | data2              |   | data3                  |
 * |switch--netoutput1  |   |hcomallreduce-netoutput2|  hcomallreduce need p2p output
 * +--------------------+   +------------------------+
 *
 *  ||
 *  \/
 *
 *     data1
 *       |
 *       if
 *       |
 *      op3
 *       |
 *    netoutput
 *
 * then_subgraph          else_subgraph
 * +-----------------------------+   +------------------------+
 * | data2                       |   | data3                  |
 * |switch-identity1-netoutput1  |   |hcomallreduce-netoutput2|  hcomallreduce need p2p output
 * +-----------------------------+   +------------------------+
 */
TEST_F(UtestMemLayoutConflictNotSupportRefreshOut,
       NotSupportRefreshOutAndRtsSpecailOut_InsertIdentityWhenFeatureMapRefreshable_Success) {
  auto graph = MemConflictShareGraph::BuildNotSupportRefreshOutAndRtsSpecailOutGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  FeatureMapRefreshOptionGuarder guarder;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"netoutput1"}}), GRAPH_SUCCESS);
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
 * +--------------------+   +---------------+
 * | data2              |   | data3         |
 * |switch--netoutput1  |   |add-netoutput2 |
 * +--------------------+   +---------------+
 *
 *           ||
 *           \/
 *
 *     data1
 *       |
 *       if
 *       |
 *      op3
 *       |
 *    netoutput
 *
 * then_subgraph          else_subgraph
 * +-------------------+   +---------------+
 * | data2             |   | data3         |
 * |switch-netoutput1  |   |add-netoutput2 |
 * +-------------------+   +---------------+
 */
TEST_F(UtestMemLayoutConflictNotSupportRefreshOut, NotSupportRefreshOutAndNormalOut_NotInsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildNotSupportRefreshOutAndNormalOutGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 0U), GRAPH_SUCCESS);
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
 * +--------------------+   +---------------+
 * | data2              |   | data3         |
 * |switch--netoutput1  |   |add-netoutput2 |
 * +--------------------+   +---------------+
 *
 *           ||
 *           \/
 *
 *     data1
 *       |
 *       if
 *       |
 *      op3
 *       |
 *    netoutput
 *
 * then_subgraph          else_subgraph
 * +----------------------------+   +---------------+
 * | data2                      |   | data3         |
 * |switch-identity-netoutput1  |   |add-netoutput2 |
 * +----------------------------+   +---------------+
 */
TEST_F(UtestMemLayoutConflictNotSupportRefreshOut,
       NotSupportRefreshOutAndNormalOut_InsertIdentityWhenFeatureMapRefreshable_Success) {
  auto graph = MemConflictShareGraph::BuildNotSupportRefreshOutAndNormalOutGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  FeatureMapRefreshOptionGuarder guarder;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"netoutput1"}}), GRAPH_SUCCESS);
}

/*
 *     data1
 *       |
 *       if
 *      / \
 *      op3
 *       |
 *    netoutput
 *
 * then_subgraph            else_subgraph
 * +--------------------+   +---------------+
 * |data2               |   |data3          |
 * |switch+--netoutput1 |   |add--netoutput2|
 * |      +---+         |   |add2--+        |
 * +--------------------+   +---------------+
 *
 *                         ||
 *                         \/
 *
 *     data1
 *       |
 *       if
 *      / \
 *      op3
 *       |
 *    netoutput
 *
 * then_subgraph                    else_subgraph
 * +-----------------------------+   +---------------+
 * |data2                        |   |data3          |
 * |switch-identity+--netoutput1 |   |add--netoutput2|
 * |               +---+         |   |add2--+        |
 * +-----------------------------+   +---------------+
 */
TEST_F(UtestMemLayoutConflictNotSupportRefreshOut,
       NotSupportRefreshOutMultiReferenceAndNormalOut_InsertIdentityWhenFeatureMapRefreshable_Success) {
  auto graph = MemConflictShareGraph::BuildNotSupportRefreshOutMultiReferenceAndNormalOutGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  FeatureMapRefreshOptionGuarder guarder;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"netoutput1"}}), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 2U), GRAPH_SUCCESS);
}
/*
 *  swt1   swt2
 *   \     /
 *    concat
 */
TEST_F(UtestMemLayoutConflictNotSupportRefreshOut,
       NotSupportRefreshOutAndContinuousInput_InsertIdentityWhenFeatureMapRefreshable_Success) {
  auto graph = MemConflictShareGraph::BuildNotSupportRefreshOutAndContinuousInputGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  FeatureMapRefreshOptionGuarder guarder;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"concat", 0}, {"concat", 1}}), GRAPH_SUCCESS);
}
/*
 *  swt1   swt2
 *   \     /
 *  phony_concat
 */
TEST_F(UtestMemLayoutConflictNotSupportRefreshOut,
       NotSupportRefreshOutAndNoPaddingContinuousInput_InsertIdentityWhenFeatureMapRefreshable_Success) {
  auto graph = MemConflictShareGraph::BuildNotSupportRefreshOutAndNoPaddingContinuousInputGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  FeatureMapRefreshOptionGuarder guarder;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"phony_concat", 0}, {"phony_concat", 1}}), GRAPH_SUCCESS);
}

/*
 *    op1
 *     |
 *    if           then_sub1         else_sub1
 *    /\           +--------------+  +--------------+
 *  op3 op4        | data1        |  | data1        |
 *   |   |         |  |           |  |              |
 *  netoutput      | memcpy1      |  |              |
 *                 |  |           |  |              |
 *                 | swt1    swt2 |  | phony_split  |
 *                 |  \       /   |  |  |  |        |
 *                 |   netoutput1 |  | netoutput1   |
 *                 +--------------+  +--------------+
 */
TEST_F(UtestMemLayoutConflictNotSupportRefreshOut,
       NotSupportRefreshOutAndNoPaddingContinuousOutput_InsertIdentityWhenFeatureMapRefreshable_Success) {
  auto graph = MemConflictShareGraph::BuildNotSupportRefreshOutAndNoPaddingContinuousOutputGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  FeatureMapRefreshOptionGuarder guarder;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"netoutput1", 0}, {"netoutput1", 1}}), GRAPH_SUCCESS);
}

/*
 *    op1
 *     |
 *    if           then_sub1         else_sub1
 *    /\           +--------------+  +--------------+
 *  op3 op4        | data1        |  | data1        |
 *   |   |         |  |           |  |              |
 *  netoutput      | memcpy1      |  |              |
 *                 |  |           |  |              |
 *                 | swt1    swt2 |  | split        |
 *                 |  \       /   |  |  |  |        |
 *                 |   netoutput1 |  | netoutput1   |
 *                 +--------------+  +--------------+
 */
TEST_F(UtestMemLayoutConflictNotSupportRefreshOut,
       NotSupportRefreshOutAndContinuousOutput_InsertIdentityWhenFeatureMapRefreshable_Success) {
  auto graph = MemConflictShareGraph::BuildNotSupportRefreshOutAndContinuousOutputGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  FeatureMapRefreshOptionGuarder guarder;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"netoutput1", 0}, {"netoutput1", 1}}), GRAPH_SUCCESS);
}
/*
 *    op1                                             op1
 *     |                                               |
 *    if        then_sub1       else_sub1             if        then_sub1       else_sub1
 *     |        +------------+  +------------+         |        +------------+  +------------+
 *    op3       |   data1    |  | data1      |        op3       |   data1    |  | data1      |
 *     |        |    |       |  |            | ==>     |        |    |       |  |            |
 *  netoutput   |   memcpy1  |  |            |      netoutput   |   memcpy1  |  |            |
 *              |    |       |  |            |                  |    |       |  |            |
 *              |   swt1     |  |    swt2    |                  |   swt1     |  |    swt2    |
 *              |    |       |  |     |      |                  |    |       |  |     |      |
 *              | netoutput1 |  | netoutput2 |                  | identity   |  |  identity  |
 *              +------------+  +------------+                  |    |       |  |     |      |
 *                                                              | netoutput1 |  | netoutput2 |
 *                                                              +------------+  +------------+
 */
TEST_F(UtestMemLayoutConflictNotSupportRefreshOut,
       NotSupportRefreshOutAndNotSupportRefreshOut_InsertIdentityWhenFeatureMapRefreshable_Success) {
  auto graph = MemConflictShareGraph::BuildNotSupportRefreshOutAndNotSupportRefreshOutGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  FeatureMapRefreshOptionGuarder guarder;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 2), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"netoutput1"}, {"netoutput2"}}), GRAPH_SUCCESS);
}
/*
 *   data         data
 *    |            |
 *    a            a
 *    |            |
 *   swt1        identity
 *    |            |
 *   swt2    ==>  swt1
 *    |            |
 *    b          identity
 *    |            |
 *  netoutput     swt2
 *                 |
 *               identity
 *                 |
 *               Netoutput
 */
TEST_F(UtestMemLayoutConflictNotSupportRefreshOut,
       NotSupportRefreshOutAndNotSupportRefreshOutByRef_InsertIdentityWhenFeatureMapRefreshable_Success) {
  auto graph = MemConflictShareGraph::BuildNotSupportRefreshOutAndNotSupportRefreshOutByRefGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  FeatureMapRefreshOptionGuarder guarder;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 3), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"swt1", 0}, {"swt2", 0}, {"b", 0}}), GRAPH_SUCCESS);
}
/*
 *   data         data
 *    |            |
 *    a            a
 *    |            |
 *   swt1        identity
 *    |            |
 *   swt2    ==>  swt1
 *    |            |
 *    b          identity
 *    |            |
 *  netoutput     swt2
 *                 |
 *               identity
 *                 |
 *               Netoutput
 */
TEST_F(UtestMemLayoutConflictNotSupportRefreshOut,
       NotSupportRefreshOutAndNotSupportRefreshIn_InsertIdentityWhenFeatureMapRefreshable_Success) {
  auto graph = MemConflictShareGraph::BuildNotSupportRefreshOutAndNotSupportRefreshInGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  FeatureMapRefreshOptionGuarder guarder;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 3), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"swt1", 0}, {"swt2", 0}, {"b", 0}}), GRAPH_SUCCESS);
}

/*
 *  const     const
 *    |         |
 *   swt ==> identity
 *    |         |
 *    a        swt
 *              |
 *            identity
 *              |
 *              a
 */
TEST_F(UtestMemLayoutConflictNotSupportRefreshOut,
       ImmutableOutAndNotSupportedRefreshOutByRef_InsertIdentityWhenFeatureMapRefreshable_Success) {
  auto graph = MemConflictShareGraph::BuildImmutableOutAndNotSupportedRefreshOutByRefGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  FeatureMapRefreshOptionGuarder guarder;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 2), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"swt", 0}, {"a", 0}}), GRAPH_SUCCESS);
}
/*
 *                   const  swt
 *                     |     |
 *   const  swt    ientity identity
 *     \    /   ==>    \    /
 *   phonyconcat     phonyconcat
 */
TEST_F(UtestMemLayoutConflictNotSupportRefreshOut,
       ImmutableOutAndNotSupportedRefreshOutByContinuousIn_InsertIdentityWhenFeatureMapRefreshable_Success) {
  auto graph = MemConflictShareGraph::BuildImmutableOutAndNotSupportedRefreshOutByContinuousInGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  FeatureMapRefreshOptionGuarder guarder;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 2), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"phonyconcat", 0}, {"phonyconcat", 1}}), GRAPH_SUCCESS);
}


} // namespace ge
