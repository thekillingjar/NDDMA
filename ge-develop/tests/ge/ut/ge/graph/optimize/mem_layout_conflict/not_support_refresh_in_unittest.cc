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
#include "compiler/graph/optimize/mem_layout_conflict_optimize/mem_layout_conflict_optimizer.h"
#include "ge_context.h"

namespace ge {
class UtestMemLayoutConflictNotSupportRefreshIn : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

/*
 *    op1
 *     +--hcomallreduce (p2p input)
 *     |
 *   switch
 *
 *   ||
 *   \/
 *
 *    op1
 *     +--hcomallreduce (p2p input)
 *     |
 *   switch
 */
TEST_F(UtestMemLayoutConflictNotSupportRefreshIn, NotSupportRefreshInAndRtsSpecailIn_NotInsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildNotSupportRefreshInAndRtsSpecialInGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 0U), GRAPH_SUCCESS);
}

/*
 *    op1
 *     +--hcomallreduce (p2p input)
 *     |
 *   switch
 *
 *   ||
 *   \/
 *
 *    op1
 *     +--hcomallreduce (p2p input)
 *     |
 *   identity
 *     |
 *   switch
 */
TEST_F(UtestMemLayoutConflictNotSupportRefreshIn,
       NotSupportRefreshInAndRtsSpecailIn_InsertIdentityWhenFeatureMapRefreshable_Success) {
  auto graph = MemConflictShareGraph::BuildNotSupportRefreshInAndRtsSpecialInGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  FeatureMapRefreshOptionGuarder fm_refresh_guarder;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"switch"}}), GRAPH_SUCCESS);
}

/*
 *   hcomallreduce (p2p output)
 *     |
 *   swtich
 *
 *   ||
 *   \/
 *
 *   hcomallreduce (p2p output)
 *     |
 *   swtich
 */
TEST_F(UtestMemLayoutConflictNotSupportRefreshIn, NotSupportRefreshInAndRtsSpecailOut_NotInsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildNotSupportRefreshInAndRtsSpecialOutGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 0U), GRAPH_SUCCESS);
}

/*
 *   hcomallreduce (p2p output)
 *     |
 *   swtich
 *
 *   ||
 *   \/
 *
 *   hcomallreduce (p2p output)
 *     |
 *   identity
 *     |
 *   swtich
 */
TEST_F(UtestMemLayoutConflictNotSupportRefreshIn,
       NotSupportRefreshInAndRtsSpecailOut_InsertIdentityWhenFeatureMapRefreshable_Success) {
  auto graph = MemConflictShareGraph::BuildNotSupportRefreshInAndRtsSpecialOutGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  FeatureMapRefreshOptionGuarder fm_refresh_guarder;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"switch"}}), GRAPH_SUCCESS);
}

/*
 *    add
 *     |
 *   swtich
 *
 *   ||
 *   \/
 *
 *    add
 *     |
 *   swtich
 */
TEST_F(UtestMemLayoutConflictNotSupportRefreshIn, NotSupportRefreshInAndNoramlOut_NotInsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildNotSupportRefreshInAndNormalOutGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 0U), GRAPH_SUCCESS);
}

/*
 *    add
 *     |
 *   swtich
 *
 *   ||
 *   \/
 *
 *    add
 *     |
 *   identity
 *     |
 *   swtich
 */
TEST_F(UtestMemLayoutConflictNotSupportRefreshIn,
       NotSupportRefreshInAndNoramlOut_InsertIdentityWhenFeatureMapRefreshable_Success) {
  auto graph = MemConflictShareGraph::BuildNotSupportRefreshInAndNormalOutGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  FeatureMapRefreshOptionGuarder fm_refresh_guarder;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"switch"}}), GRAPH_SUCCESS);
}

/*
 *    add
 *     |
 *   swtich
 *
 *   ||
 *   \/
 *
 *    add
 *     |
 *   identity
 *     |
 *   swtich
 */
TEST_F(UtestMemLayoutConflictNotSupportRefreshIn,
       NotSupportRefreshInAndNoramlOut_InsertIdentityWhenDynamicAndStaticGraphMemoryReuse_Success) {
  auto graph = MemConflictShareGraph::BuildNotSupportRefreshInAndNormalOutGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  StaticMemoryPolicy4Guarder reuse_guarder;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"switch"}}), GRAPH_SUCCESS);
}

/*
 *    add
 *     |
 *   hcom
 */
TEST_F(UtestMemLayoutConflictNotSupportRefreshIn,
       NotSupportRefreshInAndNoramlOut_NotInsertIdentityWhenDynamicAndStaticGraphMemoryReuse_Success) {
  auto graph = MemConflictShareGraph::BuildPhysicalRefreshableInAndNormalOutGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  StaticMemoryPolicy4Guarder reuse_guarder;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 0), GRAPH_SUCCESS);
}

/*
 * unknown graph
 *      op1
 *       |               known sub graph
 * partitioned_call      +-----------+
 *       |               |   data    |
 *      op2              |    |      |
 *                       |   add1    |
 *                       |    |      |
 *                       |streamswitch|
 *                       |    |      |
 *                       |   add2    |
 *                       |    |      |
 *                       | netoutput |
 *                       +-----------+
 *
 *
 * unknown graph
 *      op1
 *       |               known sub graph
 * partitioned_call      +-----------+
 *       |               |   data    |
 *      op2              |    |      |
 *                       |   add1    |
 *                       |    |      |
 *                       |   identity|
 *                       |    |      |
 *                       |streamswitch|
 *                       |    |      |
 *                       |   add2    |
 *                       |    |      |
 *                       | netoutput |
 *                       +-----------+
 *
 */
TEST_F(UtestMemLayoutConflictNotSupportRefreshIn, NotSupportRefreshInAndNoramlOut_InsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildNotSupportRefreshInAndNormalOutInKnowSubgraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"switch"}}), GRAPH_SUCCESS);
}

/*
 *   swt1  swt3
 *    /\     /\
 * swt2 concat swt4
 */
TEST_F(UtestMemLayoutConflictNotSupportRefreshIn, NotSupportRefreshInAndContinuousInput_InsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildNotSupportRefreshInputAndContinuousInputGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  FeatureMapRefreshOptionGuarder fm_refresh_guarder;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"concat", 0}, {"concat", 1}}), GRAPH_SUCCESS);
}
/*
 *    a            b
 *    /\           /\
 * swt2 phony_concat swt4
 */
TEST_F(UtestMemLayoutConflictNotSupportRefreshIn, NotSupportRefreshInAndNoPaddingContinuousInput_InsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildNotSupportRefreshInputAndNoPaddingContinuousInputGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  FeatureMapRefreshOptionGuarder fm_refresh_guarder;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"swt2", 0}, {"swt4", 0}}), GRAPH_SUCCESS);
}
/*
 *    split
 *     /\
 *  swt1 swt2
 */
TEST_F(UtestMemLayoutConflictNotSupportRefreshIn, NotSupportRefreshInAndContinuousOutput_InsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildNotSupportRefreshInputAndContinuousOutputGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  FeatureMapRefreshOptionGuarder fm_refresh_guarder;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"swt1", 0}, {"swt1", 0}}), GRAPH_SUCCESS);
}
/*
 *  phony_split
 *     / \
 *  swt1  swt2
 */
TEST_F(UtestMemLayoutConflictNotSupportRefreshIn, NotSupportRefreshInAndNoPaddingContinuousOutnput_InsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildNotSupportRefreshInputAndNoPaddingContinuousOutputGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  FeatureMapRefreshOptionGuarder fm_refresh_guarder;

  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"swt1", 0}, {"swt2", 0}}), GRAPH_SUCCESS);
}

/*
 *   const     const
 *    |   ==>   |
 *   swt       identity
 *              |
 *             swt
 */
TEST_F(UtestMemLayoutConflictNotSupportRefreshIn, ImmutableOutAndNotSupportedRefreshIn_InsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildImmutableOutAndNotSupportedRefreshInGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  FeatureMapRefreshOptionGuarder fm_refresh_guarder;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"swt", 0},}), GRAPH_SUCCESS);
}
} // namespace ge
