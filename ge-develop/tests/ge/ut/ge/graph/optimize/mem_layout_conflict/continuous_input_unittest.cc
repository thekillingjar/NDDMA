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
#include "runtime/mem.h"

namespace ge {
class UtestMemLayoutConflictContinuousIn : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};
/*
 *   a       b           a                b
 *   /\      /\          /\               /\
 *  | concat1  | ==>    | identity identity |
 *  \          /        \        \  /      /
 *    concat2             \     concat1  /
 *                           \         /
 *                             concat2
 */
TEST_F(UtestMemLayoutConflictContinuousIn, ContinuousInAndContinuousIn_InsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildContinuousInAndContinuousInGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 2), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"concat2"}, {"concat2", 1}}), GRAPH_SUCCESS);
}
/*
 *   a       b         a       b
 *   \      /          \      /
 * c  concat1  d ==> c  concat1  d
 *  \   |      /     |    |      |
 *    concat2        \  identity /
 *                    \   |     /
 *                      concat2
 */
TEST_F(UtestMemLayoutConflictContinuousIn, ContinuousInAndContinuousInByRef_InsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildContinuousInAndContinuousInByRefGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 1), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"concat2", 1}}), GRAPH_SUCCESS);
}

/*
 *  a  split
 *  \   / \
 *  concat  b
 */
TEST_F(UtestMemLayoutConflictContinuousIn, ContinuousInAndContinuousOut_NotInsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildContinuousInAndContinuousOutGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 0), GRAPH_SUCCESS);
}

/*
 *  split
 *  /   \  |  /
 *       concat
 */
TEST_F(UtestMemLayoutConflictContinuousIn, ContinuousInHeadSameWithContinuousOutHeadGraph_NotInsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildContinuousInHeadSameWithContinuousOutTailGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 0), GRAPH_SUCCESS);
}

/*
 *          split
 *    \  |  /  \
 *      concat
 */
TEST_F(UtestMemLayoutConflictContinuousIn, ContinuousInTailSameWithContinuousOutHead_NotInsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildContinuousInTailSameWithContinuousOutHeadGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 0), GRAPH_SUCCESS);
}

/*
 * 不冲突场景1：子集
 * node_a所有输出anchor     ： anchor1 anchor2 anchor3 anchor4
 * node_b所有输入anchor的对端：         anchor2 anchor3
 */
TEST_F(UtestMemLayoutConflictContinuousIn, ContinuousInIsSubSetOfContinuousOut_NotInsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildContinuousInIsSubSetOfContinuousOutGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 0), GRAPH_SUCCESS);
}

/*
 * 不冲突场景1：子集
 * node_a所有输出anchor     ：         anchor2 anchor3
 * node_b所有输入anchor的对端： anchor1 anchor2 anchor3 anchor4
 */
TEST_F(UtestMemLayoutConflictContinuousIn, ContinuousOutIsSubSetOfContinuousIn_NotInsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildContinuousOutIsSubSetOfContinuousInGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 0), GRAPH_SUCCESS);
}

/*
 * 不冲突场景2：完全一样
 * node_a所有输出anchor     ： anchor1 anchor2 anchor3
 * node_b所有输入anchor的对端： anchor1 anchor2 anchor3
 */
TEST_F(UtestMemLayoutConflictContinuousIn, ContinuousInIsTheSameAsContinuousIn_NotInsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildContinuousInIsTheSameAsContinuousOutGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 0), GRAPH_SUCCESS);
}

/*
 * 冲突
 * node_a所有输出anchor     ： anchor1 anchor2
 * node_b所有输入anchor的对端：         anchor2 anchor1
 */
TEST_F(UtestMemLayoutConflictContinuousIn, ContinuousInReverseWithContinuousOut_InsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildContinuousInReverseWithContinuousOutGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 1), GRAPH_SUCCESS);
}

/*
 * 冲突
 * node_a所有输出anchor     ：                  anchor1 anchor2
 * node_b所有输入anchor的对端：  anchor3 anchor2 anchor1
 */
TEST_F(UtestMemLayoutConflictContinuousIn, ContinuousInReverseWithContinuousOut2_InsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildContinuousInReverseWithContinuousOut2Graph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 2), GRAPH_SUCCESS);
}

/*
 * 冲突
 * node_a所有输出anchor     ： anchor1 anchor2 anchor3 anchor4
 * node_b所有输入anchor的对端：         anchor2         anchor4
 */
TEST_F(UtestMemLayoutConflictContinuousIn, ContinuousInAndContinuousOutPartialSame1_InsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildContinuousInAndContinuousOutPartialSame1Graph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 2), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"concat", 0}, {"concat", 1}}), GRAPH_SUCCESS);
}

/*
 * 冲突
 * node_a所有输出anchor     ： anchor1 anchor2 anchor3 anchor4
 * node_b所有输入anchor的对端：         anchor2                 anchor5
 */
TEST_F(UtestMemLayoutConflictContinuousIn, ContinuousInAndContinuousOutPartialSame2_InsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildContinuousInAndContinuousOutPartialSame2Graph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 1), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"concat", 0}}), GRAPH_SUCCESS);
}

/*
 * 冲突
 * node_a所有输出anchor     ： anchor1 anchor2 anchor3 anchor4
 * node_b所有输入anchor的对端：                                 anchor5  anchor2
 */
TEST_F(UtestMemLayoutConflictContinuousIn, ContinuousInAndContinuousOutPartialSame3_InsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildContinuousInAndContinuousOutPartialSame3Graph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 1), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"concat", 1}}), GRAPH_SUCCESS);
}

/*
 * 冲突
 * node_a所有输出anchor     ：                anchor1 anchor2 anchor3 anchor4
 * node_b所有输入anchor的对端： anchor5 anchor6                anchor3 anchor4
 */
TEST_F(UtestMemLayoutConflictContinuousIn, ContinuousInAndContinuousOutPartialSame4_InsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildContinuousInAndContinuousOutPartialSame4Graph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 2), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"concat", 2}, {"concat", 3}}), GRAPH_SUCCESS);
}

/*
 *   a       b          a       b
 *   /\      /\         /\      /\
 *  | concat   | ==>   | concat   |
 *  \          /     identity   identity
 *   phony_concat       \        /
 *                     phony_concat
 */
TEST_F(UtestMemLayoutConflictContinuousIn, ContinuousInAndNoPaddingContinuousIn_InsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildContinuousInAndNoPaddingContinuousInGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 2), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"phony_concat"}, {"phony_concat", 1}}), GRAPH_SUCCESS);
}
/*
 *   a       b           a       b
 *   \      /            \      /
 * c  concat   d  ==>  c  concat   d
 *  \   |      /       |   |       |
 *   phony_concat      \  identity /
 *                      \  |      /
 *                       phony_concat
 */
TEST_F(UtestMemLayoutConflictContinuousIn, ContinuousInAndNoPaddingContinuousInByRef_InsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildContinuousInAndNoPaddingContinuousInByRefGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 1), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"phony_concat", 1}}), GRAPH_SUCCESS);
}
/*
 *  a  phony_split
 *  \   / \
 *  concat  b
 */
TEST_F(UtestMemLayoutConflictContinuousIn, ContinuousInAndNoPaddingContinuousOut_InsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildContinuousInAndNoPaddingContinuousOutGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"concat", 1}}), GRAPH_SUCCESS);
}
/*
 *   a   b                         b
 *   \   /\    ==>                 /\
 *  concat hcom1         a  identity hcom1
 *                        \   /
 *                       concat
 */
TEST_F(UtestMemLayoutConflictContinuousIn, ContinuousInAndRtsSpecailIn_InsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildContinuousInAndRtsSpecailInGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 1U), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"concat", 1}}), GRAPH_SUCCESS);
}
/*
 *                        hcom1
 *                          |
 *   a    hcom1      a   identity
 *     \  /    ==>    \  /
 *     concat         concat
 */
TEST_F(UtestMemLayoutConflictContinuousIn, ContinuousInAndRtsSpecailOut_InsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildContinuousInAndRtsSpecailOutGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 1U), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"concat", 1}}), GRAPH_SUCCESS);
}
/*
 *   a   b       a   b
 *    \ /         \ /
 *    hcom1 ==>   hcom1
 *     |           |
 *   switch      identity
 *                 |
 *                switch
 */
TEST_F(UtestMemLayoutConflictContinuousIn, ContinuousInAndRtsSpecailInByRef_InsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildContinuousInAndRtsSpecailInByRefGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 1U), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"switch"}}), GRAPH_SUCCESS);
}
} // namespace ge
