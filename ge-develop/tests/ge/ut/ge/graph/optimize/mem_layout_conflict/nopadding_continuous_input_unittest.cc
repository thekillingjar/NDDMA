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
#include "ge_local_context.h"

namespace ge {
class UtestMemLayoutConflictNoPaddingContinuousIn : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};
/*
 *   a             b
 *   /\            /\
 *  | phony_concat1  |
 *  \                /
 *    phony_concat2
 */
TEST_F(UtestMemLayoutConflictNoPaddingContinuousIn,
       NoPaddingContinuousInAndNoPaddingContinuousIn_NotInsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildNoPaddingContinuousInAndNoPaddingContinuousInGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 0U), GRAPH_SUCCESS);
}
/*
 *    a        b
 *     \      /
 * c  phony_concat1  d
 *  \      |        /
 *    phony_concat2
 */
TEST_F(UtestMemLayoutConflictNoPaddingContinuousIn,
       NoPaddingContinuousInAndNoPaddingContinuousInByRef_NotInsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildNoPaddingContinuousInAndNoPaddingContinuousInByRefGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 0U), GRAPH_SUCCESS);
}
/*
 *  a         phony_split              phony_split
 *   \         / \                      / \
 *  phony_concat  b         a    identity  b
 *                           \         /
 *                          phony_concat
 */
TEST_F(UtestMemLayoutConflictNoPaddingContinuousIn,
       NoPaddingContinuousInAndNoPaddingContinuousOut_InsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildNoPaddingContinuousInAndNoPaddingContinuousOutGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"phony_concat", 1}}), GRAPH_SUCCESS);
}

/*
 *    a    b
 *     \   /
 *  PhonyConcat
 *       |
 *   PhonySplit
 *      /  \
 *     c    d
 */
TEST_F(UtestMemLayoutConflictNoPaddingContinuousIn,
       NoPaddingContinuousInAndNoPaddingContinuousOutConnect_NotInsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildNoPaddingContinuousInAndNoPaddingContinuousOutConnectGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 0U), GRAPH_SUCCESS);
}

/*
 *   a            b          a                 b
 *   /\1         0/\         /\                /\
 *  | phony_concat1 |  ==>  |   \            /   |
 *  \0            1/        \     \1       0/     |
 *    phony_concat2    identity  phony_concat1  identity
 *                            \0              1/
 *                               phony_concat2
 */
TEST_F(UtestMemLayoutConflictNoPaddingContinuousIn,
       NoPaddingContinuousInAndNoPaddingContinuousInCross_InsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildNoPaddingContinuousInAndNoPaddingContinuousInCrossGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  dlog_setlevel(GE_MODULE_NAME, 1, 0); // 有些代码必须开info日志才能走到
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 2), GRAPH_SUCCESS);

  const auto concat1 = graph->FindNode("phony_concat1");
  ASSERT_NE(concat1, nullptr);
  const auto concat2 = graph->FindNode("phony_concat2");
  ASSERT_NE(concat2, nullptr);
  if (concat1->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetType() == IDENTITY) {
    EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"phony_concat1", 1}, {"phony_concat1", 0}}), GRAPH_SUCCESS);
  } else {
    EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"phony_concat2", 1}, {"phony_concat2", 0}}), GRAPH_SUCCESS);
  }
  bool check_success = false;
  if ((concat1->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetType() == IDENTITY)
      || (concat1->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetType() == IDENTITY)
      ||(concat2->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetType() == IDENTITY)
      || (concat2->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetType() == IDENTITY)) {
    check_success = true;
  }
  EXPECT_TRUE(check_success);
  dlog_setlevel(GE_MODULE_NAME, 3, 0); // 有些代码必须开info日志才能走到
}
/*
 *     a b c d      c d
 *     | | | |      | |
 *   phonyconcat1 phonyconcat2
 */
TEST_F(UtestMemLayoutConflictNoPaddingContinuousIn,
       NoPaddingContinuousInAndNoPaddingContinuousInPartialSameInputs_NotInsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildNoPaddingContinuousInAndNoPaddingContinuousInPartialSameInputsGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 0), GRAPH_SUCCESS);
}
/*
 *     a b c d      a d
 *     | | | |      | |
 *   phonyconcat1 phonyconcat2
 */
TEST_F(UtestMemLayoutConflictNoPaddingContinuousIn,
       NoPaddingContinuousInAndNoPaddingContinuousInPartialSameInputsConflict_InsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildNoPaddingContinuousInAndNoPaddingContinuousInPartialSameInputsConflictGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 2), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"phony_concat2"}, {"phony_concat2", 1}}), GRAPH_SUCCESS);
}

/*
 *     a b c d      e f c d
 *     | | | |      | | | |
 *   phonyconcat1 phonyconcat2
 */
TEST_F(UtestMemLayoutConflictNoPaddingContinuousIn,
       NoPaddingContinuousInAndNoPaddingContinuousInTailPartialSameInputs_InsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildNoPaddingContinuousInAndNoPaddingContinuousInTailPartialSameInputsGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 2), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"phony_concat2", 2}, {"phony_concat2", 3}}), GRAPH_SUCCESS);
}
/*
 *     a b c d      e b
 *     | | | |      | |
 *   phonyconcat1 phonyconcat2
 */
TEST_F(UtestMemLayoutConflictNoPaddingContinuousIn,
       NoPaddingContinuousInAndNoPaddingContinuousInTailIntersectionG_InsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildNoPaddingContinuousInAndNoPaddingContinuousInTailIntersectionGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 1), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"phony_concat2", 1}}), GRAPH_SUCCESS);
}

/*
 *        a   b
 *         \ /
 *     c  PhonyConcat
 *     \    |
 *      hcom
 *      |   |
 *      d
 */
TEST_F(UtestMemLayoutConflictNoPaddingContinuousIn,
       NoPaddingContinuousInAndRtsSpecailInByRefAndOutAnchorSuspended_NotInsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildNoPaddingContinuousInAndRtsSpecailInByRefAndOutAnchorSuspendedGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 0U), GRAPH_SUCCESS);
}

/*
 *              data
 *               /\
 *  assign_slice0 assign_slice1 (inplace)
 *             \   /
 *              pc
 *              |
 *              b
 *  aot调优会导致这种结构，data只有一个输出，但是assign_slice1会设置input offset，GE也做了配合，
 *  最终结果是data的输出内存分为2半，一半给assign_slice0，一半给assign_slice1. 后面再通过PhonyConcat节点合并。
 *  因此PhonyConcat的输入虽然是用户输入，但是所有输入都是同一个Data节点，可以不插入identity.
 *
 *  另外，GE会校验PhonyConcat的输入节点有没有打_reuse_input_on_dim_index属性，中间插入的identity没有这个属性，会校验报错。
 *  更重要的是，输入节点可能在一块大的内存中会跳写，若后面跟着identity，就无法分配大的连续内存了。
 */
TEST_F(UtestMemLayoutConflictNoPaddingContinuousIn,
       BuildNoPaddingContinuousInWithSameAnchor_NotInsertIdentity) {
  auto graph = MemConflictShareGraph::BuildNoPaddingContinuousInWithSameAnchorData();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 0U), GRAPH_SUCCESS);
}

/*
 *              var
 *               /\
 *  assign_slice0 assign_slice1 (inplace)
 *             \   /
 *              pc
 *              |
 *              b
 *  aot调优会导致这种结构，var只有一个输出，但是assign_slice1会设置input offset，GE也做了配合，
 *  最终结果是var的输出内存分为2半，一半给assign_slice0，一半给assign_slice1. 后面再通过PhonyConcat节点合并。
 *  因此PhonyConcat的输入虽然是用户输入，但是所有输入都是同一个var节点，可以不插入identity.
 *
 *  另外，GE会校验PhonyConcat的输入节点有没有打_reuse_input_on_dim_index属性，中间插入的identity没有这个属性，会校验报错。
 *  更重要的是，输入节点可能在一块大的内存中会跳写，若后面跟着identity，就无法分配大的连续内存了。
 */
TEST_F(UtestMemLayoutConflictNoPaddingContinuousIn,
       BuildNoPaddingContinuousInWithSameAnchorVariable_NotInsertIdentity) {
  auto graph = MemConflictShareGraph::BuildNoPaddingContinuousInWithSameAnchorVariable();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 0U), GRAPH_SUCCESS);
}
} // namespace ge
