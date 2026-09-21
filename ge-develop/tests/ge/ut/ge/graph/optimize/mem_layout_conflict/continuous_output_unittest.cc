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
class UtestMemLayoutConflictContinuousOut : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};
/*
 *        split1             split1
 *         /  \               /  \
 *     split2  c  ==>    identity c
 *     /   \                |
 *    a     b             split2
 *                        /   \
 *                       a     b
 */
TEST_F(UtestMemLayoutConflictContinuousOut, ContinuousOutAndContinuousOutByRef_InsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildContinuousOutAndContinuousOutByRefGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"split2"}}), GRAPH_SUCCESS);
}
/*
 *  split              split
 *  |  |       ==>     |   |
 * phony_concat   identity identity
 *                   |        |
 *                  phony_concat
 */
TEST_F(UtestMemLayoutConflictContinuousOut, ContinuousOutAndNoPaddingContinuousIn_InsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildContinuousOutAndNoPaddingContinuousInGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"phony_concat"}, {"phony_concat", 1}}), GRAPH_SUCCESS);
}
/*
 *        split              split
 *         /  \               /  \
 * phony_plit  c ==>    identity  c
 *     /   \               |
 *    a     b           phony_plit
 *                        /   \
 *                       a     b
 */
TEST_F(UtestMemLayoutConflictContinuousOut, ContinuousOutAndNoPaddingContinuousOutByRef_InsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildContinuousOutAndNoPaddingContinuousOutByRefGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 1U), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"phony_plit"}}), GRAPH_SUCCESS);
}
} // namespace ge
