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
class UtestMemLayoutConflictRtsSpecailIn : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

/*
 *            add
 *           /     \
 *  hcomallreduce rtMemCpyAsync
 *   (p2p in)       (RT_MEMORY_TS input)
 *
 *              ||
 *              \/
 *
 *            add
 *           /     \
 *       identity   \
 *         /         \
 *  hcomallreduce rtMemCpyAsync
 *   (p2p in)       (RT_MEMORY_TS input)
 */
TEST_F(UtestMemLayoutConflictRtsSpecailIn, RtsSpecailInAndRtsSpecailInNotSame_InsertTwoIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildRtsSpecialInAndRtsSpecialInGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  auto hcomallreduce = graph->FindNode("hcomallreduce");
  auto rtMemCpyAsync = graph->FindNode("rtMemCpyAsync");
  if (hcomallreduce->GetOpDesc()->GetId() < rtMemCpyAsync->GetOpDesc()->GetId()) {
    EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"hcomallreduce"}}), GRAPH_SUCCESS);
  } else {
    EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"rtMemCpyAsync"}}), GRAPH_SUCCESS);
  }
}

/*
 *            add
 *           /     \
 *  hcomallreduce hcomallreduce2
 *   (p2p in)       (p2p in)
 *
 *              ||
 *              \/
 *
 *            add
 *           /     \
 *  hcomallreduce hcomallreduce2
 *   (p2p in)       (p2p in)
 */
TEST_F(UtestMemLayoutConflictRtsSpecailIn, RtsSpecailInAndRtsSpecailInSameType_NotInsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildRtsSpecialInAndRtsSpecialInSameTypeGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 0U), GRAPH_SUCCESS);
}

/*
 *   rtMemCpyAsync (RT_MEMORY_TS out)
 *      |
 *  hcomallreduce  (p2p in)
 *
 *     ||
 *     \/
 *
 *   rtMemCpyAsync (RT_MEMORY_TS out)
 *      |
 *     idenity
 *      |
 *  hcomallreduce  (p2p in)
 */
TEST_F(UtestMemLayoutConflictRtsSpecailIn, RtsSpecailInAndRtsSpecailOut_InsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildRtsSpecialInAndRtsSpecialOutGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"hcomallreduce"}}), GRAPH_SUCCESS);
}

/*
 *   hcomallreduce1 (p2p out)
 *      |
 *  hcomallreduce2  (p2p in)
 */
TEST_F(UtestMemLayoutConflictRtsSpecailIn, RtsSpecailInAndRtsSpecailOutSameType_NotInsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildRtsSpecialInAndRtsSpecialOutSameTypeGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 0U), GRAPH_SUCCESS);
}

/*
 *     add
 *      |
 *  hcomallreduce  (p2p in)
 */
TEST_F(UtestMemLayoutConflictRtsSpecailIn, RtsSpecailInAndNormalOut_NotInsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildRtsSpecialInAndNormalOutGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 0U), GRAPH_SUCCESS);
}

/*
*   a         a
 *  |         |
 * hcom1 ==> hcom1
 *  |         |
 *  b       identity
 *            |
 *            b
*/
TEST_F(UtestMemLayoutConflictRtsSpecailIn, RtsSpecialInAndRtsSpecialInByRef_InsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildRtsSpecialInAndRtsSpecialInByRefGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 1U), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"b"}}), GRAPH_SUCCESS);
}

/*
 *              a
 *              |
 *   a        identity
 *   |          |
 *  hcom1 ==>  hcom1
 *   |          |
 *  hcom2      hcom2
 */
TEST_F(UtestMemLayoutConflictRtsSpecailIn, RtsSpecialInAndRtsSpecialOutByRef_InsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildRtsSpecialInAndRtsSpecialOutByRefGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 1U), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"hcom1"}}), GRAPH_SUCCESS);
}
} // namespace ge
