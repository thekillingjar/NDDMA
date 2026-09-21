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
class UtestMemLayoutConflictNoPaddingContinuousOut : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

/*
 *   phony_split1         phony_split1
 *         /  \                 /  \
 * phony_plit2  c ==>     identity  c
 *     /   \                /
 *    a     b       phony_plit2
 *                      /   \
 *                     a     b
 */
TEST_F(UtestMemLayoutConflictNoPaddingContinuousOut,
       NoPaddingContinuousOutAndNoPaddingContinuousOut_InsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildNoPaddingContinuousOutAndNoPaddingContinuousOutGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 1), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"phony_plit2"}}), GRAPH_SUCCESS);
}
/*
 *       c             c
 *       |             |
 * phony_plit ==> phony_plit
 *     /   \         /   \
 *    a     b       a     b
 */
TEST_F(UtestMemLayoutConflictNoPaddingContinuousOut,
       NoPaddingContinuousOutAndNormalOut_NotInsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildNoPaddingContinuousOutAndNormalOutGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 0), GRAPH_SUCCESS);
}
} // namespace ge
