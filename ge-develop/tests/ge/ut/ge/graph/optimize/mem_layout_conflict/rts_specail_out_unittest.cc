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
class UtestMemLayoutConflictRtsSpecailOut : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

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
 * +---------------------------+   +------------------------+
 * | data2                     |   | data3                  |
 * |rtMemCpyAsync--netoutput1  |   |hcomallreduce-netoutput2|  hcomallreduce need p2p output
 * +---------------------------+   +------------------------+
 */
TEST_F(UtestMemLayoutConflictRtsSpecailOut, RtsSpecailOutAndRtsSpecailOut_InsertOneIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildRtsSpecialOutAndRtsSpecialOutGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  NodePtr netoutput1 = nullptr;
  NodePtr netoutput2 = nullptr;
  for (auto &node : graph->GetAllNodes()) {
    if (node->GetName() == "netoutput1") {
      netoutput1 = node;
    }
    if (node->GetName() == "netoutput2") {
      netoutput2 = node;
    }
  }
  ASSERT_NE(netoutput1, nullptr);
  ASSERT_NE(netoutput2, nullptr);
  if (netoutput1->GetOpDesc()->GetId() < netoutput2->GetOpDesc()->GetId()) {
    EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"netoutput1"}}), GRAPH_SUCCESS);
  } else {
    EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"netoutput2"}}), GRAPH_SUCCESS);
  }
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 1U), GRAPH_SUCCESS);
}

/*
 *    a
 *    |
 *   hcom1 (p2p in/out)
 *    |
 *   hcom2 (p2p in/out, output ref input)
 *    |
 *    b
 *    |
 *  netoutput
 *  
 *  not conflict, not insert identity
 */
TEST_F(UtestMemLayoutConflictRtsSpecailOut, RtsSpecialOutAndRtsSpecialOutByRef_SameTypeNotInsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildRtsSpecialOutAndRtsSpecialOutByRefGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 0U), GRAPH_SUCCESS);
}
} // namespace ge
