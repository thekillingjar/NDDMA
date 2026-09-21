/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <random>
#include <gtest/gtest.h>
#include "common/mem_conflict_share_graph.h"
#include "result_checker.h"
#include "graph/compute_graph.h"
#include "compiler/graph/optimize/mem_layout_conflict_optimize/mem_layout_conflict_optimizer.h"

namespace ge {
class UtestMemLayoutConflictUserOutput : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

/*
 *    add
 *     /\
 *   netoutput
 *
 *     ||
 *     \/
 *
 *    add
 *     /\
 *   netoutput
 *
 */
TEST_F(UtestMemLayoutConflictUserOutput, UserOutAndUserOut_NodeInserIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildUserOutAndUserOutGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 0U), GRAPH_SUCCESS);
}

/*
 *      add
 *       /\
 *   partitioned_call1  +-------------+
 *      |  |            | data1  data2|
 *     netoutput        |  |      |   |
 *                      |  netoutput1 |
 *                      +-------------+
 *
 *                   ||
 *                   \/
 *      add
 *       /\
 *   partitioned_call1  +-------------+
 *      |  |            | data1  data2|
 *     netoutput        |  |      |   |
 *                      |  netoutput1 |
 *                      +-------------+
 */
TEST_F(UtestMemLayoutConflictUserOutput, UserOutAndUserOutThroughSubGraph_NotInserIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildUserOutAndUserOutGraphWithSubGraph();
  MemLayoutConflictOptimizer mem_check_pass;

  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 0U), GRAPH_SUCCESS);
}

/*
 * unknown graph
 *      op1
 *       |               known sub graph
 * partitioned_call      +-------------+
 *       |               |data  op3    |
 *      op2              |      / \    |
 *                       |    netoutput|
 *                       +-------------+
 *           ||
 *           \/
 *
 * unknown graph
 *      op1
 *       |               known sub graph
 * partitioned_call      +-------------+
 *       |               |data  op3    |
 *      op2              |      / \    |
 *                       |    netoutput|
 *                       +-------------+
 */
TEST_F(UtestMemLayoutConflictUserOutput, UserOutAndUserOutInSubgraph_NotInserIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildUserOutAndUserOutGraphInSubGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 0U), GRAPH_SUCCESS);
}

/*
 *      const
 *       |
 *     netoutput
 *
 *      ||
 *      \/
 *      const
 *       |
 *     identity
 *       |
 *     netoutput
 */
TEST_F(UtestMemLayoutConflictUserOutput, UserOutAndImmutableOut_InserIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildUserOutAndConstOutGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"netoutput"}}), GRAPH_SUCCESS);
}

/*
 *      variable
 *       |
 *     netoutput
 *
 *      ||
 *      \/
 *      variable
 *       |
 *     netoutput
 */
TEST_F(UtestMemLayoutConflictUserOutput, UserOutAndVariable_NotInserIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildUserOutAndVariableGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 0U), GRAPH_SUCCESS);
}

/*
 *     refata
 *       |
 *     netoutput
 *
 *      ||
 *      \/
 *     refata
 *       |
 *     netoutput
 */
TEST_F(UtestMemLayoutConflictUserOutput, UserOutAndRefdata_NotInserIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildUserOutAndRefdataGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 0U), GRAPH_SUCCESS);
}

/*
 *   partitioned_call1
 *         |           +-----------+
 *         |           |  const1   |
 *         |           |   |       |
 *         |           | netoutput1|
 *         |           +-----------+
 *       netoutput
 *
 *           ||
 *           \/
 *   partitioned_call1
 *         |           +-----------+
 *         |           |  const1   |
 *         |           |   |       |
 *         |           | netoutput1|
 *         |           +-----------+
 *        identity
 *         |
 *       netoutput
 */
TEST_F(UtestMemLayoutConflictUserOutput, UserOutAndImmutableOutInSubGraph_InserIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildUserOutAndConstOutGraphWithSubGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"netoutput"}}), GRAPH_SUCCESS);
}

/*
 * unknown graph
 *      op1
 *       |               known sub graph
 * partitioned_call      +-------------+
 *       |               |data constant|
 *      op2              |       |     |
 *                       |    netoutput|
 *                       +-------------+
 *           ||
 *           \/
 *
 * unknown graph
 *      op1
 *       |               known sub graph
 * partitioned_call      +-------------+
 *       |               |data constant|
 *      op2              |       |     |
 *                       |    netoutput|
 *                       +-------------+
 */
TEST_F(UtestMemLayoutConflictUserOutput, UserOutAndImmutableInSubgraph_NotInserIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildUserOutAndConstOutGraphInSubgraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 0U), GRAPH_SUCCESS);
}

/*
 *
 * unknown graph
 *      op1
 *       |               known sub graph
 * partitioned_call      +-----------+
 *       |               |   data    |
 *      op2              |    |      |
 *                       |    op3    |
 *                       |    |      |
 *                       |  switch   |
 *                       |    |      |
 *                       | netoutput |
 *                       +-----------+
 *                ||
 *                \/
 *
 * unknown graph
 *      op1
 *       |               known sub graph
 * partitioned_call      +-----------+
 *       |               |   data    |
 *      op2              |    |      |
 *                       |    op3    |
 *                       |    |      |
 *                       |  switch   |
 *                       |    |      |
 *                       |  identity |
 *                       |    |      |
 *                       | netoutput |
 *                       +-----------+
 */
TEST_F(UtestMemLayoutConflictUserOutput, UserOutAndNotSupportRefreshOutputInSubgraph_InserIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildUserOutAndNotSupportRefreshOutWithSubgraphDataGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"netoutput"}, {"switch"}}), GRAPH_SUCCESS);
}

/*
 *    op1
 *     +--switch (not support refresh address)
 *     |
 *   netoutput
 *
 *    ||
 *    \/
 *
 *    op1
 *     +--identity-switch (not support refresh address)
 *     |
 *   netoutput
 */
TEST_F(UtestMemLayoutConflictUserOutput, UserOutAndNotSupportRefreshIn_InserIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildUserOutAndNotSupportRefreshInGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 1), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"netoutput"}}), GRAPH_SUCCESS);
}

/*
 *    op1
 *     +--switch (not support refresh address)
 *     |
 *   netoutput
 *
 *    ||
 *    \/
 *
 *    op1
 *     +--identity-switch (not support refresh address)
 *     |
 *   netoutput
 */
TEST_F(UtestMemLayoutConflictUserOutput, UserOutAndNotSupportRefreshIn_FeatureMapBaseRefreshable_Success) {
  auto graph = MemConflictShareGraph::BuildUserOutAndNotSupportRefreshInGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  FeatureMapRefreshOptionGuarder guarder;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 1), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"switch"}}), GRAPH_SUCCESS);
}

/*
 *    a
 *    |
 * hcombroadcast 输入不支持地址刷新，输出不支持地址刷新，且输出引用输入
 *    |
 *  Netoutput
 *
 *   ||
 *   \/
 *    a
 *    |
 * hcombroadcast 输入不支持地址刷新，输出不支持地址刷新，且输出引用输入
 *    |
 *  identity
 *    |
 *  Netoutput
 *  校验点：
 *  1 hcombroadcast 前不能插入拷贝。
 *  2 hcombroadcast 后面得插入拷贝。
 */
TEST_F(UtestMemLayoutConflictUserOutput, OutAndHcomBroadcast_FeatureMapBaseNotRefreshable_Success) {
  auto graph = MemConflictShareGraph::BuildUserOutAndNotSupportRefreshInByRefGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"netoutput"}}), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 1), GRAPH_SUCCESS);
}

/*
 *    a
 *    |
 * hcombroadcast 输入不支持地址刷新，输出不支持地址刷新，且输出引用输入
 *    |
 *  Netoutput
 *
 *   ||
 *   \/
 *    a
 *    |
 * hcombroadcast 输入不支持地址刷新，输出不支持地址刷新，且输出引用输入
 *    |
 *  identity
 *    |
 *  Netoutput
 *  物理地址不可刷新
 *  校验点：
 *  1 hcombroadcast 前不能插入拷贝。
 *  2 hcombroadcast 后面得插入拷贝。
 */
TEST_F(UtestMemLayoutConflictUserOutput, OutAndHcomBroadcast_StaticMemoryPolicy4_Success) {
  auto graph = MemConflictShareGraph::BuildUserOutAndNotSupportRefreshInByRefGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  StaticMemoryPolicy4Guarder guarder;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"netoutput"}}), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 1), GRAPH_SUCCESS);
}
/*
 *    op1
 *     +--hcombroadcast (p2p input)
 *     |
 *   netoutput
 *
 *    ||
 *    \/
 *
 *    op1
 *     +--identity--hcombroadcast (p2p input)
 *     |            /
 *     |    identity
 *     |    /
 *   netoutput
 */
TEST_F(UtestMemLayoutConflictUserOutput, UserOutAndRtsSpecialIn_InserIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildUserOutAndRtsSpecialInGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  const auto hcomallreduce = graph->FindNode("hcombroadcast");
  const auto netoutput = graph->FindNode("netoutput");
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"netoutput"}, {"netoutput", 1}}), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 2U), GRAPH_SUCCESS);
}
/*
 *          a
 *         / \
 * netoutput swt
 *             \
 *              b
 */
TEST_F(UtestMemLayoutConflictUserOutput, UserOutAndNotSupportRefreshOutByRef_FeatureMapBaseNotRefreshable_Success) {
  auto graph = MemConflictShareGraph::BuildUserOutAndNotSupportRefreshOutOneOutMultiRefGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"netoutput"}}), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 1U), GRAPH_SUCCESS);
}

/*
 *          a
 *         / \
 * netoutput swt
 *             \
 *              b
 */
TEST_F(UtestMemLayoutConflictUserOutput, UserOutAndNotSupportRefreshOutByRef_FeatureMapBaseRefreshable_Success) {
  auto graph = MemConflictShareGraph::BuildUserOutAndNotSupportRefreshOutOneOutMultiRefGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  FeatureMapRefreshOptionGuarder guarder;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"swt"}, {"b"}}), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 2U), GRAPH_SUCCESS);
}

/*
 *                                a
 *                              /  \
 *          a             identity  \
 *         / \               /       \
 * netoutput hcom ===>   netoutput   hcom
 *             \                       \
 *              b                      b
 *
 * StaticMemoryPolicy4，会给hcom算子分配fixed内存，所以hcom和a的输出间没有冲突。
 * 此时在netoutput前面或者在hcom前面插入identity都可以解决问题，并且也没有多插入identity
 */
TEST_F(UtestMemLayoutConflictUserOutput, UserOutAndNotSupportRefreshOutByRef_StaticMemoryPolicy4_Success) {
  auto graph = MemConflictShareGraph::BuildUserOutAndNotSupportRefreshOutByRefGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  StaticMemoryPolicy4Guarder guarder;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"netoutput"}}), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 1U), GRAPH_SUCCESS);
}
/*
 *   hcomallreduce (p2p output)
 *     |
 *   netoutput
 *
 *   ||
 *   \/
 *   hcomallreduce (p2p output)
 *     |
 *   identity
 *     |
 *   netoutput
 *
 */
TEST_F(UtestMemLayoutConflictUserOutput, UserOutAndRtsSpecialOut_InserIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildUserOutAndRtsSpecialOutGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"netoutput"}}), GRAPH_SUCCESS);
}
/*
 *   a             b             a             b
 *   |             |             |             |
 *   +-------+     |             +-------+     |
 *   |        \    /      ==>    |        \    /
 *   |      hcomallreduce        |      hcomallreduce
 * netoutput                   identity
 *                               |
 *                            netoutput
 */
TEST_F(UtestMemLayoutConflictUserOutput, UserOutAndContinuouInput_InserIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildUserOutAndContinuousInputGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"netoutput"}}), GRAPH_SUCCESS);
}
/*
 *   a             b             a             b
 *   |             |             |             |
 *   +-------+     |             +-------+     |
 *   |        \    /      ==>    |        \    /
 *   |      phonyconcat          |      phonyconcat
 * netoutput                   identity
 *                               |
 *                            netoutput
 */
TEST_F(UtestMemLayoutConflictUserOutput, UserOutAndNoPaddingContinuouInput_InserIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildUserOutAndNoPaddingContinuousInputGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"netoutput"}}), GRAPH_SUCCESS);
}
/*
 *       hcom             hcom
 *      /    \  ==>      /    \
 * netoutput  b       identity  b
 *                       |
 *                    netoutput
 */
TEST_F(UtestMemLayoutConflictUserOutput, UserOutAndContinuouOutput_InserIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildUserOutAndContinuousOutputGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"netoutput"}}), GRAPH_SUCCESS);
}
/*
 *   hcom             hcom
 *   ||..    ==>      /    \ ..
 * netoutput     identity  identity ..
 *                     \   /
 *                    netoutput
 */
TEST_F(UtestMemLayoutConflictUserOutput, UserOutAndALotOfContinuouOutput_InserIdentityInOrder_Success) {
  std::vector<NodePtr> identity1;
  std::vector<NodePtr> identity2;
  {
    auto graph = MemConflictShareGraph::BuildUserOutAndALotOfContinuousOutputGraph();
    graph->TopologicalSortingGraph(true);
    // 设置不同的哈希种子
    std::mt19937 rng2(67890);
    MemLayoutConflictOptimizer mem_check_pass;
    ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
    graph->TopologicalSortingGraph(true);
    auto netoutput = graph->FindFirstNodeMatchType(NETOUTPUT);
    const auto in_nodes = netoutput->GetInDataNodes();
    identity1.insert(identity1.end(), in_nodes.begin(), in_nodes.end());
  }
  {
    auto graph = MemConflictShareGraph::BuildUserOutAndALotOfContinuousOutputGraph();
    graph->TopologicalSortingGraph(true);
    // 设置不同的哈希种子
    std::mt19937 rng2(123456);
    MemLayoutConflictOptimizer mem_check_pass;
    ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
    graph->TopologicalSortingGraph(true);
    auto netoutput = graph->FindFirstNodeMatchType(NETOUTPUT);
    const auto in_nodes = netoutput->GetInDataNodes();
    identity2.insert(identity2.end(), in_nodes.begin(), in_nodes.end());
  }
  ASSERT_TRUE(identity1.size() == identity2.size());
  for (size_t i = 0U; i < identity1.size(); ++i) {
    EXPECT_EQ(identity1.at(i)->GetOpDesc()->GetId(), identity2.at(i)->GetOpDesc()->GetId());
  }
}
/*
 *    phony_split      phony_split
 *      /    \  ==>      /    \
 * netoutput  b       identity  b
 *                       |
 *                    netoutput
 */
TEST_F(UtestMemLayoutConflictUserOutput, UserOutAndNoPaddingContinuouOutput_InserIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildUserOutAndNoPaddingContinuousOutputGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"netoutput"}}), GRAPH_SUCCESS);
}
/*
 * data1  data2
 *   |      |
 *   a      b
 *   \      /
 *   phony_concat (NoPaddingContinuousInput,
 *      |           and otuput ref input)
 *    netoutput
 */
TEST_F(UtestMemLayoutConflictUserOutput, UserOutAndNoPaddingContinuouInputByRef_NotInserIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildUserOutAndNoPaddingContinuousInputByReferenceGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 0U), GRAPH_SUCCESS);
}
/*
 * data1  data2
 *   |      |
 *   a      b
 *   \      /
 *   concat (ContinuousInput,
 *      |    and otuput ref input)
 *    netoutput
 */
TEST_F(UtestMemLayoutConflictUserOutput, UserOutAndContinuouInputByRef_NotInserIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildUserOutAndContinuousInputByReferenceGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 0U), GRAPH_SUCCESS);
}

} // namespace ge
