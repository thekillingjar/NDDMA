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
#include "runtime/mem.h"

namespace ge {
class UtestMemLayoutConflictUserInput : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};
/*
 *  data1  data2
 *     \    /
 *       if
 *       |
 *      op3
 *       |
 *    netoutput
 *
 * then_subgraph          else_subgraph
 * +------------------+   +------------------+
 * | data3--netoutput1|   | data5  netoutput2|
 * | data4            |   | data6-+/         |
 * +------------------+   +------------------+
 *
 *             ||
 *             \/
 *
 *  data1  data2
 *     \    /
 *       if
 *       |
 *      op3
 *       |
 *    netoutput
 *
 * then_subgraph                     else_subgraph
 * +-----------------------------+   +-----------------------------+
 * | data3--identity1--netoutput1|   | data5             netoutput2|
 * | data4                       |   | data6--identity2-+/         |
 * +-----------------------------+   +-----------------------------+
 */
TEST_F(UtestMemLayoutConflictUserInput, UserInAndUserIn_InsertTwoIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildUserInAndUserInGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"netoutput1"}, {"netoutput2"}}), GRAPH_SUCCESS);
}

/*
 *      data
 *       |
 *     netoutput
 *
 *        ||
 *        \/
 *      data
 *       |
 *     netoutput
 */
TEST_F(UtestMemLayoutConflictUserInput, UserInAndUserOut_NotInsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildUserInAndUserOutGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 0U), GRAPH_SUCCESS);
}

/*
 * unknown graph
 *      op1
 *       |               known sub graph
 * partitioned_call      +-----------+
 *       |               |   data    |
 *      op2              |    |      |
 *                       | netoutput |
 *                       +-----------+
 *           ||
 *           \/
 *
 * unknown graph
 *      op1
 *       |               known sub graph
 * partitioned_call      +-----------+
 *       |               |   data    |
 *      op2              |    |      |
 *                       | netoutput |
 *                       +-----------+
 */
TEST_F(UtestMemLayoutConflictUserInput, UserInAndUserOutInKnownSubGraph_NotInsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildUserInAndUserOutGraphInSubGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 0U), GRAPH_SUCCESS);
}

/*
 * known graph
 *     data1
 *       |               known sub graph
 * partitioned_call      +-----------+
 *       |               |   data2   |
 *      op2              |    |      |
 *                       | netoutput |
 *                       +-----------+
 *           ||
 *           \/
 * known graph
 *     data1
 *       |               known sub graph
 * partitioned_call      +-----------+
 *       |               |   data2   |
 *      op2              |    |      |
 *                       | netoutput |
 *                       +-----------+
 */
TEST_F(UtestMemLayoutConflictUserInput, UserInAndNormalInWithSubGraph_NotInsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildUserInAndNormalInWithSubGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 0U), GRAPH_SUCCESS);
}

/*
 *  data1  const
 *     \    /
 *       if
 *       |
 *      op3
 *       |
 *    netoutput
 *
 * then_subgraph          else_subgraph
 * +------------------+   +------------------+
 * | data3--netoutput1|   | data5  netoutput2|
 * | data4            |   | data6-+/         |
 * +------------------+   +------------------+
 *
 *             ||
 *             \/
 *
 *  data1  const
 *     \    /
 *       if
 *       |
 *      op3
 *       |
 *    netoutput
 *
 * then_subgraph                     else_subgraph
 * +-----------------------------+   +-----------------------------+
 * | data3--identity1--netoutput1|   | data5             netoutput2|
 * | data4                       |   | data6--identity2-+/         |
 * +-----------------------------+   +-----------------------------+
 */
TEST_F(UtestMemLayoutConflictUserInput, UserInAndImmutableOut_InsertTwoIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildUserInAndUserInGraph();

  // 把一个用户输入改成长量
  auto data2 = graph->FindNode("data2");
  data2->GetOpDesc()->SetType(CONSTANT);
  data2->GetOpDesc()->SetName("const");

  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"netoutput1"}, {"netoutput2"}}), GRAPH_SUCCESS);
}

/*
 * unknown graph
 *      op1
 *       |               known sub graph
 * partitioned_call      +-----------+
 *       |               |   data    |
 *      op2              |    |      |
 *                       |  switch   |
 *                       |    |      |
 *                       |   op3     |
 *                       |    |      |
 *                       | netoutput |
 *                       +-----------+
 *
 *             ||
 *             \/
 * unknown graph
 *      op1
 *       |               known sub graph
 * partitioned_call      +-----------+
 *       |               |   data    |
 *      op2              |    |      |
 *                       |  identity1|
 *                       |    |      |
 *                       |  switch   |
 *                       |    |      |
 *                       |   op3     |
 *                       |    |      |
 *                       | netoutput |
 *                       +-----------+
 */
TEST_F(UtestMemLayoutConflictUserInput, UserInAndNotSupportRefreshInputInKnownSubGraph_InsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildUserInAndNotSupportRefreshInGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"switch"}}), GRAPH_SUCCESS);
}
/*
 *   data        data
 *    |           |
 *   swt  ==>  identity
 *    |           |
 *    a          swt
 *    |           |
 *  netoutput     a
 *                |
 *             netoutput
 *  feature map base不可刷新场景下
 *  校验dsa后面不插入identity.
 *  data后面插入identity
 */
TEST_F(UtestMemLayoutConflictUserInput, UserInAndNotSupportRefreshOutByRef_FeatureMapNotRefreshable_Success) {
  auto graph = MemConflictShareGraph::BuildUserInAndNotSupportRefreshOutByRefGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 1), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"swt"}}), GRAPH_SUCCESS);
}
/*
 *   data        data
 *    |           |
 *   hcom  ==>  identity
 *    |           |
 *    a          hcom
 *    |           |
 *  netoutput     a
 *                |
 *             netoutput
 *  StaticMemoryPolicy4场景下
 *  校验hcom后面不插入identity.
 *  data后面插入identity
 */
TEST_F(UtestMemLayoutConflictUserInput, UserInAndNotSupportRefreshOutByRef_StaticMemoryPolicy4_Success) {
  auto graph = MemConflictShareGraph::BuildUserInAndNotSupportRefreshOutByRefWithHcomGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  StaticMemoryPolicy4Guarder guarder;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 1), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"hcom"}}), GRAPH_SUCCESS);
}
/*
 *                      data
 *                       |
 *    data    swt      identity    swt
 *      \     /   ==>    \        /
 *     phonyconcat       phonyconcat
 *  dsa在feature map不可刷新的场景下，不需要插入identity，所以校验只有data后面才插入identity
 */
TEST_F(UtestMemLayoutConflictUserInput, UserInAndNotSupportRefreshOutByContinuousIn_FeatureMapNotRefreshable_Success) {
  auto graph = MemConflictShareGraph::BuildUserInAndNotSupportRefreshOutByContinuousInGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 1), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"phonyconcat"}}), GRAPH_SUCCESS);
}

/*
 *  data1  streamswitch
 *     \    /
 *       if
 *       |
 *      op3
 *       |
 *    netoutput
 *
 * then_subgraph          else_subgraph
 * +------------------+   +------------------+
 * | data3--netoutput1|   | data5  netoutput2|
 * | data4            |   | data6-+/         |
 * +------------------+   +------------------+
 *
 *             ||
 *             \/
 *
 *  data1  streamswitch
 *     \    /
 *       if
 *       |
 *      op3
 *       |
 *    netoutput
 *
 * then_subgraph                     else_subgraph
 * +-----------------------------+   +-----------------------------+
 * | data3--identity1--netoutput1|   | data5             netoutput2|
 * | data4                       |   | data6--identity2-+/         |
 * +-----------------------------+   +-----------------------------+
 */
TEST_F(UtestMemLayoutConflictUserInput, UserInAndNotSupportRefreshOut_InsertTwoIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildUserInAndUserInGraph();

  // 把一个用户输入改成Streamswitch
  auto data2 = graph->FindNode("data2");
  data2->GetOpDesc()->SetType(STREAMSWITCH);
  data2->GetOpDesc()->SetName("streamswitch");

  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"netoutput1"}, {"netoutput2"}}), GRAPH_SUCCESS);
}

/*
 *      data
 *       |
 *      op1 need p2p input
 *
 *        ||
 *        \/
 *      data
 *       |
 *     identity
 *       |
 *     op1 need p2p input
 */
TEST_F(UtestMemLayoutConflictUserInput, UserInAndRtsSpecialIn_InsertOneIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildUserInAndRtsSpecialInGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"op1"}}), GRAPH_SUCCESS);
}
/*
 *                                             data
 *                                              |
 *      data                                  identity
 *       |                                      |
 *      hcom (输出引用输入，输出需要p2p类型内存) ==> hcom (输出引用输入，输出需要p2p类型内存)
 *       |                                      |
 *       a                                      a
 */
TEST_F(UtestMemLayoutConflictUserInput, UserInAndRtsSpecialOut_InsertOneIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildUserInAndRtsSpecialOutGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"hcom"}}), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 1), GRAPH_SUCCESS);
}

/*
 *  data1  hcomallreduce
 *     \    /
 *       if
 *       |
 *      op3
 *       |
 *    netoutput
 *
 * then_subgraph          else_subgraph
 * +------------------+   +------------------+
 * | data3--netoutput1|   | data5  netoutput2|
 * | data4            |   | data6-+/         |
 * +------------------+   +------------------+
 *
 *             ||
 *             \/
 *
 *  data1  hcomallreduce
 *     \    /
 *       if
 *       |
 *      op3
 *       |
 *    netoutput
 *
 * then_subgraph                     else_subgraph
 * +-----------------------------+   +-----------------------------+
 * | data3--identity1--netoutput1|   | data5             netoutput2|
 * | data4                       |   | data6------------+/         |
 * +-----------------------------+   +-----------------------------+
 */
TEST_F(UtestMemLayoutConflictUserInput, UserInAndRtsSpecialOut_InsertTwoIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildUserInAndUserInGraph();

  // 把一个用户输入改成hcomallreduce
  auto data2 = graph->FindNode("data2");
  data2->GetOpDesc()->SetType(HCOMALLREDUCE);
  data2->GetOpDesc()->SetName("hcomallreduce");
  std::vector<int64_t> mem_type {RT_MEMORY_P2P_DDR};
  (void)ge::AttrUtils::SetListInt(data2->GetOpDescBarePtr(), ATTR_NAME_OUTPUT_MEM_TYPE_LIST,
                                  mem_type);

  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"netoutput1"}, {"netoutput2"}}), GRAPH_SUCCESS);
}

/*
 *  data1  add1
 *     \    /
 *       if
 *       |
 *      op3
 *       |
 *    netoutput
 *
 * then_subgraph          else_subgraph
 * +------------------+   +------------------+
 * | data3--netoutput1|   | data5  netoutput2|
 * | data4            |   | data6-+/         |
 * +------------------+   +------------------+
 *
 *             ||
 *             \/
 *
 *  data1  add1
 *     \    /
 *       if
 *       |
 *      op3
 *       |
 *    netoutput
 *
 * then_subgraph                     else_subgraph
 * +-----------------------------+   +-----------------------------+
 * | data3--identity1--netoutput1|   | data5             netoutput2|
 * | data4                       |   | data6------------+/         |
 * +-----------------------------+   +-----------------------------+
 */
TEST_F(UtestMemLayoutConflictUserInput, UserInAndNormalOut_InsertOneIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildUserInAndUserInGraph();

  // 把一个用户输入改成 add1
  auto data2 = graph->FindNode("data2");
  data2->GetOpDesc()->SetType(ADD);
  data2->GetOpDesc()->SetName("add1");

  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"netoutput1"}, {"netoutput2"}}), GRAPH_SUCCESS);
}

/*
 * know graph
 *      data1
 *       |               known sub graph
 * partitioned_call      +-----------+
 *       |               |   data2   |
 *      op2              |    |      |
 *                       |   add     |
 *                       |    |      |
 *                       | netoutput |
 *                       +-----------+
 */
TEST_F(UtestMemLayoutConflictUserInput, UserInAndNormalOut_NotInsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildUserInAndNormalOutWithSubGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 0U), GRAPH_SUCCESS);
}

/*
 *                             data1  data2
 *                               |      |
 *    data1  data2            identity identity
 *      \     /                  \     /
 *        hcom         ===>        hcom
 *         |                        |
 *         a                        a
 *
 */
TEST_F(UtestMemLayoutConflictUserInput, UserInAndContinuousInput_InsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildUserInConnectContinuousInputGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"hcom", 0}, {"hcom", 1}}), GRAPH_SUCCESS);
}

/*
 *                                          data1  data2
 *                                           |      |
 *   data1 data2                          identity identity
 *    \   /                                  \     /
 *     hcom (输出引用输入，且连续输出内存)  ==>     hcom (输出引用输入，且连续输出内存)
 *     / \                                     / \
 *    a   b                                   a   b
 */
TEST_F(UtestMemLayoutConflictUserInput, UserInAndContinuousOutput_InsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildUserInConnectContinuousOutputGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 2U), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"hcom"}, {"hcom", 1}}), GRAPH_SUCCESS);
}
/*
 *                             data1  data2
 *                               |      |
 *    data1  data2            identity identity
 *      \     /                  \     /
 *     phony_concat   ===>      phony_concat
 *         |                        |
 *         a                        a
 *
 */
TEST_F(UtestMemLayoutConflictUserInput, UserInAndNoPaddingContinuousInput_InsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildUserInConnectNoPaddingContinuousInputGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 2U), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"phony_concat", 0}, {"phony_concat", 1}}), GRAPH_SUCCESS);
}

/*
 *    data
 *      |
 *   phony_split (输出引用输入，
 *     /  \      且NoPadding连续输出内存)
 *    a    b
 */
TEST_F(UtestMemLayoutConflictUserInput, UserInAndNoPaddingContinuousOutput_NotInsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildUserInConnectNoPaddingContinuousOutputGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 0U), GRAPH_SUCCESS);
}

/*
 *    data
 *      |
 *   phony_split (输出引用输入，
 *     /  \      且NoPadding连续输出内存)
 *    a    b
 */
TEST_F(UtestMemLayoutConflictUserInput, UserInAndNoPaddingContinuousOutput_SingleOp_InsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildUserInConnectNoPaddingContinuousOutputGraph();
  (void)AttrUtils::SetBool(graph, ATTR_SINGLE_OP_SCENE, true);
  auto ps = graph->FindNode("phony_split");
  ASSERT_NE(ps, nullptr);
  AttrUtils::SetBool(ps->GetOpDesc(), ATTR_NAME_REFERENCE, true);
  ps->GetOpDesc()->MutableAllInputName()["x"] = 0;
  ps->GetOpDesc()->MutableAllOutputName()["x"] = 0;
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 1U), GRAPH_SUCCESS);
}

/*
 *   refdata  const        refdata  const
 *     \    /                \    /
 *     assign   a            assign
 *        \    /                \
 *          pc    ==>       identity a
 *           |                     \ /
 *           b                     pc
 *           |                     |
 *       netoutput                 b
 *                                 |
 *                              netoutput
 */
TEST_F(UtestMemLayoutConflictUserInput, RefdataAndNoPaddingContinuousOutput_InsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildUserInAndNoPaddingContinuousInByAssignOutput();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 1U), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"pc", 0}}), GRAPH_SUCCESS);
}

/*
 *   refdata const            refdata const
 *       \   /                    \   /
 *      assign       ==>        assign
 *       |                         |
 *      op1 need p2p input       identity
 *                                  |
 *                                 op1 need p2p input
 */
TEST_F(UtestMemLayoutConflictUserInput, RefdataAndRtsSpecialIn_InsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildUserInRefDataAndRtsSpecialInGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 1U), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"op1", 0}}), GRAPH_SUCCESS);
}
/*
 *   refdata const            refdata const
 *       \   /                    \   /
 *      assign       ==>        assign
 *       |                         |
 *      hcom need p2p output       identity
 *                                  |
 *                                 hcom need p2p output
 */
TEST_F(UtestMemLayoutConflictUserInput, RefdataAndRtsSpecialOut_InsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildUserInRefDataAndRtsSpecialOutGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 2U), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"hcom", 0}, {"netoutput", 0}}), GRAPH_SUCCESS);
}
/*
 *   refdata const            refdata const
 *       \   /                    \   /
 *      assign       ==>        assign
 *       |                         |
 *      swt                      identity
 *                                  |
 *                                 swt
 */
TEST_F(UtestMemLayoutConflictUserInput, RefdataAndNotRefreshIn_InsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildUserInRefDataAndNotSupportRefreshInGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 2U), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"swt", 0}, {"netoutput", 0}}), GRAPH_SUCCESS);
}
/*
 *   refdata const            refdata const
 *       \   /                    \   /
 *      assign       ==>        assign
 *       |                         |
 *      swt                      identity
 *                                  |
 *                                 swt
 */
TEST_F(UtestMemLayoutConflictUserInput, RefdataAndNotRefreshOut_InsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildUserInRefDataAndNotSupportRefreshOutGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityNum(graph, 2U), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"swt", 0}, {"netoutput", 0}}), GRAPH_SUCCESS);
}
} // namespace ge
