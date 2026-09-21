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
#include "graph/compute_graph.h"
#include "compiler/graph/optimize/mem_layout_conflict_optimize/mem_layout_conflict_util.h"

namespace ge {
class UtestMemLayoutConflictUtil : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};
/*
 *    var  const
 *     \    /
 *     assign
 *       |
 *   hcombroadcast
 *       |
 *       a
 *       |
 *     netoutput
 */
TEST_F(UtestMemLayoutConflictUtil, IdentityCannotHasRefVarNameAttr) {
  auto graph = MemConflictShareGraph::BuildImmutableOutAndRtsSpecailInByAssignGraph();
  auto assign_node = graph->FindNode("assign");
  ASSERT_NE(assign_node, nullptr);
  auto tensor_desc = assign_node->GetOpDesc()->MutableOutputDesc(0);
  AttrUtils::SetStr(tensor_desc, REF_VAR_SRC_VAR_NAME, "var");

  auto hcombroadcast = graph->FindNode("hcombroadcast");
  ASSERT_NE(hcombroadcast, nullptr);
  ASSERT_NE(hcombroadcast->GetInDataAnchor(0), nullptr);

  OpDescPtr identity_op = nullptr;
  ASSERT_EQ(MemLayoutConflictUtil::CreateIdentityOpDesc({hcombroadcast->GetInDataAnchor(0)}, identity_op), SUCCESS);
  ASSERT_NE(identity_op, nullptr);
  ASSERT_NE(identity_op->MutableOutputDesc(0), nullptr);

  EXPECT_FALSE(AttrUtils::HasAttr(identity_op->MutableOutputDesc(0), REF_VAR_SRC_VAR_NAME));
}

TEST_F(UtestMemLayoutConflictUtil, UnknownShpae_SkipInsert) {
  auto graph = MemConflictShareGraph::BuildUnknowShapeGraph();
  auto netoutput_node = graph->FindNode("netoutput");
  ASSERT_NE(netoutput_node, nullptr);

  ASSERT_TRUE(MemLayoutConflictUtil::IsSkipInsert(netoutput_node->GetInDataAnchor(0)));
}

TEST_F(UtestMemLayoutConflictUtil, GetRefInput_ReturnDumyNode) {
  auto out_anchor = NodeIndexIO(static_cast<Node *>(nullptr), 0, kOut);
  auto in_anchor = MemLayoutConflictUtil::GetRefInput(out_anchor, {});
  EXPECT_EQ(in_anchor.node_, nullptr);
}

TEST_F(UtestMemLayoutConflictUtil, Get_while_conflict_anchors_failed) {
  auto graph = MemConflictShareGraph::BuildWhileInNodeConnectToMultiNodesGraph();
  EXPECT_FALSE(MemLayoutConflictUtil::IsCtrlNodeSubgraphExistMemConflictSymbol(graph));
}

TEST_F(UtestMemLayoutConflictUtil, Get_while_conflict_anchors_success) {
  auto graph = MemConflictShareGraph::BuildWhileDataMulNetoutputGraph();
  EXPECT_TRUE(MemLayoutConflictUtil::IsCtrlNodeSubgraphExistMemConflictSymbol(graph));
}

TEST_F(UtestMemLayoutConflictUtil, Get_while_conflict_anchors_success_02) {
  auto graph = MemConflictShareGraph::BuildWhileTwoDataToNetoutputExchangeGraph();
  EXPECT_TRUE(MemLayoutConflictUtil::IsCtrlNodeSubgraphExistMemConflictSymbol(graph));
}

TEST_F(UtestMemLayoutConflictUtil, Get_if_conflict_anchors_success) {
  auto graph = MemConflictShareGraph::BuildIfSingleOutMultiRefToNetoutputSubGraphWithPartitionedCall();
  auto netoutput_node = graph->FindNode("netoutput");
  ASSERT_NE(netoutput_node, nullptr);
  EXPECT_TRUE(MemLayoutConflictUtil::IsCtrlNodeSubgraphExistMemConflictSymbol(graph));
}

/*
 *  data1 data2 data3
 *     \    |    /
 *      \   |   /
 *         case
 *          |
 *         op3
 *          |
 *       netoutput
 *
 * subgraph x
 * +----------------------+ 
 * | datax_1--netoutputx |
 * | datax_2             |
 * | datax_3             |
 * +----------------------+
 *
 */
TEST_F(UtestMemLayoutConflictUtil, Get_case_conflict_anchors_success) {
  const auto data1_1 = OP_CFG(DATA).ParentNodeIndex(0);
  const auto data1_2 = OP_CFG(DATA).ParentNodeIndex(1);
  const auto data1_3 = OP_CFG(DATA).ParentNodeIndex(2);
  DEF_GRAPH(subgraph1) {
                        CHAIN(NODE("data1_1", data1_1)->NODE("netoutput1", NETOUTPUT));
                        CHAIN(NODE("data1_2", data1_2));
                        CHAIN(NODE("data1_3", data1_3));
                      };
  const auto data2_1 = OP_CFG(DATA).ParentNodeIndex(0);
  const auto data2_2 = OP_CFG(DATA).ParentNodeIndex(1);
  const auto data2_3 = OP_CFG(DATA).ParentNodeIndex(2);
  DEF_GRAPH(subgraph2) {
                        CHAIN(NODE("data2_1", data2_1));
                        CHAIN(NODE("data2_2", data2_2)->NODE("netoutput2", NETOUTPUT));
                        CHAIN(NODE("data2_3", data2_3));
                      };
  const auto data3_1 = OP_CFG(DATA).ParentNodeIndex(0);
  const auto data3_2 = OP_CFG(DATA).ParentNodeIndex(1);
  const auto data3_3 = OP_CFG(DATA).ParentNodeIndex(2);
  DEF_GRAPH(subgraph3) {
                        CHAIN(NODE("data3_1", data3_1));
                        CHAIN(NODE("data3_2", data3_2));
                        CHAIN(NODE("data3_3", data3_3)->NODE("netoutput3", NETOUTPUT));
                      };

  DEF_GRAPH(g1) {
                  CHAIN(NODE("data1", DATA)->EDGE(0, 0)->NODE("case", CASE, subgraph1, subgraph2, subgraph3)
                            ->EDGE(0, 0)->NODE("op3", ADD)->EDGE(0, 0)->NODE("netoutput0", NETOUTPUT));
                  CHAIN(NODE("data2", DATA)->EDGE(0, 1)->NODE("case", CASE, subgraph1, subgraph2, subgraph3));
                  CHAIN(NODE("data3", DATA)->EDGE(0, 2)->NODE("case", CASE, subgraph1, subgraph2, subgraph3));
                };
  auto graph = ToComputeGraph(g1);

  EXPECT_TRUE(MemLayoutConflictUtil::IsCtrlNodeSubgraphExistMemConflictSymbol(graph));
}

TEST_F(UtestMemLayoutConflictUtil, HasNotSupportPhysicalMemoryRefreshNode_return_false) {
  auto graph = MemConflictShareGraph::BuildUserOutAndNotSupportRefreshInByRefGraph();
  auto hcombroadcast = graph->FindNode("hcombroadcast");
  ASSERT_NE(hcombroadcast, nullptr);
  auto a = graph->FindNode("a");
  ASSERT_NE(a, nullptr);

  NodeIndexIO node_a{a, 0, IOType::kOut};
  NodeIndexIO node_hcom{hcombroadcast, 0, IOType::kIn};
  SmallVector<AnchorAttribute, ATTR_BIT_MAX_LEN> type_a(ANCHOR_ATTR_NORMAL_OUTPUT);
  SmallVector<AnchorAttribute, ATTR_BIT_MAX_LEN> type_b(ANCHOR_ATTR_UNSUPPORTED_ADDRESS_REFRESH_OPERATOR_INPUT);
  GraphInfo graph_info;
  graph_info.is_physical_memory_refreshable = true;
  std::set<AnchorPtr> result;
  CheckFuncContext context{node_a, node_hcom, 0, 0, type_a, type_b, {node_a, node_hcom}, graph_info, result};
  EXPECT_FALSE(MemLayoutConflictUtil::HasNotSupportPhysicalMemoryRefreshNode(context));
}

/*
 *    a
 *    |
 *   refnode   b
 *        \  /
 *         pc
 */
TEST_F(UtestMemLayoutConflictUtil, TraverseRefChainReverse_Success) {
  auto graph = MemConflictShareGraph::BuildConnectToNoPaddingContinuousInputThroughtRefNodeGraph();
  auto refnode = graph->FindNode("refnode");
  ASSERT_NE(refnode, nullptr);
  bool ret = MemLayoutConflictUtil::TraverseRefChainReverse(refnode.get(), 0, [] (const Node *node, int32_t out_index) {
    return node->GetName() == "a";
  });
  EXPECT_TRUE(ret);

  auto pc = graph->FindNode("pc");
  ASSERT_NE(pc, nullptr);
  ret = MemLayoutConflictUtil::TraverseRefChainReverse(pc.get(), 0, [] (const Node *node, int32_t out_index) {
    return node->GetName() == "a";
  });
  EXPECT_TRUE(ret);

  auto b = graph->FindNode("b");
  ASSERT_NE(b, nullptr);
  ret = MemLayoutConflictUtil::TraverseRefChainReverse(b.get(), 0, [] (const Node *node, int32_t out_index) {
    return node->GetName() == "a";
  });
  EXPECT_FALSE(ret);
}

/*
 *    a
 *    |
 *   refnode   b
 *        \  /
 *         pc
 */
TEST_F(UtestMemLayoutConflictUtil, TraverseRefChain_Success) {
  auto graph = MemConflictShareGraph::BuildConnectToNoPaddingContinuousInputThroughtRefNodeGraph();
  auto refnode = graph->FindNode("refnode");
  ASSERT_NE(refnode, nullptr);
  bool ret = MemLayoutConflictUtil::TraverseRefChain(refnode.get(), 0, [] (const Node *node, int32_t out_index) {
    return node->GetName() == "pc";
  });
  EXPECT_TRUE(ret);

  auto pc = graph->FindNode("pc");
  ASSERT_NE(pc, nullptr);
  ret = MemLayoutConflictUtil::TraverseRefChain(pc.get(), 0, [] (const Node *node, int32_t out_index) {
    return node->GetName() == "pc";
  });
  EXPECT_TRUE(ret);

  ret = MemLayoutConflictUtil::TraverseRefChain(pc.get(), 0, [] (const Node *node, int32_t out_index) {
    return node->GetName() == "B";
  });
  EXPECT_FALSE(ret);
}

/*
 *    a
 *    |
 *   refnode   b
 *        \  /
 *         pc
 */
TEST_F(UtestMemLayoutConflictUtil, IsConnectToNoPaddingContinuousInputNode_Success) {
  auto graph = MemConflictShareGraph::BuildConnectToNoPaddingContinuousInputThroughtRefNodeGraph();
  auto a = graph->FindNode("a");
  ASSERT_NE(a, nullptr);
  InDataAnchor *continuous_input_node = nullptr;
  EXPECT_TRUE(MemLayoutConflictUtil::IsContinuousInputThroughRefNode(
      a->GetAllOutDataAnchorsPtr().at(0)->GetPeerInDataAnchors().at(0).get(), true, continuous_input_node));

  ASSERT_NE(continuous_input_node, nullptr);
  auto b = graph->FindNode("b");
  ASSERT_NE(b, nullptr);
  EXPECT_TRUE(MemLayoutConflictUtil::IsContinuousInputThroughRefNode(
      b->GetAllOutDataAnchorsPtr().at(0)->GetPeerInDataAnchors().at(0).get(), true, continuous_input_node));
  (void)continuous_input_node;
}

/*
 *    a    b   c
 *    |___|___|
 *        |
 *    d  pc1  f
 *    |___|___|
 *        |
 *       pc2
 */
TEST_F(UtestMemLayoutConflictUtil, IsConnectToNoPaddingContinuousInputNode_Cascaded_Success) {
  auto graph = MemConflictShareGraph::BuildNoPaddingContinuousInCascadedGraph();
  auto a = graph->FindNode("a");
  ASSERT_NE(a, nullptr);
  InDataAnchor *continuous_input_node = nullptr;
  EXPECT_TRUE(MemLayoutConflictUtil::IsContinuousInputThroughRefNode(
      a->GetAllOutDataAnchorsPtr().at(0)->GetPeerInDataAnchors().at(0).get(), true, continuous_input_node));

  ASSERT_NE(continuous_input_node, nullptr);
  EXPECT_EQ(continuous_input_node->GetOwnerNode()->GetName(), "pc2");

  auto b = graph->FindNode("b");
  ASSERT_NE(b, nullptr);
  EXPECT_TRUE(MemLayoutConflictUtil::IsContinuousInputThroughRefNode(
      b->GetAllOutDataAnchorsPtr().at(0)->GetPeerInDataAnchors().at(0).get(), true, continuous_input_node));
  EXPECT_EQ(continuous_input_node->GetOwnerNode()->GetName(), "pc1");

  auto c = graph->FindNode("c");
  ASSERT_NE(c, nullptr);
  EXPECT_TRUE(MemLayoutConflictUtil::IsContinuousInputThroughRefNode(
      c->GetAllOutDataAnchorsPtr().at(0)->GetPeerInDataAnchors().at(0).get(), true, continuous_input_node));
  EXPECT_EQ(continuous_input_node->GetOwnerNode()->GetName(), "pc1");

  auto d = graph->FindNode("d");
  ASSERT_NE(d, nullptr);
  EXPECT_TRUE(MemLayoutConflictUtil::IsContinuousInputThroughRefNode(
      d->GetAllOutDataAnchorsPtr().at(0)->GetPeerInDataAnchors().at(0).get(), true, continuous_input_node));
  EXPECT_EQ(continuous_input_node->GetOwnerNode()->GetName(), "pc2");
}
/*
 *    a
 *    |
 *   refnode   b
 *        \  /
 *         hcom
 */
TEST_F(UtestMemLayoutConflictUtil, IsConnectToContinuousInputNode_Success) {
  auto graph = MemConflictShareGraph::BuildConnectToContinuousInputThroughtRefNodeGraph();
  auto a = graph->FindNode("a");
  ASSERT_NE(a, nullptr);
  InDataAnchor *continuous_input_node = nullptr;
  EXPECT_TRUE(MemLayoutConflictUtil::IsContinuousInputThroughRefNode(
      a->GetAllOutDataAnchorsPtr().at(0)->GetPeerInDataAnchors().at(0).get(), false, continuous_input_node));
  auto b = graph->FindNode("b");
  ASSERT_NE(b, nullptr);
  EXPECT_TRUE(MemLayoutConflictUtil::IsContinuousInputThroughRefNode(
      b->GetAllOutDataAnchorsPtr().at(0)->GetPeerInDataAnchors().at(0).get(), false, continuous_input_node));
  (void)continuous_input_node;
}
/*
 *         hcom
 *        /   \
 *   refnode   a
 *        \
 *         b
 */
TEST_F(UtestMemLayoutConflictUtil, IsConnectToContinuousOutputNode_Success) {
  auto graph = MemConflictShareGraph::BuildContinousOutConnectRefNodeGraph();
  auto b = graph->FindNode("b");
  ASSERT_NE(b, nullptr);
  OutDataAnchor *continuous_output_node = nullptr;
  EXPECT_TRUE(MemLayoutConflictUtil::IsContinuousOutputThroughRefNode(
      b->GetInDataAnchor(0)->GetPeerOutAnchor().get(), false, continuous_output_node));
  EXPECT_STRCASEEQ(continuous_output_node->GetOwnerNodeBarePtr()->GetName().c_str(), "hcom");
}
/*
 *         split
 *        /   \
 *   refnode   a
 *        \
 *         b
 */
TEST_F(UtestMemLayoutConflictUtil, IsConnectToContinuousOutputNode_Success_NopaddingOut) {
  auto graph = MemConflictShareGraph::BuildNopaddingContinousOutConnectRefNodeGraph();
  auto b = graph->FindNode("b");
  ASSERT_NE(b, nullptr);
  OutDataAnchor *continuous_output_node = nullptr;
  EXPECT_TRUE(MemLayoutConflictUtil::IsContinuousOutputThroughRefNode(
      b->GetInDataAnchor(0)->GetPeerOutAnchor().get(), true, continuous_output_node));
  EXPECT_STRCASEEQ(continuous_output_node->GetOwnerNodeBarePtr()->GetName().c_str(), "split");
}

/*
 *    var  const
 *     \    /
 *     assign
 *       |
 *   hcombroadcast
 *       |
 *       a
 *       |
 *     netoutput
 */
TEST_F(UtestMemLayoutConflictUtil, GetRealInputs_Success) {
  auto graph = MemConflictShareGraph::BuildImmutableOutAndRtsSpecailInByAssignGraph();
  auto hcombroadcast = graph->FindNode("hcombroadcast");
  ASSERT_NE(hcombroadcast, nullptr);

  auto in_nodes = MemLayoutConflictUtil::GetAllRealInPeer(hcombroadcast);
  ASSERT_EQ(in_nodes.size(), 1U);
}
} // namespace ge
