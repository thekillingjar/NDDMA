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
class UtestMemLayoutConflictIfCaseSubgraph : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

void AddParentIndexForSubGraphNetoutput(ComputeGraphPtr &root_graph) {
  for (auto &node : root_graph->GetAllNodes()) {
    if (node->GetType() == NETOUTPUT) {
      if (node->GetOwnerComputeGraph()->GetParentNode() == nullptr) {
        continue;
      }
      auto op_desc = node->GetOpDesc();
      for (uint32_t index = 0U; index < op_desc->GetInputsSize(); ++index) {
        AttrUtils::SetInt(op_desc->MutableInputDesc(index), ATTR_NAME_PARENT_NODE_INDEX, index);
      }
    }
  }
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
TEST_F(UtestMemLayoutConflictIfCaseSubgraph, CaseSubGraph_EachInsertIdentity_Success) {

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
  AddParentIndexForSubGraphNetoutput(graph);

  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"netoutput1"}, {"netoutput2"}, {"netoutput3"}}), GRAPH_SUCCESS);
}

/*
 *  data1 data2 data3
 *     \    |    /
 *      \   |   /
 *     statelesscase
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
TEST_F(UtestMemLayoutConflictIfCaseSubgraph, StatelessCase_EachInsertIdentity_Success) {

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
                  CHAIN(NODE("data1", DATA)->EDGE(0, 0)->NODE("case", STATELESSCASE, subgraph1, subgraph2, subgraph3)
                            ->EDGE(0, 0)->NODE("op3", ADD)->EDGE(0, 0)->NODE("netoutput0", NETOUTPUT));
                  CHAIN(NODE("data2", DATA)->EDGE(0, 1)->NODE("case"));
                  CHAIN(NODE("data3", DATA)->EDGE(0, 2)->NODE("case"));
                };
  auto graph = ToComputeGraph(g1);
  AddParentIndexForSubGraphNetoutput(graph);

  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"netoutput1"}, {"netoutput2"}, {"netoutput3"}}), GRAPH_SUCCESS);
}

/*
 *  data1  data2
 *     \    /
 *       if
 *       |
 *       op
 *       |
 *    netoutput0
 *
 * then_subgraph                             else_subgraph
 * +-------------------------------------+   +------------------------------------+
 * |    data3                            |   |                                    |
 * |     |                               |   |                                    |
 * |    cast                             |   |    data5                           |
 * |     |                               |   |     |                              |
 * | partitioned_call1   +------------+  |   | partitioned_call2  +------------+  |
 * |     |               |   data4    |  |   |     |              |    data6   |  |
 * |  netoutput1         |     |      |  |   |  netoutput3        |      |     |  |
 * |                     | netoutput2 |  |   |                    | netoutput4 |  |
 * |                     +------------+  |   |                    +------------+  |
 * +-------------------------------------+   +------------------------------------+
 *
 */
TEST_F(UtestMemLayoutConflictIfCaseSubgraph, IfSubGraph_DataNetoutputNotInSameGraph_Success) {
  const auto data4 = OP_CFG(DATA).ParentNodeIndex(0);
  const auto data3 = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(sub_1) {
                     CHAIN(NODE("data4", data4)->NODE("netoutput2", NETOUTPUT));
                   };
  DEF_GRAPH(then_) {
                  CHAIN(NODE("data3", data3)->NODE("cast", CAST)->NODE("partitioned_call1", PARTITIONEDCALL, sub_1)->NODE("netoutput1", NETOUTPUT));
                };

  const auto data6 = OP_CFG(DATA).ParentNodeIndex(0);
  const auto data5 = OP_CFG(DATA).ParentNodeIndex(1);
  DEF_GRAPH(sub_2) {
                     CHAIN(NODE("data6", data6)->NODE("netoutput4", NETOUTPUT));
                   };
  DEF_GRAPH(else_) {
                  CHAIN(NODE("data5", data5)->NODE("partitioned_call2", PARTITIONEDCALL, sub_2)->NODE("netoutput3", NETOUTPUT));
                };

  DEF_GRAPH(g1) {
                  CHAIN(NODE("data1", DATA)->EDGE(0, 0)->NODE("if", IF, then_, else_)
                            ->EDGE(0, 0)->NODE("op", CAST)->EDGE(0, 0)->NODE("netoutput0", NETOUTPUT));
                  CHAIN(NODE("data2", DATA)->EDGE(0, 1)->NODE("if"));
                };
  auto graph = ToComputeGraph(g1);

  auto partitioned_call1_graph = ToComputeGraph(sub_1);
  auto partitioned_call2_graph = ToComputeGraph(sub_2);
  then_.Layout();
  else_.Layout();
  auto then_subgraph = graph->GetSubgraph("then_");
  auto else_subgraph = graph->GetSubgraph("else_");
  auto partitioned_call1_node = then_subgraph->FindNode("partitioned_call1");
  auto partitioned_call2_node = else_subgraph->FindNode("partitioned_call2");
  then_subgraph->SetParentGraph(graph);
  else_subgraph->SetParentGraph(graph);
  partitioned_call1_graph->SetParentNode(partitioned_call1_node);
  partitioned_call2_graph->SetParentNode(partitioned_call2_node);
  graph->AddSubGraph(partitioned_call1_graph);
  graph->AddSubGraph(partitioned_call2_graph);

  AddParentIndexForSubGraphNetoutput(graph);

  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"netoutput3"}}), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(graph, {{"netoutput1"}}), GRAPH_FAILED);
}

/*
 *  unknow sub graph
 *     data
 *      |                  know sub graph
 *   partitioned_call1   +-----------------+
 *      |                |          data2  |
 *    netoutput          |  op1 op2        |
 *                       |   \  /          |      then_subgraph       else_subgraph
 *                       |    if           |     +-----------------+ +-----------------+
 *                       |     |           |     | data3  data4    | | data5  data6    |
 *                       |    op3          |     |          |      | |          |      |
 *                       |     |           |     |      netoutput2 | |      netoutput3 |
 *                       |   netoutput1    |     +-----------------+ +-----------------+
 *                       +-----------------+
 *   构图要点：partitioned_call1是动态shape静态子图，静态子图中有if子图的场景
 *   看护场景：如果对子图调用拓扑排序，那么子图的子图不会被跳过。因为子图的子图也是挂在根图上的。后续GetRefMapping可能会报错。
 */
TEST_F(UtestMemLayoutConflictIfCaseSubgraph, IfSubGraphInKnowSubgraph_CheckReturnSuccess) {
  auto root_graph = MemConflictShareGraph::BuildIfInKnowSubGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(root_graph), GRAPH_SUCCESS);
  EXPECT_EQ(mem_check::ResultChecker::CheckIdentityBefore(root_graph, {{"netoutput2"}, {"netoutput3"}}), GRAPH_SUCCESS);
}

/*
 *    pred
 *     |
 *    cast   input
 *      \    /
 *        if
 *        \/
 *    netoutput
 *
 * then_subgraph        else_subgraph
 * +----------------+   +---------------+
 * |     data1      |   |    data2      |
 * |      |         |   |      |        |
 * |     mul1       |   |     / \       |
 * |      |         |   |   mu2  mu3    |
 * |     / \        |   |    |    |     |
 * |   netoutput1   |   |   netoutput2  |
 * +----------------+   +---------------+
 */
TEST_F(UtestMemLayoutConflictIfCaseSubgraph, IfSingleOutMultiRefToNetoutput_CheckReturnSuccess) {
  auto root_graph = MemConflictShareGraph::BuildIfSingleOutMultiRefToNetoutputSubGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(root_graph), GRAPH_SUCCESS);
  
  auto then = root_graph->GetSubgraph("then");
  auto netoutput1 = then->FindNode("netoutput1");

  auto in_anchor = netoutput1->GetInDataAnchor(1);
  auto peer_node = in_anchor->GetPeerOutAnchor()->GetOwnerNode();
  ASSERT_EQ(peer_node->GetType(), IDENTITY);
}

/*
 *    pred
 *     |
 *    cast   input
 *      \    /
 *        if
 *        \/
 *    netoutput
 *
 * then_subgraph                            else_subgraph
 * +------------------------------------+   +---------------+
 * |     data1                          |   |    data2      |
 * |      |                             |   |      |        |
 * |     mul1                           |   |     / \       |
 * |      |                             |   |   mu2  mu3    |
 * |     / \                            |   |    |    |     |
 * |    |  partitionedcall +----------+ |   |   netoutput2  |
 * |    |    |             |  data3   | |   +---------------+
 * |   netoutput1          |    |     | |
 * |                       |netoutput3| |
 * |                       +----------+ |
 * +------------------------------------+
 */
TEST_F(UtestMemLayoutConflictIfCaseSubgraph, IfSingleOutMultiRefToNetoutputWithPartitionedCall_CheckReturnSuccess) {
  auto root_graph = MemConflictShareGraph::BuildIfSingleOutMultiRefToNetoutputSubGraphWithPartitionedCall();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(root_graph), GRAPH_SUCCESS);
  
  auto then = root_graph->GetSubgraph("then");
  auto netoutput1 = then->FindNode("netoutput1");

  auto in_anchor = netoutput1->GetInDataAnchor(1);
  auto peer_node = in_anchor->GetPeerOutAnchor()->GetOwnerNode();
  ASSERT_EQ(peer_node->GetType(), IDENTITY);
}
}
