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
#include <string>
#include "common/ge_inner_error_codes.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph_builder_utils.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/utils/node_adapter.h"
#include "graph/passes/multi_batch/subexpression_migration_pass.h"

using namespace ge;

class UtestSubexpressionMigrationPass : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

static void PrepareGraphForTest(const ComputeGraphPtr &graph) {
  const auto case_node = graph->FindNode("case");
  if (case_node != nullptr) {
    (void)AttrUtils::SetInt(case_node->GetOpDesc(), ATTR_NAME_BATCH_NUM, 2);
  }
  for (const auto &subgraph : graph->GetAllSubgraphs()) {
    const auto add_node = subgraph->FindFirstNodeMatchType(ADD);
    if (add_node != nullptr) {
      add_node->SetHostNode(true);
    }
    const auto less_node = subgraph->FindFirstNodeMatchType(LESS);
    if (less_node != nullptr) {
      less_node->SetHostNode(true);
    }
  }
}

static ComputeGraphPtr BuildNormalGraph() {

  const auto sub1_data_0 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0);
  const auto sub1_data_1 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 1);
  DEF_GRAPH(sub_1) {
    CHAIN(NODE("sub1_data_0", sub1_data_0)->NODE("add", ADD)->NODE("less", LESS));
    CHAIN(NODE("sub1_data_1", sub1_data_1)->NODE("add", ADD));
    CHAIN(NODE("sub1_const_0", CONSTANTOP)->NODE("less", LESS)->NODE("netoutput", NETOUTPUT));
  };

  const auto sub2_data_0 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0);
  const auto sub2_data_1 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 1);
  DEF_GRAPH(sub_2) {
    CHAIN(NODE("sub2_data_0", sub2_data_0)->NODE("add", ADD)->NODE("less", LESS));
    CHAIN(NODE("sub2_data_1", sub2_data_1)->NODE("add", ADD));
    CHAIN(NODE("sub2_const_0", CONSTANTOP)->NODE("less", LESS)->NODE("netoutput", NETOUTPUT));
  };

  DEF_GRAPH(g1) {
    CHAIN(NODE("data_0", DATA)->NODE("case", CASE, sub_1, sub_2)->NODE("netoutput", NETOUTPUT));
    CHAIN(NODE("data_1", DATA)->NODE("case"));
  };

  sub_1.Layout();
  sub_2.Layout();
  const auto compute_graph = ToComputeGraph(g1);
  PrepareGraphForTest(compute_graph);
  return compute_graph;
}

static ComputeGraphPtr BuildGraphMultiMigration() {

  const auto sub1_data_0 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0);
  const auto sub1_data_1 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 1);
  DEF_GRAPH(sub_1) {
    CHAIN(NODE("sub1_data_0", sub1_data_0)->NODE("sub1/add", ADD)->NODE("sub1/less", LESS));
    CHAIN(NODE("sub1_data_1", sub1_data_1)->NODE("sub1/add", ADD));
    CHAIN(NODE("sub1_data_1", sub1_data_1)->NODE("sub1/less", LESS)->NODE("sub1/netoutput", NETOUTPUT));
  };

  const auto sub2_data_0 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0);
  const auto sub2_data_1 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 1);
  DEF_GRAPH(sub_2) {
    CHAIN(NODE("sub2_data_0", sub2_data_0)->NODE("sub2/add", ADD)->NODE("sub2/less", LESS));
    CHAIN(NODE("sub2_data_1", sub2_data_1)->NODE("sub2/add", ADD));
    CHAIN(NODE("sub2_data_1", sub2_data_1)->NODE("sub2/less", LESS)->NODE("sub2/netoutput", NETOUTPUT));
  };

  DEF_GRAPH(g1) {
    CHAIN(NODE("data_0", DATA)->NODE("case", CASE, sub_1, sub_2)->NODE("netoutput", NETOUTPUT));
    CHAIN(NODE("data_1", DATA)->NODE("case"));
  };

  sub_1.Layout();
  sub_2.Layout();
  const auto compute_graph = ToComputeGraph(g1);
  PrepareGraphForTest(compute_graph);
  return compute_graph;
}

static ComputeGraphPtr BuildIdentityGraph() {

  const auto sub1_data_0 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0);
  const auto sub1_data_1 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 1);
  DEF_GRAPH(sub_1) {
    CHAIN(NODE("sub1_data_0", sub1_data_0)->NODE("sub1/identity1", IDENTITY)->NODE("sub1/identityn", IDENTITYN)->EDGE(0, 0)->NODE("sub1/add", ADD)->NODE("sub1/less", LESS));
    CHAIN(NODE("sub1_data_1", sub1_data_1)->NODE("sub1/identity2", IDENTITY)->EDGE(0, 1)->NODE("sub1/identityn", IDENTITYN)->EDGE(1, 1)->NODE("sub1/add", ADD));
    CHAIN(NODE("sub1/variale", VARIABLE)->NODE("sub1/identity3", IDENTITY)->EDGE(0, 2)->NODE("sub1/identityn", IDENTITYN)->EDGE(2, 1)->NODE("sub1/less", LESS)->NODE("sub1/netoutput", NETOUTPUT));
  };

  const auto sub2_data_0 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0);
  const auto sub2_data_1 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 1);
  DEF_GRAPH(sub_2) {
    CHAIN(NODE("sub2_data_0", sub2_data_0)->NODE("sub2/identity1", IDENTITY)->NODE("sub2/identityn", IDENTITYN)->EDGE(0, 0)->NODE("sub2/add", ADD)->NODE("sub2/less", LESS));
    CHAIN(NODE("sub2_data_1", sub2_data_1)->NODE("sub2/identity2", IDENTITY)->EDGE(0, 1)->NODE("sub2/identityn", IDENTITYN)->EDGE(1, 1)->NODE("sub2/add", ADD));
    CHAIN(NODE("sub2/variale", VARIABLE)->NODE("sub2/identity3", IDENTITY)->EDGE(0, 2)->NODE("sub2/identityn", IDENTITYN)->EDGE(2, 1)->NODE("sub2/less", LESS)->NODE("sub2/netoutput", NETOUTPUT));
  };

  DEF_GRAPH(g1) {
    CHAIN(NODE("data_0", DATA)->NODE("case", CASE, sub_1, sub_2)->NODE("netoutput", NETOUTPUT));
    CHAIN(NODE("data_1", DATA)->NODE("case"));
  };

  sub_1.Layout();
  sub_2.Layout();
  const auto compute_graph = ToComputeGraph(g1);
  PrepareGraphForTest(compute_graph);
  return compute_graph;
}

static ComputeGraphPtr BuildIdentityGraph2() {

  const auto sub1_data_0 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0);
  const auto sub1_data_1 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 1);
  DEF_GRAPH(sub_1) {
    CHAIN(NODE("sub1_data_0", sub1_data_0)->NODE("sub1/identityn", IDENTITYN)->EDGE(0, 0)->NODE("sub1/add", ADD)->NODE("sub1/less", LESS));
    CHAIN(NODE("sub1_data_1", sub1_data_1)->EDGE(0, 1)->NODE("sub1/identityn", IDENTITYN)->EDGE(1, 1)->NODE("sub1/add", ADD));
    CHAIN(NODE("sub1_data_1", sub1_data_1)->EDGE(0, 0)->NODE("sub1/add2", ADD)->EDGE(0, 0)->NODE("sub1/add3", ADD)->NODE("sub1/netoutput", NETOUTPUT));
    CHAIN(NODE("sub1/variale", VARIABLE)->EDGE(0, 2)->NODE("sub1/identityn", IDENTITYN)->EDGE(2, 1)->NODE("sub1/less", LESS)->EDGE(0, 1)->NODE("sub1/add3", ADD));
  };

  const auto sub2_data_0 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0);
  const auto sub2_data_1 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 1);
  DEF_GRAPH(sub_2) {
    CHAIN(NODE("sub2_data_0", sub2_data_0)->NODE("sub2/identityn", IDENTITYN)->EDGE(0, 0)->NODE("sub2/add", ADD)->NODE("sub2/less", LESS));
    CHAIN(NODE("sub2_data_1", sub2_data_1)->EDGE(0, 1)->NODE("sub2/identityn", IDENTITYN)->EDGE(1, 1)->NODE("sub2/add", ADD));
    CHAIN(NODE("sub2_data_1", sub1_data_1)->EDGE(0, 0)->NODE("sub2/add2", ADD)->EDGE(0, 0)->NODE("sub2/add3", ADD)->NODE("sub2/netoutput", NETOUTPUT));
    CHAIN(NODE("sub2/variale", VARIABLE)->EDGE(0, 2)->NODE("sub2/identityn", IDENTITYN)->EDGE(2, 1)->NODE("sub2/less", LESS)->EDGE(0, 1)->NODE("sub2/add3", ADD));
  };

  DEF_GRAPH(g1) {
    CHAIN(NODE("data_0", DATA)->NODE("case", CASE, sub_1, sub_2)->NODE("netoutput", NETOUTPUT));
    CHAIN(NODE("data_1", DATA)->NODE("case"));
  };

  sub_1.Layout();
  sub_2.Layout();
  const auto compute_graph = ToComputeGraph(g1);
  PrepareGraphForTest(compute_graph);
  return compute_graph;
}

static ComputeGraphPtr BuildGraphNoNeedMigration() {

  const auto sub1_data_0 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0);
  DEF_GRAPH(sub_1) {
    CHAIN(NODE("sub1_data_0", sub1_data_0)->NODE("add", ADD)->NODE("netoutput", NETOUTPUT));
    CHAIN(NODE("sub1_const_0", CONSTANTOP)->NODE("add", ADD));
  };

  const auto sub2_data_0 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0);
  DEF_GRAPH(sub_2) {
    CHAIN(NODE("sub2_data_0", sub2_data_0)->NODE("add", ADD)->NODE("netoutput", NETOUTPUT));
    CHAIN(NODE("sub2_const_0", CONSTANTOP)->NODE("add", ADD));
  };

  DEF_GRAPH(g1) {
    CHAIN(NODE("data_0", DATA)->NODE("case", CASE, sub_1, sub_2)->NODE("netoutput", NETOUTPUT));
  };

  sub_1.Layout();
  sub_2.Layout();
  const auto compute_graph = ToComputeGraph(g1);
  PrepareGraphForTest(compute_graph);
  return compute_graph;
}
/*
                                          sub_1

┌─────────────┐  (0,0)   ┌─────────────┐          ┌───────────┐  (0,0)   ┌────────────────┐
│ sub1_data_0 │ ───────> │  sub1/add   │          │ sub1/less │ ───────> │ sub1/netoutput │
└─────────────┘          └─────────────┘          └───────────┘          └────────────────┘
                           ∧                        ∧
                           │ (0,1)                  │
                           │                        │
                         ┌─────────────┐  (0,1)     │
                         │ sub1_data_1 │ ───────────┘
                         └─────────────┘

                                          sub_2

┌─────────────┐  (0,0)   ┌─────────────┐  (0,0)   ┌───────────┐  (0,0)   ┌────────────────┐
│ sub2_data_0 │ ───────> │  sub2/add   │ ───────> │ sub2/cast │ ───────> │ sub2/netoutput │
└─────────────┘          └─────────────┘          └───────────┘          └────────────────┘
                           ∧
                           │ (0,1)
                           │
                         ┌─────────────┐
                         │ sub2_data_1 │
                         └─────────────┘

                               g1

                      ┌───────────┐
                      │  data_1   │
                      └───────────┘
                        │
                        │ (0,1)
                        ∨
                    ┌−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−┐
                    ╎ case:                                     ╎
                    ╎                                           ╎
┌────────┐  (0,0)   ╎ ┌───────────┐     ╔ ═ ═ ═ ╗     ╔ ═ ═ ═ ╗ ╎
│ data_0 │ ───────> ╎ │   case    │ ─── ∥ sub_1 ∥ ─── ∥ sub_2 ∥ ╎
└────────┘          ╎ └───────────┘     ╚ ═ ═ ═ ╝     ╚ ═ ═ ═ ╝ ╎
                    ╎                                           ╎
                    └−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−┘
                        │
                        │ (0,0)
                        ∨
                      ┌───────────┐
                      │ netoutput │
                      └───────────┘
*/
static ComputeGraphPtr BuildGraphNoNeedMigration1() {
  ut::GraphBuilder builder = ut::GraphBuilder("root_graph");
  auto data_0 = builder.AddNode("data_0", "Data", 1, 1);
  auto data_1 = builder.AddNode("data_1", "Data", 1, 1);
  auto case_1 = builder.AddNode("case", "Case", 2, 1);
  auto netoutput = builder.AddNode("netoutput", "NetOutput", 1, 0);

  builder.AddDataEdge(data_0, 0, case_1, 0);
  builder.AddDataEdge(data_1, 0, case_1, 1);
  builder.AddDataEdge(case_1, 0, netoutput, 0);
  auto root_graph = builder.GetGraph();

  ut::GraphBuilder sub_builder1 = ut::GraphBuilder("sub_graph1");
  auto sub1_data0 = sub_builder1.AddNode("sub1_data0", "Data", 1, 1);
  auto sub1_data1 = sub_builder1.AddNode("sub1_data1", "Data", 1, 1);
  auto sub1_add = sub_builder1.AddNode("sub1_add", "Add", 2, 1);
  auto sub1_less = sub_builder1.AddNode("sub1_less", "Less", 1, 1);
  auto sub1_netoutput = sub_builder1.AddNode("sub1_netoutput", "NetOutput", 1, 0);
  AttrUtils::SetInt(sub1_data0->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 1);
  AttrUtils::SetInt(sub1_data1->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
  AttrUtils::SetInt(sub1_netoutput->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_PARENT_NODE_INDEX, 0);

  auto input_desc0 = sub1_add->GetOpDesc()->MutableInputDesc(0);
  input_desc0->SetFormat(FORMAT_NDC1HWC0);
 
  sub_builder1.AddDataEdge(sub1_data0, 0, sub1_add, 0);
  sub_builder1.AddDataEdge(sub1_data1, 0, sub1_add, 1);
  sub_builder1.AddDataEdge(sub1_data1, 0, sub1_less, 0);
  sub_builder1.AddDataEdge(sub1_less, 0, sub1_netoutput, 1);
  auto subgraph1 = sub_builder1.GetGraph();

  ut::GraphBuilder sub_builder2 = ut::GraphBuilder("sub_graph2");
  auto sub2_data0 = sub_builder2.AddNode("sub2_data0", "Data", 1, 1);
  auto sub2_data1 = sub_builder2.AddNode("sub2_data1", "Data", 1, 1);
  auto sub2_add = sub_builder1.AddNode("sub2_add", "Add", 2, 1);
  auto sub2_cast = sub_builder2.AddNode("sub2_cast", "Cast", 1, 1);
  auto sub2_netoutput = sub_builder2.AddNode("sub2_netoutput", "NetOutput", 1, 0);
  AttrUtils::SetInt(sub2_data0->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
  AttrUtils::SetInt(sub2_data1->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 1);
  AttrUtils::SetInt(sub2_netoutput->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_PARENT_NODE_INDEX, 0);
 
  sub_builder2.AddDataEdge(sub2_data0, 0, sub2_add, 0);
  sub_builder2.AddDataEdge(sub2_data1, 0, sub2_add, 1);
  sub_builder2.AddDataEdge(sub2_add, 0, sub2_cast, 0);
  sub_builder2.AddDataEdge(sub2_cast, 0, sub2_netoutput, 0);
  auto subgraph2 = sub_builder2.GetGraph();

  auto case_node1 = root_graph->FindNode("case");
  auto case_desc1 = case_node1->GetOpDesc();
  case_desc1->AddSubgraphName("sub_graph1");
  case_desc1->AddSubgraphName("sub_graph2");
  case_desc1->SetSubgraphInstanceName(0, "sub_graph1");
  case_desc1->SetSubgraphInstanceName(1, "sub_graph2");


  subgraph1->SetParentNode(case_node1);
  subgraph1->SetParentGraph(root_graph);
  subgraph2->SetParentNode(case_node1);
  subgraph2->SetParentGraph(root_graph);
  root_graph->AddSubgraph("sub_graph1", subgraph1);
  root_graph->AddSubgraph("sub_graph2", subgraph2);
  PrepareGraphForTest(root_graph);
  return root_graph;
}

TEST_F(UtestSubexpressionMigrationPass, Run0) {
  Status retStatus;
  SubexpressionMigrationPass csubExpMigPass;

  //string name, string type, int in_cnt, int out_cnt, Format format,
  //DataType data_type, vector<int64_t> shape
  DEF_GRAPH(g1) {
    CHAIN(NODE("arg_0", DATA)->NODE("addNode", ADD)->NODE("output", NETOUTPUT));
    CHAIN(NODE("arg_1", DATA)->NODE("addNode"));
  };
  ComputeGraphPtr graph = ToComputeGraph(g1);

  retStatus = csubExpMigPass.Run(graph);
  EXPECT_EQ(retStatus, SUCCESS);
}

TEST_F(UtestSubexpressionMigrationPass, normal_migration_succ) {
  ComputeGraphPtr graph = BuildNormalGraph();
  EXPECT_EQ(graph->GetDirectNodesSize(), 4);
  SubexpressionMigrationPass pass;
  GE_DUMP(graph, "before_migration");
  const auto ret = pass.Run(graph);
  GE_DUMP(graph, "after_migration");
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(), 5);
  const auto add_node = graph->FindFirstNodeMatchType(ADD);
  EXPECT_NE(add_node, nullptr);
  const auto case_node = graph->FindFirstNodeMatchType(CASE);
  ASSERT_NE(case_node, nullptr);
  EXPECT_EQ(case_node->GetAllInDataAnchorsSize(), 2);
}

TEST_F(UtestSubexpressionMigrationPass, identity_migration_succ) {
  ComputeGraphPtr graph = BuildIdentityGraph();
  EXPECT_EQ(graph->GetDirectNodesSize(), 4);
  SubexpressionMigrationPass pass;
  GE_DUMP(graph, "before_migration");
  const auto ret = pass.Run(graph);
  GE_DUMP(graph, "after_migration");
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(), 8);
  const auto add_node = graph->FindFirstNodeMatchType(ADD);
  EXPECT_NE(add_node, nullptr);
  const auto case_node = graph->FindFirstNodeMatchType(CASE);
  ASSERT_NE(case_node, nullptr);
  EXPECT_EQ(case_node->GetAllInDataAnchorsSize(), 2);
  int32_t index = 0;
  for (const auto &node : graph->GetDirectNode()) {
    if (node->GetType() == IDENTITY) {
      index++;
    }
  }
  EXPECT_EQ(index, 2);
  index = 0;
  for (const auto &node : graph->GetDirectNode()) {
    if (node->GetType() == IDENTITYN) {
      index++;
    }
  }
  EXPECT_EQ(index, 1);

  index = 0;
  for (const auto &node : graph->GetAllNodes()) {
    if (node->GetType() == IDENTITYN) {
      index++;
    }
  }
  EXPECT_EQ(index, 3);
}

TEST_F(UtestSubexpressionMigrationPass, identityn_migration_succ) {
  ComputeGraphPtr graph = BuildIdentityGraph2();
  EXPECT_EQ(graph->GetDirectNodesSize(), 4);
  SubexpressionMigrationPass pass;
  GE_DUMP(graph, "before_migration");
  const auto ret = pass.Run(graph);
  GE_DUMP(graph, "after_migration");
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(), 6);
  const auto add_node = graph->FindFirstNodeMatchType(ADD);
  EXPECT_NE(add_node, nullptr);
  const auto case_node = graph->FindFirstNodeMatchType(CASE);
  ASSERT_NE(case_node, nullptr);
  EXPECT_EQ(case_node->GetAllInDataAnchorsSize(), 3);
  int32_t index = 0;
  for (const auto &node : graph->GetDirectNode()) {
    if (node->GetType() == IDENTITYN) {
      index++;
    }
  }
  EXPECT_EQ(index, 1);

  index = 0;
  for (const auto &node : graph->GetAllNodes()) {
    if (node->GetType() == IDENTITYN) {
      index++;
    }
  }
  EXPECT_EQ(index, 3);
}

TEST_F(UtestSubexpressionMigrationPass, multi_migration_succ) {
  ComputeGraphPtr graph = BuildGraphMultiMigration();
  EXPECT_EQ(graph->GetDirectNodesSize(), 4);
  SubexpressionMigrationPass pass;
  const auto ret = pass.Run(graph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(), 6);
  const auto add_node = graph->FindFirstNodeMatchType(ADD);
  EXPECT_NE(add_node, nullptr);
  const auto case_node = graph->FindFirstNodeMatchType(CASE);
  ASSERT_NE(case_node, nullptr);
  const auto less_node = graph->FindFirstNodeMatchType(LESS);
  ASSERT_NE(less_node, nullptr);
  EXPECT_EQ(case_node->GetAllInDataAnchorsSize(), 2);
}

TEST_F(UtestSubexpressionMigrationPass, graph_no_need_migration) {
  ComputeGraphPtr graph = BuildGraphNoNeedMigration();
  EXPECT_EQ(graph->GetDirectNodesSize(), 3);
  SubexpressionMigrationPass pass;
  const auto ret = pass.Run(graph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(), 3);
  const auto add_node = graph->FindFirstNodeMatchType(ADD);
  EXPECT_EQ(add_node, nullptr);
  const auto case_node = graph->FindFirstNodeMatchType(CASE);
  ASSERT_NE(case_node, nullptr);
  EXPECT_EQ(case_node->GetAllInDataAnchorsSize(), 1);
}

TEST_F(UtestSubexpressionMigrationPass, graph_no_need_migration_data_anchor_index_outof_range) {
  ComputeGraphPtr graph = BuildGraphNoNeedMigration1();
  GE_DUMP(graph, "test_migration");
  EXPECT_EQ(graph->GetDirectNodesSize(), 4);
  SubexpressionMigrationPass pass;
  const auto ret = pass.Run(graph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(), 4);
  const auto add_node = graph->FindFirstNodeMatchType(ADD);
  EXPECT_EQ(add_node, nullptr);
  const auto case_node = graph->FindFirstNodeMatchType(CASE);
  ASSERT_NE(case_node, nullptr);
  EXPECT_EQ(case_node->GetAllInDataAnchorsSize(), 2);
}
