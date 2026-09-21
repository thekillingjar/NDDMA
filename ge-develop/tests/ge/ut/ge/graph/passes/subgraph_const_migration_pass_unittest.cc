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

#include <set>
#include <string>

#include "framework/omg/omg_inner_types.h"
#include "common/context/local_context.h"
#include "graph/passes/multi_batch/subgraph_const_migration_pass.h"
#include "graph/passes/multi_batch/subexpression_migration_pass.h"
#include "graph/passes/pass_manager.h"
#include "register/op_registry.h"
#include "ge_graph_dsl/graph_dsl.h"

namespace ge {
class UtestSubgraphConstMigrationPass : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}

 public:
  NodePtr MakeNode(const ComputeGraphPtr &graph, int in_num, int out_num, string name, string type) {
    GeTensorDesc test_desc(GeShape(), FORMAT_NCHW, DT_FLOAT);
    auto op_desc = std::make_shared<OpDesc>(name, type);
    for (auto i = 0; i < in_num; ++i) {
      op_desc->AddInputDesc(test_desc);
    }
    for (auto i = 0; i < out_num; ++i) {
      op_desc->AddOutputDesc(test_desc);
    }
    if (type == "Const") {
      uint64_t const_value = 101;
      auto weight = std::make_shared<GeTensor>(op_desc->GetOutputDesc(0), (uint8_t *)&const_value, sizeof(uint64_t));
      AttrUtils::SetTensor(op_desc, ge::ATTR_NAME_WEIGHTS, weight);
    }
    return graph->AddNode(op_desc);
  }

  void make_original_graph(const ComputeGraphPtr &graph) {
    auto data = MakeNode(graph, 1, 1, "data", "Data");
    {
      AttrUtils::SetInt(data->GetOpDesc(), ATTR_NAME_INDEX, 0);
      AttrUtils::SetInt(data->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 1);
    }
    auto const1 = MakeNode(graph, 0, 1, "const1", "Const");
    {
      auto data1 = MakeNode(graph, 1, 1, "data1", "Data");
      AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_INDEX, 1);
      AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 2);
      GraphUtils::AddEdge(data1->GetOutControlAnchor(), const1->GetInControlAnchor());
    }

    auto const2 = MakeNode(graph, 0, 1, "const2", "Const");
    {
      auto data2 = MakeNode(graph, 1, 1, "data2", "Data");
      AttrUtils::SetInt(data2->GetOpDesc(), ATTR_NAME_INDEX, 2);
      AttrUtils::SetInt(data2->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 3);
      GraphUtils::AddEdge(data2->GetOutControlAnchor(), const2->GetInControlAnchor());
    }

    auto conv2d_node = MakeNode(graph, 3, 1, "conv1", "Conv2D");
    GraphUtils::AddEdge(data->GetOutDataAnchor(0), conv2d_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(const1->GetOutDataAnchor(0), conv2d_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(const2->GetOutDataAnchor(0), conv2d_node->GetInDataAnchor(2));
  }

  void make_multibatch_graph(const ComputeGraphPtr &graph) {
    auto index = MakeNode(graph, 1, 1, "index", "Data");
    auto data = MakeNode(graph, 1, 1, "data", "Data");
    auto data1 = MakeNode(graph, 1, 1, "data1", "Data");
    auto data2 = MakeNode(graph, 1, 1, "data2", "Data");
    AttrUtils::SetInt(data->GetOpDesc(), ATTR_NAME_INDEX, 0);
    AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_INDEX, 1);
    AttrUtils::SetInt(data2->GetOpDesc(), ATTR_NAME_INDEX, 2);

    auto case1 = MakeNode(graph, 4, 1, "case", "Case");
    GraphUtils::AddEdge(index->GetOutDataAnchor(0), case1->GetInDataAnchor(0));
    GraphUtils::AddEdge(data->GetOutDataAnchor(0), case1->GetInDataAnchor(1));
    GraphUtils::AddEdge(data1->GetOutDataAnchor(0), case1->GetInDataAnchor(2));
    GraphUtils::AddEdge(data2->GetOutDataAnchor(0), case1->GetInDataAnchor(3));
    auto output_node = MakeNode(graph, 1, 0, "output", "NetOutput");
    GraphUtils::AddEdge(case1->GetOutDataAnchor(0), output_node->GetInDataAnchor(0));

    AttrUtils::SetInt(case1->GetOpDesc(), ATTR_NAME_BATCH_NUM, 2);
    case1->GetOpDesc()->RegisterSubgraphIrName("branches", kDynamic);
    ComputeGraphPtr branch = std::make_shared<ComputeGraph>("test_branch");
    make_original_graph(branch);
    for (int i = 0; i < 2; ++i) {
      std::string name("_ascend_mbatch_batch_" + std::to_string(i));
      std::vector<NodePtr> input_nodes;
      std::vector<NodePtr> output_nodes;
      ComputeGraphPtr subgraph = GraphUtils::CloneGraph(branch, name, input_nodes, output_nodes);

      subgraph->SetName(name);
      subgraph->SetParentNode(case1);
      subgraph->SetParentGraph(graph);
      graph->AddSubgraph(subgraph->GetName(), subgraph);

      case1->GetOpDesc()->AddSubgraphName(name);
      case1->GetOpDesc()->SetSubgraphInstanceName(i, subgraph->GetName());
    }
  }

  void make_original_graph_subexpression(const ComputeGraphPtr &graph) {
    auto data = MakeNode(graph, 1, 1, "data", "Data");
    {
      AttrUtils::SetInt(data->GetOpDesc(), ATTR_NAME_INDEX, 0);
      AttrUtils::SetInt(data->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 1);
    }
    auto const1 = MakeNode(graph, 0, 1, "const1", "Const");
    {
      auto data1 = MakeNode(graph, 1, 1, "data1", "Data");
      AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_INDEX, 1);
      AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 2);
      GraphUtils::AddEdge(data1->GetOutControlAnchor(), const1->GetInControlAnchor());
      auto cast_node1 = MakeNode(graph, 1, 1, "cast1", "Cast");
      GraphUtils::AddEdge(data1->GetOutDataAnchor(0), cast_node1->GetInDataAnchor(0));
    }

    auto const2 = MakeNode(graph, 0, 1, "const2", "Const");
    {
      auto data2 = MakeNode(graph, 1, 1, "data2", "Data");
      AttrUtils::SetInt(data2->GetOpDesc(), ATTR_NAME_INDEX, 2);
      AttrUtils::SetInt(data2->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 3);
      GraphUtils::AddEdge(data2->GetOutControlAnchor(), const2->GetInControlAnchor());
    }

    auto cast_node2 = MakeNode(graph, 1, 1, "cast2", "Cast");
    GraphUtils::AddEdge(data->GetOutDataAnchor(0), cast_node2->GetInDataAnchor(0));
    auto cast_node3 = MakeNode(graph, 1, 1, "cast3", "Cast");
    GraphUtils::AddEdge(data->GetOutDataAnchor(0), cast_node3->GetInDataAnchor(0));

    auto conv2d_node = MakeNode(graph, 3, 1, "conv1", "Conv2D");
    GraphUtils::AddEdge(cast_node2->GetOutDataAnchor(0), conv2d_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(const1->GetOutDataAnchor(0), conv2d_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(const2->GetOutDataAnchor(0), conv2d_node->GetInDataAnchor(2));

    auto out = MakeNode(graph, 1, 0, "out1", "NetOutput");
    GraphUtils::AddEdge(conv2d_node->GetOutDataAnchor(0), out->GetInDataAnchor(0));
  }

  void make_multibatch_graph_subexpression(const ComputeGraphPtr &graph) {
    auto index = MakeNode(graph, 1, 1, "index", "Data");
    auto data = MakeNode(graph, 1, 1, "data", "Data");
    auto data1 = MakeNode(graph, 1, 1, "data1", "Data");
    auto data2 = MakeNode(graph, 1, 1, "data2", "Data");
    AttrUtils::SetInt(data->GetOpDesc(), ATTR_NAME_INDEX, 0);
    AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_INDEX, 1);
    AttrUtils::SetInt(data2->GetOpDesc(), ATTR_NAME_INDEX, 2);

    auto case1 = MakeNode(graph, 4, 1, "case", "Case");
    GraphUtils::AddEdge(index->GetOutDataAnchor(0), case1->GetInDataAnchor(0));
    GraphUtils::AddEdge(data->GetOutDataAnchor(0), case1->GetInDataAnchor(1));
    GraphUtils::AddEdge(data1->GetOutDataAnchor(0), case1->GetInDataAnchor(2));
    GraphUtils::AddEdge(data2->GetOutDataAnchor(0), case1->GetInDataAnchor(3));
    auto output_node = MakeNode(graph, 1, 0, "output", "NetOutput");
    GraphUtils::AddEdge(case1->GetOutDataAnchor(0), output_node->GetInDataAnchor(0));

    AttrUtils::SetInt(case1->GetOpDesc(), ATTR_NAME_BATCH_NUM, 2);
    case1->GetOpDesc()->RegisterSubgraphIrName("branches", kDynamic);
    ComputeGraphPtr branch = std::make_shared<ComputeGraph>("test_branch");
    make_original_graph_subexpression(branch);
    for (int i = 0; i < 2; ++i) {
      std::string name("_ascend_mbatch_batch_" + std::to_string(i));
      std::vector<NodePtr> input_nodes;
      std::vector<NodePtr> output_nodes;
      ComputeGraphPtr subgraph = GraphUtils::CloneGraph(branch, name, input_nodes, output_nodes);

      subgraph->SetName(name);
      subgraph->SetParentNode(case1);
      subgraph->SetParentGraph(graph);
      std::string host_node_name = "cast2" + name;
      auto host_node = subgraph->FindNode(host_node_name);
      host_node->SetHostNode(true);
      graph->AddSubgraph(subgraph->GetName(), subgraph);

      case1->GetOpDesc()->AddSubgraphName(name);
      case1->GetOpDesc()->SetSubgraphInstanceName(i, subgraph->GetName());
    }
  }
};

namespace {
static void PrepareGraphForTest(const ComputeGraphPtr &graph) {
  const auto case_node = graph->FindNode("case");
  if (case_node != nullptr) {
    (void)AttrUtils::SetInt(case_node->GetOpDesc(), ATTR_NAME_BATCH_NUM, 2);
  }
}

void InitConstantNode(const ComputeGraphPtr &graph, const std::string &op_name, int32_t const_value) {
  const auto node = graph->FindNode(op_name);
  EXPECT_NE(node, nullptr);
  const auto op_desc = node->GetOpDesc();

  GeTensorDesc data_desc = op_desc->GetOutputDesc(0);
  GeTensorPtr weight_value =
      make_shared<GeTensor>(data_desc, reinterpret_cast<uint8_t *>(&const_value), sizeof(int32_t));
  EXPECT_TRUE(AttrUtils::SetTensor(op_desc, ATTR_NAME_WEIGHTS, weight_value));
}

ComputeGraphPtr BuildNormalGraph() {
  const auto sub1_data_0 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0);
  const auto sub1_data_1 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 1);
  DEF_GRAPH(sub_1) {
    CTRL_CHAIN(NODE("data_0_ascend_mbatch_batch_0", sub1_data_0)->NODE("const_0_ascend_mbatch_batch_0", CONSTANT));
    CHAIN(NODE("const_0_ascend_mbatch_batch_0", CONSTANT)->NODE("add_ascend_mbatch_batch_0", ADD));
    CHAIN(NODE("data_1_ascend_mbatch_batch_0", sub1_data_1)->NODE("add_ascend_mbatch_batch_0", ADD));
    CTRL_CHAIN(NODE("data_1_ascend_mbatch_batch_0", sub1_data_1)->NODE("const_1_ascend_mbatch_batch_0", CONSTANT));
    CHAIN(NODE("add_ascend_mbatch_batch_0", ADD)->NODE("netoutput_ascend_mbatch_batch_0", NETOUTPUT));
    CHAIN(NODE("const_1_ascend_mbatch_batch_0", CONSTANT)->NODE("identity_ascend_mbatch_batch_0", IDENTITY));
    CHAIN(NODE("identity_ascend_mbatch_batch_0", IDENTITY)->NODE("netoutput_ascend_mbatch_batch_0", NETOUTPUT));
  };

  const auto sub2_data_0 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0);
  const auto sub2_data_1 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 1);
  DEF_GRAPH(sub_2) {
    CTRL_CHAIN(NODE("data_0_ascend_mbatch_batch_1", sub2_data_0)->NODE("const_0_ascend_mbatch_batch_1", CONSTANT));
    CHAIN(NODE("const_0_ascend_mbatch_batch_1", CONSTANT)->NODE("add_ascend_mbatch_batch_1", ADD));
    CHAIN(NODE("data_1_ascend_mbatch_batch_1", sub2_data_1)->NODE("add_ascend_mbatch_batch_1", ADD));
    CTRL_CHAIN(NODE("data_1_ascend_mbatch_batch_1", sub2_data_1)->NODE("const_1_ascend_mbatch_batch_1", CONSTANT));
    CHAIN(NODE("add_ascend_mbatch_batch_1", ADD)->NODE("netoutput_ascend_mbatch_batch_1", NETOUTPUT));
    CHAIN(NODE("const_1_ascend_mbatch_batch_1", CONSTANT)->NODE("identity_ascend_mbatch_batch_1", IDENTITY));
    CHAIN(NODE("identity_ascend_mbatch_batch_1", IDENTITY)->NODE("netoutput_ascend_mbatch_batch_1", NETOUTPUT));
  };

  DEF_GRAPH(g1) {
    CHAIN(NODE("data_0", DATA)->NODE("case", CASE, sub_1, sub_2)->NODE("netoutput", NETOUTPUT));
    CHAIN(NODE("data_1", DATA)->NODE("case"));
  };

  sub_1.Layout();
  sub_2.Layout();
  const auto compute_graph = ToComputeGraph(g1);
  PrepareGraphForTest(compute_graph);
  // value equal
  InitConstantNode(compute_graph->GetSubgraph("sub_1"), "const_0_ascend_mbatch_batch_0", 100);
  InitConstantNode(compute_graph->GetSubgraph("sub_2"), "const_0_ascend_mbatch_batch_1", 100);
  // value diff
  InitConstantNode(compute_graph->GetSubgraph("sub_1"), "const_1_ascend_mbatch_batch_0", 10);
  InitConstantNode(compute_graph->GetSubgraph("sub_2"), "const_1_ascend_mbatch_batch_1", 11);
  return compute_graph;
}

ComputeGraphPtr BuildNormalGraphWithUselessConst() {
  const auto sub1_data_0 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0);
  const auto sub1_data_1 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 1);
  DEF_GRAPH(sub_1) {
    CTRL_CHAIN(NODE("data_0_ascend_mbatch_batch_0", sub1_data_0)->NODE("const_0_ascend_mbatch_batch_0", CONSTANT));
    CHAIN(NODE("data_0_ascend_mbatch_batch_0", sub1_data_0)->NODE("add_ascend_mbatch_batch_0", ADD));
    CHAIN(NODE("data_1_ascend_mbatch_batch_0", sub1_data_1)->NODE("add_ascend_mbatch_batch_0", ADD));
    CTRL_CHAIN(NODE("data_1_ascend_mbatch_batch_0", sub1_data_1)->NODE("const_1_ascend_mbatch_batch_0", CONSTANT));
    CHAIN(NODE("add_ascend_mbatch_batch_0", ADD)->NODE("netoutput_ascend_mbatch_batch_0", NETOUTPUT));
    CHAIN(NODE("const_1_ascend_mbatch_batch_0", CONSTANT)->NODE("identity_ascend_mbatch_batch_0", IDENTITY));
    CHAIN(NODE("identity_ascend_mbatch_batch_0", IDENTITY)->NODE("netoutput_ascend_mbatch_batch_0", NETOUTPUT));
  };

  const auto sub2_data_0 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0);
  const auto sub2_data_1 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 1);
  DEF_GRAPH(sub_2) {
    CTRL_CHAIN(NODE("data_0_ascend_mbatch_batch_1", sub2_data_0)->NODE("const_0_ascend_mbatch_batch_1", CONSTANT));
    CHAIN(NODE("data_0_ascend_mbatch_batch_1", sub2_data_0)->NODE("add_ascend_mbatch_batch_1", ADD));
    CHAIN(NODE("data_1_ascend_mbatch_batch_1", sub2_data_1)->NODE("add_ascend_mbatch_batch_1", ADD));
    CTRL_CHAIN(NODE("data_1_ascend_mbatch_batch_1", sub2_data_1)->NODE("const_1_ascend_mbatch_batch_1", CONSTANT));
    CHAIN(NODE("add_ascend_mbatch_batch_1", ADD)->NODE("netoutput_ascend_mbatch_batch_1", NETOUTPUT));
    CHAIN(NODE("const_1_ascend_mbatch_batch_1", CONSTANT)->NODE("identity_ascend_mbatch_batch_1", IDENTITY));
    CHAIN(NODE("identity_ascend_mbatch_batch_1", IDENTITY)->NODE("netoutput_ascend_mbatch_batch_1", NETOUTPUT));
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

} // namespace

TEST_F(UtestSubgraphConstMigrationPass, subgraph_const_migration) {
  PassManager pass_manager;
  pass_manager.AddPass("SubgraphConstMigrationPass", new (std::nothrow) SubgraphConstMigrationPass);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  make_multibatch_graph(graph);
  EXPECT_EQ(pass_manager.Run(graph), SUCCESS);
}

TEST_F(UtestSubgraphConstMigrationPass, subgraph_const_migration_2) {
  PassManager pass_manager;
  pass_manager.AddPass("SubgraphConstMigrationPass", new (std::nothrow) SubgraphConstMigrationPass);
  ComputeGraphPtr graph = BuildNormalGraph();
  EXPECT_EQ(pass_manager.Run(graph), SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(), 4);
  const auto case_node = graph->FindNode("case");
  ASSERT_NE(case_node, nullptr);
  EXPECT_EQ(case_node->GetInDataNodes().size(), 2);
  EXPECT_EQ(case_node->GetInDataNodes().at(0)->GetType(), CONSTANT);
}

TEST_F(UtestSubgraphConstMigrationPass, subgraph_const_migration_2_with_useless_const) {
  PassManager pass_manager;
  pass_manager.AddPass("SubgraphConstMigrationPass", new (std::nothrow) SubgraphConstMigrationPass);
  ComputeGraphPtr graph = BuildNormalGraphWithUselessConst();
  EXPECT_EQ(pass_manager.Run(graph), SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(), 5);
  const auto case_node = graph->FindNode("case");
  ASSERT_NE(case_node, nullptr);
  EXPECT_EQ(case_node->GetInDataNodes().size(), 3);
  EXPECT_EQ(case_node->GetInDataNodes().at(2)->GetType(), CONSTANT);
}

TEST_F(UtestSubgraphConstMigrationPass, subgraph_const_exception_test) {
  const auto sub1_data_0 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0);
  const auto sub1_data_1 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 1);
  DEF_GRAPH(sub_1) {
    CHAIN(NODE("data_0_ascend_mbatch_batch_0", sub1_data_0)->NODE("add_ascend_mbatch_batch_0", ADD));
    CHAIN(NODE("data_1_ascend_mbatch_batch_0", sub1_data_1)->NODE("add_ascend_mbatch_batch_0", ADD));
    CHAIN(NODE("add_ascend_mbatch_batch_0", ADD)->NODE("netoutput_ascend_mbatch_batch_0", NETOUTPUT));
  };

  const auto sub2_data_0 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0);
  const auto sub2_data_1 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 1);
  DEF_GRAPH(sub_2) {
    CHAIN(NODE("data_0_ascend_mbatch_batch_1", sub1_data_0)->NODE("add_ascend_mbatch_batch_1", ADD));
    CHAIN(NODE("data_1_ascend_mbatch_batch_1", sub1_data_1)->NODE("add_ascend_mbatch_batch_1", ADD));
    CHAIN(NODE("add_ascend_mbatch_batch_1", ADD)->NODE("netoutput_ascend_mbatch_batch_1", NETOUTPUT));
  };

  DEF_GRAPH(g1) {
    CHAIN(NODE("data_0", DATA)->NODE("case", CASE, sub_1, sub_2)->NODE("netoutput", NETOUTPUT));
    CHAIN(NODE("data_1", DATA)->NODE("case"));
  };

  sub_1.Layout();
  sub_2.Layout();
  const auto compute_graph = ToComputeGraph(g1);

  PassManager pass_manager;
  pass_manager.AddPass("SubgraphConstMigrationPass", new (std::nothrow) SubgraphConstMigrationPass);

  // 1. has not ATTR_NAME_BATCH_NUM
  EXPECT_EQ(pass_manager.Run(compute_graph), SUCCESS);
  EXPECT_EQ(compute_graph->GetDirectNodesSize(), 4);

  // Add ATTR_NAME_BATCH_NUM
  const auto case_node = compute_graph->FindNode("case");
  ASSERT_NE(case_node, nullptr);
  (void)AttrUtils::SetInt(case_node->GetOpDesc(), ATTR_NAME_BATCH_NUM, 2);

  // 2. has no const node
  EXPECT_EQ(pass_manager.Run(compute_graph), SUCCESS);
  EXPECT_EQ(compute_graph->GetDirectNodesSize(), 4);

  // 3. Data od subgraph has no parent_node_index
  std::vector<ComputeGraphPtr> subgraphs;
  for (const auto &subgraph : compute_graph->GetAllSubgraphs()) {
    subgraphs.emplace_back(subgraph);
    const auto &data = subgraph->FindNode("data_0_ascend_mbatch_batch_0");
    if (data != nullptr) {
      (void)data->GetOpDesc()->DelAttr(ATTR_NAME_PARENT_NODE_INDEX);
    }
  }
  EXPECT_EQ(pass_manager.Run(compute_graph), FAILED);

  // 4. has not subgraph
  for (const auto &subgraph : subgraphs) {
    compute_graph->RemoveSubGraph(subgraph);
  }
  EXPECT_EQ(pass_manager.Run(compute_graph), FAILED);
}


TEST_F(UtestSubgraphConstMigrationPass, subgraph_expression_migration) {
  PassManager pass_manager;
  pass_manager.AddPass("SubexpressionMigrationPass", new (std::nothrow) SubexpressionMigrationPass);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  make_multibatch_graph_subexpression(graph);
  EXPECT_EQ(pass_manager.Run(graph), SUCCESS);
}
}  // namespace ge
