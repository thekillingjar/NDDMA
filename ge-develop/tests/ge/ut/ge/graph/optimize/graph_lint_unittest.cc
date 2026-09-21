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
#include "graph/preprocess/checker/graph_lint.h"
#include "common/share_graph.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "framework/common/types.h"

using namespace testing;
namespace ge {
namespace {
auto data0 = OP_CFG(DATA)
                .TensorDesc(FORMAT_ND, DT_FLOAT, {1, 2, 3, 4})
                .InCnt(1)
                .OutCnt(1)
                .InNames({"x"})
                .OutNames({"y"})
                .Attr(ATTR_NAME_INDEX, 0)
                .Build("data");
auto var = OP_CFG(VARIABLE)
                    .TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4})
                    .InCnt(1)
                    .OutCnt(1)
                    .InNames({"x"})
                    .OutNames({"y"})
                    .Build("var");
auto relu = OP_CFG(RELU)
                  .TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4})
                  .InCnt(1)
                  .OutCnt(1)
                  .InNames({"x"})
                  .OutNames({"y"})
                  .Build("relu");
auto relu2 = OP_CFG(RELU)
                  .TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4})
                  .InCnt(1)
                  .OutCnt(1)
                  .InNames({"x"})
                  .OutNames({"y"})
                  .Build("relu2");
auto cmo = OP_CFG(CMO)
                  .TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4})
                  .InCnt(1)
                  .OutCnt(1)
                  .InNames({"x"})
                  .OutNames({"y"})
                  .Build("cmo");
auto assign = OP_CFG(ASSIGN)
                  .TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4})
                  .InCnt(2)
                  .OutCnt(1)
                  .InNames({"ref", "value"})
                  .OutNames({"ref"})
                  .Build("assign");

vector<int32_t> data_value(1 * 2 * 3 * 4, 0);
GeTensorDesc data_tensor_desc(GeShape({1,2,3,4}), FORMAT_NCHW, DT_INT32);
GeTensorPtr tensor = std::make_shared<GeTensor>(data_tensor_desc, (uint8_t *)data_value.data(), sizeof(int32_t));
auto const_op = OP_CFG(CONSTANTOP).InCnt(0).OutCnt(1).Weight(tensor).Build("const_op");

/*
 *
 *     var      const
 *    /  \      /
 *  relu  assign
 *     \   /
 *     netoutput
 */
ComputeGraphPtr ReadWriteNoControlGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE(var)->NODE(relu)->NODE("NetOutput", NETOUTPUT));
    CHAIN(NODE(var)->EDGE(0, 0)->NODE(assign)->NODE("NetOutput"));
    CHAIN(NODE(const_op)->EDGE(0, 1)->NODE(assign));
  };
  auto graph = ToComputeGraph(g1);

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"relu", "assign"});
  net_output->GetOpDesc()->SetSrcIndex({0, 1});
  return graph;
}

/*
 *
 *     var      const
 *    /    \      /
 *  assign  assign
 *     \   /
 *     netoutput
 */
ComputeGraphPtr TwoWriteNoControlGraph() {
  auto assign1 = OP_CFG(ASSIGN)
                  .TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4})
                  .InCnt(2)
                  .OutCnt(1)
                  .InNames({"ref", "value"})
                  .OutNames({"ref"})
                  .Build("assign1");
  DEF_GRAPH(g1) {
    CHAIN(NODE(var)->EDGE(0, 0)->NODE(assign1)->NODE("NetOutput", NETOUTPUT));
    CHAIN(NODE(const_op)->EDGE(0, 1)->NODE(assign1));
    CHAIN(NODE(var)->EDGE(0, 0)->NODE(assign)->NODE("NetOutput"));
    CHAIN(NODE(const_op)->EDGE(0, 1)->NODE(assign));
  };
  auto graph = ToComputeGraph(g1);

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"assign1", "assign"});
  net_output->GetOpDesc()->SetSrcIndex({0, 1});
  return graph;
}
/*
 *
 *     var      const
 *    /  \      /
 *  cmo  assign
 *     \   /
 *     netoutput
 */
ComputeGraphPtr CmoWriteNoControlGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE(var)->NODE(cmo)->NODE("NetOutput", NETOUTPUT));
    CHAIN(NODE(var)->EDGE(0, 0)->NODE(assign)->NODE("NetOutput"));
    CHAIN(NODE(const_op)->EDGE(0, 1)->NODE(assign));
  };
  auto graph = ToComputeGraph(g1);

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"cmo", "assign"});
  net_output->GetOpDesc()->SetSrcIndex({0, 1});
  return graph;
}
/*
 *
 *      var      const
 *    /     \      /
 *  relu-->  assign
 *     \   /
 *     netoutput
 */
ComputeGraphPtr ReadWriteDirectControlGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE(var)->NODE(relu)->NODE("NetOutput", NETOUTPUT));
    CHAIN(NODE(var)->EDGE(0, 0)->NODE(assign)->NODE("NetOutput"));
    CHAIN(NODE(const_op)->EDGE(0, 1)->NODE(assign));
    CHAIN(NODE(relu)->Ctrl()->NODE(assign));
  };
  auto graph = ToComputeGraph(g1);

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"relu", "assign"});
  net_output->GetOpDesc()->SetSrcIndex({0, 1});
  return graph;
}

/*
 *
 *     var      const
 *    |  \      /
 *    |   assign
 *    |     |
 *    |     relu2
 *    |    /c
 *    relu
 *      |
 *     netoutput
 */
ComputeGraphPtr ReadWriteIndirectControlGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE(var)->NODE(relu)->NODE("NetOutput", NETOUTPUT));
    CHAIN(NODE(var)->EDGE(0, 0)->NODE(assign)->NODE(relu2)->Ctrl()->NODE(relu));
    CHAIN(NODE(const_op)->EDGE(0, 1)->NODE(assign));
  };
  auto graph = ToComputeGraph(g1);

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"relu"});
  net_output->GetOpDesc()->SetSrcIndex({0});
  return graph;
}

/*
*      var    data         then:                  else:
 *    /     \ |               data  const             data
 *  relu    if                |   /                   |
 *     \   /                 assign                  relu
 *     netoutput               |                      |
 *                           netoutput               netoutput
 */
ComputeGraphPtr WriteInSubgraphReadInRootgraph(bool with_control_in_root) {
  auto main_graph = [&with_control_in_root]() {
    DEF_GRAPH(g) {
      CHAIN(NODE("pred", DATA)->NODE("if", "If")->EDGE(0,1)->NODE("NetOutput", NETOUTPUT));
      CHAIN(NODE(var)->NODE(relu)->NODE("NetOutput"));
      CHAIN(NODE(var)->EDGE(0, 1)->NODE("if"));
      if (with_control_in_root) {
        CHAIN(NODE(relu)->Ctrl()->NODE("if"));
      }
    };
    return ToComputeGraph(g);
  }();
  main_graph->SetName("main");
  ge::AttrUtils::SetInt(main_graph->FindNode("pred")->GetOpDesc(), "index", 0);

  auto if_node = main_graph->FindNode("if");
  if_node->GetOpDesc()->AppendIrInput("cond", kIrInputRequired);
  if_node->GetOpDesc()->AppendIrInput("input", kIrInputDynamic);
  auto &names_indexes = if_node->GetOpDesc()->MutableAllInputName();
  names_indexes.clear();
  names_indexes["cond"] = 0;
  names_indexes["input"] = 1;

  auto then_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE(const_op)->EDGE(0, 1)->NODE(assign)->NODE("NetOutput", "NetOutput"));
      CHAIN(NODE(data0)->EDGE(0, 0)->NODE(assign));
    };
    return ToComputeGraph(g);
  }();
  then_graph->SetName("then");
  ge::AttrUtils::SetInt(then_graph->FindFirstNodeMatchType("Data")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(then_graph->FindFirstNodeMatchType("Data")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);

  auto else_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE(data0)->NODE(relu)->NODE("NetOutput", "NetOutput"));
    };
    return ToComputeGraph(g);
  }();
  else_graph->SetName("else");
  ge::AttrUtils::SetInt(else_graph->FindFirstNodeMatchType("Data")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(else_graph->FindFirstNodeMatchType("Data")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);

  then_graph->SetParentGraph(main_graph);
  then_graph->SetParentNode(if_node);
  else_graph->SetParentGraph(main_graph);
  else_graph->SetParentNode(if_node);

  main_graph->AddSubgraph(then_graph);
  main_graph->AddSubgraph(else_graph);
  if_node->GetOpDesc()->AddSubgraphName("then");
  if_node->GetOpDesc()->AddSubgraphName("else");
  if_node->GetOpDesc()->SetSubgraphInstanceName(0, "then");
  if_node->GetOpDesc()->SetSubgraphInstanceName(1, "else");
  main_graph->TopologicalSorting();

  auto net_output = main_graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"relu", "if"});
  net_output->GetOpDesc()->SetSrcIndex({0, 1});

  return main_graph;
}

/*
*      var    data         then:                  else:
 *    /     \ |               data  const               data
 *  relu    if               /   \   /                    |
 *     \   /               relu assign                  relu
 *     netoutput             \  /                         |
 *                           netoutput               netoutput
 */
ComputeGraphPtr BothConflictInSubgraphAndRootgraph() {
  auto main_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("pred", DATA)->NODE("if", "If")->EDGE(0,1)->NODE("NetOutput", NETOUTPUT));
      CHAIN(NODE(var)->NODE(relu)->NODE("NetOutput"));
      CHAIN(NODE(var)->EDGE(0, 1)->NODE("if"));
    };
    return ToComputeGraph(g);
  }();
  main_graph->SetName("main");
  ge::AttrUtils::SetInt(main_graph->FindNode("pred")->GetOpDesc(), "index", 0);

  auto if_node = main_graph->FindNode("if");
  if_node->GetOpDesc()->AppendIrInput("cond", kIrInputRequired);
  if_node->GetOpDesc()->AppendIrInput("input", kIrInputDynamic);
  auto &names_indexes = if_node->GetOpDesc()->MutableAllInputName();
  names_indexes.clear();
  names_indexes["cond"] = 0;
  names_indexes["input"] = 1;

  auto then_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE(const_op)->EDGE(0, 1)->NODE(assign)->NODE("NetOutput", "NetOutput"));
      CHAIN(NODE(data0)->EDGE(0, 0)->NODE(assign));
      CHAIN(NODE(data0)->EDGE(0, 0)->NODE(relu)->Ctrl()->NODE("NetOutput"));
    };
    return ToComputeGraph(g);
  }();
  then_graph->SetName("then");
  ge::AttrUtils::SetInt(then_graph->FindFirstNodeMatchType("Data")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(then_graph->FindFirstNodeMatchType("Data")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);

  auto else_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE(data0)->NODE(relu)->NODE("NetOutput", "NetOutput"));
    };
    return ToComputeGraph(g);
  }();
  else_graph->SetName("else");
  ge::AttrUtils::SetInt(else_graph->FindFirstNodeMatchType("Data")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(else_graph->FindFirstNodeMatchType("Data")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);

  then_graph->SetParentGraph(main_graph);
  then_graph->SetParentNode(if_node);
  else_graph->SetParentGraph(main_graph);
  else_graph->SetParentNode(if_node);

  main_graph->AddSubgraph(then_graph);
  main_graph->AddSubgraph(else_graph);
  if_node->GetOpDesc()->AddSubgraphName("then");
  if_node->GetOpDesc()->AddSubgraphName("else");
  if_node->GetOpDesc()->SetSubgraphInstanceName(0, "then");
  if_node->GetOpDesc()->SetSubgraphInstanceName(1, "else");
  main_graph->TopologicalSorting();

  auto net_output = main_graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"relu", "if"});
  net_output->GetOpDesc()->SetSrcIndex({0, 1});

  return main_graph;
}

/*
*      var    data         then:                  else:
 *    /     \ |               data                 data
 *  assign    if                |                      |
 *     \     /                 relu                  relu
 *     netoutput               |                      |
 *                           netoutput               netoutput
 */
ComputeGraphPtr ReadInSubgraphWriteInRootgraphNoControlGraph() {
  auto main_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("pred", DATA)->NODE("if", "If")->EDGE(0,1)->NODE("NetOutput", NETOUTPUT));
      CHAIN(NODE(var)->NODE(assign)->NODE("NetOutput"));
      CHAIN(NODE(var)->EDGE(0,1)->NODE("if"));
      CHAIN(NODE(const_op)->EDGE(0, 1)->NODE(assign));
    };
    return ToComputeGraph(g);
  }();
  main_graph->SetName("main");
  ge::AttrUtils::SetInt(main_graph->FindNode("pred")->GetOpDesc(), "index", 0);

  auto if_node = main_graph->FindNode("if");
  if_node->GetOpDesc()->AppendIrInput("cond", kIrInputRequired);
  if_node->GetOpDesc()->AppendIrInput("input", kIrInputDynamic);
  auto &names_indexes = if_node->GetOpDesc()->MutableAllInputName();
  names_indexes.clear();
  names_indexes["cond"] = 0;
  names_indexes["input"] = 1;

  auto then_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE(data0)->NODE(relu)->NODE("NetOutput", "NetOutput"));
    };
    return ToComputeGraph(g);
  }();
  then_graph->SetName("then");
  ge::AttrUtils::SetInt(then_graph->FindFirstNodeMatchType("Data")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(then_graph->FindFirstNodeMatchType("Data")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);

  auto else_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE(data0)->NODE(relu)->NODE("NetOutput", "NetOutput"));
    };
    return ToComputeGraph(g);
  }();
  else_graph->SetName("else");
  ge::AttrUtils::SetInt(else_graph->FindFirstNodeMatchType("Data")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(else_graph->FindFirstNodeMatchType("Data")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);

  then_graph->SetParentGraph(main_graph);
  then_graph->SetParentNode(if_node);
  else_graph->SetParentGraph(main_graph);
  else_graph->SetParentNode(if_node);

  main_graph->AddSubgraph(then_graph);
  main_graph->AddSubgraph(else_graph);
  if_node->GetOpDesc()->AddSubgraphName("then");
  if_node->GetOpDesc()->AddSubgraphName("else");
  if_node->GetOpDesc()->SetSubgraphInstanceName(0, "then");
  if_node->GetOpDesc()->SetSubgraphInstanceName(1, "else");
  main_graph->TopologicalSorting();

  auto net_output = main_graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"if", "assign"});
  net_output->GetOpDesc()->SetSrcIndex({0, 1});

  return main_graph;
}
} // namespace
class UtestGraphLintTest : public testing::Test {
 protected:
  void SetUp() {
    origin_log_level_ = dlog_getlevel(GE_MODULE_NAME, &origin_event_level_);
    dlog_setlevel(GE_MODULE_NAME, 1, 0);
  }

  void TearDown() {
    dlog_setlevel(GE_MODULE_NAME, origin_log_level_, origin_event_level_);
  }
private:
  int32_t origin_log_level_ = -1;
  int32_t origin_event_level_ = -1;
};
/*
┌───────┐  (0,0)   ┌────────┐  (0,0)   ┌───────────┐
│ data1 │ ───────> │  add1  │ ───────> │ NetOutput │
└───────┘          └────────┘          └───────────┘
                     ∧
                     │ (0,1)
                     │
                   ┌────────┐
                   │ data2  │
                   └────────┘
 */
TEST_F(UtestGraphLintTest, testValid_simple_graph_success) {
  auto compute_graph = gert::ShareGraph::AicoreGraph();
  GraphLint graph_lint;
  EXPECT_EQ(graph_lint.Verify(compute_graph), GRAPH_SUCCESS);
}
/*
g

           (0,1)
  ┌───────────────────┐
  │                   ∨
┌────────┐  (0,0)   ┌──────┐  (0,0)   ┌────────┐  (0,1)   ┌────────┐  (0,0)   ┌────────┐  (0,0)   ┌───────────┐
│ input0 │ ───────> │ add0 │ ───────> │  add1  │ ───────> │   if   │ ───────> │  add2  │ ───────> │ NetOutput │
└────────┘          └──────┘          └────────┘          └────────┘          └────────┘          └───────────┘
                                        ∧                   ∧                   ∧
                                        │ (0,1)             │ (0,0)             │ (0,1)
                                        │                   │                   │
                                      ┌────────┐          ┌────────┐          ┌────────┐
                                      │ input1 │          │  pred  │          │ input2 │
                                      └────────┘          └────────┘          └────────┘

                       g

┌────┐  (0,0)   ┌────────┐  (0,0)   ┌───────────┐
│ c0 │ ───────> │  add3  │ ───────> │ NetOutput │
└────┘          └────────┘          └───────────┘
                  ∧
                  │ (0,1)
                  │
                ┌────────┐
                │  data  │
                └────────┘

                       g

         (0,1)
  ┌─────────────────┐
  │                 ∨
┌──────┐  (0,0)   ┌──────┐  (0,0)   ┌───────────┐
│ data │ ───────> │ add4 │ ───────> │ NetOutput │
└──────┘          └──────┘          └───────────┘
 */
TEST_F(UtestGraphLintTest, testValid_graph_with_subgraph_success) {
  auto compute_graph = gert::ShareGraph::IfGraph4();
  GraphLint graph_lint;
  EXPECT_EQ(graph_lint.Verify(compute_graph), GRAPH_SUCCESS);
}
/*

                                g

┌──────────────┐  (0,0)   ┌───────────┐  (0,1)   ┌────────────────┐
│ loop_counter │ ───────> │ NetOutput │ <─────── │ max_iterations │
└──────────────┘          └───────────┘          └────────────────┘

                                g

                                             (1,1)
                            ┌───────────────────────────┐
                            │                           ∨
┌──────────────┐  (0,0)   ┌────────────────┐  (2,2)   ┌───────────┐
│    input     │ ───────> │                │ ───────> │ NetOutput │
└──────────────┘          │                │          └───────────┘
                          │                │  (0,0)     ∧
                          │     while      │ ───────────┘
                          │                │
┌──────────────┐  (0,1)   │                │
│ loop_counter │ ───────> │                │
└──────────────┘          └────────────────┘
                            ∧
                            │ (0,2)
                            │
                          ┌────────────────┐
                          │ max_iterations │
                          └────────────────┘

                          g

┌───────┐  (0,0)   ┌───────────┐  (0,0)   ┌───────────┐
│ input │ ───────> │    foo    │ ───────> │ NetOutput │
└───────┘          └───────────┘          └───────────┘
                     ∧
                     │ (0,1)
                     │
                   ┌───────────┐
                   │ max_value │
                   └───────────┘

                                   g

┌────────┐  (0,0)   ┌────────┐  (0,0)   ┌───────────┐  (0,2)   ┌────────┐
│ input0 │ ───────> │  bar   │ ───────> │ NetOutput │ <─────── │ input2 │
└────────┘          └────────┘          └───────────┘          └────────┘
                                          ∧
                                          │
                                          │
┌────────┐  (0,1)   ┌────────┐  (0,1)     │
│  one   │ ───────> │  add   │ ───────────┘
└────────┘          └────────┘
                      ∧
                      │ (0,0)
                      │
                    ┌────────┐
                    │ input1 │
                    └────────┘

                             g

                                      (2,2)
                     ┌───────────────────────────┐
                     │                           ∨
┌───────┐  (0,0)   ┌────────────────┐  (0,0)   ┌───────────┐
│ input │ ───────> │                │ ───────> │ NetOutput │
└───────┘          │                │          └───────────┘
                   │                │  (1,1)     ∧
                   │ partitioncall2 │ ───────────┘
                   │                │
                   │                │
                   │                │ <┐
                   └────────────────┘  │
                     ∧                 │
                     │ (0,1)           │ (1,2)
                     │                 │
                   ┌────────────────┐  │
                   │ partitioncall1 │ ─┘
                   └────────────────┘
 */
TEST_F(UtestGraphLintTest, testValid_graph_with_while_success) {
  auto compute_graph = gert::ShareGraph::WhileGraphInPartitionCall(true);
  GraphLint graph_lint;
  EXPECT_EQ(graph_lint.Verify(compute_graph), GRAPH_SUCCESS);
}
/*
                        ┌································┐
                        ∨                                :
┌──────────┐  (0,1)   ┌────────┐  (0,1)   ┌───────────┐  :
│ const_op │ ───────> │ assign │ ───────> │ NetOutput │  :
└──────────┘          └────────┘          └───────────┘  :
                        ∧                   ∧            :
                        │ (0,0)             │ (0,0)      :
                        │                   │            :
                      ┌────────┐  (0,0)   ┌───────────┐  :
                      │  var   │ ───────> │   relu    │ ·┘
                      └────────┘          └───────────┘
 */
TEST_F(UtestGraphLintTest, testValid_simple_graph_read_write_direct_control_success) {
  auto compute_graph = ReadWriteDirectControlGraph();
  GraphLint graph_lint;
  EXPECT_EQ(graph_lint.Verify(compute_graph), GRAPH_SUCCESS);
}
/*

┌──────────┐  (0,1)   ┌────────┐  (0,0)   ┌───────┐     ┌──────┐  (0,0)   ┌───────────┐
│ const_op │ ───────> │ assign │ ───────> │ relu2 │ ··> │ relu │ ───────> │ NetOutput │
└──────────┘          └────────┘          └───────┘     └──────┘          └───────────┘
                        ∧                                 ∧
                        │ (0,0)                           │
                        │                                 │
                      ┌────────┐  (0,0)                   │
                      │  var   │ ─────────────────────────┘
                      └────────┘
 */
TEST_F(UtestGraphLintTest, testValid_simple_graph_read_write_indirect_control_success) {
  auto compute_graph = ReadWriteIndirectControlGraph();
  GraphLint graph_lint;
  EXPECT_EQ(graph_lint.Verify(compute_graph), GRAPH_SUCCESS);
}

/**
*                           main graph


                    ┌································┐
                    ∨                                :
┌──────┐  (0,0)   ┌────────┐  (0,1)   ┌───────────┐  :
│ pred │ ───────> │   if   │ ───────> │ NetOutput │  :
└──────┘          └────────┘          └───────────┘  :
                    ∧                   ∧            :
                    │ (0,1)             │ (0,2)      :
                    │                   │            :
                  ┌────────┐  (0,0)   ┌───────────┐  :
                  │  var   │ ───────> │   relu    │ ·┘
                  └────────┘          └───────────┘

                        then graph

┌──────────┐  (0,1)   ┌────────┐  (0,0)   ┌───────────┐
│ const_op │ ───────> │ assign │ ───────> │ NetOutput │
└──────────┘          └────────┘          └───────────┘
                        ∧
                        │ (0,0)
                        │
                      ┌────────┐
                      │  data  │
                      └────────┘

                       else graph

┌──────┐  (0,0)   ┌──────┐  (0,0)   ┌───────────┐
│ data │ ───────> │ relu │ ───────> │ NetOutput │
└──────┘          └──────┘          └───────────┘
 **/
TEST_F(UtestGraphLintTest, testValid_if_graph_writeInSub_ReadInRoot_WithControl_success) {
  auto compute_graph = WriteInSubgraphReadInRootgraph(true);
  GraphLint graph_lint;
  EXPECT_EQ(graph_lint.Verify(compute_graph), GRAPH_SUCCESS);
}
/**
g1

┌──────────┐  (0,1)   ┌────────┐  (0,1)   ┌───────────┐
│ const_op │ ───────> │ assign │ ───────> │ NetOutput │
└──────────┘          └────────┘          └───────────┘
                        ∧                   ∧
                        │ (0,0)             │ (0,0)
                        │                   │
                      ┌────────┐  (0,0)   ┌───────────┐
                      │  var   │ ───────> │    cmo    │
                      └────────┘          └───────────┘
 */
TEST_F(UtestGraphLintTest, testValid_cmo_Write_NoControl_success) {
  auto compute_graph = CmoWriteNoControlGraph();
  GraphLint graph_lint;
  EXPECT_EQ(graph_lint.Verify(compute_graph), GRAPH_SUCCESS);
}

/**
*                          g1

┌──────────┐  (0,1)   ┌────────┐  (0,1)   ┌───────────┐
│ const_op │ ───────> │ assign │ ───────> │ NetOutput │
└──────────┘          └────────┘          └───────────┘
                        ∧                   ∧
                        │ (0,0)             │ (0,0)
                        │                   │
                      ┌────────┐  (0,0)   ┌───────────┐
                      │  var   │ ───────> │   relu    │
                      └────────┘          └───────────┘
 **/
TEST_F(UtestGraphLintTest, testInvalid_simple_graph_ReadWriteNoControl) {
  auto compute_graph = ReadWriteNoControlGraph();
  GraphLint graph_lint;
  EXPECT_NE(graph_lint.Verify(compute_graph), GRAPH_SUCCESS);
}
/**
*                        main graph

                             (1,3)
                    ┌───────────────────┐
                    │                   ∨
┌──────┐  (0,0)   ┌────────┐  (0,1)   ┌───────────┐
│ pred │ ───────> │   if   │ ───────> │ NetOutput │
└──────┘          └────────┘          └───────────┘
                    ∧                   ∧
                    │ (0,1)             │ (0,2)
                    │                   │
                  ┌────────┐  (0,0)   ┌───────────┐
                  │  var   │ ───────> │   relu    │
                  └────────┘          └───────────┘

                          then graph

┌──────────┐  (0,1)   ┌────────┐  (0,0)   ┌───────────┐
│ const_op │ ───────> │ assign │ ───────> │ NetOutput │
└──────────┘          └────────┘          └───────────┘
                        ∧
                        │ (0,0)
                        │
                      ┌────────┐
                      │  data  │
                      └────────┘

                       else graph

┌──────┐  (0,0)   ┌──────┐  (0,0)   ┌───────────┐
│ data │ ───────> │ relu │ ───────> │ NetOutput │
└──────┘          └──────┘          └───────────┘
 **/
TEST_F(UtestGraphLintTest, testInvalid_if_graph_writeInSub_ReadInRoot_NoControl_success) {
  auto compute_graph = WriteInSubgraphReadInRootgraph(false);
  GraphLint graph_lint;
  EXPECT_NE(graph_lint.Verify(compute_graph), GRAPH_SUCCESS);
}
/*
┌──────────┐  (0,1)   ┌────────┐  (0,2)   ┌───────────┐
          │ const_op │ ───────> │ assign │ ───────> │ NetOutput │
          └──────────┘          └────────┘          └───────────┘
                                  ∧                   ∧
  ┌─────────────────────┐         │ (0,0)             │
  │                     │         │                   │
  │       ┌──────────┐  │       ┌────────┐            │
  │       │   pred   │  └────── │  var   │            │
  │       └──────────┘          └────────┘            │
  │ (0,1)   │                                         │
  │         │ (0,0)                                   │
  │         ∨                                         │
  │       ┌──────────┐  (0,1)                         │
  └─────> │    if    │ ───────────────────────────────┘
          └──────────┘

                       g

┌──────┐  (0,0)   ┌──────┐  (0,0)   ┌───────────┐
│ data │ ───────> │ relu │ ───────> │ NetOutput │
└──────┘          └──────┘          └───────────┘

                       g

┌──────┐  (0,0)   ┌──────┐  (0,0)   ┌───────────┐
│ data │ ───────> │ relu │ ───────> │ NetOutput │
└──────┘          └──────┘          └───────────┘
 */
TEST_F(UtestGraphLintTest, testInvalid_if_graph_ReadInSub_WriteInRoot_NoControl_success) {
  auto compute_graph = ReadInSubgraphWriteInRootgraphNoControlGraph();
  GraphLint graph_lint;
  EXPECT_NE(graph_lint.Verify(compute_graph), GRAPH_SUCCESS);
}

/*
root graph

┌──────┐  (0,0)   ┌────────┐  (0,1)   ┌───────────┐
│ pred │ ───────> │   if   │ ───────> │ NetOutput │
└──────┘          └────────┘          └───────────┘
                    ∧                   ∧
                    │ (0,1)             │ (0,2)
                    │                   │
                  ┌────────┐  (0,0)   ┌───────────┐
                  │  var   │ ───────> │   relu    │
                  └────────┘          └───────────┘

                          then graph

┌──────────┐  (0,1)   ┌────────┐  (0,0)   ┌───────────┐
│ const_op │ ───────> │ assign │ ───────> │ NetOutput │
└──────────┘          └────────┘          └───────────┘
                        ∧                   ∧
                        │ (0,0)             :
                        │                   :
                      ┌────────┐  (0,0)   ┌───────────┐
                      │  data  │ ───────> │   relu    │
                      └────────┘          └───────────┘

                       else graph

┌──────┐  (0,0)   ┌──────┐  (0,0)   ┌───────────┐
│ data │ ───────> │ relu │ ───────> │ NetOutput │
└──────┘          └──────┘          └───────────┘
 */
TEST_F(UtestGraphLintTest, testInvalid_if_graph_BothConflict_SubGraph_Root_NoControl_success) {
  auto compute_graph = BothConflictInSubgraphAndRootgraph();
  GraphLint graph_lint;
  EXPECT_NE(graph_lint.Verify(compute_graph), GRAPH_SUCCESS);
}

TEST_F(UtestGraphLintTest, testInvalid_simple_graph_TwoWriteNode_NoControl_success) {
  auto compute_graph = TwoWriteNoControlGraph();
  GraphLint graph_lint;
  EXPECT_NE(graph_lint.Verify(compute_graph), GRAPH_SUCCESS);
}

TEST_F(UtestGraphLintTest, testInvalid_subgraph_not_support) {
  auto compute_graph = BothConflictInSubgraphAndRootgraph();
  auto subgraph = compute_graph->GetAllSubgraphs().at(0);
  GraphLint graph_lint;
  EXPECT_NE(graph_lint.Verify(subgraph), GRAPH_SUCCESS);
}
} // namespace ge
