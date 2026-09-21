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
#include <nlohmann/json.hpp>
#include <string>
#include "easy_graph/graph/box.h"
#include "easy_graph/graph/node.h"
#include "easy_graph/builder/graph_dsl.h"
#include "easy_graph/builder/box_builder.h"
#include "easy_graph/layout/graph_layout.h"
#include "easy_graph/layout/engines/graph_easy/graph_easy_option.h"
#include "easy_graph/layout/engines/graph_easy/graph_easy_executor.h"
#include "graph/graph.h"
#include "graph/compute_graph.h"
#include "framework/common/types.h"
#include "graph/debug/ge_attr_define.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "ge_graph_dsl/op_desc/op_desc_cfg_box.h"
#include "ge_graph_dsl/op_desc/op_desc_node_builder.h"
USING_GE_NS

class GraphDslSTest : public testing::Test {
 private:
  EG_NS::GraphEasyExecutor executor;

 protected:
  void SetUp() { EG_NS::GraphLayout::GetInstance().Config(executor, nullptr); }

  void TearDown() {}
};

TEST_F(GraphDslSTest, test_build_graph_from_optype_with_name) {
  DEF_GRAPH(g1) { CHAIN(NODE("data1", DATA)->NODE("add", ADD)); };

  auto geGraph = ToGeGraph(g1);
  auto computeGraph = ToComputeGraph(g1);

  ASSERT_EQ(geGraph.GetAllNodes().size(), 2);
  ASSERT_EQ(computeGraph->GetAllNodesSize(), 2);
}

TEST_F(GraphDslSTest, test_build_graph_with_name) {
  DEF_GRAPH(g1, "sample_graph") { CHAIN(NODE("data1", DATA)->NODE("add", ADD)); };

  auto geGraph = ToGeGraph(g1);

  ASSERT_EQ(geGraph.GetAllNodes().size(), 2);
  ASSERT_EQ(geGraph.GetName(), "sample_graph");
}

TEST_F(GraphDslSTest, test_build_from_from_op_desc_ptr) {
  DEF_GRAPH(g1) {
                  auto data = std::make_shared<OpDesc>("data1", DATA);
                  auto add = std::make_shared<OpDesc>("Add", ADD);
                  CHAIN(NODE(data)->NODE(add));
                };

  auto geGraph = ToGeGraph(g1);

  ASSERT_EQ(geGraph.GetAllNodes().size(), 2);
}

TEST_F(GraphDslSTest, test_build_from_op_desc_cfg) {
  DEF_GRAPH(g1) {
                  auto datCfg = OP_CFG(DATA).InCnt(1).OutCnt(1);
                  auto addCfg = OP_CFG(DATA).InCnt(1).OutCnt(1);
                  CHAIN(NODE("data1", datCfg)->NODE("add", addCfg));
                };

  auto geGraph = ToGeGraph(g1);

  ASSERT_EQ(geGraph.GetAllNodes().size(), 2);
}

TEST_F(GraphDslSTest, test_build_from_op_desc_cfg_inline) {
  DEF_GRAPH(g1) { CHAIN(NODE("data1", OP_CFG(DATA).InCnt(1).OutCnt(1))->NODE("add", OP_CFG(ADD).InCnt(2).OutCnt(1))); };

  auto geGraph = ToGeGraph(g1);

  ASSERT_EQ(geGraph.GetAllNodes().size(), 2);
}

TEST_F(GraphDslSTest, test_build_from_control_chain) {
  DEF_GRAPH(g1) { CTRL_CHAIN(NODE("data1", DATA)->NODE("add", ADD)); };

  auto geGraph = ToGeGraph(g1);

  ASSERT_EQ(geGraph.GetAllNodes().size(), 2);
}

TEST_F(GraphDslSTest, test_build_from_data_chain) {
  DEF_GRAPH(g1) { DATA_CHAIN(NODE("data1", DATA)->NODE("add", ADD)); };

  auto geGraph = ToGeGraph(g1);

  ASSERT_EQ(geGraph.GetAllNodes().size(), 2);
}

TEST_F(GraphDslSTest, test_build_from_data_chain_with_edge) {
  DEF_GRAPH(g1) {
                  CTRL_CHAIN(NODE("data1", DATA)->NODE("add", ADD));
                  CHAIN(NODE("data1", DATA)->EDGE(2, 2)->NODE("add"));
                };

  auto geGraph = ToGeGraph(g1);

  ASSERT_EQ(geGraph.GetAllNodes().size(), 2);
}

TEST_F(GraphDslSTest, test_build_graph_reused_before_node) {
  DEF_GRAPH(g1) {
                  CTRL_CHAIN(NODE("data1", DATA)->NODE("add", ADD));
                  CHAIN(NODE("data1")->EDGE(2, 2)->NODE("add"));
                };

  auto geGraph = ToGeGraph(g1);

  ASSERT_EQ(geGraph.GetAllNodes().size(), 2);
}

TEST_F(GraphDslSTest, test_build_graph_with_constant_folding) {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("data1", DATA)->NODE("add", ADD));
                  CHAIN(NODE("data2", DATA)->NODE("add"));
                };

  auto geGraph = ToGeGraph(g1);

  ASSERT_EQ(geGraph.GetAllNodes().size(), 3);
}

TEST_F(GraphDslSTest, test_build_complex_normal_graph_build_suggested) {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("w1", VARIABLE)->NODE("prefetch1", HCOMALLGATHER)->NODE("Add1", ADD));
                  CHAIN(NODE("w2", VARIABLE)->NODE("prefetch2", HCOMALLGATHER)->NODE("Add2", ADD));
                  CHAIN(NODE("w3", VARIABLE)->NODE("prefetch3", HCOMALLGATHER)->NODE("Add3", ADD));
                  CHAIN(NODE("w4", VARIABLE)->NODE("prefetch4", HCOMALLGATHER)->NODE("Add4", ADD));
                  CHAIN(NODE("w5", VARIABLE)->NODE("prefetch5", HCOMALLGATHER)->NODE("Add5", ADD));
                  CHAIN(NODE("const1", CONSTANTOP)
                            ->NODE("Add1")
                            ->NODE("Add2")
                            ->NODE("Add3")
                            ->NODE("Add4")
                            ->NODE("Add5")
                            ->NODE("net_output", NETOUTPUT));
                };

  auto geGraph = ToGeGraph(g1);

  ASSERT_EQ(geGraph.GetAllNodes().size(), 17);
}

TEST_F(GraphDslSTest, test_build_complex_mult_normal_graph_build) {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("w1", VARIABLE)->NODE("prefetch1", HCOMALLGATHER)->NODE("add1", ADD));
                  CHAIN(NODE("w2", VARIABLE)->NODE("prefetch2", HCOMALLGATHER)->NODE("add1"));
                  CHAIN(NODE("w3", VARIABLE)->NODE("prefetch3", HCOMALLGATHER)->NODE("add2", ADD));
                  CHAIN(NODE("w4", VARIABLE)->NODE("prefetch4", HCOMALLGATHER)->NODE("add2"));
                  CHAIN(NODE("w5", VARIABLE)->NODE("prefetch5", HCOMALLGATHER)->NODE("add3", ADD));
                  CHAIN(NODE("const1", CONSTANTOP)->NODE("add3"));
                  CHAIN(NODE("add1")->NODE("net_output", NETOUTPUT));
                  CHAIN(NODE("add2")->NODE("net_output"));
                  CHAIN(NODE("add3")->NODE("net_output"));
                  CTRL_CHAIN(NODE("add1")->NODE("add2")->NODE("add3"));
                };

  auto geGraph = ToGeGraph(g1);

  ASSERT_EQ(geGraph.GetAllNodes().size(), 15);
}

TEST_F(GraphDslSTest, test_build_graph_with_sub_graph) {
  DEF_GRAPH(sub_1) {
                     CHAIN(NODE("data_i", DATA)->NODE("less", LESS)->NODE("netoutput", NETOUTPUT));
                     CHAIN(NODE("const_5", CONSTANTOP)->NODE("less"));
                   };

  DEF_GRAPH(sub_2) {
                     CHAIN(NODE("data_a", DATA)->NODE("mul", MUL)->NODE("netoutput", NETOUTPUT));
                     CHAIN(NODE("const_2", CONSTANTOP)->NODE("mul"));
                   };

  DEF_GRAPH(g1) {
                  CHAIN(NODE("data_a", DATA)->NODE("while", WHILE, sub_1, sub_2)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("data_i", DATA)->NODE("while"));
                };

  sub_1.Layout();
  sub_2.Layout();
  auto geGraph = ToGeGraph(g1);

  ASSERT_EQ(geGraph.GetAllNodes().size(), 12);
}
