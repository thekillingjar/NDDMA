/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/shape_optimize/reshape_remove_pass.h"

#include <gtest/gtest.h>
#include "ge_graph_dsl/graph_dsl.h"

namespace ge {
class UtestReshapeRemovePass : public testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {
    for (auto &name_to_pass : names_to_pass) {
      delete name_to_pass.second;
    }
  }

  NamesToPass names_to_pass;
};


TEST_F(UtestReshapeRemovePass, reshape_remove_succ) {
  DEF_GRAPH(g1) {
    CHAIN(NODE("var1", VARIABLE)->EDGE(0, 0)->NODE("add1", ADD));
    CHAIN(NODE("var2", VARIABLE)->EDGE(0, 1)->NODE("add1", ADD));

    CHAIN(NODE("add1", ADD)->EDGE(0, 0)->NODE("reshape", RESHAPE));
    CHAIN(NODE("const1", CONSTANT)->EDGE(0, 1)->NODE("reshape", RESHAPE));
    CHAIN(NODE("reshape", RESHAPE)->EDGE(0, 0)->NODE("net_output", NETOUTPUT));
  };
  auto compute_graph = ToComputeGraph(g1);
  EXPECT_NE(compute_graph->FindNode("reshape"), nullptr);
  names_to_pass.push_back({"ReshapeRemovePass", new ReshapeRemovePass});
  GEPass pass(compute_graph);
  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);
  EXPECT_EQ(compute_graph->FindNode("reshape"), nullptr);
}

TEST_F(UtestReshapeRemovePass, unknown_shape_reshape_not_remove) {
  DEF_GRAPH(g1) {
    CHAIN(NODE("var1", VARIABLE)->EDGE(0, 0)->NODE("add1", ADD));
    CHAIN(NODE("var2", VARIABLE)->EDGE(0, 1)->NODE("add1", ADD));

    CHAIN(NODE("add1", ADD)->EDGE(0, 0)->NODE("reshape", RESHAPE));
    CHAIN(NODE("const1", CONSTANT)->EDGE(0, 1)->NODE("reshape", RESHAPE));
    CHAIN(NODE("reshape", RESHAPE)->EDGE(0, 0)->NODE("net_output", NETOUTPUT));
  };
  auto compute_graph = ToComputeGraph(g1);
  const auto reshape_node = compute_graph->FindNode("reshape");
  ASSERT_NE(reshape_node, nullptr);
  reshape_node->GetOpDesc()->MutableInputDesc(0)->SetShape(GeShape({-1, 2, 2}));
  names_to_pass.push_back({"ReshapeRemovePass", new ReshapeRemovePass});
  GEPass pass(compute_graph);
  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);
  EXPECT_NE(compute_graph->FindNode("reshape"), nullptr);
}

TEST_F(UtestReshapeRemovePass, reformat_remove_succ) {
  DEF_GRAPH(g1) {
    CHAIN(NODE("var1", VARIABLE)->EDGE(0, 0)->NODE("add1", ADD));
    CHAIN(NODE("var2", VARIABLE)->EDGE(0, 1)->NODE("add1", ADD));

    CHAIN(NODE("add1", ADD)->EDGE(0, 0)->NODE("reformat", REFORMAT));
    CHAIN(NODE("const1", CONSTANT)->EDGE(0, 1)->NODE("reformat", REFORMAT));
    CHAIN(NODE("reformat", REFORMAT)->EDGE(0, 0)->NODE("net_output", NETOUTPUT));
  };
  auto compute_graph = ToComputeGraph(g1);
  EXPECT_NE(compute_graph->FindNode("reformat"), nullptr);
  names_to_pass.push_back({"ReshapeRemovePass", new ReshapeRemovePass});
  GEPass pass(compute_graph);
  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);
  EXPECT_EQ(compute_graph->FindNode("reformat"), nullptr);
}

// when delete reshape, maybe let graph traversal stop, which cause
// repass too many times
TEST_F(UtestReshapeRemovePass, reshape_multi_remove_succ) {
  DEF_GRAPH(g1) {
    CHAIN(NODE("var1", VARIABLE)->EDGE(0, 0)->NODE("add1", ADD));
    CHAIN(NODE("var2", VARIABLE)->EDGE(0, 1)->NODE("add1", ADD));
    CHAIN(NODE("add1", ADD)->EDGE(0, 0)->NODE("reshape1", RESHAPE));
    CHAIN(NODE("const1", CONSTANT)->EDGE(0, 1)->NODE("reshape1", RESHAPE));
    CHAIN(NODE("reshape1", RESHAPE)->EDGE(0, 0)->NODE("reshape2", RESHAPE));
    CHAIN(NODE("const2", CONSTANT)->EDGE(0, 1)->NODE("reshape2", RESHAPE));
    CHAIN(NODE("reshape2", RESHAPE)->EDGE(0, 0)->NODE("reshape3", RESHAPE));
    CHAIN(NODE("const3", CONSTANT)->EDGE(0, 1)->NODE("reshape3", RESHAPE));
    CHAIN(NODE("reshape3", RESHAPE)->EDGE(0, 0)->NODE("cast", CAST));
    CHAIN(NODE("cast", CAST)->EDGE(0, 0)->NODE("net_output", NETOUTPUT));
  };
  auto compute_graph = ToComputeGraph(g1);
  EXPECT_NE(compute_graph->FindNode("reshape1"), nullptr);
  EXPECT_NE(compute_graph->FindNode("reshape2"), nullptr);
  EXPECT_NE(compute_graph->FindNode("reshape3"), nullptr);
  names_to_pass.push_back({"ReshapeRemovePass", new ReshapeRemovePass});
  GEPass pass(compute_graph);
  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);
  EXPECT_EQ(compute_graph->FindNode("reshape1"), nullptr);
  EXPECT_EQ(compute_graph->FindNode("reshape2"), nullptr);
  EXPECT_EQ(compute_graph->FindNode("reshape3"), nullptr);
  EXPECT_EQ(compute_graph->FindNode("const1"), nullptr);
}

TEST_F(UtestReshapeRemovePass, keep_static_reshape_on_dynamic_graph) {
  DEF_GRAPH(g1) {
    CHAIN(NODE("var1", VARIABLE)->EDGE(0, 0)->NODE("add1", ADD));
    CHAIN(NODE("var2", VARIABLE)->EDGE(0, 1)->NODE("add1", ADD));

    CHAIN(NODE("add1", ADD)->EDGE(0, 0)->NODE("reshape", RESHAPE));
    CHAIN(NODE("const1", CONSTANT)->EDGE(0, 1)->NODE("reshape", RESHAPE));
    CHAIN(NODE("reshape", RESHAPE)->EDGE(0, 0)->NODE("net_output", NETOUTPUT));
  };
  auto compute_graph = ToComputeGraph(g1);
  compute_graph->SetGraphUnknownFlag(true);
  const auto reshape_node = compute_graph->FindNode("reshape");
  ASSERT_NE(reshape_node, nullptr);
  names_to_pass.push_back({"ReshapeRemovePass", new ReshapeRemovePass});
  GEPass pass(compute_graph);
  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);
  EXPECT_NE(compute_graph->FindNode("reshape"), nullptr);
  
}

TEST_F(UtestReshapeRemovePass, keep_reshape_when_output_require_input_continuous) {
  DEF_GRAPH(g1) {
    CHAIN(NODE("var1", VARIABLE)->EDGE(0, 0)->NODE("reshape", RESHAPE));
    CHAIN(NODE("const1", CONSTANT)->EDGE(0, 1)->NODE("reshape", RESHAPE));
    CHAIN(NODE("reshape", RESHAPE)->EDGE(0, 0)->NODE("gathershape", GATHERSHAPES)->NODE("net_output", NETOUTPUT));
  };
  auto compute_graph = ToComputeGraph(g1);
  EXPECT_NE(compute_graph->FindNode("reshape"), nullptr);
  names_to_pass.push_back({"ReshapeRemovePass", new ReshapeRemovePass});
  GEPass pass(compute_graph);
  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);
  EXPECT_NE(compute_graph->FindNode("reshape"), nullptr);
}

TEST_F(UtestReshapeRemovePass, keep_reshape_output_of_subgraph) {
  DEF_GRAPH(sub) {
    CHAIN(NODE("var1", VARIABLE)->EDGE(0, 0)->NODE("reshape", RESHAPE));
    CHAIN(NODE("const1", CONSTANT)->EDGE(0, 1)->NODE("reshape", RESHAPE));
    CHAIN(NODE("reshape", RESHAPE)->EDGE(0, 0)->NODE("net_output", NETOUTPUT));
  };
  DEF_GRAPH(root) {
    CHAIN(NODE("data_0", DATA)->NODE("partiton", PARTITIONEDCALL, sub)->NODE("netoutput", NETOUTPUT));
  };
  auto compute_graph = ToComputeGraph(root);
  names_to_pass.push_back({"ReshapeRemovePass", new ReshapeRemovePass});
  GEPass pass(compute_graph);
  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);
  bool has_reshape = false;
  for (const auto &node : compute_graph->GetAllNodes()) {
    if (node->GetType() == RESHAPE) {
      has_reshape = true;
      break;
    }
  }
  EXPECT_TRUE(has_reshape == true);
}
}  // namespace ge
