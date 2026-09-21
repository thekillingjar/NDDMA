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
#include <map>
#include "macro_utils/dt_public_scope.h"
#include "graph/passes/memory_conflict/inplace_support_check_pass.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "api/gelib/gelib.h"
#include "graph/node.h"
#include "macro_utils/dt_public_unscope.h"

using namespace ge;

class UtestGraphPassesInplaceSupportCheckPass : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

static ComputeGraphPtr BuildGraphInplaceSupportCheckPass() {
  DEF_GRAPH(inplaceSupportCheckGraph) {
    CHAIN(NODE("cast0", CAST)->EDGE(0U, 0U)->NODE("cast1", CAST));
    CHAIN(NODE("cast1", CAST)->EDGE(0U, 0U)->NODE("translate1", TRANSLATE));
    CHAIN(NODE("translate1", TRANSLATE)->EDGE(0U, 0U)->NODE("cast2", CAST));
    CHAIN(NODE("translate1", TRANSLATE)->EDGE(1U, 0U)->NODE("cast3", CAST));
    CHAIN(NODE("cast4", CAST)->EDGE(0U, 0U)->NODE("cast2", CAST));
    CHAIN(NODE("cast2", CAST)->EDGE(0U, 0U)->NODE("cast5", CAST));
    CHAIN(NODE("cast4", CAST)->EDGE(1U, 0U)->NODE("cast5", CAST));
  };

  const auto graph = ToGeGraph(inplaceSupportCheckGraph);
  const auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  compute_graph->TopologicalSorting();

  return compute_graph;
}

TEST_F(UtestGraphPassesInplaceSupportCheckPass, cast_optimize) {
  auto graph = BuildGraphInplaceSupportCheckPass();
  ge::InplaceSupportCheckPass pass_;
  auto node_cast1 = graph->FindNode("cast1");

  Status status = pass_.Run(node_cast1);
  EXPECT_EQ(SUCCESS, status);

  // out_anchor_num > 1
  auto node_translate1 = graph->FindNode("translate1");
  status = pass_.Run(node_translate1);
  EXPECT_EQ(SUCCESS, status);

  auto node_cast2 = graph->FindNode("cast2");
  status = pass_.Run(node_cast2);
  EXPECT_EQ(SUCCESS, status);

  node_cast2->GetOpDesc()->UpdateInputDesc(0U, GeTensorDesc(GeShape({0,1}), FORMAT_NCHW, DT_INT8));
  status = pass_.Run(node_cast2);
  EXPECT_EQ(SUCCESS, status);

  node_cast2->GetOpDesc()->UpdateOutputDesc(0U, GeTensorDesc(GeShape(), FORMAT_NCHW, DT_INT8));
  status = pass_.Run(node_cast2);
  EXPECT_EQ(SUCCESS, status);
}

