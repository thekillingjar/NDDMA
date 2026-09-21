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
#include "graph/passes/feature/hccl_tailing_optimization_pass.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "api/gelib/gelib.h"
#include "macro_utils/dt_public_unscope.h"

using namespace ge;

class UtestGraphPassesHcclTailingOptimizationPass : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

static ComputeGraphPtr BuildGraphHcclTailingPass() {
  DEF_GRAPH(castTranslateGraph) {
    CHAIN(NODE("allReduce", HCOMALLREDUCE)->EDGE(0U, 0U)->NODE("cast0", CAST));
    CHAIN(NODE("cast0", CAST)->EDGE(0U, 0U)->NODE("translate1", TRANSLATE));
    CHAIN(NODE("cast0", CAST)->EDGE(0U, 0U)->NODE("translate2", TRANSLATE));
    CHAIN(NODE("translate1", TRANSLATE)->EDGE(0U, 0U)->NODE("cast2", CAST));
  };

  const auto graph = ToGeGraph(castTranslateGraph);
  const auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  compute_graph->TopologicalSorting();
  return compute_graph;
}

TEST_F(UtestGraphPassesHcclTailingOptimizationPass, cast_optimize) {
  auto graph = BuildGraphHcclTailingPass();
  auto cast0 = graph->FindNode("cast0");
  auto translate1 = graph->FindNode("translate1");
  GraphUtils::AddEdge(cast0->GetOutControlAnchor(), translate1->GetInControlAnchor());
  ge::HcclTailingOptimizationPass pass_;
  Status status = pass_.Run(graph);
  EXPECT_EQ(SUCCESS, status);
}
