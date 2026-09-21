/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/graph.h"
#include "graph/compute_graph.h"

USING_GE_NS

class GeGraphDslCheckTest : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};


TEST_F(GeGraphDslCheckTest, test_build_graph_from_optype_with_name) {
  DEF_GRAPH(g1) { CHAIN(NODE("data1", "Data")->NODE("add", "Add")); };

  auto geGraph = ToGeGraph(g1);
  auto computeGraph = ToComputeGraph(g1);

  ASSERT_EQ(geGraph.GetAllNodes().size(), 2);
  ASSERT_EQ(computeGraph->GetAllNodesSize(), 2);
}
