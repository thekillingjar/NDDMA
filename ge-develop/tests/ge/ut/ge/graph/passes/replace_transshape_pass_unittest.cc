/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// To test the ReplaceTransShapePass

#include <gtest/gtest.h>
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/passes/shape_optimize/replace_transshape_pass.h"


using namespace testing;
using namespace ge;
namespace ge {
class UtestReplaceTransShapePass : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

namespace {
/**
 * 
 *          netoutput1
 *           /  c   \
 *  trans_shape - identity
 *         |   \c  |
 *       add1  add2
 *         / \/ \
 *        /  /\  \
 *       var1  var2
 */
static ComputeGraphPtr BuildGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("var1", VARIABLE)->EDGE(0, 0)->NODE("add1", ADD));
    CHAIN(NODE("var1", VARIABLE)->EDGE(0, 0)->NODE("add2", ADD));
    CHAIN(NODE("var2", VARIABLE)->EDGE(0, 1)->NODE("add1", ADD));
    CHAIN(NODE("var2", VARIABLE)->EDGE(0, 1)->NODE("add2", ADD));

    CHAIN(NODE("add1", ADD)->EDGE(0, 0)->NODE("trans_shape", TRANSSHAPE));
    CHAIN(NODE("trans_shape", TRANSSHAPE)->EDGE(0, 0)->NODE("net_output", NETOUTPUT));
    CHAIN(NODE("add2", ADD)->EDGE(0, 0)->NODE("identity", IDENTITY));
    CHAIN(NODE("identity", IDENTITY)->EDGE(0, 1)->NODE("net_output", NETOUTPUT));

    CTRL_CHAIN(NODE("add2", ADD)->NODE("trans_shape", TRANSSHAPE));
    CTRL_CHAIN(NODE("trans_shape", TRANSSHAPE)->NODE("identity", IDENTITY));
  };
  return ToComputeGraph(g1);
}
} // namespace

TEST_F(UtestReplaceTransShapePass, test_normal_succ) {
  const auto graph = BuildGraph();
  ReplaceTransShapePass pass;
  auto ret = pass.Run(graph);
  EXPECT_EQ(ret, SUCCESS);
  const auto mem_copy = graph->FindFirstNodeMatchType(MEMCPYASYNC);
  ASSERT_NE(mem_copy, nullptr);
  EXPECT_EQ(mem_copy->GetOutDataNodesSize(), 1);
  EXPECT_EQ(mem_copy->GetInDataNodes().size(), 1);
  EXPECT_EQ(mem_copy->GetOutControlNodes().size(), 1);
  EXPECT_EQ(mem_copy->GetInControlNodes().size(), 1);
  EXPECT_EQ(mem_copy->GetInDataNodes().at(0)->GetName(), "add1");
  const auto trans_shape = graph->FindFirstNodeMatchType(TRANSSHAPE);
  ASSERT_NE(trans_shape, nullptr);
  EXPECT_EQ(trans_shape->GetInAllNodes().size(), 0);
  EXPECT_EQ(trans_shape->GetOutAllNodes().size(), 0);
}
} // namespace ge
