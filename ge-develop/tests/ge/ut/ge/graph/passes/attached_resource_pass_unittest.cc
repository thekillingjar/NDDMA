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
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/passes/feature/attached_resource_pass.h"

using namespace testing;
using namespace ge;
namespace ge {
class UtestAttachedResourcePass : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

namespace {
ComputeGraphPtr BuildNormalGraph() {
  const auto sub1_data_0 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0);
  const auto sub1_data_1 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 1);
  DEF_GRAPH(sub_1) {
    auto add_1 = OP_CFG(ADD).Attr(ATTR_NAME_STREAM_LABEL, "label1");
    CHAIN(NODE("const_0_ascend_mbatch_batch_0", CONSTANT)->NODE("add_ascend_mbatch_batch_0", add_1));
    CHAIN(NODE("data_1_ascend_mbatch_batch_0", sub1_data_1)->NODE("add_ascend_mbatch_batch_0", add_1));
    CHAIN(NODE("add_ascend_mbatch_batch_0", add_1)
              ->NODE("switch", SWITCH)
              ->NODE("netoutput_ascend_mbatch_batch_0", NETOUTPUT));
  };

  const auto sub2_data_0 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0);
  const auto sub2_data_1 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 1);
  DEF_GRAPH(sub_2) {
    auto add_2 = OP_CFG(ADD).Attr(ATTR_NAME_STREAM_LABEL, "label2");
    CHAIN(NODE("const_0_ascend_mbatch_batch_1", CONSTANT)->NODE("add_ascend_mbatch_batch_1", add_2));
    CHAIN(NODE("data_1_ascend_mbatch_batch_1", sub2_data_1)->NODE("add_ascend_mbatch_batch_1", add_2));
    CHAIN(NODE("add_ascend_mbatch_batch_1", add_2)->NODE("netoutput_ascend_mbatch_batch_1", NETOUTPUT));
  };

  DEF_GRAPH(g1) {
    CHAIN(NODE("data_0", DATA)->NODE("case", CASE, sub_1, sub_2)->NODE("netoutput", NETOUTPUT));
    CHAIN(NODE("data_1", DATA)->NODE("relu", RELU)->NODE("case"));
  };

  sub_1.Layout();
  sub_2.Layout();
  return ToComputeGraph(g1);
}
} // namespace

TEST_F(UtestAttachedResourcePass, test_SetAttachedResource_succ) {
  const auto graph = BuildNormalGraph();
  AttachedResourcePass pass;
  EXPECT_EQ(pass.Run(graph), SUCCESS);
  for (const auto &node : graph->GetDirectNode()) {
    EXPECT_TRUE(!node->GetOpDesc()->HasAttr("_disable_attached_resource"));
  }
  auto sub1 = graph->GetSubgraph("sub_1");
  ASSERT_TRUE(sub1 != nullptr);
  EXPECT_EQ(pass.Run(sub1), SUCCESS);
  for (const auto &node : sub1->GetDirectNode()) {
    EXPECT_TRUE(node->GetOpDesc()->HasAttr("_disable_attached_resource"));
  }
  auto sub2 = graph->GetSubgraph("sub_2");
  ASSERT_TRUE(sub2 != nullptr);
  EXPECT_EQ(pass.Run(sub2), SUCCESS);
  for (const auto &node : sub2->GetDirectNode()) {
    EXPECT_TRUE(!node->GetOpDesc()->HasAttr("_disable_attached_resource"));
  }
}
} // namespace ge
