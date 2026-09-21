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
#include "graph/passes/feature/hccl_sequence_adjust_pass.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/utils/graph_utils.h"
#include "api/gelib/gelib.h"
#include "macro_utils/dt_public_unscope.h"

namespace ge {
class UtestHcclSequenceAdjustPassPass : public testing::Test {
protected:
    void SetUp() {}

    void TearDown() {}
};

static ComputeGraphPtr BuildGraph() {
  DEF_GRAPH(sub_1) {
    CHAIN(NODE("data1", DATA)->NODE("output", NETOUTPUT));
  };
  DEF_GRAPH(sub_2) {
    CHAIN(NODE("data1", DATA)->NODE("output", NETOUTPUT));
  };
  DEF_GRAPH(root_graph) {
    auto if1 = OP_CFG(IF).InCnt(1).OutCnt(1).Attr(ATTR_NAME_HCCL_FUSED_GROUP, "hccl");
    CTRL_CHAIN(NODE("allreduce1", HCOMALLREDUCE)->NODE("allreduce2", HCOMALLREDUCE)->NODE("allreduce3", HCOMALLREDUCE));
    CTRL_CHAIN(NODE("allreducen", HCOMALLREDUCE)->NODE("If", if1, sub_1, sub_2));
  };

  return ToComputeGraph(root_graph);
}

static ComputeGraphPtr BuildClipGraph() {
  DEF_GRAPH(sub_1) {
    CHAIN(NODE("data1", DATA)->NODE("output", NETOUTPUT));
  };
  DEF_GRAPH(sub_2) {
    CHAIN(NODE("data1", DATA)->NODE("output", NETOUTPUT));
  };
  DEF_GRAPH(root_graph) {
    auto if1 = OP_CFG(IF).InCnt(1).OutCnt(1).Attr(ATTR_NAME_HCCL_FUSED_GROUP, "hccl");
    CTRL_CHAIN(NODE("allreduce1", HCOMALLREDUCE)->NODE("allreduce2", HCOMALLREDUCE)->NODE("allreduce3", HCOMALLREDUCE));
    CTRL_CHAIN(NODE("allreducen", HCOMALLREDUCE)->NODE("If", if1, sub_1, sub_2));
    CHAIN(NODE("allreduce3")->NODE("clip", "Clip")->NODE("allreducen"));
  };

  return ToComputeGraph(root_graph);
}

TEST_F(UtestHcclSequenceAdjustPassPass, hccl_sequence_adjust_succ) {
  ComputeGraphPtr graph = BuildGraph();
  HcclSequenceAdjustPass pass;
  auto ret = pass.Run(graph);
  EXPECT_EQ(ret, SUCCESS);

  auto allreduce3 = graph->FindNode("allreduce3");
  EXPECT_NE(allreduce3, nullptr);
  const auto &in_ctrl_nodes = allreduce3->GetInControlNodes();
  EXPECT_EQ(in_ctrl_nodes.size(), 2);
}

TEST_F(UtestHcclSequenceAdjustPassPass, hccl_sequence_adjust_clips_succ) {
  ComputeGraphPtr graph = BuildClipGraph();
  HcclSequenceAdjustPass pass;
  auto ret = pass.Run(graph);
  EXPECT_EQ(ret, SUCCESS);

  auto allreduce3 = graph->FindNode("allreduce3");
  EXPECT_NE(allreduce3, nullptr);
  const auto &in_ctrl_nodes = allreduce3->GetInControlNodes();
  EXPECT_EQ(in_ctrl_nodes.size(), 1);
}
}
