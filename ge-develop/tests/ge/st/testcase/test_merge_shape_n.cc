/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "utils/graph_factory.h"
#include <gtest/gtest.h>
#include <map>
#include "ge/ge_api.h"
#include "utils/synchronizer.h"
#include "ge_running_env/tensor_utils.h"
#include "utils/tensor_adapter.h"
#include "graph/utils/graph_utils.h"
#include "mmpa/mmpa_api.h"
#include "init_ge.h"
#include "ge_graph_dsl/assert/graph_assert.h"

namespace ge {
class MergeShapeNSt : public testing::Test {
 protected:
  void SetUp() override {
    ReInitGe();
  }
};

TEST_F(MergeShapeNSt, merge_unknown_shapeN) {
  Graph graph = GraphFactory::BuildGraphForMergeShapeNPass();
  
  DUMP_GRAPH_WHEN("OptimizeStage1_1");
  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);
  std::vector<InputTensorInfo> inputs;
  session.BuildGraph(1, inputs);

  CHECK_GRAPH(OptimizeStage1_1) {
    auto ret = graph->TopologicalSorting();
    EXPECT_EQ(ret, SUCCESS);
    int n = 0;
    for (const auto &node : graph->GetDirectNode()) {
      if (node->GetType() != SHAPEN) {
        continue;
      }
      n++;
    }
    ASSERT_EQ(n, 1);
    ASSERT_EQ(graph->FindNode("shapeN1"), nullptr);
    ASSERT_EQ(graph->FindNode("shapeN2"), nullptr);
  };  
}
}  // namespace ge
