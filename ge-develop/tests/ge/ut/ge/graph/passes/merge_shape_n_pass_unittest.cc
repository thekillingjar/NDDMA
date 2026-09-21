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
#include "graph/utils/graph_utils.h"
#include "macro_utils/dt_public_scope.h"
#include "graph/passes/shape_optimize/merge_unknown_shape_n_pass.h"
#include "macro_utils/dt_public_unscope.h"
#include "graph/compute_graph.h"
#include "graph_builder_utils.h"

namespace ge {
  class UtestDataPass : public testing::Test {
   protected:
    void SetUp() {}
    void TearDown() {}
};

ComputeGraphPtr MakeMergeGraph() {
  ge::ut::GraphBuilder builder("graph");
  auto data1 = builder.AddNode("data1", "Data", 0, 1, FORMAT_NCHW, DT_INT32, {});
  auto data2 = builder.AddNode("data2", "Data", 0, 1, FORMAT_NCHW, DT_INT32, {});
  auto relu1 = builder.AddNode("relu1", RELU, 2, 2, FORMAT_NCHW, DT_INT32, {});
  auto shapeN1 = builder.AddNode("shapeN1", SHAPEN, 1, 1, FORMAT_NCHW, DT_INT32, {});
  auto shapeN2 = builder.AddNode("shapeN2", SHAPEN, 1, 1, FORMAT_NCHW, DT_INT32, {});
  auto relu2 = builder.AddNode("relu2", RELU, 2, 1, FORMAT_NCHW, DT_INT32, {});
  auto netoutput = builder.AddNode("relu2", "NetOutput", 1, 0, FORMAT_NCHW, DT_INT32, {});

  auto output_desc_data1 = data1->GetOpDesc()->GetOutputDesc(0);
  auto output_desc_data2 = data2->GetOpDesc()->GetOutputDesc(0);
  output_desc_data1.SetShape(GeShape({-1, 1, 2}));
  output_desc_data2.SetShape(GeShape({1, 1, 2}));
  data1->GetOpDesc()->UpdateOutputDesc(0, output_desc_data1);
  data1->GetOpDesc()->UpdateOutputDesc(0, output_desc_data2);
  
  auto output_desc_relu1_0 = relu1->GetOpDesc()->GetOutputDesc(0);
  auto output_desc_relu1_1 = relu1->GetOpDesc()->GetOutputDesc(1);
  auto input_desc_relu1_0 = relu1->GetOpDesc()->GetInputDesc(0);
  auto input_desc_relu1_1 = relu1->GetOpDesc()->GetInputDesc(1);
  input_desc_relu1_0.SetShape(GeShape({-1, 1, 2}));
  input_desc_relu1_1.SetShape(GeShape({-1, 1, 2}));
  output_desc_relu1_0.SetShape(GeShape({-1, 1, 2}));
  output_desc_relu1_1.SetShape(GeShape({-1, 1, 2}));
  relu1->GetOpDesc()->UpdateInputDesc(0, input_desc_relu1_0);
  relu1->GetOpDesc()->UpdateInputDesc(1, input_desc_relu1_1);
  relu1->GetOpDesc()->UpdateOutputDesc(0, output_desc_relu1_0);
  relu1->GetOpDesc()->UpdateOutputDesc(1, output_desc_relu1_1);

  auto input_desc_shapeN1 = shapeN1->GetOpDesc()->GetInputDesc(0);
  auto output_desc_shapeN1 = shapeN1->GetOpDesc()->GetOutputDesc(0);
  ge::AttrUtils::SetStr(shapeN1->GetOpDesc(), "_split_shapen_origin_name", "shapeN_unknown");
  input_desc_shapeN1.SetShape(GeShape({-1, 1 ,2}));
  output_desc_shapeN1.SetShape(GeShape({-1, 1, 2}));
  shapeN1->GetOpDesc()->UpdateInputDesc(0, input_desc_shapeN1);
  shapeN1->GetOpDesc()->UpdateOutputDesc(0, output_desc_shapeN1);

  auto input_desc_shapeN2 = shapeN2->GetOpDesc()->GetInputDesc(0);
  auto output_desc_shapeN2 = shapeN2->GetOpDesc()->GetOutputDesc(0);
  ge::AttrUtils::SetStr(shapeN2->GetOpDesc(), "_split_shapen_origin_name", "shapeN_unknown");
  input_desc_shapeN2.SetShape(GeShape({-1, 1 ,2}));
  output_desc_shapeN2.SetShape(GeShape({-1, 1, 2}));
  shapeN2->GetOpDesc()->UpdateInputDesc(0, input_desc_shapeN2);
  shapeN2->GetOpDesc()->UpdateOutputDesc(0, output_desc_shapeN2);

  auto output_desc_relu2_0 = relu2->GetOpDesc()->GetOutputDesc(0);
  auto input_desc_relu2_0 = relu2->GetOpDesc()->GetInputDesc(0);
  auto input_desc_relu2_1 = relu2->GetOpDesc()->GetInputDesc(1);
  input_desc_relu2_0.SetShape(GeShape({-1, 1, 2}));
  input_desc_relu2_1.SetShape(GeShape({-1, 1, 2}));
  output_desc_relu2_0.SetShape(GeShape({-1, 1, 2}));
  relu2->GetOpDesc()->UpdateInputDesc(0, input_desc_relu2_0);
  relu2->GetOpDesc()->UpdateInputDesc(1, input_desc_relu2_1);
  relu2->GetOpDesc()->UpdateOutputDesc(0, output_desc_relu2_0);

  auto input_desc_netoutput = netoutput->GetOpDesc()->GetInputDesc(0);
  input_desc_netoutput.SetShape(GeShape({-1, 1 ,2}));
  netoutput->GetOpDesc()->UpdateInputDesc(0, input_desc_netoutput);

  builder.AddDataEdge(data1, 0, relu1, 0);
  builder.AddDataEdge(data2, 0, relu1, 1);
  builder.AddDataEdge(relu1, 0, shapeN1, 0);
  builder.AddDataEdge(relu1, 1, shapeN2, 0);
  builder.AddDataEdge(shapeN1, 0, relu2, 0);
  builder.AddDataEdge(shapeN2, 0, relu2, 1);
  builder.AddDataEdge(relu2, 1, netoutput, 0);
  builder.AddControlEdge(relu1, shapeN1);
  builder.AddControlEdge(relu1, shapeN2);
  builder.AddControlEdge(shapeN1, relu2);
  builder.AddControlEdge(shapeN2, relu2);
  return builder.GetGraph();
}

TEST_F(UtestDataPass, MergeShapeNPassRun) {
  auto graph = MakeMergeGraph();
  MergeUnknownShapeNPass pass;
  size_t input_index = 0;
  size_t output_index = 0;
  EXPECT_NE(pass.ReplaceMergeNodeAnchors(nullptr, nullptr, input_index, output_index), SUCCESS);
  EXPECT_EQ(pass.Run(graph), SUCCESS);
  int n = 0;
  for (const auto &node : graph->GetDirectNode()) {
    if (node->GetType() != SHAPEN) {
      continue;
    }
    n++;
  }
  ASSERT_EQ(n, 1);
}
}  // namespace ge
