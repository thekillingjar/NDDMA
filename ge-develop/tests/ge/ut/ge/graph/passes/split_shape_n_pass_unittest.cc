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
#include "graph/passes/shape_optimize/split_shape_n_pass.h"
#include "graph/compute_graph.h"
#include "graph_builder_utils.h"

namespace ge {
  class UtestSplitPass : public testing::Test {
   protected:
    void SetUp() {}
    void TearDown() {}
};

ComputeGraphPtr BuildForSplitGraph1() {
  ut::GraphBuilder builder("g1");
  auto data1 = builder.AddNode("data1", "Data", 1, 1, FORMAT_NCHW, DT_FLOAT, {-1, 3, 224, 224});
  auto addn1 = builder.AddNode("shapeN", SHAPEN, 1, 1);
  auto netoutput1 = builder.AddNode("netoutput1", "NetOutput", 1, 0);

  EXPECT_TRUE(AttrUtils::SetInt(addn1->GetOpDesc(), "TEST_OP_ATTR", 512));
  auto input_desc = addn1->GetOpDesc()->GetInputDesc(0);
  EXPECT_TRUE(AttrUtils::SetInt(input_desc, "TEST_ATTR", 1024));
  addn1->GetOpDesc()->UpdateInputDesc(0, input_desc);

  builder.AddDataEdge(data1, 0, addn1, 0);
  builder.AddDataEdge(addn1, 0, netoutput1, 0);
  auto graph = builder.GetGraph();

  graph->SetInputSize(1);
  graph->SetInputsOrder({"data1"});
  graph->AddInputNode(data1);

  return graph;
}

ComputeGraphPtr BuildForSplitGraph2() {
  ut::GraphBuilder builder("g1");
  auto data1 = builder.AddNode("data1", "Data", 1, 1, FORMAT_NCHW, DT_FLOAT, {-1, 3, 224, 224});
  auto addn1 = builder.AddNode("shapeN", SHAPEN, 1, 1);
  auto netoutput1 = builder.AddNode("netoutput1", "NetOutput", 1, 0);

  EXPECT_TRUE(AttrUtils::SetInt(addn1->GetOpDesc(), "TEST_OP_ATTR", 512));
  auto input_desc = addn1->GetOpDesc()->GetInputDesc(0);
  auto output_desc = addn1->GetOpDesc()->GetOutputDesc(0);
  input_desc.SetShape(GeShape({-1, 1 ,2}));
  output_desc.SetShape(GeShape({-1, 1, 2}));
  EXPECT_TRUE(AttrUtils::SetInt(input_desc, "TEST_ATTR", 1024));
  addn1->GetOpDesc()->UpdateInputDesc(0, input_desc);
  addn1->GetOpDesc()->UpdateOutputDesc(0, output_desc);

  builder.AddDataEdge(data1, 0, addn1, 0);
  builder.AddDataEdge(addn1, 0, netoutput1, 0);
  auto graph = builder.GetGraph();

  graph->SetInputSize(1);
  graph->SetInputsOrder({"data1"});
  graph->AddInputNode(data1);

  return graph;
}

ComputeGraphPtr BuildForSplitGraph3() {
  ut::GraphBuilder builder("g1");
  auto data1 = builder.AddNode("data1", "Data", 1, 2, FORMAT_NCHW, DT_INT32, {});
  auto addn1 = builder.AddNode("shapeN", SHAPEN, 2, 2);
  auto netoutput1 = builder.AddNode("netoutput1", "NetOutput", 1, 0);
  auto netoutput2 = builder.AddNode("netoutput2", "NetOutput", 1, 0);
  
  auto input_desc_netoutput1 = netoutput1->GetOpDesc()->GetInputDesc(0);
  auto input_desc_netoutput2 = netoutput2->GetOpDesc()->GetInputDesc(0);
  input_desc_netoutput1.SetShape(GeShape({-1, 1 ,2}));
  input_desc_netoutput2.SetShape(GeShape({1, 1 ,2}));
  netoutput1->GetOpDesc()->UpdateInputDesc(0, input_desc_netoutput1);
  netoutput1->GetOpDesc()->UpdateInputDesc(0, input_desc_netoutput2);

  auto output_desc_data1_0 = data1->GetOpDesc()->GetOutputDesc(0);
  auto output_desc_data1_1 = data1->GetOpDesc()->GetOutputDesc(1);
  output_desc_data1_0.SetShape(GeShape({-1, 1, 2}));
  output_desc_data1_1.SetShape(GeShape({1, 1, 2}));
  data1->GetOpDesc()->UpdateOutputDesc(0, output_desc_data1_0);
  data1->GetOpDesc()->UpdateOutputDesc(1, output_desc_data1_1);

  auto input_desc0 = addn1->GetOpDesc()->GetInputDesc(0);
  auto output_desc0 = addn1->GetOpDesc()->GetOutputDesc(0);
  input_desc0.SetShape(GeShape({-1, 1 ,2}));
  output_desc0.SetShape(GeShape({-1, 1, 2}));
  addn1->GetOpDesc()->UpdateInputDesc(0, input_desc0);
  addn1->GetOpDesc()->UpdateOutputDesc(0, output_desc0);

  EXPECT_TRUE(AttrUtils::SetInt(addn1->GetOpDesc(), "TEST_OP_ATTR", 512));
  auto input_desc1 = addn1->GetOpDesc()->GetInputDesc(1);
  auto output_desc1 = addn1->GetOpDesc()->GetOutputDesc(1);
  input_desc1.SetShape(GeShape({1, 1 ,2}));
  output_desc1.SetShape(GeShape({1, 1, 2}));
  EXPECT_TRUE(AttrUtils::SetInt(input_desc0, "TEST_ATTR", 1024));
  addn1->GetOpDesc()->UpdateInputDesc(1, input_desc1);
  addn1->GetOpDesc()->UpdateOutputDesc(1, output_desc1);

  builder.AddDataEdge(data1, 0, addn1, 0);
  builder.AddDataEdge(data1, 1, addn1, 1);
  builder.AddDataEdge(addn1, 0, netoutput1, 0);
  builder.AddDataEdge(addn1, 1, netoutput2, 0);
  auto graph = builder.GetGraph();

  graph->SetInputSize(1);
  graph->SetInputsOrder({"data1"});
  graph->AddInputNode(data1);

  return graph;
}

TEST_F(UtestSplitPass, SplitShapeNPass) {
  auto graph1 = BuildForSplitGraph1();
  SplitShapeNPass pass;
  auto node1 = graph1->FindNode("shapeN");
  EXPECT_EQ(pass.Run(node1), SUCCESS);
  auto graph2 = BuildForSplitGraph2(); 
  auto node2 = graph1->FindNode("shapeN");
  EXPECT_EQ(pass.Run(node2), SUCCESS);

  auto graph3 = BuildForSplitGraph3(); 
  auto node3 = graph3->FindNode("shapeN");
  EXPECT_EQ(pass.Run(node3), SUCCESS);
  auto unknown_node = graph3->FindNode("shapeN_unknown");
  auto known_node = graph3->FindNode("shapeN_known");
  auto old_node = graph3->FindNode("shapeN");
  ASSERT_NE(unknown_node, nullptr);
  ASSERT_NE(known_node, nullptr);
  ASSERT_EQ(old_node, nullptr);
}
}  // namespace ge
