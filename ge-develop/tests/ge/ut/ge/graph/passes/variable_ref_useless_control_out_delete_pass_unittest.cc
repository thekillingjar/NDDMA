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
#include "graph/passes/variable_optimize/variable_ref_useless_control_out_delete_pass.h"
#include "graph_builder_utils.h"
namespace ge {
class UtestVariableRefUselessControlOutDeletePass : public testing::Test
{
 protected:
  void SetUp()
  {
  }

  void TearDown()
  {
  }
};

namespace {
/*
 *      c
 * var1 -> addn2
 *   \    /
 *    addn1
 *     |
 *    data1
 */
ComputeGraphPtr BuildGraph1() {
  ut::GraphBuilder builder("g1");
  auto data1 = builder.AddNode("data1", "Data", 1, 1);
  auto addn1 = builder.AddNode("addn1", "AddN", 1, 1);
  auto var1 = builder.AddNode("var1", "Variable", 1, 1);
  auto addn2 = builder.AddNode("addn2", "AddN", 1, 1);
  AttrUtils::SetStr(var1->GetOpDesc(), REF_VAR_SRC_VAR_NAME, "TestVar");
  builder.AddDataEdge(data1, 0, addn1, 0);
  builder.AddDataEdge(addn1, 0, var1, 0);
  builder.AddDataEdge(addn1, 0, addn2, 0);
  builder.AddControlEdge(var1, addn2);
  return builder.GetGraph();
}

/*
 *      c
 * var1 -> addn2
 *   \       |
 *    addn1  |
 *        \  |
 *        data1
 */
ComputeGraphPtr BuildGraph2() {
  ut::GraphBuilder builder("g2");
  auto data1 = builder.AddNode("data1", "Data", 1, 1);
  auto addn1 = builder.AddNode("addn1", "AddN", 1, 1);
  auto var1 = builder.AddNode("var1", "Variable", 1, 1);
  auto addn2 = builder.AddNode("addn2", "AddN", 1, 1);

  AttrUtils::SetStr(var1->GetOpDesc(), REF_VAR_SRC_VAR_NAME, "TestVar");

  builder.AddDataEdge(data1, 0, addn1, 0);
  builder.AddDataEdge(addn1, 0, var1, 0);
  builder.AddDataEdge(data1, 0, addn2, 0);
  builder.AddControlEdge(var1, addn2);

  return builder.GetGraph();
}

/*
 *   netoutput1
 *   /    \
 * var1  addn2
 *   \    /
 *    addn1
 *     |
 *    data1
 */
ComputeGraphPtr BuildGraph3() {
  ut::GraphBuilder builder("g3");
  auto data1 = builder.AddNode("data1", "Data", 1, 1);
  auto addn1 = builder.AddNode("addn1", "AddN", 1, 1);
  auto var1 = builder.AddNode("var1", "Variable", 1, 1);
  auto addn2 = builder.AddNode("addn2", "AddN", 1, 1);
  auto netoutput1 = builder.AddNode("netoutput1", "NetOutput", 2, 2);

  AttrUtils::SetStr(var1->GetOpDesc(), REF_VAR_SRC_VAR_NAME, "TestVar");

  builder.AddDataEdge(data1, 0, addn1, 0);
  builder.AddDataEdge(addn1, 0, var1, 0);
  builder.AddDataEdge(addn1, 0, addn2, 0);
  builder.AddDataEdge(var1, 0, netoutput1, 0);
  builder.AddDataEdge(addn2, 0, netoutput1, 1);

  return builder.GetGraph();
}
}

TEST_F(UtestVariableRefUselessControlOutDeletePass, DeleteControlOutSuccess) {
  auto graph = BuildGraph1();
  VariableRefUselessControlOutDeletePass pass;

  auto var1 = graph->FindNode("var1");
  EXPECT_EQ(var1->GetOutControlNodes().size(), 1);
  EXPECT_EQ(pass.Run(graph), SUCCESS);
  EXPECT_EQ(graph->GetAllNodes().size(), 4);
  EXPECT_EQ(var1->GetOutControlNodes().size(), 0);
}

TEST_F(UtestVariableRefUselessControlOutDeletePass, KeepControlOutSuccess) {
  auto graph = BuildGraph2();
  VariableRefUselessControlOutDeletePass pass;

  auto var1 = graph->FindNode("var1");
  EXPECT_EQ(var1->GetOutControlNodes().size(), 1);
  EXPECT_EQ(pass.Run(graph), SUCCESS);
  EXPECT_EQ(graph->GetAllNodes().size(), 4);
  EXPECT_EQ(var1->GetOutControlNodes().size(), 1);
}

TEST_F(UtestVariableRefUselessControlOutDeletePass, NothingChanged) {
  auto graph = BuildGraph3();
  VariableRefUselessControlOutDeletePass pass;

  auto var1 = graph->FindNode("var1");
  EXPECT_EQ(var1->GetOutNodes().size(), 1);
  EXPECT_EQ(pass.Run(graph), SUCCESS);
  EXPECT_EQ(graph->GetAllNodes().size(), 5);
  EXPECT_EQ(var1->GetOutNodes().size(), 1);
}
}
