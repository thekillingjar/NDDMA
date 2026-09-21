/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/node_converter_registry.h"
#include <gtest/gtest.h>
#include "common/bg_test.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "exe_graph/lowering/lowering_definitions.h"
#include "lowering/placement/placed_lowering_result.h"
#include "common/topo_checker.h"
#include "faker/global_data_faker.h"
namespace gert {
namespace {
ge::ComputeGraphPtr BuildGraph1() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data0", "Data")->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data1", "Data")->NODE("NetOutput", "NetOutput"));
  };
  return ge::ToComputeGraph(g1);
}
}  // namespace

LowerResult LoweringNetOutput(const ge::NodePtr &node, const LowerInput &lower_input);
LowerResult LoweringReshape(const ge::NodePtr &node, const LowerInput &lower_input);
class NetOutputConverterUT : public bg::BgTestAutoCreateFrame {};
TEST_F(NetOutputConverterUT, LoweringNetOutput_NodesAndConnectionsCorrect) {
  auto data0 = bg::ValueHolder::CreateFeed(0);
  auto data1 = bg::ValueHolder::CreateFeed(2);
  auto shape_and_addr_0 = bg::DevMemValueHolder::CreateDataOutput("SplitTensor", {data0}, 2, 0);
  auto shape_and_addr_1 = bg::DevMemValueHolder::CreateDataOutput("SplitTensor", {data1}, 2, 0);

  auto compute_graph = BuildGraph1();
  ASSERT_NE(compute_graph, nullptr);
  auto netoutput_node = compute_graph->FindNode("NetOutput");
  ASSERT_NE(netoutput_node, nullptr);
  auto data0_node = compute_graph->FindNode("data0");
  auto data1_node = compute_graph->FindNode("data1");
  ASSERT_NE(data0_node, nullptr);
  ASSERT_NE(data1_node, nullptr);
  data0_node->GetOpDesc()->SetExtAttr(
      kLoweringResult,
      PlacedLoweringResult(data0_node,
                           LowerResult{HyperStatus::Success(), {}, {shape_and_addr_0[0]}, {shape_and_addr_0[1]}}));
  data1_node->GetOpDesc()->SetExtAttr(
      kLoweringResult,
      PlacedLoweringResult(data1_node,
                           LowerResult{HyperStatus::Success(), {}, {shape_and_addr_1[0]}, {shape_and_addr_1[1]}}));

  auto root_model = GeModelBuilder(compute_graph).BuildGeRootModel();
  LoweringGlobalData gd = GlobalDataFaker(root_model).Build();
  auto result = LoweringNetOutput(
      netoutput_node, {{shape_and_addr_0[0], shape_and_addr_1[0]}, {shape_and_addr_0[1], shape_and_addr_1[1]}, &gd});
  ASSERT_TRUE(result.result.IsSuccess());

  ASSERT_EQ(result.out_shapes.size(), 2U);
  EXPECT_EQ(result.out_shapes[0]->GetFastNode()->GetType(), "OutputData");
  EXPECT_EQ(result.out_shapes[1]->GetFastNode()->GetType(), "OutputData");

  // todo 待恢复
  // ASSERT_EQ(result.order_holders.size(), 2U);
  // EXPECT_EQ(result.order_holders[0]->GetFastNode()->GetType(), "EnsureTensorAtOutMemory");
  // EXPECT_EQ(result.order_holders[1]->GetFastNode()->GetType(), "EnsureTensorAtOutMemory");

  EXPECT_EQ(FastNodeTopoChecker(result.out_shapes[0]).StrictConnectTo(0, {{"EnsureTensorAtOutMemory", 4}}), "success");
  EXPECT_EQ(FastNodeTopoChecker(result.out_shapes[0]).StrictConnectTo(1, {{"EnsureTensorAtOutMemory", 4}}), "success");

  auto out_nodes = result.out_shapes[0]->GetFastNode()->GetOutDataNodesByIndex(0);
  ASSERT_EQ(out_nodes.size(), 1);
  EXPECT_EQ(FastNodeTopoChecker(out_nodes.at(0))
                .StrictConnectFrom(
                    {{shape_and_addr_0[0], shape_and_addr_0[1], {"Const"}, gd.GetStreamById(0), result.out_shapes[0]}}),
            "success");

  out_nodes = result.out_shapes[0]->GetFastNode()->GetOutDataNodesByIndex(1);
  ASSERT_EQ(out_nodes.size(), 1);
  EXPECT_EQ(FastNodeTopoChecker(out_nodes.at(0))
                .StrictConnectFrom(
                    {{shape_and_addr_1[0], shape_and_addr_1[1], {"Const"}, gd.GetStreamById(0), result.out_shapes[1]}}),
            "success");
}
}  // namespace gert