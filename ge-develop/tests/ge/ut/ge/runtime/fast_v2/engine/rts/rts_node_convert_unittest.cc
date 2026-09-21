/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "faker/node_faker.h"
#include "common/bg_test.h"
#include "common/topo_checker.h"
#include <gtest/gtest.h>
#include "macro_utils/dt_public_scope.h"
#include "register/node_converter_registry.h"
#include "exe_graph/lowering/lowering_global_data.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/utils/attr_utils.h"
#include "framework/common/types.h"
#include "faker/space_registry_faker.h"
#include "lowering/graph_converter.h"
#include "lowering/model_converter.h"
#include "faker/global_data_faker.h"

namespace gert {
using namespace bg;
using namespace ge;
LowerResult LoweringRtsNode(const ge::NodePtr &node, const LowerInput &lower_inpput);
LowerResult LoweringCmoNode(const ge::NodePtr &node, const LowerInput &lower_input);

ComputeGraphPtr BuildOverflowDetectionGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")->NODE("GetFloatStatusV2", ge::NPUGETFLOATSTATUSV2)->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data1", "Data")
              ->CTRL_EDGE()
              ->NODE("ClearFloatStatusV2", ge::NPUCLEARFLOATSTATUSV2)
              ->CTRL_EDGE()
              ->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data1", "Data")->NODE("GetFloatStatus", ge::NPUGETFLOATSTATUS)->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data1", "Data")
              ->CTRL_EDGE()
              ->NODE("ClearFloatStatus", ge::NPUCLEARFLOATSTATUS)
              ->CTRL_EDGE()
              ->NODE("NetOutput", "NetOutput"));
  };

  auto graph = ToComputeGraph(g1);
  auto data1 = graph->FindNode("data1");
  data1->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({8}));
  data1->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape({8}));
  data1->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_INT32);
  data1->GetOpDesc()->MutableOutputDesc(0)->SetOriginDataType(DT_INT32);
  data1->GetOpDesc()->MutableOutputDesc(0)->SetFormat(ge::FORMAT_ND);
  data1->GetOpDesc()->MutableOutputDesc(0)->SetOriginFormat(ge::FORMAT_ND);

  return graph;
};

ComputeGraphPtr BuildDebugOverflowDetectionGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")
              ->NODE("GetFloatDebugStatus1", ge::NPUGETFLOATDEBUGSTATUS)
              ->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data1", "Data")
              ->CTRL_EDGE()
              ->NODE("ClearFloatDebugStatus1", ge::NPUCLEARFLOATDEBUGSTATUS)
              ->CTRL_EDGE()
              ->NODE("NetOutput", "NetOutput"));
  };

  auto graph = ToComputeGraph(g1);
  auto data1 = graph->FindNode("data1");
  data1->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({8}));
  data1->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape({8}));
  data1->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_INT32);
  data1->GetOpDesc()->MutableOutputDesc(0)->SetOriginDataType(DT_INT32);
  data1->GetOpDesc()->MutableOutputDesc(0)->SetFormat(ge::FORMAT_ND);
  data1->GetOpDesc()->MutableOutputDesc(0)->SetOriginFormat(ge::FORMAT_ND);

  return graph;
};

ComputeGraphPtr BuildCmoGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")
              ->NODE("flash_attention", "FlashAttention")
              ->EDGE(0, 0)
              ->NODE("matmul", ge::MATMUL)
              ->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("weight", ge::CONSTANT)->EDGE(0, 1)->NODE("matmul", ge::MATMUL));
    CHAIN(NODE("weight", ge::CONSTANT)->EDGE(0, 0)->NODE("cmo", "Cmo"));
  };

  auto graph = ToComputeGraph(g1);
  auto data1 = graph->FindNode("data1");
  data1->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({8}));
  data1->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape({8}));
  data1->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_INT32);
  data1->GetOpDesc()->MutableOutputDesc(0)->SetOriginDataType(DT_INT32);
  data1->GetOpDesc()->MutableOutputDesc(0)->SetFormat(ge::FORMAT_ND);
  data1->GetOpDesc()->MutableOutputDesc(0)->SetOriginFormat(ge::FORMAT_ND);

  return graph;
};

class RtsNodeConvertUT : public bg::BgTest {};
TEST_F(RtsNodeConvertUT, NpuOverFlowDetectionOk) {
  auto graph = BuildOverflowDetectionGraph();
  EXPECT_NE(graph, nullptr);
  auto shape_holder = ValueHolder::CreateFeed(0);
  auto address_holder = DevMemValueHolder::CreateSingleDataOutput("Data", {}, 0);
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  LoweringGlobalData psg_lgd = GlobalDataFaker(root_model).Build();
    auto space_registry_array = OpImplSpaceRegistryV2Array();
  space_registry_array[static_cast<size_t>(gert::OppImplVersionTag::kOpp)] = SpaceRegistryFaker().Build();
  psg_lgd.SetSpaceRegistriesV2(space_registry_array);
  LowerInput lower_input{{shape_holder}, {address_holder}, &psg_lgd};

  auto get_float_status_v2 = graph->FindNode("GetFloatStatusV2");
  EXPECT_NE(get_float_status_v2, nullptr);
  auto ret = LoweringRtsNode(get_float_status_v2, lower_input);
  ASSERT_TRUE(ret.result.IsSuccess());
  ASSERT_EQ(ret.order_holders.size(), 1);
  ASSERT_EQ(ret.out_shapes.size(), 1);
  ASSERT_EQ(ret.out_addrs.size(), 1);

  auto clear_float_status_v2 = graph->FindNode("ClearFloatStatusV2");
  EXPECT_NE(clear_float_status_v2, nullptr);
  ret = LoweringRtsNode(clear_float_status_v2, lower_input);
  ASSERT_TRUE(ret.result.IsSuccess());
  ASSERT_EQ(ret.order_holders.size(), 1);
  ASSERT_EQ(ret.out_shapes.size(), 0);
  ASSERT_EQ(ret.out_addrs.size(), 0);

  auto get_float_status = graph->FindNode("GetFloatStatus");
  EXPECT_NE(get_float_status, nullptr);
  ret = LoweringRtsNode(get_float_status, lower_input);
  ASSERT_TRUE(ret.result.IsSuccess());
  ASSERT_EQ(ret.order_holders.size(), 1);
  ASSERT_EQ(ret.out_shapes.size(), 1);
  ASSERT_EQ(ret.out_addrs.size(), 1);

  auto clear_float_status = graph->FindNode("ClearFloatStatus");
  EXPECT_NE(clear_float_status, nullptr);
  ret = LoweringRtsNode(clear_float_status, lower_input);
  ASSERT_TRUE(ret.result.IsSuccess());
  ASSERT_EQ(ret.order_holders.size(), 1);
  ASSERT_EQ(ret.out_shapes.size(), 1);
  ASSERT_EQ(ret.out_addrs.size(), 1);
}

TEST_F(RtsNodeConvertUT, NpuDebugOverFlowDetectionOk) {
  auto graph = BuildDebugOverflowDetectionGraph();
  EXPECT_NE(graph, nullptr);
  auto shape_holder = ValueHolder::CreateFeed(0);
  auto address_holder = DevMemValueHolder::CreateSingleDataOutput("Data", {}, 0);
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  LoweringGlobalData psg_lgd = GlobalDataFaker(root_model).Build();
    auto space_registry_array = OpImplSpaceRegistryV2Array();
  space_registry_array[static_cast<size_t>(gert::OppImplVersionTag::kOpp)] = SpaceRegistryFaker().Build();
  psg_lgd.SetSpaceRegistriesV2(space_registry_array);
  LowerInput lower_input{{shape_holder}, {address_holder}, &psg_lgd};

  auto get_float_status_v2 = graph->FindNode("GetFloatDebugStatus1");
  EXPECT_NE(get_float_status_v2, nullptr);
  auto ret = LoweringRtsNode(get_float_status_v2, lower_input);
  ASSERT_TRUE(ret.result.IsSuccess());
  ASSERT_EQ(ret.order_holders.size(), 1);
  ASSERT_EQ(ret.out_shapes.size(), 1);
  ASSERT_EQ(ret.out_addrs.size(), 1);

  auto clear_float_status_v2 = graph->FindNode("ClearFloatDebugStatus1");
  EXPECT_NE(clear_float_status_v2, nullptr);
  ret = LoweringRtsNode(clear_float_status_v2, lower_input);
  ASSERT_TRUE(ret.result.IsSuccess());
  ASSERT_EQ(ret.order_holders.size(), 1);
  ASSERT_EQ(ret.out_shapes.size(), 1);
  ASSERT_EQ(ret.out_addrs.size(), 1);
}

TEST_F(RtsNodeConvertUT, LoweringCmoOk) {
  auto graph = BuildCmoGraph();
  EXPECT_NE(graph, nullptr);
  auto shape_holder = ValueHolder::CreateFeed(0);
  auto address_holder = DevMemValueHolder::CreateSingleDataOutput("Data", {}, 0);
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  LoweringGlobalData global_data = GlobalDataFaker(root_model).Build();
    auto space_registry_array = OpImplSpaceRegistryV2Array();
  space_registry_array[static_cast<size_t>(gert::OppImplVersionTag::kOpp)] = SpaceRegistryFaker().Build();
  global_data.SetSpaceRegistriesV2(space_registry_array);
  auto cmo_node = graph->FindNode("cmo");
  EXPECT_NE(cmo_node, nullptr);
  ge::AttrUtils::SetInt(cmo_node->GetOpDesc(), "type", 6);

  LowerInput lower_input{{shape_holder}, {address_holder}, &global_data};

  auto ret = LoweringCmoNode(cmo_node, lower_input);
  ASSERT_TRUE(ret.result.IsSuccess());
  ASSERT_EQ(ret.order_holders.size(), 1UL);
  ASSERT_EQ(ret.out_shapes.size(), 0UL);
  ASSERT_EQ(ret.out_addrs.size(), 0UL);
}

TEST_F(RtsNodeConvertUT, LoweringCmoOk_with_max_size) {
  auto graph = BuildCmoGraph();
  EXPECT_NE(graph, nullptr);
  auto shape_holder = ValueHolder::CreateFeed(0);
  auto address_holder = DevMemValueHolder::CreateSingleDataOutput("Data", {}, 0);
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  LoweringGlobalData global_data = GlobalDataFaker(root_model).Build();
    auto space_registry_array = OpImplSpaceRegistryV2Array();
  space_registry_array[static_cast<size_t>(gert::OppImplVersionTag::kOpp)] = SpaceRegistryFaker().Build();
  global_data.SetSpaceRegistriesV2(space_registry_array);
  auto cmo_node = graph->FindNode("cmo");
  EXPECT_NE(cmo_node, nullptr);
  ge::AttrUtils::SetInt(cmo_node->GetOpDesc(), "type", 6);
  ge::AttrUtils::SetInt(cmo_node->GetOpDesc(), "max_size", 1024);

  LowerInput lower_input{{shape_holder}, {address_holder}, &global_data};

  auto ret = LoweringCmoNode(cmo_node, lower_input);
  ASSERT_TRUE(ret.result.IsSuccess());
  ASSERT_EQ(ret.order_holders.size(), 1UL);
  ASSERT_EQ(ret.out_shapes.size(), 0UL);
  ASSERT_EQ(ret.out_addrs.size(), 0UL);
}
}  // namespace gert