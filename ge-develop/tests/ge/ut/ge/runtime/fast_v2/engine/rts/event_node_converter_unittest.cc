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
#include <memory>
#include "graph/node.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "kernel/tensor_attr.h"
#include "faker/global_data_faker.h"
#include "common/bg_test.h"
#include "common/topo_checker.h"
#include "graph/utils/graph_utils.h"
#include "register/node_converter_registry.h"
namespace gert {
using namespace ge;
using namespace bg;
LowerResult LoweringEventNode(const ge::NodePtr &node, const LowerInput &lower_input);
class EventNodeConverterUt : public bg::BgTest {};
namespace {
/*
 *
 * netoutput
 *   |   |
 * SEND RECV
 *   |  /
 * data1
 */

ComputeGraphPtr BuildSendGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")->CTRL_EDGE()->NODE("send", "Send")->CTRL_EDGE()->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data1", "Data")->CTRL_EDGE()->NODE("recv", "Recv")->CTRL_EDGE()->NODE("NetOutput", "NetOutput"));
  };
  auto graph = ToComputeGraph(g1);
  auto data1 = graph->FindNode("data1");
  data1->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({8,3,224,224}));
  data1->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape({8,3,224,224}));
  data1->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_FLOAT);
  data1->GetOpDesc()->MutableOutputDesc(0)->SetOriginDataType(DT_FLOAT);
  data1->GetOpDesc()->MutableOutputDesc(0)->SetFormat(ge::FORMAT_NCHW);
  data1->GetOpDesc()->MutableOutputDesc(0)->SetOriginFormat(ge::FORMAT_NCHW);
  if (!AttrUtils::SetInt(data1->GetOpDesc(), "index", 0)) {
    return nullptr;
  }
  return graph;
}
}

TEST_F(EventNodeConverterUt, ConvertSend_Ok) {
  auto graph = BuildSendGraph();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  LoweringGlobalData lgd = GlobalDataFaker(root_model).Build();;
  auto send = graph->FindNode("send");
  auto recv = graph->FindNode("recv");

  auto shape_holder = ValueHolder::CreateFeed(0);
  auto address_holder = ValueHolder::CreateFeed(1);
  auto space_registry_array = OpImplSpaceRegistryV2Array();
  space_registry_array[static_cast<size_t>(gert::OppImplVersionTag::kOpp)] = SpaceRegistryFaker().Build();
  lgd.SetSpaceRegistriesV2(space_registry_array);
  LowerInput lower_input{{}, {}, &lgd};
  auto ret = LoweringEventNode(send, lower_input);
  ASSERT_TRUE(ret.result.IsSuccess());
  EXPECT_EQ(ret.order_holders.size(), 0);
  EXPECT_EQ(ret.out_addrs.size(), 0);
  EXPECT_EQ(ret.out_shapes.size(), 0);

  ret = LoweringEventNode(recv, lower_input);
  ASSERT_TRUE(ret.result.IsSuccess());
  EXPECT_EQ(ret.order_holders.size(), 0);
  EXPECT_EQ(ret.out_addrs.size(), 0);
  EXPECT_EQ(ret.out_shapes.size(), 0);
}
}  // namespace gert