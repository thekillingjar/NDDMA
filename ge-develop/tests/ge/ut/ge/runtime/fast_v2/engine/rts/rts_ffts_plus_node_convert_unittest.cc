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
#include "register/node_converter_registry.h"
#include "register/ffts_node_converter_registry.h"
#include <gtest/gtest.h>
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/utils/attr_utils.h"
#include "framework/common/types.h"
#include "common/const_data_helper.h"
#include "faker/space_registry_faker.h"
#include "graph_tuner/rt2_src/graph_tunner_rt2_stub.h"
#include "graph_builder/bg_memory.h"


namespace gert {
using namespace bg;
using namespace ge;
LowerResult LoweringIdentityLikeNode(const ge::NodePtr &node, const FFTSLowerInput &input);

/*
 *
 * netoutput
 *   |
 * identity
 *   |
 * data1
 */
ComputeGraphPtr BuildIdentityGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")->NODE("identity", "Identity")->NODE("NetOutput", "NetOutput"));
  };
  auto graph = ToComputeGraph(g1);
  auto data1 = graph->FindNode("data1");
  data1->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({8, 3, 224, 224}));
  data1->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape({8, 3, 224, 224}));
  data1->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_FLOAT);
  data1->GetOpDesc()->MutableOutputDesc(0)->SetOriginDataType(DT_FLOAT);
  data1->GetOpDesc()->MutableOutputDesc(0)->SetFormat(ge::FORMAT_NCHW);
  data1->GetOpDesc()->MutableOutputDesc(0)->SetOriginFormat(ge::FORMAT_NCHW);
  if (!AttrUtils::SetInt(data1->GetOpDesc(), "index", 0)) {
    return nullptr;
  }
  return graph;
}

class RtsFftsPlusNodeConvertUT : public bg::BgTestAutoCreate3StageFrame {};
TEST_F(RtsFftsPlusNodeConvertUT, MemcpyAsyncLoweringOk) {
  auto graph = BuildIdentityGraph();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  LoweringGlobalData lgd = GlobalDataFaker(root_model).Build();
  bg::LowerConstDataNode(lgd);
  auto identity = graph->FindNode("identity");

  auto shape_holder = ValueHolder::CreateFeed(0);
  ASSERT_NE(shape_holder, nullptr);
  auto address_holder = DevMemValueHolder::CreateSingleDataOutput("Data", {}, 0);
  ASSERT_NE(address_holder, nullptr);
  address_holder->SetPlacement(kTensorPlacementEnd);

  FFTSLowerInput lower_input;
  lower_input.input_shapes.push_back(shape_holder);
  lower_input.input_addrs.push_back(address_holder);
  lower_input.global_data = &lgd;
  uint32_t val = 2;
  lower_input.thread_dim = bg::ValueHolder::CreateConst(&val, sizeof(val));
  lower_input.window_size = bg::ValueHolder::CreateConst(&val, sizeof(val));
  rtFftsPlusTaskInfo_t task_info;
  lower_input.task_info = bg::ValueHolder::CreateConst(&task_info, sizeof(rtFftsPlusTaskInfo_t));
  lower_input.ffts_mem_allocator = bg::CreateFftsMemAllocator(lower_input.window_size, *lower_input.global_data);
  lower_input.ffts_thread_fun = tune::FFTSNodeThreadV2;
  lower_input.mem_pool_types={1};

  auto result = LoweringIdentityLikeNode(identity, lower_input);
  EXPECT_TRUE(result.result.IsSuccess());
  EXPECT_TRUE(!result.order_holders.empty());
  ASSERT_EQ(result.out_shapes.size(), 1);
  ASSERT_EQ(result.out_addrs.size(), 1);
}

}  // namespace gert