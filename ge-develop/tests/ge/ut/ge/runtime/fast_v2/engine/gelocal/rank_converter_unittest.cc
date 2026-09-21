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
#include <gtest/gtest.h>
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/utils/attr_utils.h"
#include "engine/gelocal/inputs_converter.h"
#include "graph/utils/graph_utils.h"
#include "graph_builder/exe_graph_comparer.h"
#include "faker/global_data_faker.h"
#include "common/share_graph.h"
#include "exe_graph/lowering/frame_selector.h"

namespace gert {
using namespace bg;
LowerResult LoweringRankNode(const ge::NodePtr &node, const LowerInput &lower_input);

class RankConverterUT : public BgTestAutoCreateFrame {};
TEST_F(RankConverterUT, ConvertRankOk) {
  Create3StageFrames();
  DEF_GRAPH(g1) {
                  CHAIN(NODE("data1", "Data")->NODE("Rank", "Rank")->NODE("NetOutput", "NetOutput"));
                };
  auto graph = ge::ToComputeGraph(g1);
auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).Build();
  auto data1 = graph->FindNode("data1");
  ge::AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);
  auto shape_node = graph->FindNode("Rank");
  const auto &input_desc = shape_node->GetOpDesc()->MutableInputDesc(static_cast<uint32_t>(0));
  input_desc->SetShape(ge::GeShape({1,2}));

  auto shape_holder = ValueHolder::CreateFeed(0);
  auto address_holder = ValueHolder::CreateFeed(1);

  LowerInput lower_input{{shape_holder}, {std::dynamic_pointer_cast<bg::DevMemValueHolder>(address_holder)}, &global_data};

  auto result = LoweringRankNode(shape_node, lower_input);

  EXPECT_TRUE(result.result.IsSuccess());
  EXPECT_EQ(result.order_holders.size(), 1);

  ASSERT_EQ(result.out_shapes.size(), 1);
  ASSERT_EQ(result.out_addrs.size(), 1);
  EXPECT_EQ(result.out_addrs[0]->GetFastNode()->GetType(), "RankKernel");
  EXPECT_EQ(result.out_addrs[0]->GetPlacement(), kOnHost);
}
}  // namespace gert