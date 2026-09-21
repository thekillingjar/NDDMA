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
LowerResult LoweringSizeNode(const ge::NodePtr &node, const LowerInput &lower_input);

class SizeConverterUT : public BgTestAutoCreateFrame {};
TEST_F(SizeConverterUT, ConvertSizeOk) {
  Create3StageFrames();
  auto graph = ShareGraph::BuildSizeGraph();
  graph->TopologicalSorting();
auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).Build();

  auto shape_holder = ValueHolder::CreateFeed(0);
  auto address_holder = DevMemValueHolder::CreateSingleDataOutput("Data", {}, 0);

  LowerInput lower_input{{shape_holder}, {address_holder}, &global_data};

  auto ret = LoweringSizeNode(graph->FindNode("Size"), lower_input);
  ASSERT_TRUE(ret.result.IsSuccess());
  ASSERT_EQ(ret.out_addrs.size(), 1);
  ASSERT_EQ(ret.out_shapes.size(), 1);

  auto exe_graph = ret.out_addrs[0]->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr();
  ASSERT_NE(exe_graph, nullptr);
  // graph compare
}
}  // namespace gert