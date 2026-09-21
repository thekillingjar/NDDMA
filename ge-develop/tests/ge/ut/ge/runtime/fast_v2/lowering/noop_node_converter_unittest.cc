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
#include "common/share_graph.h"
namespace gert {
using namespace ge;
using namespace bg;
extern LowerResult LoweringNoOpNode(const ge::NodePtr &node, const LowerInput &lower_input);
class NoOpNodeConverterUt : public bg::BgTest {};

TEST_F(NoOpNodeConverterUt, ConvertNoOp_Ok) {
  auto graph = ShareGraph::BuildNoOpGraph();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  LoweringGlobalData lgd = GlobalDataFaker(root_model).Build();
  auto noop = graph->FindNode("noop");
  ASSERT_NE(noop, nullptr);
  LowerInput lower_input{{}, {}, &lgd};
  auto ret = LoweringNoOpNode(noop, lower_input);
  ASSERT_TRUE(ret.result.IsSuccess());
  EXPECT_EQ(ret.order_holders.size(), 0);
  EXPECT_EQ(ret.out_addrs.size(), 0);
  EXPECT_EQ(ret.out_shapes.size(), 0);
}

TEST_F(NoOpNodeConverterUt, Test_ConvertNoOp_Register_Ok) {
  auto graph = ShareGraph::BuildNoOpGraph();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  LoweringGlobalData lgd = GlobalDataFaker(root_model).Build();
  auto noop = graph->FindNode("noop");
  ASSERT_NE(noop, nullptr);
  LowerInput lower_input{{}, {}, &lgd};
  auto register_converter_data = NodeConverterRegistry::GetInstance().FindRegisterData("NoOp");
  ASSERT_NE(register_converter_data, nullptr);
  auto lower_func = register_converter_data->converter;
  ASSERT_NE(lower_func, nullptr);
  auto ret = lower_func(noop, lower_input);
  ASSERT_TRUE(ret.result.IsSuccess());
  EXPECT_EQ(ret.order_holders.size(), 0);
  EXPECT_EQ(ret.out_addrs.size(), 0);
  EXPECT_EQ(ret.out_shapes.size(), 0);
}
}  // namespace gert