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

namespace gert {
using namespace bg;
LowerResult LoweringUnsupportedV1ControlOps(const ge::NodePtr &node, const LowerInput &lower_input);

class StreamControlConverterUT : public BgTestAutoCreateFrame {};
TEST_F(StreamControlConverterUT, TestLoweringUnsupportedV1ControlOps) {
  DEF_GRAPH(g) {
                  CHAIN(NODE("data", "Data")->NODE("Enter", "Enter")->NODE("NetOutput", "NetOutput"));
  };
  auto graph = ge::ToComputeGraph(g);
  auto v1_node = graph->FindNode("Enter");
  auto shape_holder = ValueHolder::CreateFeed(0);
  auto address_holder = DevMemValueHolder::CreateSingleDataOutput("Data", {}, 0);

  LoweringGlobalData lgd;
  LowerInput lower_input{{shape_holder}, {address_holder}, &lgd};

  auto result = LoweringUnsupportedV1ControlOps(v1_node, lower_input);

  EXPECT_FALSE(result.result.IsSuccess());
}
}  // namespace gert