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
#include "kernel/common_kernel_impl/build_tensor.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/utils/attr_utils.h"
#include "faker/space_registry_faker.h"
#include "common/const_data_helper.h"

namespace gert {
using namespace bg;
LowerResult LoweringShapeNode(const ge::NodePtr &node, const LowerInput &lower_input);

class ShapeConverterUT : public BgTestAutoCreate3StageFrame {};
TEST_F(ShapeConverterUT, ConvertShapeOk) {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")->NODE("ShapeNode", "Shape")->NODE("NetOutput", "NetOutput"));
  };
  auto graph = ge::ToComputeGraph(g1);
  auto data1 = graph->FindNode("data1");
  ge::AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);
  auto shape_node = graph->FindNode("ShapeNode");

  auto shape_holder = ValueHolder::CreateFeed(0);
  auto address_holder = DevMemValueHolder::CreateSingleDataOutput("Data", {}, 0);

  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto lgd = GlobalDataFaker(root_model).Build();
  bg::LowerConstDataNode(lgd);
  LowerInput lower_input{{shape_holder}, {address_holder}, &lgd};

  auto result = LoweringShapeNode(shape_node, lower_input);

  EXPECT_TRUE(result.result.IsSuccess());
  EXPECT_TRUE(result.order_holders.empty());

  ASSERT_EQ(result.out_shapes.size(), 1);
  EXPECT_EQ(FastNodeTopoChecker(result.out_shapes[0]).InChecker().DataFrom(shape_holder).Result(), "success");

  ASSERT_EQ(result.out_addrs.size(), 1);
  EXPECT_EQ(result.out_addrs[0]->GetFastNode()->GetType(), kernel::kBuildShapeTensorData);
  EXPECT_EQ(result.out_addrs[0]->GetPlacement(), kOnHost);
  EXPECT_EQ(FastNodeTopoChecker(result.out_addrs[0]).StrictConnectFrom(std::vector<FastSrcNode>({shape_holder})), "success");

  // shape的lowering中，不使用address
  EXPECT_EQ(FastNodeTopoChecker(address_holder).StrictConnectTo(0, {}), "success");
}
}  // namespace gert