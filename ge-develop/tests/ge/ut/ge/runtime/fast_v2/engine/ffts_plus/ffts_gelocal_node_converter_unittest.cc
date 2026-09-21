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
#include "kernel/common_kernel_impl/build_tensor.h"
#include "register/node_converter_registry.h"
#include "register/ffts_node_converter_registry.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/utils/attr_utils.h"
#include "common/const_data_helper.h"
#include "register/op_impl_registry.h"

namespace gert {
using namespace bg;

ge::graphStatus InferShapeStub(InferShapeContext *context) {
  return ge::GRAPH_SUCCESS;
}
IMPL_OP(Reshape).InferShape(InferShapeStub);

LowerResult LoweringReshape(const ge::NodePtr &node, const FFTSLowerInput &lower_input);
class ReshapeFftsConverterUT : public BgTestAutoCreate3StageFrame {};
TEST_F(ReshapeFftsConverterUT, ConvertReshapeOk) {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")->EDGE(0, 0)->NODE("reshape", "Reshape")->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("shape", "Const")->EDGE(0, 1)->NODE("reshape", "Reshape"));
  };
  auto graph = ge::ToComputeGraph(g1);
  auto data1 = graph->FindNode("data1");
  ge::AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);
  auto reshape_node = graph->FindNode("reshape");
  auto const_node = graph->FindNode("shape");

  auto tensor = std::make_shared<ge::GeTensor>(const_node->GetOpDesc()->GetOutputDesc(0));
  std::vector<int64_t> data(4, 1);
  tensor->SetData(reinterpret_cast<uint8_t *>(data.data()), data.size() * sizeof(int64_t));
  ge::AttrUtils::SetTensor(const_node->GetOpDesc(), "value", tensor);

  auto shape_holder = ValueHolder::CreateFeed(0);
  auto address_holder = DevMemValueHolder::CreateSingleDataOutput("Data", {}, 0);

  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  ASSERT_NE(root_model, nullptr);
  auto lgd = GlobalDataFaker(root_model).Build();
  bg::LowerConstDataNode(lgd);
  FFTSLowerInput lower_input;
  lower_input.input_shapes.push_back(shape_holder);
  lower_input.input_addrs.push_back(address_holder);
  lower_input.global_data = &lgd;
  lower_input.mem_pool_types = {1, 0};

  auto result = LoweringReshape(reshape_node, lower_input);
  EXPECT_TRUE(result.result.IsSuccess());
  EXPECT_TRUE(result.order_holders.empty());
  ASSERT_EQ(result.out_shapes.size(), 1);
  ASSERT_EQ(result.out_addrs.size(), 1);
  const auto *mem_pool_types = reshape_node->GetOpDesc()->GetExtAttr<std::vector<uint32_t>>("_ffts_memory_pool_type");
  ASSERT_NE(mem_pool_types, nullptr);
  ASSERT_EQ(mem_pool_types->size(), 1UL);
  ASSERT_EQ((*mem_pool_types)[0], 1);
}
}  // namespace gert
