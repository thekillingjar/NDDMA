/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include <gtest/gtest.h>

#include "macro_utils/dt_public_scope.h"
#include "graph/passes/standard_optimize/constant_fuse_same_pass.h"
#include "graph_builder_utils.h"
#include "macro_utils/dt_public_unscope.h"

#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "common/ge_inner_error_codes.h"
#include "common/types.h"

using namespace std;
using namespace testing;
using namespace ge;
const char *const kOriginElementNumAttrName = "origin_element_num";

class UtestGraphPassesConstantFuseSamePass : public testing::Test {
protected:
  void SetUp() {}
  void TearDown() {}
};

//   const1      const2
//      |          |
//           add1
// change to be
//   const1
//     |
//   |   |
//    add1
namespace {
ComputeGraphPtr BuildGraph1() {
  auto builder = ut::GraphBuilder("g1");
  auto const1 = builder.AddNode("const1", CONSTANT, 0, 1, FORMAT_NCHW, DT_FLOAT, {1});
  auto const2 = builder.AddNode("const2", CONSTANT, 0, 1, FORMAT_NCHW, DT_FLOAT, {1});
  auto add1 = builder.AddNode("add1", ADD, 2, 1);

  float weight[] = {0.0f};
  GeTensorDesc weight_desc(GeShape({1}), FORMAT_NHWC, DT_FLOAT);
  GeTensorPtr tensor = std::make_shared<GeTensor>(weight_desc, (uint8_t *) weight, sizeof(weight));
  int64_t origin_val_size = 1;
  (void)ge::AttrUtils::SetInt(tensor->MutableTensorDesc(), kOriginElementNumAttrName, origin_val_size);
  OpDescUtils::SetWeights(const1, {tensor});
  OpDescUtils::SetWeights(const2, {tensor});

  builder.AddDataEdge(const1, 0, add1, 0);
  builder.AddDataEdge(const2, 0, add1, 1);

  return builder.GetGraph();
}

//   const1      const2
//      |          |
//           add1
// not fuse
ComputeGraphPtr BuildGraph2() {
  auto builder = ut::GraphBuilder("g1");
  auto const1 = builder.AddNode("const1", CONSTANT, 0, 1, FORMAT_NCHW, DT_FLOAT, {1});
  auto const2 = builder.AddNode("const2", CONSTANT, 0, 1, FORMAT_NCHW, DT_FLOAT, {1});
  auto add1 = builder.AddNode("add1", ADD, 2, 1);

  float weight[] = {0.0f};
  GeTensorDesc weight_desc(GeShape({1}), FORMAT_NHWC, DT_FLOAT);
  GeTensorPtr tensor = std::make_shared<GeTensor>(weight_desc, (uint8_t *) weight, sizeof(weight));
  OpDescUtils::SetWeights(const1, {tensor});

  float weight2[] = {1.23f};
  GeTensorDesc weight2_desc(GeShape({1}), FORMAT_NHWC, DT_FLOAT);
  GeTensorPtr tensor2 = std::make_shared<GeTensor>(weight2_desc, (uint8_t *) weight2, sizeof(weight2));
  OpDescUtils::SetWeights(const2, {tensor2});

  builder.AddDataEdge(const1, 0, add1, 0);
  builder.AddDataEdge(const2, 0, add1, 1);

  return builder.GetGraph();
}

//   const1      const2           const3      const4
//      |          |                 |          |
//           add1                        add2
// change to be
//           const1
//             |
//    |   |         |   |
//     add1          add2
ComputeGraphPtr BuildGraph3() {
  auto builder = ut::GraphBuilder("g1");
  auto const1 = builder.AddNode("const1", CONSTANT, 0, 1, FORMAT_NCHW, DT_FLOAT, {1});
  auto const2 = builder.AddNode("const2", CONSTANT, 0, 1, FORMAT_NCHW, DT_FLOAT, {1});
  auto add1 = builder.AddNode("add1", ADD, 2, 1);
  auto const3 = builder.AddNode("const3", CONSTANT, 0, 1, FORMAT_NCHW, DT_FLOAT, {1});
  auto const4 = builder.AddNode("const4", CONSTANT, 0, 1, FORMAT_NCHW, DT_FLOAT, {1});
  auto add2 = builder.AddNode("add2", ADD, 2, 1);

  float weight[] = {0.0f};
  GeTensorDesc weight_desc(GeShape({1}), FORMAT_NHWC, DT_FLOAT);
  GeTensorPtr tensor = std::make_shared<GeTensor>(weight_desc, (uint8_t *) weight, sizeof(weight));
  int64_t origin_val_size = 1;
  (void)ge::AttrUtils::SetInt(tensor->MutableTensorDesc(), kOriginElementNumAttrName, origin_val_size);
  OpDescUtils::SetWeights(const1, {tensor});
  OpDescUtils::SetWeights(const2, {tensor});
  OpDescUtils::SetWeights(const3, {tensor});
  OpDescUtils::SetWeights(const4, {tensor});

  builder.AddDataEdge(const1, 0, add1, 0);
  builder.AddDataEdge(const2, 0, add1, 1);
  builder.AddDataEdge(const3, 0, add2, 0);
  builder.AddDataEdge(const4, 0, add2, 1);

  return builder.GetGraph();
}

//   const1      const2           const3      const4
//      |          |                 |          |
//           add1                        add2
// change to be
//    const1       const3
//      |             |
//    |   |         |   |
//     add1          add2
ComputeGraphPtr BuildGraph4() {
  auto builder = ut::GraphBuilder("g1");
  auto const1 = builder.AddNode("const1", CONSTANT, 0, 1, FORMAT_NCHW, DT_FLOAT, {1});
  auto const2 = builder.AddNode("const2", CONSTANT, 0, 1, FORMAT_NCHW, DT_FLOAT, {1});
  auto add1 = builder.AddNode("add1", ADD, 2, 1);
  auto const3 = builder.AddNode("const3", CONSTANT, 0, 1, FORMAT_NCHW, DT_FLOAT, {1});
  auto const4 = builder.AddNode("const4", CONSTANT, 0, 1, FORMAT_NCHW, DT_FLOAT, {1});
  auto add2 = builder.AddNode("add2", ADD, 2, 1);

  float weight[] = {0.0f};
  GeTensorDesc weight_desc(GeShape({1}), FORMAT_NHWC, DT_FLOAT);
  GeTensorPtr tensor = std::make_shared<GeTensor>(weight_desc, (uint8_t *) weight, sizeof(weight));
  int64_t origin_val_size = 1;
  (void)ge::AttrUtils::SetInt(tensor->MutableTensorDesc(), kOriginElementNumAttrName, origin_val_size);
  OpDescUtils::SetWeights(const1, {tensor});
  OpDescUtils::SetWeights(const2, {tensor});

  float weight2[] = {1.23f};
  GeTensorDesc weight2_desc(GeShape({1}), FORMAT_NHWC, DT_FLOAT);
  GeTensorPtr tensor2 = std::make_shared<GeTensor>(weight2_desc, (uint8_t *) weight2, sizeof(weight2));
  (void)ge::AttrUtils::SetInt(tensor2->MutableTensorDesc(), kOriginElementNumAttrName, origin_val_size);
  OpDescUtils::SetWeights(const3, {tensor2});
  OpDescUtils::SetWeights(const4, {tensor2});

  builder.AddDataEdge(const1, 0, add1, 0);
  builder.AddDataEdge(const2, 0, add1, 1);
  builder.AddDataEdge(const3, 0, add2, 0);
  builder.AddDataEdge(const4, 0, add2, 1);

  return builder.GetGraph();
}

ComputeGraphPtr BuildGraph4_1() {
  auto builder = ut::GraphBuilder("g1");
  auto const1 = builder.AddNode("const1", CONSTANT, 0, 1, FORMAT_NCHW, DT_FLOAT, {1});
  auto const2 = builder.AddNode("const2", CONSTANT, 0, 1, FORMAT_NCHW, DT_FLOAT, {1});
  auto add1 = builder.AddNode("add1", ADD, 2, 1);
  auto const3 = builder.AddNode("const3", CONSTANT, 0, 1, FORMAT_NHWC, DT_FLOAT, {1});
  auto const4 = builder.AddNode("const4", CONSTANT, 0, 1, FORMAT_NHWC, DT_FLOAT, {1});
  auto add2 = builder.AddNode("add2", ADD, 2, 1);

  float weight[] = {0.0f};
  GeTensorDesc weight_desc(GeShape({1}), FORMAT_NCHW, DT_FLOAT);
  GeTensorPtr tensor = std::make_shared<GeTensor>(weight_desc, (uint8_t *) weight, sizeof(weight));
  int64_t origin_val_size = 1;
  (void)ge::AttrUtils::SetInt(tensor->MutableTensorDesc(), kOriginElementNumAttrName, origin_val_size);
  OpDescUtils::SetWeights(const1, {tensor});
  OpDescUtils::SetWeights(const2, {tensor});

  float weight2[] = {0.0f};
  GeTensorDesc weight2_desc(GeShape({1}), FORMAT_NHWC, DT_FLOAT);
  GeTensorPtr tensor2 = std::make_shared<GeTensor>(weight2_desc, (uint8_t *) weight2, sizeof(weight2));
  (void)ge::AttrUtils::SetInt(tensor2->MutableTensorDesc(), kOriginElementNumAttrName, origin_val_size);
  OpDescUtils::SetWeights(const3, {tensor2});
  OpDescUtils::SetWeights(const4, {tensor2});

  builder.AddDataEdge(const1, 0, add1, 0);
  builder.AddDataEdge(const2, 0, add1, 1);
  builder.AddDataEdge(const3, 0, add2, 0);
  builder.AddDataEdge(const4, 0, add2, 1);

  return builder.GetGraph();
}

ComputeGraphPtr BuildGraph4_2() {
  auto builder = ut::GraphBuilder("g1");
  auto const1 = builder.AddNode("const1", CONSTANT, 0, 1, FORMAT_NCHW, DT_FLOAT, {1});
  auto const2 = builder.AddNode("const2", CONSTANT, 0, 1, FORMAT_NCHW, DT_FLOAT, {1});
  auto add1 = builder.AddNode("add1", ADD, 2, 1);
  auto const3 = builder.AddNode("const3", CONSTANT, 0, 1, FORMAT_NCHW, DT_FLOAT, {1, 1});
  auto const4 = builder.AddNode("const4", CONSTANT, 0, 1, FORMAT_NCHW, DT_FLOAT, {1, 1});
  auto add2 = builder.AddNode("add2", ADD, 2, 1);

  float weight[] = {0.0f};
  GeTensorDesc weight_desc(GeShape({1}), FORMAT_NCHW, DT_FLOAT);
  GeTensorPtr tensor = std::make_shared<GeTensor>(weight_desc, (uint8_t *) weight, sizeof(weight));
  int64_t origin_val_size = 1;
  (void)ge::AttrUtils::SetInt(tensor->MutableTensorDesc(), kOriginElementNumAttrName, origin_val_size);
  OpDescUtils::SetWeights(const1, {tensor});
  OpDescUtils::SetWeights(const2, {tensor});

  float weight2[] = {0.0f};
  GeTensorDesc weight2_desc(GeShape({1, 1}), FORMAT_NCHW, DT_FLOAT);
  GeTensorPtr tensor2 = std::make_shared<GeTensor>(weight2_desc, (uint8_t *) weight2, sizeof(weight2));
  (void)ge::AttrUtils::SetInt(tensor2->MutableTensorDesc(), kOriginElementNumAttrName, origin_val_size);
  OpDescUtils::SetWeights(const3, {tensor2});
  OpDescUtils::SetWeights(const4, {tensor2});

  builder.AddDataEdge(const1, 0, add1, 0);
  builder.AddDataEdge(const2, 0, add1, 1);
  builder.AddDataEdge(const3, 0, add2, 0);
  builder.AddDataEdge(const4, 0, add2, 1);

  return builder.GetGraph();
}

//   const1      const2           const3      const4 --->reshape1
//      |          |                 |          |
//           add1                        add2
// change to be
//    const1       const3 --->reshape1
//      |             |
//    |   |         |   |
//     add1          add2
ComputeGraphPtr BuildGraph4_3() {
  auto builder = ut::GraphBuilder("g1");
  auto const1 = builder.AddNode("const1", CONSTANT, 0, 1, FORMAT_NCHW, DT_FLOAT, {1});
  auto const2 = builder.AddNode("const2", CONSTANT, 0, 1, FORMAT_NCHW, DT_FLOAT, {1});
  auto add1 = builder.AddNode("add1", ADD, 2, 1);
  auto const3 = builder.AddNode("const3", CONSTANT, 0, 1, FORMAT_NCHW, DT_FLOAT, {1});
  auto const4 = builder.AddNode("const4", CONSTANT, 0, 1, FORMAT_NCHW, DT_FLOAT, {1});
  auto add2 = builder.AddNode("add2", ADD, 2, 1);
  auto reshape1 = builder.AddNode("reshape1", RESHAPE, 1, 1);

  float weight[] = {0.0f};
  GeTensorDesc weight_desc(GeShape({1}), FORMAT_NHWC, DT_FLOAT);
  GeTensorPtr tensor = std::make_shared<GeTensor>(weight_desc, (uint8_t *) weight, sizeof(weight));
  int64_t origin_val_size = 1;
  (void)ge::AttrUtils::SetInt(tensor->MutableTensorDesc(), kOriginElementNumAttrName, origin_val_size);
  OpDescUtils::SetWeights(const1, {tensor});
  OpDescUtils::SetWeights(const2, {tensor});

  float weight2[] = {1.23f};
  GeTensorDesc weight2_desc(GeShape({1}), FORMAT_NHWC, DT_FLOAT);
  GeTensorPtr tensor2 = std::make_shared<GeTensor>(weight2_desc, (uint8_t *) weight2, sizeof(weight2));
  (void)ge::AttrUtils::SetInt(tensor2->MutableTensorDesc(), kOriginElementNumAttrName, origin_val_size);
  OpDescUtils::SetWeights(const3, {tensor2});
  OpDescUtils::SetWeights(const4, {tensor2});

  builder.AddDataEdge(const1, 0, add1, 0);
  builder.AddDataEdge(const2, 0, add1, 1);
  builder.AddDataEdge(const3, 0, add2, 0);
  builder.AddDataEdge(const4, 0, add2, 1);
  builder.AddControlEdge(const4, reshape1);

  return builder.GetGraph();
}

} // namespace

TEST_F(UtestGraphPassesConstantFuseSamePass, success_const_has_data) {
  ge::ComputeGraphPtr graph = BuildGraph1();
  EXPECT_EQ(graph->GetDirectNodesSize(), 3);

  ConstantFuseSamePass pass;
  Status status = pass.Run(graph);
  EXPECT_EQ(status, SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(), 2);
}

TEST_F(UtestGraphPassesConstantFuseSamePass, not_fuse_const_has_data) {
  ge::ComputeGraphPtr graph = BuildGraph2();
  EXPECT_EQ(graph->GetDirectNodesSize(), 3);

  ConstantFuseSamePass pass;
  Status status = pass.Run(graph);
  EXPECT_EQ(status, SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(), 3);
}

TEST_F(UtestGraphPassesConstantFuseSamePass, success_const_has_data_1) {
  ge::ComputeGraphPtr graph = BuildGraph3();
  EXPECT_EQ(graph->GetDirectNodesSize(), 6);

  ConstantFuseSamePass pass;
  Status status = pass.Run(graph);
  EXPECT_EQ(status, SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(), 3);
}

TEST_F(UtestGraphPassesConstantFuseSamePass, success_const_has_data_2) {
  ge::ComputeGraphPtr graph = BuildGraph4();
  EXPECT_EQ(graph->GetDirectNodesSize(), 6);

  ConstantFuseSamePass pass;
  Status status = pass.Run(graph);
  EXPECT_EQ(status, SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(), 4);
}

TEST_F(UtestGraphPassesConstantFuseSamePass, success_const_has_data_3) {
  ge::ComputeGraphPtr graph = BuildGraph4_1();
  EXPECT_EQ(graph->GetDirectNodesSize(), 6);

  ConstantFuseSamePass pass;
  Status status = pass.Run(graph);
  EXPECT_EQ(status, SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(), 4);
}

TEST_F(UtestGraphPassesConstantFuseSamePass, success_const_has_data_4) {
  ge::ComputeGraphPtr graph = nullptr;
  ConstantFuseSamePass pass;
  Status status = pass.Run(graph);
  EXPECT_EQ(status, GE_GRAPH_PARAM_NULLPTR);

  graph = BuildGraph4_2();
  EXPECT_EQ(graph->GetDirectNodesSize(), 6);

  status = pass.Run(graph);
  EXPECT_EQ(status, SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(), 4);
}

TEST_F(UtestGraphPassesConstantFuseSamePass, success_const_has_data_5) {
  ge::ComputeGraphPtr graph = BuildGraph4_3();
  EXPECT_EQ(graph->GetDirectNodesSize(), 7);

  ConstantFuseSamePass pass;
  Status status = pass.Run(graph);
  EXPECT_EQ(status, SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(), 5);

  auto reshape1 = graph->FindNode("reshape1");
  for (auto &node : reshape1->GetInControlNodes()) {
    EXPECT_EQ(node->GetName(), "const3");
  }
  auto const4 = graph->FindNode("const4");
  EXPECT_EQ(const4, nullptr);
}
