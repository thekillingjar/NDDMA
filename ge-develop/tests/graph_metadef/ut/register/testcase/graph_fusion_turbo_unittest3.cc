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

#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"

#include "register/graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "register/graph_optimizer/graph_fusion/graph_fusion_pass_base.h"
#include "register/graph_optimizer/fusion_common/pattern_fusion_base_pass.h"
#include "register/graph_optimizer/fusion_common/fusion_turbo.h"

#include "graph/operator_factory.h"
#include "graph/operator_reg.h"
#include "graph/operator_factory_impl.h"
#include "framework/common/debug/ge_log.h"

using namespace testing;
using namespace ge;
using namespace fe;

namespace fe {
REG_OP(Const)
    .OUTPUT(y,
            TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_UINT32,
                        DT_UINT64, DT_BOOL, DT_DOUBLE}))
        .ATTR(value, Tensor, Tensor())
        .OP_END_FACTORY_REG(Const);

REG_OP(Transpose)
    .INPUT(x,
           TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_UINT32,
                       DT_UINT64, DT_BOOL, DT_DOUBLE}))
        .INPUT(shape, TensorType({DT_INT32, DT_INT64}))
        .OUTPUT(y,
                TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_UINT32,
                            DT_UINT64, DT_BOOL, DT_DOUBLE}))
        .ATTR(axis, Int, 0)
        .ATTR(num_axes, Int, -1)
        .OP_END_FACTORY_REG(Transpose);

REG_OP(Add)
    .INPUT(x1, TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_INT16,
                           DT_INT8, DT_UINT8, DT_DOUBLE, DT_COMPLEX128,
                           DT_COMPLEX64, DT_STRING}))
        .INPUT(x2, TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_INT16,
                               DT_INT8, DT_UINT8, DT_DOUBLE, DT_COMPLEX128,
                               DT_COMPLEX64, DT_STRING}))
        .OUTPUT(y, TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_INT16,
                               DT_INT8, DT_UINT8, DT_DOUBLE, DT_COMPLEX128,
                               DT_COMPLEX64, DT_STRING}))
        .OP_END_FACTORY_REG(Add)

REG_OP(MultiAdd)
    .INPUT(x1, TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_INT16,
                           DT_INT8, DT_UINT8, DT_DOUBLE, DT_COMPLEX128,
                           DT_COMPLEX64, DT_STRING}))
        .INPUT(x2, TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_INT16,
                               DT_INT8, DT_UINT8, DT_DOUBLE, DT_COMPLEX128,
                               DT_COMPLEX64, DT_STRING}))
        .INPUT(x3, TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_INT16,
                               DT_INT8, DT_UINT8, DT_DOUBLE, DT_COMPLEX128,
                               DT_COMPLEX64, DT_STRING}))
        .INPUT(x4, TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_INT16,
                               DT_INT8, DT_UINT8, DT_DOUBLE, DT_COMPLEX128,
                               DT_COMPLEX64, DT_STRING}))
        .OUTPUT(y, TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_INT16,
                               DT_INT8, DT_UINT8, DT_DOUBLE, DT_COMPLEX128,
                               DT_COMPLEX64, DT_STRING}))
        .OP_END_FACTORY_REG(MultiAdd)

REG_OP(Relu)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE,
                          DT_INT8, DT_INT32, DT_INT16, DT_INT64,
                          DT_UINT8, DT_UINT16, DT_QINT8}))
        .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE,
                               DT_INT8, DT_INT32, DT_INT16, DT_INT64,
                               DT_UINT8, DT_UINT16, DT_QINT8}))
        .OP_END_FACTORY_REG(Relu)

REG_OP(End)
    .INPUT(x, TensorType::ALL())
        .OUTPUT(y, TensorType::ALL())
        .ATTR(peerIndex, Int, 0)
        .ATTR(parentOpType, String, "")
        .OP_END_FACTORY_REG(End)

REG_OP(LarsV2Update)
    .INPUT(w, TensorType(DT_FLOAT))
        .INPUT(g, TensorType(DT_FLOAT))
        .INPUT(w_square_sum, TensorType(DT_FLOAT))
        .INPUT(g_square_sum, TensorType(DT_FLOAT))
        .INPUT(weight_decay, TensorType(DT_FLOAT))
        .INPUT(learning_rate, TensorType(DT_FLOAT))
        .OUTPUT(g_new, TensorType(DT_FLOAT))
        .ATTR(hyperpara, Float, 0.001)
        .ATTR(epsilon, Float, 0.00001)
        .ATTR(use_clip, Bool, false)
        .OP_END_FACTORY_REG(LarsV2Update)

REG_OP(SquareSumAll)
    .INPUT(x1, TensorType({DT_FLOAT}))
        .INPUT(x2, TensorType({DT_FLOAT}))
        .OUTPUT(y1, TensorType({DT_FLOAT}))
        .OUTPUT(y2, TensorType({DT_FLOAT}))
        .OP_END_FACTORY_REG(SquareSumAll)

REG_OP(LarsV2)
    .INPUT(w, TensorType(DT_FLOAT))
        .INPUT(g, TensorType(DT_FLOAT))
        .INPUT(weight_decay, TensorType(DT_FLOAT))
        .INPUT(learning_rate, TensorType(DT_FLOAT))
        .OUTPUT(g_new, TensorType(DT_FLOAT))
        .ATTR(hyperpara, Float, 0.001)
        .ATTR(epsilon, Float, 0.00001)
        .ATTR(use_clip, Bool, false)
        .OP_END_FACTORY_REG(LarsV2)

class UTestFusionTurbo3 : public testing::Test {
 public:

 protected:


  void SetUp() {
  }

  void TearDown() {
  }

  ge::NodePtr GetNode(ComputeGraphPtr &graph, const string &name) {
    for (auto &node : graph->GetDirectNode()) {
      if (node->GetName() == name) {
        return node;
      }
    }
    return nullptr;
  }

  ComputeGraphPtr CreateComplexGraph() {
    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test1");

    OpDescPtr op_desc_relu1 = std::make_shared<OpDesc>("relu1", "Relu");
    OpDescPtr op_desc_relu2 = std::make_shared<OpDesc>("relu2", "Relu");
    OpDescPtr op_desc_output = std::make_shared<OpDesc>("output", "NetOutput");

    //add descriptor
    vector<int64_t> dim_a = {8, 4, 16, 16};
    GeShape shape_a(dim_a);
    GeTensorDesc tensor_desc_a(shape_a);
    tensor_desc_a.SetFormat(FORMAT_NCHW);
    tensor_desc_a.SetOriginFormat(FORMAT_NCHW);
    tensor_desc_a.SetDataType(DT_FLOAT16);
    tensor_desc_a.SetOriginDataType(DT_FLOAT);

    vector<int64_t> dim_b = {1, 4, 64, 64};
    GeShape shape_b(dim_b);
    GeTensorDesc tensor_desc_b(shape_b);
    tensor_desc_b.SetFormat(FORMAT_NCHW);
    tensor_desc_b.SetOriginFormat(FORMAT_NCHW);
    tensor_desc_b.SetDataType(DT_FLOAT);
    tensor_desc_b.SetOriginDataType(DT_FLOAT);

    op_desc_relu1->AddInputDesc(tensor_desc_a);
    op_desc_relu1->AddOutputDesc(tensor_desc_b);

    op_desc_relu2->AddInputDesc(tensor_desc_a);
    op_desc_relu2->AddOutputDesc(tensor_desc_b);

    op_desc_output->AddInputDesc(tensor_desc_b);
    op_desc_output->AddInputDesc(tensor_desc_b);

    NodePtr node_relu1 = graph->AddNode(op_desc_relu1);
    NodePtr node_relu2 = graph->AddNode(op_desc_relu2);
    NodePtr node_netoutput = graph->AddNode(op_desc_output);

    GraphUtils::AddEdge(node_relu2->GetOutDataAnchor(0), node_netoutput->GetInDataAnchor(1));

    FusionTurbo acc(graph);
    auto node_add = acc.InsertNodeAfter("add", "Add", node_relu2, 0, 1);
    EXPECT_NE(node_add, nullptr);
    Relations rl(0, {node_relu1, 0});
    acc.LinkInput(rl, node_add);

    unique_ptr<int32_t[]> data(new(std::nothrow) int32_t[4096]);
    WeightInfo w(tensor_desc_a, data.get());
    acc.AddWeight(node_relu1, 0, w);
    acc.AddWeight(node_relu2, 0, w);
    return graph;
  }

  ComputeGraphPtr CreateComplexGraph2() {
    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test2");

    OpDescPtr op_desc_relu1 = std::make_shared<OpDesc>("relu1", "Relu");
    OpDescPtr op_desc_relu2 = std::make_shared<OpDesc>("relu2", "Relu");
    OpDescPtr op_desc_output = std::make_shared<OpDesc>("output", "NetOutput");

    //add descriptor
    vector<int64_t> dim_a = {8, 4, 16, 16};
    GeShape shape_a(dim_a);
    GeTensorDesc tensor_desc_a(shape_a);
    tensor_desc_a.SetFormat(FORMAT_NCHW);
    tensor_desc_a.SetOriginFormat(FORMAT_NCHW);
    tensor_desc_a.SetDataType(DT_FLOAT16);
    tensor_desc_a.SetOriginDataType(DT_FLOAT);

    vector<int64_t> dim_b = {1, 4, 64, 64};
    GeShape shape_b(dim_b);
    GeTensorDesc tensor_desc_b(shape_b);
    tensor_desc_b.SetFormat(FORMAT_NCHW);
    tensor_desc_b.SetOriginFormat(FORMAT_NCHW);
    tensor_desc_b.SetDataType(DT_FLOAT);
    tensor_desc_b.SetOriginDataType(DT_FLOAT);

    op_desc_relu1->AddInputDesc(tensor_desc_a);
    op_desc_relu1->AddOutputDesc(tensor_desc_b);

    op_desc_relu2->AddInputDesc(tensor_desc_a);
    op_desc_relu2->AddOutputDesc(tensor_desc_b);

    op_desc_output->AddInputDesc(tensor_desc_b);
    op_desc_output->AddInputDesc(tensor_desc_b);

    NodePtr node_relu1 = graph->AddNode(op_desc_relu1);
    NodePtr node_relu2 = graph->AddNode(op_desc_relu2);
    NodePtr node_netoutput = graph->AddNode(op_desc_output);

    GraphUtils::AddEdge(node_relu2->GetOutDataAnchor(0), node_netoutput->GetInDataAnchor(1));

    FusionTurbo acc(graph);
    auto node_add = acc.InsertNodeAfter("add", "Add", node_relu2, 0, 0);
    EXPECT_NE(node_add, nullptr);
    Relations rl(1, {node_relu1, 0});
    acc.LinkInput(rl, node_add);

    auto relu1_front = acc.InsertNodeBefore("relu1_front", "Relu", node_relu1, 0);

    auto relu2_front = acc.InsertNodeBefore("relu2_front", "Relu", node_relu2, 0);

    auto relu_top = acc.AddNodeOnly("relu_top", "Relu");
    Relations output_relation(0, {{relu1_front, 0},
                                  {relu2_front, 0}});
    acc.LinkOutput(output_relation, relu_top);
    return graph;
  }
};

TEST_F(UTestFusionTurbo3, test_case_01) {
  auto graph = CreateComplexGraph();
  auto relu_node = graph->FindFirstNodeMatchType("Relu");
  bool has_data_out = FusionTurbo::HasOutData(relu_node);
  EXPECT_EQ(has_data_out, true);
  has_data_out = FusionTurbo::HasOutData(nullptr);
  EXPECT_EQ(has_data_out, false);
  auto net_out_node = graph->FindFirstNodeMatchType("NetOutput");
  ASSERT_NE(net_out_node, nullptr);
  has_data_out = FusionTurbo::HasOutData(net_out_node);
  EXPECT_EQ(has_data_out, false);
}

TEST_F(UTestFusionTurbo3, test_case_02) {
  auto graph = CreateComplexGraph();
  FusionTurbo ft(graph);
  auto relu_node = graph->FindFirstNodeMatchType("Relu");
  Status ret = ft.RemoveDanglingNode(relu_node);
  EXPECT_EQ(ret, FAILED);

  auto net_out_node = graph->FindFirstNodeMatchType("NetOutput");
  ASSERT_NE(net_out_node, nullptr);
  ge::GraphUtils::AddEdge(net_out_node->GetOutControlAnchor(), relu_node->GetInControlAnchor());
  ret = ft.RemoveDanglingNode(net_out_node);
  EXPECT_EQ(ret, FAILED);
  auto remain_net_out_node = graph->FindFirstNodeMatchType("NetOutput");
  EXPECT_TRUE(remain_net_out_node == net_out_node);

  ret = ft.RemoveDanglingNode(net_out_node, true);
  EXPECT_EQ(ret, SUCCESS);
  remain_net_out_node = graph->FindFirstNodeMatchType("NetOutput");
  EXPECT_EQ(remain_net_out_node, nullptr);
}

TEST_F(UTestFusionTurbo3, test_case_03) {
  auto graph = CreateComplexGraph2();
  ge::NodePtr relu_top = nullptr;
  for (const auto &node : graph->GetDirectNode()) {
    if (node->GetName() == "relu_top") {
      auto out_data_nodes = node->GetOutNodes();
      ASSERT_EQ(out_data_nodes.size(), 2);
      EXPECT_EQ(out_data_nodes.at(0)->GetName(), "relu1_front");
      EXPECT_EQ(out_data_nodes.at(1)->GetName(), "relu2_front");
      relu_top = node;
    }
  }
  EXPECT_NE(relu_top, nullptr);

  FusionTurbo ft(graph);

  auto &tensor_desc = relu_top->GetOpDesc()->GetOutputDesc(0);
  unique_ptr<int32_t[]> data(new(std::nothrow) int32_t[4096]);
  WeightInfo w(tensor_desc, data.get());
  auto const_node = ft.AddWeightAfter(relu_top, 0, w);
  ASSERT_NE(const_node, nullptr);
  auto const_out_data_nodes = const_node->GetOutNodes();
  ASSERT_EQ(const_out_data_nodes.size(), 2);
  EXPECT_EQ(const_out_data_nodes.at(0)->GetName(), "relu1_front");
  EXPECT_EQ(const_out_data_nodes.at(1)->GetName(), "relu2_front");

  auto relu_top_out_data_nodes = relu_top->GetOutNodes();
  ASSERT_EQ(relu_top_out_data_nodes.size(), 0);
}
}
