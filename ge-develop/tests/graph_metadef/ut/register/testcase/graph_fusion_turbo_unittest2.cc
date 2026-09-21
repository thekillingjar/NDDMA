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

class UTestFusionTurbo2 : public testing::Test {
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

    unique_ptr<int32_t[]> data(new(std::nothrow) int32_t[4096]);
    WeightInfo w(tensor_desc_a, data.get());
    acc.AddWeight(relu1_front, 0, w);
    acc.AddWeight(relu2_front, 0, w);
    return graph;
  }

  static void DumpGraph(const ge::ComputeGraphPtr graph, string graph_name) {
    printf("start to dump graph %s...\n", graph_name.c_str());
    for (ge::NodePtr node : graph->GetAllNodes()) {
      printf("node name = %s.\n", node->GetName().c_str());
      for (ge::OutDataAnchorPtr anchor : node->GetAllOutDataAnchors()) {
        for (ge::InDataAnchorPtr peer_in_anchor : anchor->GetPeerInDataAnchors()) {
          printf("    node name = %s[%d], out data node name = %s[%d].\n",
                 node->GetName().c_str(),
                 anchor->GetIdx(),
                 peer_in_anchor->GetOwnerNode()->GetName().c_str(),
                 peer_in_anchor->GetIdx());
        }
      }
      if (node->GetOutControlAnchor() != nullptr) {
        for (ge::InControlAnchorPtr peer_in_anchor : node->GetOutControlAnchor()->GetPeerInControlAnchors()) {
          printf("    node name = %s, out control node name = %s.\n", node->GetName().c_str(),
                 peer_in_anchor->GetOwnerNode()->GetName().c_str());
        }
      }
    }
  }

};

TEST_F(UTestFusionTurbo2, test_case_01) {
  auto graph = CreateComplexGraph();
  FusionTurbo acc(graph);
  string name = "add_new";
  string type = "Add";
  auto node = acc.AddNodeOnly(name, type);
  ASSERT_NE(node, nullptr);

  auto relu1 = GetNode(graph, "relu1");
  auto relu2 = GetNode(graph, "relu2");
  auto add = GetNode(graph, "add");
  auto out = GetNode(graph, "output");

  auto relu1_input = relu1->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();
  auto relu2_input = relu2->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();

  Relations input_relations = {{0, {relu1_input, 0}},
                               {1, {relu2_input, 0}}};
  Relations output_relations = {0, {out, 0}};
  Status ret = acc.MultiInOne(node, input_relations, output_relations, {relu1, relu2, add}, true);
  EXPECT_EQ(ret, SUCCESS);

  auto relu1_out_nodes = relu1_input->GetOutDataNodes();
  auto relu2_out_nodes = relu2_input->GetOutDataNodes();
  ASSERT_EQ(relu1_out_nodes.size(), 1);
  ASSERT_EQ(relu2_out_nodes.size(), 1);
  EXPECT_EQ(relu1_out_nodes.at(0)->GetName(), "add_new");

  EXPECT_EQ(relu2_out_nodes.at(0)->GetName(), "add_new");
  auto out_in_nodes = out->GetInDataNodes();
  EXPECT_EQ(out_in_nodes.size(), 1);
  EXPECT_EQ(out_in_nodes.at(0)->GetName(), "add_new");
  EXPECT_EQ(graph->GetDirectNodesSize(), 4);
}

TEST_F(UTestFusionTurbo2, test_case_01_1) {
  auto graph = CreateComplexGraph();
  FusionTurbo acc(graph);
  string name = "add_new";
  string type = "Add";
  auto node = acc.AddNodeOnly(name, type);
  ASSERT_NE(node, nullptr);

  auto relu1 = GetNode(graph, "relu1");
  auto relu2 = GetNode(graph, "relu2");
  auto add = GetNode(graph, "add");
  auto out = GetNode(graph, "output");

  auto relu1_input = relu1->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();
  auto relu2_input = relu2->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();

  Relations input_relations = {{0, {relu1_input, 0}},
                               {1, {relu2_input, 0}}};
  Relations output_relations = {0, {out, 0}};
  Status ret = acc.MultiInOne(node, input_relations, output_relations, {}, false);
  EXPECT_EQ(ret, SUCCESS);

  auto relu1_out_nodes = relu1_input->GetOutDataNodes();
  auto relu2_out_nodes = relu2_input->GetOutDataNodes();
  ASSERT_EQ(relu1_out_nodes.size(), 2);
  ASSERT_EQ(relu2_out_nodes.size(), 2);
  EXPECT_EQ(relu1_out_nodes.at(0)->GetName(), "relu1");
  EXPECT_EQ(relu1_out_nodes.at(1)->GetName(), "add_new");

  EXPECT_EQ(relu2_out_nodes.at(0)->GetName(), "relu2");
  EXPECT_EQ(relu2_out_nodes.at(1)->GetName(), "add_new");

  auto out_in_nodes = out->GetInDataNodes();
  EXPECT_EQ(out_in_nodes.size(), 2);
  EXPECT_EQ(out_in_nodes.at(0)->GetName(), "add_new");
  EXPECT_EQ(out_in_nodes.at(1)->GetName(), "add");

  auto add_in_nodes = add->GetInDataNodes();
  EXPECT_EQ(add_in_nodes.size(), 2);
  EXPECT_EQ(add_in_nodes.at(0)->GetName(), "relu1");
  EXPECT_EQ(add_in_nodes.at(1)->GetName(), "relu2");
}

TEST_F(UTestFusionTurbo2, test_case_01_2) {
  auto graph = CreateComplexGraph();
  FusionTurbo acc(graph);
  string name = "add_new";
  string type = "Add";
  auto node = acc.AddNodeOnly(name, type);
  ASSERT_NE(node, nullptr);

  auto relu1 = GetNode(graph, "relu1");
  auto relu2 = GetNode(graph, "relu2");
  auto add = GetNode(graph, "add");
  auto out = GetNode(graph, "output");

  auto relu1_input = relu1->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();
  auto relu2_input = relu2->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();

  Relations input_relations = {{0, {relu1_input, 0}},
                               {0, {relu1_input, 0}},
                               {1, {relu2_input, 0}}};
  Relations output_relations = {0, {out, 0}};
  Status ret = acc.MultiInOne(node, input_relations, output_relations, {}, false);
  EXPECT_EQ(ret, SUCCESS);

  auto relu1_out_nodes = relu1_input->GetOutDataNodes();
  auto relu2_out_nodes = relu2_input->GetOutDataNodes();
  ASSERT_EQ(relu1_out_nodes.size(), 2);
  ASSERT_EQ(relu2_out_nodes.size(), 2);
  EXPECT_EQ(relu1_out_nodes.at(0)->GetName(), "relu1");
  EXPECT_EQ(relu1_out_nodes.at(1)->GetName(), "add_new");

  EXPECT_EQ(relu2_out_nodes.at(0)->GetName(), "relu2");
  EXPECT_EQ(relu2_out_nodes.at(1)->GetName(), "add_new");

  auto out_in_nodes = out->GetInDataNodes();
  EXPECT_EQ(out_in_nodes.size(), 2);
  EXPECT_EQ(out_in_nodes.at(0)->GetName(), "add_new");
  EXPECT_EQ(out_in_nodes.at(1)->GetName(), "add");

  auto add_in_nodes = add->GetInDataNodes();
  EXPECT_EQ(add_in_nodes.size(), 2);
  EXPECT_EQ(add_in_nodes.at(0)->GetName(), "relu1");
  EXPECT_EQ(add_in_nodes.at(1)->GetName(), "relu2");
}

TEST_F(UTestFusionTurbo2, test_case_01_3) {
  auto graph = CreateComplexGraph();
  FusionTurbo acc(graph);
  string name = "add_new";
  string type = "Add";
  auto node = acc.AddNodeOnly(name, type);
  ASSERT_NE(node, nullptr);

  auto relu1 = GetNode(graph, "relu1");
  auto relu2 = GetNode(graph, "relu2");
  auto add = GetNode(graph, "add");
  auto out = GetNode(graph, "output");

  auto relu1_input = relu1->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();
  auto relu2_input = relu2->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();

  Relations input_relations = {{0, {relu1_input, 0}},
                               {2, {relu1_input, 0}},
                               {1, {relu2_input, 0}}};
  Relations output_relations = {0, {out, 0}};
  Status ret = acc.MultiInOne(node, input_relations, output_relations, {}, false);
  EXPECT_EQ(ret, FAILED);

  auto relu1_out_nodes = relu1_input->GetOutDataNodes();
  auto relu2_out_nodes = relu2_input->GetOutDataNodes();
  ASSERT_EQ(relu1_out_nodes.size(), 1);
  ASSERT_EQ(relu2_out_nodes.size(), 1);
  EXPECT_EQ(relu1_out_nodes.at(0)->GetName(), "relu1");

  EXPECT_EQ(relu2_out_nodes.at(0)->GetName(), "relu2");

  auto out_in_nodes = out->GetInDataNodes();
  ASSERT_EQ(out_in_nodes.size(), 1);
  EXPECT_EQ(out_in_nodes.at(0)->GetName(), "add");

  auto add_in_nodes = add->GetInDataNodes();
  ASSERT_EQ(add_in_nodes.size(), 2);
  EXPECT_EQ(add_in_nodes.at(0)->GetName(), "relu1");
  EXPECT_EQ(add_in_nodes.at(1)->GetName(), "relu2");
}


TEST_F(UTestFusionTurbo2, test_case_01_4) {
  auto graph = CreateComplexGraph();
  FusionTurbo acc(graph);
  string name = "add_new";
  string type = "Add";
  auto node = acc.AddNodeOnly(name, type);
  ASSERT_NE(node, nullptr);

  auto relu1 = GetNode(graph, "relu1");
  auto relu2 = GetNode(graph, "relu2");
  auto add = GetNode(graph, "add");
  auto out = GetNode(graph, "output");

  auto relu1_input = relu1->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();
  auto relu2_input = relu2->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();

  Relations input_relations = {{0, {relu1_input, 0}},
                               {1, {relu2_input, 0}}};
  Relations output_relations = {{0, {out, 0}},
                                {1, {out, 1}}};
  Status ret = acc.MultiInOne(node, input_relations, output_relations, {}, false);
  EXPECT_EQ(ret, FAILED);

  auto relu1_out_nodes = relu1_input->GetOutDataNodes();
  auto relu2_out_nodes = relu2_input->GetOutDataNodes();
  ASSERT_EQ(relu1_out_nodes.size(), 1);
  ASSERT_EQ(relu2_out_nodes.size(), 1);
  EXPECT_EQ(relu1_out_nodes.at(0)->GetName(), "relu1");

  EXPECT_EQ(relu2_out_nodes.at(0)->GetName(), "relu2");

  auto out_in_nodes = out->GetInDataNodes();
  ASSERT_EQ(out_in_nodes.size(), 1);
  EXPECT_EQ(out_in_nodes.at(0)->GetName(), "add");

  auto add_in_nodes = add->GetInDataNodes();
  ASSERT_EQ(add_in_nodes.size(), 2);
  EXPECT_EQ(add_in_nodes.at(0)->GetName(), "relu1");
  EXPECT_EQ(add_in_nodes.at(1)->GetName(), "relu2");
  EXPECT_EQ(graph->GetDirectNodesSize(), 7);
}


TEST_F(UTestFusionTurbo2, test_case_01_5) {
  auto graph = CreateComplexGraph();
  FusionTurbo acc(graph);
  string name = "add_new";
  string type = "Add";
  auto node = acc.AddNodeOnly(name, type);
  ASSERT_NE(node, nullptr);

  auto relu1 = GetNode(graph, "relu1");
  auto relu2 = GetNode(graph, "relu2");
  auto add = GetNode(graph, "add");
  auto out = GetNode(graph, "output");

  auto relu1_input = relu1->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();
  auto relu2_input = relu2->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();

  Relations input_relations = {{0, {relu1_input, 0}},
                               {1, {relu2_input, 0}}};
  Relations output_relations = {0, {out, 0}};
  ge::GraphUtils::AddEdge(relu1->GetOutControlAnchor(), add->GetInControlAnchor());
  ge::GraphUtils::AddEdge(node->GetOutControlAnchor(), relu1->GetInControlAnchor());
  // This is a very special case! node is in old_nodes!!
  Status ret = acc.MultiInOne(node, input_relations, output_relations, {relu1, relu2, add, node}, false);
  EXPECT_EQ(ret, SUCCESS);

  auto relu1_out_nodes = relu1_input->GetOutDataNodes();
  auto relu2_out_nodes = relu2_input->GetOutDataNodes();
  ASSERT_EQ(relu1_out_nodes.size(), 2);
  ASSERT_EQ(relu2_out_nodes.size(), 2);
  EXPECT_EQ(relu1_out_nodes.at(0)->GetName(), "relu1");
  EXPECT_EQ(relu1_out_nodes.at(1)->GetName(), "add_new");

  EXPECT_EQ(relu1_out_nodes.at(0)->GetOutControlNodes().size(), 3);
  EXPECT_EQ(relu1_out_nodes.at(1)->GetOutControlNodes().size(), 4);

  EXPECT_EQ(relu2_out_nodes.at(0)->GetName(), "relu2");
  EXPECT_EQ(relu2_out_nodes.at(1)->GetName(), "add_new");
  auto out_in_nodes = out->GetInDataNodes();
  EXPECT_EQ(out_in_nodes.size(), 2);
  EXPECT_EQ(out_in_nodes.at(0)->GetName(), "add_new");
  EXPECT_EQ(graph->GetDirectNodesSize(), 7);
}

TEST_F(UTestFusionTurbo2, test_case_2) {
  auto graph = CreateComplexGraph2();
  FusionTurbo acc(graph);
  string name = "add_new";
  string type = "Add";
  auto node = acc.AddNodeOnly(name, type);
  ASSERT_NE(node, nullptr);

  auto relu1 = GetNode(graph, "relu1");
  auto relu2 = GetNode(graph, "relu2");
  auto relu2_front = GetNode(graph, "relu2_front");
  auto add = GetNode(graph, "add");
  auto out = GetNode(graph, "output");

  auto relu1_input = relu1->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();
  auto relu2_front_input = relu2_front->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();

  Relations input_relations = {{0, {relu1_input, 0}},
                               {1, {relu2_front_input, 0}}};
  Relations output_relations = {{0, {add, 0}},
                                {0, {relu2, 0}}};
  Status ret = acc.MultiInOne(node, input_relations, output_relations, {relu1, relu2_front}, true);
  EXPECT_EQ(ret, SUCCESS);

  auto out_nodes1 = relu1_input->GetOutDataNodes();
  auto out_nodes2 = relu2_front_input->GetOutDataNodes();
  ASSERT_EQ(out_nodes1.size(), 1);
  ASSERT_EQ(out_nodes2.size(), 1);

  EXPECT_EQ(out_nodes1.at(0)->GetName(), "add_new");
  EXPECT_EQ(out_nodes2.at(0)->GetName(), "add_new");
  auto out_in_nodes = out->GetInDataNodes();
  EXPECT_EQ(out_in_nodes.size(), 1);
  EXPECT_EQ(out_in_nodes.at(0)->GetName(), "add");

  auto add_new_out_nodes = node->GetOutDataNodes();
  EXPECT_EQ(add_new_out_nodes.size(), 2);
  EXPECT_EQ(add_new_out_nodes.at(0)->GetName(), "add");
  EXPECT_EQ(add_new_out_nodes.at(1)->GetName(), "relu2");
}


TEST_F(UTestFusionTurbo2, test_case_3) {
  auto graph = CreateComplexGraph2();
  FusionTurbo acc(graph);
  string name = "add_new";
  string type = "MultiAdd";
  auto node = acc.AddNodeOnly(name, type);
  ASSERT_NE(node, nullptr);

  auto relu1 = GetNode(graph, "relu1");
  auto relu1_front = GetNode(graph, "relu1_front");
  auto relu2 = GetNode(graph, "relu2");
  auto relu2_front = GetNode(graph, "relu2_front");
  auto add = GetNode(graph, "add");
  auto out = GetNode(graph, "output");

  auto relu1_input = relu1->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();
  auto relu1_front_input = relu1_front->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();
  auto relu2_input = relu2->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();
  auto relu2_front_input = relu2_front->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();

  Relations input_relations = {{0, {relu1_input, 0}},
                               {1, {relu1_front_input, 0}},
                               {2, {relu2_input, 0}},
                               {3, {relu2_front_input, 0}}};
  Relations output_relations = {{0, {add, 0}},
                                {0, {add, 1}}};
  Status ret = acc.MultiInOne(node, input_relations, output_relations,
                              {relu1, relu1_front, relu2, relu2_front}, true);
  EXPECT_EQ(ret, SUCCESS);

  auto out_nodes1 = relu1_front_input->GetOutDataNodes();
  auto out_nodes2 = relu2_front_input->GetOutDataNodes();
  ASSERT_EQ(out_nodes1.size(), 1);
  ASSERT_EQ(out_nodes2.size(), 1);

  EXPECT_EQ(out_nodes1.at(0)->GetName(), "add_new");
  EXPECT_EQ(out_nodes2.at(0)->GetName(), "add_new");
  auto out_in_nodes = out->GetInDataNodes();
  EXPECT_EQ(out_in_nodes.size(), 1);
  EXPECT_EQ(out_in_nodes.at(0)->GetName(), "add");

  auto add_new_out_nodes = node->GetOutDataNodes();
  EXPECT_EQ(add_new_out_nodes.size(), 2);
  EXPECT_EQ(add_new_out_nodes.at(0)->GetName(), "add");
  EXPECT_EQ(add_new_out_nodes.at(1)->GetName(), "add");
  EXPECT_EQ(graph->GetDirectNodesSize(), 5);
}

TEST_F(UTestFusionTurbo2, test_case_4) {
  auto graph = CreateComplexGraph2();
  FusionTurbo acc(graph);
  string name = "add_new";
  string type = "MultiAdd";

  auto relu1 = GetNode(graph, "relu1");
  auto relu1_front = GetNode(graph, "relu1_front");
  auto relu2 = GetNode(graph, "relu2");
  auto relu2_front = GetNode(graph, "relu2_front");
  auto add = GetNode(graph, "add");
  auto out = GetNode(graph, "output");

  auto relu1_input = relu1->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();
  auto relu1_front_input = relu1_front->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();
  auto relu2_input = relu2->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();
  auto relu2_front_input = relu2_front->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();

  Relations input_relations = {{0, {relu1_input, 0}},
                               {1, {relu1_front_input, 0}},
                               {2, {relu2_input, 0}},
                               {3, {relu2_front_input, 0}}};
  Relations output_relations = {{0, {add, 0}},
                                {0, {add, 1}}};
  auto node = acc.MultiInOne(name, type, input_relations, output_relations,
                              {relu1, relu1_front, relu2, relu2_front}, true);
  EXPECT_NE(node, nullptr);

  auto out_nodes1 = relu1_front_input->GetOutDataNodes();
  auto out_nodes2 = relu2_front_input->GetOutDataNodes();
  ASSERT_EQ(out_nodes1.size(), 1);
  ASSERT_EQ(out_nodes2.size(), 1);

  EXPECT_EQ(out_nodes1.at(0)->GetName(), "add_new");
  EXPECT_EQ(out_nodes2.at(0)->GetName(), "add_new");
  auto out_in_nodes = out->GetInDataNodes();
  EXPECT_EQ(out_in_nodes.size(), 1);
  EXPECT_EQ(out_in_nodes.at(0)->GetName(), "add");

  auto add_new_out_nodes = node->GetOutDataNodes();
  EXPECT_EQ(add_new_out_nodes.size(), 2);
  EXPECT_EQ(add_new_out_nodes.at(0)->GetName(), "add");
  EXPECT_EQ(add_new_out_nodes.at(1)->GetName(), "add");
  EXPECT_EQ(graph->GetDirectNodesSize(), 5);
}

TEST_F(UTestFusionTurbo2, test_case_4_1) {
  auto graph = CreateComplexGraph2();
  FusionTurbo acc(graph);
  string name = "add_new";
  string type = "MultiAdd";

  auto relu1 = GetNode(graph, "relu1");
  auto relu1_front = GetNode(graph, "relu1_front");
  auto relu2 = GetNode(graph, "relu2");
  auto relu2_front = GetNode(graph, "relu2_front");
  auto add = GetNode(graph, "add");
  auto out = GetNode(graph, "output");

  auto relu1_front_input = relu1_front->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();
  auto relu2_front_input = relu2_front->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();

  Relations input_relations = {{0, {relu1, 0, PEER}},
                               {1, {relu1_front, 0, PEER}},
                               {2, {relu2, 0, PEER}},
                               {3, {relu2_front, 0, PEER}}};
  Relations output_relations = {{0, {add, 1}},
                                {0, {relu2, 0, PEER}}};
  auto node = acc.MultiInOne(name, type, input_relations, output_relations,
                             {relu1, relu1_front, relu2, relu2_front}, true);
  EXPECT_NE(node, nullptr);

  auto out_nodes1 = relu1_front_input->GetOutDataNodes();
  auto out_nodes2 = relu2_front_input->GetOutDataNodes();
  ASSERT_EQ(out_nodes1.size(), 1);
  ASSERT_EQ(out_nodes2.size(), 1);

  EXPECT_EQ(out_nodes1.at(0)->GetName(), "add_new");
  EXPECT_EQ(out_nodes2.at(0)->GetName(), "add_new");
  auto out_in_nodes = out->GetInDataNodes();
  EXPECT_EQ(out_in_nodes.size(), 1);
  EXPECT_EQ(out_in_nodes.at(0)->GetName(), "add");

  auto add_new_out_nodes = node->GetOutDataNodes();
  EXPECT_EQ(add_new_out_nodes.size(), 2);
  EXPECT_EQ(add_new_out_nodes.at(0)->GetName(), "add");
  EXPECT_EQ(add_new_out_nodes.at(1)->GetName(), "add");
  EXPECT_EQ(graph->GetDirectNodesSize(), 5);
}


TEST_F(UTestFusionTurbo2, test_case_4_2) {
  auto graph = CreateComplexGraph2();
  FusionTurbo acc(graph);
  string name = "add_new";
  string type = "MultiAdd";

  auto relu1 = GetNode(graph, "relu1");
  auto relu1_front = GetNode(graph, "relu1_front");
  auto relu2 = GetNode(graph, "relu2");
  auto relu2_front = GetNode(graph, "relu2_front");
  auto add = GetNode(graph, "add");
  auto out = GetNode(graph, "output");

  auto relu1_front_input = relu1_front->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();
  auto relu2_front_input = relu2_front->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();
  Relations input_relations = {{0, {relu1, 0, PEER_SINGLE}},
                               {1, {relu1_front, 0, PEER_SINGLE}},
                               {2, {relu2, 0, PEER_SINGLE}},
                               {3, {relu2_front, 0, PEER_SINGLE}}};
  Relations output_relations = {{0, {add, 1, PEER}},
                                {0, {relu2, 0, PEER}}};
  auto node = acc.MultiInOne(name, type, input_relations, output_relations,
                             {relu1, relu1_front, relu2, relu2_front}, true);
  EXPECT_NE(node, nullptr);

  auto input_nodes = node->GetInDataNodes();
  ASSERT_EQ(input_nodes.size(), 2);

  auto out_nodes1 = relu1_front_input->GetOutDataNodes();
  auto out_nodes2 = relu2_front_input->GetOutDataNodes();
  ASSERT_EQ(out_nodes1.size(), 1);
  ASSERT_EQ(out_nodes2.size(), 1);

  EXPECT_EQ(out_nodes1.at(0)->GetName(), "add_new");
  EXPECT_EQ(out_nodes2.at(0)->GetName(), "add_new");
  auto out_in_nodes = out->GetInDataNodes();
  EXPECT_EQ(out_in_nodes.size(), 1);
  EXPECT_EQ(out_in_nodes.at(0)->GetName(), "add");

  auto add_new_out_nodes = node->GetOutDataNodes();
  EXPECT_EQ(add_new_out_nodes.size(), 1);
  EXPECT_EQ(add_new_out_nodes.at(0)->GetName(), "add");
  EXPECT_EQ(graph->GetDirectNodesSize(), 5);
}


TEST_F(UTestFusionTurbo2, test_case_4_3) {
  auto graph = CreateComplexGraph2();
  FusionTurbo acc(graph);
  string name = "add_new";
  string type = "MultiAdd";

  auto relu1 = GetNode(graph, "relu1");
  auto relu1_front = GetNode(graph, "relu1_front");
  auto relu2 = GetNode(graph, "relu2");
  auto relu2_front = GetNode(graph, "relu2_front");
  auto add = GetNode(graph, "add");
  auto out = GetNode(graph, "output");

  auto relu1_front_input = relu1_front->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();
  auto relu2_front_input = relu2_front->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();
  Relations input_relations = {{0, {relu1, 0, PEER_SINGLE}},
                               {1, {relu1_front, 0, PEER_SINGLE}},
                               {2, {relu2, 0, PEER_SINGLE}},
                               {3, {relu2_front, 0, PEER_SINGLE}}};
  Relations output_relations = {{0, {add, 1, PEER_SINGLE}},
                                {0, {relu2, 0, PEER_SINGLE}}};
  auto node = acc.MultiInOne(name, type, input_relations, output_relations,
                             {relu1, relu1_front, relu2, relu2_front}, true);
  EXPECT_NE(node, nullptr);

  auto out_nodes1 = relu1_front_input->GetOutDataNodes();
  auto out_nodes2 = relu2_front_input->GetOutDataNodes();
  ASSERT_EQ(out_nodes1.size(), 1);
  ASSERT_EQ(out_nodes2.size(), 1);

  EXPECT_EQ(out_nodes1.at(0)->GetName(), "add_new");
  EXPECT_EQ(out_nodes2.at(0)->GetName(), "add_new");
  auto out_in_nodes = out->GetInDataNodes();
  EXPECT_EQ(out_in_nodes.size(), 1);
  EXPECT_EQ(out_in_nodes.at(0)->GetName(), "add");

  auto add_new_out_nodes = node->GetOutDataNodes();
  EXPECT_EQ(add_new_out_nodes.size(), 1);
  EXPECT_EQ(add_new_out_nodes.at(0)->GetName(), "add");
  EXPECT_EQ(graph->GetDirectNodesSize(), 5);
}

/* Test RemoveMultiNodesOnly. */
TEST_F(UTestFusionTurbo2, test_case_4_4) {
  auto graph = CreateComplexGraph2();
  FusionTurbo acc(graph);
  string name = "add_new";
  string type = "MultiAdd";

  auto relu1 = GetNode(graph, "relu1");
  auto relu1_front = GetNode(graph, "relu1_front");
  auto relu2 = GetNode(graph, "relu2");
  auto relu2_front = GetNode(graph, "relu2_front");
  auto add = GetNode(graph, "add");
  auto out = GetNode(graph, "output");
  FusionTurbo::GetPeerInFirstPair(add, 0);
  FusionTurbo::GetPeerOutPair(add, 0);
  auto relu1_front_input = relu1_front->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();
  auto relu2_front_input = relu2_front->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();
  Relations input_relations = {{0, {relu1, 0, PEER_SINGLE}},
                               {1, {relu1_front, 0, PEER_SINGLE}},
                               {2, {relu2, 0, PEER_SINGLE}},
                               {3, {relu2_front, 0, PEER_SINGLE}}};
  Relations output_relations = {{0, {add, 1, PEER_SINGLE}},
                                {0, {relu2, 0, PEER_SINGLE}}};
  auto node = acc.MultiInOne(name, type, input_relations, output_relations);
  EXPECT_NE(node, nullptr);
  acc.RemoveMultiNodesOnly({nullptr});
  acc.RemoveMultiNodesOnly({relu1, relu1_front, relu2, relu2_front});
  auto out_nodes1 = relu1_front_input->GetOutDataNodes();
  auto out_nodes2 = relu2_front_input->GetOutDataNodes();
  ASSERT_EQ(out_nodes1.size(), 1);
  ASSERT_EQ(out_nodes2.size(), 1);

  EXPECT_EQ(out_nodes1.at(0)->GetName(), "add_new");
  EXPECT_EQ(out_nodes2.at(0)->GetName(), "add_new");
  auto out_in_nodes = out->GetInDataNodes();
  EXPECT_EQ(out_in_nodes.size(), 1);
  EXPECT_EQ(out_in_nodes.at(0)->GetName(), "add");

  auto add_new_out_nodes = node->GetOutDataNodes();
  EXPECT_EQ(add_new_out_nodes.size(), 1);
  EXPECT_EQ(add_new_out_nodes.at(0)->GetName(), "add");
  EXPECT_EQ(graph->GetDirectNodesSize(), 5);
}

void LarsV2UpdateFusion(const ComputeGraphPtr &graph) {
  FusionTurbo ft(graph);
  NodePtr fused_node = graph->FindFirstNodeMatchType("LarsV2");

  OpDescPtr fused_desc = fused_node->GetOpDesc();

  // new the square_sum_all node
  NodePtr square_sum_all_node = ft.AddNodeOnly(fused_desc->GetName() + "/SquareSumAll", "SquareSumAll");
  NodePtr lars_v2_update_node = ft.AddNodeOnly(fused_desc->GetName() + "/LarsUpdate", "LarsV2Update");

  auto square_sum_all_op_desc = square_sum_all_node->GetOpDesc();

  Relations square_sum_input_relation;
  square_sum_input_relation.Add(0, {fused_node, 0, PEER})
                           .Add(1, {fused_node, 1, PEER});

  Relations lars_v2_input_relation = {{2, {square_sum_all_node, 0}}, {3, {square_sum_all_node, 1}},
                                      {0, {fused_node, 0, PEER}}, {1, {fused_node, 1, PEER}},
                                      {4, {fused_node, 2, PEER}}, {5, {fused_node, 3, PEER}}};

  Relations lars_v2_output_relation = {0, {fused_node, 0, PEER}};
  FusionTurbo::LinkInput(square_sum_input_relation, square_sum_all_node);
  FusionTurbo::LinkOutput(lars_v2_output_relation, lars_v2_update_node);
  FusionTurbo::LinkInput(lars_v2_input_relation, lars_v2_update_node);
  FusionTurbo::TransferInCtrlEdges({fused_node}, lars_v2_update_node);
  FusionTurbo::TransferOutCtrlEdges({fused_node}, lars_v2_update_node);

  ft.RemoveNodeOnly(fused_node);
}

TEST_F(UTestFusionTurbo2, test_case_4_5) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test1");
  fe::FusionTurbo ft(graph);
  auto data0 = ft.AddNodeOnly("data0", "Data");
  auto data0_output = data0->GetOpDesc()->MutableOutputDesc(0);
  vector<int64_t> data_0_output_shape = {1, 2, 3, 4};
  data0_output->SetShape(ge::GeShape(data_0_output_shape));

  auto data1 = ft.AddNodeOnly("data1", "Data");
  auto data1_output = data1->GetOpDesc()->MutableOutputDesc(0);
  vector<int64_t> data_1_output_shape = {2, 4, 6, 8};
  data1_output->SetShape(ge::GeShape(data_1_output_shape));

  auto data2 = ft.AddNodeOnly("data2", "Data");
  auto data2_output = data2->GetOpDesc()->MutableOutputDesc(0);
  vector<int64_t> data_2_output_shape = {3, 6, 9, 12};
  data2_output->SetShape(ge::GeShape(data_2_output_shape));

  auto data3 = ft.AddNodeOnly("data3", "Data");
  auto data3_output = data3->GetOpDesc()->MutableOutputDesc(0);
  vector<int64_t> data_3_output_shape = {4, 8, 12, 16};
  data3_output->SetShape(ge::GeShape(data_3_output_shape));

  auto end = ft.AddNodeOnly("end", "End");
  auto end_input = end->GetOpDesc()->MutableInputDesc(0);
  vector<int64_t> end_input_shape = {100};
  end_input->SetShape(ge::GeShape(end_input_shape));

  auto lars_v2 = ft.AddNodeOnly("lars_v2", "LarsV2");
  fe::Relations input_relation({{0, {data0, 0}}, {1, {data1, 0}},
                                {2, {data2, 0}}, {3, {data3, 0}}});
  fe::Relations output_relation;
  output_relation.Add(0, {end, 0});
  fe::FusionTurbo::LinkInput(input_relation, lars_v2);
  fe::FusionTurbo::LinkOutput(output_relation, lars_v2);

  LarsV2UpdateFusion(graph);

  EXPECT_EQ(graph->GetDirectNodesSize(), 7);
  size_t expected_op = 0;
  ge::NodePtr square_sum_all;
  for (const auto &node: graph->GetDirectNode()) {
    if (node->GetType() == "SquareSumAll") {
      expected_op++;
      auto op_desc = node->GetOpDesc();
      square_sum_all = node;
      ASSERT_EQ(op_desc->GetInputsSize(), 2);
      auto input0 = op_desc->MutableInputDesc(0);
      EXPECT_EQ(input0->GetShape().GetDims(), data_0_output_shape);

      auto input1 = op_desc->MutableInputDesc(1);
      EXPECT_EQ(input1->GetShape().GetDims(), data_1_output_shape);

      ASSERT_EQ(op_desc->GetOutputsSize(), 2);
      auto output0 = op_desc->MutableOutputDesc(0);
      EXPECT_EQ(output0->GetShape(), ge::GeShape());

      auto output1 = op_desc->MutableOutputDesc(1);
      EXPECT_EQ(output1->GetShape(), ge::GeShape());

      auto input_nodes = node->GetInDataNodes();
      EXPECT_EQ(input_nodes.size(), 2);
      EXPECT_EQ(input_nodes.at(0), data0);
      EXPECT_EQ(input_nodes.at(1), data1);
    }
  }

  for (const auto &node: graph->GetDirectNode()) {
    if (node->GetType() == "LarsV2Update") {
      expected_op++;
      auto op_desc = node->GetOpDesc();
      ASSERT_EQ(op_desc->GetInputsSize(), 6);
      auto input0 = op_desc->MutableInputDesc(0);
      EXPECT_EQ(input0->GetShape().GetDims(), data_0_output_shape);

      auto input1 = op_desc->MutableInputDesc(1);
      EXPECT_EQ(input1->GetShape().GetDims(), data_1_output_shape);

      auto input2 = op_desc->MutableInputDesc(2);
      EXPECT_EQ(input2->GetShape(), ge::GeShape());

      auto input3 = op_desc->MutableInputDesc(3);
      EXPECT_EQ(input3->GetShape(), ge::GeShape());

      auto input4 = op_desc->MutableInputDesc(4);
      EXPECT_EQ(input4->GetShape().GetDims(), data_2_output_shape);

      auto input5 = op_desc->MutableInputDesc(5);
      EXPECT_EQ(input5->GetShape().GetDims(), data_3_output_shape);

      ASSERT_EQ(op_desc->GetOutputsSize(), 1);
      auto output = op_desc->MutableOutputDesc(0);
      EXPECT_EQ(output->GetShape().GetDims(), end_input_shape);

      auto input_nodes = node->GetInDataNodes();
      EXPECT_EQ(input_nodes.size(), 6);
      EXPECT_EQ(input_nodes.at(0), data0);
      EXPECT_EQ(input_nodes.at(1), data1);

      EXPECT_EQ(input_nodes.at(2), square_sum_all);
      EXPECT_EQ(input_nodes.at(3), square_sum_all);

      EXPECT_EQ(input_nodes.at(4), data2);
      EXPECT_EQ(input_nodes.at(5), data3);

      auto output_nodes = node->GetOutDataNodes();
      EXPECT_EQ(output_nodes.size(), 1);
      EXPECT_EQ(output_nodes.at(0), end);
    }
  }
  EXPECT_EQ(expected_op, 2);
}

TEST_F(UTestFusionTurbo2, test_case_multiinone_out_relaitons_empty) {
  auto graph = CreateComplexGraph();
  FusionTurbo acc(graph);
  string name = "add_new";
  string type = "Add";
  auto node = acc.AddNodeOnly(name, type);
  ASSERT_NE(node, nullptr);

  auto relu1 = GetNode(graph, "relu1");
  auto relu2 = GetNode(graph, "relu2");
  auto add = GetNode(graph, "add");
  auto out = GetNode(graph, "output");

  auto relu1_input = relu1->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();
  auto relu2_input = relu2->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();

  Relations input_relations = {{0, {relu1_input, 0}},
                               {1, {relu2_input, 0}}};
  Relations output_relations = {};
  Status ret = acc.MultiInOne(node, input_relations, output_relations, {}, false);
  ASSERT_EQ(ret, SUCCESS);
}
}
