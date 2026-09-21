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
#include "graph/utils/node_utils.h"

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
REG_OP(Data)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .ATTR(index, Int, 0)
    .OP_END_FACTORY_REG(Data)


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

REG_OP(Relu)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE,
                          DT_INT8, DT_INT32, DT_INT16, DT_INT64,
                          DT_UINT8, DT_UINT16, DT_QINT8}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE,
                           DT_INT8, DT_INT32, DT_INT16, DT_INT64,
                           DT_UINT8, DT_UINT16, DT_QINT8}))
    .OP_END_FACTORY_REG(Relu)

REG_OP(Split)
    .INPUT(split_dim, TensorType({DT_INT32}))
    .INPUT(x, TensorType::BasicType())
    .DYNAMIC_OUTPUT(y, TensorType::BasicType())
    .REQUIRED_ATTR(num_split, Int)
    .OP_END_FACTORY_REG(Split)

REG_OP(Concat)
    .DYNAMIC_INPUT(x, TensorType::BasicType())
    .INPUT(concat_dim, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::BasicType())
    .ATTR(N, Int, 1)
    .OP_END_FACTORY_REG(Concat)

REG_OP(RaggedTensorFromVariant)
    .INPUT(encoded_ragged, TensorType({DT_VARIANT}))
    .DYNAMIC_OUTPUT(output_nested_splits, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(output_dense_values, TensorType::BasicType())
    .REQUIRED_ATTR(input_ragged_rank, Int)
    .REQUIRED_ATTR(output_ragged_rank, Int)
    .REQUIRED_ATTR(Tvalues, Type)
    .ATTR(Tsplits, Type, DT_INT64)
    .OP_END_FACTORY_REG(RaggedTensorFromVariant)

class UTestFusionTurbo : public testing::Test {
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

  ComputeGraphPtr CreateGraphSingleInAndOut() {
    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test1");
    OpDescPtr op_desc_cast1 = std::make_shared<OpDesc>("cast1", "Cast");
    OpDescPtr op_desc_relu = std::make_shared<OpDesc>("relu", "Relu");
    OpDescPtr op_desc_cast2 = std::make_shared<OpDesc>("cast2", "Cast");
    OpDescPtr op_desc_output = std::make_shared<OpDesc>("output", "NetOutput");
    OpDescPtr op_desc_input = std::make_shared<OpDesc>("other", "Other");

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

    vector<int64_t> dim_c = {1, 4, 64, 64};
    GeShape shape_c(dim_c);
    GeTensorDesc tensor_desc_c(shape_c);
    tensor_desc_c.SetFormat(FORMAT_NCHW);
    tensor_desc_c.SetOriginFormat(FORMAT_NCHW);
    tensor_desc_c.SetDataType(DT_FLOAT);
    tensor_desc_c.SetOriginDataType(DT_FLOAT);

    //vector<int64_t> dim_d;
    GeShape shape_d(dim_a);
    GeTensorDesc tensor_desc_d(shape_d);
    tensor_desc_d.SetFormat(FORMAT_NCHW);
    tensor_desc_d.SetOriginFormat(FORMAT_NCHW);
    tensor_desc_d.SetDataType(DT_FLOAT16);
    tensor_desc_d.SetOriginDataType(DT_FLOAT);

    op_desc_input->AddOutputDesc(tensor_desc_a);

    op_desc_cast1->AddInputDesc(tensor_desc_a);
    op_desc_cast1->AddOutputDesc(tensor_desc_b);

    op_desc_relu->AddInputDesc(tensor_desc_b);
    op_desc_relu->AddOutputDesc(tensor_desc_c);

    op_desc_cast2->AddInputDesc(tensor_desc_c);
    op_desc_cast2->AddOutputDesc(tensor_desc_d);

    op_desc_output->AddInputDesc(tensor_desc_d);

    ge::AttrUtils::SetStr(op_desc_relu, "_op_compile_strategy", "{}");
    ge::AttrUtils::SetInt(op_desc_relu, "_keep_dtype", 1);

    NodePtr node_cast1 = graph->AddNode(op_desc_cast1);
    NodePtr node_relu = graph->AddNode(op_desc_relu);
    NodePtr node_cast2 = graph->AddNode(op_desc_cast2);
    NodePtr node_netoutput = graph->AddNode(op_desc_output);
    NodePtr node_other = graph->AddNode(op_desc_input);

    GraphUtils::AddEdge(node_other->GetOutDataAnchor(0), node_cast1->GetInDataAnchor(0));
    GraphUtils::AddEdge(node_cast1->GetOutDataAnchor(0), node_relu->GetInDataAnchor(0));
    GraphUtils::AddEdge(node_relu->GetOutDataAnchor(0), node_cast2->GetInDataAnchor(0));
    GraphUtils::AddEdge(node_cast2->GetOutDataAnchor(0), node_netoutput->GetInDataAnchor(0));

    return graph;
  }

  ComputeGraphPtr CreateGraphParentAndSub() {

    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    OpDescPtr op_desc_cast1 = std::make_shared<OpDesc>("cast1", "Cast");
    OpDescPtr op_desc_add1 = std::make_shared<OpDesc>("add1", "Add");
    OpDescPtr op_desc_partcall = std::make_shared<OpDesc>("partioncall", "PartionCall");
    OpDescPtr op_desc_partout = std::make_shared<OpDesc>("partout", "PartionOut");
    OpDescPtr op_desc_add2 = std::make_shared<OpDesc>("add2", "Add");
    OpDescPtr op_desc_output = std::make_shared<OpDesc>("output", "NetOutput");
    OpDescPtr op_desc_output1 = std::make_shared<OpDesc>("output1", "NetOutput");
    OpDescPtr op_desc_input = std::make_shared<OpDesc>("other", "Other");
    OpDescPtr op_desc_input1 = std::make_shared<OpDesc>("other1", "Other");

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

    vector<int64_t> dim_c = {1, 4, 64, 64};
    GeShape shape_c(dim_c);
    GeTensorDesc tensor_desc_c(shape_c);
    tensor_desc_c.SetFormat(FORMAT_NCHW);
    tensor_desc_c.SetOriginFormat(FORMAT_NCHW);
    tensor_desc_c.SetDataType(DT_FLOAT);
    tensor_desc_c.SetOriginDataType(DT_FLOAT);

    //vector<int64_t> dim_d;
    GeShape shape_d(dim_a);
    GeTensorDesc tensor_desc_d(shape_d);
    tensor_desc_d.SetFormat(FORMAT_NCHW);
    tensor_desc_d.SetOriginFormat(FORMAT_NCHW);
    tensor_desc_d.SetDataType(DT_FLOAT16);
    tensor_desc_d.SetOriginDataType(DT_FLOAT);

    GeShape shape_e(dim_a);
    GeTensorDesc tensor_desc_e(shape_e);
    tensor_desc_e.SetFormat(FORMAT_NCHW);
    tensor_desc_e.SetOriginFormat(FORMAT_NCHW);
    tensor_desc_e.SetDataType(DT_FLOAT16);
    tensor_desc_e.SetOriginDataType(DT_FLOAT);


    op_desc_input->AddOutputDesc(tensor_desc_a);

    op_desc_cast1->AddInputDesc(tensor_desc_a);
    op_desc_cast1->AddOutputDesc(tensor_desc_b);

    op_desc_add1->AddInputDesc(tensor_desc_b);
    op_desc_add1->AddInputDesc(tensor_desc_b);
    op_desc_add1->AddOutputDesc(tensor_desc_c);

    op_desc_partcall->AddInputDesc(tensor_desc_c);
    op_desc_partcall->AddOutputDesc(tensor_desc_d);
    op_desc_partcall->AddOutputDesc(tensor_desc_d);
    op_desc_partout->AddInputDesc(tensor_desc_d);

    op_desc_add2->AddInputDesc(tensor_desc_d);
    op_desc_add2->AddInputDesc(tensor_desc_d);
    op_desc_add2->AddOutputDesc(tensor_desc_e);

    op_desc_input1->AddOutputDesc(tensor_desc_d);

    op_desc_output->AddInputDesc(tensor_desc_e);
    op_desc_output1->AddInputDesc(tensor_desc_e);

    NodePtr node_cast1 = graph->AddNode(op_desc_cast1);
    NodePtr node_add1 = graph->AddNode(op_desc_add1);
    NodePtr node_add2 = graph->AddNode(op_desc_add2);
    NodePtr node_netoutput = graph->AddNode(op_desc_output);
    NodePtr node_netoutput1 = graph->AddNode(op_desc_output1);
    NodePtr node_other = graph->AddNode(op_desc_input);
    NodePtr node_partcall = graph->AddNode(op_desc_partcall);
    NodePtr node_partout = graph->AddNode(op_desc_partout);
    NodePtr node_other1 = graph->AddNode(op_desc_input1);

    GraphUtils::AddEdge(node_other->GetOutDataAnchor(0), node_cast1->GetInDataAnchor(0));
    GraphUtils::AddEdge(node_cast1->GetOutDataAnchor(0), node_add1->GetInDataAnchor(0));
    GraphUtils::AddEdge(node_other->GetOutDataAnchor(0), node_add1->GetInDataAnchor(1));
    GraphUtils::AddEdge(node_add1->GetOutDataAnchor(0), node_partcall->GetInDataAnchor(0));
    GraphUtils::AddEdge(node_partcall->GetOutDataAnchor(0), node_partout->GetInDataAnchor(0));
    GraphUtils::AddEdge(node_partcall->GetOutDataAnchor(1), node_add2->GetInDataAnchor(0));
    GraphUtils::AddEdge(node_other1->GetOutDataAnchor(0), node_add2->GetInDataAnchor(1));
    GraphUtils::AddEdge(node_add2->GetOutDataAnchor(0), node_netoutput->GetInDataAnchor(0));
    GraphUtils::AddEdge(node_add2->GetOutDataAnchor(0), node_netoutput1->GetInDataAnchor(0));
    
    // subgraph
    ComputeGraphPtr subgraph = std::make_shared<ComputeGraph>("subgraph");
    OpDescPtr op_desc_sub_data1 = std::make_shared<OpDesc>("data1", "Data");
    OpDescPtr op_desc_sub_input = std::make_shared<OpDesc>("other", "Other");
    OpDescPtr op_desc_sub_add = std::make_shared<OpDesc>("sub_add", "Add");
    OpDescPtr op_desc_net_in = std::make_shared<OpDesc>("net_in", "NetOutInput");
    OpDescPtr op_desc_sub_output = std::make_shared<OpDesc>("output", "NetOutput");
    
    op_desc_sub_data1->AddInputDesc(tensor_desc_c);
    op_desc_sub_data1->AddOutputDesc(tensor_desc_c);
    op_desc_sub_input->AddOutputDesc(tensor_desc_c);

    op_desc_sub_add->AddInputDesc(tensor_desc_c);
    op_desc_sub_add->AddInputDesc(tensor_desc_c);
    op_desc_sub_add->AddOutputDesc(tensor_desc_d);

    op_desc_net_in->AddOutputDesc(tensor_desc_d);
    op_desc_sub_output->AddInputDesc(tensor_desc_d);
    op_desc_sub_output->AddInputDesc(tensor_desc_d);

    NodePtr sub_data_node1 = subgraph->AddNode(op_desc_sub_data1);
    NodePtr sub_input_node = subgraph->AddNode(op_desc_sub_input);
    NodePtr sub_add_node = subgraph->AddNode(op_desc_sub_add);
    NodePtr sub_netin_node = subgraph->AddNode(op_desc_net_in);
    NodePtr sub_sub_output_node = subgraph->AddNode(op_desc_sub_output);

    GraphUtils::AddEdge(sub_data_node1->GetOutDataAnchor(0), sub_add_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(sub_input_node->GetOutDataAnchor(0), sub_add_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(sub_add_node->GetOutDataAnchor(0), sub_sub_output_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(sub_netin_node->GetOutDataAnchor(0), sub_sub_output_node->GetInDataAnchor(1));
    ge::AttrUtils::SetInt(sub_data_node1->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
    ge::AttrUtils::SetInt(sub_sub_output_node->GetOpDesc()->MutableInputDesc(0), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
    ge::AttrUtils::SetInt(sub_sub_output_node->GetOpDesc()->MutableInputDesc(1), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);
    node_partcall->GetOpDesc()->AddSubgraphName("subgraph1");
    ge::NodeUtils::SetSubgraph(*node_partcall, 0, subgraph);
    subgraph->SetParentNode(node_partcall);
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

TEST_F(UTestFusionTurbo, test_case_01) {
  auto graph = CreateGraphSingleInAndOut();
  FusionTurbo acc(graph);
  string name = "transpose";
  string type = "Transpose";

  auto cast2 = GetNode(graph, "cast2");
  auto node = acc.InsertNodeBefore(name, type, cast2, 0);
  ASSERT_NE(node, nullptr);
  EXPECT_EQ(node->GetName(), name);
  EXPECT_EQ(node->GetType(), type);
  vector<int64_t> dims = {1, 4, 64, 64};
  auto input = node->GetOpDesc()->MutableInputDesc(0);
  EXPECT_EQ(input->GetShape().GetDims(), dims);
  EXPECT_EQ(input->GetFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(input->GetDataType(), ge::DT_FLOAT);

  auto output = node->GetOpDesc()->MutableOutputDesc(0);
  EXPECT_EQ(output->GetShape().GetDims(), dims);
  EXPECT_EQ(output->GetFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(output->GetDataType(), ge::DT_FLOAT);

  EXPECT_EQ(node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "relu");
  EXPECT_EQ(node->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode()->GetName(), "cast2");
}


TEST_F(UTestFusionTurbo, test_case_01_1) {
  auto graph = CreateGraphSingleInAndOut();
  FusionTurbo acc(graph);
  string name = "transpose";
  string type = "Transpose";

  auto cast2 = GetNode(graph, "cast2");
  acc.BreakInput(cast2, {0});

  auto node = acc.InsertNodeBefore(name, type, cast2, 0);
  ASSERT_NE(node, nullptr);
  EXPECT_EQ(node->GetName(), name);
  EXPECT_EQ(node->GetType(), type);
  vector<int64_t> dims_null = {};
  vector<int64_t> dims = {1, 4, 64, 64};
  auto input = node->GetOpDesc()->MutableInputDesc(0);
  EXPECT_EQ(input->GetShape().GetDims(), dims_null);
  EXPECT_EQ(input->GetFormat(), ge::FORMAT_ND);
  EXPECT_EQ(input->GetDataType(), ge::DT_FLOAT);

  auto output = node->GetOpDesc()->MutableOutputDesc(0);
  EXPECT_EQ(output->GetShape().GetDims(), dims);
  EXPECT_EQ(output->GetFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(output->GetDataType(), ge::DT_FLOAT);

  EXPECT_EQ(node->GetInDataAnchor(0)->GetPeerOutAnchor(), nullptr);
  EXPECT_EQ(node->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode()->GetName(), "cast2");
}

TEST_F(UTestFusionTurbo, test_case_02) {
  auto graph = CreateGraphSingleInAndOut();
  FusionTurbo acc(graph);
  string name = "transpose";
  string type = "Transpose";

  auto relu = GetNode(graph, "relu");
  auto node = acc.InsertNodeAfter(name, type, relu, 0);

  ASSERT_NE(node, nullptr);
  EXPECT_EQ(node->GetName(), name);
  EXPECT_EQ(node->GetType(), type);
  vector<int64_t> dims = {1, 4, 64, 64};
  auto input = node->GetOpDesc()->MutableInputDesc(0);
  EXPECT_EQ(input->GetShape().GetDims(), dims);
  EXPECT_EQ(input->GetFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(input->GetDataType(), ge::DT_FLOAT);

  auto output = node->GetOpDesc()->MutableOutputDesc(0);
  EXPECT_EQ(output->GetShape().GetDims(), dims);
  EXPECT_EQ(output->GetFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(output->GetDataType(), ge::DT_FLOAT);

  EXPECT_EQ(node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "relu");
  EXPECT_EQ(node->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode()->GetName(), "cast2");
}


TEST_F(UTestFusionTurbo, test_case_02_1) {
  auto graph = CreateGraphSingleInAndOut();
  FusionTurbo acc(graph);
  string name = "transpose";
  string type = "Transpose";

  auto relu = GetNode(graph, "relu");
  acc.BreakAllOutput(relu);
  auto node = acc.InsertNodeAfter(name, type, relu, 0);

  ASSERT_NE(node, nullptr);
  EXPECT_EQ(node->GetName(), name);
  EXPECT_EQ(node->GetType(), type);
  vector<int64_t> dims = {1, 4, 64, 64};
  vector<int64_t> dims_null = {};
  auto input = node->GetOpDesc()->MutableInputDesc(0);
  EXPECT_EQ(input->GetShape().GetDims(), dims);
  EXPECT_EQ(input->GetFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(input->GetDataType(), ge::DT_FLOAT);

  auto output = node->GetOpDesc()->MutableOutputDesc(0);
  EXPECT_EQ(output->GetShape().GetDims(), dims_null);
  EXPECT_EQ(output->GetFormat(), ge::FORMAT_ND);
  EXPECT_EQ(output->GetDataType(), ge::DT_FLOAT);

  EXPECT_EQ(node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "relu");
  EXPECT_EQ(node->GetOutDataAnchor(0)->GetPeerInDataAnchors().size(), 0);
}

TEST_F(UTestFusionTurbo, test_concat_and_split_node) {
  auto graph = std::make_shared<ComputeGraph>("test1");
  FusionTurbo acc(graph);
  string name1 = "split";
  string type1 = "Split";
  auto split = acc.AddNodeOnly(name1, type1, 32);

  string name2 = "concat";
  string type2 = "Concat";
  auto concat = acc.AddNodeOnly(name2, type2, 32);
  ASSERT_NE(split, nullptr);
  ASSERT_NE(concat, nullptr);

  auto split_input_size = split->GetOpDesc()->GetAllInputsSize();
  EXPECT_EQ(split_input_size, 2);

  auto split_output_size = split->GetOpDesc()->GetOutputsSize();
  EXPECT_EQ(split_output_size, 32);

  auto concat_input_size = concat->GetOpDesc()->GetAllInputsSize();
  EXPECT_EQ(concat_input_size, 33);

  auto concat_output_size = concat->GetOpDesc()->GetOutputsSize();
  EXPECT_EQ(concat_output_size, 1);

  Relations output_relation;
  for (size_t i = 0; i < 32; ++i) {
    output_relation.Add(i, {concat, static_cast<int32_t>(i)});
  }
  acc.LinkOutput(output_relation, split);
  auto out_data_nodes = split->GetOutDataNodes();
  ASSERT_EQ(out_data_nodes.size(), 32);
  for (size_t i = 0; i < 32; ++i) {
    EXPECT_EQ(out_data_nodes.at(i)->GetName(), "concat");
  }

  auto split_input_0_name = split->GetOpDesc()->GetInputNameByIndex(0);
  EXPECT_EQ(split_input_0_name, "split_dim");

  auto split_input_1_name = split->GetOpDesc()->GetInputNameByIndex(1);
  EXPECT_EQ(split_input_1_name, "x");

  for (size_t i = 0; i < 32; ++i) {
    auto name = concat->GetOpDesc()->GetInputNameByIndex(i);
    EXPECT_EQ(name, "x" + std::to_string(i));
  }

  for (size_t i = 0; i < 32; ++i) {
    auto name = split->GetOpDesc()->GetOutputNameByIndex(i);
    EXPECT_EQ(name, "y" + std::to_string(i));
  }
}

TEST_F(UTestFusionTurbo, test_concat_and_ragged_node) {
  auto graph = std::make_shared<ComputeGraph>("test1");
  FusionTurbo acc(graph);
  string name1 = "ragged";
  string type1 = "RaggedTensorFromVariant";
  auto ragged = acc.AddNodeOnly(name1, type1, 32);

  string name2 = "concat";
  string type2 = "Concat";
  auto concat = acc.AddNodeOnly(name2, type2, 32);
  ASSERT_NE(ragged, nullptr);
  ASSERT_NE(concat, nullptr);

  auto split_input_size = ragged->GetOpDesc()->GetAllInputsSize();
  EXPECT_EQ(split_input_size, 1);

  auto split_output_size = ragged->GetOpDesc()->GetOutputsSize();
  EXPECT_EQ(split_output_size, 33);

  auto concat_input_size = concat->GetOpDesc()->GetAllInputsSize();
  EXPECT_EQ(concat_input_size, 33);

  auto concat_output_size = concat->GetOpDesc()->GetOutputsSize();
  EXPECT_EQ(concat_output_size, 1);

  Relations output_relation;
  for (size_t i = 1; i < 33; ++i) {
    output_relation.Add(i, {concat, static_cast<int32_t>(i)});
  }
  acc.LinkOutput(output_relation, ragged);
  auto out_data_nodes = ragged->GetOutDataNodes();
  ASSERT_EQ(out_data_nodes.size(), 32);
  for (size_t i = 0; i < 32; ++i) {
    EXPECT_EQ(out_data_nodes.at(i)->GetName(), "concat");
  }

  for (size_t i = 0; i < 32; ++i) {
    auto name = concat->GetOpDesc()->GetInputNameByIndex(i);
    EXPECT_EQ(name, "x" + std::to_string(i));
  }

  for (size_t i = 0; i < 32; ++i) {
    auto name = ragged->GetOpDesc()->GetOutputNameByIndex(i);
    EXPECT_EQ(name, "output_nested_splits" + std::to_string(i));
  }
  auto output_32_name = ragged->GetOpDesc()->GetOutputNameByIndex(32);
  EXPECT_EQ(output_32_name, "output_dense_values");
}

TEST_F(UTestFusionTurbo, test_case_03) {
  auto graph = CreateGraphSingleInAndOut();
  FusionTurbo acc(graph);
  string name = "transpose";
  string type = "Transpose";
  auto node1 = acc.AddNodeOnly(name, type);
  auto node2 = acc.AddNodeOnly(name, type);
  ASSERT_NE(node1, nullptr);
  ASSERT_NE(node2, nullptr);
}

/* cast2 already has input so Transpose will not have peer out. */
TEST_F(UTestFusionTurbo, test_case_04) {
  auto graph = CreateGraphSingleInAndOut();
  FusionTurbo acc(graph);
  string name = "transpose";
  string type = "Transpose";
  auto node = acc.AddNodeOnly(name, type);
  ASSERT_NE(node, nullptr);

  auto relu = GetNode(graph, "relu");
  Relations src_list = {{relu, 0}};
  acc.LinkInput(src_list, node, UPDATE_THIS);

  auto cast2 = GetNode(graph, "cast2");
  Relations dst_list = {{cast2, 0}};
  Status ret = acc.LinkOutput(dst_list, node, UPDATE_THIS);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(node->GetName(), name);
  EXPECT_EQ(node->GetType(), type);
  vector<int64_t> dims = {1, 4, 64, 64};
  auto input = node->GetOpDesc()->MutableInputDesc(0);
  EXPECT_EQ(input->GetShape().GetDims(), dims);
  EXPECT_EQ(input->GetFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(input->GetDataType(), ge::DT_FLOAT);

  auto output = node->GetOpDesc()->MutableOutputDesc(0);
  EXPECT_EQ(output->GetShape().GetDims(), dims);
  EXPECT_EQ(output->GetFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(output->GetDataType(), ge::DT_FLOAT);

  EXPECT_EQ(node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "relu");
  EXPECT_EQ(node->GetOutDataAnchor(0)->GetPeerInDataAnchors().size(), 1);
}

TEST_F(UTestFusionTurbo, test_case_04_1) {
  auto graph = CreateGraphSingleInAndOut();
  FusionTurbo acc(graph);
  string name = "transpose";
  string type = "Transpose";
  auto node = acc.AddNodeOnly(name, type);
  ASSERT_NE(node, nullptr);

  auto relu = GetNode(graph, "relu");
  Relations src_list;
  src_list.Add(0, {relu, 0});
  acc.LinkInput(src_list, node, UPDATE_THIS);

  auto cast2 = GetNode(graph, "cast2");
  Relations dst_list = {{cast2, 0}};
  dst_list.Add(0, {cast2, 0});
  Status ret = acc.LinkOutput(dst_list, node, UPDATE_THIS);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(node->GetName(), name);
  EXPECT_EQ(node->GetType(), type);
  vector<int64_t> dims = {1, 4, 64, 64};
  auto input = node->GetOpDesc()->MutableInputDesc(0);
  EXPECT_EQ(input->GetShape().GetDims(), dims);
  EXPECT_EQ(input->GetFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(input->GetDataType(), ge::DT_FLOAT);

  auto output = node->GetOpDesc()->MutableOutputDesc(0);
  EXPECT_EQ(output->GetShape().GetDims(), dims);
  EXPECT_EQ(output->GetFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(output->GetDataType(), ge::DT_FLOAT);

  EXPECT_EQ(node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "relu");
  EXPECT_EQ(node->GetOutDataAnchor(0)->GetPeerInDataAnchors().size(), 1);
}

TEST_F(UTestFusionTurbo, test_case_04_2) {
  auto graph = CreateGraphSingleInAndOut();
  FusionTurbo acc(graph);
  string name = "transpose";
  string type = "Transpose";
  auto node = acc.AddNodeOnly(name, type);
  ASSERT_NE(node, nullptr);

  auto relu = GetNode(graph, "relu");
  Relations src_list;
  src_list.Add(0, {{relu, 0}});
  acc.LinkInput(src_list, node);

  auto cast2 = GetNode(graph, "cast2");
  Relations dst_list;
  dst_list.Add(0, {{cast2, 0}});
  Status ret = acc.LinkOutput(dst_list, node);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(node->GetName(), name);
  EXPECT_EQ(node->GetType(), type);
  vector<int64_t> dims = {1, 4, 64, 64};
  auto input = node->GetOpDesc()->MutableInputDesc(0);
  EXPECT_EQ(input->GetShape().GetDims(), dims);
  EXPECT_EQ(input->GetFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(input->GetDataType(), ge::DT_FLOAT);

  auto output = node->GetOpDesc()->MutableOutputDesc(0);
  EXPECT_EQ(output->GetShape().GetDims(), dims);
  EXPECT_EQ(output->GetFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(output->GetDataType(), ge::DT_FLOAT);

  EXPECT_EQ(node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "relu");
  EXPECT_EQ(node->GetOutDataAnchor(0)->GetPeerInDataAnchors().size(), 1);
}


TEST_F(UTestFusionTurbo, test_case_04_3) {
  auto graph = CreateGraphSingleInAndOut();
  FusionTurbo acc(graph);
  string name = "transpose";
  string type = "Transpose";
  auto node = acc.AddNodeOnly(name, type);
  ASSERT_NE(node, nullptr);

  auto relu = GetNode(graph, "relu");
  Relations src_list = {{relu, 0}};
  acc.LinkInput(src_list, node);

  auto cast2 = GetNode(graph, "cast2");
  auto output_node = GetNode(graph, "output");
  Relations dst_list = {{cast2,       0},
                        {output_node, 0}};
  Status ret = FusionTurbo::LinkOutput(dst_list, node);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(node->GetName(), name);
  EXPECT_EQ(node->GetType(), type);
  vector<int64_t> dims = {1, 4, 64, 64};
  auto input = node->GetOpDesc()->MutableInputDesc(0);
  EXPECT_EQ(input->GetShape().GetDims(), dims);
  EXPECT_EQ(input->GetFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(input->GetDataType(), ge::DT_FLOAT);

  auto output = node->GetOpDesc()->MutableOutputDesc(0);
  EXPECT_EQ(output->GetShape().GetDims(), dims);
  EXPECT_EQ(output->GetFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(output->GetDataType(), ge::DT_FLOAT);

  EXPECT_EQ(node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "relu");
  EXPECT_EQ(node->GetOutDataAnchor(0)->GetPeerInDataAnchors().size(), 1);
}

TEST_F(UTestFusionTurbo, test_case_04_4) {
  auto graph = CreateGraphSingleInAndOut();
  FusionTurbo acc(graph);
  string name = "transpose";
  string type = "Transpose";
  auto node = acc.AddNodeOnly(name, type);
  ASSERT_NE(node, nullptr);
  auto input = node->GetOpDesc()->MutableInputDesc(0);
  input->SetFormat(ge::FORMAT_ND);
  input->SetDataType(ge::DT_UNDEFINED);
  auto output = node->GetOpDesc()->MutableOutputDesc(0);
  output->SetFormat(ge::FORMAT_ND);
  output->SetDataType(ge::DT_UNDEFINED);

  auto relu = GetNode(graph, "relu");
  Relations src_list = {{relu, 0}};
  acc.LinkInput(src_list, node, UPDATE_PEER);

  auto cast2 = GetNode(graph, "cast2");
  Relations dst_list = {{cast2, 0}};
  Status ret = acc.LinkOutput(dst_list, node, UPDATE_PEER);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(node->GetName(), name);
  EXPECT_EQ(node->GetType(), type);
  vector<int64_t> dims = {1, 4, 64, 64};
  vector<int64_t> dims_null = {};

  EXPECT_EQ(input->GetShape().GetDims(), dims_null);
  EXPECT_EQ(input->GetFormat(), ge::FORMAT_ND);
  EXPECT_EQ(input->GetDataType(), ge::DT_UNDEFINED);

  EXPECT_EQ(output->GetShape().GetDims(), dims_null);
  EXPECT_EQ(output->GetFormat(), ge::FORMAT_ND);
  EXPECT_EQ(output->GetDataType(), ge::DT_UNDEFINED);

  auto relu_input = relu->GetOpDesc()->MutableInputDesc(0);
  EXPECT_EQ(relu_input->GetShape().GetDims(), dims);
  EXPECT_EQ(relu_input->GetFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(relu_input->GetDataType(), ge::DT_FLOAT);

  auto relu_output = relu->GetOpDesc()->MutableOutputDesc(0);
  EXPECT_EQ(relu_output->GetShape().GetDims(), dims_null);
  EXPECT_EQ(relu_output->GetFormat(), ge::FORMAT_ND);
  EXPECT_EQ(relu_output->GetDataType(), ge::DT_UNDEFINED);

  auto cast_input = cast2->GetOpDesc()->MutableInputDesc(0);
  EXPECT_EQ(cast_input->GetShape().GetDims(), dims_null);
  EXPECT_EQ(cast_input->GetFormat(), ge::FORMAT_ND);
  EXPECT_EQ(cast_input->GetDataType(), ge::DT_UNDEFINED);

  vector<int64_t> dims_cast = {8, 4, 16, 16};
  auto cast_output = cast2->GetOpDesc()->MutableOutputDesc(0);
  EXPECT_EQ(cast_output->GetShape().GetDims(), dims_cast);
  EXPECT_EQ(cast_output->GetFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(cast_output->GetDataType(), ge::DT_FLOAT16);

  EXPECT_EQ(node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "relu");
  EXPECT_EQ(node->GetOutDataAnchor(0)->GetPeerInDataAnchors().size(), 1);
}

TEST_F(UTestFusionTurbo, test_case_05) {
  auto graph = CreateGraphSingleInAndOut();
  FusionTurbo acc(graph);
  string name = "transpose";
  string type = "Transpose";
  auto node = acc.AddNodeOnly(name, type);
  ASSERT_NE(node, nullptr);

  auto relu = GetNode(graph, "relu");
  Relations src_list = {{relu, 0}};
  acc.LinkInput(src_list, node, UPDATE_THIS);

  auto cast2 = GetNode(graph, "cast2");
  Relations dst_list = {{cast2, 0}};
  acc.BreakInput(cast2, {0});
  Status ret = acc.LinkOutput(dst_list, node, UPDATE_THIS);
  EXPECT_EQ(ret, SUCCESS);

  EXPECT_EQ(node->GetName(), name);
  EXPECT_EQ(node->GetType(), type);
  vector<int64_t> dims = {1, 4, 64, 64};
  auto input = node->GetOpDesc()->MutableInputDesc(0);
  EXPECT_EQ(input->GetShape().GetDims(), dims);
  EXPECT_EQ(input->GetFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(input->GetDataType(), ge::DT_FLOAT);

  auto output = node->GetOpDesc()->MutableOutputDesc(0);
  EXPECT_EQ(output->GetShape().GetDims(), dims);
  EXPECT_EQ(output->GetFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(output->GetDataType(), ge::DT_FLOAT);

  EXPECT_EQ(node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "relu");
  ASSERT_EQ(node->GetOutDataAnchor(0)->GetPeerInDataAnchors().size(), 1);
  EXPECT_EQ(node->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode()->GetName(), "cast2");
}

TEST_F(UTestFusionTurbo, test_case_05_1) {
  auto graph = CreateGraphSingleInAndOut();
  FusionTurbo acc(graph);
  string name = "transpose";
  string type = "Transpose";
  auto node = acc.AddNodeOnly(name, type);
  ASSERT_NE(node, nullptr);

  auto relu = GetNode(graph, "relu");
  Relations src_list;
  src_list.Add(0, {relu, 0});
  acc.LinkInput(src_list, node, UPDATE_THIS);

  auto cast2 = GetNode(graph, "cast2");
  Relations dst_list;
  dst_list.Add(0, {cast2, 0});
  acc.BreakInput(cast2, {0});
  Status ret = acc.LinkOutput(dst_list, node, UPDATE_THIS);
  EXPECT_EQ(ret, SUCCESS);

  EXPECT_EQ(node->GetName(), name);
  EXPECT_EQ(node->GetType(), type);
  vector<int64_t> dims = {1, 4, 64, 64};
  auto input = node->GetOpDesc()->MutableInputDesc(0);
  EXPECT_EQ(input->GetShape().GetDims(), dims);
  EXPECT_EQ(input->GetFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(input->GetDataType(), ge::DT_FLOAT);

  auto output = node->GetOpDesc()->MutableOutputDesc(0);
  EXPECT_EQ(output->GetShape().GetDims(), dims);
  EXPECT_EQ(output->GetFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(output->GetDataType(), ge::DT_FLOAT);

  EXPECT_EQ(node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "relu");
  ASSERT_EQ(node->GetOutDataAnchor(0)->GetPeerInDataAnchors().size(), 1);
  EXPECT_EQ(node->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode()->GetName(), "cast2");
}

TEST_F(UTestFusionTurbo, test_case_05_2) {
  auto graph = CreateGraphSingleInAndOut();
  FusionTurbo acc(graph);
  string name = "transpose";
  string type = "Transpose";
  auto node = acc.AddNodeOnly(name, type);
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  auto input0 = op_desc->GetInputDescPtr(0);
  auto input1 = op_desc->GetInputDescPtr(1);
  EXPECT_EQ(input0->GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(input0->GetShape().GetDimNum(), 0);
  EXPECT_EQ(input0->GetFormat(), ge::FORMAT_ND);

  auto relu = GetNode(graph, "relu");
  Relations src_list;
  src_list.Add(0, {relu, 0});
  src_list.Add(0, {{relu, 0}, {relu, 0}});
  Relations src_list_1 = src_list;

  Relations dst_list;
  dst_list.Add(0, {relu, 0, PEER});
  dst_list.Add(0, {{relu, 0, PEER}, {relu, 0, PEER}});

  acc.LinkInput(src_list_1, node, UPDATE_THIS);
  auto cast2 = GetNode(graph, "cast2");
  acc.BreakInput(cast2, {0});
  Status ret = acc.LinkOutput(dst_list, node, UPDATE_THIS);
  EXPECT_EQ(ret, SUCCESS);

  EXPECT_EQ(node->GetName(), name);
  EXPECT_EQ(node->GetType(), type);
  vector<int64_t> dims = {1, 4, 64, 64};
  auto input = node->GetOpDesc()->MutableInputDesc(0);
  EXPECT_EQ(input->GetShape().GetDims(), dims);
  EXPECT_EQ(input->GetFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(input->GetDataType(), ge::DT_FLOAT);

  auto output = node->GetOpDesc()->MutableOutputDesc(0);
  EXPECT_EQ(output->GetShape().GetDims(), dims);
  EXPECT_EQ(output->GetFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(output->GetDataType(), ge::DT_FLOAT);

  EXPECT_EQ(node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "relu");
  ASSERT_EQ(node->GetOutDataAnchor(0)->GetPeerInDataAnchors().size(), 1);
  EXPECT_EQ(node->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode()->GetName(), "cast2");
}

TEST_F(UTestFusionTurbo, test_case_06) {
  auto graph = CreateGraphSingleInAndOut();
  FusionTurbo acc(graph);
  string name = "transpose";
  string type = "Transpose";
  auto node = acc.AddNodeOnly(name, type);

  ASSERT_NE(node, nullptr);

  auto relu = GetNode(graph, "relu");
  Relations src_list = {{relu, 0}};
  acc.LinkInput(src_list, node, UPDATE_NONE);

  EXPECT_EQ(node->GetName(), name);
  EXPECT_EQ(node->GetType(), type);
  vector<int64_t> dims = {};
  auto input = node->GetOpDesc()->MutableInputDesc(0);
  EXPECT_EQ(input->GetShape().GetDims(), dims);
  EXPECT_EQ(input->GetFormat(), ge::FORMAT_ND);
  EXPECT_EQ(input->GetDataType(), ge::DT_FLOAT);

  EXPECT_EQ(node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "relu");
}

TEST_F(UTestFusionTurbo, test_case_07) {
  auto graph = CreateGraphSingleInAndOut();
  FusionTurbo acc(graph);
  string name = "transpose";
  string type = "Transpose";
  auto node = acc.AddNodeOnly(name, type);
  ASSERT_NE(node, nullptr);

  auto cast2 = GetNode(graph, "cast2");
  Relations dst_list = {{cast2, 0}};
  acc.BreakAllInput(cast2);
  Status ret = acc.LinkOutput(dst_list, node, UPDATE_NONE);
  EXPECT_EQ(ret, SUCCESS);

  EXPECT_EQ(node->GetName(), name);
  EXPECT_EQ(node->GetType(), type);

  vector<int64_t> dims = {};
  auto output = node->GetOpDesc()->MutableOutputDesc(0);
  EXPECT_EQ(output->GetShape().GetDims(), dims);
  EXPECT_EQ(output->GetFormat(), ge::FORMAT_ND);
  EXPECT_EQ(output->GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(node->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode()->GetName(), "cast2");
}

TEST_F(UTestFusionTurbo, test_case_08) {
  auto graph = CreateGraphSingleInAndOut();
  FusionTurbo acc(graph);

  auto cast2 = GetNode(graph, "cast2");
  acc.BreakOutput(cast2, {0});
  acc.BreakOutput(cast2, {1});
  EXPECT_EQ(acc.RemoveNodeWithRelink(cast2, {0}), SUCCESS);
}

TEST_F(UTestFusionTurbo, test_case_09) {
  auto graph = CreateGraphSingleInAndOut();
  FusionTurbo acc(graph);

  auto cast2 = GetNode(graph, "cast2");
  auto cast1 = GetNode(graph, "cast1");
  EXPECT_EQ(acc.RemoveNodeOnly(cast2), SUCCESS);
  EXPECT_EQ(acc.RemoveNodeWithRelink(cast1, {0}), SUCCESS);
}

TEST_F(UTestFusionTurbo, test_case_10) {
  auto graph = CreateGraphSingleInAndOut();
  FusionTurbo acc(graph);

  auto cast2 = GetNode(graph, "cast2");
  cast2->GetOpDesc()->MutableOutputDesc(0)->SetShape(ge::GeShape({-1}));
  cast2->GetOpDesc()->MutableInputDesc(0)->SetOriginShape(ge::GeShape({-1}));
  EXPECT_EQ(false, acc.IsUnknownShape(cast2, 0));
  EXPECT_EQ(false, acc.IsUnknownShape(cast2, 0, true));
  EXPECT_EQ(true, acc.IsUnknownShape(cast2, 0, false));

  EXPECT_EQ(true, acc.IsUnknownOriShape(cast2, 0));
  EXPECT_EQ(true, acc.IsUnknownOriShape(cast2, 0, true));
  EXPECT_EQ(false, acc.IsUnknownOriShape(cast2, 0, false));
}

TEST_F(UTestFusionTurbo, test_case_11) {
  auto graph = CreateGraphSingleInAndOut();
  FusionTurbo acc(graph);
  string name = "transpose";
  string type = "Transpose";
  auto node = acc.AddNodeOnly(name, type);
  ASSERT_NE(node, nullptr);

  auto relu = GetNode(graph, "relu");
  Relations src_list;
  src_list.Add(0, {relu, 0});
  src_list.Add(0, {});
  EXPECT_EQ(SUCCESS, acc.LinkInput(src_list, node, UPDATE_NONE));

  auto cast2 = GetNode(graph, "cast2");

  Relations dst_list;
  dst_list.Add(0, {cast2, 0});
  dst_list.Add(0, {});
  EXPECT_EQ(SUCCESS, acc.LinkOutput(dst_list, node, UPDATE_NONE));
  auto input = node->GetOpDesc()->MutableInputDesc(0);

  // Update input desc
  vector<int64_t> dims = {};
  vector<int64_t> dims_new = {1, 4, 64, 64};
  EXPECT_EQ(SUCCESS, acc.UpdateInputByPeer(node, 0, relu, 0));
  EXPECT_EQ(input->GetShape().GetDims(), dims_new);
  EXPECT_EQ(input->GetFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(input->GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(input->GetOriginFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(input->GetOriginDataType(), ge::DT_FLOAT);

  input->SetShape(ge::GeShape(dims));
  input->SetFormat(ge::FORMAT_ND);

  acc.UpdateInputByPeer(node, 0, relu, 0);
  EXPECT_EQ(input->GetShape().GetDims(), dims_new);
  EXPECT_EQ(input->GetFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(input->GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(input->GetOriginFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(input->GetOriginDataType(), ge::DT_FLOAT);

  input->SetShape(ge::GeShape(dims));
  input->SetFormat(ge::FORMAT_ND);

  acc.UpdateInputByPeer(node, 0 , relu, 0);
  EXPECT_EQ(input->GetShape().GetDims(), dims_new);
  EXPECT_EQ(input->GetFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(input->GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(input->GetOriginFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(input->GetOriginDataType(), ge::DT_FLOAT);

  input->SetShape(ge::GeShape(dims));
  input->SetFormat(ge::FORMAT_ND);
}

TEST_F(UTestFusionTurbo, test_case_12) {
  auto graph = CreateGraphSingleInAndOut();
  FusionTurbo acc(graph);
  string name = "transpose";
  string type = "Transpose";
  auto node = acc.AddNodeOnly(name, type);
  ASSERT_NE(node, nullptr);

  auto relu = GetNode(graph, "relu");
  Relations src_list;
  NodeIndex pi = {relu, 0};
  src_list.Add(0, pi);
  src_list.Add(0, pi);
  acc.LinkInput(src_list, node, UPDATE_NONE);

  auto cast2 = GetNode(graph, "cast2");
  Relations dst_list;
  dst_list.Add(0, {{cast2, 0}});
  acc.LinkOutput(dst_list, node, UPDATE_NONE);

  auto output = node->GetOpDesc()->MutableOutputDesc(0);
  auto cast1 = GetNode(graph, "cast1");
  // Update output desc
  vector<int64_t> dims = {};
  vector<int64_t> dims_new = {8, 4, 16, 16};
  EXPECT_EQ(SUCCESS, acc.UpdateOutputByPeer(node, 0, cast1, 0));
  EXPECT_EQ(output->GetShape().GetDims(), dims_new);
  EXPECT_EQ(output->GetFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(output->GetOriginFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(output->GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(output->GetOriginDataType(), ge::DT_FLOAT);
  output->SetShape(ge::GeShape(dims));
  output->SetFormat(ge::FORMAT_ND);

  acc.UpdateOutputByPeer(node, 0, cast1, 0);
  EXPECT_EQ(output->GetShape().GetDims(), dims_new);
  EXPECT_EQ(output->GetFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(output->GetOriginFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(output->GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(output->GetOriginDataType(), ge::DT_FLOAT);

  output->SetShape(ge::GeShape(dims));
  output->SetFormat(ge::FORMAT_ND);

  acc.UpdateOutputByPeer(node, 0, cast1, 0);
  EXPECT_EQ(output->GetShape().GetDims(), dims_new);
  EXPECT_EQ(output->GetFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(output->GetOriginFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(output->GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(output->GetOriginDataType(), ge::DT_FLOAT);

  output->SetShape(ge::GeShape(dims));
  output->SetFormat(ge::FORMAT_ND);
}

TEST_F(UTestFusionTurbo, test_case_13) {
  auto graph = CreateGraphSingleInAndOut();
  FusionTurbo acc(graph);
  string name = "transpose";
  string type = "Transpose";
  auto node = acc.AddNodeOnly(name, type);
  ASSERT_NE(node, nullptr);

  auto relu = GetNode(graph, "relu");
  Relations src_list;
  src_list.Add(0, {relu, 0});
  src_list.Add(0, {relu, 0});
  acc.LinkInput(src_list, node, UPDATE_THIS);

  auto cast2 = GetNode(graph, "cast2");
  Relations dst_list;
  NodeIndex pi = {cast2, 0};
  dst_list.Add(0, pi);
  dst_list.Add(0, pi);
  acc.BreakInput(cast2, {0});
  Status ret = acc.LinkOutput(dst_list, node, UPDATE_THIS);
  EXPECT_EQ(ret, SUCCESS);

  unique_ptr<int32_t[]> value(new(std::nothrow) int32_t[24]);
  auto data_ptr = (uint8_t *) (value.get());
  for (size_t i = 0; i < 96; i++) {
    data_ptr[i] = i;
  }
  WeightInfo w = {ge::GeShape({1, 2, 3, 4}), ge::DT_INT32, ge::FORMAT_NCHW, value.get()};
  /* coverage code */
  auto shape = ge::GeShape({1, 2, 3, 4});
  WeightInfo w1 = {shape, ge::DT_INT32, ge::FORMAT_NCHW, value.get()};
  WeightInfo w2 = {ge::GeShape({1, 2, 3, 4}), ge::GeShape({1, 2, 3, 4}),
                   ge::DT_INT32, ge::DT_INT32, ge::FORMAT_NCHW,  ge::FORMAT_NCHW,
                   value.get()};

  ASSERT_NE(nullptr, acc.AddWeight(node, w));
  ASSERT_EQ(node->GetAllInDataAnchorsSize(), 3);
  EXPECT_EQ(node->GetInDataAnchor(2)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), "Const");
  auto new_weight = FusionTurbo::MutableWeight(node, 2);
  ASSERT_NE(nullptr, new_weight);

  EXPECT_EQ(new_weight->GetTensorDesc().GetShape(), ge::GeShape({1, 2, 3, 4}));
  EXPECT_EQ(new_weight->GetTensorDesc().GetOriginShape(), ge::GeShape({1, 2, 3, 4}));
  EXPECT_EQ(new_weight->GetTensorDesc().GetDataType(), ge::DT_INT32);
  EXPECT_EQ(new_weight->GetTensorDesc().GetOriginDataType(), ge::DT_INT32);
  EXPECT_EQ(new_weight->GetTensorDesc().GetFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(new_weight->GetTensorDesc().GetOriginFormat(), ge::FORMAT_NCHW);
  const uint8_t *data = new_weight->GetData().GetData();
  auto data_size = new_weight->GetData().size();
  EXPECT_EQ(data_size, 96);
  for (size_t i = 0; i < 96; i++) {
    EXPECT_EQ(data[i], i);
  }
}


TEST_F(UTestFusionTurbo, test_case_13_1) {
  auto graph = CreateGraphSingleInAndOut();
  FusionTurbo acc(graph);
  string name = "transpose";
  string type = "Transpose";
  auto node = acc.AddNodeOnly(name, type);
  ASSERT_NE(node, nullptr);

  auto relu = GetNode(graph, "relu");
  Relations src_list = {{relu, 0}};
  acc.LinkInput(src_list, node, UPDATE_THIS);

  auto cast2 = GetNode(graph, "cast2");
  Relations dst_list = {{cast2, 0}};
  acc.BreakInput(cast2, {0});
  Status ret = acc.LinkOutput(dst_list, node, UPDATE_THIS);
  EXPECT_EQ(ret, SUCCESS);

  unique_ptr<int32_t[]> value(new(std::nothrow) int32_t[24]);
  auto data_ptr = (uint8_t *) (value.get());
  for (size_t i = 0; i < 96; i++) {
    data_ptr[i] = i;
  }
  WeightInfo w = {ge::GeShape({1, 2, 3, 4}), ge::DT_INT32, ge::FORMAT_NCHW, value.get()};

  ASSERT_NE(nullptr, acc.AddWeight(node, 3, w));
  ASSERT_EQ(node->GetAllInDataAnchorsSize(), 3);
  EXPECT_EQ(node->GetInDataAnchor(2)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), "Const");
  auto new_weight = FusionTurbo::MutableWeight(node, 2);
  ASSERT_NE(nullptr, new_weight);

  EXPECT_EQ(new_weight->GetTensorDesc().GetShape(), ge::GeShape({1, 2, 3, 4}));
  EXPECT_EQ(new_weight->GetTensorDesc().GetOriginShape(), ge::GeShape({1, 2, 3, 4}));
  EXPECT_EQ(new_weight->GetTensorDesc().GetDataType(), ge::DT_INT32);
  EXPECT_EQ(new_weight->GetTensorDesc().GetOriginDataType(), ge::DT_INT32);
  EXPECT_EQ(new_weight->GetTensorDesc().GetFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(new_weight->GetTensorDesc().GetOriginFormat(), ge::FORMAT_NCHW);
  const uint8_t *data = new_weight->GetData().GetData();
  auto data_size = new_weight->GetData().size();
  EXPECT_EQ(data_size, 96);
  for (size_t i = 0; i < 96; i++) {
    EXPECT_EQ(data[i], i);
  }
}

TEST_F(UTestFusionTurbo, test_case_14) {
  auto graph = CreateGraphSingleInAndOut();
  FusionTurbo acc(graph);
  string name = "transpose";
  string type = "Transpose";
  auto node = acc.AddNodeOnly(name, type);
  ASSERT_NE(node, nullptr);

  unique_ptr<int32_t[]> value(new(std::nothrow) int32_t[24]);
  auto data_ptr = (uint8_t *) (value.get());
  for (size_t i = 0; i < 96; i++) {
    data_ptr[i] = i;
  }

  auto input1 = node->GetOpDesc()->MutableInputDesc(1);
  input1->SetDataType(ge::DT_INT32);
  input1->SetOriginDataType(ge::DT_INT16);
  input1->SetFormat(ge::FORMAT_NCHW);
  input1->SetOriginFormat(ge::FORMAT_NHWC);
  input1->SetShape(ge::GeShape({1, 2, 3, 4}));
  input1->SetOriginShape(ge::GeShape({1, 3, 4, 2}));

  WeightInfo w(node, 1, value.get());

  ASSERT_NE(nullptr, acc.AddWeight(node, 1, w));
  ASSERT_EQ(node->GetAllInDataAnchorsSize(), 2);
  EXPECT_EQ(node->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), "Const");
  auto new_weight = FusionTurbo::MutableWeight(node, 1);
  ASSERT_NE(nullptr, new_weight);

  EXPECT_EQ(new_weight->GetTensorDesc().GetShape(), ge::GeShape({1, 2, 3, 4}));
  EXPECT_EQ(new_weight->GetTensorDesc().GetOriginShape(), ge::GeShape({1, 3, 4, 2}));
  EXPECT_EQ(new_weight->GetTensorDesc().GetDataType(), ge::DT_INT32);
  EXPECT_EQ(new_weight->GetTensorDesc().GetOriginDataType(), ge::DT_INT16);
  EXPECT_EQ(new_weight->GetTensorDesc().GetFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(new_weight->GetTensorDesc().GetOriginFormat(), ge::FORMAT_NHWC);
  const uint8_t *data = new_weight->GetData().GetData();
  auto data_size = new_weight->GetData().size();
  EXPECT_EQ(data_size, 96);
  for (size_t i = 0; i < 96; i++) {
    EXPECT_EQ(data[i], i);
  }

  w.shape = ge::GeShape({5, 2, 7, 1});
  w.ori_shape = ge::GeShape({5, 1, 2, 7});
  w.datatype = ge::DT_FLOAT16;
  w.ori_datatype = ge::DT_FLOAT;
  w.format = ge::FORMAT_NHWC;
  w.ori_format = ge::FORMAT_NCHW;

  w.total_data_size = 140;
  unique_ptr<int32_t[]> value1(new(std::nothrow) int32_t[140]);
  auto data_ptr1 = (uint8_t *) (value1.get());
  for (size_t i = 0; i < 140; i++) {
    data_ptr1[i] = i + 1;
  }
  w.data = (uint8_t *) value1.get();
  /* Update const value and tensor when const node and weight both exist. */
  ASSERT_NE(nullptr, acc.AddWeight(node, 1, w));
  ASSERT_EQ(node->GetAllInDataAnchorsSize(), 2);
  EXPECT_EQ(node->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), "Const");
  new_weight = FusionTurbo::MutableWeight(node, 1);
  ASSERT_NE(nullptr, new_weight);

  EXPECT_EQ(new_weight->GetTensorDesc().GetShape(), ge::GeShape({5, 2, 7, 1}));
  EXPECT_EQ(new_weight->GetTensorDesc().GetOriginShape(), ge::GeShape({5, 1, 2, 7}));
  EXPECT_EQ(new_weight->GetTensorDesc().GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(new_weight->GetTensorDesc().GetOriginDataType(), ge::DT_FLOAT);
  EXPECT_EQ(new_weight->GetTensorDesc().GetFormat(), ge::FORMAT_NHWC);
  EXPECT_EQ(new_weight->GetTensorDesc().GetOriginFormat(), ge::FORMAT_NCHW);
  data = new_weight->GetData().GetData();
  data_size = new_weight->GetData().size();
  EXPECT_EQ(data_size, 140);
  for (size_t i = 0; i < 140; i++) {
    EXPECT_EQ(data[i], i + 1);
  }
}

TEST_F(UTestFusionTurbo, test_case_14_1) {
  auto graph = CreateGraphSingleInAndOut();
  FusionTurbo acc(graph);
  string name = "transpose";
  string type = "Transpose";
  auto node = acc.AddNodeOnly(name, type);
  ASSERT_NE(node, nullptr);

  unique_ptr<int32_t[]> value(new(std::nothrow) int32_t[24]);
  auto data_ptr = (uint8_t *) (value.get());
  for (size_t i = 0; i < 96; i++) {
    data_ptr[i] = i;
  }

  auto input1 = node->GetOpDesc()->MutableInputDesc(1);
  input1->SetDataType(ge::DT_INT32);
  input1->SetOriginDataType(ge::DT_INT16);
  input1->SetFormat(ge::FORMAT_NCHW);
  input1->SetOriginFormat(ge::FORMAT_NHWC);
  input1->SetShape(ge::GeShape({1, 2, 3, 4}));
  input1->SetOriginShape(ge::GeShape({1, 3, 4, 2}));

  WeightInfo w(node, 1, value.get());

  ASSERT_NE(nullptr, acc.AddWeight(node, "shape", w));
  ASSERT_EQ(node->GetAllInDataAnchorsSize(), 2);
  EXPECT_EQ(node->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), "Const");
  auto new_weight = FusionTurbo::MutableWeight(node, 1);
  ASSERT_NE(nullptr, new_weight);

  EXPECT_EQ(new_weight->GetTensorDesc().GetShape(), ge::GeShape({1, 2, 3, 4}));
  EXPECT_EQ(new_weight->GetTensorDesc().GetOriginShape(), ge::GeShape({1, 3, 4, 2}));
  EXPECT_EQ(new_weight->GetTensorDesc().GetDataType(), ge::DT_INT32);
  EXPECT_EQ(new_weight->GetTensorDesc().GetOriginDataType(), ge::DT_INT16);
  EXPECT_EQ(new_weight->GetTensorDesc().GetFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(new_weight->GetTensorDesc().GetOriginFormat(), ge::FORMAT_NHWC);
  const uint8_t *data = new_weight->GetData().GetData();
  auto data_size = new_weight->GetData().size();
  EXPECT_EQ(data_size, 96);
  for (size_t i = 0; i < 96; i++) {
    EXPECT_EQ(data[i], i);
  }

  w.shape = ge::GeShape({5, 2, 7, 1});
  w.ori_shape = ge::GeShape({5, 1, 2, 7});
  w.datatype = ge::DT_FLOAT16;
  w.ori_datatype = ge::DT_FLOAT;
  w.format = ge::FORMAT_NHWC;
  w.ori_format = ge::FORMAT_NCHW;

  w.total_data_size = 140;
  unique_ptr<int32_t[]> value1(new(std::nothrow) int32_t[140]);
  auto data_ptr1 = (uint8_t *) (value1.get());
  for (size_t i = 0; i < 140; i++) {
    data_ptr1[i] = i + 1;
  }
  w.data = (uint8_t *) value1.get();
  /* Update const value and tensor when const node and weight both exist. */
  ASSERT_NE(nullptr, acc.AddWeight(node, "shape", w));
  ASSERT_EQ(node->GetAllInDataAnchorsSize(), 2);
  EXPECT_EQ(node->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), "Const");
  new_weight = FusionTurbo::MutableWeight(node, 1);
  ASSERT_NE(nullptr, new_weight);

  EXPECT_EQ(new_weight->GetTensorDesc().GetShape(), ge::GeShape({5, 2, 7, 1}));
  EXPECT_EQ(new_weight->GetTensorDesc().GetOriginShape(), ge::GeShape({5, 1, 2, 7}));
  EXPECT_EQ(new_weight->GetTensorDesc().GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(new_weight->GetTensorDesc().GetOriginDataType(), ge::DT_FLOAT);
  EXPECT_EQ(new_weight->GetTensorDesc().GetFormat(), ge::FORMAT_NHWC);
  EXPECT_EQ(new_weight->GetTensorDesc().GetOriginFormat(), ge::FORMAT_NCHW);
  data = new_weight->GetData().GetData();
  data_size = new_weight->GetData().size();
  EXPECT_EQ(data_size, 140);
  for (size_t i = 0; i < 140; i++) {
    EXPECT_EQ(data[i], i + 1);
  }
}


TEST_F(UTestFusionTurbo, test_case_14_1_1) {
  auto graph = CreateGraphSingleInAndOut();
  FusionTurbo acc(graph);
  string name = "transpose";
  string type = "Transpose";
  auto node = acc.AddNodeOnly(name, type);
  ASSERT_NE(node, nullptr);

  unique_ptr<int32_t[]> value(new(std::nothrow) int32_t[24]);
  auto data_ptr = (uint8_t *) (value.get());
  for (size_t i = 0; i < 96; i++) {
    data_ptr[i] = i;
  }

  auto input1 = node->GetOpDesc()->MutableInputDesc(1);
  input1->SetDataType(ge::DT_INT32);
  input1->SetOriginDataType(ge::DT_INT16);
  input1->SetFormat(ge::FORMAT_NCHW);
  input1->SetOriginFormat(ge::FORMAT_NHWC);
  input1->SetShape(ge::GeShape({1, 2, 3, 4}));
  input1->SetOriginShape(ge::GeShape({1, 3, 4, 2}));

  WeightInfo w(node, 1, value.get());

  ASSERT_NE(nullptr, acc.AddWeight(node, "shape", w));
  ASSERT_EQ(node->GetAllInDataAnchorsSize(), 2);
  EXPECT_EQ(node->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), "Const");
  auto new_weight = FusionTurbo::MutableWeight(node, 1);
  ASSERT_NE(nullptr, new_weight);

  EXPECT_EQ(new_weight->GetTensorDesc().GetShape(), ge::GeShape({1, 2, 3, 4}));
  EXPECT_EQ(new_weight->GetTensorDesc().GetOriginShape(), ge::GeShape({1, 3, 4, 2}));
  EXPECT_EQ(new_weight->GetTensorDesc().GetDataType(), ge::DT_INT32);
  EXPECT_EQ(new_weight->GetTensorDesc().GetOriginDataType(), ge::DT_INT16);
  EXPECT_EQ(new_weight->GetTensorDesc().GetFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(new_weight->GetTensorDesc().GetOriginFormat(), ge::FORMAT_NHWC);
  const uint8_t *data = new_weight->GetData().GetData();
  auto data_size = new_weight->GetData().size();
  EXPECT_EQ(data_size, 96);
  for (size_t i = 0; i < 96; i++) {
    EXPECT_EQ(data[i], i);
  }

  w.shape = ge::GeShape({5, 2, 7, 1});
  w.ori_shape = ge::GeShape({5, 1, 2, 7});
  w.datatype = ge::DT_FLOAT16;
  w.ori_datatype = ge::DT_FLOAT;
  w.format = ge::FORMAT_NHWC;
  w.ori_format = ge::FORMAT_NCHW;

  w.total_data_size = 140;
  unique_ptr<int32_t[]> value1(new(std::nothrow) int32_t[140]);
  auto data_ptr1 = (uint8_t *) (value1.get());
  for (size_t i = 0; i < 140; i++) {
    data_ptr1[i] = i + 1;
  }
  w.data = (uint8_t *) value1.get();
  /* Update const value and tensor when const node and weight both exist. */
  ASSERT_EQ(nullptr, acc.AddWeight(node, "xxxx", w));
}

TEST_F(UTestFusionTurbo, test_case_14_2) {
  auto graph = CreateGraphSingleInAndOut();
  FusionTurbo acc(graph);
  string name = "transpose";
  string type = "Transpose";
  auto node = acc.AddNodeOnly(name, type);
  ASSERT_NE(node, nullptr);

  unique_ptr<int32_t[]> value(new(std::nothrow) int32_t[24]);
  auto data_ptr = (uint8_t *) (value.get());
  for (size_t i = 0; i < 96; i++) {
    data_ptr[i] = i;
  }
  auto const_node = acc.InsertNodeBefore("const_1", "Const", node, 1);
  ASSERT_NE(nullptr, const_node);
  auto const_out_desc = const_node->GetOpDesc()->MutableOutputDesc(0);
  EXPECT_EQ(const_out_desc->GetShape(), ge::GeShape());
  EXPECT_EQ(const_out_desc->GetOriginShape(), ge::GeShape());
  EXPECT_EQ(const_out_desc->GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(const_out_desc->GetOriginDataType(), ge::DT_UNDEFINED);
  EXPECT_EQ(const_out_desc->GetFormat(), ge::FORMAT_ND);
  EXPECT_EQ(const_out_desc->GetOriginFormat(), ge::FORMAT_ND);

  auto weight = FusionTurbo::MutableWeight(node, 1);
  ASSERT_NE(nullptr, weight);
  auto &weight_tensor = weight->GetTensorDesc();
  EXPECT_EQ(weight_tensor.GetShape(), ge::GeShape());
  EXPECT_EQ(weight_tensor.GetOriginShape(), ge::GeShape());
  EXPECT_EQ(weight_tensor.GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(weight_tensor.GetOriginDataType(), ge::DT_UNDEFINED);
  EXPECT_EQ(weight_tensor.GetFormat(), ge::FORMAT_ND);
  EXPECT_EQ(weight_tensor.GetOriginFormat(), ge::FORMAT_ND);
  const uint8_t *data = weight->GetData().GetData();
  auto data_size = weight->GetData().size();
  EXPECT_EQ(data_size, 0);
  EXPECT_NE(data, nullptr);


  auto input1 = node->GetOpDesc()->MutableInputDesc(1);
  input1->SetDataType(ge::DT_INT32);
  input1->SetOriginDataType(ge::DT_INT16);
  input1->SetFormat(ge::FORMAT_NCHW);
  input1->SetOriginFormat(ge::FORMAT_NHWC);
  input1->SetShape(ge::GeShape({1, 2, 3, 4}));
  input1->SetOriginShape(ge::GeShape({1, 3, 4, 2}));

  WeightInfo w(node, 1, value.get());
  ASSERT_NE(nullptr, acc.AddWeight(node, 1, w));
  ASSERT_EQ(node->GetAllInDataAnchorsSize(), 2);
  EXPECT_EQ(node->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), "Const");
  auto new_weight = FusionTurbo::MutableWeight(node, 1);
  ASSERT_NE(nullptr, new_weight);

  auto &new_weight_tensor = new_weight->GetTensorDesc();
  EXPECT_EQ(new_weight_tensor.GetShape(), ge::GeShape({1, 2, 3, 4}));
  EXPECT_EQ(new_weight_tensor.GetOriginShape(), ge::GeShape({1, 3, 4, 2}));
  EXPECT_EQ(new_weight_tensor.GetDataType(), ge::DT_INT32);
  EXPECT_EQ(new_weight_tensor.GetOriginDataType(), ge::DT_INT16);
  EXPECT_EQ(new_weight_tensor.GetFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(new_weight_tensor.GetOriginFormat(), ge::FORMAT_NHWC);
  data = new_weight->GetData().GetData();
  data_size = new_weight->GetData().size();
  EXPECT_EQ(data_size, 96);
  for (size_t i = 0; i < 96; i++) {
    EXPECT_EQ(data[i], i);
  }
}

TEST_F(UTestFusionTurbo, test_case_15) {
  auto graph = CreateGraphSingleInAndOut();
  FusionTurbo acc(graph);
  string name = "transpose";
  string type = "Transpose";
  auto node = acc.AddNodeOnly(name, type);
  ASSERT_NE(node, nullptr);

  auto relu = GetNode(graph, "relu");
  Relations src_list = {{relu, 0}};
  acc.LinkInput(src_list, node, UPDATE_THIS);

  auto cast2 = GetNode(graph, "cast2");
  Relations dst_list = {{cast2, 0}};
  acc.BreakInput(cast2, {0});
  Status ret = acc.LinkOutput(dst_list, node, UPDATE_THIS);
  EXPECT_EQ(ret, SUCCESS);

  unique_ptr<int32_t[]> value(new(std::nothrow) int32_t[24]);
  auto data_ptr = (uint8_t *) (value.get());
  for (size_t i = 0; i < 96; i++) {
    data_ptr[i] = i;
  }
  WeightInfo w = {ge::GeShape({1, 2, 3, 4}), ge::DT_INT32, ge::FORMAT_NCHW, value.get()};
  WeightInfo w2 = {ge::GeShape({1, 3, 2, 4}), ge::DT_INT32, ge::FORMAT_NCHW, value.get()};
  WeightInfo w3 = {ge::GeShape({4, 1, 3, 2}), ge::DT_INT32, ge::FORMAT_NCHW, value.get()};
  std::vector<WeightInfo> weight_all = {std::move(w), std::move(w2), std::move(w3)};

  auto const_nodes = acc.AddWeights(node, weight_all);
  ASSERT_EQ(const_nodes.size(), 3);

  ASSERT_EQ(node->GetAllInDataAnchorsSize(), 5);
  EXPECT_EQ(node->GetInDataAnchor(2)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), "Const");
  EXPECT_EQ(node->GetInDataAnchor(3)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), "Const");
  EXPECT_EQ(node->GetInDataAnchor(4)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), "Const");

  for (size_t i = 2; i < 5; i++) {
    auto new_weight = FusionTurbo::MutableWeight(node, i);
    ASSERT_NE(nullptr, new_weight);
    if (i == 2) {
      EXPECT_EQ(new_weight->GetTensorDesc().GetShape(), ge::GeShape({1, 2, 3, 4}));
    } else if (i == 3) {
      EXPECT_EQ(new_weight->GetTensorDesc().GetShape(), ge::GeShape({1, 3, 2, 4}));
    } else {
      EXPECT_EQ(new_weight->GetTensorDesc().GetShape(), ge::GeShape({4, 1, 3, 2}));
    }

    EXPECT_EQ(new_weight->GetTensorDesc().GetDataType(), ge::DT_INT32);
    EXPECT_EQ(new_weight->GetTensorDesc().GetOriginDataType(), ge::DT_INT32);
    EXPECT_EQ(new_weight->GetTensorDesc().GetFormat(), ge::FORMAT_NCHW);
    EXPECT_EQ(new_weight->GetTensorDesc().GetOriginFormat(), ge::FORMAT_NCHW);
    const uint8_t *data = new_weight->GetData().GetData();
    auto data_size = new_weight->GetData().size();
    EXPECT_EQ(data_size, 96);
    for (size_t j = 0; j < 96; j++) {
      EXPECT_EQ(data[j], j);
    }
  }
}


TEST_F(UTestFusionTurbo, test_case_15_1) {
  auto graph = CreateGraphSingleInAndOut();
  FusionTurbo acc(graph);
  string name = "transpose";
  string type = "Transpose";
  auto node = acc.AddNodeOnly(name, type);
  ASSERT_NE(node, nullptr);

  auto relu = GetNode(graph, "relu");
  Relations src_list = {{relu, 0}};
  acc.LinkInput(src_list, node, UPDATE_THIS);

  auto cast2 = GetNode(graph, "cast2");
  Relations dst_list = {{cast2, 0}};
  acc.BreakInput(cast2, {0});
  Status ret = acc.LinkOutput(dst_list, node, UPDATE_THIS);
  EXPECT_EQ(ret, SUCCESS);

  unique_ptr<int32_t[]> value(new(std::nothrow) int32_t[24]);
  auto data_ptr = (uint8_t *) (value.get());
  for (size_t i = 0; i < 96; i++) {
    data_ptr[i] = i;
  }
  WeightInfo w = {ge::GeShape({1, 2, 3, 4}), ge::DT_INT32, ge::FORMAT_NCHW, value.get()};
  WeightInfo w2 = {ge::GeShape({1, 3, 2, 4}), ge::DT_INT32, ge::FORMAT_NCHW, value.get()};
  WeightInfo w3 = {ge::GeShape({4, 1, 3, 2}), ge::DT_INT32, ge::FORMAT_NCHW, value.get()};
  acc.AddWeight(node, 1, w);
  std::vector<WeightInfo> weight_all = {std::move(w), std::move(w2), std::move(w3)};

  auto const_nodes = acc.AddWeights(node, weight_all);
  ASSERT_EQ(const_nodes.size(), 3);

  ASSERT_EQ(node->GetAllInDataAnchorsSize(), 5);
  EXPECT_EQ(node->GetInDataAnchor(2)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), "Const");
  EXPECT_EQ(node->GetInDataAnchor(3)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), "Const");
  EXPECT_EQ(node->GetInDataAnchor(4)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), "Const");
  auto all_weights = ge::OpDescUtils::MutableWeights(node);
  ASSERT_EQ(all_weights.size(), 4);

  for (size_t i = 1; i < 5; i++) {
    auto new_weight = FusionTurbo::MutableWeight(node, i);
    ASSERT_NE(nullptr, new_weight);
    if (i == 1 || i == 2) {
      EXPECT_EQ(new_weight->GetTensorDesc().GetShape(), ge::GeShape({1, 2, 3, 4}));
    } else if (i == 3) {
      EXPECT_EQ(new_weight->GetTensorDesc().GetShape(), ge::GeShape({1, 3, 2, 4}));
    } else {
      EXPECT_EQ(new_weight->GetTensorDesc().GetShape(), ge::GeShape({4, 1, 3, 2}));
    }

    EXPECT_EQ(new_weight->GetTensorDesc().GetDataType(), ge::DT_INT32);
    EXPECT_EQ(new_weight->GetTensorDesc().GetOriginDataType(), ge::DT_INT32);
    EXPECT_EQ(new_weight->GetTensorDesc().GetFormat(), ge::FORMAT_NCHW);
    EXPECT_EQ(new_weight->GetTensorDesc().GetOriginFormat(), ge::FORMAT_NCHW);
    const uint8_t *data = new_weight->GetData().GetData();
    auto data_size = new_weight->GetData().size();
    EXPECT_EQ(data_size, 96);
    for (size_t j = 0; j < 96; j++) {
      EXPECT_EQ(data[j], j);
    }
  }
}

TEST_F(UTestFusionTurbo, test_case_15_3) {
  auto graph = CreateGraphSingleInAndOut();
  FusionTurbo acc(graph);
  string name = "transpose";
  string type = "Transpose";
  auto node = acc.AddNodeOnly(name, type);
  ASSERT_NE(node, nullptr);

  auto relu = GetNode(graph, "relu");
  Relations src_list = {{relu, 0}};
  acc.LinkInput(src_list, node, UPDATE_THIS);

  auto cast2 = GetNode(graph, "cast2");
  Relations dst_list = {{cast2, 0}};
  acc.BreakInput(cast2, {0});
  Status ret = acc.LinkOutput(dst_list, node, UPDATE_THIS);
  EXPECT_EQ(ret, SUCCESS);

  unique_ptr<int32_t[]> value(new(std::nothrow) int32_t[24]);
  auto data_ptr = (uint8_t *) (value.get());
  for (size_t i = 0; i < 96; i++) {
    data_ptr[i] = i;
  }

  auto node_input_1 = node->GetOpDesc()->MutableInputDesc(1);
  auto weight_shape = ge::GeShape({1, 2, 3, 4});
  node_input_1->SetOriginShape(weight_shape);
  node_input_1->SetShape(weight_shape);
  node_input_1->SetDataType(ge::DT_INT32);
  node_input_1->SetOriginDataType(ge::DT_INT32);
  node_input_1->SetFormat(ge::FORMAT_NCHW);
  node_input_1->SetOriginFormat(ge::FORMAT_NCHW);

  WeightInfo w = {*node_input_1, value.get()};
  WeightInfo w2 = {ge::GeShape({1, 3, 2, 4}), ge::GeShape({1, 3, 2, 4}),
                   ge::DT_INT32, ge::DT_INT32, ge::FORMAT_NCHW, ge::FORMAT_NCHW, value.get()};
  WeightInfo w3 = {ge::GeShape({4, 1, 3, 2}), ge::DT_INT32, ge::FORMAT_NCHW, value.get()};
  acc.AddWeight(node, 1, w);
  std::vector<WeightInfo> weight_all = {std::move(w), std::move(w2), std::move(w3)};

  auto const_nodes = acc.AddWeights(node, weight_all);
  ASSERT_EQ(const_nodes.size(), 3);

  ASSERT_EQ(node->GetAllInDataAnchorsSize(), 5);
  EXPECT_EQ(node->GetInDataAnchor(2)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), "Const");
  EXPECT_EQ(node->GetInDataAnchor(3)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), "Const");
  EXPECT_EQ(node->GetInDataAnchor(4)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), "Const");
  auto all_weights = ge::OpDescUtils::MutableWeights(node);
  ASSERT_EQ(all_weights.size(), 4);

  for (size_t i = 1; i < 5; i++) {
    auto new_weight = FusionTurbo::MutableWeight(node, i);
    ASSERT_NE(nullptr, new_weight);
    if (i == 1 || i == 2) {
      EXPECT_EQ(new_weight->GetTensorDesc().GetShape(), ge::GeShape({1, 2, 3, 4}));
    } else if (i == 3) {
      EXPECT_EQ(new_weight->GetTensorDesc().GetShape(), ge::GeShape({1, 3, 2, 4}));
    } else {
      EXPECT_EQ(new_weight->GetTensorDesc().GetShape(), ge::GeShape({4, 1, 3, 2}));
    }

    EXPECT_EQ(new_weight->GetTensorDesc().GetDataType(), ge::DT_INT32);
    EXPECT_EQ(new_weight->GetTensorDesc().GetOriginDataType(), ge::DT_INT32);
    EXPECT_EQ(new_weight->GetTensorDesc().GetFormat(), ge::FORMAT_NCHW);
    EXPECT_EQ(new_weight->GetTensorDesc().GetOriginFormat(), ge::FORMAT_NCHW);
    const uint8_t *data = new_weight->GetData().GetData();
    auto data_size = new_weight->GetData().size();
    EXPECT_EQ(data_size, 96);
    for (size_t j = 0; j < 96; j++) {
      EXPECT_EQ(data[j], j);
    }
  }
}

TEST_F(UTestFusionTurbo, test_case_15_4) {
  auto graph = CreateGraphSingleInAndOut();
  FusionTurbo acc(graph);
  string name = "transpose";
  string type = "Transpose";
  auto node = acc.AddNodeOnly(name, type);
  ASSERT_NE(node, nullptr);

  auto relu = GetNode(graph, "relu");
  Relations src_list = {{relu, 0}};
  acc.LinkInput(src_list, node, UPDATE_THIS);

  auto cast2 = GetNode(graph, "cast2");
  Relations dst_list = {{cast2, 0}};
  acc.BreakInput(cast2, {0});
  Status ret = acc.LinkOutput(dst_list, node, UPDATE_THIS);
  EXPECT_EQ(ret, SUCCESS);

  unique_ptr<int32_t[]> value(new(std::nothrow) int32_t[24]);
  auto data_ptr = (uint8_t *) (value.get());
  for (size_t i = 0; i < 96; i++) {
    data_ptr[i] = i;
  }

  auto node_input_1 = node->GetOpDesc()->MutableInputDesc(1);
  auto weight_shape = ge::GeShape({1, 2, 3, 4});
  node_input_1->SetOriginShape(weight_shape);
  node_input_1->SetShape(weight_shape);
  node_input_1->SetDataType(ge::DT_INT32);
  node_input_1->SetOriginDataType(ge::DT_INT32);
  node_input_1->SetFormat(ge::FORMAT_NCHW);
  node_input_1->SetOriginFormat(ge::FORMAT_NCHW);

  WeightInfo w = {*node_input_1, value.get()};
  WeightInfo w2 = {ge::GeShape({1, 3, 2, 4}), ge::GeShape({1, 3, 2, 4}),
                   ge::DT_INT32, ge::DT_INT32, ge::FORMAT_NCHW, ge::FORMAT_NCHW, value.get()};
  WeightInfo w3 = {ge::GeShape({4, 1, 3, 2}), ge::DT_INT32, ge::FORMAT_NCHW, value.get()};
  acc.AddWeight(node, 1, w);
  std::vector<WeightInfo> weight_all = {std::move(w), std::move(w2), std::move(w3)};

  auto const_nodes = acc.AddWeights(node, weight_all);
  ASSERT_EQ(const_nodes.size(), 3);

  ASSERT_EQ(node->GetAllInDataAnchorsSize(), 5);
  auto const_2 = FusionTurbo::GetPeerOutNode(node, 2);
  auto const_2_peer_in = FusionTurbo::GetPeerInNodes(const_2, 0);
  ASSERT_EQ(const_2_peer_in.size(), 1);
  auto node_temp = const_2_peer_in.at(0);

  EXPECT_EQ(node_temp, node);
  EXPECT_EQ(FusionTurbo::CheckConnected(const_2, node), true);
  EXPECT_EQ(FusionTurbo::CheckConnected(const_2, node, 0), true);
  EXPECT_EQ(const_2->GetType(), "Const");
  EXPECT_EQ(FusionTurbo::GetPeerOutNode(node, 3)->GetType(), "Const");
  EXPECT_EQ(FusionTurbo::GetPeerOutNode(node, 4)->GetType(), "Const");
  auto all_weights = ge::OpDescUtils::MutableWeights(node);
  ASSERT_EQ(all_weights.size(), 4);

  for (size_t i = 1; i < 5; i++) {
    auto new_weight = FusionTurbo::MutableWeight(node, i);
    ASSERT_NE(nullptr, new_weight);
    if (i == 1 || i == 2) {
      EXPECT_EQ(new_weight->GetTensorDesc().GetShape(), ge::GeShape({1, 2, 3, 4}));
    } else if (i == 3) {
      EXPECT_EQ(new_weight->GetTensorDesc().GetShape(), ge::GeShape({1, 3, 2, 4}));
    } else {
      EXPECT_EQ(new_weight->GetTensorDesc().GetShape(), ge::GeShape({4, 1, 3, 2}));
    }

    EXPECT_EQ(new_weight->GetTensorDesc().GetDataType(), ge::DT_INT32);
    EXPECT_EQ(new_weight->GetTensorDesc().GetOriginDataType(), ge::DT_INT32);
    EXPECT_EQ(new_weight->GetTensorDesc().GetFormat(), ge::FORMAT_NCHW);
    EXPECT_EQ(new_weight->GetTensorDesc().GetOriginFormat(), ge::FORMAT_NCHW);
    const uint8_t *data = new_weight->GetData().GetData();
    auto data_size = new_weight->GetData().size();
    EXPECT_EQ(data_size, 96);
    for (size_t j = 0; j < 96; j++) {
      EXPECT_EQ(data[j], j);
    }
  }
}

TEST_F(UTestFusionTurbo, test_case_16_1) {
  /*
   *  input                                       data    input
   *    |  \                                         \     /
   *   cast \                                          add   input1
   *    |  /                                            |     /
   *   add1                                          netoutput
   *    |
   * partioncall     input2        
   *    /       \     /
   * partionout  add2
   *            /   \
   *      netout  netout1
   */
  auto graph = CreateGraphParentAndSub();
  FusionTurbo acc(graph);
  auto movnode = graph->FindNode("add2");
  acc.GraphNodeUpMigration(movnode, 0);
  /*
   *  input                               data   input input1 data1
   *    |  \                                 \    /     \    /
   *   cast \                                  add      add2
   *    |  /                                     \      /
   *    add1  input2                              \    /
   *    |     /                                    \  /  
   *   partioncall                                netouput
   *    /        \      \
   * partionout netout netout1
   */
  auto aftermovnode = graph->FindNode("add2");
  EXPECT_EQ(aftermovnode, nullptr);

  auto partioncall_node = graph->FindNode("partioncall");
  EXPECT_NE(partioncall_node, nullptr);
  auto sub_graph = ge::NodeUtils::GetSubgraph(*partioncall_node, 0);

  auto add2 = sub_graph->FindNode("add2");
  EXPECT_NE(add2, nullptr);

  auto in_nodes = add2->GetInDataNodes();
  ASSERT_EQ(in_nodes.size(), 2);
  EXPECT_EQ(in_nodes.at(0)->GetType(), "NetOutInput");
  EXPECT_EQ(in_nodes.at(1)->GetType(), "Data");

  auto out_nodes = add2->GetOutDataNodes();
  EXPECT_EQ(out_nodes.at(0)->GetType(), "NetOutput");

  auto data1 = in_nodes.at(1);
  int64_t index;
  ge::AttrUtils::GetInt(data1->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, index);
  EXPECT_EQ(index, 1);
  EXPECT_EQ(partioncall_node->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "other1");
}

TEST_F(UTestFusionTurbo, test_case_16_2) {
  /*
   *  input                                          data    input
   *    |  \                                            \     /
   *   cast \                                             add input1
   *    |  /                                               |  /
   *   add1                                             netoutput
   *    |
   * partioncall   input2        
   *     /      \   /
   * partionout  add2
   *            /   \
   *      netout  netout1
   */
  auto graph = CreateGraphParentAndSub();
  FusionTurbo acc(graph);
  auto movnode = graph->FindNode("add1");
  acc.GraphNodeDownMigration(movnode, 0);
  /*
   *  input                                        data    data1
   *    |  \                                          \     /
   *   cast \                                           add1  input
   *    |    \                                             \   /
   *    |   /                                                add input1
   *    |  /                                                  |  /
   * partioncall   input2                                   netout
   *     /     \    /
   * partionout add2
   *           /   \
   *      netout  netout1
   */
  auto aftermovnode = graph->FindNode("add1");
  EXPECT_EQ(aftermovnode, nullptr);
  auto partioncall_node = graph->FindNode("partioncall");
  EXPECT_NE(partioncall_node, nullptr);
  auto sub_graph = ge::NodeUtils::GetSubgraph(*partioncall_node, 0);

  auto add1 = sub_graph->FindNode("add1");
  EXPECT_NE(add1, nullptr);

  auto in_nodes = add1->GetInDataNodes();
  EXPECT_EQ(in_nodes.at(0)->GetType(), "Data");
  EXPECT_EQ(in_nodes.at(1)->GetType(), "Data");

  auto out_nodes = add1->GetOutDataNodes();
  EXPECT_EQ(out_nodes.at(0)->GetType(), "Add");
}

TEST_F(UTestFusionTurbo, test_case_17) {
  auto graph = CreateGraphSingleInAndOut();
  FusionTurbo acc(graph);
  string name = "transpose";
  string type = "Transpose";
  auto node = acc.AddNodeOnly(name, type);
  ASSERT_NE(node, nullptr);

  auto relu = GetNode(graph, "relu");
  auto cast2 = GetNode(graph, "cast2");
  Relations dst_list = {{relu, 0, PEER}};
  Relations src_list = {{relu, 0}};

  acc.LinkInput(src_list, node);

  Status ret = acc.LinkOutput(dst_list, node, UPDATE_THIS);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(node->GetName(), name);
  EXPECT_EQ(node->GetType(), type);
  vector<int64_t> dims = {1, 4, 64, 64};
  auto input = node->GetOpDesc()->MutableInputDesc(0);
  EXPECT_EQ(input->GetShape().GetDims(), dims);
  EXPECT_EQ(input->GetFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(input->GetDataType(), ge::DT_FLOAT);

  auto output = node->GetOpDesc()->MutableOutputDesc(0);
  EXPECT_EQ(output->GetShape().GetDims(), dims);
  EXPECT_EQ(output->GetFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(output->GetDataType(), ge::DT_FLOAT);

  EXPECT_EQ(node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "relu");
  EXPECT_EQ(node->GetOutDataAnchor(0)->GetPeerInDataAnchors().size(), 1);
  EXPECT_EQ(node->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode()->GetName(), "cast2");
}

TEST_F(UTestFusionTurbo, test_case_17_1) {
  auto graph = CreateGraphSingleInAndOut();
  FusionTurbo acc(graph);
  string name = "transpose";
  string type = "Transpose";
  auto node = acc.AddNodeOnly(name, type);
  ASSERT_NE(node, nullptr);

  auto relu = GetNode(graph, "relu");
  auto cast2 = GetNode(graph, "cast2");
  Relations dst_list = {{relu, 0, PEER}};
  Relations src_list = {{relu, 0}};

  Status ret = acc.LinkOutput(dst_list, node, UPDATE_THIS);

  acc.LinkInput(src_list, node);


  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(node->GetName(), name);
  EXPECT_EQ(node->GetType(), type);
  vector<int64_t> dims = {1, 4, 64, 64};
  auto input = node->GetOpDesc()->MutableInputDesc(0);
  EXPECT_EQ(input->GetShape().GetDims(), dims);
  EXPECT_EQ(input->GetFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(input->GetDataType(), ge::DT_FLOAT);

  auto output = node->GetOpDesc()->MutableOutputDesc(0);
  EXPECT_EQ(output->GetShape().GetDims(), dims);
  EXPECT_EQ(output->GetFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(output->GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "relu");
  EXPECT_EQ(node->GetOutDataAnchor(0)->GetPeerInDataAnchors().size(), 1);
  EXPECT_EQ(node->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode()->GetName(), "cast2");
}
}
