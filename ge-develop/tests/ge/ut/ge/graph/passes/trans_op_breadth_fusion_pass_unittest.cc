/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/format_optimize/transop_breadth_fusion_pass.h"

#include <gtest/gtest.h>
#include <string>

#include "common/ge_inner_error_codes.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_builder_utils.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/utils/node_adapter.h"
#include "graph/utils/graph_utils_ex.h"
using namespace ge;

class UtestGraphPassesTransOpBreadthFusionPass : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

class NodeBuilder {
 public:
  NodeBuilder(const std::string &name, const std::string &type) { op_desc_ = std::make_shared<OpDesc>(name, type); }

  NodeBuilder &AddInputDesc(std::initializer_list<int64_t> shape, ge::Format format = FORMAT_NCHW,
                            ge::DataType data_type = DT_FLOAT) {
    op_desc_->AddInputDesc(CreateTensorDesc(shape, format, data_type)->Clone());
    return *this;
  }

  NodeBuilder &AddOutputDesc(std::initializer_list<int64_t> shape, ge::Format format = FORMAT_NCHW,
                             ge::DataType data_type = DT_FLOAT) {
    op_desc_->AddOutputDesc(CreateTensorDesc(shape, format, data_type)->Clone());
    return *this;
  }

  ge::NodePtr Build(const ge::ComputeGraphPtr &graph) { return graph->AddNode(op_desc_); }

 private:
  ge::GeTensorDescPtr CreateTensorDesc(std::initializer_list<int64_t> shape, ge::Format format = FORMAT_NCHW,
                                       ge::DataType data_type = DT_FLOAT) {
    GeShape ge_shape{std::vector<int64_t>(shape)};
    ge::GeTensorDescPtr tensor_desc = std::make_shared<ge::GeTensorDesc>();
    tensor_desc->SetShape(ge_shape);
    tensor_desc->SetFormat(format);
    tensor_desc->SetDataType(data_type);
    return tensor_desc;
  }

  ge::OpDescPtr op_desc_;
};

/**
 *       relu1    relu2                relu1  relu2
 *        |         |                    \    /    \
 *     reshape1  reshape2      -->       reshape1   |
 *      /   \  /      \                 /   \       |(ctrl edge)
 *   const1  data      const2         const1  data const2
 * */
TEST_F(UtestGraphPassesTransOpBreadthFusionPass, delete_reshape_and_relink_const_to_next_node) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");

  ge::NodePtr node1 = NodeBuilder("node1", DATA).AddOutputDesc({1}, FORMAT_NCHW, DT_INT32).Build(graph);

  ge::NodePtr reshape_node_1 = NodeBuilder("reshape_node_1", RESHAPE)
      .AddInputDesc({1}, FORMAT_NCHW, DT_FLOAT)
      .AddInputDesc({1}, FORMAT_NCHW, DT_INT32)
      .AddOutputDesc({1, 1}, FORMAT_NC1HWC0, DT_INT32)
      .Build(graph);

  ge::NodePtr reshape_node_2 = NodeBuilder("reshape_node_2", RESHAPE)
      .AddInputDesc({1}, FORMAT_NCHW, DT_FLOAT)
      .AddInputDesc({1}, FORMAT_NCHW, DT_INT32)
      .AddOutputDesc({1, 1}, FORMAT_NC1HWC0, DT_INT32)
      .Build(graph);

  ge::NodePtr node_2 = NodeBuilder("node2", RELU).AddInputDesc({1, 1}, FORMAT_NC1HWC0, DT_INT32).Build(graph);

  ge::NodePtr node_3 = NodeBuilder("node3", RELU).AddInputDesc({1, 1}, FORMAT_NC1HWC0, DT_INT32).Build(graph);

  ge::NodePtr constant_1 = NodeBuilder("constant1", CONSTANT).AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT).Build(graph);

  ge::NodePtr constant_2 = NodeBuilder("constant2", CONSTANT).AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT).Build(graph);

  ge::GraphUtils::AddEdge(node1->GetOutDataAnchor(0), reshape_node_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node1->GetOutDataAnchor(0), reshape_node_2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(reshape_node_1->GetOutDataAnchor(0), node_2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(reshape_node_2->GetOutDataAnchor(0), node_3->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(constant_1->GetOutDataAnchor(0), reshape_node_1->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(constant_2->GetOutDataAnchor(0), reshape_node_2->GetInDataAnchor(1));

  ge::TransOpBreadthFusionPass pass;
  Status status = pass.Run(graph);
  EXPECT_EQ(SUCCESS, status);
  ASSERT_EQ(node1->GetOutDataNodesSize(), 1U);
  ASSERT_EQ(constant_2->GetOutControlNodes().size(), 1U);
  ASSERT_EQ(constant_2->GetOutControlNodes().at(0), node_3);
}

/*
 *        +----------------+             +---------------+
 *        |                |             |               |
 *       relu1    relu2    |            relu1  relu2     |
 *        |         |      |              \    /    \    |
 *     reshape1  reshape2  |    -->       reshape1   |   |
 *      /   \  /      \   \|/             /   \       | \|/(ctrl edge)
 *   const1  data      const2         const1  data   const2
 * */
TEST_F(UtestGraphPassesTransOpBreadthFusionPass, check_loop_success) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");

  ge::NodePtr node1 = NodeBuilder("node1", DATA).AddOutputDesc({1}, FORMAT_NCHW, DT_INT32).Build(graph);

  ge::NodePtr reshape_node_1 = NodeBuilder("reshape_node_1", RESHAPE)
      .AddInputDesc({1}, FORMAT_NCHW, DT_FLOAT)
      .AddInputDesc({1}, FORMAT_NCHW, DT_INT32)
      .AddOutputDesc({1, 1}, FORMAT_NC1HWC0, DT_INT32)
      .Build(graph);

  ge::NodePtr reshape_node_2 = NodeBuilder("reshape_node_2", RESHAPE)
      .AddInputDesc({1}, FORMAT_NCHW, DT_FLOAT)
      .AddInputDesc({1}, FORMAT_NCHW, DT_INT32)
      .AddOutputDesc({1, 1}, FORMAT_NC1HWC0, DT_INT32)
      .Build(graph);

  ge::NodePtr node_2 = NodeBuilder("node2", RELU).AddInputDesc({1, 1}, FORMAT_NC1HWC0, DT_INT32).Build(graph);

  ge::NodePtr node_3 = NodeBuilder("node3", RELU).AddInputDesc({1, 1}, FORMAT_NC1HWC0, DT_INT32).Build(graph);

  ge::NodePtr constant_1 = NodeBuilder("constant1", CONSTANT).AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT).Build(graph);

  ge::NodePtr constant_2 = NodeBuilder("constant2", CONSTANT).AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT).Build(graph);

  ge::GraphUtils::AddEdge(node1->GetOutDataAnchor(0), reshape_node_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node1->GetOutDataAnchor(0), reshape_node_2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(reshape_node_1->GetOutDataAnchor(0), node_2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(reshape_node_2->GetOutDataAnchor(0), node_3->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(constant_1->GetOutDataAnchor(0), reshape_node_1->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(constant_2->GetOutDataAnchor(0), reshape_node_2->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(node_2->GetOutControlAnchor(), constant_2->GetInControlAnchor());

  ge::TransOpBreadthFusionPass pass;
  Status status = pass.Run(graph);
  EXPECT_EQ(SUCCESS, status);
  ASSERT_EQ(graph->TopologicalSorting(), SUCCESS); // check loop
  ASSERT_EQ(node1->GetOutDataNodesSize(), 1U);
  ASSERT_EQ(constant_2->GetOutControlNodes().size(), 1U);
  ASSERT_EQ(constant_2->GetOutControlNodes().at(0), node_3);
}

TEST_F(UtestGraphPassesTransOpBreadthFusionPass, test_reshape_op_failed) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");

  ge::NodePtr data1 = NodeBuilder("data1", DATA).AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT).Build(graph);

  ge::NodePtr data2 = NodeBuilder("data2", DATA).AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT).Build(graph);

  ge::NodePtr constant = NodeBuilder("constant", CONSTANT).AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT).Build(graph);

  ge::NodePtr exp1 = NodeBuilder("exp1", EXP)
                         .AddInputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                         .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                         .Build(graph);

  ge::NodePtr exp2 = NodeBuilder("exp2", EXP)
                         .AddInputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                         .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                         .Build(graph);

  ge::NodePtr reshape1 = NodeBuilder("reshape1", RESHAPE)
                             .AddInputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                             .AddInputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                             .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                             .Build(graph);

  ge::NodePtr reshape2 = NodeBuilder("reshape2", RESHAPE)
                             .AddInputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                             .AddInputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                             .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                             .Build(graph);

  ge::NodePtr relu1 = NodeBuilder("relu1", RELU).AddInputDesc({1}, FORMAT_NCHW, DT_FLOAT).Build(graph);

  ge::NodePtr relu2 = NodeBuilder("relu2", RELU).AddInputDesc({1}, FORMAT_NCHW, DT_FLOAT).Build(graph);

  ge::GraphUtils::AddEdge(data1->GetOutDataAnchor(0), exp1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(data2->GetOutDataAnchor(0), exp2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(exp1->GetOutDataAnchor(0), reshape1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(exp2->GetOutDataAnchor(0), reshape2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(constant->GetOutDataAnchor(0), reshape1->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(constant->GetOutDataAnchor(0), reshape2->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(reshape1->GetOutDataAnchor(0), relu1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(reshape2->GetOutDataAnchor(0), relu2->GetInDataAnchor(0));

  ge::TransOpBreadthFusionPass pass;
  Status status = pass.Run(graph);
  EXPECT_EQ(SUCCESS, status);
  EXPECT_EQ(reshape1->GetOutDataNodes().size(), 1);
  EXPECT_EQ(reshape2->GetOutDataNodes().size(), 1);
}

TEST_F(UtestGraphPassesTransOpBreadthFusionPass, test_multi_anchor_case) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");

  ge::NodePtr data1 = NodeBuilder("data1", DATA)
                          .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                          .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                          .Build(graph);

  ge::NodePtr cast1 = NodeBuilder("cast1", CAST)
                          .AddInputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                          .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT16)
                          .Build(graph);

  ge::NodePtr cast2 = NodeBuilder("cast2", CAST)
                          .AddInputDesc({1}, FORMAT_NCHW, DT_FLOAT16)
                          .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                          .Build(graph);

  ge::NodePtr relu1 = NodeBuilder("relu1", RELU).AddInputDesc({1}, FORMAT_NCHW, DT_FLOAT16).Build(graph);

  ge::NodePtr relu2 = NodeBuilder("relu2", RELU).AddInputDesc({1}, FORMAT_NCHW, DT_FLOAT16).Build(graph);

  ge::GraphUtils::AddEdge(data1->GetOutDataAnchor(0), cast1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(data1->GetOutDataAnchor(1), cast2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast1->GetOutDataAnchor(0), relu1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast2->GetOutDataAnchor(0), relu2->GetInDataAnchor(0));

  ge::TransOpBreadthFusionPass pass;
  Status status = pass.Run(graph);

  EXPECT_EQ(SUCCESS, status);
  EXPECT_EQ(cast1->GetOutDataNodes().size(), 1);
  EXPECT_EQ(cast1->GetOutDataNodes().size(), 1);
}

/**
 *            ----> netoutput1
 *          /        |       \.
 *  transdata1    transdata2  transdata3
 *           \   /             |
 *            var1--------------
 */
static ComputeGraphPtr BuildGraph1() {
  ut::GraphBuilder builder("g1");
  auto var1 = builder.AddNode("var1", "Variable", 0, 1);
  auto transdata1 = builder.AddNode("transdata1", "TransData", 1, 1);
  transdata1->GetOpDesc()->MutableOutputDesc(0)->SetFormat(FORMAT_NC1HWC0);
  transdata1->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 1, 224, 224, 16})));
  AttrUtils::SetStr(transdata1->GetOpDesc(), ATTR_NAME_STREAM_LABEL, "label1");
  auto transdata2 = builder.AddNode("transdata2", "TransData", 1, 1);
  transdata2->GetOpDesc()->MutableOutputDesc(0)->SetFormat(FORMAT_NC1HWC0);
  transdata2->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 1, 224, 224, 16})));
  auto transdata3 = builder.AddNode("transdata3", "TransData", 1, 1);
  transdata3->GetOpDesc()->MutableOutputDesc(0)->SetFormat(FORMAT_NC1HWC0);
  transdata3->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 1, 224, 224, 16})));
  auto netoutput1 = builder.AddNode("netoutput1", "NetOutput1", 10, 0);

  builder.AddDataEdge(var1, 0, transdata1, 0);
  builder.AddDataEdge(var1, 0, transdata2, 0);
  builder.AddDataEdge(var1, 0, transdata3, 0);
  builder.AddDataEdge(transdata1, 0, netoutput1, 0);
  builder.AddDataEdge(transdata2, 0, netoutput1, 1);
  builder.AddDataEdge(transdata3, 0, netoutput1, 2);

  return builder.GetGraph();
}

/**
 *            --------->   netoutput1
 *          /              |       \.
 *  transdata1  transdata2(l1)  transdata3(l1)
 *           \   /                  |
 *            var1------------------
 */
static ComputeGraphPtr BuildGraph2() {
  ut::GraphBuilder builder("g2");
  auto var1 = builder.AddNode("var1", "Variable", 0, 1);
  auto transdata1 = builder.AddNode("transdata1", "TransData", 1, 1);
  transdata1->GetOpDesc()->MutableOutputDesc(0)->SetFormat(FORMAT_NC1HWC0);
  transdata1->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 1, 224, 224, 16})));
  auto transdata2 = builder.AddNode("transdata2", "TransData", 1, 1);
  transdata2->GetOpDesc()->MutableOutputDesc(0)->SetFormat(FORMAT_NC1HWC0);
  transdata2->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 1, 224, 224, 16})));
  AttrUtils::SetStr(transdata2->GetOpDesc(), ATTR_NAME_STREAM_LABEL, "label1");
  auto transdata3 = builder.AddNode("transdata3", "TransData", 1, 1);
  transdata3->GetOpDesc()->MutableOutputDesc(0)->SetFormat(FORMAT_NC1HWC0);
  transdata3->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 1, 224, 224, 16})));
  AttrUtils::SetStr(transdata3->GetOpDesc(), ATTR_NAME_STREAM_LABEL, "label1");
  auto netoutput1 = builder.AddNode("netoutput1", "NetOutput1", 10, 0);

  builder.AddDataEdge(var1, 0, transdata1, 0);
  builder.AddDataEdge(var1, 0, transdata2, 0);
  builder.AddDataEdge(var1, 0, transdata3, 0);
  builder.AddDataEdge(transdata1, 0, netoutput1, 0);
  builder.AddDataEdge(transdata2, 0, netoutput1, 1);
  builder.AddDataEdge(transdata3, 0, netoutput1, 2);

  return builder.GetGraph();
}

/**
 *                     netoutput
 *                     /       \
 *  const1  --  transpose1    transpose2  --  const2
 *                        \   /
 *                         var
 */
static ComputeGraphPtr BuildGraph3() {
 vector<int64_t> perm1{0, 3, 1, 2};
  GeTensorDesc tensor_desc1(GeShape(vector<int64_t>{4}), FORMAT_ND, DT_INT64);
  GeTensorPtr const_tensor1 = 
    std::make_shared<GeTensor>(tensor_desc1, reinterpret_cast<uint8_t *>(perm1.data()) , sizeof(int64_t)*perm1.size());
  auto const1 = OP_CFG(CONSTANT).Weight(const_tensor1);

  vector<int32_t> perm2{0, 2, 1, 3};
  GeTensorDesc tensor_desc2(GeShape(vector<int64_t>{4}), FORMAT_ND, DT_INT32);
  GeTensorPtr const_tensor2 = 
    std::make_shared<GeTensor>(tensor_desc2, reinterpret_cast<uint8_t *>(perm2.data()), sizeof(int32_t)*perm2.size());
  auto const2 = OP_CFG(CONSTANT).Weight(const_tensor2);

  auto transpose1 = OP_CFG(TRANSPOSE).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 128, 52, 52});
  auto transpose2 = OP_CFG(TRANSPOSE).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 128, 52, 52});

  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", DATA)->EDGE(0, 0)->NODE("transpose1", transpose1));
    CHAIN(NODE("data1", DATA)->EDGE(0, 0)->NODE("transpose2", transpose2));
    CHAIN(NODE("const1", const1)->EDGE(0, 1)->NODE("transpose1", transpose1));
    CHAIN(NODE("const2", const2)->EDGE(0, 1)->NODE("transpose2", transpose2));
    CHAIN(NODE("transpose1", transpose1)->EDGE(0, 0)->NODE("netoutput1", NETOUTPUT));
    CHAIN(NODE("transpose2", transpose2)->EDGE(0, 0)->NODE("netoutput1", NETOUTPUT));
  };

  auto graph = ToGeGraph(g1);

  map<string, uint32_t> name_index;
  name_index.emplace("x", 0);
  name_index.emplace("perm", 1);
  for (auto &gn : graph.GetAllNodes()) {
    AscendString type;
    (void)gn.GetType(type);
    if (type == TRANSPOSE) {
      auto node = NodeAdapter::GNode2Node(gn);
      if (node != nullptr && node->GetOpDesc()) {
        node->GetOpDesc()->UpdateInputName(name_index);
      }
    }
  }

  return ge::GraphUtilsEx::GetComputeGraph(graph);
}

/*
*  if we want to fusion cast1 and cast2
*  it will cause a cycle between fusion_cast and transdata
*       data1
*       /    \
*      /      \
*    cast1     \
*      |        \
*   trandata---> cast2
*                 |
*                relu
*/
static ComputeGraphPtr BuildGraphMayCauseCycleWhenFusion() {
  auto root_builder = ut::GraphBuilder("root");
  const auto &data1 = root_builder.AddNode("data1", "Data", 1, 1);
  const auto &cast1 = root_builder.AddNode("cast1", "Cast", 1, 1);
  const auto &cast2 = root_builder.AddNode("cast2", "Cast", 1, 1);
  const auto &transdata = root_builder.AddNode("transdata", "TransData", 1, 1);
  const auto &relu = root_builder.AddNode("relu", "Relu", 1, 1);

  root_builder.AddDataEdge(data1, 0, cast1, 0);
  root_builder.AddDataEdge(data1, 0, cast2, 0);
  root_builder.AddDataEdge(cast1, 0, transdata, 0);
  root_builder.AddDataEdge(cast2, 0, relu, 0);
  root_builder.AddControlEdge(transdata, cast2);
  return root_builder.GetGraph();
}

/*
*  if we want to fusion cast1 and cast2
*  it will cause a cycle between fusion_cast and transdata
*       data1
*       /     \
*      /       \
*    cast1 --->cast2
*      |         |
*  trandata1    relu
*/
static ComputeGraphPtr BuildGraphWithCtrlEdgeBetweenTransop() {
  auto root_builder = ut::GraphBuilder("root");
  const auto &data1 = root_builder.AddNode("data1", "Data", 1, 1);
  const auto &cast1 = root_builder.AddNode("cast1", "Cast", 1, 1);
  const auto &cast2 = root_builder.AddNode("cast2", "Cast", 1, 1);
  const auto &transdata1 = root_builder.AddNode("transdata1", "TransData", 1, 1);
  const auto &relu = root_builder.AddNode("relu", "Relu", 1, 1);

  root_builder.AddDataEdge(data1, 0, cast1, 0);
  root_builder.AddDataEdge(data1, 0, cast2, 0);
  root_builder.AddDataEdge(cast1, 0, transdata1, 0);
  root_builder.AddDataEdge(cast2, 0, relu, 0);
  root_builder.AddControlEdge(cast1, cast2);
  return root_builder.GetGraph();
}

TEST_F(UtestGraphPassesTransOpBreadthFusionPass, diff_stream1) {
  auto graph = BuildGraph1();

  ge::TransOpBreadthFusionPass pass;
  Status status = pass.Run(graph);
  EXPECT_EQ(SUCCESS, status);

  auto transdata1 = graph->FindNode("transdata1");
  auto transdata2 = graph->FindNode("transdata2");
  auto transdata3 = graph->FindNode("transdata3");

  EXPECT_EQ(transdata1->GetOutNodes().size(), 1);
  EXPECT_EQ(transdata1->GetOutDataNodes().at(0)->GetName(), "netoutput1");
  EXPECT_EQ(transdata1->GetInNodes().size(), 1);
  EXPECT_EQ(transdata1->GetInDataNodes().at(0)->GetName(), "var1");

  EXPECT_TRUE(transdata2 == nullptr || transdata3 == nullptr);
  EXPECT_FALSE(transdata2 == nullptr && transdata3 == nullptr);
  auto not_empty_node = transdata2 != nullptr ? transdata2 : transdata3;
  EXPECT_FALSE(not_empty_node->GetInNodes().empty());
  EXPECT_EQ(not_empty_node->GetInDataNodes().at(0)->GetName(), "var1");
  EXPECT_FALSE(not_empty_node->GetOutNodes().empty());
  EXPECT_EQ(not_empty_node->GetOutDataNodes().at(0)->GetName(), "netoutput1");
}

TEST_F(UtestGraphPassesTransOpBreadthFusionPass, diff_stream2) {
  auto graph = BuildGraph2();

  ge::TransOpBreadthFusionPass pass;
  Status status = pass.Run(graph);
  EXPECT_EQ(SUCCESS, status);

  auto transdata1 = graph->FindNode("transdata1");
  auto transdata2 = graph->FindNode("transdata2");
  auto transdata3 = graph->FindNode("transdata3");

  EXPECT_EQ(transdata1->GetOutNodes().size(), 1);
  EXPECT_EQ(transdata1->GetOutDataNodes().at(0)->GetName(), "netoutput1");
  EXPECT_EQ(transdata1->GetInNodes().size(), 1);
  EXPECT_EQ(transdata1->GetInDataNodes().at(0)->GetName(), "var1");

  EXPECT_TRUE(transdata2 == nullptr || transdata3 == nullptr);
  EXPECT_FALSE(transdata2 == nullptr && transdata3 == nullptr);
  auto not_empty_node = transdata2 != nullptr ? transdata2 : transdata3;
  EXPECT_FALSE(not_empty_node->GetInNodes().empty());
  EXPECT_EQ(not_empty_node->GetInDataNodes().at(0)->GetName(), "var1");
  EXPECT_FALSE(not_empty_node->GetOutNodes().empty());
  EXPECT_EQ(not_empty_node->GetOutDataNodes().at(0)->GetName(), "netoutput1");
}

TEST_F(UtestGraphPassesTransOpBreadthFusionPass, diff_transpose) {
  auto graph = BuildGraph3();

  ge::TransOpBreadthFusionPass pass;
  Status status = pass.Run(graph);
  EXPECT_EQ(SUCCESS, status);

  auto transpose1 = graph->FindNode("transpose1");
  auto transpose2 = graph->FindNode("transpose2");

  EXPECT_FALSE(transpose1 == nullptr);
  EXPECT_FALSE(transpose2 == nullptr);
}

TEST_F(UtestGraphPassesTransOpBreadthFusionPass, undirect_control_relation_may_cause_cycle) {
  auto graph = BuildGraphMayCauseCycleWhenFusion();
  EXPECT_EQ(graph->GetDirectNodesSize(), 5);

  ge::TransOpBreadthFusionPass pass;
  Status status = pass.Run(graph);
  EXPECT_EQ(SUCCESS, status);
  EXPECT_EQ(graph->GetDirectNodesSize(), 4);
  EXPECT_EQ(graph->TopologicalSorting(), SUCCESS); // topo success, no cycle
  // check relu in_ctrl now is transdata
  const auto &relu = graph->FindNode("relu");
  EXPECT_EQ(relu->GetInControlNodes().at(0)->GetName(), "transdata");
}

TEST_F(UtestGraphPassesTransOpBreadthFusionPass, direct_control_relation_may_cause_cycle) {
  auto graph = BuildGraphWithCtrlEdgeBetweenTransop();
  EXPECT_EQ(graph->GetDirectNodesSize(), 5);

  ge::TransOpBreadthFusionPass pass;
  Status status = pass.Run(graph);
  EXPECT_EQ(SUCCESS, status);
  EXPECT_EQ(graph->GetDirectNodesSize(), 4);
  EXPECT_EQ(graph->TopologicalSorting(), SUCCESS); // topo success, no cycle
}

TEST_F(UtestGraphPassesTransOpBreadthFusionPass, reshape_with_unknown_input) {
  DEF_GRAPH(reshape_unknown_input) {
    auto data0 = OP_CFG(DATA)
      .InCnt(1)
      .OutCnt(1)
      .TensorDesc(FORMAT_ND, DT_FLOAT, {1})
      .Build("data0");
    auto data1 = OP_CFG(DATA)
      .InCnt(1)
      .OutCnt(1)
      .TensorDesc(FORMAT_ND, DT_FLOAT, {1})
      .Build("data1");
    auto data2 = OP_CFG(DATA)
      .InCnt(1)
      .OutCnt(1)
      .TensorDesc(FORMAT_ND, DT_FLOAT, {1})
      .Build("data2");
    auto reshape0 = OP_CFG(RESHAPE)
      .InCnt(2)
      .OutCnt(1)
      .Build("reshape0");
    auto reshape1 = OP_CFG(RESHAPE)
      .InCnt(2)
      .OutCnt(1)
      .Build("reshape1");
    auto netoutput = OP_CFG(NETOUTPUT)
      .InCnt(1)
      .OutCnt(1)
      .Build("reshnetoutputape");
    CHAIN(NODE(data0)->EDGE(0, 0)->NODE(reshape0)->EDGE(0, 0)->NODE(netoutput));
    CHAIN(NODE(data0)->EDGE(0, 0)->NODE(reshape1)->EDGE(0, 1)->NODE(netoutput));
    CHAIN(NODE(data1)->EDGE(0, 1)->NODE(reshape0));
    CHAIN(NODE(data2)->EDGE(0, 1)->NODE(reshape1));
  };
  const auto graph = ToGeGraph(reshape_unknown_input);
  const auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);

  ge::TransOpBreadthFusionPass pass;
  auto ret = pass.Run(compute_graph);
  EXPECT_EQ(SUCCESS, ret);

  // reshape is not fused.
  auto reshape_node0 = compute_graph->FindNode("reshape0");
  EXPECT_NE(reshape_node0, nullptr);
  auto reshape_node1 = compute_graph->FindNode("reshape1");
  EXPECT_NE(reshape_node1, nullptr);
}
