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
#include "graph/shape_refiner.h"
#include "graph/operator_factory_impl.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_builder_utils.h"
#include "graph/debug/ge_op_types.h"
#include "graph/debug/ge_attr_define.h"

#include <dlog_pub.h>
#include <ge_common/debug/ge_log.h>
#include <graph/operator_reg.h>

namespace ge {
namespace {
static NodePtr CreateNode(const ComputeGraphPtr &graph, const string &name, const string &type, int in_num, int out_num) {
  OpDescPtr op_desc = std::make_shared<OpDesc>(name, type);
  op_desc->SetStreamId(0);
  static int32_t index = 0;
  op_desc->SetId(index++);

  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  tensor.SetOriginFormat(FORMAT_NCHW);
  tensor.SetOriginDataType(DT_FLOAT);
  TensorUtils::SetSize(tensor, 512);
  vector<int64_t> input_offset;
  for (int i = 0; i < in_num; i++) {
    op_desc->AddInputDesc(tensor);
    input_offset.emplace_back(1024);
  }
  op_desc->SetInputOffset(input_offset);

  vector<int64_t> output_offset;
  for (int i = 0; i < out_num; i++) {
    op_desc->AddOutputDesc(tensor);
    output_offset.emplace_back(1024);
  }
  op_desc->SetOutputOffset(output_offset);

  op_desc->SetWorkspace({});
  op_desc->SetWorkspaceBytes({});
  op_desc->SetOpKernelLibName("DNN_VM_RTS_OP_STORE");

  const auto stub_func = [](Operator &op) { return GRAPH_SUCCESS; };
  op_desc->AddInferFunc(stub_func);
  op_desc->AddInferFormatFunc(stub_func);
  op_desc->AddVerifierFunc(stub_func);

  return graph->AddNode(op_desc);
}

/*
 *                                 Data1
 *       sub_data1                   |                     sub_data2              sub_data3
 *           |               PartitionedCall2   ===>           |                      |
 *         relu1                     |                  PartitionedCall3   ===>     relu2
 *           |        <===   PartitionedCall1                  |                      |
 *      sub_output1                  |                     sub_output2           sub_output3
 *                               netoutput
 */
ComputeGraphPtr CreateGraphWithMultiSubgraph() {
  ut::GraphBuilder builder = ut::GraphBuilder("root_graph");
  auto data = builder.AddNode("Data1", "Data", 1, 1);
  auto partcall1 = builder.AddNode("partcall1", "PartitionedCall", 1, 1);
  auto partcall2 = builder.AddNode("partcall2", "PartitionedCall", 1, 1);
  auto netoutput = builder.AddNode("netoutput", "NetOutput", 1, 0);

  builder.AddDataEdge(data, 0, partcall2, 0);
  builder.AddDataEdge(partcall2, 0, partcall1, 0);
  builder.AddDataEdge(partcall1, 0, netoutput, 0);
  auto root_graph = builder.GetGraph();

  ut::GraphBuilder sub_builder1 = ut::GraphBuilder("sub_graph1");
  auto sub_data1 = sub_builder1.AddNode("sub_data1", "Data", 1, 1);
  auto data1_desc = sub_data1->GetOpDesc();
  AttrUtils::SetInt(data1_desc, "_parent_node_index", 0);
  auto sub_relu1 = sub_builder1.AddNode("sub_relu1", "Relu", 1, 1);
  auto sub_output1 = sub_builder1.AddNode("sub_output1", "NetOutput", 1, 0);
  sub_builder1.AddDataEdge(sub_data1, 0, sub_relu1, 0);
  sub_builder1.AddDataEdge(sub_relu1, 0, sub_output1, 0);
  auto subgraph1 = sub_builder1.GetGraph();

  ut::GraphBuilder sub_builder2 = ut::GraphBuilder("sub_graph2");
  auto sub_data2 = sub_builder2.AddNode("sub_data2", "Data", 1, 1);
  auto partcall3 = sub_builder2.AddNode("partcall3", "PartitionedCall", 1, 1);
  auto sub_output2 = sub_builder2.AddNode("sub_output2", "NetOutput", 1, 0);
  auto output2_desc = sub_output2->GetOpDesc();
  auto output2_desc_in = output2_desc->MutableInputDesc(0);
  AttrUtils::SetInt(output2_desc_in, "_parent_node_index", 0);
  sub_builder2.AddDataEdge(sub_data2, 0, partcall3, 0);
  sub_builder2.AddDataEdge(partcall3, 0, sub_output2, 0);
  auto subgraph2 = sub_builder2.GetGraph();

  ut::GraphBuilder sub_builder3 = ut::GraphBuilder("sub_graph3");
  auto sub_data3 = sub_builder3.AddNode("sub_data3", "Data", 1, 1);
  auto sub_relu2 = sub_builder3.AddNode("sub_relu2", "Relu", 1, 1);
  auto sub_output3 = sub_builder3.AddNode("sub_output3", "NetOutput", 1, 0);
  auto output3_desc = sub_output3->GetOpDesc();
  auto output3_desc_in = output3_desc->MutableInputDesc(0);
  AttrUtils::SetInt(output3_desc_in, "_parent_node_index", 0);
  sub_builder3.AddDataEdge(sub_data3, 0, sub_relu2, 0);
  sub_builder3.AddDataEdge(sub_relu2, 0, sub_output3, 0);
  auto subgraph3 = sub_builder3.GetGraph();

  auto part_node1 = root_graph->FindNode("partcall1");
  auto part_desc1 = part_node1->GetOpDesc();
  part_desc1->AddSubgraphName("sub_graph1");
  part_desc1->SetSubgraphInstanceName(0, "sub_graph1");

  subgraph1->SetParentNode(part_node1);
  subgraph1->SetParentGraph(root_graph);
  root_graph->AddSubgraph("sub_graph1", subgraph1);

  auto part_node2 = root_graph->FindNode("partcall2");
  auto part_desc2 = part_node2->GetOpDesc();
  part_desc2->AddSubgraphName("sub_graph2");
  part_desc2->SetSubgraphInstanceName(0, "sub_graph2");

  subgraph2->SetParentNode(part_node2);
  subgraph2->SetParentGraph(root_graph);
  root_graph->AddSubgraph("sub_graph2", subgraph2);

  auto part_node3 = subgraph2->FindNode("partcall3");
  auto part_desc3 = part_node3->GetOpDesc();
  part_desc3->AddSubgraphName("sub_graph3");
  part_desc3->SetSubgraphInstanceName(0, "sub_graph3");

  subgraph3->SetParentNode(part_node3);
  subgraph3->SetParentGraph(subgraph2);
  root_graph->AddSubgraph(subgraph3);

  return root_graph;
}

/*
 *              Data1
 *                |
 *              relu1                   sub_data0
 *                |                         |
 *        PartitionedCall0     ===>     sub_output0
 *                |                     sub_data1
 *        PartitionedCall1     ===>         |
 *                |                     sub_output1
 *              relu2
 *                |
 *            netoutput
 */
ComputeGraphPtr CreateGraphWithSubgraphDataToNetoutput() {
  ut::GraphBuilder builder = ut::GraphBuilder("root_graph");
  auto data = builder.AddNode("Data1", "Data", 1, 1);
  auto relu1 = builder.AddNode("relu1", "Relu", 1, 1);
  auto partcall0 = builder.AddNode("partcall0", "PartitionedCall", 1, 1);
  auto partcall1 = builder.AddNode("partcall1", "PartitionedCall", 1, 1);
  auto relu2 = builder.AddNode("relu2", "Relu", 1, 1);
  auto netoutput = builder.AddNode("netoutput", "NetOutput", 1, 0);

  builder.AddDataEdge(data, 0, relu1, 0);
  builder.AddDataEdge(relu1, 0, partcall0, 0);
  builder.AddDataEdge(partcall0, 0, partcall1, 0);
  builder.AddDataEdge(partcall1, 0, relu2, 0);
  builder.AddDataEdge(relu2, 0, netoutput, 0);
  auto root_graph = builder.GetGraph();

  ut::GraphBuilder sub_builder1 = ut::GraphBuilder("sub_graph1");
  auto sub_data1 = sub_builder1.AddNode("sub_data1", "Data", 1, 1);
  auto data1_desc = sub_data1->GetOpDesc();
  AttrUtils::SetInt(data1_desc, "_parent_node_index", 0);
  auto sub_output1 = sub_builder1.AddNode("sub_output1", "NetOutput", 1, 0);
  auto output1_desc = sub_output1->GetOpDesc();
  auto output1_desc_in = output1_desc->MutableInputDesc(0);
  AttrUtils::SetInt(output1_desc_in, "_parent_node_index", 0);
  sub_builder1.AddDataEdge(sub_data1, 0, sub_output1, 0);
  auto subgraph1 = sub_builder1.GetGraph();

  auto part_node1 = root_graph->FindNode("partcall1");
  auto part_desc1 = part_node1->GetOpDesc();
  part_desc1->AddSubgraphName("sub_graph1");
  part_desc1->SetSubgraphInstanceName(0, "sub_graph1");

  subgraph1->SetParentNode(part_node1);
  subgraph1->SetParentGraph(root_graph);
  root_graph->AddSubgraph("sub_graph1", subgraph1);

  ut::GraphBuilder sub_builder0 = ut::GraphBuilder("sub_graph0");
  auto sub_data0 = sub_builder0.AddNode("sub_data0", "Data", 1, 1);
  auto data0_desc = sub_data0->GetOpDesc();
  AttrUtils::SetInt(data0_desc, "_parent_node_index", 0);
  auto sub_output0 = sub_builder0.AddNode("sub_output0", "NetOutput", 1, 0);
  auto output0_desc = sub_output0->GetOpDesc();
  auto output0_desc_in = output0_desc->MutableInputDesc(0);
  AttrUtils::SetInt(output0_desc_in, "_parent_node_index", 0);
  sub_builder0.AddDataEdge(sub_data0, 0, sub_output0, 0);
  auto subgraph0 = sub_builder0.GetGraph();

  auto part_node0 = root_graph->FindNode("partcall0");
  auto part_desc0 = part_node0->GetOpDesc();
  part_desc0->AddSubgraphName("sub_graph0");
  part_desc0->SetSubgraphInstanceName(0, "sub_graph0");

  subgraph0->SetParentNode(part_node0);
  subgraph0->SetParentGraph(root_graph);
  root_graph->AddSubgraph("sub_graph0", subgraph0);
  return root_graph;
}

/* cond_graph and body_graph share the same input tensor
 *             +-------------+   +-------------+
 *             | Cond Graph  |   | Body Graph  |
 *             | NetOutput   |   | NetOutput   |
 *             |    |        |   |    |        |
 * NetOutput   | LessThan_5  |   |  Add_1      |
 *   |         |   |         |   |   |         |
 * while  -----+ input       |   + input       |
 *   |         +-------------+   +-------------+
 * input
 */

ComputeGraphPtr BuildSimpleWhileGraph2() {
  const auto stub_func = [](Operator &op) { return GRAPH_SUCCESS; };
  const std::vector<int64_t> shape{-1, -1, 224, 224};
  // build main graph
  ut::GraphBuilder main_builder("main_graph");
  auto data_1 = main_builder.AddNode("data_1", "Data", 1, 1);
  auto data_2 = main_builder.AddNode("data_2", "Data", 1, 1);
  auto while_1 = main_builder.AddNode("while_1", "While", 1, 1);
  auto output_1 = main_builder.AddNode("output_1", "NetOutput", 1, 1);
  main_builder.AddDataEdge(data_1, 0, while_1, 0);
  main_builder.AddDataEdge(while_1, 0, output_1, 0);
  while_1->GetOpDesc()->AddInferFunc(stub_func);
  auto main_graph = main_builder.GetGraph();
  AttrUtils::SetInt(data_1->GetOpDesc(), ATTR_NAME_INDEX, 0);
  output_1->GetOpDesc()->SetSrcName({"while_1"});
  output_1->GetOpDesc()->SetSrcIndex({0, 1});

  // build condition graph
  ut::GraphBuilder cond_builder("cond_graph");
  auto cond_data_1 = cond_builder.AddNode("cond_data_1", "Data", 1, 1);
  auto cond_less_1 = cond_builder.AddNode("foo", "LessThan_5", 1, 1);
  auto cond_output_1 = cond_builder.AddNode("cond_output_1", "NetOutput", 1, 1);
  cond_builder.AddDataEdge(cond_data_1, 0, cond_less_1, 0);
  cond_builder.AddDataEdge(cond_less_1, 0, cond_output_1, 0);
  auto cond_graph = cond_builder.GetGraph();
  cond_output_1->GetOpDesc()->SetSrcName({"foo"});
  cond_output_1->GetOpDesc()->SetSrcIndex({0});
  cond_output_1->GetOpDesc()->UpdateInputDesc(0, GeTensorDesc(GeShape(), FORMAT_ND, DT_BOOL));
  AttrUtils::SetInt(cond_data_1->GetOpDesc(), ATTR_NAME_INDEX, 0);
  AttrUtils::SetInt(cond_data_1->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);

  // build body graph
  ut::GraphBuilder body_builder("body_graph");
  auto body_data_1 = body_builder.AddNode("body_data_1", "Data", 1, 1);
  auto body_add_1 = body_builder.AddNode("bar", "Add_1", 1, 1);
  // out_shape contains unknown dims (-1)
  auto body_output_1 = body_builder.AddNode("body_output_1", "NetOutput", 1, 1, FORMAT_NCHW, DT_FLOAT, shape);
  body_builder.AddDataEdge(body_data_1, 0, body_add_1, 0);
  body_builder.AddDataEdge(body_add_1, 0, body_output_1, 0);
  auto body_graph = body_builder.GetGraph();
  AttrUtils::SetInt(body_data_1->GetOpDesc(), ATTR_NAME_INDEX, 0);
  AttrUtils::SetInt(body_data_1->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
  AttrUtils::SetInt(body_output_1->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_PARENT_NODE_INDEX, 0);

  // setup parent graph and sub-graphs
  cond_graph->SetParentGraph(main_graph);
  cond_graph->SetParentNode(main_graph->FindNode("while_1"));
  body_graph->SetParentGraph(main_graph);
  body_graph->SetParentNode(main_graph->FindNode("while_1"));
  main_graph->FindNode("while_1")->GetOpDesc()->AddSubgraphName("cond");
  main_graph->FindNode("while_1")->GetOpDesc()->SetSubgraphInstanceName(0, cond_graph->GetName());
  main_graph->AddSubgraph("cond_graph", cond_graph);
  main_graph->FindNode("while_1")->GetOpDesc()->AddSubgraphName("body");
  main_graph->FindNode("while_1")->GetOpDesc()->SetSubgraphInstanceName(1, body_graph->GetName());
  main_graph->AddSubgraph("body_graph", body_graph);

  main_graph->SetGraphUnknownFlag(true);
  for (auto &subgraph : main_graph->GetAllSubgraphs()) {
    subgraph->SetGraphUnknownFlag(true);
  }

  return main_graph;
}
} // namespace

class UtestShapeRefiner : public testing::Test {
 protected:
  void SetUp() {
    dlog_setlevel(GE_MODULE_NAME, 0, 1);
    operator_infershape_funcs_bak_ = OperatorFactoryImpl::operator_infershape_funcs_;
    OperatorFactoryImpl::operator_infershape_funcs_.reset(new (std::nothrow) std::map<string, InferShapeFunc>());
  }
  void TearDown() {
    OperatorFactoryImpl::operator_infershape_funcs_ = operator_infershape_funcs_bak_;
  }
private:
  shared_ptr<map<string, InferShapeFunc>> operator_infershape_funcs_bak_;
};

TEST_F(UtestShapeRefiner, InferShapeAndTypeForRunning_Success) {
  OperatorFactoryImpl::operator_infershape_funcs_->emplace("Merge", [](Operator &op) { return GRAPH_SUCCESS; });
  OperatorFactoryImpl::operator_infershape_funcs_->emplace("Enter", [](Operator &op) { return GRAPH_SUCCESS; });

  const auto graph = std::make_shared<ComputeGraph>("test_infer_shape");
  auto enter1 = CreateNode(graph, "enter", "Enter", 1, 1);
  auto op_enter = OpDescUtils::CreateOperatorFromNode(enter1);
  EXPECT_EQ(ShapeRefiner::InferShapeAndTypeForRunning(enter1, op_enter, true), GRAPH_SUCCESS);

  auto merge1 = CreateNode(graph, "merge1", "StreamMerge", 2, 2);
  auto op = OpDescUtils::CreateOperatorFromNode(merge1);
  merge1->GetOpDesc()->AddInferFunc(nullptr);
  EXPECT_EQ(ShapeRefiner::InferShapeAndTypeForRunning(merge1, op, true), GRAPH_SUCCESS);
}

TEST_F(UtestShapeRefiner, InferShapeAndTypeForRunning_Failure_NullInferFunc) {
  const auto graph = std::make_shared<ComputeGraph>("test_infer_shape");
  
  OperatorFactoryImpl::operator_infershape_funcs_.reset(new (std::nothrow) std::map<string, InferShapeFunc>());
  auto merge1 = CreateNode(graph, "merge1", "StreamMerge", 2, 2);
  auto op = OpDescUtils::CreateOperatorFromNode(merge1);
  merge1->GetOpDesc()->AddInferFunc(nullptr);
  EXPECT_EQ(ShapeRefiner::InferShapeAndTypeForRunning(merge1, op, true), GRAPH_FAILED);
}

TEST_F(UtestShapeRefiner, CreateInferenceContext_Success_CrossSubgraph) {
  OperatorFactoryImpl::operator_infershape_funcs_->emplace("Relu", [](Operator &op) { return GRAPH_SUCCESS; });
  auto graph = CreateGraphWithMultiSubgraph();
  graph->SetGraphUnknownFlag(false);
  auto subgraph = graph->GetSubgraph("sub_graph1");
  auto relu = subgraph->FindNode("sub_relu1");

  EXPECT_EQ(ShapeRefiner::InferShapeAndType(relu, false), GRAPH_SUCCESS);
  auto in_data_node = relu->GetInDataNodes().at(0);
  int32_t out_idx = 0;
  std::map<NodePtr, int32_t> nodes_idx;
  auto ret = ShapeRefiner::GetRealInNodesAndIndex(in_data_node, out_idx, nodes_idx);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(nodes_idx.size(), 1);
  for (const auto &node_idx : nodes_idx) {
    EXPECT_EQ(node_idx.first->GetName(), "sub_relu2");
  }
}

TEST_F(UtestShapeRefiner, CreateInferenceContext_Success_CrossSubgraphDataToNetoutput) {
  OperatorFactoryImpl::operator_infershape_funcs_->emplace("Relu", [](Operator &op) { return GRAPH_SUCCESS; });
  auto graph = CreateGraphWithSubgraphDataToNetoutput();
  auto relu = graph->FindNode("relu2");

  EXPECT_EQ(ShapeRefiner::InferShapeAndType(relu, false), GRAPH_SUCCESS);
  auto in_data_node = relu->GetInDataNodes().at(0);
  int32_t out_idx = 0;
  std::map<NodePtr, int32_t> nodes_idx;
  auto ret = ShapeRefiner::GetRealInNodesAndIndex(in_data_node, out_idx, nodes_idx);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(nodes_idx.size(), 1);
  for (const auto &node_idx : nodes_idx) {
    EXPECT_EQ(node_idx.first->GetName(), "relu1");
  }
}

TEST_F(UtestShapeRefiner, InferShapeAndType_Failure_InvalidNode) {
  const auto graph = std::make_shared<ComputeGraph>("test_infer_shape");
  auto enter1 = CreateNode(graph, "enter", "Enter", 1, 1);

  EXPECT_EQ(ShapeRefiner::InferShapeAndType(enter1, true), GRAPH_FAILED);
}

// 看起来是无效ut
TEST_F(UtestShapeRefiner, UpdateOutputForMultiBatch) {
  auto graph = CreateGraphWithMultiSubgraph();
  graph->SetGraphUnknownFlag(false);
  auto subgraph = graph->GetSubgraph("sub_graph1");
  auto relu = subgraph->FindNode("sub_relu1");

  auto op = OpDescUtils::CreateOperatorFromNode(relu);
  auto ret = ShapeRefiner::InferShapeAndType(relu, op, false);
  EXPECT_EQ(ret, GRAPH_PARAM_INVALID);
}

TEST_F(UtestShapeRefiner, InferShapeAndType_Success_WithMultiSubgraphs) {
  OperatorFactoryImpl::operator_infershape_funcs_->emplace("Relu", [](Operator &op) { return GRAPH_SUCCESS; });
  auto graph = CreateGraphWithMultiSubgraph();
  graph->SetGraphUnknownFlag(false);
  auto subgraph = graph->GetSubgraph("sub_graph1");
  auto relu = subgraph->FindNode("sub_relu1");

  ShapeRefiner::ClearContextMap();

  auto subgraph3 = graph->GetSubgraph("sub_graph3");
  auto relu2 = subgraph3->FindNode("sub_relu2");

  InferenceContextPtr inference_context;
  ShapeRefiner::CreateInferenceContext(relu2, inference_context);
  ShapeRefiner::PushToContextMap(relu2, inference_context);

  auto ret = ShapeRefiner::InferShapeAndType(relu);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestShapeRefiner, InferShapeAndType_Success_SingleNode) {
  auto graph = std::make_shared<ComputeGraph>("test_infer_shape");
  auto node = CreateNode(graph, "enter", "Enter", 1, 1);
  auto op = OpDescUtils::CreateOperatorFromNode(node);
  bool before_subgraph = false;

  auto ret = ShapeRefiner::InferShapeAndType(node, op, before_subgraph);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestShapeRefiner, InferShapeAndType_Success_WithEmptySubgraph) {
  auto root_graph = std::make_shared<ComputeGraph>("test_infer_shape");
  auto root_node = CreateNode(root_graph, "enter", "Enter", 1, 1);
  auto op_desc = root_node->GetOpDesc();
  op_desc->AddSubgraphName("sub_graph");
  op_desc->SetSubgraphInstanceName(0, "sub_graph");

  auto subgraph = std::make_shared<ComputeGraph>("sub_graph");
  subgraph->SetParentNode(root_node);
  subgraph->SetParentGraph(root_graph);
  root_graph->AddSubgraph("sub_graph", subgraph);

  Operator op = OpDescUtils::CreateOperatorFromNode(root_node);

  auto ret = ShapeRefiner::InferShapeAndType(root_node, op, false);
  EXPECT_NE(ret, GRAPH_SUCCESS);
}

TEST_F(UtestShapeRefiner, InferShapeAndType_Success_WithSubgraph) {
  auto root_graph = std::make_shared<ComputeGraph>("test_infer_shape");
  NodePtr root_node = CreateNode(root_graph, "enter", "Enter", 1, 1);
  auto op_desc = root_node->GetOpDesc();
  op_desc->AddSubgraphName("sub_graph");
  op_desc->SetSubgraphInstanceName(0, "sub_graph");

  auto subgraph = std::make_shared<ComputeGraph>("sub_graph");
  NodePtr sub_node = CreateNode(subgraph, "netoutput", "Netoutput", 1, 1);
  auto sub_op_desc = sub_node->GetOpDesc();
  sub_op_desc->SetType(NETOUTPUT);

  subgraph->SetParentNode(root_node);
  subgraph->SetParentGraph(root_graph);
  root_graph->AddSubgraph("sub_graph", subgraph);

  Operator op = OpDescUtils::CreateOperatorFromNode(root_node);

  auto ret = ShapeRefiner::InferShapeAndType(root_node, op, false);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestShapeRefiner, InferShapeAndType_Success_CheckRangeForWhile) {
  auto root_graph = BuildSimpleWhileGraph2();
  auto while_node = root_graph->FindNode("while_1");
  auto op = OpDescUtils::CreateOperatorFromNode(while_node);
  EXPECT_EQ(ShapeRefiner::InferShapeAndType(while_node, op, false), GRAPH_SUCCESS);
  // verify the shape ranges of While's output tensor
  std::vector<std::pair<int64_t, int64_t>> x_range;
  std::vector<std::pair<int64_t, int64_t>> expected_range = {{0, -1}, {0, -1}, {224, 224}, {224, 224}};
  while_node->GetOpDesc()->MutableOutputDesc(0)->GetShapeRange(x_range);
  EXPECT_EQ(x_range, expected_range);
}

TEST_F(UtestShapeRefiner, InferShapeAndType_UpdateSubGraphDataNodes) {
  auto graph = CreateGraphWithMultiSubgraph();
  auto p1_node = graph->FindNode("partcall1");
  EXPECT_NE(p1_node, nullptr);
  (void)AttrUtils::SetBool(p1_node->GetOpDesc(), ATTR_NAME_NEED_INFER_AGAIN, true);
  EXPECT_EQ(ShapeRefiner::InferShapeAndType(p1_node, true), GRAPH_SUCCESS);
}
} // namespace ge
