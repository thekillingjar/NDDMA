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
#include <gmock/gmock.h>
#include <vector>
#include "common/ge_common/ge_inner_error_codes.h"
#include "common/checker.h"
#include "framework/common/types.h"
#include "macro_utils/dt_public_scope.h"
#include "engines/local_engine/ops_kernel_store/ge_local_ops_kernel_builder.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/passes/graph_builder_utils.h"
#include "macro_utils/dt_public_unscope.h"

using namespace std;
using namespace testing;

namespace ge {
namespace ge_local{
namespace {
/*
 *    data0   data1
 *       \    /
 *       add_0
 *         |
 *      bitcast_0
 *         |
 *      netoutput
 */
ComputeGraphPtr GetBitcastGraph(const std::vector<int64_t> &input_shape, const ge::DataType input_dtype,
                                const std::vector<int64_t> &bitcast_out_shape, const ge::DataType bitcast_out_dtype) {
  auto data_0 = OP_CFG("Data").TensorDesc(FORMAT_NCHW, input_dtype, input_shape);
  auto data_1 = OP_CFG("Data").TensorDesc(FORMAT_NCHW, input_dtype, input_shape);
  auto add_0 = OP_CFG("Add").TensorDesc(FORMAT_NCHW, input_dtype, input_shape);

  auto bitcast_0 =
      OP_CFG("Bitcast").TensorDesc(FORMAT_NCHW, bitcast_out_dtype, bitcast_out_shape).Attr("type", bitcast_out_dtype);

  DEF_GRAPH(g) {
                 CHAIN(NODE("data_0", data_0)->EDGE(0, 0)->NODE("add", add_0));
                 CHAIN(NODE("data_1", data_1)->EDGE(0, 1)->NODE("add", add_0));
                 CHAIN(NODE("add", add_0)->NODE("bitcast", bitcast_0)->NODE("netoutput", "Netoutput"));
               };

  auto graph = ToGeGraph(g);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  return compute_graph;
}

Status TestBitcastWhenCalcOpRunningParam(const std::vector<int64_t> &input_shape, const ge::DataType input_dtype,
                                         const std::vector<int64_t> &bitcast_out_shape,
                                         const ge::DataType bitcast_out_dtype, const bool graph_unknown_flag = false) {
  auto compute_graph = GetBitcastGraph(input_shape, input_dtype, bitcast_out_shape, bitcast_out_dtype);
  if (graph_unknown_flag) {
    compute_graph->SetGraphUnknownFlag(true);
  }
  auto node_bitcast = compute_graph->FindNode("bitcast");
  GE_ASSERT_NOTNULL(node_bitcast);
  auto p = std::make_shared<GeLocalOpsKernelBuilder>();
  GE_ASSERT_SUCCESS(p->CalcOpRunningParam(*node_bitcast));
  auto bitcast_opdesc = node_bitcast->GetOpDesc();
  GE_ASSERT_NOTNULL(bitcast_opdesc);
  bool reuse_input_flag = false;
  TensorUtils::GetReuseInput(bitcast_opdesc->GetOutputDesc(0), reuse_input_flag);
  GE_ASSERT_EQ(reuse_input_flag, !graph_unknown_flag);
  return SUCCESS;
}
} // namespace

class UtestGeLocalOpsKernelBuilder : public testing::Test {
 protected:
  void SetUp() {
  }
  void TearDown() {
  }
};

TEST_F(UtestGeLocalOpsKernelBuilder, Normal) {
  auto p = std::make_shared<GeLocalOpsKernelBuilder>();
  const std::map<std::string, std::string> options = {};
  EXPECT_EQ(p->Initialize(options), SUCCESS);
  OpDescPtr test_opdesc = std::make_shared<OpDesc>("test", "test");
  OpDescPtr test_opdesc1 = std::make_shared<OpDesc>("test1", "test1");
  OpDescPtr const_opdesc = std::make_shared<OpDesc>("Const", "Constant");
  ComputeGraphPtr test_graph;
  GeTensorDesc te_desc1(GeShape({4, 5, 6, 7}), FORMAT_NCHW, DT_FLOAT);
  GeTensorDesc te_desc2(GeShape({4, 5, 6, 7}), FORMAT_NCHW, DT_STRING);
  GeTensorDesc te_desc3(GeShape({-1, 5, 6, 7}), FORMAT_NCHW, DT_FLOAT);
  GeTensorDesc te_desc4(GeShape({4, 5, 6, 7}), FORMAT_NCHW, DT_FLOAT);
  GeTensorDesc te_desc5(GeShape({-4, 5, 6, 7}), FORMAT_NCHW, DT_FLOAT);
  test_opdesc->AddInputDesc("x", te_desc3);
  NodePtr test_node = std::make_shared<Node>(test_opdesc, test_graph);
  test_opdesc1->AddInputDesc("x", te_desc1);
  test_opdesc1->AddOutputDesc("y1", te_desc1);
  TensorUtils::SetSize(te_desc4, 3360);
  test_opdesc1->AddOutputDesc("y2", te_desc4);
  test_opdesc1->AddOutputDesc("y3", te_desc5);
  NodePtr test_node1 = std::make_shared<Node>(test_opdesc1, test_graph);
  NodePtr test_node2 = std::make_shared<Node>(const_opdesc, test_graph);
  const_opdesc->AddOutputDesc("y", te_desc2);
  Node node1;
  EXPECT_EQ(p->CalcOpRunningParam(node1), FAILED);
  EXPECT_EQ(p->CalcOpRunningParam(*test_node), SUCCESS);
  EXPECT_EQ(p->CalcOpRunningParam(*test_node1), FAILED);
  EXPECT_EQ(p->CalcOpRunningParam(*test_node2), FAILED);
  ConstGeTensorPtr value = make_shared<const GeTensor>();
  int32_t const_value = 0;
  GeTensorPtr weight_value = make_shared<GeTensor>(te_desc1, reinterpret_cast<uint8_t *>(&const_value), sizeof(int32_t));
  AttrUtils::SetTensor(const_opdesc, "value", weight_value);
  OpDescPtr date_opdesc = std::make_shared<OpDesc>("Data", "Data");
  OpDescPtr empty_opdesc = nullptr;
  ComputeGraphPtr const_graph;
  NodePtr const_node = std::make_shared<Node>(const_opdesc, const_graph);
  int64_t outputsize;
  EXPECT_EQ(p->CalcMemSizeByNodeType(empty_opdesc, te_desc1, outputsize, "Constant"), SUCCESS);
  EXPECT_EQ(p->CalcMemSizeByNodeType(empty_opdesc, te_desc2, outputsize, "Constant"), PARAM_INVALID);
  EXPECT_EQ(p->CalcMemSizeByNodeType(const_opdesc, te_desc2, outputsize, "Constant"), SUCCESS);
  EXPECT_EQ(p->CalcMemSizeByNodeType(const_opdesc, te_desc1, outputsize, "Constant"), SUCCESS);
  EXPECT_EQ(p->CalcMemSizeByNodeType(date_opdesc, te_desc1, outputsize, "Data"), SUCCESS);

  ge::AttrUtils::SetBool(te_desc1, "_tensor_no_tiling_mem_type", true);
  EXPECT_EQ(p->CalcMemSizeByNodeType(date_opdesc, te_desc1, outputsize, "Constant"), SUCCESS);
}

TEST_F(UtestGeLocalOpsKernelBuilder, GenerateTask) {
  auto p = std::make_shared<GeLocalOpsKernelBuilder>();
  RunContext runContext;
  std::vector<domi::TaskDef> tasks;
  OpDescPtr test_opdesc = std::make_shared<OpDesc>("test", "test");
  ComputeGraphPtr test_graph;
  NodePtr test_node = std::make_shared<Node>(test_opdesc, test_graph);
  EXPECT_EQ(p->GenerateTask(*test_node, runContext, tasks), FAILED);
  OpDescPtr test_opdesc1 = std::make_shared<OpDesc>("test", "StackPop");
  NodePtr test_node1 = std::make_shared<Node>(test_opdesc1, test_graph);
  EXPECT_EQ(p->GenerateTask(*test_node1, runContext, tasks), SUCCESS);

  GeTensorDesc te_desc(GeShape({-1, 5, 6, 7}), FORMAT_NCHW, DT_FLOAT);
  test_opdesc->AddInputDesc(te_desc);
  EXPECT_EQ(p->GenerateTask(*test_node, runContext, tasks), SUCCESS);

  OpDescPtr test_opdesc2 = std::make_shared<OpDesc>("test", "NoOp");
  NodePtr test_node2 = std::make_shared<Node>(test_opdesc2, test_graph);
  EXPECT_EQ(p->GenerateTask(*test_node2, runContext, tasks), SUCCESS);

  OpDescPtr test_opdesc3 = std::make_shared<OpDesc>("test", "Reshape");
  NodePtr test_node3 = std::make_shared<Node>(test_opdesc3, test_graph);
  EXPECT_EQ(p->GenerateTask(*test_node3, runContext, tasks), SUCCESS);

  OpDescPtr test_opdesc4 = std::make_shared<OpDesc>("test", "ExpandDims");
  NodePtr test_node4 = std::make_shared<Node>(test_opdesc4, test_graph);
  EXPECT_EQ(p->GenerateTask(*test_node4, runContext, tasks), SUCCESS);

  OpDescPtr test_opdesc5 = std::make_shared<OpDesc>("test", "ReFormat");
  NodePtr test_node5 = std::make_shared<Node>(test_opdesc5, test_graph);
  EXPECT_EQ(p->GenerateTask(*test_node5, runContext, tasks), SUCCESS);

  OpDescPtr test_opdesc6 = std::make_shared<OpDesc>("test", "Squeeze");
  NodePtr test_node6 = std::make_shared<Node>(test_opdesc6, test_graph);
  EXPECT_EQ(p->GenerateTask(*test_node6, runContext, tasks), SUCCESS);

  OpDescPtr test_opdesc7 = std::make_shared<OpDesc>("test", "Unsqueeze");
  NodePtr test_node7 = std::make_shared<Node>(test_opdesc7, test_graph);
  EXPECT_EQ(p->GenerateTask(*test_node7, runContext, tasks), SUCCESS);

  OpDescPtr test_opdesc8 = std::make_shared<OpDesc>("test", "SqueezeV2");
  NodePtr test_node8 = std::make_shared<Node>(test_opdesc8, test_graph);
  EXPECT_EQ(p->GenerateTask(*test_node8, runContext, tasks), SUCCESS);

  OpDescPtr test_opdesc9 = std::make_shared<OpDesc>("test", "UnsqueezeV2");
  NodePtr test_node9 = std::make_shared<Node>(test_opdesc9, test_graph);
  EXPECT_EQ(p->GenerateTask(*test_node9, runContext, tasks), SUCCESS);

  OpDescPtr test_opdesc10 = std::make_shared<OpDesc>("test", "SqueezeV3");
  NodePtr test_node10 = std::make_shared<Node>(test_opdesc10, test_graph);
  EXPECT_EQ(p->GenerateTask(*test_node10, runContext, tasks), SUCCESS);

  OpDescPtr test_opdesc11 = std::make_shared<OpDesc>("test", "UnsqueezeV3");
  NodePtr test_node11 = std::make_shared<Node>(test_opdesc11, test_graph);
  EXPECT_EQ(p->GenerateTask(*test_node11, runContext, tasks), SUCCESS);

  OpDescPtr test_opdesc12 = std::make_shared<OpDesc>("test", "FlattenV2");
  NodePtr test_node12 = std::make_shared<Node>(test_opdesc12, test_graph);
  EXPECT_EQ(p->GenerateTask(*test_node12, runContext, tasks), SUCCESS);
}

TEST_F(UtestGeLocalOpsKernelBuilder, SetSizeForStringInput) {
  auto p = std::make_shared<GeLocalOpsKernelBuilder>();
  DEF_GRAPH(test1) {
    auto add1 = OP_CFG("Add").TensorDesc(FORMAT_ND, DT_STRING, {200, 200, 3})
                           .Attr("_op_max_shape", "200,200,3");
    auto data1 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_STRING, {0});
    auto data2 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_STRING, {0});
    CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("add_1", add1));
    CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("add_1", add1));
  };

  auto graph = ToGeGraph(test1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto node = compute_graph->FindNode("add_1");
  EXPECT_EQ(ge::AttrUtils::SetListInt(node->GetOpDesc(), "_op_max_size", {1000}), true);

  auto node_data = compute_graph->FindNode("data_2");
  GeTensorDesc output_tensor = node_data->GetOpDesc()->GetOutputDesc(0);
  EXPECT_EQ(p->SetSizeForStringInput(*node_data, output_tensor, 0), FAILED);

  EXPECT_EQ(ge::AttrUtils::SetListInt(node->GetOpDesc(), "_op_max_size", {1000, 1000}), true);
  EXPECT_EQ(p->SetSizeForStringInput(*node_data, output_tensor, 0), SUCCESS);
}

TEST_F(UtestGeLocalOpsKernelBuilder, StaticGraph_Bitcast_OutputReuseInput) {
  const std::vector<int64_t> input_shape = {1, 5, 4, 4};
  const ge::DataType input_dtype = DT_INT32;
  const std::vector<int64_t> bitcast_out_shape = {1, 5, 4, 4, 8};
  const ge::DataType bitcast_out_dtype = DT_INT4;
  bool graph_unknown_flag = false;
  EXPECT_EQ(TestBitcastWhenCalcOpRunningParam(input_shape, input_dtype, bitcast_out_shape, bitcast_out_dtype,
                                              graph_unknown_flag),
            SUCCESS);
}

TEST_F(UtestGeLocalOpsKernelBuilder, DynamicGraph_Bitcast_OutputNotReuseInput) {
  const std::vector<int64_t> input_shape = {-1, 5, 4, -1};
  const ge::DataType input_dtype = DT_INT4;
  const std::vector<int64_t> bitcast_out_shape = {-1, 5, 4};
  const ge::DataType bitcast_out_dtype = DT_INT32;
  bool graph_unknown_flag = true;
  EXPECT_EQ(TestBitcastWhenCalcOpRunningParam(input_shape, input_dtype, bitcast_out_shape, bitcast_out_dtype,
                                              graph_unknown_flag),
            SUCCESS);
}

/*
 *  mul1 mul2 mul3 mul4  
 *    \    |   |   /
 *     phony_concat
 *          |
 *         add
 *          |
 *      netoutput
 */
ComputeGraphPtr GetPhonyConcatComputeGraph() {
  DEF_GRAPH(test1) {
    auto add = OP_CFG("Add").TensorDesc(FORMAT_ND, DT_FLOAT16, {1, 4, 2, 8, 8});
    auto mul1 = OP_CFG("Mul").TensorDesc(FORMAT_ND, DT_FLOAT16, {1, 2, 2, 4, 8});
    auto mul2 = OP_CFG("Mul").TensorDesc(FORMAT_ND, DT_FLOAT16, {1, 2, 2, 4, 8});
    auto mul3 = OP_CFG("Mul").TensorDesc(FORMAT_ND, DT_FLOAT16, {1, 2, 2, 4, 8});
    auto mul4 = OP_CFG("Mul").TensorDesc(FORMAT_ND, DT_FLOAT16, {1, 2, 2, 4, 8});
    auto phony_concat = OP_CFG("PhonyConcat").Attr("concat_dim", std::vector<int64_t>{3, 1}).Attr("N", std::vector<int64_t>{2, 2});

    CHAIN(NODE("mul1", mul1)->EDGE(0, 0)->NODE("phony_concat", phony_concat)->
          NODE("add", add)->NODE("netoutput", "NetOutput"));
    CHAIN(NODE("mul2", mul2)->EDGE(0, 1)->NODE("phony_concat", phony_concat));
    CHAIN(NODE("mul3", mul3)->EDGE(0, 2)->NODE("phony_concat", phony_concat));
    CHAIN(NODE("mul4", mul4)->EDGE(0, 3)->NODE("phony_concat", phony_concat));
  };

  auto graph = ToGeGraph(test1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);

  auto node_mul1 = compute_graph->FindNode("mul1");
  auto node_mul2 = compute_graph->FindNode("mul2");
  auto node_mul3 = compute_graph->FindNode("mul3");
  auto node_mul4 = compute_graph->FindNode("mul4");
  node_mul1->GetOpDesc()->MutableOutputDesc(0)->SetFormat(FORMAT_ND);
  node_mul1->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_FLOAT16);
  node_mul1->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({1, 2, 2, 4, 8}));
  node_mul2->GetOpDesc()->MutableOutputDesc(0)->SetFormat(FORMAT_ND);
  node_mul2->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_FLOAT16);
  node_mul2->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({1, 2, 2, 4, 8}));
  node_mul3->GetOpDesc()->MutableOutputDesc(0)->SetFormat(FORMAT_ND);
  node_mul3->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_FLOAT16);
  node_mul3->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({1, 2, 2, 4, 8}));
  node_mul4->GetOpDesc()->MutableOutputDesc(0)->SetFormat(FORMAT_ND);
  node_mul4->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_FLOAT16);
  node_mul4->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({1, 2, 2, 4, 8}));

  auto node_phony_concat = compute_graph->FindNode("phony_concat");
  node_phony_concat->GetOpDesc()->MutableInputDesc(0)->SetShape(GeShape({1, 2, 2, 4, 8}));
  node_phony_concat->GetOpDesc()->MutableInputDesc(1)->SetShape(GeShape({1, 2, 2, 4, 8}));
  node_phony_concat->GetOpDesc()->MutableInputDesc(2)->SetShape(GeShape({1, 2, 2, 4, 8}));
  node_phony_concat->GetOpDesc()->MutableInputDesc(3)->SetShape(GeShape({1, 2, 2, 4, 8}));
  return compute_graph;
}

/*
 *     mul1   mul2  
 *       \    /
 *     phony_concat
 *          |
 *         add
 *          |
 *      netoutput
 */
ComputeGraphPtr GetPhonyConcatNot32AlignComputeGraph() {
  DEF_GRAPH(test1) {
    auto add = OP_CFG("Add").TensorDesc(FORMAT_ND, DT_INT8, {4, 3});
    auto mul1 = OP_CFG("Mul").TensorDesc(FORMAT_ND, DT_INT8, {2, 3});
    auto mul2 = OP_CFG("Mul").TensorDesc(FORMAT_ND, DT_INT8, {2, 3});
    auto phony_concat = OP_CFG("PhonyConcat").Attr("concat_dim", std::vector<int64_t>{0}).Attr("N", std::vector<int64_t>{2});

    CHAIN(NODE("mul1", mul1)->EDGE(0, 0)->NODE("phony_concat", phony_concat)->
          NODE("add", add)->NODE("netoutput", "NetOutput"));
    CHAIN(NODE("mul2", mul2)->EDGE(0, 1)->NODE("phony_concat", phony_concat));
  };

  auto graph = ToGeGraph(test1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);

  auto node_mul1 = compute_graph->FindNode("mul1");
  auto node_mul2 = compute_graph->FindNode("mul2");
  node_mul1->GetOpDesc()->MutableOutputDesc(0)->SetFormat(FORMAT_ND);
  node_mul1->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_INT8);
  node_mul1->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({2, 3}));
  node_mul2->GetOpDesc()->MutableOutputDesc(0)->SetFormat(FORMAT_ND);
  node_mul2->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_INT8);
  node_mul2->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({2, 3}));

  auto node_phony_concat = compute_graph->FindNode("phony_concat");
  node_phony_concat->GetOpDesc()->MutableInputDesc(0)->SetShape(GeShape({2, 3}));
  node_phony_concat->GetOpDesc()->MutableInputDesc(0)->SetDataType(DT_INT8);
  node_phony_concat->GetOpDesc()->MutableInputDesc(1)->SetShape(GeShape({2, 3}));
  node_phony_concat->GetOpDesc()->MutableInputDesc(1)->SetDataType(DT_INT8);
  return compute_graph;
}

TEST_F(UtestGeLocalOpsKernelBuilder, PhonyConcatCalcOffsetNot32Align) {
  auto p = std::make_shared<GeLocalOpsKernelBuilder>();

  auto compute_graph = GetPhonyConcatNot32AlignComputeGraph();
  auto node_phony_concat = compute_graph->FindNode("phony_concat");

  EXPECT_NE(p->CalcOpRunningParam(*node_phony_concat), SUCCESS);
}

/*
 *        add
 *         |
 *     phony_split
 *     /   |  |   \
 *  mul1 mul2 mul3 mul4
 *     \   |  |   /
 *      netoutput
 */
ComputeGraphPtr GetPhonySplitComputeGraph() {
  DEF_GRAPH(test1) {
    auto add = OP_CFG("Add").TensorDesc(FORMAT_ND, DT_FLOAT16, {1, 4, 2, 8, 8});
    auto mul1 = OP_CFG("Mul").TensorDesc(FORMAT_ND, DT_FLOAT16, {1, 2, 2, 4, 8});
    auto mul2 = OP_CFG("Mul").TensorDesc(FORMAT_ND, DT_FLOAT16, {1, 2, 2, 4, 8});
    auto mul3 = OP_CFG("Mul").TensorDesc(FORMAT_ND, DT_FLOAT16, {1, 2, 2, 4, 8});
    auto mul4 = OP_CFG("Mul").TensorDesc(FORMAT_ND, DT_FLOAT16, {1, 2, 2, 4, 8});
    auto phony_split = OP_CFG("PhonySplit").Attr("split_dim", std::vector<int64_t>{1, 3}).Attr("num_split", std::vector<int64_t>{2, 2});

    CHAIN(NODE("add", add)->NODE("phony_split", phony_split)->NODE("mul1", mul1)->NODE("netoutput", "NetOutput"));
    CHAIN(NODE("phony_split", phony_split)->NODE("mul2", mul2)->NODE("netoutput", "NetOutput"));
    CHAIN(NODE("phony_split", phony_split)->NODE("mul3", mul3)->NODE("netoutput", "NetOutput"));
    CHAIN(NODE("phony_split", phony_split)->NODE("mul4", mul4)->NODE("netoutput", "NetOutput"));
  };

  auto graph = ToGeGraph(test1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);

  auto node_mul1 = compute_graph->FindNode("mul1");
  auto node_mul2 = compute_graph->FindNode("mul2");
  auto node_mul3 = compute_graph->FindNode("mul3");
  auto node_mul4 = compute_graph->FindNode("mul4");
  node_mul1->GetOpDesc()->MutableInputDesc(0)->SetFormat(FORMAT_ND);
  node_mul1->GetOpDesc()->MutableInputDesc(0)->SetDataType(DT_FLOAT16);
  node_mul1->GetOpDesc()->MutableInputDesc(0)->SetShape(GeShape({1, 2, 2, 4, 8}));
  node_mul2->GetOpDesc()->MutableInputDesc(0)->SetFormat(FORMAT_ND);
  node_mul2->GetOpDesc()->MutableInputDesc(0)->SetDataType(DT_FLOAT16);
  node_mul2->GetOpDesc()->MutableInputDesc(0)->SetShape(GeShape({1, 2, 2, 4, 8}));
  node_mul3->GetOpDesc()->MutableInputDesc(0)->SetFormat(FORMAT_ND);
  node_mul3->GetOpDesc()->MutableInputDesc(0)->SetDataType(DT_FLOAT16);
  node_mul3->GetOpDesc()->MutableInputDesc(0)->SetShape(GeShape({1, 2, 2, 4, 8}));
  node_mul4->GetOpDesc()->MutableInputDesc(0)->SetFormat(FORMAT_ND);
  node_mul4->GetOpDesc()->MutableInputDesc(0)->SetDataType(DT_FLOAT16);
  node_mul4->GetOpDesc()->MutableInputDesc(0)->SetShape(GeShape({1, 2, 2, 4, 8}));

  auto node_phony_split = compute_graph->FindNode("phony_split");
  node_phony_split->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({1, 2, 2, 4, 8}));
  node_phony_split->GetOpDesc()->MutableOutputDesc(1)->SetShape(GeShape({1, 2, 2, 4, 8}));
  node_phony_split->GetOpDesc()->MutableOutputDesc(2)->SetShape(GeShape({1, 2, 2, 4, 8}));
  node_phony_split->GetOpDesc()->MutableOutputDesc(3)->SetShape(GeShape({1, 2, 2, 4, 8}));
  return compute_graph;
}

TEST_F(UtestGeLocalOpsKernelBuilder, PhonyConcatCalcOffset) {
  auto p = std::make_shared<GeLocalOpsKernelBuilder>();

  auto compute_graph = GetPhonyConcatComputeGraph();
  auto node_phony_concat = compute_graph->FindNode("phony_concat");
  auto node_mul1 = compute_graph->FindNode("mul1");
  auto node_mul2 = compute_graph->FindNode("mul2");
  auto node_mul3 = compute_graph->FindNode("mul3");
  auto node_mul4 = compute_graph->FindNode("mul4");

  EXPECT_EQ(p->CalcOpRunningParam(*node_phony_concat), SUCCESS);

  std::vector<int64_t> outputs_offset;
  EXPECT_EQ(ge::AttrUtils::GetListInt(node_mul1->GetOpDesc(), "_output_offset_list_for_continuous", outputs_offset), true);
  EXPECT_EQ(outputs_offset.size(), 1);
  if (outputs_offset.size() == 1) {
    EXPECT_EQ(outputs_offset[0], 0);
  }
  EXPECT_EQ(ge::AttrUtils::GetListInt(node_mul2->GetOpDesc(), "_output_offset_list_for_continuous", outputs_offset), true);
  EXPECT_EQ(outputs_offset.size(), 1);
  if (outputs_offset.size() == 1) {
    EXPECT_EQ(outputs_offset[0], 64);
  }
  EXPECT_EQ(ge::AttrUtils::GetListInt(node_mul3->GetOpDesc(), "_output_offset_list_for_continuous", outputs_offset), true);
  EXPECT_EQ(outputs_offset.size(), 1);
  if (outputs_offset.size() == 1) {
    EXPECT_EQ(outputs_offset[0], 512);
  }
  EXPECT_EQ(ge::AttrUtils::GetListInt(node_mul4->GetOpDesc(), "_output_offset_list_for_continuous", outputs_offset), true);
  EXPECT_EQ(outputs_offset.size(), 1);
  if (outputs_offset.size() == 1) {
    EXPECT_EQ(outputs_offset[0], 576);
  }
}

TEST_F(UtestGeLocalOpsKernelBuilder, PhonyConcatCalcOffsetInvalidDim) {
  auto p = std::make_shared<GeLocalOpsKernelBuilder>();

  auto compute_graph = GetPhonyConcatComputeGraph();
  auto node_phony_concat = compute_graph->FindNode("phony_concat");
  EXPECT_EQ(ge::AttrUtils::SetListInt(node_phony_concat->GetOpDesc(), "N", {2, 1}), true);
  EXPECT_NE(p->CalcOpRunningParam(*node_phony_concat), SUCCESS);

  EXPECT_EQ(ge::AttrUtils::SetListInt(node_phony_concat->GetOpDesc(), "N", {2, 2}), true);
  EXPECT_EQ(ge::AttrUtils::SetListInt(node_phony_concat->GetOpDesc(), "concat_dim", {-2, 1}), true);
  EXPECT_EQ(p->CalcOpRunningParam(*node_phony_concat), SUCCESS);
}

TEST_F(UtestGeLocalOpsKernelBuilder, PhonyConcatCalcOffsetNotKeepOffset) {
  auto p = std::make_shared<GeLocalOpsKernelBuilder>();

  auto compute_graph = GetPhonyConcatComputeGraph();
  auto node_phony_concat = compute_graph->FindNode("phony_concat");
  EXPECT_EQ(ge::AttrUtils::SetBool(node_phony_concat->GetOpDesc(), "keep_input_offset", false), true);

  auto node_mul1 = compute_graph->FindNode("mul1");
  auto node_mul2 = compute_graph->FindNode("mul2");
  auto node_mul3 = compute_graph->FindNode("mul3");
  auto node_mul4 = compute_graph->FindNode("mul4");

  EXPECT_EQ(p->CalcOpRunningParam(*node_phony_concat), SUCCESS);

  std::vector<int64_t> outputs_offset;
  EXPECT_EQ(ge::AttrUtils::GetListInt(node_mul1->GetOpDesc(), "_output_offset_list_for_continuous", outputs_offset), true);
  EXPECT_EQ(outputs_offset.size(), 1);
  if (outputs_offset.size() == 1) {
    EXPECT_EQ(outputs_offset[0], 0);
  }
  EXPECT_EQ(ge::AttrUtils::GetListInt(node_mul2->GetOpDesc(), "_output_offset_list_for_continuous", outputs_offset), true);
  EXPECT_EQ(outputs_offset.size(), 1);
  if (outputs_offset.size() == 1) {
    EXPECT_EQ(outputs_offset[0], 0);
  }
  EXPECT_EQ(ge::AttrUtils::GetListInt(node_mul3->GetOpDesc(), "_output_offset_list_for_continuous", outputs_offset), true);
  EXPECT_EQ(outputs_offset.size(), 1);
  if (outputs_offset.size() == 1) {
    EXPECT_EQ(outputs_offset[0], 0);
  }
  EXPECT_EQ(ge::AttrUtils::GetListInt(node_mul4->GetOpDesc(), "_output_offset_list_for_continuous", outputs_offset), true);
  EXPECT_EQ(outputs_offset.size(), 1);
  if (outputs_offset.size() == 1) {
    EXPECT_EQ(outputs_offset[0], 0);
  }
}

TEST_F(UtestGeLocalOpsKernelBuilder, PhonyConcatSkipProcess) {
  auto p = std::make_shared<GeLocalOpsKernelBuilder>();

  auto compute_graph = GetPhonyConcatComputeGraph();
  auto node_phony_concat = compute_graph->FindNode("phony_concat");
  auto node_mul1 = compute_graph->FindNode("mul1");
  auto node_mul2 = compute_graph->FindNode("mul2");
  auto node_mul3 = compute_graph->FindNode("mul3");
  auto node_mul4 = compute_graph->FindNode("mul4");
  EXPECT_EQ(ge::AttrUtils::SetListInt(node_mul1->GetOpDesc(), "_output_offset_for_buffer_fusion", {0}), true);
  EXPECT_EQ(ge::AttrUtils::SetListInt(node_mul2->GetOpDesc(), "_output_offset_for_buffer_fusion", {0}), true);
  EXPECT_EQ(ge::AttrUtils::SetListInt(node_mul3->GetOpDesc(), "_output_offset_for_buffer_fusion", {0}), true);
  EXPECT_EQ(ge::AttrUtils::SetListInt(node_mul4->GetOpDesc(), "_output_offset_for_buffer_fusion", {0}), true);

  EXPECT_EQ(p->CalcOpRunningParam(*node_phony_concat), SUCCESS);

  std::vector<int64_t> outputs_offset;
  EXPECT_EQ(ge::AttrUtils::GetListInt(node_mul1->GetOpDesc(), "_output_offset_list_for_continuous", outputs_offset), false);
  EXPECT_EQ(ge::AttrUtils::GetListInt(node_mul2->GetOpDesc(), "_output_offset_list_for_continuous", outputs_offset), false);
  EXPECT_EQ(ge::AttrUtils::GetListInt(node_mul3->GetOpDesc(), "_output_offset_list_for_continuous", outputs_offset), false);
  EXPECT_EQ(ge::AttrUtils::GetListInt(node_mul4->GetOpDesc(), "_output_offset_list_for_continuous", outputs_offset), false);
}

TEST_F(UtestGeLocalOpsKernelBuilder, PhonySplitCalcOffset) {
  auto p = std::make_shared<GeLocalOpsKernelBuilder>();

  auto compute_graph = GetPhonySplitComputeGraph();
  auto node_phony_split = compute_graph->FindNode("phony_split");
  auto node_mul1 = compute_graph->FindNode("mul1");
  auto node_mul2 = compute_graph->FindNode("mul2");
  auto node_mul3 = compute_graph->FindNode("mul3");
  auto node_mul4 = compute_graph->FindNode("mul4");

  EXPECT_EQ(p->CalcOpRunningParam(*node_phony_split), SUCCESS);

  std::vector<int64_t> inputs_offset;
  EXPECT_EQ(ge::AttrUtils::GetListInt(node_mul1->GetOpDesc(), "_input_offset_list_for_continuous", inputs_offset), true);
  EXPECT_EQ(inputs_offset.size(), 1);
  if (inputs_offset.size() == 1) {
    EXPECT_EQ(inputs_offset[0], 0);
  }
  EXPECT_EQ(ge::AttrUtils::GetListInt(node_mul2->GetOpDesc(), "_input_offset_list_for_continuous", inputs_offset), true);
  EXPECT_EQ(inputs_offset.size(), 1);
  if (inputs_offset.size() == 1) {
    EXPECT_EQ(inputs_offset[0], 64);
  }
  EXPECT_EQ(ge::AttrUtils::GetListInt(node_mul3->GetOpDesc(), "_input_offset_list_for_continuous", inputs_offset), true);
  EXPECT_EQ(inputs_offset.size(), 1);
  if (inputs_offset.size() == 1) {
    EXPECT_EQ(inputs_offset[0], 512);
  }
  EXPECT_EQ(ge::AttrUtils::GetListInt(node_mul4->GetOpDesc(), "_input_offset_list_for_continuous", inputs_offset), true);
  EXPECT_EQ(inputs_offset.size(), 1);
  if (inputs_offset.size() == 1) {
    EXPECT_EQ(inputs_offset[0], 576);
  }
}

TEST_F(UtestGeLocalOpsKernelBuilder, PhonySplitCalcOffsetNotKeepOffset) {
  auto p = std::make_shared<GeLocalOpsKernelBuilder>();

  auto compute_graph = GetPhonySplitComputeGraph();
  auto node_phony_split = compute_graph->FindNode("phony_split");
  EXPECT_EQ(ge::AttrUtils::SetBool(node_phony_split->GetOpDesc(), "keep_output_offset", false), true);

  auto node_mul1 = compute_graph->FindNode("mul1");
  auto node_mul2 = compute_graph->FindNode("mul2");
  auto node_mul3 = compute_graph->FindNode("mul3");
  auto node_mul4 = compute_graph->FindNode("mul4");

  EXPECT_EQ(p->CalcOpRunningParam(*node_phony_split), SUCCESS);

  std::vector<int64_t> outputs_offset;
  EXPECT_EQ(ge::AttrUtils::GetListInt(node_mul1->GetOpDesc(), "_input_offset_list_for_continuous", outputs_offset), true);
  EXPECT_EQ(outputs_offset.size(), 1);
  if (outputs_offset.size() == 1) {
    EXPECT_EQ(outputs_offset[0], 0);
  }
  EXPECT_EQ(ge::AttrUtils::GetListInt(node_mul2->GetOpDesc(), "_input_offset_list_for_continuous", outputs_offset), true);
  EXPECT_EQ(outputs_offset.size(), 1);
  if (outputs_offset.size() == 1) {
    EXPECT_EQ(outputs_offset[0], 0);
  }
  EXPECT_EQ(ge::AttrUtils::GetListInt(node_mul3->GetOpDesc(), "_input_offset_list_for_continuous", outputs_offset), true);
  EXPECT_EQ(outputs_offset.size(), 1);
  if (outputs_offset.size() == 1) {
    EXPECT_EQ(outputs_offset[0], 0);
  }
  EXPECT_EQ(ge::AttrUtils::GetListInt(node_mul4->GetOpDesc(), "_input_offset_list_for_continuous", outputs_offset), true);
  EXPECT_EQ(outputs_offset.size(), 1);
  if (outputs_offset.size() == 1) {
    EXPECT_EQ(outputs_offset[0], 0);
  }
}

TEST_F(UtestGeLocalOpsKernelBuilder, PhonyConcatSplitInDynamicGraph) {
  auto p = std::make_shared<GeLocalOpsKernelBuilder>();

  auto compute_graph1 = GetPhonySplitComputeGraph();
  compute_graph1->SetGraphUnknownFlag(true);
  auto node_phony_split = compute_graph1->FindNode("phony_split");
  EXPECT_NE(p->CalcOpRunningParam(*node_phony_split), SUCCESS);

  auto compute_graph2 = GetPhonyConcatComputeGraph();
  compute_graph2->SetGraphUnknownFlag(true);
  auto node_phony_concat = compute_graph2->FindNode("phony_concat");
  EXPECT_NE(p->CalcOpRunningParam(*node_phony_concat), SUCCESS);
}

TEST_F(UtestGeLocalOpsKernelBuilder, PhonyConcat_CheckSize_32Padding) {
  auto p = std::make_shared<GeLocalOpsKernelBuilder>();
  auto compute_graph = GetPhonyConcatComputeGraph();
  auto node_phony_concat = compute_graph->FindNode("phony_concat");
  node_phony_concat->GetOpDesc()->MutableOutputDesc(0)->SetFormat(FORMAT_ND);
  node_phony_concat->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_FLOAT16);
  node_phony_concat->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({4, 2, 2, 4, 8}));
  EXPECT_EQ(p->CalcOpRunningParam(*node_phony_concat), SUCCESS);
  const auto output_tensor_desc = node_phony_concat->GetOpDescBarePtr()->MutableOutputDesc(static_cast<uint32_t>(0));
  int64_t mem_size = 0;
  ge::TensorUtils::GetSize(*output_tensor_desc, mem_size);
  EXPECT_EQ(mem_size, 1056);
}

TEST_F(UtestGeLocalOpsKernelBuilder, PartitionedCall_CheckSize_32Padding) {
  auto partitioned_call = OP_CFG(PARTITIONEDCALL)
                              .InCnt(2)
                              .OutCnt(1)
                              .TensorDesc(FORMAT_NCHW, DT_FLOAT, {4, 50, 50, 8})
                              .Attr(ATTR_NAME_SUBGRAPH_MULTI_DIMS_INDEX, 0);
  auto data_1 = OP_CFG(DATA).TensorDesc(FORMAT_NCHW, DT_FLOAT, {4, 50, 50, 8});
  auto data_2 = OP_CFG(DATA).TensorDesc(FORMAT_NCHW, DT_FLOAT, {4, 50, 50, 8});
  auto data_3 = OP_CFG(DATA).TensorDesc(FORMAT_NCHW, DT_FLOAT, {4, 50, 50, 8});
  auto add = OP_CFG(ADD).TensorDesc(FORMAT_NCHW, DT_FLOAT, {4, 50, 50, 8});
  auto cast = OP_CFG(CAST).TensorDesc(FORMAT_NCHW, DT_INT32, {4, 50, 50, 8});
  auto net_output = OP_CFG(NETOUTPUT).TensorDesc(FORMAT_NCHW, DT_FLOAT, {4, 50, 50, 8});
  DEF_GRAPH(g1) {
    CHAIN(NODE("data_1", data_1)->EDGE(0, 0)->NODE("add", add));
    CHAIN(NODE("data_2", data_2)->EDGE(0, 1)->NODE("add"));
    CHAIN(NODE("data_3", data_3)->EDGE(0, 0)->NODE("partitioned_call", partitioned_call));
    CHAIN(NODE("add")->EDGE(0, 1)->NODE("partitioned_call")->EDGE(0, 0)->NODE("cast", cast));
    CHAIN(NODE("cast")->EDGE(0, 0)->NODE("net_output", net_output));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);

  auto p = std::make_shared<GeLocalOpsKernelBuilder>();
  auto partitioned_call_node = compute_graph->FindNode("partitioned_call");
  EXPECT_EQ(p->CalcOpRunningParam(*partitioned_call_node), SUCCESS);
  const auto output_tensor_desc =
      partitioned_call_node->GetOpDescBarePtr()->MutableOutputDesc(static_cast<uint32_t>(0));
  int64_t mem_size = 0;
  ge::TensorUtils::GetSize(*output_tensor_desc, mem_size);
  EXPECT_EQ(mem_size, 320032);
}
}
} // namespace ge
