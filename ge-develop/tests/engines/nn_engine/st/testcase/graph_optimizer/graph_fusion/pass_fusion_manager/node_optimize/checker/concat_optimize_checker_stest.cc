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
#include <nlohmann/json.hpp>
#define private public
#define protected public
#include "common/fe_utils.h"
#include "pass_manager.h"
#include "common/util/op_info_util.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/builtin_pass/node_optimize/conv_concat_fusion_pass.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/builtin_pass/node_optimize/checker/node_optimize_checker_base.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/builtin_pass/node_optimize/checker/concat_optimize_checker.h"
#undef protected
#undef private

using namespace std;
using namespace ge;
using namespace fe;

namespace fe {

class STEST_concat_optimize_checker : public testing::Test {
 public:
  const string GRAPH_NAME = "test";
  const string CONCATD = "ConcatD";
  const string DEQUANT = "AscendDequant";
  const string CONV2D = "Conv2D";
  const string RELU = "Relu";
  const string STRIDEDWRITE = "StridedWrite";
  const string STRIDEDREAD = "StridedRead";
  const string STRIDE_ATTR_STRIDE = "stride";
  const string STRIDE_ATTR_AXIS = "axis";

 protected:
  void SetUp() {}
  void TearDown() {}
  void InitGraph1(ComputeGraphPtr& graph) {
    OpDescPtr conv = std::make_shared<OpDesc>("conv", CONV2D);
    OpDescPtr concat = std::make_shared<OpDesc>("concat", CONCATD);
    (void)ge::AttrUtils::SetInt(concat, CONCAT_DIM, 1);

    // add descriptor
    ge::GeShape shape1({1, 16, 14, 14});
    GeTensorDesc out_desc(shape1, ge::FORMAT_NCHW, ge::DT_FLOAT);
    out_desc.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc.SetOriginDataType(ge::DT_FLOAT);
    out_desc.SetOriginShape(shape1);
    conv->AddOutputDesc(out_desc);
    concat->AddInputDesc(out_desc);
    concat->AddInputDesc(out_desc);

    // create nodes
    NodePtr conv_node = graph->AddNode(conv);
    NodePtr concat_node = graph->AddNode(concat);
    ge::GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0),
                            concat_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0),
                            concat_node->GetInDataAnchor(1));
  }

  void InitGraph2(ComputeGraphPtr& graph) {
    OpDescPtr conv1 = std::make_shared<OpDesc>("conv1", CONV2D);
    OpDescPtr conv2 = std::make_shared<OpDesc>("conv2", CONV2D);
    OpDescPtr concat = std::make_shared<OpDesc>("concat", CONCATD);
    (void)ge::AttrUtils::SetInt(concat, CONCAT_DIM, 0);

    // add descriptor
    ge::GeShape shape1({1, 16, 14, 14});
    ge::GeShape shape2({1, 32, 14, 14});
    GeTensorDesc out_desc1(shape1, ge::FORMAT_NCHW, ge::DT_FLOAT);
    out_desc1.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc1.SetOriginDataType(ge::DT_FLOAT);
    out_desc1.SetOriginShape(shape1);
    GeTensorDesc out_desc2(shape2, ge::FORMAT_NCHW, ge::DT_FLOAT);
    out_desc2.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc2.SetOriginDataType(ge::DT_FLOAT);
    out_desc2.SetOriginShape(shape2);
    conv1->AddOutputDesc(out_desc1);
    conv2->AddOutputDesc(out_desc2);
    concat->AddInputDesc(out_desc1);
    concat->AddInputDesc(out_desc2);

    // create nodes
    NodePtr conv1_node = graph->AddNode(conv1);
    NodePtr conv2_node = graph->AddNode(conv2);
    NodePtr concat_node = graph->AddNode(concat);
    /*
     *  Conv2d     Conv2d
     *      \       /
     *        Concat(concat_dim=0)
     *          |
     */
    ge::GraphUtils::AddEdge(conv1_node->GetOutDataAnchor(0),
                            concat_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(conv2_node->GetOutDataAnchor(0),
                            concat_node->GetInDataAnchor(1));
  }

  void InitGraph3(ComputeGraphPtr& graph) {
    OpDescPtr conv1 = std::make_shared<OpDesc>("conv1", CONV2D);
    OpDescPtr conv2 = std::make_shared<OpDesc>("conv2", CONV2D);
    OpDescPtr concat = std::make_shared<OpDesc>("concat", CONCATD);
    (void)ge::AttrUtils::SetInt(concat, CONCAT_DIM, 1);

    // add descriptor
    ge::GeShape shape1({1, 3, 14, 14});
    ge::GeShape shape2({1, 32, 14, 14});
    GeTensorDesc out_desc1(shape1, ge::FORMAT_NCHW, ge::DT_FLOAT);
    out_desc1.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc1.SetOriginDataType(ge::DT_FLOAT);
    out_desc1.SetOriginShape(shape1);
    GeTensorDesc out_desc2(shape2, ge::FORMAT_NCHW, ge::DT_FLOAT);
    out_desc2.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc2.SetOriginDataType(ge::DT_FLOAT);
    out_desc2.SetOriginShape(shape2);
    conv1->AddOutputDesc(out_desc1);
    conv2->AddOutputDesc(out_desc2);
    concat->AddInputDesc(out_desc1);
    concat->AddInputDesc(out_desc2);

    // create nodes
    NodePtr conv1_node = graph->AddNode(conv1);
    NodePtr conv2_node = graph->AddNode(conv2);
    NodePtr concat_node = graph->AddNode(concat);
    /*
     *  Conv2d     Conv2d
     *      \       /
     *        Concat
     *          |
     */
    ge::GraphUtils::AddEdge(conv1_node->GetOutDataAnchor(0),
                            concat_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(conv2_node->GetOutDataAnchor(0),
                            concat_node->GetInDataAnchor(1));
  }

  void InitGraph4(ComputeGraphPtr& graph) {
    OpDescPtr conv1 = std::make_shared<OpDesc>("conv1", CONV2D);
    OpDescPtr conv2 = std::make_shared<OpDesc>("conv2", CONV2D);
    OpDescPtr concat = std::make_shared<OpDesc>("concat", CONCATD);
    (void)ge::AttrUtils::SetInt(concat, CONCAT_DIM, 1);
    (void)ge::AttrUtils::SetBool(conv1, ge::ATTR_NAME_NOTASK, true);

    // add descriptor
    ge::GeShape shape1({1, 16, 14, 14});
    ge::GeShape shape2({1, 32, 14, 14});
    GeTensorDesc out_desc1(shape1, ge::FORMAT_NCHW, ge::DT_FLOAT);
    out_desc1.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc1.SetOriginDataType(ge::DT_FLOAT);
    out_desc1.SetOriginShape(shape1);
    GeTensorDesc out_desc2(shape2, ge::FORMAT_NCHW, ge::DT_FLOAT);
    out_desc2.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc2.SetOriginDataType(ge::DT_FLOAT);
    out_desc2.SetOriginShape(shape2);
    conv1->AddOutputDesc(out_desc1);
    conv2->AddOutputDesc(out_desc2);
    concat->AddInputDesc(out_desc1);
    concat->AddInputDesc(out_desc2);

    // create nodes
    NodePtr conv1_node = graph->AddNode(conv1);
    NodePtr conv2_node = graph->AddNode(conv2);
    NodePtr concat_node = graph->AddNode(concat);
    /*
     *  Conv2d     Conv2d
     *      \       /
     *        Concat
     *          |
     */
    ge::GraphUtils::AddEdge(conv1_node->GetOutDataAnchor(0),
                            concat_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(conv2_node->GetOutDataAnchor(0),
                            concat_node->GetInDataAnchor(1));
  }

void InitGraph5QuantNotAligned(ComputeGraphPtr& graph) {
  OpDescPtr quant1 = std::make_shared<OpDesc>("quant1", ASCEND_QUANT);
  OpDescPtr quant2 = std::make_shared<OpDesc>("quant2", ASCEND_QUANT);
  OpDescPtr concat = std::make_shared<OpDesc>("concat", CONCATD);
  (void)ge::AttrUtils::SetInt(concat, CONCAT_DIM, 1);

  // add descriptor
  ge::GeShape shape1({1, 64, 14, 14});
  ge::GeShape shape2({1, 32, 14, 14});
  GeTensorDesc out_desc1(shape1, ge::FORMAT_NCHW, ge::DT_INT4);
  out_desc1.SetOriginFormat(ge::FORMAT_NCHW);
  out_desc1.SetOriginDataType(ge::DT_FLOAT);
  out_desc1.SetOriginShape(shape1);
  GeTensorDesc out_desc2(shape2, ge::FORMAT_NCHW, ge::DT_INT4);
  out_desc2.SetOriginFormat(ge::FORMAT_NCHW);
  out_desc2.SetOriginDataType(ge::DT_FLOAT);
  out_desc2.SetOriginShape(shape2);

  quant1->AddOutputDesc(out_desc1);
  quant2->AddOutputDesc(out_desc2);
  concat->AddInputDesc(out_desc1);
  concat->AddInputDesc(out_desc2);

  // create nodes
  NodePtr quant1_node = graph->AddNode(quant1);
  NodePtr quant2_node = graph->AddNode(quant2);
   NodePtr concat_node = graph->AddNode(concat);
  /*
  * quant(INT4) quant(INT4)
  *      \       /
  *        Concat
  *          |
  */
    ge::GraphUtils::AddEdge(quant1_node->GetOutDataAnchor(0),
                            concat_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(quant2_node->GetOutDataAnchor(0),
                            concat_node->GetInDataAnchor(1));
  }
};

TEST_F(STEST_concat_optimize_checker, check_fail_input_from_same_node) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>(GRAPH_NAME);
  InitGraph1(graph);
  ConvConcatFusionPass pass;
  vector<GraphPass*> passes = {&pass};
  Status status = PassManager::Run(*graph, passes);
  EXPECT_EQ(status, NOT_CHANGED);
}

TEST_F(STEST_concat_optimize_checker, check_fail_not_dim_c) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>(GRAPH_NAME);
  InitGraph2(graph);
  ConvConcatFusionPass pass;
  vector<GraphPass*> passes = {&pass};
  Status status = PassManager::Run(*graph, passes);
  EXPECT_EQ(status, NOT_CHANGED);
}

TEST_F(STEST_concat_optimize_checker, check_fail_not_dim_c_aligned) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>(GRAPH_NAME);
  InitGraph3(graph);
  ConvConcatFusionPass pass;
  vector<GraphPass*> passes = {&pass};
  Status status = PassManager::Run(*graph, passes);
  EXPECT_EQ(status, NOT_CHANGED);
}

TEST_F(STEST_concat_optimize_checker, check_fail_pre_node_invalid) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>(GRAPH_NAME);
  InitGraph4(graph);
  ConvConcatFusionPass pass;
  vector<GraphPass*> passes = {&pass};
  Status status = PassManager::Run(*graph, passes);
  EXPECT_EQ(status, NOT_CHANGED);
}

TEST_F(STEST_concat_optimize_checker, check_fail_int4_not_dim_c_aligned) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>(GRAPH_NAME);
  InitGraph5QuantNotAligned(graph);
  ConvConcatFusionPass pass;
  vector<GraphPass*> passes = {&pass};
  Status status = PassManager::Run(*graph, passes);
  EXPECT_EQ(status, NOT_CHANGED);
}

TEST_F(STEST_concat_optimize_checker, is_dimc_test){
  string dim_attr = "test";
  bool is_input = true;
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr op1 = std::make_shared<OpDesc>("test", "test");
  vector<int64_t> dim_input({4, 33, 12, 16, 64});
  GeShape shape(dim_input);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_ND);
  tensor_desc.SetOriginShape(shape);
  tensor_desc.SetFormat(FORMAT_ND);
  tensor_desc.SetDataType(DT_FLOAT);
  op1->AddInputDesc("x", tensor_desc);
  op1->AddOutputDesc("y", tensor_desc);
  ge::NodePtr node_ptr = graph->AddNode(op1);
  int dim_attr_value_0 = 13;
  ge::AttrUtils::SetInt(op1, dim_attr, dim_attr_value_0);
  NodeOptimizeCheckerBase nodeoptimizecheckerbase;
  bool ret = nodeoptimizecheckerbase.IsDimC(node_ptr, dim_attr, is_input);
  EXPECT_EQ(ret, false);
}

TEST_F(STEST_concat_optimize_checker, test_is_dim_c_asigned){
  string dim_attr = "test";
  bool is_input = true;
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr op1 = std::make_shared<OpDesc>("test", "test");
  vector<int64_t> dim_input({4, 33, 12, 16, 64});
  GeShape shape(dim_input);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_ND);
  tensor_desc.SetOriginShape(shape);
  tensor_desc.SetFormat(FORMAT_ND);
  tensor_desc.SetDataType(DT_FLOAT);
  op1->AddInputDesc("x", tensor_desc);
  op1->AddInputDesc("x1", tensor_desc);
  op1->AddOutputDesc("y", tensor_desc);
  ge::NodePtr node_ptr = graph->AddNode(op1);
  int dim_attr_value_0 = 13;
  ge::AttrUtils::SetInt(op1, dim_attr, dim_attr_value_0);
  ConcatOptimizeChecker check;
  check.IsDimCAligned(node_ptr);
  check.IsDimCAlignedWithQuant(node_ptr);
  NodeOptimizeCheckerBase nodeoptimizecheckerbase;
  nodeoptimizecheckerbase.IsDimC(node_ptr, "aaa", true);
}
}  // namespace fe
