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

#define protected public
#define private public
#include "graph_optimizer/node_optimizer/stridedwrite_optimizer.h"
#include "graph_optimizer/fe_graph_optimizer.h"
#include "common/fe_utils.h"
#include "pass_manager.h"
#include "common/util/op_info_util.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/debug/ge_attr_define.h"

#undef protected
#undef private

using namespace std;
using namespace ge;
using namespace fe;

namespace fe {
using FEGraphOptimizerPtr = std::shared_ptr<fe::FEGraphOptimizer>;

class STEST_stridedwrite_optimize : public testing::Test {
 public:
  const string GRAPH_NAME = "test";
  const string CONCATD = "ConcatD";
  const string DEQUANT = "AscendDequant";
  const string CONV2D = "Conv2D";
  const string RELU = "Relu";
  const string LEAKYRELU = "LeakyRelu";
  const string STRIDEDWRITE = "StridedWrite";
  const string STRIDEDREAD = "StridedRead";
  const string STRIDE_ATTR_STRIDE = "stride";
  const string STRIDE_ATTR_AXIS = "axis";
  const string NETOUTPUT = "NetOutput";
  const int NCHW_DIM_N = 1;
  const int NCHW_DIM_H = 14;
  const int NCHW_DIM_W = 12;

 protected:
  void SetUp() {}
  void TearDown() {}

  void InitGraph1(ComputeGraphPtr& graph) {
    OpDescPtr conv1 = std::make_shared<OpDesc>("conv1", CONV2D);
    OpDescPtr conv2 = std::make_shared<OpDesc>("conv2", CONV2D);
    OpDescPtr concat = std::make_shared<OpDesc>("concat", CONCATD);
    (void)ge::AttrUtils::SetBool(concat, "_support_stridedwrite_optimize", true);
    (void)ge::AttrUtils::SetInt(concat, CONCAT_DIM, 1);
    (void)ge::AttrUtils::SetInt(conv1, FE_IMPLY_TYPE, fe::OpImplType::EN_IMPL_HW_TBE);
    (void)ge::AttrUtils::SetInt(conv2, FE_IMPLY_TYPE, fe::OpImplType::EN_IMPL_HW_TBE);

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
    conv2->AddOutputDesc(out_desc1);
    concat->AddInputDesc(out_desc1);
    concat->AddInputDesc(out_desc1);
    concat->AddOutputDesc(out_desc2);

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

  void InitGraph2(ComputeGraphPtr& graph) {
    OpDescPtr conv1 = std::make_shared<OpDesc>("conv1", CONV2D);
    OpDescPtr conv2 = std::make_shared<OpDesc>("conv2", CONV2D);
    OpDescPtr sw1 = std::make_shared<OpDesc>("sw1", STRIDEDWRITE);
    OpDescPtr sw2 = std::make_shared<OpDesc>("sw2", STRIDEDWRITE);
    OpDescPtr concat = std::make_shared<OpDesc>("concat", CONCATD);
    OpDescPtr netoutput = std::make_shared<OpDesc>("netoutput", NETOUTPUT);
    (void)ge::AttrUtils::SetBool(concat, "_support_stridedwrite_optimize", true);
    (void)ge::AttrUtils::SetInt(concat, CONCAT_DIM, 1);

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
    (void)ge::AttrUtils::SetInt(conv1, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
    (void)ge::AttrUtils::SetInt(conv2, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
    (void)ge::AttrUtils::SetInt(concat, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);

    conv1->AddOutputDesc(out_desc1);
    conv2->AddOutputDesc(out_desc2);
    sw1->AddInputDesc(out_desc2);
    sw1->AddOutputDesc(out_desc2);
    sw2->AddInputDesc(out_desc2);
    sw2->AddOutputDesc(out_desc2);
    concat->AddInputDesc(out_desc1);
    concat->AddInputDesc(out_desc2);

    ge::GeShape shape3({1, 48, 14, 14});
    GeTensorDesc out_desc3(shape3, ge::FORMAT_NCHW, ge::DT_FLOAT);
    out_desc3.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc3.SetOriginDataType(ge::DT_FLOAT);
    out_desc3.SetOriginShape(shape3);
    concat->AddOutputDesc(out_desc3);
    netoutput->AddInputDesc(out_desc3);

    // create nodes
    NodePtr conv1_node = graph->AddNode(conv1);
    NodePtr conv2_node = graph->AddNode(conv2);
    NodePtr sw1_node = graph->AddNode(sw1);
    NodePtr sw2_node = graph->AddNode(sw2);
    NodePtr concat_node = graph->AddNode(concat);
    NodePtr output_node = graph->AddNode(netoutput);
    /*
     *  Conv2d     Conv2d
     *      \       /
     *        Concat(concat_dim=1)
     *          |
     *       netoutput
     */
    ge::GraphUtils::AddEdge(conv1_node->GetOutDataAnchor(0),
                            sw1_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(conv2_node->GetOutDataAnchor(0),
                            sw2_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(sw1_node->GetOutDataAnchor(0),
                            concat_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(sw2_node->GetOutDataAnchor(0),
                            concat_node->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(concat_node->GetOutDataAnchor(0),
                            output_node->GetInDataAnchor(0));
  }
  void InitGraph3(ComputeGraphPtr& graph) {
    OpDescPtr conv1 = std::make_shared<OpDesc>("conv1", CONV2D);
    OpDescPtr conv2 = std::make_shared<OpDesc>("conv2", CONV2D);
    OpDescPtr conv3 = std::make_shared<OpDesc>("conv3", CONV2D);
    OpDescPtr pconcat = std::make_shared<OpDesc>("pconcat", "PhonyConcat");
    OpDescPtr concat = std::make_shared<OpDesc>("concat", CONCATD);
    (void)ge::AttrUtils::SetBool(concat, "_support_stridedwrite_optimize", true);
    (void)ge::AttrUtils::SetInt(concat, CONCAT_DIM, 1);
    (void)ge::AttrUtils::SetInt(conv1, FE_IMPLY_TYPE, fe::OpImplType::EN_IMPL_HW_TBE);
    (void)ge::AttrUtils::SetInt(conv2, FE_IMPLY_TYPE, fe::OpImplType::EN_IMPL_HW_TBE);

    // add descriptor
    ge::GeShape shape1({2, 16, 14, 14});
    ge::GeShape shape2({2, 32, 14, 14});
    ge::GeShape shape3({1, 16, 14, 14});
    GeTensorDesc out_desc1(shape1, ge::FORMAT_NCHW, ge::DT_FLOAT);
    out_desc1.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc1.SetOriginDataType(ge::DT_FLOAT);
    out_desc1.SetOriginShape(shape1);
    GeTensorDesc out_desc2(shape2, ge::FORMAT_NCHW, ge::DT_FLOAT);
    out_desc2.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc2.SetOriginDataType(ge::DT_FLOAT);
    out_desc2.SetOriginShape(shape2);
    GeTensorDesc out_desc3(shape3, ge::FORMAT_NCHW, ge::DT_FLOAT);
    out_desc3.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc3.SetOriginDataType(ge::DT_FLOAT);
    out_desc3.SetOriginShape(shape3);
    conv1->AddOutputDesc(out_desc1);
    conv2->AddOutputDesc(out_desc3);
    conv3->AddOutputDesc(out_desc3);
    pconcat->AddInputDesc(out_desc3);
    pconcat->AddInputDesc(out_desc3);
    pconcat->AddOutputDesc(out_desc1);
    concat->AddInputDesc(out_desc1);
    concat->AddInputDesc(out_desc1);
    concat->AddOutputDesc(out_desc2);

    // create nodes
    NodePtr conv1_node = graph->AddNode(conv1);
    NodePtr conv2_node = graph->AddNode(conv2);
    NodePtr conv3_node = graph->AddNode(conv3);
    NodePtr pconcat_node = graph->AddNode(pconcat);
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
                            pconcat_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(conv3_node->GetOutDataAnchor(0),
                            pconcat_node->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(pconcat_node->GetOutDataAnchor(0),
                            concat_node->GetInDataAnchor(1));
  }
};

TEST_F(STEST_stridedwrite_optimize, stridedwrite_optimize_01) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  InitGraph1(graph);
  StridedWriteOptimizer stridedWriteOptimizer;
  stridedWriteOptimizer.DoOptimizeForConcat(*graph);
  auto concat_node = graph->FindNode("concat");
  bool attr_notask = false;
  (void)ge::AttrUtils::GetBool(concat_node->GetOpDesc(), ATTR_NAME_NOTASK, attr_notask);
  EXPECT_EQ(attr_notask, false);
}

TEST_F(STEST_stridedwrite_optimize, stridedwrite_optimize_02) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  InitGraph2(graph);
  StridedWriteOptimizer stridedWriteOptimizer;
  stridedWriteOptimizer.DoOptimizeForConcat(*graph);
  auto concat_node = graph->FindNode("concat");
  bool attr_notask = false;
  (void)ge::AttrUtils::GetBool(concat_node->GetOpDesc(), ATTR_NAME_NOTASK, attr_notask);
  EXPECT_EQ(attr_notask, false);
}

TEST_F(STEST_stridedwrite_optimize, stridedwrite_optimize_03) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  InitGraph3(graph);
  StridedWriteOptimizer stridedWriteOptimizer;
  stridedWriteOptimizer.DoOptimizeForConcat(*graph);
  auto concat_node = graph->FindNode("concat");
  bool attr_notask = false;
  (void)ge::AttrUtils::GetBool(concat_node->GetOpDesc(), ATTR_NAME_NOTASK, attr_notask);
  EXPECT_EQ(attr_notask, false);
}

TEST_F(STEST_stridedwrite_optimize, stridedwrite_optimize_04) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  InitGraph3(graph);
  StridedWriteOptimizer stridedWriteOptimizer;
  auto concat_node = graph->FindNode("concat");
  ge::OpDescPtr op_desc = concat_node->GetOpDesc();
  size_t idx = 0;
  std::vector<int64_t> concat_out_shape = {2, 16, 14, 14};
  bool is_last_input = false;
  int32_t concat_dim = 0;
  int64_t phony_concat_offset = 0;
  bool set_offset = false;
  stridedWriteOptimizer.FeedToOpStructInfo(op_desc, idx, concat_out_shape, is_last_input, concat_dim,
                                           phony_concat_offset, set_offset);
}
}
