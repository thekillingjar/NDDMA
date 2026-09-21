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
#include "graph_optimizer/node_optimizer/stridedread_optimizer.h"
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

class STEST_stridedread_optimize : public testing::Test {
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
    OpDescPtr split = std::make_shared<OpDesc>("split", SPLITD);
    (void)ge::AttrUtils::SetBool(split, "_support_stridedread_optimize", true);
    (void)ge::AttrUtils::SetInt(split, SPLIT_DIM, 1);
    (void)ge::AttrUtils::SetInt(conv1, FE_IMPLY_TYPE, fe::OpImplType::EN_IMPL_HW_TBE);
    (void)ge::AttrUtils::SetInt(conv2, FE_IMPLY_TYPE, fe::OpImplType::EN_IMPL_HW_TBE);

    // add descriptor
    ge::GeShape shape1({1, 16, 14, 14});
    ge::GeShape shape2({1, 32, 14, 14});
    GeTensorDesc out_desc1(shape1, ge::FORMAT_NC1HWC0, ge::DT_FLOAT);
    out_desc1.SetOriginFormat(ge::FORMAT_NC1HWC0);
    out_desc1.SetOriginDataType(ge::DT_FLOAT);
    out_desc1.SetOriginShape(shape1);
    GeTensorDesc out_desc2(shape2, ge::FORMAT_NC1HWC0, ge::DT_FLOAT);
    out_desc2.SetOriginFormat(ge::FORMAT_NC1HWC0);
    out_desc2.SetOriginDataType(ge::DT_FLOAT);
    out_desc2.SetOriginShape(shape2);
    conv1->AddInputDesc(out_desc1);
    conv1->SetInputOffset({0});
    conv2->AddInputDesc(out_desc1);
    conv2->SetInputOffset({256});
    split->AddInputDesc(out_desc2);
    split->AddOutputDesc(out_desc1);
    split->AddOutputDesc(out_desc1);

    // create nodes
    NodePtr conv1_node = graph->AddNode(conv1);
    NodePtr conv2_node = graph->AddNode(conv2);
    NodePtr split_node = graph->AddNode(split);
    ge::GraphUtils::AddEdge(split_node->GetOutDataAnchor(0),
                            conv1_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(split_node->GetOutDataAnchor(1),
                            conv1_node->GetInDataAnchor(0));
  }
  void InitGraph2(ComputeGraphPtr& graph) {
    OpDescPtr conv1 = std::make_shared<OpDesc>("conv1", CONV2D);
    OpDescPtr conv2 = std::make_shared<OpDesc>("conv2", CONV2D);
    OpDescPtr split = std::make_shared<OpDesc>("split", SPLITD);
    OpDescPtr data = std::make_shared<OpDesc>("data", DATA);
    OpDescPtr sr1 = std::make_shared<OpDesc>("sw1", ASCEND_QUANT);
    OpDescPtr sr2 = std::make_shared<OpDesc>("sw1", ASCEND_QUANT);
    (void)ge::AttrUtils::SetBool(split, "_support_stridedread_optimize", true);
    (void)ge::AttrUtils::SetInt(split, SPLIT_DIM, 1);
    (void)ge::AttrUtils::SetInt(conv1, FE_IMPLY_TYPE, fe::OpImplType::EN_IMPL_HW_TBE);
    (void)ge::AttrUtils::SetInt(conv2, FE_IMPLY_TYPE, fe::OpImplType::EN_IMPL_HW_TBE);

    // add descriptor
    ge::GeShape shape1({1, 2, 14, 14, 16});
    ge::GeShape orishape1({1, 32, 14, 14});
    ge::GeShape shape2({1, 1, 14, 14, 16});
    ge::GeShape orishape2({1, 16, 14, 14});
    ge::GeShape shape3({1, 2, 14, 14, 8});
    GeTensorDesc out_desc1(shape1, ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
    out_desc1.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc1.SetOriginDataType(ge::DT_FLOAT);
    out_desc1.SetOriginShape(orishape1);
    GeTensorDesc out_desc2(shape2, ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
    out_desc2.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc2.SetOriginDataType(ge::DT_FLOAT);
    out_desc2.SetOriginShape(orishape2);
    GeTensorDesc out_desc3(shape3, ge::FORMAT_NC1HWC0, ge::DT_INT8);
    out_desc2.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc2.SetOriginDataType(ge::DT_FLOAT);
    out_desc2.SetOriginShape(orishape2);
    data->AddOutputDesc(out_desc1);
    split->AddInputDesc(out_desc1);
    split->AddOutputDesc(out_desc2);
    split->AddOutputDesc(out_desc2);
    sr1->AddInputDesc(out_desc2);
    sr1->AddOutputDesc(out_desc3);
    sr2->AddInputDesc(out_desc2);
    sr2->AddOutputDesc(out_desc3);
    conv1->AddInputDesc(out_desc3);
    conv2->AddInputDesc(out_desc3);


    // create nodes
    NodePtr data_node = graph->AddNode(data);
    NodePtr conv1_node = graph->AddNode(conv1);
    NodePtr conv2_node = graph->AddNode(conv2);
    NodePtr split_node = graph->AddNode(split);
    NodePtr sr1_node = graph->AddNode(sr1);
    NodePtr sr2_node = graph->AddNode(sr2);
    ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(0),
                            split_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(split_node->GetOutDataAnchor(0),
                            sr1_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(split_node->GetOutDataAnchor(1),
                            sr2_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(sr1_node->GetOutDataAnchor(0),
                            conv1_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(sr2_node->GetOutDataAnchor(0),
                            conv2_node->GetInDataAnchor(0));
  }
  void InitGraph3(ComputeGraphPtr& graph) {
    OpDescPtr conv1 = std::make_shared<OpDesc>("conv1", CONV2D);
    OpDescPtr conv2 = std::make_shared<OpDesc>("conv2", CONV2D);
    OpDescPtr split = std::make_shared<OpDesc>("split", SPLITD);
    (void)ge::AttrUtils::SetBool(split, "_support_stridedread_optimize", true);
    (void)ge::AttrUtils::SetInt(split, SPLIT_DIM, 1);
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
    conv1->AddInputDesc(out_desc1);
    conv1->SetInputOffset({0});
    conv2->AddInputDesc(out_desc1);
    conv2->SetInputOffset({256});
    split->AddInputDesc(out_desc2);
    split->AddOutputDesc(out_desc1);
    split->AddOutputDesc(out_desc1);

    // create nodes
    NodePtr conv1_node = graph->AddNode(conv1);
    NodePtr conv2_node = graph->AddNode(conv2);
    NodePtr split_node = graph->AddNode(split);
    ge::GraphUtils::AddEdge(split_node->GetOutDataAnchor(0),
                            conv1_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(split_node->GetOutDataAnchor(1),
                            conv1_node->GetInDataAnchor(0));
  }
};

TEST_F(STEST_stridedread_optimize, stridedread_optimize_01) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>(GRAPH_NAME);
  InitGraph1(graph);
  StridedReadOptimizer stridedReadOptimizer;
  stridedReadOptimizer.DoOptimizeForSplit(*graph);
  auto concat_node = graph->FindNode("split");
  bool attr_notask = false;
  (void)ge::AttrUtils::GetBool(concat_node->GetOpDesc(), ATTR_NAME_NOTASK, attr_notask);
  EXPECT_EQ(attr_notask, true);
}

TEST_F(STEST_stridedread_optimize, stridedread_optimize_02) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>(GRAPH_NAME);
  InitGraph2(graph);
  StridedReadOptimizer stridedReadOptimizer;
  stridedReadOptimizer.DoOptimizeForSplit(*graph);
  auto concat_node = graph->FindNode("split");
  bool attr_notask = false;
  (void)ge::AttrUtils::GetBool(concat_node->GetOpDesc(), ATTR_NAME_NOTASK, attr_notask);
  EXPECT_EQ(attr_notask, true);
}

TEST_F(STEST_stridedread_optimize, stridedread_optimize_03) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>(GRAPH_NAME);
  InitGraph3(graph);
  StridedReadOptimizer stridedReadOptimizer;
  stridedReadOptimizer.DoOptimizeForSplit(*graph);
  auto concat_node = graph->FindNode("split");
  bool attr_notask = false;
  (void)ge::AttrUtils::GetBool(concat_node->GetOpDesc(), ATTR_NAME_NOTASK, attr_notask);
  EXPECT_EQ(attr_notask, false);
}
}
