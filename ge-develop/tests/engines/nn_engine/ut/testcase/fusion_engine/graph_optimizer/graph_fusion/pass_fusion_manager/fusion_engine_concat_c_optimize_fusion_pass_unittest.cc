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
#include "graph_optimizer/graph_fusion/fusion_pass_manager/builtin_pass/concat_c_optimize_fusion_pass.h"
#include "graph_optimizer/fe_graph_optimizer.h"
#include "common/fe_utils.h"
#include "pass_manager.h"
#include "common/util/op_info_util.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "ops_store/ops_kernel_manager.h"

#undef protected
#undef private

using namespace std;
using namespace ge;
using namespace fe;

namespace fe {
using FEGraphOptimizerPtr = std::shared_ptr<fe::FEGraphOptimizer>;

class UTEST_concat_c_optimize : public testing::Test {
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
  FEOpsKernelInfoStorePtr ops_kernel_info_store_ptr;

protected:
  void SetUp() {}
  void TearDown() {}
  void InitGraph1(ComputeGraphPtr& graph) {
    OpDescPtr conv1 = std::make_shared<OpDesc>("conv1", CONV2D);
    OpDescPtr conv2 = std::make_shared<OpDesc>("conv2", CONV2D);
    OpDescPtr concat = std::make_shared<OpDesc>("concat", CONCATD);
    (void)ge::AttrUtils::SetInt(concat, CONCAT_DIM, 1);

    // add descriptor
    ge::GeShape shape1({1, -1, 14, 14});
    ge::GeShape shape2({1, -1, 14, 14});
    GeTensorDesc out_desc1(shape1, ge::FORMAT_NC1HWC0, ge::DT_FLOAT);
    out_desc1.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc1.SetOriginDataType(ge::DT_FLOAT);
    out_desc1.SetOriginShape(shape1);
    GeTensorDesc out_desc2(shape2, ge::FORMAT_NC1HWC0, ge::DT_FLOAT);
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

  void InitGraph2(ComputeGraphPtr& graph) {
    OpDescPtr conv1 = std::make_shared<OpDesc>("conv1", CONV2D);
    OpDescPtr conv2 = std::make_shared<OpDesc>("conv2", CONV2D);
    OpDescPtr concat = std::make_shared<OpDesc>("concat", CONCATD);
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

  void InitGraph3(ComputeGraphPtr& graph) {
    OpDescPtr conv1 = std::make_shared<OpDesc>("conv1", CONV2D);
    OpDescPtr conv2 = std::make_shared<OpDesc>("conv2", CONV2D);
    OpDescPtr concat = std::make_shared<OpDesc>("concat", CONCATD);
    (void)ge::AttrUtils::SetInt(concat, CONCAT_DIM, 1);

    // add descriptor
    ge::GeShape shape1({-1, 16, 14, 14});
    ge::GeShape shape2({-1, 32, 14, 14});
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
    concat->AddInputDesc(out_desc1);
    concat->AddInputDesc(out_desc2);

    // create nodes
    NodePtr conv1_node = graph->AddNode(conv1);
    NodePtr conv2_node = graph->AddNode(conv2);
    NodePtr concat_node = graph->AddNode(concat);
    /*
     *  Conv2d     Conv2d
     *      \       /
     *        Concat(concat_dim=1)
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
    OpDescPtr netoutput = std::make_shared<OpDesc>("netoutput", NETOUTPUT);
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
                            concat_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(conv2_node->GetOutDataAnchor(0),
                            concat_node->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(concat_node->GetOutDataAnchor(0),
                            output_node->GetInDataAnchor(0));
  }

  void InitGraph5(ComputeGraphPtr& graph) {
    OpDescPtr conv1 = std::make_shared<OpDesc>("conv1", CONV2D);
    OpDescPtr reshape = std::make_shared<OpDesc>("reshape", "Reshape");
    OpDescPtr concat = std::make_shared<OpDesc>("concat", CONCATD);
    (void)ge::AttrUtils::SetInt(concat, CONCAT_DIM, 1);
    (void)ge::AttrUtils::SetInt(conv1, FE_IMPLY_TYPE, fe::OpImplType::EN_IMPL_HW_TBE);
    (void)ge::AttrUtils::SetInt(reshape, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);

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
    reshape->AddOutputDesc(out_desc1);
    concat->AddInputDesc(out_desc1);
    concat->AddInputDesc(out_desc1);
    concat->AddOutputDesc(out_desc2);

    // create nodes
    NodePtr conv1_node = graph->AddNode(conv1);
    NodePtr reshape_node = graph->AddNode(reshape);
    NodePtr concat_node = graph->AddNode(concat);
    /*
     *  Conv2d     Conv2d
     *      \       /
     *        Concat(concat_dim=0)
     *          |
     */
    ge::GraphUtils::AddEdge(conv1_node->GetOutDataAnchor(0),
                            concat_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(reshape_node->GetOutDataAnchor(0),
                            concat_node->GetInDataAnchor(1));
  }

  void InitGraph6(ComputeGraphPtr& graph) {
    OpDescPtr conv1 = std::make_shared<OpDesc>("conv1", CONV2D);
    OpDescPtr conv2 = std::make_shared<OpDesc>("conv2", CONV2D);
    OpDescPtr conv3 = std::make_shared<OpDesc>("conv3", CONV2D);
    OpDescPtr concat = std::make_shared<OpDesc>("concat", CONCATD);
    (void)ge::AttrUtils::SetInt(concat, CONCAT_DIM, 1);
    (void)ge::AttrUtils::SetInt(conv1, FE_IMPLY_TYPE, fe::OpImplType::EN_IMPL_HW_TBE);
    (void)ge::AttrUtils::SetInt(conv2, FE_IMPLY_TYPE, fe::OpImplType::EN_IMPL_HW_TBE);
    (void)ge::AttrUtils::SetInt(conv3, FE_IMPLY_TYPE, fe::OpImplType::EN_IMPL_HW_TBE);

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
    conv3->AddOutputDesc(out_desc1);
    concat->AddInputDesc(out_desc1);
    concat->AddInputDesc(out_desc1);
    concat->AddOutputDesc(out_desc2);
    conv3->AddInputDesc(out_desc2);

    // create nodes
    NodePtr conv1_node = graph->AddNode(conv1);
    NodePtr conv2_node = graph->AddNode(conv2);
    NodePtr concat_node = graph->AddNode(concat);
    NodePtr conv3_node = graph->AddNode(conv3);
    /*
     *  Conv2d     Conv2d
     *      \       /
     *        Concat(concat_dim=0)
     *          |
     *        Conv2d
     */
    ge::GraphUtils::AddEdge(conv1_node->GetOutDataAnchor(0),
                            concat_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(conv2_node->GetOutDataAnchor(0),
                            concat_node->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(concat_node->GetOutDataAnchor(0),
                            conv3_node->GetInDataAnchor(0));
  }

  void InitGraph7(ComputeGraphPtr& graph) {
    OpDescPtr conv1 = std::make_shared<OpDesc>("conv1", CONV2D);
    OpDescPtr conv2 = std::make_shared<OpDesc>("conv2", CONV2D);
    OpDescPtr end = std::make_shared<OpDesc>("end", "End");
    OpDescPtr concat = std::make_shared<OpDesc>("concat", CONCATD);
    (void)ge::AttrUtils::SetInt(concat, CONCAT_DIM, 1);
    (void)ge::AttrUtils::SetInt(conv1, FE_IMPLY_TYPE, fe::OpImplType::EN_IMPL_HW_TBE);
    (void)ge::AttrUtils::SetInt(conv2, FE_IMPLY_TYPE, fe::OpImplType::EN_IMPL_HW_TBE);
    (void)ge::AttrUtils::SetInt(end, FE_IMPLY_TYPE, fe::OpImplType::EN_IMPL_HW_TBE);

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
    end->AddOutputDesc(out_desc1);
    concat->AddInputDesc(out_desc1);
    concat->AddInputDesc(out_desc1);
    concat->AddOutputDesc(out_desc2);
    end->AddInputDesc(out_desc2);
    (void)ge::AttrUtils::SetStr(end, "parentOpType", "NetOutput");

    // create nodes
    NodePtr conv1_node = graph->AddNode(conv1);
    NodePtr conv2_node = graph->AddNode(conv2);
    NodePtr concat_node = graph->AddNode(concat);
    NodePtr end_node = graph->AddNode(end);
    /*
     *  Conv2d     Conv2d
     *      \       /
     *        Concat(concat_dim=0)
     *          |
     *        End
     */
    ge::GraphUtils::AddEdge(conv1_node->GetOutDataAnchor(0),
                            concat_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(conv2_node->GetOutDataAnchor(0),
                            concat_node->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(concat_node->GetOutDataAnchor(0),
                            end_node->GetInDataAnchor(0));
  }

  void InitGraph8(ComputeGraphPtr& graph) {
    OpDescPtr conv1 = std::make_shared<OpDesc>("conv1", CONV2D);
    OpDescPtr conv2 = std::make_shared<OpDesc>("conv2", CONV2D);
    OpDescPtr end = std::make_shared<OpDesc>("end", "End");
    OpDescPtr concat = std::make_shared<OpDesc>("concat", CONCATD);
    (void)ge::AttrUtils::SetInt(concat, CONCAT_DIM, 1);
    (void)ge::AttrUtils::SetInt(conv1, FE_IMPLY_TYPE, fe::OpImplType::EN_IMPL_HW_TBE);
    (void)ge::AttrUtils::SetInt(conv2, FE_IMPLY_TYPE, fe::OpImplType::EN_IMPL_HW_TBE);
    (void)ge::AttrUtils::SetInt(end, FE_IMPLY_TYPE, fe::OpImplType::EN_IMPL_HW_TBE);

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
    end->AddOutputDesc(out_desc1);
    concat->AddInputDesc(out_desc1);
    concat->AddInputDesc(out_desc1);
    concat->AddOutputDesc(out_desc2);
    end->AddInputDesc(out_desc2);
    (void)ge::AttrUtils::SetStr(end, "parentOpType", "NetOutput");
    (void)ge::AttrUtils::SetBool(conv1, "can_reused_for_concat_optimize", false);

    // create nodes
    NodePtr conv1_node = graph->AddNode(conv1);
    NodePtr conv2_node = graph->AddNode(conv2);
    NodePtr concat_node = graph->AddNode(concat);
    NodePtr end_node = graph->AddNode(end);
    /*
     *  Conv2d     Conv2d
     *      \       /
     *        Concat(concat_dim=0)
     *          |
     *        End
     */
    ge::GraphUtils::AddEdge(conv1_node->GetOutDataAnchor(0),
                            concat_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(conv2_node->GetOutDataAnchor(0),
                            concat_node->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(concat_node->GetOutDataAnchor(0),
                            end_node->GetInDataAnchor(0));
  }

  void InitGraph9(ComputeGraphPtr& graph) {
    OpDescPtr opdesc_ptr = std::make_shared<OpDesc>("test", "ReduceAllD");
    OpDescPtr concat = std::make_shared<OpDesc>("concat", CONCATD);
    OpDescPtr reshape1 = std::make_shared<OpDesc>("reshape1", RESHAPE);
    OpDescPtr reshape2 = std::make_shared<OpDesc>("reshape2", RESHAPE);

    (void)ge::AttrUtils::SetInt(concat, CONCAT_DIM, 1);

    // add descriptor
    ge::GeShape shape1({1, 32, 64, 64});
    ge::GeShape shape2({1, 32, 64, 64});
    GeTensorDesc out_desc1(shape1, ge::FORMAT_NCHW, ge::DT_FLOAT);
    out_desc1.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc1.SetOriginDataType(ge::DT_FLOAT);
    out_desc1.SetOriginShape(shape1);
    GeTensorDesc out_desc2(shape2, ge::FORMAT_NCHW, ge::DT_FLOAT);
    out_desc2.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc2.SetOriginDataType(ge::DT_FLOAT);
    out_desc2.SetOriginShape(shape2);
    (void)ge::AttrUtils::SetInt(concat, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
    (void)ge::AttrUtils::SetInt(reshape1, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
    (void)ge::AttrUtils::SetInt(reshape2, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
    (void)ge::AttrUtils::SetInt(concat, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);


    opdesc_ptr->AddOutputDesc(out_desc1);
    reshape1->AddOutputDesc(out_desc1);
    reshape2->AddOutputDesc(out_desc1);
    reshape1->AddInputDesc(out_desc1);
    reshape2->AddInputDesc(out_desc1);
    concat->AddInputDesc(out_desc1);
    concat->AddInputDesc(out_desc2);

    // create nodes
    NodePtr test_node = graph->AddNode(opdesc_ptr);
    NodePtr reshape1_node = graph->AddNode(reshape1);
    NodePtr reshape2_node = graph->AddNode(reshape2);
    NodePtr concat_node = graph->AddNode(concat);
    /*
     *         node
     *      /        \
     *  Reshape    Reshape
     *      \       /
     *        Concat(concat_dim=1)
     *          |
     */
    ge::GraphUtils::AddEdge(reshape1_node->GetOutDataAnchor(0),
                            concat_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(reshape2_node->GetOutDataAnchor(0),
                            concat_node->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(test_node->GetOutDataAnchor(0),
                            reshape1_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(test_node->GetOutDataAnchor(0),
                            reshape2_node->GetInDataAnchor(0));
  }

  void InitGraph10(ComputeGraphPtr& graph) {
    OpDescPtr conv = std::make_shared<OpDesc>("conv", CONV2D);
    OpDescPtr op_a = std::make_shared<OpDesc>("a", "A");
    OpDescPtr op_b = std::make_shared<OpDesc>("b", "B");

    OpDescPtr concat_1= std::make_shared<OpDesc>("concat_1", CONCATD);
    OpDescPtr concat_2= std::make_shared<OpDesc>("concat_2", CONCATD);
    OpDescPtr concat_3= std::make_shared<OpDesc>("concat_3", CONCATD);
    (void)ge::AttrUtils::SetInt(concat_1, CONCAT_DIM, 1);
    (void)ge::AttrUtils::SetInt(concat_2, CONCAT_DIM, 1);
    (void)ge::AttrUtils::SetInt(concat_3, CONCAT_DIM, 1);
    // add descriptor
    ge::GeShape shape1({1, 32, 14, 14});
    ge::GeShape shape2({1, 32, 14, 14});
    GeTensorDesc out_desc1(shape1, ge::FORMAT_NCHW, ge::DT_FLOAT);
    out_desc1.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc1.SetOriginDataType(ge::DT_FLOAT);
    out_desc1.SetOriginShape(shape1);
    GeTensorDesc out_desc2(shape2, ge::FORMAT_NCHW, ge::DT_FLOAT);
    out_desc2.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc2.SetOriginDataType(ge::DT_FLOAT);
    out_desc2.SetOriginShape(shape2);
    GeTensorDesc in_desc(shape2, ge::FORMAT_NCHW, ge::DT_FLOAT);
    in_desc.SetOriginFormat(ge::FORMAT_NCHW);
    in_desc.SetOriginDataType(ge::DT_FLOAT);
    in_desc.SetOriginShape(shape2);
    (void)ge::AttrUtils::SetInt(conv, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
    (void)ge::AttrUtils::SetInt(op_a, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
    (void)ge::AttrUtils::SetInt(op_b, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
    (void)ge::AttrUtils::SetInt(concat_1, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
    (void)ge::AttrUtils::SetInt(concat_2, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
    (void)ge::AttrUtils::SetInt(concat_3, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);

    conv->AddOutputDesc(out_desc1);
    op_a->AddOutputDesc(out_desc2);
    op_b->AddOutputDesc(out_desc1);
    concat_1->AddInputDesc(in_desc);
    concat_1->AddInputDesc(in_desc);
    concat_1->AddOutputDesc(out_desc1);
    concat_2->AddInputDesc(in_desc);
    concat_2->AddInputDesc(in_desc);
    concat_2->AddOutputDesc(out_desc2);
    concat_3->AddInputDesc(in_desc);
    concat_3->AddInputDesc(in_desc);
    concat_3->AddOutputDesc(out_desc2);

    // create nodes
    NodePtr conv_node = graph->AddNode(conv);
    NodePtr a_node = graph->AddNode(op_a);
    NodePtr b_node = graph->AddNode(op_b);
    NodePtr concat1_node = graph->AddNode(concat_1);
    NodePtr concat2_node = graph->AddNode(concat_2);
    NodePtr concat3_node = graph->AddNode(concat_3);
    /*
     *  A      Conv2d      B
     *   \       / \      /
     *    \     /   \    /
     *     Concat   Concat(concat_dim=1)
     *        \        /
     *         \      /
     *          Concat
     */
    ge::GraphUtils::AddEdge(a_node->GetOutDataAnchor(0),
                            concat1_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0),
                            concat1_node->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0),
                            concat2_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(b_node->GetOutDataAnchor(0),
                            concat2_node->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(concat1_node->GetOutDataAnchor(0),
                            concat3_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(concat2_node->GetOutDataAnchor(0),
                            concat3_node->GetInDataAnchor(1));
  }

  void InitGraph11(ComputeGraphPtr& graph) {
    OpDescPtr conv = std::make_shared<OpDesc>("conv", CONV2D);
    OpDescPtr op_a = std::make_shared<OpDesc>("a", "A");
    OpDescPtr op_b = std::make_shared<OpDesc>("b", "B");

    OpDescPtr concat_1= std::make_shared<OpDesc>("concat_1", CONCATD);
    OpDescPtr concat_2= std::make_shared<OpDesc>("concat_2", CONCATD);
    OpDescPtr concat_3= std::make_shared<OpDesc>("concat_3", CONCATD);
    (void)ge::AttrUtils::SetInt(concat_1, CONCAT_DIM, 1);
    (void)ge::AttrUtils::SetInt(concat_2, CONCAT_DIM, 1);
    (void)ge::AttrUtils::SetInt(concat_3, CONCAT_DIM, 1);
    // add descriptor
    ge::GeShape shape1({1, 32, 14, 14});
    ge::GeShape shape2({1, 32, 14, 14});
    GeTensorDesc out_desc1(shape1, ge::FORMAT_NCHW, ge::DT_FLOAT);
    out_desc1.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc1.SetOriginDataType(ge::DT_FLOAT);
    out_desc1.SetOriginShape(shape1);
    GeTensorDesc out_desc2(shape2, ge::FORMAT_NCHW, ge::DT_FLOAT);
    out_desc2.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc2.SetOriginDataType(ge::DT_FLOAT);
    out_desc2.SetOriginShape(shape2);
    GeTensorDesc in_desc(shape2, ge::FORMAT_NCHW, ge::DT_FLOAT);
    in_desc.SetOriginFormat(ge::FORMAT_NCHW);
    in_desc.SetOriginDataType(ge::DT_FLOAT);
    in_desc.SetOriginShape(shape2);
    (void)ge::AttrUtils::SetInt(conv, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
    (void)ge::AttrUtils::SetInt(op_a, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
    (void)ge::AttrUtils::SetInt(op_b, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
    (void)ge::AttrUtils::SetInt(concat_1, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
    (void)ge::AttrUtils::SetInt(concat_2, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
    (void)ge::AttrUtils::SetInt(concat_3, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);

    conv->AddOutputDesc(out_desc1);
    op_a->AddOutputDesc(out_desc2);
    op_b->AddOutputDesc(out_desc1);
    concat_1->AddInputDesc(in_desc);
    concat_1->AddInputDesc(in_desc);
    concat_1->AddOutputDesc(out_desc1);
    concat_2->AddInputDesc(in_desc);
    concat_2->AddInputDesc(in_desc);
    concat_2->AddOutputDesc(out_desc2);
    concat_3->AddInputDesc(in_desc);
    concat_3->AddInputDesc(in_desc);
    concat_3->AddOutputDesc(out_desc2);

    // create nodes
    NodePtr conv_node = graph->AddNode(conv);
    NodePtr a_node = graph->AddNode(op_a);
    NodePtr b_node = graph->AddNode(op_b);
    NodePtr concat1_node = graph->AddNode(concat_1);
    NodePtr concat2_node = graph->AddNode(concat_2);
    NodePtr concat3_node = graph->AddNode(concat_3);
    /*
     *  A      Conv2d       B
     *   \       / \       /
     *    \     /   \ (1) /(0)
     *     Concat   Concat(concat_dim=1)
     *        \        /
     *         \      /
     *          Concat
     */
    ge::GraphUtils::AddEdge(a_node->GetOutDataAnchor(0),
                            concat1_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0),
                            concat1_node->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0),
                            concat2_node->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(b_node->GetOutDataAnchor(0),
                            concat2_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(concat1_node->GetOutDataAnchor(0),
                            concat3_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(concat2_node->GetOutDataAnchor(0),
                            concat3_node->GetInDataAnchor(1));
  }

  void InitGraph12(ComputeGraphPtr& graph) {
    OpDescPtr relu1 = std::make_shared<OpDesc>("relu1", RELU);
    OpDescPtr relu2 = std::make_shared<OpDesc>("relu2", RELU);
    OpDescPtr concat = std::make_shared<OpDesc>("concat", CONCATD);
    (void)ge::AttrUtils::SetInt(concat, CONCAT_DIM, 1);

    // add descriptor
    ge::GeShape shape1({1, -1, 14, 14});
    ge::GeShape shape2({1, -1, 14, 14});
    GeTensorDesc out_desc1(shape1, ge::FORMAT_NC1HWC0, ge::DT_FLOAT);
    out_desc1.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc1.SetOriginDataType(ge::DT_FLOAT);
    out_desc1.SetOriginShape(shape1);
    GeTensorDesc out_desc2(shape2, ge::FORMAT_NC1HWC0, ge::DT_FLOAT);
    out_desc2.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc2.SetOriginDataType(ge::DT_FLOAT);
    out_desc2.SetOriginShape(shape2);
    relu1->AddOutputDesc(out_desc1);
    relu2->AddOutputDesc(out_desc2);
    concat->AddInputDesc(out_desc1);
    concat->AddInputDesc(out_desc2);

    // create nodes
    NodePtr relu1_node = graph->AddNode(relu1);
    NodePtr relu2_node = graph->AddNode(relu2);
    NodePtr concat_node = graph->AddNode(concat);
    /*
     *    Relu     Relu
     *      \       /
     *        Concat(concat_dim=0)
     *          |
     */
    ge::GraphUtils::AddEdge(relu1_node->GetOutDataAnchor(0),
                            concat_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(relu2_node->GetOutDataAnchor(0),
                            concat_node->GetInDataAnchor(1));
  }

  void InitGraph13(ComputeGraphPtr& graph) {
    OpDescPtr conv1 = std::make_shared<OpDesc>("conv1", CONV2D);
    OpDescPtr conv2 = std::make_shared<OpDesc>("conv2", CONV2D);
    OpDescPtr concat = std::make_shared<OpDesc>("concat", CONCATD);
    (void)ge::AttrUtils::SetInt(concat, CONCAT_DIM, 1);
    // add descriptor
    ge::GeShape ori_shape1({1, 16, 16, 16});
    ge::GeShape ori_shape2({1, 16, 16, 16});
    ge::GeShape ori_shape3({1, 32, 16, 16});
    ge::GeShape shape({1, 1, 16, 16, 16});
    ge::GeShape shape2({1, 2, 16, 16, 16});
    GeTensorDesc out_desc1(shape, ge::FORMAT_NC1HWC0, ge::DT_FLOAT);
    out_desc1.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc1.SetOriginDataType(ge::DT_FLOAT);
    out_desc1.SetOriginShape(ori_shape1);
    GeTensorDesc out_desc2(shape, ge::FORMAT_NC1HWC0, ge::DT_FLOAT);
    out_desc2.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc2.SetOriginDataType(ge::DT_FLOAT);
    out_desc2.SetOriginShape(ori_shape2);
    GeTensorDesc out_desc3(shape2, ge::FORMAT_NC1HWC0, ge::DT_FLOAT);
    out_desc2.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc2.SetOriginDataType(ge::DT_FLOAT);
    out_desc2.SetOriginShape(ori_shape3);
    conv1->AddInputDesc(out_desc1);
    conv1->AddOutputDesc(out_desc1);
    conv2->AddInputDesc(out_desc2);
    conv2->AddOutputDesc(out_desc2);
    concat->AddInputDesc(out_desc1);
    concat->AddInputDesc(out_desc2);
    concat->AddOutputDesc(out_desc3);
    // create nodes
    NodePtr conv1_node = graph->AddNode(conv1);
    NodePtr conv2_node = graph->AddNode(conv2);
    NodePtr concat_node = graph->AddNode(concat);
    /*
     *     conv2d   conv2d
     *        \   /
     *        Concat(concat_dim=0)
     *          |
     */
    ge::GraphUtils::AddEdge(conv1_node->GetOutDataAnchor(0),
                            concat_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(conv2_node->GetOutDataAnchor(0),
                            concat_node->GetInDataAnchor(1));
  }
};

TEST_F(UTEST_concat_c_optimize, do_optimize_success_0) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>(GRAPH_NAME);
  InitGraph1(graph);
  ConcatCOptimizeFusionPass pass;
  vector<GraphPass*> passes = {&pass};
  Status ret = PassManager::Run(*graph, passes, ops_kernel_info_store_ptr);
  EXPECT_EQ(ret, fe::NOT_CHANGED);
  auto concat_node = graph->FindNode("concat");
  bool attr_notask = false;
  (void)ge::AttrUtils::GetBool(concat_node->GetOpDesc(), ATTR_NAME_NOTASK, attr_notask);
  EXPECT_EQ(attr_notask, false);

  ge::ComputeGraphPtr graph1 = std::make_shared<ComputeGraph>(GRAPH_NAME);
  InitGraph2(graph1);
  ret = PassManager::Run(*graph1, passes, ops_kernel_info_store_ptr);
  EXPECT_EQ(ret, fe::NOT_CHANGED);
  auto concat_node1 = graph1->FindNode("concat");
  (void)ge::AttrUtils::GetBool(concat_node1->GetOpDesc(), ATTR_NAME_NOTASK, attr_notask);
  EXPECT_EQ(attr_notask, false);
}

TEST_F(UTEST_concat_c_optimize, do_optimize_fail_0) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>(GRAPH_NAME);
  InitGraph9(graph);
  ConcatCOptimizeFusionPass pass;
  vector<GraphPass*> passes = {&pass};
  Status ret = PassManager::Run(*graph, passes, ops_kernel_info_store_ptr);
  EXPECT_EQ(ret, fe::NOT_CHANGED);
  auto concat_node = graph->FindNode("concat");
  bool attr_notask = false;
  (void)ge::AttrUtils::GetBool(concat_node->GetOpDesc(), ATTR_NAME_NOTASK, attr_notask);
  EXPECT_EQ(attr_notask, false);

  ge::ComputeGraphPtr graph1 = std::make_shared<ComputeGraph>(GRAPH_NAME);
  InitGraph3(graph1);
  ret = PassManager::Run(*graph1, passes, ops_kernel_info_store_ptr);
  EXPECT_EQ(ret, fe::NOT_CHANGED);
  auto concat_node1 = graph1->FindNode("concat");
  (void)ge::AttrUtils::GetBool(concat_node1->GetOpDesc(), ATTR_NAME_NOTASK, attr_notask);
  EXPECT_EQ(attr_notask, false);

  ge::ComputeGraphPtr graph2 = std::make_shared<ComputeGraph>(GRAPH_NAME);
  InitGraph4(graph2);
  ret = PassManager::Run(*graph2, passes, ops_kernel_info_store_ptr);
  EXPECT_EQ(ret, fe::NOT_CHANGED);
  auto concat_node2 = graph2->FindNode("concat");
  (void)ge::AttrUtils::GetBool(concat_node2->GetOpDesc(), ATTR_NAME_NOTASK, attr_notask);
  EXPECT_EQ(attr_notask, false);
}

TEST_F(UTEST_concat_c_optimize, check_is_valid_concat) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>(GRAPH_NAME);
  InitGraph5(graph);
  ConcatCOptimizeFusionPass pass;
  vector<GraphPass*> passes = {&pass};
  Status ret = PassManager::Run(*graph, passes, ops_kernel_info_store_ptr);
  EXPECT_EQ(ret, fe::NOT_CHANGED);
  bool attr_notask = false;
  auto concat_node1 = graph->FindNode("concat");
  (void)ge::AttrUtils::GetBool(concat_node1->GetOpDesc(), ATTR_NAME_NOTASK, attr_notask);
  EXPECT_EQ(attr_notask, false);
}

TEST_F(UTEST_concat_c_optimize, check_aligned) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>(GRAPH_NAME);
  InitGraph6(graph);
  ConcatCOptimizeFusionPass pass;

  auto concat_node = graph->FindNode("concat");
  bool res = pass.CheckOutput(concat_node);
  EXPECT_EQ(res, true);

  ge::ComputeGraphPtr graph1 = std::make_shared<ComputeGraph>(GRAPH_NAME);
  InitGraph6(graph1);

  auto concat_node1 = graph1->FindNode("concat");
  res = pass.CheckOutput(concat_node1);
  EXPECT_EQ(res, true);

  ge::ComputeGraphPtr graph2 = std::make_shared<ComputeGraph>(GRAPH_NAME);
  InitGraph7(graph2);
  auto concat_node2 = graph2->FindNode("concat");
  res = pass.CheckOutput(concat_node2);
  EXPECT_EQ(res, false);

  ge::ComputeGraphPtr graph3 = std::make_shared<ComputeGraph>(GRAPH_NAME);
  InitGraph8(graph3);
  auto concat_node3 = graph3->FindNode("concat");
  res = pass.CheckPeerNodeCanReUsed(concat_node3);
  EXPECT_EQ(res, false);

  bool bres = false;
  pass.CheckAndSetAttrForConcat(concat_node3, concat_node3->GetOpDesc(), bres);
  pass.CheckIsLxSlicedOp(concat_node3->GetOpDesc(), bres);
  pass.SetStrideWriteInfoForInputs(concat_node3, bres);
}

TEST_F(UTEST_concat_c_optimize, do_optimize_success_1) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>(GRAPH_NAME);
  OpDescPtr conv1 = std::make_shared<OpDesc>("conv1", CONV2D);
  OpDescPtr conv2 = std::make_shared<OpDesc>("conv2", "Conv2DTransposeD");
  OpDescPtr relu1 = std::make_shared<OpDesc>("relu1", "Relu");
  OpDescPtr relu2 = std::make_shared<OpDesc>("relu2", "Relu");
  OpDescPtr concat = std::make_shared<OpDesc>("concat", CONCATD);
  (void)ge::AttrUtils::SetInt(concat, CONCAT_DIM, 1);
  // add descriptor
  ge::GeShape ori_shape1({1, 16, 16, 16});
  ge::GeShape ori_shape2({1, 16, 16, 16});
  ge::GeShape ori_shape3({1, 32, 16, 16});
  ge::GeShape shape({1, 1, 16, 16, 16});
  ge::GeShape shape2({1, 2, 16, 16, 16});
  GeTensorDesc out_desc1(shape, ge::FORMAT_NC1HWC0, ge::DT_FLOAT);
  out_desc1.SetOriginFormat(ge::FORMAT_NCHW);
  out_desc1.SetOriginDataType(ge::DT_FLOAT);
  out_desc1.SetOriginShape(ori_shape1);
  GeTensorDesc out_desc2(shape, ge::FORMAT_NC1HWC0, ge::DT_FLOAT);
  out_desc2.SetOriginFormat(ge::FORMAT_NCHW);
  out_desc2.SetOriginDataType(ge::DT_FLOAT);
  out_desc2.SetOriginShape(ori_shape2);
  GeTensorDesc out_desc3(shape2, ge::FORMAT_NC1HWC0, ge::DT_FLOAT);
  out_desc2.SetOriginFormat(ge::FORMAT_NCHW);
  out_desc2.SetOriginDataType(ge::DT_FLOAT);
  out_desc2.SetOriginShape(ori_shape3);
  conv1->AddInputDesc(out_desc1);
  conv1->AddOutputDesc(out_desc1);
  conv2->AddInputDesc(out_desc2);
  conv2->AddOutputDesc(out_desc2);
  relu1->AddInputDesc(out_desc1);
  relu1->AddOutputDesc(out_desc1);
  relu2->AddInputDesc(out_desc2);
  relu2->AddOutputDesc(out_desc2);
  concat->AddInputDesc(out_desc1);
  concat->AddInputDesc(out_desc2);
  concat->AddOutputDesc(out_desc3);

  // create nodes
  NodePtr conv1_node = graph->AddNode(conv1);
  NodePtr conv2_node = graph->AddNode(conv2);
  NodePtr relu1_node = graph->AddNode(relu1);
  NodePtr relu2_node = graph->AddNode(relu2);
  NodePtr concat_node = graph->AddNode(concat);
  /*
   *  Conv2D     Conv2DTransposeD
   *      \       /
   *     Relu   Relu
   *        \   /
   *        Concat(concat_dim=0)
   *          |
   */
  ge::GraphUtils::AddEdge(conv1_node->GetOutDataAnchor(0),
                          relu1_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(conv2_node->GetOutDataAnchor(0),
                          relu2_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(relu1_node->GetOutDataAnchor(0),
                          concat_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(relu2_node->GetOutDataAnchor(0),
                          concat_node->GetInDataAnchor(1));
  ConcatCOptimizeFusionPass pass;
  vector<GraphPass*> passes = {&pass};
  Status ret = PassManager::Run(*graph, passes, ops_kernel_info_store_ptr);
  EXPECT_EQ(ret, fe::SUCCESS);
  bool attr_notask = false;
  (void)ge::AttrUtils::GetBool(concat_node->GetOpDesc(), ATTR_NAME_NOTASK, attr_notask);
  EXPECT_EQ(attr_notask, true);
}

TEST_F(UTEST_concat_c_optimize, do_optimize_not_changed) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>(GRAPH_NAME);
  OpDescPtr conv1 = std::make_shared<OpDesc>("conv1", CONV2D);
  OpDescPtr conv2 = std::make_shared<OpDesc>("conv2", "Conv2DTransposeD");
  OpDescPtr relu1 = std::make_shared<OpDesc>("relu1", "Relu");
  OpDescPtr abs1 = std::make_shared<OpDesc>("abs1", "Abs");
  OpDescPtr concat = std::make_shared<OpDesc>("concat", CONCATD);
  (void)ge::AttrUtils::SetInt(concat, CONCAT_DIM, 1);
  // add descriptor
  ge::GeShape shape1({1, 1, 16, 16});
  ge::GeShape shape2({1, 2, 16, 16});
  GeTensorDesc out_desc1(shape1, ge::FORMAT_NC1HWC0, ge::DT_FLOAT);
  out_desc1.SetOriginFormat(ge::FORMAT_NCHW);
  out_desc1.SetOriginDataType(ge::DT_FLOAT);
  out_desc1.SetOriginShape(shape1);
  GeTensorDesc out_desc2(shape2, ge::FORMAT_NC1HWC0, ge::DT_FLOAT);
  out_desc2.SetOriginFormat(ge::FORMAT_NCHW);
  out_desc2.SetOriginDataType(ge::DT_FLOAT);
  out_desc2.SetOriginShape(shape2);
  conv1->AddInputDesc(out_desc1);
  conv1->AddOutputDesc(out_desc1);
  conv2->AddInputDesc(out_desc1);
  conv2->AddOutputDesc(out_desc1);
  relu1->AddInputDesc(out_desc1);
  relu1->AddOutputDesc(out_desc1);
  abs1->AddInputDesc(out_desc1);
  abs1->AddOutputDesc(out_desc1);
  concat->AddInputDesc(out_desc1);
  concat->AddInputDesc(out_desc1);
  concat->AddOutputDesc(out_desc2);
  // create nodes
  NodePtr conv1_node = graph->AddNode(conv1);
  NodePtr conv2_node = graph->AddNode(conv2);
  NodePtr relu1_node = graph->AddNode(relu1);
  NodePtr abs1_node = graph->AddNode(abs1);
  NodePtr concat_node = graph->AddNode(concat);
  /*
   *  Conv2D     Conv2DTransposeD
   *      \       /
   *     Relu   Abs
   *        \   /
   *        Concat(concat_dim=0)
   *          |
   */
  ge::GraphUtils::AddEdge(conv1_node->GetOutDataAnchor(0),
                          relu1_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(conv2_node->GetOutDataAnchor(0),
                          abs1_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(relu1_node->GetOutDataAnchor(0),
                          concat_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(abs1_node->GetOutDataAnchor(0),
                          concat_node->GetInDataAnchor(1));
  ConcatCOptimizeFusionPass pass;
  vector<GraphPass*> passes = {&pass};
  Status ret = PassManager::Run(*graph, passes, ops_kernel_info_store_ptr);
  EXPECT_EQ(ret, fe::NOT_CHANGED);
  bool attr_notask = false;
  (void)ge::AttrUtils::GetBool(concat_node->GetOpDesc(), ATTR_NAME_NOTASK, attr_notask);
  EXPECT_EQ(attr_notask, false);
}

TEST_F(UTEST_concat_c_optimize, check_alignment_fail) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>(GRAPH_NAME);
  OpDescPtr conv1 = std::make_shared<OpDesc>("conv1", CONV2D);
  OpDescPtr conv2 = std::make_shared<OpDesc>("conv2", CONV2D);
  OpDescPtr concat = std::make_shared<OpDesc>("concat", CONCATD);
  (void)ge::AttrUtils::SetInt(concat, CONCAT_DIM, 1);
  // add descriptor
  ge::GeShape ori_shape1({1, 5, 16, 16});
  ge::GeShape ori_shape2({1, 5, 16, 16});
  ge::GeShape ori_shape3({1, 10, 16, 16});
  ge::GeShape shape({1, 1, 16, 16, 16});
  GeTensorDesc out_desc1(shape, ge::FORMAT_NC1HWC0, ge::DT_FLOAT);
  out_desc1.SetOriginFormat(ge::FORMAT_NCHW);
  out_desc1.SetOriginDataType(ge::DT_FLOAT);
  out_desc1.SetOriginShape(ori_shape1);
  GeTensorDesc out_desc2(shape, ge::FORMAT_NC1HWC0, ge::DT_FLOAT);
  out_desc2.SetOriginFormat(ge::FORMAT_NCHW);
  out_desc2.SetOriginDataType(ge::DT_FLOAT);
  out_desc2.SetOriginShape(ori_shape2);
  GeTensorDesc out_desc3(shape, ge::FORMAT_NC1HWC0, ge::DT_FLOAT);
  out_desc2.SetOriginFormat(ge::FORMAT_NCHW);
  out_desc2.SetOriginDataType(ge::DT_FLOAT);
  out_desc2.SetOriginShape(ori_shape3);
  conv1->AddInputDesc(out_desc1);
  conv1->AddOutputDesc(out_desc1);
  conv2->AddInputDesc(out_desc2);
  conv2->AddOutputDesc(out_desc2);
  concat->AddInputDesc(out_desc1);
  concat->AddInputDesc(out_desc2);
  concat->AddOutputDesc(out_desc3);
  // create nodes
  NodePtr conv1_node = graph->AddNode(conv1);
  NodePtr conv2_node = graph->AddNode(conv2);
  NodePtr concat_node = graph->AddNode(concat);
  /*
   *     conv2d   conv2d
   *        \   /
   *        Concat(concat_dim=0)
   *          |
   */
  ge::GraphUtils::AddEdge(conv1_node->GetOutDataAnchor(0),
                          concat_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(conv2_node->GetOutDataAnchor(0),
                          concat_node->GetInDataAnchor(1));
  ConcatCOptimizeFusionPass pass;
  vector<GraphPass*> passes = {&pass};
  Status ret = PassManager::Run(*graph, passes, ops_kernel_info_store_ptr);
  EXPECT_EQ(ret, fe::NOT_CHANGED);
  bool attr_notask = false;
  (void)ge::AttrUtils::GetBool(concat_node->GetOpDesc(), ATTR_NAME_NOTASK, attr_notask);
  EXPECT_EQ(attr_notask, false);
}

TEST_F(UTEST_concat_c_optimize, check_alignment_succ) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>(GRAPH_NAME);
  OpDescPtr conv1 = std::make_shared<OpDesc>("conv1", CONV2D);
  OpDescPtr conv2 = std::make_shared<OpDesc>("conv2", CONV2D);
  OpDescPtr concat = std::make_shared<OpDesc>("concat", CONCATD);
  (void)ge::AttrUtils::SetInt(concat, CONCAT_DIM, 1);
  // add descriptor
  ge::GeShape ori_shape1({1, 16, 16, 16});
  ge::GeShape ori_shape2({1, 16, 16, 16});
  ge::GeShape ori_shape3({1, 32, 16, 16});
  ge::GeShape shape({1, 1, 16, 16, 16});
  ge::GeShape shape2({1, 2, 16, 16, 16});
  GeTensorDesc out_desc1(shape, ge::FORMAT_NC1HWC0, ge::DT_FLOAT);
  out_desc1.SetOriginFormat(ge::FORMAT_NCHW);
  out_desc1.SetOriginDataType(ge::DT_FLOAT);
  out_desc1.SetOriginShape(ori_shape1);
  GeTensorDesc out_desc2(shape, ge::FORMAT_NC1HWC0, ge::DT_FLOAT);
  out_desc2.SetOriginFormat(ge::FORMAT_NCHW);
  out_desc2.SetOriginDataType(ge::DT_FLOAT);
  out_desc2.SetOriginShape(ori_shape2);
  GeTensorDesc out_desc3(shape2, ge::FORMAT_NC1HWC0, ge::DT_FLOAT);
  out_desc2.SetOriginFormat(ge::FORMAT_NCHW);
  out_desc2.SetOriginDataType(ge::DT_FLOAT);
  out_desc2.SetOriginShape(ori_shape3);
  conv1->AddInputDesc(out_desc1);
  conv1->AddOutputDesc(out_desc1);
  conv2->AddInputDesc(out_desc2);
  conv2->AddOutputDesc(out_desc2);
  concat->AddInputDesc(out_desc1);
  concat->AddInputDesc(out_desc2);
  concat->AddOutputDesc(out_desc3);
  // create nodes
  NodePtr conv1_node = graph->AddNode(conv1);
  NodePtr conv2_node = graph->AddNode(conv2);
  NodePtr concat_node = graph->AddNode(concat);
  /*
   *     conv2d   conv2d
   *        \   /
   *        Concat(concat_dim=0)
   *          |
   */
  ge::GraphUtils::AddEdge(conv1_node->GetOutDataAnchor(0),
                          concat_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(conv2_node->GetOutDataAnchor(0),
                          concat_node->GetInDataAnchor(1));
  ConcatCOptimizeFusionPass pass;
  vector<GraphPass*> passes = {&pass};
  Status ret = PassManager::Run(*graph, passes, ops_kernel_info_store_ptr);
  EXPECT_EQ(ret, fe::SUCCESS);
  bool attr_notask = false;
  (void)ge::AttrUtils::GetBool(concat_node->GetOpDesc(), ATTR_NAME_NOTASK, attr_notask);
  EXPECT_EQ(attr_notask, true);
}

TEST_F(UTEST_concat_c_optimize, get_concat_dim_fail_not_changed) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>(GRAPH_NAME);
  OpDescPtr conv1 = std::make_shared<OpDesc>("conv1", CONV2D);
  OpDescPtr conv2 = std::make_shared<OpDesc>("conv2", CONV2D);
  OpDescPtr concat = std::make_shared<OpDesc>("concat", CONCATD);
  // add descriptor
  ge::GeShape ori_shape1({1, 16, 16, 16});
  ge::GeShape ori_shape2({1, 16, 16, 16});
  ge::GeShape ori_shape3({1, 32, 16, 16});
  ge::GeShape shape({1, 1, 16, 16, 16});
  ge::GeShape shape2({1, 2, 16, 16, 16});
  GeTensorDesc out_desc1(shape, ge::FORMAT_NC1HWC0, ge::DT_FLOAT);
  out_desc1.SetOriginFormat(ge::FORMAT_NCHW);
  out_desc1.SetOriginDataType(ge::DT_FLOAT);
  out_desc1.SetOriginShape(ori_shape1);
  GeTensorDesc out_desc2(shape, ge::FORMAT_NC1HWC0, ge::DT_FLOAT);
  out_desc2.SetOriginFormat(ge::FORMAT_NCHW);
  out_desc2.SetOriginDataType(ge::DT_FLOAT);
  out_desc2.SetOriginShape(ori_shape2);
  GeTensorDesc out_desc3(shape2, ge::FORMAT_NC1HWC0, ge::DT_FLOAT);
  out_desc2.SetOriginFormat(ge::FORMAT_NCHW);
  out_desc2.SetOriginDataType(ge::DT_FLOAT);
  out_desc2.SetOriginShape(ori_shape3);
  conv1->AddInputDesc(out_desc1);
  conv1->AddOutputDesc(out_desc1);
  conv2->AddInputDesc(out_desc2);
  conv2->AddOutputDesc(out_desc2);
  concat->AddInputDesc(out_desc1);
  concat->AddInputDesc(out_desc2);
  concat->AddOutputDesc(out_desc3);
  // create nodes
  NodePtr conv1_node = graph->AddNode(conv1);
  NodePtr conv2_node = graph->AddNode(conv2);
  NodePtr concat_node = graph->AddNode(concat);
  /*
   *     conv2d   conv2d
   *        \   /
   *        Concat(concat_dim=0)
   *          |
   */
  ge::GraphUtils::AddEdge(conv1_node->GetOutDataAnchor(0),
                          concat_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(conv2_node->GetOutDataAnchor(0),
                          concat_node->GetInDataAnchor(1));
  ConcatCOptimizeFusionPass pass;
  vector<GraphPass*> passes = {&pass};
  Status ret = PassManager::Run(*graph, passes, ops_kernel_info_store_ptr);
  EXPECT_EQ(ret, fe::NOT_CHANGED);
  bool attr_notask = false;
  (void)ge::AttrUtils::GetBool(concat_node->GetOpDesc(), ATTR_NAME_NOTASK, attr_notask);
  EXPECT_EQ(attr_notask, false);
}

TEST_F(UTEST_concat_c_optimize, get_concat_dim_negative_succ) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>(GRAPH_NAME);
  OpDescPtr conv1 = std::make_shared<OpDesc>("conv1", CONV2D);
  OpDescPtr conv2 = std::make_shared<OpDesc>("conv2", CONV2D);
  OpDescPtr concat = std::make_shared<OpDesc>("concat", CONCATD);
  (void)ge::AttrUtils::SetInt(concat, CONCAT_DIM, -3);
  // add descriptor
  ge::GeShape ori_shape1({1, 16, 16, 16});
  ge::GeShape ori_shape2({1, 16, 16, 16});
  ge::GeShape ori_shape3({1, 32, 16, 16});
  ge::GeShape shape({1, 1, 16, 16, 16});
  ge::GeShape shape2({1, 2, 16, 16, 16});
  GeTensorDesc out_desc1(shape, ge::FORMAT_NC1HWC0, ge::DT_FLOAT);
  out_desc1.SetOriginFormat(ge::FORMAT_NCHW);
  out_desc1.SetOriginDataType(ge::DT_FLOAT);
  out_desc1.SetOriginShape(ori_shape1);
  GeTensorDesc out_desc2(shape, ge::FORMAT_NC1HWC0, ge::DT_FLOAT);
  out_desc2.SetOriginFormat(ge::FORMAT_NCHW);
  out_desc2.SetOriginDataType(ge::DT_FLOAT);
  out_desc2.SetOriginShape(ori_shape2);
  GeTensorDesc out_desc3(shape2, ge::FORMAT_NC1HWC0, ge::DT_FLOAT);
  out_desc2.SetOriginFormat(ge::FORMAT_NCHW);
  out_desc2.SetOriginDataType(ge::DT_FLOAT);
  out_desc2.SetOriginShape(ori_shape3);
  conv1->AddInputDesc(out_desc1);
  conv1->AddOutputDesc(out_desc1);
  conv2->AddInputDesc(out_desc2);
  conv2->AddOutputDesc(out_desc2);
  concat->AddInputDesc(out_desc1);
  concat->AddInputDesc(out_desc2);
  concat->AddOutputDesc(out_desc3);
  // create nodes
  NodePtr conv1_node = graph->AddNode(conv1);
  NodePtr conv2_node = graph->AddNode(conv2);
  NodePtr concat_node = graph->AddNode(concat);
  /*
   *     conv2d   conv2d
   *        \   /
   *        Concat(concat_dim=0)
   *          |
   */
  ge::GraphUtils::AddEdge(conv1_node->GetOutDataAnchor(0),
                          concat_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(conv2_node->GetOutDataAnchor(0),
                          concat_node->GetInDataAnchor(1));
  ConcatCOptimizeFusionPass pass;
  vector<GraphPass*> passes = {&pass};
  Status ret = PassManager::Run(*graph, passes, ops_kernel_info_store_ptr);
  EXPECT_EQ(ret, fe::SUCCESS);
  bool attr_notask = false;
  (void)ge::AttrUtils::GetBool(concat_node->GetOpDesc(), ATTR_NAME_NOTASK, attr_notask);
  EXPECT_EQ(attr_notask, true);
}

TEST_F(UTEST_concat_c_optimize, root_graph_unknown_succ) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>(GRAPH_NAME);
  InitGraph13(graph);

  ge::ComputeGraphPtr root_graph = std::make_shared<ge::ComputeGraph>("root_graph");
  ge::AttrUtils::SetBool(root_graph, "_graph_unknown_flag", true);
  graph->SetParentGraph(root_graph);

  ConcatCOptimizeFusionPass pass;
  vector<GraphPass*> passes = {&pass};
  Status ret = PassManager::Run(*graph, passes, ops_kernel_info_store_ptr);
  EXPECT_EQ(ret, fe::SUCCESS);
  auto concat_node = graph->FindNode("concat");
  bool attr_notask = false;
  (void)ge::AttrUtils::GetBool(concat_node->GetOpDesc(), ATTR_NAME_NOTASK, attr_notask);
  EXPECT_EQ(attr_notask, true);
}

TEST_F(UTEST_concat_c_optimize, owner_graph_unknown_no_changed_1) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>(GRAPH_NAME);
  InitGraph13(graph);

  ge::AttrUtils::SetBool(graph, "_dynamic_shape_partitioned", true);

  ConcatCOptimizeFusionPass pass;
  vector<GraphPass*> passes = {&pass};
  Status ret = PassManager::Run(*graph, passes, ops_kernel_info_store_ptr);
  EXPECT_EQ(ret, fe::NOT_CHANGED);
  auto concat_node = graph->FindNode("concat");
  bool attr_notask = false;
  (void)ge::AttrUtils::GetBool(concat_node->GetOpDesc(), ATTR_NAME_NOTASK, attr_notask);
  EXPECT_EQ(attr_notask, false);
}

TEST_F(UTEST_concat_c_optimize, owner_graph_unknown_no_changed_2) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>(GRAPH_NAME);
  InitGraph13(graph);

  ge::AttrUtils::SetBool(graph, "_graph_unknown_flag", true);

  ConcatCOptimizeFusionPass pass;
  vector<GraphPass*> passes = {&pass};
  Status ret = PassManager::Run(*graph, passes, ops_kernel_info_store_ptr);
  EXPECT_EQ(ret, fe::NOT_CHANGED);
  auto concat_node = graph->FindNode("concat");
  bool attr_notask = false;
  (void)ge::AttrUtils::GetBool(concat_node->GetOpDesc(), ATTR_NAME_NOTASK, attr_notask);
  EXPECT_EQ(attr_notask, false);
}
}
