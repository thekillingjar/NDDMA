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
#include "graph_optimizer/node_optimizer/split_c_to_n_optimizer.h"

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

class STEST_split_c_to_n_optimize : public testing::Test {
 public:
  const string GRAPH_NAME = "test";
  const string SPLITD = "SplitD";
  const string CONCATD = "ConcatD";
  const string DEQUANT = "AscendDequant";
  const string QUANT = "AscendQuant";
  const string CONV2D = "Conv2D";
  const string RELU = "Relu";
  const string STRIDEDWRITE = "StridedWrite";
  const string STRIDEDREAD = "StridedRead";
  const string STRIDE_ATTR_STRIDE = "stride";
  const string STRIDE_ATTR_AXIS = "axis";
  const string DATA = "Data";
  const int NCHW_DIM_N = 1;
  const int NCHW_DIM_H = 14;
  const int NCHW_DIM_W = 12;

protected:
  void SetUp() {}
  void TearDown() {}
  void InitGraph1(ComputeGraphPtr& graph) {
    //format连续
    OpDescPtr quant1 = std::make_shared<OpDesc>("quant1", QUANT);
    OpDescPtr quant2 = std::make_shared<OpDesc>("quant2", QUANT);
    OpDescPtr split = std::make_shared<OpDesc>("split", SPLITD);
    (void)ge::AttrUtils::SetInt(split, SPLIT_DIM, 1);
    (void)ge::AttrUtils::SetInt(quant1, ge::ATTR_NAME_IMPLY_TYPE, static_cast<int>(domi::ImplyType::TVM));
    (void)ge::AttrUtils::SetInt(quant2, ge::ATTR_NAME_IMPLY_TYPE, static_cast<int>(domi::ImplyType::TVM));

    // add descriptor
    ge::GeShape shape1({1, 32, 14, 14});
    ge::GeShape shape2({1, 32, 14, 14});
    ge::GeShape shape3({1, 64, 14, 14});
    GeTensorDesc out_desc1(shape1, ge::FORMAT_NCHW, ge::DT_INT32);
    GeTensorDesc out_desc2(shape2, ge::FORMAT_NCHW, ge::DT_INT32);
    GeTensorDesc out_desc5(shape3, ge::FORMAT_NCHW, ge::DT_INT32);

    out_desc1.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc1.SetOriginDataType(ge::DT_FLOAT);
    out_desc1.SetOriginShape(shape1);
    out_desc2.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc2.SetOriginDataType(ge::DT_FLOAT);
    out_desc2.SetOriginShape(shape2);

    out_desc5.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc5.SetOriginDataType(ge::DT_FLOAT16);
    out_desc5.SetOriginShape(shape3);

    quant1->AddOutputDesc(out_desc1);
    quant2->AddOutputDesc(out_desc2);
    split->AddInputDesc(out_desc1);

    GeTensorDesc out_desc3(shape1, ge::FORMAT_NCHW, ge::DT_FLOAT16);
    out_desc3.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc3.SetOriginDataType(ge::DT_FLOAT16);
    out_desc3.SetOriginShape(shape1);

    GeTensorDesc out_desc4(shape2, ge::FORMAT_NCHW, ge::DT_FLOAT16);
    out_desc4.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc4.SetOriginDataType(ge::DT_FLOAT16);
    out_desc4.SetOriginShape(shape2);

    quant1->AddInputDesc(out_desc3);
    quant2->AddInputDesc(out_desc4);

    split->AddOutputDesc(out_desc3);
    split->AddOutputDesc(out_desc4);


    // create nodes
    NodePtr quant1_node = graph->AddNode(quant1);
    NodePtr quant2_node = graph->AddNode(quant2);
    NodePtr split_node = graph->AddNode(split);
    /*
     *        split
     *       /     \
     *     /         \
     * AcscendQuant   AcscendQuant (split_dim=1)
     *          
     */
    ge::GraphUtils::AddEdge(split_node->GetOutDataAnchor(0),
                            quant1_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(split_node->GetOutDataAnchor(1),
                            quant2_node->GetInDataAnchor(0));
  }

  void InitGraph2(ComputeGraphPtr& graph) {
    //ND->NZ
    OpDescPtr quant1 = std::make_shared<OpDesc>("quant1", QUANT);
    OpDescPtr quant2 = std::make_shared<OpDesc>("quant2", QUANT);
    OpDescPtr split = std::make_shared<OpDesc>("split", SPLITD);
    (void)ge::AttrUtils::SetInt(split, SPLIT_DIM, 1);
    (void)ge::AttrUtils::SetInt(quant1, ge::ATTR_NAME_IMPLY_TYPE, static_cast<int>(domi::ImplyType::TVM));
    (void)ge::AttrUtils::SetInt(quant2, ge::ATTR_NAME_IMPLY_TYPE, static_cast<int>(domi::ImplyType::TVM));
    // add descriptor
    ge::GeShape shape1({1, 32, 14, 14});
    ge::GeShape shape2({1, 32, 14, 14});
    ge::GeShape shape3({1, 64, 14, 14});
    GeTensorDesc out_desc1(shape1, ge::FORMAT_NCHW, ge::DT_INT32);
    GeTensorDesc out_desc2(shape2, ge::FORMAT_NCHW, ge::DT_INT32);
    GeTensorDesc out_desc5(shape3, ge::FORMAT_FRACTAL_NZ, ge::DT_INT32);

    out_desc1.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc1.SetOriginDataType(ge::DT_FLOAT);
    out_desc1.SetOriginShape(shape1);
    out_desc2.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc2.SetOriginDataType(ge::DT_FLOAT);
    out_desc2.SetOriginShape(shape2);

    out_desc5.SetOriginFormat(ge::FORMAT_ND);
    out_desc5.SetOriginDataType(ge::DT_FLOAT16);
    out_desc5.SetOriginShape(shape3);

    quant1->AddOutputDesc(out_desc1);
    quant2->AddOutputDesc(out_desc2);
    split->AddInputDesc(out_desc5);

    GeTensorDesc out_desc3(shape1, ge::FORMAT_NCHW, ge::DT_FLOAT16);
    out_desc3.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc3.SetOriginDataType(ge::DT_FLOAT16);
    out_desc3.SetOriginShape(shape1);

    GeTensorDesc out_desc4(shape2, ge::FORMAT_NCHW, ge::DT_FLOAT16);
    out_desc4.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc4.SetOriginDataType(ge::DT_FLOAT16);
    out_desc4.SetOriginShape(shape2);

    quant1->AddInputDesc(out_desc3);
    quant2->AddInputDesc(out_desc4);

    split->AddOutputDesc(out_desc3);
    split->AddOutputDesc(out_desc4);


    // create nodes
    NodePtr quant1_node = graph->AddNode(quant1);
    NodePtr quant2_node = graph->AddNode(quant2);
    NodePtr split_node = graph->AddNode(split);
    /*
     *        split
     *       /     \
     *     /         \
     * AcscendQuant   AcscendQuant (split_dim=1)
     *          
     */
    ge::GraphUtils::AddEdge(split_node->GetOutDataAnchor(0),
                            quant1_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(split_node->GetOutDataAnchor(1),
                            quant2_node->GetInDataAnchor(0));
  }

  void InitGraph3(ComputeGraphPtr& graph) {
    //ND->NZ origin shape < 4
    OpDescPtr quant1 = std::make_shared<OpDesc>("quant1", QUANT);
    OpDescPtr quant2 = std::make_shared<OpDesc>("quant2", QUANT);
    OpDescPtr split = std::make_shared<OpDesc>("split", SPLITD);
    (void)ge::AttrUtils::SetInt(split, SPLIT_DIM, 1);
    (void)ge::AttrUtils::SetInt(quant1, ge::ATTR_NAME_IMPLY_TYPE, static_cast<int>(domi::ImplyType::TVM));
    (void)ge::AttrUtils::SetInt(quant2, ge::ATTR_NAME_IMPLY_TYPE, static_cast<int>(domi::ImplyType::TVM));
    // add descriptor
    ge::GeShape shape1({1, 32, 14, 14});
    ge::GeShape shape2({1, 32, 14, 14});
    ge::GeShape shape3({1, 64, 14});
    GeTensorDesc out_desc1(shape1, ge::FORMAT_NCHW, ge::DT_INT32);
    GeTensorDesc out_desc2(shape2, ge::FORMAT_NCHW, ge::DT_INT32);
    GeTensorDesc out_desc5(shape3, ge::FORMAT_FRACTAL_NZ, ge::DT_INT32);

    out_desc1.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc1.SetOriginDataType(ge::DT_FLOAT);
    out_desc1.SetOriginShape(shape1);
    out_desc2.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc2.SetOriginDataType(ge::DT_FLOAT);
    out_desc2.SetOriginShape(shape2);

    out_desc5.SetOriginFormat(ge::FORMAT_ND);
    out_desc5.SetOriginDataType(ge::DT_FLOAT16);
    out_desc5.SetOriginShape(shape3);

    quant1->AddOutputDesc(out_desc1);
    quant2->AddOutputDesc(out_desc2);
    split->AddInputDesc(out_desc5);

    GeTensorDesc out_desc3(shape1, ge::FORMAT_NCHW, ge::DT_FLOAT16);
    out_desc3.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc3.SetOriginDataType(ge::DT_FLOAT16);
    out_desc3.SetOriginShape(shape1);

    GeTensorDesc out_desc4(shape2, ge::FORMAT_NCHW, ge::DT_FLOAT16);
    out_desc4.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc4.SetOriginDataType(ge::DT_FLOAT16);
    out_desc4.SetOriginShape(shape2);

    quant1->AddInputDesc(out_desc3);
    quant2->AddInputDesc(out_desc4);

    split->AddOutputDesc(out_desc3);
    split->AddOutputDesc(out_desc4);


    // create nodes
    NodePtr quant1_node = graph->AddNode(quant1);
    NodePtr quant2_node = graph->AddNode(quant2);
    NodePtr split_node = graph->AddNode(split);
    /*
     *        split
     *       /     \
     *     /         \
     * AcscendQuant   AcscendQuant (split_dim=1)
     *          
     */
    ge::GraphUtils::AddEdge(split_node->GetOutDataAnchor(0),
                            quant1_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(split_node->GetOutDataAnchor(1),
                            quant2_node->GetInDataAnchor(0));
  }

  void InitGraph4(ComputeGraphPtr& graph) {
    //NCHW->5HD
    OpDescPtr quant1 = std::make_shared<OpDesc>("quant1", QUANT);
    OpDescPtr quant2 = std::make_shared<OpDesc>("quant2", QUANT);
    OpDescPtr split = std::make_shared<OpDesc>("split", SPLITD);
    (void)ge::AttrUtils::SetInt(split, SPLIT_DIM, 1);
    (void)ge::AttrUtils::SetInt(quant1, ge::ATTR_NAME_IMPLY_TYPE, static_cast<int>(domi::ImplyType::TVM));
    (void)ge::AttrUtils::SetInt(quant2, ge::ATTR_NAME_IMPLY_TYPE, static_cast<int>(domi::ImplyType::TVM));

    // add descriptor
    ge::GeShape shape1({1, 32, 14, 14});
    ge::GeShape shape2({1, 32, 14, 14});
    ge::GeShape shape3({1, 80, 14, 14});
    GeTensorDesc out_desc1(shape1, ge::FORMAT_NCHW, ge::DT_INT32);
    GeTensorDesc out_desc2(shape2, ge::FORMAT_NCHW, ge::DT_INT32);
    GeTensorDesc out_desc5(shape3, ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);

    out_desc1.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc1.SetOriginDataType(ge::DT_FLOAT);
    out_desc1.SetOriginShape(shape1);
    out_desc2.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc2.SetOriginDataType(ge::DT_FLOAT);
    out_desc2.SetOriginShape(shape2);

    out_desc5.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc5.SetOriginDataType(ge::DT_FLOAT16);
    out_desc5.SetOriginShape(shape3);

    quant1->AddOutputDesc(out_desc1);
    quant2->AddOutputDesc(out_desc2);
    split->AddInputDesc(out_desc5);

    GeTensorDesc out_desc3(shape1, ge::FORMAT_NCHW, ge::DT_FLOAT16);
    out_desc3.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc3.SetOriginDataType(ge::DT_FLOAT16);
    out_desc3.SetOriginShape(shape1);

    GeTensorDesc out_desc4(shape2, ge::FORMAT_NCHW, ge::DT_FLOAT16);
    out_desc4.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc4.SetOriginDataType(ge::DT_FLOAT16);
    out_desc4.SetOriginShape(shape2);

    quant1->AddInputDesc(out_desc3);
    quant2->AddInputDesc(out_desc4);

    split->AddOutputDesc(out_desc5);
    split->AddOutputDesc(out_desc5);


    // create nodes
    NodePtr quant1_node = graph->AddNode(quant1);
    NodePtr quant2_node = graph->AddNode(quant2);
    NodePtr split_node = graph->AddNode(split);
    /*
     *        split
     *       /     \
     *     /         \
     * AcscendQuant   AcscendQuant (split_dim=1)
     *          
     */
    ge::GraphUtils::AddEdge(split_node->GetOutDataAnchor(0),
                            quant1_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(split_node->GetOutDataAnchor(1),
                            quant2_node->GetInDataAnchor(0));
  }

  void InitGraph5(ComputeGraphPtr& graph) {
    //NCHW->5HD Not meet condition_nchw_to_nc1hwc0
    OpDescPtr quant1 = std::make_shared<OpDesc>("quant1", QUANT);
    OpDescPtr quant2 = std::make_shared<OpDesc>("quant2", QUANT);
    OpDescPtr split = std::make_shared<OpDesc>("split", SPLITD);
    (void)ge::AttrUtils::SetInt(split, SPLIT_DIM, 1);
    (void)ge::AttrUtils::SetInt(quant1, ge::ATTR_NAME_IMPLY_TYPE, static_cast<int>(domi::ImplyType::TVM));
    (void)ge::AttrUtils::SetInt(quant2, ge::ATTR_NAME_IMPLY_TYPE, static_cast<int>(domi::ImplyType::TVM));

    // add descriptor
    ge::GeShape shape1({1, 32, 14, 14});
    ge::GeShape shape2({1, 32, 14, 14});
    ge::GeShape shape3({1, 14, 14, 14});
    GeTensorDesc out_desc1(shape1, ge::FORMAT_NCHW, ge::DT_INT32);
    GeTensorDesc out_desc2(shape2, ge::FORMAT_NCHW, ge::DT_INT32);
    GeTensorDesc out_desc5(shape3, ge::FORMAT_NC1HWC0, ge::DT_INT32);

    out_desc1.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc1.SetOriginDataType(ge::DT_FLOAT);
    out_desc1.SetOriginShape(shape1);
    out_desc2.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc2.SetOriginDataType(ge::DT_FLOAT);
    out_desc2.SetOriginShape(shape2);

    out_desc5.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc5.SetOriginDataType(ge::DT_FLOAT16);
    out_desc5.SetOriginShape(shape3);

    quant1->AddOutputDesc(out_desc1);
    quant2->AddOutputDesc(out_desc2);
    split->AddInputDesc(out_desc5);

    GeTensorDesc out_desc3(shape1, ge::FORMAT_NCHW, ge::DT_FLOAT16);
    out_desc3.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc3.SetOriginDataType(ge::DT_FLOAT16);
    out_desc3.SetOriginShape(shape1);

    GeTensorDesc out_desc4(shape2, ge::FORMAT_NCHW, ge::DT_FLOAT16);
    out_desc4.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc4.SetOriginDataType(ge::DT_FLOAT16);
    out_desc4.SetOriginShape(shape2);

    quant1->AddInputDesc(out_desc5);
    quant2->AddInputDesc(out_desc5);

    split->AddOutputDesc(out_desc5);
    split->AddOutputDesc(out_desc5);


    // create nodes
    NodePtr quant1_node = graph->AddNode(quant1);
    NodePtr quant2_node = graph->AddNode(quant2);
    NodePtr split_node = graph->AddNode(split);
    /*
     *        split
     *       /     \
     *     /         \
     * AcscendQuant   AcscendQuant (split_dim=1)
     *          
     */
    ge::GraphUtils::AddEdge(split_node->GetOutDataAnchor(0),
                            quant1_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(split_node->GetOutDataAnchor(1),
                            quant2_node->GetInDataAnchor(0));
  }

  void InitGraph6(ComputeGraphPtr& graph) {
    //IsDynamicGraph
    OpDescPtr quant1 = std::make_shared<OpDesc>("quant1", QUANT);
    OpDescPtr quant2 = std::make_shared<OpDesc>("quant2", QUANT);
    OpDescPtr split = std::make_shared<OpDesc>("split", SPLITD);
    (void)ge::AttrUtils::SetInt(split, SPLIT_DIM, -2);
    (void)ge::AttrUtils::SetInt(quant1, ge::ATTR_NAME_IMPLY_TYPE, static_cast<int>(domi::ImplyType::TVM));
    (void)ge::AttrUtils::SetInt(quant2, ge::ATTR_NAME_IMPLY_TYPE, static_cast<int>(domi::ImplyType::TVM));
    // add descriptor
    ge::GeShape shape1({1, 32, 14, 14});
    ge::GeShape shape2({1, 32, 14, 14});
    ge::GeShape shape3({1, 64, 14, 14});
    GeTensorDesc out_desc1(shape1, ge::FORMAT_NCHW, ge::DT_INT32);
    GeTensorDesc out_desc2(shape2, ge::FORMAT_NCHW, ge::DT_INT32);
    GeTensorDesc out_desc5(shape3, ge::FORMAT_NCHW, ge::DT_INT32);

    out_desc1.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc1.SetOriginDataType(ge::DT_FLOAT);
    out_desc1.SetOriginShape(shape1);
    out_desc2.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc2.SetOriginDataType(ge::DT_FLOAT);
    out_desc2.SetOriginShape(shape2);

    out_desc5.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc5.SetOriginDataType(ge::DT_FLOAT16);
    out_desc5.SetOriginShape(shape3);

    quant1->AddOutputDesc(out_desc1);
    quant2->AddOutputDesc(out_desc2);
    split->AddInputDesc(out_desc1);

    GeTensorDesc out_desc3(shape1, ge::FORMAT_NCHW, ge::DT_FLOAT16);
    out_desc3.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc3.SetOriginDataType(ge::DT_FLOAT16);
    out_desc3.SetOriginShape(shape1);

    GeTensorDesc out_desc4(shape2, ge::FORMAT_NCHW, ge::DT_FLOAT16);
    out_desc4.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc4.SetOriginDataType(ge::DT_FLOAT16);
    out_desc4.SetOriginShape(shape2);

    quant1->AddInputDesc(out_desc3);
    quant2->AddInputDesc(out_desc4);

    split->AddOutputDesc(out_desc3);
    split->AddOutputDesc(out_desc4);


    // create nodes
    NodePtr quant1_node = graph->AddNode(quant1);
    NodePtr quant2_node = graph->AddNode(quant2);
    NodePtr split_node = graph->AddNode(split);
    /*
     *        split
     *       /     \
     *     /         \
     * AcscendQuant   AcscendQuant (split_dim = -2)
     *          
     */
    ge::GraphUtils::AddEdge(split_node->GetOutDataAnchor(0),
                            quant1_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(split_node->GetOutDataAnchor(1),
                            quant2_node->GetInDataAnchor(0));
  }

  void InitGraph7(ComputeGraphPtr& graph) {
    //asix N != 1
    OpDescPtr quant1 = std::make_shared<OpDesc>("quant1", QUANT);
    OpDescPtr quant2 = std::make_shared<OpDesc>("quant2", QUANT);
    OpDescPtr split = std::make_shared<OpDesc>("split", SPLITD);
    (void)ge::AttrUtils::SetInt(split, SPLIT_DIM, 1);
    (void)ge::AttrUtils::SetInt(quant1, ge::ATTR_NAME_IMPLY_TYPE, static_cast<int>(domi::ImplyType::TVM));
    (void)ge::AttrUtils::SetInt(quant2, ge::ATTR_NAME_IMPLY_TYPE, static_cast<int>(domi::ImplyType::TVM));
    // add descriptor
    ge::GeShape shape1({1, 32, 14, 14});
    ge::GeShape shape2({1, 32, 14, 14});
    ge::GeShape shape3({2, 64, 14, 14});
    GeTensorDesc out_desc1(shape1, ge::FORMAT_NCHW, ge::DT_INT32);
    GeTensorDesc out_desc2(shape2, ge::FORMAT_NCHW, ge::DT_INT32);
    GeTensorDesc out_desc5(shape3, ge::FORMAT_NCHW, ge::DT_INT32);

    out_desc1.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc1.SetOriginDataType(ge::DT_FLOAT);
    out_desc1.SetOriginShape(shape1);
    out_desc2.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc2.SetOriginDataType(ge::DT_FLOAT);
    out_desc2.SetOriginShape(shape2);

    out_desc5.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc5.SetOriginDataType(ge::DT_FLOAT16);
    out_desc5.SetOriginShape(shape3);

    quant1->AddOutputDesc(out_desc1);
    quant2->AddOutputDesc(out_desc2);
    split->AddInputDesc(out_desc5);

    GeTensorDesc out_desc3(shape1, ge::FORMAT_NCHW, ge::DT_FLOAT16);
    out_desc3.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc3.SetOriginDataType(ge::DT_FLOAT16);
    out_desc3.SetOriginShape(shape1);

    GeTensorDesc out_desc4(shape2, ge::FORMAT_NCHW, ge::DT_FLOAT16);
    out_desc4.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc4.SetOriginDataType(ge::DT_FLOAT16);
    out_desc4.SetOriginShape(shape2);

    quant1->AddInputDesc(out_desc3);
    quant2->AddInputDesc(out_desc4);

    split->AddOutputDesc(out_desc5);
    split->AddOutputDesc(out_desc5);


    // create nodes
    NodePtr quant1_node = graph->AddNode(quant1);
    NodePtr quant2_node = graph->AddNode(quant2);
    NodePtr split_node = graph->AddNode(split);
    /*
     *        split
     *       /     \
     *     /         \
     * AcscendQuant   AcscendQuant (split_dim=1)
     *          
     */
    ge::GraphUtils::AddEdge(split_node->GetOutDataAnchor(0),
                            quant1_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(split_node->GetOutDataAnchor(1),
                            quant2_node->GetInDataAnchor(0));
  }

    void InitGraph8(ComputeGraphPtr& graph) {
    OpDescPtr relu = std::make_shared<OpDesc>("relu1", RELU);
    OpDescPtr quant1 = std::make_shared<OpDesc>("quant1", QUANT);
    OpDescPtr quant2 = std::make_shared<OpDesc>("quant2", QUANT);
    OpDescPtr split = std::make_shared<OpDesc>("split", SPLITD);
    (void)ge::AttrUtils::SetInt(split, SPLIT_DIM, 1);
    (void)ge::AttrUtils::SetInt(quant1, ge::ATTR_NAME_IMPLY_TYPE, static_cast<int>(domi::ImplyType::TVM));
    (void)ge::AttrUtils::SetInt(quant2, ge::ATTR_NAME_IMPLY_TYPE, static_cast<int>(domi::ImplyType::TVM));
    // add descriptor
    ge::GeShape shape1({1, 32, 14, 14});
    ge::GeShape shape2({1, 32, 14, 14});
    ge::GeShape shape3({1, 64, 14, 14});
    ge::GeShape shape4({1, 64, 14, -1});
    GeTensorDesc out_desc1(shape1, ge::FORMAT_NCHW, ge::DT_INT32);
    GeTensorDesc out_desc2(shape2, ge::FORMAT_NCHW, ge::DT_INT32);
    GeTensorDesc out_desc5(shape3, ge::FORMAT_NCHW, ge::DT_INT32);
    GeTensorDesc out_desc6(shape4, ge::FORMAT_NCHW, ge::DT_INT32);

    out_desc1.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc1.SetOriginDataType(ge::DT_FLOAT);
    out_desc1.SetOriginShape(shape1);
    out_desc2.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc2.SetOriginDataType(ge::DT_FLOAT);
    out_desc2.SetOriginShape(shape2);

    out_desc5.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc5.SetOriginDataType(ge::DT_FLOAT16);
    out_desc5.SetOriginShape(shape3);

    quant1->AddOutputDesc(out_desc1);
    quant2->AddOutputDesc(out_desc2);
    split->AddInputDesc(out_desc1);
    relu->AddOutputDesc(out_desc6);

    GeTensorDesc out_desc3(shape1, ge::FORMAT_NCHW, ge::DT_FLOAT16);
    out_desc3.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc3.SetOriginDataType(ge::DT_FLOAT16);
    out_desc3.SetOriginShape(shape1);

    GeTensorDesc out_desc4(shape2, ge::FORMAT_NCHW, ge::DT_FLOAT16);
    out_desc4.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc4.SetOriginDataType(ge::DT_FLOAT16);
    out_desc4.SetOriginShape(shape2);

    quant1->AddInputDesc(out_desc3);
    quant2->AddInputDesc(out_desc4);

    split->AddOutputDesc(out_desc3);
    split->AddOutputDesc(out_desc4);


    // create nodes
    NodePtr relu_node = graph->AddNode(relu);
    NodePtr quant1_node = graph->AddNode(quant1);
    NodePtr quant2_node = graph->AddNode(quant2);
    NodePtr split_node = graph->AddNode(split);
    /*
     *        split
     *       /     \
     *     /         \
     * AcscendQuant   AcscendQuant (split_dim=1)
     *
     */
    ge::GraphUtils::AddEdge(relu_node->GetOutDataAnchor(0),
                            split_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(split_node->GetOutDataAnchor(0),
                            quant1_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(split_node->GetOutDataAnchor(1),
                            quant2_node->GetInDataAnchor(0));
    graph->SetGraphUnknownFlag(true);
  }

  void InitGraph9(ComputeGraphPtr& graph) {
    //out_node has invalid node attr
    OpDescPtr quant1 = std::make_shared<OpDesc>("quant1", QUANT);
    OpDescPtr quant2 = std::make_shared<OpDesc>("quant2", QUANT);
    OpDescPtr split = std::make_shared<OpDesc>("split", SPLITD);
    (void)ge::AttrUtils::SetInt(split, SPLIT_DIM, 1);

    // add descriptor
    ge::GeShape shape1({1, 32, 14, 14});
    ge::GeShape shape2({1, 32, 14, 14});
    ge::GeShape shape3({1, 64, 14, 14});
    GeTensorDesc out_desc1(shape1, ge::FORMAT_NCHW, ge::DT_INT32);
    GeTensorDesc out_desc2(shape2, ge::FORMAT_NCHW, ge::DT_INT32);
    GeTensorDesc out_desc5(shape3, ge::FORMAT_NCHW, ge::DT_INT32);

    out_desc1.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc1.SetOriginDataType(ge::DT_FLOAT);
    out_desc1.SetOriginShape(shape1);
    out_desc2.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc2.SetOriginDataType(ge::DT_FLOAT);
    out_desc2.SetOriginShape(shape2);

    out_desc5.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc5.SetOriginDataType(ge::DT_FLOAT16);
    out_desc5.SetOriginShape(shape3);

    quant1->AddOutputDesc(out_desc1);
    quant2->AddOutputDesc(out_desc2);
    split->AddInputDesc(out_desc1);

    GeTensorDesc out_desc3(shape1, ge::FORMAT_NCHW, ge::DT_FLOAT16);
    out_desc3.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc3.SetOriginDataType(ge::DT_FLOAT16);
    out_desc3.SetOriginShape(shape1);

    GeTensorDesc out_desc4(shape2, ge::FORMAT_NCHW, ge::DT_FLOAT16);
    out_desc4.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc4.SetOriginDataType(ge::DT_FLOAT16);
    out_desc4.SetOriginShape(shape2);

    quant1->AddInputDesc(out_desc3);
    quant2->AddInputDesc(out_desc4);

    split->AddOutputDesc(out_desc3);
    split->AddOutputDesc(out_desc4);


    // create nodes
    NodePtr quant1_node = graph->AddNode(quant1);
    NodePtr quant2_node = graph->AddNode(quant2);
    NodePtr split_node = graph->AddNode(split);
    /*
     *        split
     *       /     \
     *     /         \
     * AcscendQuant   AcscendQuant (split_dim=1)
     *          
     */
    ge::GraphUtils::AddEdge(split_node->GetOutDataAnchor(0),
                            quant1_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(split_node->GetOutDataAnchor(1),
                            quant2_node->GetInDataAnchor(0));
  }

  void InitGraph10(ComputeGraphPtr& graph) {
    //valid input
    OpDescPtr quant1 = std::make_shared<OpDesc>("quant1", QUANT);
    OpDescPtr quant2 = std::make_shared<OpDesc>("quant2", QUANT);
    OpDescPtr split = std::make_shared<OpDesc>("split", SPLITD);
    OpDescPtr data = std::make_shared<OpDesc>("data", DATA);

    (void)ge::AttrUtils::SetInt(split, SPLIT_DIM, 1);
    (void)ge::AttrUtils::SetInt(quant1, ge::ATTR_NAME_IMPLY_TYPE, static_cast<int>(domi::ImplyType::TVM));
    (void)ge::AttrUtils::SetInt(quant2, ge::ATTR_NAME_IMPLY_TYPE, static_cast<int>(domi::ImplyType::TVM));

    // add descriptor
    ge::GeShape shape1({1, 32, 14, 14});
    ge::GeShape shape2({1, 32, 14, 14});
    ge::GeShape shape3({1, 64, 14, 14});
    GeTensorDesc out_desc1(shape1, ge::FORMAT_NCHW, ge::DT_INT32);
    GeTensorDesc out_desc2(shape2, ge::FORMAT_NCHW, ge::DT_INT32);
    GeTensorDesc out_desc5(shape3, ge::FORMAT_NCHW, ge::DT_INT32);

    out_desc1.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc1.SetOriginDataType(ge::DT_FLOAT);
    out_desc1.SetOriginShape(shape1);
    out_desc2.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc2.SetOriginDataType(ge::DT_FLOAT);
    out_desc2.SetOriginShape(shape2);

    out_desc5.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc5.SetOriginDataType(ge::DT_FLOAT16);
    out_desc5.SetOriginShape(shape3);

    quant1->AddOutputDesc(out_desc1);
    quant2->AddOutputDesc(out_desc2);
    split->AddInputDesc(out_desc1);
    data->AddOutputDesc(out_desc1);

    GeTensorDesc out_desc3(shape1, ge::FORMAT_NCHW, ge::DT_FLOAT16);
    out_desc3.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc3.SetOriginDataType(ge::DT_FLOAT16);
    out_desc3.SetOriginShape(shape1);

    GeTensorDesc out_desc4(shape2, ge::FORMAT_NCHW, ge::DT_FLOAT16);
    out_desc4.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc4.SetOriginDataType(ge::DT_FLOAT16);
    out_desc4.SetOriginShape(shape2);

    quant1->AddInputDesc(out_desc3);
    quant2->AddInputDesc(out_desc4);

    split->AddOutputDesc(out_desc3);
    split->AddOutputDesc(out_desc4);


    // create nodes
    NodePtr quant1_node = graph->AddNode(quant1);
    NodePtr quant2_node = graph->AddNode(quant2);
    NodePtr split_node = graph->AddNode(split);
    NodePtr data_node = graph->AddNode(data);
    /*          
     *
     *        data
     *          |
     *        split
     *       /     \
     *     /         \
     * AcscendQuant   AcscendQuant (split_dim=1)
     *          
     */
    ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(0),
                            split_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(split_node->GetOutDataAnchor(0),
                            quant1_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(split_node->GetOutDataAnchor(1),
                            quant2_node->GetInDataAnchor(0));
  }

  void InitGraph11(ComputeGraphPtr& graph) {
    //format连续 ori_shape size < 2
    OpDescPtr quant1 = std::make_shared<OpDesc>("quant1", QUANT);
    OpDescPtr quant2 = std::make_shared<OpDesc>("quant2", QUANT);
    OpDescPtr split = std::make_shared<OpDesc>("split", SPLITD);
    (void)ge::AttrUtils::SetInt(split, SPLIT_DIM, 1);
    (void)ge::AttrUtils::SetInt(quant1, ge::ATTR_NAME_IMPLY_TYPE, static_cast<int>(domi::ImplyType::TVM));
    (void)ge::AttrUtils::SetInt(quant2, ge::ATTR_NAME_IMPLY_TYPE, static_cast<int>(domi::ImplyType::TVM));

    // add descriptor
    ge::GeShape shape1({1, 32, 14, 14});
    ge::GeShape shape2({1, 32, 14, 14});
    ge::GeShape shape3({1});
    GeTensorDesc out_desc1(shape1, ge::FORMAT_NCHW, ge::DT_INT32);
    GeTensorDesc out_desc2(shape2, ge::FORMAT_NCHW, ge::DT_INT32);
    GeTensorDesc out_desc5(shape3, ge::FORMAT_NCHW, ge::DT_INT32);

    out_desc1.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc1.SetOriginDataType(ge::DT_FLOAT);
    out_desc1.SetOriginShape(shape1);
    out_desc2.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc2.SetOriginDataType(ge::DT_FLOAT);
    out_desc2.SetOriginShape(shape2);

    out_desc5.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc5.SetOriginDataType(ge::DT_FLOAT16);
    out_desc5.SetOriginShape(shape3);

    quant1->AddOutputDesc(out_desc1);
    quant2->AddOutputDesc(out_desc2);
    split->AddInputDesc(out_desc1);

    GeTensorDesc out_desc3(shape1, ge::FORMAT_NCHW, ge::DT_FLOAT16);
    out_desc3.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc3.SetOriginDataType(ge::DT_FLOAT16);
    out_desc3.SetOriginShape(shape1);

    GeTensorDesc out_desc4(shape2, ge::FORMAT_NCHW, ge::DT_FLOAT16);
    out_desc4.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc4.SetOriginDataType(ge::DT_FLOAT16);
    out_desc4.SetOriginShape(shape2);

    quant1->AddInputDesc(out_desc3);
    quant2->AddInputDesc(out_desc4);

    split->AddOutputDesc(out_desc5);
    split->AddOutputDesc(out_desc5);


    // create nodes
    NodePtr quant1_node = graph->AddNode(quant1);
    NodePtr quant2_node = graph->AddNode(quant2);
    NodePtr split_node = graph->AddNode(split);
    /*
     *        split
     *       /     \
     *     /         \
     * AcscendQuant   AcscendQuant (split_dim=1)
     *          
     */
    ge::GraphUtils::AddEdge(split_node->GetOutDataAnchor(0),
                            quant1_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(split_node->GetOutDataAnchor(1),
                            quant2_node->GetInDataAnchor(0));
  }

  void InitGraph12(ComputeGraphPtr& graph) {
    //split node has been set no_task attr
    //format连续
    OpDescPtr quant1 = std::make_shared<OpDesc>("quant1", QUANT);
    OpDescPtr quant2 = std::make_shared<OpDesc>("quant2", QUANT);
    OpDescPtr split = std::make_shared<OpDesc>("split", SPLITD);
    (void)ge::AttrUtils::SetInt(split, SPLIT_DIM, 1);
    (void)ge::AttrUtils::SetInt(quant1, ge::ATTR_NAME_IMPLY_TYPE, static_cast<int>(domi::ImplyType::TVM));
    (void)ge::AttrUtils::SetInt(quant2, ge::ATTR_NAME_IMPLY_TYPE, static_cast<int>(domi::ImplyType::TVM));
    (void)ge::AttrUtils::SetBool(split, ge::ATTR_NAME_NOTASK, true);

    // add descriptor
    ge::GeShape shape1({1, 32, 14, 14});
    ge::GeShape shape2({1, 32, 14, 14});
    ge::GeShape shape3({1, 64, 14, 14});
    GeTensorDesc out_desc1(shape1, ge::FORMAT_NCHW, ge::DT_INT32);
    GeTensorDesc out_desc2(shape2, ge::FORMAT_NCHW, ge::DT_INT32);
    GeTensorDesc out_desc5(shape3, ge::FORMAT_NCHW, ge::DT_INT32);

    out_desc1.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc1.SetOriginDataType(ge::DT_FLOAT);
    out_desc1.SetOriginShape(shape1);
    out_desc2.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc2.SetOriginDataType(ge::DT_FLOAT);
    out_desc2.SetOriginShape(shape2);

    out_desc5.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc5.SetOriginDataType(ge::DT_FLOAT16);
    out_desc5.SetOriginShape(shape3);

    quant1->AddOutputDesc(out_desc1);
    quant2->AddOutputDesc(out_desc2);
    split->AddInputDesc(out_desc1);

    GeTensorDesc out_desc3(shape1, ge::FORMAT_NCHW, ge::DT_FLOAT16);
    out_desc3.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc3.SetOriginDataType(ge::DT_FLOAT16);
    out_desc3.SetOriginShape(shape1);

    GeTensorDesc out_desc4(shape2, ge::FORMAT_NCHW, ge::DT_FLOAT16);
    out_desc4.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc4.SetOriginDataType(ge::DT_FLOAT16);
    out_desc4.SetOriginShape(shape2);

    quant1->AddInputDesc(out_desc3);
    quant2->AddInputDesc(out_desc4);

    split->AddOutputDesc(out_desc3);
    split->AddOutputDesc(out_desc4);


    // create nodes
    NodePtr quant1_node = graph->AddNode(quant1);
    NodePtr quant2_node = graph->AddNode(quant2);
    NodePtr split_node = graph->AddNode(split);
    /*
     *        split
     *       /     \
     *     /         \
     * AcscendQuant   AcscendQuant (split_dim=1)
     *          
     */
    ge::GraphUtils::AddEdge(split_node->GetOutDataAnchor(0),
                            quant1_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(split_node->GetOutDataAnchor(1),
                            quant2_node->GetInDataAnchor(0));
  }

void InitGraph13(ComputeGraphPtr& graph) {
    //NCHW->5HD
    OpDescPtr quant1 = std::make_shared<OpDesc>("quant1", QUANT);
    OpDescPtr quant2 = std::make_shared<OpDesc>("quant2", QUANT);
    OpDescPtr split = std::make_shared<OpDesc>("split", SPLITD);
    (void)ge::AttrUtils::SetInt(split, SPLIT_DIM, 1);
    (void)ge::AttrUtils::SetInt(quant1, ge::ATTR_NAME_IMPLY_TYPE, static_cast<int>(domi::ImplyType::TVM));
    (void)ge::AttrUtils::SetInt(quant2, ge::ATTR_NAME_IMPLY_TYPE, static_cast<int>(domi::ImplyType::TVM));
    (void)ge::AttrUtils::SetBool(quant1, ge::ATTR_NAME_NOTASK, true);

    // add descriptor
    ge::GeShape shape1({1, 32, 14, 14});
    ge::GeShape shape2({1, 32, 14, 14});
    ge::GeShape shape3({1, 80, 14, 14});
    GeTensorDesc out_desc1(shape1, ge::FORMAT_NCHW, ge::DT_INT32);
    GeTensorDesc out_desc2(shape2, ge::FORMAT_NCHW, ge::DT_INT32);
    GeTensorDesc out_desc5(shape3, ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);

    out_desc1.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc1.SetOriginDataType(ge::DT_FLOAT);
    out_desc1.SetOriginShape(shape1);
    out_desc2.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc2.SetOriginDataType(ge::DT_FLOAT);
    out_desc2.SetOriginShape(shape2);

    out_desc5.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc5.SetOriginDataType(ge::DT_FLOAT16);
    out_desc5.SetOriginShape(shape3);

    quant1->AddOutputDesc(out_desc1);
    quant2->AddOutputDesc(out_desc2);
    split->AddInputDesc(out_desc5);

    GeTensorDesc out_desc3(shape1, ge::FORMAT_NCHW, ge::DT_FLOAT16);
    out_desc3.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc3.SetOriginDataType(ge::DT_FLOAT16);
    out_desc3.SetOriginShape(shape1);

    GeTensorDesc out_desc4(shape2, ge::FORMAT_NCHW, ge::DT_FLOAT16);
    out_desc4.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc4.SetOriginDataType(ge::DT_FLOAT16);
    out_desc4.SetOriginShape(shape2);

    quant1->AddInputDesc(out_desc3);
    quant2->AddInputDesc(out_desc4);

    split->AddOutputDesc(out_desc5);
    split->AddOutputDesc(out_desc5);


    // create nodes
    NodePtr quant1_node = graph->AddNode(quant1);
    NodePtr quant2_node = graph->AddNode(quant2);
    NodePtr split_node = graph->AddNode(split);
    /*
     *        split
     *       /     \
     *     /         \
     * AcscendQuant   AcscendQuant (split_dim=1)
     *          
     */
    ge::GraphUtils::AddEdge(split_node->GetOutDataAnchor(0),
                            quant1_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(split_node->GetOutDataAnchor(1),
                            quant2_node->GetInDataAnchor(0));
  }
};

TEST_F(STEST_split_c_to_n_optimize, split_c_to_n_no_task_suc_001) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>(GRAPH_NAME);
  InitGraph1(graph);
  SplitCToNOptimizer split_c_to_optimize;
  split_c_to_optimize.SetFusionVirtualOp(*graph);

  auto split_node = graph->FindNode("split");

  bool attr_notask = false;
  bool continues_output = false;
  bool output_reuse_input = false;
  uint32_t dim_index = 1;
  (void)ge::AttrUtils::GetBool(split_node->GetOpDesc(), ATTR_NAME_NOTASK, attr_notask);
  (void)ge::AttrUtils::GetBool(split_node->GetOpDesc(), ge::ATTR_NAME_NOPADDING_CONTINUOUS_OUTPUT, continues_output);
  (void)ge::AttrUtils::GetBool(split_node->GetOpDesc(), ge::ATTR_NAME_OUTPUT_REUSE_INPUT, output_reuse_input);
  (void)ge::AttrUtils::GetInt(split_node->GetOpDesc(), ge::ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX, dim_index);

  EXPECT_EQ(attr_notask, true);
  EXPECT_EQ(continues_output, true);
  EXPECT_EQ(output_reuse_input, true);
  EXPECT_EQ(dim_index, 0);
}

TEST_F(STEST_split_c_to_n_optimize, split_c_to_n_no_task_suc_002) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>(GRAPH_NAME);
  InitGraph2(graph);
  SplitCToNOptimizer split_c_to_optimize;
  split_c_to_optimize.SetFusionVirtualOp(*graph);
  auto split_node = graph->FindNode("split");

  bool attr_notask = false;
  bool continues_output = false;
  bool output_reuse_input = false;
  uint32_t dim_index = 1;
  (void)ge::AttrUtils::GetBool(split_node->GetOpDesc(), ATTR_NAME_NOTASK, attr_notask);
  (void)ge::AttrUtils::GetBool(split_node->GetOpDesc(), ge::ATTR_NAME_NOPADDING_CONTINUOUS_OUTPUT, continues_output);
  (void)ge::AttrUtils::GetBool(split_node->GetOpDesc(), ge::ATTR_NAME_OUTPUT_REUSE_INPUT, output_reuse_input);
  (void)ge::AttrUtils::GetInt(split_node->GetOpDesc(), ge::ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX, dim_index);

  EXPECT_EQ(attr_notask, true);
  EXPECT_EQ(continues_output, true);
  EXPECT_EQ(output_reuse_input, true);
  EXPECT_EQ(dim_index, 0);
}

TEST_F(STEST_split_c_to_n_optimize, split_c_to_n_no_task_suc_003) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>(GRAPH_NAME);
  InitGraph4(graph);
  SplitCToNOptimizer split_c_to_optimize;
  split_c_to_optimize.SetFusionVirtualOp(*graph);
  auto split_node = graph->FindNode("split");

  bool attr_notask = false;
  bool continues_output = false;
  bool output_reuse_input = false;
  uint32_t dim_index = 1;
  (void)ge::AttrUtils::GetBool(split_node->GetOpDesc(), ATTR_NAME_NOTASK, attr_notask);
  (void)ge::AttrUtils::GetBool(split_node->GetOpDesc(), ge::ATTR_NAME_NOPADDING_CONTINUOUS_OUTPUT, continues_output);
  (void)ge::AttrUtils::GetBool(split_node->GetOpDesc(), ge::ATTR_NAME_OUTPUT_REUSE_INPUT, output_reuse_input);
  (void)ge::AttrUtils::GetInt(split_node->GetOpDesc(), ge::ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX, dim_index);

  EXPECT_EQ(attr_notask, true);
  EXPECT_EQ(continues_output, true);
  EXPECT_EQ(output_reuse_input, true);
  EXPECT_EQ(dim_index, 0);
}

TEST_F(STEST_split_c_to_n_optimize, split_c_to_n_no_task_suc_004) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>(GRAPH_NAME);
  InitGraph12(graph);
  SplitCToNOptimizer split_c_to_optimize;
  split_c_to_optimize.SetFusionVirtualOp(*graph);
  auto split_node = graph->FindNode("split");

  bool attr_notask = false;
  (void)ge::AttrUtils::GetBool(split_node->GetOpDesc(), ATTR_NAME_NOTASK, attr_notask);
  EXPECT_EQ(attr_notask, true);
}

TEST_F(STEST_split_c_to_n_optimize, split_c_to_n_no_task_fail_001) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>(GRAPH_NAME);
  InitGraph3(graph);
  SplitCToNOptimizer split_c_to_optimize;
  split_c_to_optimize.SetFusionVirtualOp(*graph);
  auto split_node = graph->FindNode("split");

  bool attr_notask = false;
  (void)ge::AttrUtils::GetBool(split_node->GetOpDesc(), ATTR_NAME_NOTASK, attr_notask);
  EXPECT_EQ(attr_notask, false);
}

TEST_F(STEST_split_c_to_n_optimize, split_c_to_n_no_task_fail_002) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>(GRAPH_NAME);
  InitGraph5(graph);
  SplitCToNOptimizer split_c_to_optimize;
  split_c_to_optimize.SetFusionVirtualOp(*graph);
  auto split_node = graph->FindNode("split");

  bool attr_notask = false;
  (void)ge::AttrUtils::GetBool(split_node->GetOpDesc(), ATTR_NAME_NOTASK, attr_notask);
  EXPECT_EQ(attr_notask, false);
}

TEST_F(STEST_split_c_to_n_optimize, split_c_to_n_no_task_fail_003) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>(GRAPH_NAME);
  InitGraph6(graph);
  SplitCToNOptimizer split_c_to_optimize;
  split_c_to_optimize.SetFusionVirtualOp(*graph);
  auto split_node = graph->FindNode("split");

  bool attr_notask = false;
  (void)ge::AttrUtils::GetBool(split_node->GetOpDesc(), ATTR_NAME_NOTASK, attr_notask);
  EXPECT_EQ(attr_notask, false);
}

TEST_F(STEST_split_c_to_n_optimize, split_c_to_n_no_task_fail_004) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>(GRAPH_NAME);
  InitGraph7(graph);
  SplitCToNOptimizer split_c_to_optimize;
  split_c_to_optimize.SetFusionVirtualOp(*graph);
  auto split_node = graph->FindNode("split");

  bool attr_notask = false;
  (void)ge::AttrUtils::GetBool(split_node->GetOpDesc(), ATTR_NAME_NOTASK, attr_notask);
  EXPECT_EQ(attr_notask, false);
}

TEST_F(STEST_split_c_to_n_optimize, split_c_to_n_no_task_fail_005) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>(GRAPH_NAME);
  InitGraph8(graph);
  SplitCToNOptimizer split_c_to_optimize;
  split_c_to_optimize.SetFusionVirtualOp(*graph);
  auto split_node = graph->FindNode("split");

  bool attr_notask = false;
  (void)ge::AttrUtils::GetBool(split_node->GetOpDesc(), ATTR_NAME_NOTASK, attr_notask);
  EXPECT_EQ(attr_notask, false);
}

TEST_F(STEST_split_c_to_n_optimize, split_c_to_n_no_task_fail_006) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>(GRAPH_NAME);
  InitGraph9(graph);
  SplitCToNOptimizer split_c_to_optimize;
  split_c_to_optimize.SetFusionVirtualOp(*graph);
  auto split_node = graph->FindNode("split");

  bool attr_notask = false;
  (void)ge::AttrUtils::GetBool(split_node->GetOpDesc(), ATTR_NAME_NOTASK, attr_notask);
  EXPECT_EQ(attr_notask, false);
}

TEST_F(STEST_split_c_to_n_optimize, split_c_to_n_no_task_fail_007) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>(GRAPH_NAME);
  InitGraph10(graph);
  SplitCToNOptimizer split_c_to_optimize;
  split_c_to_optimize.SetFusionVirtualOp(*graph);
  auto split_node = graph->FindNode("split");

  bool attr_notask = false;
  (void)ge::AttrUtils::GetBool(split_node->GetOpDesc(), ATTR_NAME_NOTASK, attr_notask);
  EXPECT_EQ(attr_notask, true);
}

TEST_F(STEST_split_c_to_n_optimize, split_c_to_n_no_task_fail_008) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>(GRAPH_NAME);
  InitGraph11(graph);
  SplitCToNOptimizer split_c_to_optimize;
  split_c_to_optimize.SetFusionVirtualOp(*graph);
  auto split_node = graph->FindNode("split");

  bool attr_notask = false;
  (void)ge::AttrUtils::GetBool(split_node->GetOpDesc(), ATTR_NAME_NOTASK, attr_notask);
  EXPECT_EQ(attr_notask, false);
}

TEST_F(STEST_split_c_to_n_optimize, split_c_to_n_no_task_fail_009) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>(GRAPH_NAME);
  InitGraph13(graph);
  SplitCToNOptimizer split_c_to_optimize;
  split_c_to_optimize.SetFusionVirtualOp(*graph);
  auto split_node = graph->FindNode("split");

  bool attr_notask = false;
  (void)ge::AttrUtils::GetBool(split_node->GetOpDesc(), ATTR_NAME_NOTASK, attr_notask);
  EXPECT_EQ(attr_notask, false);
}
}
