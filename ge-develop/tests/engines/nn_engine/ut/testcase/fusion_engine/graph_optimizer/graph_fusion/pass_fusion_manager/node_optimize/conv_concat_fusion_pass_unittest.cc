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
#include "graph_optimizer/graph_fusion/fusion_pass_manager/builtin_pass/node_optimize/conv_concat_fusion_pass.h"
#include "common/platform_utils.h"
#include "common/configuration.h"
#include "common/fe_utils.h"
#include "pass_manager.h"
#include "ops_store/ops_kernel_manager.h"
#include "../../../../../../../graph_constructor/graph_constructor.h"
#include "graph/ge_local_context.h"
#undef protected
#undef private

using namespace std;
using namespace ge;
using namespace fe;

namespace fe {

class UTEST_conv_concat_fusion_pass : public testing::Test {
 public:
  const string GRAPH_NAME = "test";
  const string CONCATD = "ConcatD";
  const string DEQUANT = "AscendDequant";
  const string REQUANT = "AscendRequant";
  const string QUANT = "AscendQuant";
  const string CONV2D = "Conv2D";
  const string MAXPOOL = "MaxPool";
  const string POOLING = "Pooling";
  const string RELU = "Relu";
  const string LEAKYRELU = "LeakyRelu";
  const string STRIDEDWRITE = "StridedWrite";
  const string STRIDEDREAD = "StridedRead";
  const string STRIDE_ATTR_STRIDE = "stride";
  const string STRIDE_ATTR_AXIS = "axis";
  const int NCHW_DIM_N = 1;
  const int NCHW_DIM_H = 14;
  const int NCHW_DIM_W = 12;

protected:
  static void SetUpTestCase() {
    PlatformUtils::Instance().soc_version_ = "Ascend910A";
  }
  void SetUp() {
    FEOpsStoreInfo feOpsStoreInfo;
    OpKernelInfoPtr opKernelInfoPtr1 = std::make_shared<OpKernelInfo>(STRIDEDWRITE);
    OpKernelInfoPtr opKernelInfoPtr2 = std::make_shared<OpKernelInfo>(STRIDEDREAD);
    SubOpInfoStorePtr subOpInfoStorePtr = std::make_shared<SubOpInfoStore>(feOpsStoreInfo);
    subOpInfoStorePtr->op_kernel_info_map_.emplace(std::make_pair(STRIDEDWRITE, opKernelInfoPtr1));
    subOpInfoStorePtr->op_kernel_info_map_.emplace(std::make_pair(STRIDEDREAD, opKernelInfoPtr2));
    OpsKernelManager::Instance(AI_CORE_NAME).sub_ops_store_map_[EN_IMPL_HW_TBE] = subOpInfoStorePtr;
  }
  void TearDown() {}
  void InitNoQuantGraph(ComputeGraphPtr& graph, string relu_str, bool dyn_input) {
    OpDescPtr conv1 = std::make_shared<OpDesc>("conv1", CONV2D);
    OpDescPtr conv2 = std::make_shared<OpDesc>("conv2", CONV2D);
    OpDescPtr concat = std::make_shared<OpDesc>("concat", CONCATD);
    OpDescPtr netoutput = std::make_shared<OpDesc>("netoutput", "NetOutput");
    (void)ge::AttrUtils::SetInt(concat, CONCAT_DIM, 1);

    // add descriptor
    ge::GeShape shape1({NCHW_DIM_N, 16, NCHW_DIM_H, NCHW_DIM_W});
    ge::GeShape shape2({NCHW_DIM_N, 32, NCHW_DIM_H, NCHW_DIM_W});
    GeTensorDesc out_desc1(shape1, ge::FORMAT_NCHW, ge::DT_FLOAT);
    out_desc1.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc1.SetOriginDataType(ge::DT_FLOAT);
    out_desc1.SetOriginShape(shape1);
    GeTensorDesc out_desc2(shape2, ge::FORMAT_NCHW, ge::DT_FLOAT);
    out_desc2.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc2.SetOriginDataType(ge::DT_FLOAT);
    out_desc2.SetOriginShape(shape2);
    if (dyn_input) {
      ge::GeShape shape3({NCHW_DIM_N, 16, NCHW_DIM_H, -1});
      GeTensorDesc out_desc3(shape3, ge::FORMAT_NCHW, ge::DT_FLOAT);
      out_desc3.SetOriginFormat(ge::FORMAT_NCHW);
      out_desc3.SetOriginDataType(ge::DT_FLOAT);
      out_desc3.SetOriginShape(shape3);
      conv1->AddInputDesc(out_desc3);
    }
    conv1->AddOutputDesc(out_desc1);
    conv2->AddOutputDesc(out_desc2);
    concat->AddInputDesc(out_desc1);
    concat->AddInputDesc(out_desc2);
    ge::GeShape shape({1, 48, 14, 14});
    GeTensorDesc out_desc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT);
    out_desc.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc.SetOriginDataType(ge::DT_FLOAT);
    out_desc.SetOriginShape(shape);
    concat->AddOutputDesc(out_desc);
    netoutput->AddInputDesc(out_desc);

    // create nodes
    NodePtr conv1_node = graph->AddNode(conv1);
    NodePtr conv2_node = graph->AddNode(conv2);
    NodePtr concat_node = graph->AddNode(concat);
    NodePtr netoutput_node = graph->AddNode(netoutput);

    /*
     *  Conv2d             Conv2d
     *      \               /
     *      (Relu)      (Relu)
     *           \      /
     *            Concat
     *              |
     *          NetOutput
     */
    if (!relu_str.empty()) {
      OpDescPtr relu1 = std::make_shared<OpDesc>("relu1", relu_str);
      OpDescPtr relu2 = std::make_shared<OpDesc>("relu2", relu_str);
      relu1->AddInputDesc(out_desc1);
      relu1->AddOutputDesc(out_desc1);
      relu2->AddInputDesc(out_desc2);
      relu2->AddOutputDesc(out_desc2);
      NodePtr relu_node1 = graph->AddNode(relu1);
      NodePtr relu_node2 = graph->AddNode(relu2);
      ge::GraphUtils::AddEdge(conv1_node->GetOutDataAnchor(0),
                              relu_node1->GetInDataAnchor(0));
      ge::GraphUtils::AddEdge(conv2_node->GetOutDataAnchor(0),
                              relu_node2->GetInDataAnchor(0));

      ge::GraphUtils::AddEdge(relu_node1->GetOutDataAnchor(0),
                              concat_node->GetInDataAnchor(0));
      ge::GraphUtils::AddEdge(relu_node2->GetOutDataAnchor(0),
                              concat_node->GetInDataAnchor(1));
      ge::GraphUtils::AddEdge(concat_node->GetOutDataAnchor(0),
                              netoutput_node->GetInDataAnchor(0));
    } else {
      ge::GraphUtils::AddEdge(conv1_node->GetOutDataAnchor(0),
                              concat_node->GetInDataAnchor(0));
      ge::GraphUtils::AddEdge(conv2_node->GetOutDataAnchor(0),
                              concat_node->GetInDataAnchor(1));
      ge::GraphUtils::AddEdge(concat_node->GetOutDataAnchor(0),
                              netoutput_node->GetInDataAnchor(0));
    }
  }

  /*
   *       MaxPool   Conv2d
   *           \      /
   *            Concat
   *              |
   */
  void InitMaxPoolNoQuantGraph(ComputeGraphPtr& graph) {
    OpDescPtr conv1 = std::make_shared<OpDesc>("conv1", MAXPOOL);
    OpDescPtr conv2 = std::make_shared<OpDesc>("conv2", CONV2D);
    OpDescPtr concat = std::make_shared<OpDesc>("concat", CONCATD);
    (void)ge::AttrUtils::SetInt(concat, CONCAT_DIM, 1);

    // add descriptor
    ge::GeShape shape1({NCHW_DIM_N, 16, NCHW_DIM_H, NCHW_DIM_W});
    ge::GeShape shape2({NCHW_DIM_N, 32, NCHW_DIM_H, NCHW_DIM_W});
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
    ge::GeShape shape({1, 48, 14, 14});
    GeTensorDesc out_desc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT);
    out_desc.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc.SetOriginDataType(ge::DT_FLOAT);
    out_desc.SetOriginShape(shape);
    concat->AddOutputDesc(out_desc);

    // create nodes
    NodePtr conv1_node = graph->AddNode(conv1);
    NodePtr conv2_node = graph->AddNode(conv2);
    NodePtr concat_node = graph->AddNode(concat);

      ge::GraphUtils::AddEdge(conv1_node->GetOutDataAnchor(0),
                              concat_node->GetInDataAnchor(0));
      ge::GraphUtils::AddEdge(conv2_node->GetOutDataAnchor(0),
                              concat_node->GetInDataAnchor(1));
  }
  /*
   *       MaxPool   Conv2d
   *           \      /
   *            Concat
   *              |
   */
  void InitPoolingNoQuantGraph(ComputeGraphPtr& graph) {
    OpDescPtr conv1 = std::make_shared<OpDesc>("conv1", POOLING);
    OpDescPtr conv2 = std::make_shared<OpDesc>("conv2", CONV2D);
    OpDescPtr concat = std::make_shared<OpDesc>("concat", CONCATD);
    (void)ge::AttrUtils::SetInt(concat, CONCAT_DIM, 1);

    // add descriptor
    ge::GeShape shape1({NCHW_DIM_N, 16, NCHW_DIM_H, NCHW_DIM_W});
    ge::GeShape shape2({NCHW_DIM_N, 32, NCHW_DIM_H, NCHW_DIM_W});
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
    ge::GeShape shape({1, 48, 14, 14});
    GeTensorDesc out_desc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT);
    out_desc.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc.SetOriginDataType(ge::DT_FLOAT);
    out_desc.SetOriginShape(shape);
    concat->AddOutputDesc(out_desc);

    // create nodes
    NodePtr conv1_node = graph->AddNode(conv1);
    NodePtr conv2_node = graph->AddNode(conv2);
    NodePtr concat_node = graph->AddNode(concat);

      ge::GraphUtils::AddEdge(conv1_node->GetOutDataAnchor(0),
                              concat_node->GetInDataAnchor(0));
      ge::GraphUtils::AddEdge(conv2_node->GetOutDataAnchor(0),
                              concat_node->GetInDataAnchor(1));
  }

  void InitQuantGraph(ComputeGraphPtr& graph, string elemwise_str) {
    OpDescPtr conv1 = std::make_shared<OpDesc>("conv1", CONV2D);
    OpDescPtr conv2 = std::make_shared<OpDesc>("conv2", CONV2D);
    OpDescPtr dequant1 = std::make_shared<OpDesc>("dequant1", DEQUANT);
    OpDescPtr dequant2 = std::make_shared<OpDesc>("dequant2", DEQUANT);
    OpDescPtr concat = std::make_shared<OpDesc>("concat", CONCATD);
    OpDescPtr netoutput = std::make_shared<OpDesc>("netoutput", "NetOutput");
    (void)ge::AttrUtils::SetInt(concat, CONCAT_DIM, 1);

    // add descriptor
    ge::GeShape shape1({NCHW_DIM_N, 32, NCHW_DIM_H, NCHW_DIM_W});
    GeTensorDesc out_desc1(shape1, ge::FORMAT_NCHW, ge::DT_INT32);
    ge::GeShape shape2({NCHW_DIM_N, 64, NCHW_DIM_H, NCHW_DIM_W});
    GeTensorDesc out_desc2(shape2, ge::FORMAT_NCHW, ge::DT_INT32);
    conv1->AddOutputDesc(out_desc1);
    conv2->AddOutputDesc(out_desc2);
    dequant1->AddInputDesc(out_desc1);
    dequant2->AddInputDesc(out_desc2);

    GeTensorDesc out_desc3(shape1, ge::FORMAT_NCHW, ge::DT_INT8);
    out_desc3.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc3.SetOriginDataType(ge::DT_INT8);
    out_desc3.SetOriginShape(shape1);

    GeTensorDesc out_desc4(shape2, ge::FORMAT_NCHW, ge::DT_INT8);
    out_desc4.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc4.SetOriginDataType(ge::DT_INT8);
    out_desc4.SetOriginShape(shape2);
    dequant1->AddOutputDesc(out_desc3);
    dequant2->AddOutputDesc(out_desc4);

    concat->AddInputDesc(out_desc3);
    concat->AddInputDesc(out_desc4);
    ge::GeShape shape({NCHW_DIM_N, 96, NCHW_DIM_H, NCHW_DIM_W});
    GeTensorDesc out_desc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT);
    out_desc.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc.SetOriginDataType(ge::DT_FLOAT);
    out_desc.SetOriginShape(shape);
    concat->AddOutputDesc(out_desc);
    netoutput->AddInputDesc(out_desc);

    NodePtr conv1_node = graph->AddNode(conv1);
    NodePtr conv2_node = graph->AddNode(conv2);
    NodePtr dequant1_node = graph->AddNode(dequant1);
    NodePtr dequant2_node = graph->AddNode(dequant2);
    NodePtr concat_node = graph->AddNode(concat);
    NodePtr netoutput_node = graph->AddNode(netoutput);
    ge::GraphUtils::AddEdge(conv1_node->GetOutDataAnchor(0),
                            dequant1_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(conv2_node->GetOutDataAnchor(0),
                            dequant2_node->GetInDataAnchor(0));
    if (!elemwise_str.empty()) {
      OpDescPtr relu1 = std::make_shared<OpDesc>("relu1", elemwise_str);
      OpDescPtr relu2 = std::make_shared<OpDesc>("relu2", elemwise_str);
      relu1->AddInputDesc(out_desc1);
      relu1->AddOutputDesc(out_desc1);
      relu2->AddInputDesc(out_desc2);
      relu2->AddOutputDesc(out_desc2);
      NodePtr relu_node1 = graph->AddNode(relu1);
      NodePtr relu_node2 = graph->AddNode(relu2);
      ge::GraphUtils::AddEdge(dequant1_node->GetOutDataAnchor(0),
                              relu_node1->GetInDataAnchor(0));
      ge::GraphUtils::AddEdge(dequant2_node->GetOutDataAnchor(0),
                              relu_node2->GetInDataAnchor(0));

      ge::GraphUtils::AddEdge(relu_node1->GetOutDataAnchor(0),
                              concat_node->GetInDataAnchor(0));
      ge::GraphUtils::AddEdge(relu_node2->GetOutDataAnchor(0),
                              concat_node->GetInDataAnchor(1));
      ge::GraphUtils::AddEdge(concat_node->GetOutDataAnchor(0),
                              netoutput_node->GetInDataAnchor(0));
    } else {
      ge::GraphUtils::AddEdge(dequant1_node->GetOutDataAnchor(0),
                              concat_node->GetInDataAnchor(0));
      ge::GraphUtils::AddEdge(dequant2_node->GetOutDataAnchor(0),
                              concat_node->GetInDataAnchor(1));
      ge::GraphUtils::AddEdge(concat_node->GetOutDataAnchor(0),
                              netoutput_node->GetInDataAnchor(0));
    }
  }

  void InitRequantGraph(ComputeGraphPtr& graph) {
    OpDescPtr conv1 = std::make_shared<OpDesc>("conv1", CONV2D);
    OpDescPtr conv2 = std::make_shared<OpDesc>("conv2", CONV2D);
    OpDescPtr requant1 = std::make_shared<OpDesc>("requant1", REQUANT);
    OpDescPtr requant2 = std::make_shared<OpDesc>("requant2", REQUANT);
    OpDescPtr concat = std::make_shared<OpDesc>("concat", CONCATD);
    (void)ge::AttrUtils::SetInt(concat, CONCAT_DIM, 1);

    // add descriptor
    ge::GeShape shape1({NCHW_DIM_N, 32, NCHW_DIM_H, NCHW_DIM_W});
    GeTensorDesc out_desc1(shape1, ge::FORMAT_NCHW, ge::DT_INT32);
    ge::GeShape shape2({NCHW_DIM_N, 64, NCHW_DIM_H, NCHW_DIM_W});
    GeTensorDesc out_desc2(shape2, ge::FORMAT_NCHW, ge::DT_INT32);
    conv1->AddOutputDesc(out_desc1);
    conv2->AddOutputDesc(out_desc2);
    requant1->AddInputDesc(out_desc1);
    requant2->AddInputDesc(out_desc2);

    GeTensorDesc out_desc3(shape1, ge::FORMAT_NCHW, ge::DT_INT8);
    out_desc3.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc3.SetOriginDataType(ge::DT_INT8);
    out_desc3.SetOriginShape(shape1);

    GeTensorDesc out_desc4(shape2, ge::FORMAT_NCHW, ge::DT_INT8);
    out_desc4.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc4.SetOriginDataType(ge::DT_INT8);
    out_desc4.SetOriginShape(shape2);
    requant1->AddOutputDesc(out_desc3);
    requant2->AddOutputDesc(out_desc4);

    concat->AddInputDesc(out_desc3);
    concat->AddInputDesc(out_desc4);
    ge::GeShape shape({NCHW_DIM_N, 96, NCHW_DIM_H, NCHW_DIM_W});
    GeTensorDesc out_desc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT);
    out_desc.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc.SetOriginDataType(ge::DT_FLOAT);
    out_desc.SetOriginShape(shape);
    concat->AddOutputDesc(out_desc);

    NodePtr conv1_node = graph->AddNode(conv1);
    NodePtr conv2_node = graph->AddNode(conv2);
    NodePtr requant1_node = graph->AddNode(requant1);
    NodePtr requant2_node = graph->AddNode(requant2);
    NodePtr concat_node = graph->AddNode(concat);
    ge::GraphUtils::AddEdge(conv1_node->GetOutDataAnchor(0),
                              requant1_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(conv2_node->GetOutDataAnchor(0),
                              requant2_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(requant1_node->GetOutDataAnchor(0),
                              concat_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(requant2_node->GetOutDataAnchor(0),
                              concat_node->GetInDataAnchor(1));
  }


  void InitQuantAndConvAndReluAndLeakyreluGraph(ComputeGraphPtr& graph, vector<string> op_type_vec) {
    OpDescPtr conv1 = std::make_shared<OpDesc>("conv1", CONV2D);
    OpDescPtr dequant1 = std::make_shared<OpDesc>("dequant1", DEQUANT);
    OpDescPtr concat = std::make_shared<OpDesc>("concat", CONCATD);
    (void)ge::AttrUtils::SetInt(concat, CONCAT_DIM, 1);
    ge::GeShape shape({NCHW_DIM_N, 32, NCHW_DIM_H, NCHW_DIM_W});
    GeTensorDesc out_desc1(shape, ge::FORMAT_NCHW, ge::DT_INT32);
    out_desc1.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc1.SetOriginDataType(ge::DT_INT32);
    out_desc1.SetOriginShape(shape);
    conv1->AddOutputDesc(out_desc1);
    dequant1->AddInputDesc(out_desc1);

    GeTensorDesc out_desc2(shape, ge::FORMAT_NCHW, ge::DT_INT8);
    out_desc2.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc2.SetOriginDataType(ge::DT_INT8);
    out_desc2.SetOriginShape(shape);
    dequant1->AddOutputDesc(out_desc2);
    concat->AddInputDesc(out_desc2);

    NodePtr conv1_node = graph->AddNode(conv1);
    NodePtr dequant1_node = graph->AddNode(dequant1);

    GeTensorDesc out_desc3(shape, ge::FORMAT_NCHW, ge::DT_FLOAT);
    out_desc3.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc3.SetOriginDataType(ge::DT_FLOAT);
    out_desc3.SetOriginShape(shape);

    int op_type_vec_size = op_type_vec.size();
    for(int i = 0; i != op_type_vec_size; ++i) {
      concat->AddInputDesc(out_desc3);
    }

    ge::GeShape out_shape({NCHW_DIM_N, 32*(op_type_vec_size +1), NCHW_DIM_H, NCHW_DIM_W});
    GeTensorDesc out_desc(out_shape, ge::FORMAT_NCHW, ge::DT_FLOAT);
    out_desc.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc.SetOriginDataType(ge::DT_FLOAT);
    out_desc.SetOriginShape(out_shape);
    concat->AddOutputDesc(out_desc);

    NodePtr concat_node = graph->AddNode(concat);
    ge::GraphUtils::AddEdge(conv1_node->GetOutDataAnchor(0),
                            dequant1_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(dequant1_node->GetOutDataAnchor(0),
                            concat_node->GetInDataAnchor(0));

    for(int i = 0; i != op_type_vec_size; ++i){
      OpDescPtr conv_op = std::make_shared<OpDesc>("conv"+to_string(i), CONV2D);
      out_desc3.SetOriginFormat(ge::FORMAT_NCHW);
      out_desc3.SetOriginDataType(ge::DT_FLOAT16);
      out_desc3.SetOriginShape(shape);
      conv_op->AddOutputDesc(out_desc3);
      NodePtr conv_node = graph->AddNode(conv_op);
      if(op_type_vec[i] == CONV2D) {
        ge::GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0),
                                concat_node->GetInDataAnchor(i+1));
      } else {
        OpDescPtr cur_op =
            std::make_shared<OpDesc>(op_type_vec[i] + to_string(i), op_type_vec[i]);
        cur_op->AddInputDesc(out_desc3);
        cur_op->AddOutputDesc(out_desc3);
        if (op_type_vec[i] == LEAKYRELU) {
          (void)ge::AttrUtils::SetFloat(cur_op, "negative_slope",0.0);
        }
        NodePtr cur_node = graph->AddNode(cur_op);
        ge::GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0),
                                cur_node->GetInDataAnchor(0));
        ge::GraphUtils::AddEdge(cur_node->GetOutDataAnchor(0),
                                concat_node->GetInDataAnchor(i + 1));
      }
    }
  }

  void InitQuantGraph2(ComputeGraphPtr &graph, bool with_quant) {
    OpDescPtr relu = std::make_shared<OpDesc>("relu1", RELU);
    OpDescPtr split = std::make_shared<OpDesc>("split", SPLITD);
    OpDescPtr conv1 = std::make_shared<OpDesc>("conv1", CONV2D);
    OpDescPtr conv2 = std::make_shared<OpDesc>("conv2", CONV2D);
    OpDescPtr quant1 = std::make_shared<OpDesc>("quant1", QUANT);
    OpDescPtr quant2 = std::make_shared<OpDesc>("quant2", QUANT);
    OpDescPtr dequant1 = std::make_shared<OpDesc>("dequant1", DEQUANT);
    OpDescPtr dequant2 = std::make_shared<OpDesc>("dequant2", DEQUANT);
    OpDescPtr concat = std::make_shared<OpDesc>("concat", CONCATD);
    (void)ge::AttrUtils::SetInt(split, SPLIT_DIM, 1);
    (void)ge::AttrUtils::SetInt(concat, CONCAT_DIM, 1);

    // add descriptor
    ge::GeShape shape1({2, 32, 14, 14});
    GeTensorDesc out_desc1(shape1, ge::FORMAT_NCHW, ge::DT_INT32);
    ge::GeShape shape2({2, 32, 14, 14});
    GeTensorDesc out_desc2(shape2, ge::FORMAT_NCHW, ge::DT_INT32);
    ge::GeShape shape3({2, 64, 14, 14});
    GeTensorDesc out_desc5(shape3, ge::FORMAT_NCHW, ge::DT_INT32);
    out_desc5.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc5.SetOriginDataType(ge::DT_FLOAT16);
    out_desc5.SetOriginShape(shape3);
    split->AddInputDesc(out_desc5);
    quant1->AddOutputDesc(out_desc1);
    quant2->AddOutputDesc(out_desc2);
    conv1->AddInputDesc(out_desc1);
    conv2->AddInputDesc(out_desc2);
    conv1->AddOutputDesc(out_desc1);
    conv2->AddOutputDesc(out_desc2);
    dequant1->AddInputDesc(out_desc1);
    dequant2->AddInputDesc(out_desc2);

    GeTensorDesc out_desc3(shape1, ge::FORMAT_NCHW, ge::DT_FLOAT16);
    out_desc3.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc3.SetOriginDataType(ge::DT_FLOAT16);
    out_desc3.SetOriginShape(shape1);

    GeTensorDesc out_desc4(shape2, ge::FORMAT_NCHW, ge::DT_FLOAT16);
    out_desc4.SetOriginFormat(ge::FORMAT_NCHW);
    out_desc4.SetOriginDataType(ge::DT_FLOAT16);
    out_desc4.SetOriginShape(shape2);

    relu->AddOutputDesc(out_desc5);

    split->AddOutputDesc(out_desc3);
    split->AddOutputDesc(out_desc4);

    dequant1->AddOutputDesc(out_desc3);
    dequant2->AddOutputDesc(out_desc4);

    quant1->AddInputDesc(out_desc3);
    quant2->AddInputDesc(out_desc4);

    concat->AddInputDesc(out_desc3);
    concat->AddInputDesc(out_desc4);

    concat->AddOutputDesc(out_desc5);

    NodePtr relu_node = graph->AddNode(relu);
    NodePtr split_node = graph->AddNode(split);
    NodePtr quant1_node = graph->AddNode(quant1);
    NodePtr quant2_node = graph->AddNode(quant2);
    NodePtr conv1_node = graph->AddNode(conv1);
    NodePtr conv2_node = graph->AddNode(conv2);
    NodePtr dequant1_node = graph->AddNode(dequant1);
    NodePtr dequant2_node = graph->AddNode(dequant2);
    NodePtr concat_node = graph->AddNode(concat);

    /*             split
     *           /       \
     *         /           \
     *  AcscendQuant   AcscendQuant
     *     /                \
     *  Conv2d             Conv2d
     *      \               /
     * AcsendDequant AcsendDequant
     *         \          /
     *           \      /
     *            Concat
     *               |
     *          AcscendQuant
     */

    ge::GraphUtils::AddEdge(relu_node->GetOutDataAnchor(0),
                            split_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(split_node->GetOutDataAnchor(0),
                            quant1_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(split_node->GetOutDataAnchor(1),
                            quant2_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(quant1_node->GetOutDataAnchor(0),
                            conv1_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(quant2_node->GetOutDataAnchor(0),
                            conv2_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(conv1_node->GetOutDataAnchor(0),
                            dequant1_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(conv2_node->GetOutDataAnchor(0),
                            dequant2_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(dequant1_node->GetOutDataAnchor(0),
                            concat_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(dequant2_node->GetOutDataAnchor(0),
                            concat_node->GetInDataAnchor(1));

    if (with_quant) {
      OpDescPtr quant3 = std::make_shared<OpDesc>("quant3", QUANT);
      (void)ge::AttrUtils::SetFloat(quant3, "scale",-1);
      (void)ge::AttrUtils::SetFloat(quant3, "offset",-1);
      quant3->AddInputDesc(out_desc5);
      quant3->AddOutputDesc(out_desc5);
      NodePtr quant_node3 = graph->AddNode(quant3);
      ge::GraphUtils::AddEdge(concat_node->GetOutDataAnchor(0),
                              quant_node3->GetInDataAnchor(0));
      OpDescPtr relu = std::make_shared<OpDesc>("relu1", RELU);
      relu->AddInputDesc(out_desc5);
      NodePtr relu_node = graph->AddNode(relu);
      ge::GraphUtils::AddEdge(quant_node3->GetOutDataAnchor(0),
                              relu_node->GetInDataAnchor(0));

      OpDescPtr quant4 = std::make_shared<OpDesc>("quant4", QUANT);
      (void)ge::AttrUtils::SetFloat(quant4, "scale",-1);
      (void)ge::AttrUtils::SetFloat(quant4, "offset",-1);
      quant4->AddInputDesc(out_desc5);
      quant4->AddOutputDesc(out_desc5);
      NodePtr quant_node4 = graph->AddNode(quant4);
      ge::GraphUtils::AddEdge(concat_node->GetOutDataAnchor(0),
                              quant_node4->GetInDataAnchor(0));
      OpDescPtr relu2 = std::make_shared<OpDesc>("relu2", RELU);
      relu2->AddInputDesc(out_desc5);
      NodePtr relu_node2 = graph->AddNode(relu2);
      ge::GraphUtils::AddEdge(quant_node4->GetOutDataAnchor(0),
                              relu_node2->GetInDataAnchor(0));

    } else {
      OpDescPtr split2 = std::make_shared<OpDesc>("split2", SPLITD);
      split2->AddInputDesc(out_desc3);
      NodePtr split2_node = graph->AddNode(split2);
      ge::GraphUtils::AddEdge(concat_node->GetOutDataAnchor(0),
                              split2_node->GetInDataAnchor(0));
    }
  }

  Status CheckGraph(ComputeGraphPtr& graph, int expect_axis) {
    for (auto& node : graph->GetDirectNode()) {
      FE_CHECK_NOTNULL(node);
      if (node->GetType() == CONCATD) {
        OpDescPtr concat_op_desc_ptr = node->GetOpDesc();
        bool no_task = false;
        bool output_reuse_input = false;
        bool no_padding_continuous_input = false;
        bool no_padding_continuous_output = false;
        int dim_index = -1;
        (void)ge::AttrUtils::GetBool(concat_op_desc_ptr, ge::ATTR_NAME_NOTASK,
                                     no_task);
        (void)ge::AttrUtils::GetBool(concat_op_desc_ptr,
                                     ge::ATTR_NAME_OUTPUT_REUSE_INPUT,
                                     output_reuse_input);
        (void)ge::AttrUtils::GetBool(concat_op_desc_ptr,
                                     ge::ATTR_NAME_NOPADDING_CONTINUOUS_INPUT,
                                     no_padding_continuous_input);
        (void)ge::AttrUtils::GetBool(concat_op_desc_ptr,
                                     ge::ATTR_NAME_NOPADDING_CONTINUOUS_OUTPUT,
                                     no_padding_continuous_output);
        (void)ge::AttrUtils::GetInt(
            concat_op_desc_ptr, ge::ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX, dim_index);

        EXPECT_EQ(no_task, true);
        EXPECT_EQ(output_reuse_input, true);
        EXPECT_EQ(no_padding_continuous_input, true);
        EXPECT_EQ(no_padding_continuous_output, false);
        EXPECT_EQ(dim_index, 1);
        for (auto& input_anchor : node->GetAllInDataAnchors()) {
          EXPECT_NE(input_anchor, nullptr);
          EXPECT_NE(input_anchor->GetPeerOutAnchor(), nullptr);
          EXPECT_NE(input_anchor->GetPeerOutAnchor()->GetOwnerNode(), nullptr);
          // check stridedwride op
          auto peer_output_node = input_anchor->GetPeerOutAnchor()->GetOwnerNode();
          EXPECT_EQ(peer_output_node->GetType(), STRIDEDWRITE);
          OpDescPtr peer_output_op_desc = peer_output_node->GetOpDesc();
          int axis = 0;
          (void)ge::AttrUtils::GetInt(peer_output_op_desc, STRIDE_ATTR_AXIS, axis);
          EXPECT_EQ(axis, expect_axis);
        }
      }
    }
    return SUCCESS;
  }

  Status CheckGraph2(ComputeGraphPtr& graph) {
    for (auto &node : graph->GetDirectNode()) {
      FE_CHECK_NOTNULL(node);
      EXPECT_NE(node->GetType(), STRIDEDWRITE);
      if (node->GetType() == CONCATD) {
        OpDescPtr concat_op_desc_ptr = node->GetOpDesc();
        bool no_task = false;
        bool output_reuse_input = false;
        bool no_padding_continuous_input = false;
        bool no_padding_continuous_output = false;
        int dim_index = -1;
        (void)ge::AttrUtils::GetBool(concat_op_desc_ptr, ge::ATTR_NAME_NOTASK,
                                     no_task);
        (void)ge::AttrUtils::GetBool(concat_op_desc_ptr,
                                     ge::ATTR_NAME_OUTPUT_REUSE_INPUT,
                                     output_reuse_input);
        (void)ge::AttrUtils::GetBool(concat_op_desc_ptr,
                                     ge::ATTR_NAME_NOPADDING_CONTINUOUS_INPUT,
                                     no_padding_continuous_input);
        (void)ge::AttrUtils::GetBool(concat_op_desc_ptr,
                                     ge::ATTR_NAME_NOPADDING_CONTINUOUS_OUTPUT,
                                     no_padding_continuous_output);
        (void)ge::AttrUtils::GetInt(
            concat_op_desc_ptr, ge::ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX, dim_index);

        EXPECT_EQ(no_task, false);
        EXPECT_EQ(output_reuse_input, false);
        EXPECT_EQ(no_padding_continuous_input, false);
        EXPECT_EQ(no_padding_continuous_output, false);
      }
    }
    return SUCCESS;
  }

  void BuildGraph_1(ge::ComputeGraphPtr &graph, ge::DataType dtype = ge::DT_INT8) {
    ge::GeShape original_shape = ge::GeShape({NCHW_DIM_N, 32, NCHW_DIM_H, NCHW_DIM_W});
    GraphConstructor test(graph, "", ge::FORMAT_NCHW, dtype,
                          original_shape);
    test.AddOpDesc(EN_IMPL_HW_TBE, "Convolution", "conv0", CONV2D, 1, 1)
        .SetInputs({"Data_1"})
        .AddOpDesc(EN_IMPL_HW_TBE, "dequant", "dequant0", DEQUANT, 2, 1)
        .SetInputs({"conv0"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "LeakyRelu0", "LeakyRelu", 1, 1)
        .SetInputs({"dequant0"})
        .AddOpDesc(EN_IMPL_HW_TBE, "Convolution", "conv1", CONV2D, 1, 1)
        .SetInputs({"Data_2"})
        .AddOpDesc(EN_IMPL_HW_TBE, "dequant", "dequant1", DEQUANT, 2, 1)
        .SetInputs({"conv1"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "LeakyRelu1", "LeakyRelu", 1, 1)
        .SetInputs({"dequant1"})
        .AddOpDesc(EN_IMPL_HW_TBE, "concat", "concat", CONCATD, 2, 1)
        .SetInputs({"LeakyRelu0", "LeakyRelu1"})
        .SetInput("NetOutput", "concat");
    test.DumpGraph(graph);
    for (auto node : graph->GetDirectNode()) {
      if (node->GetType() == CONCATD) {
        (void)ge::AttrUtils::SetInt(node->GetOpDesc(), CONCAT_DIM, 1);
      }
    }

  }
};

// conv2d+concat
// con2d+concat
TEST_F(UTEST_conv_concat_fusion_pass, con2d_success) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>(GRAPH_NAME);
  InitNoQuantGraph(graph, "", false);
  ConvConcatFusionPass pass;
  vector<GraphPass*> passes = {&pass};
  Status status = PassManager::Run(*graph, passes);
  CheckGraph(graph, 1);
}

// dyn.conv2d+concat
// dyn.con2d+concat
TEST_F(UTEST_conv_concat_fusion_pass, dyn_con2d_no_pass) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>(GRAPH_NAME);
  InitNoQuantGraph(graph, "", true);
  ConvConcatFusionPass pass;
  vector<GraphPass*> passes = {&pass};
  Status status = PassManager::Run(*graph, passes);
}

// conv2d+concat
// conv2d+relu+concat
TEST_F(UTEST_conv_concat_fusion_pass, con2d_relu_success) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>(GRAPH_NAME);
  InitNoQuantGraph(graph, RELU, false);
  ConvConcatFusionPass pass;
  vector<GraphPass*> passes = {&pass};
  Status status = PassManager::Run(*graph, passes);
  CheckGraph(graph, 1);
}

// maxpool+concat
// conv2d+concat
TEST_F(UTEST_conv_concat_fusion_pass, maxpool_conv2d_success) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>(GRAPH_NAME);
  InitMaxPoolNoQuantGraph(graph);
  ConvConcatFusionPass pass;
  vector<GraphPass*> passes = {&pass};
  Status status = PassManager::Run(*graph, passes);
  CheckGraph(graph, 1);
}

// pooling+concat
// conv2d+concat
TEST_F(UTEST_conv_concat_fusion_pass, pooling_conv2d_success) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>(GRAPH_NAME);
  InitPoolingNoQuantGraph(graph);
  ConvConcatFusionPass pass;
  vector<GraphPass*> passes = {&pass};
  Status status = PassManager::Run(*graph, passes);
  CheckGraph(graph, 1);
}

// pooling+concat
// conv2d+concat
TEST_F(UTEST_conv_concat_fusion_pass, conv2d_requant_success) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>(GRAPH_NAME);
  InitRequantGraph(graph);
  ConvConcatFusionPass pass;
  vector<GraphPass*> passes = {&pass};
  Status status = PassManager::Run(*graph, passes);
  CheckGraph(graph, 1);
}


// conv2d+concat
// conv2d+leakyrelu+concat
TEST_F(UTEST_conv_concat_fusion_pass, con2d_leakyrelu_success) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>(GRAPH_NAME);
  InitNoQuantGraph(graph, RELU, false);
  ConvConcatFusionPass pass;
  vector<GraphPass*> passes = {&pass};
  Status status = PassManager::Run(*graph, passes);
  CheckGraph(graph, 1);
}

// conv2d+acsenddequant+concat
// conv2d+acsenddequant+concat
TEST_F(UTEST_conv_concat_fusion_pass, con2d_dequant_success) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>(GRAPH_NAME);
  vector<string> op_type_vec = {DEQUANT};
  InitQuantAndConvAndReluAndLeakyreluGraph(graph, op_type_vec);
  InitQuantGraph(graph, "");
  ConvConcatFusionPass pass;
  vector<GraphPass*> passes = {&pass};
  Status status = PassManager::Run(*graph, passes);
  CheckGraph(graph, 1);
}

// conv2d+acsenddequant+concat+ascendquant
// conv2d+acsenddequant+concat+ascendquant
TEST_F(UTEST_conv_concat_fusion_pass, con2d_dequant_concat_quant_success) {
ComputeGraphPtr graph = std::make_shared<ComputeGraph>(GRAPH_NAME);
InitQuantGraph2(graph, true);
ConvConcatFusionPass pass;
vector<GraphPass*> passes = {&pass};
Status status = PassManager::Run(*graph, passes);
CheckGraph(graph, 1);
}

// conv2d+acsenddequant+concat
// conv2d+concat
TEST_F(UTEST_conv_concat_fusion_pass, con2d_dequant_and_conv_success) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>(GRAPH_NAME);
  vector<string> op_type_vec = {CONV2D};
  InitQuantAndConvAndReluAndLeakyreluGraph(graph, op_type_vec);
  ConvConcatFusionPass pass;
  vector<GraphPass*> passes = {&pass};
  Status status = PassManager::Run(*graph, passes);
  CheckGraph(graph, 1);
}

// conv2d+dequant+concat
// conv2d+relu+concat
TEST_F(UTEST_conv_concat_fusion_pass, con2d_dequant_and_relu_success) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>(GRAPH_NAME);
  vector<string> op_type_vec = {RELU};
  InitQuantAndConvAndReluAndLeakyreluGraph(graph, op_type_vec);
  ConvConcatFusionPass pass;
  vector<GraphPass*> passes = {&pass};
  Status status = PassManager::Run(*graph, passes);
  CheckGraph(graph, 1);
}

// conv2d+dequant+concat
// conv2d+leakyrelu+concat
TEST_F(UTEST_conv_concat_fusion_pass, con2d_dequant_and_leakyrelu_success) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>(GRAPH_NAME);
  vector<string> op_type_vec = {LEAKYRELU};
  InitQuantAndConvAndReluAndLeakyreluGraph(graph, op_type_vec);
  ConvConcatFusionPass pass;
  vector<GraphPass*> passes = {&pass};
  Status status = PassManager::Run(*graph, passes);
  CheckGraph(graph, 1);
}

// conv2d+dequant+concat
// conv2d+relu+concat
// conv2d+leakyrelu+concat
TEST_F(UTEST_conv_concat_fusion_pass, con2d_dequant_and_relu_leakyrelu_success) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>(GRAPH_NAME);
  vector<string> op_type_vec = {RELU, LEAKYRELU};
  InitQuantAndConvAndReluAndLeakyreluGraph(graph, op_type_vec);
  ConvConcatFusionPass pass;
  vector<GraphPass*> passes = {&pass};
  Status status = PassManager::Run(*graph, passes);
  CheckGraph(graph, 1);
}

// conv2d+dequant+concat
// conv2d+concat
// conv2d+relu+concat
TEST_F(UTEST_conv_concat_fusion_pass, con2d_dequant_and_conv_relu_success) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>(GRAPH_NAME);
  vector<string> op_type_vec = {CONV2D, RELU};
  InitQuantAndConvAndReluAndLeakyreluGraph(graph, op_type_vec);
  ConvConcatFusionPass pass;
  vector<GraphPass*> passes = {&pass};
  Status status = PassManager::Run(*graph, passes);
  CheckGraph(graph, 1);
}

// conv2d+dequant+concat
// conv2d+concat
// conv2d+leakyrelu+concat
TEST_F(UTEST_conv_concat_fusion_pass, con2d_dequant_and_conv_leakyrelu_success) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>(GRAPH_NAME);
  vector<string> op_type_vec = {CONV2D, LEAKYRELU};
  InitQuantAndConvAndReluAndLeakyreluGraph(graph, op_type_vec);
  ConvConcatFusionPass pass;
  vector<GraphPass*> passes = {&pass};
  Status status = PassManager::Run(*graph, passes);
  CheckGraph(graph, 1);
}

// conv2d+dequant+concat
// conv2d+concat
// conv2d+relu+concat
// conv2d+leakyrelu+concat
TEST_F(UTEST_conv_concat_fusion_pass, con2d_dequant_and_conv_relu_leakyrelu_success) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>(GRAPH_NAME);
  vector<string> op_type_vec = {CONV2D, RELU, LEAKYRELU};
  InitQuantAndConvAndReluAndLeakyreluGraph(graph, op_type_vec);

  ConvConcatFusionPass pass;
  vector<GraphPass*> passes = {&pass};
  Status status = PassManager::Run(*graph, passes);

  CheckGraph(graph, 1);
}

TEST_F(UTEST_conv_concat_fusion_pass, con2d_dequant_leakyrelu_concat_success) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>(GRAPH_NAME);
  BuildGraph_1(graph);

  ConvConcatFusionPass pass;
  vector<GraphPass*> passes = {&pass};
  Status status = PassManager::Run(*graph, passes);

  CheckGraph(graph, 1);
}

// conv2d+mish+concat
TEST_F(UTEST_conv_concat_fusion_pass, con2d_mish_not_change) {
  PlatformUtils::Instance().soc_version_ = "Ascend910B2";
  CubeVecStateNew cube_vec_state_bak = PlatformUtils::Instance().GetCubeVecState();
  PlatformUtils::Instance().pm_item_vec_[static_cast<size_t>(PlatformUtils::PlatformInfoItem::CubeVecState)] =
      static_cast<int64_t>(CubeVecStateNew::CUBE_VEC_SPLIT);
  PlatformInfoManager::Instance().opti_compilation_info_.soc_version = "Ascend910B2";
  PlatformInfoManager::Instance().opti_compilation_infos_.SetSocVersion("Ascend910B2");
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>(GRAPH_NAME);
  InitNoQuantGraph(graph, "Mish", false);
  EXPECT_EQ(graph->GetDirectNodesSize(), 6);
  ConvConcatFusionPass pass;
  vector<GraphPass*> passes = {&pass};
  Status status = PassManager::Run(*graph, passes);
  EXPECT_EQ(status, fe::SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(), 7);
  PlatformUtils::Instance().pm_item_vec_[static_cast<size_t>(PlatformUtils::PlatformInfoItem::CubeVecState)] =
      static_cast<int64_t>(cube_vec_state_bak);
}

// conv2d+dequant+mish+concat
TEST_F(UTEST_conv_concat_fusion_pass, con2d_dequant_mish_not_change) {
  PlatformUtils::Instance().soc_version_ = "Ascend910B2";
  CubeVecStateNew cube_vec_state_bak = PlatformUtils::Instance().GetCubeVecState();
  PlatformUtils::Instance().pm_item_vec_[static_cast<size_t>(PlatformUtils::PlatformInfoItem::CubeVecState)] =
      static_cast<int64_t>(CubeVecStateNew::CUBE_VEC_SPLIT);
  PlatformInfoManager::Instance().opti_compilation_info_.soc_version = "Ascend910B2";
  PlatformInfoManager::Instance().opti_compilation_infos_.SetSocVersion("Ascend910B2");
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>(GRAPH_NAME);
  InitQuantGraph(graph, "Mish");
  EXPECT_EQ(graph->GetDirectNodesSize(), 8);
  ConvConcatFusionPass pass;
  vector<GraphPass*> passes = {&pass};
  Status status = PassManager::Run(*graph, passes);
  EXPECT_EQ(status, fe::SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(), 9);
  PlatformUtils::Instance().pm_item_vec_[static_cast<size_t>(PlatformUtils::PlatformInfoItem::CubeVecState)] =
      static_cast<int64_t>(cube_vec_state_bak);
}

TEST_F(UTEST_conv_concat_fusion_pass, con2d_dequant_leakyrelu_concat_success_force_fp32) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>(GRAPH_NAME);
  BuildGraph_1(graph);

  auto options = ge::GetThreadLocalContext().GetAllGraphOptions();
  options[ge::PRECISION_MODE] = "force_fp32";
  ge::GetThreadLocalContext().SetGraphOption(options);
  ConvConcatFusionPass pass;
  vector<GraphPass*> passes = {&pass};
  Status status = PassManager::Run(*graph, passes);

  CheckGraph2(graph);
}

TEST_F(UTEST_conv_concat_fusion_pass, con2d_dequant_leakyrelu_concat_success_must_keep_origin_dtype) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>(GRAPH_NAME);
  BuildGraph_1(graph);

  auto options = ge::GetThreadLocalContext().GetAllGraphOptions();
  options[ge::PRECISION_MODE] = "must_keep_origin_dtype";
  ge::GetThreadLocalContext().SetGraphOption(options);


  ConvConcatFusionPass pass;
  vector<GraphPass*> passes = {&pass};
  Status status = PassManager::Run(*graph, passes);

  CheckGraph(graph, 1);
}

TEST_F(UTEST_conv_concat_fusion_pass,
       con2d_dequant_leakyrelu_concat_success_must_keep_origin_dtype_origin_float) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>(GRAPH_NAME);
  BuildGraph_1(graph, ge::DT_FLOAT);

  auto options = ge::GetThreadLocalContext().GetAllGraphOptions();
  options[ge::PRECISION_MODE] = "must_keep_origin_dtype";
  ge::GetThreadLocalContext().SetGraphOption(options);

  ConvConcatFusionPass pass;
  vector<GraphPass*> passes = {&pass};
  Status status = PassManager::Run(*graph, passes);

  CheckGraph2(graph);
}

TEST_F(UTEST_conv_concat_fusion_pass, test_match_for_no_dequuant) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>(GRAPH_NAME);
  OpDescPtr conv1 = std::make_shared<OpDesc>("conv1", CONV2D);
  OpDescPtr aaa = std::make_shared<OpDesc>("concat", "aaa");
  ge::GeShape shape1({NCHW_DIM_N, 16, NCHW_DIM_H, NCHW_DIM_W});
  GeTensorDesc out_desc1(shape1, ge::FORMAT_NCHW, ge::DT_FLOAT);
  out_desc1.SetOriginFormat(ge::FORMAT_NCHW);
  out_desc1.SetOriginDataType(ge::DT_FLOAT);
  out_desc1.SetOriginShape(shape1);

  GeTensorDesc out_desc2(shape1, ge::FORMAT_NCHW, ge::DT_FLOAT);
  out_desc2.SetOriginFormat(ge::FORMAT_NCHW);
  out_desc2.SetOriginDataType(ge::DT_FLOAT);
  out_desc2.SetOriginShape(shape1);
  aaa->AddOutputDesc(out_desc2);

  conv1->AddInputDesc(out_desc1);

  NodePtr conv1_node = graph->AddNode(conv1);
  NodePtr aaa_node = graph->AddNode(aaa);
  ge::GraphUtils::AddEdge(aaa_node->GetOutDataAnchor(0), conv1_node->GetInDataAnchor(0));

  ConvConcatFusionPass pass;
  std::vector<ge::NodePtr> mish_nodes_vec;
  pass.MatchForDequant(conv1_node, mish_nodes_vec);
}
}  // namespace fe
