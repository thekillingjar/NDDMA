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
#include "graph/ge_tensor.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/op_desc.h"
#include "graph/compute_graph.h"
#include "graph_optimizer/fusion_common/graph_pass_util.h"
#include "register/graph_optimizer/graph_fusion/fusion_quant_util.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
using namespace std;
using namespace ge;

namespace fe {

class FusionQuantUtilUT : public testing::Test {
protected:
  void SetUp() {}

  void TearDown() {}

  static ge::ComputeGraphPtr CreateTestGraphWithOffset() {
    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    OpDescPtr x = std::make_shared<OpDesc>("x", "Data");
    OpDescPtr weight = std::make_shared<OpDesc>("weight", "Const");
    OpDescPtr atquant_scale = std::make_shared<OpDesc>("atquant_scale", "Const");
    OpDescPtr quant_scale = std::make_shared<OpDesc>("quant_scale", "Const");
    OpDescPtr quant_offset = std::make_shared<OpDesc>("quant_offset", "Const");
    OpDescPtr mm = std::make_shared<OpDesc>("mm", "WeightQuantBatchMatmulV2");
    OpDescPtr y = std::make_shared<OpDesc>("y", "NetOutput");

    // add descriptor
    ge::GeShape shape1({2,4,9,16});
    GeTensorDesc tensor_desc1(shape1, ge::FORMAT_NCHW, ge::DT_FLOAT16);
    tensor_desc1.SetOriginFormat(ge::FORMAT_NCHW);
    tensor_desc1.SetOriginDataType(ge::DT_FLOAT16);
    tensor_desc1.SetOriginShape(shape1);

    GeTensorDesc tensor_desc2(shape1, ge::FORMAT_NCHW, ge::DT_INT8);
    tensor_desc2.SetOriginFormat(ge::FORMAT_NCHW);
    tensor_desc2.SetOriginDataType(ge::DT_INT8);
    tensor_desc2.SetOriginShape(shape1);

    ge::GeShape shape2({1, 16});
    GeTensorDesc tensor_desc3(shape2, ge::FORMAT_ND, ge::DT_FLOAT);
    tensor_desc3.SetOriginFormat(ge::FORMAT_ND);
    tensor_desc3.SetOriginDataType(ge::DT_FLOAT);
    tensor_desc3.SetOriginShape(shape2);

    x->AddOutputDesc(tensor_desc1);
    weight->AddOutputDesc(tensor_desc2);
    atquant_scale->AddOutputDesc(tensor_desc1);
    quant_scale->AddOutputDesc(tensor_desc3);
    quant_offset->AddOutputDesc(tensor_desc3);

    mm->AddInputDesc(tensor_desc1);
    mm->AddInputDesc(tensor_desc2);
    mm->AddInputDesc(tensor_desc1);
    mm->AddInputDesc(tensor_desc3);
    mm->AddInputDesc(tensor_desc3);
    mm->AddOutputDesc(tensor_desc2);
    y->AddInputDesc(tensor_desc2);

    // create nodes
    NodePtr x_node = graph->AddNode(x);
    NodePtr weight_node = graph->AddNode(weight);
    NodePtr atquant_scale_node = graph->AddNode(atquant_scale);
    NodePtr quant_scale_node = graph->AddNode(quant_scale);
    NodePtr quant_offset_node = graph->AddNode(quant_offset);
    NodePtr mm_node = graph->AddNode(mm);
    NodePtr y_node = graph->AddNode(y);

    // link edge
    ge::GraphUtils::AddEdge(x_node->GetOutDataAnchor(0),
                            mm_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(weight_node->GetOutDataAnchor(0),
                            mm_node->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(atquant_scale_node->GetOutDataAnchor(0),
                            mm_node->GetInDataAnchor(2));
    ge::GraphUtils::AddEdge(quant_scale_node->GetOutDataAnchor(0),
                            mm_node->GetInDataAnchor(3));
    ge::GraphUtils::AddEdge(quant_offset_node->GetOutDataAnchor(0),
                            mm_node->GetInDataAnchor(4));
    ge::GraphUtils::AddEdge(mm_node->GetOutDataAnchor(0),
                            y_node->GetInDataAnchor(0));
    return graph;
  }

  static ge::ComputeGraphPtr CreateTestGraphWithOffset2() {
    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    OpDescPtr x = std::make_shared<OpDesc>("x", "Data");
    OpDescPtr weight = std::make_shared<OpDesc>("weight", "Const");
    OpDescPtr atquant_scale = std::make_shared<OpDesc>("atquant_scale", "Const");
    OpDescPtr quant_scale = std::make_shared<OpDesc>("quant_scale", "Const");
    OpDescPtr quant_offset = std::make_shared<OpDesc>("quant_offset", "Const");
    OpDescPtr mm = std::make_shared<OpDesc>("mm", "WeightQuantBatchMatmulV2");
    OpDescPtr y = std::make_shared<OpDesc>("y", "NetOutput");

    // add descriptor
    ge::GeShape shape1({2,4,9,16});
    GeTensorDesc tensor_desc1(shape1, ge::FORMAT_NCHW, ge::DT_FLOAT16);
    tensor_desc1.SetOriginFormat(ge::FORMAT_NCHW);
    tensor_desc1.SetOriginDataType(ge::DT_FLOAT16);
    tensor_desc1.SetOriginShape(shape1);

    GeTensorDesc tensor_desc2(shape1, ge::FORMAT_NCHW, ge::DT_INT8);
    tensor_desc2.SetOriginFormat(ge::FORMAT_NCHW);
    tensor_desc2.SetOriginDataType(ge::DT_INT8);
    tensor_desc2.SetOriginShape(shape1);

    ge::GeShape shape2({16});
    GeTensorDesc tensor_desc3(shape2, ge::FORMAT_ND, ge::DT_FLOAT);
    tensor_desc3.SetOriginFormat(ge::FORMAT_ND);
    tensor_desc3.SetOriginDataType(ge::DT_FLOAT);
    tensor_desc3.SetOriginShape(shape2);

    x->AddOutputDesc(tensor_desc1);
    weight->AddOutputDesc(tensor_desc2);
    atquant_scale->AddOutputDesc(tensor_desc1);
    quant_scale->AddOutputDesc(tensor_desc3);
    quant_offset->AddOutputDesc(tensor_desc3);

    mm->AddInputDesc(tensor_desc1);
    mm->AddInputDesc(tensor_desc2);
    mm->AddInputDesc(tensor_desc1);
    mm->AddInputDesc(tensor_desc3);
    mm->AddInputDesc(tensor_desc3);
    mm->AddOutputDesc(tensor_desc2);
    y->AddInputDesc(tensor_desc2);

    // create nodes
    NodePtr x_node = graph->AddNode(x);
    NodePtr weight_node = graph->AddNode(weight);
    NodePtr atquant_scale_node = graph->AddNode(atquant_scale);
    NodePtr quant_scale_node = graph->AddNode(quant_scale);
    NodePtr quant_offset_node = graph->AddNode(quant_offset);
    NodePtr mm_node = graph->AddNode(mm);
    NodePtr y_node = graph->AddNode(y);

    // link edge
    ge::GraphUtils::AddEdge(x_node->GetOutDataAnchor(0),
                            mm_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(weight_node->GetOutDataAnchor(0),
                            mm_node->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(atquant_scale_node->GetOutDataAnchor(0),
                            mm_node->GetInDataAnchor(2));
    ge::GraphUtils::AddEdge(quant_scale_node->GetOutDataAnchor(0),
                            mm_node->GetInDataAnchor(3));
    ge::GraphUtils::AddEdge(quant_offset_node->GetOutDataAnchor(0),
                            mm_node->GetInDataAnchor(4));
    ge::GraphUtils::AddEdge(mm_node->GetOutDataAnchor(0),
                            y_node->GetInDataAnchor(0));
    return graph;
  }

  static ge::ComputeGraphPtr CreateTestGraphWithoutOffset() {
    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    OpDescPtr x = std::make_shared<OpDesc>("x", "Data");
    OpDescPtr weight = std::make_shared<OpDesc>("weight", "Const");
    OpDescPtr atquant_scale = std::make_shared<OpDesc>("atquant_scale", "Const");
    OpDescPtr quant_scale = std::make_shared<OpDesc>("quant_scale", "Const");
    OpDescPtr mm = std::make_shared<OpDesc>("mm", "WeightQuantBatchMatmulV2");
    OpDescPtr y = std::make_shared<OpDesc>("y", "NetOutput");

    // add descriptor
    ge::GeShape shape1({2,4,9,16});
    GeTensorDesc tensor_desc1(shape1, ge::FORMAT_NCHW, ge::DT_FLOAT16);
    tensor_desc1.SetOriginFormat(ge::FORMAT_NCHW);
    tensor_desc1.SetOriginDataType(ge::DT_FLOAT16);
    tensor_desc1.SetOriginShape(shape1);

    GeTensorDesc tensor_desc2(shape1, ge::FORMAT_NCHW, ge::DT_INT8);
    tensor_desc2.SetOriginFormat(ge::FORMAT_NCHW);
    tensor_desc2.SetOriginDataType(ge::DT_INT8);
    tensor_desc2.SetOriginShape(shape1);

    ge::GeShape shape2({1, 16});
    GeTensorDesc tensor_desc3(shape2, ge::FORMAT_ND, ge::DT_FLOAT);
    tensor_desc3.SetOriginFormat(ge::FORMAT_ND);
    tensor_desc3.SetOriginDataType(ge::DT_FLOAT);
    tensor_desc3.SetOriginShape(shape2);

    x->AddOutputDesc(tensor_desc1);
    weight->AddOutputDesc(tensor_desc2);
    atquant_scale->AddOutputDesc(tensor_desc1);
    quant_scale->AddOutputDesc(tensor_desc3);

    mm->AddInputDesc(tensor_desc1);
    mm->AddInputDesc(tensor_desc2);
    mm->AddInputDesc(tensor_desc1);
    mm->AddInputDesc(tensor_desc3);
    mm->AddOutputDesc(tensor_desc2);
    y->AddInputDesc(tensor_desc2);

    // create nodes
    NodePtr x_node = graph->AddNode(x);
    NodePtr weight_node = graph->AddNode(weight);
    NodePtr atquant_scale_node = graph->AddNode(atquant_scale);
    NodePtr quant_scale_node = graph->AddNode(quant_scale);
    NodePtr mm_node = graph->AddNode(mm);
    NodePtr y_node = graph->AddNode(y);

    // link edge
    ge::GraphUtils::AddEdge(x_node->GetOutDataAnchor(0),
                            mm_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(weight_node->GetOutDataAnchor(0),
                            mm_node->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(atquant_scale_node->GetOutDataAnchor(0),
                            mm_node->GetInDataAnchor(2));
    ge::GraphUtils::AddEdge(quant_scale_node->GetOutDataAnchor(0),
                            mm_node->GetInDataAnchor(3));
    ge::GraphUtils::AddEdge(mm_node->GetOutDataAnchor(0),
                            y_node->GetInDataAnchor(0));
    return graph;
  }

  static void FillWeightValue2(const ge::ComputeGraphPtr &graph) {
    for (const ge::NodePtr &node : graph->GetDirectNode()) {
      if (node == nullptr) {
        continue;
      }
      if (node->GetType() != "Const") {
        continue;
      }
      std::vector<ge::ConstGeTensorPtr> weights = ge::OpDescUtils::GetWeights(node);
      if (weights.empty()) {
        ge::ConstGeTensorDescPtr out_tensor = node->GetOpDesc()->GetOutputDescPtr(0);
        int64_t shape_size = out_tensor->GetShape().GetShapeSize();
        if (shape_size <= 0) {
          continue;
        }
        ge::GeTensorPtr weight = std::make_shared<ge::GeTensor>(*out_tensor);
        if (node->GetName() == "quant_scale") {
          vector<float> data_vec =
              {-2.7065194, -4.7495637, 2.5856478, 2.533566 , -2.7307642, 0.08650689, 1.2195834, -4.520703,
               -4.902806, -4.9793777 , -3.8038466 , 4.6814585, -0.8230759, 1.4473673, 4.71265, 2.3249402};
          weight->SetData(reinterpret_cast<uint8_t *>(data_vec.data()), shape_size * sizeof(float));
          ge::OpDescUtils::SetWeights(node->GetOpDesc(), weight);
          continue;
        }
        if (node->GetName() == "quant_offset") {
          std::cout << "mmm quant_offset" << std::endl;
          vector<float> data_vec =
              {1.7815902, -0.83771265, 3.8743427, -1.129952, 3.348905, 4.898297, 2.8627427, -4.685532,
               -1.0928544, 0.0128879, 3.988301, -4.4012594, -0.15809901, 1.5274582, 3.3731332, -0.75769955};
          weight->SetData(reinterpret_cast<uint8_t *>(data_vec.data()), shape_size * sizeof(float));
          ge::OpDescUtils::SetWeights(node->GetOpDesc(), weight);
          continue;
        }
        if (out_tensor->GetDataType() == ge::DT_UINT32 || out_tensor->GetDataType() == ge::DT_INT32 ||
            out_tensor->GetDataType() == ge::DT_FLOAT) {
          vector<int32_t> data_vec(shape_size, 1);
          weight->SetData(reinterpret_cast<uint8_t *>(data_vec.data()), shape_size * sizeof(int32_t));
        }
        if (out_tensor->GetDataType() == ge::DT_UINT64 || out_tensor->GetDataType() == ge::DT_INT64 ||
            out_tensor->GetDataType() == ge::DT_DOUBLE) {
          vector<int64_t> data_vec(shape_size, 1);
          weight->SetData(reinterpret_cast<uint8_t *>(data_vec.data()), shape_size * sizeof(int64_t));
        }
        if (out_tensor->GetDataType() == ge::DT_UINT16 || out_tensor->GetDataType() == ge::DT_INT16 ||
            out_tensor->GetDataType() == ge::DT_FLOAT16) {
          vector<int16_t> data_vec(shape_size, 1);
          weight->SetData(reinterpret_cast<uint8_t *>(data_vec.data()), shape_size * sizeof(int16_t));
        }
        if (out_tensor->GetDataType() == ge::DT_UINT8 || out_tensor->GetDataType() == ge::DT_INT8) {
          vector<int8_t> data_vec(shape_size, 1);
          weight->SetData(reinterpret_cast<uint8_t *>(data_vec.data()), shape_size * sizeof(int8_t));
        }
        ge::OpDescUtils::SetWeights(node->GetOpDesc(), weight);
      }
    }
  }
};

TEST_F(FusionQuantUtilUT, insert_quant_op_succ) {
  ComputeGraphPtr graph = CreateTestGraphWithOffset();
  FillWeightValue2(graph);
  ge::NodePtr mm_node = graph->FindNode("mm");
  InDataAnchorPtr cuba_bias = mm_node->GetInDataAnchor(1);
  InDataAnchorPtr quant_scale = mm_node->GetInDataAnchor(3);
  InDataAnchorPtr quant_offset = mm_node->GetInDataAnchor(4);
  std::vector<ge::NodePtr> fusion_nodes;
  Status ret = QuantUtil::InsertQuantScaleConvert(quant_scale, quant_offset, fusion_nodes);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(FusionQuantUtilUT, insert_quant_op_succ2) {
  ComputeGraphPtr graph = CreateTestGraphWithoutOffset();
  FillWeightValue2(graph);
  ge::NodePtr mm_node = graph->FindNode("mm");
  InDataAnchorPtr cuba_bias = mm_node->GetInDataAnchor(1);
  InDataAnchorPtr quant_scale = mm_node->GetInDataAnchor(3);
  InDataAnchorPtr quant_offset = mm_node->GetInDataAnchor(4);
  std::vector<ge::NodePtr> fusion_nodes;
  Status ret = QuantUtil::InsertQuantScaleConvert(quant_scale, quant_offset, fusion_nodes);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(FusionQuantUtilUT, insert_requant_op_succ) {
  ComputeGraphPtr graph = CreateTestGraphWithOffset2();
  FillWeightValue2(graph);
  ge::NodePtr mm_node = graph->FindNode("mm");
  InDataAnchorPtr cuba_bias = mm_node->GetInDataAnchor(1);
  InDataAnchorPtr quant_scale = mm_node->GetInDataAnchor(3);
  InDataAnchorPtr quant_offset = mm_node->GetInDataAnchor(4);
  std::vector<ge::NodePtr> fusion_nodes;
  Status ret = QuantUtil::InsertRequantScaleConvert(quant_scale, quant_offset, cuba_bias, fusion_nodes);
  EXPECT_EQ(ret, SUCCESS);
}
}
