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

#include "fe_llt_utils.h"
#define protected public
#define private public

#include "graph_optimizer/graph_fusion/fusion_pass_manager/builtin_pass/depthwise_insert_transdata_fusion_pass.h"
#include "graph/compute_graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "ops_kernel_store/fe_ops_kernel_info_store.h"
#include "adapter/tbe_adapter/tbe_op_store_adapter.h"
#include "common/graph/fe_graph_utils.h"
#include "common/configuration.h"
#include <vector>
#undef protected
#undef private

using namespace testing;
using namespace ge;
using namespace fe;

namespace fe {

using FEOpsKernelInfoStorePtr = std::shared_ptr<fe::FEOpsKernelInfoStore>;


class UTEST_fusion_engine_depthwise_insert_transdata_pass : public testing::Test
{
public:
  FEOpsKernelInfoStorePtr fe_ops_kernel_info_store_ptr;
protected:
  void SetUp()
  {
    fe_ops_kernel_info_store_ptr = make_shared<fe::FEOpsKernelInfoStore>(fe::AI_CORE_NAME);
    FEOpsStoreInfo tbe_builtin {
            0,
            "tbe-builtin",
            EN_IMPL_HW_TBE,
            GetCodeDir() + "/tests/engines/nn_engine/st/testcase/ops_kernel_store/fe_config/tbe_opinfo",
            "",
            false,
            false,
            false};
    vector<FEOpsStoreInfo> store_info;
    store_info.emplace_back(tbe_builtin);
    Configuration::Instance(fe::AI_CORE_NAME).ops_store_info_vector_ = (store_info);
    std::map<std::string, std::string> options;
    fe_ops_kernel_info_store_ptr->Initialize(options);

  }

  void TearDown()
  {
    fe_ops_kernel_info_store_ptr->Finalize();

  }

  static NodePtr CreateTransDataNode(string name, GeTensorDescPtr out_desc_ptr, ComputeGraphPtr graph,
                                     const string &type)
  {
    OpDescPtr data = std::make_shared<OpDesc>(name, type);
    data->AddInputDesc(out_desc_ptr->Clone());
    //set OpDesc
    data->AddOutputDesc(out_desc_ptr->Clone());
    // set attr
    NodePtr node_const = graph->AddNode(data);

    return node_const;
  }

  static NodePtr CreateConvNode(string node_name, ComputeGraphPtr graph, std::vector<GeTensorDescPtr> &input_tensor) {
    OpDescPtr conv_op = std::make_shared<OpDesc>(node_name, "DepthwiseConv2D");
    for (auto item : input_tensor) {
      conv_op->AddInputDesc(item->Clone());
    }
    conv_op->AddOutputDesc(input_tensor[0]->Clone());
    NodePtr conv_node = graph->AddNode(conv_op);
    ge::AttrUtils::SetInt(conv_op, FE_IMPLY_TYPE, 6);

    return conv_node;
  }

  static ComputeGraphPtr CreateTestGraph_NHWC()
  {
    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    GeTensorDescPtr input_tensor_desc = std::make_shared<GeTensorDesc>();
    vector<int64_t> roi_dims{16, 1, 224, 224, 16};
    vector<int64_t> roi_dims_{16, 3, 224, 224};
    GeShape rois_shape(roi_dims);
    GeShape rois_shape_(roi_dims_);

    input_tensor_desc->SetShape(rois_shape);
    input_tensor_desc->SetOriginShape(rois_shape_);
    input_tensor_desc->SetFormat(FORMAT_NC1HWC0);
    input_tensor_desc->SetOriginFormat(FORMAT_NCHW);
    input_tensor_desc->SetDataType(DT_FLOAT16);
    NodePtr data0 = CreateTransDataNode("test/data0", input_tensor_desc, graph, "Data");
    // new a output GeTensorDesc
    GeTensorDescPtr general_ge_tensor_desc = std::make_shared<GeTensorDesc>();
    vector<int64_t> dims{1, 7, 7, 16};
    GeShape shape(dims);
    general_ge_tensor_desc->SetShape(shape);
    general_ge_tensor_desc->SetOriginShape(shape);
    general_ge_tensor_desc->SetFormat(FORMAT_NHWC);
    general_ge_tensor_desc->SetOriginFormat(FORMAT_NHWC);
    general_ge_tensor_desc->SetDataType(DT_FLOAT16);
    NodePtr data1 = CreateTransDataNode("test/data1", general_ge_tensor_desc, graph, "Const");

    GeTensorDescPtr trans_data_tensor = std::make_shared<GeTensorDesc>();
    vector<int64_t> trans_data{49, 1, 16, 16};
    GeShape trans_shape(trans_data);
    trans_data_tensor->SetShape(trans_shape);
    trans_data_tensor->SetOriginShape(shape);
    trans_data_tensor->SetFormat(FORMAT_FRACTAL_Z);
    trans_data_tensor->SetOriginFormat(FORMAT_NHWC);
    trans_data_tensor->SetDataType(DT_FLOAT16);
    NodePtr trans_node = CreateTransDataNode("trans_TransData_0", trans_data_tensor, graph, "TransData");

    trans_node->GetOpDesc()->UpdateInputDesc(0, general_ge_tensor_desc->Clone());
    std::vector<GeTensorDescPtr> input_tensors = {input_tensor_desc, trans_data_tensor};
    NodePtr depthwise_node = CreateConvNode("depthwise", graph, input_tensors);

    GraphUtils::AddEdge(data0->GetOutDataAnchor(0), depthwise_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(data1->GetOutDataAnchor(0), trans_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(trans_node->GetOutDataAnchor(0), depthwise_node->GetInDataAnchor(1));

    return graph;
  }
};

TEST_F(UTEST_fusion_engine_depthwise_insert_transdata_pass, depthwise_insert_02)
{
  ComputeGraphPtr graph = CreateTestGraph_NHWC();
  DepthwiseInsertTransDataFusionPass pass;
  fe::Status status = pass.Run(*graph, fe_ops_kernel_info_store_ptr);
  EXPECT_EQ(fe::SUCCESS, status);

  for(auto node : graph->GetDirectNode()) {
    if (node->GetType() == "DepthwiseConv2D") {
      ge::NodePtr trans1 = node->GetInDataNodes().at(1);
      ge::OpDescPtr trans1_desc = trans1->GetOpDesc();
      EXPECT_EQ(trans1_desc->GetType(), "TransData");
      EXPECT_EQ(trans1_desc->GetInputDescPtr(0)->GetFormat(), ge::FORMAT_HWCN);
      EXPECT_EQ(trans1_desc->GetInputDescPtr(0)->GetShape().GetDims()[0], 7);
      EXPECT_EQ(trans1_desc->GetInputDescPtr(0)->GetShape().GetDims()[1], 7);
      EXPECT_EQ(trans1_desc->GetInputDescPtr(0)->GetShape().GetDims()[2], 16);
      EXPECT_EQ(trans1_desc->GetInputDescPtr(0)->GetShape().GetDims()[3], 1);
      EXPECT_EQ(trans1_desc->GetOutputDescPtr(0)->GetFormat(), ge::FORMAT_FRACTAL_Z);
      EXPECT_EQ(trans1_desc->GetOutputDescPtr(0)->GetShape().GetDims()[0], 49);
      EXPECT_EQ(trans1_desc->GetOutputDescPtr(0)->GetShape().GetDims()[1], 1);
      EXPECT_EQ(trans1_desc->GetOutputDescPtr(0)->GetShape().GetDims()[2], 16);
      EXPECT_EQ(trans1_desc->GetOutputDescPtr(0)->GetShape().GetDims()[3], 16);
      ge::NodePtr trans2 = trans1->GetInDataNodes().at(0);
      ge::OpDescPtr trans2_desc = trans2->GetOpDesc();
      EXPECT_EQ(trans2_desc->GetType(), "TransData");
      EXPECT_EQ(trans2_desc->GetInputDescPtr(0)->GetFormat(), ge::FORMAT_NHWC);
      EXPECT_EQ(trans2_desc->GetInputDescPtr(0)->GetShape().GetDims()[0], 1);
      EXPECT_EQ(trans2_desc->GetInputDescPtr(0)->GetShape().GetDims()[1], 7);
      EXPECT_EQ(trans2_desc->GetInputDescPtr(0)->GetShape().GetDims()[2], 7);
      EXPECT_EQ(trans2_desc->GetInputDescPtr(0)->GetShape().GetDims()[3], 16);
      EXPECT_EQ(trans2_desc->GetOutputDescPtr(0)->GetFormat(), ge::FORMAT_HWCN);
      EXPECT_EQ(trans2_desc->GetOutputDescPtr(0)->GetShape().GetDims()[0], 7);
      EXPECT_EQ(trans2_desc->GetOutputDescPtr(0)->GetShape().GetDims()[1], 7);
      EXPECT_EQ(trans2_desc->GetOutputDescPtr(0)->GetShape().GetDims()[2], 16);
      EXPECT_EQ(trans2_desc->GetOutputDescPtr(0)->GetShape().GetDims()[3], 1);
    }
  }
}

}
