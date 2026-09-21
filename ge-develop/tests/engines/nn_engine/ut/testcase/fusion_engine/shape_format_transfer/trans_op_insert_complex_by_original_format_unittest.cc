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
#include "graph/utils/tensor_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/attr_utils.h"
#define protected public
#define private   public
#include "adapter/tbe_adapter/tbe_op_store_adapter.h"
#include "ops_kernel_store/fe_ops_kernel_info_store.h"
#include "graph_optimizer/shape_format_transfer/trans_node_manager/trans_node_manager.h"
#include "graph_optimizer/shape_format_transfer/trans_node_implementation/trans_node_base_generator.h"
#include "graph_optimizer/shape_format_transfer/trans_node_implementation/trans_node_cast_generator.h"
#include "graph_optimizer/shape_format_transfer/trans_node_implementation/trans_node_reshape_generator.h"
#include "graph_optimizer/shape_format_transfer/trans_node_implementation/trans_node_transpose_generator.h"
#include "graph_optimizer/shape_format_transfer/trans_node_implementation/trans_node_transdata_generator.h"
#include "common/configuration.h"
#include "ops_store/ops_kernel_manager.h"
#include "graph/operator_factory_impl.h"
#undef protected
#undef private

#include <iostream>

using namespace std;
using namespace ge;
using namespace fe;



class UTEST_FE_TRANSOP_INSERT_COMPLEX_BY_ORIGINAL_FORMAT : public testing::Test {
 protected:
  void SetUp()
  {
    std::map<std::string, std::string> options;
    fe_ops_kernel_info_store_ptr_ = make_shared<fe::FEOpsKernelInfoStore>(fe::AI_CORE_NAME);
    FEOpsStoreInfo tbe_custom {
        6,
        "tbe-custom",
        EN_IMPL_HW_TBE,
        GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/tbe_custom_opinfo",
        ""};
    vector<FEOpsStoreInfo> store_info;
    store_info.emplace_back(tbe_custom);
    Configuration::Instance(fe::AI_CORE_NAME).ops_store_info_vector_ = (store_info);
    OpsKernelManager::Instance(AI_CORE_NAME).Finalize();

    fe_ops_kernel_info_store_ptr_->Initialize(options);
    RegisterOpCreator("Transpose", {"x", "perm"}, {"y"});
  }

  void TearDown()
  {
    fe_ops_kernel_info_store_ptr_->Finalize();

  }

  static void RegisterOpCreator(const std::string &op_type,
                              const std::vector<std::string> &input_names,
                              const std::vector<std::string> &output_names) {
    auto op_creator = [op_type, input_names, output_names](const std::string &name) -> Operator {
      auto op_desc = make_shared<OpDesc>(name, op_type);
      for (const auto &tensor_name : input_names) {
        op_desc->AddInputDesc(tensor_name, {});
      }
      for (const auto &tensor_name : output_names) {
        op_desc->AddOutputDesc(tensor_name, {});
      }
      return OpDescUtils::CreateOperatorFromOpDesc(op_desc);
    };
    OperatorFactoryImpl::RegisterOperatorCreator(op_type, op_creator);
  }

  shared_ptr<fe::FEOpsKernelInfoStore> fe_ops_kernel_info_store_ptr_;
 protected:

};

Status QueryHighPrioOpImplTypeStubTbeOriginalFormat(
    FEOpsKernelInfoStore* This, const ge::OpDescPtr& op_desc_ptr,
    OpImplType &impl_type, OpKernelInfoPtr &op_kernel_ptr) {
  impl_type = EN_IMPL_HW_TBE;
  return fe::SUCCESS;
}

/* NC1HWC0(fp16) ->NC1HWC0(fp32) -> NCHW(fp32) -> | NHWC(fp32)->NCHW (fp32)
 * The Program will insert Transdata->Cast->Reshape ops.
 * Reshape Failed because Dimension H/W is not 1*/
TEST_F(UTEST_FE_TRANSOP_INSERT_COMPLEX_BY_ORIGINAL_FORMAT, InsertAllTransopByOriginalFormat_01) {
// src:cce op, dst:cce op
  vector<int64_t> input_dim = {100, 16, 256, 512, 16};
  vector<int64_t> output_dim = {2, 3};
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  GeTensorDesc src_tensor_desc(GeShape(input_dim),
                               ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  src_tensor_desc.SetOriginShape(GeShape(output_dim));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);
  ge::AttrUtils::SetInt(src_op, FE_IMPLY_TYPE, 6);

  OpDescPtr dst_op = std::make_shared<OpDesc>("E", "E");
  GeTensorDesc dst_tensor_desc(GeShape(output_dim), ge::FORMAT_NCHW, ge::DT_FLOAT);
  dst_tensor_desc.SetOriginShape(GeShape(output_dim));
  dst_tensor_desc.SetOriginFormat(ge::FORMAT_NHWC);
  dst_op->AddInputDesc(dst_tensor_desc);
  dst_op->AddOutputDesc(dst_tensor_desc);
  auto dst_node = graph->AddNode(dst_op);
  ge::AttrUtils::SetInt(dst_op, FE_IMPLY_TYPE, 6);
  vector<bool> input_const_vector = {false};
  dst_op->SetIsInputConst(input_const_vector);

  GraphUtils::AddEdge(src_node->GetOutDataAnchor(0),
                      dst_node->GetInDataAnchor(0));
  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status status = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));

  int count_node = 0;
  ASSERT_EQ(fe::SUCCESS, status);
  for (auto node : graph->GetDirectNode()) {
    ASSERT_NE(node, nullptr);
    count_node++;
    std::vector<int64_t > output_dim4_d = {1,2,3,1};
    if (node->GetType() == "TransData" && count_node == 2) {
      {
        vector<int64_t> input_dim_trans_data = {100, 16, 256, 512, 16};
        ge::GeShape shape = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        EXPECT_EQ(shape.GetDims(), input_dim_trans_data);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetDataType(),
                  ge::DT_FLOAT16);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetFormat(),
                  ge::FORMAT_NC1HWC0);
      }
      {

        ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        EXPECT_EQ(shape.GetDims(), output_dim4_d);
        EXPECT_EQ(node->GetOpDesc()->GetOutputDescPtr(0)->GetDataType(),
                  ge::DT_FLOAT16);
        EXPECT_EQ(node->GetOpDesc()->GetOutputDescPtr(0)->GetFormat(),
                  ge::FORMAT_NCHW);
      }
    }

    if (node->GetType() == fe::SQUEEZE_V2) {
      {
        ge::GeShape shape = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        EXPECT_EQ(shape.GetDims(), output_dim4_d);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetDataType(),
                  ge::DT_FLOAT16);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetFormat(),
                  ge::FORMAT_NCHW);
      }
      {
        ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        EXPECT_EQ(shape.GetDims(), output_dim);
        EXPECT_EQ(node->GetOpDesc()->GetOutputDescPtr(0)->GetDataType(),
                  ge::DT_FLOAT16);
        EXPECT_EQ(node->GetOpDesc()->GetOutputDescPtr(0)->GetFormat(),
                  ge::FORMAT_NCHW);
      }
    }

    if (node->GetType() == TRANSPOSE && count_node == 5) {
      {
        vector<int64_t> transpose_input = {2,1,1,3};
        ge::GeShape shape = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        //EXPECT_EQ(shape.GetDims(), transpose_input);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetDataType(),
                  ge::DT_FLOAT16);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetFormat(),
                  ge::FORMAT_NHWC);
        EXPECT_EQ(ge::AttrUtils::HasAttr(node->GetOpDesc(), "perm"), false);
        EXPECT_EQ(ge::AttrUtils::HasAttr(node->GetOpDesc(), "order"), true);
      }
      {
        vector<int64_t> transpose_output = {2, 3, 1, 1};
        ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        //EXPECT_EQ(shape.GetDims(), transpose_output);
        EXPECT_EQ(node->GetOpDesc()->GetOutputDescPtr(0)->GetDataType(),
                  ge::DT_FLOAT16);
        EXPECT_EQ(node->GetOpDesc()->GetOutputDescPtr(0)->GetFormat(),
                  ge::FORMAT_NCHW);
      }
    }

    if (node->GetType() == "Cast") {
      vector<int64_t> transpose_output = {2, 3, 1, 1};
      {
        ge::GeShape shape = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        //EXPECT_EQ(shape.GetDims(), transpose_output);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetDataType(),
                  ge::DT_FLOAT16);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetFormat(),
                  ge::FORMAT_NCHW);
      }
      {
        ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        //EXPECT_EQ(shape.GetDims(), transpose_output);
        EXPECT_EQ(node->GetOpDesc()->GetOutputDescPtr(0)->GetDataType(),
                  ge::DT_FLOAT);
        EXPECT_EQ(node->GetOpDesc()->GetOutputDescPtr(0)->GetFormat(),
                  ge::FORMAT_NCHW);
      }
    }
  }
  EXPECT_EQ(count_node, 7);
}


/* NC1HWC0(fp16) ->NC1HWC0(fp32) -> NCHW(fp32) -> | NCHW(fp32)->NC1HWC0 (fp32)
 * The Program will insert Transdata->Cast->Reshape ops.
 * Reshape Failed because Dimension H/W is not 1*/
TEST_F(UTEST_FE_TRANSOP_INSERT_COMPLEX_BY_ORIGINAL_FORMAT, InsertAllTransopByOriginalFormat_02) {
// src:cce op, dst:cce op
  vector<int64_t> current5_d_dim = {1, 4, 128, 1, 16};
  vector<int64_t> current4_d_dim = {1, 64, 128, 1};
  vector<int64_t> original_dim = {64, 128};
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  GeTensorDesc src_tensor_desc(GeShape(current5_d_dim),
                               ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  src_tensor_desc.SetOriginShape(GeShape(original_dim));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);
  ge::AttrUtils::SetInt(src_op, FE_IMPLY_TYPE, 6);

  vector<int64_t> current5_d_dim_of_e = {64, 8, 1, 1,16};
  vector<int64_t> current4_d_dim_of_e = {64, 128, 1, 1};
  OpDescPtr dst_op = std::make_shared<OpDesc>("E", "E");
  GeTensorDesc dst_tensor_desc(GeShape(current5_d_dim_of_e), ge::FORMAT_NC1HWC0, ge::DT_FLOAT);
  dst_tensor_desc.SetOriginShape(GeShape(original_dim));
  dst_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  dst_op->AddInputDesc(dst_tensor_desc);
  dst_op->AddOutputDesc(dst_tensor_desc);
  auto dst_node = graph->AddNode(dst_op);
  ge::AttrUtils::SetInt(dst_op, FE_IMPLY_TYPE, 6);
  vector<bool> input_const_vector = {false};
  dst_op->SetIsInputConst(input_const_vector);

  GraphUtils::AddEdge(src_node->GetOutDataAnchor(0),
                      dst_node->GetInDataAnchor(0));
  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status status = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));

  int count_node = 0;
  ASSERT_EQ(fe::SUCCESS, status);
  for (auto node : graph->GetDirectNode()) {
    ASSERT_NE(node, nullptr);
    count_node++;
    if (node->GetType() == "TransData" && count_node == 2) {
      {
        ge::GeShape shape = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        EXPECT_EQ(shape.GetDims(), current5_d_dim);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetDataType(),
                  ge::DT_FLOAT16);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetFormat(),
                  ge::FORMAT_NC1HWC0);
      }
      {

        ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        EXPECT_EQ(shape.GetDims(), current4_d_dim);
        EXPECT_EQ(node->GetOpDesc()->GetOutputDescPtr(0)->GetDataType(),
                  ge::DT_FLOAT16);
        EXPECT_EQ(GetPrimaryFormat(node->GetOpDesc()->GetOutputDescPtr(0)->GetFormat()),
                  ge::FORMAT_NCHW);
      }
    }

    if (node->GetType() == fe::SQUEEZE_V2) {
      {
        ge::GeShape shape = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        EXPECT_EQ(shape.GetDims(), current4_d_dim);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetDataType(),
                  ge::DT_FLOAT16);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetFormat(),
                  ge::FORMAT_NCHW);
      }
      {
        ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        EXPECT_EQ(shape.GetDims(), original_dim);
        EXPECT_EQ(node->GetOpDesc()->GetOutputDescPtr(0)->GetDataType(),
                  ge::DT_FLOAT16);
        EXPECT_EQ(node->GetOpDesc()->GetOutputDescPtr(0)->GetFormat(),
                  ge::FORMAT_NCHW);
      }
    }
    if (node->GetType() == fe::UNSQUEEZE_V2) {
      {
        ge::GeShape shape = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        EXPECT_EQ(shape.GetDims(), original_dim);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetDataType(),
                  ge::DT_FLOAT16);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetFormat(),
                  ge::FORMAT_NCHW);
      }
      {
        ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        EXPECT_EQ(shape.GetDims(), current4_d_dim_of_e);
        EXPECT_EQ(node->GetOpDesc()->GetOutputDescPtr(0)->GetDataType(),
                  ge::DT_FLOAT16);
        EXPECT_EQ(node->GetOpDesc()->GetOutputDescPtr(0)->GetFormat(),
                  ge::FORMAT_NCHW);
      }
    }
    if (node->GetType() == "TransData" && count_node == 5) {
      {
        ge::GeShape shape = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        EXPECT_EQ(shape.GetDims(), current4_d_dim_of_e);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetDataType(),
                  ge::DT_FLOAT16);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetFormat(),
                  ge::FORMAT_NCHW);
      }
      {

        ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        EXPECT_EQ(shape.GetDims(), current5_d_dim_of_e);
        EXPECT_EQ(node->GetOpDesc()->GetOutputDescPtr(0)->GetDataType(),
                  ge::DT_FLOAT16);
        EXPECT_EQ(GetPrimaryFormat(node->GetOpDesc()->GetOutputDescPtr(0)->GetFormat()),
                  ge::FORMAT_NC1HWC0);
      }
    }
    if (node->GetType() == "Cast") {
      {
        ge::GeShape shape = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        EXPECT_EQ(shape.GetDims(), current5_d_dim_of_e);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetDataType(),
                  ge::DT_FLOAT16);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetFormat(),
                  ge::FORMAT_NC1HWC0);
      }
      {
        ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        EXPECT_EQ(shape.GetDims(), current5_d_dim_of_e);
        EXPECT_EQ(node->GetOpDesc()->GetOutputDescPtr(0)->GetDataType(),
                  ge::DT_FLOAT);
        EXPECT_EQ(node->GetOpDesc()->GetOutputDescPtr(0)->GetFormat(),
                  ge::FORMAT_NC1HWC0);
      }

    }
  }
  EXPECT_EQ(count_node, 7);
}

/* | FORMAT_NCHW(fp32)->FORMAT_NC1HWC0 (fp32)
 * The Program will insert Transdata->Cast->Reshape ops.
 * Reshape Failed because Dimension H/W is not 1*/
TEST_F(UTEST_FE_TRANSOP_INSERT_COMPLEX_BY_ORIGINAL_FORMAT, InsertAllTransopByOriginalFormat_03) {
// src:cce op, dst:cce op
  vector<int64_t> current5_d_dim = {1, 4, 128, 1, 16};
  vector<int64_t> current4_d_dim = {1, 128, 1, 64};
  vector<int64_t> original_dim = {64, 128};
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  GeTensorDesc src_tensor_desc(GeShape(current4_d_dim),
                               ge::FORMAT_NHWC, ge::DT_FLOAT16);
  src_tensor_desc.SetOriginShape(GeShape(original_dim));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);
  ge::AttrUtils::SetInt(src_op, FE_IMPLY_TYPE, 6);

  vector<int64_t> current5_d_dim_of_e = {64, 8, 1, 1,16};
  vector<int64_t> current4_d_dim_of_e = {64, 128, 1, 1};
  OpDescPtr dst_op = std::make_shared<OpDesc>("E", "E");
  GeTensorDesc dst_tensor_desc(GeShape(current5_d_dim_of_e), ge::FORMAT_NC1HWC0, ge::DT_FLOAT);
  dst_tensor_desc.SetOriginShape(GeShape(original_dim));
  dst_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  dst_op->AddInputDesc(dst_tensor_desc);
  dst_op->AddOutputDesc(dst_tensor_desc);
  auto dst_node = graph->AddNode(dst_op);
  ge::AttrUtils::SetInt(dst_op, FE_IMPLY_TYPE, 6);
  vector<bool> input_const_vector = {false};
  dst_op->SetIsInputConst(input_const_vector);

  GraphUtils::AddEdge(src_node->GetOutDataAnchor(0),
                      dst_node->GetInDataAnchor(0));
  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status status = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));

  int count_node = 0;
  ASSERT_EQ(fe::SUCCESS, status);
  for (auto node : graph->GetDirectNode()) {
    ASSERT_NE(node, nullptr);
    count_node++;
    if (node->GetType() == "TransData" && count_node == 2) {
      {
        ge::GeShape shape = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        EXPECT_EQ(shape.GetDims(), current4_d_dim);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetDataType(),
                  ge::DT_FLOAT16);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetFormat(),
                  ge::FORMAT_NHWC);
      }
      {
        vector<int64_t> current4_d_dim_nch_w = {1, 64, 128, 1};
        ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        EXPECT_EQ(shape.GetDims(), current4_d_dim_nch_w);
        EXPECT_EQ(node->GetOpDesc()->GetOutputDescPtr(0)->GetDataType(),
                  ge::DT_FLOAT16);
        EXPECT_EQ(node->GetOpDesc()->GetOutputDescPtr(0)->GetFormat(),
                  ge::FORMAT_NCHW);
      }
    }
    if (node->GetType() == fe::UNSQUEEZE_V2) {
      {
        ge::GeShape shape = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        EXPECT_EQ(shape.GetDims(), original_dim);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetDataType(),
                  ge::DT_FLOAT16);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetFormat(),
                  ge::FORMAT_NCHW);
      }
      {
        ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        EXPECT_EQ(shape.GetDims(), current4_d_dim_of_e);
        EXPECT_EQ(node->GetOpDesc()->GetOutputDescPtr(0)->GetDataType(),
                  ge::DT_FLOAT16);
        EXPECT_EQ(node->GetOpDesc()->GetOutputDescPtr(0)->GetFormat(),
                  ge::FORMAT_NCHW);
      }
    }
    if (node->GetType() == "TransData" && count_node == 4) {
      {
        ge::GeShape shape = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        EXPECT_EQ(shape.GetDims(), current4_d_dim_of_e);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetDataType(),
                  ge::DT_FLOAT16);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetFormat(),
                  ge::FORMAT_NCHW);
      }
      {

        ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        EXPECT_EQ(shape.GetDims(), current5_d_dim_of_e);
        EXPECT_EQ(node->GetOpDesc()->GetOutputDescPtr(0)->GetDataType(),
                  ge::DT_FLOAT16);
        EXPECT_EQ(GetPrimaryFormat(node->GetOpDesc()->GetOutputDescPtr(0)->GetFormat()),
                  ge::FORMAT_NC1HWC0);
      }
    }
    if (node->GetType() == "Cast") {
      {
        ge::GeShape shape = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        EXPECT_EQ(shape.GetDims(), current5_d_dim_of_e);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetDataType(),
                  ge::DT_FLOAT16);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetFormat(),
                  ge::FORMAT_NC1HWC0);
      }
      {
        ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        EXPECT_EQ(shape.GetDims(), current5_d_dim_of_e);
        EXPECT_EQ(node->GetOpDesc()->GetOutputDescPtr(0)->GetDataType(),
                  ge::DT_FLOAT);
        EXPECT_EQ(node->GetOpDesc()->GetOutputDescPtr(0)->GetFormat(),
                  ge::FORMAT_NC1HWC0);
      }
    }
  }
  EXPECT_EQ(count_node, 7);
}


/* | FORMAT_HWCN(fp32)->FORMAT_NC1HWC0 (fp32)
 * The Program will insert Transdata->Cast->Reshape ops.
 * Reshape Failed because Dimension H/W is not 1*/
TEST_F(UTEST_FE_TRANSOP_INSERT_COMPLEX_BY_ORIGINAL_FORMAT, InsertAllTransopByOriginalFormat_04) {
// src:cce op, dst:cce op
  vector<int64_t> current5_d_dim = {1, 4, 128, 1, 16};
  vector<int64_t> current4_d_dim = {1, 128, 1, 64};
  vector<int64_t> original_dim = {64, 128};
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr src_op = std::make_shared<OpDesc>("A", OP_TYPE_PLACE_HOLDER);
  GeTensorDesc src_tensor_desc(GeShape(current4_d_dim),
                               ge::FORMAT_NHWC, ge::DT_FLOAT16);
  src_tensor_desc.SetOriginShape(GeShape(original_dim));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NHWC);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);
  ge::AttrUtils::SetInt(src_op, FE_IMPLY_TYPE, 6);

  vector<int64_t> current_fz_dim_of_e = {4, 8, 16, 16};
  vector<int64_t> current4_d_dim_of_e = {1, 1, 64, 128};
  OpDescPtr dst_op = std::make_shared<OpDesc>("E", "E");
  GeTensorDesc dst_tensor_desc(GeShape(current_fz_dim_of_e), ge::FORMAT_FRACTAL_Z, ge::DT_FLOAT);
  dst_tensor_desc.SetOriginShape(GeShape(original_dim));
  dst_tensor_desc.SetOriginFormat(ge::FORMAT_HWCN);
  dst_op->AddInputDesc(dst_tensor_desc);
  dst_op->AddOutputDesc(dst_tensor_desc);
  auto dst_node = graph->AddNode(dst_op);
  ge::AttrUtils::SetInt(dst_op, FE_IMPLY_TYPE, 6);
  vector<bool> input_const_vector = {false};
  dst_op->SetIsInputConst(input_const_vector);

  GraphUtils::AddEdge(src_node->GetOutDataAnchor(0),
                      dst_node->GetInDataAnchor(0));
  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status status = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));

  int count_node = 0;
  ASSERT_EQ(fe::SUCCESS, status);
  for (auto node : graph->GetDirectNode()) {
    ASSERT_NE(node, nullptr);
    count_node++;
    if (node->GetType() == "TransData" && count_node == 3) {
      {
        ge::GeShape shape = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        EXPECT_EQ(shape.GetDims(), current4_d_dim_of_e);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetDataType(),
                  ge::DT_FLOAT16);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetFormat(),
                  ge::FORMAT_HWCN);
      }
      {
        vector<int64_t> current4_d_dim_nch_w = {1, 64, 128, 1};
        ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        EXPECT_EQ(shape.GetDims(), current_fz_dim_of_e);
        EXPECT_EQ(node->GetOpDesc()->GetOutputDescPtr(0)->GetDataType(),
                  ge::DT_FLOAT16);
        EXPECT_EQ(node->GetOpDesc()->GetOutputDescPtr(0)->GetFormat(),
                  ge::FORMAT_FRACTAL_Z);
      }
    }

    if (node->GetType() == "Cast") {
      {
        ge::GeShape shape = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        EXPECT_EQ(shape.GetDims(), current_fz_dim_of_e);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetDataType(),
                  ge::DT_FLOAT16);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetFormat(),
                  ge::FORMAT_FRACTAL_Z);
      }
      {
        ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        EXPECT_EQ(shape.GetDims(), current_fz_dim_of_e);
        EXPECT_EQ(node->GetOpDesc()->GetOutputDescPtr(0)->GetDataType(),
                  ge::DT_FLOAT);
        EXPECT_EQ(node->GetOpDesc()->GetOutputDescPtr(0)->GetFormat(),
                  ge::FORMAT_FRACTAL_Z);
      }
    }
  }
  EXPECT_EQ(count_node, 6);
}


/* | FORMAT_HWCN(fp32)->FORMAT_NC1HWC0 (fp32)
 * The Program will insert Transdata->Cast->Reshape ops.
 * Reshape Failed because Dimension H/W is not 1*/
TEST_F(UTEST_FE_TRANSOP_INSERT_COMPLEX_BY_ORIGINAL_FORMAT, InsertAllTransopByOriginalFormat_05) {
// src:cce op, dst:cce op
  vector<int64_t> current5_d_dim = {1, 4, 128, 1, 16};
  vector<int64_t> current4_d_dim = {1, 128, 1, 64};
  vector<int64_t> original_dim = {64, 128};
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr src_op = std::make_shared<OpDesc>("A", OP_TYPE_PLACE_HOLDER);
  GeTensorDesc src_tensor_desc(GeShape(current4_d_dim),
                               ge::FORMAT_NHWC, ge::DT_FLOAT16);
  src_tensor_desc.SetOriginShape(GeShape(original_dim));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NHWC);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);
  ge::AttrUtils::SetInt(src_op, FE_IMPLY_TYPE, 6);
  ge::AttrUtils::SetStr(src_op, PARENT_OP_TYPE, "Variable");

  vector<int64_t> current_fz_dim_of_e = {4, 8, 16, 16};
  vector<int64_t> current4_d_dim_of_e = {1, 1, 64, 128};
  OpDescPtr dst_op = std::make_shared<OpDesc>("E", "E");
  GeTensorDesc dst_tensor_desc(GeShape(current_fz_dim_of_e), ge::FORMAT_FRACTAL_Z, ge::DT_FLOAT);
  dst_tensor_desc.SetOriginShape(GeShape(original_dim));
  dst_tensor_desc.SetOriginFormat(ge::FORMAT_HWCN);
  dst_op->AddInputDesc(dst_tensor_desc);
  dst_op->AddOutputDesc(dst_tensor_desc);
  auto dst_node = graph->AddNode(dst_op);
  ge::AttrUtils::SetInt(dst_op, FE_IMPLY_TYPE, 6);
  vector<bool> input_const_vector = {false};
  dst_op->SetIsInputConst(input_const_vector);


  GraphUtils::AddEdge(src_node->GetOutDataAnchor(0),
                      dst_node->GetInDataAnchor(0));
  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status status = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));

  int count_node = 0;
  ASSERT_EQ(fe::SUCCESS, status);
  for (auto node : graph->GetDirectNode()) {
    ASSERT_NE(node, nullptr);
    count_node++;
    if (node->GetType() == "TransData" && count_node == 3) {
      {
        ge::GeShape shape = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        EXPECT_EQ(shape.GetDims(), current4_d_dim_of_e);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetDataType(),
                  ge::DT_FLOAT16);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetFormat(),
                  ge::FORMAT_HWCN);
      }
      {
        vector<int64_t> current4_d_dim_nch_w = {1, 64, 128, 1};
        ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        EXPECT_EQ(shape.GetDims(), current_fz_dim_of_e);
        EXPECT_EQ(node->GetOpDesc()->GetOutputDescPtr(0)->GetDataType(),
                  ge::DT_FLOAT16);
        EXPECT_EQ(node->GetOpDesc()->GetOutputDescPtr(0)->GetFormat(),
                  ge::FORMAT_FRACTAL_Z);
      }
    }

    if (node->GetType() == "Cast") {
      {
        ge::GeShape shape = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        EXPECT_EQ(shape.GetDims(), current_fz_dim_of_e);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetDataType(),
                  ge::DT_FLOAT16);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetFormat(),
                  ge::FORMAT_FRACTAL_Z);
      }
      {
        ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        EXPECT_EQ(shape.GetDims(), current_fz_dim_of_e);
        EXPECT_EQ(node->GetOpDesc()->GetOutputDescPtr(0)->GetDataType(),
                  ge::DT_FLOAT);
        EXPECT_EQ(node->GetOpDesc()->GetOutputDescPtr(0)->GetFormat(),
                  ge::FORMAT_FRACTAL_Z);
      }
    }
  }
  EXPECT_EQ(count_node, 6);
}
