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
#include "adapter/common/op_store_adapter_manager.h"
#include "adapter/tbe_adapter/tbe_op_store_adapter.h"
#include "ops_kernel_store/fe_ops_kernel_info_store.h"
#include "graph_optimizer/fusion_common/fusion_pass_manager.h"
#include "graph_optimizer/graph_fusion/graph_fusion.h"
#include "graph_optimizer/fe_graph_optimizer.h"
#include "graph_optimizer/shape_format_transfer/trans_node_manager/trans_node_manager.h"
#include "graph_optimizer/shape_format_transfer/trans_node_implementation/trans_node_base_generator.h"
#include "graph_optimizer/shape_format_transfer/trans_node_implementation/trans_node_cast_generator.h"
#include "graph_optimizer/shape_format_transfer/trans_node_implementation/trans_node_reshape_generator.h"
#include "graph_optimizer/shape_format_transfer/trans_node_implementation/trans_node_transpose_generator.h"
#include "graph_optimizer/shape_format_transfer/trans_node_implementation/trans_node_transdata_generator.h"
#include "ops_store/ops_kernel_manager.h"
#include "graph/operator_factory_impl.h"
#undef protected
#undef private

#include <iostream>

using namespace std;
using namespace ge;
using namespace fe;

class UTEST_FE_TRANSOP_INSERT_COMPLEX : public testing::Test {
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
};

/* NC1HWC0(fp16) -> NC1HWC0(fp32)-> NHWC (fp32)
 * The Program will insert Cast->Transdata ops. */
TEST_F(UTEST_FE_TRANSOP_INSERT_COMPLEX, InsertAllTransop_01) {
  // src:cce op, dst:cce op
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  GeTensorDesc src_tensor_desc(GeShape({100, 8, 256, 512, 16}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  src_tensor_desc.SetOriginShape(GeShape({100, 128, 256, 512}));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);
  ge::AttrUtils::SetInt(src_op, FE_IMPLY_TYPE, 6);

  OpDescPtr dst_op = std::make_shared<OpDesc>("D", "D");
  GeTensorDesc dst_tensor_desc(GeShape({100, 256, 512, 128}), ge::FORMAT_NHWC, ge::DT_FLOAT);
  dst_tensor_desc.SetOriginShape(GeShape({100, 128, 256, 512}));
  dst_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  dst_op->AddInputDesc(dst_tensor_desc);
  dst_op->AddOutputDesc(dst_tensor_desc);
  auto dst_node = graph->AddNode(dst_op);
  ge::AttrUtils::SetInt(dst_op, FE_IMPLY_TYPE, 6);
  vector<bool> input_const_vector = {false};
  dst_op->SetIsInputConst(input_const_vector);

  GraphUtils::AddEdge(src_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(0));
  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status status = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));

  int count_node = 0;
  ASSERT_EQ(fe::SUCCESS, status);
  for (auto node : graph->GetDirectNode()) {
    ASSERT_NE(node, nullptr);
    count_node++;
    if (node->GetType() == "TransData") {
      {
        ge::GeShape shape = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        ASSERT_EQ(shape.GetDimNum(), 5);
        EXPECT_EQ(shape.GetDims()[0], 100);
        EXPECT_EQ(shape.GetDims()[1], 8);
        EXPECT_EQ(shape.GetDims()[2], 256);
        EXPECT_EQ(shape.GetDims()[3], 512);
        EXPECT_EQ(shape.GetDims()[4], 16);
      }
      {
        ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        ASSERT_EQ(shape.GetDimNum(), 4);
        EXPECT_EQ(shape.GetDims()[0], 100);
        EXPECT_EQ(shape.GetDims()[1], 256);
        EXPECT_EQ(shape.GetDims()[2], 512);
        EXPECT_EQ(shape.GetDims()[3], 128);
      }
    }
    if (node->GetType() == "Cast") {
      ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
      ASSERT_EQ(shape.GetDimNum(), 4);
      EXPECT_EQ(shape.GetDims()[0], 100);
      EXPECT_EQ(shape.GetDims()[1], 256);
      EXPECT_EQ(shape.GetDims()[2], 512);
      EXPECT_EQ(shape.GetDims()[3], 128);
      EXPECT_EQ(node->GetOpDesc()->GetOutputDescPtr(0)->GetDataType(), ge::DT_FLOAT);
    }
    if (node->GetType() == "D") {
      {
        ge::GeShape shape = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        ASSERT_EQ(shape.GetDimNum(), 4);
        EXPECT_EQ(shape.GetDims()[0], 100);
        EXPECT_EQ(shape.GetDims()[1], 256);
        EXPECT_EQ(shape.GetDims()[2], 512);
        EXPECT_EQ(shape.GetDims()[3], 128);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetDataType(), ge::DT_FLOAT);
      }
      {
        ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        ASSERT_EQ(shape.GetDimNum(), 4);
        EXPECT_EQ(shape.GetDims()[0], 100);
        EXPECT_EQ(shape.GetDims()[1], 256);
        EXPECT_EQ(shape.GetDims()[2], 512);
        EXPECT_EQ(shape.GetDims()[3], 128);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetDataType(), ge::DT_FLOAT);
      }

    }
  }
  EXPECT_EQ(count_node, 4);
}

/* NC1HWC0(fp16) -> NC1HWC0(fp32)-> NHWC (fp32)
 * The Program will insert Cast->Transdata ops.
 * Orignal format of A is NHWC*/
TEST_F(UTEST_FE_TRANSOP_INSERT_COMPLEX, InsertAllTransop_01_2) {
  // src:cce op, dst:cce op
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  GeTensorDesc src_tensor_desc(GeShape({100, 128, 256, 512, 32}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  src_tensor_desc.SetOriginShape(GeShape({100, 30, 256, 512}));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NHWC);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);
  ge::AttrUtils::SetInt(src_op, FE_IMPLY_TYPE, 6);

  OpDescPtr dst_op = std::make_shared<OpDesc>("D", "D");
  GeTensorDesc dst_tensor_desc(GeShape({1, 3, 4, 2}), ge::FORMAT_NHWC, ge::DT_FLOAT);
  dst_tensor_desc.SetOriginShape(GeShape({1, 3, 4, 2}));
  dst_tensor_desc.SetOriginFormat(ge::FORMAT_NHWC);
  dst_op->AddInputDesc(dst_tensor_desc);
  dst_op->AddOutputDesc(dst_tensor_desc);
  auto dst_node = graph->AddNode(dst_op);
  ge::AttrUtils::SetInt(dst_op, FE_IMPLY_TYPE, 6);
  vector<bool> input_const_vector = {false};
  dst_op->SetIsInputConst(input_const_vector);

  GraphUtils::AddEdge(src_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(0));
  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status status = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));

  int count_node = 0;
  ASSERT_EQ(fe::SUCCESS, status);
  for (auto node : graph->GetDirectNode()) {
    ASSERT_NE(node, nullptr);
    count_node++;
    if (node->GetType() == "TransData") {
      {
        ge::GeShape shape = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        ASSERT_EQ(shape.GetDimNum(), 5);
        EXPECT_EQ(shape.GetDims()[0], 100);
        EXPECT_EQ(shape.GetDims()[1], 128);
        EXPECT_EQ(shape.GetDims()[2], 256);
        EXPECT_EQ(shape.GetDims()[3], 512);
        EXPECT_EQ(shape.GetDims()[4], 32);
      }
      {
        ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        ASSERT_EQ(shape.GetDimNum(), 4);
        EXPECT_EQ(shape.GetDims()[0], 100);
        EXPECT_EQ(shape.GetDims()[1], 30);
        EXPECT_EQ(shape.GetDims()[2], 256);
        EXPECT_EQ(shape.GetDims()[3], 512);
      }
    }
    if (node->GetType() == "Cast") {
      ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
      ASSERT_EQ(shape.GetDimNum(), 4);
      EXPECT_EQ(shape.GetDims()[0], 1);
      EXPECT_EQ(shape.GetDims()[1], 3);
      EXPECT_EQ(shape.GetDims()[2], 4);
      EXPECT_EQ(shape.GetDims()[3], 2);
      EXPECT_EQ(node->GetOpDesc()->GetOutputDescPtr(0)->GetDataType(), ge::DT_FLOAT);
    }
    if (node->GetType() == "D") {
      {
        ge::GeShape shape = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        ASSERT_EQ(shape.GetDimNum(), 4);
        EXPECT_EQ(shape.GetDims()[0], 1);
        EXPECT_EQ(shape.GetDims()[1], 3);
        EXPECT_EQ(shape.GetDims()[2], 4);
        EXPECT_EQ(shape.GetDims()[3], 2);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetDataType(), ge::DT_FLOAT);
      }
      {
        ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        ASSERT_EQ(shape.GetDimNum(), 4);
        EXPECT_EQ(shape.GetDims()[0], 1);
        EXPECT_EQ(shape.GetDims()[1], 3);
        EXPECT_EQ(shape.GetDims()[2], 4);
        EXPECT_EQ(shape.GetDims()[3], 2);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetDataType(), ge::DT_FLOAT);
      }

    }
  }
  EXPECT_EQ(count_node, 4);
}

/* NC1HWC0(fp16) -> NC1HWC0(fp32)-> NHWC (fp32)
 * The Program will insert Cast->Transdata ops.
 * Orignal format of A is NHWC, And it's Tbe Op*/
TEST_F(UTEST_FE_TRANSOP_INSERT_COMPLEX, InsertAllTransop_01_3) {
  // src:cce op, dst:cce op
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  GeTensorDesc src_tensor_desc(GeShape({100, 8, 256, 512, 16}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  src_tensor_desc.SetOriginShape(GeShape({100, 30, 256, 512}));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NHWC);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);
  ge::AttrUtils::SetInt(src_op, FE_IMPLY_TYPE, 6);

  OpDescPtr dst_op = std::make_shared<OpDesc>("D", "D");
  GeTensorDesc dst_tensor_desc(GeShape({100, 128, 256, 512}), ge::FORMAT_NCHW, ge::DT_FLOAT);
  dst_tensor_desc.SetOriginShape(GeShape({100, 30, 256, 512}));
  dst_tensor_desc.SetOriginFormat(ge::FORMAT_NHWC);
  dst_op->AddInputDesc(dst_tensor_desc);
  dst_op->AddOutputDesc(dst_tensor_desc);
  auto dst_node = graph->AddNode(dst_op);
  ge::AttrUtils::SetInt(dst_op, FE_IMPLY_TYPE, 6);
  vector<bool> input_const_vector = {false};
  dst_op->SetIsInputConst(input_const_vector);

  GraphUtils::AddEdge(src_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(0));
  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status status = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));

  int count_node = 0;
  ASSERT_EQ(fe::SUCCESS, status);
  for (auto node : graph->GetDirectNode()) {
    ASSERT_NE(node, nullptr);
    count_node++;
    if (node->GetType() == "TransData") {
      {
        ge::GeShape shape = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        ASSERT_EQ(shape.GetDimNum(), 5);
        EXPECT_EQ(shape.GetDims()[0], 100);
        EXPECT_EQ(shape.GetDims()[1], 8);
        EXPECT_EQ(shape.GetDims()[2], 256);
        EXPECT_EQ(shape.GetDims()[3], 512);
        EXPECT_EQ(shape.GetDims()[4], 16);
      }
      {
        ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        ASSERT_EQ(shape.GetDimNum(), 4);
        EXPECT_EQ(shape.GetDims()[0], 100);
        EXPECT_EQ(shape.GetDims()[1], 512);
        EXPECT_EQ(shape.GetDims()[2], 30);
        EXPECT_EQ(shape.GetDims()[3], 256);
      }
    }
    if (node->GetType() == "Cast") {
      ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
      ASSERT_EQ(shape.GetDimNum(), 4);
      EXPECT_EQ(shape.GetDims()[0], 100);
      EXPECT_EQ(shape.GetDims()[1], 512);
      EXPECT_EQ(shape.GetDims()[2], 30);
      EXPECT_EQ(shape.GetDims()[3], 256);
      EXPECT_EQ(node->GetOpDesc()->GetOutputDescPtr(0)->GetDataType(), ge::DT_FLOAT);
    }
    if (node->GetType() == "D") {
      {
        ge::GeShape shape = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        ASSERT_EQ(shape.GetDimNum(), 4);
        EXPECT_EQ(shape.GetDims()[0], 100);
        EXPECT_EQ(shape.GetDims()[1], 128);
        EXPECT_EQ(shape.GetDims()[2], 256);
        EXPECT_EQ(shape.GetDims()[3], 512);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetDataType(), ge::DT_FLOAT);
      }
      {
        ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        ASSERT_EQ(shape.GetDimNum(), 4);
        EXPECT_EQ(shape.GetDims()[0], 100);
        EXPECT_EQ(shape.GetDims()[1], 128);
        EXPECT_EQ(shape.GetDims()[2], 256);
        EXPECT_EQ(shape.GetDims()[3], 512);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetDataType(), ge::DT_FLOAT);
      }
    }
  }
  EXPECT_EQ(count_node, 4);
}


/* NC1HWC0(fp16) -> NC1HWC0(fp32) -> NHWC (fp32)
 * The Program will insert Cast->Transdata ops.
 * Orignal format of A is NCHW, And it's Tbe Op*/
TEST_F(UTEST_FE_TRANSOP_INSERT_COMPLEX, InsertAllTransop_01_4) {
  // src:cce op, dst:cce op
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  GeTensorDesc src_tensor_desc(GeShape({32, 4, 109, 109,16}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  src_tensor_desc.SetOriginShape(GeShape({32, 64, 109, 109}));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);
  ge::AttrUtils::SetInt(src_op, FE_IMPLY_TYPE, 6);

  OpDescPtr dst_op = std::make_shared<OpDesc>("D", "D");
  GeTensorDesc dst_tensor_desc(GeShape({1, 3, 4, 2}), ge::FORMAT_NCHW, ge::DT_FLOAT);
  dst_tensor_desc.SetOriginShape(GeShape({1, 3, 4, 2}));
  dst_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  dst_op->AddInputDesc(dst_tensor_desc);
  dst_op->AddOutputDesc(dst_tensor_desc);
  auto dst_node = graph->AddNode(dst_op);
  ge::AttrUtils::SetInt(dst_op, FE_IMPLY_TYPE, 6);
  vector<bool> input_const_vector = {false};
  dst_op->SetIsInputConst(input_const_vector);

  GraphUtils::AddEdge(src_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(0));
  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status status = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));

  int count_node = 0;
  ASSERT_EQ(fe::SUCCESS, status);
  for (auto node : graph->GetDirectNode()) {
    ASSERT_NE(node, nullptr);
    count_node++;
    if (node->GetType() == "TransData") {
      {
        ge::GeShape shape = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        ASSERT_EQ(shape.GetDimNum(), 5);
        EXPECT_EQ(shape.GetDims()[0], 32);
        EXPECT_EQ(shape.GetDims()[1], 4);
        EXPECT_EQ(shape.GetDims()[2], 109);
        EXPECT_EQ(shape.GetDims()[3], 109);
        EXPECT_EQ(shape.GetDims()[4], 16);
      }
      {
        ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        ASSERT_EQ(shape.GetDimNum(), 4);
        EXPECT_EQ(shape.GetDims()[0], 32);
        EXPECT_EQ(shape.GetDims()[1], 64);
        EXPECT_EQ(shape.GetDims()[2], 109);
        EXPECT_EQ(shape.GetDims()[3], 109);

      }
    }
    if (node->GetType() == "Cast") {
      ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
      ASSERT_EQ(shape.GetDimNum(), 4);
      EXPECT_EQ(shape.GetDims()[0], 1);
      EXPECT_EQ(shape.GetDims()[1], 3);
      EXPECT_EQ(shape.GetDims()[2], 4);
      EXPECT_EQ(shape.GetDims()[3], 2);
      EXPECT_EQ(node->GetOpDesc()->GetOutputDescPtr(0)->GetDataType(), ge::DT_FLOAT);
    }
    if (node->GetType() == "D") {
      {
        ge::GeShape shape = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        ASSERT_EQ(shape.GetDimNum(), 4);
        EXPECT_EQ(shape.GetDims()[0], 1);
        EXPECT_EQ(shape.GetDims()[1], 3);
        EXPECT_EQ(shape.GetDims()[2], 4);
        EXPECT_EQ(shape.GetDims()[3], 2);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetDataType(), ge::DT_FLOAT);
      }
      {
        ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        ASSERT_EQ(shape.GetDimNum(), 4);
        EXPECT_EQ(shape.GetDims()[0], 1);
        EXPECT_EQ(shape.GetDims()[1], 3);
        EXPECT_EQ(shape.GetDims()[2], 4);
        EXPECT_EQ(shape.GetDims()[3], 2);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetDataType(), ge::DT_FLOAT);
      }

    }
  }
  EXPECT_EQ(count_node, 4);
}


/* NC1HWC0(fp16) -> NCHW(fp16) -> NHWC (fp32)
 * The Program will insert Transdata->Cast->Permute ops.
 * Orignal format of A is NCHW and original shape of A is 5D.
 * Transdata is Tbe Op*/
TEST_F(UTEST_FE_TRANSOP_INSERT_COMPLEX, InsertAllTransop_01_5) {
  // src:cce op, dst:cce op
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  GeTensorDesc src_tensor_desc(GeShape({32, 4, 109, 109, 16}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  src_tensor_desc.SetOriginShape(GeShape({32, 64, 109, 109}));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);
  ge::AttrUtils::SetInt(src_op, FE_IMPLY_TYPE, 6);

  OpDescPtr dst_op = std::make_shared<OpDesc>("D", "D");
  GeTensorDesc dst_tensor_desc(GeShape({1, 3, 4, 2}), ge::FORMAT_NCHW, ge::DT_FLOAT);
  dst_tensor_desc.SetOriginShape(GeShape({1, 3, 4, 2}));
  dst_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  dst_op->AddInputDesc(dst_tensor_desc);
  dst_op->AddOutputDesc(dst_tensor_desc);
  auto dst_node = graph->AddNode(dst_op);
  ge::AttrUtils::SetInt(dst_op, FE_IMPLY_TYPE, 6);
  vector<bool> input_const_vector = {false};
  dst_op->SetIsInputConst(input_const_vector);

  GraphUtils::AddEdge(src_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(0));
  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status status = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));

  int count_node = 0;
  ASSERT_EQ(fe::SUCCESS, status);
  for (auto node : graph->GetDirectNode()) {
    ASSERT_NE(node, nullptr);
    count_node++;
    if (node->GetType() == "TransData") {
      {
        ge::GeShape shape = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        ASSERT_EQ(shape.GetDimNum(), 5);
        EXPECT_EQ(shape.GetDims()[0], 32);
        EXPECT_EQ(shape.GetDims()[1], 4);
        EXPECT_EQ(shape.GetDims()[2], 109);
        EXPECT_EQ(shape.GetDims()[3], 109);
        EXPECT_EQ(shape.GetDims()[4], 16);
      }
      {
        ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        ASSERT_EQ(shape.GetDimNum(), 4);
        EXPECT_EQ(shape.GetDims()[0], 32);
        EXPECT_EQ(shape.GetDims()[1], 64);
        EXPECT_EQ(shape.GetDims()[2], 109);
        EXPECT_EQ(shape.GetDims()[3], 109);

      }
    }
    if (node->GetType() == "Cast") {
      ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
      ASSERT_EQ(shape.GetDimNum(), 4);
      EXPECT_EQ(shape.GetDims()[0], 1);
      EXPECT_EQ(shape.GetDims()[1], 3);
      EXPECT_EQ(shape.GetDims()[2], 4);
      EXPECT_EQ(shape.GetDims()[3], 2);
      EXPECT_EQ(node->GetOpDesc()->GetOutputDescPtr(0)->GetDataType(), ge::DT_FLOAT);
    }
    if (node->GetType() == "D") {
      {
        ge::GeShape shape = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        ASSERT_EQ(shape.GetDimNum(), 4);
        EXPECT_EQ(shape.GetDims()[0], 1);
        EXPECT_EQ(shape.GetDims()[1], 3);
        EXPECT_EQ(shape.GetDims()[2], 4);
        EXPECT_EQ(shape.GetDims()[3], 2);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetDataType(), ge::DT_FLOAT);
      }
      {
        ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        ASSERT_EQ(shape.GetDimNum(), 4);
        EXPECT_EQ(shape.GetDims()[0], 1);
        EXPECT_EQ(shape.GetDims()[1], 3);
        EXPECT_EQ(shape.GetDims()[2], 4);
        EXPECT_EQ(shape.GetDims()[3], 2);
        EXPECT_EQ(node->GetOpDesc()->GetOutputDescPtr(0)->GetDataType(), ge::DT_FLOAT);
      }

    }
  }
  EXPECT_EQ(count_node, 4);
}

/* NC1HWC0(fp16) -> NCHW(fp16) -> NCHW (fp32)
 * The Program will insert Transdata->Cast->Reshape ops.
 * Reshape Failed because Dimension H/W is not 1*/
TEST_F(UTEST_FE_TRANSOP_INSERT_COMPLEX, InsertAllTransop_02) {
  // src:cce op, dst:cce op
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  GeTensorDesc src_tensor_desc(GeShape({100, 256, 256, 512}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  src_tensor_desc.SetOriginShape(GeShape({100, 256, 256, 512}));
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);
  ge::AttrUtils::SetInt(src_op, FE_IMPLY_TYPE, 6);

  OpDescPtr dst_op = std::make_shared<OpDesc>("EE", "EE");
  GeTensorDesc dst_tensor_desc(GeShape({1, 3}), ge::FORMAT_NCHW, ge::DT_FLOAT);
  dst_tensor_desc.SetOriginShape(GeShape({100, 256, 256, 512}));
  dst_op->AddInputDesc(dst_tensor_desc);
  dst_op->AddOutputDesc(dst_tensor_desc);
  auto dst_node = graph->AddNode(dst_op);
  ge::AttrUtils::SetInt(dst_op, FE_IMPLY_TYPE, 6);
  vector<bool> input_const_vector = {false};
  dst_op->SetIsInputConst(input_const_vector);

  GraphUtils::AddEdge(src_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(0));
  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status status = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));

  int count_node = 0;
  ASSERT_EQ(fe::SUCCESS, status);
  for (auto node : graph->GetDirectNode()) {
    ASSERT_NE(node, nullptr);
    count_node++;
    if (node->GetType() == fe::SQUEEZE_V2) {
      vector<int64_t> input_dim = {100,256,256,512};
      vector<int64_t> output_dim = {1,3};
      {
        ge::GeShape shape = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        ASSERT_EQ(shape.GetDimNum(), 4);
        EXPECT_EQ(shape.GetDims(), input_dim);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetDataType(), ge::DT_FLOAT16);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetFormat(), ge::FORMAT_NCHW);
      }
      {
        ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        ASSERT_EQ(shape.GetDimNum(), 2);
        EXPECT_EQ(shape.GetDims(), output_dim);
        EXPECT_EQ(node->GetOpDesc()->GetOutputDescPtr(0)->GetDataType(), ge::DT_FLOAT16);
        EXPECT_EQ(node->GetOpDesc()->GetOutputDescPtr(0)->GetFormat(), ge::FORMAT_NCHW);
      }

    }
  }
  EXPECT_EQ(count_node, 5);
}

/* NC1HWC0(fp16) -> NCHW(fp16) -> NCHW (fp32)
 * The Program will insert Transdata->Cast->Reshape ops.
 * Reshape successfully.*/
TEST_F(UTEST_FE_TRANSOP_INSERT_COMPLEX, InsertAllTransop_03) {
  // src:cce op, dst:cce op
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  GeTensorDesc src_tensor_desc(GeShape({100, 1, 1, 258}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  src_tensor_desc.SetOriginShape(GeShape({100, 1, 1, 258}));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);
  ge::AttrUtils::SetInt(src_op, FE_IMPLY_TYPE, 6);

  OpDescPtr dst_op = std::make_shared<OpDesc>("FF", "FF");
  GeTensorDesc dst_tensor_desc(GeShape({1, 3}), ge::FORMAT_NCHW, ge::DT_FLOAT); //Reshape type nw
  dst_tensor_desc.SetOriginShape(GeShape({100, 1, 1, 258}));
  dst_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  dst_op->AddInputDesc(dst_tensor_desc);
  dst_op->AddOutputDesc(dst_tensor_desc);
  auto dst_node = graph->AddNode(dst_op);
  ge::AttrUtils::SetInt(dst_op, FE_IMPLY_TYPE, 6);
  vector<bool> input_const_vector = {false};
  dst_op->SetIsInputConst(input_const_vector);

  GraphUtils::AddEdge(src_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(0));
  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status status = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));

  int count_node = 0;
  ASSERT_EQ(fe::SUCCESS, status);
  for (auto node : graph->GetDirectNode()) {
    ASSERT_NE(node, nullptr);
    count_node++;
    if (node->GetType() == "TransData") {
      ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
      ASSERT_EQ(shape.GetDimNum(), 4);
      EXPECT_EQ(shape.GetDims()[0], 100);
      EXPECT_EQ(shape.GetDims()[1], 1);
      EXPECT_EQ(shape.GetDims()[2], 1);
      EXPECT_EQ(shape.GetDims()[3], 258);
    }
    if (node->GetType() == fe::SQUEEZE_V2) {
      ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
      ASSERT_EQ(shape.GetDimNum(), 2);
      EXPECT_EQ(shape.GetDims()[0], 1);
      EXPECT_EQ(shape.GetDims()[1], 3);
      EXPECT_EQ(node->GetOpDesc()->GetOutputDescPtr(0)->GetDataType(), ge::DT_FLOAT16);
      EXPECT_EQ(node->GetOpDesc()->GetOutputDescPtr(0)->GetFormat(), ge::FORMAT_NCHW);
    }
    if (node->GetType() == "Cast") {
      ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
      ASSERT_EQ(shape.GetDimNum(), 2);
      EXPECT_EQ(shape.GetDims()[0], 1);
      EXPECT_EQ(shape.GetDims()[1], 3);
      EXPECT_EQ(node->GetOpDesc()->GetOutputDescPtr(0)->GetDataType(), ge::DT_FLOAT);
    }
    if (node->GetType() == "F") {
      {
        ge::GeShape shape = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        ASSERT_EQ(shape.GetDimNum(), 2);
        EXPECT_EQ(shape.GetDims()[0], 1);
        EXPECT_EQ(shape.GetDims()[1], 3);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetDataType(), ge::DT_FLOAT);
      }

      {
        ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        ASSERT_EQ(shape.GetDimNum(), 2);
        EXPECT_EQ(shape.GetDims()[0], 1);
        EXPECT_EQ(shape.GetDims()[1], 3);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetDataType(), ge::DT_FLOAT);
      }

    }
  }
  EXPECT_EQ(count_node, 5);
}

/* Fractal_Z(fp16) -> Fractal_Z(fp32) -> NHWC (fp32)
 * The Program will insert Cast->Transdata ops.*/
TEST_F(UTEST_FE_TRANSOP_INSERT_COMPLEX, InsertAllTransop_06) {
  // src:cce op, dst:cce op
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  GeTensorDesc src_tensor_desc(GeShape({1, 1, 16, 16}), ge::FORMAT_FRACTAL_Z, ge::DT_FLOAT16);
  src_tensor_desc.SetOriginShape(GeShape({3, 4, 1, 2}));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);
  ge::AttrUtils::SetInt(src_op, FE_IMPLY_TYPE, 6);

  OpDescPtr dst_op = std::make_shared<OpDesc>("D", "D");
  GeTensorDesc dst_tensor_desc(GeShape({1, 3, 4, 2}), ge::FORMAT_NHWC, ge::DT_FLOAT);
  dst_tensor_desc.SetOriginShape(GeShape({1, 3, 4, 2}));
  dst_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  dst_op->AddInputDesc(dst_tensor_desc);
  dst_op->AddOutputDesc(dst_tensor_desc);
  auto dst_node = graph->AddNode(dst_op);
  ge::AttrUtils::SetInt(dst_op, FE_IMPLY_TYPE, 6);
  vector<bool> input_const_vector = {false};
  dst_op->SetIsInputConst(input_const_vector);

  GraphUtils::AddEdge(src_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(0));
  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status status = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));

  int count_node = 0;
  ASSERT_EQ(fe::SUCCESS, status);
  for (auto node : graph->GetDirectNode()) {
    ASSERT_NE(node, nullptr);
    count_node++;
    if (node->GetType() == "TransData") {
      printf("countNode %d...\n", count_node);
      ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
      ASSERT_EQ(shape.GetDimNum(), 4);
      EXPECT_EQ(shape.GetDims()[0], 3);
      EXPECT_EQ(shape.GetDims()[1], 4);
      EXPECT_EQ(shape.GetDims()[2], 1);
      EXPECT_EQ(shape.GetDims()[3], 2);
    }
    if (node->GetType() == "Cast") {
      ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
      ASSERT_EQ(shape.GetDimNum(), 4);
      EXPECT_EQ(shape.GetDims()[0], 1);
      EXPECT_EQ(shape.GetDims()[1], 4);
      EXPECT_EQ(shape.GetDims()[2], 2);
      EXPECT_EQ(shape.GetDims()[3], 3);
      EXPECT_EQ(node->GetOpDesc()->GetOutputDescPtr(0)->GetDataType(), ge::DT_FLOAT);
    }

    if (node->GetType() == "D") {
      {
        ge::GeShape shape = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        ASSERT_EQ(shape.GetDimNum(), 4);
        EXPECT_EQ(shape.GetDims()[0], 1);
        EXPECT_EQ(shape.GetDims()[1], 3);
        EXPECT_EQ(shape.GetDims()[2], 4);
        EXPECT_EQ(shape.GetDims()[3], 2);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetDataType(), ge::DT_FLOAT);
      }
      {
        ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        ASSERT_EQ(shape.GetDimNum(), 4);
        EXPECT_EQ(shape.GetDims()[0], 1);
        EXPECT_EQ(shape.GetDims()[1], 3);
        EXPECT_EQ(shape.GetDims()[2], 4);
        EXPECT_EQ(shape.GetDims()[3], 2);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetDataType(), ge::DT_FLOAT);
      }

    }
  }
  EXPECT_EQ(count_node, 6);
}


/* FRACTAL_Z(fp16) -> NCHW(fp16) -> NCHW (fp32)
 * The Program will insert Transdata->Cast->Reshape ops.
 * Reshape Failed because Dimension H/W is not 1*/
TEST_F(UTEST_FE_TRANSOP_INSERT_COMPLEX, InsertAllTransop_07) {
  // src:cce op, dst:cce op
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  GeTensorDesc src_tensor_desc(GeShape({1, 1, 16, 16}), ge::FORMAT_FRACTAL_Z, ge::DT_FLOAT16);
  src_tensor_desc.SetOriginShape(GeShape({3, 4, 1, 2}));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);
  ge::AttrUtils::SetInt(src_op, FE_IMPLY_TYPE, 6);

  OpDescPtr dst_op = std::make_shared<OpDesc>("EE", "EE");
  GeTensorDesc dst_tensor_desc(GeShape({1, 3}), ge::FORMAT_NCHW, ge::DT_FLOAT);
  dst_tensor_desc.SetOriginShape(GeShape({1, 3}));
  dst_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  dst_op->AddInputDesc(dst_tensor_desc);
  dst_op->AddOutputDesc(dst_tensor_desc);
  auto dst_node = graph->AddNode(dst_op);
  ge::AttrUtils::SetInt(dst_op, FE_IMPLY_TYPE, 6);
  vector<bool> input_const_vector = {false};
  dst_op->SetIsInputConst(input_const_vector);

  GraphUtils::AddEdge(src_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(0));
  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status status = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));

  int count_node = 0;
  ASSERT_EQ(fe::SUCCESS, status);
  for (auto node : graph->GetDirectNode()) {
    ASSERT_NE(node, nullptr);
    count_node++;
  }
  EXPECT_EQ(count_node, 4);
}


/* FRACTAL_Z(fp16) -> FRACTAL_Z(fp32) -> NCHW (fp32)
 * The Program will insert Cast->Transdata ops.
 * Reshape successfully.*/
TEST_F(UTEST_FE_TRANSOP_INSERT_COMPLEX, InsertAllTransop_08) {
  // src:cce op, dst:cce op
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  GeTensorDesc src_tensor_desc(GeShape({1, 1, 16, 16}), ge::FORMAT_FRACTAL_Z, ge::DT_FLOAT16);
  src_tensor_desc.SetOriginShape(GeShape({3, 4, 1, 2}));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);
  ge::AttrUtils::SetInt(src_op, FE_IMPLY_TYPE, 6);

  OpDescPtr dst_op = std::make_shared<OpDesc>("FF", "FF");
  GeTensorDesc dst_tensor_desc(GeShape({1, 3}), ge::FORMAT_NCHW, ge::DT_FLOAT); //Reshape type nw
  dst_tensor_desc.SetOriginShape(GeShape({1, 3}));
  dst_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  dst_op->AddInputDesc(dst_tensor_desc);
  dst_op->AddOutputDesc(dst_tensor_desc);
  auto dst_node = graph->AddNode(dst_op);
  ge::AttrUtils::SetInt(dst_op, FE_IMPLY_TYPE, 6);
  vector<bool> input_const_vector = {false};
  dst_op->SetIsInputConst(input_const_vector);

  GraphUtils::AddEdge(src_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(0));
  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status status = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));

  int count_node = 0;
  ASSERT_EQ(fe::SUCCESS, status);
  for (auto node : graph->GetDirectNode()) {
    ASSERT_NE(node, nullptr);
    count_node++;
    if (node->GetType() == "TransData") {
      ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
      ASSERT_EQ(shape.GetDimNum(), 4);
      EXPECT_EQ(shape.GetDims()[0], 3);
      EXPECT_EQ(shape.GetDims()[1], 4);
      EXPECT_EQ(shape.GetDims()[2], 1);
      EXPECT_EQ(shape.GetDims()[3], 2);
    }
    if (node->GetType() == "Cast") {
      ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
      ASSERT_EQ(shape.GetDimNum(), 2);
      EXPECT_EQ(shape.GetDims()[0], 1);
      EXPECT_EQ(shape.GetDims()[1], 3);
      EXPECT_EQ(node->GetOpDesc()->GetOutputDescPtr(0)->GetDataType(), ge::DT_FLOAT);
    }

    if (node->GetType() == "F") {
      {
        ge::GeShape shape = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        ASSERT_EQ(shape.GetDimNum(), 2);
        EXPECT_EQ(shape.GetDims()[0], 1);
        EXPECT_EQ(shape.GetDims()[1], 3);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetDataType(), ge::DT_FLOAT);
      }

      {
        ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        ASSERT_EQ(shape.GetDimNum(), 2);
        EXPECT_EQ(shape.GetDims()[0], 1);
        EXPECT_EQ(shape.GetDims()[1], 3);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetDataType(), ge::DT_FLOAT);
      }

    }
  }
  EXPECT_EQ(count_node, 4);
}

/* Fractal_Z(fp16) -> HWCN(fp16)
 * The Program will insert Transdata ops.*/
TEST_F(UTEST_FE_TRANSOP_INSERT_COMPLEX, InsertAllTransop_11) {
  // src:cce op, dst:cce op
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  GeTensorDesc src_tensor_desc(GeShape({100, 256, 128, 512}), ge::FORMAT_FRACTAL_Z, ge::DT_FLOAT16);
  src_tensor_desc.SetOriginShape(GeShape({3, 4, 1, 2}));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);
  ge::AttrUtils::SetInt(src_op, FE_IMPLY_TYPE, 6);

  OpDescPtr dst_op = std::make_shared<OpDesc>("D", "D");
  GeTensorDesc dst_tensor_desc(GeShape({1, 3, 4, 2}), ge::FORMAT_HWCN, ge::DT_FLOAT16);
  dst_tensor_desc.SetOriginShape(GeShape({3, 4, 1, 2}));
  dst_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  dst_op->AddInputDesc(dst_tensor_desc);
  dst_op->AddOutputDesc(dst_tensor_desc);
  auto dst_node = graph->AddNode(dst_op);
  ge::AttrUtils::SetInt(dst_op, FE_IMPLY_TYPE, 6);
  vector<bool> input_const_vector = {false};
  dst_op->SetIsInputConst(input_const_vector);

  GraphUtils::AddEdge(src_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(0));
  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status status = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));

  int count_node = 0;
  ASSERT_EQ(fe::SUCCESS, status);
  for (auto node : graph->GetDirectNode()) {
    ASSERT_NE(node, nullptr);
    count_node++;
    if (node->GetType() == "TransData") {
      {
        ge::GeShape shape = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        ge::Format format_in = node->GetOpDesc()->GetInputDescPtr(0)->GetFormat();
        ASSERT_EQ(shape.GetDimNum(), 4);
        EXPECT_EQ(shape.GetDims()[0], 2);
        EXPECT_EQ(shape.GetDims()[1], 1);
        EXPECT_EQ(shape.GetDims()[2], 16);
        EXPECT_EQ(shape.GetDims()[3], 16);
        EXPECT_EQ(format_in, ge::FORMAT_FRACTAL_Z);
      }
      {
        ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        ge::Format format_out = node->GetOpDesc()->GetOutputDescPtr(0)->GetFormat();
        ASSERT_EQ(shape.GetDimNum(), 4);
        EXPECT_EQ(shape.GetDims()[0], 1);
        EXPECT_EQ(shape.GetDims()[1], 2);
        EXPECT_EQ(shape.GetDims()[2], 4);
        EXPECT_EQ(shape.GetDims()[3], 3);
        EXPECT_EQ(format_out, ge::FORMAT_HWCN);
      }
    }

    if (node->GetType() == "D") {
      {
        ge::GeShape shape = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        ASSERT_EQ(shape.GetDimNum(), 4);
        EXPECT_EQ(shape.GetDims()[0], 1);
        EXPECT_EQ(shape.GetDims()[1], 3);
        EXPECT_EQ(shape.GetDims()[2], 4);
        EXPECT_EQ(shape.GetDims()[3], 2);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetDataType(), ge::DT_FLOAT16);
      }
      {
        ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        ASSERT_EQ(shape.GetDimNum(), 4);
        EXPECT_EQ(shape.GetDims()[0], 1);
        EXPECT_EQ(shape.GetDims()[1], 3);
        EXPECT_EQ(shape.GetDims()[2], 4);
        EXPECT_EQ(shape.GetDims()[3], 2);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetDataType(), ge::DT_FLOAT16);
      }

    }
  }
  EXPECT_EQ(count_node, 3);
}

/* HWCN(fp16) -> Fragz(fp16)
 * The Program will insert Transdata ops.*/
TEST_F(UTEST_FE_TRANSOP_INSERT_COMPLEX, InsertAllTransop_12) {
  // src:cce op, dst:cce op
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  GeTensorDesc src_tensor_desc(GeShape({100, 256, 128, 512}), ge::FORMAT_HWCN, ge::DT_FLOAT16);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);
  ge::AttrUtils::SetInt(src_op, FE_IMPLY_TYPE, 6);

  OpDescPtr dst_op = std::make_shared<OpDesc>("D", "D");
  GeTensorDesc dst_tensor_desc(GeShape({1, 3, 4, 2}), ge::FORMAT_FRACTAL_Z, ge::DT_FLOAT16);
  dst_op->AddInputDesc(dst_tensor_desc);
  dst_op->AddOutputDesc(dst_tensor_desc);
  auto dst_node = graph->AddNode(dst_op);
  ge::AttrUtils::SetInt(dst_op, FE_IMPLY_TYPE, 6);
  vector<bool> input_const_vector = {false};
  dst_op->SetIsInputConst(input_const_vector);

  GraphUtils::AddEdge(src_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(0));
  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status status = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));

  int count_node = 0;
  ASSERT_EQ(fe::SUCCESS, status);
  for (auto node : graph->GetDirectNode()) {
    ASSERT_NE(node, nullptr);
    count_node++;
    if (node->GetType() == "TransData") {
      {
        ge::GeShape shape = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        ge::Format format_in = node->GetOpDesc()->GetInputDescPtr(0)->GetFormat();
        ASSERT_EQ(shape.GetDimNum(), 4);
        EXPECT_EQ(shape.GetDims()[0], 100);
        EXPECT_EQ(shape.GetDims()[1], 256);
        EXPECT_EQ(shape.GetDims()[2], 128);
        EXPECT_EQ(shape.GetDims()[3], 512);
        EXPECT_EQ(format_in, ge::FORMAT_HWCN);
      }
      {
        ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        ge::Format format_out = static_cast<ge::Format>(GetPrimaryFormat(node->GetOpDesc()->GetOutputDescPtr(0)->GetFormat()));

        ASSERT_EQ(shape.GetDimNum(), 4);
        EXPECT_EQ(shape.GetDims()[0], 204800);
        EXPECT_EQ(shape.GetDims()[1], 32);
        EXPECT_EQ(shape.GetDims()[2], 16);
        EXPECT_EQ(shape.GetDims()[3], 16);
        EXPECT_EQ(format_out, ge::FORMAT_FRACTAL_Z);
      }
    }

    if (node->GetType() == "D") {
      {
        ge::GeShape shape = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        ASSERT_EQ(shape.GetDimNum(), 4);
        EXPECT_EQ(shape.GetDims()[0], 1);
        EXPECT_EQ(shape.GetDims()[1], 3);
        EXPECT_EQ(shape.GetDims()[2], 4);
        EXPECT_EQ(shape.GetDims()[3], 2);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetDataType(), ge::DT_FLOAT16);
      }
      {
        ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        ASSERT_EQ(shape.GetDimNum(), 4);
        EXPECT_EQ(shape.GetDims()[0], 1);
        EXPECT_EQ(shape.GetDims()[1], 3);
        EXPECT_EQ(shape.GetDims()[2], 4);
        EXPECT_EQ(shape.GetDims()[3], 2);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetDataType(), ge::DT_FLOAT16);
      }
    }
  }
  EXPECT_EQ(count_node, 3);
}

/* FORMAT_NC1HWC0(fp16) -> FORMAT_NC1HWC0(fp32) -> NHWC (fp32)
 * The Program will insert Cast->Transdata ops.*/
TEST_F(UTEST_FE_TRANSOP_INSERT_COMPLEX, InsertAllTransop_13) {
  // src:cce op, dst:cce op
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  GeTensorDesc src_tensor_desc(GeShape({3, 4, 5, 6, 7}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  src_tensor_desc.SetOriginShape(GeShape({3, 4, 5, 6, 7}));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NC1HWC0);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);
  ge::AttrUtils::SetInt(src_op, FE_IMPLY_TYPE, 6);

  OpDescPtr dst_op = std::make_shared<OpDesc>("D", "D");
  GeTensorDesc dst_tensor_desc(GeShape({1, 3, 4, 2}), ge::FORMAT_NHWC, ge::DT_FLOAT);
  dst_tensor_desc.SetOriginShape(GeShape({3, 4, 5, 6, 7}));
  dst_tensor_desc.SetOriginFormat(ge::FORMAT_NC1HWC0);
  dst_op->AddInputDesc(dst_tensor_desc);
  dst_op->AddOutputDesc(dst_tensor_desc);
  auto dst_node = graph->AddNode(dst_op);
  ge::AttrUtils::SetInt(dst_op, FE_IMPLY_TYPE, 6);
  vector<bool> input_const_vector = {false};
  dst_op->SetIsInputConst(input_const_vector);

  GraphUtils::AddEdge(src_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(0));
  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status status = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));

  int count_node = 0;
  ASSERT_EQ(fe::SUCCESS, status);
  for (auto node : graph->GetDirectNode()) {
    ASSERT_NE(node, nullptr);
    count_node++;
    if (node->GetType() == "TransData") {
      ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
      EXPECT_EQ(shape.GetDimNum(), 5);
      EXPECT_EQ(shape.GetDims()[0], 3);
      EXPECT_EQ(shape.GetDims()[1], 4);
      EXPECT_EQ(shape.GetDims()[2], 5);
      EXPECT_EQ(shape.GetDims()[3], 6);
      EXPECT_EQ(shape.GetDims()[4], 7);
    }
    if (node->GetType() == "Cast") {
      ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
      EXPECT_EQ(shape.GetDimNum(), 5);
      EXPECT_EQ(shape.GetDims()[0], 3);
      EXPECT_EQ(shape.GetDims()[1], 4);
      EXPECT_EQ(shape.GetDims()[2], 5);
      EXPECT_EQ(shape.GetDims()[3], 6);
      EXPECT_EQ(shape.GetDims()[4], 7);
      EXPECT_EQ(node->GetOpDesc()->GetOutputDescPtr(0)->GetDataType(), ge::DT_FLOAT);
    }

    if (node->GetType() == "D") {
      {
        ge::GeShape shape = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        EXPECT_EQ(shape.GetDimNum(), 4);
        EXPECT_EQ(shape.GetDims()[0], 1);
        EXPECT_EQ(shape.GetDims()[1], 3);
        EXPECT_EQ(shape.GetDims()[2], 4);
        EXPECT_EQ(shape.GetDims()[3], 2);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetDataType(), ge::DT_FLOAT);
      }
      {
        ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        EXPECT_EQ(shape.GetDimNum(), 4);
        EXPECT_EQ(shape.GetDims()[0], 1);
        EXPECT_EQ(shape.GetDims()[1], 3);
        EXPECT_EQ(shape.GetDims()[2], 4);
        EXPECT_EQ(shape.GetDims()[3], 2);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetDataType(), ge::DT_FLOAT);
      }

    }
  }
  EXPECT_EQ(count_node, 4);
}

/* FORMAT_NC1HWC0(fp16) -> FORMAT_NC1HWC0(fp32) -> NHWC (fp32)
 * The Program will insert Cast->Permutecast->Transdata ops.*/
TEST_F(UTEST_FE_TRANSOP_INSERT_COMPLEX, InsertAllTransop_13_1) {
  // src:cce op, dst:cce op
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  GeTensorDesc src_tensor_desc(GeShape({3, 4, 5, 6, 7}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  src_tensor_desc.SetOriginShape(GeShape({3, 4, 5, 6, 7}));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NHWC);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);
  ge::AttrUtils::SetInt(src_op, FE_IMPLY_TYPE, 6);

  OpDescPtr dst_op = std::make_shared<OpDesc>("D", "D");
  GeTensorDesc dst_tensor_desc(GeShape({1, 3, 4, 2}), ge::FORMAT_NHWC, ge::DT_FLOAT);
  dst_tensor_desc.SetOriginShape(GeShape({1, 3, 4, 2}));
  dst_tensor_desc.SetOriginFormat(ge::FORMAT_NHWC);
  dst_op->AddInputDesc(dst_tensor_desc);
  dst_op->AddOutputDesc(dst_tensor_desc);
  auto dst_node = graph->AddNode(dst_op);
  ge::AttrUtils::SetInt(dst_op, FE_IMPLY_TYPE, 6);
  vector<bool> input_const_vector = {false};
  dst_op->SetIsInputConst(input_const_vector);

  GraphUtils::AddEdge(src_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(0));
  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status status = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));

  int count_node = 0;
  ASSERT_EQ(fe::SUCCESS, status);
  for (auto node : graph->GetDirectNode()) {
    ASSERT_NE(node, nullptr);
    count_node++;
    if (node->GetType() == "TransData") {
      ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
      EXPECT_EQ(shape.GetDimNum(), 5);
      EXPECT_EQ(shape.GetDims()[0], 3);
      EXPECT_EQ(shape.GetDims()[1], 4);
      EXPECT_EQ(shape.GetDims()[2], 5);
      EXPECT_EQ(shape.GetDims()[3], 6);
      EXPECT_EQ(shape.GetDims()[4], 7);
    }
    if (node->GetType() == "Cast") {
      ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
      EXPECT_EQ(shape.GetDimNum(), 4);
      EXPECT_EQ(shape.GetDims()[0], 1);
      EXPECT_EQ(shape.GetDims()[1], 3);
      EXPECT_EQ(shape.GetDims()[2], 4);
      EXPECT_EQ(shape.GetDims()[3], 2);
      EXPECT_EQ(node->GetOpDesc()->GetOutputDescPtr(0)->GetDataType(), ge::DT_FLOAT);
    }

    if (node->GetType() == "D") {
      {
        ge::GeShape shape = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        EXPECT_EQ(shape.GetDimNum(), 4);
        EXPECT_EQ(shape.GetDims()[0], 1);
        EXPECT_EQ(shape.GetDims()[1], 3);
        EXPECT_EQ(shape.GetDims()[2], 4);
        EXPECT_EQ(shape.GetDims()[3], 2);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetDataType(), ge::DT_FLOAT);
      }
      {
        ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        EXPECT_EQ(shape.GetDimNum(), 4);
        EXPECT_EQ(shape.GetDims()[0], 1);
        EXPECT_EQ(shape.GetDims()[1], 3);
        EXPECT_EQ(shape.GetDims()[2], 4);
        EXPECT_EQ(shape.GetDims()[3], 2);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetDataType(), ge::DT_FLOAT);
      }

    }
  }
  EXPECT_EQ(count_node, 4);
}

/* FORMAT_NC1HWC0(fp16) -> FORMAT_NC1HWC0(fp32) -> NHWC (fp32)
 * The Program will insert Cast->Transdata ops.*/
TEST_F(UTEST_FE_TRANSOP_INSERT_COMPLEX, InsertAllTransop_13_2) {
  // src:cce op, dst:cce op
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  GeTensorDesc src_tensor_desc(GeShape({3, 4, 5, 6, 7}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  src_tensor_desc.SetOriginShape(GeShape({3, 4, 5, 6, 7}));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NHWC);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);
  ge::AttrUtils::SetInt(src_op, FE_IMPLY_TYPE, 6);

  OpDescPtr dst_op = std::make_shared<OpDesc>("D", "D");
  GeTensorDesc dst_tensor_desc(GeShape({1, 3, 4, 2}), ge::FORMAT_NCHW, ge::DT_FLOAT);
  dst_tensor_desc.SetOriginShape(GeShape({3, 4, 5, 6, 7}));
  dst_tensor_desc.SetOriginFormat(ge::FORMAT_NHWC);
  dst_op->AddInputDesc(dst_tensor_desc);
  dst_op->AddOutputDesc(dst_tensor_desc);
  auto dst_node = graph->AddNode(dst_op);
  ge::AttrUtils::SetInt(dst_op, FE_IMPLY_TYPE, 6);
  vector<bool> input_const_vector = {false};
  dst_op->SetIsInputConst(input_const_vector);

  GraphUtils::AddEdge(src_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(0));
  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status status = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));

  int count_node = 0;
  ASSERT_EQ(fe::SUCCESS, status);
  for (auto node : graph->GetDirectNode()) {
    ASSERT_NE(node, nullptr);
    count_node++;
    if (node->GetType() == "TransData") {
      ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
      EXPECT_EQ(shape.GetDimNum(), 5);
      EXPECT_EQ(shape.GetDims()[0], 3);
      EXPECT_EQ(shape.GetDims()[1], 4);
      EXPECT_EQ(shape.GetDims()[2], 5);
      EXPECT_EQ(shape.GetDims()[3], 6);
      EXPECT_EQ(shape.GetDims()[4], 7);
    }
    if (node->GetType() == "Cast") {
      ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
      EXPECT_EQ(shape.GetDimNum(), 5);
      EXPECT_EQ(shape.GetDims()[0], 3);
      EXPECT_EQ(shape.GetDims()[1], 4);
      EXPECT_EQ(shape.GetDims()[2], 5);
      EXPECT_EQ(shape.GetDims()[3], 6);
      EXPECT_EQ(shape.GetDims()[4], 7);
      EXPECT_EQ(node->GetOpDesc()->GetOutputDescPtr(0)->GetDataType(), ge::DT_FLOAT);
    }

    if (node->GetType() == "D") {
      {
        ge::GeShape shape = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        EXPECT_EQ(shape.GetDimNum(), 4);
        EXPECT_EQ(shape.GetDims()[0], 1);
        EXPECT_EQ(shape.GetDims()[1], 3);
        EXPECT_EQ(shape.GetDims()[2], 4);
        EXPECT_EQ(shape.GetDims()[3], 2);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetDataType(), ge::DT_FLOAT);
      }
      {
        ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        EXPECT_EQ(shape.GetDimNum(), 4);
        EXPECT_EQ(shape.GetDims()[0], 1);
        EXPECT_EQ(shape.GetDims()[1], 3);
        EXPECT_EQ(shape.GetDims()[2], 4);
        EXPECT_EQ(shape.GetDims()[3], 2);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetDataType(), ge::DT_FLOAT);
      }

    }
  }
  EXPECT_EQ(count_node, 4);
}


/* NC1HWC0(fp16) -> HWCN(fp16) -> HWCN (fp32)
 * The Program will insert Transdata->Cast->Reshape ops.
 * Reshape successfully.*/
TEST_F(UTEST_FE_TRANSOP_INSERT_COMPLEX, InsertAllTransop_14) {
  // src:cce op, dst:cce op
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  vector<int64_t> dim_a = {100, 1, 1, 258};
  GeTensorDesc src_tensor_desc(GeShape(dim_a), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  src_tensor_desc.SetOriginShape(GeShape(dim_a));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);
  ge::AttrUtils::SetInt(src_op, FE_IMPLY_TYPE, 6);
  vector<int64_t> dim_f = {1,3};
  OpDescPtr dst_op = std::make_shared<OpDesc>("FF", "FF");
  GeTensorDesc dst_tensor_desc(GeShape(dim_f), ge::FORMAT_HWCN, ge::DT_FLOAT); //Reshape type nw
  dst_tensor_desc.SetOriginShape(GeShape(dim_a));
  dst_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  dst_op->AddInputDesc(dst_tensor_desc);
  dst_op->AddOutputDesc(dst_tensor_desc);
  auto dst_node = graph->AddNode(dst_op);
  ge::AttrUtils::SetInt(dst_op, FE_IMPLY_TYPE, 6);
  vector<bool> input_const_vector = {false};
  dst_op->SetIsInputConst(input_const_vector);

  GraphUtils::AddEdge(src_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(0));
  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status status = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));

  int count_node = 0;
  ASSERT_EQ(fe::SUCCESS, status);
  for (auto node : graph->GetDirectNode()) {
    ASSERT_NE(node, nullptr);
    count_node++;
    if (node->GetType() == "TransData") {
      ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
      ASSERT_EQ(shape.GetDimNum(), 4);
      EXPECT_EQ(shape.GetDims()[0], 100);
      EXPECT_EQ(shape.GetDims()[1], 1);
      EXPECT_EQ(shape.GetDims()[2], 1);
      EXPECT_EQ(shape.GetDims()[3], 258);
      EXPECT_EQ(ge::FORMAT_NCHW, node->GetOpDesc()->GetOutputDescPtr(0)->GetFormat());

    }
    if (node->GetType() == TRANSPOSE) {
      ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
      ASSERT_EQ(shape.GetDimNum(), 4);
      EXPECT_EQ(shape.GetDims()[0], 1);
      EXPECT_EQ(shape.GetDims()[1], 258);
      EXPECT_EQ(shape.GetDims()[2], 1);
      EXPECT_EQ(shape.GetDims()[3], 100);
      EXPECT_EQ(ge::FORMAT_HWCN, node->GetOpDesc()->GetOutputDescPtr(0)->GetFormat());

    }
    if (node->GetType() == "Cast") {
      ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
      ASSERT_EQ(shape.GetDimNum(), 2);
      EXPECT_EQ(shape.GetDims(), dim_f);
      EXPECT_EQ(node->GetOpDesc()->GetOutputDescPtr(0)->GetDataType(), ge::DT_FLOAT);
      EXPECT_EQ(ge::FORMAT_HWCN, node->GetOpDesc()->GetOutputDescPtr(0)->GetFormat());
    }
    if (node->GetType() == fe::SQUEEZE_V2) {
      ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
      ASSERT_EQ(shape.GetDimNum(), 2);
      EXPECT_EQ(shape.GetDims()[0], 1);
      EXPECT_EQ(shape.GetDims()[1], 3);
      EXPECT_EQ(node->GetOpDesc()->GetOutputDescPtr(0)->GetDataType(), ge::DT_FLOAT16);
      EXPECT_EQ(node->GetOpDesc()->GetOutputDescPtr(0)->GetFormat(), ge::FORMAT_HWCN);
    }
    if (node->GetType() == "F") {
      {
        ge::GeShape shape = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        ASSERT_EQ(shape.GetDimNum(), 2);
        EXPECT_EQ(shape.GetDims()[0], 1);
        EXPECT_EQ(shape.GetDims()[1], 3);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetDataType(), ge::DT_FLOAT);
      }

      {
        ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        ASSERT_EQ(shape.GetDimNum(), 2);
        EXPECT_EQ(shape.GetDims()[0], 1);
        EXPECT_EQ(shape.GetDims()[1], 3);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetDataType(), ge::DT_FLOAT);
      }

    }
  }
  EXPECT_EQ(count_node, 7);
}

/*  HWCN(fp16) ->NCHW(fp16) -> NC1HWC0(fp16) ->  NC1HWC0(fp32)
 * The Program will insert Transpose->Transdata->Cast ops.
 * Reshape successfully.*/
TEST_F(UTEST_FE_TRANSOP_INSERT_COMPLEX, InsertAllTransop_14_1) {
  // src:cce op, dst:cce op
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  GeTensorDesc src_tensor_desc(GeShape({100, 1, 2, 258}), ge::FORMAT_HWCN, ge::DT_FLOAT16);
  src_tensor_desc.SetOriginShape(GeShape({5, 6, 7, 8}));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);
  ge::AttrUtils::SetInt(src_op, FE_IMPLY_TYPE, 6);
  vector<int64_t> dims_f = {1,2,3,4};
  OpDescPtr dst_op = std::make_shared<OpDesc>("FF", "FF");
  GeTensorDesc dst_tensor_desc(GeShape(dims_f), ge::FORMAT_NC1HWC0, ge::DT_FLOAT); //Reshape type nw
  dst_tensor_desc.SetOriginShape(GeShape({5, 6, 7, 8}));
  dst_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  dst_op->AddInputDesc(dst_tensor_desc);
  dst_op->AddOutputDesc(dst_tensor_desc);
  auto dst_node = graph->AddNode(dst_op);
  ge::AttrUtils::SetInt(dst_op, FE_IMPLY_TYPE, 6);
  vector<bool> input_const_vector = {false};
  dst_op->SetIsInputConst(input_const_vector);

  GraphUtils::AddEdge(src_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(0));
  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status status = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));

  int count_node = 0;
  ASSERT_EQ(fe::SUCCESS, status);
  for (auto node : graph->GetDirectNode()) {
    ASSERT_NE(node, nullptr);
    count_node++;
    if (node->GetType() == "TransData") {
      ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
      ASSERT_EQ(shape.GetDimNum(), 5);
      EXPECT_EQ(shape.GetDims()[0], 258);
      EXPECT_EQ(shape.GetDims()[1], 1);
      EXPECT_EQ(shape.GetDims()[2], 100);
      EXPECT_EQ(shape.GetDims()[3], 1);
      EXPECT_EQ(shape.GetDims()[4], 16);
      EXPECT_EQ(ge::FORMAT_NC1HWC0, GetPrimaryFormat(node->GetOpDesc()->GetOutputDescPtr(0)->GetFormat()));

    }
    if (node->GetType() == TRANSPOSE) {
      ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
      ASSERT_EQ(shape.GetDimNum(), 4);
      EXPECT_EQ(shape.GetDims()[0], 258);
      EXPECT_EQ(shape.GetDims()[1], 2);
      EXPECT_EQ(shape.GetDims()[2], 100);
      EXPECT_EQ(shape.GetDims()[3], 1);
      EXPECT_EQ(ge::FORMAT_NCHW, node->GetOpDesc()->GetOutputDescPtr(0)->GetFormat());

    }
    if (node->GetType() == "Cast") {
      ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
      EXPECT_EQ(shape.GetDimNum(), 5);
      EXPECT_EQ(shape.GetDims()[0], 258);
      EXPECT_EQ(shape.GetDims()[1], 1);
      EXPECT_EQ(shape.GetDims()[2], 100);
      EXPECT_EQ(shape.GetDims()[3], 1);
      EXPECT_EQ(node->GetOpDesc()->GetOutputDescPtr(0)->GetDataType(), ge::DT_FLOAT);
      EXPECT_EQ(ge::FORMAT_NC1HWC0, node->GetOpDesc()->GetOutputDescPtr(0)->GetFormat());
    }

    if (node->GetType() == "F") {
      {
        ge::GeShape shape = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        ASSERT_EQ(shape.GetDimNum(), 4);
        EXPECT_EQ(shape.GetDims(), dims_f);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetDataType(), ge::DT_FLOAT);
      }

      {
        ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        ASSERT_EQ(shape.GetDimNum(), 4);
        EXPECT_EQ(shape.GetDims(), dims_f);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetDataType(), ge::DT_FLOAT);
      }

    }
  }
  EXPECT_EQ(count_node, 6);
}

TEST_F(UTEST_FE_TRANSOP_INSERT_COMPLEX, InsertAllTransop_15) {
  // src:cce op, dst:cce op
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr src_op = std::make_shared<OpDesc>("var1", "Variable");
  vector<int64_t> dims4_d = {100,200,300,400};
  vector<int64_t> dimsfz = {380000,25,16,16};
  GeTensorDesc src_tensor_desc(GeShape(dimsfz), ge::FORMAT_FRACTAL_Z, ge::DT_FLOAT);
  src_tensor_desc.SetOriginShape(GeShape(dims4_d));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_HWCN);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);
  ge::AttrUtils::SetInt(src_op, FE_IMPLY_TYPE, -1);

  OpDescPtr trans_op_0 = std::make_shared<OpDesc>("transdata_0", "TransData");
  GeTensorDesc trans_in_tensor_desc(GeShape(dimsfz), ge::FORMAT_FRACTAL_Z, ge::DT_FLOAT);
  trans_in_tensor_desc.SetOriginShape(GeShape(dims4_d));
  trans_in_tensor_desc.SetOriginFormat(ge::FORMAT_HWCN);
  GeTensorDesc trans_out_tensor_desc(GeShape(dims4_d), ge::FORMAT_HWCN, ge::DT_FLOAT);
  trans_out_tensor_desc.SetOriginShape(GeShape(dims4_d));
  trans_out_tensor_desc.SetOriginFormat(ge::FORMAT_HWCN);
  trans_op_0->AddInputDesc(trans_in_tensor_desc);
  trans_op_0->AddOutputDesc(trans_out_tensor_desc);
  auto trans_node_0 = graph->AddNode(trans_op_0);
  ge::AttrUtils::SetInt(trans_op_0, FE_IMPLY_TYPE, 6);

  OpDescPtr apply_op = std::make_shared<OpDesc>("apply", "ApplyMomentum");
  apply_op->AddInputDesc(trans_in_tensor_desc);
  apply_op->AddOutputDesc(trans_in_tensor_desc);
  auto apply_node = graph->AddNode(apply_op);
  ge::AttrUtils::SetInt(apply_op, FE_IMPLY_TYPE, 6);

  OpDescPtr trans_op_1 = std::make_shared<OpDesc>("transdata_1", "TransData");
  trans_op_1->AddInputDesc(trans_out_tensor_desc);
  trans_op_1->AddOutputDesc(trans_in_tensor_desc);
  auto trans_node_1 = graph->AddNode(trans_op_1);
  ge::AttrUtils::SetInt(trans_op_1, FE_IMPLY_TYPE, 6);

  OpDescPtr dst_op = std::make_shared<OpDesc>("var2", "Variable");
  GeTensorDesc dst_tensor_desc(GeShape(dimsfz), ge::FORMAT_FRACTAL_Z, ge::DT_FLOAT);
  dst_tensor_desc.SetOriginShape(GeShape(dims4_d));
  dst_tensor_desc.SetOriginFormat(ge::FORMAT_HWCN);
  dst_op->AddOutputDesc(dst_tensor_desc);
  dst_op->AddInputDesc(dst_tensor_desc);
  auto dst_node = graph->AddNode(dst_op);
  ge::AttrUtils::SetInt(dst_op, FE_IMPLY_TYPE, -1);

  GraphUtils::AddEdge(src_node->GetOutDataAnchor(0), trans_node_0->GetInDataAnchor(0));
  GraphUtils::AddEdge(trans_node_0->GetOutDataAnchor(0), apply_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(apply_node->GetOutDataAnchor(0), trans_node_1->GetInDataAnchor(0));
  GraphUtils::AddEdge(trans_node_1->GetOutDataAnchor(0), dst_node->GetInDataAnchor(0));

  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status status = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));

  int count_node = 0;
  ASSERT_EQ(fe::SUCCESS, status);
  for (auto node : graph->GetDirectNode()) {
    ASSERT_NE(node, nullptr);
    count_node++;
    EXPECT_NE(node->GetName(),"transdata_0");
    EXPECT_NE(node->GetName(),"transdata_1");
    if (node->GetName() == "apply") {
      {
        ge::GeShape shape = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        ASSERT_EQ(shape.GetDimNum(), 4);
        EXPECT_EQ(shape.GetDims(), dimsfz);
        EXPECT_EQ(ge::FORMAT_FRACTAL_Z,
                  node->GetOpDesc()->GetInputDescPtr(0)->GetFormat());
      }
      {
        ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        ASSERT_EQ(shape.GetDimNum(), 4);
        EXPECT_EQ(shape.GetDims(), dimsfz);
        EXPECT_EQ(ge::FORMAT_FRACTAL_Z,
                  node->GetOpDesc()->GetOutputDescPtr(0)->GetFormat());
      }
    }
  }
  EXPECT_EQ(count_node, 3);
}


TEST_F(UTEST_FE_TRANSOP_INSERT_COMPLEX, InsertAllTransop_16) {
  // src:cce op, dst:cce op
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr src_op = std::make_shared<OpDesc>("var1", "Variable");
  vector<int64_t> dims4_d = {100,200,300,400};
  vector<int64_t> dimsfz = {380000,25,16,16};
  GeTensorDesc src_tensor_desc(GeShape(dimsfz), ge::FORMAT_FRACTAL_Z, ge::DT_FLOAT16);
  src_tensor_desc.SetOriginShape(GeShape(dims4_d));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_HWCN);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);
  ge::AttrUtils::SetInt(src_op, FE_IMPLY_TYPE, -1);

  OpDescPtr trans_op_0 = std::make_shared<OpDesc>("transdata_0", "TransData");
  GeTensorDesc trans_in_tensor_desc(GeShape(dimsfz), ge::FORMAT_FRACTAL_Z, ge::DT_FLOAT16);
  trans_in_tensor_desc.SetOriginShape(GeShape(dims4_d));
  trans_in_tensor_desc.SetOriginFormat(ge::FORMAT_HWCN);
  GeTensorDesc trans_out_tensor_desc(GeShape(dims4_d), ge::FORMAT_HWCN, ge::DT_FLOAT16);
  trans_out_tensor_desc.SetOriginShape(GeShape(dims4_d));
  trans_out_tensor_desc.SetOriginFormat(ge::FORMAT_HWCN);
  trans_op_0->AddInputDesc(trans_in_tensor_desc);
  trans_op_0->AddOutputDesc(trans_out_tensor_desc);
  auto trans_node_0 = graph->AddNode(trans_op_0);
  ge::AttrUtils::SetInt(trans_op_0, FE_IMPLY_TYPE, 6);

  OpDescPtr apply_op = std::make_shared<OpDesc>("apply", "ApplyMomentum");
  GeTensorDesc apply_tensor_desc(GeShape(dimsfz), ge::FORMAT_FRACTAL_Z, ge::DT_FLOAT);
  apply_tensor_desc.SetOriginShape(GeShape(dims4_d));
  apply_tensor_desc.SetOriginFormat(ge::FORMAT_HWCN);
  apply_op->AddInputDesc(apply_tensor_desc);
  apply_op->AddOutputDesc(apply_tensor_desc);
  auto apply_node = graph->AddNode(apply_op);
  ge::AttrUtils::SetInt(apply_op, FE_IMPLY_TYPE, 6);

  OpDescPtr trans_op_1 = std::make_shared<OpDesc>("transdata_1", "TransData");
  trans_op_1->AddInputDesc(trans_out_tensor_desc);
  trans_op_1->AddOutputDesc(trans_in_tensor_desc);
  auto trans_node_1 = graph->AddNode(trans_op_1);
  ge::AttrUtils::SetInt(trans_op_1, FE_IMPLY_TYPE, 6);

  OpDescPtr dst_op = std::make_shared<OpDesc>("var2", "Variable");
  GeTensorDesc dst_tensor_desc(GeShape(dimsfz), ge::FORMAT_FRACTAL_Z, ge::DT_FLOAT16);
  dst_tensor_desc.SetOriginShape(GeShape(dims4_d));
  dst_tensor_desc.SetOriginFormat(ge::FORMAT_HWCN);
  dst_op->AddOutputDesc(dst_tensor_desc);
  dst_op->AddInputDesc(dst_tensor_desc);
  auto dst_node = graph->AddNode(dst_op);
  ge::AttrUtils::SetInt(dst_op, FE_IMPLY_TYPE, -1);

  GraphUtils::AddEdge(src_node->GetOutDataAnchor(0), trans_node_0->GetInDataAnchor(0));
  GraphUtils::AddEdge(trans_node_0->GetOutDataAnchor(0), apply_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(apply_node->GetOutDataAnchor(0), trans_node_1->GetInDataAnchor(0));
  GraphUtils::AddEdge(trans_node_1->GetOutDataAnchor(0), dst_node->GetInDataAnchor(0));

  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status status = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));

  int count_node = 0;
  uint32_t index = 0;
  ASSERT_EQ(fe::SUCCESS, status);
  for (auto node : graph->GetDirectNode()) {
    ASSERT_NE(node, nullptr);
    count_node++;
    EXPECT_NE(node->GetName(),"transdata_0");
    EXPECT_NE(node->GetName(),"transdata_1");
    if (node->GetName() == "apply") {
      {
        ge::GeShape shape = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        ASSERT_EQ(shape.GetDimNum(), 4);
        EXPECT_EQ(shape.GetDims(), dimsfz);
        EXPECT_EQ(ge::FORMAT_FRACTAL_Z,
                  node->GetOpDesc()->GetInputDescPtr(0)->GetFormat());
        EXPECT_EQ(ge::DT_FLOAT,
                  node->GetOpDesc()->GetInputDescPtr(0)->GetDataType());
      }
      {
        ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        ASSERT_EQ(shape.GetDimNum(), 4);
        EXPECT_EQ(shape.GetDims(), dimsfz);
        EXPECT_EQ(ge::FORMAT_FRACTAL_Z,
                  node->GetOpDesc()->GetOutputDescPtr(0)->GetFormat());
        EXPECT_EQ(ge::DT_FLOAT,
                  node->GetOpDesc()->GetOutputDescPtr(0)->GetDataType());
      }
    }
    if (node->GetType() == "Cast") {
      if (index == 0) {
        {
          ge::GeShape shape = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
          ASSERT_EQ(shape.GetDimNum(), 4);
          EXPECT_EQ(shape.GetDims(), dimsfz);
          EXPECT_EQ(ge::FORMAT_FRACTAL_Z,
                    node->GetOpDesc()->GetInputDescPtr(0)->GetFormat());
          EXPECT_EQ(ge::DT_FLOAT16,
                    node->GetOpDesc()->GetInputDescPtr(0)->GetDataType());
        }
        {
          ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
          ASSERT_EQ(shape.GetDimNum(), 4);
          EXPECT_EQ(shape.GetDims(), dimsfz);
          EXPECT_EQ(ge::FORMAT_FRACTAL_Z,
                    node->GetOpDesc()->GetOutputDescPtr(0)->GetFormat());
          EXPECT_EQ(ge::DT_FLOAT,
                    node->GetOpDesc()->GetOutputDescPtr(0)->GetDataType());
        }
        index++;
      } else {
        {
          ge::GeShape shape = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
          ASSERT_EQ(shape.GetDimNum(), 4);
          EXPECT_EQ(shape.GetDims(), dimsfz);
          EXPECT_EQ(ge::FORMAT_FRACTAL_Z,
                    node->GetOpDesc()->GetInputDescPtr(0)->GetFormat());
          EXPECT_EQ(ge::DT_FLOAT,
                    node->GetOpDesc()->GetInputDescPtr(0)->GetDataType());
        }
        {
          ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
          ASSERT_EQ(shape.GetDimNum(), 4);
          EXPECT_EQ(shape.GetDims(), dimsfz);
          EXPECT_EQ(ge::FORMAT_FRACTAL_Z,
                    node->GetOpDesc()->GetOutputDescPtr(0)->GetFormat());
          EXPECT_EQ(ge::DT_FLOAT16,
                    node->GetOpDesc()->GetOutputDescPtr(0)->GetDataType());
        }
      }
    }
  }
  EXPECT_EQ(count_node, 5);
}

/* NCHW 2D(fp16) -> NC1HWC0 5D(fp16) -> NC1HWC0 5D(fp32)
 * The Program will insert Cast->TransData ops.
 * Shape of input of Cast will remain as {256,512}*/
TEST_F(UTEST_FE_TRANSOP_INSERT_COMPLEX, InsertAllTransop_Converage_03) {
  // src:cce op, dst:cce op
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr src_op = std::make_shared<OpDesc>("EE", "EE");
  GeTensorDesc src_tensor_desc(GeShape({256, 512}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);
  ge::AttrUtils::SetInt(src_op, FE_IMPLY_TYPE, 6);

  OpDescPtr dst_op = std::make_shared<OpDesc>("A", "A");
  GeTensorDesc dst_tensor_desc(GeShape({1, 256, 256, 512, 1}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT);
  dst_op->AddInputDesc(dst_tensor_desc);
  dst_op->AddOutputDesc(dst_tensor_desc);
  auto dst_node = graph->AddNode(dst_op);
  ge::AttrUtils::SetInt(dst_op, FE_IMPLY_TYPE, 6);
  vector<bool> input_const_vector = {false};
  dst_op->SetIsInputConst(input_const_vector);


  GraphUtils::AddEdge(src_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(0));
  /* Set trans_data op as tbe op */
  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status status = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));

  int count_node = 0;
  ASSERT_EQ(fe::SUCCESS, status);
  for (auto node : graph->GetDirectNode()) {
    ASSERT_NE(node, nullptr);
    count_node++;
    if (node->GetType() == fe::UNSQUEEZE_V2) {
      {
        ge::GeShape shape = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        ASSERT_EQ(shape.GetDimNum(), 2);
        EXPECT_EQ(shape.GetDims()[0], 256);
        EXPECT_EQ(shape.GetDims()[1], 512);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetDataType(),
                  ge::DT_FLOAT16);
      }
      {
        ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        ASSERT_EQ(shape.GetDimNum(), 4);
        EXPECT_EQ(shape.GetDims()[0], 1);
        EXPECT_EQ(shape.GetDims()[1], 256);
        EXPECT_EQ(shape.GetDims()[2], 512);
        EXPECT_EQ(shape.GetDims()[3], 1);
        EXPECT_EQ(node->GetOpDesc()->GetOutputDescPtr(0)->GetDataType(),
                  ge::DT_FLOAT16);
      }
    }
    if (node->GetType() == "TransData") {
      {
        ge::GeShape shape = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        ASSERT_EQ(shape.GetDimNum(), 4);
        EXPECT_EQ(shape.GetDims()[0], 1);
        EXPECT_EQ(shape.GetDims()[1], 256);
        EXPECT_EQ(shape.GetDims()[2], 512);
        EXPECT_EQ(shape.GetDims()[3], 1);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetDataType(), ge::DT_FLOAT16);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetFormat(), ge::FORMAT_NCHW);
      }
      {
        ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        ASSERT_EQ(shape.GetDimNum(), 5);
        EXPECT_EQ(shape.GetDims()[0], 1);
        EXPECT_EQ(shape.GetDims()[1], 16);
        EXPECT_EQ(shape.GetDims()[2], 512);
        EXPECT_EQ(shape.GetDims()[3], 1);
        EXPECT_EQ(shape.GetDims()[4], 16);
        EXPECT_EQ(GetPrimaryFormat(node->GetOpDesc()->GetOutputDescPtr(0)->GetFormat()), ge::FORMAT_NC1HWC0);
        EXPECT_EQ(node->GetOpDesc()->GetOutputDescPtr(0)->GetDataType(), ge::DT_FLOAT16);
      }
    }
    if (node->GetType() == "Cast") {
      {
        ge::GeShape shape = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        ASSERT_EQ(shape.GetDimNum(), 5);
        EXPECT_EQ(shape.GetDims()[0], 1);
        EXPECT_EQ(shape.GetDims()[1], 16);
        EXPECT_EQ(shape.GetDims()[2], 512);
        EXPECT_EQ(shape.GetDims()[3], 1);
        EXPECT_EQ(shape.GetDims()[4], 16);

        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetDataType(), ge::DT_FLOAT16);
      }
      {
        ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        ASSERT_EQ(shape.GetDimNum(), 5);
        EXPECT_EQ(shape.GetDims()[0], 1);
        EXPECT_EQ(shape.GetDims()[1], 16);
        EXPECT_EQ(shape.GetDims()[2], 512);
        EXPECT_EQ(shape.GetDims()[3], 1);
        EXPECT_EQ(shape.GetDims()[4], 16);
        EXPECT_EQ(node->GetOpDesc()->GetOutputDescPtr(0)->GetDataType(), ge::DT_FLOAT);
      }
    }
    if (node->GetType() == "A") {
      {
        ge::GeShape shape = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        ASSERT_EQ(shape.GetDimNum(), 5);
        EXPECT_EQ(shape.GetDims()[0], 1);
        EXPECT_EQ(shape.GetDims()[1], 256);
        EXPECT_EQ(shape.GetDims()[2], 256);
        EXPECT_EQ(shape.GetDims()[3], 512);
        EXPECT_EQ(shape.GetDims()[4], 1);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetDataType(), ge::DT_FLOAT);
      }
      {
        ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        ASSERT_EQ(shape.GetDimNum(), 5);
        EXPECT_EQ(shape.GetDims()[0], 1);
        EXPECT_EQ(shape.GetDims()[1], 256);
        EXPECT_EQ(shape.GetDims()[2], 256);
        EXPECT_EQ(shape.GetDims()[3], 512);
        EXPECT_EQ(shape.GetDims()[4], 1);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetDataType(), ge::DT_FLOAT);
      }

    }
  }
  EXPECT_EQ(count_node, 5);

}

/* NHWC(fp32) -> NC1HWC0(fp32) -> NC1HWC0(fp16)
 * The Program will insert TransData->Cas ops.
 * Shape of input of Cast will remain as {256,512}*/
TEST_F(UTEST_FE_TRANSOP_INSERT_COMPLEX, InsertAllTransop_Converage_04) {
  // src:cce op, dst:cce op
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr src_op = std::make_shared<OpDesc>("B", "B");
  GeTensorDesc src_tensor_desc(GeShape({256,512}), ge::FORMAT_NHWC, ge::DT_FLOAT16);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);
  ge::AttrUtils::SetInt(src_op, FE_IMPLY_TYPE, 6);

  OpDescPtr dst_op = std::make_shared<OpDesc>("A", "A");
  GeTensorDesc dst_tensor_desc(GeShape({1, 256, 256, 512, 1}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT);
  dst_op->AddInputDesc(dst_tensor_desc);
  dst_op->AddOutputDesc(dst_tensor_desc);
  auto dst_node = graph->AddNode(dst_op);
  ge::AttrUtils::SetInt(dst_op, FE_IMPLY_TYPE, 6);
  vector<bool> input_const_vector = {false};
  dst_op->SetIsInputConst(input_const_vector);

  GraphUtils::AddEdge(src_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(0));
  /* Set trans_data op as tbe op */
  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status status = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));

  int count_node = 0;
  ASSERT_EQ(fe::SUCCESS, status);
  for (auto node : graph->GetDirectNode()) {
    ASSERT_NE(node, nullptr);
    count_node++;
    if (node->GetType() == fe::UNSQUEEZE_V2) {
      {
        ge::GeShape shape = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        ASSERT_EQ(shape.GetDimNum(), 2);
        EXPECT_EQ(shape.GetDims()[0], 256);
        EXPECT_EQ(shape.GetDims()[1], 512);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetDataType(),
                  ge::DT_FLOAT16);
      }
      {
        ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        ASSERT_EQ(shape.GetDimNum(), 4);
        EXPECT_EQ(shape.GetDims()[0], 1);
        EXPECT_EQ(shape.GetDims()[1], 256);
        EXPECT_EQ(shape.GetDims()[2], 512);
        EXPECT_EQ(shape.GetDims()[3], 1);
        EXPECT_EQ(node->GetOpDesc()->GetOutputDescPtr(0)->GetDataType(),
                  ge::DT_FLOAT16);
      }
    }
    if (node->GetType() == "TransData") {
      {
        ge::GeShape shape = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        ASSERT_EQ(shape.GetDimNum(), 4);
        EXPECT_EQ(shape.GetDims()[0], 1);
        EXPECT_EQ(shape.GetDims()[1], 256);
        EXPECT_EQ(shape.GetDims()[2], 512);
        EXPECT_EQ(shape.GetDims()[3], 1);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetDataType(), ge::DT_FLOAT16);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetFormat(), ge::FORMAT_NHWC);
      }
      {
        ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        ASSERT_EQ(shape.GetDimNum(), 5);
        EXPECT_EQ(shape.GetDims()[0], 1);
        EXPECT_EQ(shape.GetDims()[1], 1);
        EXPECT_EQ(shape.GetDims()[2], 256);
        EXPECT_EQ(shape.GetDims()[3], 512);
        EXPECT_EQ(shape.GetDims()[4], 16);
        EXPECT_EQ(node->GetOpDesc()->GetOutputDescPtr(0)->GetDataType(), ge::DT_FLOAT16);
        EXPECT_EQ(GetPrimaryFormat(node->GetOpDesc()->GetOutputDescPtr(0)->GetFormat()), ge::FORMAT_NC1HWC0);
      }

    }
    if (node->GetType() == "Cast") {
      {
        ge::GeShape shape = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        ASSERT_EQ(shape.GetDimNum(), 5);
        EXPECT_EQ(shape.GetDims()[0], 1);
        EXPECT_EQ(shape.GetDims()[1], 1);
        EXPECT_EQ(shape.GetDims()[2], 256);
        EXPECT_EQ(shape.GetDims()[3], 512);
        EXPECT_EQ(shape.GetDims()[4], 16);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetDataType(), ge::DT_FLOAT16);
      }
      {
        ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        ASSERT_EQ(shape.GetDimNum(), 5);
        EXPECT_EQ(shape.GetDims()[0], 1);
        EXPECT_EQ(shape.GetDims()[1], 1);
        EXPECT_EQ(shape.GetDims()[2], 256);
        EXPECT_EQ(shape.GetDims()[3], 512);
        EXPECT_EQ(shape.GetDims()[4], 16);
        EXPECT_EQ(node->GetOpDesc()->GetOutputDescPtr(0)->GetDataType(), ge::DT_FLOAT);
      }
    }
    if (node->GetType() == "A") {
      {
        ge::GeShape shape = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        ASSERT_EQ(shape.GetDimNum(), 5);
        EXPECT_EQ(shape.GetDims()[0], 1);
        EXPECT_EQ(shape.GetDims()[1], 256);
        EXPECT_EQ(shape.GetDims()[2], 256);
        EXPECT_EQ(shape.GetDims()[3], 512);
        EXPECT_EQ(shape.GetDims()[4], 1);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetDataType(), ge::DT_FLOAT);
      }
      {
        ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        ASSERT_EQ(shape.GetDimNum(), 5);
        EXPECT_EQ(shape.GetDims()[0], 1);
        EXPECT_EQ(shape.GetDims()[1], 256);
        EXPECT_EQ(shape.GetDims()[2], 256);
        EXPECT_EQ(shape.GetDims()[3], 512);
        EXPECT_EQ(shape.GetDims()[4], 1);
        EXPECT_EQ(node->GetOpDesc()->GetOutputDescPtr(0)->GetDataType(), ge::DT_FLOAT);
      }

    }
  }
  EXPECT_EQ(count_node, 5);

}

/* NHWC(fp16) -> NC1HWC0(fp16) -> NC1HWC0(fp32)
 * The Program will insert TransData->Cast-> ops.*/
TEST_F(UTEST_FE_TRANSOP_INSERT_COMPLEX, InsertAllTransop_Converage_05) {
  // src:cce op, dst:cce op
  std::map<std::string, std::string> options;
  fe_ops_kernel_info_store_ptr_ = make_shared<fe::FEOpsKernelInfoStore>(fe::AI_CORE_NAME);
  FEOpsStoreInfo tbe_custom {
      2,
      "tbe-custom",
      EN_IMPL_CUSTOM_TBE,
      GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/tbe_custom_opinfo",
      ""};
  vector<FEOpsStoreInfo> store_info;
  store_info.emplace_back(tbe_custom);
  Configuration::Instance(fe::AI_CORE_NAME).ops_store_info_vector_ = (store_info);
  fe_ops_kernel_info_store_ptr_->Initialize(options);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr src_op = std::make_shared<OpDesc>("B", "B");
  GeTensorDesc src_tensor_desc(GeShape({256,512}), ge::FORMAT_NHWC, ge::DT_FLOAT16);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);
  ge::AttrUtils::SetInt(src_op, FE_IMPLY_TYPE, 6);

  OpDescPtr dst_op = std::make_shared<OpDesc>("A", "A");
  GeTensorDesc dst_tensor_desc(GeShape({1, 256, 256, 512, 1}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT);
  dst_op->AddInputDesc(dst_tensor_desc);
  dst_op->AddOutputDesc(dst_tensor_desc);
  auto dst_node = graph->AddNode(dst_op);
  ge::AttrUtils::SetInt(dst_op, FE_IMPLY_TYPE, 6);
  vector<bool> input_const_vector = {false};
  dst_op->SetIsInputConst(input_const_vector);

  GraphUtils::AddEdge(src_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(0));
  /* Set trans_data op as tbe op */
  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status status = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));

  int count_node = 0;
  ASSERT_EQ(fe::SUCCESS, status);
  for (auto node : graph->GetDirectNode()) {
    ASSERT_NE(node, nullptr);
    count_node++;
    if (node->GetType() == fe::UNSQUEEZE_V2) {
      {
        ge::GeShape shape = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        ASSERT_EQ(shape.GetDimNum(), 2);
        EXPECT_EQ(shape.GetDims()[0], 256);
        EXPECT_EQ(shape.GetDims()[1], 512);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetDataType(),
                  ge::DT_FLOAT16);
      }
      {
        ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        ASSERT_EQ(shape.GetDimNum(), 4);
        EXPECT_EQ(shape.GetDims()[0], 1);
        EXPECT_EQ(shape.GetDims()[1], 256);
        EXPECT_EQ(shape.GetDims()[2], 512);
        EXPECT_EQ(shape.GetDims()[3], 1);
        EXPECT_EQ(node->GetOpDesc()->GetOutputDescPtr(0)->GetDataType(),
                  ge::DT_FLOAT16);
      }
    }
    if (node->GetType() == "TransData") {
      {
        ge::GeShape shape = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        ASSERT_EQ(shape.GetDimNum(), 4);
        EXPECT_EQ(shape.GetDims()[0], 1);
        EXPECT_EQ(shape.GetDims()[1], 256);
        EXPECT_EQ(shape.GetDims()[2], 512);
        EXPECT_EQ(shape.GetDims()[3], 1);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetDataType(), ge::DT_FLOAT16);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetFormat(), ge::FORMAT_NHWC);
      }
      {
        ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        ASSERT_EQ(shape.GetDimNum(), 5);
        EXPECT_EQ(shape.GetDims()[0], 1);
        EXPECT_EQ(shape.GetDims()[1], 1);
        EXPECT_EQ(shape.GetDims()[2], 256);
        EXPECT_EQ(shape.GetDims()[3], 512);
        EXPECT_EQ(shape.GetDims()[4], 16);
        EXPECT_EQ(node->GetOpDesc()->GetOutputDescPtr(0)->GetDataType(), ge::DT_FLOAT16);
        EXPECT_EQ(GetPrimaryFormat(node->GetOpDesc()->GetOutputDescPtr(0)->GetFormat()), ge::FORMAT_NC1HWC0);
      }
    }
    if (node->GetType() == "Cast") {
      {
        ge::GeShape shape = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        ASSERT_EQ(shape.GetDimNum(), 5);
        EXPECT_EQ(shape.GetDims()[0], 1);
        EXPECT_EQ(shape.GetDims()[1], 1);
        EXPECT_EQ(shape.GetDims()[2], 256);
        EXPECT_EQ(shape.GetDims()[3], 512);
        EXPECT_EQ(shape.GetDims()[4], 16);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetDataType(), ge::DT_FLOAT16);
      }
      {
        ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        ASSERT_EQ(shape.GetDimNum(), 5);
        EXPECT_EQ(shape.GetDims()[0], 1);
        EXPECT_EQ(shape.GetDims()[1], 1);
        EXPECT_EQ(shape.GetDims()[2], 256);
        EXPECT_EQ(shape.GetDims()[3], 512);
        EXPECT_EQ(shape.GetDims()[4], 16);
        EXPECT_EQ(node->GetOpDesc()->GetOutputDescPtr(0)->GetDataType(), ge::DT_FLOAT);
      }
    }
    if (node->GetType() == "A") {
      {
        ge::GeShape shape = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        ASSERT_EQ(shape.GetDimNum(), 5);
        EXPECT_EQ(shape.GetDims()[0], 1);
        EXPECT_EQ(shape.GetDims()[1], 256);
        EXPECT_EQ(shape.GetDims()[2], 256);
        EXPECT_EQ(shape.GetDims()[3], 512);
        EXPECT_EQ(shape.GetDims()[4], 1);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetDataType(), ge::DT_FLOAT);
      }
      {
        ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        ASSERT_EQ(shape.GetDimNum(), 5);
        EXPECT_EQ(shape.GetDims()[0], 1);
        EXPECT_EQ(shape.GetDims()[1], 256);
        EXPECT_EQ(shape.GetDims()[2], 256);
        EXPECT_EQ(shape.GetDims()[3], 512);
        EXPECT_EQ(shape.GetDims()[4], 1);
        EXPECT_EQ(node->GetOpDesc()->GetOutputDescPtr(0)->GetDataType(), ge::DT_FLOAT);
      }

    }
  }
  EXPECT_EQ(count_node, 5);

}

/* NCHW 2D(fp16) -> NC1HWC0 2D(fp16) -> NC1HWC0 5D(fp32)
 * The Program will insert TransData->Cast->TransData ops.
 * This test case will not insert reshape op */
TEST_F(UTEST_FE_TRANSOP_INSERT_COMPLEX, InsertAllTransop_Converage_06) {
  // src:cce op, dst:cce op
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr src_op = std::make_shared<OpDesc>("B", "B");
  auto in_shape = GeShape({256,512});
  GeTensorDesc src_tensor_desc(in_shape, ge::FORMAT_NCHW, ge::DT_FLOAT16);
  src_tensor_desc.SetOriginShape(in_shape);
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);
  ge::AttrUtils::SetInt(src_op, FE_IMPLY_TYPE, 6);

  OpDescPtr dst_op = std::make_shared<OpDesc>("A", "A");
  auto out_shape = GeShape({1, 256, 256, 512, 1});
  GeTensorDesc dst_tensor_desc(out_shape, ge::FORMAT_NC1HWC0, ge::DT_FLOAT);
  dst_tensor_desc.SetOriginShape(in_shape);
  dst_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  dst_op->AddInputDesc(dst_tensor_desc);
  dst_op->AddOutputDesc(dst_tensor_desc);
  auto dst_node = graph->AddNode(dst_op);
  ge::AttrUtils::SetInt(dst_op, FE_IMPLY_TYPE, 6);
  vector<bool> input_const_vector = {false};
  dst_op->SetIsInputConst(input_const_vector);


  GraphUtils::AddEdge(src_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(0));
  /* Set trans_data op as tbe op */
  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status status = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));

  int count_node = 0;
  ASSERT_EQ(fe::SUCCESS, status);
  for (auto node : graph->GetDirectNode()) {
    ASSERT_NE(node, nullptr);
    count_node++;
    if (node->GetType() == fe::UNSQUEEZE_V2) {
      {
        ge::GeShape shape = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        ASSERT_EQ(shape.GetDimNum(), 2);
        EXPECT_EQ(shape.GetDims()[0], 256);
        EXPECT_EQ(shape.GetDims()[1], 512);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetDataType(),
                  ge::DT_FLOAT16);
      }
      {
        ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        EXPECT_EQ(shape.GetDims()[0], 1);
        EXPECT_EQ(shape.GetDims()[1], 256);
        EXPECT_EQ(shape.GetDims()[2], 512);
        EXPECT_EQ(shape.GetDims()[3], 1);
        EXPECT_EQ(node->GetOpDesc()->GetOutputDescPtr(0)->GetDataType(),
                  ge::DT_FLOAT16);

        ge::GeShape original_shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetOriginShape();
        ASSERT_EQ(original_shape.GetDimNum(), 2);
        EXPECT_EQ(original_shape.GetDims()[0], 256);
        EXPECT_EQ(original_shape.GetDims()[1], 512);
      }
    }
    if (node->GetType() == "TransData") {
      {
        ge::GeShape shape = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        ASSERT_EQ(shape.GetDimNum(), 4);
        EXPECT_EQ(shape.GetDims()[0], 1);
        EXPECT_EQ(shape.GetDims()[1], 256);
        EXPECT_EQ(shape.GetDims()[2], 512);
        EXPECT_EQ(shape.GetDims()[3], 1);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetDataType(), ge::DT_FLOAT16);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetFormat(), ge::FORMAT_NCHW);
      }
      {
        ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        ASSERT_EQ(shape.GetDimNum(), 5);
        EXPECT_EQ(shape.GetDims()[0], 1);
        EXPECT_EQ(shape.GetDims()[1], 16);
        EXPECT_EQ(shape.GetDims()[2], 512);
        EXPECT_EQ(shape.GetDims()[3], 1);
        EXPECT_EQ(shape.GetDims()[4], 16);
        EXPECT_EQ(GetPrimaryFormat(node->GetOpDesc()->GetOutputDescPtr(0)->GetFormat()), ge::FORMAT_NC1HWC0);
        EXPECT_EQ(node->GetOpDesc()->GetOutputDescPtr(0)->GetDataType(), ge::DT_FLOAT16);
      }
    }

    if (node->GetType() == "Cast") {
      {
        ge::GeShape shape = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        ASSERT_EQ(shape.GetDimNum(), 5);
        EXPECT_EQ(shape.GetDims()[0], 1);
        EXPECT_EQ(shape.GetDims()[1], 16);
        EXPECT_EQ(shape.GetDims()[2], 512);
        EXPECT_EQ(shape.GetDims()[3], 1);
        EXPECT_EQ(shape.GetDims()[4], 16);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetDataType(), ge::DT_FLOAT16);
      }
      {
        ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        ASSERT_EQ(shape.GetDimNum(), 5);
        EXPECT_EQ(shape.GetDims()[0], 1);
        EXPECT_EQ(shape.GetDims()[1], 16);
        EXPECT_EQ(shape.GetDims()[2], 512);
        EXPECT_EQ(shape.GetDims()[3], 1);
        EXPECT_EQ(shape.GetDims()[4], 16);
        EXPECT_EQ(node->GetOpDesc()->GetOutputDescPtr(0)->GetDataType(), ge::DT_FLOAT);
      }
    }

    if (node->GetType() == "A") {
      {
        ge::GeShape shape = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        ASSERT_EQ(shape.GetDimNum(), 5);
        EXPECT_EQ(shape.GetDims()[0], 1);
        EXPECT_EQ(shape.GetDims()[1], 256);
        EXPECT_EQ(shape.GetDims()[2], 256);
        EXPECT_EQ(shape.GetDims()[3], 512);
        EXPECT_EQ(shape.GetDims()[4], 1);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetDataType(), ge::DT_FLOAT);
      }
      {
        ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        ASSERT_EQ(shape.GetDimNum(), 5);
        EXPECT_EQ(shape.GetDims()[0], 1);
        EXPECT_EQ(shape.GetDims()[1], 256);
        EXPECT_EQ(shape.GetDims()[2], 256);
        EXPECT_EQ(shape.GetDims()[3], 512);
        EXPECT_EQ(shape.GetDims()[4], 1);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetDataType(), ge::DT_FLOAT);
      }

    }
  }
  EXPECT_EQ(count_node, 5);

}

/*  Test Merging transop of the following case
 *        A (16out)
 *        |
 *      Cast (16->32)
 *      /  \
 *   Cast  Cast(both are 32-16)
 *    |      |
 *    B      C (16in) */
TEST_F(UTEST_FE_TRANSOP_INSERT_COMPLEX, MergeThreeCastOp) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");

  GeTensorDesc src_tensor_desc_16(GeShape({128, 256, 512, 1024}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
  GeTensorDesc dst_tensor_desc_32(GeShape({128, 256, 512, 1024}), ge::FORMAT_NCHW, ge::DT_FLOAT);
  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  src_op->AddOutputDesc(src_tensor_desc_16);
  src_op->AddInputDesc(src_tensor_desc_16);
  auto src_node = graph->AddNode(src_op);

  OpDescPtr dst_op = std::make_shared<OpDesc>("B1", "B");
  dst_op->AddInputDesc(src_tensor_desc_16);
  dst_op->AddOutputDesc(src_tensor_desc_16);
  auto dst_node1 = graph->AddNode(dst_op);

  OpDescPtr dst_op2 = std::make_shared<OpDesc>("B2", "B");
  dst_op2->AddInputDesc(src_tensor_desc_16);
  dst_op2->AddOutputDesc(src_tensor_desc_16);
  auto dst_node2 = graph->AddNode(dst_op2);

  OpDescPtr dst_op_cast1 = std::make_shared<OpDesc>("Cast1", "Cast");
  dst_op_cast1->AddInputDesc(src_tensor_desc_16);
  dst_op_cast1->AddOutputDesc(dst_tensor_desc_32);
  auto trandata_node1 = graph->AddNode(dst_op_cast1);

  OpDescPtr dst_op_cast2 = std::make_shared<OpDesc>("Cast2", "Cast");
  dst_op_cast2->AddInputDesc(dst_tensor_desc_32);
  dst_op_cast2->AddOutputDesc(src_tensor_desc_16);
  auto trandata_node2 = graph->AddNode(dst_op_cast2);

  OpDescPtr dst_op_cast3 = std::make_shared<OpDesc>("Cast3", "Cast");
  dst_op_cast3->AddInputDesc(dst_tensor_desc_32);
  dst_op_cast3->AddOutputDesc(src_tensor_desc_16);
  auto trandata_node3 = graph->AddNode(dst_op_cast3);


  GraphUtils::AddEdge(src_node->GetOutDataAnchor(0), trandata_node1->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node1->GetOutDataAnchor(0), trandata_node2->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node1->GetOutDataAnchor(0), trandata_node3->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node2->GetOutDataAnchor(0), dst_node1->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node3->GetOutDataAnchor(0), dst_node2->GetInDataAnchor(0));

  TransNodeMerging trans_op_merger;
  trans_op_merger.MergeAllTransOps(*(graph.get()), {}, false);
  uint32_t count_node = 0;
  for (auto node: graph->GetDirectNode()) {
    EXPECT_NE(node, nullptr);
    string name = node->GetName();

    EXPECT_NE(node->GetType(), "TransData");
    EXPECT_NE(node->GetType(), "Cast");
    if (name == "A") {
      EXPECT_EQ(node->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode()->GetName(), "B1");
      EXPECT_EQ(node->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(1)->GetOwnerNode()->GetName(), "B2");
    }
    if (name == "B1") {
      EXPECT_EQ(1, node->GetOpDesc()->GetSrcName().size());
      EXPECT_EQ(0, node->GetOpDesc()->GetSrcIndex().size());
      EXPECT_EQ(1, node->GetOpDesc()->GetInputName().size());

      EXPECT_EQ("A", node->GetOpDesc()->GetSrcName().at(0));
      EXPECT_EQ("A", node->GetOpDesc()->GetInputName().at(0));
      EXPECT_EQ(node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "A");

    }
    if (name == "B2") {
      EXPECT_EQ(1, node->GetOpDesc()->GetSrcName().size());
      EXPECT_EQ(0, node->GetOpDesc()->GetSrcIndex().size());
      EXPECT_EQ(1, node->GetOpDesc()->GetInputName().size());

      EXPECT_EQ("A", node->GetOpDesc()->GetSrcName().at(0));
      EXPECT_EQ("A", node->GetOpDesc()->GetInputName().at(0));
      EXPECT_EQ(node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "A");
    }

    count_node++;
  }
  EXPECT_EQ(count_node,3);
}


/*  Test Merging transop of the following case
*        A (16out)
*        |
*      Cast1 (16->32)
*        |
*      Cast2 (32->16)
*      /  \
*    B1    B2 (16in)*/
TEST_F(UTEST_FE_TRANSOP_INSERT_COMPLEX, MergeTwoCastOp) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");

  GeTensorDesc src_tensor_desc_16(GeShape({128, 256, 512, 1024}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
  GeTensorDesc dst_tensor_desc_32(GeShape({128, 256, 512, 1024}), ge::FORMAT_NCHW, ge::DT_FLOAT);
  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  src_op->AddOutputDesc(src_tensor_desc_16);
  src_op->AddInputDesc(src_tensor_desc_16);
  auto src_node = graph->AddNode(src_op);

  OpDescPtr dst_op = std::make_shared<OpDesc>("B1", "B");
  dst_op->AddInputDesc(src_tensor_desc_16);
  dst_op->AddOutputDesc(src_tensor_desc_16);
  auto dst_node1 = graph->AddNode(dst_op);

  OpDescPtr dst_op2 = std::make_shared<OpDesc>("B2", "B");
  dst_op2->AddInputDesc(src_tensor_desc_16);
  dst_op2->AddOutputDesc(src_tensor_desc_16);
  auto dst_node2 = graph->AddNode(dst_op2);

  OpDescPtr dst_op_cast1 = std::make_shared<OpDesc>("Cast1", "Cast");
  dst_op_cast1->AddInputDesc(src_tensor_desc_16);
  dst_op_cast1->AddOutputDesc(dst_tensor_desc_32);
  auto trandata_node1 = graph->AddNode(dst_op_cast1);

  OpDescPtr dst_op_cast2 = std::make_shared<OpDesc>("Cast2", "Cast");
  dst_op_cast2->AddInputDesc(dst_tensor_desc_32);
  dst_op_cast2->AddOutputDesc(src_tensor_desc_16);
  auto trandata_node2 = graph->AddNode(dst_op_cast2);


  GraphUtils::AddEdge(src_node->GetOutDataAnchor(0), trandata_node1->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node1->GetOutDataAnchor(0), trandata_node2->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node2->GetOutDataAnchor(0), dst_node1->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node2->GetOutDataAnchor(0), dst_node2->GetInDataAnchor(0));

  TransNodeMerging trans_op_merger;
  trans_op_merger.MergeAllTransOps(*(graph.get()), {}, false);
  uint32_t count_node = 0;
  for (auto node: graph->GetDirectNode()) {
    EXPECT_NE(node, nullptr);
    string name = node->GetName();

    EXPECT_NE(node->GetType(), "Cast");
    if (name == "A") {
      EXPECT_EQ(node->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode()->GetName(), "B1");
      EXPECT_EQ(node->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(1)->GetOwnerNode()->GetName(), "B2");
    }
    if (name == "B1") {
      EXPECT_EQ(1, node->GetOpDesc()->GetSrcName().size());
      EXPECT_EQ(0, node->GetOpDesc()->GetSrcIndex().size());
      EXPECT_EQ(1, node->GetOpDesc()->GetInputName().size());

      EXPECT_EQ("A", node->GetOpDesc()->GetSrcName().at(0));
      EXPECT_EQ("A", node->GetOpDesc()->GetInputName().at(0));
      EXPECT_EQ(node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "A");

    }
    if (name == "B2") {
      EXPECT_EQ(1, node->GetOpDesc()->GetSrcName().size());
      EXPECT_EQ(0, node->GetOpDesc()->GetSrcIndex().size());
      EXPECT_EQ(1, node->GetOpDesc()->GetInputName().size());

      EXPECT_EQ("A", node->GetOpDesc()->GetSrcName().at(0));
      EXPECT_EQ("A", node->GetOpDesc()->GetInputName().at(0));
      EXPECT_EQ(node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "A");
    }

    count_node++;
  }
  EXPECT_EQ(count_node,3);
}


/*  Test Merging transop of the following case
*            A  (16out)
*            |
*           Cast (16->32)
*        /       \
*    Cast(to16)  Cast (to Int32)
*    /  \           \
*  B     C (16in)  D (INT32 in) */
TEST_F(UTEST_FE_TRANSOP_INSERT_COMPLEX, MergeThreeCastOp2) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");

  GeTensorDesc src_tensor_desc_16(GeShape({128, 256, 512, 1024}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
  GeTensorDesc dst_tensor_desc_32(GeShape({128, 256, 512, 1024}), ge::FORMAT_NCHW, ge::DT_FLOAT);
  GeTensorDesc dst_tensor_desc_int32(GeShape({128, 256, 512, 1024}), ge::FORMAT_NCHW, ge::DT_INT32);
  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  src_op->AddOutputDesc(src_tensor_desc_16);
  src_op->AddInputDesc(src_tensor_desc_16);
  auto src_node = graph->AddNode(src_op);

  OpDescPtr dst_op = std::make_shared<OpDesc>("B1", "B");
  dst_op->AddInputDesc(src_tensor_desc_16);
  dst_op->AddOutputDesc(src_tensor_desc_16);
  auto dst_node1 = graph->AddNode(dst_op);

  OpDescPtr dst_op2 = std::make_shared<OpDesc>("B2", "B");
  dst_op2->AddInputDesc(src_tensor_desc_16);
  dst_op2->AddOutputDesc(src_tensor_desc_16);
  auto dst_node2 = graph->AddNode(dst_op2);

  OpDescPtr dst_op3 = std::make_shared<OpDesc>("B3", "B");
  dst_op3->AddInputDesc(dst_tensor_desc_int32);
  dst_op3->AddOutputDesc(dst_tensor_desc_int32);
  auto dst_node3 = graph->AddNode(dst_op3);
  dst_op3->SetSrcName({"Cast3"});
  dst_op3->SetSrcIndex({0});
  dst_op3->SetInputName({"Cast3"});

  OpDescPtr dst_op_cast1 = std::make_shared<OpDesc>("Cast1", "Cast");
  dst_op_cast1->AddInputDesc(src_tensor_desc_16);
  dst_op_cast1->AddOutputDesc(dst_tensor_desc_32);
  auto trandata_node1 = graph->AddNode(dst_op_cast1);

  OpDescPtr dst_op_cast2 = std::make_shared<OpDesc>("Cast2", "Cast");
  dst_op_cast2->AddInputDesc(dst_tensor_desc_32);
  dst_op_cast2->AddOutputDesc(src_tensor_desc_16);
  auto trandata_node2 = graph->AddNode(dst_op_cast2);

  OpDescPtr dst_op_cast3 = std::make_shared<OpDesc>("Cast3", "Cast");
  dst_op_cast3->AddInputDesc(dst_tensor_desc_32);
  dst_op_cast3->AddOutputDesc(dst_tensor_desc_int32);
  auto trandata_node3 = graph->AddNode(dst_op_cast3);


  GraphUtils::AddEdge(src_node->GetOutDataAnchor(0), trandata_node1->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node1->GetOutDataAnchor(0), trandata_node2->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node1->GetOutDataAnchor(0), trandata_node3->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node2->GetOutDataAnchor(0), dst_node1->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node2->GetOutDataAnchor(0), dst_node2->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node3->GetOutDataAnchor(0), dst_node3->GetInDataAnchor(0));

  TransNodeMerging trans_op_merger;
  trans_op_merger.MergeAllTransOps(*(graph.get()), {}, false);
  uint32_t count_node = 0;
  for (auto node: graph->GetDirectNode()) {
    EXPECT_NE(node, nullptr);
    string name = node->GetName();

    EXPECT_NE(node->GetType(), "TransData");
    if (name == "A") {
      EXPECT_EQ(node->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode()->GetName(), "Cast1");
      EXPECT_EQ(node->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(1)->GetOwnerNode()->GetName(), "B1");
      EXPECT_EQ(node->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(2)->GetOwnerNode()->GetName(), "B2");
    }
    if (name == "B1") {
      EXPECT_EQ(1, node->GetOpDesc()->GetSrcName().size());
      EXPECT_EQ(0, node->GetOpDesc()->GetSrcIndex().size());
      EXPECT_EQ(1, node->GetOpDesc()->GetInputName().size());

      EXPECT_EQ("A", node->GetOpDesc()->GetSrcName().at(0));
      EXPECT_EQ("A", node->GetOpDesc()->GetInputName().at(0));
      EXPECT_EQ(node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "A");

    }
    if (name == "B2") {
      EXPECT_EQ(1, node->GetOpDesc()->GetSrcName().size());
      EXPECT_EQ(0, node->GetOpDesc()->GetSrcIndex().size());
      EXPECT_EQ(1, node->GetOpDesc()->GetInputName().size());

      EXPECT_EQ("A", node->GetOpDesc()->GetSrcName().at(0));
      EXPECT_EQ("A", node->GetOpDesc()->GetInputName().at(0));
      EXPECT_EQ(node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "A");
    }

    if (name == "B3") {
      EXPECT_EQ(1, node->GetOpDesc()->GetSrcName().size());
      EXPECT_EQ(1, node->GetOpDesc()->GetSrcIndex().size());
      EXPECT_EQ(1, node->GetOpDesc()->GetInputName().size());

      EXPECT_EQ("Cast3", node->GetOpDesc()->GetSrcName().at(0));
      EXPECT_EQ("Cast3", node->GetOpDesc()->GetInputName().at(0));
      EXPECT_EQ(node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "Cast3");
    }

    count_node++;
  }
  EXPECT_EQ(count_node,6);
}

/*  Test Merging transop of the following case
*           A (16out)
*           |
*          Cast1 (16->32)
*      /    |       \
*   Cast2  Cast3     B3
* (32->16) (32->16)  /  \
*    |      |       /    \
*    B1     B2    Cast4  Cast5(32->16)
*    |      |   (32->16)  |
*    |      |      |      |
*    \    Cast6   Cast7  Cast8(16->int32)
*     \  (16->32)(16->32) |
*      \    |    /        |
*       \   |   /         |
*        \  |  /          |
*          B4             B5(int32)
*/
TEST_F(UTEST_FE_TRANSOP_INSERT_COMPLEX, MergeMultipleCastOp) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");

  GeTensorDesc src_tensor_desc_16(GeShape({128, 256, 512, 1024}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
  GeTensorDesc dst_tensor_desc_32(GeShape({128, 256, 512, 1024}), ge::FORMAT_NCHW, ge::DT_FLOAT);
  GeTensorDesc dst_tensor_desc_int32(GeShape({128, 256, 512, 1024}), ge::FORMAT_NCHW, ge::DT_INT32);
  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  src_op->AddOutputDesc(src_tensor_desc_16);
  src_op->AddInputDesc(src_tensor_desc_16);
  auto src_node = graph->AddNode(src_op);

  OpDescPtr dst_op = std::make_shared<OpDesc>("B1", "B");
  dst_op->AddInputDesc(src_tensor_desc_16);
  dst_op->AddOutputDesc(src_tensor_desc_16);
  auto dst_node1 = graph->AddNode(dst_op);

  OpDescPtr dst_op2 = std::make_shared<OpDesc>("B2", "B");
  dst_op2->AddInputDesc(src_tensor_desc_16);
  dst_op2->AddOutputDesc(src_tensor_desc_16);
  auto dst_node2 = graph->AddNode(dst_op2);

  OpDescPtr dst_op3 = std::make_shared<OpDesc>("B3", "B");
  dst_op3->AddInputDesc(dst_tensor_desc_32);
  dst_op3->AddOutputDesc(dst_tensor_desc_32);
  dst_op3->AddOutputDesc(dst_tensor_desc_32);
  auto dst_node3 = graph->AddNode(dst_op3);

  OpDescPtr dst_op4 = std::make_shared<OpDesc>("B4", "B");
  dst_op4->AddInputDesc(src_tensor_desc_16);
  dst_op4->AddInputDesc(dst_tensor_desc_32);
  dst_op4->AddInputDesc(dst_tensor_desc_32);
  dst_op4->AddOutputDesc(src_tensor_desc_16);
  auto dst_node4 = graph->AddNode(dst_op4);
  dst_op4->SetSrcName({"B1", "Cast6", "Cast7"});
  dst_op4->SetSrcIndex({0, 1, 2});
  dst_op4->SetInputName({"B1", "Cast6", "Cast7"});

  OpDescPtr dst_op5 = std::make_shared<OpDesc>("B5", "B");
  dst_op5->AddInputDesc(dst_tensor_desc_int32);
  dst_op5->AddOutputDesc(dst_tensor_desc_int32);
  auto dst_node5 = graph->AddNode(dst_op5);

  OpDescPtr dst_op_cast1 = std::make_shared<OpDesc>("Cast1", "Cast");
  dst_op_cast1->AddInputDesc(src_tensor_desc_16);
  dst_op_cast1->AddOutputDesc(dst_tensor_desc_32);
  auto trandata_node1 = graph->AddNode(dst_op_cast1);
  OpDescPtr dst_op_cast2 = std::make_shared<OpDesc>("Cast2", "Cast");
  dst_op_cast2->AddInputDesc(dst_tensor_desc_32);
  dst_op_cast2->AddOutputDesc(src_tensor_desc_16);
  auto trandata_node2 = graph->AddNode(dst_op_cast2);
  OpDescPtr dst_op_cast3 = std::make_shared<OpDesc>("Cast3", "Cast");
  dst_op_cast3->AddInputDesc(dst_tensor_desc_32);
  dst_op_cast3->AddOutputDesc(src_tensor_desc_16);
  auto trandata_node3 = graph->AddNode(dst_op_cast3);
  OpDescPtr dst_op_cast4 = std::make_shared<OpDesc>("Cast4", "Cast");
  dst_op_cast4->AddInputDesc(dst_tensor_desc_32);
  dst_op_cast4->AddOutputDesc(src_tensor_desc_16);
  auto trandata_node4 = graph->AddNode(dst_op_cast4);
  OpDescPtr dst_op_cast5 = std::make_shared<OpDesc>("Cast5", "Cast");
  dst_op_cast5->AddInputDesc(dst_tensor_desc_32);
  dst_op_cast5->AddOutputDesc(src_tensor_desc_16);
  auto trandata_node5 = graph->AddNode(dst_op_cast5);
  OpDescPtr dst_op_cast6 = std::make_shared<OpDesc>("Cast6", "Cast");
  dst_op_cast6->AddInputDesc(src_tensor_desc_16);
  dst_op_cast6->AddOutputDesc(dst_tensor_desc_32);
  auto trandata_node6 = graph->AddNode(dst_op_cast6);
  OpDescPtr dst_op_cast7 = std::make_shared<OpDesc>("Cast7", "Cast");
  dst_op_cast7->AddInputDesc(src_tensor_desc_16);
  dst_op_cast7->AddOutputDesc(dst_tensor_desc_32);
  auto trandata_node7 = graph->AddNode(dst_op_cast7);
  OpDescPtr dst_op_cast8 = std::make_shared<OpDesc>("Cast8", "Cast");
  dst_op_cast8->AddInputDesc(src_tensor_desc_16);
  dst_op_cast8->AddOutputDesc(dst_tensor_desc_int32);
  auto trandata_node8 = graph->AddNode(dst_op_cast8);


  GraphUtils::AddEdge(src_node->GetOutDataAnchor(0), trandata_node1->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node1->GetOutDataAnchor(0), trandata_node2->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node1->GetOutDataAnchor(0), trandata_node3->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node1->GetOutDataAnchor(0), dst_node3->GetInDataAnchor(0));

  GraphUtils::AddEdge(trandata_node2->GetOutDataAnchor(0), dst_node1->GetInDataAnchor(0));

  GraphUtils::AddEdge(trandata_node3->GetOutDataAnchor(0), dst_node2->GetInDataAnchor(0));
  GraphUtils::AddEdge(dst_node3->GetOutDataAnchor(0), trandata_node4->GetInDataAnchor(0));
  GraphUtils::AddEdge(dst_node3->GetOutDataAnchor(1), trandata_node5->GetInDataAnchor(0));

  GraphUtils::AddEdge(dst_node2->GetOutDataAnchor(0), trandata_node6->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node4->GetOutDataAnchor(0), trandata_node7->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node5->GetOutDataAnchor(0), trandata_node8->GetInDataAnchor(0));

  GraphUtils::AddEdge(dst_node1->GetOutDataAnchor(0), dst_node4->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node6->GetOutDataAnchor(0), dst_node4->GetInDataAnchor(1));
  GraphUtils::AddEdge(trandata_node7->GetOutDataAnchor(0), dst_node4->GetInDataAnchor(2));
  GraphUtils::AddEdge(trandata_node8->GetOutDataAnchor(0), dst_node5->GetInDataAnchor(0));

  TransNodeMerging trans_op_merger;
  trans_op_merger.MergeAllTransOps(*(graph.get()), {}, false);
  uint32_t count_node = 0;
  uint32_t count_b4 = 0;
  for (auto node: graph->GetDirectNode()) {
    EXPECT_NE(node, nullptr);
    string name = node->GetName();
    if (name == "A") {
      EXPECT_EQ(node->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode()->GetName(), "Cast1");
      EXPECT_EQ(node->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(1)->GetOwnerNode()->GetName(), "B1");
      EXPECT_EQ(node->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(2)->GetOwnerNode()->GetName(), "B2");
    }
    if (name == "B1") {
      EXPECT_EQ(1, node->GetOpDesc()->GetSrcName().size());
      EXPECT_EQ(0, node->GetOpDesc()->GetSrcIndex().size());
      EXPECT_EQ(1, node->GetOpDesc()->GetInputName().size());

      EXPECT_EQ("A", node->GetOpDesc()->GetSrcName().at(0));
      EXPECT_EQ("A", node->GetOpDesc()->GetInputName().at(0));
      EXPECT_EQ(node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "A");

    }
    if (name == "B2") {
      EXPECT_EQ(1, node->GetOpDesc()->GetSrcName().size());
      EXPECT_EQ(0, node->GetOpDesc()->GetSrcIndex().size());
      EXPECT_EQ(1, node->GetOpDesc()->GetInputName().size());

      EXPECT_EQ("A", node->GetOpDesc()->GetSrcName().at(0));
      EXPECT_EQ("A", node->GetOpDesc()->GetInputName().at(0));
      EXPECT_EQ(node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "A");
    }
    if (name == "Cast1") {
      EXPECT_EQ(node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "A");
    }

    if (name == "Cast6") {
      EXPECT_EQ(node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "B2");
    }
    if (name == "B3") {
      EXPECT_EQ(node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "Cast1");
    }

    if (name == "Cast5") {
      EXPECT_EQ(node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "B3");
    }

    if (name == "Cast8") {
      EXPECT_EQ(node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "Cast5");
    }
    if (name == "B5") {
      EXPECT_EQ(node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "Cast8");
    }

    if (name == "B4") {
      count_b4++;
      EXPECT_EQ(3, node->GetOpDesc()->GetSrcName().size());
      EXPECT_EQ(3, node->GetOpDesc()->GetSrcIndex().size());
      EXPECT_EQ(3, node->GetOpDesc()->GetInputName().size());

      EXPECT_EQ("B1", node->GetOpDesc()->GetSrcName().at(0));
      EXPECT_EQ(0, node->GetOpDesc()->GetSrcIndex().at(0));
      EXPECT_EQ("B1", node->GetOpDesc()->GetInputName().at(0));

      EXPECT_EQ("Cast6", node->GetOpDesc()->GetSrcName().at(1));
      EXPECT_EQ(1, node->GetOpDesc()->GetSrcIndex().at(1));
      EXPECT_EQ("Cast6", node->GetOpDesc()->GetInputName().at(1));

      EXPECT_EQ("B3", node->GetOpDesc()->GetSrcName().at(2));
      EXPECT_EQ(2, node->GetOpDesc()->GetSrcIndex().at(2));
      EXPECT_EQ("B3", node->GetOpDesc()->GetInputName().at(2));
      EXPECT_EQ(node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "B1");
      EXPECT_EQ(node->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "Cast6");
      EXPECT_EQ(node->GetInDataAnchor(2)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "B3");
    }
    count_node++;
  }
  EXPECT_EQ(count_node,10);
  EXPECT_EQ(count_b4,1);
}



/*  Test Merging transop of the following case
*              A (32out)
*              |
*           Cast1 (32->16)
*       /               \
* Cast2 (16->INT32)   Cast3(16->32)
*   |                    |
*  B1                    B2*/
TEST_F(UTEST_FE_TRANSOP_INSERT_COMPLEX, MergeMultipleCastOp2) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");

  GeTensorDesc tensor_desc_16(GeShape({128, 256, 512, 1024}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
  GeTensorDesc tensor_desc_32(GeShape({128, 256, 512, 1024}), ge::FORMAT_NCHW, ge::DT_FLOAT);
  GeTensorDesc tensor_desc_int32(GeShape({128, 256, 512, 1024}), ge::FORMAT_NCHW, ge::DT_INT32);
  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  src_op->AddOutputDesc(tensor_desc_32);
  src_op->AddInputDesc(tensor_desc_32);
  auto src_node = graph->AddNode(src_op);

  OpDescPtr dst_op = std::make_shared<OpDesc>("B1", "B");
  dst_op->AddInputDesc(tensor_desc_16);
  dst_op->AddOutputDesc(tensor_desc_16);
  auto dst_node1 = graph->AddNode(dst_op);

  OpDescPtr dst_op2 = std::make_shared<OpDesc>("B2", "B");
  dst_op2->AddInputDesc(tensor_desc_int32);
  dst_op2->AddOutputDesc(tensor_desc_int32);
  auto dst_node2 = graph->AddNode(dst_op2);

  OpDescPtr dst_op_cast1 = std::make_shared<OpDesc>("Cast1", "Cast");
  dst_op_cast1->AddInputDesc(tensor_desc_32);
  dst_op_cast1->AddOutputDesc(tensor_desc_16);
  auto trandata_node1 = graph->AddNode(dst_op_cast1);

  OpDescPtr dst_op_cast2 = std::make_shared<OpDesc>("Cast2", "Cast");
  dst_op_cast2->AddInputDesc(tensor_desc_16);
  dst_op_cast2->AddOutputDesc(tensor_desc_int32);
  auto trandata_node2 = graph->AddNode(dst_op_cast2);

  OpDescPtr dst_op_cast3 = std::make_shared<OpDesc>("Cast3", "Cast");
  dst_op_cast3->AddInputDesc(tensor_desc_16);
  dst_op_cast3->AddOutputDesc(tensor_desc_32);
  auto trandata_node3 = graph->AddNode(dst_op_cast3);

  GraphUtils::AddEdge(src_node->GetOutDataAnchor(0), trandata_node1->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node1->GetOutDataAnchor(0), trandata_node2->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node1->GetOutDataAnchor(0), trandata_node3->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node2->GetOutDataAnchor(0), dst_node1->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node3->GetOutDataAnchor(0), dst_node2->GetInDataAnchor(0));

  TransNodeMerging trans_op_merger;
  trans_op_merger.MergeAllTransOps(*(graph.get()), {}, false);
  uint32_t count_node = 0;
  for (auto node: graph->GetDirectNode()) {
    EXPECT_NE(node, nullptr);
    string name = node->GetName();

    if (name == "A") {
      EXPECT_EQ(node->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode()->GetName(), "Cast1");
      EXPECT_EQ(node->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(1)->GetOwnerNode()->GetName(), "B2");
    }
    if (name == "B1") {
      EXPECT_EQ(node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "Cast2");

    }
    if (name == "B2") {
      EXPECT_EQ(1, node->GetOpDesc()->GetSrcName().size());
      EXPECT_EQ(1, node->GetOpDesc()->GetInputName().size());

      EXPECT_EQ("A", node->GetOpDesc()->GetSrcName().at(0));
      EXPECT_EQ("A", node->GetOpDesc()->GetInputName().at(0));
      EXPECT_EQ(node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "A");
    }
    if (name == "Cast1") {
      EXPECT_EQ(node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "A");
      EXPECT_EQ(node->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode()->GetName(), "Cast2");
    }
    if (name == "Cast2") {
      EXPECT_EQ(node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "Cast1");
      EXPECT_EQ(node->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode()->GetName(), "B1");
    }
    count_node++;
  }
  EXPECT_EQ(count_node,5);
}


/*  Test Merging transop of the following case
*                       A (32out)
*                       |
*                      Cast1 (32->16)
*       /               |              \
* Cast2 (16->INT32)   Cast3(16->32)   Cast4(16->32)
*   |                    |              |
*  B1                    B2             B3*/
TEST_F(UTEST_FE_TRANSOP_INSERT_COMPLEX, MergeMultipleCastOp3) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");

  GeTensorDesc tensor_desc_16(GeShape({128, 256, 512, 1024}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
  GeTensorDesc tensor_desc_32(GeShape({128, 256, 512, 1024}), ge::FORMAT_NCHW, ge::DT_FLOAT);
  GeTensorDesc tensor_desc_int32(GeShape({128, 256, 512, 1024}), ge::FORMAT_NCHW, ge::DT_INT32);
  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  src_op->AddOutputDesc(tensor_desc_32);
  src_op->AddInputDesc(tensor_desc_32);
  auto src_node = graph->AddNode(src_op);

  OpDescPtr dst_op = std::make_shared<OpDesc>("B1", "B");
  dst_op->AddInputDesc(tensor_desc_16);
  dst_op->AddOutputDesc(tensor_desc_16);
  auto dst_node1 = graph->AddNode(dst_op);

  OpDescPtr dst_op2 = std::make_shared<OpDesc>("B2", "B");
  dst_op2->AddInputDesc(tensor_desc_int32);
  dst_op2->AddOutputDesc(tensor_desc_int32);
  auto dst_node2 = graph->AddNode(dst_op2);

  OpDescPtr dst_op3 = std::make_shared<OpDesc>("B3", "B");
  dst_op3->AddInputDesc(tensor_desc_int32);
  dst_op3->AddOutputDesc(tensor_desc_int32);
  auto dst_node3 = graph->AddNode(dst_op3);

  OpDescPtr dst_op_cast1 = std::make_shared<OpDesc>("Cast1", "Cast");
  dst_op_cast1->AddInputDesc(tensor_desc_32);
  dst_op_cast1->AddOutputDesc(tensor_desc_16);
  auto trandata_node1 = graph->AddNode(dst_op_cast1);

  OpDescPtr dst_op_cast2 = std::make_shared<OpDesc>("Cast2", "Cast");
  dst_op_cast2->AddInputDesc(tensor_desc_16);
  dst_op_cast2->AddOutputDesc(tensor_desc_int32);
  auto trandata_node2 = graph->AddNode(dst_op_cast2);

  OpDescPtr dst_op_cast3 = std::make_shared<OpDesc>("Cast3", "Cast");
  dst_op_cast3->AddInputDesc(tensor_desc_16);
  dst_op_cast3->AddOutputDesc(tensor_desc_32);
  auto trandata_node3 = graph->AddNode(dst_op_cast3);

  OpDescPtr dst_op_cast4 = std::make_shared<OpDesc>("Cast4", "Cast");
  dst_op_cast4->AddInputDesc(tensor_desc_16);
  dst_op_cast4->AddOutputDesc(tensor_desc_32);
  auto trandata_node4 = graph->AddNode(dst_op_cast4);

  GraphUtils::AddEdge(src_node->GetOutDataAnchor(0), trandata_node1->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node1->GetOutDataAnchor(0), trandata_node2->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node1->GetOutDataAnchor(0), trandata_node3->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node1->GetOutDataAnchor(0), trandata_node4->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node2->GetOutDataAnchor(0), dst_node1->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node3->GetOutDataAnchor(0), dst_node2->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node4->GetOutDataAnchor(0), dst_node3->GetInDataAnchor(0));

  TransNodeMerging trans_op_merger;
  trans_op_merger.MergeAllTransOps(*(graph.get()), {}, false);
  uint32_t count_node = 0;
  for (auto node: graph->GetDirectNode()) {
    EXPECT_NE(node, nullptr);
    string name = node->GetName();

    if (name == "A") {
      EXPECT_EQ(node->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode()->GetName(), "Cast1");
      EXPECT_EQ(node->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(1)->GetOwnerNode()->GetName(), "B2");
      EXPECT_EQ(node->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(2)->GetOwnerNode()->GetName(), "B3");
    }
    if (name == "B1") {
      EXPECT_EQ(node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "Cast2");

    }
    if (name == "B2") {
      EXPECT_EQ(1, node->GetOpDesc()->GetSrcName().size());
      EXPECT_EQ(1, node->GetOpDesc()->GetInputName().size());

      EXPECT_EQ("A", node->GetOpDesc()->GetSrcName().at(0));
      EXPECT_EQ("A", node->GetOpDesc()->GetInputName().at(0));
      EXPECT_EQ(node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "A");
    }

    if (name == "B3") {
      EXPECT_EQ(1, node->GetOpDesc()->GetSrcName().size());
      EXPECT_EQ(1, node->GetOpDesc()->GetInputName().size());

      EXPECT_EQ("A", node->GetOpDesc()->GetSrcName().at(0));
      EXPECT_EQ("A", node->GetOpDesc()->GetInputName().at(0));
      EXPECT_EQ(node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "A");
    }
    if (name == "Cast1") {
      EXPECT_EQ(node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "A");
      EXPECT_EQ(node->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode()->GetName(), "Cast2");
    }
    if (name == "Cast2") {
      EXPECT_EQ(node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "Cast1");
      EXPECT_EQ(node->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode()->GetName(), "B1");
    }

    count_node++;
  }
  EXPECT_EQ(count_node,6);
}


/*  Test Merging transop of the following case
*                       A (32out)
*                       |
*                      Cast1 (32->16)
*       /               |                     \
* Cast2 (16->INT32)   Cast3(16->32)          Cast4(16->32)
*   |                    |                /      |         \
*  B1                    B2         Cast5   Cast7(32->Int) Cast8
*                                     |          |          |
*                                   Cast6       B4        Cast9
*                                     |                     |
*                                     B3                    B5
*/
TEST_F(UTEST_FE_TRANSOP_INSERT_COMPLEX, MergeMultipleCastOp4) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");

  GeTensorDesc tensor_desc_16(GeShape({128, 256, 512, 1024}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
  GeTensorDesc tensor_desc_32(GeShape({128, 256, 512, 1024}), ge::FORMAT_NCHW, ge::DT_FLOAT);
  GeTensorDesc tensor_desc_int32(GeShape({128, 256, 512, 1024}), ge::FORMAT_NCHW, ge::DT_INT32);
  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  src_op->AddOutputDesc(tensor_desc_32);
  src_op->AddInputDesc(tensor_desc_32);
  auto src_node = graph->AddNode(src_op);

  OpDescPtr dst_op = std::make_shared<OpDesc>("B1", "B");
  dst_op->AddInputDesc(tensor_desc_16);
  dst_op->AddOutputDesc(tensor_desc_16);
  auto dst_node1 = graph->AddNode(dst_op);

  OpDescPtr dst_op2 = std::make_shared<OpDesc>("B2", "B");
  dst_op2->AddInputDesc(tensor_desc_32);
  dst_op2->AddOutputDesc(tensor_desc_32);
  auto dst_node2 = graph->AddNode(dst_op2);

  OpDescPtr dst_op3 = std::make_shared<OpDesc>("B3", "B");
  dst_op3->AddInputDesc(tensor_desc_32);
  dst_op3->AddOutputDesc(tensor_desc_32);
  auto dst_node3 = graph->AddNode(dst_op3);

  OpDescPtr dst_op4 = std::make_shared<OpDesc>("B4", "B");
  dst_op4->AddInputDesc(tensor_desc_int32);
  dst_op4->AddOutputDesc(tensor_desc_int32);
  auto dst_node4 = graph->AddNode(dst_op4);

  OpDescPtr dst_op5 = std::make_shared<OpDesc>("B5", "B");
  dst_op5->AddInputDesc(tensor_desc_32);
  dst_op5->AddOutputDesc(tensor_desc_32);
  auto dst_node5 = graph->AddNode(dst_op5);

  OpDescPtr dst_op_cast1 = std::make_shared<OpDesc>("Cast1", "Cast");
  dst_op_cast1->AddInputDesc(tensor_desc_32);
  dst_op_cast1->AddOutputDesc(tensor_desc_16);
  auto trandata_node1 = graph->AddNode(dst_op_cast1);

  OpDescPtr dst_op_cast2 = std::make_shared<OpDesc>("Cast2", "Cast");
  dst_op_cast2->AddInputDesc(tensor_desc_16);
  dst_op_cast2->AddOutputDesc(tensor_desc_int32);
  auto trandata_node2 = graph->AddNode(dst_op_cast2);

  OpDescPtr dst_op_cast3 = std::make_shared<OpDesc>("Cast3", "Cast");
  dst_op_cast3->AddInputDesc(tensor_desc_16);
  dst_op_cast3->AddOutputDesc(tensor_desc_32);
  auto trandata_node3 = graph->AddNode(dst_op_cast3);

  OpDescPtr dst_op_cast4 = std::make_shared<OpDesc>("Cast4", "Cast");
  dst_op_cast4->AddInputDesc(tensor_desc_16);
  dst_op_cast4->AddOutputDesc(tensor_desc_32);
  auto trandata_node4 = graph->AddNode(dst_op_cast4);

  OpDescPtr dst_op_cast5 = std::make_shared<OpDesc>("Cast5", "Cast");
  dst_op_cast5->AddInputDesc(tensor_desc_32);
  dst_op_cast5->AddOutputDesc(tensor_desc_16);
  auto trandata_node5 = graph->AddNode(dst_op_cast5);

  OpDescPtr dst_op_cast6 = std::make_shared<OpDesc>("Cast6", "Cast");
  dst_op_cast6->AddInputDesc(tensor_desc_16);
  dst_op_cast6->AddOutputDesc(tensor_desc_32);
  auto trandata_node6 = graph->AddNode(dst_op_cast6);

  OpDescPtr dst_op_cast7 = std::make_shared<OpDesc>("Cast7", "Cast");
  dst_op_cast7->AddInputDesc(tensor_desc_32);
  dst_op_cast7->AddOutputDesc(tensor_desc_int32);
  auto trandata_node7 = graph->AddNode(dst_op_cast7);

  OpDescPtr dst_op_cast8 = std::make_shared<OpDesc>("Cast8", "Cast");
  dst_op_cast8->AddInputDesc(tensor_desc_32);
  dst_op_cast8->AddOutputDesc(tensor_desc_16);
  auto trandata_node8 = graph->AddNode(dst_op_cast8);

  OpDescPtr dst_op_cast9 = std::make_shared<OpDesc>("Cast9", "Cast");
  dst_op_cast9->AddInputDesc(tensor_desc_16);
  dst_op_cast9->AddOutputDesc(tensor_desc_32);
  auto trandata_node9 = graph->AddNode(dst_op_cast9);

  GraphUtils::AddEdge(src_node->GetOutDataAnchor(0), trandata_node1->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node1->GetOutDataAnchor(0), trandata_node2->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node1->GetOutDataAnchor(0), trandata_node3->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node1->GetOutDataAnchor(0), trandata_node4->GetInDataAnchor(0));

  GraphUtils::AddEdge(trandata_node4->GetOutDataAnchor(0), trandata_node5->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node4->GetOutDataAnchor(0), trandata_node7->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node4->GetOutDataAnchor(0), trandata_node8->GetInDataAnchor(0));

  GraphUtils::AddEdge(trandata_node2->GetOutDataAnchor(0), dst_node1->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node3->GetOutDataAnchor(0), dst_node2->GetInDataAnchor(0));

  GraphUtils::AddEdge(trandata_node5->GetOutDataAnchor(0), trandata_node6->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node7->GetOutDataAnchor(0), dst_node4->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node8->GetOutDataAnchor(0), trandata_node9->GetInDataAnchor(0));

  GraphUtils::AddEdge(trandata_node6->GetOutDataAnchor(0), dst_node3->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node9->GetOutDataAnchor(0), dst_node5->GetInDataAnchor(0));


  TransNodeMerging trans_op_merger;
  trans_op_merger.MergeAllTransOps(*(graph.get()), {}, false);
  uint32_t count_node = 0;
  for (auto node: graph->GetDirectNode()) {
    EXPECT_NE(node, nullptr);
    string name = node->GetName();

    if (name == "A") {
      EXPECT_EQ(node->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode()->GetName(), "Cast1");
      EXPECT_EQ(node->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(1)->GetOwnerNode()->GetName(), "B2");
      EXPECT_EQ(node->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(2)->GetOwnerNode()->GetName(), "B3");
      EXPECT_EQ(node->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(3)->GetOwnerNode()->GetName(), "Cast7");
      EXPECT_EQ(node->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(4)->GetOwnerNode()->GetName(), "B5");

    }
    if (name == "B1") {
      EXPECT_EQ(node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "Cast2");

    }
    if (name == "B2") {
      EXPECT_EQ(1, node->GetOpDesc()->GetSrcName().size());
      EXPECT_EQ(1, node->GetOpDesc()->GetInputName().size());

      EXPECT_EQ("A", node->GetOpDesc()->GetSrcName().at(0));
      EXPECT_EQ("A", node->GetOpDesc()->GetInputName().at(0));
      EXPECT_EQ(node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "A");
    }

    if (name == "B3") {
      EXPECT_EQ(1, node->GetOpDesc()->GetSrcName().size());
      EXPECT_EQ(1, node->GetOpDesc()->GetInputName().size());

      EXPECT_EQ("A", node->GetOpDesc()->GetSrcName().at(0));
      EXPECT_EQ("A", node->GetOpDesc()->GetInputName().at(0));
      EXPECT_EQ(node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "A");
    }

    if (name == "B4") {
      EXPECT_EQ(node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "Cast7");
    }

    if (name == "B5") {
      EXPECT_EQ(1, node->GetOpDesc()->GetSrcName().size());
      EXPECT_EQ(1, node->GetOpDesc()->GetInputName().size());

      EXPECT_EQ("A", node->GetOpDesc()->GetSrcName().at(0));
      EXPECT_EQ("A", node->GetOpDesc()->GetInputName().at(0));
      EXPECT_EQ(node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "A");
    }
    if (name == "Cast1") {
      EXPECT_EQ(node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "A");
      EXPECT_EQ(node->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode()->GetName(), "Cast2");
    }
    if (name == "Cast2") {
      EXPECT_EQ(node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "Cast1");
      EXPECT_EQ(node->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode()->GetName(), "B1");
    }
    if (name == "Cast7") {
      EXPECT_EQ(node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "A");
      EXPECT_EQ(node->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode()->GetName(), "B4");
    }
    count_node++;
  }
  EXPECT_EQ(count_node,9);
}

TEST_F(UTEST_FE_TRANSOP_INSERT_COMPLEX, MergeMultipleCastOp5) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");

  GeTensorDesc tensor_desc_16(GeShape({128, 256, 512, 1024}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
  GeTensorDesc tensor_desc_32(GeShape({128, 256, 512, 1024}), ge::FORMAT_NCHW, ge::DT_FLOAT);
  GeTensorDesc tensor_desc_int32(GeShape({128, 256, 512, 1024}), ge::FORMAT_NCHW, ge::DT_INT32);
  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  src_op->AddOutputDesc(tensor_desc_32);
  src_op->AddInputDesc(tensor_desc_32);
  auto src_node = graph->AddNode(src_op);

  OpDescPtr dst_op = std::make_shared<OpDesc>("B1", "B");
  dst_op->AddInputDesc(tensor_desc_32);
  dst_op->AddOutputDesc(tensor_desc_32);
  auto dst_node1 = graph->AddNode(dst_op);

  OpDescPtr dst_op2 = std::make_shared<OpDesc>("B2", "B");
  dst_op2->AddInputDesc(tensor_desc_32);
  dst_op2->AddOutputDesc(tensor_desc_32);
  auto dst_node2 = graph->AddNode(dst_op2);

  OpDescPtr dst_op3 = std::make_shared<OpDesc>("B3", "B");
  dst_op3->AddInputDesc(tensor_desc_32);
  dst_op3->AddOutputDesc(tensor_desc_32);
  auto dst_node3 = graph->AddNode(dst_op3);

  OpDescPtr dst_op4 = std::make_shared<OpDesc>("B4", "B");
  dst_op4->AddInputDesc(tensor_desc_32);
  dst_op4->AddOutputDesc(tensor_desc_32);
  auto dst_node4 = graph->AddNode(dst_op4);


  OpDescPtr dst_op_cast1 = std::make_shared<OpDesc>("Cast1", "Cast");
  dst_op_cast1->AddInputDesc(tensor_desc_32);
  dst_op_cast1->AddOutputDesc(tensor_desc_16);
  auto trandata_node1 = graph->AddNode(dst_op_cast1);

  OpDescPtr dst_op_cast2 = std::make_shared<OpDesc>("Cast2", "Cast");
  dst_op_cast2->AddInputDesc(tensor_desc_16);
  dst_op_cast2->AddOutputDesc(tensor_desc_int32);
  auto trandata_node2 = graph->AddNode(dst_op_cast2);

  OpDescPtr dst_op_cast3 = std::make_shared<OpDesc>("Cast3", "Cast");
  dst_op_cast3->AddInputDesc(tensor_desc_16);
  dst_op_cast3->AddOutputDesc(tensor_desc_int32);
  auto trandata_node3 = graph->AddNode(dst_op_cast3);

  OpDescPtr dst_op_cast4 = std::make_shared<OpDesc>("Cast4", "Cast");
  dst_op_cast4->AddInputDesc(tensor_desc_int32);
  dst_op_cast4->AddOutputDesc(tensor_desc_16);
  auto trandata_node4 = graph->AddNode(dst_op_cast4);

  OpDescPtr dst_op_cast5 = std::make_shared<OpDesc>("Cast5", "Cast");
  dst_op_cast5->AddInputDesc(tensor_desc_int32);
  dst_op_cast5->AddOutputDesc(tensor_desc_16);
  auto trandata_node5 = graph->AddNode(dst_op_cast5);

  OpDescPtr dst_op_cast6 = std::make_shared<OpDesc>("Cast6", "Cast");
  dst_op_cast6->AddInputDesc(tensor_desc_int32);
  dst_op_cast6->AddOutputDesc(tensor_desc_16);
  auto trandata_node6 = graph->AddNode(dst_op_cast6);

  OpDescPtr dst_op_cast7 = std::make_shared<OpDesc>("Cast7", "Cast");
  dst_op_cast7->AddInputDesc(tensor_desc_int32);
  dst_op_cast7->AddOutputDesc(tensor_desc_16);
  auto trandata_node7 = graph->AddNode(dst_op_cast7);

  OpDescPtr dst_op_cast8 = std::make_shared<OpDesc>("Cast8", "Cast");
  dst_op_cast8->AddInputDesc(tensor_desc_16);
  dst_op_cast8->AddOutputDesc(tensor_desc_32);
  auto trandata_node8 = graph->AddNode(dst_op_cast8);

  OpDescPtr dst_op_cast9 = std::make_shared<OpDesc>("Cast9", "Cast");
  dst_op_cast9->AddInputDesc(tensor_desc_16);
  dst_op_cast9->AddOutputDesc(tensor_desc_32);
  auto trandata_node9 = graph->AddNode(dst_op_cast9);

  OpDescPtr dst_op_cast10 = std::make_shared<OpDesc>("Cast10", "Cast");
  dst_op_cast10->AddInputDesc(tensor_desc_16);
  dst_op_cast10->AddOutputDesc(tensor_desc_32);
  auto trandata_node10 = graph->AddNode(dst_op_cast10);

  OpDescPtr dst_op_cast11 = std::make_shared<OpDesc>("Cast11", "Cast");
  dst_op_cast11->AddInputDesc(tensor_desc_16);
  dst_op_cast11->AddOutputDesc(tensor_desc_32);
  auto trandata_node11 = graph->AddNode(dst_op_cast11);
  GraphUtils::AddEdge(src_node->GetOutDataAnchor(0), trandata_node1->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node1->GetOutDataAnchor(0), trandata_node2->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node1->GetOutDataAnchor(0), trandata_node3->GetInDataAnchor(0));

  GraphUtils::AddEdge(trandata_node2->GetOutDataAnchor(0), trandata_node4->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node2->GetOutDataAnchor(0), trandata_node5->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node3->GetOutDataAnchor(0), trandata_node6->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node3->GetOutDataAnchor(0), trandata_node7->GetInDataAnchor(0));

  GraphUtils::AddEdge(trandata_node4->GetOutDataAnchor(0), trandata_node8->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node5->GetOutDataAnchor(0), trandata_node9->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node6->GetOutDataAnchor(0), trandata_node10->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node7->GetOutDataAnchor(0), trandata_node11->GetInDataAnchor(0));

  GraphUtils::AddEdge(trandata_node8->GetOutDataAnchor(0), dst_node1->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node9->GetOutDataAnchor(0), dst_node2->GetInDataAnchor(0));

  GraphUtils::AddEdge(trandata_node10->GetOutDataAnchor(0), dst_node3->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node11->GetOutDataAnchor(0), dst_node4->GetInDataAnchor(0));

  TransNodeMerging trans_op_merger;
  trans_op_merger.MergeAllTransOps(*(graph.get()), {}, false);
  uint32_t count_node = 0;
  for (auto node: graph->GetDirectNode()) {
    EXPECT_NE(node, nullptr);
    string name = node->GetName();

    if (name == "A") {
      EXPECT_EQ(node->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode()->GetName(), "Cast1");

    }
    if (name == "B1") {
      EXPECT_EQ(node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "Cast8");

    }
    if (name == "B2") {
      EXPECT_EQ(node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "Cast9");
    }

    if (name == "B3") {
      EXPECT_EQ(node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "Cast10");
    }

    if (name == "B4") {
      EXPECT_EQ(node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "Cast11");
    }

    count_node++;
  }
  EXPECT_EQ(count_node,16);
}


TEST_F(UTEST_FE_TRANSOP_INSERT_COMPLEX, MergeMultipleCastOp6) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  GeTensorDesc tensor_desc_16(GeShape({128, 256, 512, 1024}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
  GeTensorDesc tensor_desc_32(GeShape({128, 256, 512, 1024}), ge::FORMAT_NCHW, ge::DT_FLOAT);
  GeTensorDesc tensor_desc_int32(GeShape({128, 256, 512, 1024}), ge::FORMAT_NCHW, ge::DT_INT32);
  GeTensorDesc tensor_desc_int16(GeShape({128, 256, 512, 1024}), ge::FORMAT_NCHW, ge::DT_INT16);
  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  src_op->AddOutputDesc(tensor_desc_32);
  src_op->AddInputDesc(tensor_desc_32);
  auto src_node = graph->AddNode(src_op);

  OpDescPtr dst_op = std::make_shared<OpDesc>("B1", "B");
  dst_op->AddInputDesc(tensor_desc_32);
  dst_op->AddOutputDesc(tensor_desc_32);
  auto dst_node1 = graph->AddNode(dst_op);

  OpDescPtr dst_op2 = std::make_shared<OpDesc>("B2", "B");
  dst_op2->AddInputDesc(tensor_desc_32);
  dst_op2->AddOutputDesc(tensor_desc_32);
  auto dst_node2 = graph->AddNode(dst_op2);

  OpDescPtr dst_op3 = std::make_shared<OpDesc>("B3", "B");
  dst_op3->AddInputDesc(tensor_desc_32);
  dst_op3->AddOutputDesc(tensor_desc_32);
  auto dst_node3 = graph->AddNode(dst_op3);

  OpDescPtr dst_op4 = std::make_shared<OpDesc>("B4", "B");
  dst_op4->AddInputDesc(tensor_desc_32);
  dst_op4->AddOutputDesc(tensor_desc_32);
  auto dst_node4 = graph->AddNode(dst_op4);

  OpDescPtr dst_op_cast1 = std::make_shared<OpDesc>("Cast1", "Cast");
  dst_op_cast1->AddInputDesc(tensor_desc_32);
  dst_op_cast1->AddOutputDesc(tensor_desc_16);
  auto trandata_node1 = graph->AddNode(dst_op_cast1);

  OpDescPtr dst_op_cast2 = std::make_shared<OpDesc>("Cast2", "Cast");
  dst_op_cast2->AddInputDesc(tensor_desc_16);
  dst_op_cast2->AddOutputDesc(tensor_desc_int32);
  auto trandata_node2 = graph->AddNode(dst_op_cast2);

  OpDescPtr dst_op_cast3 = std::make_shared<OpDesc>("Cast3", "Cast");
  dst_op_cast3->AddInputDesc(tensor_desc_16);
  dst_op_cast3->AddOutputDesc(tensor_desc_int32);
  auto trandata_node3 = graph->AddNode(dst_op_cast3);

  OpDescPtr dst_op_cast4 = std::make_shared<OpDesc>("Cast4", "Cast");
  dst_op_cast4->AddInputDesc(tensor_desc_int32);
  dst_op_cast4->AddOutputDesc(tensor_desc_16);
  auto trandata_node4 = graph->AddNode(dst_op_cast4);

  OpDescPtr dst_op_cast5 = std::make_shared<OpDesc>("Cast5", "Cast");
  dst_op_cast5->AddInputDesc(tensor_desc_int32);
  dst_op_cast5->AddOutputDesc(tensor_desc_int16);
  auto trandata_node5 = graph->AddNode(dst_op_cast5);

  OpDescPtr dst_op_cast6 = std::make_shared<OpDesc>("Cast6", "Cast");
  dst_op_cast6->AddInputDesc(tensor_desc_int32);
  dst_op_cast6->AddOutputDesc(tensor_desc_int16);
  auto trandata_node6 = graph->AddNode(dst_op_cast6);

  OpDescPtr dst_op_cast7 = std::make_shared<OpDesc>("Cast7", "Cast");
  dst_op_cast7->AddInputDesc(tensor_desc_int32);
  dst_op_cast7->AddOutputDesc(tensor_desc_16);
  auto trandata_node7 = graph->AddNode(dst_op_cast7);

  OpDescPtr dst_op_cast8 = std::make_shared<OpDesc>("Cast8", "Cast");
  dst_op_cast8->AddInputDesc(tensor_desc_16);
  dst_op_cast8->AddOutputDesc(tensor_desc_32);
  auto trandata_node8 = graph->AddNode(dst_op_cast8);

  OpDescPtr dst_op_cast9 = std::make_shared<OpDesc>("Cast9", "Cast");
  dst_op_cast9->AddInputDesc(tensor_desc_int16);
  dst_op_cast9->AddOutputDesc(tensor_desc_int32);
  auto trandata_node9 = graph->AddNode(dst_op_cast9);

  OpDescPtr dst_op_cast10 = std::make_shared<OpDesc>("Cast10", "Cast");
  dst_op_cast10->AddInputDesc(tensor_desc_int16);
  dst_op_cast10->AddOutputDesc(tensor_desc_int32);
  auto trandata_node10 = graph->AddNode(dst_op_cast10);

  OpDescPtr dst_op_cast11 = std::make_shared<OpDesc>("Cast11", "Cast");
  dst_op_cast11->AddInputDesc(tensor_desc_16);
  dst_op_cast11->AddOutputDesc(tensor_desc_32);
  auto trandata_node11 = graph->AddNode(dst_op_cast11);

  OpDescPtr dst_op_cast12 = std::make_shared<OpDesc>("Cast12", "Cast");
  dst_op_cast12->AddInputDesc(tensor_desc_int32);
  dst_op_cast12->AddOutputDesc(tensor_desc_16);
  auto trandata_node12 = graph->AddNode(dst_op_cast12);

  OpDescPtr dst_op_cast13 = std::make_shared<OpDesc>("Cast13", "Cast");
  dst_op_cast13->AddInputDesc(tensor_desc_16);
  dst_op_cast13->AddOutputDesc(tensor_desc_32);
  auto trandata_node13 = graph->AddNode(dst_op_cast13);

  OpDescPtr dst_op_cast14 = std::make_shared<OpDesc>("Cast14", "Cast");
  dst_op_cast14->AddInputDesc(tensor_desc_int32);
  dst_op_cast14->AddOutputDesc(tensor_desc_16);
  auto trandata_node14 = graph->AddNode(dst_op_cast14);

  OpDescPtr dst_op_cast15 = std::make_shared<OpDesc>("Cast15", "Cast");
  dst_op_cast15->AddInputDesc(tensor_desc_16);
  dst_op_cast15->AddOutputDesc(tensor_desc_32);
  auto trandata_node15 = graph->AddNode(dst_op_cast15);
  GraphUtils::AddEdge(src_node->GetOutDataAnchor(0), trandata_node1->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node1->GetOutDataAnchor(0), trandata_node2->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node1->GetOutDataAnchor(0), trandata_node3->GetInDataAnchor(0));

  GraphUtils::AddEdge(trandata_node2->GetOutDataAnchor(0), trandata_node4->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node2->GetOutDataAnchor(0), trandata_node5->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node3->GetOutDataAnchor(0), trandata_node6->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node3->GetOutDataAnchor(0), trandata_node7->GetInDataAnchor(0));

  GraphUtils::AddEdge(trandata_node4->GetOutDataAnchor(0), trandata_node8->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node5->GetOutDataAnchor(0), trandata_node9->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node6->GetOutDataAnchor(0), trandata_node10->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node7->GetOutDataAnchor(0), trandata_node11->GetInDataAnchor(0));

  GraphUtils::AddEdge(trandata_node9->GetOutDataAnchor(0), trandata_node12->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node12->GetOutDataAnchor(0), trandata_node13->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node10->GetOutDataAnchor(0), trandata_node14->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node14->GetOutDataAnchor(0), trandata_node15->GetInDataAnchor(0));

  GraphUtils::AddEdge(trandata_node8->GetOutDataAnchor(0), dst_node1->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node13->GetOutDataAnchor(0), dst_node2->GetInDataAnchor(0));

  GraphUtils::AddEdge(trandata_node15->GetOutDataAnchor(0), dst_node3->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node11->GetOutDataAnchor(0), dst_node4->GetInDataAnchor(0));

  TransNodeMerging trans_op_merger;
  trans_op_merger.MergeAllTransOps(*(graph.get()), {}, true);
  uint32_t count_node = 0;
  for (auto node: graph->GetDirectNode()) {
    EXPECT_NE(node, nullptr);
    string name = node->GetName();

    if (name == "A") {
      EXPECT_EQ(node->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode()->GetName(), "B1");
      EXPECT_EQ(node->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(1)->GetOwnerNode()->GetName(), "B2");
      EXPECT_EQ(node->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(2)->GetOwnerNode()->GetName(), "B3");
      EXPECT_EQ(node->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(3)->GetOwnerNode()->GetName(), "B4");

    }
    if (name == "B1") {
      EXPECT_EQ(node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "A");

    }
    if (name == "B2") {
      EXPECT_EQ(1, node->GetOpDesc()->GetSrcName().size());
      EXPECT_EQ(1, node->GetOpDesc()->GetInputName().size());

      EXPECT_EQ("A", node->GetOpDesc()->GetSrcName().at(0));
      EXPECT_EQ("A", node->GetOpDesc()->GetInputName().at(0));
      EXPECT_EQ(node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "A");
    }

    if (name == "B3") {
      EXPECT_EQ(1, node->GetOpDesc()->GetSrcName().size());
      EXPECT_EQ(1, node->GetOpDesc()->GetInputName().size());

      EXPECT_EQ("A", node->GetOpDesc()->GetSrcName().at(0));
      EXPECT_EQ("A", node->GetOpDesc()->GetInputName().at(0));
      EXPECT_EQ(node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "A");
    }

    if (name == "B4") {
      EXPECT_EQ(node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "A");
    }

    count_node++;
  }
  EXPECT_EQ(count_node,5);
}


TEST_F(UTEST_FE_TRANSOP_INSERT_COMPLEX, MergeMultipleCastOp7) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  GeTensorDesc tensor_desc_16(GeShape({128, 256, 512, 1024}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
  GeTensorDesc tensor_desc_32(GeShape({128, 256, 512, 1024}), ge::FORMAT_NCHW, ge::DT_FLOAT);
  GeTensorDesc tensor_desc_int32(GeShape({128, 256, 512, 1024}), ge::FORMAT_NCHW, ge::DT_INT32);
  GeTensorDesc tensor_desc_int16(GeShape({128, 256, 512, 1024}), ge::FORMAT_NCHW, ge::DT_INT16);
  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  src_op->AddOutputDesc(tensor_desc_32);
  src_op->AddInputDesc(tensor_desc_32);
  auto src_node = graph->AddNode(src_op);

  OpDescPtr dst_op = std::make_shared<OpDesc>("B1", "B");
  dst_op->AddInputDesc(tensor_desc_32);
  dst_op->AddOutputDesc(tensor_desc_32);
  auto dst_node1 = graph->AddNode(dst_op);

  OpDescPtr dst_op_cast1 = std::make_shared<OpDesc>("Cast1", "Cast");
  dst_op_cast1->AddInputDesc(tensor_desc_32);
  dst_op_cast1->AddOutputDesc(tensor_desc_16);
  auto trandata_node1 = graph->AddNode(dst_op_cast1);

  OpDescPtr dst_op_cast2 = std::make_shared<OpDesc>("Cast2", "Cast");
  dst_op_cast2->AddInputDesc(tensor_desc_16);
  dst_op_cast2->AddOutputDesc(tensor_desc_32);
  auto trandata_node2 = graph->AddNode(dst_op_cast2);

  OpDescPtr dst_op_cast3 = std::make_shared<OpDesc>("Cast3", "Cast");
  dst_op_cast3->AddInputDesc(tensor_desc_32);
  dst_op_cast3->AddOutputDesc(tensor_desc_16);
  auto trandata_node3 = graph->AddNode(dst_op_cast3);

  OpDescPtr dst_op_cast4 = std::make_shared<OpDesc>("Cast4", "Cast");
  dst_op_cast4->AddInputDesc(tensor_desc_16);
  dst_op_cast4->AddOutputDesc(tensor_desc_32);
  auto trandata_node4 = graph->AddNode(dst_op_cast4);

  OpDescPtr dst_op_reshape5 = std::make_shared<OpDesc>("Reshape", "Reshape");
  dst_op_reshape5->AddInputDesc(tensor_desc_32);
  dst_op_reshape5->AddOutputDesc(tensor_desc_32);
  auto trandata_node5 = graph->AddNode(dst_op_reshape5);

  GraphUtils::AddEdge(src_node->GetOutDataAnchor(0), trandata_node1->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node1->GetOutDataAnchor(0), trandata_node2->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node2->GetOutDataAnchor(0), trandata_node3->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node3->GetOutDataAnchor(0), trandata_node4->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node4->GetOutDataAnchor(0), trandata_node5->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node5->GetOutDataAnchor(0), dst_node1->GetInDataAnchor(0));

  TransNodeMerging trans_op_merger;
  trans_op_merger.MergeAllTransOps(*(graph.get()), {}, false);
  uint32_t count_node = 0;
  for (auto node: graph->GetDirectNode()) {
    EXPECT_NE(node, nullptr);
    string name = node->GetName();

    if (name == "A") {
      EXPECT_EQ(node->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode()->GetName(), "Reshape");

    }
    if (name == "B1") {
      EXPECT_EQ(node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "Reshape");

    }
    if (name == "Reshape") {
      EXPECT_EQ(1, node->GetOpDesc()->GetSrcName().size());
      EXPECT_EQ(1, node->GetOpDesc()->GetInputName().size());

      EXPECT_EQ("A", node->GetOpDesc()->GetSrcName().at(0));
      EXPECT_EQ("A", node->GetOpDesc()->GetInputName().at(0));
      EXPECT_EQ(node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "A");
    }

    count_node++;
  }
  EXPECT_EQ(count_node,3);
}

/* FRACTAL_NZ(fp16) -> NCDHW(fp16)
 * The Program will insert Transdata->Reformat. */
TEST_F(UTEST_FE_TRANSOP_INSERT_COMPLEX, InsertAllTransopNzToNcdhw) {
  // src:cce op, dst:cce op
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  GeTensorDesc src_tensor_desc(GeShape({100, 2, 3, 16, 16}), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
  src_tensor_desc.SetOriginShape(GeShape({100, 48, 32}));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NCDHW);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);
  ge::AttrUtils::SetInt(src_op, FE_IMPLY_TYPE, 6);

  OpDescPtr dst_op = std::make_shared<OpDesc>("FF", "FF");
  GeTensorDesc dst_tensor_desc(GeShape({100, 48, 32}), ge::FORMAT_NCDHW, ge::DT_FLOAT); //Reshape type nw
  dst_tensor_desc.SetOriginShape(GeShape({100, 48, 32}));
  dst_tensor_desc.SetOriginFormat(ge::FORMAT_NCDHW);
  dst_op->AddInputDesc(dst_tensor_desc);
  dst_op->AddOutputDesc(dst_tensor_desc);
  auto dst_node = graph->AddNode(dst_op);
  ge::AttrUtils::SetInt(dst_op, FE_IMPLY_TYPE, 6);
  vector<bool> input_const_vector = {false};
  dst_op->SetIsInputConst(input_const_vector);

  GraphUtils::AddEdge(src_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(0));
  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status status = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));

  int count_node = 0;
  ASSERT_EQ(fe::SUCCESS, status);
  for (auto node : graph->GetDirectNode()) {
    ASSERT_NE(node, nullptr);
    count_node++;
    if (node->GetType() == TRANSDATA) {
      auto output = node->GetOpDesc()->GetOutputDescPtr(0);
      const ge::GeShape &shape = output->GetShape();
      ASSERT_EQ(shape.GetDimNum(), 3);
      EXPECT_EQ(shape.GetDim(0), 100);
      EXPECT_EQ(shape.GetDim(1), 48);
      EXPECT_EQ(shape.GetDim(2), 32);
      EXPECT_EQ(output->GetFormat(), ge::FORMAT_ND);
    }
    if (node->GetType() == fe::REFORMAT) {
      auto input = node->GetOpDesc()->GetInputDescPtr(0);
      const ge::GeShape &shape_in = input->GetShape();
      ASSERT_EQ(shape_in.GetDimNum(), 3);
      EXPECT_EQ(shape_in.GetDim(0), 100);
      EXPECT_EQ(shape_in.GetDim(1), 48);
      EXPECT_EQ(shape_in.GetDim(2), 32);
      EXPECT_EQ(input->GetFormat(), ge::FORMAT_ND);

      auto output = node->GetOpDesc()->GetOutputDescPtr(0);
      const ge::GeShape &shape_out = output->GetShape();
      ASSERT_EQ(shape_out.GetDimNum(), 3);
      EXPECT_EQ(shape_out.GetDim(0), 100);
      EXPECT_EQ(shape_out.GetDim(1), 48);
      EXPECT_EQ(shape_out.GetDim(2), 32);
      EXPECT_EQ(output->GetFormat(), ge::FORMAT_NCDHW);
    }

  }
  EXPECT_EQ(count_node, 5);
}

/* FRACTAL_NZ(fp16) -> NDHWC(fp16)
 * The Program will insert Transdata->Reformat. */
TEST_F(UTEST_FE_TRANSOP_INSERT_COMPLEX, InsertAllTransopNzToNdhwc) {
  // src:cce op, dst:cce op
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  GeTensorDesc src_tensor_desc(GeShape({100, 2, 3, 16, 16}), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
  src_tensor_desc.SetOriginShape(GeShape({100, 48, 32}));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NDHWC);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);
  ge::AttrUtils::SetInt(src_op, FE_IMPLY_TYPE, 6);

  OpDescPtr dst_op = std::make_shared<OpDesc>("FF", "FF");
  GeTensorDesc dst_tensor_desc(GeShape({100, 48, 32}), ge::FORMAT_NDHWC, ge::DT_FLOAT); //Reshape type nw
  dst_tensor_desc.SetOriginShape(GeShape({100, 48, 32}));
  dst_tensor_desc.SetOriginFormat(ge::FORMAT_NDHWC);
  dst_op->AddInputDesc(dst_tensor_desc);
  dst_op->AddOutputDesc(dst_tensor_desc);
  auto dst_node = graph->AddNode(dst_op);
  ge::AttrUtils::SetInt(dst_op, FE_IMPLY_TYPE, 6);
  vector<bool> input_const_vector = {false};
  dst_op->SetIsInputConst(input_const_vector);

  GraphUtils::AddEdge(src_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(0));
  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status status = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));

  int count_node = 0;
  ASSERT_EQ(fe::SUCCESS, status);
  for (auto node : graph->GetDirectNode()) {
    ASSERT_NE(node, nullptr);
    count_node++;
    if (node->GetType() == TRANSDATA) {
      auto output = node->GetOpDesc()->GetOutputDescPtr(0);
      const auto &shape = output->GetShape();
      ASSERT_EQ(shape.GetDimNum(), 3);
      EXPECT_EQ(shape.GetDim(0), 100);
      EXPECT_EQ(shape.GetDim(1), 48);
      EXPECT_EQ(shape.GetDim(2), 32);
      EXPECT_EQ(output->GetFormat(), ge::FORMAT_ND);
    }
    if (node->GetType() == fe::REFORMAT) {
      auto input = node->GetOpDesc()->GetInputDescPtr(0);
      const auto &shape_in = input->GetShape();
      ASSERT_EQ(shape_in.GetDimNum(), 3);
      EXPECT_EQ(shape_in.GetDim(0), 100);
      EXPECT_EQ(shape_in.GetDim(1), 48);
      EXPECT_EQ(shape_in.GetDim(2), 32);
      EXPECT_EQ(input->GetFormat(), ge::FORMAT_ND);

      auto output = node->GetOpDesc()->GetOutputDescPtr(0);
      const auto &shape_out = output->GetShape();
      ASSERT_EQ(shape_out.GetDimNum(), 3);
      EXPECT_EQ(shape_out.GetDim(0), 100);
      EXPECT_EQ(shape_out.GetDim(1), 48);
      EXPECT_EQ(shape_out.GetDim(2), 32);
      EXPECT_EQ(output->GetFormat(), ge::FORMAT_NDHWC);
    }

  }
  EXPECT_EQ(count_node, 5);
}

/* NDHWC(fp16) -> FRACTAL_NZ(fp16)
 * The Program will insert Reformat->Transdata. */
TEST_F(UTEST_FE_TRANSOP_INSERT_COMPLEX, InsertAllTransopNdhwcToNz) {
  // src:cce op, dst:cce op
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  GeTensorDesc src_tensor_desc(GeShape({100, 48, 32}), ge::FORMAT_NDHWC, ge::DT_FLOAT);
  src_tensor_desc.SetOriginShape(GeShape({100, 48, 32}));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NDHWC);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);
  ge::AttrUtils::SetInt(src_op, FE_IMPLY_TYPE, 6);

  OpDescPtr dst_op = std::make_shared<OpDesc>("FF", "FF");
  GeTensorDesc dst_tensor_desc(GeShape({100, 2, 3, 16, 16}), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
  dst_tensor_desc.SetOriginShape(GeShape({100, 48, 32}));
  dst_tensor_desc.SetOriginFormat(ge::FORMAT_NDHWC);
  dst_op->AddInputDesc(dst_tensor_desc);
  dst_op->AddOutputDesc(dst_tensor_desc);
  auto dst_node = graph->AddNode(dst_op);
  ge::AttrUtils::SetInt(dst_op, FE_IMPLY_TYPE, 6);
  vector<bool> input_const_vector = {false};
  dst_op->SetIsInputConst(input_const_vector);

  GraphUtils::AddEdge(src_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(0));
  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status status = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));

  int count_node = 0;
  ASSERT_EQ(fe::SUCCESS, status);
  for (auto node : graph->GetDirectNode()) {
    ASSERT_NE(node, nullptr);
    count_node++;
    if (node->GetType() == TRANSDATA) {
      auto input = node->GetOpDesc()->GetInputDescPtr(0);
      const ge::GeShape &shape = input->GetShape();
      ASSERT_EQ(shape.GetDimNum(), 3);
      EXPECT_EQ(shape.GetDim(0), 100);
      EXPECT_EQ(shape.GetDim(1), 48);
      EXPECT_EQ(shape.GetDim(2), 32);
      EXPECT_EQ(input->GetFormat(), ge::FORMAT_ND);

      auto output = node->GetOpDesc()->GetOutputDescPtr(0);
      const ge::GeShape &output_shape = output->GetShape();
      ASSERT_EQ(output_shape.GetDimNum(), 5);
      EXPECT_EQ(output_shape.GetDim(0), 100);
      EXPECT_EQ(output_shape.GetDim(1), 2);
      EXPECT_EQ(output_shape.GetDim(2), 3);
      EXPECT_EQ(output_shape.GetDim(3), 16);
      EXPECT_EQ(output_shape.GetDim(4), 16);
      EXPECT_EQ(GetPrimaryFormat(output->GetFormat()), ge::FORMAT_FRACTAL_NZ);
    }
    if (node->GetType() == fe::REFORMAT) {
      auto input = node->GetOpDesc()->GetInputDescPtr(0);
      const ge::GeShape &shape_in = input->GetShape();
      ASSERT_EQ(shape_in.GetDimNum(), 3);
      EXPECT_EQ(shape_in.GetDim(0), 100);
      EXPECT_EQ(shape_in.GetDim(1), 48);
      EXPECT_EQ(shape_in.GetDim(2), 32);
      EXPECT_EQ(input->GetFormat(), ge::FORMAT_NDHWC);

      auto output = node->GetOpDesc()->GetOutputDescPtr(0);
      const ge::GeShape &shape_out = output->GetShape();
      ASSERT_EQ(shape_out.GetDimNum(), 3);
      EXPECT_EQ(shape_out.GetDim(0), 100);
      EXPECT_EQ(shape_out.GetDim(1), 48);
      EXPECT_EQ(shape_out.GetDim(2), 32);
      EXPECT_EQ(output->GetFormat(), ge::FORMAT_ND);
    }

  }
  EXPECT_EQ(count_node, 5);
}

/* Ncdhw(fp16) -> FRACTAL_NZ(fp16)
 * The Program will insert Reformat->Transdata. */
TEST_F(UTEST_FE_TRANSOP_INSERT_COMPLEX, InsertAllTransopNcdhwToNz) {
  // src:cce op, dst:cce op
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  GeTensorDesc src_tensor_desc(GeShape({100, 48, 32}), ge::FORMAT_NCDHW, ge::DT_FLOAT);
  src_tensor_desc.SetOriginShape(GeShape({100, 48, 32}));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NCDHW);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);
  ge::AttrUtils::SetInt(src_op, FE_IMPLY_TYPE, 6);

  OpDescPtr dst_op = std::make_shared<OpDesc>("FF", "FF");
  GeTensorDesc dst_tensor_desc(GeShape({100, 2, 3, 16, 16}), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
  dst_tensor_desc.SetOriginShape(GeShape({100, 48, 32}));
  dst_tensor_desc.SetOriginFormat(ge::FORMAT_NCDHW);
  dst_op->AddInputDesc(dst_tensor_desc);
  dst_op->AddOutputDesc(dst_tensor_desc);
  auto dst_node = graph->AddNode(dst_op);
  ge::AttrUtils::SetInt(dst_op, FE_IMPLY_TYPE, 6);
  vector<bool> input_const_vector = {false};
  dst_op->SetIsInputConst(input_const_vector);

  GraphUtils::AddEdge(src_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(0));
  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status status = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));

  int count_node = 0;
  ASSERT_EQ(fe::SUCCESS, status);
  for (auto node : graph->GetDirectNode()) {
    ASSERT_NE(node, nullptr);
    count_node++;
    if (node->GetType() == TRANSDATA) {
      auto input = node->GetOpDesc()->GetInputDescPtr(0);
      const ge::GeShape &shape = input->GetShape();
      ASSERT_EQ(shape.GetDimNum(), 3);
      EXPECT_EQ(shape.GetDim(0), 100);
      EXPECT_EQ(shape.GetDim(1), 48);
      EXPECT_EQ(shape.GetDim(2), 32);
      EXPECT_EQ(input->GetFormat(), ge::FORMAT_ND);

      auto output = node->GetOpDesc()->GetOutputDescPtr(0);
      const ge::GeShape &output_shape = output->GetShape();
      ASSERT_EQ(output_shape.GetDimNum(), 5);
      EXPECT_EQ(output_shape.GetDim(0), 100);
      EXPECT_EQ(output_shape.GetDim(1), 2);
      EXPECT_EQ(output_shape.GetDim(2), 3);
      EXPECT_EQ(output_shape.GetDim(3), 16);
      EXPECT_EQ(output_shape.GetDim(4), 16);
      EXPECT_EQ(GetPrimaryFormat(output->GetFormat()), ge::FORMAT_FRACTAL_NZ);
    }
    if (node->GetType() == fe::REFORMAT) {
      auto input = node->GetOpDesc()->GetInputDescPtr(0);
      const ge::GeShape &shape_in = input->GetShape();
      ASSERT_EQ(shape_in.GetDimNum(), 3);
      EXPECT_EQ(shape_in.GetDim(0), 100);
      EXPECT_EQ(shape_in.GetDim(1), 48);
      EXPECT_EQ(shape_in.GetDim(2), 32);
      EXPECT_EQ(input->GetFormat(), ge::FORMAT_NCDHW);

      auto output = node->GetOpDesc()->GetOutputDescPtr(0);
      const ge::GeShape &shape_out = output->GetShape();
      ASSERT_EQ(shape_out.GetDimNum(), 3);
      EXPECT_EQ(shape_out.GetDim(0), 100);
      EXPECT_EQ(shape_out.GetDim(1), 48);
      EXPECT_EQ(shape_out.GetDim(2), 32);
      EXPECT_EQ(output->GetFormat(), ge::FORMAT_ND);
    }
  }
  EXPECT_EQ(count_node, 5);
}
