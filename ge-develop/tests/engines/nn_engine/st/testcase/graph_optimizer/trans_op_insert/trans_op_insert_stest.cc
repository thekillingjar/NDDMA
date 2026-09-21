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
#include "common/platform_utils.h"
#include "common/configuration.h"
#include "adapter/common/op_store_adapter_manager.h"
#include "adapter/tbe_adapter/tbe_op_store_adapter.h"
#include "ops_kernel_store/fe_ops_kernel_info_store.h"
#include "ops_store/ops_kernel_manager.h"
#include "graph_optimizer/shape_format_transfer/trans_node_manager/trans_node_manager.h"
#include "graph_optimizer/shape_format_transfer/trans_node_implementation/trans_node_base_generator.h"
#include "graph_optimizer/shape_format_transfer/trans_node_implementation/trans_node_cast_generator.h"
#include "graph_optimizer/shape_format_transfer/trans_node_implementation/trans_node_reshape_generator.h"
#include "graph_optimizer/shape_format_transfer/trans_node_implementation/trans_node_transpose_generator.h"
#include "graph_optimizer/shape_format_transfer/trans_node_implementation/trans_node_transdata_generator.h"
#include "graph/operator_factory_impl.h"

#undef protected
#undef private

#include <iostream>

using namespace std;
using namespace ge;
using namespace fe;



class STEST_FE_TRANSOP_INSERT : public testing::Test {
 protected:
  void SetUp()
  {
    std::map<std::string, std::string> options;
    fe_ops_kernel_info_store_ptr_ = make_shared<fe::FEOpsKernelInfoStore>();
    FEOpsStoreInfo tbe_custom {
        2,
        "tbe-custom",
        EN_IMPL_CUSTOM_TBE,
        GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/tbe_custom_opinfo",
        ""};
    vector<FEOpsStoreInfo> store_info;
    store_info.emplace_back(tbe_custom);
    Configuration::Instance(fe::AI_CORE_NAME).ops_store_info_vector_ = (store_info);
    OpsKernelManager::Instance(AI_CORE_NAME).Finalize();

    fe_ops_kernel_info_store_ptr_->Initialize(options);
  }

  void TearDown()
  {
    fe_ops_kernel_info_store_ptr_->Finalize();

  }

  shared_ptr<fe::FEOpsKernelInfoStore> fe_ops_kernel_info_store_ptr_;
};

Status QueryHighPrioOpImplTypeStubTbe(FEOpsKernelInfoStore* This, const ge::OpDescPtr& op_desc_ptr,
                                      OpImplType &impl_type, OpKernelInfoPtr &op_kernel_ptr) {
  impl_type = EN_IMPL_HW_TBE;
  return fe::SUCCESS;
}

Status QueryHighPrioOpImplTypeStubCce(FEOpsKernelInfoStore* This, const ge::OpDescPtr& op_desc_ptr,
                                      OpImplType &impl_type, OpKernelInfoPtr &op_kernel_ptr) {
  impl_type = EN_IMPL_HW_GENERAL_CCE;
  return fe::SUCCESS;
}

Status ConstructComputeGraph(ComputeGraphPtr graph, ComputeGraphPtr graph_check)
{
  return fe::SUCCESS;
}
bool CheckComputeGraph(ComputeGraphPtr graph, ComputeGraphPtr graph_check)
{
  //EXPECT_EQ(ge::GRAPH_SUCCESS, fused_graph.TopologicalSorting());
  return true;
}

TEST_F(STEST_FE_TRANSOP_INSERT, InsertTransDataNode_01)
{
  // src:cce op, dst:cce op
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  GeTensorDesc src_tensor_desc(GeShape({100, 256, 256, 512}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);
  ge::AttrUtils::SetInt(src_op, FE_IMPLY_TYPE, 6);

  OpDescPtr dst_op = std::make_shared<OpDesc>("B", "B");
  GeTensorDesc dst_tensor_desc(GeShape({1, 256, 256, 512}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
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
      EXPECT_EQ(shape.GetDimNum(), 4);
      EXPECT_EQ(shape.GetDims()[0], 100);
      EXPECT_EQ(shape.GetDims()[1], 256);
      EXPECT_EQ(shape.GetDims()[2], 256);
      EXPECT_EQ(shape.GetDims()[3], 512);
    }
    if (node->GetType() == "B") {
      ge::GeShape shape = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
      EXPECT_EQ(shape.GetDimNum(), 4);
      EXPECT_EQ(shape.GetDims()[0], 1);
      EXPECT_EQ(shape.GetDims()[1], 256);
      EXPECT_EQ(shape.GetDims()[2], 256);
      EXPECT_EQ(shape.GetDims()[3], 512);
    }
  }
  EXPECT_EQ(count_node, 3);

}

TEST_F(STEST_FE_TRANSOP_INSERT, InsertTransDataNode_02)
{
  // src:cce op, dst:cce op
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  ComputeGraphPtr graph_check =  std::make_shared<ComputeGraph>("test_graph_check");
  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  GeTensorDesc src_tensor_desc(GeShape({1, 256, 256, 512}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);
  ge::AttrUtils::SetInt(src_op, FE_IMPLY_TYPE, 6);

  OpDescPtr dst_op = std::make_shared<OpDesc>("B", "B");
  GeTensorDesc dst_tensor_desc(GeShape({1, 2, 3, 4}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
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
  ASSERT_EQ(fe::SUCCESS, status);
  int count_node = 0;
  for (auto node : graph->GetDirectNode()) {
    ASSERT_NE(node, nullptr);
    count_node++;
    if (node->GetType() == "B") {
      {
        ge::GeShape shape_check = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        auto size = shape_check.GetDimNum();
        ASSERT_EQ(size, 4);
        vector<int64_t> input_vec_of_b =shape_check.GetDims();
        EXPECT_EQ(input_vec_of_b[0], 1);
        EXPECT_EQ(input_vec_of_b[1], 2);
        EXPECT_EQ(input_vec_of_b[2], 3);
        EXPECT_EQ(input_vec_of_b[3], 4);
      }
      {
        ge::GeShape shape_check = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        auto size = shape_check.GetDimNum();
        ASSERT_EQ(size, 4);
        vector<int64_t> input_vec_of_b =shape_check.GetDims();
        EXPECT_EQ(input_vec_of_b[0], 1);
        EXPECT_EQ(input_vec_of_b[1], 2);
        EXPECT_EQ(input_vec_of_b[2], 3);
        EXPECT_EQ(input_vec_of_b[3], 4);
      }
    }
  }
  EXPECT_EQ(count_node, 3);
}

TEST_F(STEST_FE_TRANSOP_INSERT, InsertTransDataNode_05)
{
  // src:cce op, dst:cce op
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  ComputeGraphPtr graph_check =  std::make_shared<ComputeGraph>("test_graph_check");
  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  GeTensorDesc src_tensor_desc(GeShape({1, 256, 256, 512, 32}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  src_tensor_desc.SetOriginShape(ge::GeShape({123,456}));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NHWC);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);
  ge::AttrUtils::SetInt(src_op, FE_IMPLY_TYPE, 6);

  OpDescPtr dst_op = std::make_shared<OpDesc>("B", "B");
  GeTensorDesc dst_tensor_desc(GeShape({1, 2, 3, 4}), ge::FORMAT_NHWC, ge::DT_FLOAT16);
  dst_tensor_desc.SetOriginShape(ge::GeShape({123,456}));
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
  ASSERT_EQ(fe::SUCCESS, status);
  int count_node = 0;
  for (auto node : graph->GetDirectNode()) {
    ASSERT_NE(node, nullptr);
    count_node++;
    if (node->GetType() == "B") {
      {
        ge::GeShape shape_check = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        auto size = shape_check.GetDimNum();
        ASSERT_EQ(size, 4);
        vector<int64_t> input_vec_of_b =shape_check.GetDims();
        EXPECT_EQ(input_vec_of_b[0], 1);
        EXPECT_EQ(input_vec_of_b[1], 2);
        EXPECT_EQ(input_vec_of_b[2], 3);
        EXPECT_EQ(input_vec_of_b[3], 4);
      }
      {
        ge::GeShape shape_check = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        auto size = shape_check.GetDimNum();
        ASSERT_EQ(size, 4);
        vector<int64_t> input_vec_of_b =shape_check.GetDims();
        EXPECT_EQ(input_vec_of_b[0], 1);
        EXPECT_EQ(input_vec_of_b[1], 2);
        EXPECT_EQ(input_vec_of_b[2], 3);
        EXPECT_EQ(input_vec_of_b[3], 4);
      }
    }
    if (node->GetType() == "TransData") {
      {
        ge::GeShape shape_check = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        auto size = shape_check.GetDimNum();
        ASSERT_EQ(size, 5);
        vector<int64_t> input_vec_of_b =shape_check.GetDims();
        EXPECT_EQ(input_vec_of_b[0], 1);
        EXPECT_EQ(input_vec_of_b[1], 256);
        EXPECT_EQ(input_vec_of_b[2], 256);
        EXPECT_EQ(input_vec_of_b[3], 512);
        EXPECT_EQ(input_vec_of_b[4], 32);
      }
      {
        ge::GeShape shape_check = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        auto size = shape_check.GetDimNum();
        ASSERT_EQ(size, 4);
        vector<int64_t> input_vec_of_b =shape_check.GetDims();
        EXPECT_EQ(input_vec_of_b[0], 1);
        EXPECT_EQ(input_vec_of_b[1], 123);
        EXPECT_EQ(input_vec_of_b[2], 456);
        EXPECT_EQ(input_vec_of_b[3], 1);
      }

    }
  }
  EXPECT_EQ(count_node, 3);
}

TEST_F(STEST_FE_TRANSOP_INSERT, InsertTransDataNode_06)
{
  // src:cce op, dst:cce op
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  ComputeGraphPtr graph_check =  std::make_shared<ComputeGraph>("test_graph_check");
  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  GeTensorDesc src_tensor_desc(GeShape({1, 256, 256, 512, 32}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  src_tensor_desc.SetOriginShape(ge::GeShape({123,456}));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);
  ge::AttrUtils::SetInt(src_op, FE_IMPLY_TYPE, 6);

  OpDescPtr dst_op = std::make_shared<OpDesc>("B", "B");
  GeTensorDesc dst_tensor_desc(GeShape({1, 2, 3, 4}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
  dst_tensor_desc.SetOriginShape(ge::GeShape({123,456}));
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
  ASSERT_EQ(fe::SUCCESS, status);
  int count_node = 0;
  for (auto node : graph->GetDirectNode()) {
    ASSERT_NE(node, nullptr);
    count_node++;
    if (node->GetType() == "B") {
      {
        ge::GeShape shape_check = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        auto size = shape_check.GetDimNum();
        ASSERT_EQ(size, 4);
        vector<int64_t> input_vec_of_b =shape_check.GetDims();
        EXPECT_EQ(input_vec_of_b[0], 1);
        EXPECT_EQ(input_vec_of_b[1], 2);
        EXPECT_EQ(input_vec_of_b[2], 3);
        EXPECT_EQ(input_vec_of_b[3], 4);
      }
      {
        ge::GeShape shape_check = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        auto size = shape_check.GetDimNum();
        ASSERT_EQ(size, 4);
        vector<int64_t> input_vec_of_b =shape_check.GetDims();
        EXPECT_EQ(input_vec_of_b[0], 1);
        EXPECT_EQ(input_vec_of_b[1], 2);
        EXPECT_EQ(input_vec_of_b[2], 3);
        EXPECT_EQ(input_vec_of_b[3], 4);
      }
    }
    if (node->GetType() == "TransData") {
      {
        ge::GeShape shape_check = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        auto size = shape_check.GetDimNum();
        ASSERT_EQ(size, 5);
        vector<int64_t> input_vec_of_b =shape_check.GetDims();
        EXPECT_EQ(input_vec_of_b[0], 1);
        EXPECT_EQ(input_vec_of_b[1], 256);
        EXPECT_EQ(input_vec_of_b[2], 256);
        EXPECT_EQ(input_vec_of_b[3], 512);
        EXPECT_EQ(input_vec_of_b[4], 32);
      }
      {
        ge::GeShape shape_check = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        auto size = shape_check.GetDimNum();
        ASSERT_EQ(size, 4);
        vector<int64_t> input_vec_of_b =shape_check.GetDims();
        EXPECT_EQ(input_vec_of_b[0], 1);
        EXPECT_EQ(input_vec_of_b[1], 123);
        EXPECT_EQ(input_vec_of_b[2], 456);
        EXPECT_EQ(input_vec_of_b[3], 1);
      }

    }
  }
  EXPECT_EQ(count_node, 3);
}


TEST_F(STEST_FE_TRANSOP_INSERT, InsertTransDataNode_07)
{
  // src:cce op, dst:cce op
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  ComputeGraphPtr graph_check =  std::make_shared<ComputeGraph>("test_graph_check");
  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  GeTensorDesc src_tensor_desc(GeShape({1, 256, 256, 512, 32}), ge::FORMAT_FRACTAL_Z, ge::DT_FLOAT16);
  src_tensor_desc.SetOriginShape(GeShape({100,200,100,200}));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);
  ge::AttrUtils::SetInt(src_op, FE_IMPLY_TYPE, 6);

  OpDescPtr dst_op = std::make_shared<OpDesc>("B", "B");
  GeTensorDesc dst_tensor_desc(GeShape({1, 2, 3, 4}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
  dst_tensor_desc.SetOriginShape(GeShape({1, 2, 3, 4}));
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
  ASSERT_EQ(fe::SUCCESS, status);
  int count_node = 0;
  for (auto node : graph->GetDirectNode()) {
    ASSERT_NE(node, nullptr);
    count_node++;
    if (node->GetType() == "B") {
      {
        ge::GeShape shape_check = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        auto size = shape_check.GetDimNum();
        ASSERT_EQ(size, 4);
        vector<int64_t> input_vec_of_b =shape_check.GetDims();
        EXPECT_EQ(input_vec_of_b[0], 1);
        EXPECT_EQ(input_vec_of_b[1], 2);
        EXPECT_EQ(input_vec_of_b[2], 3);
        EXPECT_EQ(input_vec_of_b[3], 4);
      }
      {
        ge::GeShape shape_check = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        auto size = shape_check.GetDimNum();
        ASSERT_EQ(size, 4);
        vector<int64_t> input_vec_of_b =shape_check.GetDims();
        EXPECT_EQ(input_vec_of_b[0], 1);
        EXPECT_EQ(input_vec_of_b[1], 2);
        EXPECT_EQ(input_vec_of_b[2], 3);
        EXPECT_EQ(input_vec_of_b[3], 4);
      }
    }
    if (node->GetType() == "TransData") {
      {
        ge::GeShape shape_check = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        auto size = shape_check.GetDimNum();
        ASSERT_EQ(size, 4);
        vector<int64_t> input_vec_of_b =shape_check.GetDims();
        EXPECT_EQ(input_vec_of_b[0], 260000);
        EXPECT_EQ(input_vec_of_b[1], 7);
        EXPECT_EQ(input_vec_of_b[2], 16);
        EXPECT_EQ(input_vec_of_b[3], 16);
      }
      {
        ge::GeShape shape_check = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        auto size = shape_check.GetDimNum();
        ASSERT_EQ(size, 4);
        vector<int64_t> input_vec_of_b =shape_check.GetDims();
        EXPECT_EQ(input_vec_of_b[0], 100);
        EXPECT_EQ(input_vec_of_b[1], 200);
        EXPECT_EQ(input_vec_of_b[2], 100);
        EXPECT_EQ(input_vec_of_b[3], 200);
      }

    }
  }
  EXPECT_EQ(count_node, 3);
}

TEST_F(STEST_FE_TRANSOP_INSERT, InsertTransDataNode_07_1)
{
  // src:cce op, dst:cce op
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  ComputeGraphPtr graph_check =  std::make_shared<ComputeGraph>("test_graph_check");
  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  GeTensorDesc src_tensor_desc(GeShape({100,200}), ge::FORMAT_HWCN, ge::DT_FLOAT16);
  src_tensor_desc.SetOriginShape(GeShape({100,200}));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_HWCN);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);
  ge::AttrUtils::SetInt(src_op, FE_IMPLY_TYPE, 6);

  OpDescPtr dst_op = std::make_shared<OpDesc>("BB", "BB");
  GeTensorDesc dst_tensor_desc(GeShape({7, 13, 16, 16}), ge::FORMAT_FRACTAL_Z, ge::DT_FLOAT16);
  dst_tensor_desc.SetOriginShape(GeShape({100, 200}));
  dst_tensor_desc.SetOriginFormat(ge::FORMAT_HWCN);
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
  ASSERT_EQ(fe::SUCCESS, status);
  int count_node = 0;
  for (auto node : graph->GetDirectNode()) {
    ASSERT_NE(node, nullptr);
    count_node++;
    if (node->GetType() == "TransData") {
      {
        ge::GeShape shape_check = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        auto size = shape_check.GetDimNum();
        ASSERT_EQ(size, 4);
        vector<int64_t> input_vec_of_b =shape_check.GetDims();
        EXPECT_EQ(input_vec_of_b[0], 1);
        EXPECT_EQ(input_vec_of_b[1], 1);
        EXPECT_EQ(input_vec_of_b[2], 100);
        EXPECT_EQ(input_vec_of_b[3], 200);
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetFormat(), ge::FORMAT_HWCN);
      }
      {
        ge::GeShape shape_check = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        auto size = shape_check.GetDimNum();
        ASSERT_EQ(size, 4);
        vector<int64_t> input_vec_of_b =shape_check.GetDims();
        EXPECT_EQ(input_vec_of_b[0], 7);
        EXPECT_EQ(input_vec_of_b[1], 13);
        EXPECT_EQ(input_vec_of_b[2], 16);
        EXPECT_EQ(input_vec_of_b[3], 16);
        EXPECT_EQ(GetPrimaryFormat(node->GetOpDesc()->GetOutputDescPtr(0)->GetFormat()), ge::FORMAT_FRACTAL_Z);
      }

    }
  }
  EXPECT_EQ(count_node, 4);
}

TEST_F(STEST_FE_TRANSOP_INSERT, InsertTransDataNode_08)
{
  // src:cce op, dst:cce op
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  ComputeGraphPtr graph_check =  std::make_shared<ComputeGraph>("test_graph_check");
  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  GeTensorDesc src_tensor_desc(GeShape({1, 256, 256, 512, 32}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  src_tensor_desc.SetOriginShape(GeShape({100,200}));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NHWC);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);
  ge::AttrUtils::SetInt(src_op, FE_IMPLY_TYPE, 6);

  OpDescPtr dst_op = std::make_shared<OpDesc>("B", "B");
  GeTensorDesc dst_tensor_desc(GeShape({1, 2, 3, 4}), ge::FORMAT_NHWC, ge::DT_FLOAT16);
  dst_tensor_desc.SetOriginShape(GeShape({100,200}));
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
  ASSERT_EQ(fe::SUCCESS, status);
  int count_node = 0;
  for (auto node : graph->GetDirectNode()) {
    ASSERT_NE(node, nullptr);
    count_node++;
    if (node->GetType() == "B") {
      {
        ge::GeShape shape_check = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        auto size = shape_check.GetDimNum();
        ASSERT_EQ(size, 4);
        vector<int64_t> input_vec_of_b =shape_check.GetDims();
        EXPECT_EQ(input_vec_of_b[0], 1);
        EXPECT_EQ(input_vec_of_b[1], 2);
        EXPECT_EQ(input_vec_of_b[2], 3);
        EXPECT_EQ(input_vec_of_b[3], 4);
      }
      {
        ge::GeShape shape_check = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        auto size = shape_check.GetDimNum();
        ASSERT_EQ(size, 4);
        vector<int64_t> input_vec_of_b =shape_check.GetDims();
        EXPECT_EQ(input_vec_of_b[0], 1);
        EXPECT_EQ(input_vec_of_b[1], 2);
        EXPECT_EQ(input_vec_of_b[2], 3);
        EXPECT_EQ(input_vec_of_b[3], 4);
      }
    }
    if (node->GetType() == "TransData") {
      {
        ge::GeShape shape_check = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        auto size = shape_check.GetDimNum();
        ASSERT_EQ(size, 5);
        vector<int64_t> input_vec_of_b =shape_check.GetDims();
        EXPECT_EQ(input_vec_of_b[0], 1);
        EXPECT_EQ(input_vec_of_b[1], 256);
        EXPECT_EQ(input_vec_of_b[2], 256);
        EXPECT_EQ(input_vec_of_b[3], 512);
        EXPECT_EQ(input_vec_of_b[4], 32);
      }
      {
        ge::GeShape shape_check = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        auto size = shape_check.GetDimNum();
        ASSERT_EQ(size, 4);
        vector<int64_t> input_vec_of_b =shape_check.GetDims();
        EXPECT_EQ(input_vec_of_b[0], 1);
        EXPECT_EQ(input_vec_of_b[1], 100);
        EXPECT_EQ(input_vec_of_b[2], 200);
        EXPECT_EQ(input_vec_of_b[3], 1);
      }

    }
  }
  EXPECT_EQ(count_node, 3);
}

TEST_F(STEST_FE_TRANSOP_INSERT, InsertTransDataNode_09)
{
  // src:FORMAT_FRACTAL_Z, dst:FORMAT_NCHW  when C==1 && N==groups && groups > 1, insert HWCN
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  ComputeGraphPtr graph_check =  std::make_shared<ComputeGraph>("test_graph_check");
  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  int64_t group = 2; // set groups 2  sub format
  GeTensorDesc src_tensor_desc(GeShape({1, 256, 256, 512, 32}), static_cast<ge::Format>(ge::GetFormatFromSub(ge::FORMAT_FRACTAL_Z, group)), ge::DT_FLOAT);
  src_tensor_desc.SetOriginShape(GeShape({2,1,3,4}));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);
  ge::AttrUtils::SetInt(src_op, FE_IMPLY_TYPE, 6);

  OpDescPtr dst_op = std::make_shared<OpDesc>("B", "B");
  GeTensorDesc dst_tensor_desc(GeShape({1, 2, 3, 4}), static_cast<ge::Format>(ge::GetFormatFromSub(ge::FORMAT_NCHW, 0)), ge::DT_FLOAT);
  dst_tensor_desc.SetOriginShape(GeShape({2,1,3,4}));
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
  ASSERT_EQ(fe::SUCCESS, status);
  int count_node = 0;

  printf("ge::GetPrimaryFormat(src_op->GetInputDesc(0).GetFormat()) %d...\n", ge::GetPrimaryFormat(src_op->GetInputDesc(0).GetFormat()));
  printf("ge::GetSubFormat(op1->GetInputDesc(0).GetFormat() %d...\n", ge::GetSubFormat(src_op->GetInputDesc(0).GetFormat()));
  printf("ge::GetPrimaryFormat(dst_op->GetInputDesc(0).GetFormat()) %d...\n", ge::GetPrimaryFormat(dst_op->GetInputDesc(0).GetFormat()));
  printf("ge::GetSubFormat(dst_op->GetInputDesc(0).GetFormat() %d...\n", ge::GetSubFormat(dst_op->GetInputDesc(0).GetFormat()));

  for (auto node : graph->GetDirectNode()) {
    ASSERT_NE(node, nullptr);
    count_node++;
    printf("countNode %d...\n", count_node);
	printf("node->GetType() %s...\n", node->GetType().c_str());

    if (node->GetType() == "A") {
      {
        ge::GeShape shape_check = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
		printf("InputDescPtr(0).GetFormat() %d...\n", node->GetOpDesc()->GetInputDescPtr(0)->GetFormat());
        auto size = shape_check.GetDimNum();
        ASSERT_EQ(size, 5);
        vector<int64_t> input_vec_of_b =shape_check.GetDims();
        EXPECT_EQ(input_vec_of_b[0], 1);
        EXPECT_EQ(input_vec_of_b[1], 256);
        EXPECT_EQ(input_vec_of_b[2], 256);
        EXPECT_EQ(input_vec_of_b[3], 512);
		EXPECT_EQ(input_vec_of_b[4], 32);
      }
      {
        ge::GeShape shape_check = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
		printf("GetOutputDescPtr(0).GetFormat() %d...\n", node->GetOpDesc()->GetOutputDescPtr(0)->GetFormat());
        auto size = shape_check.GetDimNum();
        ASSERT_EQ(size, 5);
        vector<int64_t> input_vec_of_b =shape_check.GetDims();
        EXPECT_EQ(input_vec_of_b[0], 1);
        EXPECT_EQ(input_vec_of_b[1], 256);
        EXPECT_EQ(input_vec_of_b[2], 256);
        EXPECT_EQ(input_vec_of_b[3], 512);
		EXPECT_EQ(input_vec_of_b[4], 32);
      }
    }
	if (node->GetType() == "TransData") {
      {
        ge::GeShape shape_check = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
		printf("InputDescPtr(0).GetFormat() %d...\n", node->GetOpDesc()->GetInputDescPtr(0)->GetFormat());
        auto size = shape_check.GetDimNum();
        ASSERT_EQ(size, 4);
        vector<int64_t> input_vec_of_b =shape_check.GetDims();
        EXPECT_EQ(input_vec_of_b[0], 12);
        EXPECT_EQ(input_vec_of_b[1], 1);
        EXPECT_EQ(input_vec_of_b[2], 16);
        EXPECT_EQ(input_vec_of_b[3], 16);
      }
      {
        ge::GeShape shape_check = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
		printf("GetOutputDescPtr(0).GetFormat() %d...\n", node->GetOpDesc()->GetOutputDescPtr(0)->GetFormat());
        auto size = shape_check.GetDimNum();
        ASSERT_EQ(size, 4);
        vector<int64_t> input_vec_of_b =shape_check.GetDims();
        EXPECT_EQ(input_vec_of_b[0], 3);
        EXPECT_EQ(input_vec_of_b[1], 4);
        EXPECT_EQ(input_vec_of_b[2], 1);
        EXPECT_EQ(input_vec_of_b[3], 2);
      }
    }
	if (node->GetType() == TRANSPOSE) {
      {
        ge::GeShape shape_check = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
		printf("InputDescPtr(0).GetFormat() %d...\n", node->GetOpDesc()->GetInputDescPtr(0)->GetFormat());
        auto size = shape_check.GetDimNum();
        ASSERT_EQ(size, 4);
        vector<int64_t> input_vec_of_b =shape_check.GetDims();
        EXPECT_EQ(input_vec_of_b[0], 3);
        EXPECT_EQ(input_vec_of_b[1], 4);
        EXPECT_EQ(input_vec_of_b[2], 1);
        EXPECT_EQ(input_vec_of_b[3], 2);
      }
      {
        ge::GeShape shape_check = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
		printf("GetOutputDescPtr(0).GetFormat() %d...\n", node->GetOpDesc()->GetOutputDescPtr(0)->GetFormat());
        auto size = shape_check.GetDimNum();
        ASSERT_EQ(size, 4);
        vector<int64_t> input_vec_of_b =shape_check.GetDims();
        EXPECT_EQ(input_vec_of_b[0], 2);
        EXPECT_EQ(input_vec_of_b[1], 1);
        EXPECT_EQ(input_vec_of_b[2], 3);
        EXPECT_EQ(input_vec_of_b[3], 4);
      }
    }
    if (node->GetType() == "B") {
      {
        ge::GeShape shape_check = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
		printf("InputDescPtr(0).GetFormat() %d...\n", node->GetOpDesc()->GetInputDescPtr(0)->GetFormat());
        auto size = shape_check.GetDimNum();
        ASSERT_EQ(size, 4);
        vector<int64_t> input_vec_of_b =shape_check.GetDims();
        EXPECT_EQ(input_vec_of_b[0], 1);
        EXPECT_EQ(input_vec_of_b[1], 2);
        EXPECT_EQ(input_vec_of_b[2], 3);
        EXPECT_EQ(input_vec_of_b[3], 4);
      }
      {
        ge::GeShape shape_check = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
		printf("GetOutputDescPtr(0).GetFormat() %d...\n", node->GetOpDesc()->GetOutputDescPtr(0)->GetFormat());
        auto size = shape_check.GetDimNum();
        ASSERT_EQ(size, 4);
        vector<int64_t> input_vec_of_b =shape_check.GetDims();
        EXPECT_EQ(input_vec_of_b[0], 1);
        EXPECT_EQ(input_vec_of_b[1], 2);
        EXPECT_EQ(input_vec_of_b[2], 3);
        EXPECT_EQ(input_vec_of_b[3], 4);
      }
    }
  }
  EXPECT_EQ(count_node, 5);
}



TEST_F(STEST_FE_TRANSOP_INSERT, InsertTransDataNode_09_1)
{
  // src:FORMAT_NCHW, dst: FORMAT_FRACTAL_Z  when C==1 && N==groups && groups > 1, insert HWCN
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  ComputeGraphPtr graph_check =  std::make_shared<ComputeGraph>("test_graph_check");

  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  int64_t group = 2; // set groups 2  sub format
  GeTensorDesc src_tensor_desc(GeShape({2,1,3,4}), static_cast<ge::Format>(ge::GetFormatFromSub(ge::FORMAT_NCHW, 0)), ge::DT_FLOAT);
  src_tensor_desc.SetOriginShape(GeShape({2,1,3,4}));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  src_op->AddInputDesc(src_tensor_desc);
  src_op->AddOutputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);
  ge::AttrUtils::SetInt(src_op, FE_IMPLY_TYPE, 6);
  vector<bool> input_const_vector = {false};
  src_op->SetIsInputConst(input_const_vector);

  OpDescPtr dst_op = std::make_shared<OpDesc>("B", "B");
  GeTensorDesc dst_tensor_desc(GeShape({1, 256, 256, 512, 32}), static_cast<ge::Format>(ge::GetFormatFromSub(ge::FORMAT_FRACTAL_Z, group)), ge::DT_FLOAT);
  dst_tensor_desc.SetOriginShape(GeShape({2,1,3,4}));
  dst_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  dst_op->AddOutputDesc(dst_tensor_desc);
  dst_op->AddInputDesc(dst_tensor_desc);
  auto dst_node = graph->AddNode(dst_op);
  ge::AttrUtils::SetInt(dst_op, FE_IMPLY_TYPE, 6);

  GraphUtils::AddEdge(src_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(0));

  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status status = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));
  ASSERT_EQ(fe::SUCCESS, status);
  int count_node = 0;

  printf("ge::GetPrimaryFormat(src_op->GetInputDesc(0).GetFormat()) %d...\n", ge::GetPrimaryFormat(src_op->GetInputDesc(0).GetFormat()));
  printf("ge::GetSubFormat(op1->GetInputDesc(0).GetFormat() %d...\n", ge::GetSubFormat(src_op->GetInputDesc(0).GetFormat()));
  printf("ge::GetPrimaryFormat(dst_op->GetInputDesc(0).GetFormat()) %d...\n", ge::GetPrimaryFormat(dst_op->GetInputDesc(0).GetFormat()));
  printf("ge::GetSubFormat(dst_op->GetInputDesc(0).GetFormat() %d...\n", ge::GetSubFormat(dst_op->GetInputDesc(0).GetFormat()));

  for (auto node : graph->GetDirectNode()) {
    ASSERT_NE(node, nullptr);
    count_node++;
    printf("countNode %d...\n", count_node);
	printf("node->GetType() %s...\n", node->GetType().c_str());

    if (node->GetType() == "A") {
      {
        ge::GeShape shape_check = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        auto size = shape_check.GetDimNum();
        ASSERT_EQ(size, 4);
        vector<int64_t> input_vec_of_b =shape_check.GetDims();
        EXPECT_EQ(input_vec_of_b[0], 2);
        EXPECT_EQ(input_vec_of_b[1], 1);
        EXPECT_EQ(input_vec_of_b[2], 3);
        EXPECT_EQ(input_vec_of_b[3], 4);
      }
      {
        ge::GeShape shape_check = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        auto size = shape_check.GetDimNum();
        ASSERT_EQ(size, 4);
        vector<int64_t> input_vec_of_b =shape_check.GetDims();
        EXPECT_EQ(input_vec_of_b[0], 2);
        EXPECT_EQ(input_vec_of_b[1], 1);
        EXPECT_EQ(input_vec_of_b[2], 3);
        EXPECT_EQ(input_vec_of_b[3], 4);
      }
    }
	if (node->GetType() == TRANSPOSE) {
      {
        ge::GeShape shape_check = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        auto size = shape_check.GetDimNum();
        ASSERT_EQ(size, 4);
        vector<int64_t> input_vec_of_b =shape_check.GetDims();
        EXPECT_EQ(input_vec_of_b[0], 2);
        EXPECT_EQ(input_vec_of_b[1], 1);
        EXPECT_EQ(input_vec_of_b[2], 3);
        EXPECT_EQ(input_vec_of_b[3], 4);
      }
      {
        ge::GeShape shape_check = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        auto size = shape_check.GetDimNum();
        ASSERT_EQ(size, 4);
        vector<int64_t> input_vec_of_b =shape_check.GetDims();
        EXPECT_EQ(input_vec_of_b[0], 3);
        EXPECT_EQ(input_vec_of_b[1], 4);
        EXPECT_EQ(input_vec_of_b[2], 1);
        EXPECT_EQ(input_vec_of_b[3], 2);
      }
	}
    if (node->GetType() == "TransData") {
      {
        ge::GeShape shape_check = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        auto size = shape_check.GetDimNum();
        ASSERT_EQ(size, 4);
        vector<int64_t> input_vec_of_b =shape_check.GetDims();
        EXPECT_EQ(input_vec_of_b[0], 3);
        EXPECT_EQ(input_vec_of_b[1], 4);
        EXPECT_EQ(input_vec_of_b[2], 1);
        EXPECT_EQ(input_vec_of_b[3], 2);
      }
      {
        ge::GeShape shape_check = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        auto size = shape_check.GetDimNum();
        ASSERT_EQ(size, 4);
        vector<int64_t> input_vec_of_b =shape_check.GetDims();
        EXPECT_EQ(input_vec_of_b[0], 12);
        EXPECT_EQ(input_vec_of_b[1], 1);
        EXPECT_EQ(input_vec_of_b[2], 16);
        EXPECT_EQ(input_vec_of_b[3], 16);
      }
    }
    if (node->GetType() == "B") {
      {
        ge::GeShape shape_check = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        auto size = shape_check.GetDimNum();
        ASSERT_EQ(size, 5);
        vector<int64_t> input_vec_of_b =shape_check.GetDims();
        EXPECT_EQ(input_vec_of_b[0], 1);
        EXPECT_EQ(input_vec_of_b[1], 256);
        EXPECT_EQ(input_vec_of_b[2], 256);
        EXPECT_EQ(input_vec_of_b[3], 512);
		EXPECT_EQ(input_vec_of_b[4], 32);
      }
      {
        ge::GeShape shape_check = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        auto size = shape_check.GetDimNum();
        ASSERT_EQ(size, 5);
        vector<int64_t> input_vec_of_b =shape_check.GetDims();
        EXPECT_EQ(input_vec_of_b[0], 1);
        EXPECT_EQ(input_vec_of_b[1], 256);
        EXPECT_EQ(input_vec_of_b[2], 256);
        EXPECT_EQ(input_vec_of_b[3], 512);
		EXPECT_EQ(input_vec_of_b[4], 32);
      }
    }
  }
  EXPECT_EQ(count_node, 5);
}

TEST_F(STEST_FE_TRANSOP_INSERT, InsertTransDataNode_10)
{
  // src:cce op, dst:cce op
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  ComputeGraphPtr graph_check =  std::make_shared<ComputeGraph>("test_graph_check");
  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  GeTensorDesc src_tensor_desc(GeShape({-1, 1, -1, -1 , 16}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  src_tensor_desc.SetOriginShape(GeShape({-1, -1, -1, 4}));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NHWC);
  src_tensor_desc.SetShapeRange({{2, 3}, {1, 1}, {1, 3}, {4, 15}, {16, 16}});
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);
  ge::AttrUtils::SetInt(src_op, FE_IMPLY_TYPE, 6);

  OpDescPtr dst_op = std::make_shared<OpDesc>("B", "B");
  GeTensorDesc dst_tensor_desc(GeShape({-1, -1, -1, 4}), ge::FORMAT_NHWC, ge::DT_FLOAT16);
  dst_tensor_desc.SetOriginShape(GeShape({-1, -1, -1, 4}));
  dst_tensor_desc.SetOriginFormat(ge::FORMAT_NHWC);
  dst_tensor_desc.SetShapeRange({{2, 3}, {1, 3}, {4, 15}, {4, 4}});
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
  ASSERT_EQ(fe::SUCCESS, status);
  int count_node = 0;
  for (auto node : graph->GetDirectNode()) {
    ASSERT_NE(node, nullptr);
    count_node++;
    if (node->GetType() == "B") {
      {
        ge::GeShape shape_check = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        auto size = shape_check.GetDimNum();
        ASSERT_EQ(size, 4);
        vector<int64_t> input_vec_of_b =shape_check.GetDims();
        vector<int64_t> exp_vec = {-1, -1, -1, 4};
        EXPECT_EQ(input_vec_of_b, exp_vec);
      }
      {
        ge::GeShape shape_check = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        auto size = shape_check.GetDimNum();
        ASSERT_EQ(size, 4);
        vector<int64_t> input_vec_of_b =shape_check.GetDims();
        vector<int64_t> exp_vec = {-1, -1, -1, 4};
        EXPECT_EQ(input_vec_of_b, exp_vec);
      }
    }
    if (node->GetType() == "TransData") {
      {
        ge::GeShape shape_check = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        std::vector<std::pair<int64_t, int64_t>> shape_range;
        node->GetOpDesc()->GetInputDescPtr(0)->GetShapeRange(shape_range);
        auto size = shape_check.GetDimNum();
        ASSERT_EQ(size, 5);
        vector<int64_t> input_vec_of_b =shape_check.GetDims();
        vector<int64_t> exp_vec = {-1, 1, -1, -1, 16};
        std::vector<std::pair<int64_t, int64_t>> exp_range = {{2, 3}, {1, 1}, {1, 3}, {4, 15}, {16, 16}};
        EXPECT_EQ(input_vec_of_b, exp_vec);
        EXPECT_EQ(shape_range, exp_range);
      }
      {
        ge::GeShape shape_check = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        std::vector<std::pair<int64_t, int64_t>> shape_range;
        node->GetOpDesc()->GetOutputDescPtr(0)->GetShapeRange(shape_range);
        auto size = shape_check.GetDimNum();
        ASSERT_EQ(size, 4);
        vector<int64_t> exp_vec = {-1, -1, -1, 4};
        std::vector<std::pair<int64_t, int64_t>> exp_range = {{2, 3}, {1, 3}, {4, 15}, {4, 4}};
        vector<int64_t> input_vec_of_b =shape_check.GetDims();
        EXPECT_EQ(input_vec_of_b, exp_vec);
        EXPECT_EQ(shape_range, exp_range);
      }

    }
  }
  EXPECT_EQ(count_node, 3);
}

TEST_F(STEST_FE_TRANSOP_INSERT, InsertCastNode)
{
  // src:cce op, dst:cce op
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");

  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  GeTensorDesc src_tensor_desc(GeShape({1, 256, 256, 512, 1}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);
  ge::AttrUtils::SetInt(src_op, FE_IMPLY_TYPE, 6);

  OpDescPtr dst_op = std::make_shared<OpDesc>("C", "C");
  GeTensorDesc dst_tensor_desc(GeShape({1, 256, 256, 512, 1}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
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
  ASSERT_EQ(fe::SUCCESS, status);
  int count_node = 0;
  for (auto node : graph->GetDirectNode()) {

    ASSERT_NE(node, nullptr);
    if (node->GetType() == "A") {
      auto dim_vec = node->GetOpDesc()->GetInputDescPtr(0)->GetShape().GetDims();
      ASSERT_EQ(dim_vec.size(), 5);
      EXPECT_EQ(dim_vec[0], 1);
      EXPECT_EQ(dim_vec[1], 256);
      EXPECT_EQ(dim_vec[2], 256);
      EXPECT_EQ(dim_vec[3], 512);
      EXPECT_EQ(dim_vec[4], 1);
    }
    count_node++;
  }
  EXPECT_EQ(count_node, 3);
}

TEST_F(STEST_FE_TRANSOP_INSERT, InsertCastNode_not_necessary)
{
  // src:cce op, dst:cce op
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");

  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  GeTensorDesc src_tensor_desc(GeShape({1, 256, 256, 512, 1}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);
  ge::AttrUtils::SetInt(src_op, FE_IMPLY_TYPE, 6);

  OpDescPtr dst_op = std::make_shared<OpDesc>("C", "C");
  GeTensorDesc dst_tensor_desc(GeShape({1, 256, 256, 512, 1}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
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
  ASSERT_EQ(fe::SUCCESS, status);
  int count_node = 0;
  for (auto node : graph->GetDirectNode()) {
    ASSERT_NE(node, nullptr);
    count_node++;
  }
  EXPECT_EQ(count_node, 2);
}

TEST_F(STEST_FE_TRANSOP_INSERT, InsertPermuteNodeWithInputConst)
{
  // src:cce op, dst:cce op
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");

  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  GeTensorDesc src_tensor_desc(GeShape({1, 256, 256, 512}), ge::FORMAT_NCHW, ge::DT_FLOAT);
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);
  ge::AttrUtils::SetInt(src_op, FE_IMPLY_TYPE, 6);

  OpDescPtr dst_op = std::make_shared<OpDesc>("C", "C");
  GeTensorDesc dst_tensor_desc(GeShape({1, 256, 256, 512}), ge::FORMAT_NHWC, ge::DT_FLOAT);
  dst_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  dst_op->AddInputDesc(dst_tensor_desc);
  dst_op->AddOutputDesc(dst_tensor_desc);
  auto dst_node = graph->AddNode(dst_op);
  ge::AttrUtils::SetInt(dst_op, FE_IMPLY_TYPE, 6);
  GraphUtils::AddEdge(src_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(0));

  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status status = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));
  ASSERT_EQ(fe::SUCCESS, status);
  int count_node = 0;
  for (auto node : graph->GetDirectNode()) {

    ASSERT_NE(node, nullptr);
    if (node->GetType() == TRANSPOSE) {
      auto dim_vec = node->GetOpDesc()->GetInputDescPtr(0)->GetShape().GetDims();
      ASSERT_EQ(dim_vec.size(), 4);
      EXPECT_EQ(dim_vec[0], 1);
      EXPECT_EQ(dim_vec[1], 256);
      EXPECT_EQ(dim_vec[2], 256);
      EXPECT_EQ(dim_vec[3], 512);
    }
    if (node->GetType() == "A") {
      auto dim_vec = node->GetOpDesc()->GetInputDescPtr(0)->GetShape().GetDims();
      ASSERT_EQ(dim_vec.size(), 4);
      EXPECT_EQ(dim_vec[0], 1);
      EXPECT_EQ(dim_vec[1], 256);
      EXPECT_EQ(dim_vec[2], 256);
      EXPECT_EQ(dim_vec[3], 512);
    }
    count_node++;
  }
  EXPECT_EQ(count_node, 4);
}

TEST_F(STEST_FE_TRANSOP_INSERT, InsertPermuteNode)
{
  // src:cce op, dst:cce op
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");

  OpDescPtr src_op = std::make_shared<OpDesc>("B", "B");
  GeTensorDesc src_tensor_desc(GeShape({1, 1024, 256, 512}), ge::FORMAT_NCHW, ge::DT_FLOAT);
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);
  ge::AttrUtils::SetInt(src_op, FE_IMPLY_TYPE, 6);

  OpDescPtr dst_op = std::make_shared<OpDesc>("D", "D");
  GeTensorDesc dst_tensor_desc(GeShape({1, 3, 4, 2}), ge::FORMAT_NHWC, ge::DT_FLOAT);
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

  ASSERT_EQ(fe::SUCCESS, status);
  int count_node = 0;
  for (auto node : graph->GetDirectNode()) {
    ASSERT_NE(node, nullptr);
    count_node++;
    if (node->GetType() == TRANSPOSE) {
      ge::GeShape shape_check = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
      auto size = shape_check.GetDimNum();
      ASSERT_EQ(size, 4);
      vector<int64_t> input_vec_of_b =shape_check.GetDims();
      EXPECT_EQ(input_vec_of_b[0], 1);
      EXPECT_EQ(input_vec_of_b[1], 256);
      EXPECT_EQ(input_vec_of_b[2], 512);
      EXPECT_EQ(input_vec_of_b[3], 1024);
    }
    if (node->GetType() == "D") {
      {
        ge::GeShape shape_check = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        auto size = shape_check.GetDimNum();
        ASSERT_EQ(size, 4);
        vector<int64_t> input_vec_of_b =shape_check.GetDims();
        EXPECT_EQ(input_vec_of_b[0], 1);
        EXPECT_EQ(input_vec_of_b[1], 3);
        EXPECT_EQ(input_vec_of_b[2], 4);
        EXPECT_EQ(input_vec_of_b[3], 2);
      }
      {
        ge::GeShape shape_check = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        auto size = shape_check.GetDimNum();
        ASSERT_EQ(size, 4);
        vector<int64_t> input_vec_of_b =shape_check.GetDims();
        EXPECT_EQ(input_vec_of_b[0], 1);
        EXPECT_EQ(input_vec_of_b[1], 3);
        EXPECT_EQ(input_vec_of_b[2], 4);
        EXPECT_EQ(input_vec_of_b[3], 2);
      }

    }

  }
  EXPECT_EQ(count_node, 4);
}

TEST_F(STEST_FE_TRANSOP_INSERT, InsertPermuteNode2)
{
  // src:cce op, dst:cce op
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");

  OpDescPtr src_op = std::make_shared<OpDesc>("D", "D");
  GeTensorDesc src_tensor_desc(GeShape({1, 3, 4, 2}), ge::FORMAT_NHWC, ge::DT_FLOAT);
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NHWC);

  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);
  ge::AttrUtils::SetInt(src_op, FE_IMPLY_TYPE, 6);

  OpDescPtr dst_op = std::make_shared<OpDesc>("B", "B");
  GeTensorDesc dst_tensor_desc(GeShape({1, 2}), ge::FORMAT_NCHW, ge::DT_FLOAT);
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

  ASSERT_EQ(fe::SUCCESS, status);
  int count_node = 0;
  for (auto node : graph->GetDirectNode()) {
    ASSERT_NE(node, nullptr);
    count_node++;
    if (node->GetType() == TRANSPOSE) {
      ge::GeShape shape_check = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
      auto size = shape_check.GetDimNum();
      ASSERT_EQ(size, 4);
      vector<int64_t> input_vec_of_b =shape_check.GetDims();
      EXPECT_EQ(input_vec_of_b[0], 1);
      EXPECT_EQ(input_vec_of_b[1], 2);
      EXPECT_EQ(input_vec_of_b[2], 3);
      EXPECT_EQ(input_vec_of_b[3], 4);
      EXPECT_EQ(ge::AttrUtils::HasAttr(node->GetOpDesc(), "perm"), false);
      EXPECT_EQ(ge::AttrUtils::HasAttr(node->GetOpDesc(), "order"), true);
    }
    if (node->GetType() == "B") {
      ge::GeShape input_shape_check = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
      ge::GeShape output_shape_check = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
      auto size_in = input_shape_check.GetDimNum();
      auto size_out = output_shape_check.GetDimNum();
      ASSERT_EQ(size_in, 2);
      ASSERT_EQ(size_out, 2);
      vector<int64_t> input_vec_of_b = input_shape_check.GetDims();
      vector<int64_t> output_vec_of_b = output_shape_check.GetDims();
      EXPECT_EQ(input_vec_of_b[0], 1);
      EXPECT_EQ(input_vec_of_b[1], 2);

      EXPECT_EQ(output_vec_of_b[0], 1);
      EXPECT_EQ(output_vec_of_b[1], 2);

    }
  }
  EXPECT_EQ(count_node, 4);
}



TEST_F(STEST_FE_TRANSOP_INSERT, InsertPermuteNodeWithInputConst2)
{
  // src:cce op, dst:cce op
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");

  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  GeTensorDesc src_tensor_desc(GeShape({256,512}), ge::FORMAT_NCHW, ge::DT_FLOAT);
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);

  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);
  ge::AttrUtils::SetInt(src_op, FE_IMPLY_TYPE, 6);

  OpDescPtr dst_op = std::make_shared<OpDesc>("C", "C");
  GeTensorDesc dst_tensor_desc(GeShape({1, 256, 256, 512}), ge::FORMAT_NHWC, ge::DT_FLOAT);
  dst_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);

  dst_op->AddInputDesc(dst_tensor_desc);
  dst_op->AddOutputDesc(dst_tensor_desc);
  auto dst_node = graph->AddNode(dst_op);
  ge::AttrUtils::SetInt(dst_op, FE_IMPLY_TYPE, 6);
  GraphUtils::AddEdge(src_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(0));

  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status status = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));
  ASSERT_EQ(fe::SUCCESS, status);
  int count_node = 0;
  for (auto node : graph->GetDirectNode()) {

    ASSERT_NE(node, nullptr);
    if (node->GetType() == TRANSPOSE) {
      {
        auto dim_vec = node->GetOpDesc()->GetInputDescPtr(0)->GetShape().GetDims();
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetFormat(), ge::FORMAT_NCHW);
        ASSERT_EQ(dim_vec.size(), 4);
        EXPECT_EQ(dim_vec[0], 1);
        EXPECT_EQ(dim_vec[1], 256);
        EXPECT_EQ(dim_vec[2], 512);
        EXPECT_EQ(dim_vec[3], 1);
      }
      {
        auto dim_vec = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape().GetDims();
        EXPECT_EQ(node->GetOpDesc()->GetOutputDescPtr(0)->GetFormat(), ge::FORMAT_NHWC);
        ASSERT_EQ(dim_vec.size(), 4);
        EXPECT_EQ(dim_vec[0], 1);
        EXPECT_EQ(dim_vec[1], 512);
        EXPECT_EQ(dim_vec[2], 1);
        EXPECT_EQ(dim_vec[3], 256);
      }
    }
    if (node->GetType() == "A") {
      auto dim_vec = node->GetOpDesc()->GetInputDescPtr(0)->GetShape().GetDims();
      ASSERT_EQ(dim_vec.size(), 2);
      EXPECT_EQ(dim_vec[0], 256);
      EXPECT_EQ(dim_vec[1], 512);
    }
    if (node->GetType() == "C") {
      {
        auto dim_vec = node->GetOpDesc()->GetInputDescPtr(0)->GetShape().GetDims();
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetFormat(), ge::FORMAT_NHWC);
        ASSERT_EQ(dim_vec.size(), 4);
        EXPECT_EQ(dim_vec[0], 1);
        EXPECT_EQ(dim_vec[1], 256);
        EXPECT_EQ(dim_vec[2], 256);
        EXPECT_EQ(dim_vec[3], 512);
      }
      {
        auto dim_vec = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape().GetDims();
        EXPECT_EQ(node->GetOpDesc()->GetOutputDescPtr(0)->GetFormat(), ge::FORMAT_NHWC);
        ASSERT_EQ(dim_vec.size(), 4);
        EXPECT_EQ(dim_vec[0], 1);
        EXPECT_EQ(dim_vec[1], 256);
        EXPECT_EQ(dim_vec[2], 256);
        EXPECT_EQ(dim_vec[3], 512);
      }
    }
    count_node++;
  }
  EXPECT_EQ(count_node, 4);
}

TEST_F(STEST_FE_TRANSOP_INSERT, InsertPermuteNode4)
{
  // src:cce op, dst:cce op
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");

  OpDescPtr src_op = std::make_shared<OpDesc>("B", "B");
  GeTensorDesc src_tensor_desc(GeShape({1024, 256, 2}), ge::FORMAT_NCHW, ge::DT_FLOAT);
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);

  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);
  ge::AttrUtils::SetInt(src_op, FE_IMPLY_TYPE, 6);

  OpDescPtr dst_op = std::make_shared<OpDesc>("D", "D");
  GeTensorDesc dst_tensor_desc(GeShape({1, 3, 4, 2}), ge::FORMAT_NHWC, ge::DT_FLOAT);
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

  ASSERT_EQ(fe::SUCCESS, status);
  int count_node = 0;
  for (auto node : graph->GetDirectNode()) {
    ASSERT_NE(node, nullptr);
    count_node++;
    if (node->GetType() == TRANSPOSE) {
      {
        ge::GeShape shape_check = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        auto size = shape_check.GetDimNum();
        ASSERT_EQ(size, 4);
        vector<int64_t> input_vec_of_b = shape_check.GetDims();
        EXPECT_EQ(input_vec_of_b[0], 1);
        EXPECT_EQ(input_vec_of_b[1], 1024);
        EXPECT_EQ(input_vec_of_b[2], 256);
        EXPECT_EQ(input_vec_of_b[3], 2);
      }
      {
        ge::GeShape shape_check = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        auto size = shape_check.GetDimNum();
        ASSERT_EQ(size, 4);
        vector<int64_t> input_vec_of_b = shape_check.GetDims();
        EXPECT_EQ(input_vec_of_b[0], 1);
        EXPECT_EQ(input_vec_of_b[1], 256);
        EXPECT_EQ(input_vec_of_b[2], 2);
        EXPECT_EQ(input_vec_of_b[3], 1024);
      }
    }
    if (node->GetType() == "D") {
      {
        ge::GeShape shape_check = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        auto size = shape_check.GetDimNum();
        ASSERT_EQ(size, 4);
        vector<int64_t> input_vec_of_b =shape_check.GetDims();
        EXPECT_EQ(input_vec_of_b[0], 1);
        EXPECT_EQ(input_vec_of_b[1], 3);
        EXPECT_EQ(input_vec_of_b[2], 4);
        EXPECT_EQ(input_vec_of_b[3], 2);
      }
      {
        ge::GeShape shape_check = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        auto size = shape_check.GetDimNum();
        ASSERT_EQ(size, 4);
        vector<int64_t> input_vec_of_b =shape_check.GetDims();
        EXPECT_EQ(input_vec_of_b[0], 1);
        EXPECT_EQ(input_vec_of_b[1], 3);
        EXPECT_EQ(input_vec_of_b[2], 4);
        EXPECT_EQ(input_vec_of_b[3], 2);
      }

    }

  }
  EXPECT_EQ(count_node, 4);
}

TEST_F(STEST_FE_TRANSOP_INSERT, InsertPermuteNode5)
{
  // src:cce op, dst:cce op
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");

  OpDescPtr src_op = std::make_shared<OpDesc>("D", "D");
  GeTensorDesc src_tensor_desc(GeShape({5}), ge::FORMAT_NHWC, ge::DT_FLOAT);
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NHWC);

  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);
  ge::AttrUtils::SetInt(src_op, FE_IMPLY_TYPE, 6);

  OpDescPtr dst_op = std::make_shared<OpDesc>("B", "B");
  GeTensorDesc dst_tensor_desc(GeShape({1, 2}), ge::FORMAT_NCHW, ge::DT_FLOAT);
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

  ASSERT_EQ(fe::SUCCESS, status);
  int count_node = 0;
  for (auto node : graph->GetDirectNode()) {
    ASSERT_NE(node, nullptr);
    count_node++;
    if (node->GetType() == TRANSPOSE) {
      {
        ge::GeShape shape_check = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        auto size = shape_check.GetDimNum();
        ASSERT_EQ(size, 4);
        vector<int64_t> input_vec_of_b = shape_check.GetDims();
        EXPECT_EQ(input_vec_of_b[0], 1);
        EXPECT_EQ(input_vec_of_b[1], 1);
        EXPECT_EQ(input_vec_of_b[2], 1);
        EXPECT_EQ(input_vec_of_b[3], 5);
      }
      {
        ge::GeShape shape_check = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        auto size = shape_check.GetDimNum();
        ASSERT_EQ(size, 4);
        vector<int64_t> input_vec_of_b = shape_check.GetDims();
        EXPECT_EQ(input_vec_of_b[0], 1);
        EXPECT_EQ(input_vec_of_b[1], 5);
        EXPECT_EQ(input_vec_of_b[2], 1);
        EXPECT_EQ(input_vec_of_b[3], 1);
      }
    }
    if (node->GetType() == "B") {
      ge::GeShape input_shape_check = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
      ge::GeShape output_shape_check = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
      auto size_in = input_shape_check.GetDimNum();
      auto size_out = output_shape_check.GetDimNum();
      ASSERT_EQ(size_in, 2);
      ASSERT_EQ(size_out, 2);
      vector<int64_t> input_vec_of_b = input_shape_check.GetDims();
      vector<int64_t> output_vec_of_b = output_shape_check.GetDims();
      EXPECT_EQ(input_vec_of_b[0], 1);
      EXPECT_EQ(input_vec_of_b[1], 2);

      EXPECT_EQ(output_vec_of_b[0], 1);
      EXPECT_EQ(output_vec_of_b[1], 2);

    }
  }
  EXPECT_EQ(count_node, 4);
}


TEST_F(STEST_FE_TRANSOP_INSERT, InsertPermuteNode6)
{
  // src:cce op, dst:cce op
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");

  OpDescPtr src_op = std::make_shared<OpDesc>("D", "D");
  GeTensorDesc src_tensor_desc(GeShape({5,7}), ge::FORMAT_NHWC, ge::DT_FLOAT);
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NHWC);

  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);
  ge::AttrUtils::SetInt(src_op, FE_IMPLY_TYPE, 6);

  OpDescPtr dst_op = std::make_shared<OpDesc>("B", "B");
  GeTensorDesc dst_tensor_desc(GeShape({1, 2}), ge::FORMAT_NCHW, ge::DT_FLOAT);
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

  ASSERT_EQ(fe::SUCCESS, status);
  int count_node = 0;
  for (auto node : graph->GetDirectNode()) {
    ASSERT_NE(node, nullptr);
    count_node++;
    if (node->GetType() == TRANSPOSE) {
      {
        ge::GeShape shape_check = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        auto size = shape_check.GetDimNum();
        ASSERT_EQ(size, 4);
        vector<int64_t> input_vec_of_b = shape_check.GetDims();
        EXPECT_EQ(input_vec_of_b[0], 1);
        EXPECT_EQ(input_vec_of_b[1], 5);
        EXPECT_EQ(input_vec_of_b[2], 7);
        EXPECT_EQ(input_vec_of_b[3], 1);
      }
      {
        ge::GeShape shape_check = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        auto size = shape_check.GetDimNum();
        ASSERT_EQ(size, 4);
        vector<int64_t> input_vec_of_b = shape_check.GetDims();
        EXPECT_EQ(input_vec_of_b[0], 1);
        EXPECT_EQ(input_vec_of_b[1], 1);
        EXPECT_EQ(input_vec_of_b[2], 5);
        EXPECT_EQ(input_vec_of_b[3], 7);
      }
    }
    if (node->GetType() == "B") {
      ge::GeShape input_shape_check = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
      ge::GeShape output_shape_check = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
      auto size_in = input_shape_check.GetDimNum();
      auto size_out = output_shape_check.GetDimNum();
      ASSERT_EQ(size_in, 2);
      ASSERT_EQ(size_out, 2);
      vector<int64_t> input_vec_of_b = input_shape_check.GetDims();
      vector<int64_t> output_vec_of_b = output_shape_check.GetDims();
      EXPECT_EQ(input_vec_of_b[0], 1);
      EXPECT_EQ(input_vec_of_b[1], 2);

      EXPECT_EQ(output_vec_of_b[0], 1);
      EXPECT_EQ(output_vec_of_b[1], 2);

    }
  }
  EXPECT_EQ(count_node, 4);
}

TEST_F(STEST_FE_TRANSOP_INSERT, InsertTransposDNode_NCDHW)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");

  OpDescPtr src_x_op = std::make_shared<OpDesc>("X", "BatchA");
  GeTensorDesc src_x_tensor_desc(GeShape({5,7}), ge::FORMAT_NCDHW, ge::DT_FLOAT);
  src_x_tensor_desc.SetOriginFormat(ge::FORMAT_NCDHW);
  src_x_op->AddOutputDesc(src_x_tensor_desc);
  src_x_op->AddInputDesc(src_x_tensor_desc);
  auto src_x_node = graph->AddNode(src_x_op);
  ge::AttrUtils::SetInt(src_x_op, FE_IMPLY_TYPE, 6);

  OpDescPtr src_w_op = std::make_shared<OpDesc>("W", "BatchA");
  GeTensorDesc src_w_tensor_desc(GeShape({5,7}), ge::FORMAT_NCDHW, ge::DT_FLOAT);
  src_w_tensor_desc.SetOriginFormat(ge::FORMAT_NDHWC);
  src_w_op->AddOutputDesc(src_w_tensor_desc);
  src_w_op->AddInputDesc(src_w_tensor_desc);
  auto src_w_node = graph->AddNode(src_w_op);
  ge::AttrUtils::SetInt(src_w_op, FE_IMPLY_TYPE, 6);

  OpDescPtr src_b_op = std::make_shared<OpDesc>("B", "BatchA");
  GeTensorDesc src_b_tensor_desc(GeShape({5,6,7}), ge::FORMAT_NCDHW, ge::DT_FLOAT);
  src_b_tensor_desc.SetOriginFormat(ge::FORMAT_DHWCN);
  src_b_op->AddOutputDesc(src_b_tensor_desc);
  src_b_op->AddInputDesc(src_b_tensor_desc);
  auto src_b_node = graph->AddNode(src_b_op);
  ge::AttrUtils::SetInt(src_b_op, FE_IMPLY_TYPE, 6);

  OpDescPtr src_c_op = std::make_shared<OpDesc>("C", "BatchA");
  GeTensorDesc src_c_tensor_desc(GeShape({5,6,7,8,9}), ge::FORMAT_NCDHW, ge::DT_FLOAT);
  src_c_tensor_desc.SetOriginFormat(ge::FORMAT_DHWNC);
  src_c_op->AddOutputDesc(src_c_tensor_desc);
  src_c_op->AddInputDesc(src_c_tensor_desc);
  auto src_c_node = graph->AddNode(src_c_op);
  ge::AttrUtils::SetInt(src_c_op, FE_IMPLY_TYPE, 6);

  OpDescPtr dst_op = std::make_shared<OpDesc>("batchnorm", "BatchNormV9");
  GeTensorDesc dst_x_tensor_desc(GeShape({1, 2, 3, 4, 5}), ge::FORMAT_NCDHW, ge::DT_FLOAT);
  dst_x_tensor_desc.SetOriginFormat(ge::FORMAT_NCDHW);
  GeTensorDesc dst_w_tensor_desc(GeShape({1, 2, 3, 4, 5}), ge::FORMAT_NDHWC, ge::DT_FLOAT);
  dst_w_tensor_desc.SetOriginFormat(ge::FORMAT_NDHWC);
  GeTensorDesc dst_b_tensor_desc(GeShape({1, 2, 3, 4, 5}), ge::FORMAT_DHWCN, ge::DT_FLOAT);
  dst_b_tensor_desc.SetOriginFormat(ge::FORMAT_DHWCN);
  GeTensorDesc dst_c_tensor_desc(GeShape({1, 2, 3, 4, 5}), ge::FORMAT_DHWNC, ge::DT_FLOAT);
  dst_c_tensor_desc.SetOriginFormat(ge::FORMAT_DHWNC);

  dst_op->AddInputDesc("x", dst_x_tensor_desc);
  dst_op->AddInputDesc("w", dst_w_tensor_desc);
  dst_op->AddInputDesc("b", dst_b_tensor_desc);
  dst_op->AddInputDesc("c", dst_c_tensor_desc);
  dst_op->AddOutputDesc(dst_x_tensor_desc);
  auto dst_node = graph->AddNode(dst_op);
  ge::AttrUtils::SetInt(dst_op, FE_IMPLY_TYPE, 6);
  vector<bool> input_const_vector = {false};
  dst_op->SetIsInputConst(input_const_vector);

  GraphUtils::AddEdge(src_x_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(src_w_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(1));
  GraphUtils::AddEdge(src_b_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(2));
  GraphUtils::AddEdge(src_c_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(3));

  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status status = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));

  ASSERT_EQ(fe::SUCCESS, status);
  int count_node = 0;
  int trans_count = 0;
  int unsqueeze_count = 0;
  vector<vector<int64_t>> trans_dim_result{{1,1,1,5,7}, {1,1,5,6,7}, {5,6,7,8,9}};

  for (auto node : graph->GetDirectNode()) {
    ASSERT_NE(node, nullptr);
    count_node++;
    if (node->GetType() == TRANSPOSE) {
      EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetShape().GetDims(), trans_dim_result[trans_count++]);
      continue;
    }
    if (node->GetType() == "UnsqueezeV2") {
      EXPECT_EQ(node->GetOpDesc()->GetOutputDescPtr(0)->GetShape().GetDims(), trans_dim_result[unsqueeze_count++]);
      continue;
    }
  }
  EXPECT_EQ(count_node, 13);
}

TEST_F(STEST_FE_TRANSOP_INSERT, InsertTransposDNode_NDHWC)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");

  OpDescPtr src_x_op = std::make_shared<OpDesc>("X", "BatchA");
  GeTensorDesc src_x_tensor_desc(GeShape({5,7}), ge::FORMAT_NDHWC, ge::DT_FLOAT);
  src_x_tensor_desc.SetOriginFormat(ge::FORMAT_NCDHW);
  src_x_op->AddOutputDesc(src_x_tensor_desc);
  src_x_op->AddInputDesc(src_x_tensor_desc);
  auto src_x_node = graph->AddNode(src_x_op);
  ge::AttrUtils::SetInt(src_x_op, FE_IMPLY_TYPE, 6);

  OpDescPtr src_w_op = std::make_shared<OpDesc>("W", "BatchA");
  GeTensorDesc src_w_tensor_desc(GeShape({5,7}), ge::FORMAT_NDHWC, ge::DT_FLOAT);
  src_w_tensor_desc.SetOriginFormat(ge::FORMAT_NDHWC);
  src_w_op->AddOutputDesc(src_w_tensor_desc);
  src_w_op->AddInputDesc(src_w_tensor_desc);
  auto src_w_node = graph->AddNode(src_w_op);
  ge::AttrUtils::SetInt(src_w_op, FE_IMPLY_TYPE, 6);

  OpDescPtr src_b_op = std::make_shared<OpDesc>("B", "BatchA");
  GeTensorDesc src_b_tensor_desc(GeShape({5,6,7}), ge::FORMAT_NDHWC, ge::DT_FLOAT);
  src_b_tensor_desc.SetOriginFormat(ge::FORMAT_DHWCN);
  src_b_op->AddOutputDesc(src_b_tensor_desc);
  src_b_op->AddInputDesc(src_b_tensor_desc);
  auto src_b_node = graph->AddNode(src_b_op);
  ge::AttrUtils::SetInt(src_b_op, FE_IMPLY_TYPE, 6);

  OpDescPtr src_c_op = std::make_shared<OpDesc>("C", "BatchA");
  GeTensorDesc src_c_tensor_desc(GeShape({5,6,7,8,9}), ge::FORMAT_NDHWC, ge::DT_FLOAT);
  src_c_tensor_desc.SetOriginFormat(ge::FORMAT_DHWNC);
  src_c_op->AddOutputDesc(src_c_tensor_desc);
  src_c_op->AddInputDesc(src_c_tensor_desc);
  auto src_c_node = graph->AddNode(src_c_op);
  ge::AttrUtils::SetInt(src_c_op, FE_IMPLY_TYPE, 6);

  OpDescPtr dst_op = std::make_shared<OpDesc>("batchnorm", "BatchNormV9");
  GeTensorDesc dst_x_tensor_desc(GeShape({1, 2, 3, 4, 5}), ge::FORMAT_NCDHW, ge::DT_FLOAT);
  dst_x_tensor_desc.SetOriginFormat(ge::FORMAT_NCDHW);
  GeTensorDesc dst_w_tensor_desc(GeShape({1, 2, 3, 4, 5}), ge::FORMAT_NDHWC, ge::DT_FLOAT);
  dst_w_tensor_desc.SetOriginFormat(ge::FORMAT_NDHWC);
  GeTensorDesc dst_b_tensor_desc(GeShape({1, 2, 3, 4, 5}), ge::FORMAT_DHWCN, ge::DT_FLOAT);
  dst_b_tensor_desc.SetOriginFormat(ge::FORMAT_DHWCN);
  GeTensorDesc dst_c_tensor_desc(GeShape({1, 2, 3, 4, 5}), ge::FORMAT_DHWNC, ge::DT_FLOAT);
  dst_c_tensor_desc.SetOriginFormat(ge::FORMAT_DHWNC);

  dst_op->AddInputDesc("x", dst_x_tensor_desc);
  dst_op->AddInputDesc("w", dst_w_tensor_desc);
  dst_op->AddInputDesc("b", dst_b_tensor_desc);
  dst_op->AddInputDesc("c", dst_c_tensor_desc);
  dst_op->AddOutputDesc(dst_x_tensor_desc);
  auto dst_node = graph->AddNode(dst_op);
  ge::AttrUtils::SetInt(dst_op, FE_IMPLY_TYPE, 6);
  vector<bool> input_const_vector = {false};
  dst_op->SetIsInputConst(input_const_vector);

  GraphUtils::AddEdge(src_x_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(src_w_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(1));
  GraphUtils::AddEdge(src_b_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(2));
  GraphUtils::AddEdge(src_c_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(3));

  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status status = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));

  ASSERT_EQ(fe::SUCCESS, status);
  int count_node = 0;
  int trans_count = 0;
  int unsqueeze_count = 0;
  vector<vector<int64_t>> trans_dim_result{{1,1,1,5,7}, {1,1,5,6,7}, {5,6,7,8,9}};

  for (auto node : graph->GetDirectNode()) {
    ASSERT_NE(node, nullptr);
    count_node++;
    if (node->GetType() == TRANSPOSE) {
      EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetShape().GetDims(), trans_dim_result[trans_count++]);
      continue;
    }
    if (node->GetType() == "UnsqueezeV2") {
      EXPECT_EQ(node->GetOpDesc()->GetOutputDescPtr(0)->GetShape().GetDims(), trans_dim_result[unsqueeze_count++]);
      continue;
    }
  }
  EXPECT_EQ(count_node, 13);
}

TEST_F(STEST_FE_TRANSOP_INSERT, InsertTransposDNode_DHWCN)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");

  OpDescPtr src_x_op = std::make_shared<OpDesc>("X", "BatchA");
  GeTensorDesc src_x_tensor_desc(GeShape({5,7}), ge::FORMAT_DHWCN, ge::DT_FLOAT);
  src_x_tensor_desc.SetOriginFormat(ge::FORMAT_NCDHW);
  src_x_op->AddOutputDesc(src_x_tensor_desc);
  src_x_op->AddInputDesc(src_x_tensor_desc);
  auto src_x_node = graph->AddNode(src_x_op);
  ge::AttrUtils::SetInt(src_x_op, FE_IMPLY_TYPE, 6);

  OpDescPtr src_w_op = std::make_shared<OpDesc>("W", "BatchA");
  GeTensorDesc src_w_tensor_desc(GeShape({5,7}), ge::FORMAT_DHWCN, ge::DT_FLOAT);
  src_w_tensor_desc.SetOriginFormat(ge::FORMAT_NDHWC);
  src_w_op->AddOutputDesc(src_w_tensor_desc);
  src_w_op->AddInputDesc(src_w_tensor_desc);
  auto src_w_node = graph->AddNode(src_w_op);
  ge::AttrUtils::SetInt(src_w_op, FE_IMPLY_TYPE, 6);

  OpDescPtr src_b_op = std::make_shared<OpDesc>("B", "BatchA");
  GeTensorDesc src_b_tensor_desc(GeShape({5,6,7}), ge::FORMAT_DHWCN, ge::DT_FLOAT);
  src_b_tensor_desc.SetOriginFormat(ge::FORMAT_DHWCN);
  src_b_op->AddOutputDesc(src_b_tensor_desc);
  src_b_op->AddInputDesc(src_b_tensor_desc);
  auto src_b_node = graph->AddNode(src_b_op);
  ge::AttrUtils::SetInt(src_b_op, FE_IMPLY_TYPE, 6);

  OpDescPtr src_c_op = std::make_shared<OpDesc>("C", "BatchA");
  GeTensorDesc src_c_tensor_desc(GeShape({5,6,7,8,9}), ge::FORMAT_DHWCN, ge::DT_FLOAT);
  src_c_tensor_desc.SetOriginFormat(ge::FORMAT_DHWNC);
  src_c_op->AddOutputDesc(src_c_tensor_desc);
  src_c_op->AddInputDesc(src_c_tensor_desc);
  auto src_c_node = graph->AddNode(src_c_op);
  ge::AttrUtils::SetInt(src_c_op, FE_IMPLY_TYPE, 6);

  OpDescPtr dst_op = std::make_shared<OpDesc>("batchnorm", "BatchNormV9");
  GeTensorDesc dst_x_tensor_desc(GeShape({1, 2, 3, 4, 5}), ge::FORMAT_NCDHW, ge::DT_FLOAT);
  dst_x_tensor_desc.SetOriginFormat(ge::FORMAT_NCDHW);
  GeTensorDesc dst_w_tensor_desc(GeShape({1, 2, 3, 4, 5}), ge::FORMAT_NDHWC, ge::DT_FLOAT);
  dst_w_tensor_desc.SetOriginFormat(ge::FORMAT_NDHWC);
  GeTensorDesc dst_b_tensor_desc(GeShape({1, 2, 3, 4, 5}), ge::FORMAT_DHWCN, ge::DT_FLOAT);
  dst_b_tensor_desc.SetOriginFormat(ge::FORMAT_DHWCN);
  GeTensorDesc dst_c_tensor_desc(GeShape({1, 2, 3, 4, 5}), ge::FORMAT_DHWNC, ge::DT_FLOAT);
  dst_c_tensor_desc.SetOriginFormat(ge::FORMAT_DHWNC);

  dst_op->AddInputDesc("x", dst_x_tensor_desc);
  dst_op->AddInputDesc("w", dst_w_tensor_desc);
  dst_op->AddInputDesc("b", dst_b_tensor_desc);
  dst_op->AddInputDesc("c", dst_c_tensor_desc);
  dst_op->AddOutputDesc(dst_x_tensor_desc);
  auto dst_node = graph->AddNode(dst_op);
  ge::AttrUtils::SetInt(dst_op, FE_IMPLY_TYPE, 6);
  vector<bool> input_const_vector = {false};
  dst_op->SetIsInputConst(input_const_vector);

  GraphUtils::AddEdge(src_x_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(src_w_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(1));
  GraphUtils::AddEdge(src_b_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(2));
  GraphUtils::AddEdge(src_c_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(3));

  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status status = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));

  ASSERT_EQ(fe::SUCCESS, status);
  int count_node = 0;
  int trans_count = 0;
  int unsqueeze_count = 0;
  vector<vector<int64_t>> trans_dim_result{{1,1,1,5,7}, {1,1,1,5,7}, {5,6,7,8,9}};

  for (auto node : graph->GetDirectNode()) {
    ASSERT_NE(node, nullptr);
    count_node++;
    if (node->GetType() == TRANSPOSE) {
      EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetShape().GetDims(), trans_dim_result[trans_count++]);
      continue;
    }
    if (node->GetType() == "UnsqueezeV2") {
      EXPECT_EQ(node->GetOpDesc()->GetOutputDescPtr(0)->GetShape().GetDims(), trans_dim_result[unsqueeze_count++]);
      continue;
    }
  }
  EXPECT_EQ(count_node, 13);
}

TEST_F(STEST_FE_TRANSOP_INSERT, InsertTransposDNode_DHWNC)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");

  OpDescPtr src_x_op = std::make_shared<OpDesc>("X", "BatchA");
  GeTensorDesc src_x_tensor_desc(GeShape({5,7}), ge::FORMAT_DHWNC, ge::DT_FLOAT);
  src_x_tensor_desc.SetOriginFormat(ge::FORMAT_NCDHW);
  src_x_op->AddOutputDesc(src_x_tensor_desc);
  src_x_op->AddInputDesc(src_x_tensor_desc);
  auto src_x_node = graph->AddNode(src_x_op);
  ge::AttrUtils::SetInt(src_x_op, FE_IMPLY_TYPE, 6);

  OpDescPtr src_w_op = std::make_shared<OpDesc>("W", "BatchA");
  GeTensorDesc src_w_tensor_desc(GeShape({5,7}), ge::FORMAT_DHWNC, ge::DT_FLOAT);
  src_w_tensor_desc.SetOriginFormat(ge::FORMAT_NDHWC);
  src_w_op->AddOutputDesc(src_w_tensor_desc);
  src_w_op->AddInputDesc(src_w_tensor_desc);
  auto src_w_node = graph->AddNode(src_w_op);
  ge::AttrUtils::SetInt(src_w_op, FE_IMPLY_TYPE, 6);

  OpDescPtr src_b_op = std::make_shared<OpDesc>("B", "BatchA");
  GeTensorDesc src_b_tensor_desc(GeShape({5,6,7}), ge::FORMAT_DHWNC, ge::DT_FLOAT);
  src_b_tensor_desc.SetOriginFormat(ge::FORMAT_DHWCN);
  src_b_op->AddOutputDesc(src_b_tensor_desc);
  src_b_op->AddInputDesc(src_b_tensor_desc);
  auto src_b_node = graph->AddNode(src_b_op);
  ge::AttrUtils::SetInt(src_b_op, FE_IMPLY_TYPE, 6);

  OpDescPtr src_c_op = std::make_shared<OpDesc>("C", "BatchA");
  GeTensorDesc src_c_tensor_desc(GeShape({5,6,7,8,9}), ge::FORMAT_DHWNC, ge::DT_FLOAT);
  src_c_tensor_desc.SetOriginFormat(ge::FORMAT_DHWNC);
  src_c_op->AddOutputDesc(src_c_tensor_desc);
  src_c_op->AddInputDesc(src_c_tensor_desc);
  auto src_c_node = graph->AddNode(src_c_op);
  ge::AttrUtils::SetInt(src_c_op, FE_IMPLY_TYPE, 6);

  OpDescPtr dst_op = std::make_shared<OpDesc>("batchnorm", "BatchNormV9");
  GeTensorDesc dst_x_tensor_desc(GeShape({1, 2, 3, 4, 5}), ge::FORMAT_NCDHW, ge::DT_FLOAT);
  dst_x_tensor_desc.SetOriginFormat(ge::FORMAT_NCDHW);
  GeTensorDesc dst_w_tensor_desc(GeShape({1, 2, 3, 4, 5}), ge::FORMAT_NDHWC, ge::DT_FLOAT);
  dst_w_tensor_desc.SetOriginFormat(ge::FORMAT_NDHWC);
  GeTensorDesc dst_b_tensor_desc(GeShape({1, 2, 3, 4, 5}), ge::FORMAT_DHWCN, ge::DT_FLOAT);
  dst_b_tensor_desc.SetOriginFormat(ge::FORMAT_DHWCN);
  GeTensorDesc dst_c_tensor_desc(GeShape({1, 2, 3, 4, 5}), ge::FORMAT_DHWNC, ge::DT_FLOAT);
  dst_c_tensor_desc.SetOriginFormat(ge::FORMAT_DHWNC);

  dst_op->AddInputDesc("x", dst_x_tensor_desc);
  dst_op->AddInputDesc("w", dst_w_tensor_desc);
  dst_op->AddInputDesc("b", dst_b_tensor_desc);
  dst_op->AddInputDesc("c", dst_c_tensor_desc);
  dst_op->AddOutputDesc(dst_x_tensor_desc);
  auto dst_node = graph->AddNode(dst_op);
  ge::AttrUtils::SetInt(dst_op, FE_IMPLY_TYPE, 6);
  vector<bool> input_const_vector = {false};
  dst_op->SetIsInputConst(input_const_vector);

  GraphUtils::AddEdge(src_x_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(src_w_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(1));
  GraphUtils::AddEdge(src_b_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(2));
  GraphUtils::AddEdge(src_c_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(3));

  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status status = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));

  ASSERT_EQ(fe::SUCCESS, status);
  int count_node = 0;
  int trans_count = 0;
  int unsqueeze_count = 0;
  vector<vector<int64_t>> trans_dim_result{{1,1,1,5,7}, {1,1,1,5,7}, {1,1,5,6,7}};

  for (auto node : graph->GetDirectNode()) {
    ASSERT_NE(node, nullptr);
    count_node++;
    if (node->GetType() == TRANSPOSE) {
      EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetShape().GetDims(), trans_dim_result[trans_count++]);
      continue;
    }
    if (node->GetType() == "UnsqueezeV2") {
      EXPECT_EQ(node->GetOpDesc()->GetOutputDescPtr(0)->GetShape().GetDims(), trans_dim_result[unsqueeze_count++]);
      continue;
    }
  }
  EXPECT_EQ(count_node, 14);
}

TEST_F(STEST_FE_TRANSOP_INSERT, InsertTransposDNode_Squeeze_NCDHW)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");

  OpDescPtr src_x_op = std::make_shared<OpDesc>("X", "BatchA");
  GeTensorDesc src_x_tensor_desc(GeShape({5,6,7}), ge::FORMAT_NCDHW, ge::DT_FLOAT);
  src_x_tensor_desc.SetOriginFormat(ge::FORMAT_NCDHW);
  src_x_op->AddOutputDesc(src_x_tensor_desc);
  src_x_op->AddInputDesc(src_x_tensor_desc);
  auto src_x_node = graph->AddNode(src_x_op);
  ge::AttrUtils::SetInt(src_x_op, FE_IMPLY_TYPE, 6);

  OpDescPtr src_w_op = std::make_shared<OpDesc>("W", "BatchA");
  GeTensorDesc src_w_tensor_desc(GeShape({5,6,7}), ge::FORMAT_NCDHW, ge::DT_FLOAT);
  src_w_tensor_desc.SetOriginFormat(ge::FORMAT_NDHWC);
  src_w_op->AddOutputDesc(src_w_tensor_desc);
  src_w_op->AddInputDesc(src_w_tensor_desc);
  auto src_w_node = graph->AddNode(src_w_op);
  ge::AttrUtils::SetInt(src_w_op, FE_IMPLY_TYPE, 6);

  OpDescPtr src_b_op = std::make_shared<OpDesc>("B", "BatchA");
  GeTensorDesc src_b_tensor_desc(GeShape({5,6,7}), ge::FORMAT_NCDHW, ge::DT_FLOAT);
  src_b_tensor_desc.SetOriginFormat(ge::FORMAT_DHWCN);
  src_b_op->AddOutputDesc(src_b_tensor_desc);
  src_b_op->AddInputDesc(src_b_tensor_desc);
  auto src_b_node = graph->AddNode(src_b_op);
  ge::AttrUtils::SetInt(src_b_op, FE_IMPLY_TYPE, 6);

  OpDescPtr src_c_op = std::make_shared<OpDesc>("C", "BatchA");
  GeTensorDesc src_c_tensor_desc(GeShape({5,6,7,8,9}), ge::FORMAT_NCDHW, ge::DT_FLOAT);
  src_c_tensor_desc.SetOriginFormat(ge::FORMAT_DHWNC);
  src_c_op->AddOutputDesc(src_c_tensor_desc);
  src_c_op->AddInputDesc(src_c_tensor_desc);
  auto src_c_node = graph->AddNode(src_c_op);
  ge::AttrUtils::SetInt(src_c_op, FE_IMPLY_TYPE, 6);

  OpDescPtr dst_op = std::make_shared<OpDesc>("batchnorm", "BatchNormV9");
  GeTensorDesc dst_x_tensor_desc(GeShape({1, 2}), ge::FORMAT_NCDHW, ge::DT_FLOAT);
  dst_x_tensor_desc.SetOriginFormat(ge::FORMAT_NCDHW);
  GeTensorDesc dst_w_tensor_desc(GeShape({1, 2}), ge::FORMAT_NDHWC, ge::DT_FLOAT);
  dst_w_tensor_desc.SetOriginFormat(ge::FORMAT_NDHWC);
  GeTensorDesc dst_b_tensor_desc(GeShape({1, 2}), ge::FORMAT_DHWCN, ge::DT_FLOAT);
  dst_b_tensor_desc.SetOriginFormat(ge::FORMAT_DHWCN);
  GeTensorDesc dst_c_tensor_desc(GeShape({1, 2}), ge::FORMAT_DHWNC, ge::DT_FLOAT);
  dst_c_tensor_desc.SetOriginFormat(ge::FORMAT_DHWNC);

  dst_op->AddInputDesc("x", dst_x_tensor_desc);
  dst_op->AddInputDesc("w", dst_w_tensor_desc);
  dst_op->AddInputDesc("b", dst_b_tensor_desc);
  dst_op->AddInputDesc("c", dst_c_tensor_desc);
  dst_op->AddOutputDesc(dst_x_tensor_desc);
  auto dst_node = graph->AddNode(dst_op);
  ge::AttrUtils::SetInt(dst_op, FE_IMPLY_TYPE, 6);
  vector<bool> input_const_vector = {false};
  dst_op->SetIsInputConst(input_const_vector);

  GraphUtils::AddEdge(src_x_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(src_w_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(1));
  GraphUtils::AddEdge(src_b_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(2));
  GraphUtils::AddEdge(src_c_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(3));

  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status status = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));

  ASSERT_EQ(fe::SUCCESS, status);
  int count_node = 0;
  int trans_count = 0;
  int unsqueeze_count = 0;
  vector<vector<int64_t>> trans_dim_result{{1,1,5,6,7}, {1,1,5,6,7}, {5,6,7,8,9}};
  vector<vector<int64_t>> squeeze_dim_result{{1,5,6,7,1}, {5,6,7,1,1}, {7,8,9,5,6}};

  for (auto node : graph->GetDirectNode()) {
    ASSERT_NE(node, nullptr);
    count_node++;
    if (node->GetType() == TRANSPOSE) {
      EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetShape().GetDims(), trans_dim_result[trans_count++]);
      continue;
    }
    if (node->GetType() == "SqueezeV2") {
      EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetShape().GetDims(), squeeze_dim_result[unsqueeze_count++]);
      continue;
    }
  }
  EXPECT_EQ(count_node, 14);
}

TEST_F(STEST_FE_TRANSOP_INSERT, InsertTransposDNode_Squeeze_NDHWC)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");

  OpDescPtr src_x_op = std::make_shared<OpDesc>("X", "BatchA");
  GeTensorDesc src_x_tensor_desc(GeShape({5,6,7}), ge::FORMAT_NDHWC, ge::DT_FLOAT);
  src_x_tensor_desc.SetOriginFormat(ge::FORMAT_NCDHW);
  src_x_op->AddOutputDesc(src_x_tensor_desc);
  src_x_op->AddInputDesc(src_x_tensor_desc);
  auto src_x_node = graph->AddNode(src_x_op);
  ge::AttrUtils::SetInt(src_x_op, FE_IMPLY_TYPE, 6);

  OpDescPtr src_w_op = std::make_shared<OpDesc>("W", "BatchA");
  GeTensorDesc src_w_tensor_desc(GeShape({5,6,7}), ge::FORMAT_NDHWC, ge::DT_FLOAT);
  src_w_tensor_desc.SetOriginFormat(ge::FORMAT_NDHWC);
  src_w_op->AddOutputDesc(src_w_tensor_desc);
  src_w_op->AddInputDesc(src_w_tensor_desc);
  auto src_w_node = graph->AddNode(src_w_op);
  ge::AttrUtils::SetInt(src_w_op, FE_IMPLY_TYPE, 6);

  OpDescPtr src_b_op = std::make_shared<OpDesc>("B", "BatchA");
  GeTensorDesc src_b_tensor_desc(GeShape({5,6,7}), ge::FORMAT_NDHWC, ge::DT_FLOAT);
  src_b_tensor_desc.SetOriginFormat(ge::FORMAT_DHWCN);
  src_b_op->AddOutputDesc(src_b_tensor_desc);
  src_b_op->AddInputDesc(src_b_tensor_desc);
  auto src_b_node = graph->AddNode(src_b_op);
  ge::AttrUtils::SetInt(src_b_op, FE_IMPLY_TYPE, 6);

  OpDescPtr src_c_op = std::make_shared<OpDesc>("C", "BatchA");
  GeTensorDesc src_c_tensor_desc(GeShape({5,6,7,8,9}), ge::FORMAT_NDHWC, ge::DT_FLOAT);
  src_c_tensor_desc.SetOriginFormat(ge::FORMAT_DHWNC);
  src_c_op->AddOutputDesc(src_c_tensor_desc);
  src_c_op->AddInputDesc(src_c_tensor_desc);
  auto src_c_node = graph->AddNode(src_c_op);
  ge::AttrUtils::SetInt(src_c_op, FE_IMPLY_TYPE, 6);

  OpDescPtr dst_op = std::make_shared<OpDesc>("batchnorm", "BatchNormV9");
  GeTensorDesc dst_x_tensor_desc(GeShape({1, 2}), ge::FORMAT_NCDHW, ge::DT_FLOAT);
  dst_x_tensor_desc.SetOriginFormat(ge::FORMAT_NCDHW);
  GeTensorDesc dst_w_tensor_desc(GeShape({1, 2}), ge::FORMAT_NDHWC, ge::DT_FLOAT);
  dst_w_tensor_desc.SetOriginFormat(ge::FORMAT_NDHWC);
  GeTensorDesc dst_b_tensor_desc(GeShape({1, 2}), ge::FORMAT_DHWCN, ge::DT_FLOAT);
  dst_b_tensor_desc.SetOriginFormat(ge::FORMAT_DHWCN);
  GeTensorDesc dst_c_tensor_desc(GeShape({1, 2}), ge::FORMAT_DHWNC, ge::DT_FLOAT);
  dst_c_tensor_desc.SetOriginFormat(ge::FORMAT_DHWNC);

  dst_op->AddInputDesc("x", dst_x_tensor_desc);
  dst_op->AddInputDesc("w", dst_w_tensor_desc);
  dst_op->AddInputDesc("b", dst_b_tensor_desc);
  dst_op->AddInputDesc("c", dst_c_tensor_desc);
  dst_op->AddOutputDesc(dst_x_tensor_desc);
  auto dst_node = graph->AddNode(dst_op);
  ge::AttrUtils::SetInt(dst_op, FE_IMPLY_TYPE, 6);
  vector<bool> input_const_vector = {false};
  dst_op->SetIsInputConst(input_const_vector);

  GraphUtils::AddEdge(src_x_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(src_w_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(1));
  GraphUtils::AddEdge(src_b_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(2));
  GraphUtils::AddEdge(src_c_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(3));

  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status status = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));

  ASSERT_EQ(fe::SUCCESS, status);
  int count_node = 0;
  int trans_count = 0;
  int unsqueeze_count = 0;
  vector<vector<int64_t>> trans_dim_result{{1,1,5,6,7}, {1,1,5,6,7}, {5,6,7,8,9}};
  vector<vector<int64_t>> squeeze_dim_result{{1,7,1,5,6}, {1,5,6,7,1}, {6,7,8,5,9}};

  for (auto node : graph->GetDirectNode()) {
    ASSERT_NE(node, nullptr);
    count_node++;
    if (node->GetType() == TRANSPOSE) {
      EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetShape().GetDims(), trans_dim_result[trans_count++]);
      continue;
    }
    if (node->GetType() == "SqueezeV2") {
      EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetShape().GetDims(), squeeze_dim_result[unsqueeze_count++]);
      continue;
    }
  }
  EXPECT_EQ(count_node, 14);
}

TEST_F(STEST_FE_TRANSOP_INSERT, InsertTransposDNode_Squeeze_DHWCN)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");

  OpDescPtr src_x_op = std::make_shared<OpDesc>("X", "BatchA");
  GeTensorDesc src_x_tensor_desc(GeShape({5,6,7}), ge::FORMAT_DHWCN, ge::DT_FLOAT);
  src_x_tensor_desc.SetOriginFormat(ge::FORMAT_NCDHW);
  src_x_op->AddOutputDesc(src_x_tensor_desc);
  src_x_op->AddInputDesc(src_x_tensor_desc);
  auto src_x_node = graph->AddNode(src_x_op);
  ge::AttrUtils::SetInt(src_x_op, FE_IMPLY_TYPE, 6);

  OpDescPtr src_w_op = std::make_shared<OpDesc>("W", "BatchA");
  GeTensorDesc src_w_tensor_desc(GeShape({5,6,7}), ge::FORMAT_DHWCN, ge::DT_FLOAT);
  src_w_tensor_desc.SetOriginFormat(ge::FORMAT_NDHWC);
  src_w_op->AddOutputDesc(src_w_tensor_desc);
  src_w_op->AddInputDesc(src_w_tensor_desc);
  auto src_w_node = graph->AddNode(src_w_op);
  ge::AttrUtils::SetInt(src_w_op, FE_IMPLY_TYPE, 6);

  OpDescPtr src_b_op = std::make_shared<OpDesc>("B", "BatchA");
  GeTensorDesc src_b_tensor_desc(GeShape({5,6,7}), ge::FORMAT_DHWCN, ge::DT_FLOAT);
  src_b_tensor_desc.SetOriginFormat(ge::FORMAT_DHWCN);
  src_b_op->AddOutputDesc(src_b_tensor_desc);
  src_b_op->AddInputDesc(src_b_tensor_desc);
  auto src_b_node = graph->AddNode(src_b_op);
  ge::AttrUtils::SetInt(src_b_op, FE_IMPLY_TYPE, 6);

  OpDescPtr src_c_op = std::make_shared<OpDesc>("C", "BatchA");
  GeTensorDesc src_c_tensor_desc(GeShape({5,6,7,8,9}), ge::FORMAT_DHWCN, ge::DT_FLOAT);
  src_c_tensor_desc.SetOriginFormat(ge::FORMAT_DHWNC);
  src_c_op->AddOutputDesc(src_c_tensor_desc);
  src_c_op->AddInputDesc(src_c_tensor_desc);
  auto src_c_node = graph->AddNode(src_c_op);
  ge::AttrUtils::SetInt(src_c_op, FE_IMPLY_TYPE, 6);

  OpDescPtr dst_op = std::make_shared<OpDesc>("batchnorm", "BatchNormV9");
  GeTensorDesc dst_x_tensor_desc(GeShape({1, 2}), ge::FORMAT_NCDHW, ge::DT_FLOAT);
  dst_x_tensor_desc.SetOriginFormat(ge::FORMAT_NCDHW);
  GeTensorDesc dst_w_tensor_desc(GeShape({1, 2}), ge::FORMAT_NDHWC, ge::DT_FLOAT);
  dst_w_tensor_desc.SetOriginFormat(ge::FORMAT_NDHWC);
  GeTensorDesc dst_b_tensor_desc(GeShape({1, 2}), ge::FORMAT_DHWCN, ge::DT_FLOAT);
  dst_b_tensor_desc.SetOriginFormat(ge::FORMAT_DHWCN);
  GeTensorDesc dst_c_tensor_desc(GeShape({1, 2}), ge::FORMAT_DHWNC, ge::DT_FLOAT);
  dst_c_tensor_desc.SetOriginFormat(ge::FORMAT_DHWNC);

  dst_op->AddInputDesc("x", dst_x_tensor_desc);
  dst_op->AddInputDesc("w", dst_w_tensor_desc);
  dst_op->AddInputDesc("b", dst_b_tensor_desc);
  dst_op->AddInputDesc("c", dst_c_tensor_desc);
  dst_op->AddOutputDesc(dst_x_tensor_desc);
  auto dst_node = graph->AddNode(dst_op);
  ge::AttrUtils::SetInt(dst_op, FE_IMPLY_TYPE, 6);
  vector<bool> input_const_vector = {false};
  dst_op->SetIsInputConst(input_const_vector);

  GraphUtils::AddEdge(src_x_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(src_w_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(1));
  GraphUtils::AddEdge(src_b_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(2));
  GraphUtils::AddEdge(src_c_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(3));

  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status status = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));

  ASSERT_EQ(fe::SUCCESS, status);
  int count_node = 0;
  int trans_count = 0;
  int unsqueeze_count = 0;
  vector<vector<int64_t>> trans_dim_result{{1,1,5,6,7}, {1,1,5,6,7}, {5,6,7,8,9}};
  vector<vector<int64_t>> squeeze_dim_result{{7,6,1,1,5}, {7,1,1,5,6}, {5,6,7,9,8}};

  for (auto node : graph->GetDirectNode()) {
    ASSERT_NE(node, nullptr);
    count_node++;
    if (node->GetType() == TRANSPOSE) {
      EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetShape().GetDims(), trans_dim_result[trans_count++]);
      continue;
    }
    if (node->GetType() == "SqueezeV2") {
      EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetShape().GetDims(), squeeze_dim_result[unsqueeze_count++]);
      continue;
    }
  }
  EXPECT_EQ(count_node, 14);
}

TEST_F(STEST_FE_TRANSOP_INSERT, InsertTransposDNode_Squeeze_DHWNC)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");

  OpDescPtr src_x_op = std::make_shared<OpDesc>("X", "BatchA");
  GeTensorDesc src_x_tensor_desc(GeShape({5,6,7}), ge::FORMAT_DHWNC, ge::DT_FLOAT);
  src_x_tensor_desc.SetOriginFormat(ge::FORMAT_NCDHW);
  src_x_op->AddOutputDesc(src_x_tensor_desc);
  src_x_op->AddInputDesc(src_x_tensor_desc);
  auto src_x_node = graph->AddNode(src_x_op);
  ge::AttrUtils::SetInt(src_x_op, FE_IMPLY_TYPE, 6);

  OpDescPtr src_w_op = std::make_shared<OpDesc>("W", "BatchA");
  GeTensorDesc src_w_tensor_desc(GeShape({5,6,7}), ge::FORMAT_DHWNC, ge::DT_FLOAT);
  src_w_tensor_desc.SetOriginFormat(ge::FORMAT_NDHWC);
  src_w_op->AddOutputDesc(src_w_tensor_desc);
  src_w_op->AddInputDesc(src_w_tensor_desc);
  auto src_w_node = graph->AddNode(src_w_op);
  ge::AttrUtils::SetInt(src_w_op, FE_IMPLY_TYPE, 6);

  OpDescPtr src_b_op = std::make_shared<OpDesc>("B", "BatchA");
  GeTensorDesc src_b_tensor_desc(GeShape({5,6,7}), ge::FORMAT_DHWNC, ge::DT_FLOAT);
  src_b_tensor_desc.SetOriginFormat(ge::FORMAT_DHWCN);
  src_b_op->AddOutputDesc(src_b_tensor_desc);
  src_b_op->AddInputDesc(src_b_tensor_desc);
  auto src_b_node = graph->AddNode(src_b_op);
  ge::AttrUtils::SetInt(src_b_op, FE_IMPLY_TYPE, 6);

  OpDescPtr src_c_op = std::make_shared<OpDesc>("C", "BatchA");
  GeTensorDesc src_c_tensor_desc(GeShape({5,6,7,8,9}), ge::FORMAT_DHWNC, ge::DT_FLOAT);
  src_c_tensor_desc.SetOriginFormat(ge::FORMAT_DHWNC);
  src_c_op->AddOutputDesc(src_c_tensor_desc);
  src_c_op->AddInputDesc(src_c_tensor_desc);
  auto src_c_node = graph->AddNode(src_c_op);
  ge::AttrUtils::SetInt(src_c_op, FE_IMPLY_TYPE, 6);

  OpDescPtr dst_op = std::make_shared<OpDesc>("batchnorm", "BatchNormV9");
  GeTensorDesc dst_x_tensor_desc(GeShape({1, 2}), ge::FORMAT_NCDHW, ge::DT_FLOAT);
  dst_x_tensor_desc.SetOriginFormat(ge::FORMAT_NCDHW);
  GeTensorDesc dst_w_tensor_desc(GeShape({1, 2}), ge::FORMAT_NDHWC, ge::DT_FLOAT);
  dst_w_tensor_desc.SetOriginFormat(ge::FORMAT_NDHWC);
  GeTensorDesc dst_b_tensor_desc(GeShape({1, 2}), ge::FORMAT_DHWCN, ge::DT_FLOAT);
  dst_b_tensor_desc.SetOriginFormat(ge::FORMAT_DHWCN);
  GeTensorDesc dst_c_tensor_desc(GeShape({1, 2}), ge::FORMAT_DHWNC, ge::DT_FLOAT);
  dst_c_tensor_desc.SetOriginFormat(ge::FORMAT_DHWNC);

  dst_op->AddInputDesc("x", dst_x_tensor_desc);
  dst_op->AddInputDesc("w", dst_w_tensor_desc);
  dst_op->AddInputDesc("b", dst_b_tensor_desc);
  dst_op->AddInputDesc("c", dst_c_tensor_desc);
  dst_op->AddOutputDesc(dst_x_tensor_desc);
  auto dst_node = graph->AddNode(dst_op);
  ge::AttrUtils::SetInt(dst_op, FE_IMPLY_TYPE, 6);
  vector<bool> input_const_vector = {false};
  dst_op->SetIsInputConst(input_const_vector);

  GraphUtils::AddEdge(src_x_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(src_w_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(1));
  GraphUtils::AddEdge(src_b_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(2));
  GraphUtils::AddEdge(src_c_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(3));

  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status status = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));

  ASSERT_EQ(fe::SUCCESS, status);
  int count_node = 0;
  int trans_count = 0;
  int unsqueeze_count = 0;
  vector<vector<int64_t>> trans_dim_result{{1,1,5,6,7}, {1,1,5,6,7}, {1,1,5,6,7}};
  vector<vector<int64_t>> squeeze_dim_result{{6,7,1,1,5}, {6,1,1,5,7}, {1,1,5,7,6}};

  for (auto node : graph->GetDirectNode()) {
    ASSERT_NE(node, nullptr);
    count_node++;
    if (node->GetType() == TRANSPOSE) {
      EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetShape().GetDims(), trans_dim_result[trans_count++]);
      continue;
    }
    if (node->GetType() == "SqueezeV2") {
      EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetShape().GetDims(), squeeze_dim_result[unsqueeze_count++]);
      continue;
    }
  }
  EXPECT_EQ(count_node, 14);
}

TEST_F(STEST_FE_TRANSOP_INSERT, InsertTransposDNode_2DTransTo3D)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");

  OpDescPtr src_x_op = std::make_shared<OpDesc>("b", "B");
  GeTensorDesc src_x_tensor_desc(GeShape({5,7}), ge::FORMAT_NCHW, ge::DT_FLOAT);
  src_x_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  src_x_op->AddOutputDesc(src_x_tensor_desc);
  src_x_op->AddInputDesc(src_x_tensor_desc);
  auto src_x_node = graph->AddNode(src_x_op);
  ge::AttrUtils::SetInt(src_x_op, FE_IMPLY_TYPE, 6);

  OpDescPtr dst_op = std::make_shared<OpDesc>("conv3d", "BatchNormV9");
  GeTensorDesc dst_x_tensor_desc(GeShape({1, 2, 3, 4}), ge::FORMAT_NCDHW, ge::DT_FLOAT);
  dst_x_tensor_desc.SetOriginFormat(ge::FORMAT_NCDHW);

  dst_op->AddInputDesc("x", dst_x_tensor_desc);
  dst_op->AddOutputDesc(dst_x_tensor_desc);
  auto dst_node = graph->AddNode(dst_op);
  ge::AttrUtils::SetInt(dst_op, FE_IMPLY_TYPE, 6);
  vector<bool> input_const_vector = {false};
  dst_op->SetIsInputConst(input_const_vector);

  GraphUtils::AddEdge(src_x_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(0));

  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status status = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));

  ASSERT_EQ(fe::SUCCESS, status);
  int count_node = 0;
  for (auto node : graph->GetDirectNode()) {
    ASSERT_NE(node, nullptr);
    count_node++;
  }
  EXPECT_EQ(count_node, 2);
}

TEST_F(STEST_FE_TRANSOP_INSERT, InsertTransposDNode_3DTransTo2D)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");

  OpDescPtr src_x_op = std::make_shared<OpDesc>("X", "BatchA");
  GeTensorDesc src_x_tensor_desc(GeShape({5,7}), ge::FORMAT_DHWNC, ge::DT_FLOAT);
  src_x_tensor_desc.SetOriginFormat(ge::FORMAT_NCDHW);
  src_x_op->AddOutputDesc(src_x_tensor_desc);
  src_x_op->AddInputDesc(src_x_tensor_desc);
  auto src_x_node = graph->AddNode(src_x_op);
  ge::AttrUtils::SetInt(src_x_op, FE_IMPLY_TYPE, 6);

  OpDescPtr dst_op = std::make_shared<OpDesc>("conv2d", "Conv");
  GeTensorDesc dst_x_tensor_desc(GeShape({1, 2, 3, 4}), ge::FORMAT_NCHW, ge::DT_FLOAT);
  dst_x_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);

  dst_op->AddInputDesc("x", dst_x_tensor_desc);
  dst_op->AddOutputDesc(dst_x_tensor_desc);
  auto dst_node = graph->AddNode(dst_op);
  ge::AttrUtils::SetInt(dst_op, FE_IMPLY_TYPE, 6);
  vector<bool> input_const_vector = {false};
  dst_op->SetIsInputConst(input_const_vector);

  GraphUtils::AddEdge(src_x_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(0));

  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status status = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));

  ASSERT_EQ(fe::SUCCESS, status);
  int count_node = 0;
  for (auto node : graph->GetDirectNode()) {
    ASSERT_NE(node, nullptr);
    count_node++;
    if (node->GetType() == TRANSPOSE) {
      vector<int64_t> trans_dim_result({1,1,1,5,7});
      EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(0)->GetShape().GetDims(), trans_dim_result);
      continue;
    }
  }

  EXPECT_EQ(count_node, 4);
}

/* 2D->NCHW->NHWC, reshape type of E is nc*/
TEST_F(STEST_FE_TRANSOP_INSERT, InsertReshapeNode) {
  // src:cce op, dst:cce op
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");

  OpDescPtr src_op = std::make_shared<OpDesc>("EE", "EE");
  GeTensorDesc src_tensor_desc(GeShape({5, 2}), ge::FORMAT_NCHW, ge::DT_FLOAT);
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);

  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);
  ge::AttrUtils::SetInt(src_op, FE_IMPLY_TYPE, 6);

  OpDescPtr dst_op = std::make_shared<OpDesc>("D", "D");
  GeTensorDesc dst_tensor_desc(GeShape({5, 5, 5, 5}), ge::FORMAT_NHWC, ge::DT_FLOAT);
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

  ASSERT_EQ(fe::SUCCESS, status);
  int count_node = 0;
  for (auto node : graph->GetDirectNode()) {
    ASSERT_NE(node, nullptr);
    count_node++;
    if (node->GetType() == TRANSPOSE) {
      ge::GeShape shape_check = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
      auto size = shape_check.GetDimNum();
      ASSERT_EQ(size, 4);
      vector<int64_t> input_vec_of_b =shape_check.GetDims();
      EXPECT_EQ(input_vec_of_b[0], 1);
      EXPECT_EQ(input_vec_of_b[1], 2);
      EXPECT_EQ(input_vec_of_b[2], 1);
      EXPECT_EQ(input_vec_of_b[3], 5);
    }
    if (node->GetType() == "D") {
      {
        ge::GeShape shape_check = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        auto size = shape_check.GetDimNum();
        ASSERT_EQ(size, 4);
        vector<int64_t> input_vec_of_b =shape_check.GetDims();
        EXPECT_EQ(input_vec_of_b[0], 5);
        EXPECT_EQ(input_vec_of_b[1], 5);
        EXPECT_EQ(input_vec_of_b[2], 5);
        EXPECT_EQ(input_vec_of_b[3], 5);
      }
      {
        ge::GeShape shape_check = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        auto size = shape_check.GetDimNum();
        ASSERT_EQ(size, 4);
        vector<int64_t> input_vec_of_b =shape_check.GetDims();
        EXPECT_EQ(input_vec_of_b[0], 5);
        EXPECT_EQ(input_vec_of_b[1], 5);
        EXPECT_EQ(input_vec_of_b[2], 5);
        EXPECT_EQ(input_vec_of_b[3], 5);
      }
    }
  }
  EXPECT_EQ(count_node, 4);
}

TEST_F(STEST_FE_TRANSOP_INSERT, AddReshapeOp_01) {
  string reshape_type = "NC";
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  GeTensorDesc src_tensor_desc(GeShape({1, 256}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
  GeTensorDesc dst_tensor_desc(GeShape({1, 256, 1, 1}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);

  OpDescPtr dst_op = std::make_shared<OpDesc>("B", "B");
  dst_op->AddInputDesc(dst_tensor_desc);
  dst_op->AddOutputDesc(dst_tensor_desc);
  auto dst_node = graph->AddNode(dst_op);

  GraphUtils::AddEdge(src_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(0));

  TransInfoPtr trans_info_ptr = std::make_shared<TransInfo>();
  trans_info_ptr->src_out_shape = GeShape({1, 256});
  trans_info_ptr->dst_in_shape = GeShape({1, 256,1,1});
  trans_info_ptr->src_reshape_type = reshape_type;
  trans_info_ptr->dst_reshape_type = reshape_type;
  trans_info_ptr->src_out_data_type = ge::DT_FLOAT16;
  trans_info_ptr->dst_in_data_type = ge::DT_FLOAT16;

  trans_info_ptr->src_op_desc = src_op;
  trans_info_ptr->dst_op_desc = dst_op;
  trans_info_ptr->src_node_ptr = src_node;
  trans_info_ptr->dst_node_ptr = dst_node;
  trans_info_ptr->src_anchor = src_node->GetOutDataAnchor(0);
  trans_info_ptr->dst_anchor = dst_node->GetInDataAnchor(0);


  TransNodeReshapeGenerator trans_op_insert(fe_ops_kernel_info_store_ptr_, trans_info_ptr);
  Status ret = trans_op_insert.AddTransNode(*graph.get(), trans_info_ptr);
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(STEST_FE_TRANSOP_INSERT, AddReshapeOp_02) {
  string reshape_type = "NC";
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  GeTensorDesc src_tensor_desc(GeShape({1, 256}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
  GeTensorDesc dst_tensor_desc(GeShape({1, 256, 1, 1}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);

  OpDescPtr dst_op = std::make_shared<OpDesc>("B", "B");
  dst_op->AddInputDesc(dst_tensor_desc);
  dst_op->AddOutputDesc(dst_tensor_desc);
  auto dst_node = graph->AddNode(dst_op);


  TransInfoPtr trans_info_ptr = std::make_shared<TransInfo>();
  trans_info_ptr->src_out_shape = GeShape({1, 256});
  trans_info_ptr->dst_in_shape = GeShape({1, 256,1,1});
  trans_info_ptr->src_reshape_type = reshape_type;
  trans_info_ptr->dst_reshape_type = reshape_type;
  trans_info_ptr->src_out_data_type = ge::DT_FLOAT16;
  trans_info_ptr->dst_in_data_type = ge::DT_FLOAT16;

  trans_info_ptr->src_op_desc = src_op;
  trans_info_ptr->dst_op_desc = dst_op;
  trans_info_ptr->src_node_ptr = src_node;
  trans_info_ptr->dst_node_ptr = dst_node;
  trans_info_ptr->src_anchor = src_node->GetOutDataAnchor(0);
  trans_info_ptr->dst_anchor = dst_node->GetInDataAnchor(0);


  TransNodeReshapeGenerator trans_op_insert(fe_ops_kernel_info_store_ptr_, trans_info_ptr);
  Status ret = trans_op_insert.AddTransNode(*graph.get(), trans_info_ptr);
  /* If two anchors is not connected, it will still return success. */
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(STEST_FE_TRANSOP_INSERT, MergeTwoTransDataOp) {
  // src:cce op, dst:cce op
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");

  GeTensorDesc src_tensor_desc(GeShape({1, 256, 256, 512, 4}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  GeTensorDesc dst_tensor_desc(GeShape({1, 1024, 256, 512}), ge::FORMAT_NCHW, ge::DT_FLOAT16);

  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);

  OpDescPtr dst_op = std::make_shared<OpDesc>("B", "B");
  dst_op->AddInputDesc(src_tensor_desc);
  dst_op->AddOutputDesc(src_tensor_desc);
  auto dst_node = graph->AddNode(dst_op);

  OpDescPtr dst_op_trans_data1 = std::make_shared<OpDesc>("Transdata1", "TransData");
  dst_op_trans_data1->AddInputDesc(src_tensor_desc);
  dst_op_trans_data1->AddOutputDesc(dst_tensor_desc);
  auto trandata_node1 = graph->AddNode(dst_op_trans_data1);

  OpDescPtr dst_op_trans_data2 = std::make_shared<OpDesc>("Transdata2", "TransData");
  dst_op_trans_data2->AddInputDesc(dst_tensor_desc);
  dst_op_trans_data2->AddOutputDesc(src_tensor_desc);
  auto trandata_node2 = graph->AddNode(dst_op_trans_data2);

  GraphUtils::AddEdge(src_node->GetOutDataAnchor(0), trandata_node1->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node1->GetOutDataAnchor(0), trandata_node2->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node2->GetOutDataAnchor(0), dst_node->GetInDataAnchor(0));

  TransNodeMerging trans_op_merger;
  trans_op_merger.MergeAllTransOps(*(graph.get()), {}, false);
  uint32_t count_node = 0;
  for (auto node: graph->GetDirectNode()) {
    EXPECT_NE(node, nullptr);
    EXPECT_NE(node->GetType(), "TransData");
    count_node++;
  }
  EXPECT_EQ(count_node,2);
}

TEST_F(STEST_FE_TRANSOP_INSERT, MergeTwoTransposeOp) {
  // src:cce op, dst:cce op
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");

  GeTensorDesc src_tensor_desc(GeShape({7, 3, 10, 11}), ge::FORMAT_NHWC, ge::DT_FLOAT16);
  GeTensorDesc dst_tensor_desc(GeShape({7, 11, 3, 10}), ge::FORMAT_NCHW, ge::DT_FLOAT16);

  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);

  OpDescPtr dst_op = std::make_shared<OpDesc>("B", "B");
  dst_op->AddInputDesc(dst_tensor_desc);
  dst_op->AddOutputDesc(dst_tensor_desc);
  auto dst_node = graph->AddNode(dst_op);

  OpDescPtr dst_op_trans_pose1 = std::make_shared<OpDesc>("Transpose1", "TransposeD");
  dst_op_trans_pose1->AddInputDesc(src_tensor_desc);
  dst_op_trans_pose1->AddOutputDesc(dst_tensor_desc);
  auto tranpose_node1 = graph->AddNode(dst_op_trans_pose1);

  OpDescPtr dst_op_trans_pose2 = std::make_shared<OpDesc>("Transpose2", "TransposeD");
  dst_op_trans_pose2->AddInputDesc(dst_tensor_desc);
  dst_op_trans_pose2->AddOutputDesc(src_tensor_desc);
  auto tranpose_node2 = graph->AddNode(dst_op_trans_pose2);

  GraphUtils::AddEdge(src_node->GetOutDataAnchor(0), tranpose_node1->GetInDataAnchor(0));
  GraphUtils::AddEdge(tranpose_node1->GetOutDataAnchor(0), tranpose_node2->GetInDataAnchor(0));
  GraphUtils::AddEdge(tranpose_node2->GetOutDataAnchor(0), dst_node->GetInDataAnchor(0));

  TransNodeMerging trans_op_merger;
  trans_op_merger.MergeAllTransOps(*(graph.get()), {}, false);
  uint32_t count_node = 0;
  for (auto node: graph->GetDirectNode()) {
    EXPECT_NE(node, nullptr);
    EXPECT_NE(node->GetType(), "TransposeD");
    count_node++;
  }
  EXPECT_EQ(count_node,2);
}

TEST_F(STEST_FE_TRANSOP_INSERT, MergeFourTransDataOp) {
  // src:cce op, dst:cce op
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");

  GeTensorDesc src_tensor_desc(GeShape({1, 256, 256, 512, 4}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  GeTensorDesc dst_tensor_desc(GeShape({1, 1024, 256, 512}), ge::FORMAT_NCHW, ge::DT_FLOAT16);

  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);

  OpDescPtr dst_op = std::make_shared<OpDesc>("B", "B");
  dst_op->AddInputDesc(src_tensor_desc);
  dst_op->AddOutputDesc(src_tensor_desc);
  auto dst_node = graph->AddNode(dst_op);

  OpDescPtr dst_op_trans_data1 = std::make_shared<OpDesc>("Transdata1", "TransData");
  dst_op_trans_data1->AddInputDesc(src_tensor_desc);
  dst_op_trans_data1->AddOutputDesc(dst_tensor_desc);
  auto trandata_node1 = graph->AddNode(dst_op_trans_data1);

  OpDescPtr dst_op_trans_data2 = std::make_shared<OpDesc>("Transdata2", "TransData");
  dst_op_trans_data2->AddInputDesc(dst_tensor_desc);
  dst_op_trans_data2->AddOutputDesc(src_tensor_desc);
  auto trandata_node2 = graph->AddNode(dst_op_trans_data2);

  OpDescPtr dst_op_trans_data3 = std::make_shared<OpDesc>("Transdata3", "TransData");
  dst_op_trans_data3->AddInputDesc(src_tensor_desc);
  dst_op_trans_data3->AddOutputDesc(dst_tensor_desc);
  auto trandata_node3 = graph->AddNode(dst_op_trans_data3);

  OpDescPtr dst_op_trans_data4 = std::make_shared<OpDesc>("Transdata4", "TransData");
  dst_op_trans_data4->AddInputDesc(dst_tensor_desc);
  dst_op_trans_data4->AddOutputDesc(src_tensor_desc);
  auto trandata_node4 = graph->AddNode(dst_op_trans_data4);

  GraphUtils::AddEdge(src_node->GetOutDataAnchor(0), trandata_node1->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node1->GetOutDataAnchor(0), trandata_node2->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node2->GetOutDataAnchor(0), trandata_node3->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node3->GetOutDataAnchor(0), trandata_node4->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node4->GetOutDataAnchor(0), dst_node->GetInDataAnchor(0));

  TransNodeMerging trans_op_merger;
  trans_op_merger.MergeAllTransOps(*(graph.get()), {}, false);
  uint32_t count_node = 0;
  for (auto node: graph->GetDirectNode()) {
    EXPECT_NE(node, nullptr);
    EXPECT_NE(node->GetType(), "TransData");
    count_node++;
  }
  EXPECT_EQ(count_node,2);
}

void Create4TransdataMergeGraph(ComputeGraphPtr &graph, const ge::GeShape &shape1,
                                const ge::GeShape &shape2) {
  GeTensorDesc src_tensor_desc(shape1, ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  GeTensorDesc dst_tensor_desc(shape2, ge::FORMAT_NCHW, ge::DT_FLOAT16);

  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);

  OpDescPtr dst_op = std::make_shared<OpDesc>("B", "B");
  dst_op->AddInputDesc(src_tensor_desc);
  dst_op->AddOutputDesc(src_tensor_desc);
  auto dst_node = graph->AddNode(dst_op);

  OpDescPtr dst_op_trans_data1 = std::make_shared<OpDesc>("Transdata1", "TransData");
  dst_op_trans_data1->AddInputDesc(src_tensor_desc);
  dst_op_trans_data1->AddOutputDesc(dst_tensor_desc);
  auto trandata_node1 = graph->AddNode(dst_op_trans_data1);

  OpDescPtr dst_op_trans_data2 = std::make_shared<OpDesc>("Transdata2", "TransData");
  dst_op_trans_data2->AddInputDesc(dst_tensor_desc);
  dst_op_trans_data2->AddOutputDesc(src_tensor_desc);
  auto trandata_node2 = graph->AddNode(dst_op_trans_data2);

  OpDescPtr dst_op_trans_data3 = std::make_shared<OpDesc>("Transdata3", "TransData");
  dst_op_trans_data3->AddInputDesc(src_tensor_desc);
  dst_op_trans_data3->AddOutputDesc(dst_tensor_desc);
  auto trandata_node3 = graph->AddNode(dst_op_trans_data3);

  OpDescPtr dst_op_trans_data4 = std::make_shared<OpDesc>("Transdata4", "TransData");
  dst_op_trans_data4->AddInputDesc(dst_tensor_desc);
  dst_op_trans_data4->AddOutputDesc(src_tensor_desc);
  auto trandata_node4 = graph->AddNode(dst_op_trans_data4);

  GraphUtils::AddEdge(src_node->GetOutDataAnchor(0), trandata_node1->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node1->GetOutDataAnchor(0), trandata_node2->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node2->GetOutDataAnchor(0), trandata_node3->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node3->GetOutDataAnchor(0), trandata_node4->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node4->GetOutDataAnchor(0), dst_node->GetInDataAnchor(0));
}

TEST_F(STEST_FE_TRANSOP_INSERT, MergeFourTransDataOpUnknownDimNum) {
  // src:cce op, dst:cce op
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  Create4TransdataMergeGraph(graph, GeShape({1, 256, 256, 512, 4}), GeShape({-2}));

  TransNodeMerging trans_op_merger;
  trans_op_merger.MergeAllTransOps(*(graph.get()), {}, false);
  uint32_t count_node = 0;
  for (auto node: graph->GetDirectNode()) {
    EXPECT_NE(node, nullptr);
    count_node++;
  }
  EXPECT_EQ(count_node,6);
}

TEST_F(STEST_FE_TRANSOP_INSERT, MergeFourTransDataOpUnknownDimNum2) {
  // src:cce op, dst:cce op
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  Create4TransdataMergeGraph(graph, GeShape({-2}), GeShape({-2}));

  TransNodeMerging trans_op_merger;
  trans_op_merger.MergeAllTransOps(*(graph.get()), {}, false);
  uint32_t count_node = 0;
  for (auto node: graph->GetDirectNode()) {
    EXPECT_NE(node, nullptr);
    count_node++;
  }
  EXPECT_EQ(count_node,6);
}

/* Two transdata nodes are different in shape, so they can not merge.
 * So, 4 nodes left and two of which are these two transdata nodes */
TEST_F(STEST_FE_TRANSOP_INSERT, MergeTwoTransDataOp_Abnomal) {
  // src:cce op, dst:cce op
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");

  GeTensorDesc src_tensor_desc(GeShape({1, 256, 256, 512, 4}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  GeTensorDesc dst_tensor_desc(GeShape({1, 1024, 256, 512}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
  GeTensorDesc dst_tensor_desc_01(GeShape({1, 1024, 256, 333}), ge::FORMAT_NCHW, ge::DT_FLOAT16);

  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);

  OpDescPtr dst_op = std::make_shared<OpDesc>("B", "B");
  dst_op->AddInputDesc(src_tensor_desc);
  dst_op->AddOutputDesc(src_tensor_desc);
  auto dst_node = graph->AddNode(dst_op);

  OpDescPtr dst_op_trans_data1 = std::make_shared<OpDesc>("Transdata1", "TransData");
  dst_op_trans_data1->AddInputDesc(src_tensor_desc);
  dst_op_trans_data1->AddOutputDesc(dst_tensor_desc);
  auto trandata_node1 = graph->AddNode(dst_op_trans_data1);

  OpDescPtr dst_op_trans_data2 = std::make_shared<OpDesc>("Transdata2", "TransData");
  dst_op_trans_data2->AddInputDesc(dst_tensor_desc);
  dst_op_trans_data2->AddOutputDesc(src_tensor_desc);
  auto trandata_node2 = graph->AddNode(dst_op_trans_data2);

  OpDescPtr dst_op_trans_data3 = std::make_shared<OpDesc>("Transdata3", "TransData");
  dst_op_trans_data3->AddInputDesc(src_tensor_desc);
  dst_op_trans_data3->AddOutputDesc(dst_tensor_desc_01);
  auto trandata_node3 = graph->AddNode(dst_op_trans_data3);

  OpDescPtr dst_op_trans_data4 = std::make_shared<OpDesc>("Transdata4", "TransData");
  dst_op_trans_data4->AddInputDesc(dst_tensor_desc);
  dst_op_trans_data4->AddOutputDesc(src_tensor_desc);
  auto trandata_node4 = graph->AddNode(dst_op_trans_data4);

  GraphUtils::AddEdge(src_node->GetOutDataAnchor(0), trandata_node1->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node1->GetOutDataAnchor(0), trandata_node2->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node2->GetOutDataAnchor(0), trandata_node3->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node3->GetOutDataAnchor(0), trandata_node4->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node4->GetOutDataAnchor(0), dst_node->GetInDataAnchor(0));

  TransNodeMerging trans_op_merger;
  trans_op_merger.MergeAllTransOps(*(graph.get()), {}, false);
  uint32_t count_node = 0;
  for (auto node: graph->GetDirectNode()) {
    EXPECT_NE(node, nullptr);
    if(node->GetType() == "TransData") {
      EXPECT_NE(node->GetName(), "Transdata1");
      EXPECT_NE(node->GetName(), "Transdata2");
    }
    count_node++;
  }
  EXPECT_EQ(count_node,4);
}

/* One transdata node can not find its lover, so it can not be merged.
 * So, 3 nodes left and one of which is the transdata nodes. */
TEST_F(STEST_FE_TRANSOP_INSERT, MergeThreeTransDataOp_Abnomal) {
  // src:cce op, dst:cce op
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");

  GeTensorDesc src_tensor_desc(GeShape({1, 256, 256, 512, 4}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  GeTensorDesc dst_tensor_desc(GeShape({1, 1024, 256, 512}), ge::FORMAT_NCHW, ge::DT_FLOAT16);

  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);

  OpDescPtr dst_op = std::make_shared<OpDesc>("B", "B");
  dst_op->AddInputDesc(src_tensor_desc);
  dst_op->AddOutputDesc(src_tensor_desc);
  auto dst_node = graph->AddNode(dst_op);

  OpDescPtr dst_op_trans_data1 = std::make_shared<OpDesc>("Transdata1", "TransData");
  dst_op_trans_data1->AddInputDesc(src_tensor_desc);
  dst_op_trans_data1->AddOutputDesc(dst_tensor_desc);
  auto trandata_node1 = graph->AddNode(dst_op_trans_data1);

  OpDescPtr dst_op_trans_data2 = std::make_shared<OpDesc>("Transdata2", "TransData");
  dst_op_trans_data2->AddInputDesc(dst_tensor_desc);
  dst_op_trans_data2->AddOutputDesc(src_tensor_desc);
  auto trandata_node2 = graph->AddNode(dst_op_trans_data2);

  OpDescPtr dst_op_trans_data3 = std::make_shared<OpDesc>("Transdata3", "TransData");
  dst_op_trans_data3->AddInputDesc(src_tensor_desc);
  dst_op_trans_data3->AddOutputDesc(src_tensor_desc);
  auto trandata_node3 = graph->AddNode(dst_op_trans_data3);

  GraphUtils::AddEdge(src_node->GetOutDataAnchor(0), trandata_node1->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node1->GetOutDataAnchor(0), trandata_node2->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node2->GetOutDataAnchor(0), trandata_node3->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node3->GetOutDataAnchor(0), dst_node->GetInDataAnchor(0));

  TransNodeMerging trans_op_merger;
  trans_op_merger.MergeAllTransOps(*(graph.get()), {}, false);
  uint32_t count_node = 0;
  for (auto node: graph->GetDirectNode()) {
    EXPECT_NE(node, nullptr);
    if(node->GetType() == "TransData") {
      EXPECT_EQ(node->GetName(), "Transdata3");
      int64_t topo_id = node->GetOpDesc()->GetId();
      EXPECT_EQ(topo_id, 4);
    }
    count_node++;
  }
  EXPECT_EQ(count_node,3);
}

TEST_F(STEST_FE_TRANSOP_INSERT, MergeTwoTransDataAndTwoCastOp) {
  // src:cce op, dst:cce op
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");

  GeTensorDesc src_tensor_desc(GeShape({128, 256, 256, 512, 4}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  GeTensorDesc dst_tensor_desc(GeShape({128, 1024, 256, 512}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
  GeTensorDesc dst_tensor_desc_cast(GeShape({128, 256, 512, 1024}), ge::FORMAT_NHWC, ge::DT_FLOAT16);
  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);

  OpDescPtr dst_op = std::make_shared<OpDesc>("B", "B");
  dst_op->AddInputDesc(src_tensor_desc);
  dst_op->AddOutputDesc(src_tensor_desc);
  auto dst_node = graph->AddNode(dst_op);

  OpDescPtr dst_op_trans_data1 = std::make_shared<OpDesc>("Transdata1", "TransData");
  dst_op_trans_data1->AddInputDesc(src_tensor_desc);
  dst_op_trans_data1->AddOutputDesc(dst_tensor_desc);
  auto trandata_node1 = graph->AddNode(dst_op_trans_data1);

  OpDescPtr dst_op_trans_data2 = std::make_shared<OpDesc>("Cast1", "Cast");
  dst_op_trans_data2->AddInputDesc(dst_tensor_desc);
  dst_op_trans_data2->AddOutputDesc(dst_tensor_desc_cast);
  auto trandata_node2 = graph->AddNode(dst_op_trans_data2);

  OpDescPtr dst_op_trans_data3 = std::make_shared<OpDesc>("Cast2", "Cast");
  dst_op_trans_data3->AddInputDesc(dst_tensor_desc_cast);
  dst_op_trans_data3->AddOutputDesc(dst_tensor_desc);
  auto trandata_node3 = graph->AddNode(dst_op_trans_data3);

  OpDescPtr dst_op_trans_data4 = std::make_shared<OpDesc>("Transdata4", "TransData");
  dst_op_trans_data4->AddInputDesc(dst_tensor_desc);
  dst_op_trans_data4->AddOutputDesc(src_tensor_desc);
  auto trandata_node4 = graph->AddNode(dst_op_trans_data4);

  GraphUtils::AddEdge(src_node->GetOutDataAnchor(0), trandata_node1->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node1->GetOutDataAnchor(0), trandata_node2->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node2->GetOutDataAnchor(0), trandata_node3->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node3->GetOutDataAnchor(0), trandata_node4->GetInDataAnchor(0));
  GraphUtils::AddEdge(trandata_node4->GetOutDataAnchor(0), dst_node->GetInDataAnchor(0));

  TransNodeMerging trans_op_merger;
  trans_op_merger.MergeAllTransOps(*(graph.get()), {}, false);
  uint32_t count_node = 0;
  for (auto node: graph->GetDirectNode()) {
    EXPECT_NE(node, nullptr);
    EXPECT_NE(node->GetType(), "TransData");
    count_node++;
  }
  EXPECT_EQ(count_node,2);
}

TEST_F(STEST_FE_TRANSOP_INSERT, InsertCastAfterPlaceHolder) {
  // src:cce op, dst:cce op
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");

  OpDescPtr src_op = std::make_shared<OpDesc>("PlaceHolder", "PlaceHolder");
  GeTensorDesc src_tensor_desc(GeShape({1, 1024, 256, 512}), ge::FORMAT_NHWC, ge::DT_FLOAT16);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);
  ge::AttrUtils::SetInt(src_op, FE_IMPLY_TYPE, 6);

  OpDescPtr dst_op = std::make_shared<OpDesc>("D", "D");
  GeTensorDesc dst_tensor_desc(GeShape({1, 3, 4, 2}), ge::FORMAT_NHWC, ge::DT_FLOAT);
  dst_op->AddInputDesc(dst_tensor_desc);
  dst_op->AddOutputDesc(dst_tensor_desc);
  auto dst_node = graph->AddNode(dst_op);
  ge::AttrUtils::SetInt(dst_op, FE_IMPLY_TYPE, 4);
  vector<bool> input_const_vector = {false};
  dst_op->SetIsInputConst(input_const_vector);

  GraphUtils::AddEdge(src_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(0));

  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status status = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));
  uint32_t count_node = 0;
  uint32_t count_cast_node = 0;
  for (auto node: graph->GetDirectNode()) {
    EXPECT_NE(node, nullptr);
    if (node->GetType() == "Cast") {
      count_cast_node++;
    }
    count_node++;
  }
  EXPECT_EQ(count_node,3);
  EXPECT_EQ(count_cast_node,1);
}

TEST_F(STEST_FE_TRANSOP_INSERT, AddReduceReshapeOp_01) {
  string reshape_type = "NC";
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  GeTensorDesc src_tensor_desc(GeShape({18, 1, 16, 16}), ge::FORMAT_FRACTAL_Z, ge::DT_FLOAT16);
  GeTensorDesc dst_tensor_desc(GeShape({16, 32, 3, 3}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
  OpDescPtr src_op = std::make_shared<OpDesc>("reduce1", "ReduceOp");
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);

  OpDescPtr dst_op = std::make_shared<OpDesc>("B", "B");
  dst_op->AddInputDesc(dst_tensor_desc);
  dst_op->AddOutputDesc(dst_tensor_desc);
  auto dst_node = graph->AddNode(dst_op);

  GraphUtils::AddEdge(src_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(0));

  TransInfoPtr trans_info_ptr = std::make_shared<TransInfo>();
  trans_info_ptr->src_out_shape = GeShape({2, 3, 3, 1, 16, 16});
  trans_info_ptr->dst_in_shape = GeShape({16, 32, 3, 3});
  trans_info_ptr->src_reshape_type = reshape_type;
  trans_info_ptr->dst_reshape_type = reshape_type;
  trans_info_ptr->src_out_data_type = ge::DT_FLOAT16;
  trans_info_ptr->dst_in_data_type = ge::DT_FLOAT16;
  trans_info_ptr->src_op_pattern = OP_PATTERN_REDUCE;
  trans_info_ptr->src_out_primary_format = ge::FORMAT_FRACTAL_Z;
  trans_info_ptr->dst_in_primary_format = ge::FORMAT_NCHW;

  trans_info_ptr->src_op_desc = src_op;
  trans_info_ptr->dst_op_desc = dst_op;
  trans_info_ptr->src_node_ptr = src_node;
  trans_info_ptr->dst_node_ptr = dst_node;
  trans_info_ptr->src_anchor = src_node->GetOutDataAnchor(0);
  trans_info_ptr->dst_anchor = dst_node->GetInDataAnchor(0);

  TransNodeReshapeGenerator trans_op_insert(fe_ops_kernel_info_store_ptr_, trans_info_ptr);
  Status ret = trans_op_insert.AddTransNode(*graph.get(), trans_info_ptr);
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(STEST_FE_TRANSOP_INSERT, AddReduceReshapeOp_02) {
  string reshape_type = "NC";
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  GeTensorDesc src_tensor_desc(GeShape({16, 32, 3, 3}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
  GeTensorDesc dst_tensor_desc(GeShape({2, 3, 3, 1, 16, 16}), ge::FORMAT_FRACTAL_Z, ge::DT_FLOAT16);
  OpDescPtr src_op = std::make_shared<OpDesc>("B", "B");
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);

  OpDescPtr dst_op = std::make_shared<OpDesc>("reduce2", "ReduceOp");
  dst_op->AddInputDesc(dst_tensor_desc);
  dst_op->AddOutputDesc(dst_tensor_desc);
  auto dst_node = graph->AddNode(dst_op);

  GraphUtils::AddEdge(src_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(0));

  TransInfoPtr trans_info_ptr = std::make_shared<TransInfo>();
  trans_info_ptr->src_out_shape = GeShape({16, 32, 3, 3});
  trans_info_ptr->dst_in_shape = GeShape({2, 3, 3, 1, 16, 16});
  trans_info_ptr->src_reshape_type = reshape_type;
  trans_info_ptr->dst_reshape_type = reshape_type;
  trans_info_ptr->src_out_data_type = ge::DT_FLOAT16;
  trans_info_ptr->dst_in_data_type = ge::DT_FLOAT16;
  trans_info_ptr->dst_op_pattern = OP_PATTERN_REDUCE;
  trans_info_ptr->dst_in_primary_format = ge::FORMAT_FRACTAL_Z;
  trans_info_ptr->src_out_primary_format = ge::FORMAT_NCHW;

  trans_info_ptr->src_op_desc = src_op;
  trans_info_ptr->dst_op_desc = dst_op;
  trans_info_ptr->src_node_ptr = src_node;
  trans_info_ptr->dst_node_ptr = dst_node;
  trans_info_ptr->src_anchor = src_node->GetOutDataAnchor(0);
  trans_info_ptr->dst_anchor = dst_node->GetInDataAnchor(0);

  TransNodeReshapeGenerator trans_op_insert(fe_ops_kernel_info_store_ptr_, trans_info_ptr);
  Status ret = trans_op_insert.AddTransNode(*graph.get(), trans_info_ptr);
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(STEST_FE_TRANSOP_INSERT, InsertTransDataNodeBeforeEndOfNetOutput_01)
{
  // src:cce op, dst:cce op
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  vector<int64_t> nchw_dims = {1, 256, 256, 512};
  vector<int64_t> nc1hwc0_dims = {100, 256, 256, 512, 16};
  GeTensorDesc src_tensor_desc(GeShape({100, 256, 256, 512, 16}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  src_tensor_desc.SetOriginShape(GeShape(nchw_dims));
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);
  ge::AttrUtils::SetInt(src_op, FE_IMPLY_TYPE, 6);

  OpDescPtr dst_op = std::make_shared<OpDesc>("B", OP_TYPE_END);

  GeTensorDesc dst_tensor_desc(GeShape(nchw_dims), ge::FORMAT_NCHW, ge::DT_FLOAT);
  ge::AttrUtils::SetStr(dst_op, PARENT_OP_TYPE, fe::NETOUTPUT);
  ge::AttrUtils::SetInt(dst_op, FE_IMPLY_TYPE, 6);
  dst_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  dst_tensor_desc.SetOriginShape(GeShape(nchw_dims));
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
        EXPECT_EQ(shape.GetDimNum(), 5);
        EXPECT_EQ(shape.GetDims(), nc1hwc0_dims);
        ge::DataType data_type = node->GetOpDesc()->GetInputDescPtr(
            0)->GetDataType();
        ge::Format format = node->GetOpDesc()->GetInputDescPtr(0)->GetFormat();
        EXPECT_EQ(data_type, ge::DT_FLOAT16);
        EXPECT_EQ(format, ge::FORMAT_NC1HWC0);
      }
      {
        ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        EXPECT_EQ(shape.GetDimNum(), 4);
        EXPECT_EQ(shape.GetDims(), nchw_dims);
        ge::DataType data_type = node->GetOpDesc()->GetOutputDescPtr(
            0)->GetDataType();
        ge::Format format = node->GetOpDesc()->GetOutputDescPtr(0)->GetFormat();
        EXPECT_EQ(data_type, ge::DT_FLOAT16);
        EXPECT_EQ(format, ge::FORMAT_NCHW);
      }
    }
    if (node->GetType() == "Cast") {
      {
        ge::GeShape shape = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        EXPECT_EQ(shape.GetDimNum(), 4);
        EXPECT_EQ(shape.GetDims(), nchw_dims);
        ge::DataType data_type = node->GetOpDesc()->GetInputDescPtr(0)->GetDataType();
        ge::Format format = node->GetOpDesc()->GetInputDescPtr(0)->GetFormat();
        EXPECT_EQ(data_type, ge::DT_FLOAT16);
        EXPECT_EQ(format, ge::FORMAT_NCHW);
      }
      {
        ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        EXPECT_EQ(shape.GetDimNum(), 4);
        EXPECT_EQ(shape.GetDims(), nchw_dims);
        ge::DataType data_type = node->GetOpDesc()->GetOutputDescPtr(0)->GetDataType();
        ge::Format format = node->GetOpDesc()->GetOutputDescPtr(0)->GetFormat();
        EXPECT_EQ(data_type, ge::DT_FLOAT);
        EXPECT_EQ(format, ge::FORMAT_NCHW);
      }
    }
  }
  EXPECT_EQ(count_node, 4);
}


TEST_F(STEST_FE_TRANSOP_INSERT, InsertTransDataNodeBeforeEndOfNetOutput_02)
{
  // src:cce op, dst:cce op
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr src_op = std::make_shared<OpDesc>("A", OP_TYPE_PLACE_HOLDER);
  vector<int64_t> nchw_dims = {1, 256, 256, 512};
  vector<int64_t> nc1hwc0_dims = {100, 256, 256, 512, 16};
  GeTensorDesc src_tensor_desc(GeShape({100, 256, 256, 512, 16}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT);
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  src_tensor_desc.SetOriginShape(GeShape(nchw_dims));
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);
  ge::AttrUtils::SetInt(src_op, FE_IMPLY_TYPE, 6);

  OpDescPtr dst_op = std::make_shared<OpDesc>("B", "B");

  GeTensorDesc dst_tensor_desc(GeShape(nchw_dims), ge::FORMAT_NCHW, ge::DT_FLOAT16);
  ge::AttrUtils::SetStr(dst_op, PARENT_OP_TYPE, fe::NETOUTPUT);
  ge::AttrUtils::SetInt(dst_op, FE_IMPLY_TYPE, 6);
  dst_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  dst_tensor_desc.SetOriginShape(GeShape(nchw_dims));
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

    if (node->GetType() == "Cast") {
      {
        ge::GeShape shape = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        EXPECT_EQ(shape.GetDimNum(), 5);
        EXPECT_EQ(shape.GetDims(), nc1hwc0_dims);
        ge::DataType data_type = node->GetOpDesc()->GetInputDescPtr(0)->GetDataType();
        ge::Format format = node->GetOpDesc()->GetInputDescPtr(0)->GetFormat();
        EXPECT_EQ(data_type, ge::DT_FLOAT);
        EXPECT_EQ(format, ge::FORMAT_NC1HWC0);
      }
      {
        ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        EXPECT_EQ(shape.GetDimNum(), 5);
        EXPECT_EQ(shape.GetDims(), nc1hwc0_dims);
        ge::DataType data_type = node->GetOpDesc()->GetOutputDescPtr(0)->GetDataType();
        ge::Format format = node->GetOpDesc()->GetOutputDescPtr(0)->GetFormat();
        EXPECT_EQ(data_type, ge::DT_FLOAT16);
        EXPECT_EQ(format, ge::FORMAT_NC1HWC0);
      }
    }
    if (node->GetType() == "TransData") {
      {
        ge::GeShape shape = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        EXPECT_EQ(shape.GetDimNum(), 5);
        EXPECT_EQ(shape.GetDims(), nc1hwc0_dims);
        ge::DataType data_type = node->GetOpDesc()->GetInputDescPtr(
            0)->GetDataType();
        ge::Format format = node->GetOpDesc()->GetInputDescPtr(0)->GetFormat();
        EXPECT_EQ(data_type, ge::DT_FLOAT16);
        EXPECT_EQ(format, ge::FORMAT_NC1HWC0);
      }
      {
        ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        EXPECT_EQ(shape.GetDimNum(), 4);
        EXPECT_EQ(shape.GetDims(), nchw_dims);
        ge::DataType data_type = node->GetOpDesc()->GetOutputDescPtr(
            0)->GetDataType();
        ge::Format format = node->GetOpDesc()->GetOutputDescPtr(0)->GetFormat();
        EXPECT_EQ(data_type, ge::DT_FLOAT16);
        EXPECT_EQ(format, ge::FORMAT_NCHW);
      }
    }
  }
  EXPECT_EQ(count_node, 4);
}

TEST_F(STEST_FE_TRANSOP_INSERT, InsertTransDataNodeBeforeEndOfNetOutput_03) {
  // src:cce op, dst:cce op
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");

  OpDescPtr src_op = std::make_shared<OpDesc>("D", "D");
  GeTensorDesc src_tensor_desc(GeShape({512, 1, 5, 5}), ge::FORMAT_NCHW, ge::DT_FLOAT);
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  src_tensor_desc.SetOriginShape(GeShape({512, 1, 5, 5}));

  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);
  ge::AttrUtils::SetInt(src_op, FE_IMPLY_TYPE, 6);

  OpDescPtr dst_op = std::make_shared<OpDesc>("B", "B");
  GeTensorDesc dst_tensor_desc(GeShape({32, 5, 5, 16}), ge::FORMAT_C1HWC0, ge::DT_FLOAT);
  dst_tensor_desc.SetOriginFormat(ge::FORMAT_C1HWC0);
  dst_tensor_desc.SetOriginShape(GeShape({32, 5, 5, 16}));

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

  ASSERT_EQ(fe::SUCCESS, status);
  int count_node = 0;
  for (auto node : graph->GetDirectNode()) {
    ASSERT_NE(node, nullptr);
    count_node++;
    if (node->GetType() == "TransData") {
      {
        ge::GeShape shape_check = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        auto size = shape_check.GetDimNum();
        ASSERT_EQ(size, 4);
        vector<int64_t> input_vec_of_b = shape_check.GetDims();
        EXPECT_EQ(input_vec_of_b[0], 512);
        EXPECT_EQ(input_vec_of_b[1], 1);
        EXPECT_EQ(input_vec_of_b[2], 5);
        EXPECT_EQ(input_vec_of_b[3], 5);
      }
      {
        ge::GeShape shape_check = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        auto size = shape_check.GetDimNum();
        ASSERT_EQ(size, 4);
        vector<int64_t> input_vec_of_b = shape_check.GetDims();
        EXPECT_EQ(input_vec_of_b[0], 32);
        EXPECT_EQ(input_vec_of_b[1], 5);
        EXPECT_EQ(input_vec_of_b[2], 5);
        EXPECT_EQ(input_vec_of_b[3], 16);
      }
    }
  }
  EXPECT_EQ(count_node, 2);
}

TEST_F(STEST_FE_TRANSOP_INSERT, test_dynamic_shape_not_same)
{
  // src:cce op, dst:cce op
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");

  OpDescPtr src_op = std::make_shared<OpDesc>("D", "D");
  GeTensorDesc src_tensor_desc(GeShape({1, 1, -1}), ge::FORMAT_ND, ge::DT_FLOAT);
  src_tensor_desc.SetOriginFormat(ge::FORMAT_ND);
  src_tensor_desc.SetOriginShape(GeShape({1, 1, -1}));

  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);
  ge::AttrUtils::SetInt(src_op, FE_IMPLY_TYPE, 6);

  OpDescPtr dst_op = std::make_shared<OpDesc>("B", "B");
  GeTensorDesc dst_tensor_desc(GeShape({1, 2, 1, 16, 16}), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
  dst_tensor_desc.SetOriginFormat(ge::FORMAT_ND);
  dst_tensor_desc.SetOriginShape(GeShape({1, 1, 32}));

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

  ASSERT_EQ(fe::SUCCESS, status);
  int count_node = 0;
  for (auto node : graph->GetDirectNode()) {
    ASSERT_NE(node, nullptr);
    count_node++;
    if (node->GetType() == "Cast") {
      {
        ge::GeShape shape_check = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        auto size = shape_check.GetDimNum();
        ASSERT_EQ(size, 3);
        vector<int64_t> input_vec_of_b = shape_check.GetDims();
        EXPECT_EQ(input_vec_of_b[0], 1);
        EXPECT_EQ(input_vec_of_b[1], 1);
        EXPECT_EQ(input_vec_of_b[2], -1);

        auto &original_shape_check = node->GetOpDesc()->GetInputDescPtr(0)->GetOriginShape();
        size = original_shape_check.GetDimNum();
        ASSERT_EQ(size, 3);
        auto ori = original_shape_check.GetDims();
        EXPECT_EQ(ori[0], 1);
        EXPECT_EQ(ori[1], 1);
        EXPECT_EQ(ori[2], -1);
      }
      {
        ge::GeShape shape_check = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        auto size = shape_check.GetDimNum();
        ASSERT_EQ(size, 3);
        vector<int64_t> input_vec_of_b = shape_check.GetDims();
        EXPECT_EQ(input_vec_of_b[0], 1);
        EXPECT_EQ(input_vec_of_b[1], 1);
        EXPECT_EQ(input_vec_of_b[2], -1);

        auto &ori_shape_check = node->GetOpDesc()->GetOutputDescPtr(0)->GetOriginShape();
        size = ori_shape_check.GetDimNum();
        ASSERT_EQ(size, 3);
        auto ori = ori_shape_check.GetDims();
        EXPECT_EQ(ori[0], 1);
        EXPECT_EQ(ori[1], 1);
        EXPECT_EQ(ori[2], -1);
      }
    }
  }
  EXPECT_EQ(count_node, 4);
}

TEST_F(STEST_FE_TRANSOP_INSERT, test_dynamic_shape_not_same_bn_case)
{
  // src:cce op, dst:cce op
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");

  OpDescPtr src_op = std::make_shared<OpDesc>("D", "D");
  GeTensorDesc src_tensor_desc(GeShape({1, -1, 1, 1}), ge::FORMAT_ND, ge::DT_FLOAT);
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  src_tensor_desc.SetOriginShape(GeShape({-1}));

  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);
  ge::AttrUtils::SetInt(src_op, FE_IMPLY_TYPE, 6);

  OpDescPtr dst_op = std::make_shared<OpDesc>("B", "B");
  GeTensorDesc dst_tensor_desc(GeShape({1, -1, 1, 1, 16}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT);
  dst_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  dst_tensor_desc.SetOriginShape(GeShape({-1}));

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

  ASSERT_EQ(fe::SUCCESS, status);
  int count_node = 0;
  for (auto node : graph->GetDirectNode()) {
    ASSERT_NE(node, nullptr);
    count_node++;
    if (node->GetType() == "UnsqueezeV2") {
      {
        ge::GeShape shape_check = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        auto size = shape_check.GetDimNum();
        ASSERT_EQ(size, 1);
        vector<int64_t> input_vec_of_b = shape_check.GetDims();
        EXPECT_EQ(input_vec_of_b[0], -1);

        auto &original_shape_check = node->GetOpDesc()->GetInputDescPtr(0)->GetOriginShape();
        size = original_shape_check.GetDimNum();
        ASSERT_EQ(size, 1);
        auto ori = original_shape_check.GetDims();
        EXPECT_EQ(ori[0], -1);
      }
      {
        ge::GeShape shape_check = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        auto size = shape_check.GetDimNum();
        ASSERT_EQ(size, 4);
        vector<int64_t> input_vec_of_b = shape_check.GetDims();
        EXPECT_EQ(input_vec_of_b[0], 1);
        EXPECT_EQ(input_vec_of_b[1], -1);
        EXPECT_EQ(input_vec_of_b[2], 1);
        EXPECT_EQ(input_vec_of_b[3], 1);

        auto &ori_shape_check = node->GetOpDesc()->GetOutputDescPtr(0)->GetOriginShape();
        size = ori_shape_check.GetDimNum();
        ASSERT_EQ(size, 1);
        auto ori = ori_shape_check.GetDims();
        EXPECT_EQ(ori[0], -1);
      }
    }
    if (node->GetType() == "TransData") {
      {
        ge::GeShape shape_check = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        auto size = shape_check.GetDimNum();
        ASSERT_EQ(size, 4);
        vector<int64_t> input_vec_of_b = shape_check.GetDims();
        EXPECT_EQ(input_vec_of_b[0], 1);
        EXPECT_EQ(input_vec_of_b[1], -1);
        EXPECT_EQ(input_vec_of_b[2], 1);
        EXPECT_EQ(input_vec_of_b[3], 1);

        auto &original_shape_check = node->GetOpDesc()->GetInputDescPtr(0)->GetOriginShape();
        size = original_shape_check.GetDimNum();
        ASSERT_EQ(size, 1);
        auto ori = original_shape_check.GetDims();
        EXPECT_EQ(ori[0], -1);
      }
      {
        ge::GeShape shape_check = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        auto size = shape_check.GetDimNum();
        ASSERT_EQ(size, 5);
        vector<int64_t> input_vec_of_b = shape_check.GetDims();
        EXPECT_EQ(input_vec_of_b[0], 1);
        EXPECT_EQ(input_vec_of_b[1], -1);
        EXPECT_EQ(input_vec_of_b[2], 1);
        EXPECT_EQ(input_vec_of_b[3], 1);
        EXPECT_EQ(input_vec_of_b[4], 16);

        auto &ori_shape_check = node->GetOpDesc()->GetOutputDescPtr(0)->GetOriginShape();
        size = ori_shape_check.GetDimNum();
        ASSERT_EQ(size, 1);
        auto ori = ori_shape_check.GetDims();
        EXPECT_EQ(ori[0], -1);
      }
    }
  }
  EXPECT_EQ(count_node, 4);
}


TEST_F(STEST_FE_TRANSOP_INSERT, test_dynamic_shape_not_same_bn_case_reverse)
{
  // src:cce op, dst:cce op
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");

  OpDescPtr src_op = std::make_shared<OpDesc>("D", "D");
  GeTensorDesc src_tensor_desc(GeShape({1, -1, 1, 1, 16}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT);
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  src_tensor_desc.SetOriginShape(GeShape({-1}));



  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);
  ge::AttrUtils::SetInt(src_op, FE_IMPLY_TYPE, 6);

  OpDescPtr dst_op = std::make_shared<OpDesc>("B", "B");
  GeTensorDesc dst_tensor_desc(GeShape({1, -1, 1, 1}), ge::FORMAT_ND, ge::DT_FLOAT);
  dst_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  dst_tensor_desc.SetOriginShape(GeShape({-1}));

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

  ASSERT_EQ(fe::SUCCESS, status);
  int count_node = 0;
  for (auto node : graph->GetDirectNode()) {
    ASSERT_NE(node, nullptr);
    count_node++;
    if (node->GetType() == "SqueezeV2") {
      {
        ge::GeShape shape_check = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        auto size = shape_check.GetDimNum();
        ASSERT_EQ(size, 4);
        vector<int64_t> input_vec_of_b = shape_check.GetDims();
        EXPECT_EQ(input_vec_of_b[0], 1);
        EXPECT_EQ(input_vec_of_b[1], -1);
        EXPECT_EQ(input_vec_of_b[2], 1);
        EXPECT_EQ(input_vec_of_b[3], 1);

        auto &original_shape_check = node->GetOpDesc()->GetInputDescPtr(0)->GetOriginShape();
        size = original_shape_check.GetDimNum();
        ASSERT_EQ(size, 1);
        auto ori = original_shape_check.GetDims();
        EXPECT_EQ(ori[0], -1);
      }
      {
        ge::GeShape shape_check = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        auto size = shape_check.GetDimNum();
        ASSERT_EQ(size, 1);
        vector<int64_t> input_vec_of_b = shape_check.GetDims();
        EXPECT_EQ(input_vec_of_b[0], -1);

        auto &ori_shape_check = node->GetOpDesc()->GetOutputDescPtr(0)->GetOriginShape();
        size = ori_shape_check.GetDimNum();
        ASSERT_EQ(size, 1);
        auto ori = ori_shape_check.GetDims();
        EXPECT_EQ(ori[0], -1);
      }
    }
    if (node->GetType() == "TransData") {
      {
        ge::GeShape shape_check = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        auto size = shape_check.GetDimNum();
        ASSERT_EQ(size, 5);
        vector<int64_t> input_vec_of_b = shape_check.GetDims();
        EXPECT_EQ(input_vec_of_b[0], 1);
        EXPECT_EQ(input_vec_of_b[1], -1);
        EXPECT_EQ(input_vec_of_b[2], 1);
        EXPECT_EQ(input_vec_of_b[3], 1);
        EXPECT_EQ(input_vec_of_b[4], 16);

        auto &original_shape_check = node->GetOpDesc()->GetInputDescPtr(0)->GetOriginShape();
        size = original_shape_check.GetDimNum();
        ASSERT_EQ(size, 1);
        auto ori = original_shape_check.GetDims();
        EXPECT_EQ(ori[0], -1);
      }
      {
        ge::GeShape shape_check = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        auto size = shape_check.GetDimNum();
        ASSERT_EQ(size, 4);
        vector<int64_t> input_vec_of_b = shape_check.GetDims();
        EXPECT_EQ(input_vec_of_b[0], 1);
        EXPECT_EQ(input_vec_of_b[1], -1);
        EXPECT_EQ(input_vec_of_b[2], 1);
        EXPECT_EQ(input_vec_of_b[3], 1);

        auto &ori_shape_check = node->GetOpDesc()->GetOutputDescPtr(0)->GetOriginShape();
        size = ori_shape_check.GetDimNum();
        ASSERT_EQ(size, 1);
        auto ori = ori_shape_check.GetDims();
        EXPECT_EQ(ori[0], -1);
      }
    }
  }
  EXPECT_EQ(count_node, 4);
}

TEST_F(STEST_FE_TRANSOP_INSERT, test_dynamic_shape_not_same_reverse)
{
  // src:cce op, dst:cce op
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");

  OpDescPtr src_op = std::make_shared<OpDesc>("D", "D");
  GeTensorDesc src_tensor_desc(GeShape({1, 2, 1, 16, 16}), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT);
  src_tensor_desc.SetOriginFormat(ge::FORMAT_ND);
  src_tensor_desc.SetOriginShape(GeShape({1, 1, 32}));

  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);
  ge::AttrUtils::SetInt(src_op, FE_IMPLY_TYPE, 6);

  OpDescPtr dst_op = std::make_shared<OpDesc>("B", "B");
  GeTensorDesc dst_tensor_desc(GeShape({1, 1, -1}), ge::FORMAT_ND, ge::DT_FLOAT16);
  dst_tensor_desc.SetOriginFormat(ge::FORMAT_ND);
  dst_tensor_desc.SetOriginShape(GeShape({1, 1, -1}));

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

  ASSERT_EQ(fe::SUCCESS, status);
  int count_node = 0;
  for (auto node : graph->GetDirectNode()) {
    ASSERT_NE(node, nullptr);
    count_node++;
    if (node->GetType() == "Cast") {
      {
        ge::GeShape shape_check = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        auto size = shape_check.GetDimNum();
        ASSERT_EQ(size, 5);
        vector<int64_t> input_vec_of_b = shape_check.GetDims();
        EXPECT_EQ(input_vec_of_b[0], 1);
        EXPECT_EQ(input_vec_of_b[1], 2);
        EXPECT_EQ(input_vec_of_b[2], 1);
        EXPECT_EQ(input_vec_of_b[3], 16);
        EXPECT_EQ(input_vec_of_b[4], 16);

        auto &original_shape_check = node->GetOpDesc()->GetInputDescPtr(0)->GetOriginShape();
        size = original_shape_check.GetDimNum();
        ASSERT_EQ(size, 3);
        auto ori = original_shape_check.GetDims();
        EXPECT_EQ(ori[0], 1);
        EXPECT_EQ(ori[1], 1);
        EXPECT_EQ(ori[2], 32);
      }
      {
        ge::GeShape shape_check = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        auto size = shape_check.GetDimNum();
        ASSERT_EQ(size, 5);
        vector<int64_t> input_vec_of_b = shape_check.GetDims();
        EXPECT_EQ(input_vec_of_b[0], 1);
        EXPECT_EQ(input_vec_of_b[1], 2);
        EXPECT_EQ(input_vec_of_b[2], 1);
        EXPECT_EQ(input_vec_of_b[3], 16);
        EXPECT_EQ(input_vec_of_b[4], 16);

        auto &ori_shape_check = node->GetOpDesc()->GetOutputDescPtr(0)->GetOriginShape();
        size = ori_shape_check.GetDimNum();
        ASSERT_EQ(size, 3);
        auto ori = ori_shape_check.GetDims();
        EXPECT_EQ(ori[0], 1);
        EXPECT_EQ(ori[1], 1);
        EXPECT_EQ(ori[2], 32);
      }
    }
  }
  EXPECT_EQ(count_node, 4);
}

TEST_F(STEST_FE_TRANSOP_INSERT, insert_transnode_between_cube_and_vec)
{
  // src:cce op, dst:cce op
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  ComputeGraphPtr graph_check =  std::make_shared<ComputeGraph>("test_graph_check");
  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  ge::Format format = static_cast<ge::Format>(ge::GetFormatFromC0(ge::FORMAT_NC1HWC0, GetC0BitValFromC0(8)));
  GeTensorDesc src_tensor_desc(GeShape({3, 2, 32, 32, 8}), format, ge::DT_FLOAT);
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  src_tensor_desc.SetOriginShape(GeShape({3, 16, 32, 32}));
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);
  ge::AttrUtils::SetInt(src_op, FE_IMPLY_TYPE, 6);

  OpDescPtr dst_op = std::make_shared<OpDesc>("B", "B");
  GeTensorDesc dst_tensor_desc(GeShape({3, 16, 32, 32}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
  dst_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  dst_tensor_desc.SetOriginShape(GeShape({3, 16, 32, 32}));
  dst_op->AddInputDesc(dst_tensor_desc);
  dst_op->AddOutputDesc(dst_tensor_desc);
  auto dst_node = graph->AddNode(dst_op);
  ge::AttrUtils::SetInt(dst_op, FE_IMPLY_TYPE, 6);

  GraphUtils::AddEdge(src_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(0));

  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status status = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));
  ASSERT_EQ(fe::SUCCESS, status);
  int count_node = 0;
  for (auto node : graph->GetDirectNode()) {
    ASSERT_NE(node, nullptr);
    count_node++;
    if (node->GetType() == "TransData") {
      {
        ge::GeShape shape_check = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        ge::DataType dtype_check = node->GetOpDesc()->GetInputDescPtr(0)->GetDataType();
        auto size = shape_check.GetDimNum();
        ASSERT_EQ(size, 5);
        ASSERT_EQ(dtype_check, ge::DT_FLOAT);
        vector<int64_t> input_vec_of_b =shape_check.GetDims();
        EXPECT_EQ(input_vec_of_b[0], 3);
        EXPECT_EQ(input_vec_of_b[1], 2);
        EXPECT_EQ(input_vec_of_b[2], 32);
        EXPECT_EQ(input_vec_of_b[3], 32);
        EXPECT_EQ(input_vec_of_b[4], 8);
      }
      {
        ge::GeShape shape_check = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        ge::DataType dtype_check = node->GetOpDesc()->GetOutputDescPtr(0)->GetDataType();
        auto size = shape_check.GetDimNum();
        ASSERT_EQ(size, 4);
        ASSERT_EQ(dtype_check, ge::DT_FLOAT);
        vector<int64_t> input_vec_of_b =shape_check.GetDims();
        EXPECT_EQ(input_vec_of_b[0], 3);
        EXPECT_EQ(input_vec_of_b[1], 16);
        EXPECT_EQ(input_vec_of_b[2], 32);
        EXPECT_EQ(input_vec_of_b[3], 32);
      }
    }
    if (node->GetType() == "Cast") {
      {
        ge::GeShape shape_check = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        ge::DataType dtype_check = node->GetOpDesc()->GetInputDescPtr(0)->GetDataType();
        auto size = shape_check.GetDimNum();
        ASSERT_EQ(size, 4);
        ASSERT_EQ(dtype_check, ge::DT_FLOAT);
        vector<int64_t> input_vec_of_b =shape_check.GetDims();
        EXPECT_EQ(input_vec_of_b[0], 3);
        EXPECT_EQ(input_vec_of_b[1], 16);
        EXPECT_EQ(input_vec_of_b[2], 32);
        EXPECT_EQ(input_vec_of_b[3], 32);
      }
      {
        ge::GeShape shape_check = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        ge::DataType dtype_check = node->GetOpDesc()->GetOutputDescPtr(0)->GetDataType();
        auto size = shape_check.GetDimNum();
        ASSERT_EQ(size, 4);
        ASSERT_EQ(dtype_check, ge::DT_FLOAT16);
        vector<int64_t> input_vec_of_b =shape_check.GetDims();
        EXPECT_EQ(input_vec_of_b[0], 3);
        EXPECT_EQ(input_vec_of_b[1], 16);
        EXPECT_EQ(input_vec_of_b[2], 32);
        EXPECT_EQ(input_vec_of_b[3], 32);
      }
    }
  }
  EXPECT_EQ(count_node, 4);
}

TEST_F(STEST_FE_TRANSOP_INSERT, insert_transnode_between_cube_and_vec_1)
{
  // src:cce op, dst:cce op
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  ComputeGraphPtr graph_check =  std::make_shared<ComputeGraph>("test_graph_check");
  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  ge::Format format = static_cast<ge::Format>(ge::GetFormatFromC0(ge::FORMAT_NC1HWC0, GetC0BitValFromC0(8)));
  GeTensorDesc src_tensor_desc(GeShape({3, 2, 32, 32, 8}), format, ge::DT_FLOAT);
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  src_tensor_desc.SetOriginShape(GeShape({3, 16, 32, 32}));
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);
  ge::AttrUtils::SetInt(src_op, FE_IMPLY_TYPE, 6);

  OpDescPtr dst_op = std::make_shared<OpDesc>("B", "B");
  GeTensorDesc dst_tensor_desc(GeShape({3, 16, 32, 32}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
  dst_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  dst_tensor_desc.SetOriginShape(GeShape({3, 16, 32, 32}));
  dst_op->AddInputDesc(dst_tensor_desc);
  dst_op->AddOutputDesc(dst_tensor_desc);
  auto dst_node = graph->AddNode(dst_op);
  ge::AttrUtils::SetInt(dst_op, FE_IMPLY_TYPE, 6);

  GraphUtils::AddEdge(dst_node->GetOutDataAnchor(0), src_node->GetInDataAnchor(0));

  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status status = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));
  ASSERT_EQ(fe::SUCCESS, status);
  int count_node = 0;
  for (auto node : graph->GetDirectNode()) {
    ASSERT_NE(node, nullptr);
    count_node++;
    if (node->GetType() == "TransData") {
      {
        ge::GeShape shape_check = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        ge::DataType dtype_check = node->GetOpDesc()->GetInputDescPtr(0)->GetDataType();
        auto size = shape_check.GetDimNum();
        ASSERT_EQ(size, 4);
        ASSERT_EQ(dtype_check, ge::DT_FLOAT);
        vector<int64_t> input_vec_of_b =shape_check.GetDims();
        EXPECT_EQ(input_vec_of_b[0], 3);
        EXPECT_EQ(input_vec_of_b[1], 16);
        EXPECT_EQ(input_vec_of_b[2], 32);
        EXPECT_EQ(input_vec_of_b[3], 32);
      }
      {
        ge::GeShape shape_check = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        ge::DataType dtype_check = node->GetOpDesc()->GetOutputDescPtr(0)->GetDataType();
        auto size = shape_check.GetDimNum();
        ASSERT_EQ(size, 5);
        ASSERT_EQ(dtype_check, ge::DT_FLOAT);
        vector<int64_t> input_vec_of_b =shape_check.GetDims();
        EXPECT_EQ(input_vec_of_b[0], 3);
        EXPECT_EQ(input_vec_of_b[1], 2);
        EXPECT_EQ(input_vec_of_b[2], 32);
        EXPECT_EQ(input_vec_of_b[3], 32);
        EXPECT_EQ(input_vec_of_b[4], 8);
      }
    }
    if (node->GetType() == "Cast") {
      {
        ge::GeShape shape_check = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        ge::DataType dtype_check = node->GetOpDesc()->GetInputDescPtr(0)->GetDataType();
        auto size = shape_check.GetDimNum();
        ASSERT_EQ(size, 4);
        ASSERT_EQ(dtype_check, ge::DT_FLOAT16);
        vector<int64_t> input_vec_of_b =shape_check.GetDims();
        EXPECT_EQ(input_vec_of_b[0], 3);
        EXPECT_EQ(input_vec_of_b[1], 16);
        EXPECT_EQ(input_vec_of_b[2], 32);
        EXPECT_EQ(input_vec_of_b[3], 32);
      }
      {
        ge::GeShape shape_check = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        ge::DataType dtype_check = node->GetOpDesc()->GetOutputDescPtr(0)->GetDataType();
        auto size = shape_check.GetDimNum();
        ASSERT_EQ(size, 4);
        ASSERT_EQ(dtype_check, ge::DT_FLOAT);
        vector<int64_t> input_vec_of_b =shape_check.GetDims();
        EXPECT_EQ(input_vec_of_b[0], 3);
        EXPECT_EQ(input_vec_of_b[1], 16);
        EXPECT_EQ(input_vec_of_b[2], 32);
        EXPECT_EQ(input_vec_of_b[3], 32);
      }
    }
  }
  EXPECT_EQ(count_node, 4);
}

TEST_F(STEST_FE_TRANSOP_INSERT, merge_two_tansposed_case1) {
  vector<int64_t> dims1 = {4,8,1,8,8};
  vector<int64_t> dims2 = {4,8,8,1,8};
  ge::GeShape shape1(dims1);
  ge::GeShape shape2(dims2);
  ge::GeTensorDesc tensor_desc1(shape1, ge::FORMAT_ND, ge::DT_FLOAT);
  tensor_desc1.SetOriginFormat(ge::FORMAT_ND);
  tensor_desc1.SetOriginDataType(ge::DT_FLOAT);
  tensor_desc1.SetOriginShape(shape1);

  ge::GeTensorDesc tensor_desc2(shape2, ge::FORMAT_ND, ge::DT_FLOAT);
  tensor_desc2.SetOriginFormat(ge::FORMAT_ND);
  tensor_desc2.SetOriginDataType(ge::DT_FLOAT);
  tensor_desc2.SetOriginShape(shape2);

  ge::OpDescPtr data_op = std::make_shared<ge::OpDesc>("data", "Data");
  ge::OpDescPtr transpose1_op = std::make_shared<ge::OpDesc>("transpose1", TRANSPOSED);
  ge::OpDescPtr transpose2_op = std::make_shared<ge::OpDesc>("transpose2", TRANSPOSED);
  ge::OpDescPtr netout_op = std::make_shared<ge::OpDesc>("nteout", "NetOutput");

  data_op->AddOutputDesc(tensor_desc1);
  transpose1_op->AddInputDesc(tensor_desc1);
  transpose1_op->AddOutputDesc(tensor_desc2);
  transpose2_op->AddInputDesc(tensor_desc2);
  transpose2_op->AddOutputDesc(tensor_desc1);
  netout_op->AddInputDesc(tensor_desc1);

  vector<int64_t> perm_vec1 = {0,1,3,2,4};
  vector<int64_t> perm_vec2 = {0,2,3,1,4};
  ge::AttrUtils::SetListInt(transpose1_op, "perm", perm_vec1);
  ge::AttrUtils::SetListInt(transpose2_op, "perm", perm_vec2);

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  ge::NodePtr data_node = graph->AddNode(data_op);
  ge::NodePtr transpose1_node = graph->AddNode(transpose1_op);
  ge::NodePtr transpose2_node = graph->AddNode(transpose2_op);
  ge::NodePtr netout_node = graph->AddNode(netout_op);
  GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), transpose1_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(transpose1_node->GetOutDataAnchor(0), transpose2_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(transpose2_node->GetOutDataAnchor(0), netout_node->GetInDataAnchor(0));

  TransNodeMerging trans_op_merger;
  EXPECT_EQ(trans_op_merger.MergeAllTransOps(*(graph.get()), {}, false), fe::SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(), 4);
  size_t transpose_count = 0;
  for (const ge::NodePtr &node : graph->GetDirectNode()) {
    if (node->GetType() == TRANSPOSED) {
      transpose_count++;
    }
  }
  EXPECT_EQ(transpose_count, 2);
}

TEST_F(STEST_FE_TRANSOP_INSERT, merge_two_tansposed_case2) {
  vector<int64_t> dims1 = {4,8,1,8,8};
  vector<int64_t> dims2 = {4,8,8,1,8};
  ge::GeShape shape1(dims1);
  ge::GeShape shape2(dims2);
  ge::GeTensorDesc tensor_desc1(shape1, ge::FORMAT_ND, ge::DT_FLOAT);
  tensor_desc1.SetOriginFormat(ge::FORMAT_ND);
  tensor_desc1.SetOriginDataType(ge::DT_FLOAT);
  tensor_desc1.SetOriginShape(shape1);

  ge::GeTensorDesc tensor_desc2(shape2, ge::FORMAT_ND, ge::DT_FLOAT);
  tensor_desc2.SetOriginFormat(ge::FORMAT_ND);
  tensor_desc2.SetOriginDataType(ge::DT_FLOAT);
  tensor_desc2.SetOriginShape(shape2);

  ge::OpDescPtr data_op = std::make_shared<ge::OpDesc>("data", "Data");
  ge::OpDescPtr transpose1_op = std::make_shared<ge::OpDesc>("transpose1", TRANSPOSED);
  ge::OpDescPtr transpose2_op = std::make_shared<ge::OpDesc>("transpose2", TRANSPOSED);
  ge::OpDescPtr netout_op = std::make_shared<ge::OpDesc>("nteout", "NetOutput");

  data_op->AddOutputDesc(tensor_desc1);
  transpose1_op->AddInputDesc(tensor_desc1);
  transpose1_op->AddOutputDesc(tensor_desc2);
  transpose2_op->AddInputDesc(tensor_desc2);
  transpose2_op->AddOutputDesc(tensor_desc1);
  netout_op->AddInputDesc(tensor_desc1);

  vector<int64_t> perm_vec1 = {0,1,3,2,4};
  vector<int64_t> perm_vec2 = {0,1,3,2,4};
  ge::AttrUtils::SetListInt(transpose1_op, "perm", perm_vec1);
  ge::AttrUtils::SetListInt(transpose2_op, "perm", perm_vec2);

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  ge::NodePtr data_node = graph->AddNode(data_op);
  ge::NodePtr transpose1_node = graph->AddNode(transpose1_op);
  ge::NodePtr transpose2_node = graph->AddNode(transpose2_op);
  ge::NodePtr netout_node = graph->AddNode(netout_op);
  GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), transpose1_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(transpose1_node->GetOutDataAnchor(0), transpose2_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(transpose2_node->GetOutDataAnchor(0), netout_node->GetInDataAnchor(0));

  TransNodeMerging trans_op_merger;
  EXPECT_EQ(trans_op_merger.MergeAllTransOps(*(graph.get()), {}, false), fe::SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(), 2);
  size_t transpose_count = 0;
  for (const ge::NodePtr &node : graph->GetDirectNode()) {
    if (node->GetType() == TRANSPOSED) {
      transpose_count++;
    }
  }
  EXPECT_EQ(transpose_count, 0);
}

TEST_F(STEST_FE_TRANSOP_INSERT, merge_two_tansposed_case3) {
  vector<int64_t> dims1 = {4,8,1,8,8};
  vector<int64_t> dims2 = {4,8,8,1,8};
  ge::GeShape shape1(dims1);
  ge::GeShape shape2(dims2);
  ge::GeTensorDesc tensor_desc1(shape1, ge::FORMAT_ND, ge::DT_FLOAT);
  tensor_desc1.SetOriginFormat(ge::FORMAT_ND);
  tensor_desc1.SetOriginDataType(ge::DT_FLOAT);
  tensor_desc1.SetOriginShape(shape1);

  ge::GeTensorDesc tensor_desc2(shape2, ge::FORMAT_ND, ge::DT_FLOAT);
  tensor_desc2.SetOriginFormat(ge::FORMAT_ND);
  tensor_desc2.SetOriginDataType(ge::DT_FLOAT);
  tensor_desc2.SetOriginShape(shape2);

  ge::OpDescPtr data_op = std::make_shared<ge::OpDesc>("data", "Data");
  ge::OpDescPtr transpose1_op = std::make_shared<ge::OpDesc>("transpose1", TRANSPOSE);
  ge::OpDescPtr transpose2_op = std::make_shared<ge::OpDesc>("transpose2", TRANSPOSE);
  ge::OpDescPtr netout_op = std::make_shared<ge::OpDesc>("nteout", "NetOutput");

  data_op->AddOutputDesc(tensor_desc1);
  transpose1_op->AddInputDesc("x", tensor_desc1);
  transpose1_op->AddOutputDesc(tensor_desc2);
  transpose2_op->AddInputDesc("x", tensor_desc2);
  transpose2_op->AddOutputDesc(tensor_desc1);
  netout_op->AddInputDesc(tensor_desc1);

  ge::GeTensorDesc outTensor1(ge::GeShape({5}), ge::FORMAT_ND, ge::DT_INT32);
  transpose1_op->AddInputDesc("perm", outTensor1);
  ge::GeTensorPtr const_out_tenosr1 = nullptr;
  const_out_tenosr1 = std::make_shared<ge::GeTensor>(outTensor1);
  vector<int32_t> perm_vec1 = {0,1,3,2,4};
  const_out_tenosr1->SetData(reinterpret_cast<uint8_t *>(perm_vec1.data()), perm_vec1.size() * sizeof(int32_t));
  ge::OpDescPtr outOpDesc1 = ge::OpDescUtils::CreateConstOp(const_out_tenosr1);

  ge::GeTensorDesc outTensor2(ge::GeShape({5}), ge::FORMAT_ND, ge::DT_INT32);
  transpose2_op->AddInputDesc("perm", outTensor2);
  ge::GeTensorPtr const_out_tenosr2 = nullptr;
  const_out_tenosr2 = std::make_shared<ge::GeTensor>(outTensor2);
  vector<int32_t> perm_vec2 = {0,1,3,2,4};
  const_out_tenosr2->SetData(reinterpret_cast<uint8_t *>(perm_vec2.data()), perm_vec2.size() * sizeof(int32_t));
  ge::OpDescPtr outOpDesc2 = ge::OpDescUtils::CreateConstOp(const_out_tenosr2);

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  ge::NodePtr data_node = graph->AddNode(data_op);
  ge::NodePtr constNode1 = graph->AddNode(outOpDesc1);
  ge::NodePtr transpose1_node = graph->AddNode(transpose1_op);
  ge::NodePtr constNode2 = graph->AddNode(outOpDesc2);
  ge::NodePtr transpose2_node = graph->AddNode(transpose2_op);
  ge::NodePtr netout_node = graph->AddNode(netout_op);
  GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), transpose1_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(constNode1->GetOutDataAnchor(0), transpose1_node->GetInDataAnchor(1));
  GraphUtils::AddEdge(transpose1_node->GetOutDataAnchor(0), transpose2_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(constNode2->GetOutDataAnchor(0), transpose2_node->GetInDataAnchor(1));
  GraphUtils::AddEdge(transpose2_node->GetOutDataAnchor(0), netout_node->GetInDataAnchor(0));

  TransNodeMerging trans_op_merger;
  EXPECT_EQ(trans_op_merger.MergeAllTransOps(*(graph.get()), {}, false), fe::SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(), 2);
  size_t transpose_count = 0;
  for (const ge::NodePtr &node : graph->GetDirectNode()) {
    if (node->GetType() == TRANSPOSE || node->GetType() == TRANSPOSED) {
      transpose_count++;
    }
  }
  EXPECT_EQ(transpose_count, 0);
}

TEST_F(STEST_FE_TRANSOP_INSERT, merge_two_tansposed_case4) {
  vector<int64_t> dims1 = {4,8,1,8,8};
  vector<int64_t> dims2 = {4,8,8,1,8};
  ge::GeShape shape1(dims1);
  ge::GeShape shape2(dims2);
  ge::GeTensorDesc tensor_desc1(shape1, ge::FORMAT_ND, ge::DT_FLOAT);
  tensor_desc1.SetOriginFormat(ge::FORMAT_ND);
  tensor_desc1.SetOriginDataType(ge::DT_FLOAT);
  tensor_desc1.SetOriginShape(shape1);

  ge::GeTensorDesc tensor_desc2(shape2, ge::FORMAT_ND, ge::DT_FLOAT);
  tensor_desc2.SetOriginFormat(ge::FORMAT_ND);
  tensor_desc2.SetOriginDataType(ge::DT_FLOAT);
  tensor_desc2.SetOriginShape(shape2);

  ge::OpDescPtr data_op = std::make_shared<ge::OpDesc>("data", "Data");
  ge::OpDescPtr transpose1_op = std::make_shared<ge::OpDesc>("transpose1", TRANSPOSED);
  ge::OpDescPtr transpose2_op = std::make_shared<ge::OpDesc>("transpose2", TRANSPOSE);
  ge::OpDescPtr netout_op = std::make_shared<ge::OpDesc>("nteout", "NetOutput");

  data_op->AddOutputDesc(tensor_desc1);
  transpose1_op->AddInputDesc("x", tensor_desc1);
  transpose1_op->AddOutputDesc(tensor_desc2);
  transpose2_op->AddInputDesc("x", tensor_desc2);
  transpose2_op->AddOutputDesc(tensor_desc1);
  netout_op->AddInputDesc(tensor_desc1);

  vector<int64_t> perm_vec1 = {0,1,3,2,4};
  ge::AttrUtils::SetListInt(transpose1_op, "perm", perm_vec1);

  ge::GeTensorDesc outTensor2(ge::GeShape({5}), ge::FORMAT_ND, ge::DT_INT32);
  transpose2_op->AddInputDesc("perm", outTensor2);
  ge::GeTensorPtr const_out_tenosr2 = nullptr;
  const_out_tenosr2 = std::make_shared<ge::GeTensor>(outTensor2);
  vector<int32_t> perm_vec2 = {0,1,3,2,4};
  const_out_tenosr2->SetData(reinterpret_cast<uint8_t *>(perm_vec2.data()), perm_vec2.size() * sizeof(int32_t));
  ge::OpDescPtr outOpDesc2 = ge::OpDescUtils::CreateConstOp(const_out_tenosr2);

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  ge::NodePtr data_node = graph->AddNode(data_op);
  ge::NodePtr transpose1_node = graph->AddNode(transpose1_op);
  ge::NodePtr constNode = graph->AddNode(outOpDesc2);
  ge::NodePtr transpose2_node = graph->AddNode(transpose2_op);
  ge::NodePtr netout_node = graph->AddNode(netout_op);
  GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), transpose1_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(transpose1_node->GetOutDataAnchor(0), transpose2_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(constNode->GetOutDataAnchor(0), transpose2_node->GetInDataAnchor(1));
  GraphUtils::AddEdge(transpose2_node->GetOutDataAnchor(0), netout_node->GetInDataAnchor(0));

  TransNodeMerging trans_op_merger;
  EXPECT_EQ(trans_op_merger.MergeAllTransOps(*(graph.get()), {}, false), fe::SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(), 2);
  size_t transpose_count = 0;
  for (const ge::NodePtr &node : graph->GetDirectNode()) {
    if (node->GetType() == TRANSPOSE || node->GetType() == TRANSPOSED) {
      transpose_count++;
    }
  }
  EXPECT_EQ(transpose_count, 0);
}

TEST_F(STEST_FE_TRANSOP_INSERT, merge_two_tansposed_case5) {
  vector<int64_t> dims1 = {4,8,1,8,8};
  vector<int64_t> dims2 = {4,8,8,1,8};
  ge::GeShape shape1(dims1);
  ge::GeShape shape2(dims2);
  ge::GeTensorDesc tensor_desc1(shape1, ge::FORMAT_ND, ge::DT_FLOAT);
  tensor_desc1.SetOriginFormat(ge::FORMAT_ND);
  tensor_desc1.SetOriginDataType(ge::DT_FLOAT);
  tensor_desc1.SetOriginShape(shape1);

  ge::GeTensorDesc tensor_desc2(shape2, ge::FORMAT_ND, ge::DT_FLOAT);
  tensor_desc2.SetOriginFormat(ge::FORMAT_ND);
  tensor_desc2.SetOriginDataType(ge::DT_FLOAT);
  tensor_desc2.SetOriginShape(shape2);

  ge::OpDescPtr data_op = std::make_shared<ge::OpDesc>("data", "Data");
  ge::OpDescPtr transpose1_op = std::make_shared<ge::OpDesc>("transpose1", TRANSPOSE);
  ge::OpDescPtr transpose2_op = std::make_shared<ge::OpDesc>("transpose2", TRANSPOSE);
  ge::OpDescPtr netout_op = std::make_shared<ge::OpDesc>("nteout", "NetOutput");

  data_op->AddOutputDesc(tensor_desc1);
  transpose1_op->AddInputDesc("x", tensor_desc1);
  transpose1_op->AddOutputDesc(tensor_desc2);
  transpose2_op->AddInputDesc("x", tensor_desc2);
  transpose2_op->AddOutputDesc(tensor_desc1);
  netout_op->AddInputDesc(tensor_desc1);

  ge::GeTensorDesc outTensor1(ge::GeShape({5}), ge::FORMAT_ND, ge::DT_INT32);
  transpose1_op->AddInputDesc("perm", outTensor1);
  transpose2_op->AddInputDesc("perm", outTensor1);
  ge::GeTensorPtr const_out_tenosr1 = nullptr;
  const_out_tenosr1 = std::make_shared<ge::GeTensor>(outTensor1);
  vector<int32_t> perm_vec1 = {0,1,3,2,4};
  const_out_tenosr1->SetData(reinterpret_cast<uint8_t *>(perm_vec1.data()), perm_vec1.size() * sizeof(int32_t));
  ge::OpDescPtr outOpDesc1 = ge::OpDescUtils::CreateConstOp(const_out_tenosr1);

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  ge::NodePtr data_node = graph->AddNode(data_op);
  ge::NodePtr constNode1 = graph->AddNode(outOpDesc1);
  ge::NodePtr transpose1_node = graph->AddNode(transpose1_op);
  ge::NodePtr transpose2_node = graph->AddNode(transpose2_op);
  ge::NodePtr netout_node = graph->AddNode(netout_op);
  GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), transpose1_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(constNode1->GetOutDataAnchor(0), transpose1_node->GetInDataAnchor(1));
  GraphUtils::AddEdge(transpose1_node->GetOutDataAnchor(0), transpose2_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(constNode1->GetOutDataAnchor(0), transpose2_node->GetInDataAnchor(1));
  GraphUtils::AddEdge(transpose2_node->GetOutDataAnchor(0), netout_node->GetInDataAnchor(0));
  
  GraphUtils::RemoveJustNode(graph, constNode1);

  TransNodeMerging trans_op_merger;
  EXPECT_EQ(trans_op_merger.MergeAllTransOps(*(graph.get()), {}, false), fe::FAILED);
}

TEST_F(STEST_FE_TRANSOP_INSERT, stable_toposorting_of_insert_node)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  ComputeGraphPtr graph_check =  std::make_shared<ComputeGraph>("test_graph_check");
  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  GeTensorDesc src_tensor_desc(GeShape({100,200}), ge::FORMAT_HWCN, ge::DT_FLOAT16);
  src_tensor_desc.SetOriginShape(GeShape({100,200}));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_HWCN);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);
  ge::AttrUtils::SetInt(src_op, FE_IMPLY_TYPE, 6);

  OpDescPtr dst_op = std::make_shared<OpDesc>("BB", "BB");
  GeTensorDesc dst_tensor_desc(GeShape({7, 13, 16, 16}), ge::FORMAT_FRACTAL_Z, ge::DT_FLOAT16);
  dst_tensor_desc.SetOriginShape(GeShape({100, 200}));
  dst_tensor_desc.SetOriginFormat(ge::FORMAT_HWCN);
  dst_op->AddInputDesc(dst_tensor_desc);
  dst_op->AddOutputDesc(dst_tensor_desc);
  auto dst_node = graph->AddNode(dst_op);
  ge::AttrUtils::SetInt(dst_op, FE_IMPLY_TYPE, 6);
  vector<bool> input_const_vector = {false};
  dst_op->SetIsInputConst(input_const_vector);

  GraphUtils::AddEdge(src_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(0));

  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status status = trans_op_insert.InsertTransNodes(*(graph.get()), dst_node);
  ASSERT_EQ(fe::SUCCESS, status);

  for (auto &node : graph->GetDirectNode()) {
    ASSERT_NE(node, nullptr);
    if (node->GetType() == "UnsqueezeV2" || node->GetType() == "TransData") {
      int64_t topo_id = node->GetOpDesc()->GetId();;
      EXPECT_EQ(topo_id, 0);
    }
  }
}

TEST_F(STEST_FE_TRANSOP_INSERT, insert_cast_and_add_control_edge_1)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr refdata1 = std::make_shared<OpDesc>("refdata1", "RefData");
  OpDescPtr refdata2 = std::make_shared<OpDesc>("refdata2", "RefData");
  OpDescPtr assign = std::make_shared<OpDesc>("assign", "Assign");
  OpDescPtr no_op = std::make_shared<OpDesc>("no_op", "NoOp");
  OpDescPtr adamax1 = std::make_shared<OpDesc>("adamax1", "ApplyAdaMaxD");
  OpDescPtr adamax2 = std::make_shared<OpDesc>("adamax2", "ApplyAdaMaxD");
  OpDescPtr relu = std::make_shared<OpDesc>("relu", "Relu");
  // add descriptor
  ge::GeShape shape({1});
  GeTensorDesc tensor_desc1(shape, ge::FORMAT_ND, ge::DT_FLOAT);
  tensor_desc1.SetOriginFormat(ge::FORMAT_ND);
  tensor_desc1.SetOriginDataType(ge::DT_FLOAT);
  tensor_desc1.SetOriginShape(shape);
  GeTensorDesc tensor_desc2(shape, ge::FORMAT_ND, ge::DT_FLOAT16);
  tensor_desc2.SetOriginFormat(ge::FORMAT_ND);
  tensor_desc2.SetOriginDataType(ge::DT_FLOAT16);
  tensor_desc2.SetOriginShape(shape);
  refdata1->AddOutputDesc(tensor_desc1);
  refdata1->AddOutputDesc(tensor_desc1);
  refdata1->AddOutputDesc(tensor_desc1);
  assign->AddOutputDesc(tensor_desc1);
  refdata2->AddInputDesc(tensor_desc1);
  adamax1->AddInputDesc(tensor_desc2);
  adamax1->AddOutputDesc(tensor_desc2);
  adamax2->AddInputDesc(tensor_desc2);
  adamax2->AddOutputDesc(tensor_desc2);
  relu->AddInputDesc(tensor_desc2);
  relu->AddOutputDesc(tensor_desc2);
  // create nodes
  NodePtr refdata1_node = graph->AddNode(refdata1);
  NodePtr refdata2_node = graph->AddNode(refdata2);
  NodePtr assign_node = graph->AddNode(assign);
  NodePtr adamax1_node = graph->AddNode(adamax1);
  NodePtr adamax2_node = graph->AddNode(adamax2);
  NodePtr no_op_node = graph->AddNode(no_op);
  NodePtr relu_node = graph->AddNode(relu);
  // link edge
  ge::GraphUtils::AddEdge(refdata1_node->GetOutDataAnchor(0),
                          adamax1_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(refdata1_node->GetOutDataAnchor(0),
                          adamax2_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(assign_node->GetOutDataAnchor(0),
                          refdata2_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(assign_node->GetOutControlAnchor(),
                          adamax1_node->GetInControlAnchor());
  ge::GraphUtils::AddEdge(assign_node->GetOutControlAnchor(),
                          no_op_node->GetInControlAnchor());
  ge::GraphUtils::AddEdge(refdata2_node->GetOutControlAnchor(),
                          no_op_node->GetInControlAnchor());
  ge::GraphUtils::AddEdge(no_op_node->GetOutControlAnchor(),
                          adamax2_node->GetInControlAnchor());
  ge::GraphUtils::AddEdge(relu_node->GetOutControlAnchor(),
                          adamax2_node->GetInControlAnchor());
  ge::AttrUtils::SetStr(refdata2, "ref_var_src_var_name", "refdata1");

  EXPECT_EQ(graph->GetAllNodesSize(), 7);
  EXPECT_EQ(adamax2_node->GetInControlNodes().size(), 2);

  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status status = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));
  EXPECT_EQ(graph->GetAllNodesSize(), 9);
  EXPECT_EQ(adamax2_node->GetInControlNodes().size(), 0);
}

TEST_F(STEST_FE_TRANSOP_INSERT, insert_cast_and_add_control_edge_2)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr refdata1 = std::make_shared<OpDesc>("refdata1", "RefData");
  OpDescPtr refdata2 = std::make_shared<OpDesc>("refdata2", "RefData");
  OpDescPtr assign = std::make_shared<OpDesc>("assign", "Assign");
  OpDescPtr no_op = std::make_shared<OpDesc>("no_op", "NoOp");
  OpDescPtr adamax1 = std::make_shared<OpDesc>("adamax1", "ApplyAdaMaxD");
  OpDescPtr netoutput = std::make_shared<OpDesc>("netoutput", "NetOutput");
  OpDescPtr relu = std::make_shared<OpDesc>("relu", "Relu");
  // add descriptor
  ge::GeShape shape({1});
  GeTensorDesc tensor_desc1(shape, ge::FORMAT_ND, ge::DT_FLOAT);
  tensor_desc1.SetOriginFormat(ge::FORMAT_ND);
  tensor_desc1.SetOriginDataType(ge::DT_FLOAT);
  tensor_desc1.SetOriginShape(shape);
  GeTensorDesc tensor_desc2(shape, ge::FORMAT_ND, ge::DT_FLOAT16);
  tensor_desc2.SetOriginFormat(ge::FORMAT_ND);
  tensor_desc2.SetOriginDataType(ge::DT_FLOAT16);
  tensor_desc2.SetOriginShape(shape);
  refdata1->AddOutputDesc(tensor_desc1);
  refdata1->AddOutputDesc(tensor_desc1);
  refdata1->AddOutputDesc(tensor_desc1);
  assign->AddOutputDesc(tensor_desc1);
  refdata2->AddInputDesc(tensor_desc1);
  adamax1->AddInputDesc(tensor_desc2);
  adamax1->AddOutputDesc(tensor_desc2);
  netoutput->AddInputDesc(tensor_desc2);
  netoutput->AddOutputDesc(tensor_desc2);
  relu->AddInputDesc(tensor_desc2);
  relu->AddOutputDesc(tensor_desc2);
  // create nodes
  NodePtr refdata1_node = graph->AddNode(refdata1);
  NodePtr refdata2_node = graph->AddNode(refdata2);
  NodePtr assign_node = graph->AddNode(assign);
  NodePtr adamax1_node = graph->AddNode(adamax1);
  NodePtr netoutput_node = graph->AddNode(netoutput);
  NodePtr no_op_node = graph->AddNode(no_op);
  NodePtr relu_node = graph->AddNode(relu);
  // link edge
  ge::GraphUtils::AddEdge(refdata1_node->GetOutDataAnchor(0),
                          adamax1_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(refdata1_node->GetOutDataAnchor(0),
                          netoutput_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(assign_node->GetOutDataAnchor(0),
                          refdata2_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(assign_node->GetOutControlAnchor(),
                          adamax1_node->GetInControlAnchor());
  ge::GraphUtils::AddEdge(assign_node->GetOutControlAnchor(),
                          no_op_node->GetInControlAnchor());
  ge::GraphUtils::AddEdge(refdata2_node->GetOutControlAnchor(),
                          no_op_node->GetInControlAnchor());
  ge::GraphUtils::AddEdge(no_op_node->GetOutControlAnchor(),
                          netoutput_node->GetInControlAnchor());
  ge::GraphUtils::AddEdge(relu_node->GetOutControlAnchor(),
                          netoutput_node->GetInControlAnchor());
  ge::AttrUtils::SetStr(refdata2, "ref_var_src_var_name", "refdata1");

  EXPECT_EQ(graph->GetAllNodesSize(), 7);
  EXPECT_EQ(netoutput_node->GetInControlNodes().size(), 2);

  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status status = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));
  EXPECT_EQ(graph->GetAllNodesSize(), 9);
  EXPECT_EQ(netoutput_node->GetInControlNodes().size(), 2);
}

TEST_F(STEST_FE_TRANSOP_INSERT, trans_hwcn_to_c1hwc0)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  ComputeGraphPtr graph_check =  std::make_shared<ComputeGraph>("test_graph_check");
  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  GeTensorDesc src_tensor_desc(GeShape({100,200}), ge::FORMAT_HWCN, ge::DT_FLOAT16);
  src_tensor_desc.SetOriginShape(GeShape({100,200}));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_HWCN);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);
  ge::AttrUtils::SetInt(src_op, FE_IMPLY_TYPE, 6);

  OpDescPtr dst_op = std::make_shared<OpDesc>("BB", "BB");
  GeTensorDesc dst_tensor_desc(GeShape({1,2,4,8,16}), ge::FORMAT_C1HWC0, ge::DT_FLOAT16);
  dst_tensor_desc.SetOriginShape(GeShape({100, 200}));
  dst_tensor_desc.SetOriginFormat(ge::FORMAT_HWCN);
  dst_op->AddInputDesc(dst_tensor_desc);
  dst_op->AddOutputDesc(dst_tensor_desc);
  auto dst_node = graph->AddNode(dst_op);
  ge::AttrUtils::SetInt(dst_op, FE_IMPLY_TYPE, 6);
  vector<bool> input_const_vector = {false};
  dst_op->SetIsInputConst(input_const_vector);

  GraphUtils::AddEdge(src_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(0));

  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  EXPECT_EQ(graph->GetAllNodesSize(), 2);
  Status status = trans_op_insert.InsertTransNodes(*(graph.get()), dst_node);
  ASSERT_EQ(fe::SUCCESS, status);
  EXPECT_EQ(graph->GetAllNodesSize(), 4);
}

TEST_F(STEST_FE_TRANSOP_INSERT, trans_hwcn_to_c1hwc0_src_failed)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  GeTensorDesc src_tensor_desc(GeShape({100,200}), ge::FORMAT_RESERVED, ge::DT_UNDEFINED);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);
  ge::AttrUtils::SetInt(src_op, FE_IMPLY_TYPE, 6);

  OpDescPtr dst_op = std::make_shared<OpDesc>("BB", "BB");
  GeTensorDesc dst_tensor_desc(GeShape({7, 13, 16, 16}), ge::FORMAT_C1HWC0, ge::DT_UNDEFINED);
  dst_op->AddInputDesc(dst_tensor_desc);
  dst_op->AddOutputDesc(dst_tensor_desc);
  auto dst_node = graph->AddNode(dst_op);
  ge::AttrUtils::SetInt(dst_op, FE_IMPLY_TYPE, 6);
  vector<bool> input_const_vector = {false};
  dst_op->SetIsInputConst(input_const_vector);
  GraphUtils::AddEdge(src_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(0));
  src_node->GetOutDataAnchor(0)->SetIdx(1);

  ConcecutivePrinciple use_concecutive_principle = ConcecutivePrinciple::kConcecutive;
  TransNodeInsertion trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.global_trans_info_ptr_ = std::make_shared<TransInfo>();
  Status status = trans_op_insert.FillTransInfo(dst_node->GetInDataAnchor(0), src_node->GetOutDataAnchor(0), src_node, dst_node, use_concecutive_principle);
  ASSERT_EQ(fe::FAILED, status);
}

TEST_F(STEST_FE_TRANSOP_INSERT, trans_hwcn_to_c1hwc0_dst_failed)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  ComputeGraphPtr graph_check =  std::make_shared<ComputeGraph>("test_graph_check");
  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  GeTensorDesc src_tensor_desc(GeShape({100,200}), ge::FORMAT_HWCN, ge::DT_FLOAT16);
  src_tensor_desc.SetOriginShape(GeShape({100,200}));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_HWCN);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);
  ge::AttrUtils::SetInt(src_op, FE_IMPLY_TYPE, 6);

  OpDescPtr dst_op = std::make_shared<OpDesc>("BB", "BB");
  GeTensorDesc dst_tensor_desc(GeShape({1,2,4,8,16}), ge::FORMAT_RESERVED, ge::DT_UNDEFINED);
  dst_tensor_desc.SetOriginShape(GeShape({100, 200}));
  dst_tensor_desc.SetOriginFormat(ge::FORMAT_HWCN);
  dst_op->AddInputDesc(dst_tensor_desc);
  dst_op->AddOutputDesc(dst_tensor_desc);
  auto dst_node = graph->AddNode(dst_op);
  ge::AttrUtils::SetInt(dst_op, FE_IMPLY_TYPE, 6);
  vector<bool> input_const_vector = {false};
  dst_op->SetIsInputConst(input_const_vector);

  GraphUtils::AddEdge(src_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(0));

  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  EXPECT_EQ(graph->GetAllNodesSize(), 2);
  Status status = trans_op_insert.InsertTransNodes(*(graph.get()), dst_node);
  ASSERT_EQ(fe::FAILED, status);
}

TEST_F(STEST_FE_TRANSOP_INSERT, merge_one_node_test_1)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  ComputeGraphPtr graph_check =  std::make_shared<ComputeGraph>("test_graph_check");
  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  GeTensorDesc src_tensor_desc(GeShape({100,200}), ge::FORMAT_HWCN, ge::DT_FLOAT16);
  src_tensor_desc.SetOriginShape(GeShape({100,200}));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_HWCN);
  src_op->AddInputDesc(src_tensor_desc);
  src_op->AddOutputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);
  ge::AttrUtils::SetInt(src_op, FE_IMPLY_TYPE, 6);

  uint32_t current_in_anchor_index;
  std::stack<uint32_t> in_anchor_index_stack;
  TransNodeMerging trans_op_merger;
  trans_op_merger.MergeOneNode(*(graph.get()), src_node, current_in_anchor_index, in_anchor_index_stack);
}