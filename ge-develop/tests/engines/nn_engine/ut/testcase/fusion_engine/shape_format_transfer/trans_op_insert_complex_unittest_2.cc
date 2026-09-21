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
#include "graph/debug/ge_attr_define.h"

#define protected public
#define private   public
#include "graph_optimizer/fusion_common/fusion_pass_manager.h"
#include "graph_optimizer/graph_fusion/graph_fusion.h"
#include "graph_optimizer/fe_graph_optimizer.h"
#include "../../../../graph_constructor/graph_constructor.h"
#include "adapter/common/op_store_adapter_manager.h"
#include "adapter/tbe_adapter/tbe_op_store_adapter.h"
#include "ops_kernel_store/fe_ops_kernel_info_store.h"
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


int k_clear_atomic_id_flag = false;
uint64_t GetTransAtomicIdFromZero() {
  static std::atomic<uint64_t> global_trans_atomic_id(0);
  if (k_clear_atomic_id_flag) {
    global_trans_atomic_id = 0;
    return 0;
  }
  return global_trans_atomic_id.fetch_add(1, std::memory_order_relaxed);
}

class UTEST_FE_TRANSOP_INSERT_COMPLEX_2 : public testing::Test {
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

    k_clear_atomic_id_flag = true;
    GetTransAtomicIdFromZero();
    k_clear_atomic_id_flag = false;
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

TEST_F(UTEST_FE_TRANSOP_INSERT_COMPLEX_2, InsertAllTransop_1) {
  // src:cce op, dst:cce op
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr src_op = std::make_shared<OpDesc>("var1", fe::VARIABLE);
  vector<int64_t> dims4_d = {100,200,300,400};
  vector<int64_t> dimsfz = {380000,25,16,16};
  GeTensorDesc src_tensor_desc(GeShape(dims4_d), ge::FORMAT_HWCN, ge::DT_FLOAT16);
  src_tensor_desc.SetOriginShape(GeShape(dims4_d));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_HWCN);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);
  ge::AttrUtils::SetInt(src_op, FE_IMPLY_TYPE, -1);
  ge::AttrUtils::SetStr(src_op, PARENT_OP_TYPE, "Variable");

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
  ge::AttrUtils::SetBool(trans_op_0, ge::ATTR_INSERTED_BY_GE, true);

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
  ge::AttrUtils::SetBool(trans_op_1, ge::ATTR_INSERTED_BY_GE, true);
  
  OpDescPtr dst_op = std::make_shared<OpDesc>("var2", fe::VARIABLE);
  GeTensorDesc dst_tensor_desc(GeShape(dims4_d), ge::FORMAT_HWCN, ge::DT_FLOAT16);
  dst_tensor_desc.SetOriginShape(GeShape(dims4_d));
  dst_tensor_desc.SetOriginFormat(ge::FORMAT_HWCN);
  dst_op->AddOutputDesc(dst_tensor_desc);
  dst_op->AddInputDesc(dst_tensor_desc);
  auto dst_node = graph->AddNode(dst_op);
  ge::AttrUtils::SetInt(dst_op, FE_IMPLY_TYPE, -1);
  ge::AttrUtils::SetStr(dst_op, PARENT_OP_TYPE, "Variable");

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
        EXPECT_EQ(shape.GetDimNum(), 4);
        EXPECT_EQ(shape.GetDims(), dimsfz);
        EXPECT_EQ(ge::FORMAT_FRACTAL_Z,
                  node->GetOpDesc()->GetInputDescPtr(0)->GetFormat());
        EXPECT_EQ(ge::DT_FLOAT,
                  node->GetOpDesc()->GetInputDescPtr(0)->GetDataType());
      }
      {
        ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        EXPECT_EQ(shape.GetDimNum(), 4);
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
          EXPECT_EQ(shape.GetDimNum(), 4);
          EXPECT_EQ(shape.GetDims(), dimsfz);
          EXPECT_EQ(ge::FORMAT_FRACTAL_Z,
                    node->GetOpDesc()->GetInputDescPtr(0)->GetFormat());
          EXPECT_EQ(ge::DT_FLOAT16,
                    node->GetOpDesc()->GetInputDescPtr(0)->GetDataType());
        }
        {
          ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
          EXPECT_EQ(shape.GetDimNum(), 4);
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
          EXPECT_EQ(shape.GetDimNum(), 4);
          EXPECT_EQ(shape.GetDims(), dimsfz);
          EXPECT_EQ(ge::FORMAT_FRACTAL_Z,
                    node->GetOpDesc()->GetInputDescPtr(0)->GetFormat());
          EXPECT_EQ(ge::DT_FLOAT,
                    node->GetOpDesc()->GetInputDescPtr(0)->GetDataType());
        }
        {
          ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
          EXPECT_EQ(shape.GetDimNum(), 4);
          EXPECT_EQ(shape.GetDims(), dimsfz);
          EXPECT_EQ(ge::FORMAT_FRACTAL_Z,
                    node->GetOpDesc()->GetOutputDescPtr(0)->GetFormat());
          EXPECT_EQ(ge::DT_FLOAT16,
                    node->GetOpDesc()->GetOutputDescPtr(0)->GetDataType());
        }
      }
    }
  }
  EXPECT_EQ(count_node, 7);
}



TEST_F(UTEST_FE_TRANSOP_INSERT_COMPLEX_2, InsertAllTransop_2) {
  // src:cce op, dst:cce op
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr src_op = std::make_shared<OpDesc>("var1", OP_TYPE_PLACE_HOLDER);
  vector<int64_t> dims4_d = {100,200,300,400};
  vector<int64_t> dimsfz = {380000,25,16,16};
  GeTensorDesc src_tensor_desc(GeShape(dims4_d), ge::FORMAT_HWCN, ge::DT_FLOAT16);
  src_tensor_desc.SetOriginShape(GeShape(dims4_d));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_HWCN);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);
  ge::AttrUtils::SetInt(src_op, FE_IMPLY_TYPE, -1);
  ge::AttrUtils::SetStr(src_op, PARENT_OP_TYPE, "Variable");

  OpDescPtr trans_op_0 = std::make_shared<OpDesc>("transdata_0", "TransData");
  GeTensorDesc trans_in_tensor_desc(GeShape(dimsfz), ge::FORMAT_FRACTAL_Z, ge::DT_FLOAT16);
  trans_in_tensor_desc.SetOriginShape(GeShape(dims4_d));
  trans_in_tensor_desc.SetOriginFormat(ge::FORMAT_HWCN);
  trans_in_tensor_desc.SetOriginDataType(ge::DT_FLOAT16);
  GeTensorDesc trans_out_tensor_desc(GeShape(dims4_d), ge::FORMAT_HWCN, ge::DT_FLOAT16);
  trans_out_tensor_desc.SetOriginShape(GeShape(dims4_d));
  trans_out_tensor_desc.SetOriginFormat(ge::FORMAT_HWCN);
  trans_out_tensor_desc.SetOriginDataType(ge::DT_FLOAT16);
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

  OpDescPtr dst_op = std::make_shared<OpDesc>("var2", OP_TYPE_END);
  GeTensorDesc dst_tensor_desc(GeShape(dims4_d), ge::FORMAT_HWCN, ge::DT_FLOAT16);
  dst_tensor_desc.SetOriginShape(GeShape(dims4_d));
  dst_tensor_desc.SetOriginFormat(ge::FORMAT_HWCN);
  dst_op->AddOutputDesc(dst_tensor_desc);
  dst_op->AddInputDesc(dst_tensor_desc);
  auto dst_node = graph->AddNode(dst_op);
  ge::AttrUtils::SetInt(dst_op, FE_IMPLY_TYPE, -1);
  ge::AttrUtils::SetStr(dst_op, PARENT_OP_TYPE, "Variable");

  GraphUtils::AddEdge(src_node->GetOutDataAnchor(0), trans_node_0->GetInDataAnchor(0));
  GraphUtils::AddEdge(trans_node_0->GetOutDataAnchor(0), apply_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(apply_node->GetOutDataAnchor(0), trans_node_1->GetInDataAnchor(0));
  GraphUtils::AddEdge(trans_node_1->GetOutDataAnchor(0), dst_node->GetInDataAnchor(0));

  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status status = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));

  int count_node = 0;
  uint32_t index_cast = 0;
  uint32_t index_transdata = 0;
  ASSERT_EQ(fe::SUCCESS, status);
  for (auto node : graph->GetDirectNode()) {
    ASSERT_NE(node, nullptr);
    count_node++;
    EXPECT_NE(node->GetName(),"transdata_0");
    EXPECT_NE(node->GetName(),"transdata_1");
    if (node->GetName() == "apply") {
      {
        ge::GeShape shape = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
        EXPECT_EQ(shape.GetDimNum(), 4);
        EXPECT_EQ(shape.GetDims(), dimsfz);
        EXPECT_EQ(ge::FORMAT_FRACTAL_Z,
                  node->GetOpDesc()->GetInputDescPtr(0)->GetFormat());
        EXPECT_EQ(ge::DT_FLOAT,
                  node->GetOpDesc()->GetInputDescPtr(0)->GetDataType());
      }
      {
        ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
        EXPECT_EQ(shape.GetDimNum(), 4);
        EXPECT_EQ(shape.GetDims(), dimsfz);
        EXPECT_EQ(ge::FORMAT_FRACTAL_Z,
                  node->GetOpDesc()->GetOutputDescPtr(0)->GetFormat());
        EXPECT_EQ(ge::DT_FLOAT,
                  node->GetOpDesc()->GetOutputDescPtr(0)->GetDataType());
      }
    }
    if (node->GetType() == "Cast") {
      if (index_cast == 0) {
        {
          ge::GeShape shape = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
          EXPECT_EQ(shape.GetDimNum(), 4);
          EXPECT_EQ(shape.GetDims(), dimsfz);
          EXPECT_EQ(ge::FORMAT_FRACTAL_Z,
                    node->GetOpDesc()->GetInputDescPtr(0)->GetFormat());
          EXPECT_EQ(ge::DT_FLOAT16,
                    node->GetOpDesc()->GetInputDescPtr(0)->GetDataType());
        }
        {
          ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
          EXPECT_EQ(shape.GetDimNum(), 4);
          EXPECT_EQ(shape.GetDims(), dimsfz);
          EXPECT_EQ(ge::FORMAT_FRACTAL_Z,
                    node->GetOpDesc()->GetOutputDescPtr(0)->GetFormat());
          EXPECT_EQ(ge::DT_FLOAT,
                    node->GetOpDesc()->GetOutputDescPtr(0)->GetDataType());
        }
        index_cast++;
      } else {
        {
          ge::GeShape shape = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
          EXPECT_EQ(shape.GetDimNum(), 4);
          EXPECT_EQ(shape.GetDims(), dimsfz);
          EXPECT_EQ(ge::FORMAT_FRACTAL_Z,
                    node->GetOpDesc()->GetInputDescPtr(0)->GetFormat());
          EXPECT_EQ(ge::DT_FLOAT,
                    node->GetOpDesc()->GetInputDescPtr(0)->GetDataType());
        }
        {
          ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
          EXPECT_EQ(shape.GetDimNum(), 4);
          EXPECT_EQ(shape.GetDims(), dimsfz);
          EXPECT_EQ(ge::FORMAT_FRACTAL_Z,
                    node->GetOpDesc()->GetOutputDescPtr(0)->GetFormat());
          EXPECT_EQ(ge::DT_FLOAT16,
                    node->GetOpDesc()->GetOutputDescPtr(0)->GetDataType());
        }
      }
    }
    if (node->GetType() == "TransData") {
      if (index_transdata == 0) {
        {
          ge::GeShape shape = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
          EXPECT_EQ(shape.GetDimNum(), 4);
          EXPECT_EQ(shape.GetDims(), dims4_d);
          EXPECT_EQ(ge::FORMAT_HWCN,
                    node->GetOpDesc()->GetInputDescPtr(0)->GetFormat());
          EXPECT_EQ(ge::DT_FLOAT16,
                    node->GetOpDesc()->GetInputDescPtr(0)->GetDataType());
        }
        {
          ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
          EXPECT_EQ(shape.GetDimNum(), 4);
          EXPECT_EQ(shape.GetDims(), dimsfz);
          EXPECT_EQ(ge::FORMAT_FRACTAL_Z,
                    GetPrimaryFormat(node->GetOpDesc()->GetOutputDescPtr(0)->GetFormat()));
          EXPECT_EQ(ge::DT_FLOAT16,
                    node->GetOpDesc()->GetOutputDescPtr(0)->GetDataType());
        }
        index_transdata++;
      } else {
        {
          ge::GeShape shape = node->GetOpDesc()->GetInputDescPtr(0)->GetShape();
          EXPECT_EQ(shape.GetDimNum(), 4);
          EXPECT_EQ(shape.GetDims(), dimsfz);
          EXPECT_EQ(ge::FORMAT_FRACTAL_Z,
                    node->GetOpDesc()->GetInputDescPtr(0)->GetFormat());
          EXPECT_EQ(ge::DT_FLOAT16,
                    node->GetOpDesc()->GetInputDescPtr(0)->GetDataType());
        }
        {
          ge::GeShape shape = node->GetOpDesc()->GetOutputDescPtr(0)->GetShape();
          EXPECT_EQ(shape.GetDimNum(), 4);
          EXPECT_EQ(shape.GetDims(), dims4_d);
          EXPECT_EQ(ge::FORMAT_HWCN,
                    node->GetOpDesc()->GetOutputDescPtr(0)->GetFormat());
          EXPECT_EQ(ge::DT_FLOAT16,
                    node->GetOpDesc()->GetOutputDescPtr(0)->GetDataType());
        }
      }
    }
  }
  EXPECT_EQ(count_node, 7);
}



TEST_F(UTEST_FE_TRANSOP_INSERT_COMPLEX_2, InsertPermuteNode)
{
  // src:cce op, dst:cce op
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  std::vector<int64_t> dim_src = {1, 1024, 256, 512};
  OpDescPtr src_op = std::make_shared<OpDesc>("B", "B");
  GeTensorDesc src_tensor_desc(GeShape(dim_src), ge::FORMAT_NCHW, ge::DT_FLOAT);
  src_tensor_desc.SetOriginShape(GeShape(dim_src));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);
  ge::AttrUtils::SetInt(src_op, FE_IMPLY_TYPE, 6);

  std::vector<int64_t> dim_dst = {1, 3, 4, 2};
  OpDescPtr dst_op = std::make_shared<OpDesc>("D", "D");
  GeTensorDesc dst_tensor_desc(GeShape(dim_dst), ge::FORMAT_NHWC, ge::DT_FLOAT);
  dst_tensor_desc.SetOriginShape(GeShape(dim_dst));
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
    EXPECT_NE(node->GetType(), "TransposeD");
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
  EXPECT_EQ(count_node, 2);
}


TEST_F(UTEST_FE_TRANSOP_INSERT_COMPLEX_2, merge_two_transdata_with_diff_shape) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  GraphConstructor test(graph);
  test.SetInput("A", ge::FORMAT_NC1HWC0, "B", ge::FORMAT_NC1HWC0,
                ge::FORMAT_NCHW, ge::FORMAT_NCHW, {8, 288, 28, 28},
                {72, 32, 28, 28});

  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status status = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));
  EXPECT_EQ(status, fe::SUCCESS);
  uint32_t count_node = 0;
  for (auto &node : graph->GetDirectNode()) {
    EXPECT_NE(node->GetName(), fe::TRANSDATA);
    count_node++;
  }
  EXPECT_EQ(count_node, 2);
}


/* shape's total product is the same, but the sequence is not the same and
 * does not meet the requirements of reshape*/
TEST_F(UTEST_FE_TRANSOP_INSERT_COMPLEX_2, merge_two_transdata_with_diff_shape_02) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  GraphConstructor test(graph);
  test.SetInput("A", ge::FORMAT_NC1HWC0, "B", ge::FORMAT_NC1HWC0,
                ge::FORMAT_NCHW, ge::FORMAT_NCHW, {8, 288, 14, 15},
                {72, 32, 15, 14});

  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status status = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));
  EXPECT_EQ(status, fe::SUCCESS);
  uint32_t count_node = 0;
  for (auto &node : graph->GetDirectNode()) {
    count_node++;
  }
  EXPECT_EQ(count_node, 4);
}

/* shape's total product is not the same */
TEST_F(UTEST_FE_TRANSOP_INSERT_COMPLEX_2, merge_two_transdata_with_diff_shape_03) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  GraphConstructor test(graph);
  test.SetInput("A", ge::FORMAT_NC1HWC0, "B", ge::FORMAT_NC1HWC0,
                ge::FORMAT_NCHW, ge::FORMAT_NCHW, {8, 288, 14, 15},
                {72, 32, 14, 30});

  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status status = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));
  EXPECT_EQ(status, fe::SUCCESS);
  uint32_t count_node = 0;
  for (auto &node : graph->GetDirectNode()) {
    count_node++;
  }
  EXPECT_EQ(count_node, 4);
}

/* shape's total product is the same, N * C is also the same, but axis of node
 * A cannot be divided by 16. */
TEST_F(UTEST_FE_TRANSOP_INSERT_COMPLEX_2, merge_two_transdata_with_diff_shape_04) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  GraphConstructor test(graph);
  test.SetInput("A", ge::FORMAT_NC1HWC0, "B", ge::FORMAT_NC1HWC0,
                ge::FORMAT_NCHW, ge::FORMAT_NCHW, {64, 4*9, 14, 30},
                {32, 72, 14, 30});

  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status status = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));
  EXPECT_EQ(status, fe::SUCCESS);
  uint32_t count_node = 0;
  for (auto &node : graph->GetDirectNode()) {
    count_node++;
  }
  EXPECT_EQ(count_node, 4);
}


/* shape's total product is the same, N * C is also the same, but axis of node
 * A cannot be divided by 32. */
TEST_F(UTEST_FE_TRANSOP_INSERT_COMPLEX_2, merge_two_transdata_with_diff_shape_05) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_INT8);
  test.SetInput("A", ge::FORMAT_NC1HWC0, "B", ge::FORMAT_NC1HWC0,
                ge::FORMAT_NCHW, ge::FORMAT_NCHW, {16, 9*16, 14, 30},
                {32, 72, 14, 30});

  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status status = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));
  EXPECT_EQ(status, fe::SUCCESS);
  uint32_t count_node = 0;
  for (auto &node : graph->GetDirectNode()) {
    count_node++;
  }
  EXPECT_EQ(count_node, 4);
}


/* shape's total product is the same, N * C is also the same, and axis of node
 * A can be divided by 32. */
TEST_F(UTEST_FE_TRANSOP_INSERT_COMPLEX_2, merge_two_transdata_with_diff_shape_06) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_INT8);
  test.SetInput("A", ge::FORMAT_NC1HWC0, "B", ge::FORMAT_NC1HWC0,
                ge::FORMAT_NCHW, ge::FORMAT_NCHW, {36, 64, 14, 30},
                {72, 32, 14, 30});

  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status status = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));
  EXPECT_EQ(status, fe::SUCCESS);
  uint32_t count_node = 0;
  for (auto &node : graph->GetDirectNode()) {
    count_node++;
  }
  EXPECT_EQ(count_node, 2);
}


/* shape's total product is the same, N * C is not the same */
TEST_F(UTEST_FE_TRANSOP_INSERT_COMPLEX_2, merge_two_transdata_with_diff_shape_07) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_INT8);
  test.SetInput("A", ge::FORMAT_NC1HWC0, "B", ge::FORMAT_NC1HWC0,
                ge::FORMAT_NCHW, ge::FORMAT_NCHW, {1, 64, 128, 1},
                {64, 128, 1, 1});

  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status status = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));
  EXPECT_EQ(status, fe::SUCCESS);
  uint32_t count_node = 0;
  for (auto &node : graph->GetDirectNode()) {
    count_node++;
  }
  EXPECT_EQ(count_node, 4);
}


/* shape's total product is the same, NHWC is the same */
TEST_F(UTEST_FE_TRANSOP_INSERT_COMPLEX_2, merge_two_transdata_with_diff_shape_08) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  GraphConstructor test(graph, "", ge::FORMAT_NHWC, ge::DT_INT8);
  test.SetInput("A", ge::FORMAT_NC1HWC0, "B", ge::FORMAT_NC1HWC0,
                ge::FORMAT_NHWC, ge::FORMAT_NHWC, {1, 64, 128, 1},
                {64, 128, 1, 1});

  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status status = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));
  EXPECT_EQ(status, fe::SUCCESS);
  uint32_t count_node = 0;
  for (auto &node : graph->GetDirectNode()) {
    count_node++;
  }
  EXPECT_EQ(count_node, 2);
}



/* shape's total product is the same, NHWC is the same, but C is not the same
 * and they can not be divided by 16. */
TEST_F(UTEST_FE_TRANSOP_INSERT_COMPLEX_2, merge_two_transdata_with_diff_shape_09) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  GraphConstructor test(graph, "", ge::FORMAT_NHWC, ge::DT_INT8);
  test.SetInput("A", ge::FORMAT_NC1HWC0, "B", ge::FORMAT_NC1HWC0,
                ge::FORMAT_NHWC, ge::FORMAT_NHWC, {1, 64, 128, 7},
                {64, 128, 7, 1});

  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status status = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));
  EXPECT_EQ(status, fe::SUCCESS);
  uint32_t count_node = 0;
  for (auto &node : graph->GetDirectNode()) {
    count_node++;
  }
  EXPECT_EQ(count_node, 4);
}

/* shape's total product is the same, HWC is the same.
 * Without transpose the format of two transdata is not the same. */
TEST_F(UTEST_FE_TRANSOP_INSERT_COMPLEX_2, merge_two_transdata_with_diff_shape_10) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  GraphConstructor test(graph, "", ge::FORMAT_HWCN, ge::DT_INT8);
  test.SetInput("A", ge::FORMAT_NC1HWC0, "B", ge::FORMAT_NC1HWC0,
                ge::FORMAT_HWCN, ge::FORMAT_HWCN, {32, 64, 128, 1},
                {64, 128, 32, 1});

  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status status = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));
  EXPECT_EQ(status, fe::SUCCESS);
  uint32_t count_node = 0;
  for (auto &node : graph->GetDirectNode()) {
    count_node++;
  }
  EXPECT_EQ(count_node, 8);
}



/* A -> TransData(4->5, bool) -> Cast(5HD, bool->fp16) -> B (5HD, fp16)
* will be changed to :
* A -> Cast(4D, bool->fp16) -> TransData(4->5, fp16) -> B (5HD, fp16)
*/
TEST_F(UTEST_FE_TRANSOP_INSERT_COMPLEX_2, SwitchCastAndTransdata) {
  ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::GeShape original_shape = ge::GeShape({3, 12, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NHWC, ge::DT_FLOAT,
                        original_shape);
  test.AddOpDesc("a", "A", 1, 1)
      .AddOpDesc("transdata", fe::TRANSDATA, 1, 1)
      .AddOpDesc("cast", fe::CAST, 1, 1)
      .AddOpDesc("b", "B", 1, 1);

  test.SetInput("transdata", ge::FORMAT_NHWC, ge::DT_BOOL, "a", ge::FORMAT_NHWC, ge::DT_BOOL)
      .SetInput("cast", ge::FORMAT_NC1HWC0, ge::DT_BOOL, "transdata", ge::FORMAT_NC1HWC0, ge::DT_BOOL)
      .SetInput("b", ge::FORMAT_NC1HWC0, ge::DT_FLOAT16, "cast", ge::FORMAT_NC1HWC0,  ge::DT_FLOAT16);
  test.DumpGraph(graph);

  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status ret = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));
  EXPECT_EQ(ret, fe::SUCCESS);

  FusionRuleManagerPtr fusion_rule_mgr_ptr = make_shared<FusionRuleManager>(fe_ops_kernel_info_store_ptr_);
  auto graph_fusion_ptr = std::make_shared<GraphFusion>(fusion_rule_mgr_ptr, fe_ops_kernel_info_store_ptr_,
                                                        nullptr);
  graph_fusion_ptr->SetEngineName(AI_CORE_NAME);
  ret = graph_fusion_ptr->SwitchTransDataAndCast(*(graph.get()), trans_op_insert.GetOptimizableCast());
  EXPECT_EQ(ret, fe::SUCCESS);
  test.DumpGraph(graph);
  ge::NodePtr cast;
  ge::NodePtr transdata;
  test.GetNodeByName("cast", cast);
  test.GetNodeByName("transdata", transdata);
  ASSERT_EQ(cast->GetInAllNodes().size(), 1);
  ASSERT_EQ(cast->GetInAllNodes().at(0)->GetType(), "A");
  ASSERT_EQ(transdata->GetInAllNodes().size(), 1);
  ASSERT_EQ(transdata->GetInAllNodes().at(0)->GetType(), fe::CAST);

  EXPECT_EQ(cast->GetOpDesc()->GetInputDesc(0).GetDataType(), ge::DT_BOOL);
  EXPECT_EQ(cast->GetOpDesc()->GetOutputDesc(0).GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(cast->GetOpDesc()->GetInputDesc(0).GetFormat(), ge::FORMAT_NHWC);
  EXPECT_EQ(cast->GetOpDesc()->GetOutputDesc(0).GetFormat(), ge::FORMAT_NHWC);

  EXPECT_EQ(transdata->GetOpDesc()->GetInputDesc(0).GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(transdata->GetOpDesc()->GetOutputDesc(0).GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(transdata->GetOpDesc()->GetInputDesc(0).GetFormat(), ge::FORMAT_NHWC);
  EXPECT_EQ(transdata->GetOpDesc()->GetOutputDesc(0).GetFormat(), ge::FORMAT_NC1HWC0);
}


/* The following case is RARE.
* A -> TransData(5->4, bool) -> Cast(4D, bool->fp16) -> B (4D, fp16)
* will be changed to :
* A -> Cast(5D, bool->fp16) -> TransData(5->4, fp16) -> B (4D, fp16) */
TEST_F(UTEST_FE_TRANSOP_INSERT_COMPLEX_2, SwitchCastAndTransdata_2) {
  ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::GeShape original_shape = ge::GeShape({3, 12, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NHWC, ge::DT_FLOAT,
                        original_shape);
  test.AddOpDesc("a", "A", 1, 1)
      .AddOpDesc("transdata", fe::TRANSDATA, 1, 1)
      .AddOpDesc("cast", fe::CAST, 1, 1)
      .AddOpDesc("b", "B", 1, 1);

  test.SetInput("transdata", ge::FORMAT_NC1HWC0, ge::DT_BOOL, "a", ge::FORMAT_NC1HWC0, ge::DT_BOOL)
      .SetInput("cast", ge::FORMAT_NHWC, ge::DT_BOOL, "transdata", ge::FORMAT_NHWC, ge::DT_BOOL)
      .SetInput("b", ge::FORMAT_NHWC, ge::DT_FLOAT16, "cast", ge::FORMAT_NHWC,  ge::DT_FLOAT16);
  test.DumpGraph(graph);

  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status ret = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));
  EXPECT_EQ(ret, fe::SUCCESS);

  FusionRuleManagerPtr fusion_rule_mgr_ptr = make_shared<FusionRuleManager>(fe_ops_kernel_info_store_ptr_);
  auto graph_fusion_ptr = std::make_shared<GraphFusion>(fusion_rule_mgr_ptr, fe_ops_kernel_info_store_ptr_,
                                                        nullptr);
  graph_fusion_ptr->SetEngineName(AI_CORE_NAME);
  ret = graph_fusion_ptr->SwitchTransDataAndCast(*(graph.get()), trans_op_insert.GetOptimizableCast());
  EXPECT_EQ(ret, fe::SUCCESS);
  test.DumpGraph(graph);
  ge::NodePtr cast;
  ge::NodePtr transdata;
  test.GetNodeByName("cast", cast);
  test.GetNodeByName("transdata", transdata);
  ASSERT_EQ(cast->GetInAllNodes().size(), 1);
  ASSERT_EQ(cast->GetInAllNodes().at(0)->GetType(), "A");
  ASSERT_EQ(transdata->GetInAllNodes().size(), 1);
  ASSERT_EQ(transdata->GetInAllNodes().at(0)->GetType(), fe::CAST);

  EXPECT_EQ(cast->GetOpDesc()->GetInputDesc(0).GetDataType(), ge::DT_BOOL);
  EXPECT_EQ(cast->GetOpDesc()->GetOutputDesc(0).GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(cast->GetOpDesc()->GetInputDesc(0).GetFormat(), ge::FORMAT_NC1HWC0);
  EXPECT_EQ(cast->GetOpDesc()->GetOutputDesc(0).GetFormat(), ge::FORMAT_NC1HWC0);

  EXPECT_EQ(transdata->GetOpDesc()->GetInputDesc(0).GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(transdata->GetOpDesc()->GetOutputDesc(0).GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(transdata->GetOpDesc()->GetInputDesc(0).GetFormat(), ge::FORMAT_NC1HWC0);
  EXPECT_EQ(transdata->GetOpDesc()->GetOutputDesc(0).GetFormat(), ge::FORMAT_NHWC);
}



/* A -> Cast(5HD, fp16->bool) -> TransData(5->4, bool) -> B (4D, bool)
* will be changed to :
* A -> TransData(5->4, fp16) -> Cast(4D, fp16->bool) -> B (4D, bool)
*/
TEST_F(UTEST_FE_TRANSOP_INSERT_COMPLEX_2, SwitchCastAndTransdata_3) {
  ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::GeShape original_shape = ge::GeShape({3, 12, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NHWC, ge::DT_FLOAT,
                        original_shape);
  test.AddOpDesc("a", "A", 1, 1)
      .AddOpDesc("transdata", fe::TRANSDATA, 1, 1)
      .AddOpDesc("cast", fe::CAST, 1, 1)
      .AddOpDesc("b", "B", 1, 1);

  test.SetInput("cast", ge::FORMAT_NC1HWC0, ge::DT_FLOAT16, "a", ge::FORMAT_NC1HWC0, ge::DT_FLOAT16)
      .SetInput("transdata", ge::FORMAT_NC1HWC0, ge::DT_BOOL, "cast", ge::FORMAT_NC1HWC0, ge::DT_BOOL)
      .SetInput("b", ge::FORMAT_NHWC, ge::DT_BOOL, "transdata", ge::FORMAT_NHWC,  ge::DT_BOOL);
  test.DumpGraph(graph);

  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status ret = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));
  EXPECT_EQ(ret, fe::SUCCESS);

  FusionRuleManagerPtr fusion_rule_mgr_ptr = make_shared<FusionRuleManager>(fe_ops_kernel_info_store_ptr_);
  auto graph_fusion_ptr = std::make_shared<GraphFusion>(fusion_rule_mgr_ptr, fe_ops_kernel_info_store_ptr_,
                                                        nullptr);
  graph_fusion_ptr->SetEngineName(AI_CORE_NAME);
  ret = graph_fusion_ptr->SwitchTransDataAndCast(*(graph.get()), trans_op_insert.GetOptimizableCast());
  EXPECT_EQ(ret, fe::SUCCESS);
  test.DumpGraph(graph);
  ge::NodePtr cast;
  ge::NodePtr transdata;
  test.GetNodeByName("cast", cast);
  test.GetNodeByName("transdata", transdata);
  ASSERT_EQ(cast->GetInAllNodes().size(), 1);
  ASSERT_EQ(cast->GetInAllNodes().at(0)->GetType(), fe::TRANSDATA);
  EXPECT_EQ(cast->GetOpDesc()->GetInputDesc(0).GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(cast->GetOpDesc()->GetOutputDesc(0).GetDataType(), ge::DT_BOOL);
  EXPECT_EQ(cast->GetOpDesc()->GetInputDesc(0).GetFormat(), ge::FORMAT_NHWC);
  EXPECT_EQ(cast->GetOpDesc()->GetOutputDesc(0).GetFormat(), ge::FORMAT_NHWC);

  EXPECT_EQ(transdata->GetOpDesc()->GetInputDesc(0).GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(transdata->GetOpDesc()->GetOutputDesc(0).GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(transdata->GetOpDesc()->GetInputDesc(0).GetFormat(), ge::FORMAT_NC1HWC0);
  EXPECT_EQ(transdata->GetOpDesc()->GetOutputDesc(0).GetFormat(), ge::FORMAT_NHWC);
}

/* A -> Cast(4D, fp16->bool) -> TransData(4->5, bool) -> B (5HD, bool)
 * will be changed to :
 * A -> TransData(4->5, fp16) -> Cast(5HD, fp16->bool) -> B (5HD, bool)
 */
TEST_F(UTEST_FE_TRANSOP_INSERT_COMPLEX_2, SwitchCastAndTransdata_4) {
  ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::GeShape original_shape = ge::GeShape({3, 12, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NHWC, ge::DT_FLOAT,
                        original_shape);
  test.AddOpDesc("a", "A", 1, 1)
      .AddOpDesc("transdata", fe::TRANSDATA, 1, 1)
      .AddOpDesc("cast", fe::CAST, 1, 1)
      .AddOpDesc("b", "B", 1, 1);

  test.SetInput("cast", ge::FORMAT_NHWC, ge::DT_FLOAT16, "a", ge::FORMAT_NHWC, ge::DT_FLOAT16)
      .SetInput("transdata", ge::FORMAT_NHWC, ge::DT_BOOL, "cast", ge::FORMAT_NHWC, ge::DT_BOOL)
      .SetInput("b", ge::FORMAT_NC1HWC0, ge::DT_BOOL, "transdata", ge::FORMAT_NC1HWC0,  ge::DT_BOOL);
  test.DumpGraph(graph);

  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status ret = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));
  EXPECT_EQ(ret, fe::SUCCESS);

  FusionRuleManagerPtr fusion_rule_mgr_ptr = make_shared<FusionRuleManager>(fe_ops_kernel_info_store_ptr_);
  auto graph_fusion_ptr = std::make_shared<GraphFusion>(fusion_rule_mgr_ptr, fe_ops_kernel_info_store_ptr_,
                                                        nullptr);
  graph_fusion_ptr->SetEngineName(AI_CORE_NAME);
  ret = graph_fusion_ptr->SwitchTransDataAndCast(*(graph.get()), trans_op_insert.GetOptimizableCast());
  EXPECT_EQ(ret, fe::SUCCESS);
  test.DumpGraph(graph);
  ge::NodePtr cast;
  ge::NodePtr transdata;
  test.GetNodeByName("cast", cast);
  test.GetNodeByName("transdata", transdata);
  ASSERT_EQ(cast->GetInAllNodes().size(), 1);
  ASSERT_EQ(cast->GetInAllNodes().at(0)->GetType(), fe::TRANSDATA);
  ASSERT_EQ(transdata->GetInAllNodes().size(), 1);
  ASSERT_EQ(transdata->GetInAllNodes().at(0)->GetType(), "A");
  EXPECT_EQ(cast->GetOpDesc()->GetInputDesc(0).GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(cast->GetOpDesc()->GetOutputDesc(0).GetDataType(), ge::DT_BOOL);
  EXPECT_EQ(cast->GetOpDesc()->GetInputDesc(0).GetFormat(), ge::FORMAT_NC1HWC0);
  EXPECT_EQ(cast->GetOpDesc()->GetOutputDesc(0).GetFormat(), ge::FORMAT_NC1HWC0);

  EXPECT_EQ(transdata->GetOpDesc()->GetInputDesc(0).GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(transdata->GetOpDesc()->GetOutputDesc(0).GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(transdata->GetOpDesc()->GetInputDesc(0).GetFormat(), ge::FORMAT_NHWC);
  EXPECT_EQ(transdata->GetOpDesc()->GetOutputDesc(0).GetFormat(), ge::FORMAT_NC1HWC0);
}


/* An Special Case is :
* A -> TransData(4->5, bool) -> Cast1(5HD, bool->fp32)-> Cast2(5HD, fp32->fp16)
* -> B(5HD, fp16)
* This case will be optimized as:
* A -> TransData(4->5, bool) -> Cast(5HD, bool->fp16)-> B(5HD, fp16)
* then:
* A -> Cast(4D, bool->fp16) -> TransData(4->5, fp16) -> B(5HD, fp16)
* */
TEST_F(UTEST_FE_TRANSOP_INSERT_COMPLEX_2, SwitchCastAndTransdata_5) {
  ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::GeShape original_shape = ge::GeShape({3, 12, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NHWC, ge::DT_FLOAT,
                        original_shape);
  test.AddOpDesc("a", "A", 1, 1)
      .AddOpDesc("transdata", fe::TRANSDATA, 1, 1)
      .AddOpDesc("cast1", fe::CAST, 1, 1)
      .AddOpDesc("cast2", fe::CAST, 1, 1)
      .AddOpDesc("b", "B", 1, 1);

  test.SetInput("transdata", ge::FORMAT_NHWC, ge::DT_BOOL, "a", ge::FORMAT_NHWC, ge::DT_BOOL)
      .SetInput("cast1", ge::FORMAT_NC1HWC0, ge::DT_BOOL, "transdata", ge::FORMAT_NC1HWC0, ge::DT_BOOL)
      .SetInput("cast2", ge::FORMAT_NC1HWC0, ge::DT_FLOAT, "cast1", ge::FORMAT_NC1HWC0,  ge::DT_FLOAT)
      .SetInput("b", ge::FORMAT_NC1HWC0, ge::DT_FLOAT16, "cast2", ge::FORMAT_NC1HWC0,  ge::DT_FLOAT16);
  test.DumpGraph(graph);

  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status ret = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));
  EXPECT_EQ(ret, fe::SUCCESS);

  FusionRuleManagerPtr fusion_rule_mgr_ptr = make_shared<FusionRuleManager>(fe_ops_kernel_info_store_ptr_);
  auto graph_fusion_ptr = std::make_shared<GraphFusion>(fusion_rule_mgr_ptr, fe_ops_kernel_info_store_ptr_,
                                                        nullptr);
  graph_fusion_ptr->SetEngineName(AI_CORE_NAME);
  ret = graph_fusion_ptr->SwitchTransDataAndCast(*(graph.get()), trans_op_insert.GetOptimizableCast());
  EXPECT_EQ(ret, fe::SUCCESS);
  test.DumpGraph(graph);
  ge::NodePtr cast1;
  ge::NodePtr cast2;
  ge::NodePtr transdata;
  test.GetNodeByName("cast1", cast1);
  test.GetNodeByName("cast2", cast2);
  test.GetNodeByName("transdata", transdata);
  ASSERT_EQ(cast1->GetInAllNodes().size(), 1);
  ASSERT_EQ(cast1->GetInAllNodes().at(0)->GetType(), "A");
  ASSERT_EQ(transdata->GetInAllNodes().size(), 1);
  ASSERT_EQ(transdata->GetInAllNodes().at(0)->GetType(), fe::CAST);
  EXPECT_EQ(cast2, nullptr);
  EXPECT_EQ(cast1->GetOpDesc()->GetInputDesc(0).GetDataType(), ge::DT_BOOL);
  EXPECT_EQ(cast1->GetOpDesc()->GetOutputDesc(0).GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(cast1->GetOpDesc()->GetInputDesc(0).GetFormat(), ge::FORMAT_NHWC);
  EXPECT_EQ(cast1->GetOpDesc()->GetOutputDesc(0).GetFormat(), ge::FORMAT_NHWC);

  EXPECT_EQ(transdata->GetOpDesc()->GetInputDesc(0).GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(transdata->GetOpDesc()->GetOutputDesc(0).GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(transdata->GetOpDesc()->GetInputDesc(0).GetFormat(), ge::FORMAT_NHWC);
  EXPECT_EQ(transdata->GetOpDesc()->GetOutputDesc(0).GetFormat(), ge::FORMAT_NC1HWC0);
}

/* An Special Case is :
* A -> TransData(4->5, bool) -> Cast1(5HD, bool->fp32)-> Cast2(5HD, fp32->fp16) -> B(5HD, fp16)
*                                                    \-> Cast3(5HD, fp32->fp16) -> C(5HD, fp16)
 *                                                   \-> Cast4(5HD, fp32->fp16) -> D(5HD, fp16)
* This case will be optimized as:
* A -> Cast1(4D, bool->fp16)-> TransData(4->5, 16) ----> B(5HD, fp16)
*                                                    \-> C(5HD, fp16)
 *                                                   \-> D(5HD, fp16)
* */
TEST_F(UTEST_FE_TRANSOP_INSERT_COMPLEX_2, SwitchCastAndTransdata_6) {
  ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::GeShape original_shape = ge::GeShape({3, 12, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NHWC, ge::DT_FLOAT,
                        original_shape);
  test.SetInput("TransData", ge::FORMAT_NHWC, ge::DT_BOOL, "a", ge::FORMAT_NHWC, ge::DT_BOOL)
      .SetInput("Cast_1", ge::FORMAT_NC1HWC0, ge::DT_BOOL, "TransData", ge::FORMAT_NC1HWC0, ge::DT_BOOL)
      .SetInput("Cast_2", ge::FORMAT_NC1HWC0, ge::DT_FLOAT, "Cast_1:0", ge::FORMAT_NC1HWC0,  ge::DT_FLOAT)
      .SetInput("Cast_3", ge::FORMAT_NC1HWC0, ge::DT_FLOAT, "Cast_1:0", ge::FORMAT_NC1HWC0,  ge::DT_FLOAT)
      .SetInput("Cast_4", ge::FORMAT_NC1HWC0, ge::DT_FLOAT, "Cast_1:0", ge::FORMAT_NC1HWC0,  ge::DT_FLOAT)

      .SetInput("b", ge::FORMAT_NC1HWC0, ge::DT_FLOAT16, "Cast_2", ge::FORMAT_NC1HWC0,  ge::DT_FLOAT16)
      .SetInput("c", ge::FORMAT_NC1HWC0, ge::DT_FLOAT16, "Cast_3", ge::FORMAT_NC1HWC0,  ge::DT_FLOAT16)
      .SetInput("d", ge::FORMAT_NC1HWC0, ge::DT_FLOAT16, "Cast_4", ge::FORMAT_NC1HWC0,  ge::DT_FLOAT16);

  test.DumpGraph(graph);

  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status ret = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));
  EXPECT_EQ(ret, fe::SUCCESS);

  FusionRuleManagerPtr fusion_rule_mgr_ptr = make_shared<FusionRuleManager>(fe_ops_kernel_info_store_ptr_);
  auto graph_fusion_ptr = std::make_shared<GraphFusion>(fusion_rule_mgr_ptr, fe_ops_kernel_info_store_ptr_,
                                                        nullptr);
  graph_fusion_ptr->SetEngineName(AI_CORE_NAME);
  ret = graph_fusion_ptr->SwitchTransDataAndCast(*(graph.get()), trans_op_insert.GetOptimizableCast());
  EXPECT_EQ(ret, fe::SUCCESS);
  test.DumpGraph(graph);
  ge::NodePtr cast1;
  ge::NodePtr cast2;
  ge::NodePtr cast3;
  ge::NodePtr cast4;
  ge::NodePtr transdata;
  test.GetNodeByName("Cast_1", cast1);
  test.GetNodeByName("Cast_2", cast2);
  test.GetNodeByName("Cast_3", cast3);
  test.GetNodeByName("Cast_4", cast4);

  test.GetNodeByName("TransData", transdata);
  EXPECT_EQ(cast2, nullptr);
  EXPECT_EQ(cast3, nullptr);
  EXPECT_EQ(cast4, nullptr);
  EXPECT_EQ(cast1->GetOutAllNodes().size(), 1);
  EXPECT_EQ(transdata->GetOutAllNodes().size(), 3);

  for (auto bcd : transdata->GetOutAllNodes()) {
    EXPECT_NE(bcd->GetType(), fe::CAST);
  }

  EXPECT_EQ(cast1->GetOpDesc()->GetInputDesc(0).GetDataType(), ge::DT_BOOL);
  EXPECT_EQ(cast1->GetOpDesc()->GetOutputDesc(0).GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(cast1->GetOpDesc()->GetInputDesc(0).GetFormat(), ge::FORMAT_NHWC);
  EXPECT_EQ(cast1->GetOpDesc()->GetOutputDesc(0).GetFormat(), ge::FORMAT_NHWC);

  EXPECT_EQ(transdata->GetOpDesc()->GetInputDesc(0).GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(transdata->GetOpDesc()->GetOutputDesc(0).GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(transdata->GetOpDesc()->GetInputDesc(0).GetFormat(), ge::FORMAT_NHWC);
  EXPECT_EQ(transdata->GetOpDesc()->GetOutputDesc(0).GetFormat(), ge::FORMAT_NC1HWC0);
}

/* An Special Case is :
* A -> TransData(4->5, bool) -> Cast1(5HD, bool->fp32)-> Cast2(5HD, fp32->int32) -> B(5HD, int32)
*                                                    \-> Cast3(5HD, fp32->int32) -> C(5HD, int32)
*                                                    \-> Cast4(5HD, fp32->int32) -> D(5HD, int32)
* This case will be optimized to:
* A -> Cast1(4D, bool->fp32) -> TransData(4->5, fp32)-> Cast2(5HD, fp32->int32) -> B(5HD, int32)
*                                                    \-> Cast3(5HD, fp32->int32) -> C(5HD, int32)
*                                                    \-> Cast4(5HD, fp32->int32) -> D(5HD, int32)
* */

TEST_F(UTEST_FE_TRANSOP_INSERT_COMPLEX_2, SwitchCastAndTransdata_7) {
  ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::GeShape original_shape = ge::GeShape({3, 12, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NHWC, ge::DT_FLOAT,
                        original_shape);

  test.SetInput("TransData", ge::FORMAT_NHWC, ge::DT_BOOL, "a", ge::FORMAT_NHWC, ge::DT_BOOL)
      .SetInput("Cast_1", ge::FORMAT_NC1HWC0, ge::DT_BOOL, "TransData", ge::FORMAT_NC1HWC0, ge::DT_BOOL)
      .SetInput("Cast_2", ge::FORMAT_NC1HWC0, ge::DT_FLOAT, "Cast_1:0", ge::FORMAT_NC1HWC0,  ge::DT_FLOAT)
      .SetInput("Cast_3", ge::FORMAT_NC1HWC0, ge::DT_FLOAT, "Cast_1:0", ge::FORMAT_NC1HWC0,  ge::DT_FLOAT)
      .SetInput("Cast_4", ge::FORMAT_NC1HWC0, ge::DT_FLOAT, "Cast_1:0", ge::FORMAT_NC1HWC0,  ge::DT_FLOAT)

      .SetInput("b", ge::FORMAT_NC1HWC0, ge::DT_INT32, "Cast_2", ge::FORMAT_NC1HWC0,  ge::DT_INT32)
      .SetInput("c", ge::FORMAT_NC1HWC0, ge::DT_INT32, "Cast_3", ge::FORMAT_NC1HWC0,  ge::DT_INT32)
      .SetInput("d", ge::FORMAT_NC1HWC0, ge::DT_INT32, "Cast_4", ge::FORMAT_NC1HWC0,  ge::DT_INT32);

  test.DumpGraph(graph);

  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status ret = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));
  EXPECT_EQ(ret, fe::SUCCESS);

  FusionRuleManagerPtr fusion_rule_mgr_ptr = make_shared<FusionRuleManager>(fe_ops_kernel_info_store_ptr_);
  auto graph_fusion_ptr = std::make_shared<GraphFusion>(fusion_rule_mgr_ptr, fe_ops_kernel_info_store_ptr_,
                                                        nullptr);
  graph_fusion_ptr->SetEngineName(AI_CORE_NAME);
  ret = graph_fusion_ptr->SwitchTransDataAndCast(*(graph.get()), trans_op_insert.GetOptimizableCast());
  EXPECT_EQ(ret, fe::SUCCESS);
  test.DumpGraph(graph);
  ge::NodePtr cast1;
  ge::NodePtr cast2;
  ge::NodePtr cast3;
  ge::NodePtr cast4;
  ge::NodePtr transdata;
  test.GetNodeByName("Cast_1", cast1);
  test.GetNodeByName("Cast_2", cast2);
  test.GetNodeByName("Cast_3", cast3);
  test.GetNodeByName("Cast_4", cast4);

  test.GetNodeByName("TransData", transdata);
  EXPECT_NE(cast2, nullptr);
  EXPECT_NE(cast3, nullptr);
  EXPECT_NE(cast4, nullptr);
  EXPECT_EQ(cast1->GetOutAllNodes().size(), 1);
  EXPECT_EQ(transdata->GetOutAllNodes().size(), 3);

  for (auto cast234 : transdata->GetOutAllNodes()) {
    EXPECT_EQ(cast234->GetType(), fe::CAST);
  }

  EXPECT_EQ(cast1->GetOpDesc()->GetInputDesc(0).GetDataType(), ge::DT_BOOL);
  EXPECT_EQ(cast1->GetOpDesc()->GetOutputDesc(0).GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(cast1->GetOpDesc()->GetInputDesc(0).GetFormat(), ge::FORMAT_NHWC);
  EXPECT_EQ(cast1->GetOpDesc()->GetOutputDesc(0).GetFormat(), ge::FORMAT_NHWC);

  EXPECT_EQ(transdata->GetOpDesc()->GetInputDesc(0).GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(transdata->GetOpDesc()->GetOutputDesc(0).GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(transdata->GetOpDesc()->GetInputDesc(0).GetFormat(), ge::FORMAT_NHWC);
  EXPECT_EQ(transdata->GetOpDesc()->GetOutputDesc(0).GetFormat(), ge::FORMAT_NC1HWC0);
}

/* An Special Case is :
 * A -> TransData(4->5, bool) -> Cast1(5HD, bool->fp32)-> Cast2(5HD, fp32->fp16) -> B(5HD, fp16)
 *                                                                              \-> C(5HD, fp16)
 *                                                    \-> Cast3(5HD, fp32->fp16) -> D(5HD, fp16)
 *                                                                              \-> E(5HD, fp16)
 * This case will be optimized as:
 * A -> Cast1(4D, bool->fp16)-> TransData(4->5, 16) ----> B(5HD, fp16)
 *                                                    \-> C(5HD, fp16)
 *                                                    \-> D(5HD, fp16)
 *                                                    \-> E(5HD, fp16)
 */
TEST_F(UTEST_FE_TRANSOP_INSERT_COMPLEX_2, SwitchCastAndTransdata_8) {
  ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::GeShape original_shape = ge::GeShape({3, 12, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NHWC, ge::DT_FLOAT,
                        original_shape);

  test.SetInput("TransData", ge::FORMAT_NHWC, ge::DT_BOOL, "a", ge::FORMAT_NHWC, ge::DT_BOOL)
      .SetInput("Cast_1", ge::FORMAT_NC1HWC0, ge::DT_BOOL, "TransData", ge::FORMAT_NC1HWC0, ge::DT_BOOL)
      .SetInput("Cast_2", ge::FORMAT_NC1HWC0, ge::DT_FLOAT, "Cast_1:0", ge::FORMAT_NC1HWC0,  ge::DT_FLOAT)
      .SetInput("Cast_3", ge::FORMAT_NC1HWC0, ge::DT_FLOAT, "Cast_1:0", ge::FORMAT_NC1HWC0,  ge::DT_FLOAT)
      .SetInput("Cast_4", ge::FORMAT_NC1HWC0, ge::DT_FLOAT, "Cast_1:0", ge::FORMAT_NC1HWC0,  ge::DT_FLOAT)

      .SetInput("b", ge::FORMAT_NC1HWC0, ge::DT_FLOAT16, "Cast_2", ge::FORMAT_NC1HWC0,  ge::DT_FLOAT16)
      .SetInput("c", ge::FORMAT_NC1HWC0, ge::DT_FLOAT16, "Cast_3", ge::FORMAT_NC1HWC0,  ge::DT_FLOAT16)
      .SetInput("d", ge::FORMAT_NC1HWC0, ge::DT_FLOAT16, "Cast_4", ge::FORMAT_NC1HWC0,  ge::DT_FLOAT16);

  test.DumpGraph(graph);

  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status ret = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));
  EXPECT_EQ(ret, fe::SUCCESS);

  FusionRuleManagerPtr fusion_rule_mgr_ptr = make_shared<FusionRuleManager>(fe_ops_kernel_info_store_ptr_);
  auto graph_fusion_ptr = std::make_shared<GraphFusion>(fusion_rule_mgr_ptr, fe_ops_kernel_info_store_ptr_,
                                                        nullptr);
  graph_fusion_ptr->SetEngineName(AI_CORE_NAME);
  ret = graph_fusion_ptr->SwitchTransDataAndCast(*(graph.get()), trans_op_insert.GetOptimizableCast());
  EXPECT_EQ(ret, fe::SUCCESS);
  test.DumpGraph(graph);
  ge::NodePtr cast1;
  ge::NodePtr cast2;
  ge::NodePtr cast3;
  ge::NodePtr cast4;
  ge::NodePtr transdata;
  test.GetNodeByName("Cast_1", cast1);
  test.GetNodeByName("Cast_2", cast2);
  test.GetNodeByName("Cast_3", cast3);
  test.GetNodeByName("Cast_4", cast4);

  test.GetNodeByName("TransData", transdata);
  EXPECT_EQ(cast2, nullptr);
  EXPECT_EQ(cast3, nullptr);
  EXPECT_EQ(cast4, nullptr);
  EXPECT_EQ(cast1->GetOutAllNodes().size(), 1);
  EXPECT_EQ(transdata->GetOutAllNodes().size(), 3);

  for (auto bcd : transdata->GetOutAllNodes()) {
    EXPECT_NE(bcd->GetType(), fe::CAST);
  }

  EXPECT_EQ(cast1->GetOpDesc()->GetInputDesc(0).GetDataType(), ge::DT_BOOL);
  EXPECT_EQ(cast1->GetOpDesc()->GetOutputDesc(0).GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(cast1->GetOpDesc()->GetInputDesc(0).GetFormat(), ge::FORMAT_NHWC);
  EXPECT_EQ(cast1->GetOpDesc()->GetOutputDesc(0).GetFormat(), ge::FORMAT_NHWC);
  int64_t dst_type = -1;
  ge::AttrUtils::GetInt(cast1->GetOpDesc(), ge::CAST_ATTR_DST_TYPE, dst_type);
  EXPECT_EQ(dst_type, 1);
  EXPECT_EQ(transdata->GetOpDesc()->GetInputDesc(0).GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(transdata->GetOpDesc()->GetOutputDesc(0).GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(transdata->GetOpDesc()->GetInputDesc(0).GetFormat(), ge::FORMAT_NHWC);
  EXPECT_EQ(transdata->GetOpDesc()->GetOutputDesc(0).GetFormat(), ge::FORMAT_NC1HWC0);
}

/* Another Special Case is :
* Fuse two concecutive cast if they meet the following pattern
* ----> Cast2(16 -> fp32) ----> Cast1 (fp32 -> x) ----> TransData ---->
* This case will become:
* ----> Cast1(16 -> x) ----> TransData ---->
* then:
* ----> TransData (5HD -> 4D, fp16)----> Cast1(4D, 16 -> x) ---->
* */
TEST_F(UTEST_FE_TRANSOP_INSERT_COMPLEX_2, SwitchCastAndTransdata_9) {
  ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::GeShape original_shape = ge::GeShape({3, 12, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NHWC, ge::DT_FLOAT,
                        original_shape);

  test.SetInput("Cast_2", ge::FORMAT_NC1HWC0, ge::DT_FLOAT16, "a", ge::FORMAT_NC1HWC0, ge::DT_FLOAT16)
      .SetInput("Cast_1", ge::FORMAT_NC1HWC0, ge::DT_FLOAT, "Cast_2", ge::FORMAT_NC1HWC0, ge::DT_FLOAT)
      .SetInput("TransData", ge::FORMAT_NC1HWC0, ge::DT_BOOL, "Cast_1", ge::FORMAT_NC1HWC0,  ge::DT_BOOL)
      .SetInput("b", ge::FORMAT_NHWC, ge::DT_BOOL, "TransData", ge::FORMAT_NHWC,  ge::DT_BOOL);
  test.DumpGraph(graph);

  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status ret = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));
  EXPECT_EQ(ret, fe::SUCCESS);

  FusionRuleManagerPtr fusion_rule_mgr_ptr = make_shared<FusionRuleManager>(fe_ops_kernel_info_store_ptr_);
  auto graph_fusion_ptr = std::make_shared<GraphFusion>(fusion_rule_mgr_ptr, fe_ops_kernel_info_store_ptr_,
                                                        nullptr);
  graph_fusion_ptr->SetEngineName(AI_CORE_NAME);
  ret = graph_fusion_ptr->SwitchTransDataAndCast(*(graph.get()), trans_op_insert.GetOptimizableCast());
  EXPECT_EQ(ret, fe::SUCCESS);
  test.DumpGraph(graph);
  ge::NodePtr cast1;
  ge::NodePtr cast2;
  ge::NodePtr transdata;
  test.GetNodeByName("Cast_1", cast1);
  test.GetNodeByName("Cast_2", cast2);
  test.GetNodeByName("TransData", transdata);
  ASSERT_EQ(cast1->GetInAllNodes().size(), 1);
  ASSERT_EQ(cast1->GetInAllNodes().at(0)->GetType(), fe::TRANSDATA);
  ASSERT_EQ(transdata->GetInAllNodes().size(), 1);
  ASSERT_EQ(transdata->GetInAllNodes().at(0)->GetType(), "a");
  EXPECT_EQ(cast2, nullptr);
  EXPECT_EQ(cast1->GetOpDesc()->GetInputDesc(0).GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(cast1->GetOpDesc()->GetOutputDesc(0).GetDataType(), ge::DT_BOOL);
  EXPECT_EQ(cast1->GetOpDesc()->GetInputDesc(0).GetFormat(), ge::FORMAT_NHWC);
  EXPECT_EQ(cast1->GetOpDesc()->GetOutputDesc(0).GetFormat(), ge::FORMAT_NHWC);

  EXPECT_EQ(transdata->GetOpDesc()->GetInputDesc(0).GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(transdata->GetOpDesc()->GetOutputDesc(0).GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(transdata->GetOpDesc()->GetInputDesc(0).GetFormat(), ge::FORMAT_NC1HWC0);
  EXPECT_EQ(transdata->GetOpDesc()->GetOutputDesc(0).GetFormat(), ge::FORMAT_NHWC);
}


/* Another Special Case is :
 * Fuse two concecutive cast if they meet the following pattern
 * ----> Cast2(16 -> fp32) ----> Cast1 (fp32 -> x) ----> TransData1 ---->
 *                         \---> Cast3 (fp32 -> x) ----> TransData2 ---->
 * This case will be optimized to :
 *  * ----> Cast2(16 -> fp32) ----> TransData1(5->4) ----> Cast1(fp32 -> x) ---->
 *                            \---> TransData2(5->4) ----> Cast2(fp32 -> x) ---->
 * */
TEST_F(UTEST_FE_TRANSOP_INSERT_COMPLEX_2, SwitchCastAndTransdata_10) {
  ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::GeShape original_shape = ge::GeShape({3, 12, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NHWC, ge::DT_FLOAT,
                        original_shape);

  test.SetInput("Cast_2", ge::FORMAT_NHWC, ge::DT_FLOAT16, "a", ge::FORMAT_NHWC, ge::DT_FLOAT16)
      .SetInput("Cast_1", ge::FORMAT_NC1HWC0, ge::DT_FLOAT, "Cast_2:0", ge::FORMAT_NC1HWC0, ge::DT_FLOAT)
      .SetInput("Cast_3", ge::FORMAT_NC1HWC0, ge::DT_FLOAT, "Cast_2:0", ge::FORMAT_NC1HWC0,  ge::DT_FLOAT)
      .SetInput("TransData_1", ge::FORMAT_NC1HWC0, ge::DT_BOOL, "Cast_1", ge::FORMAT_NC1HWC0,  ge::DT_BOOL)
      .SetInput("TransData_2", ge::FORMAT_NC1HWC0, ge::DT_BOOL, "Cast_3", ge::FORMAT_NC1HWC0,  ge::DT_BOOL)
      .SetInput("b", ge::FORMAT_NHWC, ge::DT_BOOL, "TransData_1", ge::FORMAT_NHWC,  ge::DT_BOOL)
      .SetInput("c", ge::FORMAT_NHWC, ge::DT_BOOL, "TransData_2", ge::FORMAT_NHWC,  ge::DT_BOOL);

  test.DumpGraph(graph);

  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status ret = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));
  EXPECT_EQ(ret, fe::SUCCESS);

  FusionRuleManagerPtr fusion_rule_mgr_ptr = make_shared<FusionRuleManager>(fe_ops_kernel_info_store_ptr_);
  auto graph_fusion_ptr = std::make_shared<GraphFusion>(fusion_rule_mgr_ptr, fe_ops_kernel_info_store_ptr_,
                                                        nullptr);
  graph_fusion_ptr->SetEngineName(AI_CORE_NAME);
  ret = graph_fusion_ptr->SwitchTransDataAndCast(*(graph.get()), trans_op_insert.GetOptimizableCast());
  EXPECT_EQ(ret, fe::SUCCESS);
  test.DumpGraph(graph);
  ge::NodePtr cast1;
  ge::NodePtr cast2;
  ge::NodePtr cast3;
  ge::NodePtr transdata;
  test.GetNodeByName("Cast_1", cast1);
  test.GetNodeByName("Cast_2", cast2);
  test.GetNodeByName("Cast_3", cast3);
  test.GetNodeByName("TransData_1", transdata);
  EXPECT_NE(cast2, nullptr);
  EXPECT_NE(cast3, nullptr);

  EXPECT_EQ(cast1->GetOpDesc()->GetInputDesc(0).GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(cast1->GetOpDesc()->GetOutputDesc(0).GetDataType(), ge::DT_BOOL);
  EXPECT_EQ(cast1->GetOpDesc()->GetInputDesc(0).GetFormat(), ge::FORMAT_NHWC);
  EXPECT_EQ(cast1->GetOpDesc()->GetOutputDesc(0).GetFormat(), ge::FORMAT_NHWC);

  EXPECT_EQ(transdata->GetOpDesc()->GetInputDesc(0).GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(transdata->GetOpDesc()->GetOutputDesc(0).GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(transdata->GetOpDesc()->GetInputDesc(0).GetFormat(), ge::FORMAT_NC1HWC0);
  EXPECT_EQ(transdata->GetOpDesc()->GetOutputDesc(0).GetFormat(), ge::FORMAT_NHWC);
}


/* Another Special Case is :
 * Fuse two concecutive cast if they meet the following pattern
 * ----> Cast2(16 -> fp32) ----> Cast1 (fp32 -> Bool) ----> TransData1 (5->4, Bool)---->
 *                                                  \ ----> TransData2 (5->4, Bool)---->
 * This case will be optimized as:
 *
 * */
TEST_F(UTEST_FE_TRANSOP_INSERT_COMPLEX_2, SwitchCastAndTransdata_11) {
  ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::GeShape original_shape = ge::GeShape({3, 12, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NHWC, ge::DT_FLOAT,
                        original_shape);

  test.SetInput("Cast_2", ge::FORMAT_NHWC, ge::DT_FLOAT16, "a", ge::FORMAT_NHWC, ge::DT_FLOAT16)
      .SetInput("Cast_1", ge::FORMAT_NC1HWC0, ge::DT_FLOAT, "Cast_2", ge::FORMAT_NC1HWC0, ge::DT_FLOAT)
      .SetInput("TransData_1", ge::FORMAT_NC1HWC0, ge::DT_BOOL, "Cast_1:0", ge::FORMAT_NC1HWC0,  ge::DT_BOOL)
      .SetInput("TransData_2", ge::FORMAT_NC1HWC0, ge::DT_BOOL, "Cast_1:0", ge::FORMAT_NC1HWC0,  ge::DT_BOOL)
      .SetInput("b", ge::FORMAT_NHWC, ge::DT_BOOL, "TransData_1", ge::FORMAT_NHWC,  ge::DT_BOOL)
      .SetInput("c", ge::FORMAT_NHWC, ge::DT_BOOL, "TransData_2", ge::FORMAT_NHWC,  ge::DT_BOOL);
  test.DumpGraph(graph);

  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status ret = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));
  EXPECT_EQ(ret, fe::SUCCESS);

  FusionRuleManagerPtr fusion_rule_mgr_ptr = make_shared<FusionRuleManager>(fe_ops_kernel_info_store_ptr_);
  auto graph_fusion_ptr = std::make_shared<GraphFusion>(fusion_rule_mgr_ptr, fe_ops_kernel_info_store_ptr_,
                                                        nullptr);
  graph_fusion_ptr->SetEngineName(AI_CORE_NAME);
  ret = graph_fusion_ptr->SwitchTransDataAndCast(*(graph.get()), trans_op_insert.GetOptimizableCast());
  EXPECT_EQ(ret, fe::SUCCESS);
  test.DumpGraph(graph);

  ge::NodePtr cast1 = GraphConstructor::GetNodeByName("Cast_1", graph);
  ge::NodePtr cast2 = GraphConstructor::GetNodeByName("Cast_2", graph);
  ge::NodePtr transdata = GraphConstructor::GetNodeByName("TransData_1", graph);

  EXPECT_EQ(cast2, nullptr);
  EXPECT_EQ(cast1->GetOpDesc()->GetInputDesc(0).GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(cast1->GetOpDesc()->GetOutputDesc(0).GetDataType(), ge::DT_BOOL);
  EXPECT_EQ(cast1->GetOpDesc()->GetInputDesc(0).GetFormat(), ge::FORMAT_NHWC);
  EXPECT_EQ(cast1->GetOpDesc()->GetOutputDesc(0).GetFormat(), ge::FORMAT_NHWC);

  EXPECT_EQ(transdata->GetOpDesc()->GetInputDesc(0).GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(transdata->GetOpDesc()->GetOutputDesc(0).GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(transdata->GetOpDesc()->GetInputDesc(0).GetFormat(), ge::FORMAT_NC1HWC0);
  EXPECT_EQ(transdata->GetOpDesc()->GetOutputDesc(0).GetFormat(), ge::FORMAT_NHWC);
}


/* A->Transadata(5d->6HD, fp32)->Cast(fp32 to fp16, 6HD)-> Conv3D(fp16,6HD)
 * will be changed to :
 * A -> Cast(fp32 to fp16, 5d) -> Transadata(5d->6HD, fp16) -> Conv3D(fp16,6HD)
 */
TEST_F(UTEST_FE_TRANSOP_INSERT_COMPLEX_2, SwitchCastAndTransdata_12) {
  ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::GeShape original_shape = ge::GeShape({3, 12, 5, 6, 32});
  GraphConstructor test(graph, "", ge::FORMAT_NDHWC, ge::DT_FLOAT,
                        original_shape);
  test.AddOpDesc("a", "A", 1, 1)
      .AddOpDesc("transdata", fe::TRANSDATA, 1, 1)
      .AddOpDesc("cast", fe::CAST, 1, 1)
      .AddOpDesc("conv3d", "Conv3D", 1, 1);

  test.SetInput("transdata", ge::FORMAT_NDHWC, ge::DT_FLOAT, "a", ge::FORMAT_NDHWC, ge::DT_FLOAT)
      .SetInput("cast", ge::FORMAT_NDC1HWC0, ge::DT_FLOAT, "transdata", ge::FORMAT_NDC1HWC0, ge::DT_FLOAT)
      .SetInput("conv3d", ge::FORMAT_NDC1HWC0, ge::DT_FLOAT16, "cast", ge::FORMAT_NDC1HWC0,  ge::DT_FLOAT16);
  test.DumpGraph(graph);

  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status ret = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));
  EXPECT_EQ(ret, fe::SUCCESS);

  FusionRuleManagerPtr fusion_rule_mgr_ptr = make_shared<FusionRuleManager>(fe_ops_kernel_info_store_ptr_);
  auto graph_fusion_ptr = std::make_shared<GraphFusion>(fusion_rule_mgr_ptr, fe_ops_kernel_info_store_ptr_,
                                                        nullptr);
  graph_fusion_ptr->SetEngineName(AI_CORE_NAME);
  ret = graph_fusion_ptr->SwitchTransDataAndCast(*(graph.get()), trans_op_insert.GetOptimizableCast());
  EXPECT_EQ(ret, fe::SUCCESS);
  test.DumpGraph(graph);
  ge::NodePtr cast = GraphConstructor::GetNodeByName("cast", graph);
  ge::NodePtr transdata = GraphConstructor::GetNodeByName("transdata", graph);

  ASSERT_EQ(cast->GetInAllNodes().size(), 1);
  ASSERT_EQ(cast->GetInAllNodes().at(0)->GetType(), "A");
  EXPECT_EQ(cast->GetOpDesc()->GetInputDesc(0).GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(cast->GetOpDesc()->GetOutputDesc(0).GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(cast->GetOpDesc()->GetInputDesc(0).GetFormat(), ge::FORMAT_NDHWC);
  EXPECT_EQ(cast->GetOpDesc()->GetOutputDesc(0).GetFormat(), ge::FORMAT_NDHWC);

  ASSERT_EQ(transdata->GetInAllNodes().size(), 1);
  ASSERT_EQ(transdata->GetInAllNodes().at(0)->GetType(), fe::CAST);
  EXPECT_EQ(transdata->GetOpDesc()->GetInputDesc(0).GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(transdata->GetOpDesc()->GetOutputDesc(0).GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(transdata->GetOpDesc()->GetInputDesc(0).GetFormat(), ge::FORMAT_NDHWC);
  EXPECT_EQ(transdata->GetOpDesc()->GetOutputDesc(0).GetFormat(), ge::FORMAT_NDC1HWC0);
}

/* A(uint8, 4D) -> Cast(5HD, uint8->fp32) -> Conv2D(fp16, 5HD)
 * will be changed to :
 * A -> Cast(uint8 to fp16, 4d) -> Transdata(4d->5HD, fp16) -> Conv2D(fp16, 5HD)
 */
TEST_F(UTEST_FE_TRANSOP_INSERT_COMPLEX_2, SwitchCastAndTransdata_12_1) {
  ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::GeShape original_shape = ge::GeShape({3, 6, 6, 32});
  GraphConstructor test(graph, "", ge::FORMAT_NHWC, ge::DT_UINT8,
                        original_shape);
  test.AddOpDesc("a", "A", 1, 1)
  .AddOpDesc("cast", fe::CAST, 1, 1)
  .AddOpDesc("conv2d", "Conv2D", 1, 1);

  test.SetInput("cast", ge::FORMAT_NC1HWC0, ge::DT_UINT8, "a", ge::FORMAT_NHWC, ge::DT_UINT8)
  .SetInput("conv2d", ge::FORMAT_NC1HWC0, ge::DT_FLOAT16, "cast", ge::FORMAT_NC1HWC0, ge::DT_FLOAT);
  test.DumpGraph(graph);
  ge::NodePtr cast = GraphConstructor::GetNodeByName("cast", graph);
  ge::NodePtr conv2d = GraphConstructor::GetNodeByName("conv2d", graph);
  ge::Format format = static_cast<ge::Format>(ge::GetFormatFromC0(ge::FORMAT_NC1HWC0, 6));
  ge::Format format1 = static_cast<ge::Format>(ge::GetFormatFromC0(ge::FORMAT_NC1HWC0, 5));
  cast->GetOpDesc()->MutableInputDesc(0)->SetFormat(format);
  cast->GetOpDesc()->MutableOutputDesc(0)->SetFormat(format1);
  conv2d->GetOpDesc()->MutableInputDesc(0)->SetFormat(format1);
  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status ret = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));
  EXPECT_EQ(ret, fe::SUCCESS);
  FusionRuleManagerPtr fusion_rule_mgr_ptr = make_shared<FusionRuleManager>(fe_ops_kernel_info_store_ptr_);
  auto graph_fusion_ptr = std::make_shared<GraphFusion>(fusion_rule_mgr_ptr, fe_ops_kernel_info_store_ptr_,
                                                        nullptr);
  graph_fusion_ptr->SetEngineName(AI_CORE_NAME);
  ret = graph_fusion_ptr->SwitchTransDataAndCast(*(graph.get()), trans_op_insert.GetOptimizableCast());
  EXPECT_EQ(ret, fe::SUCCESS);
  test.DumpGraph(graph);

  ASSERT_EQ(cast->GetInAllNodes().size(), 1);
  EXPECT_EQ(cast->GetInAllNodes().at(0)->GetType(), "A");
  EXPECT_EQ(cast->GetOpDesc()->GetInputDesc(0).GetDataType(), ge::DT_UINT8);
  EXPECT_EQ(cast->GetOpDesc()->GetOutputDesc(0).GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(ge::GetPrimaryFormat(cast->GetOpDesc()->GetInputDesc(0).GetFormat()), ge::FORMAT_NHWC);
  EXPECT_EQ(ge::GetPrimaryFormat(cast->GetOpDesc()->GetOutputDesc(0).GetFormat()), ge::FORMAT_NHWC);

  ge::NodePtr transdata = cast->GetOutDataNodes().at(0);
  ASSERT_EQ(transdata->GetType(), "TransData");
  ASSERT_EQ(transdata->GetInAllNodes().size(), 1);
  ASSERT_EQ(transdata->GetInAllNodes().at(0)->GetType(), fe::CAST);
  EXPECT_EQ(transdata->GetOpDesc()->GetInputDesc(0).GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(transdata->GetOpDesc()->GetOutputDesc(0).GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(ge::GetPrimaryFormat(transdata->GetOpDesc()->GetInputDesc(0).GetFormat()), ge::FORMAT_NHWC);
  EXPECT_EQ(ge::GetPrimaryFormat(transdata->GetOpDesc()->GetOutputDesc(0).GetFormat()), ge::FORMAT_NC1HWC0);
  EXPECT_EQ(ge::GetC0Value(transdata->GetOpDesc()->GetOutputDesc(0).GetFormat()), 16);
}

/* Conv2D(fp16, 5HD) -> Cast(5HD, fp32->uint8) -> a(uint8, 4d)
 * will be changed to :
 * Conv2D(fp16, 5hd) -> Transdata(fp16, 5hd-> 4d) -> Cast(4d, uint8) -> Conv2D(fp16, 5HD)
 */
TEST_F(UTEST_FE_TRANSOP_INSERT_COMPLEX_2, SwitchCastAndTransdata_12_2) {
  ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::GeShape original_shape = ge::GeShape({3, 6, 6, 32});
  GraphConstructor test(graph, "", ge::FORMAT_NHWC, ge::DT_UINT8,
                        original_shape);
  test.AddOpDesc("a", "A", 1, 1)
  .AddOpDesc("cast", fe::CAST, 1, 1)
  .AddOpDesc("conv2d", "Conv2D", 1, 1);

  test.SetInput("cast", ge::FORMAT_NC1HWC0, ge::DT_FLOAT, "conv2d", ge::FORMAT_NC1HWC0, ge::DT_FLOAT16)
  .SetInput("a", ge::FORMAT_NHWC, ge::DT_UINT8, "cast", ge::FORMAT_NC1HWC0, ge::DT_UINT8);
  test.DumpGraph(graph);
  ge::NodePtr cast = GraphConstructor::GetNodeByName("cast", graph);
  ge::NodePtr conv2d = GraphConstructor::GetNodeByName("conv2d", graph);
  ge::Format format = static_cast<ge::Format>(ge::GetFormatFromC0(ge::FORMAT_NC1HWC0, 6));
  ge::Format format1 = static_cast<ge::Format>(ge::GetFormatFromC0(ge::FORMAT_NC1HWC0, 5));
  cast->GetOpDesc()->MutableInputDesc(0)->SetFormat(format1);
  cast->GetOpDesc()->MutableOutputDesc(0)->SetFormat(format);
  conv2d->GetOpDesc()->MutableOutputDesc(0)->SetFormat(format1);
  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status ret = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));
  EXPECT_EQ(ret, fe::SUCCESS);
  FusionRuleManagerPtr fusion_rule_mgr_ptr = make_shared<FusionRuleManager>(fe_ops_kernel_info_store_ptr_);
  auto graph_fusion_ptr = std::make_shared<GraphFusion>(fusion_rule_mgr_ptr, fe_ops_kernel_info_store_ptr_,
                                                        nullptr);
  graph_fusion_ptr->SetEngineName(AI_CORE_NAME);
  ret = graph_fusion_ptr->SwitchTransDataAndCast(*(graph.get()), trans_op_insert.GetOptimizableCast());
  EXPECT_EQ(ret, fe::SUCCESS);
  test.DumpGraph(graph);

  ASSERT_EQ(cast->GetInAllNodes().size(), 1);
  EXPECT_EQ(cast->GetOutDataNodes().at(0)->GetType(), "A");
  EXPECT_EQ(cast->GetOpDesc()->GetOutputDesc(0).GetDataType(), ge::DT_UINT8);
  EXPECT_EQ(cast->GetOpDesc()->GetInputDesc(0).GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(ge::GetPrimaryFormat(cast->GetOpDesc()->GetInputDesc(0).GetFormat()), ge::FORMAT_NHWC);
  EXPECT_EQ(ge::GetPrimaryFormat(cast->GetOpDesc()->GetOutputDesc(0).GetFormat()), ge::FORMAT_NHWC);

  ge::NodePtr transdata = cast->GetInDataNodes().at(0);
  EXPECT_EQ(transdata->GetType(), "TransData");
  ASSERT_EQ(transdata->GetInAllNodes().size(), 1);
  EXPECT_EQ(transdata->GetOpDesc()->GetInputDesc(0).GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(transdata->GetOpDesc()->GetOutputDesc(0).GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(ge::GetPrimaryFormat(transdata->GetOpDesc()->GetOutputDesc(0).GetFormat()), ge::FORMAT_NHWC);
  EXPECT_EQ(ge::GetPrimaryFormat(transdata->GetOpDesc()->GetInputDesc(0).GetFormat()), ge::FORMAT_NC1HWC0);
  EXPECT_EQ(ge::GetC0Value(transdata->GetOpDesc()->GetInputDesc(0).GetFormat()), 16);
}

/* A-> Cast(fp16 to fp32 NDC1HWC0 -> Transadata(NDC1HWC0 to NDHWC fp32) -> B (fp32, NDHWC)
 * will be changed to :
 * A -> Transadata(NDC1HWC0 to NDHWC fp16) -> Cast(fp16 to fp32 NDHWC) -> B(fp32,NDHWC)
 */
TEST_F(UTEST_FE_TRANSOP_INSERT_COMPLEX_2, SwitchCastAndTransdata_13) {
  ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::GeShape original_shape = ge::GeShape({3, 12, 5, 6, 32});
  GraphConstructor test(graph, "", ge::FORMAT_NDHWC, ge::DT_FLOAT,
                        original_shape);
  test.AddOpDesc("a", "A", 1, 1)
      .AddOpDesc("transdata", fe::TRANSDATA, 1, 1)
      .AddOpDesc("cast", fe::CAST, 1, 1)
      .AddOpDesc("b", "B", 1, 1);

  test.SetInput("cast", ge::FORMAT_NDC1HWC0, ge::DT_FLOAT16, "a", ge::FORMAT_NDC1HWC0, ge::DT_FLOAT16)
      .SetInput("transdata", ge::FORMAT_NDC1HWC0, ge::DT_FLOAT, "cast", ge::FORMAT_NDC1HWC0, ge::DT_FLOAT)
      .SetInput("b", ge::FORMAT_NDHWC, ge::DT_FLOAT, "transdata", ge::FORMAT_NDHWC,  ge::DT_FLOAT);
  test.DumpGraph(graph);

  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status ret = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));
  EXPECT_EQ(ret, fe::SUCCESS);

  FusionRuleManagerPtr fusion_rule_mgr_ptr = make_shared<FusionRuleManager>(fe_ops_kernel_info_store_ptr_);
  auto graph_fusion_ptr = std::make_shared<GraphFusion>(fusion_rule_mgr_ptr, fe_ops_kernel_info_store_ptr_,
                                                        nullptr);
  graph_fusion_ptr->SetEngineName(AI_CORE_NAME);
  ret = graph_fusion_ptr->SwitchTransDataAndCast(*(graph.get()), trans_op_insert.GetOptimizableCast());
  EXPECT_EQ(ret, fe::SUCCESS);
  test.DumpGraph(graph);
  ge::NodePtr cast = GraphConstructor::GetNodeByName("cast", graph);
  ge::NodePtr transdata = GraphConstructor::GetNodeByName("transdata", graph);

  ASSERT_EQ(cast->GetInAllNodes().size(), 1);
  ASSERT_EQ(cast->GetInAllNodes().at(0)->GetType(), fe::TRANSDATA);
  EXPECT_EQ(cast->GetOpDesc()->GetInputDesc(0).GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(cast->GetOpDesc()->GetOutputDesc(0).GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(cast->GetOpDesc()->GetInputDesc(0).GetFormat(), ge::FORMAT_NDHWC);
  EXPECT_EQ(cast->GetOpDesc()->GetOutputDesc(0).GetFormat(), ge::FORMAT_NDHWC);

  ASSERT_EQ(transdata->GetInAllNodes().size(), 1);
  ASSERT_EQ(transdata->GetInAllNodes().at(0)->GetType(), "A");
  EXPECT_EQ(transdata->GetOpDesc()->GetInputDesc(0).GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(transdata->GetOpDesc()->GetOutputDesc(0).GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(transdata->GetOpDesc()->GetInputDesc(0).GetFormat(), ge::FORMAT_NDC1HWC0);
  EXPECT_EQ(transdata->GetOpDesc()->GetOutputDesc(0).GetFormat(), ge::FORMAT_NDHWC);
}

/* A-> Cast(fp16 to fp32 NDC1HWC0 -> Transadata1(NDC1HWC0 to NDHWC fp32) -> B (fp32, NDHWC)
 *                                -> Transadata2(NDC1HWC0 to NDHWC fp32) -> C (fp32, NDHWC)
 *                                -> Transadata3(NDC1HWC0 to NDHWC fp32) -> D (fp32, NDHWC)
 * will be changed to :
 * A -> Transadata(NDC1HWC0 to NDHWC fp16) -> Cast(fp16 to fp32 NDHWC) -> B (fp32, NDHWC)
 *                                                                     -> C (fp32, NDHWC)
 *                                                                     -> D (fp32, NDHWC)
 * Transdata merged
 */
TEST_F(UTEST_FE_TRANSOP_INSERT_COMPLEX_2, SwitchCastAndTransdata_14) {
  ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::GeShape original_shape = ge::GeShape({3, 12, 5, 6, 32});
  GraphConstructor test(graph, "", ge::FORMAT_NDHWC, ge::DT_FLOAT,
                        original_shape);
  test.AddOpDesc("a", "A", 1, 1)
      .AddOpDesc("transdata1", fe::TRANSDATA, 1, 1)
      .AddOpDesc("transdata2", fe::TRANSDATA, 1, 1)
      .AddOpDesc("transdata3", fe::TRANSDATA, 1, 1)
      .AddOpDesc("cast", fe::CAST, 1, 1)
      .AddOpDesc("b", "B", 1, 1)
      .AddOpDesc("c", "C", 1, 1)
      .AddOpDesc("d", "D", 1, 1);

  test.SetInput("cast", ge::FORMAT_NDC1HWC0, ge::DT_FLOAT16, "a", ge::FORMAT_NDC1HWC0, ge::DT_FLOAT16)
      .SetInput("transdata1", ge::FORMAT_NDC1HWC0, ge::DT_FLOAT, "cast:0", ge::FORMAT_NDC1HWC0, ge::DT_FLOAT)
      .SetInput("transdata2", ge::FORMAT_NDC1HWC0, ge::DT_FLOAT, "cast:0", ge::FORMAT_NDC1HWC0, ge::DT_FLOAT)
      .SetInput("transdata3", ge::FORMAT_NDC1HWC0, ge::DT_FLOAT, "cast:0", ge::FORMAT_NDC1HWC0, ge::DT_FLOAT)
      .SetInput("b", ge::FORMAT_NDHWC, ge::DT_FLOAT, "transdata1", ge::FORMAT_NDHWC,  ge::DT_FLOAT)
      .SetInput("c", ge::FORMAT_NDHWC, ge::DT_FLOAT, "transdata2", ge::FORMAT_NDHWC,  ge::DT_FLOAT)
      .SetInput("d", ge::FORMAT_NDHWC, ge::DT_FLOAT, "transdata3", ge::FORMAT_NDHWC,  ge::DT_FLOAT);
  test.DumpGraph(graph);

  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status ret = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));
  EXPECT_EQ(ret, fe::SUCCESS);

  FusionRuleManagerPtr fusion_rule_mgr_ptr = make_shared<FusionRuleManager>(fe_ops_kernel_info_store_ptr_);
  auto graph_fusion_ptr = std::make_shared<GraphFusion>(fusion_rule_mgr_ptr, fe_ops_kernel_info_store_ptr_,
                                                        nullptr);
  graph_fusion_ptr->SetEngineName(AI_CORE_NAME);
  ret = graph_fusion_ptr->SwitchTransDataAndCast(*(graph.get()), trans_op_insert.GetOptimizableCast());
  EXPECT_EQ(ret, fe::SUCCESS);
  test.DumpGraph(graph);
  ge::NodePtr cast = GraphConstructor::GetNodeByName("cast", graph);
  ge::NodePtr transdata1 = GraphConstructor::GetNodeByName("transdata1", graph);
  ge::NodePtr transdata2 = GraphConstructor::GetNodeByName("transdata2", graph);
  ge::NodePtr transdata3 = GraphConstructor::GetNodeByName("transdata3", graph);

  ASSERT_EQ(cast->GetInAllNodes().size(), 1);
  ASSERT_EQ(cast->GetInAllNodes().at(0)->GetName(), "transdata1");
  EXPECT_EQ(cast->GetOpDesc()->GetInputDesc(0).GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(cast->GetOpDesc()->GetOutputDesc(0).GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(cast->GetOpDesc()->GetInputDesc(0).GetFormat(), ge::FORMAT_NDHWC);
  EXPECT_EQ(cast->GetOpDesc()->GetOutputDesc(0).GetFormat(), ge::FORMAT_NDHWC);

  ASSERT_EQ(cast->GetOutAllNodes().size(), 3);
  ASSERT_EQ(cast->GetOutAllNodes().at(0)->GetName(), "b");
  ASSERT_EQ(cast->GetOutAllNodes().at(1)->GetName(), "c");
  ASSERT_EQ(cast->GetOutAllNodes().at(2)->GetName(), "d");


  ASSERT_EQ(transdata1->GetInAllNodes().size(), 1);
  ASSERT_EQ(transdata1->GetInAllNodes().at(0)->GetType(), "A");
  EXPECT_EQ(transdata1->GetOpDesc()->GetInputDesc(0).GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(transdata1->GetOpDesc()->GetOutputDesc(0).GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(transdata1->GetOpDesc()->GetInputDesc(0).GetFormat(), ge::FORMAT_NDC1HWC0);
  EXPECT_EQ(transdata1->GetOpDesc()->GetOutputDesc(0).GetFormat(), ge::FORMAT_NDHWC);

  ASSERT_EQ(transdata2, nullptr);
  ASSERT_EQ(transdata3, nullptr);
}


/* A-> Cast(fp16 to fp32 NDC1HWC0 -> Transadata1(NDC1HWC0 to NDHWC fp32) -> B (fp32, NDHWC)
 *                                -> Transadata2(NDC1HWC0 to NDHWC fp32) -> C (fp32, NDHWC)
 *                                -> Transadata3(NDC1HWC0 to DHWCN fp32) -> D (fp32, DHWCN)
 * will not be optimized.
 */
TEST_F(UTEST_FE_TRANSOP_INSERT_COMPLEX_2, SwitchCastAndTransdata_15) {
  ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::GeShape original_shape = ge::GeShape({3, 12, 5, 6, 32});
  GraphConstructor test(graph, "", ge::FORMAT_NDHWC, ge::DT_FLOAT,
                        original_shape);
  test.AddOpDesc("a", "A", 1, 1)
      .AddOpDesc("transdata1", fe::TRANSDATA, 1, 1)
      .AddOpDesc("transdata2", fe::TRANSDATA, 1, 1)
      .AddOpDesc("transdata3", fe::TRANSDATA, 1, 1)
      .AddOpDesc("cast", fe::CAST, 1, 1)
      .AddOpDesc("b", "B", 1, 1)
      .AddOpDesc("c", "C", 1, 1)
      .AddOpDesc("d", "D", 1, 1);

  test.SetInput("cast", ge::FORMAT_NDC1HWC0, ge::DT_FLOAT16, "a", ge::FORMAT_NDC1HWC0, ge::DT_FLOAT16)
      .SetInput("transdata1", ge::FORMAT_NDC1HWC0, ge::DT_FLOAT, "cast:0", ge::FORMAT_NDC1HWC0, ge::DT_FLOAT)
      .SetInput("transdata2", ge::FORMAT_NDC1HWC0, ge::DT_FLOAT, "cast:0", ge::FORMAT_NDC1HWC0, ge::DT_FLOAT)
      .SetInput("transdata3", ge::FORMAT_NDC1HWC0, ge::DT_FLOAT, "cast:0", ge::FORMAT_NDC1HWC0, ge::DT_FLOAT)
      .SetInput("b", ge::FORMAT_NDHWC, ge::DT_FLOAT, "transdata1", ge::FORMAT_NDHWC,  ge::DT_FLOAT)
      .SetInput("c", ge::FORMAT_NDHWC, ge::DT_FLOAT, "transdata2", ge::FORMAT_NDHWC,  ge::DT_FLOAT)
      .SetInput("d", ge::FORMAT_DHWCN, ge::DT_FLOAT, "transdata3", ge::FORMAT_DHWCN,  ge::DT_FLOAT);
  test.DumpGraph(graph);

  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status ret = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));
  EXPECT_EQ(ret, fe::SUCCESS);

  FusionRuleManagerPtr fusion_rule_mgr_ptr = make_shared<FusionRuleManager>(fe_ops_kernel_info_store_ptr_);
  auto graph_fusion_ptr = std::make_shared<GraphFusion>(fusion_rule_mgr_ptr, fe_ops_kernel_info_store_ptr_,
                                                        nullptr);
  graph_fusion_ptr->SetEngineName(AI_CORE_NAME);
  ret = graph_fusion_ptr->SwitchTransDataAndCast(*(graph.get()), trans_op_insert.GetOptimizableCast());
  EXPECT_EQ(ret, fe::SUCCESS);
  test.DumpGraph(graph);
  ge::NodePtr cast = GraphConstructor::GetNodeByName("cast", graph);
  ge::NodePtr transdata1 = GraphConstructor::GetNodeByName("transdata1", graph);
  ge::NodePtr transdata2 = GraphConstructor::GetNodeByName("transdata2", graph);
  ge::NodePtr transdata3 = GraphConstructor::GetNodeByName("transdata3", graph);


  ASSERT_EQ(cast->GetInAllNodes().size(), 1);
  ASSERT_EQ(cast->GetInAllNodes().at(0)->GetType(), "A");
  EXPECT_EQ(cast->GetOpDesc()->GetInputDesc(0).GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(cast->GetOpDesc()->GetOutputDesc(0).GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(cast->GetOpDesc()->GetInputDesc(0).GetFormat(), ge::FORMAT_NDC1HWC0);
  EXPECT_EQ(cast->GetOpDesc()->GetOutputDesc(0).GetFormat(), ge::FORMAT_NDC1HWC0);
  ASSERT_EQ(cast->GetOutAllNodes().size(), 3);
  ASSERT_EQ(cast->GetOutAllNodes().at(0)->GetName(), "transdata1");
  ASSERT_EQ(cast->GetOutAllNodes().at(1)->GetName(), "transdata2");
  ASSERT_EQ(cast->GetOutAllNodes().at(2)->GetName(), "transdata3");

  ASSERT_EQ(transdata1->GetInAllNodes().size(), 1);
  ASSERT_EQ(transdata1->GetInAllNodes().at(0)->GetType(), fe::CAST);
  EXPECT_EQ(transdata1->GetOpDesc()->GetInputDesc(0).GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(transdata1->GetOpDesc()->GetOutputDesc(0).GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(transdata1->GetOpDesc()->GetInputDesc(0).GetFormat(), ge::FORMAT_NDC1HWC0);
  EXPECT_EQ(transdata1->GetOpDesc()->GetOutputDesc(0).GetFormat(), ge::FORMAT_NDHWC);

  ASSERT_EQ(transdata2->GetInAllNodes().size(), 1);
  ASSERT_EQ(transdata2->GetInAllNodes().at(0)->GetType(), fe::CAST);
  EXPECT_EQ(transdata2->GetOpDesc()->GetInputDesc(0).GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(transdata2->GetOpDesc()->GetOutputDesc(0).GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(transdata2->GetOpDesc()->GetInputDesc(0).GetFormat(), ge::FORMAT_NDC1HWC0);
  EXPECT_EQ(transdata2->GetOpDesc()->GetOutputDesc(0).GetFormat(), ge::FORMAT_NDHWC);

  ASSERT_EQ(transdata3->GetInAllNodes().size(), 1);
  ASSERT_EQ(transdata3->GetInAllNodes().at(0)->GetType(), fe::CAST);
  EXPECT_EQ(transdata3->GetOpDesc()->GetInputDesc(0).GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(transdata3->GetOpDesc()->GetOutputDesc(0).GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(transdata3->GetOpDesc()->GetInputDesc(0).GetFormat(), ge::FORMAT_NDC1HWC0);
  EXPECT_EQ(transdata3->GetOpDesc()->GetOutputDesc(0).GetFormat(), ge::FORMAT_DHWCN);
}

/* A-> Cast(fp16 to fp32 NDC1HWC0 ---------------------------------------> B (fp32, NDC1HWC0)
 *                                -> Transadata2(NDC1HWC0 to NDHWC fp32) -> C (fp32, NDHWC)
 *                                -> Transadata3(NDC1HWC0 to NDHWC fp32) -> D (fp32, NDHWC)
 * will not will optimized
 */
TEST_F(UTEST_FE_TRANSOP_INSERT_COMPLEX_2, SwitchCastAndTransdata_16) {
  ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::GeShape original_shape = ge::GeShape({3, 12, 5, 6, 32});
  GraphConstructor test(graph, "", ge::FORMAT_NDHWC, ge::DT_FLOAT,
                        original_shape);
  test.AddOpDesc("a", "A", 1, 1)
      .AddOpDesc("transdata2", fe::TRANSDATA, 1, 1)
      .AddOpDesc("transdata3", fe::TRANSDATA, 1, 1)
      .AddOpDesc("cast", fe::CAST, 1, 1)
      .AddOpDesc("b", "B", 1, 1)
      .AddOpDesc("c", "C", 1, 1)
      .AddOpDesc("d", "D", 1, 1);

  test.SetInput("cast", ge::FORMAT_NDC1HWC0, ge::DT_FLOAT16, "a", ge::FORMAT_NDC1HWC0, ge::DT_FLOAT16)
      .SetInput("b", ge::FORMAT_NDC1HWC0, ge::DT_FLOAT, "cast:0", ge::FORMAT_NDC1HWC0, ge::DT_FLOAT)
      .SetInput("transdata2", ge::FORMAT_NDC1HWC0, ge::DT_FLOAT, "cast:0", ge::FORMAT_NDC1HWC0, ge::DT_FLOAT)
      .SetInput("transdata3", ge::FORMAT_NDC1HWC0, ge::DT_FLOAT, "cast:0", ge::FORMAT_NDC1HWC0, ge::DT_FLOAT)
      .SetInput("d", ge::FORMAT_NDHWC, ge::DT_FLOAT, "transdata2", ge::FORMAT_NDHWC,  ge::DT_FLOAT)
      .SetInput("d", ge::FORMAT_NDHWC, ge::DT_FLOAT, "transdata3", ge::FORMAT_NDHWC,  ge::DT_FLOAT);
  test.DumpGraph(graph);

  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status ret = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));
  EXPECT_EQ(ret, fe::SUCCESS);

  FusionRuleManagerPtr fusion_rule_mgr_ptr = make_shared<FusionRuleManager>(fe_ops_kernel_info_store_ptr_);
  auto graph_fusion_ptr = std::make_shared<GraphFusion>(fusion_rule_mgr_ptr, fe_ops_kernel_info_store_ptr_,
                                                        nullptr);
  graph_fusion_ptr->SetEngineName(AI_CORE_NAME);
  ret = graph_fusion_ptr->SwitchTransDataAndCast(*(graph.get()), trans_op_insert.GetOptimizableCast());
  EXPECT_EQ(ret, fe::SUCCESS);
  test.DumpGraph(graph);

  ge::NodePtr cast = GraphConstructor::GetNodeByName("cast", graph);;
  ge::NodePtr transdata2 = GraphConstructor::GetNodeByName("transdata2", graph);
  ge::NodePtr transdata3 = GraphConstructor::GetNodeByName("transdata3", graph);

  ASSERT_EQ(cast->GetInAllNodes().size(), 1);
  ASSERT_EQ(cast->GetInAllNodes().at(0)->GetType(), "A");
  EXPECT_EQ(cast->GetOpDesc()->GetInputDesc(0).GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(cast->GetOpDesc()->GetOutputDesc(0).GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(cast->GetOpDesc()->GetInputDesc(0).GetFormat(), ge::FORMAT_NDC1HWC0);
  EXPECT_EQ(cast->GetOpDesc()->GetOutputDesc(0).GetFormat(), ge::FORMAT_NDC1HWC0);
  ASSERT_EQ(cast->GetOutAllNodes().size(), 3);
  ASSERT_EQ(cast->GetOutAllNodes().at(0)->GetName(), "b");
  ASSERT_EQ(cast->GetOutAllNodes().at(1)->GetName(), "transdata2");
  ASSERT_EQ(cast->GetOutAllNodes().at(2)->GetName(), "transdata3");

  ASSERT_EQ(transdata2->GetInAllNodes().size(), 1);
  ASSERT_EQ(transdata2->GetInAllNodes().at(0)->GetType(), fe::CAST);
  EXPECT_EQ(transdata2->GetOpDesc()->GetInputDesc(0).GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(transdata2->GetOpDesc()->GetOutputDesc(0).GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(transdata2->GetOpDesc()->GetInputDesc(0).GetFormat(), ge::FORMAT_NDC1HWC0);
  EXPECT_EQ(transdata2->GetOpDesc()->GetOutputDesc(0).GetFormat(), ge::FORMAT_NDHWC);

  ASSERT_EQ(transdata3->GetInAllNodes().size(), 1);
  ASSERT_EQ(transdata3->GetInAllNodes().at(0)->GetType(), fe::CAST);
  EXPECT_EQ(transdata3->GetOpDesc()->GetInputDesc(0).GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(transdata3->GetOpDesc()->GetOutputDesc(0).GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(transdata3->GetOpDesc()->GetInputDesc(0).GetFormat(), ge::FORMAT_NDC1HWC0);
  EXPECT_EQ(transdata3->GetOpDesc()->GetOutputDesc(0).GetFormat(), ge::FORMAT_NDHWC);
}

/*              -------->X------->
 *             /                  \
 * A -> TransData(4->5, bool) -> Cast(5HD, bool->fp16) -> B (5HD, fp16)
 * will be changed to :
 *              -------->X------->
 *             /                  \
 * A -> Cast(4D, bool->fp16) -> TransData(4->5, fp16) -> B (5HD, fp16)
 *
 * Transdata has a out control edge to X and X has
 * a out control edge to Cast(from TransData to Cast).
 *
 * After switching the control flow will start from Cast and
 * end at TransData.
 */
TEST_F(UTEST_FE_TRANSOP_INSERT_COMPLEX_2, SwitchCastAndTransdataCtrlEdge_01) {
  ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::GeShape original_shape = ge::GeShape({3, 12, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NHWC, ge::DT_FLOAT,
                        original_shape);
  test.AddOpDesc("a", "A", 1, 1)
      .AddOpDesc("transdata", fe::TRANSDATA, 1, 1)
      .AddOpDesc("cast", fe::CAST, 1, 1)
      .AddOpDesc("b", "B", 1, 1)
      .AddOpDesc("x", "X", 1, 1);

  test.SetInput("transdata", ge::FORMAT_NHWC, ge::DT_BOOL, "a", ge::FORMAT_NHWC, ge::DT_BOOL)
      .SetInput("cast", ge::FORMAT_NC1HWC0, ge::DT_BOOL, "transdata", ge::FORMAT_NC1HWC0, ge::DT_BOOL)
      .SetInput("b", ge::FORMAT_NC1HWC0, ge::DT_FLOAT16, "cast", ge::FORMAT_NC1HWC0,  ge::DT_FLOAT16)
      .SetInput("x:-1", "transdata:-1")
      .SetInput("cast:-1", "x:-1");

  test.DumpGraph(graph);

  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status ret = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));
  EXPECT_EQ(ret, fe::SUCCESS);

  FusionRuleManagerPtr fusion_rule_mgr_ptr = make_shared<FusionRuleManager>(fe_ops_kernel_info_store_ptr_);
  auto graph_fusion_ptr = std::make_shared<GraphFusion>(fusion_rule_mgr_ptr, fe_ops_kernel_info_store_ptr_,
                                                        nullptr);
  graph_fusion_ptr->SetEngineName(AI_CORE_NAME);
  ret = graph_fusion_ptr->SwitchTransDataAndCast(*(graph.get()), trans_op_insert.GetOptimizableCast());
  EXPECT_EQ(ret, fe::SUCCESS);
  test.DumpGraph(graph);
  ge::NodePtr cast;
  ge::NodePtr transdata;
  test.GetNodeByName("cast", cast);
  test.GetNodeByName("transdata", transdata);
  ASSERT_EQ(cast->GetInAllNodes().size(), 1);
  ASSERT_EQ(cast->GetInAllNodes().at(0)->GetType(), "A");
  ASSERT_EQ(transdata->GetInAllNodes().size(), 2);
  ASSERT_EQ(transdata->GetInAllNodes().at(0)->GetType(), fe::CAST);

  EXPECT_EQ(cast->GetOpDesc()->GetInputDesc(0).GetDataType(), ge::DT_BOOL);
  EXPECT_EQ(cast->GetOpDesc()->GetOutputDesc(0).GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(cast->GetOpDesc()->GetInputDesc(0).GetFormat(), ge::FORMAT_NHWC);
  EXPECT_EQ(cast->GetOpDesc()->GetOutputDesc(0).GetFormat(), ge::FORMAT_NHWC);

  EXPECT_EQ(transdata->GetOpDesc()->GetInputDesc(0).GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(transdata->GetOpDesc()->GetOutputDesc(0).GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(transdata->GetOpDesc()->GetInputDesc(0).GetFormat(), ge::FORMAT_NHWC);
  EXPECT_EQ(transdata->GetOpDesc()->GetOutputDesc(0).GetFormat(), ge::FORMAT_NC1HWC0);

  auto cast_out_control_anchor = cast->GetOutControlAnchor();
  auto cast_peer_in = cast_out_control_anchor->GetPeerInControlAnchors();

  ASSERT_EQ(cast_peer_in.size(), 1);

  auto transdata_in = cast_peer_in.at(0);
  ASSERT_NE(transdata_in, nullptr);

  auto transdata_node = transdata_in->GetOwnerNode();
  ASSERT_NE(transdata_node, nullptr);
  ASSERT_EQ(transdata_node->GetName(), "x");

  auto cast_in_control_anchor = cast->GetInControlAnchor();
  auto cast_in_peer_out = cast_in_control_anchor->GetPeerOutControlAnchors();

  ASSERT_EQ(cast_in_peer_out.size(), 0);
  ASSERT_EQ(graph->TopologicalSorting(), ge::GRAPH_SUCCESS);
}

/*              -------->X------->
 *             /                  \
 * A -> Cast(5HD, fp16->bool) -> TransData(5->4, bool) -> B (4D, bool)
 * will be changed to :
 *              -------->X------->
 *             /                  \
 * A -> TransData(5->4,fp16) -> Cast(4D, fp16->bool) -> B (4D, bool)
 *
 * Transdata has a out control edge to X and X has
 * a out control edge to Cast(from TransData to Cast).
 *
 * After switching the control flow will start from Cast and
 * end at TransData.
 */
TEST_F(UTEST_FE_TRANSOP_INSERT_COMPLEX_2, SwitchCastAndTransdataCtrlEdge_02) {
  ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::GeShape original_shape = ge::GeShape({3, 12, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NHWC, ge::DT_FLOAT,
                        original_shape);
  test.AddOpDesc("a", "A", 1, 1)
      .AddOpDesc("transdata", fe::TRANSDATA, 1, 1)
      .AddOpDesc("cast", fe::CAST, 1, 1)
      .AddOpDesc("b", "B", 1, 1)
      .AddOpDesc("x", "X", 1, 1);

  test.SetInput("cast", ge::FORMAT_NC1HWC0, ge::DT_FLOAT16, "a", ge::FORMAT_NC1HWC0, ge::DT_FLOAT16)
      .SetInput("transdata", ge::FORMAT_NC1HWC0, ge::DT_BOOL, "cast", ge::FORMAT_NC1HWC0, ge::DT_BOOL)
      .SetInput("b", ge::FORMAT_NHWC, ge::DT_BOOL, "transdata", ge::FORMAT_NHWC,  ge::DT_BOOL)
      .SetInput("x:-1", "cast:-1")
      .SetInput("transdata:-1", "x:-1");

  test.DumpGraph(graph);

  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status ret = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));
  EXPECT_EQ(ret, fe::SUCCESS);

  FusionRuleManagerPtr fusion_rule_mgr_ptr = make_shared<FusionRuleManager>(fe_ops_kernel_info_store_ptr_);
  auto graph_fusion_ptr = std::make_shared<GraphFusion>(fusion_rule_mgr_ptr, fe_ops_kernel_info_store_ptr_,
                                                        nullptr);
  graph_fusion_ptr->SetEngineName(AI_CORE_NAME);
  ret = graph_fusion_ptr->SwitchTransDataAndCast(*(graph.get()), trans_op_insert.GetOptimizableCast());
  EXPECT_EQ(ret, fe::SUCCESS);
  test.DumpGraph(graph);
  ge::NodePtr cast;
  ge::NodePtr transdata;
  test.GetNodeByName("cast", cast);
  test.GetNodeByName("transdata", transdata);
  ASSERT_EQ(cast->GetInAllNodes().size(), 2);
  ASSERT_EQ(cast->GetInAllNodes().at(0)->GetType(), fe::TRANSDATA);
  ASSERT_EQ(transdata->GetInAllNodes().size(), 1);
  ASSERT_EQ(transdata->GetInAllNodes().at(0)->GetType(), "A");

  EXPECT_EQ(cast->GetOpDesc()->GetInputDesc(0).GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(cast->GetOpDesc()->GetOutputDesc(0).GetDataType(), ge::DT_BOOL);
  EXPECT_EQ(cast->GetOpDesc()->GetInputDesc(0).GetFormat(), ge::FORMAT_NHWC);
  EXPECT_EQ(cast->GetOpDesc()->GetOutputDesc(0).GetFormat(), ge::FORMAT_NHWC);

  EXPECT_EQ(transdata->GetOpDesc()->GetInputDesc(0).GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(transdata->GetOpDesc()->GetOutputDesc(0).GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(transdata->GetOpDesc()->GetInputDesc(0).GetFormat(), ge::FORMAT_NC1HWC0);
  EXPECT_EQ(transdata->GetOpDesc()->GetOutputDesc(0).GetFormat(), ge::FORMAT_NHWC);

  auto transdata_out_control_anchor = transdata->GetOutControlAnchor();
  auto transdata_peer_in = transdata_out_control_anchor->GetPeerInControlAnchors();

  ASSERT_EQ(transdata_peer_in.size(), 1);

  auto cast_in = transdata_peer_in.at(0);
  ASSERT_NE(cast_in, nullptr);

  auto cast_node = cast_in->GetOwnerNode();
  ASSERT_NE(cast_node, nullptr);
  ASSERT_EQ(cast_node->GetName(), "x");

  auto transdata_in_control_anchor = transdata->GetInControlAnchor();
  auto transdata_in_peer_out = transdata_in_control_anchor->GetPeerOutControlAnchors();

  ASSERT_EQ(transdata_in_peer_out.size(), 0);
  ASSERT_EQ(graph->TopologicalSorting(), ge::GRAPH_SUCCESS);
}

/*
 *            C----\
 * original: A----->B, C->B is a control edge
 * After inserting transdata the graph will become:
 *
 *                    C-----\
 * A---->TransData1---->TransData2---->B
 *
 * After transdata-optimization, control edge will be
 * from C to B again
 */
TEST_F(UTEST_FE_TRANSOP_INSERT_COMPLEX_2, TransdataCtrlEdge_01) {
  ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::GeShape original_shape = ge::GeShape({3, 12, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NHWC, ge::DT_FLOAT,
                        original_shape);
  test.AddOpDesc("a", "A", 1, 1)
      .AddOpDesc("b", "B", 1, 1)
      .AddOpDesc("c", "C", 0, 0);

  test.SetInput("b", ge::FORMAT_NC1HWC0, ge::DT_FLOAT16, "a", ge::FORMAT_NC1HWC0, ge::DT_FLOAT16)
      .SetInput("b:-1", "c:-1");

  ge::NodePtr b;
  test.GetNodeByName("b", b);

  auto op_b = b->GetOpDesc();
  ge::AttrUtils::SetStr(op_b->MutableInputDesc(0), ge::ATTR_NAME_RESHAPE_INFER_TYPE, "NWC");

  test.DumpGraph(graph);

  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status ret = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));
  EXPECT_EQ(ret, fe::SUCCESS);

  ge::NodePtr a;
  test.GetNodeByName("a", a);
  EXPECT_EQ(graph->GetDirectNodesSize(), 3);
  EXPECT_EQ(a->GetInControlAnchor()->GetPeerAnchorsSize(), 0);
  ASSERT_EQ(b->GetInControlAnchor()->GetPeerAnchorsSize(), 1);
  EXPECT_EQ(b->GetInControlAnchor()->GetPeerOutControlAnchors().at(0)->GetOwnerNode()->GetName(), "c");
}

/*
 *      C----\
 * original: A----->B, C->B is a control edge
 * After inserting transdata the graph will become:
 *
 *  C-----\
 *        A---->TransData1---->TransData2---->B
 *
 * After transdata-optimization, control edge will be
 * from C to A again
 */
TEST_F(UTEST_FE_TRANSOP_INSERT_COMPLEX_2, TransdataCtrlEdge_02) {
  ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::GeShape original_shape = ge::GeShape({3, 12, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NHWC, ge::DT_FLOAT,
                        original_shape);
  test.AddOpDesc("a", "A", 1, 1)
      .AddOpDesc("b", "B", 1, 1)
      .AddOpDesc("c", "C", 0, 0);

  test.SetInput("b", ge::FORMAT_NC1HWC0, ge::DT_FLOAT16, "a", ge::FORMAT_NC1HWC0, ge::DT_FLOAT16)
      .SetInput("a:-1", "c:-1");

  ge::NodePtr b;
  test.GetNodeByName("b", b);

  auto op_b = b->GetOpDesc();
  ge::AttrUtils::SetStr(op_b->MutableInputDesc(0), ge::ATTR_NAME_RESHAPE_INFER_TYPE, "NWC");

  test.DumpGraph(graph);

  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status ret = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));
  EXPECT_EQ(ret, fe::SUCCESS);

  ge::NodePtr a;
  test.GetNodeByName("a", a);
  EXPECT_EQ(graph->GetDirectNodesSize(), 3);
  EXPECT_EQ(a->GetInControlAnchor()->GetPeerAnchorsSize(), 1);
  ASSERT_EQ(b->GetInControlAnchor()->GetPeerAnchorsSize(), 0);
  EXPECT_EQ(a->GetInControlAnchor()->GetPeerOutControlAnchors().at(0)->GetOwnerNode()->GetName(), "c");
}

/*
 *
 *     C-----\
 * A---->TransData1---->TransData2---->B
 *
 * After transdata-optimization, control edge will be
 * from C to B
 * */
TEST_F(UTEST_FE_TRANSOP_INSERT_COMPLEX_2, TransdataCtrlEdge_03) {
  ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::GeShape original_shape = ge::GeShape({3, 12, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NHWC, ge::DT_FLOAT,
                        original_shape);
  test.AddOpDesc("a", "A", 1, 1)
      .AddOpDesc("b", "B", 1, 1)
      .AddOpDesc("c", "C", 0, 0)
      .AddOpDesc("transdata1", "TransData", 1, 1)
      .AddOpDesc("transdata2", "TransData", 1, 1);

  test.SetInput("transdata1", ge::FORMAT_NC1HWC0, ge::DT_FLOAT16, "a", ge::FORMAT_NC1HWC0, ge::DT_FLOAT16)
      .SetInput("transdata2", ge::FORMAT_NHWC, ge::DT_FLOAT16, "transdata1", ge::FORMAT_NHWC, ge::DT_FLOAT16)
      .SetInput("b", ge::FORMAT_NC1HWC0, ge::DT_FLOAT16, "transdata2", ge::FORMAT_NC1HWC0, ge::DT_FLOAT16)
      .SetInput("transdata1:-1", "c:-1");

  ge::NodePtr b;
  test.GetNodeByName("b", b);

  auto op_b = b->GetOpDesc();
  ge::AttrUtils::SetStr(op_b->MutableInputDesc(0), ge::ATTR_NAME_RESHAPE_INFER_TYPE, "NWC");

  test.DumpGraph(graph);

  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status ret = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));
  EXPECT_EQ(ret, fe::SUCCESS);

  ge::NodePtr a;
  test.GetNodeByName("a", a);
  EXPECT_EQ(graph->GetDirectNodesSize(), 3);
  EXPECT_EQ(a->GetInControlAnchor()->GetPeerAnchorsSize(), 0);
  ASSERT_EQ(b->GetInControlAnchor()->GetPeerAnchorsSize(), 1);
  EXPECT_EQ(b->GetInControlAnchor()->GetPeerOutControlAnchors().at(0)->GetOwnerNode()->GetName(), "c");
}
