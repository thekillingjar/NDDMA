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
#include "graph_optimizer/shape_format_transfer/trans_node_manager/trans_node_manager.h"
#include "graph_optimizer/shape_format_transfer/trans_node_implementation/trans_node_base_generator.h"
#include "graph_optimizer/shape_format_transfer/trans_node_implementation/trans_node_cast_generator.h"
#include "graph_optimizer/shape_format_transfer/trans_node_implementation/trans_node_reshape_generator.h"
#include "graph_optimizer/shape_format_transfer/trans_node_implementation/trans_node_transpose_generator.h"
#include "graph_optimizer/shape_format_transfer/trans_node_implementation/trans_node_transdata_generator.h"
#include "common/configuration.h"
#include "ops_store/ops_kernel_manager.h"
#include "graph_optimizer/op_axis_update/op_axis_update_desc.h"
#include "../../../../graph_constructor/graph_constructor.h"
#undef protected
#undef private

#include <iostream>

using namespace std;
using namespace ge;
using namespace fe;



class UTEST_FE_TRANSOP_INSERT_FZ_3D_WITH_RESHAPE : public testing::Test {
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
    op_axis_update_desc_ptr_ = std::make_shared<fe::OpAxisUpdateDesc>(fe::AI_CORE_NAME);
  }

  void TearDown()
  {
    fe_ops_kernel_info_store_ptr_->Finalize();

  }

  shared_ptr<fe::FEOpsKernelInfoStore> fe_ops_kernel_info_store_ptr_;
  std::shared_ptr<fe::OpAxisUpdateDesc> op_axis_update_desc_ptr_;
 protected:

};


TEST_F(UTEST_FE_TRANSOP_INSERT_FZ_3D_WITH_RESHAPE, insert_reshape_01) {
  ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::GeShape original_shape = GeShape({128, 12, 5, 6, 16});
  GraphConstructor test(graph, "", ge::FORMAT_NDHWC, ge::DT_FLOAT16, original_shape);

  test.AddOpDesc("a", "A")
      .AddOpDesc("reduce", "ReduceOp")
      .AddOpDesc("b", "B")

      .SetInput("reduce:0", ge::FORMAT_FRACTAL_Z_3D, "a", ge::FORMAT_NDHWC)
      .SetInput("b", ge::FORMAT_NDHWC, "reduce", ge::FORMAT_FRACTAL_Z_3D);
  ge::NodePtr reduce;
  test.GetNodeByName("reduce", reduce);
  ge::AttrUtils::SetListInt(reduce->GetOpDesc(), AXES_ATTR_NAME, {2,3});
  op_axis_update_desc_ptr_->UpdateAxis(*(graph.get()));

  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status status = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, status);

  ge::NodePtr a;
  for (auto &node: graph->GetDirectNode()) {
    if (node->GetName() == "a") {
      a = node;
    }
  }
  std::vector<int64_t> shape_fz_3d = {360,8,16,16};
  std::vector<int64_t> shape_7d = {1,12,5,6,8,16,16};
  // Check First Transdata
  ge::NodePtr transdata_1 = a->GetOutAllNodes().at(0);
  {
    EXPECT_EQ(transdata_1->GetType(), fe::TRANSDATA);
    auto input_0 = transdata_1->GetOpDesc()->MutableInputDesc(0);
    EXPECT_EQ(input_0->GetFormat(), ge::FORMAT_NDHWC);
    EXPECT_EQ(input_0->GetShape().GetDims(), original_shape.GetDims());
    auto output_0 = transdata_1->GetOpDesc()->MutableOutputDesc(0);
    EXPECT_EQ(GetPrimaryFormat(output_0->GetFormat()), ge::FORMAT_FRACTAL_Z_3D);
    EXPECT_EQ(output_0->GetShape().GetDims(), shape_fz_3d);
  }

  // Check First Reshape
  ge::NodePtr reshape_1 = transdata_1->GetOutAllNodes().at(0);
  {
    EXPECT_EQ(reshape_1->GetType(), fe::RESHAPE);
    auto input_0 = reshape_1->GetOpDesc()->MutableInputDesc(0);
    EXPECT_EQ(input_0->GetFormat(), ge::FORMAT_FRACTAL_Z_3D);
    EXPECT_EQ(input_0->GetShape().GetDims(), shape_fz_3d);
    auto output_0 = reshape_1->GetOpDesc()->MutableOutputDesc(0);
    EXPECT_EQ(output_0->GetFormat(), ge::FORMAT_FRACTAL_Z_3D);
    EXPECT_EQ(output_0->GetShape().GetDims(), shape_7d);
  }

  // Check Second Reshape
  ge::NodePtr reshape_2 = reshape_1->GetOutAllNodes().at(0)->GetOutAllNodes().at(0);
  {
    EXPECT_EQ(reshape_2->GetType(), fe::RESHAPE);
    auto input_0 = reshape_2->GetOpDesc()->MutableInputDesc(0);
    std::vector<int64_t> shape = {360,8,16,16};
    EXPECT_EQ(input_0->GetFormat(), ge::FORMAT_FRACTAL_Z_3D);
    EXPECT_EQ(input_0->GetShape().GetDims(), shape_7d);
    auto output_0 = reshape_2->GetOpDesc()->MutableOutputDesc(0);
    EXPECT_EQ(output_0->GetFormat(), ge::FORMAT_FRACTAL_Z_3D);
    EXPECT_EQ(output_0->GetShape().GetDims(), shape_fz_3d);
  }

  // Check Second Transdata
  ge::NodePtr transdata_2 = reshape_2->GetOutAllNodes().at(0);
  {
    EXPECT_EQ(transdata_2->GetType(), fe::TRANSDATA);
    auto input_0 = transdata_2->GetOpDesc()->MutableInputDesc(0);
    std::vector<int64_t> shape = {360,8,16,16};
    EXPECT_EQ(input_0->GetFormat(), ge::FORMAT_FRACTAL_Z_3D);
    EXPECT_EQ(input_0->GetShape().GetDims(), shape);
    auto output_0 = transdata_2->GetOpDesc()->MutableOutputDesc(0);
    EXPECT_EQ(output_0->GetFormat(), ge::FORMAT_NDHWC);
    EXPECT_EQ(output_0->GetShape().GetDims(), original_shape.GetDims());
  }
}

TEST_F(UTEST_FE_TRANSOP_INSERT_FZ_3D_WITH_RESHAPE, insert_reshape_02) {
  ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::GeShape original_shape = GeShape({32, 12, 5, 6, 32});
  GraphConstructor test(graph, "", ge::FORMAT_DHWCN, ge::DT_FLOAT16, original_shape);

  test.AddOpDesc("a", "A")
      .AddOpDesc("reduce", "ReduceOp")
      .AddOpDesc("b", "B")

      .SetInput("reduce:0", ge::FORMAT_FRACTAL_Z_3D, "a", ge::FORMAT_DHWCN)
      .SetInput("b", ge::FORMAT_DHWCN, "reduce", ge::FORMAT_FRACTAL_Z_3D);
  ge::NodePtr reduce;
  test.GetNodeByName("reduce", reduce);
  ge::AttrUtils::SetListInt(reduce->GetOpDesc(), AXES_ATTR_NAME, {1,2});
  op_axis_update_desc_ptr_->UpdateAxis(*(graph.get()));

  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status status = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, status);

  ge::NodePtr a;
  for (auto &node: graph->GetDirectNode()) {
    if (node->GetName() == "a") {
      a = node;
    }
  }
  std::vector<int64_t> shape_fz_3d = {1*12*5*32,2,16,16};
  std::vector<int64_t> shape_7d = {1,32,12,5,2,16,16};
  // Check First Transdata
  ge::NodePtr transdata_1 = a->GetOutAllNodes().at(0);
  {
    ASSERT_EQ(transdata_1->GetType(), fe::TRANSDATA);
    auto input_0 = transdata_1->GetOpDesc()->MutableInputDesc(0);
    EXPECT_EQ(input_0->GetFormat(), ge::FORMAT_DHWCN);
    EXPECT_EQ(input_0->GetShape().GetDims(), original_shape.GetDims());
    auto output_0 = transdata_1->GetOpDesc()->MutableOutputDesc(0);
    EXPECT_EQ(GetPrimaryFormat(output_0->GetFormat()), ge::FORMAT_FRACTAL_Z_3D);
    EXPECT_EQ(output_0->GetShape().GetDims(), shape_fz_3d);
  }

  // Check First Reshape
  ge::NodePtr reshape_1 = transdata_1->GetOutAllNodes().at(0);
  {
    ASSERT_EQ(reshape_1->GetType(), fe::RESHAPE);
    auto input_0 = reshape_1->GetOpDesc()->MutableInputDesc(0);
    EXPECT_EQ(input_0->GetFormat(), ge::FORMAT_FRACTAL_Z_3D);
    EXPECT_EQ(input_0->GetShape().GetDims(), shape_fz_3d);
    auto output_0 = reshape_1->GetOpDesc()->MutableOutputDesc(0);
    EXPECT_EQ(output_0->GetFormat(), ge::FORMAT_FRACTAL_Z_3D);
    EXPECT_EQ(output_0->GetShape().GetDims(), shape_7d);
  }

  // Check Second Reshape
  ge::NodePtr reshape_2 = reshape_1->GetOutAllNodes().at(0)->GetOutAllNodes().at(0);
  {
    ASSERT_EQ(reshape_2->GetType(), fe::RESHAPE);
    auto input_0 = reshape_2->GetOpDesc()->MutableInputDesc(0);
    std::vector<int64_t> shape = {360,8,16,16};
    EXPECT_EQ(input_0->GetFormat(), ge::FORMAT_FRACTAL_Z_3D);
    EXPECT_EQ(input_0->GetShape().GetDims(), shape_7d);
    auto output_0 = reshape_2->GetOpDesc()->MutableOutputDesc(0);
    EXPECT_EQ(output_0->GetFormat(), ge::FORMAT_FRACTAL_Z_3D);
    EXPECT_EQ(output_0->GetShape().GetDims(), shape_fz_3d);
  }

  // Check Second Transdata
  ge::NodePtr transdata_2 = reshape_2->GetOutAllNodes().at(0);
  {
    ASSERT_EQ(transdata_2->GetType(), fe::TRANSDATA);
    auto input_0 = transdata_2->GetOpDesc()->MutableInputDesc(0);
    EXPECT_EQ(input_0->GetFormat(), ge::FORMAT_FRACTAL_Z_3D);
    EXPECT_EQ(input_0->GetShape().GetDims(), shape_fz_3d);
    auto output_0 = transdata_2->GetOpDesc()->MutableOutputDesc(0);
    EXPECT_EQ(output_0->GetFormat(), ge::FORMAT_DHWCN);
    EXPECT_EQ(output_0->GetShape().GetDims(), original_shape.GetDims());
  }
}


TEST_F(UTEST_FE_TRANSOP_INSERT_FZ_3D_WITH_RESHAPE, insert_reshape_03) {
  ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::GeShape original_shape = GeShape({32, 12, 5, 6, 32});
  GraphConstructor test(graph, "", ge::FORMAT_DHWCN, ge::DT_FLOAT16, original_shape);

  test.AddOpDesc("a", "A")
      .AddOpDesc("reduce", "ReduceOp")
      .AddOpDesc("reduce2", "ReduceOp")

      .SetInput("reduce:0", ge::FORMAT_FRACTAL_Z_3D, "a", ge::FORMAT_DHWCN)
      .SetInput("reduce2", ge::FORMAT_FRACTAL_Z_3D, "reduce", ge::FORMAT_FRACTAL_Z_3D);
  ge::NodePtr reduce;
  test.GetNodeByName("reduce", reduce);
  ge::AttrUtils::SetListInt(reduce->GetOpDesc(), AXES_ATTR_NAME, {1,2});
  ge::NodePtr reduce2;
  test.GetNodeByName("reduce2", reduce2);
  ge::AttrUtils::SetListInt(reduce2->GetOpDesc(), AXES_ATTR_NAME, {1,2});

  op_axis_update_desc_ptr_->UpdateAxis(*(graph.get()));

  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status status = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, status);

  ge::NodePtr a;
  for (auto &node: graph->GetDirectNode()) {
    if (node->GetName() == "a") {
      a = node;
    }
  }
  std::vector<int64_t> shape_fz_3d = {1*12*5*32, 2, 16, 16};
  std::vector<int64_t> shape_7d =  {1, 32, 12, 5, 2, 16, 16};
  // Check First Transdata
  ge::NodePtr transdata_1 = a->GetOutAllNodes().at(0);
  {
    EXPECT_EQ(transdata_1->GetType(), fe::TRANSDATA);
    auto input_0 = transdata_1->GetOpDesc()->MutableInputDesc(0);
    EXPECT_EQ(input_0->GetFormat(), ge::FORMAT_DHWCN);
    EXPECT_EQ(input_0->GetShape().GetDims(), original_shape.GetDims());
    auto output_0 = transdata_1->GetOpDesc()->MutableOutputDesc(0);
    EXPECT_EQ(GetPrimaryFormat(output_0->GetFormat()), ge::FORMAT_FRACTAL_Z_3D);
    EXPECT_EQ(output_0->GetShape().GetDims(), shape_fz_3d);
  }

  // Check First Reshape
  ge::NodePtr reshape_1 = transdata_1->GetOutAllNodes().at(0);
  {
    EXPECT_EQ(reshape_1->GetType(), fe::RESHAPE);
    auto input_0 = reshape_1->GetOpDesc()->MutableInputDesc(0);
    EXPECT_EQ(input_0->GetFormat(), ge::FORMAT_FRACTAL_Z_3D);
    EXPECT_EQ(input_0->GetShape().GetDims(), shape_fz_3d);
    auto output_0 = reshape_1->GetOpDesc()->MutableOutputDesc(0);
    EXPECT_EQ(output_0->GetFormat(), ge::FORMAT_FRACTAL_Z_3D);
    EXPECT_EQ(output_0->GetShape().GetDims(), shape_7d);
  }

  // Check reduce2
  ge::NodePtr reduce2_temp = reshape_1->GetOutAllNodes().at(0);
  EXPECT_EQ(reduce2_temp->GetType(), "ReduceOp");
}


TEST_F(UTEST_FE_TRANSOP_INSERT_FZ_3D_WITH_RESHAPE, insert_reshape_04) {
  ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::GeShape original_shape = GeShape({32, 12, 5, 6, 32});
  GraphConstructor test(graph, "", ge::FORMAT_DHWCN, ge::DT_FLOAT16, original_shape);

  test.AddOpDesc("reduce2", "ReduceOp")
      .AddOpDesc("reduce", "ReduceOp")
      .AddOpDesc("b", "B")

      .SetInput("reduce:0", ge::FORMAT_FRACTAL_Z_3D, "reduce2", ge::FORMAT_FRACTAL_Z_3D)
      .SetInput("b", ge::FORMAT_DHWCN, "reduce", ge::FORMAT_FRACTAL_Z_3D);
  ge::NodePtr reduce2;
  test.GetNodeByName("reduce2", reduce2);
  ge::AttrUtils::SetListInt(reduce2->GetOpDesc(), AXES_ATTR_NAME, {1,2});
  ge::NodePtr reduce;
  test.GetNodeByName("reduce", reduce);
  ge::AttrUtils::SetListInt(reduce->GetOpDesc(), AXES_ATTR_NAME, {1,2});

  op_axis_update_desc_ptr_->UpdateAxis(*(graph.get()));

  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status status = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, status);

  std::vector<int64_t> shape_fz_3d = {1*12*5*32, 2, 16, 16};
  std::vector<int64_t> shape_7d = {1, 32, 12, 5, 2, 16, 16};

  // Check First Reshape
  ge::NodePtr reshape_1 = reduce2->GetOutAllNodes().at(0)->GetOutAllNodes().at(0);
  {
    ASSERT_EQ(reshape_1->GetType(), fe::RESHAPE);
    auto input_0 = reshape_1->GetOpDesc()->MutableInputDesc(0);
    std::vector<int64_t> shape = {360,8,16,16};
    EXPECT_EQ(input_0->GetFormat(), ge::FORMAT_FRACTAL_Z_3D);
    EXPECT_EQ(input_0->GetShape().GetDims(), shape_7d);
    auto output_0 = reshape_1->GetOpDesc()->MutableOutputDesc(0);
    EXPECT_EQ(output_0->GetFormat(), ge::FORMAT_FRACTAL_Z_3D);
    EXPECT_EQ(output_0->GetShape().GetDims(), shape_fz_3d);
  }

  // Check First Transdata
  ge::NodePtr transdata_1 = reshape_1->GetOutAllNodes().at(0);
  {
    ASSERT_EQ(transdata_1->GetType(), fe::TRANSDATA);
    auto input_0 = transdata_1->GetOpDesc()->MutableInputDesc(0);
    EXPECT_EQ(input_0->GetFormat(), ge::FORMAT_FRACTAL_Z_3D);
    EXPECT_EQ(input_0->GetShape().GetDims(), shape_fz_3d);
    auto output_0 = transdata_1->GetOpDesc()->MutableOutputDesc(0);
    EXPECT_EQ(output_0->GetFormat(), ge::FORMAT_DHWCN);
    EXPECT_EQ(output_0->GetShape().GetDims(), original_shape.GetDims());
  }

  // Check b
  ge::NodePtr b = transdata_1->GetOutAllNodes().at(0);
  ASSERT_EQ(b->GetType(), "B");
}
