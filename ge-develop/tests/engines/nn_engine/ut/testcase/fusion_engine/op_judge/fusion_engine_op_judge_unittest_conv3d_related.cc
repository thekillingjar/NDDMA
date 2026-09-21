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
#include <memory>
#include "common/util/op_info_util.h"
#include "common/aicore_util_attr_define.h"
#include "fe_llt_utils.h"
#define private public
#define protected public
#include "common/unknown_shape_util.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/ge_local_context.h"
#include "adapter/common/op_store_adapter_manager.h"
#include "adapter/tbe_adapter/tbe_op_store_adapter.h"
#include "graph_optimizer/op_judge/imply_type/op_impl_type_judge.h"
#include "graph_optimizer/op_judge/format_and_dtype/op_format_dtype_judge.h"
#include "graph_optimizer/op_judge/format_and_dtype/strategy/matcher/dtype/op_dtype_rise_matcher.h"
#include "graph_optimizer/op_judge/format_and_dtype/strategy/matcher/format/op_format_matcher.h"

#include "graph_optimizer/shape_format_transfer/trans_node_manager/trans_node_manager.h"
#include "graph/debug/ge_attr_define.h"
#include "common/configuration.h"
#include "ops_store/ops_kernel_manager.h"
using namespace std;
using namespace ge;
using namespace fe;
#define DIMENSION_4 (4)
#define DIMENSION_1 (1)
using OpImplTypeJudgePtr = std::shared_ptr<OpImplTypeJudge>;
using OpFormatDtypeJudgePtr = std::shared_ptr<OpFormatDtypeJudge>;
using OpDtypeRiseMatcherPtr = std::shared_ptr<OpDtypeRiseMatcher>;
using OpFormatMatcherPtr = std::shared_ptr<OpFormatMatcher>;


using TransNodeManagerPtr = std::shared_ptr<TransNodeManager>;

class UTEST_fusion_engine_op_judge_unittest_conv3d : public testing::Test {
 protected:
  void SetUp() {
    std::map<std::string, std::string> options;
    fe_ops_kernel_info_store_ptr_ = make_shared<fe::FEOpsKernelInfoStore>();

    FEOpsStoreInfo tbe_custom{
        6,
        "tbe-custom",
        EN_IMPL_HW_TBE,
        GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/conv3d",
        ""};
    vector<FEOpsStoreInfo> store_info;

    store_info.emplace_back(tbe_custom);
    Configuration::Instance(fe::AI_CORE_NAME).ops_store_info_vector_ = (store_info);
    ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = ALLOW_MIX_PRECISION;
    OpsKernelManager::Instance(AI_CORE_NAME).Finalize();

    fe_ops_kernel_info_store_ptr_->Initialize(options);
    reflection_builder_ptr_ = std::make_shared<ge::RefRelations>();
    op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr_);
    op_format_dtype_judge_ptr_->Initialize();
  }

  void TearDown() {

  }
  shared_ptr<fe::FEOpsKernelInfoStore> fe_ops_kernel_info_store_ptr_;
  RefRelationsPtr reflection_builder_ptr_;
  OpFormatDtypeJudgePtr op_format_dtype_judge_ptr_;
};


TEST_F(UTEST_fusion_engine_op_judge_unittest_conv3d, test_01){
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr op1 = std::make_shared<OpDesc>("Conv3D", "Conv3D");
  OpDescPtr op2 = std::make_shared<OpDesc>("A1", "A");
  //add descriptor
  vector<int64_t> dim_input({4, 33, 12, 16, 64});
  GeShape shape(dim_input);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NDHWC);
  tensor_desc.SetOriginShape(shape);
  tensor_desc.SetFormat(FORMAT_NDHWC);
  tensor_desc.SetDataType(DT_FLOAT);
  op1->AddInputDesc("x", tensor_desc);

  vector<int64_t> dim_weight({3, 3, 3, 4, 5});
  GeShape weight_shape(dim_weight);
  GeTensorDesc weight_desc(weight_shape);
  weight_desc.SetOriginFormat(FORMAT_NDHWC);
  weight_desc.SetOriginShape(weight_shape);
  weight_desc.SetFormat(FORMAT_NDHWC);
  weight_desc.SetDataType(DT_FLOAT);
  op1->AddInputDesc("y", weight_desc);

  op1->AddOutputDesc("z", tensor_desc);
  ge::AttrUtils::SetInt(op1, FE_IMPLY_TYPE, 6);
  ge::NodePtr node1 = graph->AddNode(op1);

  op2->AddInputDesc("x", tensor_desc);
  op2->AddOutputDesc("z", tensor_desc);
  ge::AttrUtils::SetInt(op2, FE_IMPLY_TYPE, 6);
  ge::NodePtr node2 = graph->AddNode(op2);
  GraphUtils::AddEdge(node1->GetOutDataAnchor(0), node2->GetInDataAnchor(0));

  Status ret1 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(node1, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret1 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(node1, "tbe-custom");
  Status ret2 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(node2, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret2 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(node2, "tbe-custom");
  ASSERT_EQ(ret1, fe::SUCCESS);
  ASSERT_EQ(ret2, fe::SUCCESS);

  vector<int64_t> dim_result6_d({4, 33, 4, 12, 16, 16});
  vector<int64_t> dim_result_fz({36, 1, 16, 16});
  EXPECT_EQ(ge::GetPrimaryFormat(op1->GetInputDesc(0).GetFormat()), FORMAT_NDC1HWC0);
  EXPECT_EQ(op1->GetInputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op1->GetInputDesc(0).GetShape().GetDims(), dim_result6_d);

  EXPECT_EQ(ge::GetPrimaryFormat(op1->GetInputDesc(1).GetFormat()), FORMAT_FRACTAL_Z_3D);
  EXPECT_EQ(op1->GetInputDesc(1).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op1->GetInputDesc(1).GetShape().GetDims(), dim_result_fz);

  EXPECT_EQ(ge::GetPrimaryFormat(op1->GetOutputDesc(0).GetFormat()), FORMAT_NDC1HWC0);
  EXPECT_EQ(op1->GetOutputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op1->GetOutputDesc(0).GetShape().GetDims(), dim_result6_d);

  EXPECT_EQ(op2->GetInputDesc(0).GetFormat(), FORMAT_NDHWC);
  EXPECT_EQ(op2->GetInputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op2->GetInputDesc(0).GetShape().GetDims(), dim_input);
  EXPECT_EQ(op2->GetOutputDesc(0).GetFormat(), FORMAT_NDHWC);
  EXPECT_EQ(op2->GetOutputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op2->GetOutputDesc(0).GetShape().GetDims(), dim_input);
}

TEST_F(UTEST_fusion_engine_op_judge_unittest_conv3d, test_02){
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr op1 = std::make_shared<OpDesc>("Conv3DTranspose", "Conv3DTranspose");
  OpDescPtr op2 = std::make_shared<OpDesc>("A1", "A");
  //add descriptor
  vector<int64_t> dim_input({4, 33, 12, 16, 64});
  GeShape shape(dim_input);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NDHWC);
  tensor_desc.SetOriginShape(shape);
  tensor_desc.SetFormat(FORMAT_NDHWC);
  tensor_desc.SetDataType(DT_FLOAT);
  op1->AddInputDesc("x", tensor_desc);

  vector<int64_t> dim_weight({3, 3, 3, 4, 5});
  GeShape weight_shape(dim_weight);
  GeTensorDesc weight_desc(weight_shape);
  weight_desc.SetOriginFormat(FORMAT_DHWNC);
  weight_desc.SetOriginShape(weight_shape);
  weight_desc.SetFormat(FORMAT_DHWNC);
  weight_desc.SetDataType(DT_FLOAT);
  op1->AddInputDesc("y", weight_desc);

  op1->AddOutputDesc("z", tensor_desc);
  ge::AttrUtils::SetInt(op1, FE_IMPLY_TYPE, 6);
  ge::NodePtr node1 = graph->AddNode(op1);

  op2->AddInputDesc("x", tensor_desc);
  op2->AddOutputDesc("z", tensor_desc);
  ge::AttrUtils::SetInt(op2, FE_IMPLY_TYPE, 6);
  ge::NodePtr node2 = graph->AddNode(op2);
  GraphUtils::AddEdge(node1->GetOutDataAnchor(0), node2->GetInDataAnchor(0));

  Status ret1 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(node1, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret1 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(node1, "tbe-custom");
  Status ret2 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(node2, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret2 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(node2, "tbe-custom");
  ASSERT_EQ(ret1, fe::SUCCESS);
  ASSERT_EQ(ret2, fe::SUCCESS);

  vector<int64_t> dim_result6_d({4, 33, 4, 12, 16, 16});
  vector<int64_t> dim_result_fz({27, 1, 16, 16});
  EXPECT_EQ(ge::GetPrimaryFormat(op1->GetInputDesc(0).GetFormat()), FORMAT_NDC1HWC0);
  EXPECT_EQ(op1->GetInputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op1->GetInputDesc(0).GetShape().GetDims(), dim_result6_d);

  EXPECT_EQ(ge::GetPrimaryFormat(op1->GetInputDesc(1).GetFormat()), FORMAT_FRACTAL_Z_3D_TRANSPOSE);
  EXPECT_EQ(op1->GetInputDesc(1).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op1->GetInputDesc(1).GetShape().GetDims(), dim_result_fz);

  EXPECT_EQ(ge::GetPrimaryFormat(op1->GetOutputDesc(0).GetFormat()), FORMAT_NDC1HWC0);
  EXPECT_EQ(op1->GetOutputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op1->GetOutputDesc(0).GetShape().GetDims(), dim_result6_d);

  EXPECT_EQ(op2->GetInputDesc(0).GetFormat(), FORMAT_NDHWC);
  EXPECT_EQ(op2->GetInputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op2->GetInputDesc(0).GetShape().GetDims(), dim_input);
  EXPECT_EQ(op2->GetOutputDesc(0).GetFormat(), FORMAT_NDHWC);
  EXPECT_EQ(op2->GetOutputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op2->GetOutputDesc(0).GetShape().GetDims(), dim_input);
}


TEST_F(UTEST_fusion_engine_op_judge_unittest_conv3d, test_03){
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr op1 = std::make_shared<OpDesc>("Conv3D", "Conv3D");
  OpDescPtr op2 = std::make_shared<OpDesc>("A1", "A");
  //add descriptor
  vector<int64_t> dim_input({4, 33, 12, 16, 64});
  GeShape shape(dim_input);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NDHWC);
  tensor_desc.SetOriginShape(shape);
  tensor_desc.SetFormat(FORMAT_NDHWC);
  tensor_desc.SetDataType(DT_FLOAT);
  op1->AddInputDesc("x", tensor_desc);

  vector<int64_t> dim_weight({3, 3, 3, 4, 5});
  GeShape weight_shape(dim_weight);
  GeTensorDesc weight_desc(weight_shape);
  weight_desc.SetOriginFormat(FORMAT_DHWCN);
  weight_desc.SetOriginShape(weight_shape);
  weight_desc.SetFormat(FORMAT_DHWCN);
  weight_desc.SetDataType(DT_FLOAT);
  op1->AddInputDesc("y", weight_desc);

  op1->AddOutputDesc("z", tensor_desc);
  ge::AttrUtils::SetInt(op1, FE_IMPLY_TYPE, 6);
  ge::NodePtr node1 = graph->AddNode(op1);

  op2->AddInputDesc("x", tensor_desc);
  op2->AddOutputDesc("z", tensor_desc);
  ge::AttrUtils::SetInt(op2, FE_IMPLY_TYPE, 6);
  ge::NodePtr node2 = graph->AddNode(op2);
  GraphUtils::AddEdge(node1->GetOutDataAnchor(0), node2->GetInDataAnchor(0));

  Status ret1 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(node1, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret1 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(node1, "tbe-custom");
  Status ret2 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(node2, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret2 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(node2, "tbe-custom");
  ASSERT_EQ(ret1, fe::SUCCESS);
  ASSERT_EQ(ret2, fe::SUCCESS);

  vector<int64_t> dim_result6_d({4, 33, 4, 12, 16, 16});
  vector<int64_t> dim_result_fz({27, 1, 16, 16});
  EXPECT_EQ(ge::GetPrimaryFormat(op1->GetInputDesc(0).GetFormat()), FORMAT_NDC1HWC0);
  EXPECT_EQ(op1->GetInputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op1->GetInputDesc(0).GetShape().GetDims(), dim_result6_d);

  EXPECT_EQ(ge::GetPrimaryFormat(op1->GetInputDesc(1).GetFormat()), FORMAT_FRACTAL_Z_3D);
  EXPECT_EQ(op1->GetInputDesc(1).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op1->GetInputDesc(1).GetShape().GetDims(), dim_result_fz);

  EXPECT_EQ(ge::GetPrimaryFormat(op1->GetOutputDesc(0).GetFormat()), FORMAT_NDC1HWC0);
  EXPECT_EQ(op1->GetOutputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op1->GetOutputDesc(0).GetShape().GetDims(), dim_result6_d);

  EXPECT_EQ(op2->GetInputDesc(0).GetFormat(), FORMAT_NDHWC);
  EXPECT_EQ(op2->GetInputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op2->GetInputDesc(0).GetShape().GetDims(), dim_input);
  EXPECT_EQ(op2->GetOutputDesc(0).GetFormat(), FORMAT_NDHWC);
  EXPECT_EQ(op2->GetOutputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op2->GetOutputDesc(0).GetShape().GetDims(), dim_input);
}

// x: NCDHW, filter:NCDHW, y:NCDHW
TEST_F(UTEST_fusion_engine_op_judge_unittest_conv3d, test_04){
    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
    OpDescPtr op1 = std::make_shared<OpDesc>("Conv3D", "Conv3D");
    OpDescPtr op2 = std::make_shared<OpDesc>("A1", "A");
    //add descriptor
    vector<int64_t> dim_input({4, 33, 12, 16, 64});
    GeShape shape(dim_input);
    GeTensorDesc tensor_desc(shape);
    tensor_desc.SetOriginFormat(FORMAT_NCDHW);
    tensor_desc.SetOriginShape(shape);
    tensor_desc.SetFormat(FORMAT_NCDHW);
    tensor_desc.SetDataType(DT_FLOAT);
    op1->AddInputDesc("x", tensor_desc);

    vector<int64_t> dim_weight({3, 3, 3, 4, 5});
    GeShape weight_shape(dim_weight);
    GeTensorDesc weight_desc(weight_shape);
    weight_desc.SetOriginFormat(FORMAT_NCDHW);
    weight_desc.SetOriginShape(weight_shape);
    weight_desc.SetFormat(FORMAT_NCDHW);
    weight_desc.SetDataType(DT_FLOAT);
    op1->AddInputDesc("y", weight_desc);

    op1->AddOutputDesc("z", tensor_desc);
    ge::AttrUtils::SetInt(op1, FE_IMPLY_TYPE, 6);
    ge::NodePtr node1 = graph->AddNode(op1);

    op2->AddInputDesc("x", tensor_desc);
    op2->AddOutputDesc("z", tensor_desc);
    ge::AttrUtils::SetInt(op2, FE_IMPLY_TYPE, 6);
    ge::NodePtr node2 = graph->AddNode(op2);
    GraphUtils::AddEdge(node1->GetOutDataAnchor(0), node2->GetInDataAnchor(0));

    Status ret1 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(node1, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret1 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(node1, "tbe-custom");
    Status ret2 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(node2, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret2 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(node2, "tbe-custom");
    ASSERT_EQ(ret1, fe::SUCCESS);
    ASSERT_EQ(ret2, fe::SUCCESS);

    vector<int64_t> dim_result6_d({4, 12, 3, 16, 64, 16});
    vector<int64_t> dim_result_fz({60, 1, 16, 16});
    EXPECT_EQ(ge::GetPrimaryFormat(op1->GetInputDesc(0).GetFormat()), FORMAT_NDC1HWC0);
    EXPECT_EQ(op1->GetInputDesc(0).GetDataType(), DT_FLOAT16);
    EXPECT_EQ(op1->GetInputDesc(0).GetShape().GetDims(), dim_result6_d);

    EXPECT_EQ(ge::GetPrimaryFormat(op1->GetInputDesc(1).GetFormat()), FORMAT_FRACTAL_Z_3D);
    EXPECT_EQ(op1->GetInputDesc(1).GetDataType(), DT_FLOAT16);
    EXPECT_EQ(op1->GetInputDesc(1).GetShape().GetDims(), dim_result_fz);

    EXPECT_EQ(ge::GetPrimaryFormat(op1->GetOutputDesc(0).GetFormat()), FORMAT_NDC1HWC0);
    EXPECT_EQ(op1->GetOutputDesc(0).GetDataType(), DT_FLOAT16);
    EXPECT_EQ(op1->GetOutputDesc(0).GetShape().GetDims(), dim_result6_d);

    EXPECT_EQ(op2->GetInputDesc(0).GetFormat(), FORMAT_NCDHW);
    EXPECT_EQ(op2->GetInputDesc(0).GetDataType(), DT_FLOAT16);
    EXPECT_EQ(op2->GetInputDesc(0).GetShape().GetDims(), dim_input);
    EXPECT_EQ(op2->GetOutputDesc(0).GetFormat(), FORMAT_NCDHW);
    EXPECT_EQ(op2->GetOutputDesc(0).GetDataType(), DT_FLOAT16);
    EXPECT_EQ(op2->GetOutputDesc(0).GetShape().GetDims(), dim_input);
}

// x: NDCHW, filter:NDCHW, y:NDCHW
TEST_F(UTEST_fusion_engine_op_judge_unittest_conv3d, test_05){
    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
    OpDescPtr op1 = std::make_shared<OpDesc>("Conv3D", "Conv3D");
    OpDescPtr op2 = std::make_shared<OpDesc>("A1", "A");
    //add descriptor
    vector<int64_t> dim_input({4, 33, 12, 16, 64});
    GeShape shape(dim_input);
    GeTensorDesc tensor_desc(shape);
    tensor_desc.SetOriginFormat(FORMAT_NDHWC);
    tensor_desc.SetOriginShape(shape);
    tensor_desc.SetFormat(FORMAT_NDHWC);
    tensor_desc.SetDataType(DT_FLOAT);
    op1->AddInputDesc("x", tensor_desc);

    vector<int64_t> dim_weight({3, 3, 3, 4, 5});
    GeShape weight_shape(dim_weight);
    GeTensorDesc weight_desc(weight_shape);
    weight_desc.SetOriginFormat(FORMAT_NDHWC);
    weight_desc.SetOriginShape(weight_shape);
    weight_desc.SetFormat(FORMAT_NDHWC);
    weight_desc.SetDataType(DT_FLOAT);
    op1->AddInputDesc("y", weight_desc);

    op1->AddOutputDesc("z", tensor_desc);
    ge::AttrUtils::SetInt(op1, FE_IMPLY_TYPE, 6);
    ge::NodePtr node1 = graph->AddNode(op1);

    op2->AddInputDesc("x", tensor_desc);
    op2->AddOutputDesc("z", tensor_desc);
    ge::AttrUtils::SetInt(op2, FE_IMPLY_TYPE, 6);
    ge::NodePtr node2 = graph->AddNode(op2);
    GraphUtils::AddEdge(node1->GetOutDataAnchor(0), node2->GetInDataAnchor(0));

    Status ret1 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(node1, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret1 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(node1, "tbe-custom");
    Status ret2 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(node2, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret2 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(node2, "tbe-custom");
    ASSERT_EQ(ret1, fe::SUCCESS);
    ASSERT_EQ(ret2, fe::SUCCESS);

    vector<int64_t> dim_result6_d({4, 33, 4, 12, 16, 16});
    vector<int64_t> dim_result_fz({36, 1, 16, 16});
    EXPECT_EQ(ge::GetPrimaryFormat(op1->GetInputDesc(0).GetFormat()), FORMAT_NDC1HWC0);
    EXPECT_EQ(op1->GetInputDesc(0).GetDataType(), DT_FLOAT16);
    EXPECT_EQ(op1->GetInputDesc(0).GetShape().GetDims(), dim_result6_d);

    EXPECT_EQ(ge::GetPrimaryFormat(op1->GetInputDesc(1).GetFormat()), FORMAT_FRACTAL_Z_3D);
    EXPECT_EQ(op1->GetInputDesc(1).GetDataType(), DT_FLOAT16);
    EXPECT_EQ(op1->GetInputDesc(1).GetShape().GetDims(), dim_result_fz);

    EXPECT_EQ(ge::GetPrimaryFormat(op1->GetOutputDesc(0).GetFormat()), FORMAT_NDC1HWC0);
    EXPECT_EQ(op1->GetOutputDesc(0).GetDataType(), DT_FLOAT16);
    EXPECT_EQ(op1->GetOutputDesc(0).GetShape().GetDims(), dim_result6_d);

    EXPECT_EQ(op2->GetInputDesc(0).GetFormat(), FORMAT_NDHWC);
    EXPECT_EQ(op2->GetInputDesc(0).GetDataType(), DT_FLOAT16);
    EXPECT_EQ(op2->GetInputDesc(0).GetShape().GetDims(), dim_input);
    EXPECT_EQ(op2->GetOutputDesc(0).GetFormat(), FORMAT_NDHWC);
    EXPECT_EQ(op2->GetOutputDesc(0).GetDataType(), DT_FLOAT16);
    EXPECT_EQ(op2->GetOutputDesc(0).GetShape().GetDims(), dim_input);
}

TEST_F(UTEST_fusion_engine_op_judge_unittest_conv3d, test_expand_dim_NDHWC){
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr op1 = std::make_shared<OpDesc>("b1", "B");
  OpDescPtr op2 = std::make_shared<OpDesc>("b2", "B");
  OpDescPtr op3 = std::make_shared<OpDesc>("b3", "B");
  OpDescPtr op4 = std::make_shared<OpDesc>("b4", "B");

  vector<int64_t> dim_input({44});
  GeShape shape(dim_input);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NDHWC);
  tensor_desc.SetOriginShape(shape);
  tensor_desc.SetFormat(FORMAT_NDHWC);
  tensor_desc.SetDataType(DT_FLOAT);
  op1->AddInputDesc("x", tensor_desc);
  op1->AddOutputDesc("z", tensor_desc);
  ge::AttrUtils::SetInt(op1, FE_IMPLY_TYPE, 6);
  ge::NodePtr node_1 = graph->AddNode(op1);

  vector<int64_t> dim_input_2({44, 55});
  GeShape shape_2(dim_input_2);
  GeTensorDesc tensor_desc_2(shape_2);
  tensor_desc_2.SetOriginFormat(FORMAT_NDHWC);
  tensor_desc_2.SetOriginShape(shape);
  tensor_desc_2.SetFormat(FORMAT_NDHWC);
  tensor_desc_2.SetDataType(DT_FLOAT);
  op2->AddInputDesc("x", tensor_desc_2);
  op2->AddOutputDesc("z", tensor_desc_2);
  ge::AttrUtils::SetInt(op2, FE_IMPLY_TYPE, 6);
  ge::NodePtr node_2 = graph->AddNode(op2);

  vector<int64_t> dim_input_3({44, 55, 66});
  GeShape shape_3(dim_input_3);
  GeTensorDesc tensor_desc_3(shape_3);
  tensor_desc_3.SetOriginFormat(FORMAT_NDHWC);
  tensor_desc_3.SetOriginShape(shape);
  tensor_desc_3.SetFormat(FORMAT_NDHWC);
  tensor_desc_3.SetDataType(DT_FLOAT);
  op3->AddInputDesc("x", tensor_desc_3);
  op3->AddOutputDesc("z", tensor_desc_3);
  ge::AttrUtils::SetInt(op3, FE_IMPLY_TYPE, 6);
  ge::NodePtr node_3 = graph->AddNode(op3);

  vector<int64_t> dim_input_4({44, 55, 66, 77});
  GeShape shape_4(dim_input_4);
  GeTensorDesc tensor_desc_4(shape_4);
  tensor_desc_4.SetOriginFormat(FORMAT_NDHWC);
  tensor_desc_4.SetOriginShape(shape);
  tensor_desc_4.SetFormat(FORMAT_NDHWC);
  tensor_desc_4.SetDataType(DT_FLOAT);
  op4->AddInputDesc("x", tensor_desc_4);
  op4->AddOutputDesc("z", tensor_desc_4);
  ge::AttrUtils::SetInt(op4, FE_IMPLY_TYPE, 6);
  ge::NodePtr node_4 = graph->AddNode(op4);

  Status ret1 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(node_1, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret1 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(node_1, "tbe-custom");
  ASSERT_EQ(ret1, fe::SUCCESS);
  Status ret2 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(node_2, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret2 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(node_2, "tbe-custom");
  ASSERT_EQ(ret1, fe::SUCCESS);
  Status ret3 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(node_3, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret3 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(node_3, "tbe-custom");
  ASSERT_EQ(ret1, fe::SUCCESS);
  Status ret4 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(node_4, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret4 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(node_4, "tbe-custom");
  ASSERT_EQ(ret1, fe::SUCCESS);

  vector<int64_t> dim_result6_d_1({1, 1, 3, 1, 1, 16});
  EXPECT_EQ(ge::GetPrimaryFormat(op1->GetInputDesc(0).GetFormat()), FORMAT_NDC1HWC0);
  EXPECT_EQ(op1->GetInputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op1->GetInputDesc(0).GetShape().GetDims(), dim_result6_d_1);

  EXPECT_EQ(ge::GetPrimaryFormat(op1->GetOutputDesc(0).GetFormat()), FORMAT_NDC1HWC0);
  EXPECT_EQ(op1->GetOutputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op1->GetOutputDesc(0).GetShape().GetDims(), dim_result6_d_1);

  vector<int64_t> dim_result6_d_2({1, 1, 4, 1, 44, 16});
  EXPECT_EQ(ge::GetPrimaryFormat(op2->GetInputDesc(0).GetFormat()), FORMAT_NDC1HWC0);
  EXPECT_EQ(op2->GetInputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op2->GetInputDesc(0).GetShape().GetDims(), dim_result6_d_2);

  EXPECT_EQ(ge::GetPrimaryFormat(op2->GetOutputDesc(0).GetFormat()), FORMAT_NDC1HWC0);
  EXPECT_EQ(op2->GetOutputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op2->GetOutputDesc(0).GetShape().GetDims(), dim_result6_d_2);

  vector<int64_t> dim_result6_d_3({1, 1, 5, 44, 55, 16});
  EXPECT_EQ(ge::GetPrimaryFormat(op3->GetInputDesc(0).GetFormat()), FORMAT_NDC1HWC0);
  EXPECT_EQ(op3->GetInputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op3->GetInputDesc(0).GetShape().GetDims(), dim_result6_d_3);

  EXPECT_EQ(ge::GetPrimaryFormat(op3->GetOutputDesc(0).GetFormat()), FORMAT_NDC1HWC0);
  EXPECT_EQ(op3->GetOutputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op3->GetOutputDesc(0).GetShape().GetDims(), dim_result6_d_3);

  vector<int64_t> dim_result6_d_4({1, 44, 5, 55, 66, 16});
  EXPECT_EQ(ge::GetPrimaryFormat(op4->GetInputDesc(0).GetFormat()), FORMAT_NDC1HWC0);
  EXPECT_EQ(op4->GetInputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op4->GetInputDesc(0).GetShape().GetDims(), dim_result6_d_4);

  EXPECT_EQ(ge::GetPrimaryFormat(op4->GetOutputDesc(0).GetFormat()), FORMAT_NDC1HWC0);
  EXPECT_EQ(op4->GetOutputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op4->GetOutputDesc(0).GetShape().GetDims(), dim_result6_d_4);
}

TEST_F(UTEST_fusion_engine_op_judge_unittest_conv3d, test_expand_dim_NCDHW){
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr op1 = std::make_shared<OpDesc>("b1", "B");
  OpDescPtr op2 = std::make_shared<OpDesc>("b2", "B");
  OpDescPtr op3 = std::make_shared<OpDesc>("b3", "B");
  OpDescPtr op4 = std::make_shared<OpDesc>("b4", "B");

  vector<int64_t> dim_input({44});
  GeShape shape(dim_input);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NCDHW);
  tensor_desc.SetOriginShape(shape);
  tensor_desc.SetFormat(FORMAT_NCDHW);
  tensor_desc.SetDataType(DT_FLOAT);
  op1->AddInputDesc("x", tensor_desc);
  op1->AddOutputDesc("z", tensor_desc);
  ge::AttrUtils::SetInt(op1, FE_IMPLY_TYPE, 6);
  ge::NodePtr node_1 = graph->AddNode(op1);

  vector<int64_t> dim_input_2({44, 55});
  GeShape shape_2(dim_input_2);
  GeTensorDesc tensor_desc_2(shape_2);
  tensor_desc_2.SetOriginFormat(FORMAT_NCDHW);
  tensor_desc_2.SetOriginShape(shape);
  tensor_desc_2.SetFormat(FORMAT_NCDHW);
  tensor_desc_2.SetDataType(DT_FLOAT);
  op2->AddInputDesc("x", tensor_desc_2);
  op2->AddOutputDesc("z", tensor_desc_2);
  ge::AttrUtils::SetInt(op2, FE_IMPLY_TYPE, 6);
  ge::NodePtr node_2 = graph->AddNode(op2);

  vector<int64_t> dim_input_3({44, 55, 66});
  GeShape shape_3(dim_input_3);
  GeTensorDesc tensor_desc_3(shape_3);
  tensor_desc_3.SetOriginFormat(FORMAT_NCDHW);
  tensor_desc_3.SetOriginShape(shape);
  tensor_desc_3.SetFormat(FORMAT_NCDHW);
  tensor_desc_3.SetDataType(DT_FLOAT);
  op3->AddInputDesc("x", tensor_desc_3);
  op3->AddOutputDesc("z", tensor_desc_3);
  ge::AttrUtils::SetInt(op3, FE_IMPLY_TYPE, 6);
  ge::NodePtr node_3 = graph->AddNode(op3);

  vector<int64_t> dim_input_4({44, 55, 66, 77});
  GeShape shape_4(dim_input_4);
  GeTensorDesc tensor_desc_4(shape_4);
  tensor_desc_4.SetOriginFormat(FORMAT_NCDHW);
  tensor_desc_4.SetOriginShape(shape);
  tensor_desc_4.SetFormat(FORMAT_NCDHW);
  tensor_desc_4.SetDataType(DT_FLOAT);
  op4->AddInputDesc("x", tensor_desc_4);
  op4->AddOutputDesc("z", tensor_desc_4);
  ge::AttrUtils::SetInt(op4, FE_IMPLY_TYPE, 6);
  ge::NodePtr node_4 = graph->AddNode(op4);

  Status ret1 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(node_1, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret1 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(node_1, "tbe-custom");
  ASSERT_EQ(ret1, fe::SUCCESS);
  Status ret2 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(node_2, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret2 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(node_2, "tbe-custom");
  ASSERT_EQ(ret1, fe::SUCCESS);
  Status ret3 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(node_3, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret3 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(node_3, "tbe-custom");
  ASSERT_EQ(ret1, fe::SUCCESS);
  Status ret4 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(node_4, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret4 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(node_4, "tbe-custom");
  ASSERT_EQ(ret1, fe::SUCCESS);

  vector<int64_t> dim_result6_d_1({1, 1, 3, 1, 1, 16});
  EXPECT_EQ(ge::GetPrimaryFormat(op1->GetInputDesc(0).GetFormat()), FORMAT_NDC1HWC0);
  EXPECT_EQ(op1->GetInputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op1->GetInputDesc(0).GetShape().GetDims(), dim_result6_d_1);

  EXPECT_EQ(ge::GetPrimaryFormat(op1->GetOutputDesc(0).GetFormat()), FORMAT_NDC1HWC0);
  EXPECT_EQ(op1->GetOutputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op1->GetOutputDesc(0).GetShape().GetDims(), dim_result6_d_1);

  vector<int64_t> dim_result6_d_2({1, 1, 1, 44, 55, 16});
  EXPECT_EQ(ge::GetPrimaryFormat(op2->GetInputDesc(0).GetFormat()), FORMAT_NDC1HWC0);
  EXPECT_EQ(op2->GetInputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op2->GetInputDesc(0).GetShape().GetDims(), dim_result6_d_2);

  EXPECT_EQ(ge::GetPrimaryFormat(op2->GetOutputDesc(0).GetFormat()), FORMAT_NDC1HWC0);
  EXPECT_EQ(op2->GetOutputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op2->GetOutputDesc(0).GetShape().GetDims(), dim_result6_d_2);

  vector<int64_t> dim_result6_d_3({1, 44, 1, 55, 66, 16});
  EXPECT_EQ(ge::GetPrimaryFormat(op3->GetInputDesc(0).GetFormat()), FORMAT_NDC1HWC0);
  EXPECT_EQ(op3->GetInputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op3->GetInputDesc(0).GetShape().GetDims(), dim_result6_d_3);

  EXPECT_EQ(ge::GetPrimaryFormat(op3->GetOutputDesc(0).GetFormat()), FORMAT_NDC1HWC0);
  EXPECT_EQ(op3->GetOutputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op3->GetOutputDesc(0).GetShape().GetDims(), dim_result6_d_3);

  vector<int64_t> dim_result6_d_4({1, 55, 3, 66, 77, 16});
  EXPECT_EQ(ge::GetPrimaryFormat(op4->GetInputDesc(0).GetFormat()), FORMAT_NDC1HWC0);
  EXPECT_EQ(op4->GetInputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op4->GetInputDesc(0).GetShape().GetDims(), dim_result6_d_4);

  EXPECT_EQ(ge::GetPrimaryFormat(op4->GetOutputDesc(0).GetFormat()), FORMAT_NDC1HWC0);
  EXPECT_EQ(op4->GetOutputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op4->GetOutputDesc(0).GetShape().GetDims(), dim_result6_d_4);
}

TEST_F(UTEST_fusion_engine_op_judge_unittest_conv3d, test_opjudge_update_shape_NCDHW){
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr op1 = std::make_shared<OpDesc>("A1", "BatchNormV9");
  // x w b c ac bc
  // NCDHW
  vector<int64_t> dim_input_x({44});
  GeShape shape(dim_input_x);
  GeTensorDesc tensor_x_desc(shape);
  tensor_x_desc.SetOriginFormat(FORMAT_NCDHW);
  tensor_x_desc.SetOriginShape(shape);
  tensor_x_desc.SetFormat(FORMAT_NCDHW);
  tensor_x_desc.SetDataType(DT_FLOAT);
  op1->AddInputDesc("x", tensor_x_desc);

  // NDHWC
  vector<int64_t> dim_input_w({44, 55});
  GeShape shape_2(dim_input_w);
  GeTensorDesc tensor_desc_2(shape_2);
  tensor_desc_2.SetOriginFormat(FORMAT_NCDHW);
  tensor_desc_2.SetOriginShape(shape);
  tensor_desc_2.SetFormat(FORMAT_NCDHW);
  tensor_desc_2.SetDataType(DT_FLOAT);
  op1->AddInputDesc("w", tensor_desc_2);

  // DHWCN
  vector<int64_t> dim_input_b({44, 55, 66});
  GeShape shape_3(dim_input_b);
  GeTensorDesc tensor_desc_3(shape_3);
  tensor_desc_3.SetOriginFormat(FORMAT_NCDHW);
  tensor_desc_3.SetOriginShape(shape);
  tensor_desc_3.SetFormat(FORMAT_NCDHW);
  tensor_desc_3.SetDataType(DT_FLOAT);
  op1->AddInputDesc("b", tensor_desc_3);

  // DHWNC
  vector<int64_t> dim_input_c({44, 55, 66, 77});
  GeShape shape_4(dim_input_c);
  GeTensorDesc tensor_desc_4(shape_4);
  tensor_desc_4.SetOriginFormat(FORMAT_NCDHW);
  tensor_desc_4.SetOriginShape(shape);
  tensor_desc_4.SetFormat(FORMAT_NCDHW);
  tensor_desc_4.SetDataType(DT_FLOAT);
  op1->AddInputDesc("c", tensor_desc_4);

  // DHWNC
  vector<int64_t> dim_input_ac({44, 55, 66, 77, 88});
  GeShape shape_5(dim_input_ac);
  GeTensorDesc tensor_desc_5(shape_5);
  tensor_desc_5.SetOriginFormat(FORMAT_NCDHW);
  tensor_desc_5.SetOriginShape(shape);
  tensor_desc_5.SetFormat(FORMAT_NCDHW);
  tensor_desc_5.SetDataType(DT_FLOAT);
  op1->AddInputDesc("ac", tensor_desc_5);

  // DHWNC
  vector<int64_t> dim_input_bc({});
  GeShape shape_6(dim_input_bc);
  GeTensorDesc tensor_desc_6(shape_6);
  tensor_desc_6.SetOriginFormat(FORMAT_NCDHW);
  tensor_desc_6.SetOriginShape(shape);
  tensor_desc_6.SetFormat(FORMAT_NCDHW);
  tensor_desc_6.SetDataType(DT_FLOAT);
  op1->AddInputDesc("bc", tensor_desc_6);

  op1->AddOutputDesc("z", tensor_x_desc);
  ge::AttrUtils::SetInt(op1, FE_IMPLY_TYPE, 6);
  ge::NodePtr node_1 = graph->AddNode(op1);
  Status ret1 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(node_1, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret1 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(node_1, "tbe-custom");
  ASSERT_EQ(ret1, fe::SUCCESS);

  vector<int64_t> dim_result_x({44});
  vector<int64_t> dim_result_w({1, 1, 44, 55, 1});
  vector<int64_t> dim_result_b({44, 55, 66, 1, 1});
  vector<int64_t> dim_result_c({55, 66, 77, 1, 44});
  vector<int64_t> dim_result_ac({66, 77, 88, 44, 55});
  vector<int64_t> dim_result_bc({});

  EXPECT_EQ(ge::GetPrimaryFormat(op1->GetInputDesc(0).GetFormat()), FORMAT_NCDHW);
  EXPECT_EQ(op1->GetInputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op1->GetInputDesc(0).GetShape().GetDims(), dim_result_x);

  EXPECT_EQ(ge::GetPrimaryFormat(op1->GetInputDesc(1).GetFormat()), FORMAT_NDHWC);
  EXPECT_EQ(op1->GetInputDesc(1).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op1->GetInputDesc(1).GetShape().GetDims(), dim_result_w);

  EXPECT_EQ(ge::GetPrimaryFormat(op1->GetInputDesc(2).GetFormat()), FORMAT_DHWCN);
  EXPECT_EQ(op1->GetInputDesc(2).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op1->GetInputDesc(2).GetShape().GetDims(), dim_result_b);

  EXPECT_EQ(ge::GetPrimaryFormat(op1->GetInputDesc(3).GetFormat()), FORMAT_DHWNC);
  EXPECT_EQ(op1->GetInputDesc(3).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op1->GetInputDesc(3).GetShape().GetDims(), dim_result_c);

  EXPECT_EQ(ge::GetPrimaryFormat(op1->GetInputDesc(4).GetFormat()), FORMAT_DHWNC);
  EXPECT_EQ(op1->GetInputDesc(4).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op1->GetInputDesc(4).GetShape().GetDims(), dim_result_ac);

  EXPECT_EQ(ge::GetPrimaryFormat(op1->GetInputDesc(5).GetFormat()), FORMAT_NCDHW);
  EXPECT_EQ(op1->GetInputDesc(5).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op1->GetInputDesc(5).GetShape().GetDims(), dim_result_bc);
}

TEST_F(UTEST_fusion_engine_op_judge_unittest_conv3d, test_opjudge_update_shape_NDHWC){
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr op1 = std::make_shared<OpDesc>("A1", "BatchNormV9");
  // x w b c ac bc
  // NCDHW
  vector<int64_t> dim_input_x({44});
  GeShape shape(dim_input_x);
  GeTensorDesc tensor_x_desc(shape);
  tensor_x_desc.SetOriginFormat(FORMAT_NDHWC);
  tensor_x_desc.SetOriginShape(shape);
  tensor_x_desc.SetFormat(FORMAT_NDHWC);
  tensor_x_desc.SetDataType(DT_FLOAT);
  op1->AddInputDesc("x", tensor_x_desc);

  // NDHWC
  vector<int64_t> dim_input_w({44, 55});
  GeShape shape_2(dim_input_w);
  GeTensorDesc tensor_desc_2(shape_2);
  tensor_desc_2.SetOriginFormat(FORMAT_NDHWC);
  tensor_desc_2.SetOriginShape(shape);
  tensor_desc_2.SetFormat(FORMAT_NDHWC);
  tensor_desc_2.SetDataType(DT_FLOAT);
  op1->AddInputDesc("w", tensor_desc_2);

  // DHWCN
  vector<int64_t> dim_input_b({44, 55, 66});
  GeShape shape_3(dim_input_b);
  GeTensorDesc tensor_desc_3(shape_3);
  tensor_desc_3.SetOriginFormat(FORMAT_NDHWC);
  tensor_desc_3.SetOriginShape(shape);
  tensor_desc_3.SetFormat(FORMAT_NDHWC);
  tensor_desc_3.SetDataType(DT_FLOAT);
  op1->AddInputDesc("b", tensor_desc_3);

  // DHWNC
  vector<int64_t> dim_input_c({44, 55, 66, 77});
  GeShape shape_4(dim_input_c);
  GeTensorDesc tensor_desc_4(shape_4);
  tensor_desc_4.SetOriginFormat(FORMAT_NDHWC);
  tensor_desc_4.SetOriginShape(shape);
  tensor_desc_4.SetFormat(FORMAT_NDHWC);
  tensor_desc_4.SetDataType(DT_FLOAT);
  op1->AddInputDesc("c", tensor_desc_4);

  // DHWNC
  vector<int64_t> dim_input_ac({44, 55, 66, 77, 88});
  GeShape shape_5(dim_input_ac);
  GeTensorDesc tensor_desc_5(shape_5);
  tensor_desc_5.SetOriginFormat(FORMAT_NDHWC);
  tensor_desc_5.SetOriginShape(shape);
  tensor_desc_5.SetFormat(FORMAT_NDHWC);
  tensor_desc_5.SetDataType(DT_FLOAT);
  op1->AddInputDesc("ac", tensor_desc_5);

  // DHWNC
  vector<int64_t> dim_input_bc({});
  GeShape shape_6(dim_input_bc);
  GeTensorDesc tensor_desc_6(shape_6);
  tensor_desc_6.SetOriginFormat(FORMAT_NDHWC);
  tensor_desc_6.SetOriginShape(shape);
  tensor_desc_6.SetFormat(FORMAT_NDHWC);
  tensor_desc_6.SetDataType(DT_FLOAT);
  op1->AddInputDesc("bc", tensor_desc_6);

  op1->AddOutputDesc("z", tensor_x_desc);
  ge::AttrUtils::SetInt(op1, FE_IMPLY_TYPE, 6);
  ge::NodePtr node_1 = graph->AddNode(op1);
  Status ret1 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(node_1, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret1 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(node_1, "tbe-custom");
  ASSERT_EQ(ret1, fe::SUCCESS);

  vector<int64_t> dim_result_x({1, 44, 1, 1, 1});
  vector<int64_t> dim_result_w({44, 55});
  vector<int64_t> dim_result_b({1, 44, 55, 66, 1});
  vector<int64_t> dim_result_c({44, 55, 66, 1, 77});
  vector<int64_t> dim_result_ac({55, 66, 77, 44, 88});
  vector<int64_t> dim_result_bc({});

  EXPECT_EQ(ge::GetPrimaryFormat(op1->GetInputDesc(0).GetFormat()), FORMAT_NCDHW);
  EXPECT_EQ(op1->GetInputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op1->GetInputDesc(0).GetShape().GetDims(), dim_result_x);

  EXPECT_EQ(ge::GetPrimaryFormat(op1->GetInputDesc(1).GetFormat()), FORMAT_NDHWC);
  EXPECT_EQ(op1->GetInputDesc(1).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op1->GetInputDesc(1).GetShape().GetDims(), dim_result_w);

  EXPECT_EQ(ge::GetPrimaryFormat(op1->GetInputDesc(2).GetFormat()), FORMAT_DHWCN);
  EXPECT_EQ(op1->GetInputDesc(2).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op1->GetInputDesc(2).GetShape().GetDims(), dim_result_b);

  EXPECT_EQ(ge::GetPrimaryFormat(op1->GetInputDesc(3).GetFormat()), FORMAT_DHWNC);
  EXPECT_EQ(op1->GetInputDesc(3).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op1->GetInputDesc(3).GetShape().GetDims(), dim_result_c);

  EXPECT_EQ(ge::GetPrimaryFormat(op1->GetInputDesc(4).GetFormat()), FORMAT_DHWNC);
  EXPECT_EQ(op1->GetInputDesc(4).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op1->GetInputDesc(4).GetShape().GetDims(), dim_result_ac);

  EXPECT_EQ(ge::GetPrimaryFormat(op1->GetInputDesc(5).GetFormat()), FORMAT_NDHWC);
  EXPECT_EQ(op1->GetInputDesc(5).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op1->GetInputDesc(5).GetShape().GetDims(), dim_result_bc);
}

TEST_F(UTEST_fusion_engine_op_judge_unittest_conv3d, test_opjudge_update_shape_DHWCN){
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr op1 = std::make_shared<OpDesc>("A1", "BatchNormV9");
  // x w b c ac bc
  // NCDHW
  vector<int64_t> dim_input_x({44});
  GeShape shape(dim_input_x);
  GeTensorDesc tensor_x_desc(shape);
  tensor_x_desc.SetOriginFormat(FORMAT_DHWCN);
  tensor_x_desc.SetOriginShape(shape);
  tensor_x_desc.SetFormat(FORMAT_DHWCN);
  tensor_x_desc.SetDataType(DT_FLOAT);
  op1->AddInputDesc("x", tensor_x_desc);

  // NDHWC
  vector<int64_t> dim_input_w({44, 55});
  GeShape shape_2(dim_input_w);
  GeTensorDesc tensor_desc_2(shape_2);
  tensor_desc_2.SetOriginFormat(FORMAT_DHWCN);
  tensor_desc_2.SetOriginShape(shape);
  tensor_desc_2.SetFormat(FORMAT_DHWCN);
  tensor_desc_2.SetDataType(DT_FLOAT);
  op1->AddInputDesc("w", tensor_desc_2);

  // DHWCN
  vector<int64_t> dim_input_b({44, 55, 66});
  GeShape shape_3(dim_input_b);
  GeTensorDesc tensor_desc_3(shape_3);
  tensor_desc_3.SetOriginFormat(FORMAT_DHWCN);
  tensor_desc_3.SetOriginShape(shape);
  tensor_desc_3.SetFormat(FORMAT_DHWCN);
  tensor_desc_3.SetDataType(DT_FLOAT);
  op1->AddInputDesc("b", tensor_desc_3);

  // DHWNC
  vector<int64_t> dim_input_c({44, 55, 66, 77});
  GeShape shape_4(dim_input_c);
  GeTensorDesc tensor_desc_4(shape_4);
  tensor_desc_4.SetOriginFormat(FORMAT_DHWCN);
  tensor_desc_4.SetOriginShape(shape);
  tensor_desc_4.SetFormat(FORMAT_DHWCN);
  tensor_desc_4.SetDataType(DT_FLOAT);
  op1->AddInputDesc("c", tensor_desc_4);

  // DHWNC
  vector<int64_t> dim_input_ac({44, 55, 66, 77, 88});
  GeShape shape_5(dim_input_ac);
  GeTensorDesc tensor_desc_5(shape_5);
  tensor_desc_5.SetOriginFormat(FORMAT_DHWCN);
  tensor_desc_5.SetOriginShape(shape);
  tensor_desc_5.SetFormat(FORMAT_DHWCN);
  tensor_desc_5.SetDataType(DT_FLOAT);
  op1->AddInputDesc("ac", tensor_desc_5);

  // DHWNC
  vector<int64_t> dim_input_bc({});
  GeShape shape_6(dim_input_bc);
  GeTensorDesc tensor_desc_6(shape_6);
  tensor_desc_6.SetOriginFormat(FORMAT_DHWCN);
  tensor_desc_6.SetOriginShape(shape);
  tensor_desc_6.SetFormat(FORMAT_DHWCN);
  tensor_desc_6.SetDataType(DT_FLOAT);
  op1->AddInputDesc("bc", tensor_desc_6);

  op1->AddOutputDesc("z", tensor_x_desc);
  ge::AttrUtils::SetInt(op1, FE_IMPLY_TYPE, 6);
  ge::NodePtr node_1 = graph->AddNode(op1);
  Status ret1 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(node_1, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret1 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(node_1, "tbe-custom");
  ASSERT_EQ(ret1, fe::SUCCESS);

  vector<int64_t> dim_result_x({1, 44, 1, 1, 1});
  vector<int64_t> dim_result_w({55, 1, 1, 1, 44});
  vector<int64_t> dim_result_b({44, 55, 66});
  vector<int64_t> dim_result_c({1, 44, 55, 77, 66});
  vector<int64_t> dim_result_ac({44, 55, 66, 88, 77});
  vector<int64_t> dim_result_bc({});

  EXPECT_EQ(ge::GetPrimaryFormat(op1->GetInputDesc(0).GetFormat()), FORMAT_NCDHW);
  EXPECT_EQ(op1->GetInputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op1->GetInputDesc(0).GetShape().GetDims(), dim_result_x);

  EXPECT_EQ(ge::GetPrimaryFormat(op1->GetInputDesc(1).GetFormat()), FORMAT_NDHWC);
  EXPECT_EQ(op1->GetInputDesc(1).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op1->GetInputDesc(1).GetShape().GetDims(), dim_result_w);

  EXPECT_EQ(ge::GetPrimaryFormat(op1->GetInputDesc(2).GetFormat()), FORMAT_DHWCN);
  EXPECT_EQ(op1->GetInputDesc(2).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op1->GetInputDesc(2).GetShape().GetDims(), dim_result_b);

  EXPECT_EQ(ge::GetPrimaryFormat(op1->GetInputDesc(3).GetFormat()), FORMAT_DHWNC);
  EXPECT_EQ(op1->GetInputDesc(3).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op1->GetInputDesc(3).GetShape().GetDims(), dim_result_c);

  EXPECT_EQ(ge::GetPrimaryFormat(op1->GetInputDesc(4).GetFormat()), FORMAT_DHWNC);
  EXPECT_EQ(op1->GetInputDesc(4).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op1->GetInputDesc(4).GetShape().GetDims(), dim_result_ac);

  EXPECT_EQ(ge::GetPrimaryFormat(op1->GetInputDesc(5).GetFormat()), FORMAT_DHWCN);
  EXPECT_EQ(op1->GetInputDesc(5).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op1->GetInputDesc(5).GetShape().GetDims(), dim_result_bc);
}

TEST_F(UTEST_fusion_engine_op_judge_unittest_conv3d, test_opjudge_update_shape_DHWNC){
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr op1 = std::make_shared<OpDesc>("A1", "BatchNormV9");
  // x w b c ac bc
  // NCDHW
  vector<int64_t> dim_input_x({44});
  GeShape shape(dim_input_x);
  GeTensorDesc tensor_x_desc(shape);
  tensor_x_desc.SetOriginFormat(FORMAT_DHWNC);
  tensor_x_desc.SetOriginShape(shape);
  tensor_x_desc.SetFormat(FORMAT_DHWNC);
  tensor_x_desc.SetDataType(DT_FLOAT);
  op1->AddInputDesc("x", tensor_x_desc);

  // NDHWC
  vector<int64_t> dim_input_w({44, 55});
  GeShape shape_2(dim_input_w);
  GeTensorDesc tensor_desc_2(shape_2);
  tensor_desc_2.SetOriginFormat(FORMAT_DHWNC);
  tensor_desc_2.SetOriginShape(shape);
  tensor_desc_2.SetFormat(FORMAT_DHWNC);
  tensor_desc_2.SetDataType(DT_FLOAT);
  op1->AddInputDesc("w", tensor_desc_2);

  // DHWCN
  vector<int64_t> dim_input_b({44, 55, 66});
  GeShape shape_3(dim_input_b);
  GeTensorDesc tensor_desc_3(shape_3);
  tensor_desc_3.SetOriginFormat(FORMAT_DHWNC);
  tensor_desc_3.SetOriginShape(shape);
  tensor_desc_3.SetFormat(FORMAT_DHWNC);
  tensor_desc_3.SetDataType(DT_FLOAT);
  op1->AddInputDesc("b", tensor_desc_3);

  // DHWNC
  vector<int64_t> dim_input_c({44, 55, 66, 77});
  GeShape shape_4(dim_input_c);
  GeTensorDesc tensor_desc_4(shape_4);
  tensor_desc_4.SetOriginFormat(FORMAT_DHWNC);
  tensor_desc_4.SetOriginShape(shape);
  tensor_desc_4.SetFormat(FORMAT_DHWNC);
  tensor_desc_4.SetDataType(DT_FLOAT);
  op1->AddInputDesc("c", tensor_desc_4);

  // DHWNC
  vector<int64_t> dim_input_ac({44, 55, 66, 77, 88});
  GeShape shape_5(dim_input_ac);
  GeTensorDesc tensor_desc_5(shape_5);
  tensor_desc_5.SetOriginFormat(FORMAT_DHWNC);
  tensor_desc_5.SetOriginShape(shape);
  tensor_desc_5.SetFormat(FORMAT_DHWNC);
  tensor_desc_5.SetDataType(DT_FLOAT);
  op1->AddInputDesc("ac", tensor_desc_5);

  // DHWNC
  vector<int64_t> dim_input_bc({});
  GeShape shape_6(dim_input_bc);
  GeTensorDesc tensor_desc_6(shape_6);
  tensor_desc_6.SetOriginFormat(FORMAT_DHWNC);
  tensor_desc_6.SetOriginShape(shape);
  tensor_desc_6.SetFormat(FORMAT_DHWNC);
  tensor_desc_6.SetDataType(DT_FLOAT);
  op1->AddInputDesc("bc", tensor_desc_6);

  op1->AddOutputDesc("z", tensor_x_desc);
  ge::AttrUtils::SetInt(op1, FE_IMPLY_TYPE, 6);
  ge::NodePtr node_1 = graph->AddNode(op1);
  Status ret1 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(node_1, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret1 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(node_1, "tbe-custom");
  ASSERT_EQ(ret1, fe::SUCCESS);

  vector<int64_t> dim_result_x({1, 44, 1, 1, 1});
  vector<int64_t> dim_result_w({44, 1, 1, 1, 55});
  vector<int64_t> dim_result_b({1, 1, 44, 66, 55});
  vector<int64_t> dim_result_c({44, 55, 66, 77});
  vector<int64_t> dim_result_ac({44, 55, 66, 77, 88});
  vector<int64_t> dim_result_bc({});

  EXPECT_EQ(ge::GetPrimaryFormat(op1->GetInputDesc(0).GetFormat()), FORMAT_NCDHW);
  EXPECT_EQ(op1->GetInputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op1->GetInputDesc(0).GetShape().GetDims(), dim_result_x);

  EXPECT_EQ(ge::GetPrimaryFormat(op1->GetInputDesc(1).GetFormat()), FORMAT_NDHWC);
  EXPECT_EQ(op1->GetInputDesc(1).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op1->GetInputDesc(1).GetShape().GetDims(), dim_result_w);

  EXPECT_EQ(ge::GetPrimaryFormat(op1->GetInputDesc(2).GetFormat()), FORMAT_DHWCN);
  EXPECT_EQ(op1->GetInputDesc(2).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op1->GetInputDesc(2).GetShape().GetDims(), dim_result_b);

  EXPECT_EQ(ge::GetPrimaryFormat(op1->GetInputDesc(3).GetFormat()), FORMAT_DHWNC);
  EXPECT_EQ(op1->GetInputDesc(3).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op1->GetInputDesc(3).GetShape().GetDims(), dim_result_c);

  EXPECT_EQ(ge::GetPrimaryFormat(op1->GetInputDesc(4).GetFormat()), FORMAT_DHWNC);
  EXPECT_EQ(op1->GetInputDesc(4).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op1->GetInputDesc(4).GetShape().GetDims(), dim_result_ac);

  EXPECT_EQ(ge::GetPrimaryFormat(op1->GetInputDesc(5).GetFormat()), FORMAT_DHWNC);
  EXPECT_EQ(op1->GetInputDesc(5).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op1->GetInputDesc(5).GetShape().GetDims(), dim_result_bc);
}

TEST_F(UTEST_fusion_engine_op_judge_unittest_conv3d, test_unknown_shape_01){
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr op1 = std::make_shared<OpDesc>("Conv3D", "Conv3D");
  OpDescPtr op2 = std::make_shared<OpDesc>("A1", "A");
  //add descriptor
  vector<int64_t> dim_input({4, -1, -1, -1, -1});
  vector<std::pair<int64_t, int64_t>> range1({{1, 33}, {1, 12}, {1, 16}, {1, 64}});
  GeShape shape(dim_input);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NDHWC);
  tensor_desc.SetOriginShape(shape);
  tensor_desc.SetFormat(FORMAT_NDHWC);
  tensor_desc.SetDataType(DT_FLOAT);
  tensor_desc.SetShapeRange(range1);
  op1->AddInputDesc("x", tensor_desc);

  vector<int64_t> dim_weight({3, -1, -1, -1, -1});
  vector<std::pair<int64_t, int64_t>> range2({{1, 3}, {1, 3}, {1, 4}, {1, 5}});
  GeShape weight_shape(dim_weight);
  GeTensorDesc weight_desc(weight_shape);
  weight_desc.SetOriginFormat(FORMAT_NDHWC);
  weight_desc.SetOriginShape(weight_shape);
  weight_desc.SetFormat(FORMAT_NDHWC);
  weight_desc.SetDataType(DT_FLOAT);
  weight_desc.SetShapeRange(range2);
  op1->AddInputDesc("y", weight_desc);

  op1->AddOutputDesc("z", tensor_desc);
  ge::AttrUtils::SetInt(op1, FE_IMPLY_TYPE, 6);
  ge::NodePtr node1 = graph->AddNode(op1);

  op2->AddInputDesc("x", tensor_desc);
  op2->AddOutputDesc("z", tensor_desc);
  ge::AttrUtils::SetInt(op2, FE_IMPLY_TYPE, 6);
  ge::NodePtr node2 = graph->AddNode(op2);
  GraphUtils::AddEdge(node1->GetOutDataAnchor(0), node2->GetInDataAnchor(0));

  Status ret1 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(node1, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret1 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(node1, "tbe-custom");
  Status ret2 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(node2, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret2 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(node2, "tbe-custom");
  ASSERT_EQ(ret1, fe::SUCCESS);
  ASSERT_EQ(ret2, fe::SUCCESS);

  vector<int64_t> dim_result6_d({4, -1, -1, -1, -1, 16});
  vector<int64_t> dim_result_fz({-1, 1, 16, 16});
  vector<std::pair<int64_t, int64_t>> range_ex1({{4, 4}, {1, 33}, {1, 4}, {1, 12}, {1, 16}, {16, 16}});
  vector<std::pair<int64_t, int64_t>> range_ex2({{1, 36}, {1, 1}, {16, 16}, {16, 16}});
  EXPECT_EQ(ge::GetPrimaryFormat(op1->GetInputDesc(0).GetFormat()), FORMAT_NDC1HWC0);
  EXPECT_EQ(op1->GetInputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op1->GetInputDesc(0).GetShape().GetDims(), dim_result6_d);
  EXPECT_EQ(GetShapeRange(op1->GetInputDesc(0)), range_ex1);

  EXPECT_EQ(ge::GetPrimaryFormat(op1->GetInputDesc(1).GetFormat()), FORMAT_FRACTAL_Z_3D);
  EXPECT_EQ(op1->GetInputDesc(1).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op1->GetInputDesc(1).GetShape().GetDims(), dim_result_fz);
  EXPECT_EQ(GetShapeRange(op1->GetInputDesc(1)), range_ex2);

  EXPECT_EQ(ge::GetPrimaryFormat(op1->GetOutputDesc(0).GetFormat()), FORMAT_NDC1HWC0);
  EXPECT_EQ(op1->GetOutputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op1->GetOutputDesc(0).GetShape().GetDims(), dim_result6_d);
  EXPECT_EQ(GetShapeRange(op1->GetOutputDesc(0)), range_ex1);

  EXPECT_EQ(op2->GetInputDesc(0).GetFormat(), FORMAT_NDHWC);
  EXPECT_EQ(op2->GetInputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op2->GetInputDesc(0).GetShape().GetDims(), dim_input);
  EXPECT_EQ(GetShapeRange(op2->GetInputDesc(0)), range1);
  EXPECT_EQ(op2->GetOutputDesc(0).GetFormat(), FORMAT_NDHWC);
  EXPECT_EQ(op2->GetOutputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op2->GetOutputDesc(0).GetShape().GetDims(), dim_input);
  EXPECT_EQ(GetShapeRange(op2->GetOutputDesc(0)), range1);
}

TEST_F(UTEST_fusion_engine_op_judge_unittest_conv3d, test_unknown_shape_02){
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr op1 = std::make_shared<OpDesc>("Conv3DTranspose", "Conv3DTranspose");
  OpDescPtr op2 = std::make_shared<OpDesc>("A1", "A");
  //add descriptor
  vector<int64_t> dim_input({4, -1, -1, -1, -1});
  vector<std::pair<int64_t, int64_t>> range1({{1, 33}, {1, 12}, {1, 16}, {1, 64}});
  GeShape shape(dim_input);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NDHWC);
  tensor_desc.SetOriginShape(shape);
  tensor_desc.SetFormat(FORMAT_NDHWC);
  tensor_desc.SetDataType(DT_FLOAT);
  tensor_desc.SetShapeRange(range1);
  op1->AddInputDesc("x", tensor_desc);

  vector<int64_t> dim_weight({3, -1, -1, -1, -1});
  vector<std::pair<int64_t, int64_t>> range2({{1, 3}, {1, 3}, {1, 4}, {1, 5}});
  GeShape weight_shape(dim_weight);
  GeTensorDesc weight_desc(weight_shape);
  weight_desc.SetOriginFormat(FORMAT_DHWNC);
  weight_desc.SetOriginShape(weight_shape);
  weight_desc.SetFormat(FORMAT_DHWNC);
  weight_desc.SetDataType(DT_FLOAT);
  weight_desc.SetShapeRange(range2);
  op1->AddInputDesc("y", weight_desc);

  op1->AddOutputDesc("z", tensor_desc);
  ge::AttrUtils::SetInt(op1, FE_IMPLY_TYPE, 6);
  ge::NodePtr node1 = graph->AddNode(op1);

  op2->AddInputDesc("x", tensor_desc);
  op2->AddOutputDesc("z", tensor_desc);
  ge::AttrUtils::SetInt(op2, FE_IMPLY_TYPE, 6);
  ge::NodePtr node2 = graph->AddNode(op2);
  GraphUtils::AddEdge(node1->GetOutDataAnchor(0), node2->GetInDataAnchor(0));

  Status ret1 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(node1, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret1 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(node1, "tbe-custom");
  Status ret2 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(node2, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret2 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(node2, "tbe-custom");
  ASSERT_EQ(ret1, fe::SUCCESS);
  ASSERT_EQ(ret2, fe::SUCCESS);

  vector<int64_t> dim_result6_d({4, -1, -1, -1, -1, 16});
  vector<int64_t> dim_result_fz({-1, -1, 16, 16});
  vector<std::pair<int64_t, int64_t>> range_ex1({{4, 4}, {1, 33}, {1, 4}, {1, 12}, {1, 16}, {16, 16}});
  vector<std::pair<int64_t, int64_t>> range_ex2({{3, 27}, {1, 1}, {16, 16}, {16, 16}});
  EXPECT_EQ(ge::GetPrimaryFormat(op1->GetInputDesc(0).GetFormat()), FORMAT_NDC1HWC0);
  EXPECT_EQ(op1->GetInputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op1->GetInputDesc(0).GetShape().GetDims(), dim_result6_d);
  EXPECT_EQ(GetShapeRange(op1->GetInputDesc(0)), range_ex1);

  EXPECT_EQ(ge::GetPrimaryFormat(op1->GetInputDesc(1).GetFormat()), FORMAT_FRACTAL_Z_3D_TRANSPOSE);
  EXPECT_EQ(op1->GetInputDesc(1).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op1->GetInputDesc(1).GetShape().GetDims(), dim_result_fz);
  EXPECT_EQ(GetShapeRange(op1->GetInputDesc(1)), range_ex2);

  EXPECT_EQ(ge::GetPrimaryFormat(op1->GetOutputDesc(0).GetFormat()), FORMAT_NDC1HWC0);
  EXPECT_EQ(op1->GetOutputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op1->GetOutputDesc(0).GetShape().GetDims(), dim_result6_d);
  EXPECT_EQ(GetShapeRange(op1->GetOutputDesc(0)), range_ex1);

  EXPECT_EQ(op2->GetInputDesc(0).GetFormat(), FORMAT_NDHWC);
  EXPECT_EQ(op2->GetInputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op2->GetInputDesc(0).GetShape().GetDims(), dim_input);
  EXPECT_EQ(GetShapeRange(op2->GetInputDesc(0)), range1);
  EXPECT_EQ(op2->GetOutputDesc(0).GetFormat(), FORMAT_NDHWC);
  EXPECT_EQ(op2->GetOutputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op2->GetOutputDesc(0).GetShape().GetDims(), dim_input);
  EXPECT_EQ(GetShapeRange(op2->GetOutputDesc(0)), range1);
}

TEST_F(UTEST_fusion_engine_op_judge_unittest_conv3d, test_unknown_shape_03){
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr op1 = std::make_shared<OpDesc>("Conv3D", "Conv3D");
  OpDescPtr op2 = std::make_shared<OpDesc>("A1", "A");
  //add descriptor
  vector<int64_t> dim_input({4, -1, -1, -1, -1});
  vector<std::pair<int64_t, int64_t>> range1({{1, 33}, {1, 12}, {1, 16}, {1, 64}});
  GeShape shape(dim_input);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NDHWC);
  tensor_desc.SetOriginShape(shape);
  tensor_desc.SetFormat(FORMAT_NDHWC);
  tensor_desc.SetDataType(DT_FLOAT);
  tensor_desc.SetShapeRange(range1);
  op1->AddInputDesc("x", tensor_desc);

  vector<int64_t> dim_weight({3, -1, -1, -1, -1});
  vector<std::pair<int64_t, int64_t>> range2({{1, 3}, {1, 3}, {1, 4}, {1, 5}});
  GeShape weight_shape(dim_weight);
  GeTensorDesc weight_desc(weight_shape);
  weight_desc.SetOriginFormat(FORMAT_DHWCN);
  weight_desc.SetOriginShape(weight_shape);
  weight_desc.SetFormat(FORMAT_DHWCN);
  weight_desc.SetDataType(DT_FLOAT);
  weight_desc.SetShapeRange(range2);
  op1->AddInputDesc("y", weight_desc);

  op1->AddOutputDesc("z", tensor_desc);
  ge::AttrUtils::SetInt(op1, FE_IMPLY_TYPE, 6);
  ge::NodePtr node1 = graph->AddNode(op1);

  op2->AddInputDesc("x", tensor_desc);
  op2->AddOutputDesc("z", tensor_desc);
  ge::AttrUtils::SetInt(op2, FE_IMPLY_TYPE, 6);
  ge::NodePtr node2 = graph->AddNode(op2);
  GraphUtils::AddEdge(node1->GetOutDataAnchor(0), node2->GetInDataAnchor(0));

  Status ret1 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(node1, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret1 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(node1, "tbe-custom");
  Status ret2 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(node2, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret2 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(node2, "tbe-custom");
  ASSERT_EQ(ret1, fe::SUCCESS);
  ASSERT_EQ(ret2, fe::SUCCESS);

  vector<int64_t> dim_result6_d({4, -1, -1, -1, -1, 16});
  vector<int64_t> dim_result_fz({-1, -1, 16, 16});
  vector<std::pair<int64_t, int64_t>> range_ex1({{4, 4}, {1, 33}, {1, 4}, {1, 12}, {1, 16}, {16, 16}});
  vector<std::pair<int64_t, int64_t>> range_ex2({{3, 27}, {1, 1}, {16, 16}, {16, 16}});
  EXPECT_EQ(ge::GetPrimaryFormat(op1->GetInputDesc(0).GetFormat()), FORMAT_NDC1HWC0);
  EXPECT_EQ(op1->GetInputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op1->GetInputDesc(0).GetShape().GetDims(), dim_result6_d);
  EXPECT_EQ(GetShapeRange(op1->GetInputDesc(0)), range_ex1);

  EXPECT_EQ(ge::GetPrimaryFormat(op1->GetInputDesc(1).GetFormat()), FORMAT_FRACTAL_Z_3D);
  EXPECT_EQ(op1->GetInputDesc(1).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op1->GetInputDesc(1).GetShape().GetDims(), dim_result_fz);
  EXPECT_EQ(GetShapeRange(op1->GetInputDesc(1)), range_ex2);

  EXPECT_EQ(ge::GetPrimaryFormat(op1->GetOutputDesc(0).GetFormat()), FORMAT_NDC1HWC0);
  EXPECT_EQ(op1->GetOutputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op1->GetOutputDesc(0).GetShape().GetDims(), dim_result6_d);
  EXPECT_EQ(GetShapeRange(op1->GetOutputDesc(0)), range_ex1);

  EXPECT_EQ(op2->GetInputDesc(0).GetFormat(), FORMAT_NDHWC);
  EXPECT_EQ(op2->GetInputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op2->GetInputDesc(0).GetShape().GetDims(), dim_input);
  EXPECT_EQ(GetShapeRange(op2->GetInputDesc(0)), range1);
  EXPECT_EQ(op2->GetOutputDesc(0).GetFormat(), FORMAT_NDHWC);
  EXPECT_EQ(op2->GetOutputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op2->GetOutputDesc(0).GetShape().GetDims(), dim_input);
  EXPECT_EQ(GetShapeRange(op2->GetOutputDesc(0)), range1);
}

TEST_F(UTEST_fusion_engine_op_judge_unittest_conv3d, test_unknown_shape_04){
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr op1 = std::make_shared<OpDesc>("Conv3D", "Conv3D");
  OpDescPtr op2 = std::make_shared<OpDesc>("A1", "A");
  //add descriptor
  vector<int64_t> dim_input({4, -1, -1, -1, -1});
  vector<std::pair<int64_t, int64_t>> range1({{1, 33}, {1, 12}, {1, 16}, {1, 64}});
  GeShape shape(dim_input);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NCDHW);
  tensor_desc.SetOriginShape(shape);
  tensor_desc.SetFormat(FORMAT_NCDHW);
  tensor_desc.SetDataType(DT_FLOAT);
  tensor_desc.SetShapeRange(range1);
  op1->AddInputDesc("x", tensor_desc);

  vector<int64_t> dim_weight({3, -1, -1, -1, -1});
  vector<std::pair<int64_t, int64_t>> range2({{1, 3}, {1, 3}, {1, 4}, {1, 5}});
  GeShape weight_shape(dim_weight);
  GeTensorDesc weight_desc(weight_shape);
  weight_desc.SetOriginFormat(FORMAT_NCDHW);
  weight_desc.SetOriginShape(weight_shape);
  weight_desc.SetFormat(FORMAT_NCDHW);
  weight_desc.SetDataType(DT_FLOAT);
  weight_desc.SetShapeRange(range2);
  op1->AddInputDesc("y", weight_desc);

  op1->AddOutputDesc("z", tensor_desc);
  ge::AttrUtils::SetInt(op1, FE_IMPLY_TYPE, 6);
  ge::NodePtr node1 = graph->AddNode(op1);

  op2->AddInputDesc("x", tensor_desc);
  op2->AddOutputDesc("z", tensor_desc);
  ge::AttrUtils::SetInt(op2, FE_IMPLY_TYPE, 6);
  ge::NodePtr node2 = graph->AddNode(op2);
  GraphUtils::AddEdge(node1->GetOutDataAnchor(0), node2->GetInDataAnchor(0));
  
  Status ret1 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(node1, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret1 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(node1, "tbe-custom");
  Status ret2 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(node2, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret2 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(node2, "tbe-custom");
  ASSERT_EQ(ret1, fe::SUCCESS);
  ASSERT_EQ(ret2, fe::SUCCESS);

  vector<int64_t> dim_result6_d({4, -1, -1, -1, -1, 16});
  vector<int64_t> dim_result_fz({-1, 1, 16, 16});
  vector<std::pair<int64_t, int64_t>> range_ex1({{4, 4}, {1, 12}, {1, 3}, {1, 16}, {1, 64}, {16, 16}});
  vector<std::pair<int64_t, int64_t>> range_ex2({{1, 60}, {1, 1}, {16, 16}, {16, 16}});
  EXPECT_EQ(ge::GetPrimaryFormat(op1->GetInputDesc(0).GetFormat()), FORMAT_NDC1HWC0);
  EXPECT_EQ(op1->GetInputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op1->GetInputDesc(0).GetShape().GetDims(), dim_result6_d);
  EXPECT_EQ(GetShapeRange(op1->GetInputDesc(0)), range_ex1);

  EXPECT_EQ(ge::GetPrimaryFormat(op1->GetInputDesc(1).GetFormat()), FORMAT_FRACTAL_Z_3D);
  EXPECT_EQ(op1->GetInputDesc(1).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op1->GetInputDesc(1).GetShape().GetDims(), dim_result_fz);
  EXPECT_EQ(GetShapeRange(op1->GetInputDesc(1)), range_ex2);

  EXPECT_EQ(ge::GetPrimaryFormat(op1->GetOutputDesc(0).GetFormat()), FORMAT_NDC1HWC0);
  EXPECT_EQ(op1->GetOutputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op1->GetOutputDesc(0).GetShape().GetDims(), dim_result6_d);
  EXPECT_EQ(GetShapeRange(op1->GetOutputDesc(0)), range_ex1);

  EXPECT_EQ(op2->GetInputDesc(0).GetFormat(), FORMAT_NCDHW);
  EXPECT_EQ(op2->GetInputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op2->GetInputDesc(0).GetShape().GetDims(), dim_input);
  EXPECT_EQ(GetShapeRange(op2->GetInputDesc(0)), range1);
  EXPECT_EQ(op2->GetOutputDesc(0).GetFormat(), FORMAT_NCDHW);
  EXPECT_EQ(op2->GetOutputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op2->GetOutputDesc(0).GetShape().GetDims(), dim_input);
  EXPECT_EQ(GetShapeRange(op2->GetOutputDesc(0)), range1);
}

TEST_F(UTEST_fusion_engine_op_judge_unittest_conv3d, test_unknown_shape_05){
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr op1 = std::make_shared<OpDesc>("Conv3D", "Conv3D");
  OpDescPtr op2 = std::make_shared<OpDesc>("A1", "A");
  //add descriptor
  vector<int64_t> dim_input({4, -1, -1, -1, -1});
  vector<std::pair<int64_t, int64_t>> range1({{1, 33}, {1, 12}, {1, 16}, {1, 64}});
  GeShape shape(dim_input);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NDHWC);
  tensor_desc.SetOriginShape(shape);
  tensor_desc.SetFormat(FORMAT_NDHWC);
  tensor_desc.SetDataType(DT_FLOAT);
  tensor_desc.SetShapeRange(range1);
  op1->AddInputDesc("x", tensor_desc);

  vector<int64_t> dim_weight({3, -1, -1, -1, -1});
  vector<std::pair<int64_t, int64_t>> range2({{1, 3}, {1, 3}, {1, 4}, {1, 5}});
  GeShape weight_shape(dim_weight);
  GeTensorDesc weight_desc(weight_shape);
  weight_desc.SetOriginFormat(FORMAT_NDHWC);
  weight_desc.SetOriginShape(weight_shape);
  weight_desc.SetFormat(FORMAT_NDHWC);
  weight_desc.SetDataType(DT_FLOAT);
  weight_desc.SetShapeRange(range2);
  op1->AddInputDesc("y", weight_desc);

  op1->AddOutputDesc("z", tensor_desc);
  ge::AttrUtils::SetInt(op1, FE_IMPLY_TYPE, 6);
  ge::NodePtr node1 = graph->AddNode(op1);

  op2->AddInputDesc("x", tensor_desc);
  op2->AddOutputDesc("z", tensor_desc);
  ge::AttrUtils::SetInt(op2, FE_IMPLY_TYPE, 6);
  ge::NodePtr node2 = graph->AddNode(op2);
  GraphUtils::AddEdge(node1->GetOutDataAnchor(0), node2->GetInDataAnchor(0));

  Status ret1 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(node1, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret1 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(node1, "tbe-custom");
  Status ret2 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(node2, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret2 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(node2, "tbe-custom");
  ASSERT_EQ(ret1, fe::SUCCESS);
  ASSERT_EQ(ret2, fe::SUCCESS);

  vector<int64_t> dim_result6_d({4, -1, -1, -1, -1, 16});
  vector<int64_t> dim_result_fz({-1, 1, 16, 16});
  vector<std::pair<int64_t, int64_t>> range_ex1({{4, 4}, {1, 33}, {1, 4}, {1, 12}, {1, 16}, {16, 16}});
  vector<std::pair<int64_t, int64_t>> range_ex2({{1, 36}, {1, 1}, {16, 16}, {16, 16}});
  EXPECT_EQ(ge::GetPrimaryFormat(op1->GetInputDesc(0).GetFormat()), FORMAT_NDC1HWC0);
  EXPECT_EQ(op1->GetInputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op1->GetInputDesc(0).GetShape().GetDims(), dim_result6_d);
  EXPECT_EQ(GetShapeRange(op1->GetInputDesc(0)), range_ex1);

  EXPECT_EQ(ge::GetPrimaryFormat(op1->GetInputDesc(1).GetFormat()), FORMAT_FRACTAL_Z_3D);
  EXPECT_EQ(op1->GetInputDesc(1).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op1->GetInputDesc(1).GetShape().GetDims(), dim_result_fz);
  EXPECT_EQ(GetShapeRange(op1->GetInputDesc(1)), range_ex2);

  EXPECT_EQ(ge::GetPrimaryFormat(op1->GetOutputDesc(0).GetFormat()), FORMAT_NDC1HWC0);
  EXPECT_EQ(op1->GetOutputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op1->GetOutputDesc(0).GetShape().GetDims(), dim_result6_d);
  EXPECT_EQ(GetShapeRange(op1->GetOutputDesc(0)), range_ex1);

  EXPECT_EQ(op2->GetInputDesc(0).GetFormat(), FORMAT_NDHWC);
  EXPECT_EQ(op2->GetInputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op2->GetInputDesc(0).GetShape().GetDims(), dim_input);
  EXPECT_EQ(GetShapeRange(op2->GetInputDesc(0)), range1);
  EXPECT_EQ(op2->GetOutputDesc(0).GetFormat(), FORMAT_NDHWC);
  EXPECT_EQ(op2->GetOutputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op2->GetOutputDesc(0).GetShape().GetDims(), dim_input);
  EXPECT_EQ(GetShapeRange(op2->GetOutputDesc(0)), range1);
}

TEST_F(UTEST_fusion_engine_op_judge_unittest_conv3d, update_newshape_and_format_suc) {
  OpKernelInfoPtr op_kernel_info_ptr;
  InputOrOutputInfoPtr in_desc_ptr = std::make_shared<fe::InputOrOutputInfo>("x");
  in_desc_ptr->supported_formats_.emplace_back(FORMAT_NC1HWC0);
  in_desc_ptr->supported_formats_.emplace_back(FORMAT_NCHW);
  in_desc_ptr->supported_dtypes_.emplace_back(DT_FLOAT);
  in_desc_ptr->supported_dtypes_.emplace_back(DT_FLOAT16);
  in_desc_ptr->supported_sub_formats_.emplace_back(32);

  uint32_t matched_index;
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr op_desc_0 = std::make_shared<OpDesc>("add", "Add");
  vector<int64_t> dim(4, 4);
  GeShape shape(dim);
  GeTensorDesc out_desc(shape);
  op_desc_0->AddInputDesc(out_desc);
  op_desc_0->AddOutputDesc(out_desc);
  NodePtr node_ptr = graph->AddNode(op_desc_0);
  uint32_t index;
  ge::GeTensorDesc op_input_or_output_desc;
  bool is_input;
  UpdateInfo update_info = {op_kernel_info_ptr, in_desc_ptr, matched_index, node_ptr, index, op_input_or_output_desc, is_input}; 
  ge::Format op_kernel_format = ge::FORMAT_FRACTAL_Z; 
  int64_t group = 32;
  ge::GeShape original_shape; 
  ge::GeShape new_shape;
  std::string op_type;
  std::string op_name;
  Status ret = fe::UpdateNewShapeAndFormat(update_info, op_kernel_format, group, original_shape, new_shape, op_type, op_name);
  ge::Format new_format = update_info.op_input_or_output_desc.GetFormat();
  int32_t sub_format = ge::GetSubFormat(new_format);
  EXPECT_EQ(sub_format, 32);
  EXPECT_EQ(ret, ge::SUCCESS);
}

TEST_F(UTEST_fusion_engine_op_judge_unittest_conv3d, update_newshape_and_format_fail) {
  OpKernelInfoPtr op_kernel_info_ptr;
  InputOrOutputInfoPtr in_desc_ptr = std::make_shared<fe::InputOrOutputInfo>("x");
  in_desc_ptr->supported_formats_.emplace_back(FORMAT_NC1HWC0);
  in_desc_ptr->supported_formats_.emplace_back(FORMAT_NCHW);
  in_desc_ptr->supported_dtypes_.emplace_back(DT_FLOAT);
  in_desc_ptr->supported_dtypes_.emplace_back(DT_FLOAT16);
  in_desc_ptr->supported_sub_formats_.emplace_back(32);

  uint32_t matched_index;
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr op_desc_0 = std::make_shared<OpDesc>("add", "Add");
  vector<int64_t> dim(4, 4);
  GeShape shape(dim);
  GeTensorDesc out_desc(shape);
  op_desc_0->AddInputDesc(out_desc);
  op_desc_0->AddOutputDesc(out_desc);
  NodePtr node_ptr = graph->AddNode(op_desc_0);
  uint32_t index;
  ge::GeTensorDesc op_input_or_output_desc;
  bool is_input;
  UpdateInfo update_info = {op_kernel_info_ptr, in_desc_ptr, matched_index, node_ptr, index, op_input_or_output_desc, is_input};
  ge::Format op_kernel_format = ge::FORMAT_FRACTAL_Z;
  int64_t group = 2;
  ge::GeShape original_shape;
  ge::GeShape new_shape;
  std::string op_type;
  std::string op_name;
  Status ret = fe::UpdateNewShapeAndFormat(update_info, op_kernel_format, group, original_shape, new_shape, op_type, op_name);
  ge::Format new_format = update_info.op_input_or_output_desc.GetFormat();
  EXPECT_EQ(ret, ge::FAILED);
}