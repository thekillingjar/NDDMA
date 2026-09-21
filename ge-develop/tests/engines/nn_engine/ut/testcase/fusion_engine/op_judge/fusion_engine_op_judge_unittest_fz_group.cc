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
#include "fe_llt_utils.h"
#include "common/util/op_info_util.h"
#include "common/aicore_util_attr_define.h"
#define private public
#define protected public
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/attr_utils.h"
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

class UTEST_fusion_engine_op_judge_unittest_fz_group : public testing::Test {
 protected:

  void SetUp() {
    std::map<std::string, std::string> options;
    fe_ops_kernel_info_store_ptr_ = make_shared<fe::FEOpsKernelInfoStore>();

    FEOpsStoreInfo tbe_custom{
        6,
        "tbe-custom",
        EN_IMPL_HW_TBE,
        GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/tbe_group",
        "",
        false,
        false,
        false};
    vector<FEOpsStoreInfo> store_info;

    store_info.emplace_back(tbe_custom);
    Configuration::Instance(fe::AI_CORE_NAME).ops_store_info_vector_ = (store_info);
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

TEST_F(UTEST_fusion_engine_op_judge_unittest_fz_group, test_01){
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr op1 = std::make_shared<OpDesc>("A1", "A");
  OpDescPtr op2 = std::make_shared<OpDesc>("B1", "B");
  //add descriptor
  int64_t op1_groups = 2;
  int64_t op2_groups = 1;
  vector<int64_t> dim_input({40, 25, 7, 7});
  GeShape shape(dim_input);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NCHW);
  tensor_desc.SetOriginShape(shape);
  tensor_desc.SetFormat(FORMAT_NCHW);

  tensor_desc.SetDataType(DT_FLOAT);
  op1->AddInputDesc("x", tensor_desc);

  /*1, 1, i+h, 4*h
   * h = 562, i = 672*/
  vector<int64_t> dim_weight({7, 7, 25, 40});
  GeShape weight_shape(dim_weight);
  GeTensorDesc weight_desc(weight_shape);
  weight_desc.SetOriginFormat(FORMAT_HWCN);
  weight_desc.SetOriginShape(weight_shape);
  weight_desc.SetFormat(FORMAT_HWCN);
  weight_desc.SetDataType(DT_FLOAT);
  op1->AddInputDesc("y", weight_desc);

  vector<int64_t> dim_out({40, 7, 7, 25});
  GeShape out_shape(dim_out);
  GeTensorDesc out_desc(out_shape);
  out_desc.SetOriginFormat(FORMAT_NHWC);
  out_desc.SetOriginShape(out_shape);

  out_desc.SetFormat(FORMAT_NHWC);
  out_desc.SetDataType(DT_FLOAT);
  op1->AddOutputDesc("z", out_desc);
  ge::AttrUtils::SetInt(op1, FE_IMPLY_TYPE, 6);
  ge::AttrUtils::SetInt(op1, ATTR_NAME_GROUPS, op1_groups);
  ge::NodePtr node1 = graph->AddNode(op1);

  vector<int64_t> dim_input_b({4, 33, 44, 10});
  GeShape bshape(dim_input_b);
  GeTensorDesc b_tensor_desc(bshape);
  b_tensor_desc.SetOriginFormat(FORMAT_NHWC);
  b_tensor_desc.SetOriginShape(bshape);
  b_tensor_desc.SetFormat(FORMAT_NHWC);
  b_tensor_desc.SetDataType(DT_FLOAT);
  op2->AddInputDesc("x", b_tensor_desc);

  vector<int64_t> dim_input_c({4, 10, 33, 44});
  GeShape cshape(dim_input_c);
  GeTensorDesc c_tensor_desc(cshape);
  c_tensor_desc.SetOriginFormat(FORMAT_NCHW);
  c_tensor_desc.SetOriginShape(shape);
  c_tensor_desc.SetFormat(FORMAT_NCHW);
  c_tensor_desc.SetDataType(DT_FLOAT);
  op2->AddOutputDesc("z", c_tensor_desc);
  ge::AttrUtils::SetInt(op2, FE_IMPLY_TYPE, 6);
  ge::AttrUtils::SetInt(op2, ATTR_NAME_GROUPS, op2_groups);

  ge::NodePtr node2 = graph->AddNode(op2);
  GraphUtils::AddEdge(node1->GetOutDataAnchor(0), node2->GetInDataAnchor(0));

  Status ret1 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(node1, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret1 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(node1, "tbe-custom");
  Status ret2 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(node2, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret2 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(node2, "tbe-custom");
  ASSERT_EQ(ret1, fe::SUCCESS);
  ASSERT_EQ(ret2, fe::SUCCESS);

  vector<int64_t> dim_result_fzg1({196, 3, 16, 16});
  EXPECT_EQ(ge::GetPrimaryFormat(op1->GetInputDesc(0).GetFormat()), FORMAT_FRACTAL_Z);
  EXPECT_EQ(op1->GetInputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op1->GetInputDesc(0).GetShape().GetDims(), dim_result_fzg1);
  EXPECT_EQ(ge::GetSubFormat(op1->GetInputDesc(0).GetFormat()), op1_groups);

  EXPECT_EQ(ge::GetPrimaryFormat(op1->GetInputDesc(1).GetFormat()), FORMAT_FRACTAL_Z);
  EXPECT_EQ(op1->GetInputDesc(1).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op1->GetInputDesc(1).GetShape().GetDims(), dim_result_fzg1);
  EXPECT_EQ(ge::GetSubFormat(op1->GetInputDesc(1).GetFormat()), op1_groups);


  EXPECT_EQ(ge::GetPrimaryFormat(op1->GetOutputDesc(0).GetFormat()), FORMAT_FRACTAL_Z);
  EXPECT_EQ(op1->GetOutputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op1->GetOutputDesc(0).GetShape().GetDims(), dim_result_fzg1);
  EXPECT_EQ(ge::GetSubFormat(op1->GetOutputDesc(0).GetFormat()), op1_groups);

  vector<int64_t> dim_result_fzg2({1452, 1, 16, 16});
  EXPECT_EQ(ge::GetPrimaryFormat(op2->GetInputDesc(0).GetFormat()), FORMAT_FRACTAL_Z);
  EXPECT_EQ(op2->GetInputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op2->GetInputDesc(0).GetShape().GetDims(), dim_result_fzg2);
  EXPECT_EQ(ge::GetSubFormat(op2->GetInputDesc(0).GetFormat()), 0);

  EXPECT_EQ(ge::GetPrimaryFormat(op2->GetOutputDesc(0).GetFormat()), FORMAT_FRACTAL_Z);
  EXPECT_EQ(op2->GetOutputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op2->GetOutputDesc(0).GetShape().GetDims(), dim_result_fzg2);
  EXPECT_EQ(ge::GetSubFormat(op2->GetOutputDesc(0).GetFormat()), 0);
}


TEST_F(UTEST_fusion_engine_op_judge_unittest_fz_group, test_02){
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr op1 = std::make_shared<OpDesc>("E1", "E");
  OpDescPtr op2 = std::make_shared<OpDesc>("E2", "E");
  int64_t op1_groups = 2;
  int64_t op2_groups = 1;
  //add descriptor
  vector<int64_t> dim_input({40, 25, 3, 7, 7});
  GeShape shape(dim_input);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NCDHW);
  tensor_desc.SetOriginShape(shape);
  tensor_desc.SetFormat(FORMAT_NCDHW);
  tensor_desc.SetDataType(DT_FLOAT);
  op1->AddInputDesc("x", tensor_desc);

  /*1, 1, i+h, 4*h
   * h = 562, i = 672*/
  vector<int64_t> dim_weight({3, 7, 7, 25, 40});
  GeShape weight_shape(dim_weight);
  GeTensorDesc weight_desc(weight_shape);
  weight_desc.SetOriginFormat(FORMAT_DHWCN);
  weight_desc.SetOriginShape(weight_shape);
  weight_desc.SetFormat(FORMAT_DHWCN);
  weight_desc.SetDataType(DT_FLOAT);
  op1->AddInputDesc("y", weight_desc);

  vector<int64_t> dim_out({40, 3, 7, 7, 25});
  GeShape out_shape(dim_out);
  GeTensorDesc out_desc(out_shape);
  out_desc.SetOriginFormat(FORMAT_NDHWC);
  out_desc.SetOriginShape(out_shape);
  out_desc.SetFormat(FORMAT_NDHWC);
  out_desc.SetDataType(DT_FLOAT);
  op1->AddOutputDesc("z", out_desc);
  ge::AttrUtils::SetInt(op1, FE_IMPLY_TYPE, 6);
  ge::AttrUtils::SetInt(op1, ATTR_NAME_GROUPS, op1_groups);
  ge::NodePtr node1 = graph->AddNode(op1);

  GeTensorDesc out_desc2(out_shape);
  out_desc2.SetOriginFormat(FORMAT_NDHWC);
  out_desc2.SetOriginShape(out_shape);
  out_desc2.SetFormat(FORMAT_NDHWC);
  out_desc2.SetDataType(DT_FLOAT);
  op2->AddInputDesc("x", out_desc2);

  vector<int64_t> dim_input_c({7, 33, 44, 4, 10});
  GeShape cshape(dim_input_c);
  GeTensorDesc c_tensor_desc(cshape);
  c_tensor_desc.SetOriginFormat(FORMAT_DHWNC);
  c_tensor_desc.SetOriginShape(cshape);
  c_tensor_desc.SetFormat(FORMAT_DHWNC);
  c_tensor_desc.SetDataType(DT_FLOAT);
  op2->AddInputDesc("y", c_tensor_desc);

  vector<int64_t> dim_input_d({4, 7, 33, 44, 10});
  GeShape dshape(dim_input_d);
  GeTensorDesc d_tensor_desc(dshape);
  d_tensor_desc.SetOriginFormat(FORMAT_NDHWC);
  d_tensor_desc.SetOriginShape(dshape);
  d_tensor_desc.SetFormat(FORMAT_NDHWC);
  d_tensor_desc.SetDataType(DT_FLOAT);

  op2->AddOutputDesc("z", d_tensor_desc);
  ge::AttrUtils::SetInt(op2, FE_IMPLY_TYPE, 6);
  ge::AttrUtils::SetInt(op2, ATTR_NAME_GROUPS, op2_groups);

  ge::NodePtr node2 = graph->AddNode(op2);
  GraphUtils::AddEdge(node1->GetOutDataAnchor(0), node2->GetInDataAnchor(0));

  
  Status ret1 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(node1, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret1 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(node1, "tbe-custom");
  Status ret2 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(node2, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret2 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(node2, "tbe-custom");
  ASSERT_EQ(ret1, fe::SUCCESS);
  ASSERT_EQ(ret2, fe::SUCCESS);

  int64_t group_value = -1;
  vector<int64_t> dim_result_fzg1({588, 3, 16, 16});
  EXPECT_EQ(ge::GetPrimaryFormat(op1->GetInputDesc(0).GetFormat()), FORMAT_FRACTAL_Z_3D);
  EXPECT_EQ(op1->GetInputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op1->GetInputDesc(0).GetShape().GetDims(), dim_result_fzg1);
  EXPECT_EQ(ge::GetSubFormat(op1->GetInputDesc(0).GetFormat()), op1_groups);

  EXPECT_EQ(ge::GetPrimaryFormat(op1->GetInputDesc(1).GetFormat()), FORMAT_FRACTAL_Z_3D);
  EXPECT_EQ(op1->GetInputDesc(1).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op1->GetInputDesc(1).GetShape().GetDims(), dim_result_fzg1);
  EXPECT_EQ(ge::GetSubFormat(op1->GetInputDesc(1).GetFormat()), op1_groups);

  EXPECT_EQ(ge::GetPrimaryFormat(op1->GetOutputDesc(0).GetFormat()), FORMAT_FRACTAL_Z_3D);
  EXPECT_EQ(op1->GetOutputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op1->GetOutputDesc(0).GetShape().GetDims(), dim_result_fzg1);
  EXPECT_EQ(ge::GetSubFormat(op1->GetOutputDesc(0).GetFormat()), op1_groups);

  vector<int64_t> dim_result_fzg12({294, 3, 16, 16});
  EXPECT_EQ(ge::GetPrimaryFormat(op2->GetInputDesc(0).GetFormat()), FORMAT_FRACTAL_Z_3D);
  EXPECT_EQ(op2->GetInputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op2->GetInputDesc(0).GetShape().GetDims(), dim_result_fzg12);
  EXPECT_EQ(ge::GetSubFormat(op2->GetInputDesc(0).GetFormat()), 0);

  vector<int64_t> dim_result_fzg2({10164, 1, 16, 16});
  EXPECT_EQ(ge::GetPrimaryFormat(op2->GetInputDesc(1).GetFormat()), FORMAT_FRACTAL_Z_3D);
  EXPECT_EQ(op2->GetInputDesc(1).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op2->GetInputDesc(1).GetShape().GetDims(), dim_result_fzg2);
  EXPECT_EQ(ge::GetSubFormat(op2->GetInputDesc(1).GetFormat()), 0);

  EXPECT_EQ(ge::GetPrimaryFormat(op2->GetOutputDesc(0).GetFormat()), FORMAT_FRACTAL_Z_3D);
  EXPECT_EQ(op2->GetOutputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op2->GetOutputDesc(0).GetShape().GetDims(), dim_result_fzg2);
  EXPECT_EQ(ge::GetSubFormat(op2->GetOutputDesc(0).GetFormat()), 0);
}

TEST_F(UTEST_fusion_engine_op_judge_unittest_fz_group, test_03){
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr op1 = std::make_shared<OpDesc>("A1", "A");
  OpDescPtr op2 = std::make_shared<OpDesc>("B1", "B");
  //add descriptor
  vector<int64_t> dim_input({40, 25, 7, 7});
  GeShape shape(dim_input);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NCHW);
  tensor_desc.SetOriginShape(shape);
  tensor_desc.SetFormat(FORMAT_NCHW);
  tensor_desc.SetDataType(DT_FLOAT);
  op1->AddInputDesc("x", tensor_desc);

  /*1, 1, i+h, 4*h
   * h = 562, i = 672*/
  vector<int64_t> dim_weight({7, 7, 25, 40});
  GeShape weight_shape(dim_weight);
  GeTensorDesc weight_desc(weight_shape);
  weight_desc.SetOriginFormat(FORMAT_HWCN);
  weight_desc.SetOriginShape(weight_shape);
  weight_desc.SetFormat(FORMAT_HWCN);
  weight_desc.SetDataType(DT_FLOAT);
  op1->AddInputDesc("y", weight_desc);

  vector<int64_t> dim_out({40, 7, 7, 25});
  GeShape out_shape(dim_out);
  GeTensorDesc out_desc(out_shape);
  out_desc.SetOriginFormat(FORMAT_NHWC);
  out_desc.SetOriginShape(out_shape);
  out_desc.SetFormat(FORMAT_NHWC);
  out_desc.SetDataType(DT_FLOAT);
  op1->AddOutputDesc("z", out_desc);
  ge::AttrUtils::SetInt(op1, FE_IMPLY_TYPE, 6);
  ge::AttrUtils::SetInt(op1, ATTR_NAME_GROUPS, 0);
  ge::NodePtr node1 = graph->AddNode(op1);

  vector<int64_t> dim_input_b({4, 33, 44, 10});
  GeShape bshape(dim_input_b);
  GeTensorDesc b_tensor_desc(bshape);
  b_tensor_desc.SetOriginFormat(FORMAT_NHWC);
  b_tensor_desc.SetOriginShape(bshape);
  b_tensor_desc.SetFormat(FORMAT_NHWC);
  b_tensor_desc.SetDataType(DT_FLOAT);
  op2->AddInputDesc("x", b_tensor_desc);

  vector<int64_t> dim_input_c({4, 10, 33, 44});
  GeShape cshape(dim_input_c);
  GeTensorDesc c_tensor_desc(cshape);
  c_tensor_desc.SetOriginFormat(FORMAT_NCHW);
  c_tensor_desc.SetOriginShape(shape);
  c_tensor_desc.SetFormat(FORMAT_NCHW);
  c_tensor_desc.SetDataType(DT_FLOAT);
  op2->AddOutputDesc("z", c_tensor_desc);
  ge::AttrUtils::SetInt(op2, FE_IMPLY_TYPE, 6);
  ge::AttrUtils::SetInt(op2, ATTR_NAME_GROUPS, -1);

  ge::NodePtr node2 = graph->AddNode(op2);
  GraphUtils::AddEdge(node1->GetOutDataAnchor(0), node2->GetInDataAnchor(0));

  
  Status ret1 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(node1, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret1 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(node1, "tbe-custom");
  Status ret2 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(node2, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret2 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(node2, "tbe-custom");
  ASSERT_EQ(ret1, fe::FAILED);
  ASSERT_EQ(ret2, fe::FAILED);
}

TEST_F(UTEST_fusion_engine_op_judge_unittest_fz_group, test_04){
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr op1 = std::make_shared<OpDesc>("E1", "E");
  OpDescPtr op2 = std::make_shared<OpDesc>("E2", "E");
  int64_t op1_groups = 8;
  int64_t op2_groups = 1;

  //add descriptor
  vector<int64_t> dim_input({192, 3, 3, 3, 12});
  GeShape shape(dim_input);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NDHWC);
  tensor_desc.SetOriginShape(shape);
  tensor_desc.SetFormat(FORMAT_NDHWC);
  tensor_desc.SetDataType(DT_FLOAT);

  GeTensorDesc tensor_desc2(shape);
  tensor_desc2.SetOriginFormat(FORMAT_NDHWC);
  tensor_desc2.SetOriginShape(shape);
  tensor_desc2.SetFormat(FORMAT_NDHWC);
  tensor_desc2.SetDataType(DT_FLOAT);
  op1->AddInputDesc("x", tensor_desc);
  op2->AddInputDesc("x", tensor_desc2);

  /*1, 1, i+h, 4*h
   * h = 562, i = 672*/
  vector<int64_t> dim_weight({3, 3, 3, 12, 192});
  GeShape weight_shape(dim_weight);
  GeTensorDesc weight_desc(weight_shape);
  weight_desc.SetOriginFormat(FORMAT_DHWCN);
  weight_desc.SetOriginShape(weight_shape);
  weight_desc.SetFormat(FORMAT_DHWCN);
  weight_desc.SetDataType(DT_FLOAT);
  op1->AddInputDesc("y", weight_desc);

  GeTensorDesc weight_desc2(weight_shape);
  weight_desc2.SetOriginFormat(FORMAT_DHWCN);
  weight_desc2.SetOriginShape(weight_shape);
  weight_desc2.SetFormat(FORMAT_DHWCN);
  weight_desc2.SetDataType(DT_FLOAT);
  op2->AddInputDesc("y", weight_desc2);

  vector<int64_t> dim_out({192, 12, 3, 3, 3});
  GeShape out_shape(dim_out);
  GeTensorDesc out_desc(out_shape);
  out_desc.SetOriginFormat(FORMAT_NCDHW);
  out_desc.SetOriginShape(out_shape);
  out_desc.SetFormat(FORMAT_NCDHW);
  out_desc.SetDataType(DT_FLOAT);
  op1->AddOutputDesc("z", out_desc);

  GeTensorDesc out_desc2(out_shape);
  out_desc2.SetOriginFormat(FORMAT_NCDHW);
  out_desc2.SetOriginShape(out_shape);
  out_desc2.SetFormat(FORMAT_NCDHW);
  out_desc2.SetDataType(DT_FLOAT);
  op2->AddOutputDesc("z", out_desc2);

  ge::AttrUtils::SetInt(op1, FE_IMPLY_TYPE, 6);
  ge::AttrUtils::SetInt(op1, ATTR_NAME_GROUPS, op1_groups);
  ge::NodePtr node1 = graph->AddNode(op1);

  ge::AttrUtils::SetInt(op2, FE_IMPLY_TYPE, 6);
  ge::AttrUtils::SetInt(op2, ATTR_NAME_GROUPS, op2_groups);
  ge::NodePtr node2 = graph->AddNode(op2);
  GraphUtils::AddEdge(node1->GetOutDataAnchor(0), node2->GetInDataAnchor(0));

  Status ret1 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(node1, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret1 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(node1, "tbe-custom");
  Status ret2 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(node2, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret2 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(node2, "tbe-custom");
  ASSERT_EQ(ret1, fe::SUCCESS);
  ASSERT_EQ(ret2, fe::SUCCESS);

  vector<int64_t> dim_result_fzg1({162, 6, 16, 16});
  EXPECT_EQ(ge::GetPrimaryFormat(op1->GetInputDesc(0).GetFormat()), FORMAT_FRACTAL_Z_3D);
  EXPECT_EQ(op1->GetInputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op1->GetInputDesc(0).GetShape().GetDims(), dim_result_fzg1);
  EXPECT_EQ(ge::GetSubFormat(op1->GetInputDesc(0).GetFormat()), op1_groups);

  EXPECT_EQ(ge::GetPrimaryFormat(op1->GetInputDesc(1).GetFormat()), FORMAT_FRACTAL_Z_3D);
  EXPECT_EQ(op1->GetInputDesc(1).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op1->GetInputDesc(1).GetShape().GetDims(), dim_result_fzg1);
  EXPECT_EQ(ge::GetSubFormat(op1->GetInputDesc(1).GetFormat()), op1_groups);

  EXPECT_EQ(ge::GetPrimaryFormat(op1->GetOutputDesc(0).GetFormat()), FORMAT_FRACTAL_Z_3D);
  EXPECT_EQ(op1->GetOutputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op1->GetOutputDesc(0).GetShape().GetDims(), dim_result_fzg1);
  EXPECT_EQ(ge::GetSubFormat(op1->GetOutputDesc(0).GetFormat()), op1_groups);

  vector<int64_t> dim_result_fzg2({27, 12, 16, 16});
  EXPECT_EQ(ge::GetPrimaryFormat(op2->GetInputDesc(0).GetFormat()), FORMAT_FRACTAL_Z_3D);
  EXPECT_EQ(op2->GetInputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op2->GetInputDesc(0).GetShape().GetDims(), dim_result_fzg2);
  EXPECT_EQ(ge::GetSubFormat(op2->GetInputDesc(0).GetFormat()), 0);

  EXPECT_EQ(ge::GetPrimaryFormat(op2->GetInputDesc(1).GetFormat()), FORMAT_FRACTAL_Z_3D);
  EXPECT_EQ(op2->GetInputDesc(1).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op2->GetInputDesc(1).GetShape().GetDims(), dim_result_fzg2);
  EXPECT_EQ(ge::GetSubFormat(op2->GetInputDesc(1).GetFormat()), 0);

  EXPECT_EQ(ge::GetPrimaryFormat(op2->GetOutputDesc(0).GetFormat()), FORMAT_FRACTAL_Z_3D);
  EXPECT_EQ(op2->GetOutputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op2->GetOutputDesc(0).GetShape().GetDims(), dim_result_fzg2);
  EXPECT_EQ(ge::GetSubFormat(op2->GetOutputDesc(0).GetFormat()), 0);
}
