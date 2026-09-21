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
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/ge_local_context.h"

#include "adapter/tbe_adapter/tbe_op_store_adapter.h"
#include "graph_optimizer/op_judge/imply_type/op_impl_type_judge.h"
#include "graph_optimizer/op_judge/format_and_dtype/op_format_dtype_judge.h"
#include "graph_optimizer/op_judge/format_and_dtype/strategy/matcher/dtype/op_dtype_rise_matcher.h"
#include "graph_optimizer/op_judge/format_and_dtype/strategy/matcher/format/op_format_matcher.h"

#include "graph_optimizer/shape_format_transfer/trans_node_manager/trans_node_manager.h"
#include "graph/debug/ge_attr_define.h"
#include "common/configuration.h"
#include "common/unknown_shape_util.h"
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
using OpFormatDtypeStrategyManagerPtr = std::shared_ptr<OpFormatDtypeStrategyManager>;

using TransNodeManagerPtr = std::shared_ptr<TransNodeManager>;

class UTEST_fusion_engine_op_judge_unknown_shape : public testing::Test
{
protected:
  void SetUp()
  {

    std::map<std::string, std::string> options;
    fe_ops_kernel_info_store_ptr_ = make_shared<fe::FEOpsKernelInfoStore>(fe::AI_CORE_NAME);
    fe_ops_kernel_info_store_ptr_cce_ = make_shared<fe::FEOpsKernelInfoStore>(fe::AI_CORE_NAME);
    FEOpsStoreInfo cce_custom {
        4,
        "cce-custom",
        EN_IMPL_HW_TBE,
        GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/cce_general_opinfo",
        "",
        false,
        false,
        false};
    FEOpsStoreInfo tbe_custom {
        6,
        "tbe-custom",
        EN_IMPL_HW_TBE,
        GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/tbe_custom_opinfo",
        "",
        false,
        false,
        false};
    vector<FEOpsStoreInfo> store_info;
    store_info.emplace_back(tbe_custom);
    Configuration::Instance(fe::AI_CORE_NAME).ops_store_info_vector_ = (store_info);
    OpsKernelManager::Instance(AI_CORE_NAME).Finalize();

    fe_ops_kernel_info_store_ptr_->Initialize(options);
    store_info.emplace_back(cce_custom);
    Configuration::Instance(fe::AI_CORE_NAME).ops_store_info_vector_ = (store_info);
    ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = ALLOW_MIX_PRECISION;
    OpsKernelManager::Instance(AI_CORE_NAME).Finalize();

    fe_ops_kernel_info_store_ptr_cce_->Initialize(options);
    reflection_builder_ptr_ = std::make_shared<ge::RefRelations>();
    op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr_);
    op_format_dtype_judge_ptr_->Initialize();
  }

  void TearDown()
  {

  }
  shared_ptr<fe::FEOpsKernelInfoStore> fe_ops_kernel_info_store_ptr_;
  shared_ptr<fe::FEOpsKernelInfoStore> fe_ops_kernel_info_store_ptr_cce_;
  RefRelationsPtr reflection_builder_ptr_;
  OpFormatDtypeJudgePtr op_format_dtype_judge_ptr_;
protected:
  static void CreateOneOpGraph(ComputeGraphPtr graph) {

    OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "Activation");

    // add descriptor
    vector<int64_t> dim(4, -1);
    vector<std::pair<int64_t, int64_t>> range({{1, 200}});
    GeShape shape(dim);
    GeTensorDesc out_desc(shape);
    out_desc.SetOriginFormat(FORMAT_NCHW);
    out_desc.SetFormat(FORMAT_NCHW);
    out_desc.SetDataType(DT_FLOAT16);
    out_desc.SetShapeRange(range);
    relu_op->AddInputDesc("x", out_desc);
    relu_op->AddOutputDesc("y", out_desc);

    NodePtr relu_node = graph->AddNode(relu_op);
  }

  static void CreateTwoOpGraph(ComputeGraphPtr graph) {
    // 创建Node
    OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
    OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "Activation");

    // add descriptor
    vector<int64_t> dim(4, -1);
    vector<std::pair<int64_t, int64_t>> range({{1, 200}});
    GeShape shape(dim);
    GeTensorDesc out_desc(shape);
    out_desc.SetOriginFormat(FORMAT_NCHW);
    out_desc.SetFormat(FORMAT_NCHW);
    out_desc.SetDataType(DT_FLOAT16);
    out_desc.SetShapeRange(range);
    relu_op->AddInputDesc("x", out_desc);
    relu_op->AddOutputDesc("y", out_desc);

    GeTensorDesc bn_desc(shape);
    bn_desc.SetOriginFormat(FORMAT_NC1HWC0);
    bn_desc.SetFormat(FORMAT_NC1HWC0);
    bn_desc.SetDataType(DT_FLOAT16);
    bn_op->AddInputDesc("x", bn_desc);
    bn_op->AddOutputDesc("y", bn_desc);

    ge::AttrUtils::SetInt(bn_op, "_fe_imply_type", static_cast<int>(EN_IMPL_HW_TBE));
    ge::AttrUtils::SetInt(relu_op, "_fe_imply_type", static_cast<int>(EN_IMPL_HW_TBE));

    NodePtr bn_node = graph->AddNode(bn_op);
    NodePtr relu_node = graph->AddNode(relu_op);

    GraphUtils::AddEdge(bn_node->GetOutDataAnchor(0), relu_node->GetInDataAnchor(0));
  }

  static void CreateTwoOpDescGraph(ComputeGraphPtr graph) {
    // 创建Node
    OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
    OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "Activation");

    // add descriptor
    vector<int64_t> dims = {1,-1,-1,4};
    vector<std::pair<int64_t, int64_t>> range({{1, 200}, {1, 128}});
    GeShape shape(dims);

    GeTensorDesc in_desc1(shape);
    in_desc1.SetOriginFormat(FORMAT_NCHW);
    in_desc1.SetFormat(FORMAT_NCHW);
    in_desc1.SetDataType(DT_FLOAT16);
    in_desc1.SetShapeRange(range);
    relu_op->AddInputDesc("x", in_desc1);

    GeTensorDesc out_desc1(shape);
    out_desc1.SetOriginFormat(FORMAT_HWCN);
    out_desc1.SetFormat(FORMAT_HWCN);
    out_desc1.SetDataType(DT_FLOAT16);
    out_desc1.SetShapeRange(range);
    relu_op->AddOutputDesc("y", out_desc1);


    GeTensorDesc in_desc2(shape);
    in_desc2.SetOriginFormat(FORMAT_FRACTAL_Z);
    in_desc2.SetFormat(FORMAT_FRACTAL_Z);
    in_desc2.SetDataType(DT_FLOAT16);
    in_desc2.SetShapeRange(range);
    bn_op->AddInputDesc("x", in_desc2);

    GeTensorDesc out_desc2(shape);
    out_desc2.SetOriginFormat(FORMAT_NHWC);
    out_desc2.SetFormat(FORMAT_NHWC);
    out_desc2.SetDataType(DT_FLOAT16);
    out_desc2.SetShapeRange(range);
    bn_op->AddOutputDesc("y", out_desc2);

    NodePtr bn_node = graph->AddNode(bn_op);
    NodePtr relu_node = graph->AddNode(relu_op);
    GraphUtils::AddEdge(bn_node->GetOutDataAnchor(0), relu_node->GetInDataAnchor(0));
  }

  static void CreateThreeOpGraph(ComputeGraphPtr graph)
  {
    // 创建Node
    OpDescPtr square01 = std::make_shared<OpDesc>("square01", "Square");
    OpDescPtr square02 = std::make_shared<OpDesc>("square02", "Square");

    OpDescPtr max01 = std::make_shared<OpDesc>("max01", "Maximum");

    // add descriptor
    vector<int64_t> dim(DIMENSION_4, DIMENSION_1);
    vector<std::pair<int64_t, int64_t>> range({{1, 128}});
    GeShape Shape(dim);
    GeTensorDesc out_desc(Shape);
    out_desc.SetOriginFormat(FORMAT_NCHW);
    out_desc.SetFormat(FORMAT_NCHW);
    out_desc.SetDataType(DT_FLOAT16);
    out_desc.SetShapeRange(range);
    square01->AddOutputDesc("x", out_desc);
    square02->AddOutputDesc("x", out_desc);

    max01->AddInputDesc("x", out_desc);
    max01->AddInputDesc("y", out_desc);

    ge::AttrUtils::SetInt(square01, "_fe_imply_type", static_cast<int>(EN_IMPL_HW_TBE));
    ge::AttrUtils::SetInt(square02, "_fe_imply_type", static_cast<int>(EN_IMPL_HW_TBE));
    ge::AttrUtils::SetInt(max01, "_fe_imply_type", static_cast<int>(EN_IMPL_HW_TBE));

    NodePtr square01_node = graph->AddNode(square01);
    NodePtr square02_node = graph->AddNode(square02);
    NodePtr max01_node = graph->AddNode(max01);

    GraphUtils::AddEdge(square01_node->GetOutDataAnchor(0), max01_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(square02_node->GetOutDataAnchor(0), max01_node->GetInDataAnchor(1));
  }

  static void CreateTwoInvalidOpGraph(ComputeGraphPtr graph) {
    // 创建Node
    OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
    OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "Activation");

    // add descriptor
    vector<int64_t> dim(4, -1);
    vector<std::pair<int64_t, int64_t>> range({{1, 128}});
    GeShape shape(dim);
    GeTensorDesc out_desc(shape);
    out_desc.SetOriginFormat(FORMAT_NCHW);
    out_desc.SetFormat(FORMAT_NCHW);
    out_desc.SetDataType(DT_FLOAT16);
    out_desc.SetShapeRange(range);
    relu_op->AddInputDesc("x", out_desc);
    relu_op->AddOutputDesc("y", out_desc);

    GeTensorDesc bn_desc(shape);
    bn_desc.SetOriginFormat(FORMAT_NCHW);
    bn_desc.SetFormat(FORMAT_NC1HWC0);
    bn_desc.SetDataType(DT_FLOAT16);
    bn_desc.SetShapeRange(range);
    bn_op->AddInputDesc("x", bn_desc);
    bn_op->AddOutputDesc("y", bn_desc);

    ge::AttrUtils::SetInt(bn_op, "_fe_imply_type", static_cast<int>(EN_RESERVED));
    ge::AttrUtils::SetInt(relu_op, "_fe_imply_type", static_cast<int>(EN_RESERVED));

    NodePtr bn_node = graph->AddNode(bn_op);
    NodePtr relu_node = graph->AddNode(relu_op);

    GraphUtils::AddEdge(bn_node->GetOutDataAnchor(0), relu_node->GetInDataAnchor(0));
  }

  static void CreateTwoPluginTbeOpGraph(ComputeGraphPtr graph) {
    OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
    OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "Activation");

    // add descriptor
    vector<int64_t> dim(4, -1);
    vector<std::pair<int64_t, int64_t>> range({{1, 128}});
    GeShape shape(dim);
    GeTensorDesc out_desc(shape);
    out_desc.SetOriginFormat(FORMAT_NCHW);
    out_desc.SetFormat(FORMAT_NCHW);
    out_desc.SetDataType(DT_FLOAT16);
    out_desc.SetShapeRange(range);
    relu_op->AddInputDesc("x", out_desc);
    relu_op->AddOutputDesc("y", out_desc);

    GeTensorDesc bn_desc(shape);
    out_desc.SetOriginFormat(FORMAT_NCHW);
    bn_desc.SetFormat(FORMAT_NC1HWC0);
    bn_desc.SetDataType(DT_FLOAT16);
    bn_desc.SetShapeRange(range);
    bn_op->AddInputDesc("x", bn_desc);
    bn_op->AddOutputDesc("y", bn_desc);

    ge::AttrUtils::SetInt(relu_op, "_fe_imply_type", static_cast<int>(EN_IMPL_PLUGIN_TBE));
    ge::AttrUtils::SetInt(bn_op, "_fe_imply_type", static_cast<int>(EN_IMPL_PLUGIN_TBE));

    NodePtr bn_node = graph->AddNode(bn_op);
    NodePtr relu_node = graph->AddNode(relu_op);

    GraphUtils::AddEdge(bn_node->GetOutDataAnchor(0), relu_node->GetInDataAnchor(0));
  }
};

TEST_F(UTEST_fusion_engine_op_judge_unknown_shape, judge_nchw_c04_succ)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr conv_op = std::make_shared<OpDesc>("conv", "ConvTempC04");

  vector<int64_t> dim({64, -1, 7, -1});
  vector<std::pair<int64_t, int64_t>> range({{1, 16}, {1, 7}});
  GeShape shape(dim);
  GeTensorDesc in_desc1(shape);
  in_desc1.SetOriginFormat(FORMAT_NCHW);
  in_desc1.SetFormat(FORMAT_NCHW);
  in_desc1.SetDataType(DT_FLOAT16);
  in_desc1.SetShapeRange(range);
  vector<int64_t> dim2({64, -1, 7, -1});
  vector<std::pair<int64_t, int64_t>> range1({{1, 3}, {1, 7}});
  GeShape shape2(dim2);
  GeTensorDesc in_desc2(shape2);
  in_desc2.SetOriginFormat(FORMAT_NCHW);
  in_desc2.SetFormat(FORMAT_NCHW);
  in_desc2.SetDataType(DT_FLOAT16);
  in_desc2.SetShapeRange(range1);
  conv_op->AddInputDesc("x", in_desc1);
  conv_op->AddInputDesc("w", in_desc2);
  conv_op->AddOutputDesc("y", in_desc1);
  NodePtr conv_node = graph->AddNode(conv_op);
  ge::AttrUtils::SetInt(conv_op, FE_IMPLY_TYPE, 6);//TBE

  Status ret = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(conv_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(conv_node, "tbe-custom");
  EXPECT_EQ(fe::SUCCESS, ret);
  OpDescPtr conv_op_desc = conv_node->GetOpDesc();
  vector<int64_t> new_dim({64, -1, 7, -1, 4});
  vector<std::pair<int64_t, int64_t>> range_expect({{64, 64}, {1, 4}, {7, 7}, {1, 7}, {4, 4}});
  EXPECT_EQ(conv_op_desc->GetInputDesc(0).GetShape().GetDims(), new_dim);
  EXPECT_EQ(GetShapeRange(conv_op_desc->GetInputDesc(0)), range_expect);
  vector<int64_t> new_dim2({-1, 4, 16, 16});
  vector<std::pair<int64_t, int64_t>> range_expect1({{2, 13}, {4, 4}, {16, 16}, {16, 16}});
  //EXPECT_EQ(conv_op_desc->GetInputDesc(1).GetShape().GetDims(), new_dim2);
  EXPECT_EQ(conv_op_desc->GetOutputDesc(0).GetShape().GetDims(), new_dim);
  EXPECT_EQ(GetShapeRange(conv_op_desc->GetInputDesc(1)), range_expect1);
  EXPECT_EQ(GetShapeRange(conv_op_desc->GetOutputDesc(0)), range_expect);
}

TEST_F(UTEST_fusion_engine_op_judge_unknown_shape, judge_nhwc_c04_succ)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr conv_op = std::make_shared<OpDesc>("conv", "ConvTempC04");

  vector<int64_t> dim({64, -1, 7, -1});
  vector<std::pair<int64_t, int64_t>> range({{1, 7}, {1, 16}});
  GeShape shape(dim);
  GeTensorDesc in_desc1(shape);
  in_desc1.SetOriginFormat(FORMAT_NHWC);
  in_desc1.SetFormat(FORMAT_NHWC);
  in_desc1.SetDataType(DT_FLOAT16);
  in_desc1.SetShapeRange(range);
  vector<int64_t> dim2({64, -1, 7, -1});
  vector<std::pair<int64_t, int64_t>> range1({{1, 7}, {1, 3}});
  GeShape shape2(dim2);
  GeTensorDesc in_desc2(shape2);
  in_desc2.SetOriginFormat(FORMAT_NHWC);
  in_desc2.SetFormat(FORMAT_NHWC);
  in_desc2.SetDataType(DT_FLOAT16);
  in_desc2.SetShapeRange(range1);
  conv_op->AddInputDesc("x", in_desc1);
  conv_op->AddInputDesc("w", in_desc2);
  conv_op->AddOutputDesc("y", in_desc1);
  NodePtr conv_node = graph->AddNode(conv_op);
  ge::AttrUtils::SetInt(conv_op, FE_IMPLY_TYPE, 6);//TBE

  
  Status ret = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(conv_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(conv_node, "tbe-custom");
  EXPECT_EQ(fe::SUCCESS, ret);
  OpDescPtr conv_op_desc = conv_node->GetOpDesc();
  vector<int64_t> new_dim({64, -1, -1, 7, 4});
  vector<std::pair<int64_t, int64_t>> range_expect({{64, 64}, {1, 4}, {1, 7}, {7, 7}, {4, 4}});
  EXPECT_EQ(conv_op_desc->GetInputDesc(0).GetShape().GetDims(), new_dim);
  EXPECT_EQ(GetShapeRange(conv_op_desc->GetInputDesc(0)), range_expect);
  vector<int64_t> new_dim2({-1, 4, 16, 16});
  vector<std::pair<int64_t, int64_t>> range_expect1({{2, 13}, {4, 4}, {16, 16}, {16, 16}});
  //EXPECT_EQ(conv_op_desc->GetInputDesc(1).GetShape().GetDims(), new_dim2);
  EXPECT_EQ(conv_op_desc->GetOutputDesc(0).GetShape().GetDims(), new_dim);
  EXPECT_EQ(GetShapeRange(conv_op_desc->GetInputDesc(1)), range_expect1);
  EXPECT_EQ(GetShapeRange(conv_op_desc->GetOutputDesc(0)), range_expect);
}

TEST_F(UTEST_fusion_engine_op_judge_unknown_shape, judge_hwcn_c04_succ)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr conv_op = std::make_shared<OpDesc>("conv", "ConvTempC04");

  vector<int64_t> dim({-1, 7, -1, 64});
  vector<std::pair<int64_t, int64_t>> range({{1, 7}, {1, 16}});
  GeShape shape(dim);
  GeTensorDesc in_desc1(shape);
  in_desc1.SetOriginFormat(FORMAT_HWCN);
  in_desc1.SetFormat(FORMAT_HWCN);
  in_desc1.SetDataType(DT_FLOAT16);
  in_desc1.SetShapeRange(range);
  vector<int64_t> dim2({-1, -1, 3, 64});
  vector<std::pair<int64_t, int64_t>> range1({{1, 7}, {1, 7}});
  GeShape shape2(dim2);
  GeTensorDesc in_desc2(shape2);
  in_desc2.SetOriginFormat(FORMAT_HWCN);
  in_desc2.SetFormat(FORMAT_HWCN);
  in_desc2.SetDataType(DT_FLOAT16);
  in_desc2.SetShapeRange(range1);
  conv_op->AddInputDesc("x", in_desc1);
  conv_op->AddInputDesc("w", in_desc2);
  conv_op->AddOutputDesc("y", in_desc1);
  NodePtr conv_node = graph->AddNode(conv_op);
  ge::AttrUtils::SetInt(conv_op, FE_IMPLY_TYPE, 6);//TBE

  Status ret = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(conv_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(conv_node, "tbe-custom");
  EXPECT_EQ(fe::SUCCESS, ret);
  OpDescPtr conv_op_desc = conv_node->GetOpDesc();
  vector<int64_t> new_dim({64, -1, -1, 7, 4});
  vector<std::pair<int64_t, int64_t>> range_expect({{64, 64}, {1, 4}, {1, 7}, {7, 7}, {4, 4}});
  EXPECT_EQ(conv_op_desc->GetInputDesc(0).GetShape().GetDims(), new_dim);
  EXPECT_EQ(GetShapeRange(conv_op_desc->GetInputDesc(0)), range_expect);
  vector<int64_t> new_dim2({-1, 4, 16, 16});
  vector<std::pair<int64_t, int64_t>> range_expect1({{1, 13}, {4, 4}, {16, 16}, {16, 16}});
  //EXPECT_EQ(conv_op_desc->GetInputDesc(1).GetShape().GetDims(), new_dim2);
  EXPECT_EQ(conv_op_desc->GetOutputDesc(0).GetShape().GetDims(), new_dim);
  EXPECT_EQ(GetShapeRange(conv_op_desc->GetInputDesc(1)), range_expect1);
  EXPECT_EQ(GetShapeRange(conv_op_desc->GetOutputDesc(0)), range_expect);
}

/* Test SetDtypeAndFormatByPrecisionMode on op G without predecessor node
 * After OpFormatDtypeJudge, Op G1 format is FORMAT_NC1HWC0 and Dtype is Fp*/
TEST_F(UTEST_fusion_engine_op_judge_unknown_shape, set_input_format_succ)
{
    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
    OpDescPtr g_op = std::make_shared<OpDesc>("G1", "G");
    ge::NodePtr g_node = graph->AddNode(g_op);
    //add descriptor
    vector<int64_t> dim({1, -1, 3, -1, -1});
    vector<std::pair<int64_t, int64_t>> range1({{1, 2}, {1, 4}, {1, 5}});
    GeShape shape(dim);

    GeTensorDesc tensor_desc(shape);
    tensor_desc.SetOriginFormat(FORMAT_NC1HWC0);
    tensor_desc.SetFormat(FORMAT_NC1HWC0);
    tensor_desc.SetDataType(DT_FLOAT);
    tensor_desc.SetShapeRange(range1);
    g_op->AddInputDesc("x", tensor_desc);
    g_op->AddOutputDesc("z", tensor_desc);

    
    Status ret = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g_node, "tbe-custom");
    ASSERT_EQ(ret, fe::SUCCESS);
    EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetInputDesc(0).GetFormat()), FORMAT_NC1HWC0);
    EXPECT_EQ(g_op->GetInputDesc(0).GetDataType(), DT_FLOAT);
    EXPECT_EQ(g_op->GetInputDesc(0).GetDataType(), DT_FLOAT);
    EXPECT_EQ(g_op->GetInputDesc(0).GetShape().GetDims(), dim);
    EXPECT_EQ(GetShapeRange(g_op->GetInputDesc(0)), range1);

    EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
    EXPECT_EQ(g_op->GetOutputDesc(0).GetDataType(), DT_FLOAT);
    EXPECT_EQ(g_op->GetOutputDesc(0).GetShape().GetDims(), dim);
    EXPECT_EQ(GetShapeRange(g_op->GetOutputDesc(0)), range1);
}


/* Test SetDtypeAndFormatByPrecisionMode on op G without predecessor node
 * After OpFormatDtypeJudge, Op G1 format is FORMAT_NC1HWC0 and Dtype is Fp*/
TEST_F(UTEST_fusion_engine_op_judge_unknown_shape, set_input_format_succ_format_changed)
{
    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
    OpDescPtr g_op = std::make_shared<OpDesc>("G1", "G");
    ge::NodePtr g_node = graph->AddNode(g_op);
    //add descriptor
    vector<int64_t> dim({4, -1, -1, 16});
    vector<std::pair<int64_t, int64_t>> range({{1, 33}, {1, 12}});
    GeShape shape(dim);

    GeTensorDesc tensor_desc(shape);
    tensor_desc.SetOriginFormat(FORMAT_NCHW);
    tensor_desc.SetFormat(FORMAT_NCHW);
    tensor_desc.SetDataType(DT_FLOAT);
    tensor_desc.SetShapeRange(range);
    g_op->AddInputDesc("x", tensor_desc);
    g_op->AddOutputDesc("z", tensor_desc);
    ge::AttrUtils::SetInt(g_op, FE_IMPLY_TYPE, 6);//TBE

    
    Status ret = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g_node, "tbe-custom");
    ASSERT_EQ(ret, fe::SUCCESS);
    vector<int64_t> dim_result({4, -1, -1, 16,16});
    vector<std::pair<int64_t, int64_t>> range_expect({{4, 4}, {1, 3}, {1, 12}, {16, 16}, {16, 16}});
    EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetInputDesc(0).GetFormat()), FORMAT_NC1HWC0);
    EXPECT_EQ(g_op->GetInputDesc(0).GetDataType(), DT_FLOAT);
    EXPECT_EQ(g_op->GetInputDesc(0).GetShape().GetDims(), dim_result);
    EXPECT_EQ(GetShapeRange(g_op->GetInputDesc(0)), range_expect);

    EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
    EXPECT_EQ(g_op->GetOutputDesc(0).GetDataType(), DT_FLOAT);
    EXPECT_EQ(g_op->GetOutputDesc(0).GetShape().GetDims(), dim_result);
    EXPECT_EQ(GetShapeRange(g_op->GetOutputDesc(0)), range_expect);
}

/* Test SetDtypeAndFormatByPrecisionMode on op ConvTemp without predecessor node */
TEST_F(UTEST_fusion_engine_op_judge_unknown_shape, set_input_format_succ_format_changed_Conv)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g_op = std::make_shared<OpDesc>("ConvTemp", "ConvTemp");
  ge::NodePtr g_node = graph->AddNode(g_op);
  //add descriptor

  vector<int64_t> dim1({-1, -1, 64, -1});
  vector<std::pair<int64_t, int64_t>> range({{1, 1}, {1, 16}, {1, 64}});
  GeShape shape1(dim1);
  GeTensorDesc tensor_desc1(shape1);
  tensor_desc1.SetOriginFormat(FORMAT_NCHW);
  tensor_desc1.SetFormat(FORMAT_NCHW);
  tensor_desc1.SetDataType(DT_FLOAT);
  tensor_desc1.SetShapeRange(range);
  g_op->AddInputDesc("xasd", tensor_desc1);

  vector<int64_t> dim2({-1, -1, 7, -1});
  vector<std::pair<int64_t, int64_t>> range1({{1, 64}, {1, 16}, {1, 7}});
  GeShape shape2(dim2);
  GeTensorDesc tensor_desc2(shape2);
  tensor_desc2.SetOriginFormat(FORMAT_NCHW);
  tensor_desc2.SetFormat(FORMAT_NCHW);
  tensor_desc2.SetDataType(DT_FLOAT);
  tensor_desc2.SetShapeRange(range1);
  g_op->AddInputDesc("wasd", tensor_desc2);

  GeShape shape3;
  GeTensorDesc tensor_desc3(shape3);
  vector<std::pair<int64_t, int64_t>> range2;
  tensor_desc3.SetOriginFormat(FORMAT_NCHW);
  tensor_desc3.SetFormat(FORMAT_NCHW);
  tensor_desc3.SetDataType(DT_FLOAT);
  tensor_desc3.SetShapeRange(range2);
  g_op->AddInputDesc("basd", tensor_desc3);

  vector<int64_t> dimo({-1, -1, 30, -1});
  vector<std::pair<int64_t, int64_t>> range3({{1, 1}, {1, 64}, {1, 30}});
  GeShape shapeo(dimo);
  GeTensorDesc tensor_desco(shapeo);
  tensor_desco.SetOriginFormat(FORMAT_NCHW);
  tensor_desco.SetFormat(FORMAT_NCHW);
  tensor_desco.SetDataType(DT_FLOAT);
  tensor_desco.SetShapeRange(range3);
  g_op->AddOutputDesc("yasd", tensor_desco);

  ge::AttrUtils::SetInt(g_op, FE_IMPLY_TYPE, 6);//TBE

  
  Status ret = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g_node, "tbe-custom");
  ASSERT_EQ(ret, fe::SUCCESS);
  ASSERT_EQ(g_op->GetAllInputsDesc().size(), 3);
  vector<int64_t> dim1_5_d({-1, -1, 64, -1, 16});
  vector<std::pair<int64_t, int64_t>> range5_d({{1, 1}, {1, 1}, {64, 64}, {1, 64}, {16, 16}});
  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetInputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetInputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(g_op->GetInputDesc(0).GetShape().GetDims(), dim1_5_d);
  EXPECT_EQ(GetShapeRange(g_op->GetInputDesc(0)), range5_d);

  vector<int64_t> dim2_fz({-1, -1, 16, 16});
  vector<std::pair<int64_t, int64_t>> range_f_z({{7, 49}, {1, 4}, {16, 16}, {16, 16}});
  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetInputDesc(1).GetFormat()), FORMAT_FRACTAL_Z);
  EXPECT_EQ(g_op->GetInputDesc(1).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(g_op->GetInputDesc(1).GetShape().GetDims(), dim2_fz);
  EXPECT_EQ(GetShapeRange(g_op->GetInputDesc(1)), range_f_z);

  EXPECT_EQ(g_op->GetInputDesc(2).GetFormat(), FORMAT_NCHW);
  EXPECT_EQ(g_op->GetInputDesc(2).GetDataType(), DT_FLOAT);

  vector<int64_t> dimo_5_d({-1, -1, 30, -1, 16});
  vector<std::pair<int64_t, int64_t>> range1_5_d({{1, 1}, {1, 4}, {30, 30}, {1, 30}, {16, 16}});
  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetShape().GetDims(), dimo_5_d);
  EXPECT_EQ(GetShapeRange(g_op->GetOutputDesc(0)), range1_5_d);

}

/* Test SetDtypeAndFormatByPrecisionMode on op ConvTemp without predecessor node
 * The First input name is correct as ops kernel info store
 * The second and output is not correct. But we still consider they are qualified by the
 * structure of input and output. */
TEST_F(UTEST_fusion_engine_op_judge_unknown_shape, set_input_format_succ_format_changed_Conv_without_input2)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g_op = std::make_shared<OpDesc>("ConvTemp", "ConvTemp");
  ge::NodePtr g_node = graph->AddNode(g_op);
  //add descriptor

  vector<int64_t> dim1({-1, -1, 64, -1});
  vector<std::pair<int64_t, int64_t>> range({{1, 1}, {1, 16}, {1, 64}});
  GeShape shape1(dim1);
  GeTensorDesc tensor_desc1(shape1);
  tensor_desc1.SetOriginFormat(FORMAT_NCHW);
  tensor_desc1.SetFormat(FORMAT_NCHW);
  tensor_desc1.SetDataType(DT_FLOAT);
  tensor_desc1.SetShapeRange(range);
  g_op->AddInputDesc("x", tensor_desc1);

  vector<int64_t> dim2({-1, -1, 7, -1});
  vector<std::pair<int64_t, int64_t>> range1({{1, 64}, {1, 16}, {1, 7}});
  GeShape shape2(dim2);
  GeTensorDesc tensor_desc2(shape2);
  tensor_desc2.SetOriginFormat(FORMAT_NCHW);
  tensor_desc2.SetFormat(FORMAT_NCHW);
  tensor_desc2.SetDataType(DT_FLOAT);
  tensor_desc2.SetShapeRange(range1);
  g_op->AddInputDesc("wqwe", tensor_desc2);

  vector<int64_t> dimo({-1, -1, 30, -1});
  vector<std::pair<int64_t, int64_t>> range3({{1, 1}, {1, 64}, {1, 30}});
  GeShape shapeo(dimo);
  GeTensorDesc tensor_desco(shapeo);
  tensor_desco.SetOriginFormat(FORMAT_NCHW);
  tensor_desco.SetFormat(FORMAT_NCHW);
  tensor_desco.SetDataType(DT_FLOAT);
  tensor_desco.SetShapeRange(range3);
  g_op->AddOutputDesc("yqwe", tensor_desco);

  ge::AttrUtils::SetInt(g_op, FE_IMPLY_TYPE, 6);//TBE
  vector<int64_t> dim1_5_d({-1, -1, 64, -1, 16});
  vector<int64_t> dim2_fz({-1, -1, 16, 16});
  vector<int64_t> dimo_5_d({-1, -1, 30, -1, 16});
  vector<std::pair<int64_t, int64_t>> range5_d({{1, 1}, {1, 1}, {64, 64}, {1, 64}, {16, 16}});
  vector<std::pair<int64_t, int64_t>> range_f_z({{7, 49}, {1, 4}, {16, 16}, {16, 16}});
  vector<std::pair<int64_t, int64_t>> range1_5_d({{1, 1}, {1, 4}, {30, 30}, {1, 30}, {16, 16}});
  OpImplTypeJudgePtr op_impl_type_judge_ptr = std::make_shared<OpImplTypeJudge>(AI_CORE_NAME, fe_ops_kernel_info_store_ptr_);
  

  Status ret1 = op_impl_type_judge_ptr->MultiThreadJudge(*(graph.get()));
  ASSERT_EQ(ret1, fe::SUCCESS);
  Status ret2 = op_format_dtype_judge_ptr_->Judge(*(graph.get()));
  ret2 = op_format_dtype_judge_ptr_->SetFormat(*(graph.get()));
  ASSERT_EQ(ret2, fe::SUCCESS);

  ASSERT_EQ(g_op->GetAllInputsDesc().size(), 2);
  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetInputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetInputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(g_op->GetInputDesc(0).GetShape().GetDims(), dim1_5_d);
  EXPECT_EQ(GetShapeRange(g_op->GetInputDesc(0)), range5_d);

  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetInputDesc(1).GetFormat()), FORMAT_FRACTAL_Z);
  EXPECT_EQ(g_op->GetInputDesc(1).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(g_op->GetInputDesc(1).GetShape().GetDims(), dim2_fz);
  EXPECT_EQ(GetShapeRange(g_op->GetInputDesc(1)), range_f_z);

  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetShape().GetDims(), dimo_5_d);
  EXPECT_EQ(GetShapeRange(g_op->GetOutputDesc(0)), range1_5_d);
}

TEST_F(UTEST_fusion_engine_op_judge_unknown_shape, set_input_format_succ_format_changed_01)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g_op = std::make_shared<OpDesc>("G1", "G");

  //add descriptor
  vector<int64_t> dim({4, -1, -1, 16});
  vector<std::pair<int64_t, int64_t>> range({{1, 33}, {1, 12}});
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NHWC);
  tensor_desc.SetFormat(FORMAT_NHWC);
  tensor_desc.SetDataType(DT_FLOAT);
  tensor_desc.SetShapeRange(range);
  g_op->AddInputDesc("x", tensor_desc);
  g_op->AddOutputDesc("z", tensor_desc);
  ge::NodePtr g_node = graph->AddNode(g_op);

  ge::AttrUtils::SetInt(g_op, FE_IMPLY_TYPE, 6);//TBE
  OpImplTypeJudgePtr op_impl_type_judge_ptr = std::make_shared<OpImplTypeJudge>(AI_CORE_NAME, fe_ops_kernel_info_store_ptr_);
  
  Status ret1 = op_impl_type_judge_ptr->MultiThreadJudge(*(graph.get()));
  ASSERT_EQ(ret1, fe::SUCCESS);
  Status ret2 = op_format_dtype_judge_ptr_->Judge(*(graph.get()));
  ret2 = op_format_dtype_judge_ptr_->SetFormat(*(graph.get()));
  ASSERT_EQ(ret2, fe::SUCCESS);

  vector<int64_t> dim_result({4, 1, -1, -1, 16});
  vector<std::pair<int64_t, int64_t>> range_expect({{4, 4}, {1, 1}, {1, 33}, {1, 12}, {16, 16}});
  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetInputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetInputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(g_op->GetInputDesc(0).GetShape().GetDims(), dim_result);
  EXPECT_EQ(GetShapeRange(g_op->GetInputDesc(0)), range_expect);

  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetShape().GetDims(), dim_result);
  EXPECT_EQ(GetShapeRange(g_op->GetOutputDesc(0)), range_expect);
}

/* Test SetDtypeAndFormatByPrecisionMode on op G without predecessor node
 * After OpFormatDtypeJudge, Op G1 format is FORMAT_NC1HWC0 and Dtype is Fp*/
TEST_F(UTEST_fusion_engine_op_judge_unknown_shape, set_input_format_succ_format_and_dtype_changed)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g_op = std::make_shared<OpDesc>("G1", "G");
  ge::NodePtr g_node = graph->AddNode(g_op);
  //add descriptor
  vector<int64_t> dim({4, -1, -1, 16});
  vector<std::pair<int64_t, int64_t>> range({{1, 33}, {1, 12}});
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NCHW);
  tensor_desc.SetFormat(FORMAT_NCHW);
  tensor_desc.SetDataType(DT_INT32);
  tensor_desc.SetShapeRange(range);
  g_op->AddInputDesc("x", tensor_desc);
  g_op->AddOutputDesc("z", tensor_desc);
  ge::AttrUtils::SetInt(g_op, FE_IMPLY_TYPE, 6);

  
  Status ret = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g_node, "tbe-custom");
  ASSERT_EQ(ret, fe::SUCCESS);
  vector<int64_t> dim_result({4, -1, -1, 16, 32});
  vector<std::pair<int64_t, int64_t>> range_expect({{4, 4}, {1, 2}, {1, 12}, {16, 16}, {32, 32}});
  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetInputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetInputDesc(0).GetDataType(), DT_UINT8);
  EXPECT_EQ(g_op->GetInputDesc(0).GetShape().GetDims(), dim_result);
  EXPECT_EQ(GetShapeRange(g_op->GetInputDesc(0)), range_expect);

  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetDataType(), DT_UINT8);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetShape().GetDims(), dim_result);
  EXPECT_EQ(GetShapeRange(g_op->GetOutputDesc(0)), range_expect);
}

/* Test SetDtypeAndFormatByPrecisionMode on op G and H, relation in graph is
 * G->H. G is the father of H. G's format after OpFormatDtypeJudge will become NC1HWC0.
 * H's format is NCHW, and its op kernel supports NCHW and NC1HWC0. Due to
 * consecutive principle, we will pick NC1HWC0 based on its predecessor is
 * NC1HWC0. We do this operation for dtype as well. If there is no common
 * dtype between ops kernel and its father, we will still pick father's format
 * and pick the first Dtype belongs to this format.
 * After OpFormatDtypeJudge, Op G1 format is FORMAT_NC1HWC0 and Dtype is Fp
 * Op H1 format is FORMAT_NC1HWC0 and Dtype is Fp */
TEST_F(UTEST_fusion_engine_op_judge_unknown_shape, set_two_nodes_format_dtype_01)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g_op = std::make_shared<OpDesc>("G1", "G");
  OpDescPtr h_op = std::make_shared<OpDesc>("H1", "H");

  //add descriptor
  vector<int64_t> dim({4, -1, -1, 16});
  vector<std::pair<int64_t, int64_t>> range({{1, 33}, {1, 12}});
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NCHW);
  tensor_desc.SetFormat(FORMAT_NCHW);
  tensor_desc.SetDataType(DT_FLOAT);
  tensor_desc.SetShapeRange(range);
  g_op->AddInputDesc("x", tensor_desc);
  g_op->AddOutputDesc("z", tensor_desc);
  ge::AttrUtils::SetInt(g_op, FE_IMPLY_TYPE, 6);
  ge::NodePtr g_node = graph->AddNode(g_op);

  vector<int64_t> dim_h({1, -1, 3, -1});
  vector<std::pair<int64_t, int64_t>> range1({{1, 2}, {1, 4}});
  GeShape shape_h(dim_h);
  GeTensorDesc tensor_desc_h(shape_h);
  tensor_desc_h.SetOriginFormat(FORMAT_NHWC);
  tensor_desc_h.SetFormat(FORMAT_NHWC);
  tensor_desc_h.SetDataType(DT_FLOAT16);
  tensor_desc_h.SetShapeRange(range1);
  h_op->AddInputDesc("x", tensor_desc_h);
  h_op->AddOutputDesc("z", tensor_desc_h);
  ge::AttrUtils::SetInt(h_op, FE_IMPLY_TYPE, 6);
  ge::NodePtr h_node = graph->AddNode(h_op);
  GraphUtils::AddEdge(g_node->GetOutDataAnchor(0), h_node->GetInDataAnchor(0));

  
  Status ret1 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret1 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g_node, "tbe-custom");
  Status ret2 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(h_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret2 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(h_node, "tbe-custom");
  ASSERT_EQ(ret1, fe::SUCCESS);
  ASSERT_EQ(ret2, fe::SUCCESS);
  vector<int64_t> dim_result_nch_w_to_NC1_hw_c0({4, -1, -1, 16, 16});
  vector<int64_t> dim_result_nhw_c_to_NC1_hw_c0({1, -1, -1, 3, 16});
  vector<std::pair<int64_t, int64_t>> range_expect({{4, 4}, {1, 3}, {1, 12}, {16, 16}, {16, 16}});
  vector<std::pair<int64_t, int64_t>> range_expect1({{1, 1}, {1, 1}, {1, 2}, {3, 3}, {16, 16}});
  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetInputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetInputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(g_op->GetInputDesc(0).GetShape().GetDims(), dim_result_nch_w_to_NC1_hw_c0);
  EXPECT_EQ(GetShapeRange(g_op->GetInputDesc(0)), range_expect);

  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetShape().GetDims(), dim_result_nch_w_to_NC1_hw_c0);
  EXPECT_EQ(GetShapeRange(g_op->GetOutputDesc(0)), range_expect);

  EXPECT_EQ(h_op->GetInputDesc(0).GetFormat(), FORMAT_NHWC);
  EXPECT_EQ(h_op->GetInputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(h_op->GetInputDesc(0).GetShape().GetDims(), dim_h);
  EXPECT_EQ(GetShapeRange(h_op->GetInputDesc(0)), range1);

  EXPECT_EQ(h_op->GetOutputDesc(0).GetFormat(), FORMAT_NHWC);
  EXPECT_EQ(h_op->GetOutputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(h_op->GetOutputDesc(0).GetShape().GetDims(), dim_h);
  EXPECT_EQ(GetShapeRange(h_op->GetOutputDesc(0)), range1);
}


/* Test SetDtypeAndFormatByPrecisionMode on op G and H, relation in graph is
 * G->H. G is the father of H. G's format after OpFormatDtypeJudge will become NC1HWC0.
 * H's format is NCHW, and its op kernel supports NCHW and NC1HWC0. Due to
 * consecutive principle, we will pick NC1HWC0 based on its predecessor is
 * NC1HWC0. We do this operation for dtype as well. If there is no common
 * dtype between ops kernel and its father, we will still pick father's format
 * and pick the first Dtype belongs to this format.
 * After OpFormatDtypeJudge, Op G1 format is FORMAT_NC1HWC0 and Dtype is Fp
 * Op H1 format is FORMAT_NC1HWC0 and Dtype is Fp */
TEST_F(UTEST_fusion_engine_op_judge_unknown_shape, set_two_nodes_format_dtype_02)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g_op = std::make_shared<OpDesc>("G1", "G");
  OpDescPtr h_op = std::make_shared<OpDesc>("I1", "I");

  //add descriptor
  vector<int64_t> dim({4, -1, -1, 16});
  vector<std::pair<int64_t, int64_t>> range({{1, 33}, {1, 12}});
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NCHW);
  tensor_desc.SetFormat(FORMAT_NCHW);
  tensor_desc.SetDataType(DT_FLOAT);
  tensor_desc.SetShapeRange(range);
  g_op->AddInputDesc("x", tensor_desc);
  g_op->AddOutputDesc("z", tensor_desc);
  ge::AttrUtils::SetInt(g_op, FE_IMPLY_TYPE, 6);
  ge::NodePtr g_node = graph->AddNode(g_op);

  vector<int64_t> dim_i({1, -1, 3, -1});
  vector<std::pair<int64_t, int64_t>> range1({{1, 2}, {1, 4}});
  GeShape shape_i(dim_i);
  GeTensorDesc tensor_desc_i(shape_i);
  tensor_desc_i.SetOriginFormat(FORMAT_NHWC);
  tensor_desc_i.SetFormat(FORMAT_NHWC);
  tensor_desc_i.SetDataType(DT_FLOAT16);
  tensor_desc_i.SetShapeRange(range1);
  h_op->AddInputDesc("x", tensor_desc_i);
  h_op->AddOutputDesc("z", tensor_desc_i);
  ge::AttrUtils::SetInt(h_op, FE_IMPLY_TYPE, 6);
  ge::NodePtr h_node = graph->AddNode(h_op);

  GraphUtils::AddEdge(g_node->GetOutDataAnchor(0), h_node->GetInDataAnchor(0));

  
  Status ret1 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret1 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g_node, "tbe-custom");
  Status ret2 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(h_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret2 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(h_node, "tbe-custom");
  ASSERT_EQ(ret1, fe::SUCCESS);
  ASSERT_EQ(ret2, fe::SUCCESS);

  vector<int64_t> dim_result({4, -1, -1, 16, 16});
  vector<std::pair<int64_t, int64_t>> range_expect({{4, 4}, {1, 3}, {1, 12}, {16, 16}, {16, 16}});
  vector<std::pair<int64_t, int64_t>> range_expect1({{1, 1}, {1, 4}, {1, 2}, {3, 3}});
  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetInputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetInputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(g_op->GetInputDesc(0).GetShape().GetDims(), dim_result);
  EXPECT_EQ(GetShapeRange(g_op->GetInputDesc(0)), range_expect);

  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetShape().GetDims(), dim_result);
  EXPECT_EQ(GetShapeRange(g_op->GetOutputDesc(0)), range_expect);

  vector<int64_t> dim_result_h({1, -1, -1, 3});
  vector<int64_t> dim_result_h5_d({1, 1, 2, 3, 32});
  EXPECT_EQ(h_op->GetInputDesc(0).GetFormat(), FORMAT_NHWC);
  EXPECT_EQ(h_op->GetInputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(h_op->GetInputDesc(0).GetShape().GetDims(), dim_i);
  EXPECT_EQ(GetShapeRange(h_op->GetInputDesc(0)), range1);

  EXPECT_EQ(h_op->GetOutputDesc(0).GetFormat(), FORMAT_NCHW);
  EXPECT_EQ(h_op->GetOutputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(h_op->GetOutputDesc(0).GetShape().GetDims(), dim_result_h);
  EXPECT_EQ(GetShapeRange(h_op->GetOutputDesc(0)), range_expect1);
}

TEST_F(UTEST_fusion_engine_op_judge_unknown_shape, set_op_shape_dim_01)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr op = std::make_shared<OpDesc>("relu", "GWithoutReshapeType");

  //add descriptor
  vector<int64_t> dim1({-1});
  vector<std::pair<int64_t, int64_t>> range({{1, 7}});
  GeShape shape1(dim1);
  GeTensorDesc input(shape1);
  input.SetOriginFormat(FORMAT_NHWC);
  input.SetFormat(FORMAT_NHWC);
  input.SetDataType(DT_FLOAT);
  input.SetShapeRange(range);

  vector<int64_t> dim2({2,-1});
  vector<std::pair<int64_t, int64_t>> range1({{1, 3}});
  GeShape shape2(dim2);
  GeTensorDesc output(shape2);
  output.SetOriginFormat(FORMAT_NHWC);
  output.SetFormat(FORMAT_NHWC);
  output.SetDataType(DT_FLOAT);
  output.SetShapeRange(range1);

  op->AddInputDesc("x", input);
  op->AddOutputDesc("z", output);
  ge::AttrUtils::SetInt(op, "_fe_imply_type", static_cast<int>(EN_IMPL_HW_TBE));
  ge::NodePtr g_node = graph->AddNode(op);
  OpImplTypeJudgePtr op_impl_type_judge_ptr = std::make_shared<OpImplTypeJudge>(AI_CORE_NAME, fe_ops_kernel_info_store_ptr_);
  
  Status ret1 = op_impl_type_judge_ptr->MultiThreadJudge(*(graph.get()));
  ASSERT_EQ(ret1, fe::SUCCESS);
  Status ret2 = op_format_dtype_judge_ptr_->Judge(*(graph.get()));
  ret2 = op_format_dtype_judge_ptr_->SetFormat(*(graph.get()));
  ASSERT_EQ(ret2, fe::SUCCESS);

  vector<int64_t> dim_result1({1,-1,1,1,16});
  vector<std::pair<int64_t, int64_t>> range_expect({{1, 1}, {1, 1}, {1, 1}, {1, 1}, {16, 16}});
  EXPECT_EQ(op->GetInputDesc(0).GetShape().GetDims(), dim_result1);
  EXPECT_EQ(GetShapeRange(op->GetInputDesc(0)), range_expect);

  vector<int64_t> dim_result2({1,1,2,-1,16});
  vector<std::pair<int64_t, int64_t>> range_expect1({{1, 1}, {1, 1}, {2, 2}, {1, 3}, {16, 16}});
  EXPECT_EQ(op->GetOutputDesc(0).GetShape().GetDims(), dim_result2);
  EXPECT_EQ(GetShapeRange(op->GetOutputDesc(0)), range_expect1);
}


TEST_F(UTEST_fusion_engine_op_judge_unknown_shape, set_op_shape_dim_02)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr op = std::make_shared<OpDesc>("mat1", "MatMul");

  //add descriptor
  vector<int64_t> dim1({-1});
  vector<std::pair<int64_t, int64_t>> range({{1, 7}});
  GeShape shape1(dim1);
  GeTensorDesc input(shape1);
  input.SetOriginFormat(FORMAT_NHWC);
  input.SetFormat(FORMAT_NHWC);
  input.SetDataType(DT_FLOAT);
  input.SetShapeRange(range);

  vector<int64_t> dim2({2,-1});
  vector<std::pair<int64_t, int64_t>> range1({{1, 3}});
  GeShape shape2(dim2);
  GeTensorDesc output(shape2);
  output.SetOriginFormat(FORMAT_NHWC);
  output.SetFormat(FORMAT_NHWC);
  output.SetDataType(DT_FLOAT);
  output.SetShapeRange(range1);

  op->AddInputDesc("x", input);
  op->AddOutputDesc("z", output);
  ge::AttrUtils::SetInt(op, "_fe_imply_type", static_cast<int>(EN_IMPL_HW_TBE));
  ge::NodePtr g_node = graph->AddNode(op);
  OpImplTypeJudgePtr op_impl_type_judge_ptr = std::make_shared<OpImplTypeJudge>(AI_CORE_NAME, fe_ops_kernel_info_store_ptr_);
  
  Status ret1 = op_impl_type_judge_ptr->MultiThreadJudge(*(graph.get()));
  ASSERT_EQ(ret1, fe::SUCCESS);
  Status ret2 = op_format_dtype_judge_ptr_->Judge(*(graph.get()));
  ret2 = op_format_dtype_judge_ptr_->SetFormat(*(graph.get()));
  ASSERT_EQ(ret2, fe::SUCCESS);

  vector<int64_t> dim_result1({1,-1,1,1,16});
  vector<std::pair<int64_t, int64_t>> range_expect({{1, 1}, {1, 1}, {1, 1}, {1, 1}, {16, 16}});
  EXPECT_EQ(op->GetInputDesc(0).GetShape().GetDims(), dim_result1);
  EXPECT_EQ(GetShapeRange(op->GetInputDesc(0)), range_expect);

  vector<int64_t> dim_result2({1,1,2,-1,16});
  vector<std::pair<int64_t, int64_t>> range_expect1({{1, 1}, {1, 1}, {2, 2}, {1, 3}, {16, 16}});
  EXPECT_EQ(op->GetOutputDesc(0).GetShape().GetDims(), dim_result2);
  EXPECT_EQ(GetShapeRange(op->GetOutputDesc(0)), range_expect1);
}

TEST_F(UTEST_fusion_engine_op_judge_unknown_shape, set_op_shape_dim_03)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr op = std::make_shared<OpDesc>("relu", "GWithoutReshapeType");

  //add descriptor
  vector<int64_t> dim1({4,-1,-1});
  vector<std::pair<int64_t, int64_t>> range({{1, 5}, {1, 6}});
  GeShape shape1(dim1);
  GeTensorDesc input(shape1);
  input.SetOriginFormat(FORMAT_NHWC);
  input.SetFormat(FORMAT_NHWC);
  input.SetDataType(DT_FLOAT);
  input.SetShapeRange(range);

  vector<int64_t> dim2({5,-1,-1,-1});
  vector<std::pair<int64_t, int64_t>> range1({{1, 6}, {1, 7}, {1, 8}});
  GeShape shape2(dim2);
  GeTensorDesc output(shape2);
  output.SetOriginFormat(FORMAT_NHWC);
  output.SetFormat(FORMAT_NHWC);
  output.SetDataType(DT_FLOAT);
  output.SetShapeRange(range1);

  op->AddInputDesc("x", input);
  op->AddOutputDesc("z", output);
  ge::AttrUtils::SetInt(op, "_fe_imply_type", static_cast<int>(EN_IMPL_HW_TBE));
  ge::NodePtr g_node = graph->AddNode(op);

  OpImplTypeJudgePtr op_impl_type_judge_ptr = std::make_shared<OpImplTypeJudge>(AI_CORE_NAME, fe_ops_kernel_info_store_ptr_);
  
  Status ret1 = op_impl_type_judge_ptr->MultiThreadJudge(*(graph.get()));
  ASSERT_EQ(ret1, fe::SUCCESS);
  Status ret2 = op_format_dtype_judge_ptr_->Judge(*(graph.get()));
  ret2 = op_format_dtype_judge_ptr_->SetFormat(*(graph.get()));
  ASSERT_EQ(ret2, fe::SUCCESS);

  vector<int64_t> dim_result1({1,-1,4,-1,16});
  vector<std::pair<int64_t, int64_t>> range_ex({{1, 1}, {1, 1}, {4, 4}, {1, 5}, {16, 16}});
  EXPECT_EQ(op->GetInputDesc(0).GetShape().GetDims(), dim_result1);
  EXPECT_EQ(GetShapeRange(op->GetInputDesc(0)), range_ex);

  vector<int64_t> dim_result2({5,-1,-1,-1,16});
  vector<std::pair<int64_t, int64_t>> range_ex_out({{5, 5}, {1, 1}, {1, 6}, {1, 7}, {16, 16}});
  EXPECT_EQ(op->GetOutputDesc(0).GetShape().GetDims(), dim_result2);
  EXPECT_EQ(GetShapeRange(op->GetOutputDesc(0)), range_ex_out);
}


TEST_F(UTEST_fusion_engine_op_judge_unknown_shape, set_op_shape_dim_04) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr op = std::make_shared<OpDesc>("relu", "MatMul");

  //add descriptor
  vector<int64_t> dim1({-1, -1});
  vector<std::pair<int64_t, int64_t>> range({{1, 5},
                                             {1, 6}});
  GeShape shape1(dim1);
  GeTensorDesc input(shape1);
  input.SetOriginFormat(FORMAT_NHWC);
  input.SetFormat(FORMAT_NHWC);
  input.SetDataType(DT_FLOAT);
  input.SetShapeRange(range);

  vector<int64_t> dim2({5, -1, -1, -1});
  vector<std::pair<int64_t, int64_t>> range1({{1, 6},
                                              {1, 7},
                                              {1, 8}});
  GeShape shape2(dim2);
  GeTensorDesc output(shape2);
  output.SetOriginFormat(FORMAT_NHWC);
  output.SetFormat(FORMAT_NHWC);
  output.SetDataType(DT_FLOAT);
  output.SetShapeRange(range1);

  op->AddInputDesc("x", input);
  op->AddOutputDesc("z", output);
  ge::AttrUtils::SetInt(op, "_fe_imply_type", static_cast<int>(EN_IMPL_HW_TBE));
  ge::NodePtr g_node = graph->AddNode(op);

  OpImplTypeJudgePtr op_impl_type_judge_ptr = std::make_shared<OpImplTypeJudge>(
          AI_CORE_NAME, fe_ops_kernel_info_store_ptr_);

  Status ret1 = op_impl_type_judge_ptr->MultiThreadJudge(*(graph.get()));
  ASSERT_EQ(ret1, fe::SUCCESS);
  Status ret2 = op_format_dtype_judge_ptr_->Judge(*(graph.get()));
  ret2 = op_format_dtype_judge_ptr_->SetFormat(*(graph.get()));
  ASSERT_EQ(ret2, fe::SUCCESS);

  vector<int64_t> dim_result1({1, 1, -1, -1, 16});
  vector<std::pair<int64_t, int64_t>> range_ex({{1,  1},
                                               {1,  1},
                                               {1,  5},
                                               {1,  6},
                                               {16, 16}});
  EXPECT_EQ(op->GetInputDesc(0).GetShape().GetDims(), dim_result1);
  EXPECT_EQ(GetShapeRange(op->GetInputDesc(0)), range_ex);

  vector<int64_t> dim_result2({5, -1, -1, -1, 16});
  vector<std::pair<int64_t, int64_t>> range_ex_out({{5,  5},
                                                  {1,  1},
                                                  {1,  6},
                                                  {1,  7},
                                                  {16, 16}});
  EXPECT_EQ(op->GetOutputDesc(0).GetShape().GetDims(), dim_result2);
  EXPECT_EQ(GetShapeRange(op->GetOutputDesc(0)), range_ex_out);
}

/* Test SetDtypeAndFormatByPrecisionMode on op ConvTemp (TBE_builtin) without predecessor node
 * Set Shape of Fragz as {HWC1, N/16, 16, C0} from NHWC for int8 input*/
TEST_F(UTEST_fusion_engine_op_judge_unknown_shape, set_input_format_succ_format_changed_Conv_Tbe_2)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g_op = std::make_shared<OpDesc>("ConvTemp", "ConvTemp");
  ge::NodePtr g_node = graph->AddNode(g_op);
  //add descriptor

  vector<int64_t> dim1({1, -1, -1, -1});
  vector<std::pair<int64_t, int64_t>> range1({{1, 66}, {1, 64}, {1, 64}});
  GeShape shape1(dim1);
  GeTensorDesc tensor_desc1(shape1);
  tensor_desc1.SetOriginFormat(FORMAT_NCHW);
  tensor_desc1.SetFormat(FORMAT_NCHW);
  tensor_desc1.SetDataType(DT_INT8);
  tensor_desc1.SetShapeRange(range1);
  g_op->AddInputDesc("xasd", tensor_desc1);

  vector<int64_t> dim2({-1, -1, 7, -1});
  vector<std::pair<int64_t, int64_t>> range2({{1, 64}, {1, 38}, {1, 7}});
  GeShape shape2(dim2);
  GeTensorDesc tensor_desc2(shape2);
  tensor_desc2.SetOriginFormat(FORMAT_NHWC);
  tensor_desc2.SetFormat(FORMAT_NHWC);
  tensor_desc2.SetDataType(DT_INT8);
  tensor_desc2.SetShapeRange(range2);
  g_op->AddInputDesc("wasd", tensor_desc2);

  GeShape shape3;
  GeTensorDesc tensor_desc3(shape3);
  vector<std::pair<int64_t, int64_t>> range3({});
  tensor_desc3.SetOriginFormat(FORMAT_NCHW);
  tensor_desc3.SetFormat(FORMAT_NCHW);
  tensor_desc3.SetDataType(DT_FLOAT16);
  tensor_desc3.SetShapeRange(range3);
  g_op->AddInputDesc("basd", tensor_desc3);

  vector<int64_t> dimo({1, -1, -1, -1});
  vector<std::pair<int64_t, int64_t>> range4({{1, 64}, {1, 30}, {1, 30}});
  GeShape shapeo(dimo);
  GeTensorDesc tensor_desco(shapeo);
  tensor_desco.SetOriginFormat(FORMAT_NCHW);
  tensor_desco.SetFormat(FORMAT_NCHW);
  tensor_desco.SetDataType(DT_INT8);
  tensor_desco.SetShapeRange(range4);
  g_op->AddOutputDesc("yasd", tensor_desco);

  ge::AttrUtils::SetInt(g_op, FE_IMPLY_TYPE, 6);//TBE
  
  Status ret = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g_node, "tbe-custom");
  ASSERT_EQ(ret, fe::SUCCESS);
  ASSERT_EQ(g_op->GetAllInputsDesc().size(), 3);
  vector<int64_t> dim_result_x = {1,-1,-1,-1,32};
  vector<std::pair<int64_t, int64_t>> range_ex1({{1, 1}, {1, 3}, {1, 64}, {1, 64}, {32, 32}});
  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetInputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetInputDesc(0).GetDataType(), DT_INT8);
  EXPECT_EQ(g_op->GetInputDesc(0).GetShape().GetDims(), dim_result_x);
  EXPECT_EQ(GetShapeRange(g_op->GetInputDesc(0)), range_ex1);

  vector<int64_t> dim_result_w = {-1,-1,16,32};
  vector<std::pair<int64_t, int64_t>> range_ex2({{7, 266}, {1, 4}, {16, 16}, {32, 32}});
  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetInputDesc(1).GetFormat()), FORMAT_FRACTAL_Z);
  EXPECT_EQ(g_op->GetInputDesc(1).GetDataType(), DT_INT8);
  EXPECT_EQ(g_op->GetInputDesc(1).GetShape().GetDims(), dim_result_w);
  EXPECT_EQ(GetShapeRange(g_op->GetInputDesc(1)), range_ex2);

  EXPECT_EQ(g_op->GetInputDesc(2).GetFormat(), FORMAT_NCHW);
  EXPECT_EQ(g_op->GetInputDesc(2).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(GetShapeRange(g_op->GetInputDesc(2)), range3);

  vector<int64_t> dim_result_o = {1,-1,-1,-1,32};
  vector<std::pair<int64_t, int64_t>> range_ex3({{1, 1}, {1, 2}, {1, 30}, {1, 30}, {32, 32}});
  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetDataType(), DT_INT8);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetShape().GetDims(), dim_result_o);
  EXPECT_EQ(GetShapeRange(g_op->GetOutputDesc(0)), range_ex3);

}

/* For pytorch, Data's format is FRACTAL_NZ. */
TEST_F(UTEST_fusion_engine_op_judge_unknown_shape, pytorch_set_two_nodes_format_dtype_allow_fp32_to_fp16_2)
{
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = ALLOW_FP32_TO_FP16;
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr data_op = std::make_shared<OpDesc>("Data", fe::DATA);
  OpDescPtr h_op = std::make_shared<OpDesc>("bm", "BatchMatMul2");

  //add descriptor
  vector<int64_t> dim({-1, -1, -1, 3, 16, 16});
  vector<std::pair<int64_t, int64_t>> range1({{1, 4}, {1, 33}, {1, 2}});
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NCHW);
  tensor_desc.SetFormat(FORMAT_FRACTAL_NZ);
  tensor_desc.SetDataType(DT_FLOAT);
  tensor_desc.SetShapeRange(range1);
  data_op->AddInputDesc("x", tensor_desc);
  data_op->AddOutputDesc("z", tensor_desc);
  ge::AttrUtils::SetInt(data_op, FE_IMPLY_TYPE, 6);
  ge::NodePtr data_node = graph->AddNode(data_op);

  vector<int64_t> dim_h({-1, -1, -1, 32});
  vector<std::pair<int64_t, int64_t>> range2({{1, 4}, {1, 33}, {1, 48}});
  GeShape shape_h(dim_h);
  GeTensorDesc tensor_desc_h(shape_h);
  tensor_desc_h.SetOriginFormat(FORMAT_NCHW);
  tensor_desc_h.SetFormat(FORMAT_NCHW);
  tensor_desc_h.SetDataType(DT_FLOAT);
  tensor_desc_h.SetShapeRange(range2);
  h_op->AddInputDesc("x", tensor_desc_h);
  h_op->AddOutputDesc("z", tensor_desc_h);
  ge::AttrUtils::SetInt(h_op, FE_IMPLY_TYPE, 6);
  ge::NodePtr h_node = graph->AddNode(h_op);
  GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), h_node->GetInDataAnchor(0));
  
  Status ret2 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(h_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret2 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(h_node, "tbe-custom");
  vector<int64_t> dim_ex({-1, -1, 2, -1, 16, 16});
  ASSERT_EQ(ret2, fe::SUCCESS);
  EXPECT_EQ(ge::GetPrimaryFormat(data_op->GetInputDesc(0).GetFormat()), FORMAT_FRACTAL_NZ);
  EXPECT_EQ(data_op->GetInputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(data_op->GetInputDesc(0).GetShape().GetDims(), dim);
  vector<std::pair<int64_t, int64_t>> range_ex({{1, 4}, {1, 33}, {2, 2}, {1, 3}, {16, 16}, {16, 16}});
  EXPECT_EQ(GetShapeRange(data_op->GetInputDesc(0)), range1);

  EXPECT_EQ(ge::GetPrimaryFormat(data_op->GetOutputDesc(0).GetFormat()), FORMAT_FRACTAL_NZ);
  EXPECT_EQ(data_op->GetOutputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(data_op->GetOutputDesc(0).GetShape().GetDims(), dim);
  EXPECT_EQ(GetShapeRange(data_op->GetOutputDesc(0)), range1);

  EXPECT_EQ(ge::GetPrimaryFormat(h_op->GetInputDesc(0).GetFormat()), FORMAT_FRACTAL_NZ);
  EXPECT_EQ(h_op->GetInputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(h_op->GetInputDesc(0).GetShape().GetDims(), dim_ex);
  EXPECT_EQ(GetShapeRange(h_op->GetInputDesc(0)), range_ex);

  EXPECT_EQ(ge::GetPrimaryFormat(h_op->GetOutputDesc(0).GetFormat()), FORMAT_FRACTAL_NZ);
  EXPECT_EQ(h_op->GetOutputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(h_op->GetOutputDesc(0).GetShape().GetDims(), dim_ex);
  EXPECT_EQ(GetShapeRange(h_op->GetOutputDesc(0)), range_ex);
}
