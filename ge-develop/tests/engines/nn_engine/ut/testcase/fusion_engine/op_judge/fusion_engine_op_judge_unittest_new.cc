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
#include "graph/ge_context.h"
#include "fe_llt_utils.h"
#define private public
#define protected public
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/ge_local_context.h"
#include "adapter/common/op_store_adapter_manager.h"
#include "adapter/tbe_adapter/tbe_op_store_adapter.h"
#include "graph_optimizer/op_setter/op_setter.h"
#include "graph_optimizer/op_judge/imply_type/op_impl_type_judge.h"
#include "graph_optimizer/op_judge/format_and_dtype/op_format_dtype_judge.h"
#include "graph_optimizer/op_judge/format_and_dtype/strategy/matcher/dtype/op_dtype_rise_matcher.h"
#include "graph_optimizer/op_judge/format_and_dtype/strategy/matcher/format/op_format_matcher.h"

#include "graph_optimizer/shape_format_transfer/trans_node_manager/trans_node_manager.h"
#include "graph/debug/ge_attr_define.h"
#include "common/platform_utils.h"
#include "common/configuration.h"
#include "ops_store/ops_kernel_manager.h"
#include "ffts_type.h"

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
using OpSetterPtr = std::shared_ptr<OpSetter>;
using TransNodeManagerPtr = std::shared_ptr<TransNodeManager>;

class UTEST_fusion_engine_op_judge_new_unittest : public testing::Test
{
protected:
  void SetUp()
  {
    TbeOpStoreAdapterPtr tbe_adapter_ptr = std::dynamic_pointer_cast<TbeOpStoreAdapter>(OpStoreAdapterManager::Instance(AI_CORE_NAME).GetOpStoreAdapter(EN_IMPL_HW_TBE));
    tbe_adapter_ptr->Initialize(std::map<std::string, std::string>());
    tbe_adapter_ptr->GetOpSpecificInfo = GetOpSpecificInfoSub;
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

    OpsKernelManager::Instance(AI_CORE_NAME).Initialize();
    fe_ops_kernel_info_store_ptr_->Initialize(options);
    store_info.emplace_back(cce_custom);
    Configuration::Instance(fe::AI_CORE_NAME).ops_store_info_vector_ = (store_info);
    ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = ALLOW_MIX_PRECISION;
    OpsKernelManager::Instance(AI_CORE_NAME).Finalize();

    OpsKernelManager::Instance(AI_CORE_NAME).Initialize();
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
  static bool GetOpSpecificInfoSub(const te::TbeOpInfo &tbeOpInfo, std::string &opCoreTypeString) {
    nlohmann::json op_specific_info;
    op_specific_info["core_type_value"] = "AiCore,MIX";
    opCoreTypeString = op_specific_info.dump();
    return true;
  }

  static void CreateOneOpGraph(ComputeGraphPtr graph) {

    OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "Activation");

    // add descriptor
    vector<int64_t> dim(4, 1);
    GeShape shape(dim);
    GeTensorDesc out_desc(shape);
    out_desc.SetOriginFormat(FORMAT_NCHW);
    out_desc.SetFormat(FORMAT_NCHW);
    out_desc.SetDataType(DT_FLOAT16);
    relu_op->AddInputDesc("x", out_desc);
    relu_op->AddOutputDesc("y", out_desc);

    NodePtr relu_node = graph->AddNode(relu_op);
  }
  static void CreateTwoOpGraph(ComputeGraphPtr graph) {
    // 创建Node
    OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
    OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "Activation");

    // add descriptor
    vector<int64_t> dim(4, 1);
    GeShape shape(dim);
    GeTensorDesc out_desc(shape);
    out_desc.SetOriginFormat(FORMAT_NCHW);
    out_desc.SetFormat(FORMAT_NCHW);
    out_desc.SetDataType(DT_FLOAT16);
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
    vector<int64_t> dims = {1,2,3,4};
    GeShape shape(dims);

    GeTensorDesc in_desc1(shape);
    in_desc1.SetOriginFormat(FORMAT_NCHW);
    in_desc1.SetFormat(FORMAT_NCHW);
    in_desc1.SetDataType(DT_FLOAT16);
    relu_op->AddInputDesc("x", in_desc1);

    GeTensorDesc out_desc1(shape);
    out_desc1.SetOriginFormat(FORMAT_HWCN);
    out_desc1.SetFormat(FORMAT_HWCN);
    out_desc1.SetDataType(DT_FLOAT16);
    relu_op->AddOutputDesc("y", out_desc1);


    GeTensorDesc in_desc2(shape);
    in_desc2.SetOriginFormat(FORMAT_FRACTAL_Z);
    in_desc2.SetFormat(FORMAT_FRACTAL_Z);
    in_desc2.SetDataType(DT_FLOAT16);
    bn_op->AddInputDesc("x", in_desc2);

    GeTensorDesc out_desc2(shape);
    out_desc2.SetOriginFormat(FORMAT_NHWC);
    out_desc2.SetFormat(FORMAT_NHWC);
    out_desc2.SetDataType(DT_FLOAT16);
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
    GeShape Shape(dim);
    GeTensorDesc out_desc(Shape);
    out_desc.SetOriginFormat(FORMAT_NCHW);
    out_desc.SetFormat(FORMAT_NCHW);
    out_desc.SetDataType(DT_FLOAT16);
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
    vector<int64_t> dim(4, 1);
    GeShape shape(dim);
    GeTensorDesc out_desc(shape);
    out_desc.SetOriginFormat(FORMAT_NCHW);
    out_desc.SetFormat(FORMAT_NCHW);
    out_desc.SetDataType(DT_FLOAT16);
    relu_op->AddInputDesc("x", out_desc);
    relu_op->AddOutputDesc("y", out_desc);

    GeTensorDesc bn_desc(shape);
    bn_desc.SetOriginFormat(FORMAT_NCHW);
    bn_desc.SetFormat(FORMAT_NC1HWC0);
    bn_desc.SetDataType(DT_FLOAT16);
    bn_op->AddInputDesc("x", bn_desc);
    bn_op->AddOutputDesc("y", bn_desc);

    ge::AttrUtils::SetInt(bn_op, "_fe_imply_type", static_cast<int>(EN_RESERVED));
    ge::AttrUtils::SetInt(relu_op, "_fe_imply_type", static_cast<int>(EN_RESERVED));

    NodePtr bn_node = graph->AddNode(bn_op);
    NodePtr relu_node = graph->AddNode(relu_op);

    GraphUtils::AddEdge(bn_node->GetOutDataAnchor(0), relu_node->GetInDataAnchor(0));
  }


  static void CreateTwoMultiOpGraph(ComputeGraphPtr graph) {
    // 创建Node
    OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
    OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "Activation");

    // add descriptor
    vector<int64_t> dim(4, 1);
    GeShape shape(dim);
    GeTensorDesc out_desc(shape);
    out_desc.SetOriginFormat(FORMAT_NCHW);
    out_desc.SetFormat(FORMAT_NCHW);
    out_desc.SetDataType(DT_FLOAT16);
    relu_op->AddInputDesc("x", out_desc);
    relu_op->AddInputDesc("x1", out_desc);
    relu_op->AddOutputDesc("y", out_desc);
    relu_op->AddOutputDesc("y1", out_desc);

    GeTensorDesc bn_desc(shape);
    bn_desc.SetOriginFormat(FORMAT_NCHW);
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

  static void CreateTwoPluginTbeOpGraph(ComputeGraphPtr graph) {
    OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
    OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "Activation");

    // add descriptor
    vector<int64_t> dim(4, 1);
    GeShape shape(dim);
    GeTensorDesc out_desc(shape);
    out_desc.SetOriginFormat(FORMAT_NCHW);
    out_desc.SetFormat(FORMAT_NCHW);
    out_desc.SetDataType(DT_FLOAT16);
    relu_op->AddInputDesc("x", out_desc);
    relu_op->AddOutputDesc("y", out_desc);

    GeTensorDesc bn_desc(shape);
    out_desc.SetOriginFormat(FORMAT_NCHW);
    bn_desc.SetFormat(FORMAT_NC1HWC0);
    bn_desc.SetDataType(DT_FLOAT16);
    bn_op->AddInputDesc("x", bn_desc);
    bn_op->AddOutputDesc("y", bn_desc);

    ge::AttrUtils::SetInt(relu_op, "_fe_imply_type", static_cast<int>(EN_IMPL_PLUGIN_TBE));
    ge::AttrUtils::SetInt(bn_op, "_fe_imply_type", static_cast<int>(EN_IMPL_PLUGIN_TBE));

    NodePtr bn_node = graph->AddNode(bn_op);
    NodePtr relu_node = graph->AddNode(relu_op);

    GraphUtils::AddEdge(bn_node->GetOutDataAnchor(0), relu_node->GetInDataAnchor(0));
  }
};

TEST_F(UTEST_fusion_engine_op_judge_new_unittest, judge_nchw_c04_succ)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr conv_op = std::make_shared<OpDesc>("conv", "ConvTempC04");

  vector<int64_t> dim({64, 16, 7, 7});
  GeShape shape(dim);
  GeTensorDesc in_desc1(shape);
  in_desc1.SetOriginFormat(FORMAT_NCHW);
  in_desc1.SetFormat(FORMAT_NCHW);
  in_desc1.SetDataType(DT_FLOAT16);
  vector<int64_t> dim2({64, 3, 7, 7});
  GeShape shape2(dim2);
  GeTensorDesc in_desc2(shape2);
  in_desc2.SetOriginFormat(FORMAT_NCHW);
  in_desc2.SetFormat(FORMAT_NCHW);
  in_desc2.SetDataType(DT_FLOAT16);
  conv_op->AddInputDesc("x", in_desc1);
  conv_op->AddInputDesc("w", in_desc2);
  conv_op->AddOutputDesc("y", in_desc1);
  NodePtr conv_node = graph->AddNode(conv_op);
  ge::AttrUtils::SetInt(conv_op, FE_IMPLY_TYPE, 6);//TBE
  
  Status ret = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(conv_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(conv_node, "tbe-custom");
  EXPECT_EQ(fe::SUCCESS, ret);
  OpDescPtr conv_op_desc = conv_node->GetOpDesc();
  vector<int64_t> new_dim({64, 4, 7, 7, 4});
  EXPECT_EQ(conv_op_desc->GetInputDesc(0).GetShape().GetDims(), new_dim);
  vector<int64_t> new_dim2({13, 4, 16, 16});
  EXPECT_EQ(conv_op_desc->GetInputDesc(1).GetShape().GetDims(), new_dim2);
  EXPECT_EQ(conv_op_desc->GetOutputDesc(0).GetShape().GetDims(), new_dim);
}

TEST_F(UTEST_fusion_engine_op_judge_new_unittest, judge_nhwc_c04_succ)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr conv_op = std::make_shared<OpDesc>("conv", "ConvTempC04");

  vector<int64_t> dim({64, 7, 7, 16});
  GeShape shape(dim);
  GeTensorDesc in_desc1(shape);
  in_desc1.SetOriginFormat(FORMAT_NHWC);
  in_desc1.SetFormat(FORMAT_NHWC);
  in_desc1.SetDataType(DT_FLOAT16);
  vector<int64_t> dim2({64, 7, 7, 3});
  GeShape shape2(dim2);
  GeTensorDesc in_desc2(shape2);
  in_desc2.SetOriginFormat(FORMAT_NHWC);
  in_desc2.SetFormat(FORMAT_NHWC);
  in_desc2.SetDataType(DT_FLOAT16);
  conv_op->AddInputDesc("x", in_desc1);
  conv_op->AddInputDesc("w", in_desc2);
  conv_op->AddOutputDesc("y", in_desc1);
  NodePtr conv_node = graph->AddNode(conv_op);
  ge::AttrUtils::SetInt(conv_op, FE_IMPLY_TYPE, 6);//TBE

  
  Status ret = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(conv_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(conv_node, "tbe-custom");
  EXPECT_EQ(fe::SUCCESS, ret);
  OpDescPtr conv_op_desc = conv_node->GetOpDesc();
  vector<int64_t> new_dim({64, 4, 7, 7, 4});
  EXPECT_EQ(conv_op_desc->GetInputDesc(0).GetShape().GetDims(), new_dim);
  vector<int64_t> new_dim2({13, 4, 16, 16});
  EXPECT_EQ(conv_op_desc->GetInputDesc(1).GetShape().GetDims(), new_dim2);
  EXPECT_EQ(conv_op_desc->GetOutputDesc(0).GetShape().GetDims(), new_dim);
}

TEST_F(UTEST_fusion_engine_op_judge_new_unittest, judge_hwcn_c04_succ)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr conv_op = std::make_shared<OpDesc>("conv", "ConvTempC04");

  vector<int64_t> dim({7, 7, 16, 64});
  GeShape shape(dim);
  GeTensorDesc in_desc1(shape);
  in_desc1.SetOriginFormat(FORMAT_HWCN);
  in_desc1.SetFormat(FORMAT_HWCN);
  in_desc1.SetDataType(DT_FLOAT16);
  vector<int64_t> dim2({7, 7, 3, 64});
  GeShape shape2(dim2);
  GeTensorDesc in_desc2(shape2);
  in_desc2.SetOriginFormat(FORMAT_HWCN);
  in_desc2.SetFormat(FORMAT_HWCN);
  in_desc2.SetDataType(DT_FLOAT16);
  conv_op->AddInputDesc("x", in_desc1);
  conv_op->AddInputDesc("w", in_desc2);
  conv_op->AddOutputDesc("y", in_desc1);
  NodePtr conv_node = graph->AddNode(conv_op);
  ge::AttrUtils::SetInt(conv_op, FE_IMPLY_TYPE, 6);//TBE

  
  Status ret = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(conv_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(conv_node, "tbe-custom");
  EXPECT_EQ(fe::SUCCESS, ret);
  OpDescPtr conv_op_desc = conv_node->GetOpDesc();
  vector<int64_t> new_dim({64, 4, 7, 7, 4});
  EXPECT_EQ(conv_op_desc->GetInputDesc(0).GetShape().GetDims(), new_dim);
  vector<int64_t> new_dim2({13, 4, 16, 16});
  EXPECT_EQ(conv_op_desc->GetInputDesc(1).GetShape().GetDims(), new_dim2);
  EXPECT_EQ(conv_op_desc->GetOutputDesc(0).GetShape().GetDims(), new_dim);
}

/* Test SetDtypeAndFormatByPrecisionMode on op G without predecessor node
 * After OpFormatDtypeJudge, Op G1 format is FORMAT_NC1HWC0 and Dtype is Fp*/
TEST_F(UTEST_fusion_engine_op_judge_new_unittest, set_input_format_succ)
{
    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
    OpDescPtr g_op = std::make_shared<OpDesc>("G1", "G");
    ge::NodePtr g_node = graph->AddNode(g_op);
    //add descriptor
    vector<int64_t> dim({1, 2, 3, 4, 5});
    GeShape shape(dim);

    GeTensorDesc tensor_desc(shape);
    tensor_desc.SetOriginFormat(FORMAT_NC1HWC0);
    tensor_desc.SetFormat(FORMAT_NC1HWC0);
    tensor_desc.SetDataType(DT_FLOAT);
    g_op->AddInputDesc("x", tensor_desc);
    g_op->AddOutputDesc("z", tensor_desc);

    
    Status ret = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g_node, "tbe-custom");
    ASSERT_EQ(ret, fe::SUCCESS);
    EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetInputDesc(0).GetFormat()), FORMAT_NC1HWC0);
    EXPECT_EQ(g_op->GetInputDesc(0).GetDataType(), DT_FLOAT);
    EXPECT_EQ(g_op->GetInputDesc(0).GetDataType(), DT_FLOAT);
    EXPECT_EQ(g_op->GetInputDesc(0).GetShape().GetDims(), dim);

    EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
    EXPECT_EQ(g_op->GetOutputDesc(0).GetDataType(), DT_FLOAT);
    EXPECT_EQ(g_op->GetOutputDesc(0).GetShape().GetDims(), dim);
}

TEST_F(UTEST_fusion_engine_op_judge_new_unittest, set_input_format_succ_optional)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g_op = std::make_shared<OpDesc>("G1", "GV9");
  ge::NodePtr g_node = graph->AddNode(g_op);
  //add descriptor
  vector<int64_t> dim({1, 2, 3, 4, 5});
  GeShape shape(dim);

  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NC1HWC0);
  tensor_desc.SetFormat(FORMAT_NC1HWC0);
  tensor_desc.SetDataType(DT_FLOAT);
  GeTensorDesc data_desc_in_valid(shape, FORMAT_RESERVED, DT_UNDEFINED);
  g_op->AddInputDesc("x", tensor_desc);
  g_op->AddInputDesc("y", data_desc_in_valid);
  g_op->AddOutputDesc("z", tensor_desc);

  
  Status ret = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g_node, "tbe-custom");
  ASSERT_EQ(ret, fe::SUCCESS);
  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetInputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetInputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(g_op->GetInputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(g_op->GetInputDesc(0).GetShape().GetDims(), dim);

  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetShape().GetDims(), dim);
}

/* Test SetDtypeAndFormatByPrecisionMode on op G without predecessor node
 * After OpFormatDtypeJudge, Op G1 format is FORMAT_NC1HWC0 and Dtype is Fp*/
TEST_F(UTEST_fusion_engine_op_judge_new_unittest, set_input_format_succ_format_changed)
{
    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
    OpDescPtr g_op = std::make_shared<OpDesc>("G1", "G");
    ge::NodePtr g_node = graph->AddNode(g_op);
    //add descriptor
    vector<int64_t> dim({4, 33, 12, 16});
    GeShape shape(dim);

    GeTensorDesc tensor_desc(shape);
    tensor_desc.SetOriginFormat(FORMAT_NCHW);
    tensor_desc.SetFormat(FORMAT_NCHW);
    tensor_desc.SetDataType(DT_FLOAT);
    g_op->AddInputDesc("x", tensor_desc);
    g_op->AddOutputDesc("z", tensor_desc);
    ge::AttrUtils::SetInt(g_op, FE_IMPLY_TYPE, 6);//TBE

    
    Status ret = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g_node, "tbe-custom");
    ASSERT_EQ(ret, fe::SUCCESS);
    vector<int64_t> dim_result({4, 3, 12, 16,16});
    EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetInputDesc(0).GetFormat()), FORMAT_NC1HWC0);
    EXPECT_EQ(g_op->GetInputDesc(0).GetDataType(), DT_FLOAT);
    EXPECT_EQ(g_op->GetInputDesc(0).GetShape().GetDims(), dim_result);

    EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
    EXPECT_EQ(g_op->GetOutputDesc(0).GetDataType(), DT_FLOAT);
    EXPECT_EQ(g_op->GetOutputDesc(0).GetShape().GetDims(), dim_result);

}

/* Test SetDtypeAndFormatByPrecisionMode on op ConvTemp without predecessor node
 * The First input name is correct as ops kernel info store
 * The second and output is not correct. But we still consider they are qualified by the
 * structure of input and output. */
TEST_F(UTEST_fusion_engine_op_judge_new_unittest, set_input_format_succ_format_changed_Conv_without_input2)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g_op = std::make_shared<OpDesc>("ConvTemp", "ConvTemp");
  ge::NodePtr g_node = graph->AddNode(g_op);
  //add descriptor

  vector<int64_t> dim1({1, 16, 64, 64});
  GeShape shape1(dim1);
  GeTensorDesc tensor_desc1(shape1);
  tensor_desc1.SetOriginFormat(FORMAT_NCHW);
  tensor_desc1.SetFormat(FORMAT_NCHW);
  tensor_desc1.SetDataType(DT_FLOAT);
  g_op->AddInputDesc("x", tensor_desc1);

  vector<int64_t> dim2({64, 16, 7, 7});
  GeShape shape2(dim2);
  GeTensorDesc tensor_desc2(shape2);
  tensor_desc2.SetOriginFormat(FORMAT_NCHW);
  tensor_desc2.SetFormat(FORMAT_NCHW);
  tensor_desc2.SetDataType(DT_FLOAT);
  g_op->AddInputDesc("wqwe", tensor_desc2);

  vector<int64_t> dimo({1, 64, 30, 30});
  GeShape shapeo(dimo);
  GeTensorDesc tensor_desco(shapeo);
  tensor_desco.SetOriginFormat(FORMAT_NCHW);
  tensor_desco.SetFormat(FORMAT_NCHW);
  tensor_desco.SetDataType(DT_FLOAT);
  g_op->AddOutputDesc("yqwe", tensor_desco);

  ge::AttrUtils::SetInt(g_op, FE_IMPLY_TYPE, 6);//TBE
  vector<int64_t> dim1_5_d({1, 1, 64, 64, 16});
  vector<int64_t> dim2_fz({49, 4, 16, 16});
  vector<int64_t> dimo_5_d({1, 4, 30, 30, 16});
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

  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetInputDesc(1).GetFormat()), FORMAT_FRACTAL_Z);
  EXPECT_EQ(g_op->GetInputDesc(1).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(g_op->GetInputDesc(1).GetShape().GetDims(), dim2_fz);

  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetShape().GetDims(), dimo_5_d);

}

TEST_F(UTEST_fusion_engine_op_judge_new_unittest, set_input_format_succ_format_changed_01)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g_op = std::make_shared<OpDesc>("G1", "G");

  //add descriptor
  vector<int64_t> dim({4, 33, 12, 16});
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NHWC);
  tensor_desc.SetFormat(FORMAT_NHWC);
  tensor_desc.SetDataType(DT_FLOAT);
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

  vector<int64_t> dim_result({4, 1, 33, 12, 16});
  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetInputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetInputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(g_op->GetInputDesc(0).GetShape().GetDims(), dim_result);

  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetShape().GetDims(), dim_result);
}

TEST_F(UTEST_fusion_engine_op_judge_new_unittest, set_input_format_succ_format_changed_autofuse_01)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g_op = std::make_shared<OpDesc>("G1", "AscBackend");

  //add descriptor
  vector<int64_t> dim({4, 33, 12, 16});
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NHWC);
  tensor_desc.SetFormat(FORMAT_NHWC);
  tensor_desc.SetDataType(DT_FLOAT);
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

  vector<int64_t> dim_result({4, 33, 12, 16});
  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetInputDesc(0).GetFormat()), FORMAT_NHWC);
  EXPECT_EQ(g_op->GetInputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(g_op->GetInputDesc(0).GetShape().GetDims(), dim_result);
}

TEST_F(UTEST_fusion_engine_op_judge_new_unittest, set_input_format_succ_autofuse)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g_op = std::make_shared<OpDesc>("G1", "AscBackend");

  //add descriptor
  vector<int64_t> dim({4, 33, 12, 16});
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NHWC);
  tensor_desc.SetFormat(FORMAT_NHWC);
  tensor_desc.SetDataType(DT_FLOAT);
  g_op->AddInputDesc("x", tensor_desc);
  g_op->AddOutputDesc("z", tensor_desc);
  ge::NodePtr g_node = graph->AddNode(g_op);

  OpImplTypeJudgePtr op_impl_type_judge_ptr = std::make_shared<OpImplTypeJudge>(AI_CORE_NAME, fe_ops_kernel_info_store_ptr_);
  
  Status ret1 = op_impl_type_judge_ptr->JudgeInSubGraph(*(graph.get()));
  ASSERT_EQ(ret1, fe::SUCCESS);
}

/* Test SetDtypeAndFormatByPrecisionMode on op G without predecessor node
 * After OpFormatDtypeJudge, Op G1 format is FORMAT_NC1HWC0 and Dtype is Fp*/
TEST_F(UTEST_fusion_engine_op_judge_new_unittest, set_input_format_succ_format_and_dtype_changed)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g_op = std::make_shared<OpDesc>("G1", "G");
  ge::NodePtr g_node = graph->AddNode(g_op);
  //add descriptor
  vector<int64_t> dim({4, 33, 12, 16});
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NCHW);
  tensor_desc.SetFormat(FORMAT_NCHW);
  tensor_desc.SetDataType(DT_INT32);
  g_op->AddInputDesc("x", tensor_desc);
  g_op->AddOutputDesc("z", tensor_desc);
  ge::AttrUtils::SetInt(g_op, FE_IMPLY_TYPE, 6);

  
  Status ret = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g_node, "tbe-custom");
  ASSERT_EQ(ret, fe::SUCCESS);
  vector<int64_t> dim_result({4, 2, 12, 16, 32});
  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetInputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetInputDesc(0).GetDataType(), DT_UINT8);
  EXPECT_EQ(g_op->GetInputDesc(0).GetShape().GetDims(), dim_result);

  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetDataType(), DT_UINT8);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetShape().GetDims(), dim_result);
}

TEST_F(UTEST_fusion_engine_op_judge_new_unittest, set_input_format_failed_format_changed_01)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g_op = std::make_shared<OpDesc>("G1", "G");

  //add descriptor
  vector<int64_t> dim({4, 33, 12, 16});
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NHWC);
  tensor_desc.SetFormat(FORMAT_NHWC);
  tensor_desc.SetDataType(DT_FLOAT);
  g_op->AddInputDesc("x", tensor_desc);
  g_op->AddOutputDesc("z", tensor_desc);
  ge::NodePtr g_node = graph->AddNode(g_op);

  ge::AttrUtils::SetInt(g_op, FE_IMPLY_TYPE, 7);
  OpImplTypeJudgePtr op_impl_type_judge_ptr = std::make_shared<OpImplTypeJudge>(AI_CORE_NAME, fe_ops_kernel_info_store_ptr_);

  Status ret2 = op_format_dtype_judge_ptr_->SetFormat(*(graph.get()));
  ASSERT_EQ(ret2, fe::SUCCESS);

  vector<int64_t> dim_result({4, 33, 12, 16});
  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetInputDesc(0).GetFormat()), FORMAT_NHWC);
  EXPECT_EQ(g_op->GetInputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(g_op->GetInputDesc(0).GetShape().GetDims(), dim_result);

  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetOutputDesc(0).GetFormat()), FORMAT_NHWC);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetShape().GetDims(), dim_result);
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
TEST_F(UTEST_fusion_engine_op_judge_new_unittest, set_two_nodes_format_dtype_01)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g_op = std::make_shared<OpDesc>("G1", "G");
  OpDescPtr h_op = std::make_shared<OpDesc>("H1", "H");

  //add descriptor
  vector<int64_t> dim({4, 33, 12, 16});
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NCHW);
  tensor_desc.SetFormat(FORMAT_NCHW);
  tensor_desc.SetDataType(DT_FLOAT);
  g_op->AddInputDesc("x", tensor_desc);
  g_op->AddOutputDesc("z", tensor_desc);
  ge::AttrUtils::SetInt(g_op, FE_IMPLY_TYPE, 6);
  ge::NodePtr g_node = graph->AddNode(g_op);

  vector<int64_t> dim_h({1, 2, 3, 4});
  GeShape shape_h(dim_h);
  GeTensorDesc tensor_desc_h(shape_h);
  tensor_desc_h.SetOriginFormat(FORMAT_NHWC);
  tensor_desc_h.SetFormat(FORMAT_NHWC);
  tensor_desc_h.SetDataType(DT_FLOAT16);
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
  vector<int64_t> dim_result_nch_w_to_NC1_hw_c0({4, 3, 12, 16, 16});
  vector<int64_t> dim_result_nhw_c_to_NC1_hw_c0({1, 1, 2, 3, 16});
  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetInputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetInputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(g_op->GetInputDesc(0).GetShape().GetDims(), dim_result_nch_w_to_NC1_hw_c0);

  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetShape().GetDims(), dim_result_nch_w_to_NC1_hw_c0);

  EXPECT_EQ(h_op->GetInputDesc(0).GetFormat(), FORMAT_NHWC);
  EXPECT_EQ(h_op->GetInputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(h_op->GetInputDesc(0).GetShape().GetDims(), dim_h);

  EXPECT_EQ(h_op->GetOutputDesc(0).GetFormat(), FORMAT_NHWC);
  EXPECT_EQ(h_op->GetOutputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(h_op->GetOutputDesc(0).GetShape().GetDims(), dim_h);
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
TEST_F(UTEST_fusion_engine_op_judge_new_unittest, set_two_nodes_format_dtype_02)
{
  Configuration &config = Configuration::Instance(fe::AI_CORE_NAME);
  config.mix_list_parser_ = nullptr;
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g_op = std::make_shared<OpDesc>("G1", "G");
  OpDescPtr h_op = std::make_shared<OpDesc>("I1", "I");

  //add descriptor
  vector<int64_t> dim({4, 33, 12, 16});
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NCHW);
  tensor_desc.SetFormat(FORMAT_NCHW);
  tensor_desc.SetDataType(DT_FLOAT);
  g_op->AddInputDesc("x", tensor_desc);
  g_op->AddOutputDesc("z", tensor_desc);
  ge::AttrUtils::SetInt(g_op, FE_IMPLY_TYPE, 6);
  ge::NodePtr g_node = graph->AddNode(g_op);

  vector<int64_t> dim_i({1, 2, 3, 4});
  GeShape shape_i(dim_i);
  GeTensorDesc tensor_desc_i(shape_i);
  tensor_desc_i.SetOriginFormat(FORMAT_NHWC);
  tensor_desc_i.SetFormat(FORMAT_NHWC);
  tensor_desc_i.SetDataType(DT_FLOAT16);
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

  vector<int64_t> dim_result({4, 3, 12, 16, 16});
  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetInputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetInputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(g_op->GetInputDesc(0).GetShape().GetDims(), dim_result);

  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetShape().GetDims(), dim_result);

  vector<int64_t> dim_result_h({1, 4, 2, 3});
  vector<int64_t> dim_result_h5_d({1, 1, 2, 3, 32});
  EXPECT_EQ(h_op->GetInputDesc(0).GetFormat(), FORMAT_NHWC);
  EXPECT_EQ(h_op->GetInputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(h_op->GetInputDesc(0).GetShape().GetDims(), dim_i);

  EXPECT_EQ(h_op->GetOutputDesc(0).GetFormat(), FORMAT_NCHW);
  EXPECT_EQ(h_op->GetOutputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(h_op->GetOutputDesc(0).GetShape().GetDims(), dim_result_h);
}

/*  Below is a test case from interface JudgeOp(). The pre-condition and check spot
 * is the same as set_two_nodes_format_dtype_02 */
TEST_F(UTEST_fusion_engine_op_judge_new_unittest, set_two_nodes_format_dtype_03)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g_op = std::make_shared<OpDesc>("G1", "G");
  OpDescPtr h_op = std::make_shared<OpDesc>("I1", "I");

  //add descriptor
  vector<int64_t> dim({4, 33, 12, 16});
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NCHW);
  tensor_desc.SetFormat(FORMAT_NCHW);
  tensor_desc.SetDataType(DT_FLOAT);
  g_op->AddInputDesc("x", tensor_desc);
  g_op->AddOutputDesc("z", tensor_desc);
  //ge::AttrUtils::SetInt(g_op, FE_IMPLY_TYPE, 6);
  ge::NodePtr g_node = graph->AddNode(g_op);

  vector<int64_t> dim_i({1, 2, 3, 4});
  GeShape shape_i(dim_i);
  GeTensorDesc tensor_desc_i(shape_i);
  tensor_desc_i.SetOriginFormat(FORMAT_NHWC);
  tensor_desc_i.SetFormat(FORMAT_NHWC);
  tensor_desc_i.SetDataType(DT_FLOAT16);
  h_op->AddInputDesc("x", tensor_desc_i);
  h_op->AddOutputDesc("z", tensor_desc_i);
  //ge::AttrUtils::SetInt(h_op, FE_IMPLY_TYPE, 6);
  ge::NodePtr h_node = graph->AddNode(h_op);

  GraphUtils::AddEdge(g_node->GetOutDataAnchor(0), h_node->GetInDataAnchor(0));

  OpImplTypeJudgePtr op_impl_type_judge_ptr = std::make_shared<OpImplTypeJudge>(AI_CORE_NAME, fe_ops_kernel_info_store_ptr_);
  
  Status ret1 = op_impl_type_judge_ptr->MultiThreadJudge(*(graph.get()));
  ASSERT_EQ(ret1, fe::SUCCESS);
  Status ret2 = op_format_dtype_judge_ptr_->Judge(*(graph.get()));
  ret2 = op_format_dtype_judge_ptr_->SetFormat(*(graph.get()));
  ASSERT_EQ(ret2, fe::SUCCESS);

  vector<int64_t> dim_result({4, 3, 12, 16, 16});
  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetInputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetInputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(g_op->GetInputDesc(0).GetShape().GetDims(), dim_result);

  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetShape().GetDims(), dim_result);

  vector<int64_t> dim_result_h({1, 4, 2, 3});
  vector<int64_t> dim_result_h5_d({1, 1, 2, 3,32});
  EXPECT_EQ(h_op->GetInputDesc(0).GetFormat(), FORMAT_NHWC);
  EXPECT_EQ(h_op->GetInputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(h_op->GetInputDesc(0).GetShape().GetDims(), dim_i);

  EXPECT_EQ(h_op->GetOutputDesc(0).GetFormat(), FORMAT_NCHW);
  EXPECT_EQ(h_op->GetOutputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(h_op->GetOutputDesc(0).GetShape().GetDims(), dim_result_h);
}

TEST_F(UTEST_fusion_engine_op_judge_new_unittest, set_dtype_and_format_succ_dtype)
{
    Configuration &config = Configuration::Instance(fe::AI_CORE_NAME);
    config.is_init_ = false;

    map<string, string> options;
    string soc_version = "Ascend910A";
    PlatformUtils::Instance().soc_version_ = soc_version;
    config.Initialize(options);
    vector<FEOpsStoreInfo> &op_store_info_vector = Configuration::Instance(fe::AI_CORE_NAME).ops_store_info_vector_;
    for (auto it = op_store_info_vector.begin(); it != op_store_info_vector.end(); ) {
      if ((*it).op_impl_type == EN_IMPL_HW_GENERAL_CCE) {
        it = op_store_info_vector.erase(it);
      } else {
        it++;
      }
    }

    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    CreateTwoOpGraph(graph);

    FEOpsKernelInfoStorePtr store_ptr = std::make_shared<FEOpsKernelInfoStore>(fe::AI_CORE_NAME);

    OpKernelInfoPtr info_ptr_act = std::make_shared<OpKernelInfo>("Activation");
    OpKernelInfoPtr info_ptr_bn = std::make_shared<OpKernelInfo>("BatchNorm");

    InputOrOutputInfoPtr in_desc_ptr = std::make_shared<fe::InputOrOutputInfo>("x");
    in_desc_ptr->supported_formats_.emplace_back(FORMAT_NC1HWC0);
    in_desc_ptr->supported_formats_.emplace_back(FORMAT_NCHW);
    in_desc_ptr->supported_dtypes_.emplace_back(DT_FLOAT);
    in_desc_ptr->supported_dtypes_.emplace_back(DT_FLOAT16);
    info_ptr_act->input_infos_.emplace_back(in_desc_ptr);
    info_ptr_bn->input_infos_.emplace_back( in_desc_ptr);

    InputOrOutputInfoPtr out_desc_ptr = std::make_shared<fe::InputOrOutputInfo>("y");
    out_desc_ptr->supported_formats_.emplace_back(FORMAT_NC1HWC0);
    out_desc_ptr->supported_formats_.emplace_back(FORMAT_NCHW);
    out_desc_ptr->supported_dtypes_.emplace_back(DT_FLOAT);
    out_desc_ptr->supported_dtypes_.emplace_back(DT_FLOAT16);
    info_ptr_act->output_infos_.emplace_back(out_desc_ptr);
    info_ptr_bn->output_infos_.emplace_back(out_desc_ptr);

    FEOpsStoreInfo ops_store_info;
    SubOpInfoStorePtr sub_ops_kernel_ptr = std::make_shared<SubOpInfoStore>(ops_store_info);
    sub_ops_kernel_ptr->op_kernel_info_map_.emplace(std::make_pair("Activation", info_ptr_act));
    sub_ops_kernel_ptr->op_kernel_info_map_.emplace(std::make_pair("BatchNorm", info_ptr_bn));
    OpsKernelManager::Instance(AI_CORE_NAME).sub_ops_kernel_map_.emplace(std::make_pair("aicore-tbe-builtin", sub_ops_kernel_ptr));

    FEOpsKernelInfoStorePtr ops_kernel_info_store_ptr = std::make_shared<FEOpsKernelInfoStore>(fe::AI_CORE_NAME);
    OpFormatDtypeJudgePtr op_format_dtype_judge_ptr = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr_);
    Status ret = op_format_dtype_judge_ptr->Judge(*(graph.get()));
  ret = op_format_dtype_judge_ptr->SetFormat(*(graph.get()));

    for (auto node : graph->GetDirectNode()) {
      if (node->GetType() == "Activation") {
        EXPECT_EQ(DT_FLOAT16, node->GetOpDesc()->GetInputDescPtr(0)->GetDataType());
        EXPECT_EQ(DT_FLOAT16, node->GetOpDesc()->GetOutputDescPtr(0)->GetDataType());
        EXPECT_EQ(FORMAT_NCHW, node->GetOpDesc()->GetInputDescPtr(0)->GetFormat());
        EXPECT_EQ(FORMAT_NCHW, node->GetOpDesc()->GetOutputDescPtr(0)->GetFormat());

      } else if (node->GetType() == "BatchNorm") {
        EXPECT_EQ(DT_FLOAT16, node->GetOpDesc()->GetInputDescPtr(0)->GetDataType());
        EXPECT_EQ(DT_FLOAT16, node->GetOpDesc()->GetOutputDescPtr(0)->GetDataType());
        EXPECT_EQ(FORMAT_NC1HWC0, node->GetOpDesc()->GetInputDescPtr(0)->GetFormat());
        EXPECT_EQ(FORMAT_NC1HWC0, node->GetOpDesc()->GetOutputDescPtr(0)->GetFormat());
      }
    }
}

TEST_F(UTEST_fusion_engine_op_judge_new_unittest, set_dtype_and_format_fail)
{
    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    CreateTwoInvalidOpGraph(graph);

    FEOpsKernelInfoStorePtr store_ptr = std::make_shared<FEOpsKernelInfoStore>(fe::AI_CORE_NAME);

    OpKernelInfoPtr info_ptr_act = std::make_shared<OpKernelInfo>("Activation");
    OpKernelInfoPtr info_ptr_bn = std::make_shared<OpKernelInfo>("BatchNorm");

    InputOrOutputInfoPtr in_desc_ptr = std::make_shared<fe::InputOrOutputInfo>("x");
    in_desc_ptr->supported_formats_.emplace_back(FORMAT_NC1HWC0);
    in_desc_ptr->supported_formats_.emplace_back(FORMAT_NCHW);
    in_desc_ptr->supported_dtypes_.emplace_back(DT_FLOAT);
    info_ptr_act->input_infos_.emplace_back(in_desc_ptr);
    info_ptr_bn->input_infos_.emplace_back(in_desc_ptr);

    InputOrOutputInfoPtr out_desc_ptr = std::make_shared<fe::InputOrOutputInfo>("y");
    out_desc_ptr->supported_formats_.emplace_back(FORMAT_NC1HWC0);
    out_desc_ptr->supported_formats_.emplace_back(FORMAT_NCHW);
    out_desc_ptr->supported_dtypes_.emplace_back(DT_FLOAT);
    info_ptr_act->output_infos_.emplace_back(out_desc_ptr);
    info_ptr_bn->output_infos_.emplace_back(out_desc_ptr);

    FEOpsStoreInfo ops_store_info;
    SubOpInfoStorePtr sub_ops_kernel_ptr = std::make_shared<SubOpInfoStore>(ops_store_info);
    sub_ops_kernel_ptr->op_kernel_info_map_.emplace(std::make_pair("Activation", info_ptr_act));
    sub_ops_kernel_ptr->op_kernel_info_map_.emplace(std::make_pair("BatchNorm", info_ptr_bn));
    OpsKernelManager::Instance(AI_CORE_NAME).sub_ops_kernel_map_.emplace(std::make_pair("tbe-builtin", sub_ops_kernel_ptr));

    FEOpsKernelInfoStorePtr ops_kernel_info_store_ptr = std::make_shared<FEOpsKernelInfoStore>(fe::AI_CORE_NAME);
    OpFormatDtypeJudgePtr op_format_dtype_judge_ptr = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr_);
    Status ret = op_format_dtype_judge_ptr->Judge(*(graph.get()));
  ret = op_format_dtype_judge_ptr->SetFormat(*(graph.get()));

    EXPECT_EQ(fe::SUCCESS, ret);

}

TEST_F(UTEST_fusion_engine_op_judge_new_unittest, FindSuitableDtypeVec)
{
  FEOpsKernelInfoStorePtr ops_kernel_info_store_ptr = std::make_shared<FEOpsKernelInfoStore>(fe::AI_CORE_NAME);
  OpDtypeRiseMatcherPtr op_dtype_rise_matcher = std::make_shared<OpDtypeRiseMatcher>();
  ge::DataType dtype_to_be_found = ge::DT_INT64;
  vector<ge::DataType> input_dtype_vec_ops_kernel = {DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT64, DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT64, DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT64, DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT64};
  vector<uint32_t> matched_index_vec = {0, 1, 2,3,4,5,6,7,8,9,10,11,12,13,14,15};
  op_dtype_rise_matcher->FindSuitableDtype(input_dtype_vec_ops_kernel, dtype_to_be_found, matched_index_vec, ge::DT_MAX);
  EXPECT_EQ(matched_index_vec.size(), 4);
}

TEST_F(UTEST_fusion_engine_op_judge_new_unittest, FindSuitableDtypeVec2)
{
  FEOpsKernelInfoStorePtr ops_kernel_info_store_ptr = std::make_shared<FEOpsKernelInfoStore>(fe::AI_CORE_NAME);
  OpDtypeRiseMatcherPtr op_dtype_rise_matcher = std::make_shared<OpDtypeRiseMatcher>();
  ge::DataType dtype_to_be_found = ge::DT_INT64;
  vector<ge::DataType> input_dtype_vec_ops_kernel = {DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT64, DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT64, DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT64, DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT64};
  vector<uint32_t> matched_index_vec = {0, 1, 2,3};
  op_dtype_rise_matcher->FindSuitableDtype(input_dtype_vec_ops_kernel, dtype_to_be_found, matched_index_vec, ge::DT_MAX);
  EXPECT_EQ(matched_index_vec.size(), 1);

}

TEST_F(UTEST_fusion_engine_op_judge_new_unittest, FindSuitableFormatVec)
{
  FEOpsKernelInfoStorePtr ops_kernel_info_store_ptr = std::make_shared<FEOpsKernelInfoStore>(fe::AI_CORE_NAME);
  OpFormatMatcherPtr op_format_matcher = std::make_shared<OpFormatMatcher>();
  ge::Format format_to_be_found = ge::FORMAT_NHWC;
  vector<ge::Format> foramt_vec = {ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC,
  ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
  ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0,
  ge::FORMAT_NCHW,ge::FORMAT_NCHW,ge::FORMAT_NCHW,ge::FORMAT_NCHW};
  vector<uint32_t> matched_index_vec = {0, 1, 2,3};
  op_format_matcher->FindSuitableFormat(foramt_vec, format_to_be_found, FORMAT_NCHW, matched_index_vec);
  EXPECT_EQ(matched_index_vec.size(), 4);
}

TEST_F(UTEST_fusion_engine_op_judge_new_unittest, set_op_shape_dim_01)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr op = std::make_shared<OpDesc>("relu", "GWithoutReshapeType");

  //add descriptor
  vector<int64_t> dim1({7});
  GeShape shape1(dim1);
  GeTensorDesc input(shape1);
  input.SetOriginFormat(FORMAT_NHWC);
  input.SetFormat(FORMAT_NHWC);
  input.SetDataType(DT_FLOAT);

  vector<int64_t> dim2({2,3});
  GeShape shape2(dim2);
  GeTensorDesc output(shape2);
  output.SetOriginFormat(FORMAT_NHWC);
  output.SetFormat(FORMAT_NHWC);
  output.SetDataType(DT_FLOAT);

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

  vector<int64_t> dim_result1({1,1,1,1,16});
  EXPECT_EQ(op->GetInputDesc(0).GetShape().GetDims(), dim_result1);

  vector<int64_t> dim_result2({1,1,2,3,16});
  EXPECT_EQ(op->GetOutputDesc(0).GetShape().GetDims(), dim_result2);
}


TEST_F(UTEST_fusion_engine_op_judge_new_unittest, set_op_shape_dim_02)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr op = std::make_shared<OpDesc>("mat1", "MatMul");

  //add descriptor
  vector<int64_t> dim1({7});
  GeShape shape1(dim1);
  GeTensorDesc input(shape1);
  input.SetOriginFormat(FORMAT_NHWC);
  input.SetFormat(FORMAT_NHWC);
  input.SetDataType(DT_FLOAT);

  vector<int64_t> dim2({2,3});
  GeShape shape2(dim2);
  GeTensorDesc output(shape2);
  output.SetOriginFormat(FORMAT_NHWC);
  output.SetFormat(FORMAT_NHWC);
  output.SetDataType(DT_FLOAT);

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

  vector<int64_t> dim_result1({1,1,1,1,16});
  EXPECT_EQ(op->GetInputDesc(0).GetShape().GetDims(), dim_result1);

  vector<int64_t> dim_result2({1,1,2,3,16});
  EXPECT_EQ(op->GetOutputDesc(0).GetShape().GetDims(), dim_result2);
}

TEST_F(UTEST_fusion_engine_op_judge_new_unittest, set_op_shape_dim_03)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr op = std::make_shared<OpDesc>("relu", "GWithoutReshapeType");

  //add descriptor
  vector<int64_t> dim1({4,5,6});
  GeShape shape1(dim1);
  GeTensorDesc input(shape1);
  input.SetOriginFormat(FORMAT_NHWC);
  input.SetFormat(FORMAT_NHWC);
  input.SetDataType(DT_FLOAT);

  vector<int64_t> dim2({5,6,7,8});
  GeShape shape2(dim2);
  GeTensorDesc output(shape2);
  output.SetOriginFormat(FORMAT_NHWC);
  output.SetFormat(FORMAT_NHWC);
  output.SetDataType(DT_FLOAT);

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

  vector<int64_t> dim_result1({1,1,4,5,16});
  EXPECT_EQ(op->GetInputDesc(0).GetShape().GetDims(), dim_result1);

  vector<int64_t> dim_result2({5,1,6,7,16});
  EXPECT_EQ(op->GetOutputDesc(0).GetShape().GetDims(), dim_result2);
}


TEST_F(UTEST_fusion_engine_op_judge_new_unittest, set_op_shape_dim_04)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr op = std::make_shared<OpDesc>("relu", "MatMul");

  //add descriptor
  vector<int64_t> dim1({5,6});
  GeShape shape1(dim1);
  GeTensorDesc input(shape1);
  input.SetOriginFormat(FORMAT_NHWC);
  input.SetFormat(FORMAT_NHWC);
  input.SetDataType(DT_FLOAT);

  vector<int64_t> dim2({5,6,7,8});
  GeShape shape2(dim2);
  GeTensorDesc output(shape2);
  output.SetOriginFormat(FORMAT_NHWC);
  output.SetFormat(FORMAT_NHWC);
  output.SetDataType(DT_FLOAT);

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

  vector<int64_t> dim_result1({1,1,5,6,16});
  EXPECT_EQ(op->GetInputDesc(0).GetShape().GetDims(), dim_result1);

  vector<int64_t> dim_result2({5,1,6,7,16});
  EXPECT_EQ(op->GetOutputDesc(0).GetShape().GetDims(), dim_result2);
}

TEST_F(UTEST_fusion_engine_op_judge_new_unittest, set_op_shape_dim_05)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr op = std::make_shared<OpDesc>("relu", "GWithoutReshapeType");

  //add descriptor
  vector<int64_t> dim1({});
  GeShape shape1(dim1);
  GeTensorDesc input(shape1);
  input.SetOriginFormat(FORMAT_NHWC);
  input.SetFormat(FORMAT_NHWC);
  input.SetDataType(DT_FLOAT);

  vector<int64_t> dim2;
  GeShape shape2(dim2);
  GeTensorDesc output(shape2);
  input.SetOriginFormat(FORMAT_NHWC);
  input.SetFormat(FORMAT_NHWC);
  input.SetDataType(DT_FLOAT);

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

  vector<int64_t> dim_result1({});
  EXPECT_EQ(op->GetInputDesc(0).GetShape().GetDims(), dim_result1);

  vector<int64_t> dim_result2({});
  EXPECT_EQ(op->GetOutputDesc(0).GetShape().GetDims(), dim_result2);
}


TEST_F(UTEST_fusion_engine_op_judge_new_unittest, set_op_shape_dim_06)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr op = std::make_shared<OpDesc>("relu", "G");

  //add descriptor
  vector<int64_t> dim1({1,2,3,4});
  GeShape shape1(dim1);
  GeTensorDesc input(shape1);
  input.SetOriginFormat(FORMAT_NHWC);
  input.SetFormat(FORMAT_NHWC);
  input.SetDataType(DT_FLOAT);

  vector<int64_t> dim2;
  GeShape shape2(dim2);
  GeTensorDesc output(shape2);
  output.SetOriginFormat(FORMAT_NHWC);
  output.SetFormat(FORMAT_NHWC);
  output.SetDataType(DT_FLOAT);

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

  vector<int64_t> dim_result1({1,1,2,3,16});
  EXPECT_EQ(op->GetInputDesc(0).GetShape().GetDims(), dim_result1);

  vector<int64_t> dim_result2({});
  EXPECT_EQ(op->GetOutputDesc(0).GetShape().GetDims(), dim_result2);
}

TEST_F(UTEST_fusion_engine_op_judge_new_unittest, set_op_shape_dim_by_reshape_type_01)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr op = std::make_shared<OpDesc>("relu", "G");

  //add descriptor
  vector<int64_t> dim1({7});
  GeShape shape1(dim1);
  GeTensorDesc input(shape1);
  input.SetOriginFormat(FORMAT_NHWC);
  input.SetFormat(FORMAT_NHWC);
  input.SetDataType(DT_FLOAT);

  vector<int64_t> dim2({2,3});
  GeShape shape2(dim2);
  GeTensorDesc output(shape2);
  output.SetOriginFormat(FORMAT_NHWC);
  output.SetFormat(FORMAT_NHWC);
  output.SetDataType(DT_FLOAT);

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
  vector<int64_t> dim_result1({7,1,1,1,16});
  EXPECT_EQ(op->GetInputDesc(0).GetShape().GetDims(), dim_result1);

  vector<int64_t> dim_result2({1,1,2,3,16});
  EXPECT_EQ(op->GetOutputDesc(0).GetShape().GetDims(), dim_result2);
}


TEST_F(UTEST_fusion_engine_op_judge_new_unittest, set_op_shape_dim_by_reshape_type_02)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr op = std::make_shared<OpDesc>("mat1", "MatMulWithReshapeType");

  //add descriptor
  vector<int64_t> dim1({7});
  GeShape shape1(dim1);
  GeTensorDesc input(shape1);
  input.SetOriginFormat(FORMAT_NHWC);
  input.SetFormat(FORMAT_NHWC);
  input.SetDataType(DT_FLOAT);

  vector<int64_t> dim2({2,3});
  GeShape shape2(dim2);
  GeTensorDesc output(shape2);
  output.SetOriginFormat(FORMAT_NHWC);
  output.SetFormat(FORMAT_NHWC);
  output.SetDataType(DT_FLOAT);

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
  vector<int64_t> dim_result1({1,1,1,1,16});
  EXPECT_EQ(op->GetInputDesc(0).GetShape().GetDims(), dim_result1);

  vector<int64_t> dim_result2({1, 1, 2, 3, 16});
  EXPECT_EQ(op->GetOutputDesc(0).GetShape().GetDims(), dim_result2);
}

TEST_F(UTEST_fusion_engine_op_judge_new_unittest, set_op_shape_dim_by_reshape_type_03)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr op = std::make_shared<OpDesc>("relu", "G");

  //add descriptor
  vector<int64_t> dim1({4,5,6});
  GeShape shape1(dim1);
  GeTensorDesc input(shape1);
  input.SetOriginFormat(FORMAT_NHWC);
  input.SetFormat(FORMAT_NHWC);
  input.SetDataType(DT_FLOAT);

  vector<int64_t> dim2({5,6,7,8});
  GeShape shape2(dim2);
  GeTensorDesc output(shape2);
  output.SetOriginFormat(FORMAT_NHWC);
  output.SetFormat(FORMAT_NHWC);
  output.SetDataType(DT_FLOAT);

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
  vector<int64_t> dim_result1({4,5,6});
  EXPECT_EQ(op->GetInputDesc(0).GetShape().GetDims(), dim_result1);

  vector<int64_t> dim_result2({5,1,6,7,16});
  EXPECT_EQ(op->GetOutputDesc(0).GetShape().GetDims(), dim_result2);
}


TEST_F(UTEST_fusion_engine_op_judge_new_unittest, set_op_shape_dim_by_reshape_type_04)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr op = std::make_shared<OpDesc>("relu", "MatMulWithReshapeType");

  //add descriptor
  vector<int64_t> dim1({5,6});
  GeShape shape1(dim1);
  GeTensorDesc input(shape1);
  input.SetOriginFormat(FORMAT_NHWC);
  input.SetFormat(FORMAT_NHWC);
  input.SetDataType(DT_FLOAT);

  vector<int64_t> dim2({3,4});
  GeShape shape2(dim2);
  GeTensorDesc output(shape2);
  output.SetOriginFormat(FORMAT_NHWC);
  output.SetFormat(FORMAT_NHWC);
  output.SetDataType(DT_FLOAT);

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
  vector<int64_t> dim_result1({1, 1, 5, 6, 16});
  EXPECT_EQ(op->GetInputDesc(0).GetShape().GetDims(), dim_result1);

  vector<int64_t> dim_result2({1, 1, 3, 4, 16});
  EXPECT_EQ(op->GetOutputDesc(0).GetShape().GetDims(), dim_result2);
}


TEST_F(UTEST_fusion_engine_op_judge_new_unittest, set_op_shape_dim_by_reshape_type_05)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr op = std::make_shared<OpDesc>("relu", "G");

  //add descriptor
  vector<int64_t> dim1({});
  GeShape shape1(dim1);
  GeTensorDesc input(shape1);
  input.SetOriginFormat(FORMAT_NHWC);
  input.SetFormat(FORMAT_NHWC);
  input.SetDataType(DT_FLOAT);

  vector<int64_t> dim2;
  GeShape shape2(dim2);
  GeTensorDesc output(shape2);
  input.SetOriginFormat(FORMAT_NHWC);
  input.SetFormat(FORMAT_NHWC);
  input.SetDataType(DT_FLOAT);

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
  vector<int64_t> dim_result1({});
  EXPECT_EQ(op->GetInputDesc(0).GetShape().GetDims(), dim_result1);

  vector<int64_t> dim_result2({});
  EXPECT_EQ(op->GetOutputDesc(0).GetShape().GetDims(), dim_result2);
}

TEST_F(UTEST_fusion_engine_op_judge_new_unittest, set_op_shape_dim_by_reshape_type_06)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr op = std::make_shared<OpDesc>("relu", "G");

  //add descriptor
  vector<int64_t> dim1({7,8});
  GeShape shape1(dim1);
  GeTensorDesc input(shape1);
  input.SetOriginFormat(FORMAT_NHWC);
  input.SetFormat(FORMAT_NHWC);
  input.SetDataType(DT_FLOAT);

  vector<int64_t> dim2({2,3});
  GeShape shape2(dim2);
  GeTensorDesc output(shape2);
  output.SetOriginFormat(FORMAT_NHWC);
  output.SetFormat(FORMAT_NHWC);
  output.SetDataType(DT_FLOAT);

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
  vector<int64_t> dim_result1({7,1,1,1,16});
  EXPECT_EQ(op->GetInputDesc(0).GetShape().GetDims(), dim_result1);

  vector<int64_t> dim_result2({1,1,2,3,16});
  EXPECT_EQ(op->GetOutputDesc(0).GetShape().GetDims(), dim_result2);
}

TEST_F(UTEST_fusion_engine_op_judge_new_unittest, set_op_shape_dim_by_reshape_type_07)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr op = std::make_shared<OpDesc>("relu", "MatMulWithReshapeType");

  //add descriptor
  vector<int64_t> dim1({5,6});
  GeShape shape1(dim1);
  GeTensorDesc input(shape1);
  input.SetOriginFormat(FORMAT_NHWC);
  input.SetFormat(FORMAT_NHWC);
  input.SetDataType(DT_FLOAT);

  vector<int64_t> dim2({3,4});
  GeShape shape2(dim2);
  GeTensorDesc output(shape2);
  output.SetOriginFormat(FORMAT_NHWC);
  output.SetFormat(FORMAT_NHWC);
  output.SetDataType(DT_FLOAT);

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
  vector<int64_t> dim_result1({1, 1, 5, 6, 16});
  EXPECT_EQ(op->GetInputDesc(0).GetShape().GetDims(), dim_result1);

  vector<int64_t> dim_result2({1, 1, 3, 4, 16});
  EXPECT_EQ(op->GetOutputDesc(0).GetShape().GetDims(), dim_result2);
}



TEST_F(UTEST_fusion_engine_op_judge_new_unittest, set_op_shape_dim_by_reshape_type_08)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr op = std::make_shared<OpDesc>("relu", "G");

  //add descriptor
  vector<int64_t> dim1({7,8});
  GeShape shape1(dim1);
  GeTensorDesc input(shape1);
  input.SetOriginFormat(FORMAT_NHWC);
  input.SetFormat(FORMAT_NHWC);
  input.SetDataType(DT_FLOAT);

  vector<int64_t> dim2({2,3});
  GeShape shape2(dim2);
  GeTensorDesc output(shape2);
  output.SetOriginFormat(FORMAT_NHWC);
  output.SetFormat(FORMAT_NHWC);
  output.SetDataType(DT_FLOAT);

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
  vector<int64_t> dim_result1({7,1,1,1,16});
  EXPECT_EQ(op->GetInputDesc(0).GetShape().GetDims(), dim_result1);

  vector<int64_t> dim_result2({1,1,2,3,16});
  EXPECT_EQ(op->GetOutputDesc(0).GetShape().GetDims(), dim_result2);
}

/*  Test Op Convolution with  three input*/
TEST_F(UTEST_fusion_engine_op_judge_new_unittest, set_convolution_format_dtype_01)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr p1_op = std::make_shared<OpDesc>("PlaceHolder1", "PlaceHolder");
  OpDescPtr p2_op = std::make_shared<OpDesc>("PlaceHolder2", "PlaceHolder");
  OpDescPtr p3_op = std::make_shared<OpDesc>("PlaceHolder3", "PlaceHolder");
  OpDescPtr c1_op = std::make_shared<OpDesc>("ConvTemp1", "ConvTemp");

  //add descriptor
  vector<int64_t> dim_p1({3, 7, 17, 17});
  vector<int64_t> exp_dim_p1({3, 1, 17, 17, 16});
  vector<int64_t> dim_p2({6,7,2,2});
  vector<int64_t> exp_dim_p2{ 4, 1, 16, 16 };
  vector<int64_t> dim_p3({});
  vector<int64_t> dim_c1_out({3,6,5,5});
  vector<int64_t> exp_dim_c1_out{ 3, 1, 5, 5, 16 };
  vector<int64_t> dim_p4({1,1,1,1});

  GeShape shape_p1(dim_p1);
  GeShape shape_p2(dim_p2);
  GeShape shape_p3(dim_p3);
  GeShape shape_c1_out(dim_c1_out);

  GeTensorDesc tensor_desc_p1(shape_p1);
  GeTensorDesc tensor_desc_p2(shape_p2);
  GeTensorDesc tensor_desc_p3(shape_p3);
  tensor_desc_p3.SetOriginFormat(FORMAT_NCHW);
  tensor_desc_p3.SetFormat(FORMAT_NCHW);

  GeTensorDesc tensor_desc_c1_out(shape_c1_out);

  tensor_desc_p1.SetOriginFormat(FORMAT_NCHW);
  tensor_desc_p1.SetFormat(FORMAT_NCHW);
  tensor_desc_p1.SetDataType(DT_FLOAT);
  tensor_desc_p2.SetOriginFormat(FORMAT_NCHW);
  tensor_desc_p2.SetFormat(FORMAT_NCHW);
  tensor_desc_p2.SetDataType(DT_FLOAT16);

  tensor_desc_c1_out.SetOriginFormat(FORMAT_NCHW);
  tensor_desc_c1_out.SetFormat(FORMAT_NCHW);
  tensor_desc_c1_out.SetDataType(DT_FLOAT);

  p1_op->AddOutputDesc("y", tensor_desc_p1);
  p2_op->AddOutputDesc("y", tensor_desc_p2);

  //ge::AttrUtils::SetInt(g_op, FE_IMPLY_TYPE, 6);
  ge::NodePtr p1_node = graph->AddNode(p1_op);
  ge::NodePtr p2_node = graph->AddNode(p2_op);

  c1_op->AddInputDesc("f", tensor_desc_p1);
  c1_op->AddInputDesc("w", tensor_desc_p2);
  c1_op->AddInputDesc("c", tensor_desc_p3);
  c1_op->AddOutputDesc("d", tensor_desc_c1_out);

  ge::NodePtr c1_node = graph->AddNode(c1_op);

  GraphUtils::AddEdge(p1_node->GetOutDataAnchor(0), c1_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(p2_node->GetOutDataAnchor(0), c1_node->GetInDataAnchor(1));
  //GraphUtils::AddEdge(p3_node->GetOutDataAnchor(0), C1Node->GetInDataAnchor(2));

  Status ret = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(c1_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(c1_node, "tbe-custom");
  ASSERT_EQ(ret, fe::SUCCESS);

  EXPECT_EQ(ge::GetPrimaryFormat(c1_op->GetInputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(c1_op->GetInputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(c1_op->GetInputDesc(0).GetShape().GetDims(), exp_dim_p1);

  EXPECT_EQ(ge::GetPrimaryFormat(c1_op->GetInputDesc(1).GetFormat()), FORMAT_FRACTAL_Z);
  EXPECT_EQ(c1_op->GetInputDesc(1).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(c1_op->GetInputDesc(1).GetShape().GetDims(), exp_dim_p2);

  EXPECT_EQ(c1_op->GetInputDesc(2).GetFormat(), FORMAT_NCHW);
  EXPECT_EQ(c1_op->GetInputDesc(2).GetDataType(), DT_FLOAT);
  EXPECT_EQ(c1_op->GetInputDesc(2).GetShape().GetDims(), dim_p3);

  EXPECT_EQ(ge::GetPrimaryFormat(c1_op->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(c1_op->GetOutputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(c1_op->GetOutputDesc(0).GetShape().GetDims(), exp_dim_c1_out);
}

/*  Test Op Convolution with  three input*/
TEST_F(UTEST_fusion_engine_op_judge_new_unittest, set_convolution_format_dtype_02)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr p1_op = std::make_shared<OpDesc>("PlaceHolder1", "PlaceHolder");
  OpDescPtr p2_op = std::make_shared<OpDesc>("PlaceHolder2", "Op1");
  OpDescPtr p3_op = std::make_shared<OpDesc>("PlaceHolder3", "Op2");
  OpDescPtr c1_op = std::make_shared<OpDesc>("ConvTemp1", "ConvTempSeq");

  //add descriptor
  vector<int64_t> dim_p1({3, 7, 17, 17});
  vector<int64_t> dim5hd_p1_cce({17, 17, 3, 7});
  vector<int64_t> exp_dim5hd_p1_cce({17, 2, 3, 7, 16});
  vector<int64_t> dim_p2({6,7,2,2});
  vector<int64_t> dim_p3({});
  vector<int64_t> dim_c1_out({3,6,5,5});
  vector<int64_t> exp_dim_c1_out({3,1,5,5,16});
  vector<int64_t> dim_p4({1,1,1,1});

  GeShape shape_p1(dim_p1);
  GeShape shape_p2(dim_p2);
  GeShape shape_p3(dim_p3);
  GeShape shape_c1_out(dim_c1_out);

  GeTensorDesc tensor_desc_p1(shape_p1);
  GeTensorDesc tensor_desc_p2(shape_p2);
  GeTensorDesc tensor_desc_p3(shape_p3);
  GeTensorDesc tensor_desc_c1_out(shape_c1_out);

  tensor_desc_p1.SetOriginFormat(FORMAT_HWCN);
  tensor_desc_p1.SetFormat(FORMAT_HWCN);
  tensor_desc_p1.SetDataType(DT_FLOAT16);

  tensor_desc_p2.SetOriginFormat(FORMAT_FRACTAL_Z);
  tensor_desc_p2.SetFormat(FORMAT_FRACTAL_Z);
  tensor_desc_p2.SetDataType(DT_INT8);
  tensor_desc_p2.SetOriginDataType(DT_INT8);
  tensor_desc_p2.SetOriginShape(shape_p2);

  tensor_desc_p3.SetOriginFormat(FORMAT_NCHW);
  tensor_desc_p3.SetFormat(FORMAT_NCHW);
  tensor_desc_p3.SetDataType(DT_FLOAT16);

  tensor_desc_c1_out.SetOriginFormat(FORMAT_NCHW);
  tensor_desc_c1_out.SetFormat(FORMAT_NCHW);
  tensor_desc_c1_out.SetDataType(DT_INT8);

  p1_op->AddOutputDesc("y", tensor_desc_p1);
  p2_op->AddOutputDesc("y", tensor_desc_p2);
  p2_op->AddOutputDesc("y", tensor_desc_p3);

  //ge::AttrUtils::SetInt(g_op, FE_IMPLY_TYPE, 6);
  ge::NodePtr p1_node = graph->AddNode(p1_op);
  ge::NodePtr p2_node = graph->AddNode(p2_op);
  ge::NodePtr p3_node = graph->AddNode(p3_op);

  c1_op->AddInputDesc("f", tensor_desc_p1);
  c1_op->AddInputDesc("w", tensor_desc_p2);
  c1_op->AddInputDesc("c", tensor_desc_p3);
  c1_op->AddOutputDesc("d", tensor_desc_c1_out);

  ge::NodePtr c1_node = graph->AddNode(c1_op);

  GraphUtils::AddEdge(p1_node->GetOutDataAnchor(0), c1_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(p2_node->GetOutDataAnchor(0), c1_node->GetInDataAnchor(1));
  GraphUtils::AddEdge(p3_node->GetOutDataAnchor(0), c1_node->GetInDataAnchor(3));

  
  Status ret = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(c1_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(c1_node, "tbe-custom");
  ASSERT_EQ(ret, fe::SUCCESS);

  EXPECT_EQ(ge::GetPrimaryFormat(c1_op->GetInputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(c1_op->GetInputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(c1_op->GetInputDesc(0).GetShape().GetDims(), exp_dim5hd_p1_cce);

  EXPECT_EQ(ge::GetPrimaryFormat(c1_op->GetInputDesc(1).GetFormat()), FORMAT_FRACTAL_Z);
  EXPECT_EQ(c1_op->GetInputDesc(1).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(c1_op->GetInputDesc(1).GetShape().GetDims(), dim_p2);

  EXPECT_EQ(c1_op->GetInputDesc(2).GetFormat(), FORMAT_NCHW);
  EXPECT_EQ(c1_op->GetInputDesc(2).GetDataType(), DT_FLOAT);
  EXPECT_EQ(c1_op->GetInputDesc(2).GetShape().GetDims(), dim_p3);

  EXPECT_EQ(ge::GetPrimaryFormat(c1_op->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(c1_op->GetOutputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(c1_op->GetOutputDesc(0).GetShape().GetDims(), exp_dim_c1_out);
}

/*  Test Op Convolution with  three input*/
TEST_F(UTEST_fusion_engine_op_judge_new_unittest, set_convolution_format_dtype_03)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr p1_op = std::make_shared<OpDesc>("PlaceHolder1", "Const");
  OpDescPtr p2_op = std::make_shared<OpDesc>("PlaceHolder2", "PlaceHolder");
  OpDescPtr p3_op = std::make_shared<OpDesc>("PlaceHolder3", "Variable");
  OpDescPtr c1_op = std::make_shared<OpDesc>("ConvTemp1", "ConvTempSeq");

  //add descriptor
  vector<int64_t> dim_p1({3, 7, 17, 17});
  vector<int64_t> dim_p2({6,7,2,2});
  vector<int64_t> dim_p3({});
  vector<int64_t> dim_c1_out({3,6,5,5});
  vector<int64_t> exp_dim_c1_out({3,1,5,5,16});
  vector<int64_t> dim_p4({1,1,1,1});

  GeShape shape_p1(dim_p1);
  GeShape shape_p2(dim_p2);
  GeShape shape_p3(dim_p3);
  GeShape shape_c1_out(dim_c1_out);

  GeTensorDesc tensor_desc_p1(shape_p1);
  GeTensorDesc tensor_desc_p2(shape_p2);
  GeTensorDesc tensor_desc_p3(shape_p3);
  GeTensorDesc tensor_desc_c1_out(shape_c1_out);

  tensor_desc_p1.SetOriginFormat(FORMAT_HWCN);
  tensor_desc_p1.SetFormat(FORMAT_HWCN);
  tensor_desc_p1.SetDataType(DT_FLOAT16);

  tensor_desc_p2.SetOriginFormat(FORMAT_FRACTAL_Z);
  tensor_desc_p2.SetFormat(FORMAT_FRACTAL_Z);
  tensor_desc_p2.SetDataType(DT_FLOAT16);
  tensor_desc_p3.SetOriginFormat(FORMAT_NCHW);
  tensor_desc_p3.SetFormat(FORMAT_NCHW);
  tensor_desc_p3.SetDataType(DT_FLOAT16);

  tensor_desc_c1_out.SetOriginFormat(FORMAT_NCHW);
  tensor_desc_c1_out.SetFormat(FORMAT_NCHW);
  tensor_desc_c1_out.SetDataType(DT_FLOAT16);

  p1_op->AddOutputDesc("y", tensor_desc_p1);
  p2_op->AddOutputDesc("y", tensor_desc_p2);
  p2_op->AddOutputDesc("y", tensor_desc_p3);

  //ge::AttrUtils::SetInt(g_op, FE_IMPLY_TYPE, 6);
  ge::NodePtr p1_node = graph->AddNode(p1_op);
  ge::NodePtr p2_node = graph->AddNode(p2_op);
  ge::NodePtr p3_node = graph->AddNode(p3_op);

  c1_op->AddInputDesc("f", tensor_desc_p1);
  c1_op->AddInputDesc("w", tensor_desc_p2);
  c1_op->AddInputDesc("c", tensor_desc_p3);
  c1_op->AddOutputDesc("d", tensor_desc_c1_out);

  ge::NodePtr c1_node = graph->AddNode(c1_op);

  GraphUtils::AddEdge(p1_node->GetOutDataAnchor(0), c1_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(p2_node->GetOutDataAnchor(0), c1_node->GetInDataAnchor(1));
  GraphUtils::AddEdge(p3_node->GetOutDataAnchor(0), c1_node->GetInDataAnchor(2));

  
  Status ret = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(c1_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(c1_node, "tbe-custom");
  ASSERT_EQ(ret, fe::SUCCESS);
  vector<int64_t> dim_p1_result({17, 2, 3, 7, 16});

  EXPECT_EQ(ge::GetPrimaryFormat(c1_op->GetInputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(c1_op->GetInputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(c1_op->GetInputDesc(0).GetShape().GetDims(), dim_p1_result);

  EXPECT_EQ(ge::GetPrimaryFormat(c1_op->GetInputDesc(1).GetFormat()), FORMAT_FRACTAL_Z);
  EXPECT_EQ(c1_op->GetInputDesc(1).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(c1_op->GetInputDesc(1).GetShape().GetDims(), dim_p2);

  EXPECT_EQ(c1_op->GetInputDesc(2).GetFormat(), FORMAT_NCHW);
  EXPECT_EQ(c1_op->GetInputDesc(2).GetDataType(), DT_FLOAT);
  EXPECT_EQ(c1_op->GetInputDesc(2).GetShape().GetDims(), dim_p3);

  EXPECT_EQ(ge::GetPrimaryFormat(c1_op->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(c1_op->GetOutputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(c1_op->GetOutputDesc(0).GetShape().GetDims(), exp_dim_c1_out);
}

/* Below is a test case from interface JudgeOp(). Test Tbe MD
 * It will update its format to original format after op_judge and will not update to ND.
 * And there will be no cast op because Op Judge make dtype the same. */
TEST_F(UTEST_fusion_engine_op_judge_new_unittest, set_two_nodes_format_dtype_07)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr m1_op = std::make_shared<OpDesc>("M1", "M");
  OpDescPtr m2_op = std::make_shared<OpDesc>("M2", "M");

  //add descriptor
  vector<int64_t> dim({4, 33, 12, 16});
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NCHW);
  tensor_desc.SetFormat(FORMAT_NCHW);
  tensor_desc.SetDataType(DT_FLOAT);
  m1_op->AddInputDesc("x", tensor_desc);
  m1_op->AddOutputDesc("z", tensor_desc);
  ge::NodePtr m1_node = graph->AddNode(m1_op);

  vector<int64_t> dim_i({1, 2, 3, 4});
  GeShape shape_i(dim_i);
  GeTensorDesc tensor_desc_i(shape_i);
  tensor_desc_i.SetOriginFormat(FORMAT_NCHW);
  tensor_desc_i.SetFormat(FORMAT_NCHW);
  tensor_desc_i.SetDataType(DT_DOUBLE);
  m2_op->AddInputDesc("x", tensor_desc_i);
  GeTensorDesc tensor_desc_o(shape_i);
  tensor_desc_o.SetOriginFormat(FORMAT_NCHW);
  tensor_desc_o.SetFormat(FORMAT_NCHW);
  tensor_desc_o.SetDataType(DT_DOUBLE);
  m2_op->AddOutputDesc("z", tensor_desc_i);
  //ge::AttrUtils::SetInt(h_op, FE_IMPLY_TYPE, 6);
  ge::NodePtr m2_node = graph->AddNode(m2_op);

  GraphUtils::AddEdge(m1_node->GetOutDataAnchor(0), m2_node->GetInDataAnchor(0));
  FEOpsStoreInfo tbe_custom {
      6,
      "tbe-custom",
      EN_IMPL_HW_TBE,
      GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/tbe_custom_opinfo",
      ""};
  vector<FEOpsStoreInfo> store_info;
  store_info.emplace_back(tbe_custom);
  Configuration::Instance(fe::AI_CORE_NAME).ops_store_info_vector_ = (store_info);
  OpImplTypeJudgePtr op_impl_type_judge_ptr = std::make_shared<OpImplTypeJudge>(AI_CORE_NAME, fe_ops_kernel_info_store_ptr_);
  
  TransNodeManagerPtr trans_op_insert_ptr = std::make_shared<TransNodeManager>(fe_ops_kernel_info_store_ptr_);
  Status ret1 = op_impl_type_judge_ptr->MultiThreadJudge(*(graph.get()));
  ASSERT_EQ(ret1, fe::SUCCESS);
  Status ret2 = op_format_dtype_judge_ptr_->Judge(*(graph.get()));
  ret2 = op_format_dtype_judge_ptr_->SetFormat(*(graph.get()));
  ASSERT_EQ(ret2, fe::SUCCESS);

  vector<int64_t> dim_result({4, 33, 12, 16});
  EXPECT_EQ(ge::GetPrimaryFormat(m1_op->GetInputDesc(0).GetFormat()), FORMAT_MD);
  EXPECT_EQ(m1_op->GetInputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(m1_op->GetInputDesc(0).GetShape().GetDims(), dim_result);

  EXPECT_EQ(m1_op->GetOutputDesc(0).GetFormat(), FORMAT_NCHW);
  EXPECT_EQ(m1_op->GetOutputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(m1_op->GetOutputDesc(0).GetShape().GetDims(), dim_result);

  vector<int64_t> dim_result_h({1, 1, 3, 4, 16});
  EXPECT_EQ(m2_op->GetInputDesc(0).GetFormat(), FORMAT_NCHW);
  EXPECT_EQ(m2_op->GetInputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(m2_op->GetInputDesc(0).GetShape().GetDims(), dim_i);

  EXPECT_EQ(ge::GetPrimaryFormat(m2_op->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(m2_op->GetOutputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(m2_op->GetOutputDesc(0).GetShape().GetDims(), dim_result_h);
  trans_op_insert_ptr->Initialize();
  Status ret = trans_op_insert_ptr->InsertAndMergeTransNodes(*(graph.get()));
  ASSERT_EQ(ret, fe::SUCCESS);

  EXPECT_EQ(ge::GetPrimaryFormat(m1_op->GetInputDesc(0).GetFormat()), FORMAT_MD);
  EXPECT_EQ(m1_op->GetInputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(m1_op->GetInputDesc(0).GetShape().GetDims(), dim_result);

  EXPECT_EQ(m1_op->GetOutputDesc(0).GetFormat(), FORMAT_NCHW);
  EXPECT_EQ(m1_op->GetOutputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(m1_op->GetOutputDesc(0).GetShape().GetDims(), dim_result);

  EXPECT_EQ(m2_op->GetInputDesc(0).GetFormat(), FORMAT_NCHW);
  EXPECT_EQ(m2_op->GetInputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(m2_op->GetInputDesc(0).GetShape().GetDims(), dim_i);

  EXPECT_EQ(ge::GetPrimaryFormat(m2_op->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(m2_op->GetOutputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(m2_op->GetOutputDesc(0).GetShape().GetDims(), dim_result_h);
}


/* Test SetDtypeAndFormatByPrecisionMode on op G1 and G2, relation in graph is
 * G1->G2. G1 is the father of G2. G's format after OpFormatDtypeJudge will become NC1HWC0.
 * G2's format is NHWC, and its op kernel supports only NC1HWC0. Due to
 * consecutive principle, we will pick NC1HWC0 based on its predecessor is
 * NC1HWC0. We do this operation for dtype as well. If there is no common
 * dtype between ops kernel and its father, we will still pick father's format
 * and pick the first Dtype belongs to this format.
 * After OpFormatDtypeJudge, Op G1 format is FORMAT_NC1HWC0 and Dtype is Float.
 * Op G2 format is FORMAT_NC1HWC0 and Dtype is Float. And also there shape is updated. */
TEST_F(UTEST_fusion_engine_op_judge_new_unittest, set_two_nodes_format_dtype_08)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g_op = std::make_shared<OpDesc>("G1", "G");
  OpDescPtr g_op2 = std::make_shared<OpDesc>("G2", "G");

  //add descriptor
  vector<int64_t> dim({4, 33, 12, 16});
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NHWC);
  tensor_desc.SetFormat(FORMAT_NHWC);
  tensor_desc.SetDataType(DT_FLOAT);
  g_op->AddInputDesc("x", tensor_desc);
  g_op->AddOutputDesc("z", tensor_desc);
  ge::AttrUtils::SetInt(g_op, FE_IMPLY_TYPE, 6);
  ge::NodePtr g_node = graph->AddNode(g_op);

  vector<int64_t> dim_h({4, 33, 12, 16});
  GeShape shape_h(dim_h);
  GeTensorDesc tensor_desc_h(shape_h);
  tensor_desc_h.SetOriginFormat(FORMAT_NHWC);
  tensor_desc_h.SetFormat(FORMAT_NHWC);
  tensor_desc_h.SetDataType(DT_FLOAT16);
  g_op2->AddInputDesc("x", tensor_desc_h);
  g_op2->AddOutputDesc("z", tensor_desc_h);
  ge::AttrUtils::SetInt(g_op2, FE_IMPLY_TYPE, 6);
  ge::NodePtr h_node = graph->AddNode(g_op2);
  GraphUtils::AddEdge(g_node->GetOutDataAnchor(0), h_node->GetInDataAnchor(0));

  
  Status ret1 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret1 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g_node, "tbe-custom");
  Status ret2 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(h_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret2 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(h_node, "tbe-custom");
  ASSERT_EQ(ret1, fe::SUCCESS);
  ASSERT_EQ(ret2, fe::SUCCESS);
  vector<int64_t> dim_result_n_c1_hw_c0({4, 1, 33, 12, 16});
  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetInputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetInputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(g_op->GetInputDesc(0).GetShape().GetDims(), dim_result_n_c1_hw_c0);

  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetShape().GetDims(), dim_result_n_c1_hw_c0);

  EXPECT_EQ(ge::GetPrimaryFormat(g_op2->GetInputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op2->GetInputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(g_op2->GetInputDesc(0).GetShape().GetDims(), dim_result_n_c1_hw_c0);

  EXPECT_EQ(ge::GetPrimaryFormat(g_op2->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op2->GetOutputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(g_op2->GetOutputDesc(0).GetShape().GetDims(), dim_result_n_c1_hw_c0);
}

/* Almost the same Test senario as 08, but this is for tbe and will set shape to 5D*/
TEST_F(UTEST_fusion_engine_op_judge_new_unittest, set_two_nodes_format_dtype_09)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g_op = std::make_shared<OpDesc>("G1", "G");
  OpDescPtr g_op2 = std::make_shared<OpDesc>("G2", "G");

  //add descriptor
  vector<int64_t> dim({4, 33, 12, 16});
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NHWC);
  tensor_desc.SetFormat(FORMAT_NHWC);
  tensor_desc.SetDataType(DT_FLOAT);
  g_op->AddInputDesc("x", tensor_desc);
  g_op->AddOutputDesc("z", tensor_desc);
  ge::AttrUtils::SetInt(g_op, FE_IMPLY_TYPE, 6);
  ge::NodePtr g_node = graph->AddNode(g_op);

  vector<int64_t> dim_h({4, 33, 12, 16});
  GeShape shape_h(dim_h);
  GeTensorDesc tensor_desc_h(shape_h);
  tensor_desc_h.SetOriginFormat(FORMAT_NHWC);
  tensor_desc_h.SetFormat(FORMAT_NHWC);
  tensor_desc_h.SetDataType(DT_FLOAT16);
  g_op2->AddInputDesc("x", tensor_desc_h);
  g_op2->AddOutputDesc("z", tensor_desc_h);
  ge::AttrUtils::SetInt(g_op2, FE_IMPLY_TYPE, 6);

  ge::NodePtr h_node = graph->AddNode(g_op2);
  GraphUtils::AddEdge(g_node->GetOutDataAnchor(0), h_node->GetInDataAnchor(0));
  
  Status ret1 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret1 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g_node, "tbe-custom");
  Status ret2 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(h_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret2 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(h_node, "tbe-custom");
  ASSERT_EQ(ret1, fe::SUCCESS);
  ASSERT_EQ(ret2, fe::SUCCESS);
  vector<int64_t> dim_result_n_c1_hw_c0({4, 1, 33, 12, 16});
  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetInputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetInputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(g_op->GetInputDesc(0).GetShape().GetDims(), dim_result_n_c1_hw_c0);

  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetShape().GetDims(), dim_result_n_c1_hw_c0);

  EXPECT_EQ(ge::GetPrimaryFormat(g_op2->GetInputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op2->GetInputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(g_op2->GetInputDesc(0).GetShape().GetDims(), dim_result_n_c1_hw_c0);

  EXPECT_EQ(ge::GetPrimaryFormat(g_op2->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op2->GetOutputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(g_op2->GetOutputDesc(0).GetShape().GetDims(), dim_result_n_c1_hw_c0);
}

/* NCHW->NC1HWC0, this case for tbe and will set shape to 5D */
TEST_F(UTEST_fusion_engine_op_judge_new_unittest, set_two_nodes_format_dtype_10)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g_op = std::make_shared<OpDesc>("G1", "G");
  OpDescPtr g_op2 = std::make_shared<OpDesc>("G2", "G");

  //add descriptor
  vector<int64_t> dim({4, 33, 12, 16});
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NCHW);
  tensor_desc.SetFormat(FORMAT_NCHW);
  tensor_desc.SetDataType(DT_FLOAT);
  g_op->AddInputDesc("x", tensor_desc);
  g_op->AddOutputDesc("z", tensor_desc);
  ge::AttrUtils::SetInt(g_op, FE_IMPLY_TYPE, 6);
  ge::NodePtr g_node = graph->AddNode(g_op);

  vector<int64_t> dim_h({4, 33, 12, 16});
  GeShape shape_h(dim_h);
  GeTensorDesc tensor_desc_h(shape_h);
  tensor_desc_h.SetOriginFormat(FORMAT_NCHW);
  tensor_desc_h.SetFormat(FORMAT_NCHW);
  tensor_desc_h.SetDataType(DT_FLOAT16);
  g_op2->AddInputDesc("x", tensor_desc_h);
  g_op2->AddOutputDesc("z", tensor_desc_h);
  ge::AttrUtils::SetInt(g_op2, FE_IMPLY_TYPE, 6);
  ge::NodePtr h_node = graph->AddNode(g_op2);
  GraphUtils::AddEdge(g_node->GetOutDataAnchor(0), h_node->GetInDataAnchor(0));

  
  Status ret1 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret1 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g_node, "tbe-custom");
  Status ret2 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(h_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret2 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(h_node, "tbe-custom");
  ASSERT_EQ(ret1, fe::SUCCESS);
  ASSERT_EQ(ret2, fe::SUCCESS);
  vector<int64_t> dim_result_n_c1_hw_c0({4, 3, 12, 16, 16});
  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetInputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetInputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(g_op->GetInputDesc(0).GetShape().GetDims(), dim_result_n_c1_hw_c0);

  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetShape().GetDims(), dim_result_n_c1_hw_c0);

  EXPECT_EQ(ge::GetPrimaryFormat(g_op2->GetInputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op2->GetInputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(g_op2->GetInputDesc(0).GetShape().GetDims(), dim_result_n_c1_hw_c0);

  EXPECT_EQ(ge::GetPrimaryFormat(g_op2->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op2->GetOutputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(g_op2->GetOutputDesc(0).GetShape().GetDims(), dim_result_n_c1_hw_c0);
}


/* NHWC->NCHW, this case is for tbe and the shape will be transformed.*/
TEST_F(UTEST_fusion_engine_op_judge_new_unittest, set_two_nodes_format_dtype_11)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr b_op = std::make_shared<OpDesc>("B1", "B");
  OpDescPtr b_op2 = std::make_shared<OpDesc>("B2", "B");

  //add descriptor
  vector<int64_t> dim({4, 33, 12, 16});
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NHWC);
  tensor_desc.SetFormat(FORMAT_NHWC);
  tensor_desc.SetDataType(DT_FLOAT);
  b_op->AddInputDesc("x", tensor_desc);
  b_op->AddOutputDesc("z", tensor_desc);
  ge::AttrUtils::SetInt(b_op, FE_IMPLY_TYPE, 6);
  ge::NodePtr g_node = graph->AddNode(b_op);

  vector<int64_t> dim_h({4, 33, 12, 16});
  GeShape shape_h(dim_h);
  GeTensorDesc tensor_desc_h(shape_h);
  tensor_desc_h.SetOriginFormat(FORMAT_NHWC);
  tensor_desc_h.SetFormat(FORMAT_NHWC);
  tensor_desc_h.SetDataType(DT_FLOAT16);
  b_op2->AddInputDesc("x", tensor_desc_h);
  b_op2->AddOutputDesc("z", tensor_desc_h);
  ge::AttrUtils::SetInt(b_op2, FE_IMPLY_TYPE, 6);
  ge::NodePtr h_node = graph->AddNode(b_op2);
  GraphUtils::AddEdge(g_node->GetOutDataAnchor(0), h_node->GetInDataAnchor(0));

  
  Status ret1 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret1 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g_node, "tbe-custom");
  Status ret2 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(h_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret2 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(h_node, "tbe-custom");
  ASSERT_EQ(ret1, fe::SUCCESS);
  ASSERT_EQ(ret2, fe::SUCCESS);
  vector<int64_t> dim_result_nch_w({4, 16, 33, 12});
  EXPECT_EQ(b_op->GetInputDesc(0).GetFormat(), FORMAT_NCHW);
  EXPECT_EQ(b_op->GetInputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(b_op->GetInputDesc(0).GetShape().GetDims(), dim_result_nch_w);

  EXPECT_EQ(b_op->GetOutputDesc(0).GetFormat(), FORMAT_NCHW);
  EXPECT_EQ(b_op->GetOutputDesc(0).GetDataType(), DT_INT8);
  EXPECT_EQ(b_op->GetOutputDesc(0).GetShape().GetDims(), dim_result_nch_w);

  EXPECT_EQ(b_op2->GetInputDesc(0).GetFormat(), FORMAT_NCHW);
  EXPECT_EQ(b_op2->GetInputDesc(0).GetDataType(), DT_INT8);
  EXPECT_EQ(b_op2->GetInputDesc(0).GetShape().GetDims(), dim_result_nch_w);

  EXPECT_EQ(b_op2->GetOutputDesc(0).GetFormat(), FORMAT_NCHW);
  EXPECT_EQ(b_op2->GetOutputDesc(0).GetDataType(), DT_UINT8);
  EXPECT_EQ(b_op2->GetOutputDesc(0).GetShape().GetDims(), dim_result_nch_w);
}


/* NC1HWC0->NCHW, this case is for tbe and the shape will be transformed.*/
TEST_F(UTEST_fusion_engine_op_judge_new_unittest, set_two_nodes_format_dtype_12)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr b_op = std::make_shared<OpDesc>("B1", "B");
  OpDescPtr b_op2 = std::make_shared<OpDesc>("B2", "B");

  //add descriptor
  vector<int64_t> dim({4, 99, 12, 16});
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NCHW);
  tensor_desc.SetFormat(FORMAT_NCHW);
  tensor_desc.SetDataType(DT_FLOAT);
  b_op->AddInputDesc("x", tensor_desc);
  b_op->AddOutputDesc("z", tensor_desc);
  ge::AttrUtils::SetInt(b_op, FE_IMPLY_TYPE, 6);
  ge::NodePtr g_node = graph->AddNode(b_op);

  vector<int64_t> dim_h({4, 33, 12, 16});
  GeShape shape_h(dim_h);
  GeTensorDesc tensor_desc_h(shape_h);
  tensor_desc_h.SetOriginFormat(FORMAT_NCHW);
  tensor_desc_h.SetFormat(FORMAT_NCHW);
  tensor_desc_h.SetDataType(DT_FLOAT16);
  b_op2->AddInputDesc("x", tensor_desc_h);
  b_op2->AddOutputDesc("z", tensor_desc_h);
  ge::AttrUtils::SetInt(b_op2, FE_IMPLY_TYPE, 6);
  ge::NodePtr h_node = graph->AddNode(b_op2);
  GraphUtils::AddEdge(g_node->GetOutDataAnchor(0), h_node->GetInDataAnchor(0));

  
  Status ret1 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret1 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g_node, "tbe-custom");
  Status ret2 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(h_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret2 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(h_node, "tbe-custom");
  ASSERT_EQ(ret1, fe::SUCCESS);
  ASSERT_EQ(ret2, fe::SUCCESS);
  vector<int64_t> dim_result_nch_w({4, 99, 12, 16});
  EXPECT_EQ(b_op->GetInputDesc(0).GetFormat(), FORMAT_NCHW);
  EXPECT_EQ(b_op->GetInputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(b_op->GetInputDesc(0).GetShape().GetDims(), dim_result_nch_w);

  EXPECT_EQ(b_op->GetOutputDesc(0).GetFormat(), FORMAT_NCHW);
  EXPECT_EQ(b_op->GetOutputDesc(0).GetDataType(), DT_INT8);
  EXPECT_EQ(b_op->GetOutputDesc(0).GetShape().GetDims(), dim_result_nch_w);

  EXPECT_EQ(b_op2->GetInputDesc(0).GetFormat(), FORMAT_NCHW);
  EXPECT_EQ(b_op2->GetInputDesc(0).GetDataType(), DT_INT8);
  EXPECT_EQ(b_op2->GetInputDesc(0).GetShape().GetDims(), dim_h);

  EXPECT_EQ(b_op2->GetOutputDesc(0).GetFormat(), FORMAT_NCHW);
  EXPECT_EQ(b_op2->GetOutputDesc(0).GetDataType(), DT_UINT8);
  EXPECT_EQ(b_op2->GetOutputDesc(0).GetShape().GetDims(), dim_h);
}


/* NC1HWC0->NHWC, this case is for tbe and the shape will be transformed.*/
TEST_F(UTEST_fusion_engine_op_judge_new_unittest, set_two_nodes_format_dtype_13)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr d_op = std::make_shared<OpDesc>("D1", "D");
  OpDescPtr d_op2 = std::make_shared<OpDesc>("D2", "D");

  //add descriptor
  vector<int64_t> dim({4, 99, 12, 16});
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NCHW);
  tensor_desc.SetFormat(FORMAT_NCHW);
  tensor_desc.SetDataType(DT_FLOAT);
  d_op->AddInputDesc("x", tensor_desc);
  d_op->AddOutputDesc("z", tensor_desc);
  ge::AttrUtils::SetInt(d_op, FE_IMPLY_TYPE, 6);
  ge::NodePtr g_node = graph->AddNode(d_op);

  vector<int64_t> dim_h({4, 33, 12, 16});
  GeShape shape_h(dim_h);
  GeTensorDesc tensor_desc_h(shape_h);
  tensor_desc_h.SetOriginFormat(FORMAT_NCHW);
  tensor_desc_h.SetFormat(FORMAT_NCHW);
  tensor_desc_h.SetDataType(DT_FLOAT16);
  d_op2->AddInputDesc("x", tensor_desc_h);
  d_op2->AddOutputDesc("z", tensor_desc_h);
  ge::AttrUtils::SetInt(d_op2, FE_IMPLY_TYPE, 6);
  ge::NodePtr h_node = graph->AddNode(d_op2);
  GraphUtils::AddEdge(g_node->GetOutDataAnchor(0), h_node->GetInDataAnchor(0));

  
  Status ret1 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret1 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g_node, "tbe-custom");
  Status ret2 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(h_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret2 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(h_node, "tbe-custom");
  ASSERT_EQ(ret1, fe::SUCCESS);
  ASSERT_EQ(ret2, fe::SUCCESS);
  vector<int64_t> dim_result_nhw_c({4, 12, 16, 99});
  EXPECT_EQ(d_op->GetInputDesc(0).GetFormat(), FORMAT_NHWC);
  EXPECT_EQ(d_op->GetInputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(d_op->GetInputDesc(0).GetShape().GetDims(), dim_result_nhw_c);

  EXPECT_EQ(d_op->GetOutputDesc(0).GetFormat(), FORMAT_NHWC);
  EXPECT_EQ(d_op->GetOutputDesc(0).GetDataType(), DT_INT8);
  EXPECT_EQ(d_op->GetOutputDesc(0).GetShape().GetDims(), dim_result_nhw_c);
  vector<int64_t> dim_result_nhw_c2({4, 12, 16, 33});
  EXPECT_EQ(d_op2->GetInputDesc(0).GetFormat(), FORMAT_NHWC);
  EXPECT_EQ(d_op2->GetInputDesc(0).GetDataType(), DT_INT8);
  EXPECT_EQ(d_op2->GetInputDesc(0).GetShape().GetDims(), dim_result_nhw_c2);

  EXPECT_EQ(d_op2->GetOutputDesc(0).GetFormat(), FORMAT_NHWC);
  EXPECT_EQ(d_op2->GetOutputDesc(0).GetDataType(), DT_UINT8);
  EXPECT_EQ(d_op2->GetOutputDesc(0).GetShape().GetDims(), dim_result_nhw_c2);
}


/* NC1HWC0->NHWC, this case is for tbe and the shape will be transformed.*/
TEST_F(UTEST_fusion_engine_op_judge_new_unittest, set_two_nodes_format_dtype_14)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr d_op = std::make_shared<OpDesc>("D1", "D");
  OpDescPtr d_op2 = std::make_shared<OpDesc>("D2", "D");

  //add descriptor
  vector<int64_t> dim({4, 33, 12, 16});
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NCHW);
  tensor_desc.SetFormat(FORMAT_NCHW);
  tensor_desc.SetDataType(DT_FLOAT);
  d_op->AddInputDesc("x", tensor_desc);
  d_op->AddOutputDesc("z", tensor_desc);
  ge::AttrUtils::SetInt(d_op, FE_IMPLY_TYPE, 6);
  ge::NodePtr g_node = graph->AddNode(d_op);

  vector<int64_t> dim_h({4, 33, 12, 16});
  GeShape shape_h(dim_h);
  GeTensorDesc tensor_desc_h(shape_h);
  tensor_desc_h.SetOriginFormat(FORMAT_NCHW);
  tensor_desc_h.SetFormat(FORMAT_NCHW);
  tensor_desc_h.SetDataType(DT_FLOAT16);
  d_op2->AddInputDesc("x", tensor_desc_h);
  d_op2->AddOutputDesc("z", tensor_desc_h);
  ge::AttrUtils::SetInt(d_op2, FE_IMPLY_TYPE, 6);
  ge::NodePtr h_node = graph->AddNode(d_op2);
  GraphUtils::AddEdge(g_node->GetOutDataAnchor(0), h_node->GetInDataAnchor(0));

  
  Status ret1 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret1 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g_node, "tbe-custom");
  Status ret2 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(h_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret2 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(h_node, "tbe-custom");
  ASSERT_EQ(ret1, fe::SUCCESS);
  ASSERT_EQ(ret2, fe::SUCCESS);
  vector<int64_t> dim_result_nhw_c({4, 12, 16, 33});
  EXPECT_EQ(d_op->GetInputDesc(0).GetFormat(), FORMAT_NHWC);
  EXPECT_EQ(d_op->GetInputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(d_op->GetInputDesc(0).GetShape().GetDims(), dim_result_nhw_c);

  EXPECT_EQ(d_op->GetOutputDesc(0).GetFormat(), FORMAT_NHWC);
  EXPECT_EQ(d_op->GetOutputDesc(0).GetDataType(), DT_INT8);
  EXPECT_EQ(d_op->GetOutputDesc(0).GetShape().GetDims(), dim_result_nhw_c);

  EXPECT_EQ(d_op2->GetInputDesc(0).GetFormat(), FORMAT_NHWC);
  EXPECT_EQ(d_op2->GetInputDesc(0).GetDataType(), DT_INT8);
  EXPECT_EQ(d_op2->GetInputDesc(0).GetShape().GetDims(), dim_result_nhw_c);
  EXPECT_EQ(d_op2->GetOutputDesc(0).GetFormat(), FORMAT_NHWC);
  EXPECT_EQ(d_op2->GetOutputDesc(0).GetDataType(), DT_UINT8);
  EXPECT_EQ(d_op2->GetOutputDesc(0).GetShape().GetDims(), dim_result_nhw_c);
}

/* NC1HWC0->NHWC, this case is for tbe and the shape will be transformed.*/
TEST_F(UTEST_fusion_engine_op_judge_new_unittest, set_two_nodes_format_dtype_15)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr MatMulOp = std::make_shared<OpDesc>("MatMul", "MatMul");
  OpDescPtr ReluOp = std::make_shared<OpDesc>("Relu", "Relu");

  //add descriptor
  vector<int64_t> dim({4, 33, 12, 16});
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NCHW);
  tensor_desc.SetFormat(FORMAT_NCHW);
  tensor_desc.SetDataType(DT_FLOAT16);
  MatMulOp->AddInputDesc("x", tensor_desc);
  MatMulOp->AddOutputDesc("z", tensor_desc);
  ge::AttrUtils::SetInt(MatMulOp, FE_IMPLY_TYPE, 6);
  ge::NodePtr MatMulNode = graph->AddNode(MatMulOp);

  vector<int64_t> dim_h({4, 33, 12, 16});
  GeShape shape_h(dim_h);
  GeTensorDesc tensor_desc_h(shape_h);
  tensor_desc_h.SetOriginFormat(FORMAT_NCHW);
  tensor_desc_h.SetFormat(FORMAT_NCHW);
  tensor_desc_h.SetDataType(DT_FLOAT);
  ReluOp->AddInputDesc("x", tensor_desc_h);
  ReluOp->AddOutputDesc("z", tensor_desc_h);
  ge::AttrUtils::SetInt(ReluOp, FE_IMPLY_TYPE, 6);
  ge::NodePtr ReluNode = graph->AddNode(ReluOp);
  GraphUtils::AddEdge(MatMulNode->GetOutDataAnchor(0), ReluNode->GetInDataAnchor(0));

  

  Status ret1 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(MatMulNode, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret1 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(MatMulNode, "tbe-custom");
  Status ret2 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(ReluNode, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret2 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(ReluNode, "tbe-custom");
  ASSERT_EQ(ret1, fe::SUCCESS);
  ASSERT_EQ(ret2, fe::SUCCESS);
  vector<int64_t> dim_result_n_c1_hw_c0({4, 3, 12, 16, 16});
  EXPECT_EQ(ge::GetPrimaryFormat(MatMulOp->GetInputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(MatMulOp->GetInputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(MatMulOp->GetInputDesc(0).GetShape().GetDims(), dim_result_n_c1_hw_c0);

  EXPECT_EQ(ge::GetPrimaryFormat(MatMulOp->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(MatMulOp->GetOutputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(MatMulOp->GetOutputDesc(0).GetShape().GetDims(), dim_result_n_c1_hw_c0);

  EXPECT_EQ(ReluOp->GetInputDesc(0).GetFormat(), FORMAT_NCHW);
  EXPECT_EQ(ReluOp->GetInputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(ReluOp->GetInputDesc(0).GetShape().GetDims(), dim);
  EXPECT_EQ(ReluOp->GetOutputDesc(0).GetFormat(), FORMAT_NCHW);
  EXPECT_EQ(ReluOp->GetOutputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(ReluOp->GetOutputDesc(0).GetShape().GetDims(), dim);
}

/* Test SetDtypeAndFormatByPrecisionMode on op ConvTemp (TBE_builtin) without predecessor node
 * Set Shape of Fragz as {HWC1, N/16, 16, C0} from NCHW for fp16 input*/
TEST_F(UTEST_fusion_engine_op_judge_new_unittest, set_input_format_succ_format_changed_Conv_Tbe)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g_op = std::make_shared<OpDesc>("ConvTemp", "ConvTemp");
  ge::NodePtr g_node = graph->AddNode(g_op);
  //add descriptor

  vector<int64_t> dim1({1, 65, 64, 64});
  GeShape shape1(dim1);
  GeTensorDesc tensor_desc1(shape1);
  tensor_desc1.SetOriginFormat(FORMAT_NCHW);
  tensor_desc1.SetFormat(FORMAT_NCHW);
  tensor_desc1.SetDataType(DT_FLOAT);
  g_op->AddInputDesc("xasd", tensor_desc1);

  vector<int64_t> dim2({64, 17, 7, 7});
  GeShape shape2(dim2);
  GeTensorDesc tensor_desc2(shape2);
  tensor_desc2.SetOriginFormat(FORMAT_NCHW);
  tensor_desc2.SetFormat(FORMAT_NCHW);
  tensor_desc2.SetDataType(DT_FLOAT);
  g_op->AddInputDesc("wasd", tensor_desc2);

  GeShape shape3;
  GeTensorDesc tensor_desc3(shape3);
  tensor_desc3.SetOriginFormat(FORMAT_NCHW);
  tensor_desc3.SetFormat(FORMAT_NCHW);
  tensor_desc3.SetDataType(DT_FLOAT);
  g_op->AddInputDesc("basd", tensor_desc3);

  vector<int64_t> dimo({1, 64, 30, 30});
  GeShape shapeo(dimo);
  GeTensorDesc tensor_desco(shapeo);
  tensor_desco.SetOriginFormat(FORMAT_NCHW);
  tensor_desco.SetFormat(FORMAT_NCHW);
  tensor_desco.SetDataType(DT_FLOAT);
  g_op->AddOutputDesc("yasd", tensor_desco);

  ge::AttrUtils::SetInt(g_op, FE_IMPLY_TYPE, 6);//TBE
  //Set Special Fragz shape
  
  Status ret = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g_node, "tbe-custom");
  vector<int64_t> dim_result_x = {1,5,64,64,16};
  ASSERT_EQ(ret, fe::SUCCESS);
  ASSERT_EQ(g_op->GetAllInputsDesc().size(), 3);
  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetInputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetInputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(g_op->GetInputDesc(0).GetShape().GetDims(), dim_result_x);

  vector<int64_t> dim_result_w = {98, 4, 16, 16};
  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetInputDesc(1).GetFormat()), FORMAT_FRACTAL_Z);
  EXPECT_EQ(g_op->GetInputDesc(1).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(g_op->GetInputDesc(1).GetShape().GetDims(), dim_result_w);

  EXPECT_EQ(g_op->GetInputDesc(2).GetFormat(), FORMAT_NCHW);
  EXPECT_EQ(g_op->GetInputDesc(2).GetDataType(), DT_FLOAT);

  vector<int64_t> dim_result_o = {1,4,30,30,16};
  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetShape().GetDims(), dim_result_o);
}

/* Test SetDtypeAndFormatByPrecisionMode on op ConvTemp (TBE_builtin) without predecessor node
 * Set Shape of Fragz as {HWC1, N/16, 16, C0} from NHWC for int8 input*/
TEST_F(UTEST_fusion_engine_op_judge_new_unittest, set_input_format_succ_format_changed_Conv_Tbe_2)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g_op = std::make_shared<OpDesc>("ConvTemp", "ConvTemp");
  ge::NodePtr g_node = graph->AddNode(g_op);
  //add descriptor

  vector<int64_t> dim1({1, 66, 64, 64});
  GeShape shape1(dim1);
  GeTensorDesc tensor_desc1(shape1);
  tensor_desc1.SetOriginFormat(FORMAT_NCHW);
  tensor_desc1.SetFormat(FORMAT_NCHW);
  tensor_desc1.SetDataType(DT_INT8);
  g_op->AddInputDesc("xasd", tensor_desc1);

  vector<int64_t> dim2({64, 38, 7, 7});
  GeShape shape2(dim2);
  GeTensorDesc tensor_desc2(shape2);
  tensor_desc2.SetOriginFormat(FORMAT_NHWC);
  tensor_desc2.SetFormat(FORMAT_NHWC);
  tensor_desc2.SetDataType(DT_INT8);
  g_op->AddInputDesc("wasd", tensor_desc2);

  GeShape shape3;
  GeTensorDesc tensor_desc3(shape3);
  tensor_desc3.SetOriginFormat(FORMAT_NCHW);
  tensor_desc3.SetFormat(FORMAT_NCHW);
  tensor_desc3.SetDataType(DT_FLOAT16);
  g_op->AddInputDesc("basd", tensor_desc3);

  vector<int64_t> dimo({1, 64, 30, 30});
  GeShape shapeo(dimo);
  GeTensorDesc tensor_desco(shapeo);
  tensor_desco.SetOriginFormat(FORMAT_NCHW);
  tensor_desco.SetFormat(FORMAT_NCHW);
  tensor_desco.SetDataType(DT_INT8);
  g_op->AddOutputDesc("yasd", tensor_desco);

  ge::AttrUtils::SetInt(g_op, FE_IMPLY_TYPE, 6);//TBE
  
  Status ret = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g_node, "tbe-custom");
  ASSERT_EQ(ret, fe::SUCCESS);
  ASSERT_EQ(g_op->GetAllInputsDesc().size(), 3);
  vector<int64_t> dim_result_x = {1,3,64,64,32};
  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetInputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetInputDesc(0).GetDataType(), DT_INT8);
  EXPECT_EQ(g_op->GetInputDesc(0).GetShape().GetDims(), dim_result_x);

  vector<int64_t> dim_result_w = {266,4,16,32};
  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetInputDesc(1).GetFormat()), FORMAT_FRACTAL_Z);
  EXPECT_EQ(g_op->GetInputDesc(1).GetDataType(), DT_INT8);
  EXPECT_EQ(g_op->GetInputDesc(1).GetShape().GetDims(), dim_result_w);

  EXPECT_EQ(g_op->GetInputDesc(2).GetFormat(), FORMAT_NCHW);
  EXPECT_EQ(g_op->GetInputDesc(2).GetDataType(), DT_FLOAT16);

  vector<int64_t> dim_result_o = {1,2,30,30,32};
  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetDataType(), DT_INT8);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetShape().GetDims(), dim_result_o);

}

/* Test SetDtypeAndFormatByPrecisionMode on op ConvTemp (TBE_builtin) without predecessor node
 * Set Shape of Fragz as {HWC1, N/16, 16, C0} from HWCN for fp16 input*/
TEST_F(UTEST_fusion_engine_op_judge_new_unittest, set_input_format_succ_format_changed_Conv_Tbe_3)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g_op = std::make_shared<OpDesc>("ConvTemp", "ConvTemp");
  ge::NodePtr g_node = graph->AddNode(g_op);
  //add descriptor

  vector<int64_t> dim1({1, 65, 64, 64});
  GeShape shape1(dim1);
  GeTensorDesc tensor_desc1(shape1);
  tensor_desc1.SetOriginFormat(FORMAT_HWCN);
  tensor_desc1.SetFormat(FORMAT_HWCN);
  tensor_desc1.SetDataType(DT_FLOAT);
  g_op->AddInputDesc("xasd", tensor_desc1);

  vector<int64_t> dim2({64, 17, 7, 7});
  GeShape shape2(dim2);
  GeTensorDesc tensor_desc2(shape2);
  tensor_desc2.SetOriginFormat(FORMAT_HWCN);
  tensor_desc2.SetFormat(FORMAT_HWCN);
  tensor_desc2.SetDataType(DT_FLOAT);
  g_op->AddInputDesc("wasd", tensor_desc2);

  GeShape shape3;
  GeTensorDesc tensor_desc3(shape3);
  tensor_desc3.SetOriginFormat(FORMAT_HWCN);
  tensor_desc3.SetFormat(FORMAT_HWCN);
  tensor_desc3.SetDataType(DT_FLOAT);
  g_op->AddInputDesc("basd", tensor_desc3);

  vector<int64_t> dimo({1, 64, 30, 30});
  GeShape shapeo(dimo);
  GeTensorDesc tensor_desco(shapeo);
  tensor_desco.SetOriginFormat(FORMAT_HWCN);
  tensor_desco.SetFormat(FORMAT_HWCN);
  tensor_desco.SetDataType(DT_FLOAT);
  g_op->AddOutputDesc("yasd", tensor_desco);

  ge::AttrUtils::SetInt(g_op, FE_IMPLY_TYPE, 6);//TBE
  //Set Special Fragz shape
  
  Status ret = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g_node, "tbe-custom");
  vector<int64_t> dim_result_x = {64,4,1,65,16};
  ASSERT_EQ(ret, fe::SUCCESS);
  ASSERT_EQ(g_op->GetAllInputsDesc().size(), 3);
  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetInputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetInputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(g_op->GetInputDesc(0).GetShape().GetDims(), dim_result_x);

  vector<int64_t> dim_result_w = {1088, 1, 16, 16};
  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetInputDesc(1).GetFormat()), FORMAT_FRACTAL_Z);
  EXPECT_EQ(g_op->GetInputDesc(1).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(g_op->GetInputDesc(1).GetShape().GetDims(), dim_result_w);

  EXPECT_EQ(g_op->GetInputDesc(2).GetFormat(), FORMAT_HWCN);
  EXPECT_EQ(g_op->GetInputDesc(2).GetDataType(), DT_FLOAT);

  vector<int64_t> dim_result_o = {30,2,1,64,16};
  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetShape().GetDims(), dim_result_o);
}

/* Test SetDtypeAndFormatByPrecisionMode on op ConvTemp (TBE_builtin) without predecessor node
 * Set Shape of NCHW as from Fz for fp16 input*/
TEST_F(UTEST_fusion_engine_op_judge_new_unittest, set_input_format_succ_format_changed_Conv_Tbe_4)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g_op = std::make_shared<OpDesc>("ConvTempFz", "ConvTempFz");
  ge::NodePtr g_node = graph->AddNode(g_op);
  //add descriptor

  vector<int64_t> dim1({1, 65, 64, 64});
  GeShape shape1(dim1);
  GeTensorDesc tensor_desc1(shape1);
  tensor_desc1.SetOriginFormat(FORMAT_HWCN);
  tensor_desc1.SetFormat(FORMAT_HWCN);
  tensor_desc1.SetDataType(DT_FLOAT);
  g_op->AddInputDesc("xasd", tensor_desc1);

  vector<int64_t> dim2({64, 17, 7, 7});
  GeShape shape2(dim2);
  GeTensorDesc tensor_desc2(shape2);
  tensor_desc2.SetOriginFormat(FORMAT_NCHW);
  tensor_desc2.SetFormat(FORMAT_NCHW);
  tensor_desc2.SetDataType(DT_FLOAT);
  g_op->AddInputDesc("wasd", tensor_desc2);

  GeShape shape3;
  GeTensorDesc tensor_desc3(shape3);
  tensor_desc3.SetOriginFormat(FORMAT_HWCN);
  tensor_desc3.SetFormat(FORMAT_HWCN);
  tensor_desc3.SetDataType(DT_FLOAT);
  g_op->AddInputDesc("basd", tensor_desc3);

  vector<int64_t> dimo({1, 64, 30, 30});
  GeShape shapeo(dimo);
  GeTensorDesc tensor_desco(shapeo);
  tensor_desco.SetOriginFormat(FORMAT_NCHW);
  tensor_desco.SetFormat(FORMAT_NCHW);
  tensor_desco.SetDataType(DT_FLOAT);
  g_op->AddOutputDesc("yasd", tensor_desco);

  ge::AttrUtils::SetInt(g_op, FE_IMPLY_TYPE, 6);//TBE
  //Set Special Fragz shape
  
  Status ret = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g_node, "tbe-custom");
  vector<int64_t> dim_result_x = {64,4,1,65,16};
  ASSERT_EQ(ret, fe::SUCCESS);
  ASSERT_EQ(g_op->GetAllInputsDesc().size(), 3);
  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetInputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetInputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(g_op->GetInputDesc(0).GetShape().GetDims(), dim_result_x);

  vector<int64_t> dim_result_w = {7, 7, 17, 64};
  EXPECT_EQ(g_op->GetInputDesc(1).GetFormat(), FORMAT_HWCN);
  EXPECT_EQ(g_op->GetInputDesc(1).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(g_op->GetInputDesc(1).GetShape().GetDims(), dim_result_w);

  EXPECT_EQ(g_op->GetInputDesc(2).GetFormat(), FORMAT_HWCN);
  EXPECT_EQ(g_op->GetInputDesc(2).GetDataType(), DT_FLOAT);

  vector<int64_t> dim_result_o = {1,4,30,30,16};
  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetShape().GetDims(), dim_result_o);
}

TEST_F(UTEST_fusion_engine_op_judge_new_unittest, origin_format_discontinuous){
    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
    OpDescPtr d_op1 = std::make_shared<OpDesc>("D1", "Sqrt");
    OpDescPtr d_op2 = std::make_shared<OpDesc>("D2", "BiasAdd");
    //add descriptor
    vector<int64_t> dim({1, 2, 3, 4});
    GeShape shape(dim);
    GeTensorDesc tensor_desc(shape);
    tensor_desc.SetOriginFormat(FORMAT_NCHW);
    tensor_desc.SetFormat(FORMAT_ND);
    tensor_desc.SetDataType(DT_FLOAT);
    d_op1->AddInputDesc("input0", tensor_desc);
    d_op1->AddOutputDesc("output", tensor_desc);
    ge::AttrUtils::SetInt(d_op1, FE_IMPLY_TYPE, 6);
    ge::NodePtr d_node1 = graph->AddNode(d_op1);

    vector<int64_t> dim_h({1, 2, 3, 4});
    GeShape shape_h(dim_h);
    GeTensorDesc tensor_desc_h(shape_h);
    tensor_desc_h.SetOriginFormat(FORMAT_NHWC);
    tensor_desc_h.SetFormat(FORMAT_ND);
    tensor_desc_h.SetDataType(DT_FLOAT);
    d_op2->AddInputDesc("input0", tensor_desc_h);
    d_op2->AddInputDesc("input1", tensor_desc_h);
    d_op2->AddOutputDesc("output", tensor_desc_h);
    ge::AttrUtils::SetInt(d_op2, FE_IMPLY_TYPE, 6);
    ge::NodePtr d_node2 = graph->AddNode(d_op2);
    GraphUtils::AddEdge(d_node1->GetOutDataAnchor(0), d_node2->GetInDataAnchor(0));

    
    Status ret1 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(d_node1, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
    ret1 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(d_node1, "tbe-custom");
    Status ret2 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(d_node2, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
    ret2 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(d_node2, "tbe-custom");
    ASSERT_EQ(ret1, fe::SUCCESS);
    ASSERT_EQ(ret2, fe::SUCCESS);
    vector<int64_t> dim_result_nhw_c({1, 2, 3, 4});
    EXPECT_EQ(d_op1->GetInputDesc(0).GetFormat(), FORMAT_NCHW);
    EXPECT_EQ(d_op1->GetInputDesc(0).GetDataType(), DT_FLOAT);
    EXPECT_EQ(d_op1->GetInputDesc(0).GetShape().GetDims(), dim_result_nhw_c);
    EXPECT_EQ(d_op1->GetOutputDesc(0).GetFormat(), FORMAT_NCHW);
    EXPECT_EQ(d_op1->GetOutputDesc(0).GetDataType(), DT_FLOAT);
    EXPECT_EQ(d_op1->GetOutputDesc(0).GetShape().GetDims(), dim_result_nhw_c);

    EXPECT_EQ(d_op2->GetInputDesc(0).GetFormat(), FORMAT_NHWC);
    EXPECT_EQ(d_op2->GetInputDesc(0).GetDataType(), DT_FLOAT);
    EXPECT_EQ(d_op2->GetInputDesc(0).GetShape().GetDims(), dim_result_nhw_c);

    EXPECT_EQ(d_op2->GetInputDesc(1).GetFormat(), FORMAT_NHWC);
    EXPECT_EQ(d_op2->GetInputDesc(1).GetDataType(), DT_FLOAT);
    EXPECT_EQ(d_op2->GetInputDesc(1).GetShape().GetDims(), dim_result_nhw_c);

    EXPECT_EQ(d_op2->GetOutputDesc(0).GetFormat(), FORMAT_NHWC);
    EXPECT_EQ(d_op2->GetOutputDesc(0).GetDataType(), DT_FLOAT);
    EXPECT_EQ(d_op2->GetOutputDesc(0).GetShape().GetDims(), dim_result_nhw_c);
}

/* Almost the same Test senario as 08, but this is for tbe and will set shape to 5D*/
TEST_F(UTEST_fusion_engine_op_judge_new_unittest, set_two_nodes_format_reshape_type_not_equal)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g_op = std::make_shared<OpDesc>("G1", "G");
  OpDescPtr g_op2 = std::make_shared<OpDesc>("G2", "G");

  //for G's output 4,33 means HW
  //for G's input 4,33 mean NC
  vector<int64_t> dim({4, 33});
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NHWC);
  tensor_desc.SetFormat(FORMAT_NHWC);
  tensor_desc.SetDataType(DT_FLOAT);
  g_op->AddInputDesc("x", tensor_desc);
  g_op->AddOutputDesc("z", tensor_desc);
  ge::AttrUtils::SetInt(g_op, FE_IMPLY_TYPE, 6);
  ge::NodePtr g_node = graph->AddNode(g_op);

  vector<int64_t> dim_h({4, 33});
  GeShape shape_h(dim_h);
  GeTensorDesc tensor_desc_h(shape_h);
  tensor_desc_h.SetOriginFormat(FORMAT_NHWC);
  tensor_desc_h.SetFormat(FORMAT_NHWC);
  tensor_desc_h.SetDataType(DT_FLOAT16);
  g_op2->AddInputDesc("x", tensor_desc_h);
  g_op2->AddOutputDesc("z", tensor_desc_h);
  ge::AttrUtils::SetInt(g_op2, FE_IMPLY_TYPE, 6);

  ge::NodePtr h_node = graph->AddNode(g_op2);
  GraphUtils::AddEdge(g_node->GetOutDataAnchor(0), h_node->GetInDataAnchor(0));
  
  Status ret1 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret1 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g_node, "tbe-custom");
  Status ret2 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(h_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret2 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(h_node, "tbe-custom");
  ASSERT_EQ(ret1, fe::SUCCESS);
  ASSERT_EQ(ret2, fe::SUCCESS);
  vector<int64_t> dim_result_n_c1_hw_c0_n_c({4, 3, 1, 1, 16});
  vector<int64_t> dim_result_n_c1_hw_c0_h_w({1, 1, 4, 33, 16});
  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetInputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetInputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(g_op->GetInputDesc(0).GetShape().GetDims(), dim_result_n_c1_hw_c0_n_c);

  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetShape().GetDims(), dim_result_n_c1_hw_c0_h_w);

  EXPECT_EQ(ge::GetPrimaryFormat(g_op2->GetInputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op2->GetInputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(g_op2->GetInputDesc(0).GetShape().GetDims(), dim_result_n_c1_hw_c0_n_c);

  EXPECT_EQ(ge::GetPrimaryFormat(g_op2->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op2->GetOutputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(g_op2->GetOutputDesc(0).GetShape().GetDims(), dim_result_n_c1_hw_c0_h_w);
}


/* Test function GenerateInitialMatchedIndexVec */
TEST_F(UTEST_fusion_engine_op_judge_new_unittest, test_func_Generate_initial_matched_index_vec)
{
  FormatDtypeQuerierPtr format_dtype_querier_ptr =
      std::make_shared<FormatDtypeQuerier>(AI_CORE_NAME);
  OpFormatDtypeStrategyManagerPtr strategy_manager_ptr=
      std::make_shared<OpFormatDtypeStrategyManager>(fe::AI_CORE_NAME, format_dtype_querier_ptr);
  bool is_matched_index_vec_inited = false;
  vector<uint32_t> matched_index_vec;
  std::vector<ge::Format> input_format_vec = {ge::FORMAT_ND, ge::FORMAT_NCHW};
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g_op = std::make_shared<OpDesc>("matmul", "MatMul");
  //for G's output 4,33 means HW
  //for G's input 4,33 mean NC
  vector<int64_t> dim({4, 33});
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NHWC);
  tensor_desc.SetFormat(FORMAT_NHWC);
  tensor_desc.SetDataType(DT_FLOAT);
  g_op->AddInputDesc("x", tensor_desc);
  g_op->AddOutputDesc("z", tensor_desc);
  ge::AttrUtils::SetInt(g_op, FE_IMPLY_TYPE, 6);
  ge::NodePtr g_node = graph->AddNode(g_op);
  EXPECT_EQ(strategy_manager_ptr->GenerateInitialMatchedIndexVec(is_matched_index_vec_inited, matched_index_vec,
                                                                 input_format_vec),
            fe::SUCCESS);
}

TEST_F(UTEST_fusion_engine_op_judge_new_unittest, skip_speical_cast_on_non_es_board)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr a_op = std::make_shared<OpDesc>("a", "A");
  OpDescPtr cast_op = std::make_shared<OpDesc>("cast", "Cast");


  OpDescPtr netoutput_op = std::make_shared<OpDesc>("netoutput", "NetOutput");

  //add descriptor
  vector<int64_t> dim({4, 33, 12, 16});
  GeShape shape(dim);
  GeTensorDesc tensor_desc_input(shape);
  tensor_desc_input.SetOriginFormat(FORMAT_NHWC);
  tensor_desc_input.SetFormat(FORMAT_NHWC);
  tensor_desc_input.SetDataType(DT_FLOAT);

  GeTensorDesc tensor_desc_output(shape);
  tensor_desc_output.SetOriginFormat(FORMAT_NHWC);
  tensor_desc_output.SetFormat(FORMAT_NHWC);
  tensor_desc_output.SetDataType(DT_FLOAT16);

  a_op->AddInputDesc(tensor_desc_output);
  a_op->AddOutputDesc(tensor_desc_output);
  ge::NodePtr a_node = graph->AddNode(a_op);

  cast_op->AddInputDesc("x", tensor_desc_input);
  cast_op->AddOutputDesc("z", tensor_desc_output);
  ge::NodePtr cast_node = graph->AddNode(cast_op);

  netoutput_op->AddInputDesc("x", tensor_desc_output);
  netoutput_op->AddOutputDesc("z", tensor_desc_output);
  ge::NodePtr netoutput_node = graph->AddNode(netoutput_op);


  GraphUtils::AddEdge(cast_node->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(a_node->GetOutDataAnchor(0), cast_node->GetInDataAnchor(0));
  ge::AttrUtils::SetInt(netoutput_op, FE_IMPLY_TYPE, 6);//TBE

  OpImplTypeJudgePtr op_impl_type_judge_ptr = std::make_shared<OpImplTypeJudge>(AI_CORE_NAME, fe_ops_kernel_info_store_ptr_);
  
  Status ret1 = op_impl_type_judge_ptr->MultiThreadJudge(*(graph.get()));
  ASSERT_EQ(ret1, fe::SUCCESS);
  Status ret2 = op_format_dtype_judge_ptr_->Judge(*(graph.get()));
  ret2 = op_format_dtype_judge_ptr_->SetFormat(*(graph.get()));
  ASSERT_EQ(ret2, fe::SUCCESS);

  EXPECT_EQ(cast_op->GetInputDesc(0).GetFormat(), FORMAT_NHWC);
  EXPECT_EQ(cast_op->GetInputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(cast_op->GetInputDesc(0).GetShape().GetDims(), dim);

  EXPECT_EQ(cast_op->GetOutputDesc(0).GetFormat(), FORMAT_NHWC);
  EXPECT_EQ(cast_op->GetOutputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(cast_op->GetOutputDesc(0).GetShape().GetDims(), dim);
}

TEST_F(UTEST_fusion_engine_op_judge_new_unittest, force_fp16)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr a_op = std::make_shared<OpDesc>("a", "A");
  OpDescPtr a2_op = std::make_shared<OpDesc>("a2", "A");


  OpDescPtr netoutput_op = std::make_shared<OpDesc>("netoutput", "NetOutput");

  //add descriptor
  vector<int64_t> dim({100, 2, 3, 512, 4});
  GeShape shape(dim);
  GeTensorDesc tensor_desc_input(shape);
  tensor_desc_input.SetOriginFormat(FORMAT_NHWC);
  tensor_desc_input.SetFormat(FORMAT_NHWC);
  tensor_desc_input.SetDataType(DT_INT8);

  GeTensorDesc tensor_desc_output(shape);
  tensor_desc_output.SetOriginFormat(FORMAT_NHWC);
  tensor_desc_output.SetFormat(FORMAT_NHWC);
  tensor_desc_output.SetDataType(DT_FLOAT);

  a_op->AddInputDesc(tensor_desc_input);
  a_op->AddOutputDesc(tensor_desc_output);
  ge::NodePtr a_node = graph->AddNode(a_op);

  a2_op->AddInputDesc("x", tensor_desc_output);
  a2_op->AddOutputDesc("y", tensor_desc_input);
  ge::NodePtr a2_node = graph->AddNode(a2_op);

  netoutput_op->AddInputDesc("x", tensor_desc_output);
  netoutput_op->AddOutputDesc("z", tensor_desc_output);
  ge::NodePtr netoutput_node = graph->AddNode(netoutput_op);

  GraphUtils::AddEdge(a2_node->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(a_node->GetOutDataAnchor(0), a2_node->GetInDataAnchor(0));
  ge::AttrUtils::SetInt(netoutput_op, FE_IMPLY_TYPE, 6);//TBE
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = FORCE_FP16;
  op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();
  OpImplTypeJudgePtr op_impl_type_judge_ptr = std::make_shared<OpImplTypeJudge>(AI_CORE_NAME, fe_ops_kernel_info_store_ptr_);

  PlatformUtils::Instance().soc_version_ = "Ascend910B2";
  Status ret1 = op_impl_type_judge_ptr->MultiThreadJudge(*(graph.get()));
  ASSERT_EQ(ret1, fe::SUCCESS);
  Status ret2 = op_format_dtype_judge_ptr_->Judge(*(graph.get()));
  ret2 = op_format_dtype_judge_ptr_->SetFormat(*(graph.get()));
  ASSERT_EQ(ret2, fe::SUCCESS);
  vector<int64_t> dim_result({100, 16, 2, 3, 32});
  vector<int64_t> dim_result2({100, 32, 2, 3, 16});
  EXPECT_EQ(ge::GetPrimaryFormat(a_op->GetInputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(a_op->GetInputDesc(0).GetDataType(), DT_INT8);
  EXPECT_EQ(a_op->GetInputDesc(0).GetShape().GetDims(), dim_result);

  EXPECT_EQ(ge::GetPrimaryFormat(a_op->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(a_op->GetOutputDesc(0).GetDataType(), DT_UINT8);
  EXPECT_EQ(a_op->GetOutputDesc(0).GetShape().GetDims(), dim_result);

  EXPECT_EQ(ge::GetPrimaryFormat(a2_op->GetInputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(a2_op->GetInputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(a2_op->GetInputDesc(0).GetShape().GetDims(), dim_result2);

  EXPECT_EQ(ge::GetPrimaryFormat(a2_op->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(a2_op->GetOutputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(a2_op->GetOutputDesc(0).GetShape().GetDims(), dim_result2);
}


TEST_F(UTEST_fusion_engine_op_judge_new_unittest, converage_1)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr a_op = std::make_shared<OpDesc>("a", "A");
  OpDescPtr cast_op = std::make_shared<OpDesc>("cast", "Cast");
  ge::AttrUtils::SetBool(cast_op, "_fuzz_build_res", true);

  OpDescPtr netoutput_op = std::make_shared<OpDesc>("netoutput", "NetOutput");

  //add descriptor
  vector<int64_t> dim({4, 33, -1, 16});
  GeShape shape(dim);
  GeTensorDesc tensor_desc_input(shape);
  tensor_desc_input.SetOriginFormat(FORMAT_NHWC);
  tensor_desc_input.SetFormat(FORMAT_NHWC);
  tensor_desc_input.SetDataType(DT_FLOAT);

  GeTensorDesc tensor_desc_output(shape);
  tensor_desc_output.SetOriginFormat(FORMAT_NHWC);
  tensor_desc_output.SetFormat(FORMAT_NHWC);
  tensor_desc_output.SetDataType(DT_FLOAT16);

  a_op->AddInputDesc(tensor_desc_output);
  a_op->AddOutputDesc(tensor_desc_output);
  ge::NodePtr a_node = graph->AddNode(a_op);

  cast_op->AddInputDesc("x", tensor_desc_input);
  cast_op->AddOutputDesc("y", tensor_desc_output);
  ge::NodePtr cast_node = graph->AddNode(cast_op);

  netoutput_op->AddInputDesc("x", tensor_desc_output);
  netoutput_op->AddOutputDesc("z", tensor_desc_output);
  ge::NodePtr netoutput_node = graph->AddNode(netoutput_op);


  GraphUtils::AddEdge(cast_node->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(a_node->GetOutDataAnchor(0), cast_node->GetInDataAnchor(0));
  ge::AttrUtils::SetInt(netoutput_op, FE_IMPLY_TYPE, 6);//TBE

  OpImplTypeJudgePtr op_impl_type_judge_ptr = std::make_shared<OpImplTypeJudge>(AI_CORE_NAME, fe_ops_kernel_info_store_ptr_);

  PlatformUtils::Instance().soc_version_ = "Ascend910B2";
  Status ret1 = op_impl_type_judge_ptr->MultiThreadJudge(*(graph.get()));
  ASSERT_EQ(ret1, fe::SUCCESS);
  Status ret2 = op_format_dtype_judge_ptr_->Judge(*(graph.get()));
  ret2 = op_format_dtype_judge_ptr_->SetFormat(*(graph.get()));
  ASSERT_EQ(ret2, fe::SUCCESS);

  EXPECT_EQ(cast_op->GetInputDesc(0).GetFormat(), FORMAT_NHWC);
  EXPECT_EQ(cast_op->GetInputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(cast_op->GetInputDesc(0).GetShape().GetDims(), dim);

  EXPECT_EQ(cast_op->GetOutputDesc(0).GetFormat(), FORMAT_NHWC);
  EXPECT_EQ(cast_op->GetOutputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(cast_op->GetOutputDesc(0).GetShape().GetDims(), dim);
}

TEST_F(UTEST_fusion_engine_op_judge_new_unittest, get_lx_core_type_with_dynamic_success) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "Relu");
  vector<int64_t> dim(4, 1);
  GeShape shape(dim);
  GeTensorDesc out_desc(shape);
  out_desc.SetOriginFormat(FORMAT_NCHW);
  out_desc.SetFormat(FORMAT_NCHW);
  out_desc.SetDataType(DT_FLOAT16);
  relu_op->AddInputDesc("x", out_desc);
  relu_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(relu_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  ge::AttrUtils::SetInt(relu_op, "imply_type", 1);
  NodePtr relu_node = graph->AddNode(relu_op);

  OpKernelInfoPtr op_kernel_info_ptr =
      OpsKernelManager::Instance(fe::AI_CORE_NAME).GetOpKernelInfoByOpType(EN_IMPL_HW_TBE, "Relu");
  op_kernel_info_ptr->op_str_param_vec_[static_cast<size_t>(OP_KERNEL_STR_PARAM::CoreType)] = "dynamic";
  op_kernel_info_ptr->impl_type_ = EN_IMPL_HW_TBE;

  OpImplTypeJudgePtr op_impl_type_judge_ptr = std::make_shared<OpImplTypeJudge>(AI_CORE_NAME, fe_ops_kernel_info_store_ptr_);
  Status ret = op_impl_type_judge_ptr->GetLXCoreType(relu_node);
  EXPECT_EQ(fe::SUCCESS, ret);
  string core_type;
  ge::AttrUtils::GetStr(relu_node->GetOpDesc(), CORE_TYPE_VALUE, core_type);
  EXPECT_EQ(core_type, "AiCore,MIX");
}

TEST_F(UTEST_fusion_engine_op_judge_new_unittest, get_lx_core_type_with_dynamic_failed_001) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "Relu");
  vector<int64_t> dim(4, 1);
  GeShape shape(dim);
  GeTensorDesc out_desc(shape);
  out_desc.SetOriginFormat(FORMAT_NCHW);
  out_desc.SetFormat(FORMAT_NCHW);
  out_desc.SetDataType(DT_FLOAT16);
  relu_op->AddInputDesc("x", out_desc);
  relu_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(relu_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr relu_node = graph->AddNode(relu_op);

  OpKernelInfoPtr op_kernel_info_ptr =
      OpsKernelManager::Instance(fe::AI_CORE_NAME).GetOpKernelInfoByOpType(EN_IMPL_HW_TBE, "Relu");
  op_kernel_info_ptr->op_str_param_vec_[static_cast<size_t>(OP_KERNEL_STR_PARAM::CoreType)] = "dynamic";
  op_kernel_info_ptr->impl_type_ = EN_IMPL_HW_TBE;

  OpImplTypeJudgePtr op_impl_type_judge_ptr = std::make_shared<OpImplTypeJudge>(AI_CORE_NAME, fe_ops_kernel_info_store_ptr_);
  Status ret = op_impl_type_judge_ptr->GetLXCoreType(relu_node);
  EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(UTEST_fusion_engine_op_judge_new_unittest, get_lx_core_type_with_dynamic_failed_002) {
  FEOpsKernelInfoStorePtr store_ptr = std::make_shared<FEOpsKernelInfoStore>(fe::AI_CORE_NAME);
  OpStoreAdapterPtr op_store_adapter = nullptr;
  Status ret = store_ptr->GetOpStoreAdapter(EN_IMPL_CUSTOM_CONSTANT_CCE, op_store_adapter);
  EXPECT_EQ(fe::FAILED, ret);
}

TEST_F(UTEST_fusion_engine_op_judge_new_unittest, test_filter_format_ffts_only_support_nz) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  ge::AttrUtils::SetBool(graph, ffts::kFftsSwitch, true);

  OpDescPtr conv2d = std::make_shared<OpDesc>("conv2d", "ConvTemp");
  OpDescPtr conv7d = std::make_shared<OpDesc>("conv7d", "Conv7D");
  OpDescPtr bm = std::make_shared<OpDesc>("bm", "BatchMatMul");

  vector<int64_t> dim(4, 1);
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NHWC);
  tensor_desc.SetFormat(FORMAT_NHWC);
  tensor_desc.SetDataType(DT_FLOAT16);
  conv2d->AddInputDesc(tensor_desc);
  conv2d->AddInputDesc(tensor_desc);
  conv2d->AddOutputDesc(tensor_desc);

  conv7d->AddInputDesc(tensor_desc);
  conv7d->AddInputDesc(tensor_desc);
  conv7d->AddOutputDesc(tensor_desc);

  bm->AddInputDesc(tensor_desc);
  bm->AddInputDesc(tensor_desc);
  bm->AddOutputDesc(tensor_desc);

  ge::AttrUtils::SetInt(conv2d, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  ge::AttrUtils::SetInt(conv7d, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  ge::AttrUtils::SetInt(bm, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr conv2d_node = graph->AddNode(conv2d);
  NodePtr conv7d_node = graph->AddNode(conv7d);
  NodePtr bm_node = graph->AddNode(bm);

  Status ret = op_format_dtype_judge_ptr_->Judge(*graph);
  ret = op_format_dtype_judge_ptr_->SetFormat(*graph);

  EXPECT_EQ(ge::GetPrimaryFormat(conv2d->MutableInputDesc(0)->GetFormat()), ge::FORMAT_NC1HWC0);
  EXPECT_EQ(ge::GetPrimaryFormat(conv7d->MutableInputDesc(0)->GetFormat()), ge::FORMAT_NC1HWC0);
  EXPECT_EQ(ge::GetPrimaryFormat(bm->MutableInputDesc(0)->GetFormat()), ge::FORMAT_FRACTAL_NZ);
  EXPECT_EQ(fe::SUCCESS, ret);
}


TEST_F(UTEST_fusion_engine_op_judge_new_unittest, test_filter_format_no_ffts_only_support_nz) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  ge::AttrUtils::SetBool(graph, ffts::kFftsSwitch, false);

  OpDescPtr conv2d = std::make_shared<OpDesc>("conv2d", "ConvTemp");
  OpDescPtr conv7d = std::make_shared<OpDesc>("conv7d", "Conv7D");
  OpDescPtr bm = std::make_shared<OpDesc>("bm", "BatchMatMul");

  vector<int64_t> dim(4, 1);
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NHWC);
  tensor_desc.SetFormat(FORMAT_NHWC);
  tensor_desc.SetDataType(DT_FLOAT16);
  conv2d->AddInputDesc(tensor_desc);
  conv2d->AddInputDesc(tensor_desc);
  conv2d->AddOutputDesc(tensor_desc);

  conv7d->AddInputDesc(tensor_desc);
  conv7d->AddInputDesc(tensor_desc);
  conv7d->AddOutputDesc(tensor_desc);

  bm->AddInputDesc(tensor_desc);
  bm->AddInputDesc(tensor_desc);
  bm->AddOutputDesc(tensor_desc);

  ge::AttrUtils::SetInt(conv2d, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  ge::AttrUtils::SetInt(conv7d, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  ge::AttrUtils::SetInt(bm, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr conv2d_node = graph->AddNode(conv2d);
  NodePtr conv7d_node = graph->AddNode(conv7d);
  NodePtr bm_node = graph->AddNode(bm);

  Status ret = op_format_dtype_judge_ptr_->Judge(*graph);
  ret = op_format_dtype_judge_ptr_->SetFormat(*graph);

  EXPECT_EQ(ge::GetPrimaryFormat(conv2d->MutableInputDesc(0)->GetFormat()), ge::FORMAT_NC1HWC0);
  EXPECT_EQ(ge::GetPrimaryFormat(conv7d->MutableInputDesc(0)->GetFormat()), ge::FORMAT_NC1HWC0);
  EXPECT_EQ(ge::GetPrimaryFormat(bm->MutableInputDesc(0)->GetFormat()), ge::FORMAT_FRACTAL_NZ);
  EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(UTEST_fusion_engine_op_judge_new_unittest, test_filter_format_ffts_select_nhwc) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  ge::AttrUtils::SetBool(graph, ffts::kFftsSwitch, true);

  OpDescPtr bm = std::make_shared<OpDesc>("bm", "BatchMatMulFfts");

  vector<int64_t> dim(4, 1);
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NHWC);
  tensor_desc.SetFormat(FORMAT_NHWC);
  tensor_desc.SetDataType(DT_FLOAT16);

  bm->AddInputDesc(tensor_desc);
  bm->AddInputDesc(tensor_desc);
  bm->AddOutputDesc(tensor_desc);


  ge::AttrUtils::SetInt(bm, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr bm_node = graph->AddNode(bm);

  Status ret = op_format_dtype_judge_ptr_->Judge(*graph);
  ret = op_format_dtype_judge_ptr_->SetFormat(*graph);

  EXPECT_EQ(bm->MutableInputDesc(0)->GetFormat(), ge::FORMAT_NHWC);
  EXPECT_EQ(fe::SUCCESS, ret);
}


TEST_F(UTEST_fusion_engine_op_judge_new_unittest, test_filter_format_no_ffts_select_nz) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  ge::AttrUtils::SetBool(graph, ffts::kFftsSwitch, false);

  OpDescPtr bm = std::make_shared<OpDesc>("bm", "BatchMatMulFfts");

  vector<int64_t> dim(4, 1);
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NHWC);
  tensor_desc.SetFormat(FORMAT_NHWC);
  tensor_desc.SetDataType(DT_FLOAT16);

  bm->AddInputDesc(tensor_desc);
  bm->AddInputDesc(tensor_desc);
  bm->AddOutputDesc(tensor_desc);


  ge::AttrUtils::SetInt(bm, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr bm_node = graph->AddNode(bm);

  Status ret = op_format_dtype_judge_ptr_->Judge(*graph);
  ret = op_format_dtype_judge_ptr_->SetFormat(*graph);

  EXPECT_EQ(ge::GetPrimaryFormat(bm->MutableInputDesc(0)->GetFormat()), ge::FORMAT_FRACTAL_NZ);
  EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(UTEST_fusion_engine_op_judge_new_unittest, test_bf16_notsupported)
{
  PlatformInfoManager::Instance().opti_compilation_infos_.SetSocVersion("Ascend910B");
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = ALLOW_FP32_TO_BF16;
  op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr_);
  EXPECT_EQ(op_format_dtype_judge_ptr_->Initialize(), fe::FAILED);
}


TEST_F(UTEST_fusion_engine_op_judge_new_unittest, test_format_mode_nd) {
  ge::GetThreadLocalContext().graph_options_["ge.exec.formatMode"] = "1";
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr conv2d = std::make_shared<OpDesc>("conv2d", "ConvTemp");
  OpDescPtr softmax = std::make_shared<OpDesc>("softmax", "SoftMax");
  OpDescPtr bm = std::make_shared<OpDesc>("bm", "MatMulV2");

  vector<int64_t> dim(4, 1);
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NHWC);
  tensor_desc.SetFormat(FORMAT_NHWC);
  tensor_desc.SetDataType(DT_FLOAT);
  conv2d->AddInputDesc(tensor_desc);
  conv2d->AddInputDesc(tensor_desc);
  conv2d->AddOutputDesc(tensor_desc);

  softmax->AddInputDesc(tensor_desc);
  softmax->AddOutputDesc(tensor_desc);

  bm->AddInputDesc(tensor_desc);
  bm->AddInputDesc(tensor_desc);
  bm->AddOutputDesc(tensor_desc);

  ge::AttrUtils::SetInt(conv2d, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  ge::AttrUtils::SetInt(softmax, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  ge::AttrUtils::SetInt(bm, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr conv2d_node = graph->AddNode(conv2d);
  NodePtr conv7d_node = graph->AddNode(softmax);
  NodePtr bm_node = graph->AddNode(bm);

  Status ret = op_format_dtype_judge_ptr_->Judge(*graph);
  ret = op_format_dtype_judge_ptr_->SetFormat(*graph);

  EXPECT_EQ(ge::GetPrimaryFormat(conv2d->MutableInputDesc(0)->GetFormat()), ge::FORMAT_NC1HWC0);
  EXPECT_EQ(ge::GetPrimaryFormat(softmax->MutableInputDesc(0)->GetFormat()), ge::FORMAT_NHWC);
  EXPECT_EQ(ge::GetPrimaryFormat(bm->MutableInputDesc(0)->GetFormat()), ge::FORMAT_NHWC);
  EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(UTEST_fusion_engine_op_judge_new_unittest, test_format_mode_nd_02) {
  ge::GetThreadLocalContext().graph_options_["ge.exec.formatMode"] = "0";
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr conv2d = std::make_shared<OpDesc>("conv2d", "ConvTemp");
  OpDescPtr softmax = std::make_shared<OpDesc>("softmax", "SoftMax");
  OpDescPtr bm = std::make_shared<OpDesc>("bm", "BatchMatMul");

  vector<int64_t> dim(4, 1);
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NHWC);
  tensor_desc.SetFormat(FORMAT_NHWC);
  tensor_desc.SetDataType(DT_FLOAT);
  conv2d->AddInputDesc(tensor_desc);
  conv2d->AddInputDesc(tensor_desc);
  conv2d->AddOutputDesc(tensor_desc);

  softmax->AddInputDesc(tensor_desc);
  softmax->AddOutputDesc(tensor_desc);

  tensor_desc.SetDataType(DT_FLOAT16);
  bm->AddInputDesc(tensor_desc);
  bm->AddInputDesc(tensor_desc);
  bm->AddOutputDesc(tensor_desc);

  ge::AttrUtils::SetInt(conv2d, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  ge::AttrUtils::SetInt(softmax, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  ge::AttrUtils::SetInt(bm, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr conv2d_node = graph->AddNode(conv2d);
  NodePtr conv7d_node = graph->AddNode(softmax);
  NodePtr bm_node = graph->AddNode(bm);

  Status ret = op_format_dtype_judge_ptr_->Judge(*graph);
  ret = op_format_dtype_judge_ptr_->SetFormat(*graph);

  EXPECT_EQ(ge::GetPrimaryFormat(conv2d->MutableInputDesc(0)->GetFormat()), ge::FORMAT_NC1HWC0);
  EXPECT_EQ(ge::GetPrimaryFormat(softmax->MutableInputDesc(0)->GetFormat()), ge::FORMAT_NHWC);
  EXPECT_EQ(ge::GetPrimaryFormat(bm->MutableInputDesc(0)->GetFormat()), ge::FORMAT_FRACTAL_NZ);
  EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(UTEST_fusion_engine_op_judge_new_unittest, test_format_mode_NDNZ_success) {
  ge::GetThreadLocalContext().graph_options_["ge.exec.formatMode"] = "2";
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr conv2d = std::make_shared<OpDesc>("conv2d", "ConvTemp");
  OpDescPtr softmax = std::make_shared<OpDesc>("softmax", "SoftMax");
  OpDescPtr bm_v2 = std::make_shared<OpDesc>("bm_v2", "BatchMatMulV2");
  vector<int64_t> dim(4, 1);
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NHWC);
  tensor_desc.SetFormat(FORMAT_NHWC);
  tensor_desc.SetDataType(DT_FLOAT);
  conv2d->AddInputDesc(tensor_desc);
  conv2d->AddInputDesc(tensor_desc);
  conv2d->AddOutputDesc(tensor_desc);

  softmax->AddInputDesc(tensor_desc);
  softmax->AddOutputDesc(tensor_desc);

  tensor_desc.SetDataType(DT_FLOAT16);
  tensor_desc.SetFormat(FORMAT_ND);
  bm_v2->AddInputDesc(tensor_desc);
  tensor_desc.SetDataType(DT_FLOAT16);
  tensor_desc.SetFormat(FORMAT_FRACTAL_NZ);
  bm_v2->AddInputDesc(tensor_desc);
  bm_v2->AddOutputDesc(tensor_desc);
  ge::AttrUtils::SetInt(conv2d, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  ge::AttrUtils::SetInt(softmax, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  ge::AttrUtils::SetInt(bm_v2, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr conv2d_node = graph->AddNode(conv2d);
  NodePtr conv7d_node = graph->AddNode(softmax);
  NodePtr bm_node = graph->AddNode(bm_v2);

  Status ret = op_format_dtype_judge_ptr_->Judge(*graph);
  ret = op_format_dtype_judge_ptr_->SetFormat(*graph);

  EXPECT_EQ(ge::GetPrimaryFormat(conv2d->MutableInputDesc(0)->GetFormat()), ge::FORMAT_NC1HWC0);
  EXPECT_EQ(ge::GetPrimaryFormat(softmax->MutableInputDesc(0)->GetFormat()), ge::FORMAT_NHWC);
  EXPECT_EQ(ge::GetPrimaryFormat(bm_v2->MutableInputDesc(0)->GetFormat()), ge::FORMAT_NHWC);
  EXPECT_EQ(ge::GetPrimaryFormat(bm_v2->MutableInputDesc(1)->GetFormat()), ge::FORMAT_FRACTAL_NZ);
  EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(UTEST_fusion_engine_op_judge_new_unittest, test_format_mode_NDNZ_failed) {
  ge::GetThreadLocalContext().graph_options_["ge.exec.formatMode"] = "2";
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr conv2d = std::make_shared<OpDesc>("conv2d", "ConvTemp");
  OpDescPtr softmax = std::make_shared<OpDesc>("softmax", "SoftMax");
  OpDescPtr bm_v2 = std::make_shared<OpDesc>("bm_v2", "BatchMatMulV2");
  vector<int64_t> dim(4, 1);
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NHWC);
  tensor_desc.SetFormat(FORMAT_NHWC);
  tensor_desc.SetDataType(DT_FLOAT);
  conv2d->AddInputDesc(tensor_desc);
  conv2d->AddInputDesc(tensor_desc);
  conv2d->AddOutputDesc(tensor_desc);

  softmax->AddInputDesc(tensor_desc);
  softmax->AddOutputDesc(tensor_desc);

  tensor_desc.SetDataType(DT_FLOAT);
  tensor_desc.SetFormat(FORMAT_FRACTAL_NZ);
  bm_v2->AddInputDesc(tensor_desc);
  tensor_desc.SetDataType(DT_FLOAT);
  tensor_desc.SetFormat(FORMAT_FRACTAL_NZ);
  bm_v2->AddInputDesc(tensor_desc);
  bm_v2->AddOutputDesc(tensor_desc);
  ge::AttrUtils::SetInt(conv2d, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  ge::AttrUtils::SetInt(softmax, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  ge::AttrUtils::SetInt(bm_v2, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr conv2d_node = graph->AddNode(conv2d);
  NodePtr conv7d_node = graph->AddNode(softmax);
  NodePtr bm_node = graph->AddNode(bm_v2);

  Status ret = op_format_dtype_judge_ptr_->Judge(*graph);
  ret = op_format_dtype_judge_ptr_->SetFormat(*graph);

  EXPECT_EQ(ge::GetPrimaryFormat(conv2d->MutableInputDesc(0)->GetFormat()), ge::FORMAT_NC1HWC0);
  EXPECT_EQ(ge::GetPrimaryFormat(softmax->MutableInputDesc(0)->GetFormat()), ge::FORMAT_NHWC);
  EXPECT_EQ(ge::GetPrimaryFormat(bm_v2->MutableInputDesc(0)->GetFormat()), ge::FORMAT_FRACTAL_NZ);
  EXPECT_EQ(ge::GetPrimaryFormat(bm_v2->MutableInputDesc(1)->GetFormat()), ge::FORMAT_FRACTAL_NZ);
  EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(UTEST_fusion_engine_op_judge_new_unittest, test_complex32_check_support_and_select_dtype_succ)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g_op = std::make_shared<OpDesc>("Complex32Op", "Complex32Op");

  //add descriptor
  vector<int64_t> dim({4, 33, 12, 16});
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NHWC);
  tensor_desc.SetFormat(FORMAT_NHWC);
  tensor_desc.SetDataType(DT_COMPLEX32);
  g_op->AddInputDesc("x", tensor_desc);
  g_op->AddInputDesc("y", tensor_desc);
  g_op->AddOutputDesc("z", tensor_desc);
  ge::NodePtr g_node = graph->AddNode(g_op);

  ge::AttrUtils::SetInt(g_op, FE_IMPLY_TYPE, 6);//TBE
  OpImplTypeJudgePtr op_impl_type_judge_ptr = std::make_shared<OpImplTypeJudge>(AI_CORE_NAME, fe_ops_kernel_info_store_ptr_);

  Status ret1 = op_impl_type_judge_ptr->MultiThreadJudge(*(graph.get()));
  ASSERT_EQ(ret1, fe::SUCCESS);
  Status ret2 = op_format_dtype_judge_ptr_->Judge(*(graph.get()));
  ret2 = op_format_dtype_judge_ptr_->SetFormat(*(graph.get()));
  ASSERT_EQ(ret2, fe::SUCCESS);

  EXPECT_EQ(g_op->GetInputDesc(0).GetDataType(), DT_COMPLEX32);
  EXPECT_EQ(g_op->GetInputDesc(1).GetDataType(), DT_COMPLEX32);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetDataType(), DT_COMPLEX32);
}

TEST_F(UTEST_fusion_engine_op_judge_new_unittest, test_complex32_check_support_and_select_dtype_succ1)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g_op = std::make_shared<OpDesc>("Complex32Op", "Complex32Op1");

  //add descriptor
  vector<int64_t> dim({4, 33, 12, 16});
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NHWC);
  tensor_desc.SetFormat(FORMAT_NHWC);
  tensor_desc.SetDataType(DT_COMPLEX32);
  g_op->AddInputDesc("x", tensor_desc);
  g_op->AddInputDesc("y", tensor_desc);
  g_op->AddOutputDesc("z", tensor_desc);
  ge::NodePtr g_node = graph->AddNode(g_op);

  ge::AttrUtils::SetInt(g_op, FE_IMPLY_TYPE, 6);//TBE
  OpImplTypeJudgePtr op_impl_type_judge_ptr = std::make_shared<OpImplTypeJudge>(AI_CORE_NAME, fe_ops_kernel_info_store_ptr_);

  Status ret1 = op_impl_type_judge_ptr->MultiThreadJudge(*(graph.get()));
  ASSERT_EQ(ret1, fe::SUCCESS);
  Status ret2 = op_format_dtype_judge_ptr_->Judge(*(graph.get()));
  ret2 = op_format_dtype_judge_ptr_->SetFormat(*(graph.get()));
  ASSERT_EQ(ret2, fe::SUCCESS);

  EXPECT_EQ(g_op->GetInputDesc(0).GetDataType(), DT_COMPLEX32);
  EXPECT_EQ(g_op->GetInputDesc(1).GetDataType(), DT_FLOAT);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetDataType(), DT_FLOAT);
}

TEST_F(UTEST_fusion_engine_op_judge_new_unittest, test_complex32_check_support_fail)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g_op = std::make_shared<OpDesc>("Complex32Op", "Complex32Op2");

  //add descriptor
  vector<int64_t> dim({4, 33, 12, 16});
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NHWC);
  tensor_desc.SetFormat(FORMAT_NHWC);
  tensor_desc.SetDataType(DT_COMPLEX32);
  g_op->AddInputDesc("x", tensor_desc);
  g_op->AddInputDesc("y", tensor_desc);
  g_op->AddOutputDesc("z", tensor_desc);
  ge::NodePtr g_node = graph->AddNode(g_op);

  ge::AttrUtils::SetInt(g_op, FE_IMPLY_TYPE, 6);//TBE
  OpImplTypeJudgePtr op_impl_type_judge_ptr = std::make_shared<OpImplTypeJudge>(AI_CORE_NAME, fe_ops_kernel_info_store_ptr_);

  Status ret1 = op_impl_type_judge_ptr->MultiThreadJudge(*(graph.get()));
  ASSERT_EQ(ret1, fe::SUCCESS);
}

TEST_F(UTEST_fusion_engine_op_judge_new_unittest, test_complex64_check_support_and_select_dtype_succ)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g_op = std::make_shared<OpDesc>("Complex64Op", "Complex64Op");

  //add descriptor
  vector<int64_t> dim({4, 33, 12, 16});
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NHWC);
  tensor_desc.SetFormat(FORMAT_NHWC);
  tensor_desc.SetDataType(DT_COMPLEX64);
  g_op->AddInputDesc("x", tensor_desc);
  g_op->AddInputDesc("y", tensor_desc);
  g_op->AddOutputDesc("z", tensor_desc);
  ge::NodePtr g_node = graph->AddNode(g_op);

  ge::AttrUtils::SetInt(g_op, FE_IMPLY_TYPE, 6);//TBE
  OpImplTypeJudgePtr op_impl_type_judge_ptr = std::make_shared<OpImplTypeJudge>(AI_CORE_NAME, fe_ops_kernel_info_store_ptr_);

  Status ret1 = op_impl_type_judge_ptr->MultiThreadJudge(*(graph.get()));
  ASSERT_EQ(ret1, fe::SUCCESS);
  Status ret2 = op_format_dtype_judge_ptr_->Judge(*(graph.get()));
  ret2 = op_format_dtype_judge_ptr_->SetFormat(*(graph.get()));
  ASSERT_EQ(ret2, fe::SUCCESS);

  EXPECT_EQ(g_op->GetInputDesc(0).GetDataType(), DT_COMPLEX64);
  EXPECT_EQ(g_op->GetInputDesc(1).GetDataType(), DT_COMPLEX64);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetDataType(), DT_COMPLEX64);
}

TEST_F(UTEST_fusion_engine_op_judge_new_unittest, test_complex64_check_support_and_select_dtype_succ_forcefp32)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g_op = std::make_shared<OpDesc>("Complex64Op", "Complex64Op");
  string bak_precision_mode = ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE];
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = FORCE_FP32;
  //add descriptor
  vector<int64_t> dim({4, 33, 12, 16});
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NHWC);
  tensor_desc.SetFormat(FORMAT_NHWC);
  tensor_desc.SetDataType(DT_COMPLEX64);
  g_op->AddInputDesc("x", tensor_desc);
  g_op->AddInputDesc("y", tensor_desc);
  g_op->AddOutputDesc("z", tensor_desc);
  ge::NodePtr g_node = graph->AddNode(g_op);

  ge::AttrUtils::SetInt(g_op, FE_IMPLY_TYPE, 6);//TBE
  OpImplTypeJudgePtr op_impl_type_judge_ptr = std::make_shared<OpImplTypeJudge>(AI_CORE_NAME, fe_ops_kernel_info_store_ptr_);

  Status ret1 = op_impl_type_judge_ptr->MultiThreadJudge(*(graph.get()));
  ASSERT_EQ(ret1, fe::SUCCESS);
  Status ret2 = op_format_dtype_judge_ptr_->Judge(*(graph.get()));
  ret2 = op_format_dtype_judge_ptr_->SetFormat(*(graph.get()));
  ASSERT_EQ(ret2, fe::SUCCESS);

  EXPECT_EQ(g_op->GetInputDesc(0).GetDataType(), DT_COMPLEX64);
  EXPECT_EQ(g_op->GetInputDesc(1).GetDataType(), DT_COMPLEX64);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetDataType(), DT_COMPLEX64);

  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = bak_precision_mode;
}

TEST_F(UTEST_fusion_engine_op_judge_new_unittest, test_judge_format_and_dtype_by_aclnn_1) {
  Configuration &config = Configuration::Instance(fe::AI_CORE_NAME);
  EXPECT_EQ(config.IsEnableAclnn(), false);
  std::array<string, static_cast<size_t>(ENV_STR_PARAM::EnvStrParamBottom)>
      env_str_param_vec_bac = config.env_str_param_vec_;
  mmSetEnv("ENABLE_ACLNN", "true", 1);
  config.InitParamFromEnv();
  EXPECT_EQ(config.IsEnableAclnn(), true);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr bm = std::make_shared<OpDesc>("bm", "BatchMatMul");
  OpDescPtr bm_support_aclnn = std::make_shared<OpDesc>("bm_support_aclnn", "BatchMatMulSupportAclnn");
  OpDescPtr bm_aclnn_only = std::make_shared<OpDesc>("bm_aclnn_only", "BatchMatMulAclnnOnly");
  OpDescPtr bm_aclnn_xxx = std::make_shared<OpDesc>("bm_aclnn_xxx", "BatchMatMulAclnnXXX");

  vector<int64_t> dim({4, 33, 12, 16});
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NHWC);
  tensor_desc.SetFormat(FORMAT_NHWC);
  tensor_desc.SetDataType(DT_FLOAT16);
  bm->AddInputDesc(tensor_desc);
  bm->AddInputDesc(tensor_desc);
  bm->AddOutputDesc(tensor_desc);
  bm_support_aclnn->AddInputDesc(tensor_desc);
  bm_support_aclnn->AddInputDesc(tensor_desc);
  bm_support_aclnn->AddOutputDesc(tensor_desc);
  bm_aclnn_only->AddInputDesc(tensor_desc);
  bm_aclnn_only->AddInputDesc(tensor_desc);
  bm_aclnn_only->AddOutputDesc(tensor_desc);
  bm_aclnn_xxx->AddInputDesc(tensor_desc);
  bm_aclnn_xxx->AddInputDesc(tensor_desc);
  bm_aclnn_xxx->AddOutputDesc(tensor_desc);
  ge::AttrUtils::SetInt(bm, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  ge::AttrUtils::SetInt(bm_support_aclnn, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  ge::AttrUtils::SetInt(bm_aclnn_only, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  ge::AttrUtils::SetInt(bm_aclnn_xxx, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);

  NodePtr bm_node = graph->AddNode(bm);
  NodePtr bm_support_aclnn_node = graph->AddNode(bm_support_aclnn);
  NodePtr bm_aclnn_only_node = graph->AddNode(bm_aclnn_only);
  NodePtr bm_aclnn_xxx_node = graph->AddNode(bm_aclnn_xxx);
  OpSetterPtr op_setter_ptr = std::make_shared<OpSetter>(AI_CORE_NAME);
  OpKernelInfoPtr op_kernel_info_ptr =
      OpsKernelManager::Instance(fe::AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-custom", "BatchMatMul");
  EXPECT_EQ(op_kernel_info_ptr->GetAclnnSupportType(), AclnnSupportType::DEFAULT);
  EXPECT_EQ(op_setter_ptr->SetAclnnAttr(
      bm_node, 3, op_kernel_info_ptr,
      op_format_dtype_judge_ptr_->op_format_dtype_strategy_manager_ptr_->GetPrecisionMode()),
      false);
  op_kernel_info_ptr =
      OpsKernelManager::Instance(fe::AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-custom", "BatchMatMulSupportAclnn");
  EXPECT_EQ(op_kernel_info_ptr->GetAclnnSupportType(), AclnnSupportType::SUPPORT_ACLNN);
  EXPECT_EQ(op_setter_ptr->SetAclnnAttr(
      bm_support_aclnn_node, 3, op_kernel_info_ptr,
      op_format_dtype_judge_ptr_->op_format_dtype_strategy_manager_ptr_->GetPrecisionMode()),
      false);
  op_kernel_info_ptr =
      OpsKernelManager::Instance(fe::AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-custom", "BatchMatMulAclnnOnly");
  EXPECT_EQ(op_kernel_info_ptr->GetAclnnSupportType(), AclnnSupportType::ACLNN_ONLY);
  EXPECT_EQ(op_setter_ptr->SetAclnnAttr(
      bm_aclnn_only_node, 3, op_kernel_info_ptr,
      op_format_dtype_judge_ptr_->op_format_dtype_strategy_manager_ptr_->GetPrecisionMode()),
      true);
  op_kernel_info_ptr =
      OpsKernelManager::Instance(fe::AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-custom", "BatchMatMulAclnnXXX");
  EXPECT_EQ(op_kernel_info_ptr->GetAclnnSupportType(), AclnnSupportType::DEFAULT);
  EXPECT_EQ(op_setter_ptr->SetAclnnAttr(
      bm_aclnn_xxx_node, 3, op_kernel_info_ptr,
      op_format_dtype_judge_ptr_->op_format_dtype_strategy_manager_ptr_->GetPrecisionMode()),
      false);

  Status ret = op_format_dtype_judge_ptr_->Judge(*graph);
  ret = op_format_dtype_judge_ptr_->SetFormat(*graph);
  EXPECT_EQ(fe::SUCCESS, ret);
  EXPECT_EQ(ge::GetPrimaryFormat(bm->MutableInputDesc(0)->GetFormat()), ge::FORMAT_FRACTAL_NZ);
  EXPECT_EQ(ge::GetPrimaryFormat(bm_support_aclnn->MutableInputDesc(0)->GetFormat()), ge::FORMAT_FRACTAL_NZ);
  EXPECT_EQ(ge::GetPrimaryFormat(bm_aclnn_only->MutableInputDesc(0)->GetFormat()), ge::FORMAT_FRACTAL_NZ);
  EXPECT_EQ(ge::GetPrimaryFormat(bm_aclnn_xxx->MutableInputDesc(0)->GetFormat()), ge::FORMAT_FRACTAL_NZ);

  unsetenv("ENABLE_ACLNN");
  config.env_str_param_vec_ = env_str_param_vec_bac;
  config.InitParamFromEnv();
  EXPECT_EQ(config.IsEnableAclnn(), false);
}

TEST_F(UTEST_fusion_engine_op_judge_new_unittest, test_judge_format_and_dtype_by_aclnn_2) {
  Configuration &config = Configuration::Instance(fe::AI_CORE_NAME);
  EXPECT_EQ(config.IsEnableAclnn(), false);
  std::array<string, static_cast<size_t>(ENV_STR_PARAM::EnvStrParamBottom)>
      env_str_param_vec_bac = config.env_str_param_vec_;
  mmSetEnv("ENABLE_ACLNN", "true", 1);
  config.InitParamFromEnv();
  EXPECT_EQ(config.IsEnableAclnn(), true);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr bm = std::make_shared<OpDesc>("bm", "BatchMatMul");
  OpDescPtr bm_support_aclnn = std::make_shared<OpDesc>("bm_support_aclnn", "BatchMatMulSupportAclnn");
  OpDescPtr bm_aclnn_only = std::make_shared<OpDesc>("bm_aclnn_only", "BatchMatMulAclnnOnly");
  OpDescPtr bm_aclnn_xxx = std::make_shared<OpDesc>("bm_aclnn_xxx", "BatchMatMulAclnnXXX");

  // unknown shape
  vector<int64_t> dim({4, -1, 12, 16});
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NHWC);
  tensor_desc.SetFormat(FORMAT_NHWC);
  tensor_desc.SetDataType(DT_FLOAT16);
  bm->AddInputDesc(tensor_desc);
  bm->AddInputDesc(tensor_desc);
  bm->AddOutputDesc(tensor_desc);
  bm_support_aclnn->AddInputDesc(tensor_desc);
  bm_support_aclnn->AddInputDesc(tensor_desc);
  bm_support_aclnn->AddOutputDesc(tensor_desc);
  bm_aclnn_only->AddInputDesc(tensor_desc);
  bm_aclnn_only->AddInputDesc(tensor_desc);
  bm_aclnn_only->AddOutputDesc(tensor_desc);
  bm_aclnn_xxx->AddInputDesc(tensor_desc);
  bm_aclnn_xxx->AddInputDesc(tensor_desc);
  bm_aclnn_xxx->AddOutputDesc(tensor_desc);
  ge::AttrUtils::SetInt(bm, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  ge::AttrUtils::SetInt(bm_support_aclnn, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  ge::AttrUtils::SetInt(bm_aclnn_only, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  ge::AttrUtils::SetInt(bm_aclnn_xxx, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);

  NodePtr bm_node = graph->AddNode(bm);
  NodePtr bm_support_aclnn_node = graph->AddNode(bm_support_aclnn);
  NodePtr bm_aclnn_only_node = graph->AddNode(bm_aclnn_only);
  NodePtr bm_aclnn_xxx_node = graph->AddNode(bm_aclnn_xxx);
  OpSetterPtr op_setter_ptr = std::make_shared<OpSetter>(AI_CORE_NAME);
  OpKernelInfoPtr op_kernel_info_ptr =
      OpsKernelManager::Instance(fe::AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-custom", "BatchMatMul");
  EXPECT_EQ(op_kernel_info_ptr->GetAclnnSupportType(), AclnnSupportType::DEFAULT);
  EXPECT_EQ(op_setter_ptr->SetAclnnAttr(
      bm_node, 3, op_kernel_info_ptr,
      op_format_dtype_judge_ptr_->op_format_dtype_strategy_manager_ptr_->GetPrecisionMode()),
      false);
  op_kernel_info_ptr =
      OpsKernelManager::Instance(fe::AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-custom", "BatchMatMulSupportAclnn");
  EXPECT_EQ(op_kernel_info_ptr->GetAclnnSupportType(), AclnnSupportType::SUPPORT_ACLNN);
  EXPECT_EQ(op_setter_ptr->SetAclnnAttr(
      bm_support_aclnn_node, 3, op_kernel_info_ptr,
      op_format_dtype_judge_ptr_->op_format_dtype_strategy_manager_ptr_->GetPrecisionMode()),
      false);
  op_kernel_info_ptr =
      OpsKernelManager::Instance(fe::AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-custom", "BatchMatMulAclnnOnly");
  EXPECT_EQ(op_kernel_info_ptr->GetAclnnSupportType(), AclnnSupportType::ACLNN_ONLY);
  EXPECT_EQ(op_setter_ptr->SetAclnnAttr(
      bm_aclnn_only_node, 3, op_kernel_info_ptr,
      op_format_dtype_judge_ptr_->op_format_dtype_strategy_manager_ptr_->GetPrecisionMode()),
      true);
  op_kernel_info_ptr =
      OpsKernelManager::Instance(fe::AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-custom", "BatchMatMulAclnnXXX");
  EXPECT_EQ(op_kernel_info_ptr->GetAclnnSupportType(), AclnnSupportType::DEFAULT);
  EXPECT_EQ(op_setter_ptr->SetAclnnAttr(
      bm_aclnn_xxx_node, 3, op_kernel_info_ptr,
      op_format_dtype_judge_ptr_->op_format_dtype_strategy_manager_ptr_->GetPrecisionMode()),
      false);

  Status ret = op_format_dtype_judge_ptr_->Judge(*graph);
  ret = op_format_dtype_judge_ptr_->SetFormat(*graph);
  EXPECT_EQ(fe::SUCCESS, ret);
  EXPECT_EQ(ge::GetPrimaryFormat(bm->MutableInputDesc(0)->GetFormat()), ge::FORMAT_FRACTAL_NZ);
  EXPECT_EQ(ge::GetPrimaryFormat(bm_support_aclnn->MutableInputDesc(0)->GetFormat()), ge::FORMAT_FRACTAL_NZ);
  EXPECT_EQ(ge::GetPrimaryFormat(bm_aclnn_only->MutableInputDesc(0)->GetFormat()), ge::FORMAT_FRACTAL_NZ);
  EXPECT_EQ(ge::GetPrimaryFormat(bm_aclnn_xxx->MutableInputDesc(0)->GetFormat()), ge::FORMAT_FRACTAL_NZ);

  unsetenv("ENABLE_ACLNN");
  config.env_str_param_vec_ = env_str_param_vec_bac;
  config.InitParamFromEnv();
  EXPECT_EQ(config.IsEnableAclnn(), false);
}

TEST_F(UTEST_fusion_engine_op_judge_new_unittest, PromoteTypeMatch) {
  OpFormatDtypeJudgePtr op_format_dtype_judge_ptr = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr_);
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::OpDescPtr op_desc_ptr = make_shared<ge::OpDesc>("axpy", "AxpyV2");
  ge::NodePtr test_node = graph->AddNode(op_desc_ptr);

  ge::Format format_5dh_16 = static_cast<ge::Format>(ge::GetFormatFromC0(ge::FORMAT_NC1HWC0, 5));
  std::vector<int64_t> dims_nchw{10,20,15,15};
  std::vector<int64_t> dims_5hd{10,2,15,15,16};
  ge::GeShape shape_nchw(dims_nchw);
  ge::GeShape shape_5hd(dims_5hd);
  ge::GeTensorDesc tensor_desc(shape_5hd, format_5dh_16, ge::DT_FLOAT16);
  tensor_desc.SetOriginDataType(ge::DT_FLOAT);
  tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  tensor_desc.SetOriginShape(shape_nchw);
  ge::GeTensorDesc tensor_desc1 = tensor_desc;
  tensor_desc1.SetDataType(ge::DT_INT32);
  tensor_desc.SetDataType(ge::DT_FLOAT);

  op_desc_ptr->AddInputDesc("x1", tensor_desc);
  op_desc_ptr->AddInputDesc("x2", tensor_desc1);
  op_desc_ptr->AddInputDesc("x3", tensor_desc1);
  op_desc_ptr->AddOutputDesc("y", tensor_desc);
  OpJudgeParam op_judge_param(test_node, nullptr);
  std::vector<int> tmp_vec = {0, 1, 2};
  std::vector<std::vector<int>> promote_inputs_indexes;
  std::vector<uint32_t> matched_index_vec;
  promote_inputs_indexes.emplace_back(tmp_vec);
  EXPECT_EQ(op_format_dtype_judge_ptr->PromoteTypeMatch(op_judge_param, promote_inputs_indexes, matched_index_vec), fe::FAILED);
}

TEST_F(UTEST_fusion_engine_op_judge_new_unittest, test_op_setter_aclnn_fallback) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr bm = std::make_shared<OpDesc>("bm", "BatchMatMul");

  // unknown shape
  vector<int64_t> dim({4, -1, 12, 16});
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NHWC);
  tensor_desc.SetFormat(FORMAT_NHWC);
  tensor_desc.SetDataType(DT_FLOAT16);
  bm->AddInputDesc(tensor_desc);
  bm->AddInputDesc(tensor_desc);
  bm->AddOutputDesc(tensor_desc);
  ge::AttrUtils::SetInt(bm, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr bm_node = graph->AddNode(bm);

  OpSetterPtr op_setter_ptr = std::make_shared<OpSetter>(AI_CORE_NAME);
  ge::GetThreadLocalContext().graph_options_[ge::ALLOW_HF32] = "11";
  op_setter_ptr->SetAttrForAclnnLowering(
                  bm_node, op_format_dtype_judge_ptr_->op_format_dtype_strategy_manager_ptr_->GetPrecisionMode());
  std::string allow_hf32_str = "";
  (void)ge::AttrUtils::GetStr(bm_node->GetOpDesc(), ge::ALLOW_HF32, allow_hf32_str);
  EXPECT_EQ(allow_hf32_str, "11");
}
