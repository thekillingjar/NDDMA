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
#include "ops_store/ops_kernel_manager.h"
#include "ops_store/sub_op_info_store.h"
#include "ops_store/op_kernel_info.h"
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
#include "common/config_parser/op_cust_dtypes_config_parser.h"
#include "common/platform_utils.h"
#include "common/configuration.h"
#include "common/fe_op_info_common.h"
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

class UTEST_fusion_engine_op_judge_precision_mode : public testing::Test
{
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
        "",
        false,
        false,
        false};
    vector<FEOpsStoreInfo> store_info;
    store_info.emplace_back(tbe_custom);
    Configuration::Instance(fe::AI_CORE_NAME).ops_store_info_vector_ = (store_info);
    if (Configuration::Instance(fe::AI_CORE_NAME).cust_dtypes_parser_ == nullptr) {
      Configuration::Instance(fe::AI_CORE_NAME).cust_dtypes_parser_ = make_shared<OpCustDtypesConfigParser>();
    }
    op_cust_dtypes_parser_ptr_ =
            std::dynamic_pointer_cast<OpCustDtypesConfigParser>(Configuration::Instance(fe::AI_CORE_NAME).cust_dtypes_parser_);

    OpsKernelManager::Instance(AI_CORE_NAME).Finalize();

    fe_ops_kernel_info_store_ptr_->Initialize(options);
    reflection_builder_ptr_ = std::make_shared<ge::RefRelations>();

    op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr_);
    op_format_dtype_judge_ptr_->Initialize();

    OpsKernelManager::Instance(AI_CORE_NAME).Finalize();

    OpsKernelManager::Instance(AI_CORE_NAME).Initialize();

  }

  void TearDown()
  {

  }

  void ConstructAndCheck(const std::string &precision_mode) {
    op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr_);
    op_format_dtype_judge_ptr_->Initialize();
    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
    OpDescPtr g_op = std::make_shared<OpDesc>("G1", "PadV4");

    //add descriptor
    vector<int64_t> dim({4, 33, 12, 16});
    GeShape shape(dim);
    GeTensorDesc tensor_desc(shape);
    tensor_desc.SetOriginFormat(FORMAT_NCHW);
    tensor_desc.SetFormat(FORMAT_NCHW);
    tensor_desc.SetDataType(DT_BF16);
    tensor_desc.SetOriginDataType(DT_BF16);

    g_op->AddInputDesc("x", tensor_desc);
    g_op->AddOutputDesc("z", tensor_desc);
    ge::AttrUtils::SetInt(g_op, FE_IMPLY_TYPE, 6);
    ge::NodePtr g_node = graph->AddNode(g_op);
    string reason;
    bool check_result = fe_ops_kernel_info_store_ptr_->CheckSupported(g_node, reason);
    if (precision_mode == ALLOW_FP32_TO_BF16 || precision_mode == ALLOW_MIX_PRECISION_BF16) {
      ASSERT_EQ(check_result, false);
      return;
    } else {
      ASSERT_EQ(check_result, true);
    }

    Status ret1 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret1 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g_node, "tbe-custom");
    ASSERT_EQ(ret1, fe::SUCCESS);
    EXPECT_EQ(g_op->GetInputDesc(0).GetFormat(), FORMAT_NCHW);
    EXPECT_EQ(g_op->GetInputDesc(0).GetDataType(), DT_FLOAT16);
    EXPECT_EQ(g_op->GetInputDesc(0).GetShape().GetDims(), dim);

    EXPECT_EQ(g_op->GetOutputDesc(0).GetFormat(), FORMAT_NCHW);
    EXPECT_EQ(g_op->GetOutputDesc(0).GetDataType(), DT_FLOAT16);
    EXPECT_EQ(g_op->GetOutputDesc(0).GetShape().GetDims(), dim);
  }

  void ConstructAndCheck2(const std::string &op_type, ge::DataType original,
                          ge::DataType input_expect,
                          ge::DataType output_expect) {
    std::cout << op_type << std::endl;
    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
    OpDescPtr g_op = std::make_shared<OpDesc>("G1", op_type);

    //add descriptor
    vector<int64_t> dim({4, 33, 12, 16});
    GeShape shape(dim);
    GeTensorDesc tensor_desc(shape);
    tensor_desc.SetOriginFormat(FORMAT_NCHW);
    tensor_desc.SetFormat(FORMAT_NCHW);
    tensor_desc.SetDataType(original);
    tensor_desc.SetOriginDataType(original);

    g_op->AddInputDesc("x", tensor_desc);
    g_op->AddOutputDesc("z", tensor_desc);
    ge::AttrUtils::SetInt(g_op, FE_IMPLY_TYPE, 6);
    ge::NodePtr g_node = graph->AddNode(g_op);
    string reason;
    bool check_result = fe_ops_kernel_info_store_ptr_->CheckSupported(g_node, reason);
    ASSERT_EQ(check_result, true);

    Status ret1 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret1 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g_node, "tbe-custom");
    ASSERT_EQ(ret1, fe::SUCCESS);
    EXPECT_EQ(g_op->GetInputDesc(0).GetFormat(), FORMAT_NCHW);
    EXPECT_EQ(g_op->GetInputDesc(0).GetDataType(), input_expect);
    EXPECT_EQ(g_op->GetInputDesc(0).GetShape().GetDims(), dim);

    EXPECT_EQ(g_op->GetOutputDesc(0).GetFormat(), FORMAT_NCHW);
    EXPECT_EQ(g_op->GetOutputDesc(0).GetDataType(), output_expect);
    EXPECT_EQ(g_op->GetOutputDesc(0).GetShape().GetDims(), dim);
  }

    void ConstructAndCheck3(const std::string &op_type, ge::DataType original,
                          ge::DataType input_expect, ge::DataType output_expect,
                          bool check_result_expect, Status set_result_expect,
                          const std::string &op_name = "G1") {
    std::cout << op_type << std::endl;
    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
    OpDescPtr g_op = std::make_shared<OpDesc>(op_name, op_type);

    //add descriptor
    vector<int64_t> dim({4, 33, 12, 16});
    GeShape shape(dim);
    GeTensorDesc tensor_desc(shape);
    tensor_desc.SetOriginFormat(FORMAT_NCHW);
    tensor_desc.SetFormat(FORMAT_NCHW);
    tensor_desc.SetDataType(original);
    tensor_desc.SetOriginDataType(original);

    g_op->AddInputDesc("x", tensor_desc);
    g_op->AddOutputDesc("z", tensor_desc);
    ge::AttrUtils::SetInt(g_op, FE_IMPLY_TYPE, 6);
    ge::NodePtr g_node = graph->AddNode(g_op);
    string reason;
    bool check_result = fe_ops_kernel_info_store_ptr_->CheckSupported(g_node, reason);
    EXPECT_EQ(check_result, check_result_expect);

    Status ret1 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret1 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g_node, "tbe-custom");
    EXPECT_EQ(ret1, set_result_expect);
    EXPECT_EQ(g_op->GetInputDesc(0).GetFormat(), FORMAT_NCHW);
    EXPECT_EQ(g_op->GetInputDesc(0).GetDataType(), input_expect);
    EXPECT_EQ(g_op->GetInputDesc(0).GetShape().GetDims(), dim);
    EXPECT_EQ(g_op->GetOutputDesc(0).GetFormat(), FORMAT_NCHW);
    EXPECT_EQ(g_op->GetOutputDesc(0).GetDataType(), output_expect);
    EXPECT_EQ(g_op->GetOutputDesc(0).GetShape().GetDims(), dim);
  }

  void ConstructAndCheckHasFatherNode(const std::string &op_type, ge::DataType father_dtype, ge::DataType original,
                          ge::DataType input_expect, ge::DataType output_expect,
                          bool check_result_expect, Status set_result_expect,
                          const std::string &father_type = "CubeHif8SupportHif8") {
    std::cout << father_type << " -> " << op_type << std::endl;
    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
    OpDescPtr f_op = std::make_shared<OpDesc>("G0", father_type);
    OpDescPtr g_op = std::make_shared<OpDesc>("G1", op_type);

    vector<int64_t> dim_f({4, 33, 12, 16});
    GeShape shape_f(dim_f);
    GeTensorDesc tensor_desc_f(shape_f);
    tensor_desc_f.SetOriginFormat(FORMAT_NCHW);
    tensor_desc_f.SetFormat(FORMAT_NCHW);
    tensor_desc_f.SetDataType(father_dtype);
    tensor_desc_f.SetOriginDataType(father_dtype);
    f_op->AddInputDesc("x", tensor_desc_f);
    f_op->AddOutputDesc("z", tensor_desc_f);
    ge::AttrUtils::SetInt(f_op, FE_IMPLY_TYPE, 6);
    ge::NodePtr f_node = graph->AddNode(f_op);
    
    vector<int64_t> dim({4, 33, 12, 16});
    GeShape shape(dim);
    GeTensorDesc tensor_desc(shape);
    tensor_desc.SetOriginFormat(FORMAT_NCHW);
    tensor_desc.SetFormat(FORMAT_NCHW);
    tensor_desc.SetDataType(original);
    tensor_desc.SetOriginDataType(original);
    g_op->AddInputDesc("x", tensor_desc);
    g_op->AddOutputDesc("z", tensor_desc);
    ge::AttrUtils::SetInt(g_op, FE_IMPLY_TYPE, 6);
    ge::NodePtr g_node = graph->AddNode(g_op);

    GraphUtils::AddEdge(f_node->GetOutDataAnchor(0), g_node->GetInDataAnchor(0));

    string reason;
    bool check_result = fe_ops_kernel_info_store_ptr_->CheckSupported(g_node, reason);
    EXPECT_EQ(check_result, check_result_expect);

    Status ret1 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret1 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g_node, "tbe-custom");
    EXPECT_EQ(ret1, set_result_expect);
    EXPECT_EQ(g_op->GetInputDesc(0).GetFormat(), FORMAT_NCHW);
    EXPECT_EQ(g_op->GetInputDesc(0).GetDataType(), input_expect);
    EXPECT_EQ(g_op->GetInputDesc(0).GetShape().GetDims(), dim);
    EXPECT_EQ(g_op->GetOutputDesc(0).GetFormat(), FORMAT_NCHW);
    EXPECT_EQ(g_op->GetOutputDesc(0).GetDataType(), output_expect);
    EXPECT_EQ(g_op->GetOutputDesc(0).GetShape().GetDims(), dim);
  }

  static void CreateUnknownShapeGraph(ComputeGraphPtr graph, ge::Format d_format, ge::Format format, ge::GeShape unknown_shape) {
    OpDescPtr g_op = std::make_shared<OpDesc>("Data", fe::DATA);
    OpDescPtr h_op = std::make_shared<OpDesc>("UnknownShape1", "UnknownShape");

    //add descriptor
    vector<int64_t> dim({4, 33, 12, 16, 16});
    GeShape shape(dim);
    GeTensorDesc tensor_desc(shape);
    tensor_desc.SetOriginFormat(FORMAT_NCHW);
    tensor_desc.SetFormat(d_format);
    tensor_desc.SetDataType(DT_FLOAT16);
    g_op->AddInputDesc("x", tensor_desc);
    g_op->AddOutputDesc("z", tensor_desc);
    ge::AttrUtils::SetInt(g_op, FE_IMPLY_TYPE, 6);
    ge::NodePtr g_node = graph->AddNode(g_op);

    GeTensorDesc tensor_desc_h(unknown_shape);
    tensor_desc_h.SetOriginFormat(format);
    tensor_desc_h.SetFormat(format);
    tensor_desc_h.SetDataType(DT_FLOAT16);
    h_op->AddInputDesc("x", tensor_desc_h);
    h_op->AddOutputDesc("z", tensor_desc_h);
    ge::AttrUtils::SetInt(h_op, FE_IMPLY_TYPE, 6);
    ge::NodePtr h_node = graph->AddNode(h_op);
    GraphUtils::AddEdge(g_node->GetOutDataAnchor(0), h_node->GetInDataAnchor(0));
  }
  static void CreateTwoInputOneOutputNode(OpDescPtr node_op, vector<int64_t> dim, ge::DataType Dtype) {
    GeShape shape(dim);
    GeTensorDesc tensor_desc(shape);
    tensor_desc.SetOriginFormat(FORMAT_NCHW);
    tensor_desc.SetFormat(FORMAT_NCHW);
    tensor_desc.SetDataType(Dtype);
    node_op->AddInputDesc("x", tensor_desc);
    node_op->AddInputDesc("y", tensor_desc);
    node_op->AddOutputDesc("z", tensor_desc);
    ge::AttrUtils::SetInt(node_op, FE_IMPLY_TYPE, 6);
  }
  static void CreateSetOneNode(OpDescPtr node_op, vector<int64_t> dim, ge::DataType Dtype) {
    GeShape shape(dim);
    GeTensorDesc tensor_desc(shape);
    tensor_desc.SetOriginFormat(FORMAT_NCHW);
    tensor_desc.SetFormat(FORMAT_NCHW);
    tensor_desc.SetDataType(Dtype);
    node_op->AddInputDesc("x", tensor_desc);
    node_op->AddOutputDesc("z", tensor_desc);
    ge::AttrUtils::SetInt(node_op, FE_IMPLY_TYPE, 6);
  }

  static void CreatePromoteNode(OpDescPtr node_op, vector<int64_t> dim, ge::DataType dtype1, ge::DataType dtype2) {
    GeShape shape(dim);
    GeTensorDesc tensor_desc(shape);
    tensor_desc.SetOriginFormat(FORMAT_NCHW);
    tensor_desc.SetFormat(FORMAT_NCHW);
    tensor_desc.SetDataType(dtype1);
    GeTensorDesc tensor_desc2 = tensor_desc;
    tensor_desc2.SetDataType(dtype2);
    node_op->AddInputDesc("x1", tensor_desc);
    node_op->AddInputDesc("x2", tensor_desc2);
    node_op->AddOutputDesc("y", tensor_desc2);
    ge::AttrUtils::SetInt(node_op, FE_IMPLY_TYPE, 6);
  }

  shared_ptr<fe::FEOpsKernelInfoStore> fe_ops_kernel_info_store_ptr_;
  RefRelationsPtr reflection_builder_ptr_;
  OpFormatDtypeJudgePtr op_format_dtype_judge_ptr_;
  OpCustDtypesConfigParserPtr op_cust_dtypes_parser_ptr_;
};

TEST_F(UTEST_fusion_engine_op_judge_precision_mode, set_two_nodes_format_dtype_force_fp16)
{
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = FORCE_FP16;
  op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g_op = std::make_shared<OpDesc>("G1", "G");
  OpDescPtr h_op = std::make_shared<OpDesc>("G2", "G");

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
  tensor_desc_h.SetDataType(DT_INT8);
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
  vector<int64_t> dim_result_nhw_c_to_NC1_hw_c0({1, 1, 2, 3, 32});
  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetInputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetInputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(g_op->GetInputDesc(0).GetShape().GetDims(), dim_result_nch_w_to_NC1_hw_c0);

  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetShape().GetDims(), dim_result_nch_w_to_NC1_hw_c0);

  EXPECT_EQ(ge::GetPrimaryFormat(h_op->GetInputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(h_op->GetInputDesc(0).GetDataType(), DT_INT8);
  EXPECT_EQ(h_op->GetInputDesc(0).GetShape().GetDims(), dim_result_nhw_c_to_NC1_hw_c0);

  EXPECT_EQ(ge::GetPrimaryFormat(h_op->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(h_op->GetOutputDesc(0).GetDataType(), DT_INT8);
  EXPECT_EQ(h_op->GetOutputDesc(0).GetShape().GetDims(), dim_result_nhw_c_to_NC1_hw_c0);
}

TEST_F(UTEST_fusion_engine_op_judge_precision_mode, set_two_nodes_format_dtype_force_fp16_v2)
{
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = V2_FP16;
  op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME,
                                                                    reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g_op = std::make_shared<OpDesc>("G1", "G");
  OpDescPtr h_op = std::make_shared<OpDesc>("G2", "G");

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
  tensor_desc_h.SetDataType(DT_INT8);
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
  vector<int64_t> dim_result_nhw_c_to_NC1_hw_c0({1, 1, 2, 3, 32});
  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetInputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetInputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(g_op->GetInputDesc(0).GetShape().GetDims(), dim_result_nch_w_to_NC1_hw_c0);

  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetShape().GetDims(), dim_result_nch_w_to_NC1_hw_c0);

  EXPECT_EQ(ge::GetPrimaryFormat(h_op->GetInputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(h_op->GetInputDesc(0).GetDataType(), DT_INT8);
  EXPECT_EQ(h_op->GetInputDesc(0).GetShape().GetDims(), dim_result_nhw_c_to_NC1_hw_c0);

  EXPECT_EQ(ge::GetPrimaryFormat(h_op->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(h_op->GetOutputDesc(0).GetDataType(), DT_INT8);
  EXPECT_EQ(h_op->GetOutputDesc(0).GetShape().GetDims(), dim_result_nhw_c_to_NC1_hw_c0);
}

TEST_F(UTEST_fusion_engine_op_judge_precision_mode, set_two_nodes_format_dtype_force_fp16_1)
{
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = FORCE_FP16;
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g_op = std::make_shared<OpDesc>("BM", "BatchMatMul");

  //add descriptor
  vector<int64_t> dim1({32, 1, 2});
  vector<int64_t> dim2({32, 2, 2});
  vector<int64_t> dim_o({32, 1, 2});
  GeShape shape1(dim1);
  GeShape shape2(dim2);
  GeTensorDesc tensor_desc1(shape1);
  tensor_desc1.SetOriginFormat(FORMAT_NHWC);
  tensor_desc1.SetFormat(FORMAT_NHWC);
  tensor_desc1.SetDataType(DT_FLOAT);
  tensor_desc1.SetOriginDataType(DT_FLOAT);

  GeTensorDesc tensor_desc2(shape2);
  tensor_desc2.SetOriginFormat(FORMAT_NHWC);
  tensor_desc2.SetFormat(FORMAT_NHWC);
  tensor_desc2.SetDataType(DT_FLOAT);
  tensor_desc2.SetOriginDataType(DT_FLOAT);

  g_op->AddInputDesc("x1", tensor_desc2);
  g_op->AddInputDesc("x2", tensor_desc2);
  g_op->AddOutputDesc("y", tensor_desc2);
  ge::AttrUtils::SetInt(g_op, FE_IMPLY_TYPE, 6);
  ge::NodePtr g_node = graph->AddNode(g_op);


  Status ret1 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret1 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g_node, "tbe-custom");
  ASSERT_EQ(ret1, fe::SUCCESS);

  vector<int64_t> dim_result1({32, 1, 1, 16, 16});
  vector<int64_t> dim_result2({32, 1, 1, 16, 16});
  vector<int64_t> dim_result_o({32, 1, 1, 16, 16});
  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetInputDesc(0).GetFormat()), FORMAT_FRACTAL_NZ);
  EXPECT_EQ(g_op->GetInputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(g_op->GetInputDesc(0).GetShape().GetDims(), dim_result1);

  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetInputDesc(1).GetFormat()), FORMAT_FRACTAL_NZ);
  EXPECT_EQ(g_op->GetInputDesc(1).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(g_op->GetInputDesc(1).GetShape().GetDims(), dim_result2);

  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetOutputDesc(0).GetFormat()), FORMAT_FRACTAL_NZ);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetShape().GetDims(), dim_result_o);
}

/* Original format is consecutive */
TEST_F(UTEST_fusion_engine_op_judge_precision_mode, set_two_nodes_format_dtype_force_fp16_2)
{
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = FORCE_FP16;
  op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g_op = std::make_shared<OpDesc>("G1", "G");
  OpDescPtr h_op = std::make_shared<OpDesc>("G2", "G");

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
  tensor_desc_h.SetOriginFormat(FORMAT_NCHW);
  tensor_desc_h.SetFormat(FORMAT_NCHW);
  tensor_desc_h.SetDataType(DT_INT8);
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
  vector<int64_t> dim_result_nch_w_to_NC1_hw_c0_2({1, 1, 3, 4, 32});
  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetInputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetInputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(g_op->GetInputDesc(0).GetShape().GetDims(), dim_result_nch_w_to_NC1_hw_c0);

  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetShape().GetDims(), dim_result_nch_w_to_NC1_hw_c0);

  EXPECT_EQ(ge::GetPrimaryFormat(h_op->GetInputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(h_op->GetInputDesc(0).GetDataType(), DT_INT8);
  EXPECT_EQ(h_op->GetInputDesc(0).GetShape().GetDims(), dim_result_nch_w_to_NC1_hw_c0_2);

  EXPECT_EQ(ge::GetPrimaryFormat(h_op->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(h_op->GetOutputDesc(0).GetDataType(), DT_INT8);
  EXPECT_EQ(h_op->GetOutputDesc(0).GetShape().GetDims(), dim_result_nch_w_to_NC1_hw_c0_2);
}
/* Original format is consecutive */
TEST_F(UTEST_fusion_engine_op_judge_precision_mode, set_two_nodes_format_dtype_force_fp32_1)
{
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = FORCE_FP32;
  op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g_op = std::make_shared<OpDesc>("G1", "G");
  OpDescPtr h_op = std::make_shared<OpDesc>("G2", "G");

  //add descriptor
  vector<int64_t> dim({4, 33, 12, 16});
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NCHW);
  tensor_desc.SetFormat(FORMAT_NCHW);
  tensor_desc.SetDataType(DT_FLOAT16);
  g_op->AddInputDesc("x", tensor_desc);
  g_op->AddOutputDesc("z", tensor_desc);
  ge::AttrUtils::SetInt(g_op, FE_IMPLY_TYPE, 6);
  ge::NodePtr g_node = graph->AddNode(g_op);

  vector<int64_t> dim_h({1, 2, 3, 4});
  GeShape shape_h(dim_h);
  GeTensorDesc tensor_desc_h(shape_h);
  tensor_desc_h.SetOriginFormat(FORMAT_NCHW);
  tensor_desc_h.SetFormat(FORMAT_NCHW);
  tensor_desc_h.SetDataType(DT_INT8);
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
  vector<int64_t> dim_result_nch_w_to_NC1_hw_c0_2({1, 1, 3, 4, 32});
  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetInputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetInputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(g_op->GetInputDesc(0).GetShape().GetDims(), dim_result_nch_w_to_NC1_hw_c0);

  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetShape().GetDims(), dim_result_nch_w_to_NC1_hw_c0);

  EXPECT_EQ(ge::GetPrimaryFormat(h_op->GetInputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(h_op->GetInputDesc(0).GetDataType(), DT_INT8);
  EXPECT_EQ(h_op->GetInputDesc(0).GetShape().GetDims(), dim_result_nch_w_to_NC1_hw_c0_2);

  EXPECT_EQ(ge::GetPrimaryFormat(h_op->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(h_op->GetOutputDesc(0).GetDataType(), DT_INT8);
  EXPECT_EQ(h_op->GetOutputDesc(0).GetShape().GetDims(), dim_result_nch_w_to_NC1_hw_c0_2);
}
/* Original format is consecutive */
TEST_F(UTEST_fusion_engine_op_judge_precision_mode, set_two_nodes_format_dtype_force_fp32_2)
{
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = FORCE_FP32;
  op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g_op = std::make_shared<OpDesc>("G1", "G2");
  OpDescPtr h_op = std::make_shared<OpDesc>("G2", "G2");

  //add descriptor
  vector<int64_t> dim({4, 33, 12, 16});
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NCHW);
  tensor_desc.SetFormat(FORMAT_NCHW);
  tensor_desc.SetDataType(DT_FLOAT16);
  g_op->AddInputDesc("x", tensor_desc);
  g_op->AddOutputDesc("z", tensor_desc);
  ge::AttrUtils::SetInt(g_op, FE_IMPLY_TYPE, 6);
  ge::NodePtr g_node = graph->AddNode(g_op);

  vector<int64_t> dim_h({1, 2, 3, 4});
  GeShape shape_h(dim_h);
  GeTensorDesc tensor_desc_h(shape_h);
  tensor_desc_h.SetOriginFormat(FORMAT_NCHW);
  tensor_desc_h.SetFormat(FORMAT_NCHW);
  tensor_desc_h.SetDataType(DT_INT32);
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
  vector<int64_t> dim_result_nch_w_to_NC1_hw_c0_2({1, 1, 3, 4, 16});
  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetInputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetInputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(g_op->GetInputDesc(0).GetShape().GetDims(), dim_result_nch_w_to_NC1_hw_c0);

  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetShape().GetDims(), dim_result_nch_w_to_NC1_hw_c0);

  EXPECT_EQ(ge::GetPrimaryFormat(h_op->GetInputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(h_op->GetInputDesc(0).GetDataType(), DT_INT16);
  EXPECT_EQ(h_op->GetInputDesc(0).GetShape().GetDims(), dim_result_nch_w_to_NC1_hw_c0_2);

  EXPECT_EQ(ge::GetPrimaryFormat(h_op->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(h_op->GetOutputDesc(0).GetDataType(), DT_INT16);
  EXPECT_EQ(h_op->GetOutputDesc(0).GetShape().GetDims(), dim_result_nch_w_to_NC1_hw_c0_2);
}

TEST_F(UTEST_fusion_engine_op_judge_precision_mode, set_two_nodes_format_dtype_cube_fp16in_fp32out_1)
{
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = CUBE_FP16IN_FP32OUT;
  op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g_op = std::make_shared<OpDesc>("G1", "G");
  OpDescPtr h_op = std::make_shared<OpDesc>("G2", "G");

  //add descriptor
  vector<int64_t> dim({4, 33, 12, 16});
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NCHW);
  tensor_desc.SetFormat(FORMAT_NCHW);
  tensor_desc.SetDataType(DT_FLOAT16);
  g_op->AddInputDesc("x", tensor_desc);
  g_op->AddOutputDesc("z", tensor_desc);
  ge::AttrUtils::SetInt(g_op, FE_IMPLY_TYPE, 6);
  ge::NodePtr g_node = graph->AddNode(g_op);

  vector<int64_t> dim_h({1, 2, 3, 4});
  GeShape shape_h(dim_h);
  GeTensorDesc tensor_desc_h(shape_h);
  tensor_desc_h.SetOriginFormat(FORMAT_NCHW);
  tensor_desc_h.SetFormat(FORMAT_NCHW);
  tensor_desc_h.SetDataType(DT_INT8);
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
  vector<int64_t> dim_result_nch_w_to_NC1_hw_c0_2({1, 1, 3, 4, 32});
  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetInputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetInputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(g_op->GetInputDesc(0).GetShape().GetDims(), dim_result_nch_w_to_NC1_hw_c0);

  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetShape().GetDims(), dim_result_nch_w_to_NC1_hw_c0);

  EXPECT_EQ(ge::GetPrimaryFormat(h_op->GetInputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(h_op->GetInputDesc(0).GetDataType(), DT_INT8);
  EXPECT_EQ(h_op->GetInputDesc(0).GetShape().GetDims(), dim_result_nch_w_to_NC1_hw_c0_2);

  EXPECT_EQ(ge::GetPrimaryFormat(h_op->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(h_op->GetOutputDesc(0).GetDataType(), DT_INT8);
  EXPECT_EQ(h_op->GetOutputDesc(0).GetShape().GetDims(), dim_result_nch_w_to_NC1_hw_c0_2);
}

TEST_F(UTEST_fusion_engine_op_judge_precision_mode, set_two_nodes_format_dtype_cube_fp16in_fp32out_2)
{
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = CUBE_FP16IN_FP32OUT;
  op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g_op = std::make_shared<OpDesc>("G1", "G2");
  OpDescPtr h_op = std::make_shared<OpDesc>("G2", "G2");

  //add descriptor
  vector<int64_t> dim({4, 33, 12, 16});
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NCHW);
  tensor_desc.SetFormat(FORMAT_NCHW);
  tensor_desc.SetDataType(DT_FLOAT16);
  g_op->AddInputDesc("x", tensor_desc);
  g_op->AddOutputDesc("z", tensor_desc);
  ge::AttrUtils::SetInt(g_op, FE_IMPLY_TYPE, 6);
  ge::NodePtr g_node = graph->AddNode(g_op);

  vector<int64_t> dim_h({1, 2, 3, 4});
  GeShape shape_h(dim_h);
  GeTensorDesc tensor_desc_h(shape_h);
  tensor_desc_h.SetOriginFormat(FORMAT_NCHW);
  tensor_desc_h.SetFormat(FORMAT_NCHW);
  tensor_desc_h.SetDataType(DT_INT32);
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
  vector<int64_t> dim_result_nch_w_to_NC1_hw_c0_2({1, 1, 3, 4, 16});
  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetInputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetInputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(g_op->GetInputDesc(0).GetShape().GetDims(), dim_result_nch_w_to_NC1_hw_c0);

  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetShape().GetDims(), dim_result_nch_w_to_NC1_hw_c0);

  EXPECT_EQ(ge::GetPrimaryFormat(h_op->GetInputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(h_op->GetInputDesc(0).GetDataType(), DT_INT16);
  EXPECT_EQ(h_op->GetInputDesc(0).GetShape().GetDims(), dim_result_nch_w_to_NC1_hw_c0_2);

  EXPECT_EQ(ge::GetPrimaryFormat(h_op->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(h_op->GetOutputDesc(0).GetDataType(), DT_INT16);
  EXPECT_EQ(h_op->GetOutputDesc(0).GetShape().GetDims(), dim_result_nch_w_to_NC1_hw_c0_2);
}

TEST_F(UTEST_fusion_engine_op_judge_precision_mode, set_two_nodes_format_dtype_cube_fp16in_fp32out_3)
{
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = CUBE_FP16IN_FP32OUT;
  op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g_op = std::make_shared<OpDesc>("G1", "G2");
  OpDescPtr conv2d_op = std::make_shared<OpDesc>("conv2d", "Conv2D");

  //add descriptor
  vector<int64_t> dim({4, 33, 12, 16});
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NCHW);
  tensor_desc.SetFormat(FORMAT_NCHW);
  tensor_desc.SetDataType(DT_FLOAT16);
  g_op->AddInputDesc("x", tensor_desc);
  g_op->AddOutputDesc("z", tensor_desc);
  ge::AttrUtils::SetInt(g_op, FE_IMPLY_TYPE, 6);
  ge::NodePtr g_node = graph->AddNode(g_op);

  vector<int64_t> dim_h({1, 2, 3, 4});
  GeShape shape_h(dim_h);
  GeTensorDesc tensor_desc_in(shape_h);
  GeTensorDesc tensor_desc_out(shape_h);
  tensor_desc_in.SetOriginFormat(FORMAT_NCHW);
  tensor_desc_in.SetFormat(FORMAT_NCHW);
  tensor_desc_in.SetDataType(DT_FLOAT);
  tensor_desc_out.SetOriginFormat(FORMAT_NCHW);
  tensor_desc_out.SetFormat(FORMAT_NCHW);
  tensor_desc_out.SetDataType(DT_FLOAT16);

  conv2d_op->AddInputDesc("x", tensor_desc_in);
  conv2d_op->AddInputDesc("y", tensor_desc_in);
  conv2d_op->AddOutputDesc("out", tensor_desc_out);
  ge::AttrUtils::SetInt(conv2d_op, FE_IMPLY_TYPE, 6);
  ge::NodePtr conv2d_node = graph->AddNode(conv2d_op);
  GraphUtils::AddEdge(g_node->GetOutDataAnchor(0), conv2d_node->GetInDataAnchor(0));


  Status ret1 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret1 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g_node, "tbe-custom");
  Status ret2 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(conv2d_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret2 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(conv2d_node, "tbe-custom");
  ASSERT_EQ(ret1, fe::SUCCESS);
  ASSERT_EQ(ret2, fe::SUCCESS);
  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetInputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetInputDesc(0).GetDataType(), DT_FLOAT);

  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetDataType(), DT_FLOAT);

  EXPECT_EQ(ge::GetPrimaryFormat(conv2d_op->GetInputDesc(0).GetFormat()), FORMAT_NCHW);
  EXPECT_EQ(conv2d_op->GetInputDesc(0).GetDataType(), DT_FLOAT16);
   EXPECT_EQ(conv2d_op->GetInputDesc(1).GetDataType(), DT_FLOAT16);

  EXPECT_EQ(ge::GetPrimaryFormat(conv2d_op->GetOutputDesc(0).GetFormat()), FORMAT_NCHW);
  EXPECT_EQ(conv2d_op->GetOutputDesc(0).GetDataType(), DT_FLOAT);
}

TEST_F(UTEST_fusion_engine_op_judge_precision_mode, is_node_supported_16in_32out)
{
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = CUBE_FP16IN_FP32OUT;
  op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr_);
  OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-builtin", "G2");
  op_format_dtype_judge_ptr_->Initialize();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g_op = std::make_shared<OpDesc>("G1", "G2");

  //add descriptor
  vector<int64_t> dim({4, 33, 12, 16});
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NCHW);
  tensor_desc.SetFormat(FORMAT_NCHW);
  tensor_desc.SetDataType(DT_FLOAT16);
  g_op->AddInputDesc("x", tensor_desc);
  g_op->AddOutputDesc("y", tensor_desc);
  ge::AttrUtils::SetInt(g_op, FE_IMPLY_TYPE, 6);
  ge::NodePtr g_node = graph->AddNode(g_op);
  IndexNameMap input_index_map;
  IndexNameMap output_index_map;
  bool res = op_format_dtype_judge_ptr_->IsNodeSupport16In32out(g_node,  op_kernel_info_ptr, input_index_map, output_index_map);
  EXPECT_EQ(res, false);
}
TEST_F(UTEST_fusion_engine_op_judge_precision_mode, set_node_format_dtype_force_fp32)
{
  Configuration &config = Configuration::Instance(fe::AI_CORE_NAME);
  map<string, string> options;
  string soc_version = "Ascend310";
  PlatformUtils::Instance().soc_version_ = soc_version;
  config.Initialize(options);
  config.InitFp16OpType();
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = FORCE_FP32;
  op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr matmul = std::make_shared<OpDesc>("matmul", "MatMul");
  OpDescPtr relu = std::make_shared<OpDesc>("relu", "Relu");

  //add descriptor
  vector<int64_t> dim({4, 33, 12, 16});
  GeShape shape(dim);
  GeTensorDesc tensor_desc_in(shape);
  GeTensorDesc tensor_desc_out(shape);
  tensor_desc_in.SetOriginFormat(FORMAT_NCHW);
  tensor_desc_in.SetFormat(FORMAT_NCHW);
  tensor_desc_in.SetDataType(DT_FLOAT);
  tensor_desc_in.SetOriginDataType(DT_FLOAT);
  tensor_desc_out.SetOriginFormat(FORMAT_NCHW);
  tensor_desc_out.SetFormat(FORMAT_NCHW);
  tensor_desc_out.SetDataType(DT_FLOAT16);
  tensor_desc_out.SetOriginDataType(DT_FLOAT16);
  matmul->AddInputDesc("x", tensor_desc_in);
  matmul->AddOutputDesc("z", tensor_desc_out);
  ge::AttrUtils::SetInt(matmul, FE_IMPLY_TYPE, 6);
  ge::NodePtr matmul_node = graph->AddNode(matmul);

  vector<int64_t> dim_h({1, 2, 3, 4});
  GeShape shape_h(dim_h);
  GeTensorDesc tensor_desc_r_in(shape_h);
  GeTensorDesc tensor_desc_r_out(shape_h);
  tensor_desc_r_in.SetOriginFormat(FORMAT_NCHW);
  tensor_desc_r_in.SetFormat(FORMAT_NCHW);
  tensor_desc_r_in.SetDataType(DT_FLOAT16);
  tensor_desc_r_in.SetOriginDataType(DT_FLOAT16);
  tensor_desc_r_out.SetOriginFormat(FORMAT_NCHW);
  tensor_desc_r_out.SetFormat(FORMAT_NCHW);
  tensor_desc_r_out.SetOriginDataType(DT_FLOAT16);
  relu->AddInputDesc("x", tensor_desc_r_in);
  relu->AddOutputDesc("y", tensor_desc_r_out);
  ge::AttrUtils::SetInt(relu, FE_IMPLY_TYPE, 6);
  ge::NodePtr relu_node = graph->AddNode(relu);
  GraphUtils::AddEdge(matmul_node->GetOutDataAnchor(0), relu_node->GetInDataAnchor(0));


  Status ret1 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(matmul_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret1 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(matmul_node, "tbe-custom");
  Status ret2 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(relu_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret2 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(relu_node, "tbe-custom");
  ASSERT_EQ(ret1, fe::SUCCESS);
  ASSERT_EQ(ret2, fe::SUCCESS);
  EXPECT_EQ(matmul->GetInputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(matmul->GetOutputDesc(0).GetDataType(), DT_FLOAT);

  EXPECT_EQ(relu->GetInputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(relu->GetOutputDesc(0).GetDataType(), DT_FLOAT);
}

TEST_F(UTEST_fusion_engine_op_judge_precision_mode, set_two_nodes_format_auto_mixed_precision)
{
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = ALLOW_MIX_PRECISION;
  op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g_op = std::make_shared<OpDesc>("G1", "G");
  OpDescPtr h_op = std::make_shared<OpDesc>("G2", "G");

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
  tensor_desc_h.SetDataType(DT_INT8);
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

  EXPECT_EQ(ge::GetPrimaryFormat(h_op->GetInputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(h_op->GetInputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(h_op->GetInputDesc(0).GetShape().GetDims(), dim_result_nhw_c_to_NC1_hw_c0);

  EXPECT_EQ(ge::GetPrimaryFormat(h_op->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(h_op->GetOutputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(h_op->GetOutputDesc(0).GetShape().GetDims(), dim_result_nhw_c_to_NC1_hw_c0);
}

/* Original format is consecutive, Double can be the higher precision version
 * of float16 */
TEST_F(UTEST_fusion_engine_op_judge_precision_mode, set_two_nodes_format_dtype_auto_mixed_precision_2)
{
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = ALLOW_MIX_PRECISION;
  op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g_op = std::make_shared<OpDesc>("G1", "G");
  OpDescPtr h_op = std::make_shared<OpDesc>("G2", "GGray");

  //add descriptor
  vector<int64_t> dim({4, 33, 12, 16});
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NCHW);
  tensor_desc.SetFormat(FORMAT_NCHW);
  tensor_desc.SetDataType(DT_FLOAT16);
  g_op->AddInputDesc("x", tensor_desc);
  g_op->AddOutputDesc("z", tensor_desc);
  ge::AttrUtils::SetInt(g_op, FE_IMPLY_TYPE, 6);
  ge::NodePtr g_node = graph->AddNode(g_op);

  vector<int64_t> dim_h({1, 2, 3, 4});
  GeShape shape_h(dim_h);
  GeTensorDesc tensor_desc_h(shape_h);
  tensor_desc_h.SetOriginFormat(FORMAT_NCHW);
  tensor_desc_h.SetFormat(FORMAT_NCHW);
  tensor_desc_h.SetDataType(DT_INT32);
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
  vector<int64_t> dim_h_NHWC({1, 3, 4, 2});
  vector<int64_t> dim_h_NCHW({1, 2, 3, 4});
  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetInputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetInputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(g_op->GetInputDesc(0).GetShape().GetDims(), dim_result_nch_w_to_NC1_hw_c0);

  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetShape().GetDims(), dim_result_nch_w_to_NC1_hw_c0);

  EXPECT_EQ(h_op->GetInputDesc(0).GetFormat(), FORMAT_NCHW);
  EXPECT_EQ(h_op->GetInputDesc(0).GetDataType(), DT_INT8);
  EXPECT_EQ(h_op->GetInputDesc(0).GetShape().GetDims(), dim_h_NCHW);

  EXPECT_EQ(h_op->GetOutputDesc(0).GetFormat(), FORMAT_NHWC);
  EXPECT_EQ(h_op->GetOutputDesc(0).GetDataType(), DT_INT8);
  EXPECT_EQ(h_op->GetOutputDesc(0).GetShape().GetDims(), dim_h_NHWC);
}

/* G2 is in Gray list, we can decrease the precision for it. */
TEST_F(UTEST_fusion_engine_op_judge_precision_mode, set_two_nodes_format_dtype_auto_mixed_precision_3)
{
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = ALLOW_MIX_PRECISION;
  op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g_op = std::make_shared<OpDesc>("G1", "G");
  OpDescPtr h_op = std::make_shared<OpDesc>("G2", "GGray");

  //add descriptor
  vector<int64_t> dim({4, 33, 12, 16});
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NCHW);
  tensor_desc.SetFormat(FORMAT_NCHW);
  tensor_desc.SetDataType(DT_UINT8);
  g_op->AddInputDesc("x", tensor_desc);
  g_op->AddOutputDesc("z", tensor_desc);
  ge::AttrUtils::SetInt(g_op, FE_IMPLY_TYPE, 6);
  ge::NodePtr g_node = graph->AddNode(g_op);

  vector<int64_t> dim_h({1, 2, 3, 4});
  GeShape shape_h(dim_h);
  GeTensorDesc tensor_desc_h(shape_h);
  tensor_desc_h.SetOriginFormat(FORMAT_NCHW);
  tensor_desc_h.SetFormat(FORMAT_NCHW);
  tensor_desc_h.SetDataType(DT_INT32);
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

  vector<int64_t> dim_result_nch_w_to_NC1_hw_c0({4, 2, 12, 16, 32});
  vector<int64_t> dim_h_NHWC({1, 3, 4, 2});
  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetInputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetInputDesc(0).GetDataType(), DT_UINT8);
  EXPECT_EQ(g_op->GetInputDesc(0).GetShape().GetDims(), dim_result_nch_w_to_NC1_hw_c0);

  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetDataType(), DT_UINT8);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetShape().GetDims(), dim_result_nch_w_to_NC1_hw_c0);

  EXPECT_EQ(h_op->GetInputDesc(0).GetFormat(), FORMAT_NCHW);
  EXPECT_EQ(h_op->GetInputDesc(0).GetDataType(), DT_INT8);
  EXPECT_EQ(h_op->GetInputDesc(0).GetShape().GetDims(), dim_h);

  EXPECT_EQ(h_op->GetOutputDesc(0).GetFormat(), FORMAT_NHWC);
  EXPECT_EQ(h_op->GetOutputDesc(0).GetDataType(), DT_INT8);
  EXPECT_EQ(h_op->GetOutputDesc(0).GetShape().GetDims(), dim_h_NHWC);
}


/* G2 is in Black list, it must select the original dtype */
TEST_F(UTEST_fusion_engine_op_judge_precision_mode, set_two_nodes_format_dtype_auto_mixed_precision_4)
{
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = ALLOW_MIX_PRECISION;
  op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g_op = std::make_shared<OpDesc>("G1", "G");
  OpDescPtr h_op = std::make_shared<OpDesc>("G2", "GBlack");

  //add descriptor
  vector<int64_t> dim({4, 33, 12, 16});
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NCHW);
  tensor_desc.SetFormat(FORMAT_NCHW);
  tensor_desc.SetDataType(DT_DOUBLE);
  g_op->AddInputDesc("x", tensor_desc);
  g_op->AddOutputDesc("z", tensor_desc);
  ge::AttrUtils::SetInt(g_op, FE_IMPLY_TYPE, 6);
  ge::NodePtr g_node = graph->AddNode(g_op);

  vector<int64_t> dim_h({1, 2, 3, 4});
  GeShape shape_h(dim_h);
  GeTensorDesc tensor_desc_h(shape_h);
  tensor_desc_h.SetOriginFormat(FORMAT_NCHW);
  tensor_desc_h.SetFormat(FORMAT_NCHW);
  tensor_desc_h.SetDataType(DT_INT32);
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
  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetInputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetInputDesc(0).GetDataType(), DT_DOUBLE);
  EXPECT_EQ(g_op->GetInputDesc(0).GetShape().GetDims(), dim_result_nch_w_to_NC1_hw_c0);

  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetDataType(), DT_DOUBLE);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetShape().GetDims(), dim_result_nch_w_to_NC1_hw_c0);
  vector<int64_t> dim_h_5hd({1, 1, 3, 4, 32});
  EXPECT_EQ(ge::GetPrimaryFormat(h_op->GetInputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(h_op->GetInputDesc(0).GetDataType(), DT_UINT8);
  EXPECT_EQ(h_op->GetInputDesc(0).GetShape().GetDims(), dim_h_5hd);

  EXPECT_EQ(ge::GetPrimaryFormat(h_op->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(h_op->GetOutputDesc(0).GetDataType(), DT_UINT8);
  EXPECT_EQ(h_op->GetOutputDesc(0).GetShape().GetDims(), dim_h_5hd);
}

/* G2 is in the white list and we must select fp16 if the original data type is
 * fp32/fp16. And select the original dtype in default mode when dtype is not
 * fp32 or fp16. */
TEST_F(UTEST_fusion_engine_op_judge_precision_mode, set_two_nodes_format_dtype_auto_mixed_precision_5)
{
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = ALLOW_MIX_PRECISION;
  op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g_op = std::make_shared<OpDesc>("G1", "G");
  OpDescPtr h_op = std::make_shared<OpDesc>("G2", "GWhite");

  //add descriptor
  vector<int64_t> dim({4, 33, 12, 16});
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NCHW);
  tensor_desc.SetFormat(FORMAT_NCHW);
  tensor_desc.SetDataType(DT_DOUBLE);
  g_op->AddInputDesc("x", tensor_desc);
  g_op->AddOutputDesc("z", tensor_desc);
  ge::AttrUtils::SetInt(g_op, FE_IMPLY_TYPE, 6);
  ge::NodePtr g_node = graph->AddNode(g_op);

  vector<int64_t> dim_h({1, 2, 3, 4});
  GeShape shape_h(dim_h);
  GeTensorDesc tensor_desc_h(shape_h);
  tensor_desc_h.SetOriginFormat(FORMAT_NCHW);
  tensor_desc_h.SetFormat(FORMAT_NCHW);
  tensor_desc_h.SetDataType(DT_INT32);
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
  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetInputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetInputDesc(0).GetDataType(), DT_DOUBLE);
  EXPECT_EQ(g_op->GetInputDesc(0).GetShape().GetDims(), dim_result_nch_w_to_NC1_hw_c0);

  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetDataType(), DT_DOUBLE);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetShape().GetDims(), dim_result_nch_w_to_NC1_hw_c0);
  vector<int64_t> dim_h_5hd({1, 1, 3, 4, 32});
  EXPECT_EQ(ge::GetPrimaryFormat(h_op->GetInputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(h_op->GetInputDesc(0).GetDataType(), DT_UINT8);
  EXPECT_EQ(h_op->GetInputDesc(0).GetShape().GetDims(), dim_h_5hd);

  EXPECT_EQ(ge::GetPrimaryFormat(h_op->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(h_op->GetOutputDesc(0).GetDataType(), DT_UINT8);
  EXPECT_EQ(h_op->GetOutputDesc(0).GetShape().GetDims(), dim_h_5hd);
}

/* G2 is in the white list and we must select fp16 if the original data type is
 * fp32/fp16. And select the original dtype in default mode when dtype is not
 * fp32 or fp16. */
TEST_F(UTEST_fusion_engine_op_judge_precision_mode, set_two_nodes_format_dtype_auto_mixed_precision_6)
{
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = ALLOW_MIX_PRECISION;
  op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g_op = std::make_shared<OpDesc>("G1", "G");
  OpDescPtr h_op = std::make_shared<OpDesc>("G2", "GWhite");

  //add descriptor
  vector<int64_t> dim({4, 33, 12, 16});
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NCHW);
  tensor_desc.SetFormat(FORMAT_NCHW);
  tensor_desc.SetDataType(DT_DOUBLE);
  g_op->AddInputDesc("x", tensor_desc);
  g_op->AddOutputDesc("z", tensor_desc);
  ge::AttrUtils::SetInt(g_op, FE_IMPLY_TYPE, 6);
  ge::NodePtr g_node = graph->AddNode(g_op);

  vector<int64_t> dim_h({1, 2, 3, 4});
  GeShape shape_h(dim_h);
  GeTensorDesc tensor_desc_h(shape_h);
  tensor_desc_h.SetOriginFormat(FORMAT_NCHW);
  tensor_desc_h.SetFormat(FORMAT_NCHW);
  tensor_desc_h.SetDataType(DT_FLOAT);
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
  vector<int64_t> dim_result_nch_w_to_NC1_hw_c0_2({1, 1, 3, 4, 16});
  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetInputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetInputDesc(0).GetDataType(), DT_DOUBLE);
  EXPECT_EQ(g_op->GetInputDesc(0).GetShape().GetDims(), dim_result_nch_w_to_NC1_hw_c0);

  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetDataType(), DT_DOUBLE);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetShape().GetDims(), dim_result_nch_w_to_NC1_hw_c0);

  EXPECT_EQ(ge::GetPrimaryFormat(h_op->GetInputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(h_op->GetInputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(h_op->GetInputDesc(0).GetShape().GetDims(), dim_result_nch_w_to_NC1_hw_c0_2);

  EXPECT_EQ(ge::GetPrimaryFormat(h_op->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(h_op->GetOutputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(h_op->GetOutputDesc(0).GetShape().GetDims(), dim_result_nch_w_to_NC1_hw_c0_2);
}

/* G(fp16) -> Cast(fp16 to fp32) -> G, if Cast is not in Black list,
 * we will jump over cast and select fp16 for the second G op. */
TEST_F(UTEST_fusion_engine_op_judge_precision_mode, set_two_nodes_format_dtype_auto_mixed_precision_7)
{
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = ALLOW_MIX_PRECISION;
  op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g_op = std::make_shared<OpDesc>("G1", "G");
  OpDescPtr cast_op = std::make_shared<OpDesc>("Cast", "Cast");
  OpDescPtr h_op = std::make_shared<OpDesc>("G2", "GWhite");

  //add descriptor
  vector<int64_t> dim({4, 33, 12, 16});
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NCHW);
  tensor_desc.SetFormat(FORMAT_NCHW);
  tensor_desc.SetDataType(DT_FLOAT16);
  g_op->AddInputDesc("x", tensor_desc);
  g_op->AddOutputDesc("z", tensor_desc);
  ge::AttrUtils::SetInt(g_op, FE_IMPLY_TYPE, 6);
  ge::NodePtr g_node = graph->AddNode(g_op);

  vector<int64_t> dim_cast({4, 33, 12, 16});
  GeShape shape_cast(dim);
  GeTensorDesc tensor_desc_cast(shape_cast);
  tensor_desc_cast.SetOriginFormat(FORMAT_NCHW);
  tensor_desc_cast.SetFormat(FORMAT_NCHW);
  tensor_desc_cast.SetDataType(DT_FLOAT);
  cast_op->AddInputDesc("x", tensor_desc); // Float 16
  cast_op->AddOutputDesc("z", tensor_desc_cast); // Float 32
  ge::AttrUtils::SetInt(cast_op, FE_IMPLY_TYPE, 6);
  ge::NodePtr cast_node = graph->AddNode(cast_op);

  vector<int64_t> dim_h({1, 2, 3, 4});
  GeShape shape_h(dim_h);
  GeTensorDesc tensor_desc_h(shape_h);
  tensor_desc_h.SetOriginFormat(FORMAT_NCHW);
  tensor_desc_h.SetFormat(FORMAT_NCHW);
  tensor_desc_h.SetDataType(DT_FLOAT);
  h_op->AddInputDesc("x", tensor_desc_h);
  h_op->AddOutputDesc("z", tensor_desc_h);
  ge::AttrUtils::SetInt(h_op, FE_IMPLY_TYPE, 6);
  ge::NodePtr h_node = graph->AddNode(h_op);
  GraphUtils::AddEdge(g_node->GetOutDataAnchor(0), cast_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(cast_node->GetOutDataAnchor(0), h_node->GetInDataAnchor(0));


  Status ret1 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret1 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g_node, "tbe-custom");
  Status ret2 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(h_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret2 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(h_node, "tbe-custom");
  ASSERT_EQ(ret1, fe::SUCCESS);
  ASSERT_EQ(ret2, fe::SUCCESS);
  vector<int64_t> dim_result_nch_w_to_NC1_hw_c0({4, 3, 12, 16, 16});
  vector<int64_t> dim_result_nch_w_to_NC1_hw_c0_2({1, 1, 3, 4, 16});
  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetInputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetInputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(g_op->GetInputDesc(0).GetShape().GetDims(), dim_result_nch_w_to_NC1_hw_c0);

  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetShape().GetDims(), dim_result_nch_w_to_NC1_hw_c0);

  EXPECT_EQ(cast_op->GetInputDesc(0).GetFormat(), FORMAT_NCHW);
  EXPECT_EQ(cast_op->GetInputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(cast_op->GetInputDesc(0).GetShape().GetDims(), dim_cast);

  EXPECT_EQ(cast_op->GetOutputDesc(0).GetFormat(), FORMAT_NCHW);
  EXPECT_EQ(cast_op->GetOutputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(cast_op->GetOutputDesc(0).GetShape().GetDims(), dim_cast);

  EXPECT_EQ(ge::GetPrimaryFormat(h_op->GetInputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(h_op->GetInputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(h_op->GetInputDesc(0).GetShape().GetDims(), dim_result_nch_w_to_NC1_hw_c0_2);

  EXPECT_EQ(ge::GetPrimaryFormat(h_op->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(h_op->GetOutputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(h_op->GetOutputDesc(0).GetShape().GetDims(), dim_result_nch_w_to_NC1_hw_c0_2);
}

/* G(fp16) -> Cast(fp16 to fp32) -> G, if Cast is not in Black list,
 * we will jump over cast and select fp16 for the second G op. */
TEST_F(UTEST_fusion_engine_op_judge_precision_mode, set_two_nodes_format_dtype_auto_mixed_precision_8)
{
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = ALLOW_MIX_PRECISION;
  op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();
  /* Stub Cast in Black list */
  SubOpInfoStorePtr subOpInfoStorePtr = OpsKernelManager::Instance(AI_CORE_NAME).GetSubOpsKernelByStoreName("tbe-custom");
  OpKernelInfoPtr opKernelInfoPtr = subOpInfoStorePtr->GetOpKernelByOpType(fe::CAST);
  opKernelInfoPtr->op_param_vec_[static_cast<size_t>(OP_KERNEL_PARAM::PrecisionPolicy)] = static_cast<int64_t>(BLACK);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g_op = std::make_shared<OpDesc>("G1", "G");
  OpDescPtr cast_op = std::make_shared<OpDesc>("Cast", fe::CAST);
  OpDescPtr h_op = std::make_shared<OpDesc>("G2", "GGray");

  //add descriptor
  vector<int64_t> dim({4, 33, 12, 16});
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NCHW);
  tensor_desc.SetFormat(FORMAT_NCHW);
  tensor_desc.SetDataType(DT_FLOAT16);
  g_op->AddInputDesc("x", tensor_desc);
  g_op->AddOutputDesc("z", tensor_desc);
  ge::AttrUtils::SetInt(g_op, FE_IMPLY_TYPE, 6);
  ge::NodePtr g_node = graph->AddNode(g_op);

  vector<int64_t> dim_cast({4, 33, 12, 16});
  GeShape shape_cast(dim);
  GeTensorDesc tensor_desc_cast(shape_cast);
  tensor_desc_cast.SetOriginFormat(FORMAT_NCHW);
  tensor_desc_cast.SetFormat(FORMAT_NCHW);
  tensor_desc_cast.SetDataType(DT_FLOAT);
  cast_op->AddInputDesc("x", tensor_desc);
  cast_op->AddOutputDesc("z", tensor_desc_cast);
  ge::AttrUtils::SetInt(cast_op, FE_IMPLY_TYPE, 6);
  ge::NodePtr cast_node = graph->AddNode(cast_op);

  vector<int64_t> dim_h({1, 2, 3, 4});
  GeShape shape_h(dim_h);
  GeTensorDesc tensor_desc_h(shape_h);
  tensor_desc_h.SetOriginFormat(FORMAT_NCHW);
  tensor_desc_h.SetFormat(FORMAT_NCHW);
  tensor_desc_h.SetDataType(DT_FLOAT);
  h_op->AddInputDesc("x", tensor_desc_h);
  h_op->AddOutputDesc("z", tensor_desc_h);
  ge::AttrUtils::SetInt(h_op, FE_IMPLY_TYPE, 6);
  ge::NodePtr h_node = graph->AddNode(h_op);
  GraphUtils::AddEdge(g_node->GetOutDataAnchor(0), cast_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(cast_node->GetOutDataAnchor(0), h_node->GetInDataAnchor(0));



  Status ret1 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret1 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g_node, "tbe-custom");
  Status ret2 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(h_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret2 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(h_node, "tbe-custom");
  ASSERT_EQ(ret1, fe::SUCCESS);
  ASSERT_EQ(ret2, fe::SUCCESS);
  vector<int64_t> dim_result_nch_w_to_NC1_hw_c0({4, 3, 12, 16, 16});
  vector<int64_t> dim_h_NHWC({1, 3, 4, 2});
  vector<int64_t> dim_h_NCHW({1, 2, 3, 4});
  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetInputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetInputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(g_op->GetInputDesc(0).GetShape().GetDims(), dim_result_nch_w_to_NC1_hw_c0);

  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetShape().GetDims(), dim_result_nch_w_to_NC1_hw_c0);

  EXPECT_EQ(cast_op->GetInputDesc(0).GetFormat(), FORMAT_NCHW);
  EXPECT_EQ(cast_op->GetInputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(cast_op->GetInputDesc(0).GetShape().GetDims(), dim_cast);

  EXPECT_EQ(cast_op->GetOutputDesc(0).GetFormat(), FORMAT_NCHW);
  EXPECT_EQ(cast_op->GetOutputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(cast_op->GetOutputDesc(0).GetShape().GetDims(), dim_cast);

  EXPECT_EQ(h_op->GetInputDesc(0).GetFormat(), FORMAT_NCHW);
  EXPECT_EQ(h_op->GetInputDesc(0).GetDataType(), DT_INT8);
  EXPECT_EQ(h_op->GetInputDesc(0).GetShape().GetDims(), dim_h_NCHW);

  EXPECT_EQ(h_op->GetOutputDesc(0).GetFormat(), FORMAT_NHWC);
  EXPECT_EQ(h_op->GetOutputDesc(0).GetDataType(), DT_INT8);
  EXPECT_EQ(h_op->GetOutputDesc(0).GetShape().GetDims(), dim_h_NHWC);
  /* restore Cast back to Gray list */
  opKernelInfoPtr->op_param_vec_[static_cast<size_t>(OP_KERNEL_PARAM::PrecisionPolicy)] = static_cast<int64_t>(BLACK);
}

/* Op GBlackOnlyFp16 is in blacklist but it does not support fp32 and it's original data type is fp32,
 * we return failed and tell the user this op should not be configured in black list. */
TEST_F(UTEST_fusion_engine_op_judge_precision_mode, set_two_nodes_format_dtype_auto_mixed_precision_9)
{
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = ALLOW_MIX_PRECISION;
  op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g_op = std::make_shared<OpDesc>("G", "GBlackOnlyFp16");

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


  vector<int64_t> dim_5hd{ 4, 2, 12, 16, 32 };
  Status ret1 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret1 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g_node, "tbe-custom");
  ASSERT_EQ(ret1, fe::SUCCESS);
  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetInputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetInputDesc(0).GetDataType(), DT_UINT8);
  EXPECT_EQ(g_op->GetInputDesc(0).GetShape().GetDims(), dim_5hd);

  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetDataType(), DT_UINT8);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetShape().GetDims(), dim_5hd);
}

TEST_F(UTEST_fusion_engine_op_judge_precision_mode, set_two_nodes_format_auto_mixed_precision_new)
{
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = FORCE_FP16;
  op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g_op = std::make_shared<OpDesc>("G1", "G");
  OpDescPtr h_op = std::make_shared<OpDesc>("G2", "G");

  //add descriptor
  vector<int64_t> dim({4, 33, 12, 16});
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NCHW);
  tensor_desc.SetFormat(FORMAT_NCHW);
  tensor_desc.SetDataType(DT_FLOAT);
  ge::AttrUtils::SetInt(tensor_desc, FORMAT_CONTINUOUS, 1);
  g_op->AddInputDesc("x", tensor_desc);
  g_op->AddOutputDesc("z", tensor_desc);
  ge::AttrUtils::SetInt(g_op, FE_IMPLY_TYPE, 6);
  ge::NodePtr g_node = graph->AddNode(g_op);

  vector<int64_t> dim_h({1, 2, 3, 4});
  GeShape shape_h(dim_h);
  GeTensorDesc tensor_desc_h(shape_h);
  tensor_desc_h.SetOriginFormat(FORMAT_NHWC);
  tensor_desc_h.SetFormat(FORMAT_NHWC);
  tensor_desc_h.SetDataType(DT_INT8);
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
  vector<int64_t> dim_result_nhw_c_to_NC1_hw_c0({1, 1, 2, 3, 32});
  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetInputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetInputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(g_op->GetInputDesc(0).GetShape().GetDims(), dim_result_nch_w_to_NC1_hw_c0);

  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetShape().GetDims(), dim_result_nch_w_to_NC1_hw_c0);

  EXPECT_EQ(ge::GetPrimaryFormat(h_op->GetInputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(h_op->GetInputDesc(0).GetDataType(), DT_INT8);
  EXPECT_EQ(h_op->GetInputDesc(0).GetShape().GetDims(), dim_result_nhw_c_to_NC1_hw_c0);

  EXPECT_EQ(ge::GetPrimaryFormat(h_op->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(h_op->GetOutputDesc(0).GetDataType(), DT_INT8);
  EXPECT_EQ(h_op->GetOutputDesc(0).GetShape().GetDims(), dim_result_nhw_c_to_NC1_hw_c0);
}

TEST_F(UTEST_fusion_engine_op_judge_precision_mode, set_two_nodes_format_dtype_default_mode)
{
  Configuration &config = Configuration::Instance(fe::AI_CORE_NAME);
  config.enable_aclnn_ = true;
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = MUST_KEEP_ORIGIN_DTYPE;
  op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g_op = std::make_shared<OpDesc>("G1", "G");
  OpDescPtr h_op = std::make_shared<OpDesc>("G2", "G");

  //add descriptor
  vector<int64_t> dim({4, 33, 12, 16});
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NCHW);
  tensor_desc.SetFormat(FORMAT_NCHW);
  tensor_desc.SetDataType(DT_FLOAT16);
  g_op->AddInputDesc("x", tensor_desc);
  g_op->AddOutputDesc("z", tensor_desc);
  ge::AttrUtils::SetInt(g_op, FE_IMPLY_TYPE, 6);
  ge::NodePtr g_node = graph->AddNode(g_op);

  vector<int64_t> dim_h({1, 2, 3, 4});
  GeShape shape_h(dim_h);
  GeTensorDesc tensor_desc_h(shape_h);
  tensor_desc_h.SetOriginFormat(FORMAT_NHWC);
  tensor_desc_h.SetFormat(FORMAT_NHWC);
  tensor_desc_h.SetDataType(DT_INT32);
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
  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetInputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetInputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(g_op->GetInputDesc(0).GetShape().GetDims(), dim_result_nch_w_to_NC1_hw_c0);

  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetShape().GetDims(), dim_result_nch_w_to_NC1_hw_c0);
  vector<int64_t> dim_h_5hd({ 1, 1, 2, 3, 32});
  EXPECT_EQ(ge::GetPrimaryFormat(h_op->GetInputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(h_op->GetInputDesc(0).GetDataType(), DT_UINT8);
  EXPECT_EQ(h_op->GetInputDesc(0).GetShape().GetDims(), dim_h_5hd);

  EXPECT_EQ(ge::GetPrimaryFormat(h_op->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(h_op->GetOutputDesc(0).GetDataType(), DT_UINT8);
  EXPECT_EQ(h_op->GetOutputDesc(0).GetShape().GetDims(), dim_h_5hd);
  config.enable_aclnn_ = false;
}

/* Original format is consecutive */
TEST_F(UTEST_fusion_engine_op_judge_precision_mode, set_two_nodes_format_dtype_default_mode_2)
{
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = MUST_KEEP_ORIGIN_DTYPE;
  op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g_op = std::make_shared<OpDesc>("G1", "G");
  OpDescPtr h_op = std::make_shared<OpDesc>("G2", "G");

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
  tensor_desc_h.SetOriginFormat(FORMAT_NCHW);
  tensor_desc_h.SetFormat(FORMAT_NCHW);
  tensor_desc_h.SetDataType(DT_INT8);
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
  vector<int64_t> dim_result_nch_w_to_NC1_hw_c0_2({1, 1, 3, 4, 32});
  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetInputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetInputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(g_op->GetInputDesc(0).GetShape().GetDims(), dim_result_nch_w_to_NC1_hw_c0);

  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetShape().GetDims(), dim_result_nch_w_to_NC1_hw_c0);

  EXPECT_EQ(ge::GetPrimaryFormat(h_op->GetInputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(h_op->GetInputDesc(0).GetDataType(), DT_INT8);
  EXPECT_EQ(h_op->GetInputDesc(0).GetShape().GetDims(), dim_result_nch_w_to_NC1_hw_c0_2);

  EXPECT_EQ(ge::GetPrimaryFormat(h_op->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(h_op->GetOutputDesc(0).GetDataType(), DT_INT8);
  EXPECT_EQ(h_op->GetOutputDesc(0).GetShape().GetDims(), dim_result_nch_w_to_NC1_hw_c0_2);
}

/* When precisoin mode is must_keep_origin_dtype, we do
 * not allow precision reduce in CheckSupported. */
TEST_F(UTEST_fusion_engine_op_judge_precision_mode, check_support_with_must_keep_orignal_dtype)
{
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = MUST_KEEP_ORIGIN_DTYPE;
  op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g_op = std::make_shared<OpDesc>("G1", "GBlackOnlyFp16");

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

  string reason;
  bool result = fe_ops_kernel_info_store_ptr_->CheckSupported(g_node->GetOpDesc(),  reason);
  ASSERT_EQ(result, false);
  EXPECT_EQ(g_op->GetInputDesc(0).GetFormat(), FORMAT_NCHW);
  EXPECT_EQ(g_op->GetInputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(g_op->GetInputDesc(0).GetShape().GetDims(), dim);

  EXPECT_EQ(g_op->GetOutputDesc(0).GetFormat(), FORMAT_NCHW);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetShape().GetDims(), dim);
}

TEST_F(UTEST_fusion_engine_op_judge_precision_mode, check_support_with_must_keep_orignal_dtype_v2)
{
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = V2_ORIGIN;
  op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g_op = std::make_shared<OpDesc>("G1", "GBlackOnlyFp16");

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

  string reason;
  bool result = fe_ops_kernel_info_store_ptr_->CheckSupported(g_node->GetOpDesc(),  reason);
  ASSERT_EQ(result, false);
  EXPECT_EQ(g_op->GetInputDesc(0).GetFormat(), FORMAT_NCHW);
  EXPECT_EQ(g_op->GetInputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(g_op->GetInputDesc(0).GetShape().GetDims(), dim);

  EXPECT_EQ(g_op->GetOutputDesc(0).GetFormat(), FORMAT_NCHW);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetShape().GetDims(), dim);
}

TEST_F(UTEST_fusion_engine_op_judge_precision_mode, set_two_nodes_format_dtype_allow_fp32_to_fp16)
{
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = ALLOW_FP32_TO_FP16;
  op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g_op = std::make_shared<OpDesc>("G1", "G");
  OpDescPtr h_op = std::make_shared<OpDesc>("G2", "G");

  //add descriptor
  vector<int64_t> dim({4, 33, 12, 16});
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NCHW);
  tensor_desc.SetFormat(FORMAT_NCHW);
  tensor_desc.SetDataType(DT_DOUBLE);
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
  vector<int64_t> dim_result_nhw_c_to_NC1_hw_c0({1, 1, 2, 3, 16});
  vector<int64_t> dim_result_nch_w_to_NC1_hw_c0({4, 3, 12, 16, 16});
  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetInputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetInputDesc(0).GetDataType(), DT_DOUBLE);
  EXPECT_EQ(g_op->GetInputDesc(0).GetShape().GetDims(), dim_result_nch_w_to_NC1_hw_c0);

  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetDataType(), DT_DOUBLE);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetShape().GetDims(), dim_result_nch_w_to_NC1_hw_c0);

  EXPECT_EQ(ge::GetPrimaryFormat(h_op->GetInputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(h_op->GetInputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(h_op->GetInputDesc(0).GetShape().GetDims(), dim_result_nhw_c_to_NC1_hw_c0);

  EXPECT_EQ(ge::GetPrimaryFormat(h_op->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(h_op->GetOutputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(h_op->GetOutputDesc(0).GetShape().GetDims(), dim_result_nhw_c_to_NC1_hw_c0);
}

/* Original format is consecutive */
TEST_F(UTEST_fusion_engine_op_judge_precision_mode, set_two_nodes_format_dtype_allow_fp32_to_fp16_2)
{
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = ALLOW_FP32_TO_FP16;
  op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g_op = std::make_shared<OpDesc>("conv1", "ConvTemp");
  OpDescPtr h_op = std::make_shared<OpDesc>("conv2", "ConvTemp");

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
  tensor_desc_h.SetOriginFormat(FORMAT_NCHW);
  tensor_desc_h.SetFormat(FORMAT_NCHW);
  tensor_desc_h.SetDataType(DT_INT8);
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
  vector<int64_t> dim_result_nch_w_to_NC1_hw_c0_2({1, 1, 3, 4, 32});
  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetInputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetInputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(g_op->GetInputDesc(0).GetShape().GetDims(), dim_result_nch_w_to_NC1_hw_c0);

  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetShape().GetDims(), dim_result_nch_w_to_NC1_hw_c0);

  EXPECT_EQ(ge::GetPrimaryFormat(h_op->GetInputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(h_op->GetInputDesc(0).GetDataType(), DT_INT8);
  EXPECT_EQ(h_op->GetInputDesc(0).GetShape().GetDims(), dim_result_nch_w_to_NC1_hw_c0_2);

  EXPECT_EQ(ge::GetPrimaryFormat(h_op->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(h_op->GetOutputDesc(0).GetDataType(), DT_INT8);
  EXPECT_EQ(h_op->GetOutputDesc(0).GetShape().GetDims(), dim_result_nch_w_to_NC1_hw_c0_2);
}


/* Original format is consecutive,
 * and the second op is GE op, it use mixed precision as default strategy. */
TEST_F(UTEST_fusion_engine_op_judge_precision_mode,
       set_two_nodes_format_dtype_allow_fp32_to_fp16_3)
{
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = ALLOW_FP32_TO_FP16;
  op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g_op = std::make_shared<OpDesc>("conv1", "ConvTemp");
  OpDescPtr h_op = std::make_shared<OpDesc>("conv2", "ConvTemp_Ge");

  //add descriptor
  vector<int64_t> dim({4, 33, 12, 16});
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NCHW);
  tensor_desc.SetFormat(FORMAT_NCHW);
  tensor_desc.SetDataType(DT_FLOAT16);
  g_op->AddInputDesc("x", tensor_desc);
  g_op->AddOutputDesc("z", tensor_desc);
  ge::AttrUtils::SetInt(g_op, FE_IMPLY_TYPE, 6);
  ge::NodePtr g_node = graph->AddNode(g_op);

  vector<int64_t> dim_h({1, 2, 3, 4});
  GeShape shape_h(dim_h);
  GeTensorDesc tensor_desc_h(shape_h);
  tensor_desc_h.SetOriginFormat(FORMAT_NCHW);
  tensor_desc_h.SetFormat(FORMAT_NCHW);
  tensor_desc_h.SetDataType(DT_INT8);
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
  vector<int64_t> dim_result_nch_w_to_NC1_hw_c0_2({1, 1, 3, 4, 16});
  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetInputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetInputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(g_op->GetInputDesc(0).GetShape().GetDims(), dim_result_nch_w_to_NC1_hw_c0);

  EXPECT_EQ(ge::GetPrimaryFormat(g_op->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetShape().GetDims(), dim_result_nch_w_to_NC1_hw_c0);

  EXPECT_EQ(ge::GetPrimaryFormat(h_op->GetInputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(h_op->GetInputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(h_op->GetInputDesc(0).GetShape().GetDims(), dim_result_nch_w_to_NC1_hw_c0_2);

  EXPECT_EQ(ge::GetPrimaryFormat(h_op->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(h_op->GetOutputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(h_op->GetOutputDesc(0).GetShape().GetDims(), dim_result_nch_w_to_NC1_hw_c0_2);
}

/* For pytorch, Data's format is FRACTAL_NZ. */
TEST_F(UTEST_fusion_engine_op_judge_precision_mode, pytorch_set_two_nodes_format_dtype_allow_fp32_to_fp16_2)
{
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = ALLOW_FP32_TO_FP16;
  op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr data_op = std::make_shared<OpDesc>("Data", fe::DATA);
  OpDescPtr h_op = std::make_shared<OpDesc>("bm", "BatchMatMul2");

  //add descriptor
  vector<int64_t> dim({4, 33, 2, 3, 16, 16});
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NCHW);
  tensor_desc.SetFormat(FORMAT_FRACTAL_NZ);
  tensor_desc.SetDataType(DT_FLOAT);
  data_op->AddInputDesc("x", tensor_desc);
  data_op->AddOutputDesc("z", tensor_desc);
  ge::AttrUtils::SetInt(data_op, FE_IMPLY_TYPE, 6);
  ge::NodePtr data_node = graph->AddNode(data_op);

  vector<int64_t> dim_h({4, 33, 48, 32});
  GeShape shape_h(dim_h);
  GeTensorDesc tensor_desc_h(shape_h);
  tensor_desc_h.SetOriginFormat(FORMAT_NCHW);
  tensor_desc_h.SetFormat(FORMAT_NCHW);
  tensor_desc_h.SetDataType(DT_FLOAT);
  h_op->AddInputDesc("x", tensor_desc_h);
  h_op->AddOutputDesc("z", tensor_desc_h);
  ge::AttrUtils::SetInt(h_op, FE_IMPLY_TYPE, 6);
  ge::NodePtr h_node = graph->AddNode(h_op);
  GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), h_node->GetInDataAnchor(0));

  Status ret2 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(h_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret2 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(h_node, "tbe-custom");
  ASSERT_EQ(ret2, fe::SUCCESS);
  EXPECT_EQ(ge::GetPrimaryFormat(data_op->GetInputDesc(0).GetFormat()), FORMAT_FRACTAL_NZ);
  EXPECT_EQ(data_op->GetInputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(data_op->GetInputDesc(0).GetShape().GetDims(), dim);

  EXPECT_EQ(ge::GetPrimaryFormat(data_op->GetOutputDesc(0).GetFormat()), FORMAT_FRACTAL_NZ);
  EXPECT_EQ(data_op->GetOutputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(data_op->GetOutputDesc(0).GetShape().GetDims(), dim);

  EXPECT_EQ(ge::GetPrimaryFormat(h_op->GetInputDesc(0).GetFormat()), FORMAT_FRACTAL_NZ);
  EXPECT_EQ(h_op->GetInputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(h_op->GetInputDesc(0).GetShape().GetDims(), dim);

  EXPECT_EQ(ge::GetPrimaryFormat(h_op->GetOutputDesc(0).GetFormat()), FORMAT_FRACTAL_NZ);
  EXPECT_EQ(h_op->GetOutputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(h_op->GetOutputDesc(0).GetShape().GetDims(), dim);
}

//TEST_F(UTEST_fusion_engine_op_judge_precision_mode, set_two_nodes_format_dtype_unknown_shape_c)
//{
//  map<ge::Format, vector<int64_t>> format_shape_map = {
//    {FORMAT_NCHW, {1, -1, 3, 4}},
//    {FORMAT_NHWC, {1, 2, 3, -1}},
//    {FORMAT_HWCN, {1, 2, -1, 4}},
//    {FORMAT_CHWN, {-1, 2, 3, 4}},
//    {FORMAT_NDHWC, {1, 2, 3, 4, -1}},
//    {FORMAT_DHWCN, {1, 2, 3, -1, 5}},
//    {FORMAT_DHWNC, {1, -1, 3, 4, -1}},
//  };
//  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = ALLOW_FP32_TO_FP16;
//  for (auto format_shape : format_shape_map) {
//    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
//    GeShape shape_h(format_shape.second);
//    CreateUnknownShapeGraph(graph, FORMAT_NC1HWC0, format_shape.first, shape_h);
//
//    for (ge::NodePtr node : graph->GetDirectNode()) {
//      if (node->GetType() != "UnknownShape") {
//        continue;
//      }
//
//      Status ret2 = op_format_dtype_judge_ptr_->SetDtypeAndFormatByPrecisionMode(node,  "tbe-custom");
//      ASSERT_EQ(ret2, fe::SUCCESS);
//      OpDescPtr h_op = node->GetOpDesc();
//      vector<int64_t> dim_result_nhw_c_to_NC1_hw_c0({1, 1, 2, 3, 16});
//      vector<int64_t> dim_result_nch_w_to_NC1_hw_c0({4, 3, 12, 16, 16});
//
//      EXPECT_EQ(h_op->GetInputDesc(0).GetFormat(), format_shape.first);
//      EXPECT_EQ(h_op->GetInputDesc(0).GetDataType(), DT_FLOAT16);
//      EXPECT_EQ(h_op->GetInputDesc(0).GetShape().GetDims(), shape_h.GetDims());
//
//      EXPECT_EQ(h_op->GetOutputDesc(0).GetFormat(), format_shape.first);
//      EXPECT_EQ(h_op->GetOutputDesc(0).GetDataType(), DT_FLOAT16);
//      EXPECT_EQ(h_op->GetOutputDesc(0).GetShape().GetDims(), shape_h.GetDims());
//    }
//  }
//}

TEST_F(UTEST_fusion_engine_op_judge_precision_mode, set_two_nodes_format_dtype_unknown_shape_last_two)
{
  map<ge::Format, vector<int64_t>> format_shape_map = {
          {FORMAT_NCHW, {1, 2, -1, 4}},
          {FORMAT_NHWC, {1, 2, 3, -1}},
          {FORMAT_HWCN, {1, 2, -1, 4}},
          {FORMAT_CHWN, {1, 2, 3, -1}},
          {FORMAT_NDHWC, {1, 2, 3, 4, -1}},
          {FORMAT_DHWCN, {1, 2, 3, -1, 5}},
          {FORMAT_DHWNC, {1, -1, 3, 4, -1}},
  };
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = ALLOW_FP32_TO_FP16;
  op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();
  for (auto format_shape : format_shape_map) {
    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
    GeShape shape_h(format_shape.second);
    CreateUnknownShapeGraph(graph, FORMAT_FRACTAL_NZ, format_shape.first, shape_h);

    for (ge::NodePtr node : graph->GetDirectNode()) {
      if (node->GetType() != "UnknownShape") {
        continue;
      }

      Status ret2 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
      ret2 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(node, "tbe-custom");
      ASSERT_EQ(ret2, fe::SUCCESS);
      OpDescPtr h_op = node->GetOpDesc();
      vector<int64_t> dim_result_nhw_c_to_NC1_hw_c0({1, 1, 2, 3, 16});
      vector<int64_t> dim_result_nch_w_to_NC1_hw_c0({4, 3, 12, 16, 16});

      EXPECT_EQ(h_op->GetInputDesc(0).GetFormat(), format_shape.first);
      EXPECT_EQ(h_op->GetInputDesc(0).GetDataType(), DT_FLOAT16);
      EXPECT_EQ(h_op->GetInputDesc(0).GetShape().GetDims(), shape_h.GetDims());

      EXPECT_EQ(h_op->GetOutputDesc(0).GetFormat(), format_shape.first);
      EXPECT_EQ(h_op->GetOutputDesc(0).GetDataType(), DT_FLOAT16);
      EXPECT_EQ(h_op->GetOutputDesc(0).GetShape().GetDims(), shape_h.GetDims());
    }
  }
}


/* Original format is consecutive */
TEST_F(UTEST_fusion_engine_op_judge_precision_mode, keep_dtype_01)
{
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = FORCE_FP16;
  op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g1_op = std::make_shared<OpDesc>("G1", "G");
  OpDescPtr g2_op = std::make_shared<OpDesc>("G2", "G");

  //add descriptor
  vector<int64_t> dim({4, 33, 12, 16});
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NCHW);
  tensor_desc.SetFormat(FORMAT_NCHW);
  tensor_desc.SetDataType(DT_FLOAT);
  g1_op->AddInputDesc("x", tensor_desc);
  g1_op->AddOutputDesc("z", tensor_desc);
  ge::AttrUtils::SetInt(g1_op, FE_IMPLY_TYPE, 6);
  ge::AttrUtils::SetInt(g1_op, KEEP_DTYPE, 1);
  ge::NodePtr g_node = graph->AddNode(g1_op);


  vector<int64_t> dim_h({1, 2, 3, 4});
  GeShape shape_h(dim_h);
  GeTensorDesc tensor_desc_h(shape_h);
  tensor_desc_h.SetOriginFormat(FORMAT_NCHW);
  tensor_desc_h.SetFormat(FORMAT_NCHW);
  tensor_desc_h.SetDataType(DT_FLOAT);
  g2_op->AddInputDesc("x", tensor_desc_h);
  g2_op->AddOutputDesc("z", tensor_desc_h);
  ge::AttrUtils::SetInt(g2_op, FE_IMPLY_TYPE, 6);
  ge::NodePtr h_node = graph->AddNode(g2_op);
  GraphUtils::AddEdge(g_node->GetOutDataAnchor(0), h_node->GetInDataAnchor(0));


  Status ret1 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret1 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g_node, "tbe-custom");
  Status ret2 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(h_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret2 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(h_node, "tbe-custom");
  ASSERT_EQ(ret1, fe::SUCCESS);
  ASSERT_EQ(ret2, fe::SUCCESS);
  vector<int64_t> dim_result_nch_w_to_NC1_hw_c0({4, 3, 12, 16, 16});
  vector<int64_t> dim_result_nch_w_to_NC1_hw_c0_2({1, 1, 3, 4, 16});
  EXPECT_EQ(ge::GetPrimaryFormat(g1_op->GetInputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g1_op->GetInputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(g1_op->GetInputDesc(0).GetShape().GetDims(), dim_result_nch_w_to_NC1_hw_c0);

  EXPECT_EQ(ge::GetPrimaryFormat(g1_op->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g1_op->GetOutputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(g1_op->GetOutputDesc(0).GetShape().GetDims(), dim_result_nch_w_to_NC1_hw_c0);

  EXPECT_EQ(ge::GetPrimaryFormat(g2_op->GetInputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g2_op->GetInputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(g2_op->GetInputDesc(0).GetShape().GetDims(), dim_result_nch_w_to_NC1_hw_c0_2);

  EXPECT_EQ(ge::GetPrimaryFormat(g2_op->GetOutputDesc(0).GetFormat()), FORMAT_NC1HWC0);
  EXPECT_EQ(g2_op->GetOutputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(g2_op->GetOutputDesc(0).GetShape().GetDims(), dim_result_nch_w_to_NC1_hw_c0_2);
}

TEST_F(UTEST_fusion_engine_op_judge_precision_mode, set_cast_nodes_format_dtype_force_fp16)
{
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = FORCE_FP16;
  op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g_op = std::make_shared<OpDesc>("Cast1", CAST);

  //add descriptor
  vector<int64_t> dim({4, 33, 12, 16});
  GeShape shape(dim);
  GeTensorDesc float_tensor_desc(shape);
  float_tensor_desc.SetOriginFormat(FORMAT_NCHW);
  float_tensor_desc.SetFormat(FORMAT_NCHW);
  float_tensor_desc.SetDataType(DT_FLOAT);
  g_op->AddInputDesc("x", float_tensor_desc);

  GeTensorDesc int16_tensor_desc(shape);
  int16_tensor_desc.SetOriginFormat(FORMAT_NCHW);
  int16_tensor_desc.SetFormat(FORMAT_NCHW);
  int16_tensor_desc.SetDataType(DT_INT16);

  g_op->AddOutputDesc("z", int16_tensor_desc);
  ge::AttrUtils::SetInt(g_op, FE_IMPLY_TYPE, 6);
  ge::NodePtr g_node = graph->AddNode(g_op);

  Status ret1 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret1 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g_node, "tbe-custom");
  ASSERT_EQ(ret1, fe::SUCCESS);
  EXPECT_EQ(g_op->GetInputDesc(0).GetFormat(), FORMAT_NCHW);
  EXPECT_EQ(g_op->GetInputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(g_op->GetInputDesc(0).GetShape().GetDims(), dim);

  EXPECT_EQ(g_op->GetOutputDesc(0).GetFormat(), FORMAT_NCHW);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetDataType(), DT_INT16);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetShape().GetDims(), dim);
}
TEST_F(UTEST_fusion_engine_op_judge_precision_mode, set_nodes_IO_dtypes_float_force_fp16)
{
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = FORCE_FP16;
  op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g1_op = std::make_shared<OpDesc>("G1", "GOnlyfp32");
  OpDescPtr g2_op = std::make_shared<OpDesc>("G2", "GOnlyBf16");
  OpDescPtr g3_op = std::make_shared<OpDesc>("G3", "GOnlyFp16");
  OpDescPtr g4_op = std::make_shared<OpDesc>("G4", "GSupportFp32Bf16");
  OpDescPtr g5_op = std::make_shared<OpDesc>("G5", "GSupportfp32Fp16");
  OpDescPtr g6_op = std::make_shared<OpDesc>("G6", "GSupportBf16Fp16");
  OpDescPtr g7_op = std::make_shared<OpDesc>("G7", "G");

  vector<int64_t> dim({4, 33, 12, 16});
  ge::DataType dtype = DT_FLOAT;
  //add descriptor
  CreateSetOneNode(g1_op, dim, dtype);
  ge::NodePtr g1_node = graph->AddNode(g1_op);
  CreateSetOneNode(g2_op, dim, dtype);
  ge::NodePtr g2_node = graph->AddNode(g2_op);
  GraphUtils::AddEdge(g1_node->GetOutDataAnchor(0), g2_node->GetInDataAnchor(0));
  CreateSetOneNode(g3_op, dim, dtype);
  ge::NodePtr g3_node = graph->AddNode(g3_op);
  GraphUtils::AddEdge(g2_node->GetOutDataAnchor(0), g3_node->GetInDataAnchor(0));
  CreateSetOneNode(g4_op, dim, dtype);
  ge::NodePtr g4_node = graph->AddNode(g4_op);
  GraphUtils::AddEdge(g3_node->GetOutDataAnchor(0), g4_node->GetInDataAnchor(0));
  CreateSetOneNode(g5_op, dim, dtype);
  ge::NodePtr g5_node = graph->AddNode(g5_op);
  GraphUtils::AddEdge(g4_node->GetOutDataAnchor(0), g5_node->GetInDataAnchor(0));
  CreateSetOneNode(g6_op, dim, dtype);
  ge::NodePtr g6_node = graph->AddNode(g6_op);
  GraphUtils::AddEdge(g5_node->GetOutDataAnchor(0), g6_node->GetInDataAnchor(0));
  CreateSetOneNode(g7_op, dim, dtype);
  ge::NodePtr g7_node = graph->AddNode(g7_op);
  GraphUtils::AddEdge(g6_node->GetOutDataAnchor(0), g7_node->GetInDataAnchor(0));

  Status ret1 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g1_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret1 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g1_node, "tbe-custom");
  Status ret2 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g2_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret2 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g2_node, "tbe-custom");
  Status ret3 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g3_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret3 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g3_node, "tbe-custom");
  Status ret4 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g4_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret4 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g4_node, "tbe-custom");
  Status ret5 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g5_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret5 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g5_node, "tbe-custom");
  Status ret6 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g6_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret6 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g6_node, "tbe-custom");
  Status ret7 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g7_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret7 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g7_node, "tbe-custom");

  ASSERT_EQ(ret1, fe::SUCCESS);
  ASSERT_EQ(ret2, fe::SUCCESS);
  ASSERT_EQ(ret3, fe::SUCCESS);
  ASSERT_EQ(ret4, fe::SUCCESS);
  ASSERT_EQ(ret5, fe::SUCCESS);
  ASSERT_EQ(ret6, fe::SUCCESS);
  ASSERT_EQ(ret7, fe::SUCCESS);
  //G2 check no support dtypes. so return config context 1
  vector<ge::DataType> result_input = {DT_FLOAT, DT_UINT8, DT_FLOAT16, DT_FLOAT, DT_FLOAT16,
                                       DT_FLOAT16, DT_FLOAT16};
  vector<ge::DataType> result_output = {DT_FLOAT, DT_UINT8, DT_FLOAT16, DT_FLOAT, DT_FLOAT16,
                                       DT_FLOAT16, DT_FLOAT16};

  EXPECT_EQ(g1_op->GetInputDesc(0).GetDataType(), result_input[0]);
  EXPECT_EQ(g2_op->GetInputDesc(0).GetDataType(), result_input[1]);
  EXPECT_EQ(g3_op->GetInputDesc(0).GetDataType(), result_input[2]);
  EXPECT_EQ(g4_op->GetInputDesc(0).GetDataType(), result_input[3]);
  EXPECT_EQ(g5_op->GetInputDesc(0).GetDataType(), result_input[4]);
  EXPECT_EQ(g6_op->GetInputDesc(0).GetDataType(), result_input[5]);
  EXPECT_EQ(g7_op->GetInputDesc(0).GetDataType(), result_input[6]);


  EXPECT_EQ(g1_op->GetOutputDesc(0).GetDataType(), result_output[0]);
  EXPECT_EQ(g2_op->GetOutputDesc(0).GetDataType(), result_output[1]);
  EXPECT_EQ(g3_op->GetOutputDesc(0).GetDataType(), result_output[2]);
  EXPECT_EQ(g4_op->GetOutputDesc(0).GetDataType(), result_output[3]);
  EXPECT_EQ(g5_op->GetOutputDesc(0).GetDataType(), result_output[4]);
  EXPECT_EQ(g6_op->GetOutputDesc(0).GetDataType(), result_output[5]);
  EXPECT_EQ(g7_op->GetOutputDesc(0).GetDataType(), result_output[6]);
}

TEST_F(UTEST_fusion_engine_op_judge_precision_mode, set_nodes_IO_dtypes_float_force_fp16_1)
{
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = FORCE_FP16;
  op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g1_op = std::make_shared<OpDesc>("G1", "GOnlyfp32");
  OpDescPtr g2_op = std::make_shared<OpDesc>("G2", "GOnlyBf16");
  OpDescPtr g3_op = std::make_shared<OpDesc>("G3", "GOnlyFp16");
  OpDescPtr g4_op = std::make_shared<OpDesc>("G4", "GSupportFp32Bf16");
  OpDescPtr g5_op = std::make_shared<OpDesc>("G5", "GSupportfp32Fp16");
  OpDescPtr g6_op = std::make_shared<OpDesc>("G6", "GSupportBf16Fp16");
  OpDescPtr g7_op = std::make_shared<OpDesc>("G7", "G");

  vector<int64_t> dim({4, 33, 12, 16});
  ge::DataType dtype = DT_BF16;
  //add descriptor
  CreateSetOneNode(g1_op, dim, dtype);
  ge::NodePtr g1_node = graph->AddNode(g1_op);
  CreateSetOneNode(g2_op, dim, dtype);
  ge::NodePtr g2_node = graph->AddNode(g2_op);
  GraphUtils::AddEdge(g1_node->GetOutDataAnchor(0), g2_node->GetInDataAnchor(0));
  CreateSetOneNode(g3_op, dim, dtype);
  ge::NodePtr g3_node = graph->AddNode(g3_op);
  GraphUtils::AddEdge(g2_node->GetOutDataAnchor(0), g3_node->GetInDataAnchor(0));
  CreateSetOneNode(g4_op, dim, dtype);
  ge::NodePtr g4_node = graph->AddNode(g4_op);
  GraphUtils::AddEdge(g3_node->GetOutDataAnchor(0), g4_node->GetInDataAnchor(0));
  CreateSetOneNode(g5_op, dim, dtype);
  ge::NodePtr g5_node = graph->AddNode(g5_op);
  GraphUtils::AddEdge(g4_node->GetOutDataAnchor(0), g5_node->GetInDataAnchor(0));
  CreateSetOneNode(g6_op, dim, dtype);
  ge::NodePtr g6_node = graph->AddNode(g6_op);
  GraphUtils::AddEdge(g5_node->GetOutDataAnchor(0), g6_node->GetInDataAnchor(0));
  CreateSetOneNode(g7_op, dim, dtype);
  ge::NodePtr g7_node = graph->AddNode(g7_op);
  GraphUtils::AddEdge(g6_node->GetOutDataAnchor(0), g7_node->GetInDataAnchor(0));

  Status ret1 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g1_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret1 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g1_node, "tbe-custom");
  Status ret2 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g2_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret2 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g2_node, "tbe-custom");
  Status ret3 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g3_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret3 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g3_node, "tbe-custom");
  Status ret4 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g4_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret4 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g4_node, "tbe-custom");
  Status ret5 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g5_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret5 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g5_node, "tbe-custom");
  Status ret6 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g6_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret6 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g6_node, "tbe-custom");
  Status ret7 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g7_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret7 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g7_node, "tbe-custom");

  ASSERT_EQ(ret1, fe::SUCCESS);
  ASSERT_EQ(ret2, fe::SUCCESS);
  ASSERT_EQ(ret3, fe::SUCCESS);
  ASSERT_EQ(ret4, fe::SUCCESS);
  ASSERT_EQ(ret5, fe::SUCCESS);
  ASSERT_EQ(ret6, fe::SUCCESS);
  ASSERT_EQ(ret7, fe::SUCCESS);

  //G2 check no support dtypes. so return config context 1
  vector<ge::DataType> result_input = {DT_FLOAT, DT_UINT8, DT_FLOAT16, DT_FLOAT, DT_FLOAT16,
                                       DT_FLOAT16, DT_FLOAT16};
  vector<ge::DataType> result_output = {DT_FLOAT, DT_UINT8, DT_FLOAT16, DT_FLOAT, DT_FLOAT16,
                                       DT_FLOAT16, DT_FLOAT16};
  EXPECT_EQ(g1_op->GetInputDesc(0).GetDataType(), result_input[0]);
  EXPECT_EQ(g2_op->GetInputDesc(0).GetDataType(), result_input[1]);
  EXPECT_EQ(g3_op->GetInputDesc(0).GetDataType(), result_input[2]);
  EXPECT_EQ(g4_op->GetInputDesc(0).GetDataType(), result_input[3]);
  EXPECT_EQ(g5_op->GetInputDesc(0).GetDataType(), result_input[4]);
  EXPECT_EQ(g6_op->GetInputDesc(0).GetDataType(), result_input[5]);
  EXPECT_EQ(g7_op->GetInputDesc(0).GetDataType(), result_input[6]);


  EXPECT_EQ(g1_op->GetOutputDesc(0).GetDataType(), result_output[0]);
  EXPECT_EQ(g2_op->GetOutputDesc(0).GetDataType(), result_output[1]);
  EXPECT_EQ(g3_op->GetOutputDesc(0).GetDataType(), result_output[2]);
  EXPECT_EQ(g4_op->GetOutputDesc(0).GetDataType(), result_output[3]);
  EXPECT_EQ(g5_op->GetOutputDesc(0).GetDataType(), result_output[4]);
  EXPECT_EQ(g6_op->GetOutputDesc(0).GetDataType(), result_output[5]);
  EXPECT_EQ(g7_op->GetOutputDesc(0).GetDataType(), result_output[6]);
}

TEST_F(UTEST_fusion_engine_op_judge_precision_mode, set_nodes_IO_dtypes_float_force_fp16_2)
{
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = FORCE_FP16;
  op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g1_op = std::make_shared<OpDesc>("G1", "GOnlyfp32");
  OpDescPtr g2_op = std::make_shared<OpDesc>("G2", "GOnlyBf16");
  OpDescPtr g3_op = std::make_shared<OpDesc>("G3", "GOnlyFp16");
  OpDescPtr g4_op = std::make_shared<OpDesc>("G4", "GSupportFp32Bf16");
  OpDescPtr g5_op = std::make_shared<OpDesc>("G5", "GSupportfp32Fp16");
  OpDescPtr g6_op = std::make_shared<OpDesc>("G6", "GSupportBf16Fp16");
  OpDescPtr g7_op = std::make_shared<OpDesc>("G7", "G");

  vector<int64_t> dim({4, 33, 12, 16});
  ge::DataType dtype = DT_FLOAT16;
  //add descriptor
  CreateSetOneNode(g1_op, dim, dtype);
  ge::NodePtr g1_node = graph->AddNode(g1_op);
  CreateSetOneNode(g2_op, dim, dtype);
  ge::NodePtr g2_node = graph->AddNode(g2_op);
  GraphUtils::AddEdge(g1_node->GetOutDataAnchor(0), g2_node->GetInDataAnchor(0));
  CreateSetOneNode(g3_op, dim, dtype);
  ge::NodePtr g3_node = graph->AddNode(g3_op);
  GraphUtils::AddEdge(g2_node->GetOutDataAnchor(0), g3_node->GetInDataAnchor(0));
  CreateSetOneNode(g4_op, dim, dtype);
  ge::NodePtr g4_node = graph->AddNode(g4_op);
  GraphUtils::AddEdge(g3_node->GetOutDataAnchor(0), g4_node->GetInDataAnchor(0));
  CreateSetOneNode(g5_op, dim, dtype);
  ge::NodePtr g5_node = graph->AddNode(g5_op);
  GraphUtils::AddEdge(g4_node->GetOutDataAnchor(0), g5_node->GetInDataAnchor(0));
  CreateSetOneNode(g6_op, dim, dtype);
  ge::NodePtr g6_node = graph->AddNode(g6_op);
  GraphUtils::AddEdge(g5_node->GetOutDataAnchor(0), g6_node->GetInDataAnchor(0));
  CreateSetOneNode(g7_op, dim, dtype);
  ge::NodePtr g7_node = graph->AddNode(g7_op);
  GraphUtils::AddEdge(g6_node->GetOutDataAnchor(0), g7_node->GetInDataAnchor(0));

  Status ret1 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g1_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret1 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g1_node, "tbe-custom");
  Status ret2 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g2_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret2 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g2_node, "tbe-custom");
  Status ret3 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g3_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret3 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g3_node, "tbe-custom");
  Status ret4 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g4_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret4 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g4_node, "tbe-custom");
  Status ret5 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g5_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret5 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g5_node, "tbe-custom");
  Status ret6 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g6_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret6 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g6_node, "tbe-custom");
  Status ret7 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g7_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret7 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g7_node, "tbe-custom");

  ASSERT_EQ(ret1, fe::SUCCESS);
  ASSERT_EQ(ret2, fe::SUCCESS);
  ASSERT_EQ(ret3, fe::SUCCESS);
  ASSERT_EQ(ret4, fe::SUCCESS);
  ASSERT_EQ(ret5, fe::SUCCESS);
  ASSERT_EQ(ret6, fe::SUCCESS);
  ASSERT_EQ(ret7, fe::SUCCESS);
   //G2 check no support dtypes. so return config context 1
  vector<ge::DataType> result_input = {DT_FLOAT, DT_UINT8, DT_FLOAT16, DT_FLOAT, DT_FLOAT16,
                                       DT_FLOAT16, DT_FLOAT16,};
  vector<ge::DataType> result_output = {DT_FLOAT, DT_UINT8, DT_FLOAT16, DT_FLOAT, DT_FLOAT16,
                                       DT_FLOAT16, DT_FLOAT16,};
  EXPECT_EQ(g1_op->GetInputDesc(0).GetDataType(), result_input[0]);
  EXPECT_EQ(g2_op->GetInputDesc(0).GetDataType(), result_input[1]);
  EXPECT_EQ(g3_op->GetInputDesc(0).GetDataType(), result_input[2]);
  EXPECT_EQ(g4_op->GetInputDesc(0).GetDataType(), result_input[3]);
  EXPECT_EQ(g5_op->GetInputDesc(0).GetDataType(), result_input[4]);
  EXPECT_EQ(g6_op->GetInputDesc(0).GetDataType(), result_input[5]);
  EXPECT_EQ(g7_op->GetInputDesc(0).GetDataType(), result_input[6]);


  EXPECT_EQ(g1_op->GetOutputDesc(0).GetDataType(), result_output[0]);
  EXPECT_EQ(g2_op->GetOutputDesc(0).GetDataType(), result_output[1]);
  EXPECT_EQ(g3_op->GetOutputDesc(0).GetDataType(), result_output[2]);
  EXPECT_EQ(g4_op->GetOutputDesc(0).GetDataType(), result_output[3]);
  EXPECT_EQ(g5_op->GetOutputDesc(0).GetDataType(), result_output[4]);
  EXPECT_EQ(g6_op->GetOutputDesc(0).GetDataType(), result_output[5]);
  EXPECT_EQ(g7_op->GetOutputDesc(0).GetDataType(), result_output[6]);
}

TEST_F(UTEST_fusion_engine_op_judge_precision_mode, set_nodes_IO_dtypes_float_force_lowerprecision)
{
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = FORCE_LOWERPRECISION;
  op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g1_op = std::make_shared<OpDesc>("G1", "GOnlyfp32");
  OpDescPtr g2_op = std::make_shared<OpDesc>("G2", "GOnlyBf16");
  OpDescPtr g3_op = std::make_shared<OpDesc>("G3", "GOnlyFp16");
  OpDescPtr g4_op = std::make_shared<OpDesc>("G4", "GSupportFp32Bf16");
  OpDescPtr g5_op = std::make_shared<OpDesc>("G5", "GSupportfp32Fp16");
  OpDescPtr g6_op = std::make_shared<OpDesc>("G6", "GSupportBf16Fp16");
  OpDescPtr g7_op = std::make_shared<OpDesc>("G7", "G");

  vector<int64_t> dim({4, 33, 12, 16});
  ge::DataType dtype = DT_FLOAT;
  //add descriptor
  CreateSetOneNode(g1_op, dim, dtype);
  ge::NodePtr g1_node = graph->AddNode(g1_op);
  CreateSetOneNode(g2_op, dim, dtype);
  ge::NodePtr g2_node = graph->AddNode(g2_op);
  GraphUtils::AddEdge(g1_node->GetOutDataAnchor(0), g2_node->GetInDataAnchor(0));
  CreateSetOneNode(g3_op, dim, dtype);
  ge::NodePtr g3_node = graph->AddNode(g3_op);
  GraphUtils::AddEdge(g2_node->GetOutDataAnchor(0), g3_node->GetInDataAnchor(0));
  CreateSetOneNode(g4_op, dim, dtype);
  ge::NodePtr g4_node = graph->AddNode(g4_op);
  GraphUtils::AddEdge(g3_node->GetOutDataAnchor(0), g4_node->GetInDataAnchor(0));
  CreateSetOneNode(g5_op, dim, dtype);
  ge::NodePtr g5_node = graph->AddNode(g5_op);
  GraphUtils::AddEdge(g4_node->GetOutDataAnchor(0), g5_node->GetInDataAnchor(0));
  CreateSetOneNode(g6_op, dim, dtype);
  ge::NodePtr g6_node = graph->AddNode(g6_op);
  GraphUtils::AddEdge(g5_node->GetOutDataAnchor(0), g6_node->GetInDataAnchor(0));
  CreateSetOneNode(g7_op, dim, dtype);
  ge::NodePtr g7_node = graph->AddNode(g7_op);
  GraphUtils::AddEdge(g6_node->GetOutDataAnchor(0), g7_node->GetInDataAnchor(0));

  Status ret1 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g1_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret1 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g1_node, "tbe-custom");
  Status ret2 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g2_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret2 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g2_node, "tbe-custom");
  Status ret3 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g3_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret3 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g3_node, "tbe-custom");
  Status ret4 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g4_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret4 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g4_node, "tbe-custom");
  Status ret5 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g5_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret5 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g5_node, "tbe-custom");
  Status ret6 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g6_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret6 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g6_node, "tbe-custom");
  Status ret7 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g7_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret7 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g7_node, "tbe-custom");

  ASSERT_EQ(ret1, fe::SUCCESS);
  ASSERT_EQ(ret2, fe::SUCCESS);
  ASSERT_EQ(ret3, fe::SUCCESS);
  ASSERT_EQ(ret4, fe::SUCCESS);
  ASSERT_EQ(ret5, fe::SUCCESS);
  ASSERT_EQ(ret6, fe::SUCCESS);
  ASSERT_EQ(ret7, fe::SUCCESS);

  vector<ge::DataType> result_input = {DT_FLOAT, DT_BF16, DT_FLOAT16, DT_BF16, DT_FLOAT16,
                                       DT_BF16, DT_BF16};
  vector<ge::DataType> result_output = {DT_FLOAT, DT_BF16, DT_FLOAT16, DT_BF16, DT_FLOAT16,
                                       DT_BF16, DT_BF16};
  EXPECT_EQ(g1_op->GetInputDesc(0).GetDataType(), result_input[0]);
  EXPECT_EQ(g2_op->GetInputDesc(0).GetDataType(), result_input[1]);
  EXPECT_EQ(g3_op->GetInputDesc(0).GetDataType(), result_input[2]);
  EXPECT_EQ(g4_op->GetInputDesc(0).GetDataType(), result_input[3]);
  EXPECT_EQ(g5_op->GetInputDesc(0).GetDataType(), result_input[4]);
  EXPECT_EQ(g6_op->GetInputDesc(0).GetDataType(), result_input[5]);
  EXPECT_EQ(g7_op->GetInputDesc(0).GetDataType(), result_input[6]);


  EXPECT_EQ(g1_op->GetOutputDesc(0).GetDataType(), result_output[0]);
  EXPECT_EQ(g2_op->GetOutputDesc(0).GetDataType(), result_output[1]);
  EXPECT_EQ(g3_op->GetOutputDesc(0).GetDataType(), result_output[2]);
  EXPECT_EQ(g4_op->GetOutputDesc(0).GetDataType(), result_output[3]);
  EXPECT_EQ(g5_op->GetOutputDesc(0).GetDataType(), result_output[4]);
  EXPECT_EQ(g6_op->GetOutputDesc(0).GetDataType(), result_output[5]);
  EXPECT_EQ(g7_op->GetOutputDesc(0).GetDataType(), result_output[6]);
}

TEST_F(UTEST_fusion_engine_op_judge_precision_mode, set_nodes_IO_dtypes_float_force_lowerprecision_1)
{
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = FORCE_LOWERPRECISION;
  op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g1_op = std::make_shared<OpDesc>("G1", "GOnlyfp32");
  OpDescPtr g2_op = std::make_shared<OpDesc>("G2", "GOnlyBf16");
  OpDescPtr g3_op = std::make_shared<OpDesc>("G3", "GOnlyFp16");
  OpDescPtr g4_op = std::make_shared<OpDesc>("G4", "GSupportFp32Bf16");
  OpDescPtr g5_op = std::make_shared<OpDesc>("G5", "GSupportfp32Fp16");
  OpDescPtr g6_op = std::make_shared<OpDesc>("G6", "GSupportBf16Fp16");
  OpDescPtr g7_op = std::make_shared<OpDesc>("G7", "G");

  vector<int64_t> dim({4, 33, 12, 16});
  ge::DataType dtype = DT_BF16;
  //add descriptor
  CreateSetOneNode(g1_op, dim, dtype);
  ge::NodePtr g1_node = graph->AddNode(g1_op);
  CreateSetOneNode(g2_op, dim, dtype);
  ge::NodePtr g2_node = graph->AddNode(g2_op);
  GraphUtils::AddEdge(g1_node->GetOutDataAnchor(0), g2_node->GetInDataAnchor(0));
  CreateSetOneNode(g3_op, dim, dtype);
  ge::NodePtr g3_node = graph->AddNode(g3_op);
  GraphUtils::AddEdge(g2_node->GetOutDataAnchor(0), g3_node->GetInDataAnchor(0));
  CreateSetOneNode(g4_op, dim, dtype);
  ge::NodePtr g4_node = graph->AddNode(g4_op);
  GraphUtils::AddEdge(g3_node->GetOutDataAnchor(0), g4_node->GetInDataAnchor(0));
  CreateSetOneNode(g5_op, dim, dtype);
  ge::NodePtr g5_node = graph->AddNode(g5_op);
  GraphUtils::AddEdge(g4_node->GetOutDataAnchor(0), g5_node->GetInDataAnchor(0));
  CreateSetOneNode(g6_op, dim, dtype);
  ge::NodePtr g6_node = graph->AddNode(g6_op);
  GraphUtils::AddEdge(g5_node->GetOutDataAnchor(0), g6_node->GetInDataAnchor(0));
  CreateSetOneNode(g7_op, dim, dtype);
  ge::NodePtr g7_node = graph->AddNode(g7_op);
  GraphUtils::AddEdge(g6_node->GetOutDataAnchor(0), g7_node->GetInDataAnchor(0));

  Status ret1 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g1_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret1 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g1_node, "tbe-custom");
  Status ret2 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g2_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret2 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g2_node, "tbe-custom");
  Status ret3 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g3_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret3 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g3_node, "tbe-custom");
  Status ret4 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g4_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret4 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g4_node, "tbe-custom");
  Status ret5 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g5_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret5 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g5_node, "tbe-custom");
  Status ret6 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g6_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret6 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g6_node, "tbe-custom");
  Status ret7 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g7_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret7 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g7_node, "tbe-custom");

  ASSERT_EQ(ret1, fe::SUCCESS);
  ASSERT_EQ(ret2, fe::SUCCESS);
  ASSERT_EQ(ret3, fe::SUCCESS);
  ASSERT_EQ(ret4, fe::SUCCESS);
  ASSERT_EQ(ret5, fe::SUCCESS);
  ASSERT_EQ(ret6, fe::SUCCESS);
  ASSERT_EQ(ret7, fe::SUCCESS);

  vector<ge::DataType> result_input = {DT_FLOAT, DT_BF16, DT_FLOAT16, DT_BF16, DT_FLOAT16,
                                       DT_BF16, DT_BF16};
  vector<ge::DataType> result_output = {DT_FLOAT, DT_BF16, DT_FLOAT16, DT_BF16, DT_FLOAT16,
                                       DT_BF16, DT_BF16};
  EXPECT_EQ(g1_op->GetInputDesc(0).GetDataType(), result_input[0]);
  EXPECT_EQ(g2_op->GetInputDesc(0).GetDataType(), result_input[1]);
  EXPECT_EQ(g3_op->GetInputDesc(0).GetDataType(), result_input[2]);
  EXPECT_EQ(g4_op->GetInputDesc(0).GetDataType(), result_input[3]);
  EXPECT_EQ(g5_op->GetInputDesc(0).GetDataType(), result_input[4]);
  EXPECT_EQ(g6_op->GetInputDesc(0).GetDataType(), result_input[5]);
  EXPECT_EQ(g7_op->GetInputDesc(0).GetDataType(), result_input[6]);


  EXPECT_EQ(g1_op->GetOutputDesc(0).GetDataType(), result_output[0]);
  EXPECT_EQ(g2_op->GetOutputDesc(0).GetDataType(), result_output[1]);
  EXPECT_EQ(g3_op->GetOutputDesc(0).GetDataType(), result_output[2]);
  EXPECT_EQ(g4_op->GetOutputDesc(0).GetDataType(), result_output[3]);
  EXPECT_EQ(g5_op->GetOutputDesc(0).GetDataType(), result_output[4]);
  EXPECT_EQ(g6_op->GetOutputDesc(0).GetDataType(), result_output[5]);
  EXPECT_EQ(g7_op->GetOutputDesc(0).GetDataType(), result_output[6]);
}

TEST_F(UTEST_fusion_engine_op_judge_precision_mode, set_nodes_IO_dtypes_float_force_lowerprecision_2)
{
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = FORCE_LOWERPRECISION;
  op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g1_op = std::make_shared<OpDesc>("G1", "GOnlyfp32");
  OpDescPtr g2_op = std::make_shared<OpDesc>("G2", "GOnlyBf16");
  OpDescPtr g3_op = std::make_shared<OpDesc>("G3", "GOnlyFp16");
  OpDescPtr g4_op = std::make_shared<OpDesc>("G4", "GSupportFp32Bf16");
  OpDescPtr g5_op = std::make_shared<OpDesc>("G5", "GSupportfp32Fp16");
  OpDescPtr g6_op = std::make_shared<OpDesc>("G6", "GSupportBf16Fp16");
  OpDescPtr g7_op = std::make_shared<OpDesc>("G7", "G");

  vector<int64_t> dim({4, 33, 12, 16});
  ge::DataType dtype = DT_FLOAT16;
  //add descriptor
  CreateSetOneNode(g1_op, dim, dtype);
  ge::NodePtr g1_node = graph->AddNode(g1_op);
  CreateSetOneNode(g2_op, dim, dtype);
  ge::NodePtr g2_node = graph->AddNode(g2_op);
  GraphUtils::AddEdge(g1_node->GetOutDataAnchor(0), g2_node->GetInDataAnchor(0));
  CreateSetOneNode(g3_op, dim, dtype);
  ge::NodePtr g3_node = graph->AddNode(g3_op);
  GraphUtils::AddEdge(g2_node->GetOutDataAnchor(0), g3_node->GetInDataAnchor(0));
  CreateSetOneNode(g4_op, dim, dtype);
  ge::NodePtr g4_node = graph->AddNode(g4_op);
  GraphUtils::AddEdge(g3_node->GetOutDataAnchor(0), g4_node->GetInDataAnchor(0));
  CreateSetOneNode(g5_op, dim, dtype);
  ge::NodePtr g5_node = graph->AddNode(g5_op);
  GraphUtils::AddEdge(g4_node->GetOutDataAnchor(0), g5_node->GetInDataAnchor(0));
  CreateSetOneNode(g6_op, dim, dtype);
  ge::NodePtr g6_node = graph->AddNode(g6_op);
  GraphUtils::AddEdge(g5_node->GetOutDataAnchor(0), g6_node->GetInDataAnchor(0));
  CreateSetOneNode(g7_op, dim, dtype);
  ge::NodePtr g7_node = graph->AddNode(g7_op);
  GraphUtils::AddEdge(g6_node->GetOutDataAnchor(0), g7_node->GetInDataAnchor(0));

  Status ret1 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g1_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret1 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g1_node, "tbe-custom");
  Status ret2 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g2_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret2 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g2_node, "tbe-custom");
  Status ret3 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g3_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret3 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g3_node, "tbe-custom");
  Status ret4 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g4_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret4 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g4_node, "tbe-custom");
  Status ret5 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g5_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret5 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g5_node, "tbe-custom");
  Status ret6 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g6_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret6 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g6_node, "tbe-custom");
  Status ret7 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g7_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret7 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g7_node, "tbe-custom");

  ASSERT_EQ(ret1, fe::SUCCESS);
  ASSERT_EQ(ret2, fe::SUCCESS);
  ASSERT_EQ(ret3, fe::SUCCESS);
  ASSERT_EQ(ret4, fe::SUCCESS);
  ASSERT_EQ(ret5, fe::SUCCESS);
  ASSERT_EQ(ret6, fe::SUCCESS);
  ASSERT_EQ(ret7, fe::SUCCESS);

  vector<ge::DataType> result_input = {DT_FLOAT, DT_UINT8, DT_FLOAT16, DT_FLOAT, DT_FLOAT16,
                                       DT_FLOAT16, DT_FLOAT16};
  vector<ge::DataType> result_output = {DT_FLOAT, DT_UINT8, DT_FLOAT16, DT_FLOAT, DT_FLOAT16,
                                       DT_FLOAT16, DT_FLOAT16};
  EXPECT_EQ(g1_op->GetInputDesc(0).GetDataType(), result_input[0]);
  EXPECT_EQ(g2_op->GetInputDesc(0).GetDataType(), result_input[1]);
  EXPECT_EQ(g3_op->GetInputDesc(0).GetDataType(), result_input[2]);
  EXPECT_EQ(g4_op->GetInputDesc(0).GetDataType(), result_input[3]);
  EXPECT_EQ(g5_op->GetInputDesc(0).GetDataType(), result_input[4]);
  EXPECT_EQ(g6_op->GetInputDesc(0).GetDataType(), result_input[5]);
  EXPECT_EQ(g7_op->GetInputDesc(0).GetDataType(), result_input[6]);


  EXPECT_EQ(g1_op->GetOutputDesc(0).GetDataType(), result_output[0]);
  EXPECT_EQ(g2_op->GetOutputDesc(0).GetDataType(), result_output[1]);
  EXPECT_EQ(g3_op->GetOutputDesc(0).GetDataType(), result_output[2]);
  EXPECT_EQ(g4_op->GetOutputDesc(0).GetDataType(), result_output[3]);
  EXPECT_EQ(g5_op->GetOutputDesc(0).GetDataType(), result_output[4]);
  EXPECT_EQ(g6_op->GetOutputDesc(0).GetDataType(), result_output[5]);
  EXPECT_EQ(g7_op->GetOutputDesc(0).GetDataType(), result_output[6]);
}

TEST_F(UTEST_fusion_engine_op_judge_precision_mode, set_nodes_IO_dtypes_float_allow_fp32_fp16)
{
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = ALLOW_FP32_TO_FP16;
  op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g1_op = std::make_shared<OpDesc>("G1", "GOnlyfp32");
  OpDescPtr g2_op = std::make_shared<OpDesc>("G2", "GOnlyBf16");
  OpDescPtr g3_op = std::make_shared<OpDesc>("G3", "GOnlyFp16");
  OpDescPtr g4_op = std::make_shared<OpDesc>("G4", "GSupportFp32Bf16");
  OpDescPtr g5_op = std::make_shared<OpDesc>("G5", "GSupportfp32Fp16");
  OpDescPtr g6_op = std::make_shared<OpDesc>("G6", "GSupportBf16Fp16");
  OpDescPtr g7_op = std::make_shared<OpDesc>("G7", "G");

  vector<int64_t> dim({4, 33, 12, 16});
  ge::DataType dtype = DT_FLOAT;
  //add descriptor
  CreateSetOneNode(g1_op, dim, dtype);
  ge::NodePtr g1_node = graph->AddNode(g1_op);
  CreateSetOneNode(g2_op, dim, dtype);
  ge::NodePtr g2_node = graph->AddNode(g2_op);
  GraphUtils::AddEdge(g1_node->GetOutDataAnchor(0), g2_node->GetInDataAnchor(0));
  CreateSetOneNode(g3_op, dim, dtype);
  ge::NodePtr g3_node = graph->AddNode(g3_op);
  GraphUtils::AddEdge(g2_node->GetOutDataAnchor(0), g3_node->GetInDataAnchor(0));
  CreateSetOneNode(g4_op, dim, dtype);
  ge::NodePtr g4_node = graph->AddNode(g4_op);
  GraphUtils::AddEdge(g3_node->GetOutDataAnchor(0), g4_node->GetInDataAnchor(0));
  CreateSetOneNode(g5_op, dim, dtype);
  ge::NodePtr g5_node = graph->AddNode(g5_op);
  GraphUtils::AddEdge(g4_node->GetOutDataAnchor(0), g5_node->GetInDataAnchor(0));
  CreateSetOneNode(g6_op, dim, dtype);
  ge::NodePtr g6_node = graph->AddNode(g6_op);
  GraphUtils::AddEdge(g5_node->GetOutDataAnchor(0), g6_node->GetInDataAnchor(0));
  CreateSetOneNode(g7_op, dim, dtype);
  ge::NodePtr g7_node = graph->AddNode(g7_op);
  GraphUtils::AddEdge(g6_node->GetOutDataAnchor(0), g7_node->GetInDataAnchor(0));

  Status ret1 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g1_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret1 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g1_node, "tbe-custom");
  Status ret2 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g2_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret2 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g2_node, "tbe-custom");
  Status ret3 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g3_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret3 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g3_node, "tbe-custom");
  Status ret4 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g4_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret4 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g4_node, "tbe-custom");
  Status ret5 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g5_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret5 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g5_node, "tbe-custom");
  Status ret6 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g6_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret6 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g6_node, "tbe-custom");
  Status ret7 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g7_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret7 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g7_node, "tbe-custom");

  ASSERT_EQ(ret1, fe::SUCCESS);
  ASSERT_EQ(ret2, fe::SUCCESS);
  ASSERT_EQ(ret3, fe::SUCCESS);
  ASSERT_EQ(ret4, fe::SUCCESS);
  ASSERT_EQ(ret5, fe::SUCCESS);
  ASSERT_EQ(ret6, fe::SUCCESS);
  ASSERT_EQ(ret7, fe::SUCCESS);
  //G2 check no support dtypes. so return config context 1
  vector<ge::DataType> result_input = {DT_FLOAT, DT_UINT8, DT_FLOAT16, DT_FLOAT, DT_FLOAT,
                                       DT_FLOAT16, DT_FLOAT};
  vector<ge::DataType> result_output = {DT_FLOAT, DT_UINT8, DT_FLOAT16, DT_FLOAT, DT_FLOAT,
                                        DT_FLOAT16, DT_FLOAT};
  EXPECT_EQ(g1_op->GetInputDesc(0).GetDataType(), result_input[0]);
  EXPECT_EQ(g2_op->GetInputDesc(0).GetDataType(), result_input[1]);
  EXPECT_EQ(g3_op->GetInputDesc(0).GetDataType(), result_input[2]);
  EXPECT_EQ(g4_op->GetInputDesc(0).GetDataType(), result_input[3]);
  EXPECT_EQ(g5_op->GetInputDesc(0).GetDataType(), result_input[4]);
  EXPECT_EQ(g6_op->GetInputDesc(0).GetDataType(), result_input[5]);
  EXPECT_EQ(g7_op->GetInputDesc(0).GetDataType(), result_input[6]);


  EXPECT_EQ(g1_op->GetOutputDesc(0).GetDataType(), result_output[0]);
  EXPECT_EQ(g2_op->GetOutputDesc(0).GetDataType(), result_output[1]);
  EXPECT_EQ(g3_op->GetOutputDesc(0).GetDataType(), result_output[2]);
  EXPECT_EQ(g4_op->GetOutputDesc(0).GetDataType(), result_output[3]);
  EXPECT_EQ(g5_op->GetOutputDesc(0).GetDataType(), result_output[4]);
  EXPECT_EQ(g6_op->GetOutputDesc(0).GetDataType(), result_output[5]);
  EXPECT_EQ(g7_op->GetOutputDesc(0).GetDataType(), result_output[6]);
}

TEST_F(UTEST_fusion_engine_op_judge_precision_mode, set_nodes_IO_dtypes_float_allow_fp32_fp16_1)
{
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = ALLOW_FP32_TO_FP16;
  op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g1_op = std::make_shared<OpDesc>("G1", "GOnlyfp32");
  OpDescPtr g2_op = std::make_shared<OpDesc>("G2", "GOnlyBf16");
  OpDescPtr g3_op = std::make_shared<OpDesc>("G3", "GOnlyFp16");
  OpDescPtr g4_op = std::make_shared<OpDesc>("G4", "GSupportFp32Bf16");
  OpDescPtr g5_op = std::make_shared<OpDesc>("G5", "GSupportfp32Fp16");
  OpDescPtr g6_op = std::make_shared<OpDesc>("G6", "GSupportBf16Fp16");
  OpDescPtr g7_op = std::make_shared<OpDesc>("G7", "G");

  vector<int64_t> dim({4, 33, 12, 16});
  ge::DataType dtype = DT_BF16;
  //add descriptor
  CreateSetOneNode(g1_op, dim, dtype);
  ge::NodePtr g1_node = graph->AddNode(g1_op);
  CreateSetOneNode(g2_op, dim, dtype);
  ge::NodePtr g2_node = graph->AddNode(g2_op);
  GraphUtils::AddEdge(g1_node->GetOutDataAnchor(0), g2_node->GetInDataAnchor(0));
  CreateSetOneNode(g3_op, dim, dtype);
  ge::NodePtr g3_node = graph->AddNode(g3_op);
  GraphUtils::AddEdge(g2_node->GetOutDataAnchor(0), g3_node->GetInDataAnchor(0));
  CreateSetOneNode(g4_op, dim, dtype);
  ge::NodePtr g4_node = graph->AddNode(g4_op);
  GraphUtils::AddEdge(g3_node->GetOutDataAnchor(0), g4_node->GetInDataAnchor(0));
  CreateSetOneNode(g5_op, dim, dtype);
  ge::NodePtr g5_node = graph->AddNode(g5_op);
  GraphUtils::AddEdge(g4_node->GetOutDataAnchor(0), g5_node->GetInDataAnchor(0));
  CreateSetOneNode(g6_op, dim, dtype);
  ge::NodePtr g6_node = graph->AddNode(g6_op);
  GraphUtils::AddEdge(g5_node->GetOutDataAnchor(0), g6_node->GetInDataAnchor(0));
  CreateSetOneNode(g7_op, dim, dtype);
  ge::NodePtr g7_node = graph->AddNode(g7_op);
  GraphUtils::AddEdge(g6_node->GetOutDataAnchor(0), g7_node->GetInDataAnchor(0));

  Status ret1 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g1_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret1 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g1_node, "tbe-custom");
  Status ret2 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g2_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret2 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g2_node, "tbe-custom");
  Status ret3 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g3_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret3 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g3_node, "tbe-custom");
  Status ret4 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g4_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret4 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g4_node, "tbe-custom");
  Status ret5 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g5_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret5 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g5_node, "tbe-custom");
  Status ret6 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g6_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret6 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g6_node, "tbe-custom");
  Status ret7 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g7_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret7 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g7_node, "tbe-custom");

  ASSERT_EQ(ret1, fe::SUCCESS);
  ASSERT_EQ(ret2, fe::SUCCESS);
  ASSERT_EQ(ret3, fe::SUCCESS);
  ASSERT_EQ(ret4, fe::SUCCESS);
  ASSERT_EQ(ret5, fe::SUCCESS);
  ASSERT_EQ(ret6, fe::SUCCESS);
  ASSERT_EQ(ret7, fe::SUCCESS);
  //G2 check no support dtypes. so return config context 1
  vector<ge::DataType> result_input = {DT_FLOAT, DT_BF16, DT_FLOAT16, DT_BF16, DT_FLOAT,
                                       DT_BF16, DT_BF16};
  vector<ge::DataType> result_output = {DT_FLOAT, DT_BF16, DT_FLOAT16, DT_BF16, DT_FLOAT,
                                        DT_BF16, DT_BF16};

  EXPECT_EQ(g1_op->GetInputDesc(0).GetDataType(), result_input[0]);
  EXPECT_EQ(g2_op->GetInputDesc(0).GetDataType(), result_input[1]);
  EXPECT_EQ(g3_op->GetInputDesc(0).GetDataType(), result_input[2]);
  EXPECT_EQ(g4_op->GetInputDesc(0).GetDataType(), result_input[3]);
  EXPECT_EQ(g5_op->GetInputDesc(0).GetDataType(), result_input[4]);
  EXPECT_EQ(g6_op->GetInputDesc(0).GetDataType(), result_input[5]);
  EXPECT_EQ(g7_op->GetInputDesc(0).GetDataType(), result_input[6]);


  EXPECT_EQ(g1_op->GetOutputDesc(0).GetDataType(), result_output[0]);
  EXPECT_EQ(g2_op->GetOutputDesc(0).GetDataType(), result_output[1]);
  EXPECT_EQ(g3_op->GetOutputDesc(0).GetDataType(), result_output[2]);
  EXPECT_EQ(g4_op->GetOutputDesc(0).GetDataType(), result_output[3]);
  EXPECT_EQ(g5_op->GetOutputDesc(0).GetDataType(), result_output[4]);
  EXPECT_EQ(g6_op->GetOutputDesc(0).GetDataType(), result_output[5]);
  EXPECT_EQ(g7_op->GetOutputDesc(0).GetDataType(), result_output[6]);
}

TEST_F(UTEST_fusion_engine_op_judge_precision_mode, set_nodes_IO_dtypes_float_allow_fp32_fp16_2)
{
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = ALLOW_FP32_TO_FP16;
  op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g1_op = std::make_shared<OpDesc>("G1", "GOnlyfp32");
  OpDescPtr g2_op = std::make_shared<OpDesc>("G2", "GOnlyBf16");
  OpDescPtr g3_op = std::make_shared<OpDesc>("G3", "GOnlyFp16");
  OpDescPtr g4_op = std::make_shared<OpDesc>("G4", "GSupportFp32Bf16");
  OpDescPtr g5_op = std::make_shared<OpDesc>("G5", "GSupportfp32Fp16");
  OpDescPtr g6_op = std::make_shared<OpDesc>("G6", "GSupportBf16Fp16");
  OpDescPtr g7_op = std::make_shared<OpDesc>("G7", "G");

  vector<int64_t> dim({4, 33, 12, 16});
  ge::DataType dtype = DT_FLOAT16;
  //add descriptor
  CreateSetOneNode(g1_op, dim, dtype);
  ge::NodePtr g1_node = graph->AddNode(g1_op);
  CreateSetOneNode(g2_op, dim, dtype);
  ge::NodePtr g2_node = graph->AddNode(g2_op);
  GraphUtils::AddEdge(g1_node->GetOutDataAnchor(0), g2_node->GetInDataAnchor(0));
  CreateSetOneNode(g3_op, dim, dtype);
  ge::NodePtr g3_node = graph->AddNode(g3_op);
  GraphUtils::AddEdge(g2_node->GetOutDataAnchor(0), g3_node->GetInDataAnchor(0));
  CreateSetOneNode(g4_op, dim, dtype);
  ge::NodePtr g4_node = graph->AddNode(g4_op);
  GraphUtils::AddEdge(g3_node->GetOutDataAnchor(0), g4_node->GetInDataAnchor(0));
  CreateSetOneNode(g5_op, dim, dtype);
  ge::NodePtr g5_node = graph->AddNode(g5_op);
  GraphUtils::AddEdge(g4_node->GetOutDataAnchor(0), g5_node->GetInDataAnchor(0));
  CreateSetOneNode(g6_op, dim, dtype);
  ge::NodePtr g6_node = graph->AddNode(g6_op);
  GraphUtils::AddEdge(g5_node->GetOutDataAnchor(0), g6_node->GetInDataAnchor(0));
  CreateSetOneNode(g7_op, dim, dtype);
  ge::NodePtr g7_node = graph->AddNode(g7_op);
  GraphUtils::AddEdge(g6_node->GetOutDataAnchor(0), g7_node->GetInDataAnchor(0));

  Status ret1 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g1_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret1 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g1_node, "tbe-custom");
  Status ret2 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g2_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret2 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g2_node, "tbe-custom");
  Status ret3 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g3_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret3 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g3_node, "tbe-custom");
  Status ret4 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g4_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret4 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g4_node, "tbe-custom");
  Status ret5 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g5_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret5 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g5_node, "tbe-custom");
  Status ret6 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g6_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret6 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g6_node, "tbe-custom");
  Status ret7 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g7_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret7 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g7_node, "tbe-custom");

  ASSERT_EQ(ret1, fe::SUCCESS);
  ASSERT_EQ(ret2, fe::SUCCESS);
  ASSERT_EQ(ret3, fe::SUCCESS);
  ASSERT_EQ(ret4, fe::SUCCESS);
  ASSERT_EQ(ret5, fe::SUCCESS);
  ASSERT_EQ(ret6, fe::SUCCESS);
  ASSERT_EQ(ret7, fe::SUCCESS);

  vector<ge::DataType> result_input = {DT_FLOAT, DT_UINT8, DT_FLOAT16, DT_FLOAT, DT_FLOAT16,
                                       DT_FLOAT16, DT_FLOAT16};
  vector<ge::DataType> result_output = {DT_FLOAT, DT_UINT8, DT_FLOAT16, DT_FLOAT, DT_FLOAT16,
                                       DT_FLOAT16, DT_FLOAT16};
  EXPECT_EQ(g1_op->GetInputDesc(0).GetDataType(), result_input[0]);
  EXPECT_EQ(g2_op->GetInputDesc(0).GetDataType(), result_input[1]);
  EXPECT_EQ(g3_op->GetInputDesc(0).GetDataType(), result_input[2]);
  EXPECT_EQ(g4_op->GetInputDesc(0).GetDataType(), result_input[3]);
  EXPECT_EQ(g5_op->GetInputDesc(0).GetDataType(), result_input[4]);
  EXPECT_EQ(g6_op->GetInputDesc(0).GetDataType(), result_input[5]);
  EXPECT_EQ(g7_op->GetInputDesc(0).GetDataType(), result_input[6]);


  EXPECT_EQ(g1_op->GetOutputDesc(0).GetDataType(), result_output[0]);
  EXPECT_EQ(g2_op->GetOutputDesc(0).GetDataType(), result_output[1]);
  EXPECT_EQ(g3_op->GetOutputDesc(0).GetDataType(), result_output[2]);
  EXPECT_EQ(g4_op->GetOutputDesc(0).GetDataType(), result_output[3]);
  EXPECT_EQ(g5_op->GetOutputDesc(0).GetDataType(), result_output[4]);
  EXPECT_EQ(g6_op->GetOutputDesc(0).GetDataType(), result_output[5]);
  EXPECT_EQ(g7_op->GetOutputDesc(0).GetDataType(), result_output[6]);
}

TEST_F(UTEST_fusion_engine_op_judge_precision_mode, set_nodes_IO_dtypes_float_allow_fp32_bf16_3)
{
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = ALLOW_FP32_TO_FP16;
  op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g1_op = std::make_shared<OpDesc>("G1", "GFp16Fp1632Fp1632");
  vector<int64_t> dim({4, 33, 12, 16});
  ge::DataType dtype = DT_FLOAT;
  //add descriptor
  CreateTwoInputOneOutputNode(g1_op, dim, dtype);
  ge::NodePtr g1_node = graph->AddNode(g1_op);
  Status ret1 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g1_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret1 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g1_node, "tbe-custom");
  ASSERT_EQ(ret1, fe::SUCCESS);

  vector<ge::DataType> result_input = {DT_FLOAT16, DT_BF16, DT_UINT8, DT_BF16, DT_FLOAT,
                                       DT_BF16, DT_BF16};

  EXPECT_EQ(g1_op->GetInputDesc(1).GetDataType(), result_input[0]);
  EXPECT_EQ(g1_op->GetOutputDesc(0).GetDataType(), result_input[0]);
}

TEST_F(UTEST_fusion_engine_op_judge_precision_mode, set_nodes_IO_dtypes_float_allow_fp32_bf16_4)
{
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = ALLOW_FP32_TO_FP16;
  op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g1_op = std::make_shared<OpDesc>("G1", "GFp16Fp1632Fp32");
  vector<int64_t> dim({4, 33, 12, 16});
  ge::DataType dtype = DT_FLOAT;
  //add descriptor
  CreateTwoInputOneOutputNode(g1_op, dim, dtype);
  ge::NodePtr g1_node = graph->AddNode(g1_op);
  Status ret1 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g1_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret1 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g1_node, "tbe-custom");
  ASSERT_EQ(ret1, fe::SUCCESS);

  vector<ge::DataType> result_input = {DT_FLOAT16, DT_BF16, DT_UINT8, DT_BF16, DT_FLOAT,
                                       DT_BF16, DT_BF16};

  EXPECT_EQ(g1_op->GetInputDesc(1).GetDataType(), result_input[0]);
  EXPECT_EQ(g1_op->GetOutputDesc(0).GetDataType(), result_input[4]);
}

TEST_F(UTEST_fusion_engine_op_judge_precision_mode, set_nodes_IO_dtypes_float_allow_fp32_bf16)
{
  PlatformInfoManager::Instance().opti_compilation_infos_.SetSocVersion("Ascend910B2");
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = ALLOW_FP32_TO_BF16;
  op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g1_op = std::make_shared<OpDesc>("G1", "GOnlyfp32");
  OpDescPtr g2_op = std::make_shared<OpDesc>("G2", "GOnlyBf16");
  OpDescPtr g3_op = std::make_shared<OpDesc>("G3", "GOnlyFp16");
  OpDescPtr g4_op = std::make_shared<OpDesc>("G4", "GSupportFp32Bf16");
  OpDescPtr g5_op = std::make_shared<OpDesc>("G5", "GSupportfp32Fp16");
  OpDescPtr g6_op = std::make_shared<OpDesc>("G6", "GSupportBf16Fp16");
  OpDescPtr g7_op = std::make_shared<OpDesc>("G7", "G");

  vector<int64_t> dim({4, 33, 12, 16});
  ge::DataType dtype = DT_FLOAT;
  //add descriptor
  CreateSetOneNode(g1_op, dim, dtype);
  ge::NodePtr g1_node = graph->AddNode(g1_op);
  CreateSetOneNode(g2_op, dim, dtype);
  ge::NodePtr g2_node = graph->AddNode(g2_op);
  GraphUtils::AddEdge(g1_node->GetOutDataAnchor(0), g2_node->GetInDataAnchor(0));
  CreateSetOneNode(g3_op, dim, dtype);
  ge::NodePtr g3_node = graph->AddNode(g3_op);
  GraphUtils::AddEdge(g2_node->GetOutDataAnchor(0), g3_node->GetInDataAnchor(0));
  CreateSetOneNode(g4_op, dim, dtype);
  ge::NodePtr g4_node = graph->AddNode(g4_op);
  GraphUtils::AddEdge(g3_node->GetOutDataAnchor(0), g4_node->GetInDataAnchor(0));
  CreateSetOneNode(g5_op, dim, dtype);
  ge::NodePtr g5_node = graph->AddNode(g5_op);
  GraphUtils::AddEdge(g4_node->GetOutDataAnchor(0), g5_node->GetInDataAnchor(0));
  CreateSetOneNode(g6_op, dim, dtype);
  ge::NodePtr g6_node = graph->AddNode(g6_op);
  GraphUtils::AddEdge(g5_node->GetOutDataAnchor(0), g6_node->GetInDataAnchor(0));
  CreateSetOneNode(g7_op, dim, dtype);
  ge::NodePtr g7_node = graph->AddNode(g7_op);
  GraphUtils::AddEdge(g6_node->GetOutDataAnchor(0), g7_node->GetInDataAnchor(0));

  Status ret1 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g1_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret1 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g1_node, "tbe-custom");
  Status ret2 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g2_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret2 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g2_node, "tbe-custom");
  Status ret3 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g3_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret3 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g3_node, "tbe-custom");
  Status ret4 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g4_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret4 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g4_node, "tbe-custom");
  Status ret5 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g5_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret5 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g5_node, "tbe-custom");
  Status ret6 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g6_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret6 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g6_node, "tbe-custom");
  Status ret7 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g7_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret7 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g7_node, "tbe-custom");

  ASSERT_EQ(ret1, fe::SUCCESS);
  ASSERT_EQ(ret2, fe::SUCCESS);
  ASSERT_EQ(ret3, fe::SUCCESS);
  ASSERT_EQ(ret4, fe::SUCCESS);
  ASSERT_EQ(ret5, fe::SUCCESS);
  ASSERT_EQ(ret6, fe::SUCCESS);
  ASSERT_EQ(ret7, fe::SUCCESS);

  vector<ge::DataType> result_input = {DT_FLOAT, DT_BF16, DT_UINT8, DT_FLOAT, DT_FLOAT,
                                       DT_BF16, DT_FLOAT};
  vector<ge::DataType> result_output = {DT_FLOAT, DT_BF16, DT_UINT8, DT_FLOAT, DT_FLOAT,
                                       DT_BF16, DT_FLOAT};
  EXPECT_EQ(g1_op->GetInputDesc(0).GetDataType(), result_input[0]);
  EXPECT_EQ(g2_op->GetInputDesc(0).GetDataType(), result_input[1]);
  EXPECT_EQ(g3_op->GetInputDesc(0).GetDataType(), result_input[2]);
  EXPECT_EQ(g4_op->GetInputDesc(0).GetDataType(), result_input[3]);
  EXPECT_EQ(g5_op->GetInputDesc(0).GetDataType(), result_input[4]);
  EXPECT_EQ(g6_op->GetInputDesc(0).GetDataType(), result_input[5]);
  EXPECT_EQ(g7_op->GetInputDesc(0).GetDataType(), result_input[6]);


  EXPECT_EQ(g1_op->GetOutputDesc(0).GetDataType(), result_output[0]);
  EXPECT_EQ(g2_op->GetOutputDesc(0).GetDataType(), result_output[1]);
  EXPECT_EQ(g3_op->GetOutputDesc(0).GetDataType(), result_output[2]);
  EXPECT_EQ(g4_op->GetOutputDesc(0).GetDataType(), result_output[3]);
  EXPECT_EQ(g5_op->GetOutputDesc(0).GetDataType(), result_output[4]);
  EXPECT_EQ(g6_op->GetOutputDesc(0).GetDataType(), result_output[5]);
  EXPECT_EQ(g7_op->GetOutputDesc(0).GetDataType(), result_output[6]);
}

TEST_F(UTEST_fusion_engine_op_judge_precision_mode, set_nodes_IO_dtypes_float_allow_fp32_bf16_1)
{
  PlatformInfoManager::Instance().opti_compilation_infos_.SetSocVersion("Ascend910B2");
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = ALLOW_FP32_TO_BF16;
  op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g1_op = std::make_shared<OpDesc>("G1", "GOnlyfp32");
  OpDescPtr g2_op = std::make_shared<OpDesc>("G2", "GOnlyBf16");
  OpDescPtr g3_op = std::make_shared<OpDesc>("G3", "GOnlyFp16");
  OpDescPtr g4_op = std::make_shared<OpDesc>("G4", "GSupportFp32Bf16");
  OpDescPtr g5_op = std::make_shared<OpDesc>("G5", "GSupportfp32Fp16");
  OpDescPtr g6_op = std::make_shared<OpDesc>("G6", "GSupportBf16Fp16");
  OpDescPtr g7_op = std::make_shared<OpDesc>("G7", "G");

  vector<int64_t> dim({4, 33, 12, 16});
  ge::DataType dtype = DT_BF16;
  //add descriptor
  CreateSetOneNode(g1_op, dim, dtype);
  ge::NodePtr g1_node = graph->AddNode(g1_op);
  CreateSetOneNode(g2_op, dim, dtype);
  ge::NodePtr g2_node = graph->AddNode(g2_op);
  GraphUtils::AddEdge(g1_node->GetOutDataAnchor(0), g2_node->GetInDataAnchor(0));
  CreateSetOneNode(g3_op, dim, dtype);
  ge::NodePtr g3_node = graph->AddNode(g3_op);
  GraphUtils::AddEdge(g2_node->GetOutDataAnchor(0), g3_node->GetInDataAnchor(0));
  CreateSetOneNode(g4_op, dim, dtype);
  ge::NodePtr g4_node = graph->AddNode(g4_op);
  GraphUtils::AddEdge(g3_node->GetOutDataAnchor(0), g4_node->GetInDataAnchor(0));
  CreateSetOneNode(g5_op, dim, dtype);
  ge::NodePtr g5_node = graph->AddNode(g5_op);
  GraphUtils::AddEdge(g4_node->GetOutDataAnchor(0), g5_node->GetInDataAnchor(0));
  CreateSetOneNode(g6_op, dim, dtype);
  ge::NodePtr g6_node = graph->AddNode(g6_op);
  GraphUtils::AddEdge(g5_node->GetOutDataAnchor(0), g6_node->GetInDataAnchor(0));
  CreateSetOneNode(g7_op, dim, dtype);
  ge::NodePtr g7_node = graph->AddNode(g7_op);
  GraphUtils::AddEdge(g6_node->GetOutDataAnchor(0), g7_node->GetInDataAnchor(0));

  Status ret1 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g1_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret1 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g1_node, "tbe-custom");
  Status ret2 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g2_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret2 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g2_node, "tbe-custom");
  Status ret3 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g3_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret3 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g3_node, "tbe-custom");
  Status ret4 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g4_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret4 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g4_node, "tbe-custom");
  Status ret5 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g5_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret5 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g5_node, "tbe-custom");
  Status ret6 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g6_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret6 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g6_node, "tbe-custom");
  Status ret7 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g7_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret7 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g7_node, "tbe-custom");

  ASSERT_EQ(ret1, fe::SUCCESS);
  ASSERT_EQ(ret2, fe::SUCCESS);
  ASSERT_EQ(ret3, fe::SUCCESS);
  ASSERT_EQ(ret4, fe::SUCCESS);
  ASSERT_EQ(ret5, fe::SUCCESS);
  ASSERT_EQ(ret6, fe::SUCCESS);
  ASSERT_EQ(ret7, fe::SUCCESS);

  vector<ge::DataType> result_input = {DT_FLOAT, DT_BF16, DT_UINT8, DT_BF16, DT_FLOAT,
                                       DT_BF16, DT_BF16};
  vector<ge::DataType> result_output = {DT_FLOAT, DT_BF16, DT_UINT8, DT_BF16, DT_FLOAT,
                                       DT_BF16, DT_BF16};
  EXPECT_EQ(g1_op->GetInputDesc(0).GetDataType(), result_input[0]);
  EXPECT_EQ(g2_op->GetInputDesc(0).GetDataType(), result_input[1]);
  EXPECT_EQ(g3_op->GetInputDesc(0).GetDataType(), result_input[2]);
  EXPECT_EQ(g4_op->GetInputDesc(0).GetDataType(), result_input[3]);
  EXPECT_EQ(g5_op->GetInputDesc(0).GetDataType(), result_input[4]);
  EXPECT_EQ(g6_op->GetInputDesc(0).GetDataType(), result_input[5]);
  EXPECT_EQ(g7_op->GetInputDesc(0).GetDataType(), result_input[6]);


  EXPECT_EQ(g1_op->GetOutputDesc(0).GetDataType(), result_output[0]);
  EXPECT_EQ(g2_op->GetOutputDesc(0).GetDataType(), result_output[1]);
  EXPECT_EQ(g3_op->GetOutputDesc(0).GetDataType(), result_output[2]);
  EXPECT_EQ(g4_op->GetOutputDesc(0).GetDataType(), result_output[3]);
  EXPECT_EQ(g5_op->GetOutputDesc(0).GetDataType(), result_output[4]);
  EXPECT_EQ(g6_op->GetOutputDesc(0).GetDataType(), result_output[5]);
  EXPECT_EQ(g7_op->GetOutputDesc(0).GetDataType(), result_output[6]);
}

TEST_F(UTEST_fusion_engine_op_judge_precision_mode, set_nodes_IO_dtypes_float_allow_fp32_bf16_2)
{
  PlatformInfoManager::Instance().opti_compilation_infos_.SetSocVersion("Ascend910B2");
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = ALLOW_FP32_TO_BF16;
  op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g1_op = std::make_shared<OpDesc>("G1", "GOnlyfp32");
  OpDescPtr g2_op = std::make_shared<OpDesc>("G2", "GOnlyBf16");
  OpDescPtr g3_op = std::make_shared<OpDesc>("G3", "GOnlyFp16");
  OpDescPtr g4_op = std::make_shared<OpDesc>("G4", "GSupportFp32Bf16");
  OpDescPtr g5_op = std::make_shared<OpDesc>("G5", "GSupportfp32Fp16");
  OpDescPtr g6_op = std::make_shared<OpDesc>("G6", "GSupportBf16Fp16");
  OpDescPtr g7_op = std::make_shared<OpDesc>("G7", "G");

  vector<int64_t> dim({4, 33, 12, 16});
  ge::DataType dtype = DT_FLOAT16;
  //add descriptor
  CreateSetOneNode(g1_op, dim, dtype);
  ge::NodePtr g1_node = graph->AddNode(g1_op);
  CreateSetOneNode(g2_op, dim, dtype);
  ge::NodePtr g2_node = graph->AddNode(g2_op);
  GraphUtils::AddEdge(g1_node->GetOutDataAnchor(0), g2_node->GetInDataAnchor(0));
  CreateSetOneNode(g3_op, dim, dtype);
  ge::NodePtr g3_node = graph->AddNode(g3_op);
  GraphUtils::AddEdge(g2_node->GetOutDataAnchor(0), g3_node->GetInDataAnchor(0));
  CreateSetOneNode(g4_op, dim, dtype);
  ge::NodePtr g4_node = graph->AddNode(g4_op);
  GraphUtils::AddEdge(g3_node->GetOutDataAnchor(0), g4_node->GetInDataAnchor(0));
  CreateSetOneNode(g5_op, dim, dtype);
  ge::NodePtr g5_node = graph->AddNode(g5_op);
  GraphUtils::AddEdge(g4_node->GetOutDataAnchor(0), g5_node->GetInDataAnchor(0));
  CreateSetOneNode(g6_op, dim, dtype);
  ge::NodePtr g6_node = graph->AddNode(g6_op);
  GraphUtils::AddEdge(g5_node->GetOutDataAnchor(0), g6_node->GetInDataAnchor(0));
  CreateSetOneNode(g7_op, dim, dtype);
  ge::NodePtr g7_node = graph->AddNode(g7_op);
  GraphUtils::AddEdge(g6_node->GetOutDataAnchor(0), g7_node->GetInDataAnchor(0));

  Status ret1 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g1_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret1 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g1_node, "tbe-custom");
  Status ret2 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g2_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret2 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g2_node, "tbe-custom");
  Status ret3 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g3_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret3 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g3_node, "tbe-custom");
  Status ret4 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g4_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret4 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g4_node, "tbe-custom");
  Status ret5 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g5_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret5 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g5_node, "tbe-custom");
  Status ret6 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g6_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret6 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g6_node, "tbe-custom");
  Status ret7 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g7_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret7 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g7_node, "tbe-custom");

  ASSERT_EQ(ret1, fe::SUCCESS);
  ASSERT_EQ(ret2, fe::SUCCESS);
  ASSERT_EQ(ret3, fe::SUCCESS);
  ASSERT_EQ(ret4, fe::SUCCESS);
  ASSERT_EQ(ret5, fe::SUCCESS);
  ASSERT_EQ(ret6, fe::SUCCESS);
  ASSERT_EQ(ret7, fe::SUCCESS);

  vector<ge::DataType> result_input = {DT_FLOAT, DT_UINT8, DT_FLOAT16, DT_FLOAT, DT_FLOAT16,
                                       DT_FLOAT16, DT_FLOAT16};
  vector<ge::DataType> result_output = {DT_FLOAT, DT_UINT8, DT_FLOAT16, DT_FLOAT, DT_FLOAT16,
                                        DT_FLOAT16, DT_FLOAT16};
  EXPECT_EQ(g1_op->GetInputDesc(0).GetDataType(), result_input[0]);
  EXPECT_EQ(g2_op->GetInputDesc(0).GetDataType(), result_input[1]);
  EXPECT_EQ(g3_op->GetInputDesc(0).GetDataType(), result_input[2]);
  EXPECT_EQ(g4_op->GetInputDesc(0).GetDataType(), result_input[3]);
  EXPECT_EQ(g5_op->GetInputDesc(0).GetDataType(), result_input[4]);
  EXPECT_EQ(g6_op->GetInputDesc(0).GetDataType(), result_input[5]);
  EXPECT_EQ(g7_op->GetInputDesc(0).GetDataType(), result_input[6]);


  EXPECT_EQ(g1_op->GetOutputDesc(0).GetDataType(), result_output[0]);
  EXPECT_EQ(g2_op->GetOutputDesc(0).GetDataType(), result_output[1]);
  EXPECT_EQ(g3_op->GetOutputDesc(0).GetDataType(), result_output[2]);
  EXPECT_EQ(g4_op->GetOutputDesc(0).GetDataType(), result_output[3]);
  EXPECT_EQ(g5_op->GetOutputDesc(0).GetDataType(), result_output[4]);
  EXPECT_EQ(g6_op->GetOutputDesc(0).GetDataType(), result_output[5]);
  EXPECT_EQ(g7_op->GetOutputDesc(0).GetDataType(), result_output[6]);
}

TEST_F(UTEST_fusion_engine_op_judge_precision_mode, set_nodes_IO_dtypes_float_allow_fp32_to_lowprecision)
{
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = ALLOW_FP32_TO_LOWPRECISION;
  op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g1_op = std::make_shared<OpDesc>("G1", "GOnlyfp32");
  OpDescPtr g2_op = std::make_shared<OpDesc>("G2", "GOnlyBf16");
  OpDescPtr g3_op = std::make_shared<OpDesc>("G3", "GOnlyFp16");
  OpDescPtr g4_op = std::make_shared<OpDesc>("G4", "GSupportFp32Bf16");
  OpDescPtr g5_op = std::make_shared<OpDesc>("G5", "GSupportfp32Fp16");
  OpDescPtr g6_op = std::make_shared<OpDesc>("G6", "GSupportBf16Fp16");
  OpDescPtr g7_op = std::make_shared<OpDesc>("G7", "G");

  vector<int64_t> dim({4, 33, 12, 16});
  ge::DataType dtype = DT_FLOAT;
  //add descriptor
  CreateSetOneNode(g1_op, dim, dtype);
  ge::NodePtr g1_node = graph->AddNode(g1_op);
  CreateSetOneNode(g2_op, dim, dtype);
  ge::NodePtr g2_node = graph->AddNode(g2_op);
  GraphUtils::AddEdge(g1_node->GetOutDataAnchor(0), g2_node->GetInDataAnchor(0));
  CreateSetOneNode(g3_op, dim, dtype);
  ge::NodePtr g3_node = graph->AddNode(g3_op);
  GraphUtils::AddEdge(g2_node->GetOutDataAnchor(0), g3_node->GetInDataAnchor(0));
  CreateSetOneNode(g4_op, dim, dtype);
  ge::NodePtr g4_node = graph->AddNode(g4_op);
  GraphUtils::AddEdge(g3_node->GetOutDataAnchor(0), g4_node->GetInDataAnchor(0));
  CreateSetOneNode(g5_op, dim, dtype);
  ge::NodePtr g5_node = graph->AddNode(g5_op);
  GraphUtils::AddEdge(g4_node->GetOutDataAnchor(0), g5_node->GetInDataAnchor(0));
  CreateSetOneNode(g6_op, dim, dtype);
  ge::NodePtr g6_node = graph->AddNode(g6_op);
  GraphUtils::AddEdge(g5_node->GetOutDataAnchor(0), g6_node->GetInDataAnchor(0));
  CreateSetOneNode(g7_op, dim, dtype);
  ge::NodePtr g7_node = graph->AddNode(g7_op);
  GraphUtils::AddEdge(g6_node->GetOutDataAnchor(0), g7_node->GetInDataAnchor(0));

  Status ret1 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g1_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret1 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g1_node, "tbe-custom");
  Status ret2 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g2_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret2 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g2_node, "tbe-custom");
  Status ret3 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g3_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret3 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g3_node, "tbe-custom");
  Status ret4 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g4_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret4 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g4_node, "tbe-custom");
  Status ret5 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g5_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret5 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g5_node, "tbe-custom");
  Status ret6 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g6_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret6 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g6_node, "tbe-custom");
  Status ret7 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g7_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret7 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g7_node, "tbe-custom");

  ASSERT_EQ(ret1, fe::SUCCESS);
  ASSERT_EQ(ret2, fe::SUCCESS);
  ASSERT_EQ(ret3, fe::SUCCESS);
  ASSERT_EQ(ret4, fe::SUCCESS);
  ASSERT_EQ(ret5, fe::SUCCESS);
  ASSERT_EQ(ret6, fe::SUCCESS);
  ASSERT_EQ(ret7, fe::SUCCESS);

  vector<ge::DataType> result_input = {DT_FLOAT, DT_BF16, DT_FLOAT16, DT_FLOAT, DT_FLOAT,
                                       DT_BF16, DT_FLOAT};
  vector<ge::DataType> result_output = {DT_FLOAT, DT_BF16, DT_FLOAT16, DT_FLOAT, DT_FLOAT,
                                       DT_BF16, DT_FLOAT};
  EXPECT_EQ(g1_op->GetInputDesc(0).GetDataType(), result_input[0]);
  EXPECT_EQ(g2_op->GetInputDesc(0).GetDataType(), result_input[1]);
  EXPECT_EQ(g3_op->GetInputDesc(0).GetDataType(), result_input[2]);
  EXPECT_EQ(g4_op->GetInputDesc(0).GetDataType(), result_input[3]);
  EXPECT_EQ(g5_op->GetInputDesc(0).GetDataType(), result_input[4]);
  EXPECT_EQ(g6_op->GetInputDesc(0).GetDataType(), result_input[5]);
  EXPECT_EQ(g7_op->GetInputDesc(0).GetDataType(), result_input[6]);


  EXPECT_EQ(g1_op->GetOutputDesc(0).GetDataType(), result_output[0]);
  EXPECT_EQ(g2_op->GetOutputDesc(0).GetDataType(), result_output[1]);
  EXPECT_EQ(g3_op->GetOutputDesc(0).GetDataType(), result_output[2]);
  EXPECT_EQ(g4_op->GetOutputDesc(0).GetDataType(), result_output[3]);
  EXPECT_EQ(g5_op->GetOutputDesc(0).GetDataType(), result_output[4]);
  EXPECT_EQ(g6_op->GetOutputDesc(0).GetDataType(), result_output[5]);
  EXPECT_EQ(g7_op->GetOutputDesc(0).GetDataType(), result_output[6]);
}


TEST_F(UTEST_fusion_engine_op_judge_precision_mode, set_nodes_IO_dtypes_float_allow_fp32_to_lowprecision_1)
{
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = ALLOW_FP32_TO_LOWPRECISION;
  op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g1_op = std::make_shared<OpDesc>("G1", "GOnlyfp32");
  OpDescPtr g2_op = std::make_shared<OpDesc>("G2", "GOnlyBf16");
  OpDescPtr g3_op = std::make_shared<OpDesc>("G3", "GOnlyFp16");
  OpDescPtr g4_op = std::make_shared<OpDesc>("G4", "GSupportFp32Bf16");
  OpDescPtr g5_op = std::make_shared<OpDesc>("G5", "GSupportfp32Fp16");
  OpDescPtr g6_op = std::make_shared<OpDesc>("G6", "GSupportBf16Fp16");
  OpDescPtr g7_op = std::make_shared<OpDesc>("G7", "G");

  vector<int64_t> dim({4, 33, 12, 16});
  ge::DataType dtype = DT_BF16;
  //add descriptor
  CreateSetOneNode(g1_op, dim, dtype);
  ge::NodePtr g1_node = graph->AddNode(g1_op);
  CreateSetOneNode(g2_op, dim, dtype);
  ge::NodePtr g2_node = graph->AddNode(g2_op);
  GraphUtils::AddEdge(g1_node->GetOutDataAnchor(0), g2_node->GetInDataAnchor(0));
  CreateSetOneNode(g3_op, dim, dtype);
  ge::NodePtr g3_node = graph->AddNode(g3_op);
  GraphUtils::AddEdge(g2_node->GetOutDataAnchor(0), g3_node->GetInDataAnchor(0));
  CreateSetOneNode(g4_op, dim, dtype);
  ge::NodePtr g4_node = graph->AddNode(g4_op);
  GraphUtils::AddEdge(g3_node->GetOutDataAnchor(0), g4_node->GetInDataAnchor(0));
  CreateSetOneNode(g5_op, dim, dtype);
  ge::NodePtr g5_node = graph->AddNode(g5_op);
  GraphUtils::AddEdge(g4_node->GetOutDataAnchor(0), g5_node->GetInDataAnchor(0));
  CreateSetOneNode(g6_op, dim, dtype);
  ge::NodePtr g6_node = graph->AddNode(g6_op);
  GraphUtils::AddEdge(g5_node->GetOutDataAnchor(0), g6_node->GetInDataAnchor(0));
  CreateSetOneNode(g7_op, dim, dtype);
  ge::NodePtr g7_node = graph->AddNode(g7_op);
  GraphUtils::AddEdge(g6_node->GetOutDataAnchor(0), g7_node->GetInDataAnchor(0));

  Status ret1 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g1_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret1 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g1_node, "tbe-custom");
  Status ret2 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g2_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret2 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g2_node, "tbe-custom");
  Status ret3 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g3_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret3 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g3_node, "tbe-custom");
  Status ret4 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g4_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret4 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g4_node, "tbe-custom");
  Status ret5 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g5_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret5 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g5_node, "tbe-custom");
  Status ret6 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g6_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret6 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g6_node, "tbe-custom");
  Status ret7 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g7_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret7 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g7_node, "tbe-custom");

  ASSERT_EQ(ret1, fe::SUCCESS);
  ASSERT_EQ(ret2, fe::SUCCESS);
  ASSERT_EQ(ret3, fe::SUCCESS);
  ASSERT_EQ(ret4, fe::SUCCESS);
  ASSERT_EQ(ret5, fe::SUCCESS);
  ASSERT_EQ(ret6, fe::SUCCESS);
  ASSERT_EQ(ret7, fe::SUCCESS);

  vector<ge::DataType> result_input = {DT_FLOAT, DT_BF16, DT_FLOAT16, DT_BF16, DT_FLOAT,
                                       DT_BF16, DT_BF16};
  vector<ge::DataType> result_output = {DT_FLOAT, DT_BF16, DT_FLOAT16, DT_BF16, DT_FLOAT,
                                       DT_BF16, DT_BF16};
  EXPECT_EQ(g1_op->GetInputDesc(0).GetDataType(), result_input[0]);
  EXPECT_EQ(g2_op->GetInputDesc(0).GetDataType(), result_input[1]);
  EXPECT_EQ(g3_op->GetInputDesc(0).GetDataType(), result_input[2]);
  EXPECT_EQ(g4_op->GetInputDesc(0).GetDataType(), result_input[3]);
  EXPECT_EQ(g5_op->GetInputDesc(0).GetDataType(), result_input[4]);
  EXPECT_EQ(g6_op->GetInputDesc(0).GetDataType(), result_input[5]);
  EXPECT_EQ(g7_op->GetInputDesc(0).GetDataType(), result_input[6]);


  EXPECT_EQ(g1_op->GetOutputDesc(0).GetDataType(), result_output[0]);
  EXPECT_EQ(g2_op->GetOutputDesc(0).GetDataType(), result_output[1]);
  EXPECT_EQ(g3_op->GetOutputDesc(0).GetDataType(), result_output[2]);
  EXPECT_EQ(g4_op->GetOutputDesc(0).GetDataType(), result_output[3]);
  EXPECT_EQ(g5_op->GetOutputDesc(0).GetDataType(), result_output[4]);
  EXPECT_EQ(g6_op->GetOutputDesc(0).GetDataType(), result_output[5]);
  EXPECT_EQ(g7_op->GetOutputDesc(0).GetDataType(), result_output[6]);
}

TEST_F(UTEST_fusion_engine_op_judge_precision_mode, set_nodes_IO_dtypes_float_allow_fp32_to_lowprecision_2)
{
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = ALLOW_FP32_TO_LOWPRECISION;
  op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g1_op = std::make_shared<OpDesc>("G1", "GOnlyfp32");
  OpDescPtr g2_op = std::make_shared<OpDesc>("G2", "GOnlyBf16");
  OpDescPtr g3_op = std::make_shared<OpDesc>("G3", "GOnlyFp16");
  OpDescPtr g4_op = std::make_shared<OpDesc>("G4", "GSupportFp32Bf16");
  OpDescPtr g5_op = std::make_shared<OpDesc>("G5", "GSupportfp32Fp16");
  OpDescPtr g6_op = std::make_shared<OpDesc>("G6", "GSupportBf16Fp16");
  OpDescPtr g7_op = std::make_shared<OpDesc>("G7", "G");

  vector<int64_t> dim({4, 33, 12, 16});
  ge::DataType dtype = DT_FLOAT16;
  //add descriptor
  CreateSetOneNode(g1_op, dim, dtype);
  ge::NodePtr g1_node = graph->AddNode(g1_op);
  CreateSetOneNode(g2_op, dim, dtype);
  ge::NodePtr g2_node = graph->AddNode(g2_op);
  GraphUtils::AddEdge(g1_node->GetOutDataAnchor(0), g2_node->GetInDataAnchor(0));
  CreateSetOneNode(g3_op, dim, dtype);
  ge::NodePtr g3_node = graph->AddNode(g3_op);
  GraphUtils::AddEdge(g2_node->GetOutDataAnchor(0), g3_node->GetInDataAnchor(0));
  CreateSetOneNode(g4_op, dim, dtype);
  ge::NodePtr g4_node = graph->AddNode(g4_op);
  GraphUtils::AddEdge(g3_node->GetOutDataAnchor(0), g4_node->GetInDataAnchor(0));
  CreateSetOneNode(g5_op, dim, dtype);
  ge::NodePtr g5_node = graph->AddNode(g5_op);
  GraphUtils::AddEdge(g4_node->GetOutDataAnchor(0), g5_node->GetInDataAnchor(0));
  CreateSetOneNode(g6_op, dim, dtype);
  ge::NodePtr g6_node = graph->AddNode(g6_op);
  GraphUtils::AddEdge(g5_node->GetOutDataAnchor(0), g6_node->GetInDataAnchor(0));
  CreateSetOneNode(g7_op, dim, dtype);
  ge::NodePtr g7_node = graph->AddNode(g7_op);
  GraphUtils::AddEdge(g6_node->GetOutDataAnchor(0), g7_node->GetInDataAnchor(0));

  Status ret1 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g1_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret1 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g1_node, "tbe-custom");
  Status ret2 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g2_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret2 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g2_node, "tbe-custom");
  Status ret3 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g3_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret3 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g3_node, "tbe-custom");
  Status ret4 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g4_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret4 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g4_node, "tbe-custom");
  Status ret5 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g5_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret5 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g5_node, "tbe-custom");
  Status ret6 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g6_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret6 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g6_node, "tbe-custom");
  Status ret7 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g7_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret7 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g7_node, "tbe-custom");

  ASSERT_EQ(ret1, fe::SUCCESS);
  ASSERT_EQ(ret2, fe::SUCCESS);
  ASSERT_EQ(ret3, fe::SUCCESS);
  ASSERT_EQ(ret4, fe::SUCCESS);
  ASSERT_EQ(ret5, fe::SUCCESS);
  ASSERT_EQ(ret6, fe::SUCCESS);
  ASSERT_EQ(ret7, fe::SUCCESS);

  vector<ge::DataType> result_input = {DT_FLOAT, DT_UINT8, DT_FLOAT16, DT_FLOAT, DT_FLOAT16,
                                       DT_FLOAT16, DT_FLOAT16};
  vector<ge::DataType> result_output = {DT_FLOAT, DT_UINT8, DT_FLOAT16, DT_FLOAT, DT_FLOAT16,
                                       DT_FLOAT16, DT_FLOAT16};
  EXPECT_EQ(g1_op->GetInputDesc(0).GetDataType(), result_input[0]);
  EXPECT_EQ(g2_op->GetInputDesc(0).GetDataType(), result_input[1]);
  EXPECT_EQ(g3_op->GetInputDesc(0).GetDataType(), result_input[2]);
  EXPECT_EQ(g4_op->GetInputDesc(0).GetDataType(), result_input[3]);
  EXPECT_EQ(g5_op->GetInputDesc(0).GetDataType(), result_input[4]);
  EXPECT_EQ(g6_op->GetInputDesc(0).GetDataType(), result_input[5]);
  EXPECT_EQ(g7_op->GetInputDesc(0).GetDataType(), result_input[6]);


  EXPECT_EQ(g1_op->GetOutputDesc(0).GetDataType(), result_output[0]);
  EXPECT_EQ(g2_op->GetOutputDesc(0).GetDataType(), result_output[1]);
  EXPECT_EQ(g3_op->GetOutputDesc(0).GetDataType(), result_output[2]);
  EXPECT_EQ(g4_op->GetOutputDesc(0).GetDataType(), result_output[3]);
  EXPECT_EQ(g5_op->GetOutputDesc(0).GetDataType(), result_output[4]);
  EXPECT_EQ(g6_op->GetOutputDesc(0).GetDataType(), result_output[5]);
  EXPECT_EQ(g7_op->GetOutputDesc(0).GetDataType(), result_output[6]);
}

TEST_F(UTEST_fusion_engine_op_judge_precision_mode, set_nodes_IO_dtypes_float_force_fp32)
{
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = FORCE_FP32;
  op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g1_op = std::make_shared<OpDesc>("G1", "GOnlyfp32");
  OpDescPtr g2_op = std::make_shared<OpDesc>("G2", "GOnlyBf16");
  OpDescPtr g3_op = std::make_shared<OpDesc>("G3", "GOnlyFp16");
  OpDescPtr g4_op = std::make_shared<OpDesc>("G4", "GSupportFp32Bf16");
  OpDescPtr g5_op = std::make_shared<OpDesc>("G5", "GSupportfp32Fp16");
  OpDescPtr g6_op = std::make_shared<OpDesc>("G6", "GSupportBf16Fp16");
  OpDescPtr g7_op = std::make_shared<OpDesc>("G7", "G");

  vector<int64_t> dim({4, 33, 12, 16});
  ge::DataType dtype = DT_FLOAT;
  //add descriptor
  CreateSetOneNode(g1_op, dim, dtype);
  ge::NodePtr g1_node = graph->AddNode(g1_op);
  CreateSetOneNode(g2_op, dim, dtype);
  ge::NodePtr g2_node = graph->AddNode(g2_op);
  GraphUtils::AddEdge(g1_node->GetOutDataAnchor(0), g2_node->GetInDataAnchor(0));
  CreateSetOneNode(g3_op, dim, dtype);
  ge::NodePtr g3_node = graph->AddNode(g3_op);
  GraphUtils::AddEdge(g2_node->GetOutDataAnchor(0), g3_node->GetInDataAnchor(0));
  CreateSetOneNode(g4_op, dim, dtype);
  ge::NodePtr g4_node = graph->AddNode(g4_op);
  GraphUtils::AddEdge(g3_node->GetOutDataAnchor(0), g4_node->GetInDataAnchor(0));
  CreateSetOneNode(g5_op, dim, dtype);
  ge::NodePtr g5_node = graph->AddNode(g5_op);
  GraphUtils::AddEdge(g4_node->GetOutDataAnchor(0), g5_node->GetInDataAnchor(0));
  CreateSetOneNode(g6_op, dim, dtype);
  ge::NodePtr g6_node = graph->AddNode(g6_op);
  GraphUtils::AddEdge(g5_node->GetOutDataAnchor(0), g6_node->GetInDataAnchor(0));
  CreateSetOneNode(g7_op, dim, dtype);
  ge::NodePtr g7_node = graph->AddNode(g7_op);
  GraphUtils::AddEdge(g6_node->GetOutDataAnchor(0), g7_node->GetInDataAnchor(0));

  Status ret1 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g1_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret1 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g1_node, "tbe-custom");
  Status ret2 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g2_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret2 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g2_node, "tbe-custom");
  Status ret3 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g3_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret3 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g3_node, "tbe-custom");
  Status ret4 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g4_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret4 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g4_node, "tbe-custom");
  Status ret5 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g5_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret5 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g5_node, "tbe-custom");
  Status ret6 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g6_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret6 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g6_node, "tbe-custom");
  Status ret7 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g7_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret7 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g7_node, "tbe-custom");

  ASSERT_EQ(ret1, fe::SUCCESS);
  ASSERT_EQ(ret2, fe::SUCCESS);
  ASSERT_EQ(ret3, fe::SUCCESS);
  ASSERT_EQ(ret4, fe::SUCCESS);
  ASSERT_EQ(ret5, fe::SUCCESS);
  ASSERT_EQ(ret6, fe::SUCCESS);
  ASSERT_EQ(ret7, fe::SUCCESS);

  vector<ge::DataType> result_input = {DT_FLOAT, DT_UINT8, DT_FLOAT16, DT_FLOAT, DT_FLOAT,
                                       DT_FLOAT16, DT_FLOAT};
  vector<ge::DataType> result_output = {DT_FLOAT, DT_UINT8, DT_FLOAT16, DT_FLOAT, DT_FLOAT,
                                       DT_FLOAT16, DT_FLOAT};
  EXPECT_EQ(g1_op->GetInputDesc(0).GetDataType(), result_input[0]);
  EXPECT_EQ(g2_op->GetInputDesc(0).GetDataType(), result_input[1]);
  EXPECT_EQ(g3_op->GetInputDesc(0).GetDataType(), result_input[2]);
  EXPECT_EQ(g4_op->GetInputDesc(0).GetDataType(), result_input[3]);
  EXPECT_EQ(g5_op->GetInputDesc(0).GetDataType(), result_input[4]);
  EXPECT_EQ(g6_op->GetInputDesc(0).GetDataType(), result_input[5]);
  EXPECT_EQ(g7_op->GetInputDesc(0).GetDataType(), result_input[6]);


  EXPECT_EQ(g1_op->GetOutputDesc(0).GetDataType(), result_output[0]);
  EXPECT_EQ(g2_op->GetOutputDesc(0).GetDataType(), result_output[1]);
  EXPECT_EQ(g3_op->GetOutputDesc(0).GetDataType(), result_output[2]);
  EXPECT_EQ(g4_op->GetOutputDesc(0).GetDataType(), result_output[3]);
  EXPECT_EQ(g5_op->GetOutputDesc(0).GetDataType(), result_output[4]);
  EXPECT_EQ(g6_op->GetOutputDesc(0).GetDataType(), result_output[5]);
  EXPECT_EQ(g7_op->GetOutputDesc(0).GetDataType(), result_output[6]);
}

TEST_F(UTEST_fusion_engine_op_judge_precision_mode, set_nodes_IO_dtypes_keep_oringin)
{
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = MUST_KEEP_ORIGIN_DTYPE;
  op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g1_op = std::make_shared<OpDesc>("G1", "GOnlyfp32");
  OpDescPtr g2_op = std::make_shared<OpDesc>("G2", "GOnlyBf16");
  OpDescPtr g3_op = std::make_shared<OpDesc>("G3", "GOnlyFp16");
  OpDescPtr g4_op = std::make_shared<OpDesc>("G4", "GSupportFp32Bf16");
  OpDescPtr g5_op = std::make_shared<OpDesc>("G5", "GSupportfp32Fp16");
  OpDescPtr g6_op = std::make_shared<OpDesc>("G6", "GSupportBf16Fp16");
  OpDescPtr g7_op = std::make_shared<OpDesc>("G7", "G");

  vector<int64_t> dim({4, 33, 12, 16});
  ge::DataType dtype = DT_FLOAT;
  //add descriptor
  CreateSetOneNode(g1_op, dim, dtype);
  ge::NodePtr g1_node = graph->AddNode(g1_op);
  CreateSetOneNode(g2_op, dim, dtype);
  ge::NodePtr g2_node = graph->AddNode(g2_op);
  GraphUtils::AddEdge(g1_node->GetOutDataAnchor(0), g2_node->GetInDataAnchor(0));
  CreateSetOneNode(g3_op, dim, dtype);
  ge::NodePtr g3_node = graph->AddNode(g3_op);
  GraphUtils::AddEdge(g2_node->GetOutDataAnchor(0), g3_node->GetInDataAnchor(0));
  CreateSetOneNode(g4_op, dim, dtype);
  ge::NodePtr g4_node = graph->AddNode(g4_op);
  GraphUtils::AddEdge(g3_node->GetOutDataAnchor(0), g4_node->GetInDataAnchor(0));
  CreateSetOneNode(g5_op, dim, dtype);
  ge::NodePtr g5_node = graph->AddNode(g5_op);
  GraphUtils::AddEdge(g4_node->GetOutDataAnchor(0), g5_node->GetInDataAnchor(0));
  CreateSetOneNode(g6_op, dim, dtype);
  ge::NodePtr g6_node = graph->AddNode(g6_op);
  GraphUtils::AddEdge(g5_node->GetOutDataAnchor(0), g6_node->GetInDataAnchor(0));
  CreateSetOneNode(g7_op, dim, dtype);
  ge::NodePtr g7_node = graph->AddNode(g7_op);
  GraphUtils::AddEdge(g6_node->GetOutDataAnchor(0), g7_node->GetInDataAnchor(0));

  Status ret1 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g1_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret1 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g1_node, "tbe-custom");
  Status ret2 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g2_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret2 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g2_node, "tbe-custom");
  Status ret3 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g3_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret3 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g3_node, "tbe-custom");
  Status ret4 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g4_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret4 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g4_node, "tbe-custom");
  Status ret5 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g5_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret5 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g5_node, "tbe-custom");
  Status ret6 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g6_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret6 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g6_node, "tbe-custom");
  Status ret7 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g7_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret7 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g7_node, "tbe-custom");

  ASSERT_EQ(ret1, fe::SUCCESS);
  ASSERT_EQ(ret2, fe::SUCCESS);
  ASSERT_EQ(ret3, fe::SUCCESS);
  ASSERT_EQ(ret4, fe::SUCCESS);
  ASSERT_EQ(ret5, fe::SUCCESS);
  ASSERT_EQ(ret6, fe::SUCCESS);
  ASSERT_EQ(ret7, fe::SUCCESS);

  vector<ge::DataType> result_input = {DT_FLOAT, DT_BF16, DT_FLOAT16, DT_FLOAT, DT_FLOAT,
                                       DT_BF16, DT_FLOAT};
  vector<ge::DataType> result_output = {DT_FLOAT, DT_BF16, DT_FLOAT16, DT_FLOAT, DT_FLOAT,
                                       DT_BF16, DT_FLOAT};
  EXPECT_EQ(g1_op->GetInputDesc(0).GetDataType(), result_input[0]);
  //EXPECT_EQ(g2_op->GetInputDesc(0).GetDataType(), result_input[1]);
  //EXPECT_EQ(g3_op->GetInputDesc(0).GetDataType(), result_input[2]);
  EXPECT_EQ(g4_op->GetInputDesc(0).GetDataType(), result_input[3]);
  EXPECT_EQ(g5_op->GetInputDesc(0).GetDataType(), result_input[4]);
  //EXPECT_EQ(g6_op->GetInputDesc(0).GetDataType(), result_input[5]);
  EXPECT_EQ(g7_op->GetInputDesc(0).GetDataType(), result_input[6]);


  EXPECT_EQ(g1_op->GetOutputDesc(0).GetDataType(), result_output[0]);
  //EXPECT_EQ(g2_op->GetOutputDesc(0).GetDataType(), result_output[1]);
  //EXPECT_EQ(g3_op->GetOutputDesc(0).GetDataType(), result_output[2]);
  EXPECT_EQ(g4_op->GetOutputDesc(0).GetDataType(), result_output[3]);
  EXPECT_EQ(g5_op->GetOutputDesc(0).GetDataType(), result_output[4]);
  //EXPECT_EQ(g6_op->GetOutputDesc(0).GetDataType(), result_output[5]);
  EXPECT_EQ(g7_op->GetOutputDesc(0).GetDataType(), result_output[6]);
}

TEST_F(UTEST_fusion_engine_op_judge_precision_mode, set_nodes_IO_dtypes_keep_oringin_1)
{
  Configuration &config = Configuration::Instance(fe::AI_CORE_NAME);
  config.enable_aclnn_ = true;
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = MUST_KEEP_ORIGIN_DTYPE;
  op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g1_op = std::make_shared<OpDesc>("G1", "GOnlyfp32");
  OpDescPtr g2_op = std::make_shared<OpDesc>("G2", "GOnlyBf16");
  OpDescPtr g3_op = std::make_shared<OpDesc>("G3", "GOnlyFp16");
  OpDescPtr g4_op = std::make_shared<OpDesc>("G4", "GSupportFp32Bf16");
  OpDescPtr g5_op = std::make_shared<OpDesc>("G5", "GSupportfp32Fp16");
  OpDescPtr g6_op = std::make_shared<OpDesc>("G6", "GSupportBf16Fp16");
  OpDescPtr g7_op = std::make_shared<OpDesc>("G7", "G");

  vector<int64_t> dim({4, 33, 12, 16});
  ge::DataType dtype = DT_BF16;
  //add descriptor
  CreateSetOneNode(g1_op, dim, dtype);
  ge::NodePtr g1_node = graph->AddNode(g1_op);
  CreateSetOneNode(g2_op, dim, dtype);
  ge::NodePtr g2_node = graph->AddNode(g2_op);
  GraphUtils::AddEdge(g1_node->GetOutDataAnchor(0), g2_node->GetInDataAnchor(0));
  CreateSetOneNode(g3_op, dim, dtype);
  ge::NodePtr g3_node = graph->AddNode(g3_op);
  GraphUtils::AddEdge(g2_node->GetOutDataAnchor(0), g3_node->GetInDataAnchor(0));
  CreateSetOneNode(g4_op, dim, dtype);
  ge::NodePtr g4_node = graph->AddNode(g4_op);
  GraphUtils::AddEdge(g3_node->GetOutDataAnchor(0), g4_node->GetInDataAnchor(0));
  CreateSetOneNode(g5_op, dim, dtype);
  ge::NodePtr g5_node = graph->AddNode(g5_op);
  GraphUtils::AddEdge(g4_node->GetOutDataAnchor(0), g5_node->GetInDataAnchor(0));
  CreateSetOneNode(g6_op, dim, dtype);
  ge::NodePtr g6_node = graph->AddNode(g6_op);
  GraphUtils::AddEdge(g5_node->GetOutDataAnchor(0), g6_node->GetInDataAnchor(0));
  CreateSetOneNode(g7_op, dim, dtype);
  ge::NodePtr g7_node = graph->AddNode(g7_op);
  GraphUtils::AddEdge(g6_node->GetOutDataAnchor(0), g7_node->GetInDataAnchor(0));

  Status ret1 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g1_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret1 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g1_node, "tbe-custom");
  Status ret2 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g2_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret2 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g2_node, "tbe-custom");
  Status ret3 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g3_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret3 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g3_node, "tbe-custom");
  Status ret4 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g4_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret4 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g4_node, "tbe-custom");
  Status ret5 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g5_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret5 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g5_node, "tbe-custom");
  Status ret6 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g6_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret6 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g6_node, "tbe-custom");
  Status ret7 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g7_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret7 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g7_node, "tbe-custom");

  ASSERT_EQ(ret1, fe::SUCCESS);
  ASSERT_EQ(ret2, fe::SUCCESS);
  ASSERT_EQ(ret3, fe::SUCCESS);
  ASSERT_EQ(ret4, fe::SUCCESS);
  ASSERT_EQ(ret5, fe::SUCCESS);
  ASSERT_EQ(ret6, fe::SUCCESS);
  ASSERT_EQ(ret7, fe::SUCCESS);

  vector<ge::DataType> result_input = {DT_FLOAT, DT_BF16, DT_FLOAT16, DT_BF16, DT_FLOAT,
                                       DT_BF16, DT_BF16};
  vector<ge::DataType> result_output = {DT_FLOAT, DT_BF16, DT_FLOAT16, DT_BF16, DT_FLOAT,
                                       DT_BF16, DT_BF16};
  EXPECT_EQ(g1_op->GetInputDesc(0).GetDataType(), result_input[0]);
  EXPECT_EQ(g2_op->GetInputDesc(0).GetDataType(), result_input[1]);
  //EXPECT_EQ(g3_op->GetInputDesc(0).GetDataType(), result_input[2]);
  EXPECT_EQ(g4_op->GetInputDesc(0).GetDataType(), result_input[3]);
  EXPECT_EQ(g5_op->GetInputDesc(0).GetDataType(), result_input[4]);
  EXPECT_EQ(g6_op->GetInputDesc(0).GetDataType(), result_input[5]);
  EXPECT_EQ(g7_op->GetInputDesc(0).GetDataType(), result_input[6]);


  EXPECT_EQ(g1_op->GetOutputDesc(0).GetDataType(), result_output[0]);
  EXPECT_EQ(g2_op->GetOutputDesc(0).GetDataType(), result_output[1]);
  //EXPECT_EQ(g3_op->GetOutputDesc(0).GetDataType(), result_output[2]);
  EXPECT_EQ(g4_op->GetOutputDesc(0).GetDataType(), result_output[3]);
  EXPECT_EQ(g5_op->GetOutputDesc(0).GetDataType(), result_output[4]);
  EXPECT_EQ(g6_op->GetOutputDesc(0).GetDataType(), result_output[5]);
  EXPECT_EQ(g7_op->GetOutputDesc(0).GetDataType(), result_output[6]);
  config.enable_aclnn_ = false;
}

TEST_F(UTEST_fusion_engine_op_judge_precision_mode, set_nodes_IO_dtypes_keep_oringin_2)
{
  Configuration &config = Configuration::Instance(fe::AI_CORE_NAME);
  config.enable_aclnn_ = true;
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = MUST_KEEP_ORIGIN_DTYPE;
  op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g1_op = std::make_shared<OpDesc>("G1", "GOnlyfp32");
  OpDescPtr g2_op = std::make_shared<OpDesc>("G2", "GOnlyBf16");
  OpDescPtr g3_op = std::make_shared<OpDesc>("G3", "GOnlyFp16");
  OpDescPtr g4_op = std::make_shared<OpDesc>("G4", "GSupportFp32Bf16");
  OpDescPtr g5_op = std::make_shared<OpDesc>("G5", "GSupportfp32Fp16");
  OpDescPtr g6_op = std::make_shared<OpDesc>("G6", "GSupportBf16Fp16");
  OpDescPtr g7_op = std::make_shared<OpDesc>("G7", "G");

  vector<int64_t> dim({4, 33, 12, 16});
  ge::DataType dtype = DT_FLOAT16;
  //add descriptor
  CreateSetOneNode(g1_op, dim, dtype);
  ge::NodePtr g1_node = graph->AddNode(g1_op);
  CreateSetOneNode(g2_op, dim, dtype);
  ge::NodePtr g2_node = graph->AddNode(g2_op);
  GraphUtils::AddEdge(g1_node->GetOutDataAnchor(0), g2_node->GetInDataAnchor(0));
  CreateSetOneNode(g3_op, dim, dtype);
  ge::NodePtr g3_node = graph->AddNode(g3_op);
  GraphUtils::AddEdge(g2_node->GetOutDataAnchor(0), g3_node->GetInDataAnchor(0));
  CreateSetOneNode(g4_op, dim, dtype);
  ge::NodePtr g4_node = graph->AddNode(g4_op);
  GraphUtils::AddEdge(g3_node->GetOutDataAnchor(0), g4_node->GetInDataAnchor(0));
  CreateSetOneNode(g5_op, dim, dtype);
  ge::NodePtr g5_node = graph->AddNode(g5_op);
  GraphUtils::AddEdge(g4_node->GetOutDataAnchor(0), g5_node->GetInDataAnchor(0));
  CreateSetOneNode(g6_op, dim, dtype);
  ge::NodePtr g6_node = graph->AddNode(g6_op);
  GraphUtils::AddEdge(g5_node->GetOutDataAnchor(0), g6_node->GetInDataAnchor(0));
  CreateSetOneNode(g7_op, dim, dtype);
  ge::NodePtr g7_node = graph->AddNode(g7_op);
  GraphUtils::AddEdge(g6_node->GetOutDataAnchor(0), g7_node->GetInDataAnchor(0));

  Status ret1 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g1_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret1 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g1_node, "tbe-custom");
  Status ret2 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g2_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret2 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g2_node, "tbe-custom");
  Status ret3 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g3_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret3 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g3_node, "tbe-custom");
  Status ret4 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g4_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret4 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g4_node, "tbe-custom");
  Status ret5 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g5_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret5 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g5_node, "tbe-custom");
  Status ret6 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g6_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret6 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g6_node, "tbe-custom");
  Status ret7 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g7_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret7 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g7_node, "tbe-custom");

  ASSERT_EQ(ret1, fe::SUCCESS);
  ASSERT_EQ(ret2, fe::SUCCESS);
  ASSERT_EQ(ret3, fe::SUCCESS);
  ASSERT_EQ(ret4, fe::SUCCESS);
  ASSERT_EQ(ret5, fe::SUCCESS);
  ASSERT_EQ(ret6, fe::SUCCESS);
  ASSERT_EQ(ret7, fe::SUCCESS);

  vector<ge::DataType> result_input = {DT_FLOAT, DT_UINT8, DT_FLOAT16, DT_FLOAT, DT_FLOAT16,
                                       DT_FLOAT16, DT_FLOAT16};
  vector<ge::DataType> result_output = {DT_FLOAT, DT_UINT8, DT_FLOAT16, DT_FLOAT, DT_FLOAT16,
                                       DT_FLOAT16, DT_FLOAT16};
  EXPECT_EQ(g1_op->GetInputDesc(0).GetDataType(), result_input[0]);
  EXPECT_EQ(g2_op->GetInputDesc(0).GetDataType(), result_input[1]);
  EXPECT_EQ(g3_op->GetInputDesc(0).GetDataType(), result_input[2]);
  EXPECT_EQ(g4_op->GetInputDesc(0).GetDataType(), result_input[3]);
  EXPECT_EQ(g5_op->GetInputDesc(0).GetDataType(), result_input[4]);
  EXPECT_EQ(g6_op->GetInputDesc(0).GetDataType(), result_input[5]);
  EXPECT_EQ(g7_op->GetInputDesc(0).GetDataType(), result_input[6]);


  EXPECT_EQ(g1_op->GetOutputDesc(0).GetDataType(), result_output[0]);
  EXPECT_EQ(g2_op->GetOutputDesc(0).GetDataType(), result_output[1]);
  EXPECT_EQ(g3_op->GetOutputDesc(0).GetDataType(), result_output[2]);
  EXPECT_EQ(g4_op->GetOutputDesc(0).GetDataType(), result_output[3]);
  EXPECT_EQ(g5_op->GetOutputDesc(0).GetDataType(), result_output[4]);
  EXPECT_EQ(g6_op->GetOutputDesc(0).GetDataType(), result_output[5]);
  EXPECT_EQ(g7_op->GetOutputDesc(0).GetDataType(), result_output[6]);
  config.enable_aclnn_ = false;
}

TEST_F(UTEST_fusion_engine_op_judge_precision_mode, set_nodes_IO_dtypes_keep_oringin_3)
{
  Configuration &config = Configuration::Instance(fe::AI_CORE_NAME);
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = MUST_KEEP_ORIGIN_DTYPE;
  op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  OpDescPtr g3_op = std::make_shared<OpDesc>("Mul", "Mul");
  (void)ge::AttrUtils::SetBool(g3_op, kMustPromoteFlag, true);
 
  vector<int64_t> dim({4, 33, 12, 16});
  ge::DataType dtype = DT_INT32;
  ge::DataType dtype2 = DT_FLOAT16;
  CreatePromoteNode(g3_op, dim, dtype, dtype2);
  ge::NodePtr g3_node = graph->AddNode(g3_op);
  Status ret3 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g3_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret3 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g3_node, "tbe-custom");

  ASSERT_EQ(ret3, fe::SUCCESS);
  EXPECT_EQ(g3_op->GetInputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(g3_op->GetInputDesc(1).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(g3_op->GetOutputDesc(0).GetDataType(), DT_FLOAT16);
}

TEST_F(UTEST_fusion_engine_op_judge_precision_mode, set_nodes_IO_dtypes_mix_fp16)
{
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = ALLOW_MIX_PRECISION_FP16;
  op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g1_op = std::make_shared<OpDesc>("G1", "GOnlyfp32");
  OpDescPtr g2_op = std::make_shared<OpDesc>("G2", "GOnlyBf16");
  OpDescPtr g3_op = std::make_shared<OpDesc>("G3", "GBlackOnlyFp16");
  OpDescPtr g4_op = std::make_shared<OpDesc>("G4", "GWhiteOnlyFp16");
  OpDescPtr g5_op = std::make_shared<OpDesc>("G5", "GOnlyFp16");
  OpDescPtr g6_op = std::make_shared<OpDesc>("G6", "GBlackSupportFp32Bf16");
  OpDescPtr g7_op = std::make_shared<OpDesc>("G7", "GWhiteSupportFp32Bf16");
  OpDescPtr g8_op = std::make_shared<OpDesc>("G8", "GSupportFp32Bf16");
  OpDescPtr g9_op = std::make_shared<OpDesc>("G9", "GBlackSupportfp32Fp16");
  OpDescPtr g10_op = std::make_shared<OpDesc>("G10", "GWhiteSupportfp32Fp16");
  OpDescPtr g11_op = std::make_shared<OpDesc>("G11", "GSupportfp32Fp16");
  OpDescPtr g12_op = std::make_shared<OpDesc>("G12", "GBlackSupportBf16Fp16");
  OpDescPtr g13_op = std::make_shared<OpDesc>("G13", "GWhiteSupportBf16Fp16");
  OpDescPtr g14_op = std::make_shared<OpDesc>("G14", "GSupportBf16Fp16");
  OpDescPtr g15_op = std::make_shared<OpDesc>("G15", "GBlack");
  OpDescPtr g16_op = std::make_shared<OpDesc>("G16", "GWhite");
  OpDescPtr g17_op = std::make_shared<OpDesc>("G17", "G");

  vector<int64_t> dim({4, 33, 12, 16});
  ge::DataType dtype = DT_FLOAT;
  //add descriptor
  CreateSetOneNode(g1_op, dim, dtype);
  ge::NodePtr g1_node = graph->AddNode(g1_op);
  CreateSetOneNode(g2_op, dim, dtype);
  ge::NodePtr g2_node = graph->AddNode(g2_op);
  CreateSetOneNode(g3_op, dim, dtype);
  ge::NodePtr g3_node = graph->AddNode(g3_op);
  CreateSetOneNode(g4_op, dim, dtype);
  ge::NodePtr g4_node = graph->AddNode(g4_op);
  CreateSetOneNode(g5_op, dim, dtype);
  ge::NodePtr g5_node = graph->AddNode(g5_op);
  CreateSetOneNode(g6_op, dim, dtype);
  ge::NodePtr g6_node = graph->AddNode(g6_op);
  CreateSetOneNode(g7_op, dim, dtype);
  ge::NodePtr g7_node = graph->AddNode(g7_op);
  CreateSetOneNode(g8_op, dim, dtype);
  ge::NodePtr g8_node = graph->AddNode(g8_op);
  CreateSetOneNode(g9_op, dim, dtype);
  ge::NodePtr g9_node = graph->AddNode(g9_op);
  CreateSetOneNode(g10_op, dim, dtype);
  ge::NodePtr g10_node = graph->AddNode(g10_op);
  CreateSetOneNode(g11_op, dim, dtype);
  ge::NodePtr g11_node = graph->AddNode(g11_op);
  CreateSetOneNode(g12_op, dim, dtype);
  ge::NodePtr g12_node = graph->AddNode(g12_op);
  CreateSetOneNode(g13_op, dim, dtype);
  ge::NodePtr g13_node = graph->AddNode(g13_op);
  CreateSetOneNode(g14_op, dim, dtype);
  ge::NodePtr g14_node = graph->AddNode(g14_op);
  CreateSetOneNode(g15_op, dim, dtype);
  ge::NodePtr g15_node = graph->AddNode(g15_op);
  CreateSetOneNode(g16_op, dim, dtype);
  ge::NodePtr g16_node = graph->AddNode(g16_op);
  CreateSetOneNode(g17_op, dim, dtype);
  ge::NodePtr g17_node = graph->AddNode(g17_op);

  Status ret1 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g1_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret1 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g1_node, "tbe-custom");
  Status ret2 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g2_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret2 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g2_node, "tbe-custom");
  Status ret3 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g3_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret3 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g3_node, "tbe-custom");
  Status ret4 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g4_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret4 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g4_node, "tbe-custom");
  Status ret5 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g5_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret5 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g5_node, "tbe-custom");
  Status ret6 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g6_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret6 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g6_node, "tbe-custom");
  Status ret7 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g7_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret7 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g7_node, "tbe-custom");
  Status ret8 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g8_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret8 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g8_node, "tbe-custom");
  Status ret9 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g9_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret9 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g9_node, "tbe-custom");
  Status ret10 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g10_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret10 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g10_node, "tbe-custom");
  Status ret11 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g11_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret11 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g11_node, "tbe-custom");
  Status ret12 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g12_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret12 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g12_node, "tbe-custom");
  Status ret13 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g13_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret13 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g13_node, "tbe-custom");
  Status ret14 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g14_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret14 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g14_node, "tbe-custom");
  Status ret15 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g15_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret15 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g15_node, "tbe-custom");
  Status ret16 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g16_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret16 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g16_node, "tbe-custom");
  Status ret17 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g17_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret17 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g17_node, "tbe-custom");


  ASSERT_EQ(ret1, fe::SUCCESS);
  ASSERT_EQ(ret2, fe::SUCCESS);
  ASSERT_EQ(ret3, fe::SUCCESS);
  ASSERT_EQ(ret4, fe::SUCCESS);
  ASSERT_EQ(ret5, fe::SUCCESS);
  ASSERT_EQ(ret6, fe::SUCCESS);
  ASSERT_EQ(ret7, fe::SUCCESS);
  ASSERT_EQ(ret8, fe::SUCCESS);
  ASSERT_EQ(ret9, fe::SUCCESS);
  ASSERT_EQ(ret10, fe::SUCCESS);
  ASSERT_EQ(ret11, fe::SUCCESS);
  ASSERT_EQ(ret12, fe::SUCCESS);
  ASSERT_EQ(ret13, fe::SUCCESS);
  ASSERT_EQ(ret14, fe::SUCCESS);
  ASSERT_EQ(ret15, fe::SUCCESS);
  ASSERT_EQ(ret16, fe::SUCCESS);
  ASSERT_EQ(ret17, fe::SUCCESS);

  vector<ge::DataType> result_input = {DT_FLOAT, DT_UINT8, DT_MAX, DT_FLOAT16, DT_FLOAT16,
                                       DT_FLOAT, DT_UINT8, DT_FLOAT, DT_FLOAT, DT_FLOAT16,
                                       DT_FLOAT, DT_MAX, DT_FLOAT16, DT_FLOAT16, DT_FLOAT,
                                       DT_FLOAT16, DT_FLOAT};
  vector<ge::DataType> result_output = {DT_FLOAT, DT_UINT8, DT_MAX, DT_FLOAT16, DT_FLOAT16,
                                       DT_FLOAT, DT_UINT8, DT_FLOAT, DT_FLOAT, DT_FLOAT16,
                                       DT_FLOAT, DT_MAX, DT_FLOAT16, DT_FLOAT16, DT_FLOAT,
                                       DT_FLOAT16, DT_FLOAT};
  EXPECT_EQ(g1_op->GetInputDesc(0).GetDataType(), result_input[0]);
  EXPECT_EQ(g2_op->GetInputDesc(0).GetDataType(), result_input[1]);
  //EXPECT_EQ(g3_op->GetInputDesc(0).GetDataType(), result_input[2]);
  EXPECT_EQ(g4_op->GetInputDesc(0).GetDataType(), result_input[3]);
  EXPECT_EQ(g5_op->GetInputDesc(0).GetDataType(), result_input[4]);
  EXPECT_EQ(g6_op->GetInputDesc(0).GetDataType(), result_input[5]);
  EXPECT_EQ(g7_op->GetInputDesc(0).GetDataType(), result_input[6]);
  EXPECT_EQ(g8_op->GetInputDesc(0).GetDataType(), result_input[7]);
  EXPECT_EQ(g9_op->GetInputDesc(0).GetDataType(), result_input[8]);
  EXPECT_EQ(g10_op->GetInputDesc(0).GetDataType(), result_input[9]);
  EXPECT_EQ(g11_op->GetInputDesc(0).GetDataType(), result_input[10]);
  //EXPECT_EQ(g12_op->GetInputDesc(0).GetDataType(), result_input[11]);
  EXPECT_EQ(g13_op->GetInputDesc(0).GetDataType(), result_input[12]);
  EXPECT_EQ(g14_op->GetInputDesc(0).GetDataType(), result_input[13]);
  EXPECT_EQ(g15_op->GetInputDesc(0).GetDataType(), result_input[14]);
  EXPECT_EQ(g16_op->GetInputDesc(0).GetDataType(), result_input[15]);
  EXPECT_EQ(g17_op->GetInputDesc(0).GetDataType(), result_input[16]);


  EXPECT_EQ(g1_op->GetOutputDesc(0).GetDataType(), result_output[0]);
  EXPECT_EQ(g2_op->GetOutputDesc(0).GetDataType(), result_output[1]);
  //EXPECT_EQ(g3_op->GetOutputDesc(0).GetDataType(), result_output[2]);
  EXPECT_EQ(g4_op->GetOutputDesc(0).GetDataType(), result_output[3]);
  EXPECT_EQ(g5_op->GetOutputDesc(0).GetDataType(), result_output[4]);
  EXPECT_EQ(g6_op->GetOutputDesc(0).GetDataType(), result_output[5]);
  EXPECT_EQ(g7_op->GetOutputDesc(0).GetDataType(), result_output[6]);
  EXPECT_EQ(g8_op->GetOutputDesc(0).GetDataType(), result_output[7]);
  EXPECT_EQ(g9_op->GetOutputDesc(0).GetDataType(), result_output[8]);
  EXPECT_EQ(g10_op->GetOutputDesc(0).GetDataType(), result_output[9]);
  EXPECT_EQ(g11_op->GetOutputDesc(0).GetDataType(), result_output[10]);
  //EXPECT_EQ(g12_op->GetOutputDesc(0).GetDataType(), result_output[11]);
  EXPECT_EQ(g13_op->GetOutputDesc(0).GetDataType(), result_output[12]);
  EXPECT_EQ(g14_op->GetOutputDesc(0).GetDataType(), result_output[13]);
  EXPECT_EQ(g15_op->GetOutputDesc(0).GetDataType(), result_output[14]);
  EXPECT_EQ(g16_op->GetOutputDesc(0).GetDataType(), result_output[15]);
  EXPECT_EQ(g17_op->GetOutputDesc(0).GetDataType(), result_output[16]);
}


TEST_F(UTEST_fusion_engine_op_judge_precision_mode, set_nodes_IO_dtypes_mix_fp16_1)
{
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = ALLOW_MIX_PRECISION_FP16;
  op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g1_op = std::make_shared<OpDesc>("G1", "GOnlyfp32");
  OpDescPtr g2_op = std::make_shared<OpDesc>("G2", "GOnlyBf16");
  OpDescPtr g3_op = std::make_shared<OpDesc>("G3", "GBlackOnlyFp16");
  OpDescPtr g4_op = std::make_shared<OpDesc>("G4", "GWhiteOnlyFp16");
  OpDescPtr g5_op = std::make_shared<OpDesc>("G5", "GOnlyFp16");
  OpDescPtr g6_op = std::make_shared<OpDesc>("G6", "GBlackSupportFp32Bf16");
  OpDescPtr g7_op = std::make_shared<OpDesc>("G7", "GWhiteSupportFp32Bf16");
  OpDescPtr g8_op = std::make_shared<OpDesc>("G8", "GSupportFp32Bf16");
  OpDescPtr g9_op = std::make_shared<OpDesc>("G9", "GBlackSupportfp32Fp16");
  OpDescPtr g10_op = std::make_shared<OpDesc>("G10", "GWhiteSupportfp32Fp16");
  OpDescPtr g11_op = std::make_shared<OpDesc>("G11", "GSupportfp32Fp16");
  OpDescPtr g12_op = std::make_shared<OpDesc>("G12", "GBlackSupportBf16Fp16");
  OpDescPtr g13_op = std::make_shared<OpDesc>("G13", "GWhiteSupportBf16Fp16");
  OpDescPtr g14_op = std::make_shared<OpDesc>("G14", "GSupportBf16Fp16");
  OpDescPtr g15_op = std::make_shared<OpDesc>("G15", "GBlack");
  OpDescPtr g16_op = std::make_shared<OpDesc>("G16", "GWhite");
  OpDescPtr g17_op = std::make_shared<OpDesc>("G17", "G");

  vector<int64_t> dim({4, 33, 12, 16});
  ge::DataType dtype = DT_BF16;
  //add descriptor
  CreateSetOneNode(g1_op, dim, dtype);
  ge::NodePtr g1_node = graph->AddNode(g1_op);
  CreateSetOneNode(g2_op, dim, dtype);
  ge::NodePtr g2_node = graph->AddNode(g2_op);
  CreateSetOneNode(g3_op, dim, dtype);
  ge::NodePtr g3_node = graph->AddNode(g3_op);
  CreateSetOneNode(g4_op, dim, dtype);
  ge::NodePtr g4_node = graph->AddNode(g4_op);
  CreateSetOneNode(g5_op, dim, dtype);
  ge::NodePtr g5_node = graph->AddNode(g5_op);
  CreateSetOneNode(g6_op, dim, dtype);
  ge::NodePtr g6_node = graph->AddNode(g6_op);
  CreateSetOneNode(g7_op, dim, dtype);
  ge::NodePtr g7_node = graph->AddNode(g7_op);
  CreateSetOneNode(g8_op, dim, dtype);
  ge::NodePtr g8_node = graph->AddNode(g8_op);
  CreateSetOneNode(g9_op, dim, dtype);
  ge::NodePtr g9_node = graph->AddNode(g9_op);
  CreateSetOneNode(g10_op, dim, dtype);
  ge::NodePtr g10_node = graph->AddNode(g10_op);
  CreateSetOneNode(g11_op, dim, dtype);
  ge::NodePtr g11_node = graph->AddNode(g11_op);
  CreateSetOneNode(g12_op, dim, dtype);
  ge::NodePtr g12_node = graph->AddNode(g12_op);
  CreateSetOneNode(g13_op, dim, dtype);
  ge::NodePtr g13_node = graph->AddNode(g13_op);
  CreateSetOneNode(g14_op, dim, dtype);
  ge::NodePtr g14_node = graph->AddNode(g14_op);
  CreateSetOneNode(g15_op, dim, dtype);
  ge::NodePtr g15_node = graph->AddNode(g15_op);
  CreateSetOneNode(g16_op, dim, dtype);
  ge::NodePtr g16_node = graph->AddNode(g16_op);
  CreateSetOneNode(g17_op, dim, dtype);
  ge::NodePtr g17_node = graph->AddNode(g17_op);

  Status ret1 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g1_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret1 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g1_node, "tbe-custom");
  Status ret2 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g2_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret2 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g2_node, "tbe-custom");
  Status ret3 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g3_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret3 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g3_node, "tbe-custom");
  Status ret4 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g4_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret4 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g4_node, "tbe-custom");
  Status ret5 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g5_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret5 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g5_node, "tbe-custom");
  Status ret6 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g6_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret6 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g6_node, "tbe-custom");
  Status ret7 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g7_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret7 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g7_node, "tbe-custom");
  Status ret8 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g8_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret8 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g8_node, "tbe-custom");
  Status ret9 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g9_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret9 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g9_node, "tbe-custom");
  Status ret10 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g10_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret10 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g10_node, "tbe-custom");
  Status ret11 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g11_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret11 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g11_node, "tbe-custom");
  Status ret12 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g12_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret12 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g12_node, "tbe-custom");
  Status ret13 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g13_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret13 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g13_node, "tbe-custom");
  Status ret14 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g14_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret14 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g14_node, "tbe-custom");
  Status ret15 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g15_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret15 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g15_node, "tbe-custom");
  Status ret16 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g16_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret16 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g16_node, "tbe-custom");
  Status ret17 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g17_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret17 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g17_node, "tbe-custom");


  ASSERT_EQ(ret1, fe::SUCCESS);
  ASSERT_EQ(ret2, fe::SUCCESS);
  ASSERT_EQ(ret3, fe::SUCCESS);
  ASSERT_EQ(ret4, fe::SUCCESS);
  ASSERT_EQ(ret5, fe::SUCCESS);
  ASSERT_EQ(ret6, fe::SUCCESS);
  ASSERT_EQ(ret7, fe::SUCCESS);
  ASSERT_EQ(ret8, fe::SUCCESS);
  ASSERT_EQ(ret9, fe::SUCCESS);
  ASSERT_EQ(ret10, fe::SUCCESS);
  ASSERT_EQ(ret11, fe::SUCCESS);
  ASSERT_EQ(ret12, fe::SUCCESS);
  ASSERT_EQ(ret13, fe::SUCCESS);
  ASSERT_EQ(ret14, fe::SUCCESS);
  ASSERT_EQ(ret15, fe::SUCCESS);
  ASSERT_EQ(ret16, fe::SUCCESS);
  ASSERT_EQ(ret17, fe::SUCCESS);

  vector<ge::DataType> result_input = {DT_FLOAT, DT_BF16, DT_MAX, DT_FLOAT16, DT_FLOAT16,
                                       DT_BF16, DT_BF16, DT_BF16, DT_FLOAT, DT_FLOAT16,
                                       DT_FLOAT, DT_UINT8, DT_BF16, DT_BF16, DT_BF16,
                                       DT_BF16, DT_BF16};
  vector<ge::DataType> result_output = {DT_FLOAT, DT_BF16, DT_MAX, DT_FLOAT16, DT_FLOAT16,
                                        DT_BF16, DT_BF16, DT_BF16, DT_FLOAT, DT_FLOAT16,
                                       DT_FLOAT, DT_UINT8, DT_BF16, DT_BF16, DT_BF16,
                                       DT_BF16, DT_BF16};
  EXPECT_EQ(g1_op->GetInputDesc(0).GetDataType(), result_input[0]);
  EXPECT_EQ(g2_op->GetInputDesc(0).GetDataType(), result_input[1]);
  //EXPECT_EQ(g3_op->GetInputDesc(0).GetDataType(), result_input[2]);
  EXPECT_EQ(g4_op->GetInputDesc(0).GetDataType(), result_input[3]);
  EXPECT_EQ(g5_op->GetInputDesc(0).GetDataType(), result_input[4]);
  EXPECT_EQ(g6_op->GetInputDesc(0).GetDataType(), result_input[5]);
  EXPECT_EQ(g7_op->GetInputDesc(0).GetDataType(), result_input[6]);
  EXPECT_EQ(g8_op->GetInputDesc(0).GetDataType(), result_input[7]);
  EXPECT_EQ(g9_op->GetInputDesc(0).GetDataType(), result_input[8]);
  EXPECT_EQ(g10_op->GetInputDesc(0).GetDataType(), result_input[9]);
  EXPECT_EQ(g11_op->GetInputDesc(0).GetDataType(), result_input[10]);
  //EXPECT_EQ(g12_op->GetInputDesc(0).GetDataType(), result_input[11]);
  EXPECT_EQ(g13_op->GetInputDesc(0).GetDataType(), result_input[12]);
  EXPECT_EQ(g14_op->GetInputDesc(0).GetDataType(), result_input[13]);
  EXPECT_EQ(g15_op->GetInputDesc(0).GetDataType(), result_input[14]);
  EXPECT_EQ(g16_op->GetInputDesc(0).GetDataType(), result_input[15]);
  EXPECT_EQ(g17_op->GetInputDesc(0).GetDataType(), result_input[16]);


  EXPECT_EQ(g1_op->GetOutputDesc(0).GetDataType(), result_output[0]);
  EXPECT_EQ(g2_op->GetOutputDesc(0).GetDataType(), result_output[1]);
  //EXPECT_EQ(g3_op->GetOutputDesc(0).GetDataType(), result_output[2]);
  EXPECT_EQ(g4_op->GetOutputDesc(0).GetDataType(), result_output[3]);
  EXPECT_EQ(g5_op->GetOutputDesc(0).GetDataType(), result_output[4]);
  EXPECT_EQ(g6_op->GetOutputDesc(0).GetDataType(), result_output[5]);
  EXPECT_EQ(g7_op->GetOutputDesc(0).GetDataType(), result_output[6]);
  EXPECT_EQ(g8_op->GetOutputDesc(0).GetDataType(), result_output[7]);
  EXPECT_EQ(g9_op->GetOutputDesc(0).GetDataType(), result_output[8]);
  EXPECT_EQ(g10_op->GetOutputDesc(0).GetDataType(), result_output[9]);
  EXPECT_EQ(g11_op->GetOutputDesc(0).GetDataType(), result_output[10]);
  //EXPECT_EQ(g12_op->GetOutputDesc(0).GetDataType(), result_output[11]);
  EXPECT_EQ(g13_op->GetOutputDesc(0).GetDataType(), result_output[12]);
  EXPECT_EQ(g14_op->GetOutputDesc(0).GetDataType(), result_output[13]);
  EXPECT_EQ(g15_op->GetOutputDesc(0).GetDataType(), result_output[14]);
  EXPECT_EQ(g16_op->GetOutputDesc(0).GetDataType(), result_output[15]);
  EXPECT_EQ(g17_op->GetOutputDesc(0).GetDataType(), result_output[16]);
}


TEST_F(UTEST_fusion_engine_op_judge_precision_mode, set_nodes_IO_dtypes_mix_fp16_2)
{
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = ALLOW_MIX_PRECISION_FP16;
  op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g1_op = std::make_shared<OpDesc>("G1", "GOnlyfp32");
  OpDescPtr g2_op = std::make_shared<OpDesc>("G2", "GOnlyBf16");
  OpDescPtr g3_op = std::make_shared<OpDesc>("G3", "GBlackOnlyFp16");
  OpDescPtr g4_op = std::make_shared<OpDesc>("G4", "GWhiteOnlyFp16");
  OpDescPtr g5_op = std::make_shared<OpDesc>("G5", "GOnlyFp16");
  OpDescPtr g6_op = std::make_shared<OpDesc>("G6", "GBlackSupportFp32Bf16");
  OpDescPtr g7_op = std::make_shared<OpDesc>("G7", "GWhiteSupportFp32Bf16");
  OpDescPtr g8_op = std::make_shared<OpDesc>("G8", "GSupportFp32Bf16");
  OpDescPtr g9_op = std::make_shared<OpDesc>("G9", "GBlackSupportfp32Fp16");
  OpDescPtr g10_op = std::make_shared<OpDesc>("G10", "GWhiteSupportfp32Fp16");
  OpDescPtr g11_op = std::make_shared<OpDesc>("G11", "GSupportfp32Fp16");
  OpDescPtr g12_op = std::make_shared<OpDesc>("G12", "GBlackSupportBf16Fp16");
  OpDescPtr g13_op = std::make_shared<OpDesc>("G13", "GWhiteSupportBf16Fp16");
  OpDescPtr g14_op = std::make_shared<OpDesc>("G14", "GSupportBf16Fp16");
  OpDescPtr g15_op = std::make_shared<OpDesc>("G15", "GBlack");
  OpDescPtr g16_op = std::make_shared<OpDesc>("G16", "GWhite");
  OpDescPtr g17_op = std::make_shared<OpDesc>("G17", "G");

  vector<int64_t> dim({4, 33, 12, 16});
  ge::DataType dtype = DT_FLOAT16;
  //add descriptor
  CreateSetOneNode(g1_op, dim, dtype);
  ge::NodePtr g1_node = graph->AddNode(g1_op);
  CreateSetOneNode(g2_op, dim, dtype);
  ge::NodePtr g2_node = graph->AddNode(g2_op);
  CreateSetOneNode(g3_op, dim, dtype);
  ge::NodePtr g3_node = graph->AddNode(g3_op);
  CreateSetOneNode(g4_op, dim, dtype);
  ge::NodePtr g4_node = graph->AddNode(g4_op);
  CreateSetOneNode(g5_op, dim, dtype);
  ge::NodePtr g5_node = graph->AddNode(g5_op);
  CreateSetOneNode(g6_op, dim, dtype);
  ge::NodePtr g6_node = graph->AddNode(g6_op);
  CreateSetOneNode(g7_op, dim, dtype);
  ge::NodePtr g7_node = graph->AddNode(g7_op);
  CreateSetOneNode(g8_op, dim, dtype);
  ge::NodePtr g8_node = graph->AddNode(g8_op);
  CreateSetOneNode(g9_op, dim, dtype);
  ge::NodePtr g9_node = graph->AddNode(g9_op);
  CreateSetOneNode(g10_op, dim, dtype);
  ge::NodePtr g10_node = graph->AddNode(g10_op);
  CreateSetOneNode(g11_op, dim, dtype);
  ge::NodePtr g11_node = graph->AddNode(g11_op);
  CreateSetOneNode(g12_op, dim, dtype);
  ge::NodePtr g12_node = graph->AddNode(g12_op);
  CreateSetOneNode(g13_op, dim, dtype);
  ge::NodePtr g13_node = graph->AddNode(g13_op);
  CreateSetOneNode(g14_op, dim, dtype);
  ge::NodePtr g14_node = graph->AddNode(g14_op);
  CreateSetOneNode(g15_op, dim, dtype);
  ge::NodePtr g15_node = graph->AddNode(g15_op);
  CreateSetOneNode(g16_op, dim, dtype);
  ge::NodePtr g16_node = graph->AddNode(g16_op);
  CreateSetOneNode(g17_op, dim, dtype);
  ge::NodePtr g17_node = graph->AddNode(g17_op);

  Status ret1 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g1_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret1 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g1_node, "tbe-custom");
  Status ret2 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g2_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret2 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g2_node, "tbe-custom");
  Status ret3 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g3_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret3 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g3_node, "tbe-custom");
  Status ret4 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g4_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret4 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g4_node, "tbe-custom");
  Status ret5 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g5_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret5 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g5_node, "tbe-custom");
  Status ret6 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g6_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret6 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g6_node, "tbe-custom");
  Status ret7 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g7_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret7 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g7_node, "tbe-custom");
  Status ret8 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g8_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret8 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g8_node, "tbe-custom");
  Status ret9 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g9_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret9 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g9_node, "tbe-custom");
  Status ret10 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g10_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret10 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g10_node, "tbe-custom");
  Status ret11 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g11_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret11 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g11_node, "tbe-custom");
  Status ret12 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g12_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret12 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g12_node, "tbe-custom");
  Status ret13 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g13_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret13 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g13_node, "tbe-custom");
  Status ret14 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g14_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret14 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g14_node, "tbe-custom");
  Status ret15 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g15_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret15 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g15_node, "tbe-custom");
  Status ret16 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g16_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret16 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g16_node, "tbe-custom");
  Status ret17 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g17_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret17 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g17_node, "tbe-custom");


  ASSERT_EQ(ret1, fe::SUCCESS);
  ASSERT_EQ(ret2, fe::SUCCESS);
  ASSERT_EQ(ret3, fe::SUCCESS);
  ASSERT_EQ(ret4, fe::SUCCESS);
  ASSERT_EQ(ret5, fe::SUCCESS);
  ASSERT_EQ(ret6, fe::SUCCESS);
  ASSERT_EQ(ret7, fe::SUCCESS);
  ASSERT_EQ(ret8, fe::SUCCESS);
  ASSERT_EQ(ret9, fe::SUCCESS);
  ASSERT_EQ(ret10, fe::SUCCESS);
  ASSERT_EQ(ret11, fe::SUCCESS);
  ASSERT_EQ(ret12, fe::SUCCESS);
  ASSERT_EQ(ret13, fe::SUCCESS);
  ASSERT_EQ(ret14, fe::SUCCESS);
  ASSERT_EQ(ret15, fe::SUCCESS);
  ASSERT_EQ(ret16, fe::SUCCESS);
  ASSERT_EQ(ret17, fe::SUCCESS);

  vector<ge::DataType> result_input = {DT_FLOAT, DT_UINT8, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                                       DT_FLOAT, DT_UINT8, DT_FLOAT, DT_FLOAT16, DT_FLOAT16,
                                       DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                                       DT_FLOAT16, DT_FLOAT16};
  vector<ge::DataType> result_output = {DT_FLOAT, DT_UINT8, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                                       DT_FLOAT, DT_UINT8, DT_FLOAT, DT_FLOAT16, DT_FLOAT16,
                                       DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                                       DT_FLOAT16, DT_FLOAT16};
  EXPECT_EQ(g1_op->GetInputDesc(0).GetDataType(), result_input[0]);
  EXPECT_EQ(g2_op->GetInputDesc(0).GetDataType(), result_input[1]);
  EXPECT_EQ(g3_op->GetInputDesc(0).GetDataType(), result_input[2]);
  EXPECT_EQ(g4_op->GetInputDesc(0).GetDataType(), result_input[3]);
  EXPECT_EQ(g5_op->GetInputDesc(0).GetDataType(), result_input[4]);
  EXPECT_EQ(g6_op->GetInputDesc(0).GetDataType(), result_input[5]);
  EXPECT_EQ(g7_op->GetInputDesc(0).GetDataType(), result_input[6]);
  EXPECT_EQ(g8_op->GetInputDesc(0).GetDataType(), result_input[7]);
  EXPECT_EQ(g9_op->GetInputDesc(0).GetDataType(), result_input[8]);
  EXPECT_EQ(g10_op->GetInputDesc(0).GetDataType(), result_input[9]);
  EXPECT_EQ(g11_op->GetInputDesc(0).GetDataType(), result_input[10]);
  EXPECT_EQ(g12_op->GetInputDesc(0).GetDataType(), result_input[11]);
  EXPECT_EQ(g13_op->GetInputDesc(0).GetDataType(), result_input[12]);
  EXPECT_EQ(g14_op->GetInputDesc(0).GetDataType(), result_input[13]);
  EXPECT_EQ(g15_op->GetInputDesc(0).GetDataType(), result_input[14]);
  EXPECT_EQ(g16_op->GetInputDesc(0).GetDataType(), result_input[15]);
  EXPECT_EQ(g17_op->GetInputDesc(0).GetDataType(), result_input[16]);


  EXPECT_EQ(g1_op->GetOutputDesc(0).GetDataType(), result_output[0]);
  EXPECT_EQ(g2_op->GetOutputDesc(0).GetDataType(), result_output[1]);
  EXPECT_EQ(g3_op->GetOutputDesc(0).GetDataType(), result_output[2]);
  EXPECT_EQ(g4_op->GetOutputDesc(0).GetDataType(), result_output[3]);
  EXPECT_EQ(g5_op->GetOutputDesc(0).GetDataType(), result_output[4]);
  EXPECT_EQ(g6_op->GetOutputDesc(0).GetDataType(), result_output[5]);
  EXPECT_EQ(g7_op->GetOutputDesc(0).GetDataType(), result_output[6]);
  EXPECT_EQ(g8_op->GetOutputDesc(0).GetDataType(), result_output[7]);
  EXPECT_EQ(g9_op->GetOutputDesc(0).GetDataType(), result_output[8]);
  EXPECT_EQ(g10_op->GetOutputDesc(0).GetDataType(), result_output[9]);
  EXPECT_EQ(g11_op->GetOutputDesc(0).GetDataType(), result_output[10]);
  EXPECT_EQ(g12_op->GetOutputDesc(0).GetDataType(), result_output[11]);
  EXPECT_EQ(g13_op->GetOutputDesc(0).GetDataType(), result_output[12]);
  EXPECT_EQ(g14_op->GetOutputDesc(0).GetDataType(), result_output[13]);
  EXPECT_EQ(g15_op->GetOutputDesc(0).GetDataType(), result_output[14]);
  EXPECT_EQ(g16_op->GetOutputDesc(0).GetDataType(), result_output[15]);
  EXPECT_EQ(g17_op->GetOutputDesc(0).GetDataType(), result_output[16]);
}

// test for net in gray, input fp32-> GWhite -> GBlackSupportFp32Bf16 -> GWhiteSupportBf16Fp16 -> G
// the fllows output is        fp32 -> fp16 ->  fp32                  -> fp16                  -> fp16
TEST_F(UTEST_fusion_engine_op_judge_precision_mode, set_nodes_IO_dtypes_mix_fp16_3_v2)
{
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = ALLOW_MIX_PRECISION_FP16;
  op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g1_op = std::make_shared<OpDesc>("G1", "GWhite");
  OpDescPtr g2_op = std::make_shared<OpDesc>("G2", "GBlackSupportFp32Bf16");
  OpDescPtr g3_op = std::make_shared<OpDesc>("G3", "GWhiteSupportBf16Fp16");
  OpDescPtr g4_op = std::make_shared<OpDesc>("G4", "G");

  vector<int64_t> dim({4, 33, 12, 16});
  ge::DataType dtype = DT_FLOAT;
  //add descriptor
  CreateSetOneNode(g1_op, dim, dtype);
  ge::NodePtr g1_node = graph->AddNode(g1_op);
  CreateSetOneNode(g2_op, dim, dtype);
  ge::NodePtr g2_node = graph->AddNode(g2_op);
  GraphUtils::AddEdge(g1_node->GetOutDataAnchor(0), g2_node->GetInDataAnchor(0));
  CreateSetOneNode(g3_op, dim, dtype);
  ge::NodePtr g3_node = graph->AddNode(g3_op);
  GraphUtils::AddEdge(g2_node->GetOutDataAnchor(0), g3_node->GetInDataAnchor(0));
  CreateSetOneNode(g4_op, dim, dtype);
  ge::NodePtr g4_node = graph->AddNode(g4_op);
  GraphUtils::AddEdge(g3_node->GetOutDataAnchor(0), g4_node->GetInDataAnchor(0));

  Status ret1 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g1_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret1 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g1_node, "tbe-custom");
  Status ret2 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g2_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret2 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g2_node, "tbe-custom");
  Status ret3 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g3_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret3 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g3_node, "tbe-custom");
  Status ret4 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g4_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret4 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g4_node, "tbe-custom");

  ASSERT_EQ(ret1, fe::SUCCESS);
  ASSERT_EQ(ret2, fe::SUCCESS);
  ASSERT_EQ(ret3, fe::SUCCESS);
  ASSERT_EQ(ret4, fe::SUCCESS);


  vector<ge::DataType> result_input = { DT_FLOAT16, DT_FLOAT, DT_FLOAT16, DT_FLOAT16 };
  vector<ge::DataType> result_output = { DT_FLOAT16, DT_FLOAT, DT_FLOAT16, DT_FLOAT16 };
  EXPECT_EQ(g1_op->GetInputDesc(0).GetDataType(), result_input[0]);
  EXPECT_EQ(g2_op->GetInputDesc(0).GetDataType(), result_input[1]);
  EXPECT_EQ(g3_op->GetInputDesc(0).GetDataType(), result_input[2]);
  EXPECT_EQ(g4_op->GetInputDesc(0).GetDataType(), result_input[3]);

  EXPECT_EQ(g1_op->GetOutputDesc(0).GetDataType(), result_output[0]);
  EXPECT_EQ(g2_op->GetOutputDesc(0).GetDataType(), result_output[1]);
  EXPECT_EQ(g3_op->GetOutputDesc(0).GetDataType(), result_output[2]);
  EXPECT_EQ(g4_op->GetOutputDesc(0).GetDataType(), result_output[3]);
}

TEST_F(UTEST_fusion_engine_op_judge_precision_mode, set_nodes_IO_dtypes_mix_fp16_3)
{
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = V2_MIX_FP16;
  op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME,
                                                                    reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g1_op = std::make_shared<OpDesc>("G1", "GWhite");
  OpDescPtr g2_op = std::make_shared<OpDesc>("G2", "GBlackSupportFp32Bf16");
  OpDescPtr g3_op = std::make_shared<OpDesc>("G3", "GWhiteSupportBf16Fp16");
  OpDescPtr g4_op = std::make_shared<OpDesc>("G4", "G");

  vector<int64_t> dim({4, 33, 12, 16});
  ge::DataType dtype = DT_FLOAT;
  //add descriptor
  CreateSetOneNode(g1_op, dim, dtype);
  ge::NodePtr g1_node = graph->AddNode(g1_op);
  CreateSetOneNode(g2_op, dim, dtype);
  ge::NodePtr g2_node = graph->AddNode(g2_op);
  GraphUtils::AddEdge(g1_node->GetOutDataAnchor(0), g2_node->GetInDataAnchor(0));
  CreateSetOneNode(g3_op, dim, dtype);
  ge::NodePtr g3_node = graph->AddNode(g3_op);
  GraphUtils::AddEdge(g2_node->GetOutDataAnchor(0), g3_node->GetInDataAnchor(0));
  CreateSetOneNode(g4_op, dim, dtype);
  ge::NodePtr g4_node = graph->AddNode(g4_op);
  GraphUtils::AddEdge(g3_node->GetOutDataAnchor(0), g4_node->GetInDataAnchor(0));

  Status ret1 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g1_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret1 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g1_node, "tbe-custom");
  Status ret2 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g2_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret2 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g2_node, "tbe-custom");
  Status ret3 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g3_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret3 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g3_node, "tbe-custom");
  Status ret4 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g4_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret4 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g4_node, "tbe-custom");

  ASSERT_EQ(ret1, fe::SUCCESS);
  ASSERT_EQ(ret2, fe::SUCCESS);
  ASSERT_EQ(ret3, fe::SUCCESS);
  ASSERT_EQ(ret4, fe::SUCCESS);


  vector<ge::DataType> result_input = { DT_FLOAT16, DT_FLOAT, DT_FLOAT16, DT_FLOAT16 };
  vector<ge::DataType> result_output = { DT_FLOAT16, DT_FLOAT, DT_FLOAT16, DT_FLOAT16 };
  EXPECT_EQ(g1_op->GetInputDesc(0).GetDataType(), result_input[0]);
  EXPECT_EQ(g2_op->GetInputDesc(0).GetDataType(), result_input[1]);
  EXPECT_EQ(g3_op->GetInputDesc(0).GetDataType(), result_input[2]);
  EXPECT_EQ(g4_op->GetInputDesc(0).GetDataType(), result_input[3]);

  EXPECT_EQ(g1_op->GetOutputDesc(0).GetDataType(), result_output[0]);
  EXPECT_EQ(g2_op->GetOutputDesc(0).GetDataType(), result_output[1]);
  EXPECT_EQ(g3_op->GetOutputDesc(0).GetDataType(), result_output[2]);
  EXPECT_EQ(g4_op->GetOutputDesc(0).GetDataType(), result_output[3]);
}

TEST_F(UTEST_fusion_engine_op_judge_precision_mode, set_nodes_IO_dtypes_mix_bf16)
{
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = ALLOW_MIX_PRECISION_BF16;
  op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g1_op = std::make_shared<OpDesc>("G1", "GOnlyfp32");
  OpDescPtr g2_op = std::make_shared<OpDesc>("G2", "GBlackOnlyBf16");
  OpDescPtr g3_op = std::make_shared<OpDesc>("G3", "GWhiteOnlyBf16");
  OpDescPtr g4_op = std::make_shared<OpDesc>("G4", "GOnlyBf16");
  OpDescPtr g5_op = std::make_shared<OpDesc>("G5", "GOnlyFp16");
  OpDescPtr g6_op = std::make_shared<OpDesc>("G6", "GBlackSupportFp32Bf16");
  OpDescPtr g7_op = std::make_shared<OpDesc>("G7", "GWhiteSupportFp32Bf16");
  OpDescPtr g8_op = std::make_shared<OpDesc>("G8", "GSupportFp32Bf16");
  OpDescPtr g9_op = std::make_shared<OpDesc>("G9", "GBlackSupportfp32Fp16");
  OpDescPtr g10_op = std::make_shared<OpDesc>("G10", "GWhiteSupportfp32Fp16");
  OpDescPtr g11_op = std::make_shared<OpDesc>("G11", "GSupportfp32Fp16");
  OpDescPtr g12_op = std::make_shared<OpDesc>("G12", "GBlackSupportBf16Fp16");
  OpDescPtr g13_op = std::make_shared<OpDesc>("G13", "GWhiteSupportBf16Fp16");
  OpDescPtr g14_op = std::make_shared<OpDesc>("G14", "GSupportBf16Fp16");
  OpDescPtr g15_op = std::make_shared<OpDesc>("G15", "GBlack");
  OpDescPtr g16_op = std::make_shared<OpDesc>("G16", "GWhite");
  OpDescPtr g17_op = std::make_shared<OpDesc>("G17", "G");

  vector<int64_t> dim({4, 33, 12, 16});
  ge::DataType dtype = DT_FLOAT;
  //add descriptor
  CreateSetOneNode(g1_op, dim, dtype);
  ge::NodePtr g1_node = graph->AddNode(g1_op);
  CreateSetOneNode(g2_op, dim, dtype);
  ge::NodePtr g2_node = graph->AddNode(g2_op);
  CreateSetOneNode(g3_op, dim, dtype);
  ge::NodePtr g3_node = graph->AddNode(g3_op);
  CreateSetOneNode(g4_op, dim, dtype);
  ge::NodePtr g4_node = graph->AddNode(g4_op);
  CreateSetOneNode(g5_op, dim, dtype);
  ge::NodePtr g5_node = graph->AddNode(g5_op);
  CreateSetOneNode(g6_op, dim, dtype);
  ge::NodePtr g6_node = graph->AddNode(g6_op);
  CreateSetOneNode(g7_op, dim, dtype);
  ge::NodePtr g7_node = graph->AddNode(g7_op);
  CreateSetOneNode(g8_op, dim, dtype);
  ge::NodePtr g8_node = graph->AddNode(g8_op);
  CreateSetOneNode(g9_op, dim, dtype);
  ge::NodePtr g9_node = graph->AddNode(g9_op);
  CreateSetOneNode(g10_op, dim, dtype);
  ge::NodePtr g10_node = graph->AddNode(g10_op);
  CreateSetOneNode(g11_op, dim, dtype);
  ge::NodePtr g11_node = graph->AddNode(g11_op);
  CreateSetOneNode(g12_op, dim, dtype);
  ge::NodePtr g12_node = graph->AddNode(g12_op);
  CreateSetOneNode(g13_op, dim, dtype);
  ge::NodePtr g13_node = graph->AddNode(g13_op);
  CreateSetOneNode(g14_op, dim, dtype);
  ge::NodePtr g14_node = graph->AddNode(g14_op);
  CreateSetOneNode(g15_op, dim, dtype);
  ge::NodePtr g15_node = graph->AddNode(g15_op);
  CreateSetOneNode(g16_op, dim, dtype);
  ge::NodePtr g16_node = graph->AddNode(g16_op);
  CreateSetOneNode(g17_op, dim, dtype);
  ge::NodePtr g17_node = graph->AddNode(g17_op);

  Status ret1 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g1_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret1 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g1_node, "tbe-custom");
  Status ret2 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g2_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret2 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g2_node, "tbe-custom");
  Status ret3 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g3_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret3 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g3_node, "tbe-custom");
  Status ret4 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g4_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret4 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g4_node, "tbe-custom");
  Status ret5 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g5_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret5 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g5_node, "tbe-custom");
  Status ret6 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g6_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret6 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g6_node, "tbe-custom");
  Status ret7 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g7_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret7 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g7_node, "tbe-custom");
  Status ret8 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g8_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret8 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g8_node, "tbe-custom");
  Status ret9 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g9_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret9 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g9_node, "tbe-custom");
  Status ret10 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g10_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret10 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g10_node, "tbe-custom");
  Status ret11 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g11_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret11 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g11_node, "tbe-custom");
  Status ret12 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g12_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret12 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g12_node, "tbe-custom");
  Status ret13 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g13_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret13 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g13_node, "tbe-custom");
  Status ret14 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g14_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret14 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g14_node, "tbe-custom");
  Status ret15 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g15_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret15 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g15_node, "tbe-custom");
  Status ret16 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g16_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret16 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g16_node, "tbe-custom");
  Status ret17 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g17_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret17 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g17_node, "tbe-custom");


  ASSERT_EQ(ret1, fe::SUCCESS);
  ASSERT_EQ(ret2, fe::SUCCESS);
  ASSERT_EQ(ret3, fe::SUCCESS);
  ASSERT_EQ(ret4, fe::SUCCESS);
  ASSERT_EQ(ret5, fe::SUCCESS);
  ASSERT_EQ(ret6, fe::SUCCESS);
  ASSERT_EQ(ret7, fe::SUCCESS);
  ASSERT_EQ(ret8, fe::SUCCESS);
  ASSERT_EQ(ret9, fe::SUCCESS);
  ASSERT_EQ(ret10, fe::SUCCESS);
  ASSERT_EQ(ret11, fe::SUCCESS);
  ASSERT_EQ(ret12, fe::SUCCESS);
  ASSERT_EQ(ret13, fe::SUCCESS);
  ASSERT_EQ(ret14, fe::SUCCESS);
  ASSERT_EQ(ret15, fe::SUCCESS);
  ASSERT_EQ(ret16, fe::SUCCESS);
  ASSERT_EQ(ret17, fe::SUCCESS);

  vector<ge::DataType> result_input = {DT_FLOAT, DT_MAX, DT_BF16, DT_BF16, DT_UINT8,
                                       DT_FLOAT, DT_BF16, DT_FLOAT, DT_FLOAT, DT_UINT8,
                                       DT_FLOAT, DT_MAX, DT_BF16, DT_BF16, DT_FLOAT,
                                       DT_BF16, DT_FLOAT};
  vector<ge::DataType> result_output = {DT_FLOAT, DT_MAX, DT_BF16, DT_BF16, DT_UINT8,
                                       DT_FLOAT, DT_BF16, DT_FLOAT, DT_FLOAT, DT_UINT8,
                                       DT_FLOAT, DT_MAX, DT_BF16, DT_BF16, DT_FLOAT,
                                       DT_BF16, DT_FLOAT};
  EXPECT_EQ(g1_op->GetInputDesc(0).GetDataType(), result_input[0]);
  //EXPECT_EQ(g2_op->GetInputDesc(0).GetDataType(), result_input[1]);
  EXPECT_EQ(g3_op->GetInputDesc(0).GetDataType(), result_input[2]);
  EXPECT_EQ(g4_op->GetInputDesc(0).GetDataType(), result_input[3]);
  EXPECT_EQ(g5_op->GetInputDesc(0).GetDataType(), result_input[4]);
  EXPECT_EQ(g6_op->GetInputDesc(0).GetDataType(), result_input[5]);
  EXPECT_EQ(g7_op->GetInputDesc(0).GetDataType(), result_input[6]);
  EXPECT_EQ(g8_op->GetInputDesc(0).GetDataType(), result_input[7]);
  EXPECT_EQ(g9_op->GetInputDesc(0).GetDataType(), result_input[8]);
  EXPECT_EQ(g10_op->GetInputDesc(0).GetDataType(), result_input[9]);
  EXPECT_EQ(g11_op->GetInputDesc(0).GetDataType(), result_input[10]);
  //EXPECT_EQ(g12_op->GetInputDesc(0).GetDataType(), result_input[11]);
  EXPECT_EQ(g13_op->GetInputDesc(0).GetDataType(), result_input[12]);
  EXPECT_EQ(g14_op->GetInputDesc(0).GetDataType(), result_input[13]);
  EXPECT_EQ(g15_op->GetInputDesc(0).GetDataType(), result_input[14]);
  EXPECT_EQ(g16_op->GetInputDesc(0).GetDataType(), result_input[15]);
  EXPECT_EQ(g17_op->GetInputDesc(0).GetDataType(), result_input[16]);


  EXPECT_EQ(g1_op->GetOutputDesc(0).GetDataType(), result_output[0]);
  //EXPECT_EQ(g2_op->GetOutputDesc(0).GetDataType(), result_output[1]);
  EXPECT_EQ(g3_op->GetOutputDesc(0).GetDataType(), result_output[2]);
  EXPECT_EQ(g4_op->GetOutputDesc(0).GetDataType(), result_output[3]);
  EXPECT_EQ(g5_op->GetOutputDesc(0).GetDataType(), result_output[4]);
  EXPECT_EQ(g6_op->GetOutputDesc(0).GetDataType(), result_output[5]);
  EXPECT_EQ(g7_op->GetOutputDesc(0).GetDataType(), result_output[6]);
  EXPECT_EQ(g8_op->GetOutputDesc(0).GetDataType(), result_output[7]);
  EXPECT_EQ(g9_op->GetOutputDesc(0).GetDataType(), result_output[8]);
  EXPECT_EQ(g10_op->GetOutputDesc(0).GetDataType(), result_output[9]);
  EXPECT_EQ(g11_op->GetOutputDesc(0).GetDataType(), result_output[10]);
  //EXPECT_EQ(g12_op->GetOutputDesc(0).GetDataType(), result_output[11]);
  EXPECT_EQ(g13_op->GetOutputDesc(0).GetDataType(), result_output[12]);
  EXPECT_EQ(g14_op->GetOutputDesc(0).GetDataType(), result_output[13]);
  EXPECT_EQ(g15_op->GetOutputDesc(0).GetDataType(), result_output[14]);
  EXPECT_EQ(g16_op->GetOutputDesc(0).GetDataType(), result_output[15]);
  EXPECT_EQ(g17_op->GetOutputDesc(0).GetDataType(), result_output[16]);
}


TEST_F(UTEST_fusion_engine_op_judge_precision_mode, set_nodes_IO_dtypes_mix_bf16_1)
{
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = ALLOW_MIX_PRECISION_BF16;
  op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g1_op = std::make_shared<OpDesc>("G1", "GOnlyfp32");
  OpDescPtr g2_op = std::make_shared<OpDesc>("G2", "GBlackOnlyBf16");
  OpDescPtr g3_op = std::make_shared<OpDesc>("G3", "GWhiteOnlyBf16");
  OpDescPtr g4_op = std::make_shared<OpDesc>("G4", "GOnlyBf16");
  OpDescPtr g5_op = std::make_shared<OpDesc>("G5", "GOnlyFp16");
  OpDescPtr g6_op = std::make_shared<OpDesc>("G6", "GBlackSupportFp32Bf16");
  OpDescPtr g7_op = std::make_shared<OpDesc>("G7", "GWhiteSupportFp32Bf16");
  OpDescPtr g8_op = std::make_shared<OpDesc>("G8", "GSupportFp32Bf16");
  OpDescPtr g9_op = std::make_shared<OpDesc>("G9", "GBlackSupportfp32Fp16");
  OpDescPtr g10_op = std::make_shared<OpDesc>("G10", "GWhiteSupportfp32Fp16");
  OpDescPtr g11_op = std::make_shared<OpDesc>("G11", "GSupportfp32Fp16");
  OpDescPtr g12_op = std::make_shared<OpDesc>("G12", "GBlackSupportBf16Fp16");
  OpDescPtr g13_op = std::make_shared<OpDesc>("G13", "GWhiteSupportBf16Fp16");
  OpDescPtr g14_op = std::make_shared<OpDesc>("G14", "GSupportBf16Fp16");
  OpDescPtr g15_op = std::make_shared<OpDesc>("G15", "GBlack");
  OpDescPtr g16_op = std::make_shared<OpDesc>("G16", "GWhite");
  OpDescPtr g17_op = std::make_shared<OpDesc>("G17", "G");
  //test for int32 input
  OpDescPtr g18_op = std::make_shared<OpDesc>("G18", "GWhite");

  vector<int64_t> dim({4, 33, 12, 16});
  ge::DataType dtype = DT_BF16;
  //add descriptor
  CreateSetOneNode(g1_op, dim, dtype);
  ge::NodePtr g1_node = graph->AddNode(g1_op);
  CreateSetOneNode(g2_op, dim, dtype);
  ge::NodePtr g2_node = graph->AddNode(g2_op);
  CreateSetOneNode(g3_op, dim, dtype);
  ge::NodePtr g3_node = graph->AddNode(g3_op);
  CreateSetOneNode(g4_op, dim, dtype);
  ge::NodePtr g4_node = graph->AddNode(g4_op);
  CreateSetOneNode(g5_op, dim, dtype);
  ge::NodePtr g5_node = graph->AddNode(g5_op);
  CreateSetOneNode(g6_op, dim, dtype);
  ge::NodePtr g6_node = graph->AddNode(g6_op);
  CreateSetOneNode(g7_op, dim, dtype);
  ge::NodePtr g7_node = graph->AddNode(g7_op);
  CreateSetOneNode(g8_op, dim, dtype);
  ge::NodePtr g8_node = graph->AddNode(g8_op);
  CreateSetOneNode(g9_op, dim, dtype);
  ge::NodePtr g9_node = graph->AddNode(g9_op);
  CreateSetOneNode(g10_op, dim, dtype);
  ge::NodePtr g10_node = graph->AddNode(g10_op);
  CreateSetOneNode(g11_op, dim, dtype);
  ge::NodePtr g11_node = graph->AddNode(g11_op);
  CreateSetOneNode(g12_op, dim, dtype);
  ge::NodePtr g12_node = graph->AddNode(g12_op);
  CreateSetOneNode(g13_op, dim, dtype);
  ge::NodePtr g13_node = graph->AddNode(g13_op);
  CreateSetOneNode(g14_op, dim, dtype);
  ge::NodePtr g14_node = graph->AddNode(g14_op);
  CreateSetOneNode(g15_op, dim, dtype);
  ge::NodePtr g15_node = graph->AddNode(g15_op);
  CreateSetOneNode(g16_op, dim, dtype);
  ge::NodePtr g16_node = graph->AddNode(g16_op);
  CreateSetOneNode(g17_op, dim, dtype);
  ge::NodePtr g17_node = graph->AddNode(g17_op);
  CreateSetOneNode(g18_op, dim, DT_INT8);
  ge::NodePtr g18_node = graph->AddNode(g18_op);

  Status ret1 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g1_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret1 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g1_node, "tbe-custom");
  Status ret2 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g2_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret2 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g2_node, "tbe-custom");
  Status ret3 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g3_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret3 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g3_node, "tbe-custom");
  Status ret4 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g4_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret4 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g4_node, "tbe-custom");
  Status ret5 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g5_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret5 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g5_node, "tbe-custom");
  Status ret6 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g6_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret6 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g6_node, "tbe-custom");
  Status ret7 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g7_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret7 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g7_node, "tbe-custom");
  Status ret8 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g8_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret8 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g8_node, "tbe-custom");
  Status ret9 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g9_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret9 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g9_node, "tbe-custom");
  Status ret10 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g10_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret10 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g10_node, "tbe-custom");
  Status ret11 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g11_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret11 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g11_node, "tbe-custom");
  Status ret12 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g12_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret12 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g12_node, "tbe-custom");
  Status ret13 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g13_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret13 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g13_node, "tbe-custom");
  Status ret14 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g14_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret14 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g14_node, "tbe-custom");
  Status ret15 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g15_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret15 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g15_node, "tbe-custom");
  Status ret16 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g16_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret16 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g16_node, "tbe-custom");
  Status ret17 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g17_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret17 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g17_node, "tbe-custom");
  Status ret18 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g18_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret18 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g18_node, "tbe-custom");


  ASSERT_EQ(ret1, fe::SUCCESS);
  ASSERT_EQ(ret2, fe::SUCCESS);
  ASSERT_EQ(ret3, fe::SUCCESS);
  ASSERT_EQ(ret4, fe::SUCCESS);
  ASSERT_EQ(ret5, fe::SUCCESS);
  ASSERT_EQ(ret6, fe::SUCCESS);
  ASSERT_EQ(ret7, fe::SUCCESS);
  ASSERT_EQ(ret8, fe::SUCCESS);
  ASSERT_EQ(ret9, fe::SUCCESS);
  ASSERT_EQ(ret10, fe::SUCCESS);
  ASSERT_EQ(ret11, fe::SUCCESS);
  ASSERT_EQ(ret12, fe::SUCCESS);
  ASSERT_EQ(ret13, fe::SUCCESS);
  ASSERT_EQ(ret14, fe::SUCCESS);
  ASSERT_EQ(ret15, fe::SUCCESS);
  ASSERT_EQ(ret16, fe::SUCCESS);
  ASSERT_EQ(ret17, fe::SUCCESS);
  ASSERT_EQ(ret18, fe::SUCCESS);

  vector<ge::DataType> result_input = {DT_FLOAT, DT_BF16, DT_BF16, DT_BF16, DT_UINT8,
                                       DT_BF16, DT_BF16, DT_BF16, DT_FLOAT, DT_UINT8,
                                       DT_FLOAT, DT_BF16, DT_BF16, DT_BF16, DT_BF16,
                                       DT_BF16, DT_BF16};
  vector<ge::DataType> result_output = {DT_FLOAT, DT_BF16, DT_BF16, DT_BF16, DT_UINT8,
                                       DT_BF16, DT_BF16, DT_BF16, DT_FLOAT, DT_UINT8,
                                       DT_FLOAT, DT_BF16, DT_BF16, DT_BF16, DT_BF16,
                                       DT_BF16, DT_BF16};
  EXPECT_EQ(g1_op->GetInputDesc(0).GetDataType(), result_input[0]);
  EXPECT_EQ(g2_op->GetInputDesc(0).GetDataType(), result_input[1]);
  EXPECT_EQ(g3_op->GetInputDesc(0).GetDataType(), result_input[2]);
  EXPECT_EQ(g4_op->GetInputDesc(0).GetDataType(), result_input[3]);
  EXPECT_EQ(g5_op->GetInputDesc(0).GetDataType(), result_input[4]);
  EXPECT_EQ(g6_op->GetInputDesc(0).GetDataType(), result_input[5]);
  EXPECT_EQ(g7_op->GetInputDesc(0).GetDataType(), result_input[6]);
  EXPECT_EQ(g8_op->GetInputDesc(0).GetDataType(), result_input[7]);
  EXPECT_EQ(g9_op->GetInputDesc(0).GetDataType(), result_input[8]);
  EXPECT_EQ(g10_op->GetInputDesc(0).GetDataType(), result_input[9]);
  EXPECT_EQ(g11_op->GetInputDesc(0).GetDataType(), result_input[10]);
  EXPECT_EQ(g12_op->GetInputDesc(0).GetDataType(), result_input[11]);
  EXPECT_EQ(g13_op->GetInputDesc(0).GetDataType(), result_input[12]);
  EXPECT_EQ(g14_op->GetInputDesc(0).GetDataType(), result_input[13]);
  EXPECT_EQ(g15_op->GetInputDesc(0).GetDataType(), result_input[14]);
  EXPECT_EQ(g16_op->GetInputDesc(0).GetDataType(), result_input[15]);
  EXPECT_EQ(g17_op->GetInputDesc(0).GetDataType(), result_input[16]);
  EXPECT_EQ(g18_op->GetInputDesc(0).GetDataType(), DT_INT8);


  EXPECT_EQ(g1_op->GetOutputDesc(0).GetDataType(), result_output[0]);
  EXPECT_EQ(g2_op->GetOutputDesc(0).GetDataType(), result_output[1]);
  EXPECT_EQ(g3_op->GetOutputDesc(0).GetDataType(), result_output[2]);
  EXPECT_EQ(g4_op->GetOutputDesc(0).GetDataType(), result_output[3]);
  EXPECT_EQ(g5_op->GetOutputDesc(0).GetDataType(), result_output[4]);
  EXPECT_EQ(g6_op->GetOutputDesc(0).GetDataType(), result_output[5]);
  EXPECT_EQ(g7_op->GetOutputDesc(0).GetDataType(), result_output[6]);
  EXPECT_EQ(g8_op->GetOutputDesc(0).GetDataType(), result_output[7]);
  EXPECT_EQ(g9_op->GetOutputDesc(0).GetDataType(), result_output[8]);
  EXPECT_EQ(g10_op->GetOutputDesc(0).GetDataType(), result_output[9]);
  EXPECT_EQ(g11_op->GetOutputDesc(0).GetDataType(), result_output[10]);
  EXPECT_EQ(g12_op->GetOutputDesc(0).GetDataType(), result_output[11]);
  EXPECT_EQ(g13_op->GetOutputDesc(0).GetDataType(), result_output[12]);
  EXPECT_EQ(g14_op->GetOutputDesc(0).GetDataType(), result_output[13]);
  EXPECT_EQ(g15_op->GetOutputDesc(0).GetDataType(), result_output[14]);
  EXPECT_EQ(g16_op->GetOutputDesc(0).GetDataType(), result_output[15]);
  EXPECT_EQ(g17_op->GetOutputDesc(0).GetDataType(), result_output[16]);
  EXPECT_EQ(g18_op->GetInputDesc(0).GetDataType(), DT_INT8);
}


TEST_F(UTEST_fusion_engine_op_judge_precision_mode, set_nodes_IO_dtypes_mix_bf16_2)
{
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = ALLOW_MIX_PRECISION_BF16;
  op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g1_op = std::make_shared<OpDesc>("G1", "GOnlyfp32");
  OpDescPtr g2_op = std::make_shared<OpDesc>("G2", "GBlackOnlyBf16");
  OpDescPtr g3_op = std::make_shared<OpDesc>("G3", "GWhiteOnlyBf16");
  OpDescPtr g4_op = std::make_shared<OpDesc>("G4", "GOnlyBf16");
  OpDescPtr g5_op = std::make_shared<OpDesc>("G5", "GOnlyFp16");
  OpDescPtr g6_op = std::make_shared<OpDesc>("G6", "GBlackSupportFp32Bf16");
  OpDescPtr g7_op = std::make_shared<OpDesc>("G7", "GWhiteSupportFp32Bf16");
  OpDescPtr g8_op = std::make_shared<OpDesc>("G8", "GSupportFp32Bf16");
  OpDescPtr g9_op = std::make_shared<OpDesc>("G9", "GBlackSupportfp32Fp16");
  OpDescPtr g10_op = std::make_shared<OpDesc>("G10", "GWhiteSupportfp32Fp16");
  OpDescPtr g11_op = std::make_shared<OpDesc>("G11", "GSupportfp32Fp16");
  OpDescPtr g12_op = std::make_shared<OpDesc>("G12", "GBlackSupportBf16Fp16");
  OpDescPtr g13_op = std::make_shared<OpDesc>("G13", "GWhiteSupportBf16Fp16");
  OpDescPtr g14_op = std::make_shared<OpDesc>("G14", "GSupportBf16Fp16");
  OpDescPtr g15_op = std::make_shared<OpDesc>("G15", "GBlack");
  OpDescPtr g16_op = std::make_shared<OpDesc>("G16", "GWhite");
  OpDescPtr g17_op = std::make_shared<OpDesc>("G17", "G");

  vector<int64_t> dim({4, 33, 12, 16});
  ge::DataType dtype = DT_FLOAT16;
  //add descriptor
  CreateSetOneNode(g1_op, dim, dtype);
  ge::NodePtr g1_node = graph->AddNode(g1_op);
  CreateSetOneNode(g2_op, dim, dtype);
  ge::NodePtr g2_node = graph->AddNode(g2_op);
  CreateSetOneNode(g3_op, dim, dtype);
  ge::NodePtr g3_node = graph->AddNode(g3_op);
  CreateSetOneNode(g4_op, dim, dtype);
  ge::NodePtr g4_node = graph->AddNode(g4_op);
  CreateSetOneNode(g5_op, dim, dtype);
  ge::NodePtr g5_node = graph->AddNode(g5_op);
  CreateSetOneNode(g6_op, dim, dtype);
  ge::NodePtr g6_node = graph->AddNode(g6_op);
  CreateSetOneNode(g7_op, dim, dtype);
  ge::NodePtr g7_node = graph->AddNode(g7_op);
  CreateSetOneNode(g8_op, dim, dtype);
  ge::NodePtr g8_node = graph->AddNode(g8_op);
  CreateSetOneNode(g9_op, dim, dtype);
  ge::NodePtr g9_node = graph->AddNode(g9_op);
  CreateSetOneNode(g10_op, dim, dtype);
  ge::NodePtr g10_node = graph->AddNode(g10_op);
  CreateSetOneNode(g11_op, dim, dtype);
  ge::NodePtr g11_node = graph->AddNode(g11_op);
  CreateSetOneNode(g12_op, dim, dtype);
  ge::NodePtr g12_node = graph->AddNode(g12_op);
  CreateSetOneNode(g13_op, dim, dtype);
  ge::NodePtr g13_node = graph->AddNode(g13_op);
  CreateSetOneNode(g14_op, dim, dtype);
  ge::NodePtr g14_node = graph->AddNode(g14_op);
  CreateSetOneNode(g15_op, dim, dtype);
  ge::NodePtr g15_node = graph->AddNode(g15_op);
  CreateSetOneNode(g16_op, dim, dtype);
  ge::NodePtr g16_node = graph->AddNode(g16_op);
  CreateSetOneNode(g17_op, dim, dtype);
  ge::NodePtr g17_node = graph->AddNode(g17_op);

  Status ret1 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g1_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret1 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g1_node, "tbe-custom");
  Status ret2 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g2_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret2 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g2_node, "tbe-custom");
  Status ret3 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g3_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret3 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g3_node, "tbe-custom");
  Status ret4 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g4_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret4 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g4_node, "tbe-custom");
  Status ret5 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g5_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret5 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g5_node, "tbe-custom");
  Status ret6 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g6_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret6 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g6_node, "tbe-custom");
  Status ret7 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g7_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret7 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g7_node, "tbe-custom");
  Status ret8 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g8_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret8 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g8_node, "tbe-custom");
  Status ret9 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g9_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret9 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g9_node, "tbe-custom");
  Status ret10 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g10_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret10 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g10_node, "tbe-custom");
  Status ret11 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g11_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret11 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g11_node, "tbe-custom");
  Status ret12 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g12_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret12 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g12_node, "tbe-custom");
  Status ret13 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g13_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret13 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g13_node, "tbe-custom");
  Status ret14 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g14_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret14 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g14_node, "tbe-custom");
  Status ret15 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g15_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret15 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g15_node, "tbe-custom");
  Status ret16 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g16_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret16 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g16_node, "tbe-custom");
  Status ret17 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g17_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret17 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g17_node, "tbe-custom");


  ASSERT_EQ(ret1, fe::SUCCESS);
  ASSERT_EQ(ret2, fe::SUCCESS);
  ASSERT_EQ(ret3, fe::SUCCESS);
  ASSERT_EQ(ret4, fe::SUCCESS);
  ASSERT_EQ(ret5, fe::SUCCESS);
  ASSERT_EQ(ret6, fe::SUCCESS);
  ASSERT_EQ(ret7, fe::SUCCESS);
  ASSERT_EQ(ret8, fe::SUCCESS);
  ASSERT_EQ(ret9, fe::SUCCESS);
  ASSERT_EQ(ret10, fe::SUCCESS);
  ASSERT_EQ(ret11, fe::SUCCESS);
  ASSERT_EQ(ret12, fe::SUCCESS);
  ASSERT_EQ(ret13, fe::SUCCESS);
  ASSERT_EQ(ret14, fe::SUCCESS);
  ASSERT_EQ(ret15, fe::SUCCESS);
  ASSERT_EQ(ret16, fe::SUCCESS);
  ASSERT_EQ(ret17, fe::SUCCESS);

  vector<ge::DataType> result_input = {DT_FLOAT, DT_UINT8, DT_BF16, DT_UINT8, DT_FLOAT16,
                                       DT_FLOAT, DT_BF16, DT_FLOAT, DT_FLOAT16, DT_FLOAT16,
                                       DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_FLOAT16,
                                       DT_BF16, DT_FLOAT16};
  vector<ge::DataType> result_output = {DT_FLOAT, DT_UINT8, DT_BF16, DT_UINT8, DT_FLOAT16,
                                        DT_FLOAT, DT_BF16, DT_FLOAT, DT_FLOAT16, DT_FLOAT16,
                                        DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_FLOAT16,
                                        DT_BF16, DT_FLOAT16};
  EXPECT_EQ(g1_op->GetInputDesc(0).GetDataType(), result_input[0]);
  EXPECT_EQ(g2_op->GetInputDesc(0).GetDataType(), result_input[1]);
  EXPECT_EQ(g3_op->GetInputDesc(0).GetDataType(), result_input[2]);
  EXPECT_EQ(g4_op->GetInputDesc(0).GetDataType(), result_input[3]);
  EXPECT_EQ(g5_op->GetInputDesc(0).GetDataType(), result_input[4]);
  EXPECT_EQ(g6_op->GetInputDesc(0).GetDataType(), result_input[5]);
  EXPECT_EQ(g7_op->GetInputDesc(0).GetDataType(), result_input[6]);
  EXPECT_EQ(g8_op->GetInputDesc(0).GetDataType(), result_input[7]);
  EXPECT_EQ(g9_op->GetInputDesc(0).GetDataType(), result_input[8]);
  EXPECT_EQ(g10_op->GetInputDesc(0).GetDataType(), result_input[9]);
  EXPECT_EQ(g11_op->GetInputDesc(0).GetDataType(), result_input[10]);
  EXPECT_EQ(g12_op->GetInputDesc(0).GetDataType(), result_input[11]);
  EXPECT_EQ(g13_op->GetInputDesc(0).GetDataType(), result_input[12]);
  EXPECT_EQ(g14_op->GetInputDesc(0).GetDataType(), result_input[13]);
  EXPECT_EQ(g15_op->GetInputDesc(0).GetDataType(), result_input[14]);
  EXPECT_EQ(g16_op->GetInputDesc(0).GetDataType(), result_input[15]);
  EXPECT_EQ(g17_op->GetInputDesc(0).GetDataType(), result_input[16]);


  EXPECT_EQ(g1_op->GetOutputDesc(0).GetDataType(), result_output[0]);
  EXPECT_EQ(g2_op->GetOutputDesc(0).GetDataType(), result_output[1]);
  EXPECT_EQ(g3_op->GetOutputDesc(0).GetDataType(), result_output[2]);
  EXPECT_EQ(g4_op->GetOutputDesc(0).GetDataType(), result_output[3]);
  EXPECT_EQ(g5_op->GetOutputDesc(0).GetDataType(), result_output[4]);
  EXPECT_EQ(g6_op->GetOutputDesc(0).GetDataType(), result_output[5]);
  EXPECT_EQ(g7_op->GetOutputDesc(0).GetDataType(), result_output[6]);
  EXPECT_EQ(g8_op->GetOutputDesc(0).GetDataType(), result_output[7]);
  EXPECT_EQ(g9_op->GetOutputDesc(0).GetDataType(), result_output[8]);
  EXPECT_EQ(g10_op->GetOutputDesc(0).GetDataType(), result_output[9]);
  EXPECT_EQ(g11_op->GetOutputDesc(0).GetDataType(), result_output[10]);
  EXPECT_EQ(g12_op->GetOutputDesc(0).GetDataType(), result_output[11]);
  EXPECT_EQ(g13_op->GetOutputDesc(0).GetDataType(), result_output[12]);
  EXPECT_EQ(g14_op->GetOutputDesc(0).GetDataType(), result_output[13]);
  EXPECT_EQ(g15_op->GetOutputDesc(0).GetDataType(), result_output[14]);
  EXPECT_EQ(g16_op->GetOutputDesc(0).GetDataType(), result_output[15]);
  EXPECT_EQ(g17_op->GetOutputDesc(0).GetDataType(), result_output[16]);
}

// test for net in gray, input fp32-> GWhite -> GBlackSupportFp32Bf16 -> GWhiteSupportBf16Fp16 -> G
// the fllows output is        fp32 -> bf16 ->  fp32                  -> bf16                  -> bf16
TEST_F(UTEST_fusion_engine_op_judge_precision_mode, set_nodes_IO_dtypes_mix_bf16_3)
{
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = ALLOW_MIX_PRECISION_BF16;
  op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g1_op = std::make_shared<OpDesc>("G1", "GWhite");
  OpDescPtr g2_op = std::make_shared<OpDesc>("G2", "GBlackSupportFp32Bf16");
  OpDescPtr g3_op = std::make_shared<OpDesc>("G3", "GWhiteSupportBf16Fp16");
  OpDescPtr g4_op = std::make_shared<OpDesc>("G4", "G");

  vector<int64_t> dim({4, 33, 12, 16});
  ge::DataType dtype = DT_FLOAT;
  //add descriptor
  CreateSetOneNode(g1_op, dim, dtype);
  ge::NodePtr g1_node = graph->AddNode(g1_op);
  CreateSetOneNode(g2_op, dim, dtype);
  ge::NodePtr g2_node = graph->AddNode(g2_op);
  GraphUtils::AddEdge(g1_node->GetOutDataAnchor(0), g2_node->GetInDataAnchor(0));
  CreateSetOneNode(g3_op, dim, dtype);
  ge::NodePtr g3_node = graph->AddNode(g3_op);
  GraphUtils::AddEdge(g2_node->GetOutDataAnchor(0), g3_node->GetInDataAnchor(0));
  CreateSetOneNode(g4_op, dim, dtype);
  ge::NodePtr g4_node = graph->AddNode(g4_op);
  GraphUtils::AddEdge(g3_node->GetOutDataAnchor(0), g4_node->GetInDataAnchor(0));

  Status ret1 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g1_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret1 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g1_node, "tbe-custom");
  Status ret2 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g2_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret2 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g2_node, "tbe-custom");
  Status ret3 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g3_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret3 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g3_node, "tbe-custom");
  Status ret4 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g4_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret4 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g4_node, "tbe-custom");

  ASSERT_EQ(ret1, fe::SUCCESS);
  ASSERT_EQ(ret2, fe::SUCCESS);
  ASSERT_EQ(ret3, fe::SUCCESS);
  ASSERT_EQ(ret4, fe::SUCCESS);


  vector<ge::DataType> result_input = { DT_BF16, DT_FLOAT, DT_BF16, DT_BF16 };
  vector<ge::DataType> result_output = { DT_BF16, DT_FLOAT, DT_BF16, DT_BF16 };
  EXPECT_EQ(g1_op->GetInputDesc(0).GetDataType(), result_input[0]);
  EXPECT_EQ(g2_op->GetInputDesc(0).GetDataType(), result_input[1]);
  EXPECT_EQ(g3_op->GetInputDesc(0).GetDataType(), result_input[2]);
  EXPECT_EQ(g4_op->GetInputDesc(0).GetDataType(), result_input[3]);

  EXPECT_EQ(g1_op->GetOutputDesc(0).GetDataType(), result_output[0]);
  EXPECT_EQ(g2_op->GetOutputDesc(0).GetDataType(), result_output[1]);
  EXPECT_EQ(g3_op->GetOutputDesc(0).GetDataType(), result_output[2]);
  EXPECT_EQ(g4_op->GetOutputDesc(0).GetDataType(), result_output[3]);
}
TEST_F(UTEST_fusion_engine_op_judge_precision_mode, set_nodes_IO_dtypes_mix_bf16_3_v2)
{
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = V2_MIX_BF16;
  op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME,
                                                                    reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g1_op = std::make_shared<OpDesc>("G1", "GWhite");
  OpDescPtr g2_op = std::make_shared<OpDesc>("G2", "GBlackSupportFp32Bf16");
  OpDescPtr g3_op = std::make_shared<OpDesc>("G3", "GWhiteSupportBf16Fp16");
  OpDescPtr g4_op = std::make_shared<OpDesc>("G4", "G");

  vector<int64_t> dim({4, 33, 12, 16});
  ge::DataType dtype = DT_FLOAT;
  //add descriptor
  CreateSetOneNode(g1_op, dim, dtype);
  ge::NodePtr g1_node = graph->AddNode(g1_op);
  CreateSetOneNode(g2_op, dim, dtype);
  ge::NodePtr g2_node = graph->AddNode(g2_op);
  GraphUtils::AddEdge(g1_node->GetOutDataAnchor(0), g2_node->GetInDataAnchor(0));
  CreateSetOneNode(g3_op, dim, dtype);
  ge::NodePtr g3_node = graph->AddNode(g3_op);
  GraphUtils::AddEdge(g2_node->GetOutDataAnchor(0), g3_node->GetInDataAnchor(0));
  CreateSetOneNode(g4_op, dim, dtype);
  ge::NodePtr g4_node = graph->AddNode(g4_op);
  GraphUtils::AddEdge(g3_node->GetOutDataAnchor(0), g4_node->GetInDataAnchor(0));

  Status ret1 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g1_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret1 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g1_node, "tbe-custom");
  Status ret2 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g2_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret2 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g2_node, "tbe-custom");
  Status ret3 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g3_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret3 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g3_node, "tbe-custom");
  Status ret4 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g4_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret4 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g4_node, "tbe-custom");

  ASSERT_EQ(ret1, fe::SUCCESS);
  ASSERT_EQ(ret2, fe::SUCCESS);
  ASSERT_EQ(ret3, fe::SUCCESS);
  ASSERT_EQ(ret4, fe::SUCCESS);


  vector<ge::DataType> result_input = { DT_BF16, DT_FLOAT, DT_BF16, DT_BF16 };
  vector<ge::DataType> result_output = { DT_BF16, DT_FLOAT, DT_BF16, DT_BF16 };
  EXPECT_EQ(g1_op->GetInputDesc(0).GetDataType(), result_input[0]);
  EXPECT_EQ(g2_op->GetInputDesc(0).GetDataType(), result_input[1]);
  EXPECT_EQ(g3_op->GetInputDesc(0).GetDataType(), result_input[2]);
  EXPECT_EQ(g4_op->GetInputDesc(0).GetDataType(), result_input[3]);

  EXPECT_EQ(g1_op->GetOutputDesc(0).GetDataType(), result_output[0]);
  EXPECT_EQ(g2_op->GetOutputDesc(0).GetDataType(), result_output[1]);
  EXPECT_EQ(g3_op->GetOutputDesc(0).GetDataType(), result_output[2]);
  EXPECT_EQ(g4_op->GetOutputDesc(0).GetDataType(), result_output[3]);
}
// test for net in gray, input fp32-> G     -> Cast -> GWhite -> Cast1 -> GWhite
// the fllows output is        fp32 -> bf16 ->  fp32                  -> bf16                  -> bf16
TEST_F(UTEST_fusion_engine_op_judge_precision_mode, set_nodes_IO_dtypes_mix_bf16_4)
{
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = ALLOW_MIX_PRECISION_BF16;
  op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();
  /* Stub Cast in Black list */
  SubOpInfoStorePtr subOpInfoStorePtr = OpsKernelManager::Instance(AI_CORE_NAME).GetSubOpsKernelByStoreName("tbe-custom");
  OpKernelInfoPtr opKernelInfoPtr = subOpInfoStorePtr->GetOpKernelByOpType(fe::CAST);
  opKernelInfoPtr->op_param_vec_[static_cast<size_t>(OP_KERNEL_PARAM::PrecisionPolicy)] = static_cast<int64_t>(BLACK);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g_op = std::make_shared<OpDesc>("G1", "G");
  OpDescPtr cast_op = std::make_shared<OpDesc>("Cast", "Cast");
  OpDescPtr h_op = std::make_shared<OpDesc>("G2", "GWhite");
  OpDescPtr cast1_op = std::make_shared<OpDesc>("Cast1", "Cast");
  OpDescPtr u_op = std::make_shared<OpDesc>("G3", "GWhite");

  //add descriptor
  vector<int64_t> dim({4, 33, 12, 16});
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NCHW);
  tensor_desc.SetFormat(FORMAT_NCHW);
  tensor_desc.SetDataType(DT_BF16);
  g_op->AddInputDesc("x", tensor_desc);
  g_op->AddOutputDesc("z", tensor_desc);
  ge::AttrUtils::SetInt(g_op, FE_IMPLY_TYPE, 6);
  ge::NodePtr g_node = graph->AddNode(g_op);

  vector<int64_t> dim_cast({4, 33, 12, 16});
  GeShape shape_cast(dim);
  GeTensorDesc tensor_desc_cast(shape_cast);
  tensor_desc_cast.SetOriginFormat(FORMAT_NCHW);
  tensor_desc_cast.SetFormat(FORMAT_NCHW);
  tensor_desc_cast.SetDataType(DT_FLOAT);
  cast_op->AddInputDesc("x", tensor_desc);
  cast_op->AddOutputDesc("z", tensor_desc_cast);
  ge::AttrUtils::SetInt(cast_op, FE_IMPLY_TYPE, 6);
  ge::NodePtr cast_node = graph->AddNode(cast_op);

  vector<int64_t> dim_h({1, 2, 3, 4});
  GeShape shape_h(dim_h);
  GeTensorDesc tensor_desc_h(shape_h);
  tensor_desc_h.SetOriginFormat(FORMAT_NCHW);
  tensor_desc_h.SetFormat(FORMAT_NCHW);
  tensor_desc_h.SetDataType(DT_FLOAT);
  h_op->AddInputDesc("x", tensor_desc_h);
  h_op->AddOutputDesc("z", tensor_desc_h);
  ge::AttrUtils::SetInt(h_op, FE_IMPLY_TYPE, 6);
  ge::NodePtr h_node = graph->AddNode(h_op);

  vector<int64_t> dim_cast1({1, 2, 3, 4});
  GeShape shape_cast1(dim_cast1);
  GeTensorDesc tensor_desc_cast1(shape_cast1);
  tensor_desc_cast1.SetOriginFormat(FORMAT_NCHW);
  tensor_desc_cast1.SetFormat(FORMAT_NCHW);
  tensor_desc_cast1.SetDataType(DT_BF16);
  cast1_op->AddInputDesc("x", tensor_desc_cast1);
  cast1_op->AddOutputDesc("z", tensor_desc_cast1);
  ge::AttrUtils::SetInt(h_op, FE_IMPLY_TYPE, 6);
  ge::NodePtr cast1_node = graph->AddNode(cast1_op);

  vector<int64_t> dim_u({1, 2, 3, 4});
  GeShape shape_u(dim_u);
  GeTensorDesc tensor_desc_u(shape_u);
  tensor_desc_u.SetOriginFormat(FORMAT_NCHW);
  tensor_desc_u.SetFormat(FORMAT_NCHW);
  tensor_desc_u.SetDataType(DT_FLOAT);
  u_op->AddInputDesc("x", tensor_desc_u);
  u_op->AddOutputDesc("z", tensor_desc_u);
  ge::AttrUtils::SetInt(u_op, FE_IMPLY_TYPE, 6);
  ge::NodePtr u_node = graph->AddNode(u_op);
  GraphUtils::AddEdge(g_node->GetOutDataAnchor(0), cast_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(cast_node->GetOutDataAnchor(0), h_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(h_node->GetOutDataAnchor(0), cast1_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(cast1_node->GetOutDataAnchor(0), u_node->GetInDataAnchor(0));


  Status ret1 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret1 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g_node, "tbe-custom");
  Status ret2 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(h_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret2 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(h_node, "tbe-custom");
  Status ret3 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(u_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret3 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(u_node, "tbe-custom");
  ASSERT_EQ(ret1, fe::SUCCESS);
  ASSERT_EQ(ret2, fe::SUCCESS);
  ASSERT_EQ(ret3, fe::SUCCESS);

  EXPECT_EQ(g_op->GetInputDesc(0).GetDataType(), DT_BF16);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetDataType(), DT_BF16);

  EXPECT_EQ(cast_op->GetInputDesc(0).GetDataType(), DT_BF16);
  EXPECT_EQ(cast_op->GetOutputDesc(0).GetDataType(), DT_FLOAT);

  EXPECT_EQ(h_op->GetInputDesc(0).GetDataType(), DT_BF16);
  EXPECT_EQ(h_op->GetOutputDesc(0).GetDataType(), DT_BF16);

  EXPECT_EQ(cast1_op->GetInputDesc(0).GetDataType(), DT_BF16);
  EXPECT_EQ(cast1_op->GetOutputDesc(0).GetDataType(), DT_BF16);

  EXPECT_EQ(u_op->GetInputDesc(0).GetDataType(), DT_BF16);
  EXPECT_EQ(u_op->GetOutputDesc(0).GetDataType(), DT_BF16);
}

TEST_F(UTEST_fusion_engine_op_judge_precision_mode, judge_node_by_customize_dtype_1)
{
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = ALLOW_FP32_TO_FP16;
  op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();

  SubOpInfoStorePtr subOpInfoStorePtr = OpsKernelManager::Instance(AI_CORE_NAME).GetSubOpsKernelByStoreName("tbe-custom");

  OpDescPtr mat_mul_op = std::make_shared<OpDesc>("mat_mul_name", "MatMulV2");
  OpDescPtr data_op = std::make_shared<OpDesc>("data", "Data");
  OpDescPtr const_op1 = std::make_shared<OpDesc>("const1", "Const");
  OpDescPtr const_op2 = std::make_shared<OpDesc>("const2", "Const");
  OpDescPtr const_op3 = std::make_shared<OpDesc>("const3", "Const");

  //add descriptor
  vector<int64_t> dim({4, 33, 12, 16});
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_ND);
  tensor_desc.SetOriginDataType(DT_FLOAT);
  tensor_desc.SetOriginShape(shape);

  mat_mul_op->AddInputDesc("x1", tensor_desc);
  mat_mul_op->AddInputDesc("x2", tensor_desc);
  mat_mul_op->AddInputDesc("bias", tensor_desc);
  GeTensorDesc tensor_desc1(shape);
  tensor_desc1.SetDataType(DT_INT32);
  tensor_desc1.SetOriginFormat(FORMAT_ND);
  tensor_desc1.SetOriginDataType(DT_INT32);
  tensor_desc1.SetOriginShape(shape);
  mat_mul_op->AddInputDesc("offset", tensor_desc1);
  mat_mul_op->AddOutputDesc("x2", tensor_desc);
  ge::AttrUtils::SetInt(mat_mul_op, FE_IMPLY_TYPE, 6);

  data_op->AddOutputDesc(mat_mul_op->GetInputDesc(0));
  const_op1->AddOutputDesc(mat_mul_op->GetInputDesc(1));
  const_op2->AddOutputDesc(mat_mul_op->GetInputDesc(2));
  const_op3->AddOutputDesc(mat_mul_op->GetInputDesc(3));

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  ge::NodePtr mat_mul_node = graph->AddNode(mat_mul_op);
  ge::NodePtr data_node = graph->AddNode(data_op);
  ge::NodePtr const_node1 = graph->AddNode(const_op1);
  ge::NodePtr const_node2 = graph->AddNode(const_op2);
  ge::NodePtr const_node3 = graph->AddNode(const_op3);

  GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), mat_mul_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(const_node1->GetOutDataAnchor(0), mat_mul_node->GetInDataAnchor(1));
  GraphUtils::AddEdge(const_node2->GetOutDataAnchor(0), mat_mul_node->GetInDataAnchor(2));
  GraphUtils::AddEdge(const_node3->GetOutDataAnchor(0), mat_mul_node->GetInDataAnchor(3));

  op_cust_dtypes_parser_ptr_->op_type_cust_dtypes_.clear();
  op_cust_dtypes_parser_ptr_->op_name_cust_dtypes_.clear();

  Status ret = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(mat_mul_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(mat_mul_node, "tbe-custom");
  ASSERT_EQ(ret, fe::SUCCESS);

  EXPECT_EQ(mat_mul_op->GetInputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(mat_mul_op->GetInputDesc(1).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(mat_mul_op->GetInputDesc(2).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(mat_mul_op->GetInputDesc(3).GetDataType(), DT_INT32);
  EXPECT_EQ(mat_mul_op->GetOutputDesc(0).GetDataType(), DT_FLOAT16);

  OpCustomizeDtype op_cust_dtype;
  op_cust_dtype.input_dtypes = {DT_FLOAT16, DT_FLOAT16, DT_UNDEFINED};
  op_cust_dtype.output_dtypes = {DT_FLOAT16};
  op_cust_dtypes_parser_ptr_->op_type_cust_dtypes_.emplace(mat_mul_op->GetType(), op_cust_dtype);
  ret = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(mat_mul_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(mat_mul_node, "tbe-custom");
  ASSERT_EQ(ret, fe::SUCCESS);

  EXPECT_EQ(mat_mul_op->GetInputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(mat_mul_op->GetInputDesc(1).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(mat_mul_op->GetInputDesc(2).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(mat_mul_op->GetInputDesc(3).GetDataType(), DT_INT32);
  EXPECT_EQ(mat_mul_op->GetOutputDesc(0).GetDataType(), DT_FLOAT16);

  OpCustomizeDtype op_cust_dtype1;
  op_cust_dtype1.input_dtypes = {DT_FLOAT16, DT_FLOAT16};
  op_cust_dtype1.output_dtypes = {DT_FLOAT};
  op_cust_dtypes_parser_ptr_->op_name_cust_dtypes_.emplace(mat_mul_op->GetName(), op_cust_dtype1);
  ret = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(mat_mul_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ASSERT_EQ(ret, fe::FAILED);
  ret = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(mat_mul_node, "tbe-custom");
  ASSERT_EQ(ret, fe::SUCCESS);
}

TEST_F(UTEST_fusion_engine_op_judge_precision_mode, judge_node_by_customize_dtype_2)
{
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = ALLOW_FP32_TO_FP16;
  op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();

  SubOpInfoStorePtr subOpInfoStorePtr = OpsKernelManager::Instance(AI_CORE_NAME).GetSubOpsKernelByStoreName("tbe-custom");

  OpDescPtr mat_mul_op = std::make_shared<OpDesc>("mat_mul_name", "MatMulV3");
  OpDescPtr data_op = std::make_shared<OpDesc>("data", "Data");
  OpDescPtr const_op1 = std::make_shared<OpDesc>("const1", "Const");
  OpDescPtr const_op2 = std::make_shared<OpDesc>("const2", "Const");
  OpDescPtr const_op3 = std::make_shared<OpDesc>("const3", "Const");

  //add descriptor
  vector<int64_t> dim({4, 33, 12, 16});
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_ND);
  tensor_desc.SetOriginDataType(DT_FLOAT);
  tensor_desc.SetOriginShape(shape);

  mat_mul_op->AddInputDesc("x1", tensor_desc);
  mat_mul_op->AddInputDesc("x2", tensor_desc);
  mat_mul_op->AddInputDesc("bias", tensor_desc);
  GeTensorDesc tensor_desc1(shape);
  tensor_desc1.SetDataType(DT_INT32);
  tensor_desc1.SetOriginFormat(FORMAT_ND);
  tensor_desc1.SetOriginDataType(DT_INT32);
  tensor_desc1.SetOriginShape(shape);
  mat_mul_op->AddInputDesc("offset", tensor_desc1);
  mat_mul_op->AddOutputDesc("x2", tensor_desc);
  ge::AttrUtils::SetInt(mat_mul_op, FE_IMPLY_TYPE, 6);

  data_op->AddOutputDesc(mat_mul_op->GetInputDesc(0));
  const_op1->AddOutputDesc(mat_mul_op->GetInputDesc(1));
  const_op2->AddOutputDesc(mat_mul_op->GetInputDesc(2));
  const_op3->AddOutputDesc(mat_mul_op->GetInputDesc(3));

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  ge::NodePtr mat_mul_node = graph->AddNode(mat_mul_op);
  ge::NodePtr data_node = graph->AddNode(data_op);
  ge::NodePtr const_node1 = graph->AddNode(const_op1);
  ge::NodePtr const_node2 = graph->AddNode(const_op2);
  ge::NodePtr const_node3 = graph->AddNode(const_op3);

  GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), mat_mul_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(const_node1->GetOutDataAnchor(0), mat_mul_node->GetInDataAnchor(1));
  GraphUtils::AddEdge(const_node2->GetOutDataAnchor(0), mat_mul_node->GetInDataAnchor(2));
  GraphUtils::AddEdge(const_node3->GetOutDataAnchor(0), mat_mul_node->GetInDataAnchor(3));

  op_cust_dtypes_parser_ptr_->op_type_cust_dtypes_.clear();
  op_cust_dtypes_parser_ptr_->op_name_cust_dtypes_.clear();

  Status ret = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(mat_mul_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(mat_mul_node, "tbe-custom");
  ASSERT_EQ(ret, fe::SUCCESS);

  EXPECT_EQ(mat_mul_op->GetInputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(mat_mul_op->GetInputDesc(1).GetDataType(), DT_FLOAT);
  EXPECT_EQ(mat_mul_op->GetInputDesc(2).GetDataType(), DT_FLOAT);
  EXPECT_EQ(mat_mul_op->GetInputDesc(3).GetDataType(), DT_INT32);
  EXPECT_EQ(mat_mul_op->GetOutputDesc(0).GetDataType(), DT_FLOAT);

  OpCustomizeDtype op_cust_dtype;
  op_cust_dtype.input_dtypes = {DT_FLOAT16, DT_FLOAT16};
  op_cust_dtype.output_dtypes = {DT_FLOAT16};
  op_cust_dtypes_parser_ptr_->op_type_cust_dtypes_.emplace(mat_mul_op->GetType(), op_cust_dtype);
  ret = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(mat_mul_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(mat_mul_node, "tbe-custom");
  ASSERT_EQ(ret, fe::SUCCESS);

  EXPECT_EQ(mat_mul_op->GetInputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(mat_mul_op->GetInputDesc(1).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(mat_mul_op->GetInputDesc(2).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(mat_mul_op->GetInputDesc(3).GetDataType(), DT_INT32);
  EXPECT_EQ(mat_mul_op->GetOutputDesc(0).GetDataType(), DT_FLOAT16);

  OpCustomizeDtype op_cust_dtype1;
  op_cust_dtype1.input_dtypes = {DT_FLOAT16, DT_FLOAT16};
  op_cust_dtype1.output_dtypes = {DT_FLOAT};
  op_cust_dtypes_parser_ptr_->op_name_cust_dtypes_.emplace(mat_mul_op->GetName(), op_cust_dtype1);
  ret = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(mat_mul_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(mat_mul_node, "tbe-custom");
  ASSERT_EQ(ret, fe::SUCCESS);

  EXPECT_EQ(mat_mul_op->GetInputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(mat_mul_op->GetInputDesc(1).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(mat_mul_op->GetInputDesc(2).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(mat_mul_op->GetInputDesc(3).GetDataType(), DT_INT32);
  EXPECT_EQ(mat_mul_op->GetOutputDesc(0).GetDataType(), DT_FLOAT);

  OpCustomizeDtype op_cust_dtype2;
  op_cust_dtype2.input_dtypes = {DT_FLOAT16, DT_FLOAT};
  op_cust_dtype2.output_dtypes = {DT_FLOAT};
  op_cust_dtypes_parser_ptr_->op_name_cust_dtypes_[mat_mul_op->GetName()] = op_cust_dtype2;
  ret = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(mat_mul_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ASSERT_EQ(ret, fe::FAILED);
  ret = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(mat_mul_node, "tbe-custom");
  ASSERT_EQ(ret, fe::SUCCESS);
}

TEST_F(UTEST_fusion_engine_op_judge_precision_mode, judge_node_allowfp32to16_cube)
{
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = ALLOW_FP32_TO_FP16;
  PlatformUtils::Instance().soc_version_ = "Ascend910B2";
  Configuration::Instance(fe::AI_CORE_NAME).fp16_op_type_list_ = {"MatMulV3"};
  op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();

  SubOpInfoStorePtr subOpInfoStorePtr = OpsKernelManager::Instance(AI_CORE_NAME).GetSubOpsKernelByStoreName("tbe-custom");

  OpDescPtr mat_mul_op = std::make_shared<OpDesc>("mat_mul_name", "MatMulV3");
  OpDescPtr data_op = std::make_shared<OpDesc>("data", "Data");
  OpDescPtr const_op1 = std::make_shared<OpDesc>("const1", "Const");
  OpDescPtr const_op2 = std::make_shared<OpDesc>("const2", "Const");
  OpDescPtr const_op3 = std::make_shared<OpDesc>("const3", "Const");

  //add descriptor
  vector<int64_t> dim({4, 33, 12, 16});
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_ND);
  tensor_desc.SetOriginDataType(DT_FLOAT);
  tensor_desc.SetOriginShape(shape);
  (void)ge::AttrUtils::SetStr(mat_mul_op, "_sgt_cube_vector_core_type", "AiCore");
  mat_mul_op->AddInputDesc("x1", tensor_desc);
  mat_mul_op->AddInputDesc("x2", tensor_desc);
  mat_mul_op->AddInputDesc("bias", tensor_desc);
  GeTensorDesc tensor_desc1(shape);
  tensor_desc1.SetDataType(DT_INT32);
  tensor_desc1.SetOriginFormat(FORMAT_ND);
  tensor_desc1.SetOriginDataType(DT_INT32);
  tensor_desc1.SetOriginShape(shape);
  mat_mul_op->AddInputDesc("offset", tensor_desc1);
  mat_mul_op->AddOutputDesc("x2", tensor_desc);
  ge::AttrUtils::SetInt(mat_mul_op, FE_IMPLY_TYPE, 6);

  data_op->AddOutputDesc(mat_mul_op->GetInputDesc(0));
  const_op1->AddOutputDesc(mat_mul_op->GetInputDesc(1));
  const_op2->AddOutputDesc(mat_mul_op->GetInputDesc(2));
  const_op3->AddOutputDesc(mat_mul_op->GetInputDesc(3));

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  ge::NodePtr mat_mul_node = graph->AddNode(mat_mul_op);
  ge::NodePtr data_node = graph->AddNode(data_op);
  ge::NodePtr const_node1 = graph->AddNode(const_op1);
  ge::NodePtr const_node2 = graph->AddNode(const_op2);
  ge::NodePtr const_node3 = graph->AddNode(const_op3);

  GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), mat_mul_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(const_node1->GetOutDataAnchor(0), mat_mul_node->GetInDataAnchor(1));
  GraphUtils::AddEdge(const_node2->GetOutDataAnchor(0), mat_mul_node->GetInDataAnchor(2));
  GraphUtils::AddEdge(const_node3->GetOutDataAnchor(0), mat_mul_node->GetInDataAnchor(3));

  op_cust_dtypes_parser_ptr_->op_type_cust_dtypes_.clear();
  op_cust_dtypes_parser_ptr_->op_name_cust_dtypes_.clear();

  Status ret = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(mat_mul_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(mat_mul_node, "tbe-custom");
  ASSERT_EQ(ret, fe::SUCCESS);

  EXPECT_EQ(mat_mul_op->GetInputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(mat_mul_op->GetInputDesc(1).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(mat_mul_op->GetInputDesc(2).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(mat_mul_op->GetInputDesc(3).GetDataType(), DT_INT32);
  EXPECT_EQ(mat_mul_op->GetOutputDesc(0).GetDataType(), DT_FLOAT16);
}

TEST_F(UTEST_fusion_engine_op_judge_precision_mode, judge_node_allowfp32to16_cube_keepdtype)
{
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = ALLOW_FP32_TO_FP16;
  PlatformUtils::Instance().soc_version_ = "Ascend910B2";
  Configuration::Instance(fe::AI_CORE_NAME).fp16_op_type_list_ = {"MatMulV3"};
  op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();

  SubOpInfoStorePtr subOpInfoStorePtr = OpsKernelManager::Instance(AI_CORE_NAME).GetSubOpsKernelByStoreName("tbe-custom");

  OpDescPtr mat_mul_op = std::make_shared<OpDesc>("mat_mul_name", "MatMulV3");
  OpDescPtr data_op = std::make_shared<OpDesc>("data", "Data");
  OpDescPtr const_op1 = std::make_shared<OpDesc>("const1", "Const");
  OpDescPtr const_op2 = std::make_shared<OpDesc>("const2", "Const");
  OpDescPtr const_op3 = std::make_shared<OpDesc>("const3", "Const");

  //add descriptor
  vector<int64_t> dim({4, 33, 12, 16});
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_ND);
  tensor_desc.SetOriginDataType(DT_FLOAT);
  tensor_desc.SetOriginShape(shape);
  (void)ge::AttrUtils::SetStr(mat_mul_op, "_sgt_cube_vector_core_type", "AiCore");
  mat_mul_op->AddInputDesc("x1", tensor_desc);
  mat_mul_op->AddInputDesc("x2", tensor_desc);
  mat_mul_op->AddInputDesc("bias", tensor_desc);
  GeTensorDesc tensor_desc1(shape);
  tensor_desc1.SetDataType(DT_INT32);
  tensor_desc1.SetOriginFormat(FORMAT_ND);
  tensor_desc1.SetOriginDataType(DT_INT32);
  tensor_desc1.SetOriginShape(shape);
  mat_mul_op->AddInputDesc("offset", tensor_desc1);
  mat_mul_op->AddOutputDesc("x2", tensor_desc);
  ge::AttrUtils::SetInt(mat_mul_op, FE_IMPLY_TYPE, 6);
  ge::AttrUtils::SetInt(mat_mul_op, KEEP_DTYPE, 1);

  data_op->AddOutputDesc(mat_mul_op->GetInputDesc(0));
  const_op1->AddOutputDesc(mat_mul_op->GetInputDesc(1));
  const_op2->AddOutputDesc(mat_mul_op->GetInputDesc(2));
  const_op3->AddOutputDesc(mat_mul_op->GetInputDesc(3));

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  ge::NodePtr mat_mul_node = graph->AddNode(mat_mul_op);
  ge::NodePtr data_node = graph->AddNode(data_op);
  ge::NodePtr const_node1 = graph->AddNode(const_op1);
  ge::NodePtr const_node2 = graph->AddNode(const_op2);
  ge::NodePtr const_node3 = graph->AddNode(const_op3);

  GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), mat_mul_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(const_node1->GetOutDataAnchor(0), mat_mul_node->GetInDataAnchor(1));
  GraphUtils::AddEdge(const_node2->GetOutDataAnchor(0), mat_mul_node->GetInDataAnchor(2));
  GraphUtils::AddEdge(const_node3->GetOutDataAnchor(0), mat_mul_node->GetInDataAnchor(3));

  op_cust_dtypes_parser_ptr_->op_type_cust_dtypes_.clear();
  op_cust_dtypes_parser_ptr_->op_name_cust_dtypes_.clear();

  Status ret = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(mat_mul_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(mat_mul_node, "tbe-custom");
  ASSERT_EQ(ret, fe::SUCCESS);

  EXPECT_EQ(mat_mul_op->GetInputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(mat_mul_op->GetInputDesc(1).GetDataType(), DT_FLOAT);
  EXPECT_EQ(mat_mul_op->GetInputDesc(2).GetDataType(), DT_FLOAT);
  EXPECT_EQ(mat_mul_op->GetInputDesc(3).GetDataType(), DT_INT32);
  EXPECT_EQ(mat_mul_op->GetOutputDesc(0).GetDataType(), DT_FLOAT);
}

TEST_F(UTEST_fusion_engine_op_judge_precision_mode, judge_node_allowfp32to16_cube_int8)
{
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = ALLOW_FP32_TO_FP16;
  PlatformUtils::Instance().soc_version_ = "Ascend910B2";
  Configuration::Instance(fe::AI_CORE_NAME).fp16_op_type_list_ = {"MatMulV3"};
  op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();

  SubOpInfoStorePtr subOpInfoStorePtr = OpsKernelManager::Instance(AI_CORE_NAME).GetSubOpsKernelByStoreName("tbe-custom");

  OpDescPtr mat_mul_op = std::make_shared<OpDesc>("mat_mul_name", "MatMulV3");
  OpDescPtr data_op = std::make_shared<OpDesc>("data", "Data");
  OpDescPtr const_op1 = std::make_shared<OpDesc>("const1", "Const");
  OpDescPtr const_op2 = std::make_shared<OpDesc>("const2", "Const");
  OpDescPtr const_op3 = std::make_shared<OpDesc>("const3", "Const");

  //add descriptor
  vector<int64_t> dim({4, 33, 12, 16});
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_ND);
  tensor_desc.SetOriginDataType(DT_INT8);
  tensor_desc.SetDataType(DT_INT8);
  tensor_desc.SetOriginShape(shape);
  (void)ge::AttrUtils::SetStr(mat_mul_op, "_sgt_cube_vector_core_type", "AiCore");
  mat_mul_op->AddInputDesc("x1", tensor_desc);
  mat_mul_op->AddInputDesc("x2", tensor_desc);
  mat_mul_op->AddInputDesc("bias", tensor_desc);
  GeTensorDesc tensor_desc1(shape);
  tensor_desc1.SetDataType(DT_INT32);
  tensor_desc1.SetOriginFormat(FORMAT_ND);
  tensor_desc1.SetOriginDataType(DT_INT32);
  tensor_desc1.SetDataType(DT_INT32);
  tensor_desc1.SetOriginShape(shape);
  mat_mul_op->AddInputDesc("offset", tensor_desc1);
  mat_mul_op->AddOutputDesc("x2", tensor_desc);
  ge::AttrUtils::SetInt(mat_mul_op, FE_IMPLY_TYPE, 6);

  data_op->AddOutputDesc(mat_mul_op->GetInputDesc(0));
  const_op1->AddOutputDesc(mat_mul_op->GetInputDesc(1));
  const_op2->AddOutputDesc(mat_mul_op->GetInputDesc(2));
  const_op3->AddOutputDesc(mat_mul_op->GetInputDesc(3));

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  ge::NodePtr mat_mul_node = graph->AddNode(mat_mul_op);
  ge::NodePtr data_node = graph->AddNode(data_op);
  ge::NodePtr const_node1 = graph->AddNode(const_op1);
  ge::NodePtr const_node2 = graph->AddNode(const_op2);
  ge::NodePtr const_node3 = graph->AddNode(const_op3);

  GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), mat_mul_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(const_node1->GetOutDataAnchor(0), mat_mul_node->GetInDataAnchor(1));
  GraphUtils::AddEdge(const_node2->GetOutDataAnchor(0), mat_mul_node->GetInDataAnchor(2));
  GraphUtils::AddEdge(const_node3->GetOutDataAnchor(0), mat_mul_node->GetInDataAnchor(3));

  op_cust_dtypes_parser_ptr_->op_type_cust_dtypes_.clear();
  op_cust_dtypes_parser_ptr_->op_name_cust_dtypes_.clear();

  Status ret = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(mat_mul_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(mat_mul_node, "tbe-custom");
  ASSERT_EQ(ret, fe::SUCCESS);

  EXPECT_EQ(mat_mul_op->GetInputDesc(0).GetDataType(), DT_INT8);
  EXPECT_EQ(mat_mul_op->GetInputDesc(1).GetDataType(), DT_INT8);
  EXPECT_EQ(mat_mul_op->GetInputDesc(2).GetDataType(), DT_INT8);
  EXPECT_EQ(mat_mul_op->GetInputDesc(3).GetDataType(), DT_INT32);
  EXPECT_EQ(mat_mul_op->GetOutputDesc(0).GetDataType(), DT_INT8);
}

TEST_F(UTEST_fusion_engine_op_judge_precision_mode, judge_node_allowfp32to16_noncube)
{
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = ALLOW_FP32_TO_FP16;
  PlatformUtils::Instance().soc_version_ = "Ascend910B2";
  Configuration::Instance(fe::AI_CORE_NAME).fp16_op_type_list_ = {"MatMulV3"};
  op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();

  SubOpInfoStorePtr subOpInfoStorePtr = OpsKernelManager::Instance(AI_CORE_NAME).GetSubOpsKernelByStoreName("tbe-custom");

  OpDescPtr add_op = std::make_shared<OpDesc>("mat_mul_name", "AddV2");
  OpDescPtr data_op = std::make_shared<OpDesc>("data", "Data");
  OpDescPtr const_op1 = std::make_shared<OpDesc>("const1", "Const");

  //add descriptor
  vector<int64_t> dim({4, 33, 12, 16});
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_ND);
  tensor_desc.SetOriginDataType(DT_FLOAT);
  tensor_desc.SetOriginShape(shape);
  add_op->AddInputDesc("x1", tensor_desc);
  add_op->AddInputDesc("x2", tensor_desc);
  GeTensorDesc tensor_desc1(shape);
  tensor_desc1.SetDataType(DT_INT32);
  tensor_desc1.SetOriginFormat(FORMAT_ND);
  tensor_desc1.SetOriginDataType(DT_INT32);
  tensor_desc1.SetOriginShape(shape);
  add_op->AddOutputDesc("x2", tensor_desc);
  ge::AttrUtils::SetInt(add_op, FE_IMPLY_TYPE, 6);

  data_op->AddOutputDesc(add_op->GetInputDesc(0));
  const_op1->AddOutputDesc(add_op->GetInputDesc(1));

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  ge::NodePtr add_node = graph->AddNode(add_op);
  ge::NodePtr data_node = graph->AddNode(data_op);
  ge::NodePtr const_node1 = graph->AddNode(const_op1);

  GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), add_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(const_node1->GetOutDataAnchor(0), add_node->GetInDataAnchor(1));

  op_cust_dtypes_parser_ptr_->op_type_cust_dtypes_.clear();
  op_cust_dtypes_parser_ptr_->op_name_cust_dtypes_.clear();

  Status ret = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(add_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(add_node, "tbe-custom");
  ASSERT_EQ(ret, fe::SUCCESS);

  EXPECT_EQ(add_op->GetInputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(add_op->GetInputDesc(1).GetDataType(), DT_FLOAT);
  EXPECT_EQ(add_op->GetOutputDesc(0).GetDataType(), DT_FLOAT);
}

TEST_F(UTEST_fusion_engine_op_judge_precision_mode, allow_fp16_only_support_fp16_and_ori_bf16)
{
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = ALLOW_FP32_TO_FP16;
  ConstructAndCheck(ALLOW_FP32_TO_FP16);
}

TEST_F(UTEST_fusion_engine_op_judge_precision_mode, allow_bf16_only_support_fp16_and_ori_bf16)
{
  PlatformInfoManager::Instance().opti_compilation_infos_.SetSocVersion("Ascend910B2");
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = ALLOW_FP32_TO_BF16;
  ConstructAndCheck(ALLOW_FP32_TO_BF16);
}

TEST_F(UTEST_fusion_engine_op_judge_precision_mode, allow_mix_bf16_only_support_fp16_and_ori_bf16)
{
  PlatformInfoManager::Instance().opti_compilation_infos_.SetSocVersion("Ascend910B2");
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = ALLOW_FP32_TO_BF16;
  ConstructAndCheck(ALLOW_FP32_TO_BF16);
}

TEST_F(UTEST_fusion_engine_op_judge_precision_mode, force_lower_only_support_fp16_and_ori_bf16)
{
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = FORCE_LOWERPRECISION;
  ConstructAndCheck(FORCE_LOWERPRECISION);
}


TEST_F(UTEST_fusion_engine_op_judge_precision_mode, allow_bf16_ori_fp16_support_bf16_fp16)
{
  PlatformInfoManager::Instance().opti_compilation_infos_.SetSocVersion("Ascend910B2");
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = ALLOW_FP32_TO_BF16;
  op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g_op = std::make_shared<OpDesc>("G1", "SupportBf16AndFp16");

  //add descriptor
  vector<int64_t> dim({4, 33, 12, 16});
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NCHW);
  tensor_desc.SetFormat(FORMAT_NCHW);
  tensor_desc.SetDataType(DT_FLOAT16);
  tensor_desc.SetOriginDataType(DT_FLOAT16);

  g_op->AddInputDesc("x", tensor_desc);
  g_op->AddOutputDesc("z", tensor_desc);
  ge::AttrUtils::SetInt(g_op, FE_IMPLY_TYPE, 6);
  ge::NodePtr g_node = graph->AddNode(g_op);
  string reason;
  bool check_result = fe_ops_kernel_info_store_ptr_->CheckSupported(g_node, reason);
  ASSERT_EQ(check_result, true);

  Status ret1 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret1 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g_node, "tbe-custom");
  ASSERT_EQ(ret1, fe::SUCCESS);
  EXPECT_EQ(g_op->GetInputDesc(0).GetFormat(), FORMAT_NCHW);
  EXPECT_EQ(g_op->GetInputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(g_op->GetInputDesc(0).GetShape().GetDims(), dim);

  EXPECT_EQ(g_op->GetOutputDesc(0).GetFormat(), FORMAT_NCHW);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetShape().GetDims(), dim);
}

TEST_F(UTEST_fusion_engine_op_judge_precision_mode, allow_bf16_ori_float_support_bf16_fp16)
{
  PlatformInfoManager::Instance().opti_compilation_infos_.SetSocVersion("Ascend910B2");
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = ALLOW_FP32_TO_BF16;
  op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g_op = std::make_shared<OpDesc>("G1", "SupportBf16AndFp16");

  //add descriptor
  vector<int64_t> dim({4, 33, 12, 16});
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NCHW);
  tensor_desc.SetFormat(FORMAT_NCHW);
  tensor_desc.SetDataType(DT_FLOAT);
  tensor_desc.SetOriginDataType(DT_FLOAT);

  g_op->AddInputDesc("x", tensor_desc);
  g_op->AddOutputDesc("z", tensor_desc);
  ge::AttrUtils::SetInt(g_op, FE_IMPLY_TYPE, 6);
  ge::NodePtr g_node = graph->AddNode(g_op);
  string reason;
  bool check_result = fe_ops_kernel_info_store_ptr_->CheckSupported(g_node, reason);
  ASSERT_EQ(check_result, true);

  Status ret1 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret1 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g_node, "tbe-custom");
  ASSERT_EQ(ret1, fe::SUCCESS);
  EXPECT_EQ(g_op->GetInputDesc(0).GetFormat(), FORMAT_NCHW);
  EXPECT_EQ(g_op->GetInputDesc(0).GetDataType(), DT_BF16);
  EXPECT_EQ(g_op->GetInputDesc(0).GetShape().GetDims(), dim);

  EXPECT_EQ(g_op->GetOutputDesc(0).GetFormat(), FORMAT_NCHW);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetDataType(), DT_BF16);
  EXPECT_EQ(g_op->GetOutputDesc(0).GetShape().GetDims(), dim);
}

TEST_F(UTEST_fusion_engine_op_judge_precision_mode, allow_mix_precision)
{
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = ALLOW_MIX_PRECISION;
  op_format_dtype_judge_ptr_ =
      std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME,
                                           reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();
  ConstructAndCheck2("Supportbf16AndFp32GrayList", ge::DT_BF16, ge::DT_BF16, ge::DT_BF16);
  ConstructAndCheck2("Supportbf16AndFp32BlackList", ge::DT_BF16, ge::DT_BF16, ge::DT_BF16);
}

TEST_F(UTEST_fusion_engine_op_judge_precision_mode, allow_mix_precision_fp16)
{
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = ALLOW_MIX_PRECISION_FP16;
  op_format_dtype_judge_ptr_ =
      std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME,
                                           reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();
  ConstructAndCheck2("Supportbf16AndFp32GrayList", ge::DT_BF16, ge::DT_BF16, ge::DT_BF16);
  ConstructAndCheck2("Supportbf16AndFp32BlackList", ge::DT_BF16, ge::DT_BF16, ge::DT_BF16);
}

TEST_F(UTEST_fusion_engine_op_judge_precision_mode, allow_mix_precision_bf16)
{
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = ALLOW_MIX_PRECISION_BF16;
  op_format_dtype_judge_ptr_ =
      std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME,
                                           reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();
  ConstructAndCheck2("Supportbf16AndFp32GrayList", ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT);
  ConstructAndCheck2("Supportbf16AndFp32BlackList", ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT);

  ConstructAndCheck2("Supportfp16AndFp32GrayList", ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16);
  ConstructAndCheck2("Supportfp16AndFp32BlackList", ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16);

  ConstructAndCheck2("SupportBf16AndFp16", ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16);
}

TEST_F(UTEST_fusion_engine_op_judge_precision_mode, cube_hif8_for_cube)
{
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = "";
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE_V2] = kCubeHif8;
  Configuration::Instance(fe::AI_CORE_NAME).fp16_op_type_list_ = {"CubeHif8SupportHif8", "CubeHif8SupportF16",
    "CubeHif8SupportF32"};
  op_format_dtype_judge_ptr_ =
      std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME,
                                           reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();
  // original_dtype = DT_HIFLOAT8
  ConstructAndCheck3("CubeHif8SupportHif8", ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, true, fe::SUCCESS);
  ConstructAndCheck3("CubeHif8SupportF16", ge::DT_HIFLOAT8, ge::DT_FLOAT16, ge::DT_FLOAT16, false, fe::SUCCESS);
  ConstructAndCheck3("CubeHif8SupportF32", ge::DT_HIFLOAT8, ge::DT_FLOAT, ge::DT_FLOAT16, false, fe::SUCCESS);
  // original_dtype = DT_BF16
  ConstructAndCheck3("CubeHif8SupportHif8", ge::DT_BF16, ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, true, fe::SUCCESS);
  ConstructAndCheck3("CubeHif8SupportF16", ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, true, fe::SUCCESS);
  ConstructAndCheck3("CubeHif8SupportF32", ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT, true, fe::SUCCESS);
  // original_dtype = DT_FLOAT16
  ConstructAndCheck3("CubeHif8SupportHif8", ge::DT_FLOAT16, ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, true, fe::SUCCESS);
  ConstructAndCheck3("CubeHif8SupportF16", ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, true, fe::SUCCESS);
  ConstructAndCheck3("CubeHif8SupportF32", ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT16, true, fe::SUCCESS);
  // original_dtype = DT_FLOAT
  ConstructAndCheck3("CubeHif8SupportHif8", ge::DT_FLOAT, ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, true, fe::SUCCESS);
  ConstructAndCheck3("CubeHif8SupportF16", ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT16, true, fe::SUCCESS);
  ConstructAndCheck3("CubeHif8SupportF32", ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16, true, fe::SUCCESS);
  // original_dtype = DT_INT32
  ConstructAndCheck3("CubeHif8SupportF32", ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, true, fe::SUCCESS);
  // original_dtype = DT_INT8
  ConstructAndCheck3("CubeHif8SupportF32", ge::DT_INT8, ge::DT_FLOAT, ge::DT_FLOAT, false, fe::SUCCESS);
}

TEST_F(UTEST_fusion_engine_op_judge_precision_mode, cube_hif8_for_non_cube)
{
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = "";
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE_V2] = kCubeHif8;
  op_format_dtype_judge_ptr_ =
      std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME,
                                           reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();
  // original_dtype = DT_BF16
  ConstructAndCheck3("CubeHif8SupportF16", ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, true, fe::SUCCESS);
  ConstructAndCheck3("CubeHif8SupportF32", ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT, true, fe::SUCCESS);
  // original_dtype = DT_FLOAT16
  ConstructAndCheck3("CubeHif8SupportF16", ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, true, fe::SUCCESS);
  ConstructAndCheck3("CubeHif8SupportF32", ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT16, true, fe::SUCCESS);
  // original_dtype = DT_FLOAT
  ConstructAndCheck3("CubeHif8SupportF32", ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16, true, fe::SUCCESS);
  // original_dtype = DT_INT32
  ConstructAndCheck3("CubeHif8SupportF32", ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, true, fe::SUCCESS);
  // original_dtype = DT_INT8
  ConstructAndCheck3("CubeHif8SupportF32", ge::DT_INT8, ge::DT_FLOAT, ge::DT_FLOAT, false, fe::SUCCESS);
}

TEST_F(UTEST_fusion_engine_op_judge_precision_mode, mixed_hif8_white_list)
{
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = "";
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE_V2] = kMixedHif8;
  op_format_dtype_judge_ptr_ =
      std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME,
                                           reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();
  // original_dtype = DT_HIFLOAT8
  ConstructAndCheck3("Hif8SupportHif8WhiteList", ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, true, fe::SUCCESS);
  ConstructAndCheck3("Hif8SupportF16WhiteList", ge::DT_HIFLOAT8, ge::DT_FLOAT16, ge::DT_FLOAT16, false, fe::SUCCESS);
  ConstructAndCheck3("Hif8SupportF32WhiteList", ge::DT_HIFLOAT8, ge::DT_FLOAT, ge::DT_FLOAT, false, fe::SUCCESS);
  // original_dtype = DT_BF16
  ConstructAndCheck3("Hif8SupportHif8WhiteList", ge::DT_BF16, ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, true, fe::SUCCESS);
  ConstructAndCheck3("Hif8SupportF16WhiteList", ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, true, fe::SUCCESS);
  ConstructAndCheck3("Hif8SupportF32WhiteList", ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT, true, fe::SUCCESS);
  // original_dtype = DT_FLOAT16
  ConstructAndCheck3("Hif8SupportHif8WhiteList", ge::DT_FLOAT16, ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, true, fe::SUCCESS);
  ConstructAndCheck3("Hif8SupportF16WhiteList", ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, true, fe::SUCCESS);
  ConstructAndCheck3("Hif8SupportF32WhiteList", ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT, true, fe::SUCCESS);
  // original_dtype = DT_FLOAT
  ConstructAndCheck3("Hif8SupportHif8WhiteList", ge::DT_FLOAT, ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, true, fe::SUCCESS);
  ConstructAndCheck3("Hif8SupportF16WhiteList", ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT16, true, fe::SUCCESS);
  ConstructAndCheck3("Hif8SupportF32WhiteList", ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, true, fe::SUCCESS);
  // original_dtype = DT_INT32
  ConstructAndCheck3("Hif8SupportF32WhiteList", ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, true, fe::SUCCESS);
  // original_dtype = DT_INT8
  ConstructAndCheck3("Hif8SupportF32WhiteList", ge::DT_INT8, ge::DT_FLOAT, ge::DT_FLOAT, false, fe::SUCCESS);
}

TEST_F(UTEST_fusion_engine_op_judge_precision_mode, mixed_hif8_black_list)
{
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = "";
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE_V2] = kMixedHif8;
  op_format_dtype_judge_ptr_ =
      std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME,
                                           reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();
  // original_dtype = DT_BF16
  ConstructAndCheck3("Hif8SupportF16BlackList", ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, true, fe::SUCCESS);
  ConstructAndCheck3("Hif8SupportF32BlackList", ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT, true, fe::SUCCESS);
  // original_dtype = DT_FLOAT16
  ConstructAndCheck3("Hif8SupportF16BlackList", ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, true, fe::SUCCESS);
  ConstructAndCheck3("Hif8SupportF32BlackList", ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT, true, fe::SUCCESS);
  // original_dtype = DT_FLOAT
  ConstructAndCheck3("Hif8SupportF32BlackList", ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, true, fe::SUCCESS);
  // original_dtype = DT_INT32
  ConstructAndCheck3("Hif8SupportF32BlackList", ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, true, fe::SUCCESS);
  // original_dtype = DT_INT8
  ConstructAndCheck3("Hif8SupportF32BlackList", ge::DT_INT8, ge::DT_FLOAT, ge::DT_FLOAT, false, fe::SUCCESS);
}

TEST_F(UTEST_fusion_engine_op_judge_precision_mode, mixed_hif8_gray_list_has_father)
{
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = "";
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE_V2] = kMixedHif8;
  op_format_dtype_judge_ptr_ =
      std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME,
                                           reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();
  // father_dtype = DT_HIFLOAT8
  ConstructAndCheckHasFatherNode("CubeHif8SupportHif8",ge::DT_HIFLOAT8, ge::DT_FLOAT16,
                                 ge::DT_HIFLOAT8, ge::DT_FLOAT16, true, fe::SUCCESS);
  ConstructAndCheckHasFatherNode("CubeHif8SupportF16",ge::DT_HIFLOAT8, ge::DT_BF16,
                                 ge::DT_BF16, ge::DT_BF16, true, fe::SUCCESS);
  ConstructAndCheckHasFatherNode("CubeHif8SupportF32",ge::DT_HIFLOAT8, ge::DT_FLOAT,
                                 ge::DT_FLOAT, ge::DT_FLOAT, true, fe::SUCCESS);
  // father_dtype = DT_BF16
  ConstructAndCheckHasFatherNode("CubeHif8SupportHif8",ge::DT_BF16, ge::DT_HIFLOAT8,
                                 ge::DT_BF16, ge::DT_HIFLOAT8, true, fe::SUCCESS);
  ConstructAndCheckHasFatherNode("CubeHif8SupportF16",ge::DT_BF16, ge::DT_FLOAT,
                                 ge::DT_BF16, ge::DT_FLOAT, true, fe::SUCCESS);
  ConstructAndCheckHasFatherNode("CubeHif8SupportF32",ge::DT_BF16, ge::DT_FLOAT16,
                                 ge::DT_FLOAT, ge::DT_FLOAT16, true, fe::SUCCESS);
  // father_dtype = DT_FLOAT16
  ConstructAndCheckHasFatherNode("CubeHif8SupportHif8",ge::DT_FLOAT16, ge::DT_HIFLOAT8,
                                 ge::DT_FLOAT16, ge::DT_HIFLOAT8, true, fe::SUCCESS);
  ConstructAndCheckHasFatherNode("CubeHif8SupportF16",ge::DT_FLOAT16, ge::DT_FLOAT,
                                 ge::DT_FLOAT16, ge::DT_FLOAT, true, fe::SUCCESS);
  ConstructAndCheckHasFatherNode("CubeHif8SupportF32",ge::DT_FLOAT16, ge::DT_FLOAT16,
                                 ge::DT_FLOAT, ge::DT_FLOAT16, true, fe::SUCCESS);
  // father_dtype = DT_FLOAT
  ConstructAndCheckHasFatherNode("CubeHif8SupportHif8",ge::DT_FLOAT, ge::DT_HIFLOAT8,
                                 ge::DT_FLOAT, ge::DT_HIFLOAT8, true, fe::SUCCESS);
  ConstructAndCheckHasFatherNode("CubeHif8SupportF16",ge::DT_FLOAT, ge::DT_FLOAT,
                                 ge::DT_FLOAT, ge::DT_FLOAT, true, fe::SUCCESS);
  ConstructAndCheckHasFatherNode("CubeHif8SupportF32",ge::DT_FLOAT, ge::DT_FLOAT16,
                                 ge::DT_FLOAT, ge::DT_FLOAT16, true, fe::SUCCESS);
  // father_dtype = DT_INT32
  ConstructAndCheckHasFatherNode("CubeHif8SupportF32",ge::DT_INT32, ge::DT_FLOAT,
                                 ge::DT_INT32, ge::DT_FLOAT, true, fe::SUCCESS);
  // father_dtype = DT_INT8
  ConstructAndCheckHasFatherNode("CubeHif8SupportF32",ge::DT_INT8, ge::DT_INT8,
                                 ge::DT_FLOAT, ge::DT_FLOAT, false, fe::SUCCESS);
}

TEST_F(UTEST_fusion_engine_op_judge_precision_mode, mixed_hif8_gray_list_no_father)
{
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = "";
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE_V2] = kMixedHif8;
  op_format_dtype_judge_ptr_ =
      std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME,
                                           reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();
  // original_dtype = DT_BF16
  ConstructAndCheck3("CubeHif8SupportF16", ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, true, fe::SUCCESS);
  ConstructAndCheck3("CubeHif8SupportF32", ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT, true, fe::SUCCESS);
  // original_dtype = DT_FLOAT16
  ConstructAndCheck3("CubeHif8SupportF16", ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, true, fe::SUCCESS);
  ConstructAndCheck3("CubeHif8SupportF32", ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT16, true, fe::SUCCESS);
  // original_dtype = DT_FLOAT
  ConstructAndCheck3("CubeHif8SupportF32", ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, true, fe::SUCCESS);
  // original_dtype = DT_INT32
  ConstructAndCheck3("CubeHif8SupportF32", ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, true, fe::SUCCESS);
  // original_dtype = DT_INT8
  ConstructAndCheck3("CubeHif8SupportF32", ge::DT_INT8, ge::DT_FLOAT, ge::DT_FLOAT, false, fe::SUCCESS);
}

TEST_F(UTEST_fusion_engine_op_judge_precision_mode, cube_hif8_with_keep_dtype)
{
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = "";
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE_V2] = kCubeHif8;
  op_format_dtype_judge_ptr_ =
      std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME,
                                           reflection_builder_ptr_);
  op_format_dtype_judge_ptr_->Initialize();

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr g1_op = std::make_shared<OpDesc>("G1", "CubeHif8SupportHif8");
  OpDescPtr g2_op = std::make_shared<OpDesc>("G2", "CubeHif8SupportHif8");

  //add descriptor
  vector<int64_t> dim({4, 33, 12, 16});
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NCHW);
  tensor_desc.SetFormat(FORMAT_NCHW);
  tensor_desc.SetDataType(DT_FLOAT);
  tensor_desc.SetOriginDataType(DT_FLOAT);
  g1_op->AddInputDesc("x", tensor_desc);
  g1_op->AddOutputDesc("z", tensor_desc);
  ge::AttrUtils::SetInt(g1_op, FE_IMPLY_TYPE, 6);
  ge::AttrUtils::SetInt(g1_op, KEEP_DTYPE, 1);
  ge::NodePtr g_node = graph->AddNode(g1_op);


  vector<int64_t> dim_h({1, 2, 3, 4});
  GeShape shape_h(dim_h);
  GeTensorDesc tensor_desc_h(shape_h);
  tensor_desc_h.SetOriginFormat(FORMAT_NCHW);
  tensor_desc_h.SetFormat(FORMAT_NCHW);
  tensor_desc_h.SetDataType(DT_FLOAT);
  tensor_desc_h.SetOriginDataType(DT_FLOAT);
  g2_op->AddInputDesc("x", tensor_desc_h);
  g2_op->AddOutputDesc("z", tensor_desc_h);
  ge::AttrUtils::SetInt(g2_op, FE_IMPLY_TYPE, 6);
  ge::NodePtr h_node = graph->AddNode(g2_op);
  GraphUtils::AddEdge(g_node->GetOutDataAnchor(0), h_node->GetInDataAnchor(0));


  Status ret1 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(g_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret1 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(g_node, "tbe-custom");
  Status ret2 = op_format_dtype_judge_ptr_->SetDtypeByPrecisionMode(h_node, "tbe-custom", OpImplType::EN_IMPL_HW_TBE);
  ret2 = op_format_dtype_judge_ptr_->SetFormatByJudgeResult(h_node, "tbe-custom");

  EXPECT_EQ(ret1, fe::SUCCESS);
  EXPECT_EQ(ret2, fe::SUCCESS);

  EXPECT_EQ(g1_op->GetInputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(g1_op->GetOutputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(g2_op->GetInputDesc(0).GetDataType(), DT_HIFLOAT8);
  EXPECT_EQ(g2_op->GetOutputDesc(0).GetDataType(), DT_HIFLOAT8);
}
