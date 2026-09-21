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
#include "fe_llt_utils.h"
#define private public
#define protected public
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/attr_utils.h"
#include "adapter/common/op_store_adapter_manager.h"
#include "adapter/tbe_adapter/tbe_op_store_adapter.h"
#include "graph_optimizer/op_setter/op_setter.h"
#include "graph/debug/ge_attr_define.h"
#include "common/configuration.h"
#include "ops_store/ops_kernel_manager.h"
#include "ops_kernel_store/fe_ops_kernel_info_store.h"
#include "common/config_parser/op_impl_mode_config_parser.h"

using namespace std;
using namespace ge;
using namespace fe;
using OpSetterPtr = std::shared_ptr<OpSetter>;


class STEST_OP_SETTER : public testing::Test
{
protected:
  shared_ptr<fe::FEOpsKernelInfoStore> fe_ops_kernel_info_store_ptr_;
  void SetUp()
  {
    PlatformInfoManager::Instance().opti_compilation_infos_.SetSocVersion("Ascend310");
    PlatformInfoManager::Instance().opti_compilation_infos_.SetAICoreNum(8);
    std::map<std::string, std::string> options;
    fe_ops_kernel_info_store_ptr_ = make_shared<fe::FEOpsKernelInfoStore>();
    FEOpsStoreInfo tbe_custom {
            6,
            "tbe-custom",
            EN_IMPL_HW_TBE,
            GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/tbe_slice_op_info/slice_success",
            ""};
    vector<FEOpsStoreInfo> store_info;
    store_info.emplace_back(tbe_custom);
    Configuration::Instance(fe::AI_CORE_NAME).ops_store_info_vector_ = (store_info);
    OpsKernelManager::Instance(AI_CORE_NAME).Finalize();

    fe_ops_kernel_info_store_ptr_->Initialize(options);
  }

  void TearDown() {}
  static void CreateOneOpGraph(ComputeGraphPtr graph) {
    OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "Activation");
    ge::AttrUtils::SetInt(relu_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
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

  static void CreateOneInvalidOpGraph(ComputeGraphPtr graph) {
    OpDescPtr invalid_op = std::make_shared<OpDesc>("aaabbb", "cccc");
    ge::AttrUtils::SetInt(invalid_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
    vector<int64_t> op_dim(4, 1);
    GeShape op_shape(op_dim);
    GeTensorDesc odesc(op_shape);
    odesc.SetOriginFormat(FORMAT_NCHW);
    odesc.SetFormat(FORMAT_NCHW);
    odesc.SetDataType(DT_FLOAT16);
    invalid_op->AddInputDesc("x", odesc);
    invalid_op->AddOutputDesc("y", odesc);

    NodePtr invalid_node = graph->AddNode(invalid_op);
  }

  ComputeGraphPtr CreateMultiThreadGraph() {
    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    CreateOneOpGraph(graph);
    vector<int64_t> dim(4, 1);
    GeShape shape(dim);
    GeTensorDesc out_desc(shape, FORMAT_NCHW, DT_FLOAT);
    out_desc.SetOriginFormat(FORMAT_NCHW);
    out_desc.SetOriginDataType(DT_FLOAT);
    out_desc.SetOriginShape(shape);
    OpDescPtr relu_op2 = std::make_shared<OpDesc>("relu2", "Relu");
    OpDescPtr relu_op3 = std::make_shared<OpDesc>("relu3", "Relu");
    OpDescPtr relu_op4 = std::make_shared<OpDesc>("relu4", "Relu");
    OpDescPtr relu_op5 = std::make_shared<OpDesc>("relu5", "Relu");
    OpDescPtr relu_op6 = std::make_shared<OpDesc>("relu6", "Relu");
    OpDescPtr relu_op7 = std::make_shared<OpDesc>("relu7", "Relu");
    OpDescPtr relu_op8 = std::make_shared<OpDesc>("relu8", "Relu");

    relu_op2->AddInputDesc("x", out_desc);
    relu_op2->AddOutputDesc("y", out_desc);
    relu_op3->AddInputDesc("x", out_desc);
    relu_op3->AddOutputDesc("y", out_desc);
    relu_op4->AddInputDesc("x", out_desc);
    relu_op4->AddOutputDesc("y", out_desc);
    relu_op5->AddInputDesc("x", out_desc);
    relu_op5->AddOutputDesc("y", out_desc);
    relu_op6->AddInputDesc("x", out_desc);
    relu_op6->AddOutputDesc("y", out_desc);
    relu_op7->AddInputDesc("x", out_desc);
    relu_op7->AddOutputDesc("y", out_desc);
    relu_op8->AddInputDesc("x", out_desc);
    relu_op8->AddOutputDesc("y", out_desc);

    NodePtr relu_node2 = graph->AddNode(relu_op2);
    NodePtr relu_node3 = graph->AddNode(relu_op3);
    NodePtr relu_node4 = graph->AddNode(relu_op4);
    NodePtr relu_node5 = graph->AddNode(relu_op5);
    NodePtr relu_node6 = graph->AddNode(relu_op6);
    NodePtr relu_node7 = graph->AddNode(relu_op7);
    NodePtr relu_node8 = graph->AddNode(relu_op8);
    return graph;
  }

};

TEST_F(STEST_OP_SETTER, set_op_info_success) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateOneOpGraph(graph);
  OpSetterPtr op_setter_ptr = std::make_shared<OpSetter>(AI_CORE_NAME);
  Status ret2 = op_setter_ptr->SetOpInfo(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, ret2);
  string op_slice_info_str;
  for (auto node : graph->GetDirectNode()) {
    string op_type = node->GetType();
    ge::OpDescPtr op_desc_ptr = node->GetOpDesc();

	op_slice_info_str.clear();
	(void)ge::AttrUtils::GetStr(op_desc_ptr, OP_SLICE_INFO, op_slice_info_str);
	//EXPECT_EQ(op_slice_info_str.empty(), false);

    int imply_type = -1;
    if(!ge::AttrUtils::GetInt(op_desc_ptr, FE_IMPLY_TYPE, imply_type)) {
      continue;
    }
    bool is_continous_input;
    bool is_continous_output;
    ge::AttrUtils::GetBool(op_desc_ptr, ge::ATTR_NAME_CONTINUOUS_INPUT, is_continous_input);
    ge::AttrUtils::GetBool(op_desc_ptr, ge::ATTR_NAME_CONTINUOUS_OUTPUT,is_continous_output);
    if (op_type == "Activation"){
      EXPECT_EQ(is_continous_input, true);
      EXPECT_EQ(is_continous_output, true);
    } else {
      ge::AttrUtils::GetBool(op_desc_ptr, ge::ATTR_NAME_CONTINUOUS_INPUT, is_continous_input);
      ge::AttrUtils::GetBool(op_desc_ptr, ge::ATTR_NAME_CONTINUOUS_OUTPUT,is_continous_output);
      EXPECT_EQ(is_continous_input, false);
      EXPECT_EQ(is_continous_output, false);
    }
  }
}

TEST_F(STEST_OP_SETTER, set_op_impl_failed) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("op_setter_test");
  CreateOneInvalidOpGraph(graph);
  OpSetterPtr op_setter_ptr = std::make_shared<OpSetter>(AI_CORE_NAME);
  op_setter_ptr->SetOpImplMode(*graph.get());
  for (auto node : graph->GetDirectNode()) {
    int64_t op_impl_mode = 0;
    EXPECT_EQ(ge::AttrUtils::GetInt(node->GetOpDesc(), OP_IMPL_MODE_ENUM, op_impl_mode), false);
    EXPECT_EQ(op_impl_mode, 0);
  }
}

TEST_F(STEST_OP_SETTER, set_op_impl_succ) {
  Configuration::Instance(AI_CORE_NAME).impl_mode_parser_ = std::make_shared<OpImplModeConfigParser>("");
  EXPECT_NE(Configuration::Instance(AI_CORE_NAME).impl_mode_parser_, nullptr);
  auto impl_mode_parser = std::dynamic_pointer_cast<OpImplModeConfigParser>(
      Configuration::Instance(AI_CORE_NAME).impl_mode_parser_);
  EXPECT_NE(impl_mode_parser, nullptr);
  impl_mode_parser->op_precision_mode_ = "xxx";
  impl_mode_parser->op_name_select_impl_mode_map_["relu"] = "high_performance";

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("op_setter_test");
  CreateOneOpGraph(graph);
  OpSetterPtr op_setter_ptr = std::make_shared<OpSetter>(AI_CORE_NAME);
  op_setter_ptr->SetOpImplMode(*graph.get());
  for (auto node : graph->GetDirectNode()) {
    int64_t op_impl_mode = 0;
    EXPECT_EQ(ge::AttrUtils::GetInt(node->GetOpDesc(), OP_CUSTOM_IMPL_MODE_ENUM, op_impl_mode), true);
    EXPECT_EQ(op_impl_mode, 0x2);
  }
}

TEST_F(STEST_OP_SETTER, multi_set_op_info_success) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateOneOpGraph(graph);
  OpSetterPtr op_setter_ptr = std::make_shared<OpSetter>(AI_CORE_NAME);
  Status ret2 = op_setter_ptr->MultiThreadSetOpInfo(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, ret2);
  string op_slice_info_str;
  for (auto node : graph->GetDirectNode()) {
    string op_type = node->GetType();
    ge::OpDescPtr op_desc_ptr = node->GetOpDesc();

	op_slice_info_str.clear();
	(void)ge::AttrUtils::GetStr(op_desc_ptr, OP_SLICE_INFO, op_slice_info_str);

    int imply_type = -1;
    if(!ge::AttrUtils::GetInt(op_desc_ptr, FE_IMPLY_TYPE, imply_type)) {
      continue;
    }
    bool is_continous_input;
    bool is_continous_output;
    ge::AttrUtils::GetBool(op_desc_ptr, ge::ATTR_NAME_CONTINUOUS_INPUT, is_continous_input);
    ge::AttrUtils::GetBool(op_desc_ptr, ge::ATTR_NAME_CONTINUOUS_OUTPUT,is_continous_output);
    if (op_type == "Activation"){
      EXPECT_EQ(is_continous_input, true);
      EXPECT_EQ(is_continous_output, true);
    } else {
      ge::AttrUtils::GetBool(op_desc_ptr, ge::ATTR_NAME_CONTINUOUS_INPUT, is_continous_input);
      ge::AttrUtils::GetBool(op_desc_ptr, ge::ATTR_NAME_CONTINUOUS_OUTPUT,is_continous_output);
      EXPECT_EQ(is_continous_input, false);
      EXPECT_EQ(is_continous_output, false);
    }
  }
  ComputeGraphPtr graph2 = CreateMultiThreadGraph();;
  Status ret3 = op_setter_ptr->MultiThreadSetOpInfo(*(graph2.get()));
  EXPECT_EQ(fe::SUCCESS, ret3);
}
