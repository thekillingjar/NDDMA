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
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/op_desc_utils_ex.h"
#include "graph/utils/graph_utils.h"
#include "graph_constructor/pass_manager.h"
#include "common/aicore_util_constants.h"
#include "ops_store/ops_kernel_manager.h"
#define protected public
#define private public
#include "common/platform_utils.h"
#include "common/configuration.h"
#include "ops_kernel_store/fe_ops_kernel_info_store.h"
#include "format_selector/manager/format_dtype_querier.h"
#include "register/graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/builtin_pass/refresh_cube_c0_fusion_pass.h"
#undef protected
#undef private

using namespace ge;
using namespace fe;


class RefreshCubeC0FusionPassUnitTest : public testing::Test {
 protected:
  OpsKernelInfoStorePtr ops_kernel_info_store_ptr_;
  void SetUp() {
    FEOpsStoreInfo tbe_custom {
            2,
            "tbe-custom",
            EN_IMPL_CUSTOM_TBE,
            GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/conv_format",
            GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/conv_format",
            false,
            false,
            false};

    vector<FEOpsStoreInfo> store_info;
    store_info.emplace_back(tbe_custom);
    Configuration::Instance(AI_CORE_NAME).ops_store_info_vector_ = (store_info);
    OpsKernelManager::Instance(AI_CORE_NAME).Finalize();
    OpsKernelManager::Instance(AI_CORE_NAME).Initialize();
    SubOpsStorePtr sub_ops_store_ptr = std::make_shared<SubOpsStore>(AI_CORE_NAME);
    sub_ops_store_ptr->format_dtype_querier_ptr_ = std::make_shared<FormatDtypeQuerier>(AI_CORE_NAME);

    FEOpsKernelInfoStorePtr fe_ops_kernel_store_ptr = std::make_shared<FEOpsKernelInfoStore>();
    fe_ops_kernel_store_ptr->map_all_sub_store_info_.emplace("tbe-custom", sub_ops_store_ptr);
    ops_kernel_info_store_ptr_ = fe_ops_kernel_store_ptr;
    REGISTER_PASS("RefreshCubeC0FusionPass", BUILT_IN_BEFORE_TRANSNODE_INSERTION_GRAPH_PASS,
                  RefreshCubeC0FusionPass);
  }
  void TearDown() {
  }

 protected:
  static ComputeGraphPtr CreateConvGraph()
  {
    OpDescPtr op_desc_conv = std::make_shared<OpDesc>("conv", "Conv2D");

    //add descriptor
    vector<int64_t> dim_nchw = {1, 32, 8, 8};
    GeShape shape_nchw(dim_nchw);
    vector<int64_t> dim_5hd = {1, 2, 8, 8, 16};
    GeShape shape_5hd(dim_5hd);
    vector<int64_t> dim_hwcn = {8, 8, 16, 3};
    GeShape shape_hwcn(dim_hwcn);
    vector<int64_t> dim_fz = {64, 1, 16, 16 };
    GeShape shape_fz(dim_fz);
    vector<int64_t> dim_nd = {8, 8, 16, 16 };
    GeShape shape_nd(dim_nd);

    GeTensorDesc tensor_desc_a(shape_5hd, FORMAT_NC1HWC0, DT_FLOAT);
    tensor_desc_a.SetOriginFormat(FORMAT_NCHW);
    tensor_desc_a.SetOriginDataType(DT_FLOAT);
    tensor_desc_a.SetOriginShape(shape_nchw);

    GeTensorDesc tensor_desc_b(shape_fz, FORMAT_FRACTAL_Z, DT_FLOAT);
    tensor_desc_b.SetOriginFormat(FORMAT_HWCN);
    tensor_desc_b.SetOriginDataType(DT_FLOAT);
    tensor_desc_b.SetOriginShape(shape_hwcn);

    GeTensorDesc tensor_desc_c(shape_nd, FORMAT_ND, DT_FLOAT);
    tensor_desc_c.SetOriginFormat(FORMAT_ND);
    tensor_desc_c.SetOriginDataType(DT_FLOAT);
    tensor_desc_c.SetOriginShape(shape_nd);

    op_desc_conv->AddInputDesc("x", tensor_desc_a);
    op_desc_conv->AddInputDesc("filter", tensor_desc_b);
    op_desc_conv->AddInputDesc("bias", tensor_desc_c);
    op_desc_conv->AddOutputDesc("y", tensor_desc_a);

    ge::AttrUtils::SetInt(op_desc_conv, FE_IMPLY_TYPE, 2);

    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    NodePtr node_conv = graph->AddNode(op_desc_conv);
    return graph;
  }
  static ComputeGraphPtr CreateConvGraph_1()
  {
    OpDescPtr op_desc_conv = std::make_shared<OpDesc>("conv", "Conv2D");

    //add descriptor
    vector<int64_t> dim_nchw = {1, 32, 8, 8};
    GeShape shape_nchw(dim_nchw);
    vector<int64_t> dim_5hd = {1, 2, 8, 8, 16};
    GeShape shape_5hd(dim_5hd);
    vector<int64_t> dim_hwcn = {8, 8, 16, 3};
    GeShape shape_hwcn(dim_hwcn);
    vector<int64_t> dim_fz = {64, 1, 16, 16 };
    GeShape shape_fz(dim_fz);
    vector<int64_t> dim_nd = {8, 8, 16, 16 };
    GeShape shape_nd(dim_nd);

    GeTensorDesc tensor_desc_a(shape_5hd, FORMAT_NC1HWC0, DT_FLOAT);
    tensor_desc_a.SetOriginFormat(FORMAT_NCHW);
    tensor_desc_a.SetOriginDataType(DT_FLOAT);
    tensor_desc_a.SetOriginShape(shape_nchw);

    GeTensorDesc tensor_desc_b(shape_fz, FORMAT_FRACTAL_Z, DT_FLOAT);
    tensor_desc_b.SetOriginFormat(FORMAT_HWCN);
    tensor_desc_b.SetOriginDataType(DT_FLOAT);
    tensor_desc_b.SetOriginShape(shape_hwcn);

    GeTensorDesc tensor_desc_c(shape_nd, FORMAT_ND, DT_FLOAT);
    tensor_desc_c.SetOriginFormat(FORMAT_ND);
    tensor_desc_c.SetOriginDataType(DT_FLOAT);
    tensor_desc_c.SetOriginShape(shape_nd);

    op_desc_conv->AddInputDesc("xx", tensor_desc_a);
    op_desc_conv->AddInputDesc("filter1", tensor_desc_b);
    op_desc_conv->AddInputDesc("bias", tensor_desc_c);
    op_desc_conv->AddOutputDesc("y", tensor_desc_a);

    ge::AttrUtils::SetInt(op_desc_conv, FE_IMPLY_TYPE, 2);

    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    NodePtr node_conv = graph->AddNode(op_desc_conv);
    return graph;
  }
};

TEST_F(RefreshCubeC0FusionPassUnitTest, refresh_cube_c0_01) {
  ComputeGraphPtr graph = CreateConvGraph();
  std::map<string, FusionPassRegistry::CreateFn> create_fns =
          FusionPassRegistry::GetInstance().GetCreateFnByType(BUILT_IN_BEFORE_TRANSNODE_INSERTION_GRAPH_PASS);
  auto iter = create_fns.find("RefreshCubeC0FusionPass");
  Status status = fe::FAILED;
  PlatformUtils::Instance().pm_item_vec_[static_cast<size_t>(PlatformUtils::PlatformInfoItem::CubeHighPrecison)] = 1;
  if (iter != create_fns.end()) {
    auto graph_fusion_pass_base_ptr = std::unique_ptr<PatternFusionBasePass>(
            dynamic_cast<PatternFusionBasePass *>(iter->second()));
    if (graph_fusion_pass_base_ptr != nullptr) {
      graph_fusion_pass_base_ptr->SetName(iter->first);
      status = graph_fusion_pass_base_ptr->Run(*graph, ops_kernel_info_store_ptr_);
    }
  }

  EXPECT_EQ(fe::SUCCESS, status);
  vector<int64_t> dim_exp = {1, 4, 8, 8, 8};
  for (NodePtr node : graph->GetDirectNode()) {
    OpDescPtr op_desc = node->GetOpDesc();
    if (op_desc->GetType() == "Conv2D") {
      EXPECT_EQ(static_cast<ge::Format>(ge::GetPrimaryFormat(op_desc->GetInputDescPtr(0)->GetFormat())), FORMAT_NC1HWC0);
      EXPECT_EQ(ge::GetC0Value(op_desc->GetInputDescPtr(0)->GetFormat()), 8);
      EXPECT_EQ(op_desc->GetInputDescPtr(0)->GetShape().GetDims(), dim_exp);
    }
  }
}

TEST_F(RefreshCubeC0FusionPassUnitTest, refresh_cube_c0_02) {
  ComputeGraphPtr graph = CreateConvGraph_1();
  std::map<string, FusionPassRegistry::CreateFn> create_fns =
          FusionPassRegistry::GetInstance().GetCreateFnByType(BUILT_IN_BEFORE_TRANSNODE_INSERTION_GRAPH_PASS);
  auto iter = create_fns.find("RefreshCubeC0FusionPass");
  Status status = fe::FAILED;
  PlatformUtils::Instance().pm_item_vec_[static_cast<size_t>(PlatformUtils::PlatformInfoItem::CubeHighPrecison)] = 1;
  if (iter != create_fns.end()) {
    auto graph_fusion_pass_base_ptr = std::unique_ptr<PatternFusionBasePass>(
            dynamic_cast<PatternFusionBasePass *>(iter->second()));
    if (graph_fusion_pass_base_ptr != nullptr) {
      graph_fusion_pass_base_ptr->SetName(iter->first);
      status = graph_fusion_pass_base_ptr->Run(*graph, ops_kernel_info_store_ptr_);
    }
  }
  EXPECT_EQ(fe::NOT_CHANGED, status);
}

