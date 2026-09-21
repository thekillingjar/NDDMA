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
#include <iostream>
#include <list>
#include "fe_llt_utils.h"
#define protected public
#define private public
#include "ops_store/ops_kernel_manager.h"
#include "common/configuration.h"
#include "common/aicore_util_constants.h"
#include "graph/utils/op_desc_utils_ex.h"
#undef private
#undef protected

using namespace std;
using namespace fe;
using namespace ge;

class ops_kernel_manager_unit_test : public testing::Test
{
protected:
    void SetUp()
    {
      FEOpsStoreInfo tbe_custom {
              2,
              "tbe-custom",
              EN_IMPL_CUSTOM_TBE,
              GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/tbe_custom_opinfo",
              GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/tbe_custom_opinfo",
              false,
              false,
              false};

      vector<FEOpsStoreInfo> store_info;
      store_info.emplace_back(tbe_custom);
      Configuration::Instance(AI_CORE_NAME).ops_store_info_vector_ = (store_info);
    }

    void TearDown()
    {

    }

// AUTO GEN PLEASE DO NOT MODIFY IT
};

TEST_F(ops_kernel_manager_unit_test, initialize_success)
{
  OpsKernelManager opsKernelManager(AI_CORE_NAME);
  Status ret = opsKernelManager.Initialize();
  EXPECT_EQ(ret, fe::SUCCESS);

  EXPECT_EQ(opsKernelManager.sub_ops_store_map_.size(), 1);
  EXPECT_EQ(opsKernelManager.sub_ops_kernel_map_.size(), 1);

  EXPECT_NE(opsKernelManager.GetSubOpsKernelByStoreName("tbe-custom"), nullptr);
  EXPECT_EQ(opsKernelManager.GetSubOpsKernelByStoreName("cce-custom"), nullptr);

  EXPECT_NE(opsKernelManager.GetSubOpsKernelByImplType(EN_IMPL_CUSTOM_TBE), nullptr);
  EXPECT_EQ(opsKernelManager.GetSubOpsKernelByImplType(EN_IMPL_HW_TBE), nullptr);

  EXPECT_NE(opsKernelManager.GetOpKernelInfoByOpType("tbe-custom", "conv2"), nullptr);
  EXPECT_EQ(opsKernelManager.GetOpKernelInfoByOpType("tbe-custom", "conv123"), nullptr);
  EXPECT_EQ(opsKernelManager.GetOpKernelInfoByOpType("cce-custom", "conv2"), nullptr);
  EXPECT_EQ(opsKernelManager.GetOpKernelInfoByOpType("cce-custom", "conv123"), nullptr);

  EXPECT_NE(opsKernelManager.GetOpKernelInfoByOpType(EN_IMPL_CUSTOM_TBE, "conv2"), nullptr);
  EXPECT_EQ(opsKernelManager.GetOpKernelInfoByOpType(EN_IMPL_CUSTOM_TBE, "conv123"), nullptr);
  EXPECT_EQ(opsKernelManager.GetOpKernelInfoByOpType(EN_IMPL_HW_TBE, "conv2"), nullptr);
  EXPECT_EQ(opsKernelManager.GetOpKernelInfoByOpType(EN_IMPL_HW_TBE, "conv123"), nullptr);

  vector<int64_t> dim(4, 4);
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetFormat(FORMAT_NCHW);
  tensor_desc.SetOriginFormat(FORMAT_NCHW);
  tensor_desc.SetDataType(DT_FLOAT);
  tensor_desc.SetOriginDataType(DT_FLOAT);
  ge::OpDescPtr op_desc_ptr = std::make_shared<ge::OpDesc>("test_op_desc", "conv2");
  op_desc_ptr->AddInputDesc(tensor_desc);
  op_desc_ptr->AddOutputDesc(tensor_desc);

  ge::AttrUtils::SetInt(op_desc_ptr, "_fe_imply_type", 2);
  OpKernelInfoPtr opKernelInfoPtr = opsKernelManager.GetOpKernelInfoByOpDesc(op_desc_ptr);
  bool flag = opKernelInfoPtr == nullptr;
  EXPECT_EQ(flag, false);

  ge::OpDescUtilsEx::SetType(op_desc_ptr, "conv345");
  opKernelInfoPtr = opsKernelManager.GetOpKernelInfoByOpDesc(op_desc_ptr);
  flag = opKernelInfoPtr == nullptr;
  EXPECT_EQ(flag, true);

  ge::AttrUtils::SetInt(op_desc_ptr, "_fe_imply_type", 4);
  opKernelInfoPtr = opsKernelManager.GetOpKernelInfoByOpDesc(op_desc_ptr);
  flag = opKernelInfoPtr == nullptr;
  EXPECT_EQ(flag, true);

  ge::OpDescUtilsEx::SetType(op_desc_ptr, "conv2");
  opKernelInfoPtr = opsKernelManager.GetOpKernelInfoByOpDesc(op_desc_ptr);
  flag = opKernelInfoPtr == nullptr;
  EXPECT_EQ(flag, true);

  ge::AttrUtils::SetInt(op_desc_ptr, "_fe_imply_type", 2);
  opKernelInfoPtr = opsKernelManager.GetOpKernelInfoByOpDesc(op_desc_ptr);
  flag = opKernelInfoPtr == nullptr;
  ASSERT_EQ(flag, false);
  EXPECT_EQ(opKernelInfoPtr->GetDynamicCompileStatic(), DynamicCompileStatic::NOT_SUPPORT);
  EXPECT_EQ(opKernelInfoPtr->IsDynamicCompileStatic(), false);
  EXPECT_EQ(opKernelInfoPtr->GetJitCompileType(), JitCompile::DEFAULT);
  EXPECT_EQ(opKernelInfoPtr->GetEnableVectorCore(), VectorCoreType::DEFAULT);
  EXPECT_EQ(opKernelInfoPtr->GetMultiKernelSupportType(), MultiKernelSupportType::DEFAULT);
  EXPECT_EQ(opKernelInfoPtr->IsMultiKernelSupport(), false);
  EXPECT_EQ(opKernelInfoPtr->GetPrecisionPolicy(), PrecisionPolicy::GRAY);

  opKernelInfoPtr = opsKernelManager.GetOpKernelInfoByOpType("tbe-custom", "ScatterNdUpdate");
  flag = opKernelInfoPtr == nullptr;
  ASSERT_EQ(flag, false);
  std::vector<std::string> expect_refer_vec = {"var"};
  EXPECT_EQ(opKernelInfoPtr->GetReferTensorNameVec(), expect_refer_vec);
  EXPECT_EQ(opKernelInfoPtr->GetOpImplSwitch(), "dsl, tik");
  std::vector<std::string> expect_opimpl_vec = {"dsl", "tik"};
  EXPECT_EQ(opKernelInfoPtr->GetOpImplSwitchVec(), expect_opimpl_vec);
  EXPECT_EQ(opKernelInfoPtr->GetDynamicCompileStatic(), DynamicCompileStatic::TUNE);
  EXPECT_EQ(opKernelInfoPtr->IsDynamicCompileStatic(), true);
  EXPECT_EQ(opKernelInfoPtr->GetJitCompileType(), JitCompile::ONLINE);
  EXPECT_EQ(opKernelInfoPtr->GetEnableVectorCore(), VectorCoreType::DISABLE);
  EXPECT_EQ(opKernelInfoPtr->GetMultiKernelSupportType(), MultiKernelSupportType::SINGLE_SUPPORT);
  EXPECT_EQ(opKernelInfoPtr->IsMultiKernelSupport(), false);
  EXPECT_EQ(opKernelInfoPtr->GetPrecisionPolicy(), PrecisionPolicy::WHITE);
  EXPECT_EQ(opKernelInfoPtr->IsOpFileNull(), false);

  opKernelInfoPtr = opsKernelManager.GetOpKernelInfoByOpType("tbe-custom", "ScatterNdUpdate1");
  flag = opKernelInfoPtr == nullptr;
  EXPECT_EQ(flag, true);

  opKernelInfoPtr = opsKernelManager.GetOpKernelInfoByOpType("tbe-custom", "ScatterNdUpdate2");
  flag = opKernelInfoPtr == nullptr;
  ASSERT_EQ(flag, false);
  std::vector<std::string> expect_refer_vec2 = {"var", "x"};
  EXPECT_EQ(opKernelInfoPtr->GetReferTensorNameVec(), expect_refer_vec2);
  EXPECT_EQ(opKernelInfoPtr->GetDynamicCompileStatic(), DynamicCompileStatic::COMPILE);
  EXPECT_EQ(opKernelInfoPtr->IsDynamicCompileStatic(), true);
  EXPECT_EQ(opKernelInfoPtr->GetJitCompileType(), JitCompile::REUSE_BINARY);
  EXPECT_EQ(opKernelInfoPtr->GetEnableVectorCore(), VectorCoreType::ENABLE);
  EXPECT_EQ(opKernelInfoPtr->GetMultiKernelSupportType(), MultiKernelSupportType::MULTI_SUPPORT);
  EXPECT_EQ(opKernelInfoPtr->IsMultiKernelSupport(), true);
  EXPECT_EQ(opKernelInfoPtr->GetPrecisionPolicy(), PrecisionPolicy::BLACK);
  EXPECT_EQ(opKernelInfoPtr->IsOpFileNull(), true);
}
