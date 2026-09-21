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
#include "ge/ge_api_types.h"
#include "fe_llt_utils.h"
#define protected public
#define private public
#include "adapter/tbe_adapter/tbe_info/tbe_info_assembler.h"
#include "adapter/tbe_adapter/tbe_info/execution_time_estimator.h"
#include "ops_kernel_store/fe_ops_kernel_info_store.h"
#include "ops_store/ops_kernel_manager.h"
#include "fusion_manager/fusion_manager.h"
#include "common/config_parser/op_impl_mode_config_parser.h"
#include "common/platform_utils.h"
#undef protected
#undef private

using namespace std;
using namespace ge;
using namespace fe;

class STestTbeTimeEstimator : public testing::Test
{
 protected:
  static void SetUpTestCase() {
    PlatformUtils::Instance().soc_version_ = "Ascend310P1";
    PlatformUtils::Instance().short_soc_version_ = "Ascend310P";
  }
  static void TearDownTestCase() {
    PlatformUtils::Instance().soc_version_ = "Ascend910B1";
    PlatformUtils::Instance().short_soc_version_ = "Ascend910B";
  }
  void SetUp() {
    fe_ops_kernel_info_store_ptr_ = make_shared<fe::FEOpsKernelInfoStore>(AI_CORE_NAME);
    FEOpsStoreInfo tbe_builtin {
        6, "tbe-builtin", EN_IMPL_HW_TBE,
        GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/format_selector/fe_config/tbe_dynamic_opinfo", ""
    };
    vector<FEOpsStoreInfo> store_info;
    store_info.emplace_back(tbe_builtin);
    Configuration::Instance(fe::AI_CORE_NAME).ops_store_info_vector_ = (store_info);
    OpsKernelManager::Instance(AI_CORE_NAME).Finalize();
    std::map<std::string, std::string> options;
    fe_ops_kernel_info_store_ptr_->Initialize(options);
    tbe_info_assembler_ptr_ = std::make_shared<TbeInfoAssembler>();
    tbe_info_assembler_ptr_->Initialize();
  }

  void TearDown() {}
 public:
  shared_ptr<fe::FEOpsKernelInfoStore> fe_ops_kernel_info_store_ptr_;
  TbeInfoAssemblerPtr tbe_info_assembler_ptr_;
};

/* ge.virtual_type is off*/
TEST_F(STestTbeTimeEstimator, test_case_01) {
  PlatformInfoManager::Instance().opti_compilation_infos_.SetSocVersion("Ascend310P3");
  tbe_info_assembler_ptr_->Initialize();
  OpDescPtr conv = std::make_shared<OpDesc>("conv2d", fe::CONV2D);
  vector<int64_t> fm_dims = {12, 2, 21, 32, 16};
  vector<int64_t> weight_dims = {504, 2, 16, 16};
  vector<int64_t> output_dims = {12, 2, 21, 32, 16};
  GeTensorDesc fm_tensor_desc(GeShape(fm_dims), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  GeTensorDesc weight_tensor_desc(GeShape(weight_dims), ge::FORMAT_FRACTAL_Z, ge::DT_FLOAT16);
  GeTensorDesc output_tensor_desc(GeShape(output_dims), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);

  conv->AddInputDesc(fm_tensor_desc);
  conv->AddInputDesc(weight_tensor_desc);
  conv->AddInputDesc(output_tensor_desc);
  map<std::string, std::string> options;
  options[ge::AICORE_NUM] = "100";
  uint64_t exec_time = 0;
  ExecutionTimeEstimator::GetExecTime(tbe_info_assembler_ptr_->all_plat_info_.platform_info, *conv,
                                      nullptr, exec_time);
  EXPECT_EQ(exec_time, 0);
  tbe_info_assembler_ptr_->CalibrateCoreNum(*conv, AI_CORE_NAME, nullptr, options);
  EXPECT_EQ(options[ge::AICORE_NUM], "100");
}

TEST_F(STestTbeTimeEstimator, test_case_02) {
  PlatformInfoManager::Instance().opti_compilation_infos_.SetSocVersion("Ascend310P3");
  tbe_info_assembler_ptr_->Initialize();
  OpDescPtr conv = std::make_shared<OpDesc>("conv2d", fe::CONV2D);
  vector<int64_t> fm_dims = {12, 2, 21, 32, 16};
  vector<int64_t> weight_dims = {504, 2, 16, 16};
  vector<int64_t> output_dims = {12, 2, 21, 32, 16};
  GeTensorDesc fm_tensor_desc(GeShape(fm_dims), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  GeTensorDesc weight_tensor_desc(GeShape(weight_dims), ge::FORMAT_FRACTAL_Z, ge::DT_FLOAT16);
  GeTensorDesc output_tensor_desc(GeShape(output_dims), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);

  conv->AddInputDesc(fm_tensor_desc);
  conv->AddInputDesc(weight_tensor_desc);
  conv->AddOutputDesc(output_tensor_desc);
  map<std::string, std::string> options;
  options["ge.virtual_type"] = "1";
  options[ge::AICORE_NUM] = "100";
  uint64_t exec_time = 0;
  ExecutionTimeEstimator::GetExecTime(tbe_info_assembler_ptr_->all_plat_info_.platform_info, *conv,
                                      nullptr, exec_time);
  EXPECT_EQ(exec_time, 479275);
  tbe_info_assembler_ptr_->CalibrateCoreNum(*conv, AI_CORE_NAME, nullptr, options);
  EXPECT_EQ(options[ge::AICORE_NUM], "8");
}

TEST_F(STestTbeTimeEstimator, test_case_02_1) {
  PlatformInfoManager::Instance().opti_compilation_infos_.SetSocVersion("Ascend310P1");
  tbe_info_assembler_ptr_->Initialize();
  OpDescPtr conv = std::make_shared<OpDesc>("conv2d", fe::CONV2D);
  vector<int64_t> fm_dims = {12, 2, 21, 32, 16};
  vector<int64_t> weight_dims = {504, 2, 16, 16};
  vector<int64_t> output_dims = {12, 2, 21, 32, 16};
  GeTensorDesc fm_tensor_desc(GeShape(fm_dims), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  GeTensorDesc weight_tensor_desc(GeShape(weight_dims), ge::FORMAT_FRACTAL_Z, ge::DT_FLOAT16);
  GeTensorDesc output_tensor_desc(GeShape(output_dims), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);

  conv->AddInputDesc(fm_tensor_desc);
  conv->AddInputDesc(weight_tensor_desc);
  conv->AddOutputDesc(output_tensor_desc);
  map<std::string, std::string> options;
  options["ge.virtual_type"] = "1";
  options[ge::AICORE_NUM] = "100";
  uint64_t exec_time = 0;
  ExecutionTimeEstimator::GetExecTime(tbe_info_assembler_ptr_->all_plat_info_.platform_info, *conv,
                                      nullptr, exec_time);
  EXPECT_EQ(exec_time, 479275);
  tbe_info_assembler_ptr_->CalibrateCoreNum(*conv, AI_CORE_NAME, nullptr, options);
  EXPECT_EQ(options[ge::AICORE_NUM], "40");
}

TEST_F(STestTbeTimeEstimator, test_case_02_2) {
  PlatformInfoManager::Instance().opti_compilation_infos_.SetSocVersion("Ascend310P1");
  tbe_info_assembler_ptr_->Initialize();
  OpDescPtr conv = std::make_shared<OpDesc>("conv2d", fe::CONV2D);
  vector<int64_t> fm_dims = {12, 2, 21, 2, 16};
  vector<int64_t> weight_dims = {504, 2, 16, 16};
  vector<int64_t> output_dims = {12, 2, 21, 32, 16};
  GeTensorDesc fm_tensor_desc(GeShape(fm_dims), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  GeTensorDesc weight_tensor_desc(GeShape(weight_dims), ge::FORMAT_FRACTAL_Z, ge::DT_FLOAT16);
  GeTensorDesc output_tensor_desc(GeShape(output_dims), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);

  conv->AddInputDesc(fm_tensor_desc);
  conv->AddInputDesc(weight_tensor_desc);
  conv->AddOutputDesc(output_tensor_desc);
  map<std::string, std::string> options;
  options["ge.virtual_type"] = "1";
  options[ge::AICORE_NUM] = "100";
  uint64_t exec_time = 0;
  ExecutionTimeEstimator::GetExecTime(tbe_info_assembler_ptr_->all_plat_info_.platform_info, *conv,
                                      nullptr, exec_time);
  EXPECT_EQ(exec_time, 34233);
  tbe_info_assembler_ptr_->CalibrateCoreNum(*conv, AI_CORE_NAME, nullptr, options);
  EXPECT_EQ(options[ge::AICORE_NUM], "10");
}

TEST_F(STestTbeTimeEstimator, test_case_02_3) {
  PlatformInfoManager::Instance().opti_compilation_infos_.SetSocVersion("Ascend310P1");
  tbe_info_assembler_ptr_->Initialize();
  OpDescPtr conv = std::make_shared<OpDesc>("conv2d", fe::CONV2D);
  vector<int64_t> fm_dims = {12, 2, 7, 2, 16};
  vector<int64_t> weight_dims = {504, 2, 16, 16};
  vector<int64_t> output_dims = {12, 2, 21, 32, 16};
  GeTensorDesc fm_tensor_desc(GeShape(fm_dims), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  GeTensorDesc weight_tensor_desc(GeShape(weight_dims), ge::FORMAT_FRACTAL_Z, ge::DT_FLOAT16);
  GeTensorDesc output_tensor_desc(GeShape(output_dims), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);

  conv->AddInputDesc(fm_tensor_desc);
  conv->AddInputDesc(weight_tensor_desc);
  conv->AddOutputDesc(output_tensor_desc);
  map<std::string, std::string> options;
  options["ge.virtual_type"] = "1";
  options[ge::AICORE_NUM] = "100";
  uint64_t exec_time = 0;
  ExecutionTimeEstimator::GetExecTime(tbe_info_assembler_ptr_->all_plat_info_.platform_info, *conv,
                                      nullptr, exec_time);
  EXPECT_EQ(exec_time, 11411);
  tbe_info_assembler_ptr_->CalibrateCoreNum(*conv, AI_CORE_NAME, nullptr, options);
  EXPECT_EQ(options[ge::AICORE_NUM], "10");
}

TEST_F(STestTbeTimeEstimator, test_case_02_4) {
  PlatformInfoManager::Instance().opti_compilation_infos_.SetSocVersion("Ascend310P1");
  tbe_info_assembler_ptr_->Initialize();
  OpDescPtr conv = std::make_shared<OpDesc>("conv2d", fe::CONV2D);
  vector<int64_t> fm_dims = {70000, 2, 7, 2, 16};
  vector<int64_t> weight_dims = {450, 2, 16, 16};
  vector<int64_t> output_dims = {12, 2, 21, 32, 16};
  GeTensorDesc fm_tensor_desc(GeShape(fm_dims), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  GeTensorDesc weight_tensor_desc(GeShape(weight_dims), ge::FORMAT_FRACTAL_Z, ge::DT_FLOAT16);
  GeTensorDesc output_tensor_desc(GeShape(output_dims), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);

  conv->AddInputDesc(fm_tensor_desc);
  conv->AddInputDesc(weight_tensor_desc);
  conv->AddOutputDesc(output_tensor_desc);
  map<std::string, std::string> options;
  options["ge.virtual_type"] = "1";
  options[ge::AICORE_NUM] = "100";
  uint64_t exec_time = 0;
  ExecutionTimeEstimator::GetExecTime(tbe_info_assembler_ptr_->all_plat_info_.platform_info, *conv,
                                      nullptr, exec_time);
  EXPECT_EQ(exec_time, 59433962);
  tbe_info_assembler_ptr_->CalibrateCoreNum(*conv, AI_CORE_NAME, nullptr, options);
  EXPECT_EQ(options[ge::AICORE_NUM], "40");
}

TEST_F(STestTbeTimeEstimator, test_case_02_5) {
  PlatformInfoManager::Instance().opti_compilation_infos_.SetSocVersion("Ascend310P1");
  tbe_info_assembler_ptr_->Initialize();
  OpDescPtr conv = std::make_shared<OpDesc>("conv2d", fe::CONV2D);
  vector<int64_t> fm_dims = {1, 2, 7, 2, 16};
  vector<int64_t> weight_dims = {504, 2, 16, 16};
  vector<int64_t> output_dims = {2, 2, 21, 32, 16};
  GeTensorDesc fm_tensor_desc(GeShape(fm_dims), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  GeTensorDesc weight_tensor_desc(GeShape(weight_dims), ge::FORMAT_FRACTAL_Z, ge::DT_FLOAT16);
  GeTensorDesc output_tensor_desc(GeShape(output_dims), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);

  conv->AddInputDesc(fm_tensor_desc);
  conv->AddInputDesc(weight_tensor_desc);
  conv->AddOutputDesc(output_tensor_desc);
  map<std::string, std::string> options;
  options["ge.virtual_type"] = "1";
  options[ge::AICORE_NUM] = "100";
  uint64_t exec_time = 0;
  ExecutionTimeEstimator::GetExecTime(tbe_info_assembler_ptr_->all_plat_info_.platform_info, *conv,
                                      nullptr, exec_time);
  EXPECT_EQ(exec_time, 950);
  tbe_info_assembler_ptr_->CalibrateCoreNum(*conv, AI_CORE_NAME, nullptr, options);
  EXPECT_EQ(options[ge::AICORE_NUM], "10");
}

TEST_F(STestTbeTimeEstimator, test_case_03) {
  PlatformInfoManager::Instance().opti_compilation_infos_.SetSocVersion("Ascend310P3");
  tbe_info_assembler_ptr_->Initialize();
  OpDescPtr conv = std::make_shared<OpDesc>("conv2d", fe::CONV2D);
  vector<int64_t> fm_dims = {1, 2, 21, 32, 16};
  vector<int64_t> weight_dims = {504, 2, 16, 16};
  vector<int64_t> output_dims = {1, 2, 21, 32, 16};
  GeTensorDesc fm_tensor_desc(GeShape(fm_dims), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  GeTensorDesc weight_tensor_desc(GeShape(weight_dims), ge::FORMAT_FRACTAL_Z, ge::DT_FLOAT16);
  GeTensorDesc output_tensor_desc(GeShape(output_dims), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);

  conv->AddInputDesc(fm_tensor_desc);
  conv->AddInputDesc(weight_tensor_desc);
  conv->AddOutputDesc(output_tensor_desc);
  map<std::string, std::string> options;
  options["ge.virtual_type"] = "1";
  options[ge::AICORE_NUM] = "100";
  uint64_t exec_time = 0;
  ExecutionTimeEstimator::GetExecTime(tbe_info_assembler_ptr_->all_plat_info_.platform_info, *conv,
                                      nullptr, exec_time);
  EXPECT_EQ(exec_time, 39939);
  tbe_info_assembler_ptr_->CalibrateCoreNum(*conv, AI_CORE_NAME, nullptr, options);
  EXPECT_EQ(options[ge::AICORE_NUM], "8");
}

TEST_F(STestTbeTimeEstimator, test_case_04) {
  PlatformInfoManager::Instance().opti_compilation_infos_.SetSocVersion("Ascend310P3");
  tbe_info_assembler_ptr_->Initialize();
  OpDescPtr conv = std::make_shared<OpDesc>("conv2d", fe::CONV2D);
  vector<int64_t> fm_dims = {1, 2, 5, 32, 16};
  vector<int64_t> weight_dims = {504, 2, 16, 16};
  vector<int64_t> output_dims = {1, 2, 5, 32, 16};
  GeTensorDesc fm_tensor_desc(GeShape(fm_dims), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  GeTensorDesc weight_tensor_desc(GeShape(weight_dims), ge::FORMAT_FRACTAL_Z, ge::DT_FLOAT16);
  GeTensorDesc output_tensor_desc(GeShape(output_dims), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);

  conv->AddInputDesc(fm_tensor_desc);
  conv->AddInputDesc(weight_tensor_desc);
  conv->AddOutputDesc(output_tensor_desc);
  map<std::string, std::string> options;
  options["ge.virtual_type"] = "1";
  options[ge::AICORE_NUM] = "100";
  uint64_t exec_time = 0;
  ExecutionTimeEstimator::GetExecTime(tbe_info_assembler_ptr_->all_plat_info_.platform_info, *conv,
                                      nullptr, exec_time);
  EXPECT_EQ(exec_time, 9509);
  tbe_info_assembler_ptr_->CalibrateCoreNum(*conv, AI_CORE_NAME, nullptr, options);
  EXPECT_EQ(options[ge::AICORE_NUM], "8");
}

TEST_F(STestTbeTimeEstimator, test_case_05) {
  PlatformInfoManager::Instance().opti_compilation_infos_.SetSocVersion("Ascend310P3");
  tbe_info_assembler_ptr_->Initialize();
  OpDescPtr conv = std::make_shared<OpDesc>("conv2d", fe::CONV2D);
  vector<int64_t> fm_dims = {2, 2, 21, 32, 16};
  vector<int64_t> weight_dims = {504, 2, 16, 16};
  vector<int64_t> output_dims = {2, 2, 21, 32, 16};
  GeTensorDesc fm_tensor_desc(GeShape(fm_dims), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  GeTensorDesc weight_tensor_desc(GeShape(weight_dims), ge::FORMAT_FRACTAL_Z, ge::DT_FLOAT16);
  GeTensorDesc output_tensor_desc(GeShape(output_dims), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);

  conv->AddInputDesc(fm_tensor_desc);
  conv->AddInputDesc(weight_tensor_desc);
  conv->AddOutputDesc(output_tensor_desc);
  map<std::string, std::string> options;
  options["ge.virtual_type"] = "1";
  options[ge::AICORE_NUM] = "100";
  uint64_t exec_time = 0;
  ExecutionTimeEstimator::GetExecTime(tbe_info_assembler_ptr_->all_plat_info_.platform_info, *conv,
                                      nullptr, exec_time);
  EXPECT_EQ(exec_time, 79879);
  tbe_info_assembler_ptr_->CalibrateCoreNum(*conv, VECTOR_CORE_NAME, nullptr, options);
  EXPECT_EQ(options[ge::AICORE_NUM], "7");
}

/* Vector op */
TEST_F(STestTbeTimeEstimator, test_case_06) {
  PlatformInfoManager::Instance().opti_compilation_infos_.SetSocVersion("Ascend310P3");
  tbe_info_assembler_ptr_->Initialize();
  OpDescPtr relu = std::make_shared<OpDesc>("relu", fe::RELU);
  vector<int64_t> dims = {2, 2, 21, 32, 16};
  GeTensorDesc tensor_desc(GeShape(dims), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);

  relu->AddInputDesc(tensor_desc);
  relu->AddOutputDesc(tensor_desc);
  map<std::string, std::string> options;
  options["ge.virtual_type"] = "1";
  options[ge::AICORE_NUM] = "100";

  OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetHighPrioOpKernelInfoPtr(fe::RELU);
  uint64_t exec_time = 0;
  ExecutionTimeEstimator::GetExecTime(tbe_info_assembler_ptr_->all_plat_info_.platform_info, *relu,
                                      op_kernel_info_ptr, exec_time);
  EXPECT_EQ(exec_time, 950);
  tbe_info_assembler_ptr_->CalibrateCoreNum(*relu, AI_CORE_NAME, op_kernel_info_ptr, options);
  EXPECT_EQ(options[ge::AICORE_NUM], "8");
}

/* Vector op */
TEST_F(STestTbeTimeEstimator, test_case_07) {
  PlatformInfoManager::Instance().opti_compilation_infos_.SetSocVersion("Ascend310P3");
  tbe_info_assembler_ptr_->Initialize();
  OpDescPtr relu = std::make_shared<OpDesc>("relu", fe::RELU);
  vector<int64_t> dims = {12, 2, 21, 32, 16};
  GeTensorDesc tensor_desc(GeShape(dims), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);

  relu->AddInputDesc(tensor_desc);
  relu->AddOutputDesc(tensor_desc);
  map<std::string, std::string> options;
  options["ge.virtual_type"] = "1";
  options[ge::AICORE_NUM] = "100";

  OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetHighPrioOpKernelInfoPtr(fe::RELU);
  uint64_t exec_time = 0;
  ExecutionTimeEstimator::GetExecTime(tbe_info_assembler_ptr_->all_plat_info_.platform_info, *relu,
                                      op_kernel_info_ptr, exec_time);
  EXPECT_EQ(exec_time, 5705);
  tbe_info_assembler_ptr_->CalibrateCoreNum(*relu, AI_CORE_NAME, op_kernel_info_ptr, options);
  EXPECT_EQ(options[ge::AICORE_NUM], "8");
}

TEST_F(STestTbeTimeEstimator, test_case_07_1) {
  PlatformInfoManager::Instance().opti_compilation_infos_.SetSocVersion("Ascend310P3");
  tbe_info_assembler_ptr_->Initialize();
  OpDescPtr relu = std::make_shared<OpDesc>("relu", fe::RELU);
  vector<int64_t> dims = {53, 2, 8, 4, 256};
  GeTensorDesc tensor_desc(GeShape(dims), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);

  relu->AddInputDesc(tensor_desc);
  relu->AddOutputDesc(tensor_desc);
  map<std::string, std::string> options;
  options["ge.virtual_type"] = "1";
  options[ge::AICORE_NUM] = "100";

  OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetHighPrioOpKernelInfoPtr(fe::RELU);
  uint64_t exec_time = 0;
  ExecutionTimeEstimator::GetExecTime(tbe_info_assembler_ptr_->all_plat_info_.platform_info, *relu,
                                      op_kernel_info_ptr, exec_time);
  EXPECT_EQ(exec_time, 19200);
  tbe_info_assembler_ptr_->CalibrateCoreNum(*relu, AI_CORE_NAME, op_kernel_info_ptr, options);
  EXPECT_EQ(options[ge::AICORE_NUM], "8");
}

/* Vector op */
TEST_F(STestTbeTimeEstimator, test_case_08) {
  PlatformInfoManager::Instance().opti_compilation_infos_.SetSocVersion("Ascend310P3");
  tbe_info_assembler_ptr_->Initialize();
  OpDescPtr relu = std::make_shared<OpDesc>("relu", fe::RELU);
  vector<int64_t> dims = {12, 8, 21, 32, 16};
  GeTensorDesc tensor_desc(GeShape(dims), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);

  relu->AddInputDesc(tensor_desc);
  relu->AddOutputDesc(tensor_desc);
  map<std::string, std::string> options;
  options["ge.virtual_type"] = "1";
  options[ge::AICORE_NUM] = "100";

  OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetHighPrioOpKernelInfoPtr(fe::RELU);
  uint64_t exec_time = 0;
  ExecutionTimeEstimator::GetExecTime(tbe_info_assembler_ptr_->all_plat_info_.platform_info, *relu,
                                      op_kernel_info_ptr, exec_time);
  EXPECT_EQ(exec_time, 22822);
  tbe_info_assembler_ptr_->CalibrateCoreNum(*relu, AI_CORE_NAME, op_kernel_info_ptr, options);
  EXPECT_EQ(options[ge::AICORE_NUM], "8");
}

/* Vector op */
TEST_F(STestTbeTimeEstimator, test_case_09) {
  PlatformInfoManager::Instance().opti_compilation_infos_.SetSocVersion("Ascend310P3");
  tbe_info_assembler_ptr_->Initialize();
  OpDescPtr relu = std::make_shared<OpDesc>("relu", fe::RELU);
  vector<int64_t> dims = {120000, 8, 210, 32, 16};
  GeTensorDesc tensor_desc(GeShape(dims), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);

  relu->AddInputDesc(tensor_desc);
  relu->AddOutputDesc(tensor_desc);
  map<std::string, std::string> options;
  options["ge.virtual_type"] = "1";
  options[ge::AICORE_NUM] = "100";

  OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetHighPrioOpKernelInfoPtr(fe::RELU);
  uint64_t exec_time = 0;
  ExecutionTimeEstimator::GetExecTime(tbe_info_assembler_ptr_->all_plat_info_.platform_info, *relu,
                                      op_kernel_info_ptr, exec_time);
  EXPECT_EQ(exec_time, 2282264150);
  tbe_info_assembler_ptr_->CalibrateCoreNum(*relu, AI_CORE_NAME, op_kernel_info_ptr, options);
  EXPECT_EQ(options[ge::AICORE_NUM], "8");
}

TEST_F(STestTbeTimeEstimator, test_case_09_1) {
  PlatformInfoManager::Instance().opti_compilation_infos_.SetSocVersion("Ascend310P3");
  tbe_info_assembler_ptr_->Initialize();
  OpDescPtr pooling = std::make_shared<OpDesc>("relu", fe::POOLING);
  vector<int64_t> dims = {12, 8, 210, 32, 16};
  GeTensorDesc tensor_desc(GeShape(dims), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);

  pooling->AddInputDesc(tensor_desc);
  pooling->AddOutputDesc(tensor_desc);
  map<std::string, std::string> options;
  options["ge.virtual_type"] = "1";
  options[ge::AICORE_NUM] = "100";

  uint64_t exec_time = 0;
  ExecutionTimeEstimator::GetExecTime(tbe_info_assembler_ptr_->all_plat_info_.platform_info, *pooling,
                                      nullptr, exec_time);
  EXPECT_EQ(exec_time, 228226);
  tbe_info_assembler_ptr_->CalibrateCoreNum(*pooling, AI_CORE_NAME, nullptr, options);
  EXPECT_EQ(options[ge::AICORE_NUM], "8");
}

/* MatMul Op */
TEST_F(STestTbeTimeEstimator, test_case_10) {
  PlatformInfoManager::Instance().opti_compilation_infos_.SetSocVersion("Ascend310P3");
  tbe_info_assembler_ptr_->Initialize();
  OpDescPtr matmul = std::make_shared<OpDesc>("matmul", fe::MATMULV2OP);
  ge::AttrUtils::SetBool(matmul, "transpose_x1", false);
  ge::AttrUtils::SetBool(matmul, "transpose_x1", false);
  vector<int64_t> fm_dims = {12, 2, 210, 16, 16};
  vector<int64_t> weight_dims = {12, 2, 35, 16, 16};
  vector<int64_t> output_dims = {12, 25, 21, 32, 16};
  GeTensorDesc fm_tensor_desc(GeShape(fm_dims), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
  GeTensorDesc weight_tensor_desc(GeShape(weight_dims), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
  GeTensorDesc output_tensor_desc(GeShape(output_dims), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);


  matmul->AddInputDesc(fm_tensor_desc);
  matmul->AddInputDesc(weight_tensor_desc);
  matmul->AddOutputDesc(output_tensor_desc);
  map<std::string, std::string> options;
  options["ge.virtual_type"] = "1";
  options[ge::AICORE_NUM] = "100";
  uint64_t exec_time = 0;
  ExecutionTimeEstimator::GetExecTime(tbe_info_assembler_ptr_->all_plat_info_.platform_info, *matmul,
                                      nullptr, exec_time);
  EXPECT_EQ(exec_time, 9509);
  tbe_info_assembler_ptr_->CalibrateCoreNum(*matmul, AI_CORE_NAME, nullptr, options);
  EXPECT_EQ(options[ge::AICORE_NUM], "8");
}

/* MatMul Op */
TEST_F(STestTbeTimeEstimator, test_case_11) {
  PlatformInfoManager::Instance().opti_compilation_infos_.SetSocVersion("Ascend310P3");
  tbe_info_assembler_ptr_->Initialize();
  OpDescPtr matmul = std::make_shared<OpDesc>("matmul", fe::MATMULV2OP);
  ge::AttrUtils::SetBool(matmul, "transpose_x1", true);
  ge::AttrUtils::SetBool(matmul, "transpose_x2", false);
  vector<int64_t> fm_dims = {12, 2, 210, 16, 16};
  vector<int64_t> weight_dims = {12, 2, 35, 16, 16};
  vector<int64_t> output_dims = {12, 25, 21, 32, 16};
  GeTensorDesc fm_tensor_desc(GeShape(fm_dims), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
  GeTensorDesc weight_tensor_desc(GeShape(weight_dims), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
  GeTensorDesc output_tensor_desc(GeShape(output_dims), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);


  matmul->AddInputDesc(fm_tensor_desc);
  matmul->AddInputDesc(weight_tensor_desc);
  matmul->AddOutputDesc(output_tensor_desc);
  map<std::string, std::string> options;
  options["ge.virtual_type"] = "1";
  options[ge::AICORE_NUM] = "100";
  uint64_t exec_time = 0;
  ExecutionTimeEstimator::GetExecTime(tbe_info_assembler_ptr_->all_plat_info_.platform_info, *matmul,
                                      nullptr, exec_time);
  EXPECT_EQ(exec_time, 9509);
  tbe_info_assembler_ptr_->CalibrateCoreNum(*matmul, AI_CORE_NAME, nullptr, options);
  EXPECT_EQ(options[ge::AICORE_NUM], "8");
}

/* MatMul Op */
TEST_F(STestTbeTimeEstimator, test_case_12) {
  PlatformInfoManager::Instance().opti_compilation_infos_.SetSocVersion("Ascend310P3");
  tbe_info_assembler_ptr_->Initialize();
  OpDescPtr matmul = std::make_shared<OpDesc>("matmul", fe::MATMULV2OP);
  ge::AttrUtils::SetBool(matmul, "transpose_x1", true);
  ge::AttrUtils::SetBool(matmul, "transpose_x2", true);
  vector<int64_t> fm_dims = {12, 2, 210, 16, 16};
  vector<int64_t> weight_dims = {12, 2, 35, 16, 16};
  vector<int64_t> output_dims = {12, 25, 21, 32, 16};
  GeTensorDesc fm_tensor_desc(GeShape(fm_dims), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
  GeTensorDesc weight_tensor_desc(GeShape(weight_dims), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
  GeTensorDesc output_tensor_desc(GeShape(output_dims), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);


  matmul->AddInputDesc(fm_tensor_desc);
  matmul->AddInputDesc(weight_tensor_desc);
  matmul->AddOutputDesc(output_tensor_desc);
  map<std::string, std::string> options;
  options["ge.virtual_type"] = "1";
  options[ge::AICORE_NUM] = "100";
  uint64_t exec_time = 0;
  ExecutionTimeEstimator::GetExecTime(tbe_info_assembler_ptr_->all_plat_info_.platform_info, *matmul,
                                      nullptr, exec_time);
  EXPECT_EQ(exec_time, 166415);
  tbe_info_assembler_ptr_->CalibrateCoreNum(*matmul, AI_CORE_NAME, nullptr, options);
  EXPECT_EQ(options[ge::AICORE_NUM], "8");
}

/* MatMul Op */
TEST_F(STestTbeTimeEstimator, test_case_13) {
  PlatformInfoManager::Instance().opti_compilation_infos_.SetSocVersion("Ascend310P3");
  tbe_info_assembler_ptr_->Initialize();
  OpDescPtr matmul = std::make_shared<OpDesc>("matmul", fe::MATMULV2OP);
  ge::AttrUtils::SetBool(matmul, "transpose_x1", false);
  ge::AttrUtils::SetBool(matmul, "transpose_x2", true);
  vector<int64_t> fm_dims = {12, 2, 210, 16, 16};
  vector<int64_t> weight_dims = {12, 2, 35, 16, 16};
  vector<int64_t> output_dims = {12, 25, 21, 32, 16};
  GeTensorDesc fm_tensor_desc(GeShape(fm_dims), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
  GeTensorDesc weight_tensor_desc(GeShape(weight_dims), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
  GeTensorDesc output_tensor_desc(GeShape(output_dims), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);


  matmul->AddInputDesc(fm_tensor_desc);
  matmul->AddInputDesc(weight_tensor_desc);
  matmul->AddOutputDesc(output_tensor_desc);
  map<std::string, std::string> options;
  options["ge.virtual_type"] = "1";
  options[ge::AICORE_NUM] = "100";
  uint64_t exec_time = 0;
  ExecutionTimeEstimator::GetExecTime(tbe_info_assembler_ptr_->all_plat_info_.platform_info, *matmul,
                                      nullptr, exec_time);
  EXPECT_EQ(exec_time, 166415);
  tbe_info_assembler_ptr_->CalibrateCoreNum(*matmul, AI_CORE_NAME, nullptr, options);
  EXPECT_EQ(options[ge::AICORE_NUM], "8");
}

TEST_F(STestTbeTimeEstimator, test_case_14) {
  PlatformInfoManager::Instance().opti_compilation_infos_.SetSocVersion("Ascend310P3");
  tbe_info_assembler_ptr_->Initialize();
  OpDescPtr matmul = std::make_shared<OpDesc>("matmul", fe::MATMULV2OP);
  ge::AttrUtils::SetBool(matmul, "transpose_x1", false);
  ge::AttrUtils::SetBool(matmul, "transpose_x1", false);
  vector<int64_t> fm_dims = {12, 2, 210, 16, 16};
  vector<int64_t> weight_dims = {12, 2, 35, 16, 16};
  vector<int64_t> output_dims = {12, 25, 21, 32, 16};
  GeTensorDesc fm_tensor_desc(GeShape(fm_dims), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT);
  GeTensorDesc weight_tensor_desc(GeShape(weight_dims), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
  GeTensorDesc output_tensor_desc(GeShape(output_dims), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);


  matmul->AddInputDesc(fm_tensor_desc);
  matmul->AddInputDesc(weight_tensor_desc);
  matmul->AddOutputDesc(output_tensor_desc);
  map<std::string, std::string> options;
  options["ge.virtual_type"] = "1";
  options[ge::AICORE_NUM] = "100";
  uint64_t exec_time = 0;
  ExecutionTimeEstimator::GetExecTime(tbe_info_assembler_ptr_->all_plat_info_.platform_info, *matmul,
                                      nullptr, exec_time);
  EXPECT_EQ(exec_time, 0);
  tbe_info_assembler_ptr_->CalibrateCoreNum(*matmul, AI_CORE_NAME, nullptr, options);
  EXPECT_EQ(options[ge::AICORE_NUM], "100");
}


TEST_F(STestTbeTimeEstimator, test_case_14_1) {
  PlatformInfoManager::Instance().opti_compilation_infos_.SetSocVersion("Ascend310P3");
  tbe_info_assembler_ptr_->Initialize();
  OpDescPtr matmul = std::make_shared<OpDesc>("matmul", fe::MATMULV2OP);
  ge::AttrUtils::SetBool(matmul, "transpose_x1", false);
  ge::AttrUtils::SetBool(matmul, "transpose_x1", false);
  vector<int64_t> fm_dims = {12, 2, 210, 16, 16};
  vector<int64_t> weight_dims = {12, 2, 35, 16, 16};
  vector<int64_t> output_dims = {12, 25, 21, 32, 16};
  GeTensorDesc fm_tensor_desc(GeShape(fm_dims), ge::FORMAT_FRACTAL_Z, ge::DT_FLOAT16);
  GeTensorDesc weight_tensor_desc(GeShape(weight_dims), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
  GeTensorDesc output_tensor_desc(GeShape(output_dims), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);


  matmul->AddInputDesc(fm_tensor_desc);
  matmul->AddInputDesc(weight_tensor_desc);
  matmul->AddOutputDesc(output_tensor_desc);
  map<std::string, std::string> options;
  options["ge.virtual_type"] = "1";
  options[ge::AICORE_NUM] = "100";
  uint64_t exec_time = 0;
  ExecutionTimeEstimator::GetExecTime(tbe_info_assembler_ptr_->all_plat_info_.platform_info, *matmul,
                                      nullptr, exec_time);
  EXPECT_EQ(exec_time, 0);
  tbe_info_assembler_ptr_->CalibrateCoreNum(*matmul, AI_CORE_NAME, nullptr, options);
  EXPECT_EQ(options[ge::AICORE_NUM], "100");
}

TEST_F(STestTbeTimeEstimator, test_case_14_2) {
  PlatformInfoManager::Instance().opti_compilation_infos_.SetSocVersion("Ascend310P3");
  tbe_info_assembler_ptr_->Initialize();
  OpDescPtr matmul = std::make_shared<OpDesc>("matmul", fe::MATMULV2OP);
  ge::AttrUtils::SetBool(matmul, "transpose_x1", false);
  ge::AttrUtils::SetBool(matmul, "transpose_x1", false);
  vector<int64_t> fm_dims = {210, 16, 16};
  vector<int64_t> weight_dims = {12, 2, 35, 16, 16};
  vector<int64_t> output_dims = {12, 25, 21, 32, 16};
  GeTensorDesc fm_tensor_desc(GeShape(fm_dims), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
  GeTensorDesc weight_tensor_desc(GeShape(weight_dims), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
  GeTensorDesc output_tensor_desc(GeShape(output_dims), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);


  matmul->AddInputDesc(fm_tensor_desc);
  matmul->AddInputDesc(weight_tensor_desc);
  matmul->AddOutputDesc(output_tensor_desc);
  map<std::string, std::string> options;
  options["ge.virtual_type"] = "1";
  options[ge::AICORE_NUM] = "100";
  uint64_t exec_time = 0;
  ExecutionTimeEstimator::GetExecTime(tbe_info_assembler_ptr_->all_plat_info_.platform_info, *matmul,
                                      nullptr, exec_time);
  EXPECT_EQ(exec_time, 0);
  tbe_info_assembler_ptr_->CalibrateCoreNum(*matmul, AI_CORE_NAME, nullptr, options);
  EXPECT_EQ(options[ge::AICORE_NUM], "100");
}

TEST_F(STestTbeTimeEstimator, test_case_no_need_to_calibrate_corenum) {
  PlatformInfoManager::Instance().opti_compilation_infos_.SetSocVersion("Ascend310P1");
  tbe_info_assembler_ptr_->Initialize();
  OpDescPtr conv = std::make_shared<OpDesc>("conv2d", fe::CONV2D);
  vector<int64_t> fm_dims = {70000, 2, 7, 2, 16};
  vector<int64_t> weight_dims = {450, 2, 16, 16};
  vector<int64_t> output_dims = {12, 2, 21, 32, 16};
  GeTensorDesc fm_tensor_desc(GeShape(fm_dims), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  GeTensorDesc weight_tensor_desc(GeShape(weight_dims), ge::FORMAT_FRACTAL_Z, ge::DT_FLOAT16);
  GeTensorDesc output_tensor_desc(GeShape(output_dims), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);

  conv->AddInputDesc(fm_tensor_desc);
  conv->AddInputDesc(weight_tensor_desc);
  conv->AddOutputDesc(output_tensor_desc);
  map<std::string, std::string> options;
  options["ge.virtual_type"] = "1";
  options[ge::AICORE_NUM] = "10";
  uint64_t exec_time = 0;
  ExecutionTimeEstimator::GetExecTime(tbe_info_assembler_ptr_->all_plat_info_.platform_info, *conv,
                                      nullptr, exec_time);
  EXPECT_EQ(exec_time, 59433962);
  tbe_info_assembler_ptr_->CalibrateCoreNum(*conv, AI_CORE_NAME, nullptr, options);
  EXPECT_EQ(options[ge::AICORE_NUM], "40");

  PlatformInfoManager::Instance().opti_compilation_infos_.SetSocVersion("Ascend910B");
  PlatformUtils::Instance().soc_version_ = "Ascend910B";
  PlatformUtils::Instance().short_soc_version_ = "Ascend910B";
  options[ge::AICORE_NUM] = "10";
  tbe_info_assembler_ptr_->CalibrateCoreNum(*conv, AI_CORE_NAME, nullptr, options);
  EXPECT_EQ(options[ge::AICORE_NUM], "10");
}

TEST_F(STestTbeTimeEstimator, test_set_impl_mode) {
  tbe_info_assembler_ptr_->Initialize();
  OpDescPtr conv = std::make_shared<OpDesc>("conv2d_test_set_impl_mode", fe::CONV2D);
  OpDescPtr conv_invalid = std::make_shared<OpDesc>("conv2d_test_set_impl_mode_invalid", "cccc");
  vector<int64_t> fm_dims = {12, 2, 21, 32, 16};
  vector<int64_t> weight_dims = {504, 2, 16, 16};
  vector<int64_t> output_dims = {12, 2, 21, 32, 16};
  GeTensorDesc fm_tensor_desc(GeShape(fm_dims), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  GeTensorDesc weight_tensor_desc(GeShape(weight_dims), ge::FORMAT_FRACTAL_Z, ge::DT_FLOAT16);
  GeTensorDesc output_tensor_desc(GeShape(output_dims), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);

  conv->AddInputDesc(fm_tensor_desc);
  conv->AddInputDesc(weight_tensor_desc);
  conv->AddInputDesc(output_tensor_desc);
  conv_invalid->AddInputDesc(fm_tensor_desc);
  conv_invalid->AddInputDesc(weight_tensor_desc);
  conv_invalid->AddInputDesc(output_tensor_desc);
  map<std::string, std::string> options;

  Configuration::Instance(AI_CORE_NAME).impl_mode_parser_ = std::make_shared<OpImplModeConfigParser>("");
  EXPECT_NE(Configuration::Instance(AI_CORE_NAME).impl_mode_parser_, nullptr);
  te::TbeOpInfo op_info = te::TbeOpInfo("", "", "", "");
  auto impl_mode_parser = std::dynamic_pointer_cast<OpImplModeConfigParser>(Configuration::Instance(AI_CORE_NAME).impl_mode_parser_);
  EXPECT_NE(impl_mode_parser, nullptr);
  impl_mode_parser->op_name_select_impl_mode_map_["conv2d_test_set_impl_mode"] = "enable_float_32_execution";
  tbe_info_assembler_ptr_->SetOpImplMode(AI_CORE_NAME, conv, op_info);
  int64_t op_impl_mode_num = 0;
  (void)ge::AttrUtils::GetInt(conv, OP_CUSTOM_IMPL_MODE_ENUM, op_impl_mode_num);
  EXPECT_EQ(op_impl_mode_num, 0);
  EXPECT_EQ(impl_mode_parser->IsEnableCustomImplMode(), false);
  (void)ge::AttrUtils::GetInt(conv, OP_IMPL_MODE_ENUM, op_impl_mode_num);
  EXPECT_EQ(op_impl_mode_num, 0x20);
  std::string op_impl_type = op_info.GetOpImplMode();
  EXPECT_EQ(op_impl_type, "enable_float_32_execution");

  impl_mode_parser->op_precision_mode_ = "xxx";
  impl_mode_parser->op_name_select_impl_mode_map_["conv2d_test_set_impl_mode"] = "enable_hi_float_32_execution";
  tbe_info_assembler_ptr_->SetOpImplMode(AI_CORE_NAME, conv, op_info);
  (void)ge::AttrUtils::GetInt(conv, OP_CUSTOM_IMPL_MODE_ENUM, op_impl_mode_num);
  EXPECT_EQ(op_impl_mode_num, 0x20);
  EXPECT_EQ(impl_mode_parser->IsEnableCustomImplMode(), true);
  op_impl_mode_num = 0;
  (void)ge::AttrUtils::GetInt(conv, OP_IMPL_MODE_ENUM, op_impl_mode_num);
  EXPECT_EQ(op_impl_mode_num, 0x20);
  op_impl_type = op_info.GetOpImplMode();
  EXPECT_EQ(op_impl_type, "enable_float_32_execution");

  (void)conv->DelAttr(OP_IMPL_MODE_ENUM);
  impl_mode_parser->op_name_select_impl_mode_map_["conv2d_test_set_impl_mode"] = "high_performance";
  tbe_info_assembler_ptr_->SetOpImplMode(AI_CORE_NAME, conv, op_info);
  (void)ge::AttrUtils::GetInt(conv, OP_IMPL_MODE_ENUM, op_impl_mode_num);
  EXPECT_EQ(op_impl_mode_num, 0x2);

  (void)conv->DelAttr(OP_IMPL_MODE_ENUM);
  impl_mode_parser->op_name_select_impl_mode_map_["conv2d_test_set_impl_mode"] = "high_precision";
  tbe_info_assembler_ptr_->SetOpImplMode(AI_CORE_NAME, conv, op_info);
  (void)ge::AttrUtils::GetInt(conv, OP_IMPL_MODE_ENUM, op_impl_mode_num);
  EXPECT_EQ(op_impl_mode_num, 0x4);

  (void)conv->DelAttr(OP_IMPL_MODE_ENUM);
  impl_mode_parser->op_name_select_impl_mode_map_["conv2d_test_set_impl_mode"] = "super_performance";
  tbe_info_assembler_ptr_->SetOpImplMode(AI_CORE_NAME, conv, op_info);
  (void)ge::AttrUtils::GetInt(conv, OP_IMPL_MODE_ENUM, op_impl_mode_num);
  EXPECT_EQ(op_impl_mode_num, 0x8);

  (void)conv->DelAttr(OP_IMPL_MODE_ENUM);
  impl_mode_parser->op_name_select_impl_mode_map_["conv2d_test_set_impl_mode"] = "support_out_of_bound_index";
  tbe_info_assembler_ptr_->SetOpImplMode(AI_CORE_NAME, conv, op_info);
  (void)ge::AttrUtils::GetInt(conv, OP_IMPL_MODE_ENUM, op_impl_mode_num);
  EXPECT_EQ(op_impl_mode_num, 0x10);

  (void)conv->DelAttr(OP_IMPL_MODE_ENUM);
  impl_mode_parser->op_name_select_impl_mode_map_["conv2d_test_set_impl_mode"] = "enable_hi_float_32_execution";
  tbe_info_assembler_ptr_->SetOpImplMode(AI_CORE_NAME, conv, op_info);
  (void)ge::AttrUtils::GetInt(conv, OP_IMPL_MODE_ENUM, op_impl_mode_num);
  EXPECT_EQ(op_impl_mode_num, 0x40);
  op_impl_type = op_info.GetOpImplMode();
  EXPECT_EQ(op_impl_type, "enable_hi_float_32_execution");

  tbe_info_assembler_ptr_->SetOpImplMode(AI_CORE_NAME, conv_invalid, op_info);
  op_impl_mode_num = 0;
  (void)ge::AttrUtils::GetInt(conv_invalid, OP_IMPL_MODE_ENUM, op_impl_mode_num);
  EXPECT_EQ(op_impl_mode_num, 0);
}

TEST_F(STestTbeTimeEstimator, test_set_impl_mode_enum_not_equal_zero) {
  tbe_info_assembler_ptr_->Initialize();
  OpDescPtr conv = std::make_shared<OpDesc>("conv2d_test_set_impl_mode", fe::CONV2D);
  (void)ge::AttrUtils::SetInt(conv, OP_CUSTOM_IMPL_MODE_ENUM, 1);
  te::TbeOpInfo op_info = te::TbeOpInfo("", "", "", "");
  tbe_info_assembler_ptr_->SetOpImplMode(AI_CORE_NAME, conv, op_info);
  int64_t op_impl_mode_num = 0;
  (void)ge::AttrUtils::GetInt(conv, OP_IMPL_MODE_ENUM, op_impl_mode_num);
  EXPECT_EQ(op_impl_mode_num, 0);
}
