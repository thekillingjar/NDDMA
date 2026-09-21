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
#include "aicpu_engine/engine/aicpu_engine.h"
#include "aicpu_engine/kernel_info/aicpu_cust_kernel_info.h"

#include "util/util.h"
#include "config/config_file.h"
#include "stub.h"

#include "ge/ge_api_types.h"

using namespace aicpu;
using namespace ge;
using namespace std;

TEST(AicpuKernelInfo, Initialize_SUCCESS)
{
    map<string, string> options;
    options[SOC_VERSION] = "Ascend910";
    ASSERT_EQ(Initialize(options), SUCCESS);
    string kernelConfig;
    ASSERT_EQ(ConfigFile::GetInstance().GetValue("DNN_VM_AICPU_ASCENDOpsKernel", kernelConfig), true);
    ASSERT_EQ(kernelConfig, "CUSTAICPUKernel,AICPUKernel");
    string optimizerConfig;
    ASSERT_EQ(ConfigFile::GetInstance().GetValue("DNN_VM_AICPU_ASCENDGraphOptimizer", optimizerConfig), true);
    ASSERT_EQ(optimizerConfig, "AICPUOptimizer");
    map<string, OpsKernelInfoStorePtr> opsKernelInfoStores;
    GetOpsKernelInfoStores(opsKernelInfoStores);
    ASSERT_NE(opsKernelInfoStores["aicpu_ascend_kernel"], nullptr);
    ASSERT_EQ(opsKernelInfoStores["aicpu_ascend_kernel"]->Initialize(options), SUCCESS);
}

TEST(AicpuKernelInfo, GetAllOpsKernelInfo_SUCCESS)
{
    map<string, OpsKernelInfoStorePtr> opsKernelInfoStores;
    GetOpsKernelInfoStores(opsKernelInfoStores);
    map<string, OpInfo> infos;
    ASSERT_EQ(infos.size(), 0);
    opsKernelInfoStores["aicpu_ascend_kernel"]->GetAllOpsKernelInfo(infos);
    ASSERT_NE(infos.size(), 0);
}

TEST(AicpuKernelInfo, CheckSupported_SUCCESS)
{
    map<string, OpsKernelInfoStorePtr> opsKernelInfoStores;
    GetOpsKernelInfoStores(opsKernelInfoStores);

    OpDescPtr opDescPtr = make_shared<OpDesc>("Add", "Add");
    vector<int64_t> tensorShape = {1, 1, 3, 1};
    GeTensorDesc tensor1(GeShape(tensorShape), FORMAT_NCHW, DT_INT32);
    opDescPtr->AddInputDesc("x", tensor1);
    opDescPtr->AddInputDesc("y", tensor1);
    opDescPtr->AddOutputDesc("z", tensor1);
    string unSupportedReason;
    ASSERT_EQ(opsKernelInfoStores["aicpu_ascend_kernel"]->CheckSupported(opDescPtr, unSupportedReason), true);
    ASSERT_EQ(unSupportedReason, "");
}

TEST(AicpuKernelInfo, CheckSupportedTransData_SUCCESS)
{
    map<string, OpsKernelInfoStorePtr> opsKernelInfoStores;
    GetOpsKernelInfoStores(opsKernelInfoStores);

    OpDescPtr opDescPtr = make_shared<OpDesc>("TransData", "TransData");
    vector<int64_t> tensorShape1 = {1, 1, 4, 68};
    vector<int64_t> tensorShape2 = {10, 1, 16, 16};
    GeTensorDesc tensor1(GeShape(tensorShape1), FORMAT_HWCN, DT_INT32);
    GeTensorDesc tensor2(GeShape(tensorShape2), FORMAT_FRACTAL_Z, DT_INT32);
    opDescPtr->AddInputDesc("src", tensor1);
    opDescPtr->AddOutputDesc("dst", tensor2);
    AttrUtils::SetInt(opDescPtr, "groups", 16);
    string unSupportedReason;
    ASSERT_EQ(opsKernelInfoStores["aicpu_ascend_kernel"]->CheckSupported(opDescPtr, unSupportedReason), true);
    ASSERT_EQ(unSupportedReason, "");
}

TEST(AicpuKernelInfo, CheckSupportedTransData_failed_group)
{
    OpDescPtr opDescPtr = make_shared<OpDesc>("TransData", "TransData");
    vector<int64_t> tensorShape1 = {1, 1, 4, 68};
    vector<int64_t> tensorShape2 = {10, 1, 16, 16};
    GeTensorDesc tensor1(GeShape(tensorShape1), FORMAT_HWCN, DT_INT32);
    GeTensorDesc tensor2(GeShape(tensorShape2), FORMAT_FRACTAL_Z, DT_INT32);
    opDescPtr->AddInputDesc("src", tensor1);
    opDescPtr->AddOutputDesc("dst", tensor2);
    string unSupportedReason;
    const map<string, string> data_types;
    AicpuOpsKernelInfoStore aicpuOpsKernelInfoStore("DNN_VM_AICPU_ASCEND");
    ASSERT_EQ(
        aicpuOpsKernelInfoStore.CheckTransDataSupported(opDescPtr, data_types, unSupportedReason),
        false);
}

TEST(AicpuKernelInfo, CheckSupportedTransData_fail_01)
{
    OpDescPtr opDescPtr = make_shared<OpDesc>("TransData", "TransData");
    vector<int64_t> tensorShape1 = {1, 1, 4, 68};
    vector<int64_t> tensorShape2 = {10, 1, 16, 16};
    GeTensorDesc tensor1(GeShape(tensorShape1), FORMAT_ND, DT_INT32);
    GeTensorDesc tensor2(GeShape(tensorShape2), FORMAT_FRACTAL_NZ_C0_16, DT_INT32);
    opDescPtr->AddInputDesc("src", tensor1);
    opDescPtr->AddOutputDesc("dst", tensor2);
    AttrUtils::SetInt(opDescPtr, "groups", 2);
    string unSupportedReason;
    const map<string, string> data_types;
    AicpuOpsKernelInfoStore aicpuOpsKernelInfoStore("DNN_VM_HOST_CPU");
    ASSERT_EQ(
        aicpuOpsKernelInfoStore.CheckTransDataSupported(opDescPtr, data_types, unSupportedReason),
        false);
}

TEST(AicpuKernelInfo, CheckSupportedTransData_fail_02)
{
    OpDescPtr opDescPtr = make_shared<OpDesc>("TransData", "TransData");
    vector<int64_t> tensorShape1 = {1, 1, 4, 68};
    vector<int64_t> tensorShape2 = {10, 1, 16, 16};
    GeTensorDesc tensor1(GeShape(tensorShape1), FORMAT_ND, DT_INT32);
    GeTensorDesc tensor2(GeShape(tensorShape2), FORMAT_FRACTAL_NZ_C0_32, DT_INT32);
    opDescPtr->AddInputDesc("src", tensor1);
    opDescPtr->AddOutputDesc("dst", tensor2);
    AttrUtils::SetInt(opDescPtr, "groups", 2);
    string unSupportedReason;
    const map<string, string> data_types;
    AicpuOpsKernelInfoStore aicpuOpsKernelInfoStore("DNN_VM_HOST_CPU");
    ASSERT_EQ(
        aicpuOpsKernelInfoStore.CheckTransDataSupported(opDescPtr, data_types, unSupportedReason),
        false);
}

TEST(AicpuKernelInfo, CheckSupportedTransData_failed_group_num)
{
    OpDescPtr opDescPtr = make_shared<OpDesc>("TransData", "TransData");
    vector<int64_t> tensorShape1 = {1, 1, 4, 68};
    vector<int64_t> tensorShape2 = {10, 1, 16, 16};
    GeTensorDesc tensor1(GeShape(tensorShape1), FORMAT_HWCN, DT_INT32);
    GeTensorDesc tensor2(GeShape(tensorShape2), FORMAT_FRACTAL_Z, DT_INT32);
    opDescPtr->AddInputDesc("src", tensor1);
    opDescPtr->AddOutputDesc("dst", tensor2);
    AttrUtils::SetInt(opDescPtr, "groups", 1);
    string unSupportedReason;
    const map<string, string> data_types;
    AicpuOpsKernelInfoStore aicpuOpsKernelInfoStore("DNN_VM_AICPU_ASCEND");
    ASSERT_EQ(
        aicpuOpsKernelInfoStore.CheckTransDataSupported(opDescPtr, data_types, unSupportedReason),
        false);
}

TEST(AicpuKernelInfo, CheckSupportedTransData_failed_type)
{
    OpDescPtr opDescPtr = make_shared<OpDesc>("TransData", "TransData");
    vector<int64_t> tensorShape1 = {1, 1, 4, 68};
    vector<int64_t> tensorShape2 = {10, 1, 16, 16};
    GeTensorDesc tensor1(GeShape(tensorShape1), FORMAT_HWCN, DT_FLOAT);
    GeTensorDesc tensor2(GeShape(tensorShape2), FORMAT_FRACTAL_Z, DT_FLOAT);
    opDescPtr->AddInputDesc("src", tensor1);
    opDescPtr->AddOutputDesc("dst", tensor2);
    const map<string, string> data_types = {{"input0", "DT_INT32"}, {"output0", "DT_INT32"}};
    AttrUtils::SetInt(opDescPtr, "groups", 16);
    string unSupportedReason;
    AicpuOpsKernelInfoStore aicpuOpsKernelInfoStore("DNN_VM_AICPU_ASCEND");
    ASSERT_EQ(
        aicpuOpsKernelInfoStore.CheckTransDataSupported(opDescPtr, data_types, unSupportedReason),
        false);
}

TEST(AicpuKernelInfo, CheckSupportedTransData_failed_output_type)
{
    OpDescPtr opDescPtr = make_shared<OpDesc>("TransData", "TransData");
    vector<int64_t> tensorShape1 = {1, 1, 4, 68};
    vector<int64_t> tensorShape2 = {10, 1, 16, 16};
    GeTensorDesc tensor1(GeShape(tensorShape1), FORMAT_HWCN, DT_INT32);
    GeTensorDesc tensor2(GeShape(tensorShape2), FORMAT_FRACTAL_Z, DT_FLOAT);
    opDescPtr->AddInputDesc("src", tensor1);
    opDescPtr->AddOutputDesc("dst", tensor2);
    const map<string, string> data_types = {{"input0", "DT_INT32"}, {"output0", "DT_INT32"}};
    AttrUtils::SetInt(opDescPtr, "groups", 16);
    string unSupportedReason;
    AicpuOpsKernelInfoStore aicpuOpsKernelInfoStore("DNN_VM_AICPU_ASCEND");
    ASSERT_EQ(
        aicpuOpsKernelInfoStore.CheckTransDataSupported(opDescPtr, data_types, unSupportedReason),
        false);
}

TEST(AicpuKernelInfo, CheckSupportedTransData_failed_format)
{
    OpDescPtr opDescPtr = make_shared<OpDesc>("TransData", "TransData");
    vector<int64_t> tensorShape1 = {1, 1, 4, 68};
    vector<int64_t> tensorShape2 = {10, 1, 16, 16};
    GeTensorDesc tensor1(GeShape(tensorShape1), FORMAT_NHWC, DT_INT32);
    GeTensorDesc tensor2(GeShape(tensorShape2), FORMAT_FRACTAL_Z, DT_INT32);
    opDescPtr->AddInputDesc("src", tensor1);
    opDescPtr->AddOutputDesc("dst", tensor2);
    AttrUtils::SetInt(opDescPtr, "groups", 16);
    string unSupportedReason;
    const map<string, string> data_types;
    AicpuOpsKernelInfoStore aicpuOpsKernelInfoStore("DNN_VM_AICPU_ASCEND");
    ASSERT_EQ(
        aicpuOpsKernelInfoStore.CheckTransDataSupported(opDescPtr, data_types, unSupportedReason),
        false);
}

TEST(AicpuKernelInfo, CheckSupportedTransData_failed_format_2)
{
    OpDescPtr opDescPtr = make_shared<OpDesc>("TransData", "TransData");
    vector<int64_t> tensorShape1 = {1, 1, 4, 68};
    vector<int64_t> tensorShape2 = {10, 1, 16, 16};
    GeTensorDesc tensor1(GeShape(tensorShape1), FORMAT_FRACTAL_Z, DT_INT32);
    GeTensorDesc tensor2(GeShape(tensorShape2), FORMAT_FRACTAL_Z, DT_INT32);
    opDescPtr->AddInputDesc("src", tensor1);
    opDescPtr->AddOutputDesc("dst", tensor2);
    AttrUtils::SetInt(opDescPtr, "groups", 1);
    string unSupportedReason;
    const map<string, string> data_types;
    AicpuOpsKernelInfoStore aicpuOpsKernelInfoStore("DNN_VM_AICPU_ASCEND");
    ASSERT_EQ(
        aicpuOpsKernelInfoStore.CheckTransDataSupported(opDescPtr, data_types, unSupportedReason),
        false);
}

TEST(AicpuKernelInfo, opsFlagCheck_SUCCESS)
{
    map<string, OpsKernelInfoStorePtr> opsKernelInfoStores;
    GetOpsKernelInfoStores(opsKernelInfoStores);

    OpDescPtr opDescPtr = make_shared<OpDesc>("Add", "Add");
    vector<int64_t> tensorShape = {1, 1, 3, 1};
    GeTensorDesc tensor1(GeShape(tensorShape), FORMAT_NCHW, DT_INT32);
    opDescPtr->AddInputDesc("x", tensor1);
    opDescPtr->AddInputDesc("y", tensor1);
    opDescPtr->AddOutputDesc("z", tensor1);
    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_graph");
    string opsFlag;
    opsKernelInfoStores["aicpu_ascend_kernel"]->opsFlagCheck(*(graphPtr->AddNode(opDescPtr)), opsFlag);
    ASSERT_EQ(opsFlag, "");
}

TEST(AicpuKernelInfo, CompileOp_SUCCESS)
{
    map<string, string> options;
    options[SOC_VERSION] = "Ascend910";
    map<string, OpsKernelInfoStorePtr> opsKernelInfoStores;
    GetOpsKernelInfoStores(opsKernelInfoStores);
    ASSERT_NE(opsKernelInfoStores["aicpu_ascend_kernel"], nullptr);
    ASSERT_EQ(opsKernelInfoStores["aicpu_ascend_kernel"]->Initialize(options), SUCCESS);

    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_graph");
    OpDescPtr opDescPtr = make_shared<OpDesc>("Add", "Add");
    auto nodePtr = graphPtr->AddNode(opDescPtr);
    vector<int64_t> tensorShape = {1, 1, 3, 1};
    GeTensorDesc tensor1(GeShape(tensorShape), FORMAT_NCHW, DT_INT32);
    opDescPtr->AddInputDesc("x1", tensor1);
    opDescPtr->AddInputDesc("x2", tensor1);
    opDescPtr->AddOutputDesc("y", tensor1);
    nodePtr->Init();

    std::vector<ge::NodePtr> nodeList;
    nodeList.push_back(nodePtr);
    ASSERT_EQ(opsKernelInfoStores["aicpu_ascend_kernel"]->CompileOp(nodeList), SUCCESS);
}

TEST(AicpuKernelInfo, OppNewPath_001)
{
  char *build_path = getenv("BUILD_PATH");
  std::string opp_path = std::string(build_path);
  setenv("ASCEND_OPP_PATH", opp_path.c_str(), 1);

  map<string, string> options;
  options[SOC_VERSION] = "Ascend910";
  ASSERT_EQ(Initialize(options), SUCCESS);
  string kernelConfig;
  ASSERT_EQ(ConfigFile::GetInstance().GetValue("DNN_VM_AICPU_ASCENDOpsKernel", kernelConfig), true);
  ASSERT_EQ(kernelConfig, "CUSTAICPUKernel,AICPUKernel");
  string optimizerConfig;
  ASSERT_EQ(ConfigFile::GetInstance().GetValue("DNN_VM_AICPU_ASCENDGraphOptimizer", optimizerConfig), true);
  ASSERT_EQ(optimizerConfig, "AICPUOptimizer");
  map<string, OpsKernelInfoStorePtr> opsKernelInfoStores;
  GetOpsKernelInfoStores(opsKernelInfoStores);
  ASSERT_NE(opsKernelInfoStores["aicpu_ascend_kernel"], nullptr);
  ASSERT_EQ(opsKernelInfoStores["aicpu_ascend_kernel"]->Initialize(options), SUCCESS);

  unsetenv("ASCEND_OPP_PATH");
}

TEST(AicpuCustKernelInfo, OppNewPath_002)
{
    AicpuCustKernelInfo custKernelInfo;
    char *build_path = getenv("BUILD_PATH");
    std::string path = std::string(build_path) + "tests/engines/cpueng/stub/config/vendors";
    bool ret = custKernelInfo.GetCustJsonFile(path);
    ASSERT_EQ(ret, true);
}

TEST(AicpuCustKernelInfo, BuiltInCustGetJsonFile_SUCCESS)
{
    AicpuCustKernelInfo custKernelInfo;
    char *build_path = getenv("BUILD_PATH");
    std::string path = std::string(build_path) + "tests/engines/cpueng/stub/config/";
    bool ret = custKernelInfo.GetBuiltInCustJsonFile(path);
    ASSERT_EQ(ret, true);
}

TEST(AicpuCustKernelInfo, ReadCustOpInfoFromJsonFile_SUCCESS)
{
    char *build_path = getenv("BUILD_PATH");
    std::string path = "tests/engines/cpueng/stub/config/vendors/test1:tests/engines/cpueng/stub/config/vendors/test2";
    std::string opp_path = std::string(build_path) + path;
    AicpuCustKernelInfo custKernelInfo;
    bool ret = custKernelInfo.ReadCustOpInfoFromJsonFile(opp_path);
    ASSERT_EQ(ret, true);
}
