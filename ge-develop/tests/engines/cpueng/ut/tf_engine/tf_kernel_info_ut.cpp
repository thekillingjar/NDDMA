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

#define protected public
#define private public
#include "tf_engine/engine/tf_engine.h"
#include "tf_kernel_info/tf_kernel_info.h"
#undef private
#undef protected

#include "util/tf_util.h"
#include "config/config_file.h"
#include "stub.h"

#include "util/util_stub.h"
#include "ge/ge_api_types.h"
#include "config/ops_json_file.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/op_desc_utils_ex.h"
#include "runtime/context.h"
#include "runtime/stream.h"
#include "runtime/rt_model.h"
#include "runtime/kernel.h"
#include "runtime/mem.h"

using namespace aicpu;
using namespace ge;
using namespace std;


TEST(TfKernelInfo, InitializeReadJson_FAIL)
{
    map<string, string> options;
    options[SOC_VERSION] = "Ascend910";
    ASSERT_EQ(Initialize(options), SUCCESS);

    string kernelConfig;
    ASSERT_EQ(ConfigFile::GetInstance().GetValue("DNN_VM_AICPUOpsKernel", kernelConfig), true);

    ASSERT_EQ(kernelConfig, "TFKernel");
    string optimizerConfig;
    ASSERT_EQ(ConfigFile::GetInstance().GetValue("DNN_VM_AICPUGraphOptimizer", optimizerConfig), true);
    ASSERT_EQ(optimizerConfig, "TFOptimizer");
    map<string, OpsKernelInfoStorePtr> opsKernelInfoStores;
    GetOpsKernelInfoStores(opsKernelInfoStores);
    ASSERT_NE(opsKernelInfoStores["aicpu_tf_kernel"], nullptr);
}

TEST(TfKernelInfo, MapConfigReadJson_FAIL)
{
    map<string, string> options;
    options[SOC_VERSION] = "Ascend910";
    ASSERT_EQ(Initialize(options), SUCCESS);
    string kernelConfig;
    ASSERT_EQ(ConfigFile::GetInstance().GetValue("DNN_VM_AICPUOpsKernel", kernelConfig), true);
    ASSERT_EQ(kernelConfig, "TFKernel");
    string optimizerConfig;
    ASSERT_EQ(ConfigFile::GetInstance().GetValue("DNN_VM_AICPUGraphOptimizer", optimizerConfig), true);
    ASSERT_EQ(optimizerConfig, "TFOptimizer");
    map<string, OpsKernelInfoStorePtr> opsKernelInfoStores;
    GetOpsKernelInfoStores(opsKernelInfoStores);
    aicpu::State state(ge::FAILED);
    ASSERT_NE(opsKernelInfoStores["aicpu_tf_kernel"], nullptr);
}

TEST(TfKernelInfo, Initialize_SUCCESS)
{
    map<string, string> options;
    options[SOC_VERSION] = "Ascend910";
    ASSERT_EQ(Initialize(options), SUCCESS);
    string kernelConfig;
    ASSERT_EQ(ConfigFile::GetInstance().GetValue("DNN_VM_AICPUOpsKernel", kernelConfig), true);
    ASSERT_EQ(kernelConfig, "TFKernel");
    string optimizerConfig;
    ASSERT_EQ(ConfigFile::GetInstance().GetValue("DNN_VM_AICPUGraphOptimizer", optimizerConfig), true);
    ASSERT_EQ(optimizerConfig, "TFOptimizer");
    map<string, OpsKernelInfoStorePtr> opsKernelInfoStores;
    GetOpsKernelInfoStores(opsKernelInfoStores);
    ASSERT_NE(opsKernelInfoStores["aicpu_tf_kernel"], nullptr);
    ASSERT_EQ(opsKernelInfoStores["aicpu_tf_kernel"]->Initialize(options), SUCCESS);
}

TEST(TfKernelInfo, GetValue_FAIL)
{
    map<string, string> options;
    options[SOC_VERSION] = "Ascend910";
    ASSERT_EQ(Initialize(options), SUCCESS);
    string kernelConfig;
    ASSERT_EQ(ConfigFile::GetInstance().GetValue("DNN_VM_AICPUOps", kernelConfig), false);
}

TEST(TfKernelInfo, EmptyOption_FAIL)
{
    map<string, string> options;
    ASSERT_EQ(Initialize(options), SUCCESS);
    string kernelConfig;
    ASSERT_EQ(ConfigFile::GetInstance().GetValue("DNN_VM_AICPUOpsKernel", kernelConfig), true);
    ASSERT_EQ(kernelConfig, "TFKernel");
    string optimizerConfig;
    ASSERT_EQ(ConfigFile::GetInstance().GetValue("DNN_VM_AICPUGraphOptimizer", optimizerConfig), true);
    ASSERT_EQ(optimizerConfig, "TFOptimizer");
    map<string, OpsKernelInfoStorePtr> opsKernelInfoStores;
    GetOpsKernelInfoStores(opsKernelInfoStores);
    ASSERT_NE(opsKernelInfoStores["aicpu_tf_kernel"], nullptr);
    ASSERT_EQ(opsKernelInfoStores["aicpu_tf_kernel"]->Initialize(options), INPUT_PARAM_VALID);
}

TEST(TfKernelInfo, GetAllOpsKernelInfo_SUCCESS)
{
    map<string, OpsKernelInfoStorePtr> opsKernelInfoStores;
    GetOpsKernelInfoStores(opsKernelInfoStores);
    map<string, OpInfo> infos;
    ASSERT_EQ(infos.size(), 0);
    opsKernelInfoStores["aicpu_tf_kernel"]->GetAllOpsKernelInfo(infos);
    ASSERT_NE(infos.size(), 0);
}

TEST(TfKernelInfo, CheckSupported_SUCCESS)
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
    ASSERT_EQ(opsKernelInfoStores["aicpu_tf_kernel"]->CheckSupported(opDescPtr, unSupportedReason), true);
    ASSERT_EQ(unSupportedReason, "");
}

TEST(TfKernelInfo, CheckSupportedEmptyOp_FAIL)
{
    map<string, OpsKernelInfoStorePtr> opsKernelInfoStores;
    GetOpsKernelInfoStores(opsKernelInfoStores);

    OpDescPtr opDescPtr = nullptr;
    string unSupportedReason;
    ASSERT_EQ(opsKernelInfoStores["aicpu_tf_kernel"]->CheckSupported(opDescPtr, unSupportedReason), false);
}

TEST(TfKernelInfo, CheckSupportedEmptyType_FAIL)
{
    map<string, OpsKernelInfoStorePtr> opsKernelInfoStores;
    GetOpsKernelInfoStores(opsKernelInfoStores);
    OpDescPtr opDescPtr = make_shared<OpDesc>("Add", "Add");
    ge::OpDescUtilsEx::SetType(opDescPtr, "");
    string unSupportedReason;
    ASSERT_EQ(opsKernelInfoStores["aicpu_tf_kernel"]->CheckSupported(opDescPtr, unSupportedReason), false);
}

TEST(TfKernelInfo, CheckSupported_FAIL)
{
    map<string, OpsKernelInfoStorePtr> opsKernelInfoStores;
    GetOpsKernelInfoStores(opsKernelInfoStores);

    OpDescPtr opDescPtr = make_shared<OpDesc>("Add", "Ad");
    vector<int64_t> tensorShape = {1, 1, 3, 1};
    GeTensorDesc tensor1(GeShape(tensorShape), FORMAT_NCHW, DT_INT32);
    opDescPtr->AddInputDesc("x", tensor1);
    opDescPtr->AddInputDesc("y", tensor1);
    opDescPtr->AddOutputDesc("z", tensor1);

    string unSupportedReason;
    ASSERT_EQ(opsKernelInfoStores["aicpu_tf_kernel"]->CheckSupported(opDescPtr, unSupportedReason), false);
}

TEST(TfKernelInfo, CheckInputSupport_FAIL)
{
    map<string, OpsKernelInfoStorePtr> opsKernelInfoStores;
    GetOpsKernelInfoStores(opsKernelInfoStores);

    OpDescPtr opDescPtr = make_shared<OpDesc>("Add", "Add");
    vector<int64_t> tensorShape1 = {1, 1, 3, 1};
    vector<int64_t> tensorShape2 = {1, 2, 3};
    vector<int64_t> tensorShape3 = {1};
    GeTensorDesc tensor1(GeShape(tensorShape1), FORMAT_NCHW, DT_UINT64);
    GeTensorDesc tensor2(GeShape(tensorShape2), FORMAT_NCHW, DT_FLOAT);
    GeTensorDesc tensor3(GeShape(tensorShape3), FORMAT_NCHW, DT_FLOAT16);
    opDescPtr->AddInputDesc("x1", tensor1);
    opDescPtr->AddInputDesc("x2", tensor2);
    opDescPtr->AddOutputDesc("y", tensor3);
    string unSupportedReason;
    ASSERT_EQ(opsKernelInfoStores["aicpu_tf_kernel"]->CheckSupported(opDescPtr, unSupportedReason), false);
}

TEST(TfKernelInfo, CheckOutputSupport_FAIL)
{
    map<string, OpsKernelInfoStorePtr> opsKernelInfoStores;
    GetOpsKernelInfoStores(opsKernelInfoStores);

    OpDescPtr opDescPtr = make_shared<OpDesc>("Add", "Add");
    vector<int64_t> tensorShape1 = {1, 1, 3, 1};
    vector<int64_t> tensorShape2 = {1, 2, 3};
    vector<int64_t> tensorShape3 = {1};
    GeTensorDesc tensor1(GeShape(tensorShape1), FORMAT_NCHW, DT_FLOAT16);
    GeTensorDesc tensor2(GeShape(tensorShape2), FORMAT_NCHW, DT_FLOAT16);
    GeTensorDesc tensor3(GeShape(tensorShape3), FORMAT_NCHW, DT_UINT16);
    opDescPtr->AddInputDesc("x1", tensor1);
    opDescPtr->AddInputDesc("x2", tensor2);
    opDescPtr->AddOutputDesc("y", tensor3);
    string unSupportedReason;
    ASSERT_EQ(opsKernelInfoStores["aicpu_tf_kernel"]->CheckSupported(opDescPtr, unSupportedReason), false);
}

TEST(TfKernelInfo, opsFlagCheck_SUCCESS)
{
    map<string, OpsKernelInfoStorePtr> opsKernelInfoStores;
    GetOpsKernelInfoStores(opsKernelInfoStores);

    OpDescPtr opDescPtr = make_shared<OpDesc>("Add", "Add");
    vector<int64_t> tensorShape = {1, 1, 3, 1};
    GeTensorDesc tensor1(GeShape(tensorShape), FORMAT_NCHW, DT_INT32);
    opDescPtr->AddInputDesc("x1", tensor1);
    opDescPtr->AddInputDesc("x2", tensor1);
    opDescPtr->AddOutputDesc("y", tensor1);
    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_graph");
    string opsFlag;
    opsKernelInfoStores["aicpu_tf_kernel"]->opsFlagCheck(*(graphPtr->AddNode(opDescPtr)), opsFlag);
    ASSERT_EQ(opsFlag, "0");
}

TEST(TfKernelInfo, opsFlagCheck_FAIL)
{
    map<string, OpsKernelInfoStorePtr> opsKernelInfoStores;
    GetOpsKernelInfoStores(opsKernelInfoStores);

    OpDescPtr opDescPtr = make_shared<OpDesc>("Add", "Add");
    vector<int64_t> tensorShape = {1, 1, 3, 1};
    GeTensorDesc tensor1(GeShape(tensorShape), FORMAT_NCHW, DT_INT32);
    opDescPtr->AddInputDesc("x", tensor1);
    opDescPtr->AddInputDesc("y", tensor1);
    opDescPtr->AddOutputDesc("z", tensor1);
    ge::OpDescUtilsEx::SetType(opDescPtr, "");
    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_graph");
    string opsFlag;
    opsKernelInfoStores["aicpu_tf_kernel"]->opsFlagCheck(*(graphPtr->AddNode(opDescPtr)), opsFlag);
    ASSERT_EQ(opsFlag, "");
}

TEST(TfKernelInfo, CompileOp_SUCCESS)
{
    map<string, string> options;
    options[SOC_VERSION] = "Ascend910";
    TfKernelInfo tfKernelInfo;
    ASSERT_EQ(tfKernelInfo.Initialize(options), SUCCESS);

    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_graph");
    vector<int64_t> tensorShape = {2, 2, 3, 4};
    GeTensorDesc tensor1(GeShape(tensorShape), FORMAT_NCHW, DT_INT32);
    tensor1.SetOriginFormat(FORMAT_NCHW);
    tensor1.SetOriginShape(GeShape(tensorShape));

    OpDescPtr Op1DescPtr = make_shared<OpDesc>("add1", "Add");
    Op1DescPtr->AddOutputDesc("z", tensor1);
    auto node1 = graphPtr->AddNode(Op1DescPtr);

    OpDescPtr Op2DescPtr = make_shared<OpDesc>("add2", "Add");
    Op2DescPtr->AddOutputDesc("z", tensor1);
    NodePtr node2 = make_shared<Node>(Op2DescPtr, graphPtr);
    node2->Init();
    graphPtr->AddNode(node2);
    node2->AddLinkFrom(node1);

    OpDescPtr Op3DescPtr = make_shared<OpDesc>("add3", "Add");
    Op3DescPtr->AddOutputDesc("z", tensor1);
    NodePtr node3 = make_shared<Node>(Op3DescPtr, graphPtr);
    graphPtr->AddNode(node3);
    node3->AddLinkFrom(node2);

    std::vector<ge::NodePtr> nodeList;
    nodeList.push_back(node2);
    ASSERT_EQ(tfKernelInfo.CompileOp(node1), SUCCESS);
}

TEST(TfKernelInfo, NewOpp_Initialize_SUCCESS)
{
  char *build_path = getenv("BUILD_PATH");
  std::string opp_path = std::string(build_path) + "tests/engines/cpueng/stub/config";
  setenv("ASCEND_OPP_PATH", opp_path.c_str(), 1);
  map<string, string> options;
  options[SOC_VERSION] = "Ascend910";
  ASSERT_EQ(Initialize(options), SUCCESS);
  string kernelConfig;
  ASSERT_EQ(ConfigFile::GetInstance().GetValue("DNN_VM_AICPUOpsKernel", kernelConfig), true);
  ASSERT_EQ(kernelConfig, "TFKernel");
  string optimizerConfig;
  ASSERT_EQ(ConfigFile::GetInstance().GetValue("DNN_VM_AICPUGraphOptimizer", optimizerConfig), true);
  ASSERT_EQ(optimizerConfig, "TFOptimizer");
  map<string, OpsKernelInfoStorePtr> opsKernelInfoStores;
  GetOpsKernelInfoStores(opsKernelInfoStores);
  ASSERT_NE(opsKernelInfoStores["aicpu_tf_kernel"], nullptr);
  ASSERT_EQ(opsKernelInfoStores["aicpu_tf_kernel"]->Initialize(options), SUCCESS);
  unsetenv("ASCEND_OPP_PATH");
}

TEST(TfKernelInfo, CheckSupported_StridedSliceGrad1)
{
    OpDescPtr opDescPtr = make_shared<OpDesc>("StridedSliceGrad", "StridedSliceGrad");
    vector<int64_t> tensorShape = {1, 1, 3, 1};
    GeTensorDesc tensor1(GeShape(tensorShape), FORMAT_NCHW, DT_INT32);
    GeTensorDesc tensor2(GeShape(tensorShape), FORMAT_NCHW, DT_INT64);
    opDescPtr->AddInputDesc("shape", tensor2);
    opDescPtr->AddInputDesc("begin", tensor1);
    opDescPtr->AddInputDesc("end", tensor1);
    opDescPtr->AddInputDesc("strides", tensor1);
    opDescPtr->AddInputDesc("dy", tensor2);
    opDescPtr->AddOutputDesc("output", tensor2);
    string unSupportedReason;
    AicpuOpsKernelInfoStore aicpuOpsKernelInfoStore("DNN_VM_AICPU");
    ASSERT_EQ(
        aicpuOpsKernelInfoStore.CheckStridedSliceGradSupported(opDescPtr, unSupportedReason),
        true);
}

TEST(TfKernelInfo, CheckSupported_StridedSliceGrad2)
{
    OpDescPtr opDescPtr = make_shared<OpDesc>("StridedSliceGrad", "StridedSliceGrad");
    vector<int64_t> tensorShape = {1, 1, 3, 1};
    GeTensorDesc tensor1(GeShape(tensorShape), FORMAT_NCHW, DT_INT32);
    GeTensorDesc tensor2(GeShape(tensorShape), FORMAT_NCHW, DT_INT64);
    opDescPtr->AddInputDesc("shape", tensor2);
    opDescPtr->AddInputDesc("begin", tensor1);
    opDescPtr->AddInputDesc("end", tensor2);
    opDescPtr->AddInputDesc("strides", tensor1);
    opDescPtr->AddInputDesc("dy", tensor2);
    opDescPtr->AddOutputDesc("output", tensor2);
    string unSupportedReason;
    AicpuOpsKernelInfoStore aicpuOpsKernelInfoStore("DNN_VM_AICPU");
    ASSERT_EQ(
        aicpuOpsKernelInfoStore.CheckStridedSliceGradSupported(opDescPtr, unSupportedReason),
        false);
}
