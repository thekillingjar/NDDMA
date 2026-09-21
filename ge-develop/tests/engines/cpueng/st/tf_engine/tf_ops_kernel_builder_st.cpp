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
#include "tf_kernel_builder/tf_ops_kernel_builder.h"
#include "tf_kernel_builder/tf_kernel_builder.h"
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

TEST(TfOpsKernelBuilder, Initialize_SUCCESS)
{
    map<string, string> options;
    options[SOC_VERSION] = "Ascend910";
    AicpuTfOpsKernelBuilder tfOpsBuiler;
    ASSERT_EQ(tfOpsBuiler.Initialize(options), SUCCESS);
}

TEST(TfOpsKernelBuilder, CalcOpRunningParam_SUCCESS)
{
    map<string, string> options;
    options[SOC_VERSION] = "Ascend910";
    AicpuTfOpsKernelBuilder tfOpsBuiler;
    ASSERT_EQ(tfOpsBuiler.Initialize(options), SUCCESS);
    OpDescPtr opDescPtr = make_shared<OpDesc>("Add","Add");
    vector<int64_t> tensorShape = {1,1,3,1};
    GeTensorDesc tensor1(GeShape(tensorShape), FORMAT_NCHW, DT_INT32);
    opDescPtr->AddInputDesc("x1", tensor1);
    opDescPtr->AddInputDesc("x2", tensor1);
    opDescPtr->AddOutputDesc("y", tensor1);
    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_graph");
    ASSERT_EQ(tfOpsBuiler.CalcOpRunningParam(*(graphPtr->AddNode(opDescPtr))), SUCCESS);
}

TEST(TfOpsKernelBuilder, CalcOpRunningParamWithWorkspaceSize_SUCCESS)
{
    map<string, string> options;
    options[SOC_VERSION] = "Ascend910";
    AicpuTfOpsKernelBuilder tfOpsBuiler;
    ASSERT_EQ(tfOpsBuiler.Initialize(options), SUCCESS);

    OpDescPtr opDescPtr = make_shared<OpDesc>("Add","Add");
    vector<int64_t> tensorShape = {1,1,3,1};
    GeTensorDesc tensor1(GeShape(tensorShape), FORMAT_NCHW, DT_INT32);
    opDescPtr->AddInputDesc("x1", tensor1);
    opDescPtr->AddInputDesc("x2", tensor1);
    opDescPtr->AddOutputDesc("y", tensor1);

    vector<int64_t> workspaceBase = {int64_t(457568634557)};
    opDescPtr->SetWorkspace(workspaceBase);
    opDescPtr->SetWorkspaceBytes({10000});
    vector<int64_t> inputOffset = {};
    vector<int64_t> outputOffset = {};
    inputOffset.push_back(568679);
    inputOffset.push_back(56868);
    outputOffset.push_back(457568);
    opDescPtr->SetInputOffset(inputOffset);
    opDescPtr->SetOutputOffset(outputOffset);
    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_graph");

    ASSERT_EQ(tfOpsBuiler.CalcOpRunningParam(*(graphPtr->AddNode(opDescPtr))), SUCCESS);
}

TEST(TfOpsKernelBuilder, CalcOpRunningParamFMK_SUCCESS)
{
    map<string, string> options;
    options[SOC_VERSION] = "Ascend910";
    AicpuTfOpsKernelBuilder tfOpsBuiler;
    ASSERT_EQ(tfOpsBuiler.Initialize(options), SUCCESS);

    OpDescPtr opDescPtr = make_shared<OpDesc>("Add","Add");
    vector<int64_t> tensorShape = {1,1,3,1};
    GeTensorDesc tensor1(GeShape(tensorShape), FORMAT_NCHW, DT_INT32);
    opDescPtr->AddInputDesc("x1", tensor1);
    opDescPtr->AddInputDesc("x2", tensor1);
    opDescPtr->AddOutputDesc("y", tensor1);
    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_graph");
    ge::OpDescUtilsEx::SetType(opDescPtr, "FrameworkOp");
    AttrUtils::SetStr(opDescPtr, "original_type", "Add");
    ASSERT_EQ(tfOpsBuiler.CalcOpRunningParam(*(graphPtr->AddNode(opDescPtr))), SUCCESS);
}

TEST(TfOpsKernelBuilder, GenerateTask_SUCCESS)
{
    map<string, string> options;
    options[SOC_VERSION] = "Ascend910";
    AicpuTfOpsKernelBuilder tfOpsBuiler;
    ASSERT_EQ(tfOpsBuiler.Initialize(options), SUCCESS);

    OpDescPtr opDescPtr = make_shared<OpDesc>("Add","Add");
    vector<int64_t> tensorShape = {1,1,3,1};
    GeTensorDesc tensor1(GeShape(tensorShape), FORMAT_NCHW, DT_INT32);
    opDescPtr->AddInputDesc("x1", tensor1);
    opDescPtr->AddInputDesc("x2", tensor1);
    opDescPtr->AddOutputDesc("y", tensor1);
    AttrUtils::SetInt(opDescPtr, "_unknown_shape_type", 1);
    AttrUtils::SetInt(opDescPtr, "topic_type", 1);
    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_graph");
    Node *node = graphPtr->AddNode(opDescPtr).get();
    ASSERT_EQ(tfOpsBuiler.CalcOpRunningParam(*node), SUCCESS);
    RunContext context = CreateContext();
    vector<domi::TaskDef> tasks;
    ASSERT_EQ(tfOpsBuiler.GenerateTask(*node, context, tasks), SUCCESS);
    ASSERT_EQ(tfOpsBuiler.UpdateTask(*node, tasks), SUCCESS);
    DestroyContext(context);
    printf("end===================\n");
}

TEST(TfOpsKernelBuilder, GenerateTaskFMK_SUCCESS)
{
    map<string, string> options;
    options[SOC_VERSION] = "Ascend910";
    AicpuTfOpsKernelBuilder tfOpsBuiler;
    ASSERT_EQ(tfOpsBuiler.Initialize(options), SUCCESS);

    OpDescPtr opDescPtr = make_shared<OpDesc>("Add","Add");
    vector<int64_t> tensorShape = {1,1,3,1};
    GeTensorDesc tensor1(GeShape(tensorShape), FORMAT_NCHW, DT_INT32);
    opDescPtr->AddInputDesc("x1", tensor1);
    opDescPtr->AddInputDesc("x2", tensor1);
    opDescPtr->AddOutputDesc("y", tensor1);
    AttrUtils::SetInt(opDescPtr, "_unknown_shape_type", 1);
    AttrUtils::SetInt(opDescPtr, "topic_type", 1);
    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_graph");
    ge::OpDescUtilsEx::SetType(opDescPtr, "FrameworkOp");
    AttrUtils::SetStr(opDescPtr, "original_type", "Add");
    Node *node = graphPtr->AddNode(opDescPtr).get();
    ASSERT_EQ(tfOpsBuiler.CalcOpRunningParam(*node), SUCCESS);
    RunContext context = CreateContext();
    vector<domi::TaskDef> tasks;
    ASSERT_EQ(tfOpsBuiler.GenerateTask(*node, context, tasks), SUCCESS);
    DestroyContext(context);
    printf("end===================\n");
}

TEST(TfOpsKernelBuilder, GenSingleOpRunTask_SUCCESS)
{
    map<string, string> options;
    options[SOC_VERSION] = "Ascend910";
    AicpuTfOpsKernelBuilder tfOpsBuiler;
    ASSERT_EQ(tfOpsBuiler.Initialize(options), SUCCESS);

    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_graph");
    OpDescPtr opDescPtr = make_shared<OpDesc>("Add","Add");
    NodePtr nodePtr = make_shared<Node>(opDescPtr, graphPtr);
    vector<int64_t> tensorShape = {1,1,3,1};
    GeTensorDesc tensor1(GeShape(tensorShape), FORMAT_NCHW, DT_INT32);
    opDescPtr->AddInputDesc("x1", tensor1);
    opDescPtr->AddInputDesc("x2", tensor1);
    opDescPtr->AddOutputDesc("y", tensor1);
    nodePtr->Init();

    vector<int64_t> workspaceBase = {int64_t(457568634557)};
    opDescPtr->SetWorkspace(workspaceBase);
    opDescPtr->SetWorkspaceBytes({10000});
    vector<int64_t> inputOffset = {};
    vector<int64_t> outputOffset = {};
    inputOffset.push_back(568679);
    inputOffset.push_back(56868);
    outputOffset.push_back(457568);
    opDescPtr->SetInputOffset(inputOffset);
    opDescPtr->SetOutputOffset(outputOffset);

    STR_FWK_OP_KERNEL task;
    std::string task_info;
    ASSERT_EQ(SUCCESS, tfOpsBuiler.GenSingleOpRunTask(nodePtr, task, task_info));
}

TEST(TfOpsKernelBuilder, GenSingleOpRunTaskFunctionDef_SUCCESS)
{
    map<string, string> options;
    options[SOC_VERSION] = "Ascend910";
    AicpuTfOpsKernelBuilder tfOpsBuiler;
    ASSERT_EQ(tfOpsBuiler.Initialize(options), SUCCESS);

    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_graph");
    OpDescPtr opDescPtr = make_shared<OpDesc>("Add","Add");
    NodePtr nodePtr = make_shared<Node>(opDescPtr, graphPtr);
    vector<int64_t> tensorShape = {1,1,3,1};
    GeTensorDesc tensor1(GeShape(tensorShape), FORMAT_NCHW, DT_INT32);
    opDescPtr->AddInputDesc("x1", tensor1);
    opDescPtr->AddInputDesc("x2", tensor1);
    opDescPtr->AddOutputDesc("y", tensor1);
    nodePtr->Init();
    vector<int64_t> workspaceBase = {int64_t(457568634557)};
    opDescPtr->SetWorkspace(workspaceBase);
    opDescPtr->SetWorkspaceBytes({10000});
    string funcDefStr = "AddOpFunctionDef";
    const uint8_t *funcDefData = reinterpret_cast<const uint8_t*>(funcDefStr.data());
    AttrUtils::SetZeroCopyBytes(*(opDescPtr.get()), "func_def", Buffer::CopyFrom(funcDefData, funcDefStr.length()));
    vector<int64_t> inputOffset = {};
    vector<int64_t> outputOffset = {};
    inputOffset.push_back(568679);
    inputOffset.push_back(56868);
    outputOffset.push_back(457568);
    opDescPtr->SetInputOffset(inputOffset);
    opDescPtr->SetOutputOffset(outputOffset);

    STR_FWK_OP_KERNEL task;
    std::string task_info;
    ASSERT_EQ(SUCCESS, tfOpsBuiler.GenSingleOpRunTask(nodePtr, task, task_info));
}

TEST(TfOpsKernelBuilder, GenMemCopyTask_SUCCESS)
{
    STR_FWK_OP_KERNEL task;
    map<string, string> options;
    options[SOC_VERSION] = "Ascend910";
    AicpuTfOpsKernelBuilder tfOpsBuiler;
    ASSERT_EQ(tfOpsBuiler.Initialize(options), SUCCESS);

    std::string task_info;
    ASSERT_EQ(SUCCESS, tfOpsBuiler.GenMemCopyTask(2, task, task_info));
}
