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
#include "common/sgt_slice_type.h"

using namespace aicpu;
using namespace ge;
using namespace std;

TEST(TfKernelBuilder, Initialize_SUCCESS)
{
    map<string, string> options;
    options[SOC_VERSION] = "Ascend910";
    ASSERT_EQ(Initialize(options), SUCCESS);
    map<string, OpsKernelBuilderPtr> opsKernelBuilders;

    GetOpsKernelBuilderObjs(opsKernelBuilders);
    ASSERT_NE(opsKernelBuilders["aicpu_tf_builder"], nullptr);
    ASSERT_EQ(opsKernelBuilders["aicpu_tf_builder"]->Initialize(options), SUCCESS);
}

TEST(TfKernelBuilder, CalcOpRunningParam_SUCCESS)
{

    map<string, OpsKernelBuilderPtr> opsKernelBuilders;
    GetOpsKernelBuilderObjs(opsKernelBuilders);
    OpDescPtr opDescPtr = make_shared<OpDesc>("Add","Add");
    vector<int64_t> tensorShape = {1,1,3,1};
    GeTensorDesc tensor1(GeShape(tensorShape), FORMAT_NCHW, DT_INT32);
    opDescPtr->AddInputDesc("x1", tensor1);
    opDescPtr->AddInputDesc("x2", tensor1);
    opDescPtr->AddOutputDesc("y", tensor1);
    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_graph");

    ASSERT_EQ(opsKernelBuilders["aicpu_tf_builder"]->CalcOpRunningParam(*(graphPtr->AddNode(opDescPtr))), SUCCESS);
}

TEST(TfKernelBuilder, CalcOpRunningParam_OptionalFromValue_SUCCESS)
{
    map<string, OpsKernelBuilderPtr> opsKernelBuilders;
    GetOpsKernelBuilderObjs(opsKernelBuilders);
    OpDescPtr opDescPtr = make_shared<OpDesc>("OptionalFromValue","OptionalFromValue");
    vector<int64_t> tensorShape = {1,1,3,1};
    GeTensorDesc tensor1(GeShape(tensorShape), FORMAT_NCHW, DT_INT32);
    opDescPtr->AddInputDesc("x1", tensor1);
    opDescPtr->AddInputDesc("x2", tensor1);
    opDescPtr->AddOutputDesc("y", tensor1);
    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_graph");

    ASSERT_EQ(opsKernelBuilders["aicpu_tf_builder"]->CalcOpRunningParam(*(graphPtr->AddNode(opDescPtr))), SUCCESS);
}

TEST(TfKernelBuilder, CalcOpRunningParamFMK_SUCCESS)
{

    map<string, OpsKernelBuilderPtr> opsKernelBuilders;
    GetOpsKernelBuilderObjs(opsKernelBuilders);
    OpDescPtr opDescPtr = make_shared<OpDesc>("Add","Add");
    vector<int64_t> tensorShape = {1,1,3,1};
    GeTensorDesc tensor1(GeShape(tensorShape), FORMAT_NCHW, DT_INT32);
    opDescPtr->AddInputDesc("x1", tensor1);
    opDescPtr->AddInputDesc("x2", tensor1);
    opDescPtr->AddOutputDesc("y", tensor1);
    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_graph");
    ge::OpDescUtilsEx::SetType(opDescPtr, "FrameworkOp");
    AttrUtils::SetStr(opDescPtr, "original_type", "Add");
    ASSERT_EQ(opsKernelBuilders["aicpu_tf_builder"]->CalcOpRunningParam(*(graphPtr->AddNode(opDescPtr))), SUCCESS);
}

TEST(TfKernelBuilder, CalcOpRunningParamWithMaxFftsThreadSize_SUCCESS)
{
    map<string, OpsKernelBuilderPtr> opsKernelBuilders;
    GetOpsKernelBuilderObjs(opsKernelBuilders);

    OpDescPtr opDescPtr = make_shared<OpDesc>("Add", "Add");
    vector<int64_t> tensorShape = {2, 2, 2};
    GeTensorDesc tensor1(GeShape(tensorShape), FORMAT_NCHW, DT_INT32);
    opDescPtr->AddInputDesc("x1", tensor1);
    opDescPtr->AddInputDesc("x2", tensor1);
    opDescPtr->AddOutputDesc("y", tensor1);
    opDescPtr->AddOutputDesc("z", tensor1);
    ffts::ThreadSliceMap thread_slice_map;
    thread_slice_map.thread_scope_id = 1;
    thread_slice_map.slice_instance_num = 2;
    thread_slice_map.thread_mode = true; // ffts plus auto mode
    ffts::ThreadSliceMapPtr slice_ptr = make_shared<ffts::ThreadSliceMap>(thread_slice_map);
    slice_ptr->output_tensor_slice = {{{{0,1},{0,2},{0,2}},{{0,1},{0,2},{0,2}}}, {{{1,2},{0,2},{0,2}},{{1,2},{0,2},{0,2}}}};
    slice_ptr->input_tensor_slice = {{{{0,1},{0,2},{0,2}}},{{{1,2},{0,2},{0,2}}}};
    opDescPtr->SetExtAttr(kAttrNameSgtStruct, slice_ptr);

    AttrUtils::SetInt(opDescPtr, "_unknown_shape_type", 0);
    AttrUtils::SetInt(opDescPtr, kAttrNameThreadScopeId, 1);
    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_ffts_graph");
    Node *node = graphPtr->AddNode(opDescPtr).get();
    ASSERT_EQ(opsKernelBuilders["aicpu_tf_builder"]->CalcOpRunningParam(*node), SUCCESS);
}

TEST(TfKernelBuilder, GenerateTask_SUCCESS)
{
    map<string, OpsKernelBuilderPtr> opsKernelBuilders;
    GetOpsKernelBuilderObjs(opsKernelBuilders);

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
    ASSERT_EQ(opsKernelBuilders["aicpu_tf_builder"]->CalcOpRunningParam(*node), SUCCESS);
    RunContext context = CreateContext();
    vector<domi::TaskDef> tasks;
    ASSERT_EQ(opsKernelBuilders["aicpu_tf_builder"]->GenerateTask(*node, context, tasks), SUCCESS);
    ASSERT_EQ(opsKernelBuilders["aicpu_tf_builder"]->UpdateTask(*node, tasks), SUCCESS);
    DestroyContext(context);
    printf("end===================\n");
}

TEST(TfKernelBuilder, GenerateTask_SUCCESS_FFTS)
{
    map<string, OpsKernelBuilderPtr> opsKernelBuilders;
    GetOpsKernelBuilderObjs(opsKernelBuilders);

    OpDescPtr opDescPtr = make_shared<OpDesc>("Add", "Add");
    vector<int64_t> tensorShape = {1, 1, 3, 1};
    GeTensorDesc tensor1(GeShape(tensorShape), FORMAT_NCHW, DT_INT32);
    opDescPtr->AddInputDesc("x1", tensor1);
    opDescPtr->AddInputDesc("x2", tensor1);
    opDescPtr->AddOutputDesc("y", tensor1);
    ffts::ThreadSliceMap thread_slice_map;
    thread_slice_map.thread_scope_id = 1;
    thread_slice_map.slice_instance_num = 1;
    thread_slice_map.thread_mode = 0;
    ffts::ThreadSliceMapPtr slice_ptr = make_shared<ffts::ThreadSliceMap>(thread_slice_map);
    opDescPtr->SetExtAttr(kAttrNameSgtStruct, slice_ptr);
    AttrUtils::SetInt(opDescPtr, "_unknown_shape_type", 0);
    ge::AttrUtils::SetInt(opDescPtr, kAttrNameThreadScopeId, 1);
    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_graph");
    Node *node = graphPtr->AddNode(opDescPtr).get();
    ASSERT_EQ(opsKernelBuilders["aicpu_tf_builder"]->CalcOpRunningParam(*node), SUCCESS);
    RunContext context = CreateContext();
    vector<domi::TaskDef> tasks;
    ASSERT_EQ(opsKernelBuilders["aicpu_tf_builder"]->GenerateTask(*node, context, tasks), SUCCESS);
    DestroyContext(context);
    printf("ffts task end===================\n");
}

TEST(TfKernelBuilder, GenerateTaskFMK_SUCCESS)
{
    map<string, OpsKernelBuilderPtr> opsKernelBuilders;
    GetOpsKernelBuilderObjs(opsKernelBuilders);

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
    ASSERT_EQ(opsKernelBuilders["aicpu_tf_builder"]->CalcOpRunningParam(*node), SUCCESS);
    RunContext context = CreateContext();
    vector<domi::TaskDef> tasks;
    ASSERT_EQ(opsKernelBuilders["aicpu_tf_builder"]->GenerateTask(*node, context, tasks), SUCCESS);
    DestroyContext(context);
    printf("end===================\n");
}

TEST(TfKernelBuilder, GenSingleOpRunTask_SUCCESS)
{
    map<string, OpsKernelBuilderPtr> opsKernelBuilders;
    GetOpsKernelBuilderObjs(opsKernelBuilders);
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
    ASSERT_EQ(SUCCESS, opsKernelBuilders["aicpu_tf_builder"]->GenSingleOpRunTask(nodePtr, task, task_info));
}

TEST(TfKernelBuilder, GenMemCopyTask_SUCCESS)
{
    STR_FWK_OP_KERNEL task;
    map<string, OpsKernelBuilderPtr> opsKernelBuilders;
    GetOpsKernelBuilderObjs(opsKernelBuilders);
    std::string task_info;
    ASSERT_EQ(SUCCESS, opsKernelBuilders["aicpu_tf_builder"]->GenMemCopyTask(2, task, task_info));
}

TEST(TfKernelBuilder, GenerateTask_UNKNOWSHAPE_SUCCESS)
{
    map<string, OpsKernelBuilderPtr> opsKernelBuilders;
    GetOpsKernelBuilderObjs(opsKernelBuilders);

    OpDescPtr opDescPtr = make_shared<OpDesc>("Test","Test");
    vector<int64_t> tensorShape = {1,1,3,-1};
    GeTensorDesc tensor1(GeShape(tensorShape), FORMAT_NCHW, DT_INT32);
    opDescPtr->AddInputDesc("x", tensor1);
    opDescPtr->AddInputDesc("y", tensor1);
    opDescPtr->AddOutputDesc("z", tensor1);
    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_graph");
    Node *node = graphPtr->AddNode(opDescPtr).get();
    map<string, GraphOptimizerPtr> graphOptimizers;
    GetGraphOptimizerObjs(graphOptimizers);
    map<string, string> options;
    options[SOC_VERSION] = "Ascend910";
    ASSERT_EQ(graphOptimizers["aicpu_tf_optimizer"]->Initialize(options, nullptr), SUCCESS);
    ASSERT_EQ(graphOptimizers["aicpu_tf_optimizer"]->OptimizeFusedGraph(*(graphPtr.get())), SUCCESS);
    ASSERT_EQ(opsKernelBuilders["aicpu_tf_builder"]->CalcOpRunningParam(*node), SUCCESS);
    RunContext context = CreateContext();
    vector<domi::TaskDef> tasks;

    ASSERT_EQ(opsKernelBuilders["aicpu_tf_builder"]->GenerateTask(*node, context, tasks), SUCCESS);
    DestroyContext(context);
    printf("end===================\n");
}

TEST(TfKernelBuilder, CreateNodeDef_FAILED_001)
{
    OpDescPtr opDescPtr = make_shared<OpDesc>("Add","Add");
    ge::OpDescUtilsEx::SetType(opDescPtr, "FunctionOp");
    vector<int64_t> tensorShape = {1,1,3,1};
    GeTensorDesc tensor1(GeShape(tensorShape), FORMAT_NCHW, DT_INT32);
    opDescPtr->AddInputDesc("x1", tensor1);
    opDescPtr->AddInputDesc("x2", tensor1);
    opDescPtr->AddOutputDesc("y", tensor1);
    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_graph");
    TfKernelBuilder *tfKernelBuilder = (TfKernelBuilder *)(TfKernelBuilder::Instance().get());
    ASSERT_EQ(CreateNodeDef(*(graphPtr->AddNode(opDescPtr))), NODE_DEF_NOT_EXIST);
}

TEST(TfKernelBuilder, CreateNodeDef_SUCCESS)
{
    OpDescPtr opDescPtr = make_shared<OpDesc>("Add","Add");
    ge::OpDescUtilsEx::SetType(opDescPtr, "framework_type");
    AttrUtils::SetStr(opDescPtr, "original_type", "Add");
    vector<int64_t> tensorShape = {1,1,3,1};
    GeTensorDesc tensor1(GeShape(tensorShape), FORMAT_NCHW, DT_INT32);
    opDescPtr->AddInputDesc("x1", tensor1);
    opDescPtr->AddInputDesc("x2", tensor1);
    opDescPtr->AddOutputDesc("y", tensor1);
    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_graph");
    TfKernelBuilder *tfKernelBuilder = (TfKernelBuilder *)(TfKernelBuilder::Instance().get());
    ASSERT_EQ(CreateNodeDef(*(graphPtr->AddNode(opDescPtr))), SUCCESS);
}

TEST(TfKernelBuilder, SetUnknownShape_SUCCESS)
{
    map<string, string> options;
    options[SOC_VERSION] = "Ascend910";
    Initialize(options);
    string kernelConfig;
    ConfigFile::GetInstance().GetValue("DNN_VM_AICPUOpsKernel", kernelConfig);
    map<string, OpsKernelInfoStorePtr> opsKernelInfoStores;
    GetOpsKernelInfoStores(opsKernelInfoStores);
    opsKernelInfoStores["aicpu_tf_kernel"]->Initialize(options);

    TfKernelBuilder tfKernelBuiler;
    OpDescPtr opDescPtr = make_shared<OpDesc>("Add","Add");
    vector<int64_t> tensorShape = {1,1,3,1};
    GeTensorDesc tensor1(GeShape(tensorShape), FORMAT_NCHW, DT_INT32);
    opDescPtr->AddInputDesc("x1", tensor1);
    opDescPtr->AddInputDesc("x2", tensor1);
    opDescPtr->AddOutputDesc("y", tensor1);
    AttrUtils::SetInt(opDescPtr, "topic_type", 1);
    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_graph");
    AttrUtils::SetStr(opDescPtr, "original_type", "Add");
    Node *node = graphPtr->AddNode(opDescPtr).get();
    ASSERT_EQ(tfKernelBuiler.CalcOpRunningParam(*node), SUCCESS);
    RunContext context = CreateContext();
    vector<domi::TaskDef> tasks;
    ASSERT_EQ(tfKernelBuiler.GenerateTask(*node, context, tasks), SUCCESS);
    DestroyContext(context);
    printf("end===================\n");
}
