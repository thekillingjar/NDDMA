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
#include "util/util.h"
#include "config/config_file.h"
#include "stub.h"

#include "common/sgt_slice_type.h"
#include "ge/ge_api_types.h"
#include "graph/ge_context.h"
#include "graph/ge_local_context.h"
#include "graph/utils/op_desc_utils_ex.h"

using namespace aicpu;
using namespace ge;
using namespace std;

namespace {
// need fix when update libgraph.so in blue zone
const std::string ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE = "_tensor_no_tiling_mem_type";
}  // namespace

TEST(AicpuKernelBuilder, Initialize_SUCCESS)
{
    map<string, string> options;
    options[SOC_VERSION] = "Ascend910";
    ASSERT_EQ(Initialize(options), SUCCESS);
    map<string, OpsKernelBuilderPtr> opsKernelBuilders;

    GetOpsKernelBuilderObjs(opsKernelBuilders);
    ASSERT_NE(opsKernelBuilders["aicpu_ascend_builder"], nullptr);
    ASSERT_EQ(opsKernelBuilders["aicpu_ascend_builder"]->Initialize(options), SUCCESS);
}

TEST(AicpuKernelBuilder, CalcOpRunningParam_SUCCESS)
{
    map<string, string> options;
    options[SOC_VERSION] = "Ascend910";
    ASSERT_EQ(Initialize(options), SUCCESS);
    map<string, OpsKernelInfoStorePtr> opsKernelInfoStores;
    GetOpsKernelInfoStores(opsKernelInfoStores);
    ASSERT_NE(opsKernelInfoStores["aicpu_ascend_kernel"], nullptr);
    ASSERT_EQ(opsKernelInfoStores["aicpu_ascend_kernel"]->Initialize(options), SUCCESS);
    map<string, GraphOptimizerPtr> graphOptimizers;
    GetGraphOptimizerObjs(graphOptimizers);
    ASSERT_NE(graphOptimizers["aicpu_ascend_optimizer"], nullptr);
    ASSERT_EQ(graphOptimizers["aicpu_ascend_optimizer"]->Initialize(options, nullptr), SUCCESS);
    map<string, OpsKernelBuilderPtr> opsKernelBuilders;
    GetOpsKernelBuilderObjs(opsKernelBuilders);
    ASSERT_NE(opsKernelBuilders["aicpu_ascend_builder"], nullptr);
    ASSERT_EQ(opsKernelBuilders["aicpu_ascend_builder"]->Initialize(options), SUCCESS);
    
    OpDescPtr opDescPtr = make_shared<OpDesc>("Add", "Add");
    AttrUtils::SetStr(opDescPtr, "resource", "RES_CHANNEL");
    vector<int64_t> tensorShape = {1, 1, 3, 1};
    GeTensorDesc tensor1(GeShape(tensorShape), FORMAT_NCHW, DT_INT32);
    opDescPtr->AddInputDesc("x", tensor1);
    opDescPtr->AddInputDesc("y", tensor1);
    opDescPtr->AddOutputDesc("z", tensor1);
    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_graph");
    ASSERT_EQ(opsKernelBuilders["aicpu_ascend_builder"]->CalcOpRunningParam(*(graphPtr->AddNode(opDescPtr))), SUCCESS);
    OpDescPtr opDescFwk = make_shared<OpDesc>("FrameOpCase", kFrameworkOp);
    AttrUtils::SetStr(opDescFwk, kOriginalType, "");
    opDescFwk->AddInputDesc("x", tensor1);
    opDescFwk->AddInputDesc("y", tensor1);
    opDescFwk->AddOutputDesc("z", tensor1);
    shared_ptr<ComputeGraph> graphPtrFwk = make_shared<ComputeGraph>("framework_op_graph");
    ASSERT_NE(opsKernelBuilders["aicpu_ascend_builder"]->CalcOpRunningParam(*(graphPtrFwk->AddNode(opDescFwk))), SUCCESS);
    ASSERT_EQ(opsKernelBuilders["aicpu_ascend_builder"]->Finalize(), SUCCESS);
}

TEST(AicpuKernelBuilder, CalcOpRunningParamWithMaxFftsThreadSize_SUCCESS)
{
    map<string, string> options;
    options[SOC_VERSION] = "Ascend910";
    ASSERT_EQ(Initialize(options), SUCCESS);
    map<string, OpsKernelInfoStorePtr> opsKernelInfoStores;
    GetOpsKernelInfoStores(opsKernelInfoStores);
    ASSERT_NE(opsKernelInfoStores["aicpu_ascend_kernel"], nullptr);
    ASSERT_EQ(opsKernelInfoStores["aicpu_ascend_kernel"]->Initialize(options), SUCCESS);
    map<string, GraphOptimizerPtr> graphOptimizers;
    GetGraphOptimizerObjs(graphOptimizers);
    ASSERT_NE(graphOptimizers["aicpu_ascend_optimizer"], nullptr);
    ASSERT_EQ(graphOptimizers["aicpu_ascend_optimizer"]->Initialize(options, nullptr), SUCCESS);
    map<string, OpsKernelBuilderPtr> opsKernelBuilders;
    GetOpsKernelBuilderObjs(opsKernelBuilders);
    ASSERT_NE(opsKernelBuilders["aicpu_ascend_builder"], nullptr);
    ASSERT_EQ(opsKernelBuilders["aicpu_ascend_builder"]->Initialize(options), SUCCESS);

    OpDescPtr opDescPtr = make_shared<OpDesc>("Add", "Add");
    AttrUtils::SetStr(opDescPtr, "resource", "RES_CHANNEL");
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
    slice_ptr->output_tensor_slice = {{{{0,1},{0,2},{0,2}},{{0,2},{0,2},{0,2}}}, {{{1,2},{0,2},{0,2}},{{1,2},{0,2},{0,2}}}};
    slice_ptr->input_tensor_slice = {{{{0,1},{0,2},{0,2}}},{{{1,2},{0,2},{0,2}}}};
    opDescPtr->SetExtAttr(kAttrNameSgtStruct, slice_ptr);
    AttrUtils::SetStr(opDescPtr, "opKernelLib", "AICPUKernel");
    AttrUtils::SetInt(opDescPtr, "_unknown_shape_type", 0);
    AttrUtils::SetInt(opDescPtr, kAttrNameThreadScopeId, 1);
    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_ffts_graph");
    Node *node = graphPtr->AddNode(opDescPtr).get();
    ASSERT_EQ(opsKernelBuilders["aicpu_ascend_builder"]->CalcOpRunningParam(*node), SUCCESS);

    int64_t output1_tensor_size = 0;
    AttrUtils::GetInt(opDescPtr->GetOutputDesc("y"), "_ffts_sub_task_tensor_size", output1_tensor_size);
    ASSERT_EQ(output1_tensor_size, 16);

    int64_t output2_tensor_size = 0;
    AttrUtils::GetInt(opDescPtr->GetOutputDesc("z"), "_ffts_sub_task_tensor_size", output2_tensor_size);
    ASSERT_EQ(output2_tensor_size, 32);
}

TEST(AicpuKernelBuilder, GenerateTask_SUCCESS)
{
    map<string, GraphOptimizerPtr> graphOptimizers;
    GetGraphOptimizerObjs(graphOptimizers);
    ASSERT_NE(graphOptimizers["aicpu_ascend_optimizer"], nullptr);

    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_graph");
    OpDescPtr opDescPtr = make_shared<OpDesc>("placeholder", "PlaceHolder");

    vector<int64_t> tensorShape = {1, 2, 3, 4};
    GeTensorDesc tensor1(GeShape(tensorShape), ge::FORMAT_NCHW, ge::DT_INT32);
    tensor1.SetOriginFormat(ge::FORMAT_NCHW);
    tensor1.SetOriginShape(GeShape(tensorShape));

    opDescPtr->AddOutputDesc("z", tensor1);
    auto node1 = graphPtr->AddNode(opDescPtr);
    OpDescPtr Op1DescPtr = make_shared<OpDesc>("FrameOpCase", kFrameworkOp);
    AttrUtils::SetStr(Op1DescPtr, kOriginalType, "Add");

    Op1DescPtr->AddOutputDesc("z", tensor1);
    auto node2 = graphPtr->AddNode(Op1DescPtr);

    node2->AddLinkFrom(node1);

    OpDescPtr Op2DescPtr = make_shared<OpDesc>("add2", "Add");

    Op2DescPtr->AddOutputDesc("z", tensor1);
    auto node3 = graphPtr->AddNode(Op2DescPtr);

    node3->AddLinkFrom(node2);
    ASSERT_EQ(graphOptimizers["aicpu_ascend_optimizer"]->OptimizeFusedGraph(*(graphPtr.get())), SUCCESS);
    ge::OpDescUtilsEx::SetType(Op1DescPtr, kFrameworkOp);
    map<string, OpsKernelBuilderPtr> opsKernelBuilders;
    GetOpsKernelBuilderObjs(opsKernelBuilders);
    map<string, string> options;
    options[SOC_VERSION] = "Ascend910";
    ASSERT_EQ(opsKernelBuilders["aicpu_ascend_builder"]->Initialize(options), SUCCESS);
    ASSERT_EQ(opsKernelBuilders["aicpu_ascend_builder"]->CalcOpRunningParam(*node2), SUCCESS);
    RunContext context = CreateContext();
    vector<domi::TaskDef> tasks;
    ge::OpDescUtilsEx::SetType(Op1DescPtr, kFrameworkOp);
    ASSERT_EQ(opsKernelBuilders["aicpu_ascend_builder"]->GenerateTask(*node2, context, tasks), SUCCESS);
    ASSERT_EQ(opsKernelBuilders["aicpu_ascend_builder"]->UpdateTask(*node2, tasks), SUCCESS);
    DestroyContext(context);
    printf("end===================\n");
}

TEST(AicpuKernelBuilder, GenerateTask_SUCCESS_FFTS)
{
    map<string, OpsKernelBuilderPtr> opsKernelBuilders;
    GetOpsKernelBuilderObjs(opsKernelBuilders);

    OpDescPtr opDescPtr = make_shared<OpDesc>("Add", "Add");
    vector<int64_t> tensorShape = {1, 1, 1, 1};
    GeTensorDesc tensor1(GeShape(tensorShape), FORMAT_NCHW, DT_INT32);
    opDescPtr->AddInputDesc("x1", tensor1);
    opDescPtr->AddInputDesc("x2", tensor1);
    opDescPtr->AddOutputDesc("y", tensor1);
    ffts::ThreadSliceMap thread_slice_map;
    thread_slice_map.thread_scope_id = 1;
    thread_slice_map.slice_instance_num = 2;
    thread_slice_map.thread_mode = 0;
    ffts::ThreadSliceMapPtr slice_ptr = make_shared<ffts::ThreadSliceMap>(thread_slice_map);
    opDescPtr->SetExtAttr(kAttrNameSgtStruct, slice_ptr);
    AttrUtils::SetBool(opDescPtr, kAttrNameUnknownShape, true);
    AttrUtils::SetStr(opDescPtr, "opKernelLib", "AICPUKernel");
    AttrUtils::SetInt(opDescPtr, "_unknown_shape_type", 0);
    ge::AttrUtils::SetInt(opDescPtr, kAttrNameThreadScopeId, 1);
    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_ffts_graph");
    Node *node = graphPtr->AddNode(opDescPtr).get();
    ASSERT_EQ(opsKernelBuilders["aicpu_ascend_builder"]->CalcOpRunningParam(*node), SUCCESS);
    RunContext context = CreateContext();
    vector<domi::TaskDef> tasks;
    ASSERT_EQ(opsKernelBuilders["aicpu_ascend_builder"]->GenerateTask(*node, context, tasks), SUCCESS);
    slice_ptr->thread_mode = 1;
    slice_ptr->slice_instance_num = 1;
    slice_ptr->input_tensor_slice = {{{{0, 1}}, {{0, 1}}}};
    slice_ptr->output_tensor_slice = {{{{0, 1}}}};
    opDescPtr->SetExtAttr(kAttrNameSgtStruct, slice_ptr);
    ASSERT_EQ(opsKernelBuilders["aicpu_ascend_builder"]->GenerateTask(*node, context, tasks), SUCCESS);
    DestroyContext(context);
    printf("ffts task end===================\n");
}

TEST(AicpuKernelBuilder, GenerateTask_FAIL_FFTS)
{
    map<string, OpsKernelBuilderPtr> opsKernelBuilders;
    GetOpsKernelBuilderObjs(opsKernelBuilders);

    OpDescPtr opDescPtr = make_shared<OpDesc>("Add", "Add");
    vector<int64_t> tensorShape = {1, 1, 1, 1};
    GeTensorDesc tensor1(GeShape(tensorShape), FORMAT_NCHW, DT_INT32);
    opDescPtr->AddInputDesc("x1", tensor1);
    opDescPtr->AddInputDesc("x2", tensor1);
    opDescPtr->AddOutputDesc("y", tensor1);
    opDescPtr->SetExtAttr(kAttrNameSgtStruct, nullptr);
    AttrUtils::SetStr(opDescPtr, "opKernelLib", "CUSTAICPUKernel");
    AttrUtils::SetInt(opDescPtr, "_unknown_shape_type", 0);
    ge::AttrUtils::SetInt(opDescPtr, kAttrNameThreadScopeId, 1);
    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_ffts_graph");
    Node *node = graphPtr->AddNode(opDescPtr).get();
    ASSERT_NE(opsKernelBuilders["aicpu_ascend_builder"]->CalcOpRunningParam(*node), SUCCESS);
    RunContext context = CreateContext();
    vector<domi::TaskDef> tasks;
    ASSERT_NE(opsKernelBuilders["aicpu_ascend_builder"]->GenerateTask(*node, context, tasks), SUCCESS);
    DestroyContext(context);
    printf("ffts task end===================\n");
}

TEST(AicpuKernelBuilder, GenerateTask_NOTILING_SUCCESS) {
  map<string, GraphOptimizerPtr> graphOptimizers;
  GetGraphOptimizerObjs(graphOptimizers);
  ASSERT_NE(graphOptimizers["aicpu_ascend_optimizer"], nullptr);

  shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_graph");
  OpDescPtr opDescPtr = make_shared<OpDesc>("placeholder", "PlaceHolder");

  vector<int64_t> tensorShape = {1, 2, 3, 4};
  GeTensorDesc tensor1(GeShape(tensorShape), ge::FORMAT_NCHW, ge::DT_INT32);
  tensor1.SetOriginFormat(ge::FORMAT_NCHW);
  tensor1.SetOriginShape(GeShape(tensorShape));
  (void)AttrUtils::SetBool(&tensor1, ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE, true);

  opDescPtr->AddOutputDesc("z", tensor1);
  auto node1 = graphPtr->AddNode(opDescPtr);
  OpDescPtr Op1DescPtr = make_shared<OpDesc>("FrameOpCase", kFrameworkOp);
  AttrUtils::SetStr(Op1DescPtr, kOriginalType, "Add");

  Op1DescPtr->AddOutputDesc("z", tensor1);
  auto node2 = graphPtr->AddNode(Op1DescPtr);

  node2->AddLinkFrom(node1);

  OpDescPtr Op2DescPtr = make_shared<OpDesc>("add2", "Add");

  Op2DescPtr->AddOutputDesc("z", tensor1);
  auto node3 = graphPtr->AddNode(Op2DescPtr);

  node3->AddLinkFrom(node2);
  ASSERT_EQ(graphOptimizers["aicpu_ascend_optimizer"]->OptimizeFusedGraph(*(graphPtr.get())), SUCCESS);
  ge::OpDescUtilsEx::SetType(Op1DescPtr, kFrameworkOp);
  map<string, OpsKernelBuilderPtr> opsKernelBuilders;
  GetOpsKernelBuilderObjs(opsKernelBuilders);
  map<string, string> options;
  options[SOC_VERSION] = "Ascend910";
  ASSERT_EQ(opsKernelBuilders["aicpu_ascend_builder"]->Initialize(options), SUCCESS);
  ASSERT_EQ(opsKernelBuilders["aicpu_ascend_builder"]->CalcOpRunningParam(*node2), SUCCESS);
  RunContext context = CreateContext();
  vector<domi::TaskDef> tasks;
  ge::OpDescUtilsEx::SetType(Op1DescPtr, kFrameworkOp);
  ASSERT_EQ(opsKernelBuilders["aicpu_ascend_builder"]->GenerateTask(*node2, context, tasks), SUCCESS);
  DestroyContext(context);
  printf("end===================\n");
}

TEST(AicpuKernelBuilder, GenerateTask_UNKNOWSHAPE_SUCCESS)
{
    map<string, OpsKernelBuilderPtr> opsKernelBuilders;
    GetOpsKernelBuilderObjs(opsKernelBuilders);
    OpDescPtr opDescPtr = make_shared<OpDesc>("Test", "Test");
    vector<int64_t> tensorShape = {1, 1, 3, -1};
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
    ASSERT_EQ(graphOptimizers["aicpu_ascend_optimizer"]->Initialize(options, nullptr), SUCCESS);
    ASSERT_EQ(graphOptimizers["aicpu_ascend_optimizer"]->OptimizeFusedGraph(*(graphPtr.get())), SUCCESS);
    ASSERT_EQ(opsKernelBuilders["aicpu_ascend_builder"]->CalcOpRunningParam(*node), SUCCESS);
    RunContext context = CreateContext();
    vector<domi::TaskDef> tasks;
    ASSERT_EQ(opsKernelBuilders["aicpu_ascend_builder"]->GenerateTask(*node, context, tasks), SUCCESS);
    DestroyContext(context);
}

TEST(AicpuKernelBuilder, GenerateTask_RANGESHAPE_SUCCESS)
{
    map<string, OpsKernelBuilderPtr> opsKernelBuilders;
    GetOpsKernelBuilderObjs(opsKernelBuilders);
    OpDescPtr opDescPtr = make_shared<OpDesc>("Test3", "Test3");
    OpDescPtr opDescPtr1 = make_shared<OpDesc>("Test3", "Test3");
    vector<int64_t> tensorShape = {1, 1, 3, -1};
    vector<pair<int64_t, int64_t>> range;
    for (auto i : tensorShape) {
        range.push_back(make_pair(1, i));
    }
    GeTensorDesc tensor1(GeShape(tensorShape), FORMAT_NCHW, DT_INT32);
    opDescPtr->AddInputDesc("x", tensor1);
    opDescPtr1->AddInputDesc("x", tensor1);
    GeTensorDesc tensor2(GeShape({-1}), FORMAT_NCHW, DT_INT32);
    GeTensorDesc tensor3(GeShape({-1}), FORMAT_NCHW, DT_INT32);
    tensor2.SetShapeRange(range);
    opDescPtr->AddOutputDesc("z", tensor2);
    opDescPtr1->AddOutputDesc("z", tensor3);
    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_graph");
    Node *node = graphPtr->AddNode(opDescPtr).get();
    map<string, GraphOptimizerPtr> graphOptimizers;
    GetGraphOptimizerObjs(graphOptimizers);
    map<string, string> options;
    options[SOC_VERSION] = "Ascend910";
    ASSERT_EQ(graphOptimizers["aicpu_ascend_optimizer"]->Initialize(options, nullptr), SUCCESS);
    ASSERT_EQ(graphOptimizers["aicpu_ascend_optimizer"]->OptimizeFusedGraph(*(graphPtr.get())), SUCCESS);
    ASSERT_EQ(opsKernelBuilders["aicpu_ascend_builder"]->CalcOpRunningParam(*node), SUCCESS);
    RunContext context = CreateContext();
    vector<domi::TaskDef> tasks;
    ASSERT_EQ(opsKernelBuilders["aicpu_ascend_builder"]->GenerateTask(*node, context, tasks), SUCCESS);
    DestroyContext(context);
    Node *node1 = graphPtr->AddNode(opDescPtr1).get();
    ASSERT_EQ(graphOptimizers["aicpu_ascend_optimizer"]->OptimizeFusedGraph(*(graphPtr.get())), SUCCESS);
    ASSERT_EQ(opsKernelBuilders["aicpu_ascend_builder"]->CalcOpRunningParam(*node1), SUCCESS);
    RunContext context1 = CreateContext();
    vector<domi::TaskDef> tasks1;
    ASSERT_EQ(opsKernelBuilders["aicpu_ascend_builder"]->GenerateTask(*node1, context1, tasks1), SUCCESS);
    DestroyContext(context1);
}

TEST(AicpuKernelBuilder, GenerateTask_Priority_SUCCESS)
{
    map<string, GraphOptimizerPtr> graphOptimizers;
    GetGraphOptimizerObjs(graphOptimizers);
    ASSERT_NE(graphOptimizers["aicpu_ascend_optimizer"], nullptr);

    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_graph");
    OpDescPtr opDescPtr = make_shared<OpDesc>("placeholder", "PlaceHolder");

    vector<int64_t> tensorShape = {1, 2, 3, 4};
    GeTensorDesc tensor1(GeShape(tensorShape), ge::FORMAT_NCHW, ge::DT_INT32);
    tensor1.SetOriginFormat(ge::FORMAT_NCHW);
    tensor1.SetOriginShape(GeShape(tensorShape));

    opDescPtr->AddOutputDesc("z", tensor1);
    auto node1 = graphPtr->AddNode(opDescPtr);
    OpDescPtr Op1DescPtr = make_shared<OpDesc>("FrameOpCase", kFrameworkOp);
    AttrUtils::SetStr(Op1DescPtr, kOriginalType, "Add");

    Op1DescPtr->AddOutputDesc("z", tensor1);
    auto node2 = graphPtr->AddNode(Op1DescPtr);
    node2->AddLinkFrom(node1);
    OpDescPtr Op2DescPtr = make_shared<OpDesc>("add2", "Add");

    Op2DescPtr->AddOutputDesc("z", tensor1);
    auto node3 = graphPtr->AddNode(Op2DescPtr);

    node3->AddLinkFrom(node2);
    ASSERT_EQ(graphOptimizers["aicpu_ascend_optimizer"]->OptimizeFusedGraph(*(graphPtr.get())), SUCCESS);
    ge::OpDescUtilsEx::SetType(Op1DescPtr, kFrameworkOp);
    map<string, OpsKernelBuilderPtr> opsKernelBuilders;
    GetOpsKernelBuilderObjs(opsKernelBuilders);
    map<string, string> options;
    options[SOC_VERSION] = "Ascend910";
    options["ge.exec.exclude_engines"] = "AiCore";
    ge::GetThreadLocalContext().SetGraphOption(options);
    ASSERT_EQ(opsKernelBuilders["aicpu_ascend_builder"]->Initialize(options), SUCCESS);
    ASSERT_EQ(opsKernelBuilders["aicpu_ascend_builder"]->CalcOpRunningParam(*node2), SUCCESS);
    RunContext context = CreateContext();
    vector<domi::TaskDef> tasks;
    ge::OpDescUtilsEx::SetType(Op1DescPtr, kFrameworkOp);
    ASSERT_EQ(opsKernelBuilders["aicpu_ascend_builder"]->GenerateTask(*node2, context, tasks), SUCCESS);
    DestroyContext(context);
    printf("end===================\n");
}

TEST(AicpuKernelBuilder, GenSingleOpRunTask_FAIL)
{
    map<string, OpsKernelBuilderPtr> opsKernelBuilders;
    GetOpsKernelBuilderObjs(opsKernelBuilders);
    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_graph");
    OpDescPtr opDescPtr = make_shared<OpDesc>("Add", "Add");
    auto nodePtr = graphPtr->AddNode(opDescPtr);
    vector<int64_t> tensorShape = {1, 1, 3, 1};
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
    ASSERT_NE(SUCCESS, opsKernelBuilders["aicpu_ascend_builder"]->GenSingleOpRunTask(nodePtr, task, task_info));
}

TEST(AicpuKernelBuilder, GenMemCopyTask_FAIL)
{
    STR_FWK_OP_KERNEL task;
    map<string, OpsKernelBuilderPtr> opsKernelBuilders;
    GetOpsKernelBuilderObjs(opsKernelBuilders);
    std::string task_info;
    ASSERT_NE(SUCCESS, opsKernelBuilders["aicpu_ascend_builder"]->GenMemCopyTask(2, task, task_info));
}

TEST(AicpuKernelBuilder, Initialize_GetConfig_FAIL)
{
    AicpuOpsKernelBuilder aicpuOpsKBuilder("TestEngine");
    map<string, string> options;
    options[SOC_VERSION] = "Ascend910";
    ASSERT_NE(aicpuOpsKBuilder.Initialize(options), SUCCESS);
}

TEST(AicpuKernelBuilder, Initialize_Null_Ptr_FAIL)
{
    AicpuOpsKernelBuilder aicpuOpsKBuilder("DNN_VM_AICPU");
    map<string, string> options;
    options[SOC_VERSION] = "Ascend910";
    ASSERT_NE(aicpuOpsKBuilder.Initialize(options), SUCCESS);
}

TEST(AicpuKernelBuilder, Initialize_Split_FAIL)
{
    AicpuOpsKernelBuilder aicpuOpsKBuilder("DNN_VM_AICPU_EMPTY");
    map<string, string> options;
    options[SOC_VERSION] = "Ascend910";
    ASSERT_NE(aicpuOpsKBuilder.Initialize(options), SUCCESS);
    printf("end===================\n");
}