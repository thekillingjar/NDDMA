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

#include "ge/ge_api_types.h"
#include "common/sgt_slice_type.h"
#include "graph/utils/tensor_utils.h"

using namespace aicpu;
using namespace ge;
using namespace std;

TEST(AicpuCustKernelBuilder, Initialize_SUCCESS)
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
}

TEST(AicpuCustKernelBuilder, CalcOpRunningParam_SUCCESS)
{
    map<string, OpsKernelBuilderPtr> opsKernelBuilders;
    GetOpsKernelBuilderObjs(opsKernelBuilders);

    OpDescPtr opDescPtr = make_shared<OpDesc>("RandomChoiceWithMask", "RandomChoiceWithMask");
    vector<int64_t> tensorShape = {1, 1, 3, 1};
    GeTensorDesc tensor1(GeShape(tensorShape), FORMAT_NCHW, DT_INT32);
    opDescPtr->AddInputDesc("x", tensor1);
    opDescPtr->AddInputDesc("y", tensor1);
    opDescPtr->AddOutputDesc("z", tensor1);
    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_graph");
    ASSERT_EQ(opsKernelBuilders["aicpu_ascend_builder"]->CalcOpRunningParam(*(graphPtr->AddNode(opDescPtr))), SUCCESS);
}

TEST(AicpuCustKernelBuilder, GenerateTask_SUCCESS)
{
    map<string, OpsKernelBuilderPtr> opsKernelBuilders;
    GetOpsKernelBuilderObjs(opsKernelBuilders);
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
    AttrUtils::SetInt(opDescPtr, "topic_type", 0);
    auto node1 = graphPtr->AddNode(opDescPtr);
    OpDescPtr op1DescPtr = make_shared<OpDesc>("RandomChoiceWithMask", "RandomChoiceWithMask");

    op1DescPtr->AddOutputDesc("z", tensor1);
    AttrUtils::SetInt(op1DescPtr, "topic_type", 0);
    auto node2 = graphPtr->AddNode(op1DescPtr);

    node2->AddLinkFrom(node1);

    OpDescPtr op2DescPtr = make_shared<OpDesc>("RandomChoiceWithMask", "RandomChoiceWithMask");

    op1DescPtr->AddOutputDesc("z", tensor1);
    AttrUtils::SetInt(op2DescPtr, "topic_type", 0);
    auto node3 = graphPtr->AddNode(op2DescPtr);

    node3->AddLinkFrom(node2);
    ASSERT_EQ(graphOptimizers["aicpu_ascend_optimizer"]->OptimizeFusedGraph(*(graphPtr.get())), SUCCESS);
    ASSERT_EQ(opsKernelBuilders["aicpu_ascend_builder"]->CalcOpRunningParam(*node3), SUCCESS);
    RunContext context = CreateContext();
    vector<domi::TaskDef> tasks;
    ASSERT_EQ(opsKernelBuilders["aicpu_ascend_builder"]->GenerateTask(*node3, context, tasks), SUCCESS);
    ASSERT_EQ(opsKernelBuilders["aicpu_ascend_builder"]->UpdateTask(*node3, tasks), SUCCESS);
    DestroyContext(context);
    printf("end===================\n");
}

TEST(AicpuCustKernelBuilder, GenerateTask_GetOpBlockDim_SUCCESS)
{
    map<string, OpsKernelBuilderPtr> opsKernelBuilders;
    GetOpsKernelBuilderObjs(opsKernelBuilders);
    map<string, GraphOptimizerPtr> graphOptimizers;
    GetGraphOptimizerObjs(graphOptimizers);
    ASSERT_NE(graphOptimizers["aicpu_ascend_optimizer"], nullptr);
    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_graph");
    OpDescPtr opDescPtr = make_shared<OpDesc>("placeholder", "PlaceHolder");

    vector<int64_t> tensorShape = {5, 6, 7};
    GeTensorDesc tensor1(GeShape(tensorShape), ge::FORMAT_NCHW, ge::DT_INT32);
    tensor1.SetOriginFormat(ge::FORMAT_NCHW);
    tensor1.SetOriginShape(GeShape(tensorShape));
    
    opDescPtr->AddInputDesc("x1", tensor1);
    opDescPtr->AddInputDesc("x2", tensor1);
    opDescPtr->AddOutputDesc("y", tensor1);
    auto node1 = graphPtr->AddNode(opDescPtr);
    OpDescPtr op1DescPtr = make_shared<OpDesc>("TestBlock", "TestBlock");

    op1DescPtr->AddInputDesc("x1", tensor1);
    op1DescPtr->AddInputDesc("x2", tensor1);
    op1DescPtr->AddOutputDesc("y", tensor1);
    auto node2 = graphPtr->AddNode(op1DescPtr);

    node2->AddLinkFrom(node1);

    OpDescPtr op2DescPtr = make_shared<OpDesc>("TestBlock", "TestBlock");

    op2DescPtr->AddInputDesc("x1", tensor1);
    op2DescPtr->AddInputDesc("x2", tensor1);
    op2DescPtr->AddOutputDesc("y", tensor1);
    auto node3 = graphPtr->AddNode(op2DescPtr);

    node3->AddLinkFrom(node2);
    ASSERT_EQ(graphOptimizers["aicpu_ascend_optimizer"]->OptimizeFusedGraph(*(graphPtr.get())), SUCCESS);
    ASSERT_EQ(opsKernelBuilders["aicpu_ascend_builder"]->CalcOpRunningParam(*node3), SUCCESS);
    RunContext context = CreateContext();
    vector<domi::TaskDef> tasks;
    ASSERT_EQ(opsKernelBuilders["aicpu_ascend_builder"]->GenerateTask(*node3, context, tasks), SUCCESS);
    DestroyContext(context);
    printf("end===================\n");
}

TEST(AicpuKernelBuilder, GenerateTask_SUCCESS_FFTS_AUTO_BLOCKDIM)
{
    map<string, OpsKernelBuilderPtr> opsKernelBuilders;
    GetOpsKernelBuilderObjs(opsKernelBuilders);

    OpDescPtr opDescPtr = make_shared<OpDesc>("Add", "Add");
    vector<int64_t> tensorShape = {2, 2, 2};
    GeTensorDesc tensor1(GeShape(tensorShape), FORMAT_NCHW, DT_INT32);
    opDescPtr->AddInputDesc("x1", tensor1);
    opDescPtr->AddInputDesc("x2", tensor1);
    opDescPtr->AddOutputDesc("y", tensor1);
    ffts::ThreadSliceMap thread_slice_map;
    thread_slice_map.thread_scope_id = 1;
    thread_slice_map.slice_instance_num = 2;
    thread_slice_map.thread_mode = true; // ffts plus auto mode
    ffts::ThreadSliceMapPtr slice_ptr = make_shared<ffts::ThreadSliceMap>(thread_slice_map);
    slice_ptr->input_tensor_slice = {{{{0,1},{0,2},{0,2}},{{0,1},{0,2},{0,2}}}, {{{1,2},{0,2},{0,2}},{{1,2},{0,2},{0,2}}}};
    slice_ptr->output_tensor_slice = {{{{0,1},{0,2},{0,2}}},{{{1,2},{0,2},{0,2}}}};
    opDescPtr->SetExtAttr(kAttrNameSgtStruct, slice_ptr);

    AttrUtils::SetStr(opDescPtr, "opKernelLib", "AICPUKernel");
    AttrUtils::SetInt(opDescPtr, "_unknown_shape_type", 0);
    AttrUtils::SetInt(opDescPtr, kAttrNameThreadScopeId, 1);

    // add blockdim params
    AttrUtils::SetInt(opDescPtr, kBlockDimByIndex, -1);
    AttrUtils::SetBool(opDescPtr, kSupportBlockDim, true);

    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_ffts_graph_blockdim");
    Node *node = graphPtr->AddNode(opDescPtr).get();
    ASSERT_EQ(opsKernelBuilders["aicpu_ascend_builder"]->CalcOpRunningParam(*node), SUCCESS);
    RunContext context = CreateContext();
    vector<domi::TaskDef> tasks;
    ASSERT_EQ(opsKernelBuilders["aicpu_ascend_builder"]->GenerateTask(*node, context, tasks), SUCCESS);
    DestroyContext(context);
    printf("ffts task end===================\n");
}

TEST(AicpuKernelBuilder, GenerateTask_SUCCESS_FFTS_MANUAL_BLOCKDIM)
{
    map<string, OpsKernelBuilderPtr> opsKernelBuilders;
    GetOpsKernelBuilderObjs(opsKernelBuilders);

    OpDescPtr opDescPtr = make_shared<OpDesc>("Add", "Add");
    vector<int64_t> tensorShape = {2, 2, 2};
    GeTensorDesc tensor1(GeShape(tensorShape), FORMAT_NCHW, DT_INT32);
    opDescPtr->AddInputDesc("x1", tensor1);
    opDescPtr->AddInputDesc("x2", tensor1);
    opDescPtr->AddOutputDesc("y", tensor1);
    ffts::ThreadSliceMap thread_slice_map;
    thread_slice_map.thread_scope_id = 1;
    thread_slice_map.slice_instance_num = 1;
    thread_slice_map.thread_mode = false; // ffts plus manual mode
    ffts::ThreadSliceMapPtr slice_ptr = make_shared<ffts::ThreadSliceMap>(thread_slice_map);
    slice_ptr->input_tensor_slice = {{{{0,1},{0,2},{0,2}}},{{{1,2},{0,2},{0,2}}}};
    slice_ptr->output_tensor_slice = {{{{0,1},{0,2},{0,2}}},{{{1,2},{0,2},{0,2}}}};
    opDescPtr->SetExtAttr(kAttrNameSgtStruct, slice_ptr);

    AttrUtils::SetStr(opDescPtr, "opKernelLib", "AICPUKernel");
    AttrUtils::SetInt(opDescPtr, "_unknown_shape_type", 0);
    AttrUtils::SetInt(opDescPtr, kAttrNameThreadScopeId, 1);

    // add blockdim params
    AttrUtils::SetInt(opDescPtr, kBlockDimByIndex, -1);
    AttrUtils::SetBool(opDescPtr, kSupportBlockDim, true);

    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_ffts_graph_blockdim");
    Node *node = graphPtr->AddNode(opDescPtr).get();
    ASSERT_EQ(opsKernelBuilders["aicpu_ascend_builder"]->CalcOpRunningParam(*node), SUCCESS);
    RunContext context = CreateContext();
    vector<domi::TaskDef> tasks;
    ASSERT_EQ(opsKernelBuilders["aicpu_ascend_builder"]->GenerateTask(*node, context, tasks), SUCCESS);
    DestroyContext(context);
    printf("ffts task end===================\n");
}

TEST(AicpuKernelBuilder, GenerateTask_SUCCESS_FFTS_NOTSUPPORT_BLOCKDIM)
{
    map<string, OpsKernelBuilderPtr> opsKernelBuilders;
    GetOpsKernelBuilderObjs(opsKernelBuilders);

    OpDescPtr opDescPtr = make_shared<OpDesc>("Add", "Add");
    vector<int64_t> tensorShape = {2, 2, 2};
    GeTensorDesc tensor1(GeShape(tensorShape), FORMAT_NCHW, DT_INT32);
    opDescPtr->AddInputDesc("x1", tensor1);
    opDescPtr->AddInputDesc("x2", tensor1);
    opDescPtr->AddOutputDesc("y", tensor1);
    ffts::ThreadSliceMap thread_slice_map;
    thread_slice_map.thread_scope_id = 1;
    thread_slice_map.slice_instance_num = 1;
    thread_slice_map.thread_mode = false; // ffts plus manual mode
    ffts::ThreadSliceMapPtr slice_ptr = make_shared<ffts::ThreadSliceMap>(thread_slice_map);
    slice_ptr->input_tensor_slice = {{{{0,1},{0,2},{0,2}}},{{{1,2},{0,2},{0,2}}}};
    slice_ptr->output_tensor_slice = {{{{0,1},{0,2},{0,2}}},{{{1,2},{0,2},{0,2}}}};
    opDescPtr->SetExtAttr(kAttrNameSgtStruct, slice_ptr);

    AttrUtils::SetStr(opDescPtr, "opKernelLib", "AICPUKernel");
    AttrUtils::SetInt(opDescPtr, "_unknown_shape_type", 0);
    AttrUtils::SetInt(opDescPtr, kAttrNameThreadScopeId, 1);

    // add blockdim params
    AttrUtils::SetInt(opDescPtr, kBlockDimByIndex, -1);
    AttrUtils::SetBool(opDescPtr, kSupportBlockDim, false);

    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_ffts_graph_blockdim");
    Node *node = graphPtr->AddNode(opDescPtr).get();
    ASSERT_EQ(opsKernelBuilders["aicpu_ascend_builder"]->CalcOpRunningParam(*node), SUCCESS);
    RunContext context = CreateContext();
    vector<domi::TaskDef> tasks;
    ASSERT_EQ(opsKernelBuilders["aicpu_ascend_builder"]->GenerateTask(*node, context, tasks), SUCCESS);
    DestroyContext(context);
    printf("ffts task end===================\n");
}

TEST(AicpuCustKernelBuilder, NewOpp_GenerateTask_SUCCESS)
{
    char *build_path = getenv("BUILD_PATH");
    std::string opp_path = std::string(build_path) + "tests/engines/cpueng/stub/config/";
    char *path = (char *)opp_path.c_str();
    setenv("ASCEND_OPP_PATH", path, 1);
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

    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_graph");
    OpDescPtr opDescPtr = make_shared<OpDesc>("placeholder", "PlaceHolder");

    vector<int64_t> tensorShape = {1, 2, 3, 4};
    GeTensorDesc tensor1(GeShape(tensorShape), ge::FORMAT_NCHW, ge::DT_INT32);
    tensor1.SetOriginFormat(ge::FORMAT_NCHW);
    tensor1.SetOriginShape(GeShape(tensorShape));

    opDescPtr->AddOutputDesc("z", tensor1);
    AttrUtils::SetInt(opDescPtr, "topic_type", 0);
    auto node1 = graphPtr->AddNode(opDescPtr);
    OpDescPtr op1DescPtr = make_shared<OpDesc>("RandomChoiceWithMask", "RandomChoiceWithMask");

    op1DescPtr->AddOutputDesc("z", tensor1);
    AttrUtils::SetInt(op1DescPtr, "topic_type", 0);
    auto node2 = graphPtr->AddNode(op1DescPtr);

    node2->AddLinkFrom(node1);

    OpDescPtr op2DescPtr = make_shared<OpDesc>("RandomChoiceWithMask", "RandomChoiceWithMask");

    op1DescPtr->AddOutputDesc("z", tensor1);
    AttrUtils::SetInt(op2DescPtr, "topic_type", 0);
    auto node3 = graphPtr->AddNode(op2DescPtr);

    node3->AddLinkFrom(node2);
    ASSERT_EQ(graphOptimizers["aicpu_ascend_optimizer"]->OptimizeFusedGraph(*(graphPtr.get())), SUCCESS);
    ASSERT_EQ(opsKernelBuilders["aicpu_ascend_builder"]->CalcOpRunningParam(*node3), SUCCESS);
    RunContext context = CreateContext();
    vector<domi::TaskDef> tasks;
    ASSERT_EQ(opsKernelBuilders["aicpu_ascend_builder"]->GenerateTask(*node3, context, tasks), SUCCESS);
    DestroyContext(context);
    printf("end===================\n");
    unsetenv("ASCEND_OPP_PATH");
}