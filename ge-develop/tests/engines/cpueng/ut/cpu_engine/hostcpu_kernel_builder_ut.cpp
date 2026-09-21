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

#include "hostcpu_engine/engine/hostcpu_engine.h"
#include "hostcpu_engine/kernel_builder/hostcpu_ops_kernel_builder.h"
#include "util/util.h"
#include "config/config_file.h"
#include "stub.h"

#include "ge/ge_api_types.h"
#include "graph/utils/op_desc_utils_ex.h"

using namespace aicpu;
using namespace ge;
using namespace std; 


TEST(HostCpuKernelBuilder, CalcOpRunningParam_SUCCESS)
{
    map<string, string> options;
    options[SOC_VERSION] = "Ascend910";
    ASSERT_EQ(Initialize(options), SUCCESS);
    map<string, OpsKernelInfoStorePtr> opsKernelInfoStores;
    GetOpsKernelInfoStores(opsKernelInfoStores);
    ASSERT_NE(opsKernelInfoStores["DNN_VM_HOST_CPU_OP_STORE"], nullptr);
    ASSERT_EQ(opsKernelInfoStores["DNN_VM_HOST_CPU_OP_STORE"]->Initialize(options), SUCCESS);
    map<string, GraphOptimizerPtr> graphOptimizers;
    GetGraphOptimizerObjs(graphOptimizers);
    ASSERT_NE(graphOptimizers["DNN_VM_HOST_CPU_OPTIMIZER"], nullptr);
    ASSERT_EQ(graphOptimizers["DNN_VM_HOST_CPU_OPTIMIZER"]->Initialize(options, nullptr), SUCCESS);
    map<string, OpsKernelBuilderPtr> opsKernelBuilders;
    GetOpsKernelBuilderObjs(opsKernelBuilders);
    ASSERT_NE(opsKernelBuilders["host_cpu_builder"], nullptr);
    ASSERT_EQ(opsKernelBuilders["host_cpu_builder"]->Initialize(options), SUCCESS);
    OpDescPtr opDescPtr = make_shared<OpDesc>("Add", "Add");
    AttrUtils::SetStr(opDescPtr, "resource", "RES_QUEUE");
    AttrUtils::SetStr(opDescPtr, "queue_name", "test");
    vector<int64_t> tensorShape = {1, 1, 3, 1};
    GeTensorDesc tensor1(GeShape(tensorShape), FORMAT_NCHW, DT_INT32);
    opDescPtr->AddInputDesc("x", tensor1);
    opDescPtr->AddInputDesc("y", tensor1);
    opDescPtr->AddOutputDesc("z", tensor1);
    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_graph");
    ASSERT_EQ(opsKernelBuilders["host_cpu_builder"]->CalcOpRunningParam(*(graphPtr->AddNode(opDescPtr))), SUCCESS);
}

TEST(HostCpuKernelBuilder, GenerateTask_SUCCESS)
{
    map<string, GraphOptimizerPtr> graphOptimizers;
    GetGraphOptimizerObjs(graphOptimizers);
    ASSERT_NE(graphOptimizers["DNN_VM_HOST_CPU_OPTIMIZER"], nullptr);

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
    ASSERT_EQ(graphOptimizers["DNN_VM_HOST_CPU_OPTIMIZER"]->OptimizeFusedGraph(*(graphPtr.get())), SUCCESS);
    ge::OpDescUtilsEx::SetType(Op1DescPtr, kFrameworkOp);
    map<string, OpsKernelBuilderPtr> opsKernelBuilders;
    GetOpsKernelBuilderObjs(opsKernelBuilders);
    map<string, string> options;
    options[SOC_VERSION] = "Ascend910";
    ASSERT_EQ(opsKernelBuilders["host_cpu_builder"]->Initialize(options), SUCCESS);
    ASSERT_EQ(opsKernelBuilders["host_cpu_builder"]->CalcOpRunningParam(*node2), SUCCESS);
    RunContext context = CreateContext();
    vector<domi::TaskDef> tasks;
    ge::OpDescUtilsEx::SetType(Op1DescPtr, kFrameworkOp);
    ASSERT_EQ(opsKernelBuilders["host_cpu_builder"]->GenerateTask(*node2, context, tasks), SUCCESS);
    DestroyContext(context);
    printf("end===================\n");
}

TEST(HostCpuOpsKernelBuilder, GenerateTask_SUCCESS)
{
    HostCpuOpsKernelBuilder hostCpuKernelBuilder;
    map<string, string> options;
    options[SOC_VERSION] = "Ascend910";
    ASSERT_EQ(hostCpuKernelBuilder.Initialize(options), SUCCESS);
    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_graph");
    OpDescPtr opDescPtr = make_shared<OpDesc>("placeholder", "PlaceHolder");

    vector<int64_t> tensorShape = {1, 2 , 3, 4};
    GeTensorDesc tensor1(GeShape(tensorShape), ge::FORMAT_NCHW, ge::DT_INT32);
    tensor1.SetOriginFormat(ge::FORMAT_NCHW);
    tensor1.SetOriginShape(GeShape(tensorShape));

    opDescPtr->AddOutputDesc("z", tensor1);
    AttrUtils::SetInt(opDescPtr, "topic_type", 0);
    auto node1 = graphPtr->AddNode(opDescPtr);
    OpDescPtr op1DescPtr = make_shared<OpDesc>("FrameworkOpCase", kFrameworkOp);;

    op1DescPtr->AddOutputDesc("z", tensor1);
    AttrUtils::SetInt(op1DescPtr, "topic_type", 0);
    AttrUtils::SetInt(op1DescPtr, "_unknown_shape_type", 0);
    AttrUtils::SetStr(op1DescPtr, "opKernelLib", "HOSTCPUKernel");
    ge::OpDescUtilsEx::SetType(op1DescPtr, kFrameworkOp);
    AttrUtils::SetStr(op1DescPtr, kOriginalType, "MatrixDiagPartV3");
    auto node2 = graphPtr->AddNode(op1DescPtr);

    node2->AddLinkFrom(node1);
    RunContext context = CreateContext();
    vector<domi::TaskDef> tasks;
    ASSERT_EQ(hostCpuKernelBuilder.CalcOpRunningParam(*node2), SUCCESS);
    ASSERT_EQ(hostCpuKernelBuilder.GenerateTask(*node2, context, tasks), SUCCESS);
    ASSERT_EQ(hostCpuKernelBuilder.UpdateTask(*node2, tasks), SUCCESS);
    DestroyContext(context);

    OpDescPtr op2DescPtr = make_shared<OpDesc>("RandomChoiceWithMask", "RandomChoiceWithMask");

    op2DescPtr->AddOutputDesc("z", tensor1);
    AttrUtils::SetInt(op2DescPtr, "topic_type", 0);
    AttrUtils::SetInt(op2DescPtr, "_unknown_shape_type", 0);
    AttrUtils::SetStr(op2DescPtr, "opKernelLib", "HOSTCPUKernel");
    auto node3 = graphPtr->AddNode(op2DescPtr);

    ASSERT_EQ(hostCpuKernelBuilder.CalcOpRunningParam(*node3), SUCCESS);
    RunContext context1 = CreateContext();
    vector<domi::TaskDef> tasks1;
    ASSERT_EQ(hostCpuKernelBuilder.GenerateTask(*node3, context1, tasks1), SUCCESS);
    DestroyContext(context1);
    printf("end===================\n");
}

TEST(HostCpuOpsKernelBuilder, CalcOpRunningParam_Original_Unknown_FAIL)
{
   HostCpuOpsKernelBuilder hostCpuKernelBuilder;
    map<string, string> options;
    options[SOC_VERSION] = "Ascend910";
    ASSERT_EQ(hostCpuKernelBuilder.Initialize(options), SUCCESS);
    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_graph");
    OpDescPtr opDescPtr = make_shared<OpDesc>("placeholder", "PlaceHolder");

    vector<int64_t> tensorShape = {1, 2, 3, 4};
    GeTensorDesc tensor1(GeShape(tensorShape), ge::FORMAT_NCHW, ge::DT_INT32);
    tensor1.SetOriginFormat(ge::FORMAT_NCHW);
    tensor1.SetOriginShape(GeShape(tensorShape));

    opDescPtr->AddOutputDesc("z", tensor1);
    AttrUtils::SetInt(opDescPtr, "topic_type", 0);
    auto node1 = graphPtr->AddNode(opDescPtr);
    OpDescPtr op1DescPtr = make_shared<OpDesc>("FrameworkOpCase", kFrameworkOp);;

    op1DescPtr->AddOutputDesc("z", tensor1);
    AttrUtils::SetInt(op1DescPtr, "topic_type", 0);
    AttrUtils::SetInt(op1DescPtr, "_unknown_shape_type", 0);
    AttrUtils::SetStr(op1DescPtr, "opKernelLib", "HOSTCPUKernel");
    ge::OpDescUtilsEx::SetType(op1DescPtr, kFrameworkOp);
    AttrUtils::SetStr(op1DescPtr, kOriginalType, "");
    auto node2 = graphPtr->AddNode(op1DescPtr);

    node2->AddLinkFrom(node1);
    ASSERT_NE(hostCpuKernelBuilder.CalcOpRunningParam(*node2), SUCCESS);
}
