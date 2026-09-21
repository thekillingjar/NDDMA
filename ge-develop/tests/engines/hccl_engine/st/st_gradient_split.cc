/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include <mockcpp/mockcpp.hpp>
#include <stdio.h>

#define private public
#define protected public
#include "hcom_all_reduce_fusion.h"
#include "plugin_manager.h"
#include "hcom_graph_optimizer.h"
#include "auto_tuning_plugin.h"
#include "auto_tuning_hcom_ops_kernel_info_store.h"
#include "auto_tuning_hcom_ops_kernel_builder.h"
#include "hcom_ops_kernel_info_store.h"
#include "ops_kernel_info_store_base.h"
#include "hcom_ops_stores.h"
#include "auto_tuning_hcom_all_reduce_fusion.h"
#include "tuning_utils.h"
#include "hcom_op_utils.h"
#undef private
#undef protected
#include "hccl_stub.h"
#include <stdlib.h>
#include <pthread.h>
#include <runtime/rt.h>
#include <iostream>
#include <fstream>
#include "llt_hccl_stub_ge.h"
#include "llt_hccl_stub_hccl.h"

constexpr u32 INVALID_QOSCFG = 0xFFFFFFFF;

using namespace std;
using namespace hccl;

class HcomGradientSplitTuneTest : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "\033[36m--HcomGradientSplitTuneTest SetUP--\033[0m" << std::endl;
    }
    static void TearDownTestCase()
    {

        std::cout << "\033[36m--HcomGradientSplitTuneTest TearDown--\033[0m" << std::endl;
    }
    // Some expensive resource shared by all tests.
    virtual void SetUp()
    {
        std::cout << "A Test SetUP" << std::endl;
    }
    virtual void TearDown()
    {
        std::cout << "A Test TearDown" << std::endl;
    }
};

class NodeTest : public ge::Node {
public:
    NodeTest(){;};
    ~NodeTest(){;};    
};
TEST_F(HcomGradientSplitTuneTest, st_Gradient_AutoTuning_E2E_Test_Profiling)
{
    bool bErr = false;
    std::map<std::string,std::string> options;
    options.insert(pair<string,string> (ge::BUILD_MODE,"tuning"));
    options.insert(pair<string,string> (ge::TUNING_PATH,"./"));
    options.insert(pair<string,string> ("ge.jobType","4"));
    options.insert(pair<string,string> (ge::OPTION_EXEC_PROFILING_MODE,"1"));
    options.insert(pair<string,string> (ge::OPTION_EXEC_PROFILING_OPTIONS,"/tmp"));
    ge::Status ge_ret;
    HCCL_INFO("test Initialize.");
    ge_ret = Initialize(options);
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    std::map<string, OpsKernelInfoStorePtr> opKernInfos;
    std::map<string, GraphOptimizerPtr> graphOptimizers;
    GetGraphOptimizerObjs(graphOptimizers);
    GetOpsKernelInfoStores(opKernInfos);
    auto iter1 = graphOptimizers.find(HCCL_GRAPH_OPTIMIZER_NAME);
    if (iter1 == graphOptimizers.end()) {
        bErr = true;
    }
    EXPECT_EQ(bErr, false);
    if (bErr) {
        return;
    }
    auto iter2 = opKernInfos.find(AUTOTUNE_HCCL_OPS_LIB_NAME);
    if (iter2 == opKernInfos.end()) {
        bErr = true;
    }
    EXPECT_EQ(bErr, false);
    if (bErr) {
        return;
    }

    OpsKernelInfoStorePtr opsKernerInfoStorePtr = opKernInfos[AUTOTUNE_HCCL_OPS_LIB_NAME];
    GraphOptimizerPtr graphOptimizerPtr = graphOptimizers[HCCL_GRAPH_OPTIMIZER_NAME];

    HCCL_INFO("test opsKernerInfoStorePtr Initialize.");
    ge_ret = opsKernerInfoStorePtr->Initialize(options);
    EXPECT_EQ(ge_ret, ge::SUCCESS);
    HCCL_INFO("test graphOptimizerPtr Initialize.");
    ge_ret = graphOptimizerPtr->Initialize(options, nullptr);
    EXPECT_EQ(ge_ret, ge::INTERNAL_ERROR);
}

TEST_F(HcomGradientSplitTuneTest, st_Gradient_AutoTuning_E2E)
{
    bool bErr = false;
    std::map<std::string,std::string> options;
    options.insert(pair<string,string> (ge::BUILD_MODE,"tuning"));
    options.insert(pair<string,string> (ge::TUNING_PATH,"./"));
    options.insert(pair<string,string> ("ge.jobType","4"));

    ge::Status ge_ret;
    HCCL_INFO("test Initialize.");
    ge_ret = Initialize(options);
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    std::map<string, OpsKernelInfoStorePtr> opKernInfos;
    std::map<string, GraphOptimizerPtr> graphOptimizers;
    GetGraphOptimizerObjs(graphOptimizers);
    GetOpsKernelInfoStores(opKernInfos);
    auto iter1 = graphOptimizers.find(HCCL_GRAPH_OPTIMIZER_NAME);
    if (iter1 == graphOptimizers.end()) {
        bErr = true;
    }
    EXPECT_EQ(bErr, false);
    if (bErr) {
        return;
    }
    auto iter2 = opKernInfos.find(AUTOTUNE_HCCL_OPS_LIB_NAME);
    if (iter2 == opKernInfos.end()) {
        bErr = true;
    }
    EXPECT_EQ(bErr, false);
    if (bErr) {
        return;
    }

    OpsKernelInfoStorePtr opsKernerInfoStorePtr = opKernInfos[AUTOTUNE_HCCL_OPS_LIB_NAME];
    GraphOptimizerPtr graphOptimizerPtr = graphOptimizers[HCCL_GRAPH_OPTIMIZER_NAME];

    HCCL_INFO("test opsKernerInfoStorePtr Initialize.");
    ge_ret = opsKernerInfoStorePtr->Initialize(options);
    EXPECT_EQ(ge_ret, ge::SUCCESS);
    HCCL_INFO("test graphOptimizerPtr Initialize.");
    ge_ret = graphOptimizerPtr->Initialize(options, nullptr);
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    ge::ComputeGraphPtr compute_graph = std::make_shared<ge::ComputeGraph>("test_graph");
    ge::OpDescPtr opDescPtr = nullptr;
    ge::OpDesc op;
    ge::NodePtr node1 = compute_graph->AddNode(opDescPtr);
    ge::NodePtr node2 = compute_graph->AddNode(opDescPtr);
    node1->GetOutDataAnchor(0)->LinkTo(node2->GetInDataAnchor(0));
    node2->GetOutDataAnchor(0)->LinkTo(node1->GetInDataAnchor(0));
    const string type = HCCL_KERNEL_OP_TYPE_ALLREDUCE;
    op.SetType(type); 
    ge::AttrUtils::HasAttr(opDescPtr, "DUMMY_SET_TRUE_GROUP"); 
    ge::AttrUtils::HasAttr(opDescPtr, "group");
    ge::AttrUtils::SetStr(opDescPtr, "reduction", "sum"); 
    ge::AttrUtils::SetStr(opDescPtr, "group", "hccl_world_group"); 
    ge::AttrUtils::SetInt(opDescPtr, "fusion", 2);
    ge::AttrUtils::SetInt(opDescPtr, "fusion_id", 2);

    HCCL_INFO("test OptimizeGraphPrepare.");
    ge_ret = graphOptimizerPtr->OptimizeGraphPrepare(*compute_graph);
    EXPECT_EQ(ge_ret, ge::SUCCESS);
    HCCL_INFO("test OptimizeGraphPrepare.");
    std::vector<GradientDataInfo> recordInfos(2);
    recordInfos[0].dataSize = 1024;
    recordInfos[0].dataType = "float";
    recordInfos[0].graphId = 0;
    recordInfos[0].groupName = "hccl_world_group";
    recordInfos[0].gradientNodeName = "test_gradientNodeName_0";
    recordInfos[0].traceNodeName = "test_traceNodeName_0";
    recordInfos[0].allReduceNodeName = "test_allReduceNodeName_0";
    recordInfos[1].dataSize = 2048;
    recordInfos[1].dataType = "float";
    recordInfos[1].graphId = 0;
    recordInfos[1].groupName = "hccl_world_group";
    recordInfos[1].gradientNodeName = "test_gradientNodeName_1";
    recordInfos[1].traceNodeName = "test_traceNodeName_1";
    recordInfos[1].allReduceNodeName = "test_allReduceNodeName_1";

    MOCKER_CPP(&AutoTuningHcomAllReduceFusion::SetGradientInformation)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&AutoTuningHcomAllReduceFusion::RecordGradientDataInfo)
    .stubs()
    .with(mockcpp::any(), outBound(recordInfos))
    .will(returnValue(HCCL_SUCCESS));
    ge_ret = graphOptimizerPtr->OptimizeOriginalGraph(*compute_graph);
    ge::AttrUtils::HasAttr(opDescPtr, "DUMMY_SET_FALSE_GROUP"); 
    GlobalMockObject::verify();
    EXPECT_EQ(ge_ret, ge::SUCCESS);
    EXPECT_EQ(recordInfos.size(), 2);
    HCCL_INFO("test OptimizeFusedGraph.");

    ge_ret = graphOptimizerPtr->OptimizeFusedGraph(*compute_graph);
    EXPECT_EQ(ge_ret, ge::SUCCESS);
    HCCL_INFO("test OptimizeWholeGraph.");
    ge_ret = graphOptimizerPtr->OptimizeWholeGraph(*compute_graph);
    EXPECT_EQ(ge_ret, ge::SUCCESS);
    HCCL_INFO("test GetAttributes.");
    ge::GraphOptimizerAttribute attrs;
    ge_ret = graphOptimizerPtr->GetAttributes(attrs);
    EXPECT_EQ(ge_ret, ge::SUCCESS);
    EXPECT_EQ(attrs.engineName, HCCL_OPS_ENGIN);
    EXPECT_EQ(attrs.scope, ge::UNIT);

    HCCL_INFO("test CheckSupported.");
    std::string reason;
    bool bRet = opsKernerInfoStorePtr->CheckSupported(node1->GetOpDesc(), reason);
    EXPECT_EQ(bRet, true);

    HCCL_INFO("test GetAllOpsKernelInfo.");
    std::map<string, ge::OpInfo> infos;
    opsKernerInfoStorePtr->GetAllOpsKernelInfo(infos);
    EXPECT_EQ(infos.size(), AUTO_TUNING_HCOM_SUPPORTED_OP_TYPE.size());

    AutoTuningHcomOpsKernelBuilder autoTuningHcomOpsKernelBuilder;
    HCCL_INFO("test CalcOpRunningParam.");
    ge_ret = autoTuningHcomOpsKernelBuilder.CalcOpRunningParam(*node1);
    EXPECT_EQ(ge_ret, ge::SUCCESS);
    int64_t streamNum;
    ge::AttrUtils::GetInt(node1->GetOpDesc(), "used_stream_num", streamNum);
    EXPECT_EQ(streamNum, 0);
    std::vector<int64_t> workSpaceBytes = node1->GetOpDesc()->GetWorkspaceBytes();
    EXPECT_EQ(workSpaceBytes[0], 0);

    HCCL_INFO("test GenerateTask.");
    ge::Buffer tempBuffer;
    ge::RunContext runContext_dummy;
    std::vector<domi::TaskDef> taskDefList;
    int64_t streamId = 10000;
    node1->GetOpDesc()->SetStreamId((s64)streamId);
    ge_ret = autoTuningHcomOpsKernelBuilder.GenerateTask(*node1, runContext_dummy, taskDefList);
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    u32 result_type = taskDefList[0].type();
    u32 result_stream_id = taskDefList[0].stream_id();
    std::string result_hccl_hccl_type = taskDefList[0].mutable_kernel_hccl()->hccl_type();
    EXPECT_EQ(result_type, RT_MODEL_TASK_HCCL);
    EXPECT_EQ(result_stream_id, streamId);
    EXPECT_EQ(result_hccl_hccl_type, HCCL_KERNEL_OP_TYPE_ALLREDUCE);

    HCCL_INFO("test LoadTask.");
    s8* sendbuf = (s8*)sal_malloc(10 * sizeof(float));
    sal_memset(sendbuf, 10 * sizeof(float), 0, 10 * sizeof(float));
    s8* recv = (s8*)sal_malloc(10 * sizeof(float));
    sal_memset(recv, 10 * sizeof(float), 0, 10 * sizeof(float));
    rtStream_t stream;
    rtError_t rt_ret = rtStreamCreate(&stream, 5);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE); 

    ge::GETaskInfo task;
    ge::GETaskKernelHcclInfo hcclInfo;
    task.kernelHcclInfo.push_back(hcclInfo); 
    task.kernelHcclInfo[0].count=10;
    task.kernelHcclInfo[0].dataType=HCCL_DATA_TYPE_FP32;
    task.kernelHcclInfo[0].hccl_type=HCCL_KERNEL_OP_TYPE_ALLREDUCE;    
    task.kernelHcclInfo[0].inputDataAddr=sendbuf;
    task.kernelHcclInfo[0].outputDataAddr=recv;
    task.kernelHcclInfo[0].opType=HCCL_REDUCE_SUM;
    task.kernelHcclInfo[0].rootId=0;
    task.kernelHcclInfo[0].hcclQosCfg=INVALID_QOSCFG;
    task.type = RT_MODEL_TASK_HCCL;
    task.stream = stream;
    AUTO_TUNING_HCCL_KERNEL_INFO_PRIVATE_DEF privateDefBuf = {2, 0, HCCL_DATA_TYPE_INT8}; 
    task.privateDef = (void *)&privateDefBuf.rankSize;
    task.privateDefLen = sizeof(AUTO_TUNING_HCCL_KERNEL_INFO_PRIVATE_DEF); 

    ge_ret = opsKernerInfoStorePtr->LoadTask(task);
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    HCCL_INFO("test opsKernerInfoStorePtr Finalize.");
    ge_ret = opsKernerInfoStorePtr->Finalize();
    EXPECT_EQ(ge_ret, ge::SUCCESS);
    HCCL_INFO("test graphOptimizerPtr Finalize.");
    ge_ret = graphOptimizerPtr->Finalize();
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    HCCL_INFO("test Finalize.");
    ge_ret = Finalize();
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    rt_ret = rtStreamDestroy(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    sal_free(sendbuf);
    sal_free(recv); 

    std::string summaryPath = "gradient_summary.csv";
    remove(summaryPath.c_str());
}

TEST_F(HcomGradientSplitTuneTest, st_gradient_HcomOpsKernelInfoStore_AllGather_load)
{
    ge::Status ge_ret;
    ge::GETaskInfo task;
    rtStream_t stream; 
    
    AutoTuningHcomOpsKernelInfoStore athcomOpsKernelInfo;

    s8* sendbuf = (s8*)sal_malloc(10 * sizeof(float));
    HCCL_INFO("sendbuf is [%p]", sendbuf);
    sal_memset(sendbuf, 10 * sizeof(float), 0, 10 * sizeof(float));
    s8* recv = (s8*)sal_malloc(20 * sizeof(float));
    HCCL_INFO("recv is [%p]", recv);
    sal_memset(recv, 20 * sizeof(float), 0, 20 * sizeof(float));
    rtError_t rt_ret = rtStreamCreate(&stream, 5);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE); 
    ge::GETaskKernelHcclInfo hcclInfo;
    task.kernelHcclInfo.push_back(hcclInfo); 
    task.kernelHcclInfo[0].count=10;
    task.kernelHcclInfo[0].dataType=HCCL_DATA_TYPE_FP32;
    task.kernelHcclInfo[0].hccl_type=HCCL_KERNEL_OP_TYPE_ALLGATHER;    
    task.kernelHcclInfo[0].inputDataAddr=sendbuf;
    task.kernelHcclInfo[0].outputDataAddr=recv;
    task.kernelHcclInfo[0].opType=HCCL_REDUCE_SUM;
    task.kernelHcclInfo[0].rootId=0;
    task.kernelHcclInfo[0].hcclQosCfg=INVALID_QOSCFG; 
    task.type = RT_MODEL_TASK_HCCL;
    task.stream = stream;
    AUTO_TUNING_HCCL_KERNEL_INFO_PRIVATE_DEF privateDefBuf = {2, 0, HCCL_DATA_TYPE_INT8}; 
    task.privateDef = (void *)&privateDefBuf.rankSize;
    task.privateDefLen = sizeof(AUTO_TUNING_HCCL_KERNEL_INFO_PRIVATE_DEF); 

    ge_ret = athcomOpsKernelInfo.LoadTask(task);
    EXPECT_EQ(ge_ret, ge::SUCCESS);
    
    rt_ret = rtStreamDestroy(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    sal_free(sendbuf);
    sal_free(recv); 
}

TEST_F(HcomGradientSplitTuneTest, st_gradient_HcomOpsKernelInfoStore_AllReduce_load)
{
    ge::Status ge_ret;
    ge::GETaskInfo task;
    rtStream_t stream; 
    AutoTuningHcomOpsKernelInfoStore athcomOpsKernelInfo;

    s8* sendbuf = (s8*)sal_malloc(10 * sizeof(float));
    sal_memset(sendbuf, 10 * sizeof(float), 0, 10 * sizeof(float));
    s8* recv = (s8*)sal_malloc(10 * sizeof(float));
    sal_memset(recv, 10 * sizeof(float), 0, 10 * sizeof(float));
    rtError_t rt_ret = rtStreamCreate(&stream, 5);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE); 
    ge::GETaskKernelHcclInfo hcclInfo;
    task.kernelHcclInfo.push_back(hcclInfo); 
    task.kernelHcclInfo[0].count=10;
    task.kernelHcclInfo[0].dataType=HCCL_DATA_TYPE_FP32;
    task.kernelHcclInfo[0].hccl_type=HCCL_KERNEL_OP_TYPE_ALLREDUCE;    
    task.kernelHcclInfo[0].inputDataAddr=sendbuf;
    task.kernelHcclInfo[0].outputDataAddr=recv;
    task.kernelHcclInfo[0].opType=HCCL_REDUCE_SUM;
    task.kernelHcclInfo[0].rootId=0;
    task.kernelHcclInfo[0].hcclQosCfg=INVALID_QOSCFG; 
    task.type = RT_MODEL_TASK_HCCL;
    task.stream = stream;
    
    AUTO_TUNING_HCCL_KERNEL_INFO_PRIVATE_DEF privateDefBuf = {2, 0, HCCL_DATA_TYPE_INT8}; 
    task.privateDef = (void *)&privateDefBuf.rankSize;
    task.privateDefLen = sizeof(AUTO_TUNING_HCCL_KERNEL_INFO_PRIVATE_DEF); 

    ge_ret = athcomOpsKernelInfo.LoadTask(task);
    EXPECT_EQ(ge_ret, ge::SUCCESS);
    rt_ret = rtStreamDestroy(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    sal_free(sendbuf);
    sal_free(recv); 
}

TEST_F(HcomGradientSplitTuneTest, st_gradient_HcomOpsKernelInfoStore_Broadcast_load)
{
    ge::Status ge_ret;
    ge::GETaskInfo task;
    rtStream_t stream; 
    AutoTuningHcomOpsKernelInfoStore athcomOpsKernelInfo;

    s8* sendbuf = (s8*)sal_malloc(10 * sizeof(float));
    sal_memset(sendbuf, 10 * sizeof(float), 0, 10 * sizeof(float));
    s8* recv = (s8*)sal_malloc(10 * sizeof(float));
    sal_memset(recv, 10 * sizeof(float), 0, 10 * sizeof(float));
    rtError_t rt_ret = rtStreamCreate(&stream, 5);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE); 
    ge::GETaskKernelHcclInfo hcclInfo;
    task.kernelHcclInfo.push_back(hcclInfo); 
    task.kernelHcclInfo[0].count=10;
    task.kernelHcclInfo[0].dataType=HCCL_DATA_TYPE_FP32;
    task.kernelHcclInfo[0].hccl_type=HCCL_KERNEL_OP_TYPE_BROADCAST;
    task.kernelHcclInfo[0].inputDataAddr=sendbuf;
    task.kernelHcclInfo[0].outputDataAddr=recv;
    task.kernelHcclInfo[0].opType=HCCL_REDUCE_SUM;
    task.kernelHcclInfo[0].rootId=0;
    task.kernelHcclInfo[0].hcclQosCfg=INVALID_QOSCFG; 
    task.type = RT_MODEL_TASK_HCCL;
    task.stream = stream;
    
    AUTO_TUNING_HCCL_KERNEL_INFO_PRIVATE_DEF privateDefBuf = {2, 0, HCCL_DATA_TYPE_INT8}; 
    task.privateDef = (void *)&privateDefBuf.rankSize;
    task.privateDefLen = sizeof(AUTO_TUNING_HCCL_KERNEL_INFO_PRIVATE_DEF); 

    ge_ret = athcomOpsKernelInfo.LoadTask(task);
    EXPECT_EQ(ge_ret, ge::SUCCESS);
    rt_ret = rtStreamDestroy(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    sal_free(sendbuf);
    sal_free(recv); 
}

TEST_F(HcomGradientSplitTuneTest, st_gradient_HcomOpsKernelInfoStore_AlltoAllx_load)
{
    ge::Status ge_ret;
    ge::GETaskInfo task;
    rtStream_t stream; 
    AutoTuningHcomOpsKernelInfoStore athcomOpsKernelInfo;

    s8* sendbuf = (s8*)sal_malloc(10 * sizeof(float));
    sal_memset(sendbuf, 10 * sizeof(float), 0, 10 * sizeof(float));
    s8* recv = (s8*)sal_malloc(10 * sizeof(float));
    sal_memset(recv, 10 * sizeof(float), 0, 10 * sizeof(float));
    rtError_t rt_ret = rtStreamCreate(&stream, 5);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE); 
    ge::GETaskKernelHcclInfo hcclInfo;
    task.kernelHcclInfo.push_back(hcclInfo); 
    task.kernelHcclInfo[0].count=10;
    task.kernelHcclInfo[0].dataType=HCCL_DATA_TYPE_FP32;
    task.kernelHcclInfo[0].hccl_type=HCCL_KERNEL_OP_TYPE_ALLTOALLV;
    task.kernelHcclInfo[0].inputDataAddr=sendbuf;
    task.kernelHcclInfo[0].outputDataAddr=recv;
    task.kernelHcclInfo[0].opType=HCCL_REDUCE_SUM;
    task.kernelHcclInfo[0].rootId=0;
    task.kernelHcclInfo[0].hcclQosCfg=INVALID_QOSCFG; 
    task.type = RT_MODEL_TASK_HCCL;
    task.stream = stream;
    
    AUTO_TUNING_HCCL_KERNEL_INFO_PRIVATE_DEF privateDefBuf = {2, 0}; 
    task.privateDef = (void *)&privateDefBuf.rankSize;
    task.privateDefLen = sizeof(AUTO_TUNING_HCCL_KERNEL_INFO_PRIVATE_DEF); 

    ge_ret = athcomOpsKernelInfo.LoadTask(task);
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    task.kernelHcclInfo[0].hccl_type=HCCL_KERNEL_OP_TYPE_ALLTOALLVC;
    ge_ret = athcomOpsKernelInfo.LoadTask(task);
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    task.kernelHcclInfo[0].hccl_type=HCCL_KERNEL_OP_TYPE_ALLTOALL;
    ge_ret = athcomOpsKernelInfo.LoadTask(task);
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    rt_ret = rtStreamDestroy(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    sal_free(sendbuf);
    sal_free(recv); 
}

TEST_F(HcomGradientSplitTuneTest, st_gradient_HcomOpsKernelInfoStore_InitAlltoAllHostMem)
{
    AutoTuningHcomOpsKernelInfoStore athcomOpsKernelInfo;
    HcclDataType dateType;
    u64 dataBytes = 128;
    HostMem hostMem = HostMem::alloc(dataBytes);    
    uint32_t unitSize = 2;
    u64 recvCount = 0;
    dateType = HCCL_DATA_TYPE_INT8; //1
    EXPECT_EQ(HCCL_SUCCESS, SalGetDataTypeSize(dateType, unitSize));
    recvCount = dataBytes/unitSize;
    EXPECT_EQ(HCCL_SUCCESS, athcomOpsKernelInfo.InitAlltoAllHostMem(dateType, recvCount, hostMem.ptr()));

    dateType = HCCL_DATA_TYPE_INT16; //2
    EXPECT_EQ(HCCL_SUCCESS, SalGetDataTypeSize(dateType, unitSize));
    recvCount = dataBytes/unitSize;
    EXPECT_EQ(HCCL_SUCCESS, athcomOpsKernelInfo.InitAlltoAllHostMem(dateType, recvCount, hostMem.ptr()));

    dateType = HCCL_DATA_TYPE_INT32; //3
    EXPECT_EQ(HCCL_SUCCESS, SalGetDataTypeSize(dateType, unitSize));
    recvCount = dataBytes/unitSize;
    EXPECT_EQ(HCCL_SUCCESS, athcomOpsKernelInfo.InitAlltoAllHostMem(dateType, recvCount, hostMem.ptr()));

    dateType = HCCL_DATA_TYPE_FP16; //4
    EXPECT_EQ(HCCL_SUCCESS, SalGetDataTypeSize(dateType, unitSize));
    recvCount = dataBytes/unitSize;
    EXPECT_EQ(HCCL_SUCCESS, athcomOpsKernelInfo.InitAlltoAllHostMem(dateType, recvCount, hostMem.ptr()));

    dateType = HCCL_DATA_TYPE_FP32; //5
    EXPECT_EQ(HCCL_SUCCESS, SalGetDataTypeSize(dateType, unitSize));
    recvCount = dataBytes/unitSize;
    EXPECT_EQ(HCCL_SUCCESS, athcomOpsKernelInfo.InitAlltoAllHostMem(dateType, recvCount, hostMem.ptr()));

    dateType = HCCL_DATA_TYPE_INT64; //6
    EXPECT_EQ(HCCL_SUCCESS, SalGetDataTypeSize(dateType, unitSize));
    recvCount = dataBytes/unitSize;
    EXPECT_EQ(HCCL_SUCCESS, athcomOpsKernelInfo.InitAlltoAllHostMem(dateType, recvCount, hostMem.ptr()));

    dateType = HCCL_DATA_TYPE_UINT64; //7
    EXPECT_EQ(HCCL_SUCCESS, SalGetDataTypeSize(dateType, unitSize));
    recvCount = dataBytes/unitSize;
    EXPECT_EQ(HCCL_SUCCESS, athcomOpsKernelInfo.InitAlltoAllHostMem(dateType, recvCount, hostMem.ptr()));

    dateType = HCCL_DATA_TYPE_UINT8; //8
    EXPECT_EQ(HCCL_SUCCESS, SalGetDataTypeSize(dateType, unitSize));
    recvCount = dataBytes/unitSize;
    EXPECT_EQ(HCCL_SUCCESS, athcomOpsKernelInfo.InitAlltoAllHostMem(dateType, recvCount, hostMem.ptr()));

    dateType = HCCL_DATA_TYPE_UINT16; //9
    EXPECT_EQ(HCCL_SUCCESS, SalGetDataTypeSize(dateType, unitSize));
    recvCount = dataBytes/unitSize;
    EXPECT_EQ(HCCL_SUCCESS, athcomOpsKernelInfo.InitAlltoAllHostMem(dateType, recvCount, hostMem.ptr()));

    dateType = HCCL_DATA_TYPE_UINT32; //10
    EXPECT_EQ(HCCL_SUCCESS, SalGetDataTypeSize(dateType, unitSize));
    recvCount = dataBytes/unitSize;
    EXPECT_EQ(HCCL_SUCCESS, athcomOpsKernelInfo.InitAlltoAllHostMem(dateType, recvCount, hostMem.ptr()));

    dateType = HCCL_DATA_TYPE_FP64; //11
    EXPECT_EQ(HCCL_SUCCESS, SalGetDataTypeSize(dateType, unitSize));
    recvCount = dataBytes/unitSize;
    EXPECT_EQ(HCCL_SUCCESS, athcomOpsKernelInfo.InitAlltoAllHostMem(dateType, recvCount, hostMem.ptr()));

    dateType = HCCL_DATA_TYPE_BFP16; //12
    EXPECT_EQ(HCCL_SUCCESS, SalGetDataTypeSize(dateType, unitSize));
    recvCount = dataBytes/unitSize;
    EXPECT_EQ(HCCL_SUCCESS, athcomOpsKernelInfo.InitAlltoAllHostMem(dateType, recvCount, hostMem.ptr()));

    dateType = HCCL_DATA_TYPE_RESERVED; //13
    EXPECT_EQ(HCCL_E_PARA, SalGetDataTypeSize(dateType, unitSize));
    recvCount = dataBytes/unitSize;
    EXPECT_EQ(HCCL_E_NOT_SUPPORT, athcomOpsKernelInfo.InitAlltoAllHostMem(dateType, recvCount, hostMem.ptr()));
}

TEST_F(HcomGradientSplitTuneTest, st_gradient_HcomOpsKernelInfoStore_Reduce_load)
{
    ge::Status ge_ret;
    ge::GETaskInfo task;
    rtStream_t stream; 
    AutoTuningHcomOpsKernelInfoStore athcomOpsKernelInfo;

    s8* sendbuf = (s8*)sal_malloc(10 * sizeof(float));
    sal_memset(sendbuf, 10 * sizeof(float), 0, 10 * sizeof(float));
    s8* recv = (s8*)sal_malloc(10 * sizeof(float));
    sal_memset(recv, 10 * sizeof(float), 0, 10 * sizeof(float));
    rtError_t rt_ret = rtStreamCreate(&stream, 5);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE); 
    ge::GETaskKernelHcclInfo hcclInfo;
    task.kernelHcclInfo.push_back(hcclInfo); 
    task.kernelHcclInfo[0].count=10;
    task.kernelHcclInfo[0].dataType=HCCL_DATA_TYPE_FP32;
    task.kernelHcclInfo[0].hccl_type=HCCL_KERNEL_OP_TYPE_REDUCE;
    task.kernelHcclInfo[0].inputDataAddr=sendbuf;
    task.kernelHcclInfo[0].outputDataAddr=recv;
    task.kernelHcclInfo[0].opType=HCCL_REDUCE_SUM;
    task.kernelHcclInfo[0].rootId=0;
    task.kernelHcclInfo[0].hcclQosCfg=INVALID_QOSCFG; 
    task.type = RT_MODEL_TASK_HCCL;
    task.stream = stream;
    
    AUTO_TUNING_HCCL_KERNEL_INFO_PRIVATE_DEF privateDefBuf = {2, 0, HCCL_DATA_TYPE_INT8}; 
    task.privateDef = (void *)&privateDefBuf.rankSize;
    task.privateDefLen = sizeof(AUTO_TUNING_HCCL_KERNEL_INFO_PRIVATE_DEF); 

    ge_ret = athcomOpsKernelInfo.LoadTask(task);
    EXPECT_EQ(ge_ret, ge::SUCCESS);
    rt_ret = rtStreamDestroy(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    sal_free(sendbuf);
    sal_free(recv); 
}

TEST_F(HcomGradientSplitTuneTest, st_gradient_HcomOpsKernelInfoStore_ReduceScatter_load)
{
    ge::Status ge_ret;
    ge::GETaskInfo task;
    rtStream_t stream; 
    AutoTuningHcomOpsKernelInfoStore athcomOpsKernelInfo;

    s8* sendbuf = (s8*)sal_malloc(10 * sizeof(float));
    sal_memset(sendbuf, 10 * sizeof(float), 0, 10 * sizeof(float));
    s8* recv = (s8*)sal_malloc(10 * sizeof(float));
    sal_memset(recv, 10 * sizeof(float), 0, 10 * sizeof(float));
    rtError_t rt_ret = rtStreamCreate(&stream, 5);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE); 
    ge::GETaskKernelHcclInfo hcclInfo;
    task.kernelHcclInfo.push_back(hcclInfo); 
    task.kernelHcclInfo[0].count=10;
    task.kernelHcclInfo[0].dataType=HCCL_DATA_TYPE_FP32;
    task.kernelHcclInfo[0].hccl_type=HCCL_KERNEL_OP_TYPE_REDUCESCATTER;
    task.kernelHcclInfo[0].inputDataAddr=sendbuf;
    task.kernelHcclInfo[0].outputDataAddr=recv;
    task.kernelHcclInfo[0].opType=HCCL_REDUCE_SUM;
    task.kernelHcclInfo[0].rootId=0;
    task.kernelHcclInfo[0].hcclQosCfg=INVALID_QOSCFG; 
    task.type = RT_MODEL_TASK_HCCL;
    task.stream = stream;
    
    AUTO_TUNING_HCCL_KERNEL_INFO_PRIVATE_DEF privateDefBuf = {2, 0, HCCL_DATA_TYPE_INT8}; 
    task.privateDef = (void *)&privateDefBuf.rankSize;
    task.privateDefLen = sizeof(AUTO_TUNING_HCCL_KERNEL_INFO_PRIVATE_DEF); 

    ge_ret = athcomOpsKernelInfo.LoadTask(task);
    EXPECT_EQ(ge_ret, ge::SUCCESS);
    rt_ret = rtStreamDestroy(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    sal_free(sendbuf);
    sal_free(recv); 
}

TEST_F(HcomGradientSplitTuneTest, st_gradient_HcomOpsKernelInfoStore_Send_load)
{
    ge::Status ge_ret;
    ge::GETaskInfo task;
    rtStream_t stream; 
    AutoTuningHcomOpsKernelInfoStore athcomOpsKernelInfo;

    s8* sendbuf = (s8*)sal_malloc(10 * sizeof(float));
    sal_memset(sendbuf, 10 * sizeof(float), 0, 10 * sizeof(float));
    s8* recv = (s8*)sal_malloc(10 * sizeof(float));
    sal_memset(recv, 10 * sizeof(float), 0, 10 * sizeof(float));
    rtError_t rt_ret = rtStreamCreate(&stream, 5);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE); 
    ge::GETaskKernelHcclInfo hcclInfo;
    task.kernelHcclInfo.push_back(hcclInfo); 
    task.kernelHcclInfo[0].count=10;
    task.kernelHcclInfo[0].dataType=HCCL_DATA_TYPE_FP32;
    task.kernelHcclInfo[0].hccl_type=HCCL_KERNEL_OP_TYPE_SEND;
    task.kernelHcclInfo[0].inputDataAddr=sendbuf;
    task.kernelHcclInfo[0].outputDataAddr=recv;
    task.kernelHcclInfo[0].opType=HCCL_REDUCE_SUM;
    task.kernelHcclInfo[0].rootId=0;
    task.kernelHcclInfo[0].hcclQosCfg=INVALID_QOSCFG; 
    task.type = RT_MODEL_TASK_HCCL;
    task.stream = stream;
    
    AUTO_TUNING_HCCL_KERNEL_INFO_PRIVATE_DEF privateDefBuf = {2, 0, HCCL_DATA_TYPE_INT8}; 
    task.privateDef = (void *)&privateDefBuf.rankSize;
    task.privateDefLen = sizeof(AUTO_TUNING_HCCL_KERNEL_INFO_PRIVATE_DEF); 

    ge_ret = athcomOpsKernelInfo.LoadTask(task);
    EXPECT_EQ(ge_ret, ge::SUCCESS);
    rt_ret = rtStreamDestroy(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    sal_free(sendbuf);
    sal_free(recv); 
}

TEST_F(HcomGradientSplitTuneTest, st_gradient_HcomOpsKernelInfoStore_Receive_load)
{
    ge::Status ge_ret;
    ge::GETaskInfo task;
    rtStream_t stream; 
    AutoTuningHcomOpsKernelInfoStore athcomOpsKernelInfo;

    s8* sendbuf = (s8*)sal_malloc(10 * sizeof(float));
    sal_memset(sendbuf, 10 * sizeof(float), 0, 10 * sizeof(float));
    s8* recv = (s8*)sal_malloc(10 * sizeof(float));
    sal_memset(recv, 10 * sizeof(float), 0, 10 * sizeof(float));
    rtError_t rt_ret = rtStreamCreate(&stream, 5);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE); 
    ge::GETaskKernelHcclInfo hcclInfo;
    task.kernelHcclInfo.push_back(hcclInfo); 
    task.kernelHcclInfo[0].count=10;
    task.kernelHcclInfo[0].dataType=HCCL_DATA_TYPE_FP32;
    task.kernelHcclInfo[0].hccl_type=HCCL_KERNEL_OP_TYPE_RECEIVE;
    task.kernelHcclInfo[0].inputDataAddr=sendbuf;
    task.kernelHcclInfo[0].outputDataAddr=recv;
    task.kernelHcclInfo[0].opType=HCCL_REDUCE_SUM;
    task.kernelHcclInfo[0].rootId=0;
    task.kernelHcclInfo[0].hcclQosCfg=INVALID_QOSCFG; 
    task.type = RT_MODEL_TASK_HCCL;
    task.stream = stream;
    
    AUTO_TUNING_HCCL_KERNEL_INFO_PRIVATE_DEF privateDefBuf = {2, 0, HCCL_DATA_TYPE_INT8}; 
    task.privateDef = (void *)&privateDefBuf.rankSize;
    task.privateDefLen = sizeof(AUTO_TUNING_HCCL_KERNEL_INFO_PRIVATE_DEF); 

    ge_ret = athcomOpsKernelInfo.LoadTask(task);
    EXPECT_EQ(ge_ret, ge::SUCCESS);
    rt_ret = rtStreamDestroy(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    sal_free(sendbuf);
    sal_free(recv); 
}

TEST_F(HcomGradientSplitTuneTest, st_gradient_HcomGraphOptimizer){
    ge ::Status ge_ret;
    std::map<std::string,std::string> options;
    options.insert(pair<string,string> (ge::BUILD_MODE,"turning"));
    options.insert(pair<string,string> (ge::TUNING_PATH,"./"));
    options.insert(pair<string,string> ("ge.jobType","4"));
    ge::ComputeGraphPtr compute_graph = std::make_shared<ge::ComputeGraph>("test_graph");
    ge::OpDescPtr opDescPtr = nullptr;

    ge::OpDesc op;
    ge::NodePtr node1 = compute_graph->AddNode(opDescPtr);
    ge::NodePtr node2 = compute_graph->AddNode(opDescPtr);
    node1->GetOutDataAnchor(0)->LinkTo(node2->GetInDataAnchor(0));
    node2->GetOutDataAnchor(0)->LinkTo(node1->GetInDataAnchor(0));

    AutoTuningHcomGraphOptimizer hcomGraphOtimizer;
    ge_ret = hcomGraphOtimizer.Initialize(options, nullptr);
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    MOCKER_CPP(&AutoTuningHcomAllReduceFusion::RecordGradientDataInfo)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));

    const string type = "HcomAllReduce";
    op.SetType(type); 
    ge::AttrUtils::HasAttr(opDescPtr, "DUMMY_SET_TRUE_GROUP"); 
    ge::AttrUtils::SetStr(opDescPtr, "reduction", "sum"); 
    ge::AttrUtils::SetStr(opDescPtr, "group", "hccl_world_group"); 
    ge::AttrUtils::SetInt(opDescPtr, "fusion", 1); 
    ge_ret = hcomGraphOtimizer.OptimizeOriginalGraph(*compute_graph);
    ge::AttrUtils::HasAttr(opDescPtr, "DUMMY_SET_FALSE_GROUP");

    ge_ret = hcomGraphOtimizer.Finalize();
    EXPECT_EQ(ge_ret, ge::SUCCESS);
    std::string summaryPath = "gradient_summary.csv";
    remove(summaryPath.c_str());

    GlobalMockObject::verify();
}
TEST_F(HcomGradientSplitTuneTest, st_gradient_HcomGraphOptimizer_jobType){
    ge ::Status ge_ret;
    std::map<std::string,std::string> options;
    options.insert(pair<string,string> (ge::BUILD_MODE,"turning"));
    options.insert(pair<string,string> (ge::TUNING_PATH,"./"));
    options.insert(pair<string,string> ("ge.jobType","1"));
    ge::ComputeGraphPtr compute_graph = std::make_shared<ge::ComputeGraph>("test_graph");
    ge::OpDescPtr opDescPtr = nullptr;

    ge::OpDesc op;
    ge::NodePtr node1 = compute_graph->AddNode(opDescPtr);
    ge::NodePtr node2 = compute_graph->AddNode(opDescPtr);
    node1->GetOutDataAnchor(0)->LinkTo(node2->GetInDataAnchor(0));
    node2->GetOutDataAnchor(0)->LinkTo(node1->GetInDataAnchor(0));

    AutoTuningHcomGraphOptimizer hcomGraphOtimizer;
    ge_ret = hcomGraphOtimizer.Initialize(options, nullptr);
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    MOCKER_CPP(&AutoTuningHcomAllReduceFusion::RecordGradientDataInfo)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));

    const string type = "HcomAllReduce";
    op.SetType(type); 
    ge::AttrUtils::HasAttr(opDescPtr, "DUMMY_SET_TRUE_GROUP"); 
    ge::AttrUtils::SetStr(opDescPtr, "reduction", "sum"); 
    ge::AttrUtils::SetStr(opDescPtr, "group", "hccl_world_group"); 
    ge::AttrUtils::SetInt(opDescPtr, "fusion", 1); 
    ge_ret = hcomGraphOtimizer.OptimizeOriginalGraph(*compute_graph);
    ge::AttrUtils::HasAttr(opDescPtr, "DUMMY_SET_FALSE_GROUP");
}

TEST_F(HcomGradientSplitTuneTest, st_gradient_fusion_HcomGraphOptimizer_jobType){
    ge ::Status ge_ret;
    std::map<std::string,std::string> options;
    options.insert(pair<string,string> (ge::BUILD_MODE,"turning"));
    options.insert(pair<string,string> (ge::TUNING_PATH,"./"));
    options.insert(pair<string,string> ("ge.jobType","1"));
    options.insert(pair<string,string> (ge::FUSION_TENSOR_SIZE,"524288000"));
    ge::ComputeGraphPtr compute_graph = std::make_shared<ge::ComputeGraph>("test_graph");
    ge::OpDescPtr opDescPtr = nullptr;

    ge::OpDesc op;
    ge::NodePtr node1 = compute_graph->AddNode(opDescPtr);
    ge::NodePtr node2 = compute_graph->AddNode(opDescPtr);
    node1->GetOutDataAnchor(0)->LinkTo(node2->GetInDataAnchor(0));
    node2->GetOutDataAnchor(0)->LinkTo(node1->GetInDataAnchor(0));

    AutoTuningHcomGraphOptimizer hcomGraphOtimizer;
    ge_ret = hcomGraphOtimizer.Initialize(options, nullptr);
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    MOCKER_CPP(&AutoTuningHcomAllReduceFusion::RecordGradientDataInfo)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));

    const string type = "HcomAllReduce";
    op.SetType(type);
    ge::AttrUtils::HasAttr(opDescPtr, "DUMMY_SET_TRUE_GROUP");
    ge::AttrUtils::SetStr(opDescPtr, "reduction", "sum");
    ge::AttrUtils::SetStr(opDescPtr, "group", "hccl_world_group");
    ge::AttrUtils::SetInt(opDescPtr, "fusion", 1);
    ge_ret = hcomGraphOtimizer.OptimizeOriginalGraph(*compute_graph);
    ge::AttrUtils::HasAttr(opDescPtr, "DUMMY_SET_FALSE_GROUP");
}


TEST_F(HcomGradientSplitTuneTest, st_gradient_RecordGradientDataInfo)
{
    HcclResult ret;

    ge::ComputeGraphPtr compute_graph = std::make_shared<ge::ComputeGraph>("test_graph");
    ge::OpDescPtr opDescPtr = nullptr;

    ge::OpDesc op;
    ge::NodePtr node1 = compute_graph->AddNode(opDescPtr);
    ge::NodePtr node2 = compute_graph->AddNode(opDescPtr);
    node1->GetOutDataAnchor(0)->LinkTo(node2->GetInDataAnchor(0));
    node2->GetOutDataAnchor(0)->LinkTo(node1->GetInDataAnchor(0));

    std::vector<GradientDataInfo> recordInfos;
    const string type = "HcomAllReduce";
    op.SetType(type); 
    ge::AttrUtils::HasAttr(opDescPtr, "DUMMY_SET_TRUE_GROUP"); 
    ge::AttrUtils::SetStr(opDescPtr, "reduction", "sum"); 
    ge::AttrUtils::SetStr(opDescPtr, "group", "hccl_world_group"); 
    ge::AttrUtils::SetInt(opDescPtr, "fusion", 1); 
    AutoTuningHcomAllReduceFusion hcomAllReduceFusion;
    ret = hcomAllReduceFusion.RecordGradientDataInfo(*compute_graph, recordInfos);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    ge::AttrUtils::HasAttr(opDescPtr, "DUMMY_SET_FALSE_GROUP");
}

TEST_F(HcomGradientSplitTuneTest, st_gradient_PluginManager_one){
    ge ::Status ge_ret;
    std::map<std::string,std::string> options;
    options.insert(pair<string,string> (ge::BUILD_MODE,"tuning"));
    options.insert(pair<string,string> (ge::TUNING_PATH,"./"));

    ge_ret = Initialize(options);
    EXPECT_EQ(ge_ret, ge::SUCCESS);
    
    Finalize();
}

TEST_F(HcomGradientSplitTuneTest, st_gradient_PluginManager_two){
    ge ::Status ge_ret;
    std::map<std::string,std::string> options;

    options.insert(pair<string,string> (ge::BUILD_MODE,"normal"));
    ge_ret = Initialize(options);
    EXPECT_EQ(ge_ret, ge::SUCCESS);
    Finalize();
}

TEST_F(HcomGradientSplitTuneTest, st_gradient_PluginManager_three){
    ge ::Status ge_ret;
    std::map<std::string,std::string> options;
    options.insert(pair<string,string> (ge::BUILD_MODE,"tuning"));
    options.insert(pair<string,string> ("ge.jobType","4"));
    ge_ret = Initialize(options);
    EXPECT_EQ(ge_ret, ge::SUCCESS);
    
    Finalize();
}

TEST_F(HcomGradientSplitTuneTest, st_gradient_RecordGradientDataInfo_allreduce_unknownshape)
{
    HcclResult ret;

    ge::ComputeGraphPtr compute_graph = std::make_shared<ge::ComputeGraph>("test_graph");
    ge::OpDescPtr opDescPtr = nullptr;

    ge::OpDesc op;
    ge::NodePtr node1 = compute_graph->AddNode(opDescPtr);
    ge::NodePtr node2 = compute_graph->AddNode(opDescPtr);
    node1->GetOutDataAnchor(0)->LinkTo(node2->GetInDataAnchor(0));
    node2->GetOutDataAnchor(0)->LinkTo(node1->GetInDataAnchor(0));

    MOCKER(&ge::NodeUtils::GetNodeUnknownShapeStatus)
    .stubs()
    .with(outBound(true))
    .will(returnValue(ge::GRAPH_SUCCESS));

    std::vector<GradientDataInfo> recordInfos;
    const string type = HCCL_KERNEL_OP_TYPE_ALLREDUCE;
    op.SetType(type); 
    ge::AttrUtils::HasAttr(opDescPtr, "DUMMY_SET_TRUE_GROUP"); 
    ge::AttrUtils::SetStr(opDescPtr, "reduction", "sum"); 
    ge::AttrUtils::SetStr(opDescPtr, "group", "hccl_world_group"); 
    ge::AttrUtils::SetInt(opDescPtr, "fusion", 1); 
    AutoTuningHcomAllReduceFusion hcomAllReduceFusion;

    ret = hcomAllReduceFusion.RecordGradientDataInfo(*compute_graph, recordInfos);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    ge::AttrUtils::HasAttr(opDescPtr, "DUMMY_SET_FALSE_GROUP");

    EXPECT_EQ(recordInfos.size(), 0);
    GlobalMockObject::verify(); 
}

TEST_F(HcomGradientSplitTuneTest, st_gradient_RecordGradientDataInfo_allgather)
{
    HcclResult ret;

    ge::ComputeGraphPtr compute_graph = std::make_shared<ge::ComputeGraph>("test_graph");
    ge::OpDescPtr opDescPtr = nullptr;

    ge::OpDesc op;
    ge::NodePtr node1 = compute_graph->AddNode(opDescPtr);
    ge::NodePtr node2 = compute_graph->AddNode(opDescPtr);
    node1->GetOutDataAnchor(0)->LinkTo(node2->GetInDataAnchor(0));
    node2->GetOutDataAnchor(0)->LinkTo(node1->GetInDataAnchor(0));

    std::vector<GradientDataInfo> recordInfos;
    const string type = HCCL_KERNEL_OP_TYPE_ALLGATHER;
    op.SetType(type); 
    ge::AttrUtils::HasAttr(opDescPtr, "DUMMY_SET_TRUE_GROUP"); 
    ge::AttrUtils::SetStr(opDescPtr, "group", "hccl_world_group"); 
    AutoTuningHcomAllReduceFusion hcomAllReduceFusion;
    ret = hcomAllReduceFusion.RecordGradientDataInfo(*compute_graph, recordInfos);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    ge::AttrUtils::HasAttr(opDescPtr, "DUMMY_SET_FALSE_GROUP");

    EXPECT_EQ(recordInfos.size(), 0);
}

TEST_F(HcomGradientSplitTuneTest, st_gradient_RecordGradientDataInfo_broadcast)
{
    HcclResult ret;

    ge::ComputeGraphPtr compute_graph = std::make_shared<ge::ComputeGraph>("test_graph");
    ge::OpDescPtr opDescPtr = nullptr;

    ge::OpDesc op;
    ge::NodePtr node1 = compute_graph->AddNode(opDescPtr);
    ge::NodePtr node2 = compute_graph->AddNode(opDescPtr);
    node1->GetOutDataAnchor(0)->LinkTo(node2->GetInDataAnchor(0));
    node2->GetOutDataAnchor(0)->LinkTo(node1->GetInDataAnchor(0));
    std::vector<GradientDataInfo> recordInfos;
    const string type = HCCL_KERNEL_OP_TYPE_BROADCAST;
    op.SetType(type); 
    ge::AttrUtils::HasAttr(opDescPtr, "DUMMY_SET_TRUE_GROUP"); 
    ge::AttrUtils::SetStr(opDescPtr, "group", "hccl_world_group"); 
    AutoTuningHcomAllReduceFusion hcomAllReduceFusion;
    ret = hcomAllReduceFusion.RecordGradientDataInfo(*compute_graph, recordInfos);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    ge::AttrUtils::HasAttr(opDescPtr, "DUMMY_SET_FALSE_GROUP");

    EXPECT_EQ(recordInfos.size(), 0);
}
TEST_F(HcomGradientSplitTuneTest, st_gradient_RecordGradientDataInfo_reduce)
{
    HcclResult ret;

    ge::ComputeGraphPtr compute_graph = std::make_shared<ge::ComputeGraph>("test_graph");
    ge::OpDescPtr opDescPtr = nullptr;

    ge::OpDesc op;
    ge::NodePtr node1 = compute_graph->AddNode(opDescPtr);
    ge::NodePtr node2 = compute_graph->AddNode(opDescPtr);
    node1->GetOutDataAnchor(0)->LinkTo(node2->GetInDataAnchor(0));
    node2->GetOutDataAnchor(0)->LinkTo(node1->GetInDataAnchor(0));
    std::vector<GradientDataInfo> recordInfos;
    const string type = HCCL_KERNEL_OP_TYPE_REDUCE;
    op.SetType(type); 
    ge::AttrUtils::HasAttr(opDescPtr, "DUMMY_SET_TRUE_GROUP"); 
    ge::AttrUtils::SetStr(opDescPtr, "group", "hccl_world_group"); 
    AutoTuningHcomAllReduceFusion hcomAllReduceFusion;
    ret = hcomAllReduceFusion.RecordGradientDataInfo(*compute_graph, recordInfos);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    ge::AttrUtils::HasAttr(opDescPtr, "DUMMY_SET_FALSE_GROUP");

    EXPECT_EQ(recordInfos.size(), 0);
}
TEST_F(HcomGradientSplitTuneTest, st_gradient_RecordGradientDataInfo_reducescatter)
{
    HcclResult ret;

    ge::ComputeGraphPtr compute_graph = std::make_shared<ge::ComputeGraph>("test_graph");
    ge::OpDescPtr opDescPtr = nullptr;

    ge::OpDesc op;
    ge::NodePtr node1 = compute_graph->AddNode(opDescPtr);
    ge::NodePtr node2 = compute_graph->AddNode(opDescPtr);
    node1->GetOutDataAnchor(0)->LinkTo(node2->GetInDataAnchor(0));
    node2->GetOutDataAnchor(0)->LinkTo(node1->GetInDataAnchor(0));
    std::vector<GradientDataInfo> recordInfos;
    const string type = HCCL_KERNEL_OP_TYPE_REDUCESCATTER;
    op.SetType(type); 
    ge::AttrUtils::HasAttr(opDescPtr, "DUMMY_SET_TRUE_GROUP"); 
    ge::AttrUtils::SetStr(opDescPtr, "group", "hccl_world_group"); 
    AutoTuningHcomAllReduceFusion hcomAllReduceFusion;
    ret = hcomAllReduceFusion.RecordGradientDataInfo(*compute_graph, recordInfos);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    ge::AttrUtils::HasAttr(opDescPtr, "DUMMY_SET_FALSE_GROUP");

    EXPECT_EQ(recordInfos.size(), 0);
}
TEST_F(HcomGradientSplitTuneTest, st_gradient_RecordGradientDataInfo_send)
{
    HcclResult ret;

    ge::ComputeGraphPtr compute_graph = std::make_shared<ge::ComputeGraph>("test_graph");
    ge::OpDescPtr opDescPtr = nullptr;

    ge::OpDesc op;
    ge::NodePtr node1 = compute_graph->AddNode(opDescPtr);
    ge::NodePtr node2 = compute_graph->AddNode(opDescPtr);
    node1->GetOutDataAnchor(0)->LinkTo(node2->GetInDataAnchor(0));
    node2->GetOutDataAnchor(0)->LinkTo(node1->GetInDataAnchor(0));
    std::vector<GradientDataInfo> recordInfos;
    const string type = HCCL_KERNEL_OP_TYPE_SEND;
    op.SetType(type); 
    ge::AttrUtils::HasAttr(opDescPtr, "DUMMY_SET_TRUE_GROUP"); 
    ge::AttrUtils::SetStr(opDescPtr, "group", "hccl_world_group"); 
    AutoTuningHcomAllReduceFusion hcomAllReduceFusion;
    ret = hcomAllReduceFusion.RecordGradientDataInfo(*compute_graph, recordInfos);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    ge::AttrUtils::HasAttr(opDescPtr, "DUMMY_SET_FALSE_GROUP");

    EXPECT_EQ(recordInfos.size(), 0);
}
TEST_F(HcomGradientSplitTuneTest, st_gradient_RecordGradientDataInfo_receive)
{
    HcclResult ret;

    ge::ComputeGraphPtr compute_graph = std::make_shared<ge::ComputeGraph>("test_graph");
    ge::OpDescPtr opDescPtr = nullptr;

    ge::OpDesc op;
    ge::NodePtr node1 = compute_graph->AddNode(opDescPtr);
    ge::NodePtr node2 = compute_graph->AddNode(opDescPtr);
    node1->GetOutDataAnchor(0)->LinkTo(node2->GetInDataAnchor(0));
    node2->GetOutDataAnchor(0)->LinkTo(node1->GetInDataAnchor(0));
    std::vector<GradientDataInfo> recordInfos;
    const string type = HCCL_KERNEL_OP_TYPE_RECEIVE;
    op.SetType(type); 
    ge::AttrUtils::HasAttr(opDescPtr, "DUMMY_SET_TRUE_GROUP"); 
    ge::AttrUtils::SetStr(opDescPtr, "group", "hccl_world_group"); 
    AutoTuningHcomAllReduceFusion hcomAllReduceFusion;
    ret = hcomAllReduceFusion.RecordGradientDataInfo(*compute_graph, recordInfos);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    ge::AttrUtils::HasAttr(opDescPtr, "DUMMY_SET_FALSE_GROUP");

    EXPECT_EQ(recordInfos.size(), 0);
}
TEST_F(HcomGradientSplitTuneTest, st_gradient_HcomOpsKernelInfoStore_unloadTask)
{
    ge::Status ge_ret;
    ge::GETaskInfo task;
    AutoTuningHcomOpsKernelInfoStore athcomOpsKernelInfo;
    ge_ret = athcomOpsKernelInfo.UnloadTask(task);
    EXPECT_EQ(ge_ret, ge::SUCCESS);
}

TEST_F(HcomGradientSplitTuneTest, st_gradient_AddTraceNode)
{
    HcclResult ret;

    ge::ComputeGraphPtr compute_graph = std::make_shared<ge::ComputeGraph>("test_graph");
    ge::OpDescPtr opDescPtr = nullptr;

    ge::OpDesc op;
    ge::NodePtr node1 = compute_graph->AddNode(opDescPtr);
    ge::NodePtr node2 = compute_graph->AddNode(opDescPtr);
    node1->GetOutDataAnchor(0)->LinkTo(node2->GetInDataAnchor(0));
    node2->GetOutDataAnchor(0)->LinkTo(node1->GetInDataAnchor(0));

    std::vector<GradientDataInfo> recordInfos;
    const string type = "HcomAllReduce";
    op.SetType(type); 
    ge::AttrUtils::HasAttr(opDescPtr, "DUMMY_SET_TRUE_GROUP"); 
    ge::AttrUtils::SetStr(opDescPtr, "reduction", "sum"); 
    ge::AttrUtils::SetStr(opDescPtr, "group", "hccl_world_group"); 
    ge::AttrUtils::SetInt(opDescPtr, "fusion", 1); 
    AutoTuningHcomAllReduceFusion hcomAllReduceFusion;

    GradientDataInfo recordInfo;
    recordInfo.dataSize = 1024;
    recordInfo.dataType = "float";
    recordInfo.graphId = 0;
    recordInfo.groupName = "hccl_world_group";
    recordInfo.gradientNodeName = "test_gradientNodeName_0";
    recordInfo.traceNodeName = "test_traceNodeName_0";
    recordInfo.allReduceNodeName = "test_allReduceNodeName_0";
    ret = hcomAllReduceFusion.AddTraceNode(*compute_graph, node1, recordInfo);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    ret = hcomAllReduceFusion.GetGradientDataInfo(*compute_graph, node1, recordInfo);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    ge::AttrUtils::HasAttr(opDescPtr, "DUMMY_SET_FALSE_GROUP");
}