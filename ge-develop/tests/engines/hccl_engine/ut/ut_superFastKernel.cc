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
#include "llt_hccl_stub_hccl.h"
#define private public
#define protected public
#include "hcom_all_reduce_fusion.h"
#include "hcom_broadcast_fusion.h"
#include "hcom_reduce_fusion.h"
#include "hcom_alltoallvc_fusion.h"
#include "hcom_allgather_fusion.h"
#include "hcom_reducescatter_fusion.h"
#include "plugin_manager.h"
#include "hcom_fusion_optimizer.h"
#include "offline_build_config_parse.h"
#include "hcom_graph_optimizer.h"
#include "hcom_op_utils.h"
#undef private
#undef protected

#include "llt_hccl_stub.h"
#include "v80_rank_table.h"
#include "hcom_op_utils.h"
#include "hcom_ops_kernel_info_store.h"
#include "external/ge/ge_api_types.h"   // ge对内options
#include "framework/common/ge_types.h"  // ge对外options
#include "graph/ge_local_context.h"
#include "hcom_topo_info.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/node.h"
#include "graph/op_desc.h"
#include "graph/utils/tensor_utils.h"
#include "graph/ge_tensor.h"

#include "evaluator.h"
#include "model.h"
#include "cluster.h"

#include <iostream>
#include <fstream>
#include "graph/ge_context.h"
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
#include "hcom_graph_superkernel.h"
#include "hcom_graph_mc2.h"
#include "hcom_graph_optimizer.h"

using namespace std;
using namespace hccl;

namespace hccl {
extern HcclResult GetCountFromOpDesc(
    const ge::OpDescPtr &op, const std::string &sCollectiveType, HcclDataType dataType, u64 &count, u32 rankSize);
}

class SuperFastKernelTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "\033[36m--SuperFastKernelTest SetUP--\033[0m" << std::endl;
    }
    static void TearDownTestCase()
    {

        std::cout << "\033[36m--SuperFastKernelTest TearDown--\033[0m" << std::endl;
    }
    // Some expensive resource shared by all tests.
    virtual void SetUp()
    {
        std::cout << "A Test SetUP" << std::endl;
    }
    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test TearDown" << std::endl;
    }
};


HcclResult HcomGetRankSizeStub(const char *group, u32 *rankSize)
{
    *rankSize = 1;
    return HCCL_SUCCESS;
}
TEST_F(SuperFastKernelTest, ut_mc2_GetAddrFromDesc)
{
    std::map<std::string, std::string> options;
    options.insert(pair<string, string>(ge::TUNING_PATH, "./"));
    options.insert(pair<string, string>("ge.jobType", "4"));

    ge::Status ge_ret;
    HCCL_INFO("test Initialize.");
    ge_ret = Initialize(options);
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    std::string group = "group1";
    ge::OpDescPtr nodeGroup = std::make_shared<ge::OpDesc>();
    EXPECT_EQ(ge::AttrUtils::SetStr(nodeGroup, "group", group), true);
    u64 count;
    std::string sCollectiveType = HCCL_KERNEL_OP_TYPE_RECEIVE;
    std::vector<void *> contexts;
    u32 rankSize = 8;
    nodeGroup->SetType(HCCL_KERNEL_OP_TYPE_BROADCAST);
    MOCKER(HcomGetRankSize).stubs().with(mockcpp::any(), outBound(&rankSize)).will(returnValue(HCCL_SUCCESS));
    MOCKER(HcomOpUtils::GetCountFromOpDescSuperkernel).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(HcomOpUtils::GetAivCoreLimit).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(HcomGetAlgExecParam).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    EXPECT_EQ(HcomCreateSuperkernelResource(nodeGroup, contexts), ge::GRAPH_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(SuperFastKernelTest, ut_mc2_HcomCreateSuperkernelResource)
{
    DeviceMem context = DeviceMem::alloc(1024 * 1024);
    void *contextptr = context.ptr();
    MOCKER(HcclAllocComResourceByTiling)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), outBoundP(&contextptr, sizeof(contextptr)))
        .will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtStreamGetMode).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(HcomCreateComResourceByComm)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), outBoundP(&contextptr, sizeof(contextptr)))
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&ge::HcomTopoInfo::TryGetGroupTopoInfo).stubs().with(mockcpp::any()).will(returnValue(true));
    MOCKER_CPP(&HcomGetAicpuOpStreamNotify).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&ge::HcomTopoInfo::SetGroupTopoInfo).stubs().with(mockcpp::any()).will(returnValue(ge::GRAPH_SUCCESS));
    std::map<std::string, std::string> options;
    options.insert(pair<string, string>(ge::TUNING_PATH, "./"));
    options.insert(pair<string, string>("ge.jobType", "4"));

    ge::Status ge_ret;
    HCCL_INFO("test Initialize.");
    ge_ret = Initialize(options);
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    std::string group = "group1";
    const ge::OpDescPtr nodeGroup = std::make_shared<ge::OpDesc>();
    std::vector<void *> contexts;
    nodeGroup->SetType("MatmulAllReduce");
    rtStream_t stream;
    EXPECT_EQ(rtStreamCreate(&stream, 0), RT_ERROR_NONE);

    shared_ptr<std::vector<void *>> rt_resource_list = std::make_shared<std::vector<void *>>();
    rt_resource_list->push_back(stream);
    nodeGroup->SetExtAttr("_rt_resource_list", rt_resource_list);
    u64 count;
    std::string sCollectiveType = HCCL_KERNEL_OP_TYPE_RECEIVE;

    int64_t hcomComm = 0;
    HcclComm hcclComm;
    hcomComm = reinterpret_cast<int64_t>(hcclComm);
    MOCKER(HcomGetCommHandleByGroup).stubs().with(mockcpp::any(), outBound(hcomComm)).will(returnValue(HCCL_SUCCESS));
    EXPECT_EQ(HcomCreateSuperkernelResource(nodeGroup, contexts), ge::GRAPH_FAILED);
    rtStreamDestroy(stream);
}

TEST_F(SuperFastKernelTest, ut_GetAivCoreLimit)
{
    ge::ComputeGraph graph("test_graph");
    ge::OpDescPtr nodeGroup = nullptr;
    std::string group = HCCL_WORLD_GROUP;
    std::string vectorcoreNum = "1";
    EXPECT_EQ(ge::AttrUtils::SetStr(nodeGroup, "_op_vectorcore_num", vectorcoreNum), true);
    std::string sCollectiveType = HCCL_KERNEL_OP_TYPE_REDUCESCATTER;
    u32 aivCoreLimit;
    HcomOpUtils::GetAivCoreLimit(nodeGroup, sCollectiveType, aivCoreLimit);
    EXPECT_EQ(ge::AttrUtils::SetStr(nodeGroup, "vectorcore", group), true);
    ge::OpDescPtr nodeGroup1 = nullptr;
    EXPECT_EQ(ge::AttrUtils::SetStr(nodeGroup1, "group", group), true);
    HcomOpUtils::GetAivCoreLimit(nodeGroup1, sCollectiveType, aivCoreLimit);

    u64 count;
    sCollectiveType = HCCL_KERNEL_OP_TYPE_RECEIVE;
    HcclResult ret =
        HcomOpUtils::GetCountFromOpDescSuperkernel(nodeGroup, sCollectiveType, HcclDataType::HCCL_DATA_TYPE_INT64, count, 1);
    EXPECT_EQ(ret, HcclResult::HCCL_E_PARA);
    sCollectiveType = HCCL_KERNEL_OP_TYPE_REDUCESCATTER;
    ret =HcomOpUtils::GetCountFromOpDescSuperkernel(nodeGroup, sCollectiveType, HcclDataType::HCCL_DATA_TYPE_INT64, count, 1);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    sCollectiveType = HCCL_KERNEL_OP_TYPE_ALLGATHER;
    ret = HcomOpUtils::GetCountFromOpDescSuperkernel(nodeGroup, sCollectiveType, HcclDataType::HCCL_DATA_TYPE_INT64, count, 1);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    sCollectiveType = HCCL_KERNEL_OP_TYPE_ALLREDUCE;
    ret = HcomOpUtils::GetCountFromOpDescSuperkernel(nodeGroup, sCollectiveType, HcclDataType::HCCL_DATA_TYPE_INT64, count, 1);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    sCollectiveType = HCCL_KERNEL_OP_TYPE_BROADCAST;
    ret = HcomOpUtils::GetCountFromOpDescSuperkernel(nodeGroup, sCollectiveType, HcclDataType::HCCL_DATA_TYPE_INT64, count, 1);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

struct OpInfo {
    ge::NodePtr nodePtr;
    std::pair<std::string, std::string> opName;
    std::vector<std::pair<std::string, int64_t>> attrInt;
    std::vector<std::pair<std::string, std::string>> attrStr;
};

HcclResult CreateNodePtr1(OpInfo& opInfo, ge::ComputeGraph& graph)
{
    ge::OpDescPtr opDescPtr =
        std::make_shared<ge::OpDesc>(opInfo.opName.first.c_str(), opInfo.opName.second.c_str());
    EXPECT_NE(opDescPtr, nullptr);
    opDescPtr->SetName(opInfo.opName.first.c_str());

    for (auto& it : opInfo.attrInt) {
        bool bErr = ge::AttrUtils::SetInt(opDescPtr, it.first.c_str(), it.second);
        CHK_PRT_RET(!bErr, HCCL_ERROR("node[%s] set attr: %s failed", \
            opInfo.opName.first.c_str(), it.first.c_str()), HCCL_E_INTERNAL);
    }
    for (auto& it : opInfo.attrStr) {
        bool bErr = ge::AttrUtils::SetStr(opDescPtr, it.first.c_str(), it.second.c_str());
        CHK_PRT_RET(!bErr, HCCL_ERROR("node[%s] set attr: %s failed", \
            opInfo.opName.first.c_str(), it.first.c_str()), HCCL_E_INTERNAL);
    }

    opInfo.nodePtr = graph.AddNode(opDescPtr);
    CHK_PRT_RET((opInfo.nodePtr == nullptr), HCCL_ERROR("[Create]node[%s] failed",
        opInfo.opName.first.c_str()), HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

TEST_F(SuperFastKernelTest, ut_SetSuperKernelScopeAttr)
{
    int64_t hcomComm = 0;
    HcclComm hcclComm;
    hcomComm = reinterpret_cast<int64_t>(hcclComm);
    MOCKER(HcomGetCommHandleByGroup).stubs().with(mockcpp::any(), outBound(hcomComm)).will(returnValue(HCCL_SUCCESS));

    bool ifAiv = true;
    std::string algName = "AllGatherMeshAivExecutor";
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test_graph");
    u32 allgatherNum = 2;
    for (u32 i = 0; i < allgatherNum; i++) {
        // allgather
        OpInfo allgatherOpInfo;
        allgatherOpInfo.opName = std::make_pair("AllGather_" + std::to_string(i), HCCL_KERNEL_OP_TYPE_ALLGATHER);
        allgatherOpInfo.attrInt.push_back(std::make_pair("rank_size", 3));
        allgatherOpInfo.attrInt.push_back(std::make_pair("fusion", 2));
        allgatherOpInfo.attrInt.push_back(std::make_pair("fusion_id", 1));
        allgatherOpInfo.attrStr.push_back(std::make_pair("group", HCCL_WORLD_GROUP));
        auto ret = CreateNodePtr1(allgatherOpInfo, *graph);
        EXPECT_EQ(ret, HCCL_SUCCESS);

        ge::AttrUtils::HasAttr(allgatherOpInfo.nodePtr->GetOpDesc(), "DUMMY_SET_TRUE_GROUP");
        allgatherOpInfo.nodePtr->GetOpDesc()->SetType(HCCL_KERNEL_OP_TYPE_ALLGATHER);
        std::string superKernelScope = "hccl_aiv";
        EXPECT_EQ(
            ge::AttrUtils::SetStr(allgatherOpInfo.nodePtr->GetOpDesc(), "_super_kernel_scope", superKernelScope), true);
    }
    HcomGraphOptimizer graphOptimizer;
    HcclResult ge_ret = graphOptimizer.SetSuperKernelScopeAttr(*graph);
    EXPECT_EQ(ge_ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(SuperFastKernelTest, ut_SetSuperKernelScopeAttr_notsupport)
{
    int64_t hcomComm = 0;
    HcclComm hcclComm;
    hcomComm = reinterpret_cast<int64_t>(hcclComm);
    MOCKER(HcomGetCommHandleByGroup).stubs().with(mockcpp::any(), outBound(hcomComm)).will(returnValue(HCCL_SUCCESS));
    std::string algName = "AllGatherMeshAivExecutor";
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test_graph");
    u32 allgatherNum = 2;
    for (u32 i = 0; i < allgatherNum; i++) {
        // allgather
        OpInfo allgatherOpInfo;
        allgatherOpInfo.opName = std::make_pair("AllGather_" + std::to_string(i), HCCL_KERNEL_OP_TYPE_ALLTOALLV);
        allgatherOpInfo.attrInt.push_back(std::make_pair("rank_size", 3));
        allgatherOpInfo.attrInt.push_back(std::make_pair("fusion", 2));
        allgatherOpInfo.attrInt.push_back(std::make_pair("fusion_id", 1));
        allgatherOpInfo.attrStr.push_back(std::make_pair("group", HCCL_WORLD_GROUP));
        auto ret = CreateNodePtr1(allgatherOpInfo, *graph);
        EXPECT_EQ(ret, HCCL_SUCCESS);

        ge::AttrUtils::HasAttr(allgatherOpInfo.nodePtr->GetOpDesc(), "DUMMY_SET_TRUE_GROUP");
        allgatherOpInfo.nodePtr->GetOpDesc()->SetType(HCCL_KERNEL_OP_TYPE_ALLTOALLV);
        std::string superKernelScope = "hccl_aiv";
        EXPECT_EQ(
            ge::AttrUtils::SetStr(allgatherOpInfo.nodePtr->GetOpDesc(), "_super_kernel_scope", superKernelScope), true);
    }
    HcomGraphOptimizer graphOptimizer;
    HcclResult ge_ret = graphOptimizer.SetSuperKernelScopeAttr(*graph);
    EXPECT_EQ(ge_ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(SuperFastKernelTest, ut_SetSuperKernelScopeAttr_notsupport3)
{
    int64_t hcomComm = 0;
    HcclComm hcclComm;
    hcomComm = reinterpret_cast<int64_t>(hcclComm);
    std::string algName = "AllGatherMeshAivExecutor";
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test_graph");
    u32 allgatherNum = 2;
    for (u32 i = 0; i < allgatherNum; i++) {
        // allgather
        OpInfo allgatherOpInfo;
        allgatherOpInfo.opName = std::make_pair("AllGather_" + std::to_string(i), HCCL_KERNEL_OP_TYPE_ALLGATHER);
        allgatherOpInfo.attrInt.push_back(std::make_pair("rank_size", 3));
        allgatherOpInfo.attrInt.push_back(std::make_pair("fusion", 2));
        allgatherOpInfo.attrInt.push_back(std::make_pair("fusion_id", 1));
        allgatherOpInfo.attrStr.push_back(std::make_pair("group", HCCL_WORLD_GROUP));
        auto ret = CreateNodePtr1(allgatherOpInfo, *graph);
        EXPECT_EQ(ret, HCCL_SUCCESS);

        ge::AttrUtils::HasAttr(allgatherOpInfo.nodePtr->GetOpDesc(), "DUMMY_SET_TRUE_GROUP");
        allgatherOpInfo.nodePtr->GetOpDesc()->SetType(HCCL_KERNEL_OP_TYPE_ALLGATHER);
        std::string superKernelScope = "hccl_aiv";
        EXPECT_EQ(
            ge::AttrUtils::SetStr(allgatherOpInfo.nodePtr->GetOpDesc(), "_super_kernel_scope", superKernelScope), true);

        std::string vectorcoreNum = "3";
        EXPECT_EQ(ge::AttrUtils::SetStr(allgatherOpInfo.nodePtr->GetOpDesc(), "_op_vectorcore_num", vectorcoreNum), true);
    }
    HcomGraphOptimizer graphOptimizer;
    HcclResult ge_ret = graphOptimizer.SetSuperKernelScopeAttr(*graph);
    EXPECT_EQ(ge_ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

// 覆盖率
TEST_F(SuperFastKernelTest, ut_GetAivCoreLimit1)
{
    const ge::OpDescPtr nodeGroup = std::make_shared<ge::OpDesc>();
    std::string sCollectiveType = HCCL_KERNEL_OP_TYPE_SEND;
    u64 count = 0;
    GetCountFromOpDesc(nodeGroup, sCollectiveType, HcclDataType::HCCL_DATA_TYPE_INT64, count, 1);
}

TEST_F(SuperFastKernelTest, ut_mc2_creatComResourceErr)
{
    std::vector<void *> commContext{};
    ge::graphStatus ge_ret;
    HcclRootInfo id;

    const ge::OpDescPtr opDescPtr = std::make_shared<ge::OpDesc>();
    opDescPtr->SetType("MatmulAllReduce");
    ge::AttrUtils::SetStr(opDescPtr, "group", "hccl_world_group");

    ge_ret = HcomCreateComResource(opDescPtr, commContext);
    EXPECT_EQ(ge_ret, ge::GRAPH_FAILED);

    shared_ptr<std::vector<void *>> rt_resource_list = std::make_shared<std::vector<void *>>();
    opDescPtr->SetExtAttr("_rt_resource_list", rt_resource_list);
    ge_ret = HcomCreateComResource(opDescPtr, commContext);
    EXPECT_EQ(ge_ret, ge::GRAPH_FAILED);
}

TEST_F(SuperFastKernelTest, hcom_ranktable_check1)
{
    const char *ranktable1 = "fsdgfsdagsdagdsafweqqq";
    std::string realPath;
    HcclResult ret = HcomGetRanktableRealPath(ranktable1, realPath);;
    EXPECT_EQ(ret, HCCL_E_PARA);
    const char *ranktable2 = "/usr/X11R6/lib/modules/../../include/../X11R6/lib/modules/../../include/../X11R6/lib/modules/../../include/../X11R6/lib/modules/../../include/../X11R6/lib/modules/../../include/../X11R6/lib/modules/../../include/../";
    ret = HcomGetRanktableRealPath(ranktable2, realPath);
    EXPECT_EQ(ret, HCCL_E_PARA);
}