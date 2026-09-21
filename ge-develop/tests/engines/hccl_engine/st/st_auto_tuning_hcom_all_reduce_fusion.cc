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


using namespace std;
using namespace hccl;

class AutoTuningHcomAllReduceFusionTest : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "\033[36m--AutoTuningHcomAllReduceFusionTest SetUP--\033[0m" << std::endl;
    }
    static void TearDownTestCase()
    {

        std::cout << "\033[36m--AutoTuningHcomAllReduceFusionTest TearDown--\033[0m" << std::endl;
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

TEST_F(AutoTuningHcomAllReduceFusionTest, st_SetGradientInformation)
{
    HcclResult ret;

    ge::ComputeGraphPtr compute_graph = std::make_shared<ge::ComputeGraph>("test_graph");
    ge::OpDescPtr opDescPtr = nullptr;

    ge::OpDesc op;
    ge::NodePtr node1 = compute_graph->AddNode(opDescPtr);
    ge::NodePtr node2 = compute_graph->AddNode(opDescPtr);
    node1->GetOutDataAnchor(0)->LinkTo(node2->GetInDataAnchor(0));
    node2->GetOutDataAnchor(0)->LinkTo(node1->GetInDataAnchor(0));

    const string type = "HcomAllReduce";
    op.SetType(type); 
    ge::AttrUtils::HasAttr(opDescPtr, "DUMMY_SET_TRUE_GROUP"); 
    ge::AttrUtils::SetStr(opDescPtr, "reduction", "sum"); 
    ge::AttrUtils::SetStr(opDescPtr, "group", "hccl_world_group"); 
    ge::AttrUtils::SetInt(opDescPtr, "fusion", 1); 

    AutoTuningHcomAllReduceFusion hcomAllReduceFusion;
    ret = hcomAllReduceFusion.SetGradientInformation(*compute_graph);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    ge::AttrUtils::HasAttr(opDescPtr, "DUMMY_SET_FALSE_GROUP");
    GlobalMockObject::verify();
}

TEST_F(AutoTuningHcomAllReduceFusionTest, utGetFusionPathByDefault)
{
    HcclResult ret;
    std::string path;
    AutoTuningHcomAllReduceFusion object;
    ret = object.GetFusionWorkPath(path);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(AutoTuningHcomAllReduceFusionTest, utAddFusionMapInFusionJsonWithNonempty)
{
    HcclResult ret;
    std::string fusionHash = "3047522271241092779";
    std::string socVersion = "Ascend910A";

    MOCKER(HcomOpUtils::CreateFusionConfigVersion)
    .stubs()
    .with(outBound(socVersion))
    .will(returnValue(HCCL_SUCCESS));

    AutoTuningHcomAllReduceFusion object;
    ret = object.AddFusionMapInFusionJson(fusionHash);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();

    std::string filePath = "./st_Ascend910A_gradient_fusion_2.json";
    char file_name_1[] = "./st_Ascend910A_gradient_fusion_2.json";
    std::ofstream outfile_empty(file_name_1, std::ios::out | std::ios::trunc);

    if (outfile_empty.is_open()) {
        HCCL_INFO("open %s success", file_name_1);
    } else {
        HCCL_ERROR("open %s failed", file_name_1);
    }
    outfile_empty.close();

    MOCKER_CPP(&AutoTuningHcomAllReduceFusion::GetFusionWorkPath)
    .stubs()
    .with(outBound(filePath))
    .will(returnValue(HCCL_SUCCESS));

    ret = object.AddFusionMapInFusionJson(fusionHash);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
    remove(file_name_1);

    nlohmann::json fusion_json =
    {
        {"modelhash", "3047522271241092779"},
        {"modelvalue", 
            {
                {"gradientSize", {16384000,4000,67108864,16384,150994944,16384,2359296,1024,3538944,1024,2654208,1536,1843200,768,139392,384}},
                {"gradientTime", {9833418590329,15080,100303,86064,408791,246415,634287,686785,103,564542,100,1686982,117497,3490117,202901,117711}}
            }
        }
    };

    filePath = "./st_Ascend910A_gradient_fusion_3.json";
    char file_name_2[] = "./st_Ascend910A_gradient_fusion_3.json";
    std::ofstream outfile(file_name_2, std::ios::out | std::ios::trunc);

    if (outfile.is_open()) {
        outfile << std::setw(1) << fusion_json << std::endl;
        HCCL_INFO("open %s success", file_name_2);
    } else {
        HCCL_ERROR("open %s failed", file_name_2);
    }
    outfile.close();

    ret = object.AddFusionMapInFusionJson(fusionHash);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
    remove(file_name_2);
}