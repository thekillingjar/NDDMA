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
#include <gmock/gmock.h>
#include <mockcpp/mockcpp.hpp>
#include <stdio.h>
#define private public
#define protected public

#include "hcom_plugin.h"

#include <iostream>
#include <fstream>
#undef private
#undef protected

using namespace std;
using namespace hccl;

class HcomPluginTest : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "\033[36m--HcomPluginTest SetUP--\033[0m" << std::endl;
    }
    static void TearDownTestCase()
    {

        std::cout << "\033[36m--HcomPluginTest TearDown--\033[0m" << std::endl;
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

TEST_F(HcomPluginTest, ut_hcom_plugin_exec_timeout_default)
{
    ge ::Status ret;
    std::map<std::string,std::string> options;

    // 未设置 rank table：失败
    options.insert(pair<string,string> (ge::OPTION_EXEC_RANK_TABLE_FILE,"rank_table.json"));
    // 实验室场景 设置 rank id
    options.insert(pair<string,string> (ge::OPTION_EXEC_DEPLOY_MODE,"0"));
    options.insert(pair<string,string> (ge::OPTION_EXEC_RANK_ID,"1"));
    options.insert(pair<string,string> (ge::OPTION_EXEC_PROFILING_MODE,"0"));

    // hccl 算法配置
    // options.insert(pair<string,string> ("HCCL_algorithm","ring"));
    // 实验室场景 hcom_init成功：成功
    ret = HcomPlugin::Instance().Initialize(options);
    EXPECT_EQ(ret, ge::INTERNAL_ERROR);

    ret = HcomPlugin::Instance().Finalize();
    EXPECT_EQ(ret, ge::SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(HcomPluginTest, ut_hcom_plugin_exec_timeout_invalid1)
{
    ge ::Status ret;
    std::map<std::string,std::string> options;

    // 未设置 rank table：失败
    options.insert(pair<string,string> (ge::OPTION_EXEC_RANK_TABLE_FILE,"rank_table.json"));
    // 实验室场景 设置 rank id
    options.insert(pair<string,string> (ge::OPTION_EXEC_DEPLOY_MODE,"0"));
    options.insert(pair<string,string> (ge::OPTION_EXEC_RANK_ID,"1"));
    options.insert(pair<string,string> (ge::OPTION_EXEC_PROFILING_MODE,"0"));

    // hccl 算法配置
    options.insert(pair<string,string> ("ge.exec.hcclExecuteTimeOut","0x5a5a"));

    // 实验室场景 hcom_init成功：成功
    ret = HcomPlugin::Instance().Initialize(options);
    EXPECT_EQ(ret, ge::INTERNAL_ERROR);

    
    ret = HcomPlugin::Instance().Finalize();
    EXPECT_EQ(ret, ge::SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(HcomPluginTest, ut_hcom_plugin_exec_timeout_invalid2)
{
    ge ::Status ret;
    std::map<std::string,std::string> options;
    
    // 未设置 rank table：失败
    options.insert(pair<string,string> (ge::OPTION_EXEC_RANK_TABLE_FILE,"rank_table.json"));
    // 实验室场景 设置 rank id
    options.insert(pair<string,string> (ge::OPTION_EXEC_DEPLOY_MODE,"0"));
    options.insert(pair<string,string> (ge::OPTION_EXEC_RANK_ID,"1"));
    options.insert(pair<string,string> (ge::OPTION_EXEC_PROFILING_MODE,"0"));

    // hccl 算法配置
    options.insert(pair<string,string> ("ge.exec.hcclExecuteTimeOut","this is a test"));

    // 实验室场景 hcom_init成功：成功
    ret = HcomPlugin::Instance().Initialize(options);
    EXPECT_EQ(ret, ge::INTERNAL_ERROR);

    
    ret = HcomPlugin::Instance().Finalize();
    EXPECT_EQ(ret, ge::SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(HcomPluginTest, ut_hcom_plugin_exec_timeout_invalid3)
{
    ge ::Status ret;
    std::map<std::string,std::string> options;

    // 未设置 rank table：失败
    options.insert(pair<string,string> (ge::OPTION_EXEC_RANK_TABLE_FILE,"rank_table.json"));
    // 实验室场景 设置 rank id
    options.insert(pair<string,string> (ge::OPTION_EXEC_DEPLOY_MODE,"0"));
    options.insert(pair<string,string> (ge::OPTION_EXEC_RANK_ID,"1"));
    options.insert(pair<string,string> (ge::OPTION_EXEC_PROFILING_MODE,"0"));

    // hccl 算法配置
    options.insert(pair<string,string> ("ge.exec.hcclExecuteTimeOut","-2555"));

    // 实验室场景 hcom_init成功：成功
    ret = HcomPlugin::Instance().Initialize(options);
    EXPECT_EQ(ret, ge::INTERNAL_ERROR);

    
    ret = HcomPlugin::Instance().Finalize();
    EXPECT_EQ(ret, ge::SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(HcomPluginTest, ut_hcom_plugin_exec_timeout_invalid4)
{
    ge ::Status ret;
    std::map<std::string,std::string> options;

    
    
    // 未设置 rank table：失败
    options.insert(pair<string,string> (ge::OPTION_EXEC_RANK_TABLE_FILE,"rank_table.json"));
    // 实验室场景 设置 rank id
    options.insert(pair<string,string> (ge::OPTION_EXEC_DEPLOY_MODE,"0"));
    options.insert(pair<string,string> (ge::OPTION_EXEC_RANK_ID,"1"));
    options.insert(pair<string,string> (ge::OPTION_EXEC_PROFILING_MODE,"0"));

    // hccl 算法配置
    options.insert(pair<string,string> ("ge.exec.hcclExecuteTimeOut","0"));

    
    // 实验室场景 hcom_init成功：成功
    ret = HcomPlugin::Instance().Initialize(options);
    EXPECT_EQ(ret, ge::INTERNAL_ERROR);

    

    
    ret = HcomPlugin::Instance().Finalize();
    EXPECT_EQ(ret, ge::SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(HcomPluginTest, ut_hcom_plugin_exec_timeout_valid0)
{
    ge ::Status ret;
    std::map<std::string,std::string> options;
    
    // 未设置 rank table：失败
    options.insert(pair<string,string> (ge::OPTION_EXEC_RANK_TABLE_FILE,"rank_table.json"));
    // 实验室场景 设置 rank id
    options.insert(pair<string,string> (ge::OPTION_EXEC_DEPLOY_MODE,"0"));
    options.insert(pair<string,string> (ge::OPTION_EXEC_RANK_ID,"1"));
    options.insert(pair<string,string> (ge::OPTION_EXEC_PROFILING_MODE,"0"));

    // hccl 算法配置
    options.insert(pair<string,string> ("ge.exec.hcclExecuteTimeOut","68"));

    MOCKER(HcomLoadRanktableFile)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));

    // 实验室场景 hcom_init成功：成功
    ret = HcomPlugin::Instance().Initialize(options);
    EXPECT_EQ(ret, ge::SUCCESS);

    
    ret = HcomPlugin::Instance().Finalize();
    EXPECT_EQ(ret, ge::SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(HcomPluginTest, ut_hcom_plugin_exec_timeout_valid1)
{
    ge ::Status ret;
    std::map<std::string,std::string> options;
    
    // 未设置 rank table：失败
    options.insert(pair<string,string> (ge::OPTION_EXEC_RANK_TABLE_FILE,"rank_table.json"));
    // 实验室场景 设置 rank id
    options.insert(pair<string,string> (ge::OPTION_EXEC_DEPLOY_MODE,"0"));
    options.insert(pair<string,string> (ge::OPTION_EXEC_RANK_ID,"1"));
    options.insert(pair<string,string> (ge::OPTION_EXEC_PROFILING_MODE,"0"));

    // hccl 算法配置
    options.insert(pair<string,string> ("ge.exec.hcclExecuteTimeOut","3600"));

    MOCKER(HcomLoadRanktableFile)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));
    
    // 实验室场景 hcom_init成功：成功
    ret = HcomPlugin::Instance().Initialize(options);
    EXPECT_EQ(ret, ge::SUCCESS);

    
    ret = HcomPlugin::Instance().Finalize();
    EXPECT_EQ(ret, ge::SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(HcomPluginTest, ut_hcom_plugin_algo_config_hcclalgo_default)
{
    ge ::Status ret;
    std::map<std::string,std::string> options;
    
    // 未设置 rank table：失败
    options.insert(pair<string,string> (ge::OPTION_EXEC_RANK_TABLE_FILE,"rank_table.json"));
    // 实验室场景 设置 rank id
    options.insert(pair<string,string> (ge::OPTION_EXEC_DEPLOY_MODE,"0"));
    options.insert(pair<string,string> (ge::OPTION_EXEC_RANK_ID,"1"));
    options.insert(pair<string,string> (ge::OPTION_EXEC_PROFILING_MODE,"0"));

    // hccl 算法配置
    // options.insert(pair<string,string> ("HCCL_algorithm","ring"));

    MOCKER(HcomLoadRanktableFile)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));
    
    // 实验室场景 hcom_init成功：成功
    ret = HcomPlugin::Instance().Initialize(options);
    EXPECT_EQ(ret, ge::SUCCESS);
    
    ret = HcomPlugin::Instance().Finalize();
    EXPECT_EQ(ret, ge::SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(HcomPluginTest, ut_hcom_plugin_algo_config_hcclalgo_level0)
{
    ge ::Status ret;
    std::map<std::string,std::string> options;

    // 未设置 rank table：失败
    options.insert(pair<string,string> (ge::OPTION_EXEC_RANK_TABLE_FILE,"rank_table.json"));
    // 实验室场景 设置 rank id
    options.insert(pair<string,string> (ge::OPTION_EXEC_DEPLOY_MODE,"0"));
    options.insert(pair<string,string> (ge::OPTION_EXEC_RANK_ID,"1"));
    options.insert(pair<string,string> (ge::OPTION_EXEC_PROFILING_MODE,"0"));

    // hccl 算法配置
    options.insert(pair<string,string> ("HCCL_algorithm","level0:ring"));

    MOCKER(HcomLoadRanktableFile)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));    
    // 实验室场景 hcom_init成功：成功
    ret = HcomPlugin::Instance().Initialize(options);
    EXPECT_EQ(ret, ge::SUCCESS);

    
    ret = HcomPlugin::Instance().Finalize();
    EXPECT_EQ(ret, ge::SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(HcomPluginTest, ut_hcom_plugin_algo_config_hcclalgo_level1)
{
    ge ::Status ret;
    std::map<std::string,std::string> options;
    
    // 未设置 rank table：失败
    options.insert(pair<string,string> (ge::OPTION_EXEC_RANK_TABLE_FILE,"rank_table.json"));
    // 实验室场景 设置 rank id
    options.insert(pair<string,string> (ge::OPTION_EXEC_DEPLOY_MODE,"0"));
    options.insert(pair<string,string> (ge::OPTION_EXEC_RANK_ID,"1"));
    options.insert(pair<string,string> (ge::OPTION_EXEC_PROFILING_MODE,"0"));

    // hccl 算法配置
    // options.insert(pair<string,string> ("level0_algorithm","ring"));
    options.insert(pair<string,string> ("HCCL_algorithm","level1:H-D_R"));

    MOCKER(HcomLoadRanktableFile)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));
    
    // 实验室场景 hcom_init成功：成功
    ret = HcomPlugin::Instance().Initialize(options);
    EXPECT_EQ(ret, ge::SUCCESS);

    
    ret = HcomPlugin::Instance().Finalize();
    EXPECT_EQ(ret, ge::SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(HcomPluginTest, ut_hcom_plugin_algo_config_hcclalgo_level2)
{
    ge ::Status ret;
    std::map<std::string,std::string> options;    
    
    // 未设置 rank table：失败
    options.insert(pair<string,string> (ge::OPTION_EXEC_RANK_TABLE_FILE,"rank_table.json"));
    // 实验室场景 设置 rank id
    options.insert(pair<string,string> (ge::OPTION_EXEC_DEPLOY_MODE,"0"));
    options.insert(pair<string,string> (ge::OPTION_EXEC_RANK_ID,"1"));
    options.insert(pair<string,string> (ge::OPTION_EXEC_PROFILING_MODE,"0"));

    // hccl 算法配置
    options.insert(pair<string,string> ("HCCL_algorithm","level2:fullmesh"));

    MOCKER(HcomLoadRanktableFile)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));

    // 实验室场景 hcom_init成功：成功
    ret = HcomPlugin::Instance().Initialize(options);
    EXPECT_EQ(ret, ge::SUCCESS);

    
    ret = HcomPlugin::Instance().Finalize();
    EXPECT_EQ(ret, ge::SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(HcomPluginTest, ut_hcom_plugin_algo_config_hcclalgo_level3)
{
    ge ::Status ret;
    std::map<std::string,std::string> options;
        
    // 未设置 rank table：失败
    options.insert(pair<string,string> (ge::OPTION_EXEC_RANK_TABLE_FILE,"rank_table.json"));
    // 实验室场景 设置 rank id
    options.insert(pair<string,string> (ge::OPTION_EXEC_DEPLOY_MODE,"0"));
    options.insert(pair<string,string> (ge::OPTION_EXEC_RANK_ID,"1"));
    options.insert(pair<string,string> (ge::OPTION_EXEC_PROFILING_MODE,"0"));

    // hccl 算法配置
    options.insert(pair<string,string> ("HCCL_algorithm","level3:ring"));

    MOCKER(HcomLoadRanktableFile)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));

    // 实验室场景 hcom_init成功：成功
    ret = HcomPlugin::Instance().Initialize(options);
    EXPECT_EQ(ret, ge::SUCCESS);

    
    ret = HcomPlugin::Instance().Finalize();
    EXPECT_EQ(ret, ge::SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(HcomPluginTest, ut_hcom_plugin_algo_config_hcclalgo_level01)
{
    ge ::Status ret;
    std::map<std::string,std::string> options;   
    
    // 未设置 rank table：失败
    options.insert(pair<string,string> (ge::OPTION_EXEC_RANK_TABLE_FILE,"rank_table.json"));
    // 实验室场景 设置 rank id
    options.insert(pair<string,string> (ge::OPTION_EXEC_DEPLOY_MODE,"0"));
    options.insert(pair<string,string> (ge::OPTION_EXEC_RANK_ID,"1"));
    options.insert(pair<string,string> (ge::OPTION_EXEC_PROFILING_MODE,"0"));

    // hccl 算法配置
    options.insert(pair<string,string> ("HCCL_algorithm","level0:ring;level1:H-D_R"));

    MOCKER(HcomLoadRanktableFile)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));

    // 实验室场景 hcom_init成功：成功
    ret = HcomPlugin::Instance().Initialize(options);
    EXPECT_EQ(ret, ge::SUCCESS);

    
    ret = HcomPlugin::Instance().Finalize();
    EXPECT_EQ(ret, ge::SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(HcomPluginTest, ut_hcom_plugin_algo_config_hcclalgo_level03)
{
    ge ::Status ret;
    std::map<std::string,std::string> options;
   
    // 未设置 rank table：失败
    options.insert(pair<string,string> (ge::OPTION_EXEC_RANK_TABLE_FILE,"rank_table.json"));
    // 实验室场景 设置 rank id
    options.insert(pair<string,string> (ge::OPTION_EXEC_DEPLOY_MODE,"0"));
    options.insert(pair<string,string> (ge::OPTION_EXEC_RANK_ID,"1"));
    options.insert(pair<string,string> (ge::OPTION_EXEC_PROFILING_MODE,"0"));

    // hccl 算法配置
    options.insert(pair<string,string> ("HCCL_algorithm","level0:ring;level3:H-D_R"));

    MOCKER(HcomLoadRanktableFile)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));
    
    // 实验室场景 hcom_init成功：成功
    ret = HcomPlugin::Instance().Initialize(options);
    EXPECT_EQ(ret, ge::SUCCESS);


    
    ret = HcomPlugin::Instance().Finalize();
    EXPECT_EQ(ret, ge::SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(HcomPluginTest, ut_hcom_plugin_algo_config_hcclalgo_level23)
{
    ge ::Status ret;
    std::map<std::string,std::string> options;

    // 未设置 rank table：失败
    options.insert(pair<string,string> (ge::OPTION_EXEC_RANK_TABLE_FILE,"rank_table.json"));
    // 实验室场景 设置 rank id
    options.insert(pair<string,string> (ge::OPTION_EXEC_DEPLOY_MODE,"0"));
    options.insert(pair<string,string> (ge::OPTION_EXEC_RANK_ID,"1"));
    options.insert(pair<string,string> (ge::OPTION_EXEC_PROFILING_MODE,"0"));

    // hccl 算法配置
    options.insert(pair<string,string> ("HCCL_algorithm","level2:ring;level3:H-D_R"));

    MOCKER(HcomLoadRanktableFile)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));

    // 实验室场景 hcom_init成功：成功
    ret = HcomPlugin::Instance().Initialize(options);
    EXPECT_EQ(ret, ge::SUCCESS);

    
    ret = HcomPlugin::Instance().Finalize();
    EXPECT_EQ(ret, ge::SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(HcomPluginTest, ut_hcom_plugin_algo_config_hcclalgo_level0123)
{
    ge ::Status ret;
    std::map<std::string,std::string> options;
    
    // 未设置 rank table：失败
    options.insert(pair<string,string> (ge::OPTION_EXEC_RANK_TABLE_FILE,"rank_table.json"));
    // 实验室场景 设置 rank id
    options.insert(pair<string,string> (ge::OPTION_EXEC_DEPLOY_MODE,"0"));
    options.insert(pair<string,string> (ge::OPTION_EXEC_RANK_ID,"1"));
    options.insert(pair<string,string> (ge::OPTION_EXEC_PROFILING_MODE,"0"));

    // hccl 算法配置
    options.insert(pair<string,string> ("HCCL_algorithm","level0:ring;level1:H-D_R;level2:fullmesh;level3:ring"));

    MOCKER(HcomLoadRanktableFile)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));

    // 实验室场景 hcom_init成功：成功
    ret = HcomPlugin::Instance().Initialize(options);
    EXPECT_EQ(ret, ge::SUCCESS);

    
    ret = HcomPlugin::Instance().Finalize();
    EXPECT_EQ(ret, ge::SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(HcomPluginTest, ut_hcom_plugin_algo_config_hcclalgo_foramt_invalid_0)
{
    ge ::Status ret;
    std::map<std::string,std::string> options;

    
    
    // 未设置 rank table：失败
    options.insert(pair<string,string> (ge::OPTION_EXEC_RANK_TABLE_FILE,"rank_table.json"));
    // 实验室场景 设置 rank id
    options.insert(pair<string,string> (ge::OPTION_EXEC_DEPLOY_MODE,"0"));
    options.insert(pair<string,string> (ge::OPTION_EXEC_RANK_ID,"1"));
    options.insert(pair<string,string> (ge::OPTION_EXEC_PROFILING_MODE,"0"));

    // hccl 算法配置
    options.insert(pair<string,string> ("HCCL_algorithm","level:ring;level1:H-D_R;level2:fullmesh;level3:ring"));

    
    // 实验室场景 hcom_init成功：成功
    ret = HcomPlugin::Instance().Initialize(options);
    EXPECT_NE(ret, ge::SUCCESS);

    
    ret = HcomPlugin::Instance().Finalize();
    EXPECT_EQ(ret, ge::SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(HcomPluginTest, ut_hcom_plugin_algo_config_hcclalgo_foramt_invalid_1)
{
    ge ::Status ret;
    std::map<std::string,std::string> options;

    
    
    // 未设置 rank table：失败
    options.insert(pair<string,string> (ge::OPTION_EXEC_RANK_TABLE_FILE,"rank_table.json"));
    // 实验室场景 设置 rank id
    options.insert(pair<string,string> (ge::OPTION_EXEC_DEPLOY_MODE,"0"));
    options.insert(pair<string,string> (ge::OPTION_EXEC_RANK_ID,"1"));
    options.insert(pair<string,string> (ge::OPTION_EXEC_PROFILING_MODE,"0"));

    // hccl 算法配置
    options.insert(pair<string,string> ("HCCL_algorithm","levelx:ring;level1:H-D_R;level2:fullmesh;level3:ring"));

    
    // 实验室场景 hcom_init成功：成功
    ret = HcomPlugin::Instance().Initialize(options);
    EXPECT_NE(ret, ge::SUCCESS);

    
    ret = HcomPlugin::Instance().Finalize();
    EXPECT_EQ(ret, ge::SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(HcomPluginTest, ut_hcom_plugin_algo_config_hcclalgo_foramt_invalid_2)
{
    ge ::Status ret;
    std::map<std::string,std::string> options;

    
    
    // 未设置 rank table：失败
    options.insert(pair<string,string> (ge::OPTION_EXEC_RANK_TABLE_FILE,"rank_table.json"));
    // 实验室场景 设置 rank id
    options.insert(pair<string,string> (ge::OPTION_EXEC_DEPLOY_MODE,"0"));
    options.insert(pair<string,string> (ge::OPTION_EXEC_RANK_ID,"1"));
    options.insert(pair<string,string> (ge::OPTION_EXEC_PROFILING_MODE,"0"));

    // hccl 算法配置
    options.insert(pair<string,string> ("HCCL_algorithm","level0:ring;level1:H-D_R;level2:fullmesh;level4:ring"));

    
    // 实验室场景 hcom_init成功：成功
    ret = HcomPlugin::Instance().Initialize(options);
    EXPECT_NE(ret, ge::SUCCESS);

    
    ret = HcomPlugin::Instance().Finalize();
    EXPECT_EQ(ret, ge::SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(HcomPluginTest, ut_hcom_plugin_algo_config_hcclalgo_foramt_invalid_3)
{
    ge ::Status ret;
    std::map<std::string,std::string> options;

    
    
    // 未设置 rank table：失败
    options.insert(pair<string,string> (ge::OPTION_EXEC_RANK_TABLE_FILE,"rank_table.json"));
    // 实验室场景 设置 rank id
    options.insert(pair<string,string> (ge::OPTION_EXEC_DEPLOY_MODE,"0"));
    options.insert(pair<string,string> (ge::OPTION_EXEC_RANK_ID,"1"));
    options.insert(pair<string,string> (ge::OPTION_EXEC_PROFILING_MODE,"0"));

    // hccl 算法配置
    options.insert(pair<string,string> ("HCCL_algorithm","level0:ring;;level2:fullmesh;level4:ring"));

    
    // 实验室场景 hcom_init成功：成功
    ret = HcomPlugin::Instance().Initialize(options);
    EXPECT_NE(ret, ge::SUCCESS);

    
    ret = HcomPlugin::Instance().Finalize();
    EXPECT_EQ(ret, ge::SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(HcomPluginTest, ut_hcom_plugin_algo_config_hcclalgo_foramt_invalid_4)
{
    ge ::Status ret;
    std::map<std::string,std::string> options;

    
    
    // 未设置 rank table：失败
    options.insert(pair<string,string> (ge::OPTION_EXEC_RANK_TABLE_FILE,"rank_table.json"));
    // 实验室场景 设置 rank id
    options.insert(pair<string,string> (ge::OPTION_EXEC_DEPLOY_MODE,"0"));
    options.insert(pair<string,string> (ge::OPTION_EXEC_RANK_ID,"1"));
    options.insert(pair<string,string> (ge::OPTION_EXEC_PROFILING_MODE,"0"));

    // hccl 算法配置
    options.insert(pair<string,string> ("HCCL_algorithm","level0:;level1:H-D_R;level2:fullmesh;level3:ring"));

    
    // 实验室场景 hcom_init成功：成功
    ret = HcomPlugin::Instance().Initialize(options);
    EXPECT_NE(ret, ge::SUCCESS);

    
    ret = HcomPlugin::Instance().Finalize();
    EXPECT_EQ(ret, ge::SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(HcomPluginTest, ut_hcom_plugin_algo_config_hcclalgo_foramt_invalid_5)
{
    ge ::Status ret;
    std::map<std::string,std::string> options;

    
    
    // 未设置 rank table：失败
    options.insert(pair<string,string> (ge::OPTION_EXEC_RANK_TABLE_FILE,"rank_table.json"));
    // 实验室场景 设置 rank id
    options.insert(pair<string,string> (ge::OPTION_EXEC_DEPLOY_MODE,"0"));
    options.insert(pair<string,string> (ge::OPTION_EXEC_RANK_ID,"1"));
    options.insert(pair<string,string> (ge::OPTION_EXEC_PROFILING_MODE,"0"));

    // hccl 算法配置
    options.insert(pair<string,string> ("HCCL_algorithm","level0:8pring;level1:H-D_R;level2:fullmesh;level3:ring"));

    
    // 实验室场景 hcom_init成功：成功
    ret = HcomPlugin::Instance().Initialize(options);
    EXPECT_NE(ret, ge::SUCCESS);

    
    ret = HcomPlugin::Instance().Finalize();
    EXPECT_EQ(ret, ge::SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(HcomPluginTest, ut_hcom_plugin_algo_config_hcclalgo_foramt_invalid_6)
{
    ge ::Status ret;
    std::map<std::string,std::string> options;

    
    
    // 未设置 rank table：失败
    options.insert(pair<string,string> (ge::OPTION_EXEC_RANK_TABLE_FILE,"rank_table.json"));
    // 实验室场景 设置 rank id
    options.insert(pair<string,string> (ge::OPTION_EXEC_DEPLOY_MODE,"0"));
    options.insert(pair<string,string> (ge::OPTION_EXEC_RANK_ID,"1"));
    options.insert(pair<string,string> (ge::OPTION_EXEC_PROFILING_MODE,"0"));

    // hccl 算法配置
    options.insert(pair<string,string> ("HCCL_algorithm","level0:ring;level1:H-D_R;level2:4pmesh;level3:ring"));

    
    // 实验室场景 hcom_init成功：成功
    ret = HcomPlugin::Instance().Initialize(options);
    EXPECT_NE(ret, ge::SUCCESS);

    
    ret = HcomPlugin::Instance().Finalize();
    EXPECT_EQ(ret, ge::SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(HcomPluginTest, ut_hcom_plugin_algo_config_hcclalgo_foramt_invalid_7)
{
    ge ::Status ret;
    std::map<std::string,std::string> options;

    
    
    // 未设置 rank table：失败
    options.insert(pair<string,string> (ge::OPTION_EXEC_RANK_TABLE_FILE,"rank_table.json"));
    // 实验室场景 设置 rank id
    options.insert(pair<string,string> (ge::OPTION_EXEC_DEPLOY_MODE,"0"));
    options.insert(pair<string,string> (ge::OPTION_EXEC_RANK_ID,"1"));
    options.insert(pair<string,string> (ge::OPTION_EXEC_PROFILING_MODE,"0"));

    // hccl 算法配置
    options.insert(pair<string,string> ("HCCL_algorithm","level0:ring;level1:H-D;level2:fullmesh;level3:ring"));

    
    // 实验室场景 hcom_init成功：成功
    ret = HcomPlugin::Instance().Initialize(options);
    EXPECT_NE(ret, ge::SUCCESS);

    
    ret = HcomPlugin::Instance().Finalize();
    EXPECT_EQ(ret, ge::SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(HcomPluginTest, ut_hcom_plugin_algo_config_hcclalgo_foramt_invalid_8)
{
    ge ::Status ret;
    std::map<std::string,std::string> options;

    
    
    // 未设置 rank table：失败
    options.insert(pair<string,string> (ge::OPTION_EXEC_RANK_TABLE_FILE,"rank_table.json"));
    // 实验室场景 设置 rank id
    options.insert(pair<string,string> (ge::OPTION_EXEC_DEPLOY_MODE,"0"));
    options.insert(pair<string,string> (ge::OPTION_EXEC_RANK_ID,"1"));
    options.insert(pair<string,string> (ge::OPTION_EXEC_PROFILING_MODE,"0"));

    // hccl 算法配置
    options.insert(pair<string,string> ("HCCL_algorithm","level0:ring;level0:ring"));

    
    // 实验室场景 hcom_init成功：成功
    ret = HcomPlugin::Instance().Initialize(options);
    EXPECT_NE(ret, ge::SUCCESS);

    
    ret = HcomPlugin::Instance().Finalize();
    EXPECT_EQ(ret, ge::SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(HcomPluginTest, ut_hcom_plugin_initialize_by_comm_pytorch)
{
    ge ::Status ret;
    std::map<std::string,std::string> options;

    
    
    options.insert(pair<string,string> (ge::OPTION_EXEC_PROFILING_MODE,"0"));

    ret = HcomPlugin::Instance().Initialize(options);
    EXPECT_EQ(ret, ge::SUCCESS);

    
    ret = HcomPlugin::Instance().Finalize();
    EXPECT_EQ(ret, ge::SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(HcomPluginTest, ut_HcomPluginFinalize)
{
    ge ::Status ret;
    std::map<std::string,std::string> options;

    
    options.insert(pair<string,string> (ge::OPTION_EXEC_RANK_TABLE_FILE,"rank_table.json"));
    options.insert(pair<string,string> (ge::OPTION_EXEC_DEPLOY_MODE,"0"));
    options.insert(pair<string,string> (ge::OPTION_EXEC_RANK_ID,"1"));
    options.insert(pair<string,string> (ge::OPTION_EXEC_PROFILING_MODE,"0"));
 
    
    // 实验室场景 hcom_init成功：成功
    ret = HcomPlugin::Instance().Initialize(options);
   // EXPECT_NE(ret, ge::SUCCESS);

    

    ret = HcomPlugin::Instance().Finalize();
    EXPECT_EQ(ret, ge::SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(HcomPluginTest, ut_hcom_plugin_masterinfo)
{
    ge ::Status ret;
    std::map<std::string,std::string> options;

    options.insert(pair<string,string> (ge::OPTION_EXEC_DEPLOY_MODE,"0"));
    options.insert(pair<string,string> (ge::OPTION_EXEC_CM_CHIEF_IP,"127.0.0.1"));
    options.insert(pair<string,string> (ge::OPTION_EXEC_CM_CHIEF_PORT,"6000"));
    options.insert(pair<string,string> (ge::OPTION_EXEC_CM_WORKER_SIZE,"2"));
    options.insert(pair<string,string> (ge::OPTION_EXEC_CM_CHIEF_DEVICE,"0"));
    MOCKER(HcomInitByMasterInfo)
    .expects(atMost(1))
    .will(returnValue(HCCL_SUCCESS));
    // 实验室场景 hcom_init成功：成功
    ret = HcomPlugin::Instance().Initialize(options);
    EXPECT_EQ(ret, ge::SUCCESS);

    
    ret = HcomPlugin::Instance().Finalize();
    EXPECT_EQ(ret, ge::SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(HcomPluginTest, ut_hcom_plugin_congif_deterministic_0)
{
    ge ::Status ret;
    std::map<std::string,std::string> options;
    options.insert(pair<string,string> (ge::DETERMINISTIC,"0"));

    ret = HcomPlugin::Instance().Initialize(options);
    EXPECT_EQ(ret, ge::SUCCESS);

    
    ret = HcomPlugin::Instance().Finalize();
    EXPECT_EQ(ret, ge::SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(HcomPluginTest, ut_hcom_plugin_congif_deterministic_1)
{
    ge ::Status ret;
    std::map<std::string,std::string> options;
    options.insert(pair<string,string> (ge::DETERMINISTIC,"1"));

    ret = HcomPlugin::Instance().Initialize(options);
    EXPECT_EQ(ret, ge::SUCCESS);

    
    ret = HcomPlugin::Instance().Finalize();
    EXPECT_EQ(ret, ge::SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(HcomPluginTest, ut_hcom_plugin_congif_deterministic_2)
{
    ge ::Status ret;
    std::map<std::string,std::string> options;
    options.insert(pair<string,string> (ge::DETERMINISTIC,"2"));

    ret = HcomPlugin::Instance().Initialize(options);
    EXPECT_EQ(ret, ge::SUCCESS);

    ret = HcomPlugin::Instance().Finalize();
    EXPECT_EQ(ret, ge::SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(HcomPluginTest, ut_hcom_plugin_congif_deterministic_empty)
{
    ge ::Status ret;
    std::map<std::string,std::string> options;
    options.insert(pair<string,string> (ge::DETERMINISTIC,""));

    ret = HcomPlugin::Instance().Initialize(options);

    
    ret = HcomPlugin::Instance().Finalize();

    GlobalMockObject::verify();
}

TEST_F(HcomPluginTest, ut_hcom_plugin_congif_deterministic)
{
    ge ::Status ret;
    std::map<std::string,std::string> options;
    options.insert(pair<string,string> (ge::DETERMINISTIC,"1"));
    setenv("HCCL_DETERMINISTIC", "TRUE", 1);

    ret = HcomPlugin::Instance().Initialize(options);

    
    ret = HcomPlugin::Instance().Finalize();

    GlobalMockObject::verify();
    unsetenv("HCCL_DETERMINISTIC");
}

/**
 * @tc.name  : GetOpsKernelInfoPtr_ShouldReturnValidPtr_WhenPtrIsNotNull
 * @tc.number: HcomPluginTest_003
 * @tc.desc  : Test when hcomOpsKernelInfoStoreInfoPtr_ is not null, GetOpsKernelInfoPtr should return it
 */
TEST_F(HcomPluginTest, GetOpsKernelInfoPtr_ShouldReturnValidPtr_WhenPtrIsNotNull)
{
    HcomPlugin *plugin_;
    plugin_ = &HcomPlugin::Instance();
    plugin_->hcomOpsKernelInfoStoreInfoPtr_ = std::make_shared<HcomOpsKernelInfoStore>();
    HcomOpsKernelInfoStorePtr opsKernelInfoStorePtr;
    plugin_->GetOpsKernelInfoPtr(opsKernelInfoStorePtr);
    
    // 检查返回的指针是否有效
    EXPECT_NE(opsKernelInfoStorePtr, nullptr);

    GlobalMockObject::verify();
}