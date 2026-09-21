/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/util/json_util.h"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <fcntl.h>
#include "sys/stat.h"
#include "fe_llt_utils.h"
#define private public
#define protected public
#include "common/fe_utils.h"
#include "common/fe_type_utils.h"
#include "common/fe_log.h"
using namespace std;
using namespace testing;
using namespace fe;
using namespace nlohmann;

class STEST_JSON_UTIL : public testing::Test{
  protected:
    static void SetUpTestCase() {
        std::cout << "STEST_JSON_UTIL is Setting Up" << std::endl;
    }
    static void TearDownTestCase() {
        std::cout << "STEST_JSON_UTIL Is Tearing Down" << std::endl;
    }
    // Some expensive resource shared by all tests.
    virtual void SetUp(){
        std::cout << "One json util stest is setting up" << std::endl;
    }
    virtual void TearDown(){
        std::cout << "One json util stest is tearing down" << std::endl;

    }
};

TEST_F(STEST_JSON_UTIL, real_path_fail)
{
    std::string filename = "tbe_custom_opinfo.json";
    string ret = RealPath(filename);
    EXPECT_EQ(0, ret.size());
}

TEST_F(STEST_JSON_UTIL, real_path_fail_1)
{
    const size_t large_size = PATH_MAX + 100;
    std::string large_string(large_size, 'a');
    string ret = RealPath(large_string);
    EXPECT_EQ(0, ret.size());
}

TEST_F(STEST_JSON_UTIL, real_path_fail_2)
{
    const size_t large_size = PATH_MAX + 100;
    std::string large_string(large_size, 'b');
    string ret = GetRealPath(large_string);
    EXPECT_EQ(0, ret.size());
}

TEST_F(STEST_JSON_UTIL, real_path_succ)
{
    std::string filename = GetCodeDir() + "/tests/engines/nn_engine/st/testcase/ops_kernel_store/fe_config/cce_custom_opinfo/cce_custom_opinfo.json";
    string ret = RealPath(filename);
    EXPECT_NE(0, ret.size());
}

TEST_F(STEST_JSON_UTIL, read_json_file_succ)
{
    std::string filename = GetCodeDir() + "/tests/engines/nn_engine/st/testcase/ops_kernel_store/fe_config/cce_custom_opinfo/cce_custom_opinfo.json";
    json j;
    Status ret = ReadJsonFile(filename, j);
    EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(STEST_JSON_UTIL, read_json_file_fail_)
{
    std::string filename = GetCodeDir() + "/tests/engines/nn_engine/st/testcase/ops_kernel_store/fe_config/opinfo_not_exist.json";
    json j;
    Status ret = ReadJsonFile(filename, j);
    EXPECT_EQ(fe::FAILED, ret);
}

TEST_F(STEST_JSON_UTIL, read_json_file_lock_succ)
{
    std::string filename = GetCodeDir() + "/tests/engines/nn_engine/st/testcase/ops_kernel_store/fe_config/cce_custom_opinfo/cce_custom_opinfo.json";
    json j;
    Status ret = ReadJsonFileByLock(filename, j);
    EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(STEST_JSON_UTIL, read_json_file_lock_nonexist_fail)
{
    std::string filename = GetCodeDir() + "/tests/engines/nn_engine/st/testcase/ops_kernel_store/fe_config/opinfo_not_exist.json";
    json j;
    Status ret = ReadJsonFileByLock(filename, j);
    EXPECT_EQ(fe::FAILED, ret);
}

TEST_F(STEST_JSON_UTIL, read_json_file_lock_fail)
{
    std::string filename = GetCodeDir() + "/tests/engines/nn_engine/st/testcase/ops_kernel_store/fe_config/cce_custom_opinfo/cce_custom_opinfo.json";

    // manually lock first
    FILE *fp = fopen(filename.c_str(), "r");
    if (fp == nullptr) {
        EXPECT_EQ(true, false);
    }

    if (FcntlLockFile(filename, fileno(fp), -88, 0) != fe::FAILED) {
        EXPECT_EQ(true, false);
    } else {
        (void)FcntlLockFile(filename, fileno(fp), F_UNLCK, 0);
    }
    fclose(fp);
}
