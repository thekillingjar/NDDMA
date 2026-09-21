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
#include "base/base_types.h"
#include "base/model_info.h"
#include "generator/preprocess/args_manager.h"
#include "test_expr/test_stub.h"
#include "stub/stub_model_info.h"
#include "tiling_code_generator.h"
#include "reuse_group_utils/reuse_group_utils.h"
#include "common/test_common_utils.h"
#include "test_common_utils.h"

using namespace att;

class TestAtt : public ::testing::Test {
 public:
  static void SetUpTestCase()
  {
    std::cout << "Test begin." << std::endl;
  }
  static void TearDownTestCase()
  {
    std::cout << "Test end." << std::endl;
  }
  
  void SetUp() override {
    setenv("ASCEND_GLOBAL_LOG_LEVEL", "4", 1);
    model_info_ = CreateModelInfo();
  }

  void TearDown() override {
    // 清理测试生成的临时文件
    autofuse::test::CleanupTestArtifacts();
    unsetenv("ASCEND_GLOBAL_LOG_LEVEL");
  }
  ModelInfo model_info_;
};

extern void AddHeaderGuardToFile(const std::string& file_name, const std::string& macro_name);
TEST_F(TestAtt, test_generator)
{
  TilingModelInfo modelInfos;
  auto model_info = GetMatmulL2TileInfo();
  auto op_name = "OpTest0";
  modelInfos.emplace_back(model_info);
  modelInfos.emplace_back(model_info_);
  TilingCodeGenConfig config;
  config.path = "./";
  config.type = TilingImplType::HIGH_PERF;
  config.gen_extra_infos = true;
  config.tiling_data_type_name = "MMTilingData";
  EXPECT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, 0UL}, modelInfos), ge::SUCCESS);
  TilingCodeGenerator generator;
  EXPECT_EQ(generator.GenTilingCode(op_name, modelInfos, config), ge::SUCCESS);
  AddHeaderGuardToFile("autofuse_tiling_func_common.h", "__AUTOFUSE_TILING_FUNC_COMMON_H__");
  
  auto ret = std::system(std::string("cp ").append(ST_DIR).append("/testcase/tiling_func_main.cpp ./ -f").c_str());
  ret = std::system(std::string("cp ").append(ST_DIR).append("/testcase/op_log.h ./ -f").c_str());
  ret = autofuse::test::CopyStubFiles(ST_DIR, "testcase/stub/");
  EXPECT_EQ(ret, 0);

  ret = std::system("g++ tiling_func_main.cpp OpTest0_*_tiling_func.cpp -I ./ -o tiling_func_main -Werror");
  EXPECT_EQ(ret, 0);

  ret = std::system("./tiling_func_main");
  EXPECT_EQ(ret, 0);
}

TEST_F(TestAtt, test_ceiling_generator)
{
  TilingModelInfo modelInfos;
  auto model_info = CreateCeilingModel();
  auto op_name = "OpTest2";
  modelInfos.emplace_back(model_info);
  TilingCodeGenConfig config;
  config.path = "./";
  config.type = TilingImplType::HIGH_PERF;
  config.gen_extra_infos = true;
  config.tiling_data_type_name = "CeilingTilingData";
  TilingCodeGenerator generator;
  EXPECT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, 0UL}, modelInfos), ge::SUCCESS);
  EXPECT_EQ(generator.GenTilingCode(op_name, modelInfos, config), ge::SUCCESS);
  AddHeaderGuardToFile("autofuse_tiling_func_common.h", "__AUTOFUSE_TILING_FUNC_COMMON_H__");
  auto ret = std::system(std::string("cp ").append(ST_DIR).append("/testcase/tiling_func_ceiling.cpp ./ -f").c_str());
  ret = std::system(std::string("cp ").append(ST_DIR).append("/testcase/op_log.h ./ -f").c_str());
  ret = autofuse::test::CopyStubFiles(ST_DIR, "testcase/stub/");
  EXPECT_EQ(ret, 0);

  ret = std::system("g++ tiling_func_ceiling.cpp OpTest2_*_tiling_func.cpp  -I ./ -o tiling_func_main -Werror");
  EXPECT_EQ(ret, 0);

  ret = std::system("./tiling_func_main");
  EXPECT_EQ(ret, 0);
}