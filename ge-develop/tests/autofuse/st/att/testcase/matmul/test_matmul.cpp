/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <fstream>
#include <string>
#include <regex>
#include "gtest/gtest.h"
#include "matmul/stub_modelinfo.h"
#include "gen_tiling_impl.h"
#include "tiling_code_generator.h"
#include "reuse_group_utils/reuse_group_utils.h"
#include "common/test_common_utils.h"
#include "test_common_utils.h"

using namespace att;
class TestGenMatmulModelInfoE2E : public ::testing::Test {
 public:
  static void TearDownTestCase() {
    std::cout << "Test end." << std::endl;
  }
  static void SetUpTestCase() {
    std::cout << "Test begin." << std::endl;
  }
  void SetUp() override {
    // Code here will be called immediately after the constructor (right
    // before each test).
    setenv("ASCEND_GLOBAL_LOG_LEVEL", "4", 1);
  }

  void TearDown() override {
    // 清理测试生成的临时文件
    autofuse::test::CleanupTestArtifacts();
    unsetenv("ASCEND_GLOBAL_LOG_LEVEL");
  }
};

void AddHeaderGuardToFile(const std::string& file_name, const std::string& macro_name) {
    std::string content;
    std::ifstream in_file(file_name);
    if (in_file.is_open()) {
        std::string line;
        while (std::getline(in_file, line)) {
            content += line + "\n";
        }
        in_file.close();
    }

    std::ofstream out_file;
    out_file.open(file_name, std::ios::out);
    out_file << "#ifndef " << macro_name << "\n";
    out_file << "#define " << macro_name << "\n";
    out_file << "\n";
    out_file << content;
    out_file << "\n";
    out_file << "#endif // " << macro_name << "\n";
    out_file.close();
}

TEST_F(TestGenMatmulModelInfoE2E, case1) {
  TilingModelInfo modelInfos;
  ModelInfo modelInfo = GenMatmulModelInfo();
  modelInfos.emplace_back(modelInfo);
  TilingCodeGenConfig config;
  config.path = "./";
  config.type = TilingImplType::HIGH_PERF;
  config.gen_extra_infos = true;
  config.tiling_data_type_name = "MMTilingData";
  TilingCodeGenerator generator;
  EXPECT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, 0UL}, modelInfos), ge::SUCCESS);
  EXPECT_EQ(generator.GenTilingCode("Matmul", modelInfos, config), ge::SUCCESS);
  AddHeaderGuardToFile("autofuse_tiling_func_common.h", "__AUTOFUSE_TILING_FUNC_COMMON_H__");
  
  auto ret = std::system(std::string("cp ").append(ST_DIR).append("/testcase/tiling_func_matmul_main.cpp ./ -f").c_str());
  ret = std::system(std::string("cp ").append(ST_DIR).append("/testcase/op_log.h ./ -f").c_str());
  ret = autofuse::test::CopyStubFiles(ST_DIR, "testcase/stub/");
  EXPECT_EQ(ret, 0);

  ret = std::system("g++ tiling_func_matmul_main.cpp Matmul_*tiling_func.cpp -I ./ -o tiling_func_main -Werror");
  EXPECT_EQ(ret, 0);

  ret = std::system("./tiling_func_main");
  EXPECT_EQ(ret, 0);
}

TEST_F(TestGenMatmulModelInfoE2E, axes_reorder) {
  TilingModelInfo modelInfos;
  ModelInfo modelInfo = GenMatmulModelInfo();
  modelInfos.emplace_back(modelInfo);
  TilingCodeGenConfig config;
  config.path = "./";
  config.type = TilingImplType::AXES_REORDER;
  config.gen_extra_infos = true;
  config.tiling_data_type_name = "MMTilingData";
  TilingCodeGenerator generator;
  EXPECT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, 0UL}, modelInfos), ge::SUCCESS);
  EXPECT_EQ(generator.GenTilingCode("Matmul", modelInfos, config), ge::SUCCESS);
  AddHeaderGuardToFile("autofuse_tiling_func_common.h", "__AUTOFUSE_TILING_FUNC_COMMON_H__");
  
  auto ret = std::system(std::string("cp ").append(ST_DIR).append("/testcase/tiling_func_matmul_main.cpp ./ -f").c_str());
  ret = std::system(std::string("cp ").append(ST_DIR).append("/testcase/op_log.h ./ -f").c_str());
  ret = autofuse::test::CopyStubFiles(ST_DIR, "testcase/stub/");
  EXPECT_EQ(ret, 0);

  ret = std::system("g++ tiling_func_matmul_main.cpp Matmul_*tiling_func.cpp -I ./ -o tiling_func_main -Werror -DSTUB_LOG");
  EXPECT_EQ(ret, 0);

  ret = std::system("./tiling_func_main");
  EXPECT_EQ(ret, 0);
}