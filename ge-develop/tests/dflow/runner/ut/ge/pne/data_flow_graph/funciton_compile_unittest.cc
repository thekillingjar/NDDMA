/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <fstream>
#include <gtest/gtest.h>
#include "dflow/compiler/data_flow_graph/function_compile.h"
#include "proto/dflow.pb.h"
#include "dflow/compiler/data_flow_graph/compile_config_json.h"
#include "depends/mmpa/src/mmpa_stub.h"
#include "graph/ge_global_options.h"
#include "framework/common/ge_types.h"

using namespace testing;
namespace ge {
class FunctionCompileTest : public Test {
 protected:
  void SetUp() {
    std::string cmd = "mkdir -p temp; cd temp; touch libtest.so";
    (void) system(cmd.c_str());
    std::ofstream cmakefile("./temp/CMakeLists.txt");
    {
      cmakefile << "cmake_minimum_required(VERSION 3.5)\n";
      // Prevent cmake from testing the toolchain
      cmakefile << "set(CMAKE_C_COMPILER_FORCED TRUE)\n";
      cmakefile << "set(CMAKE_CXX_COMPILER_FORCED TRUE)\n";
      cmakefile << "project(test)\n";
      cmakefile << "set(BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})\n";
      cmakefile << "execute_process(\n";
      cmakefile << "\tCOMMAND cp ${BASE_DIR}/lib${UDF_TARGET_LIB}.so ${RELEASE_DIR} RESULT_VARIABLE ret\n";
      cmakefile << ")\n";
      cmakefile << "unset(CMAKE_C_COMPILER_FORCED)\n";
      cmakefile << "unset(CMAKE_CXX_COMPILER_FORCED)\n";
      cmakefile << "if (NOT ret EQUAL 0)\n";
      cmakefile << "message(FATAL_ERROR \"error\")\n";
      cmakefile << "endif()\n";
    }
    {
      auto &global_options_mutex = GetGlobalOptionsMutex();
      const std::lock_guard<std::mutex> lock(global_options_mutex);
      auto &global_options = GetMutableGlobalOptions();
      global_options[OPTION_NUMA_CONFIG] =
          R"({"cluster":[{"cluster_nodes":[{"is_local":true, "item_list":[{"item_id":0}], "node_id":0, "node_type":"TestNodeType1"}]}],"item_def":[{"aic_type":"[DAVINCI_V100:10]","item_type":"","memory":"[DDR:80GB]","resource_type":"Ascend"}],"node_def":[{"item_type":"","links_mode":"TCP:128Gb","node_type":"TestNodeType1","resource_type":"X86","support_links":"[ROCE]"}]})";
    }
  }

  void TearDown() {
    std::string cmd = "rm -rf temp";
    (void) system(cmd.c_str());
  }
};

TEST_F(FunctionCompileTest, FunctionPpCompile) {
  CompileConfigJson::FunctionPpConfig function_pp_config;
  function_pp_config.workspace = "./temp";
  function_pp_config.target_bin = "libtest.so";
  function_pp_config.cmakelist = "CMakeLists.txt";
  CompileConfigJson::RunningResourceInfo cpu_res_info = {"cpu", 2};
  CompileConfigJson::RunningResourceInfo mem_res_info = {"memory", 200};
  function_pp_config.running_resources_info.emplace_back(cpu_res_info);
  function_pp_config.running_resources_info.emplace_back(mem_res_info);
  function_pp_config.toolchain_map["X86"] = "/usr/bin/g++";
  function_pp_config.toolchain_map["Ascend"] = "/usr/bin/g++";

  FunctionCompile function_compile("pp0", function_pp_config);
  auto ret = function_compile.CompileAllResourceType();
  EXPECT_EQ(ret, SUCCESS);
  auto compile_result = function_compile.GetCompileResult();
  EXPECT_EQ(compile_result.compile_bin_info["X86"], "./temp/build/_test/X86/release/");
  EXPECT_EQ(compile_result.compile_bin_info["Ascend"], "./temp/build/_test/Ascend/release/");
  EXPECT_EQ(compile_result.running_resources_info.size(), 2);
  EXPECT_EQ(compile_result.running_resources_info[0].resource_type, "cpu");
  EXPECT_EQ(compile_result.running_resources_info[0].resource_num, 2);
  EXPECT_EQ(compile_result.running_resources_info[1].resource_type, "memory");
  EXPECT_EQ(compile_result.running_resources_info[1].resource_num, 200);
}

TEST_F(FunctionCompileTest, FunctionPpCompileAllError) {
  CompileConfigJson::FunctionPpConfig function_pp_config;
  function_pp_config.workspace = "./temp";
  function_pp_config.target_bin = "libno_so.so";
  function_pp_config.cmakelist = "CMakeLists.txt";
  CompileConfigJson::RunningResourceInfo cpu_res_info = {"cpu", 2};
  CompileConfigJson::RunningResourceInfo mem_res_info = {"memory", 200};
  function_pp_config.running_resources_info.emplace_back(cpu_res_info);
  function_pp_config.running_resources_info.emplace_back(mem_res_info);
  function_pp_config.toolchain_map["X86"] = "/usr/bin/g++";
  function_pp_config.toolchain_map["Ascend"] = "/usr/bin/g++";

  FunctionCompile function_compile("pp0", function_pp_config);
  auto ret = function_compile.CompileAllResourceType();
  EXPECT_EQ(ret, FAILED);
}

TEST_F(FunctionCompileTest, FunctionPpCompileWithInvalidToolchain) {
  CompileConfigJson::FunctionPpConfig function_pp_config;
  function_pp_config.workspace = "./temp";
  function_pp_config.target_bin = "libtest.so";
  function_pp_config.cmakelist = "./test/CMakeLists.txt";
  function_pp_config.toolchain_map["X86"] = "/usr/bin/g++;";
  function_pp_config.toolchain_map["Ascend"] = "/usr/bin/g++;";

  FunctionCompile function_compile("pp0", function_pp_config);
  auto ret = function_compile.CompileAllResourceType();
  EXPECT_EQ(ret, FAILED);
}

TEST_F(FunctionCompileTest, FunctionPpCompileWithInvalidWorkspace) {
  CompileConfigJson::FunctionPpConfig function_pp_config;
  function_pp_config.workspace = "./temp;rm -rf *";
  function_pp_config.target_bin = "libtest.so";

  FunctionCompile function_compile("pp0", function_pp_config);
  auto ret = function_compile.CompileAllResourceType();
  EXPECT_NE(ret, SUCCESS);
}

TEST_F(FunctionCompileTest, FunctionPpCompileWithInvalidTargetBin) {
  CompileConfigJson::FunctionPpConfig function_pp_config;
  function_pp_config.workspace = "./temp";
  function_pp_config.target_bin = "libtest.so;rm -rf *";

  FunctionCompile function_compile("pp0", function_pp_config);
  auto ret = function_compile.CompileAllResourceType();
  EXPECT_NE(ret, SUCCESS);
}

TEST_F(FunctionCompileTest, FunctionPpCompileWithInvalidCmakeList) {
  CompileConfigJson::FunctionPpConfig function_pp_config;
  function_pp_config.workspace = "./temp";
  function_pp_config.target_bin = "libtest.so";
  function_pp_config.cmakelist = "CMakeLists.txt;rm -rf *";

  FunctionCompile function_compile("pp0", function_pp_config);
  auto ret = function_compile.CompileAllResourceType();
  EXPECT_EQ(ret, PARAM_INVALID);
}

TEST_F(FunctionCompileTest, FunctionPpCompileWithNoMakefile) {
  CompileConfigJson::FunctionPpConfig function_pp_config;
  function_pp_config.workspace = "./temp";
  function_pp_config.target_bin = "libtest.so";
  CompileConfigJson::RunningResourceInfo cpu_res_info = {"cpu", 2};
  CompileConfigJson::RunningResourceInfo mem_res_info = {"memory", 200};
  function_pp_config.running_resources_info.emplace_back(cpu_res_info);
  function_pp_config.running_resources_info.emplace_back(mem_res_info);
  function_pp_config.toolchain_map["X86"] = "/usr/bin/g++";
  function_pp_config.toolchain_map["Aarch64"] = "/usr/bin/g++";

  FunctionCompile function_compile("pp0", function_pp_config);
  auto ret = function_compile.CompileAllResourceType();
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(FunctionCompileTest, CompileAllResourceTypeFailed) {
  class MockMmpaOpen : public ge::MmpaStubApiGe {
   public:
    virtual INT32 Open2(const CHAR *path_name, INT32 flags, MODE mode) {
      return INT32_MAX;
    }
  };
  CompileConfigJson::FunctionPpConfig function_pp_config;
  function_pp_config.workspace = "./temp";
  function_pp_config.target_bin = "libtest.so";
  function_pp_config.cmakelist = "CMakeLists.txt";
  CompileConfigJson::RunningResourceInfo cpu_res_info = {"cpu", 2};
  CompileConfigJson::RunningResourceInfo mem_res_info = {"memory", 200};
  function_pp_config.running_resources_info.emplace_back(cpu_res_info);
  function_pp_config.running_resources_info.emplace_back(mem_res_info);
  function_pp_config.toolchain_map["X86"] = "/usr/bin/g++";
  function_pp_config.toolchain_map["Ascend"] = "/no_exist_g++";
  FunctionCompile function_compile("pp0", function_pp_config);
  std::string cmd =
      "mkdir -p ./temp/build/_test/X86/build/compile_lock;mkdir -p ./temp/build/_test/Ascend/build/compile_lock;";
  (void)system(cmd.c_str());
  auto ret = function_compile.CompileAllResourceType();
  EXPECT_EQ(ret, FAILED);
  cmd = "rm -rf ./temp/build/_test/X86/build/compile_lock;rm -rf ./temp/build/_test/Ascend/build/compile_lock";
  (void)system(cmd.c_str());
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpaOpen>());
  ret = function_compile.CompileAllResourceType();
  EXPECT_EQ(ret, FAILED);
  MmpaStub::GetInstance().Reset();
}

TEST_F(FunctionCompileTest, GetAscendLatestInstallPath) {
  constexpr const char_t *kAscendHomePath = "ASCEND_HOME_PATH";
  std::string ascend_home_path("/test/ascend_path");
  mmSetEnv(kAscendHomePath, ascend_home_path.c_str(), 1);
  std::string path = FunctionCompile::GetAscendLatestInstallPath();
  EXPECT_EQ(path, ascend_home_path);

  unsetenv(kAscendHomePath);
  path = FunctionCompile::GetAscendLatestInstallPath();
  EXPECT_EQ(path, "/usr/local/Ascend/cann");
}
}  // namespace ge
