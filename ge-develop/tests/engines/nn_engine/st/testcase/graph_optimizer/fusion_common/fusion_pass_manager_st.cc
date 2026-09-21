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
#include <nlohmann/json.hpp>
#include <iostream>
#include <string>
#include <vector>

#define protected public
#define private public
#include "common/aicore_util_constants.h"
#include "common/configuration.h"
#include "fe_llt_utils.h"
#include "graph_optimizer/fusion_common/fusion_pass_manager.h"
#undef protected
#undef private

using namespace std;
using namespace fe;
using namespace ge;

class FUSION_PASS_MANAGER_STEST: public testing::Test {
 protected:
  void SetUp() {
  }

  void TearDown() {
  }
};

TEST_F(FUSION_PASS_MANAGER_STEST, initialize_suc) {
  FusionPassManager fusion_pass_manager;
  fusion_pass_manager.init_flag_ = true;
  Status ret = fusion_pass_manager.Initialize("AIcoreEngine");
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(FUSION_PASS_MANAGER_STEST, finalize_suc1) {
  FusionPassManager fusion_pass_manager;
  fusion_pass_manager.init_flag_ = false;
  Status ret = fusion_pass_manager.Finalize();
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(FUSION_PASS_MANAGER_STEST, finalize_suc2) {
  FusionPassManager fusion_pass_manager;
  Status ret = fusion_pass_manager.Finalize();
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(FUSION_PASS_MANAGER_STEST, CloseFusionPassPlugin_suc) {
  FusionPassManager fusion_pass_manager;
  PluginManagerPtr plugin_manager_ptr = std::make_shared<PluginManager>("pass_file");
  fusion_pass_manager.fusion_pass_plugin_manager_vec_ = {plugin_manager_ptr};
  fusion_pass_manager.CloseFusionPassPlugin();
  size_t size_vec = fusion_pass_manager.fusion_pass_plugin_manager_vec_.size();
  EXPECT_EQ(0, size_vec);
}

TEST_F(FUSION_PASS_MANAGER_STEST, OpenPassPath_failed) {
  FusionPassManager fusion_pass_manager;
  string file_path;
  vector<string> get_pass_files;
  Status ret = fusion_pass_manager.OpenPassPath(file_path, get_pass_files);
  EXPECT_EQ(fe::FAILED, ret);
}

TEST_F(FUSION_PASS_MANAGER_STEST, test_InitFusionPassPlugin) {
  FusionPassManager fusion_pass_manager;
  std::string engine_name = fe::AI_CORE_NAME;
  const std::string current_dir = "./";
  const std::string aicore_so_path = current_dir + "plugin/opskernel/fusion_pass/";
  std::cout << "aicore_so_path is " << aicore_so_path << std::endl;
  CreateDir(aicore_so_path);
  const std::string fileName = aicore_so_path + "libops_fusion_pass_aicore.so";
  CreateFileAndFillContent(fileName);
  Configuration::Instance(engine_name).lib_path_ = current_dir;
  Status ret = fusion_pass_manager.InitFusionPassPlugin(engine_name);
  EXPECT_EQ(fe::SUCCESS, ret);

  engine_name = fe::VECTOR_CORE_NAME;
  const std::string vectorcore_so_path = current_dir + "plugin/opskernel/fusion_pass/vector_core";
  CreateDir(vectorcore_so_path);
  const std::string vectorfileName = vectorcore_so_path + "libops_fusion_pass_vectorcore.so";
  CreateFileAndFillContent(vectorfileName);
  Configuration::Instance(engine_name).lib_path_ = current_dir;
  ret = fusion_pass_manager.InitFusionPassPlugin(engine_name);
  EXPECT_EQ(fe::SUCCESS, ret);
  system(("rm -rf " + current_dir + "plugin").c_str());
}