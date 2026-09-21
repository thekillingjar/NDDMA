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
#include <gmock/gmock.h>
#include <vector>

#include "graph/opsproto_manager.h"
#include "depends/mmpa/src/mmpa_stub.h"
#include "graph_metadef/common/plugin/plugin_manager.h"

using namespace std;
using namespace testing;

VOID *mmDlopen(const CHAR *fileName, INT32 mode) {
  return (void *) 0xffffffff;
}

namespace ge {
class OpsprotoManagerUt : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(OpsprotoManagerUt, Instance_Initialize_Finalize) {
  OpsProtoManager *opspm = nullptr;
  opspm = OpsProtoManager::Instance();
  EXPECT_NE(opspm, nullptr);

  const std::map<std::string, std::string> options = {
    {"ge.mockLibPath", "./gtest_build-prefix/src/gtest_build-build/googlemock/"},
    {"ge.opsProtoLibPath", "./protobuf_build-prefix/src/protobuf_build-build/"}};

  auto ret = opspm->Initialize(options);
  EXPECT_TRUE(ret);

  opspm->Finalize();
  EXPECT_EQ(opspm->handles_.size(), 0);
  EXPECT_EQ(opspm->is_init_, false);
}

TEST_F(OpsprotoManagerUt, LoadBuiltinOpsPluginSo) {
  OpsProtoManager *opspm = OpsProtoManager::Instance();
  opspm->LoadBuiltinOpsPluginSo("");
  opspm->LoadBuiltinOpsPluginSo("./protobuf_build-prefix/src/protobuf_build-build/");

  EXPECT_EQ(opspm->handles_.size(), 0);
}

TEST_F(OpsprotoManagerUt, LoadBuiltinOpsPluginSo_Exclude_rt) {
  std::string path = __FILE__;
  PluginManager manager;
  std::string host_env_os;
  std::string host_env_cpu;
  manager.GetCurEnvPackageOsAndCpuType(host_env_os, host_env_cpu);
  path = path.substr(0, path.rfind("/") + 1) + "opp_test/";
  system(("mkdir -p " + path + "/lib/" + host_env_os + "/" + host_env_cpu).c_str());
  system(("touch " + path + "/lib/"+ host_env_os + "/" + host_env_cpu + "/libopsproto.so").c_str());
  system(("touch " + path + "/lib/"+ host_env_os + "/" + host_env_cpu + "/libopsproto_rt2.0.so").c_str());
  system(("touch " + path + "/lib/"+ host_env_os + "/" + host_env_cpu + "/libopsproto_rt.so").c_str());

  OpsProtoManager::Instance()->handles_.clear();
  OpsProtoManager::Instance()->LoadBuiltinOpsPluginSo(path);
  ASSERT_EQ(OpsProtoManager::Instance()->handles_.size(), 1);
  system(("rm -rf " + path).c_str());
}

TEST_F(OpsprotoManagerUt, LoadOpsCustomSo_Exclude_rt) {
  std::string path = __FILE__;
  path = "";
  OpsProtoManager::Instance()->handles_.clear();
  OpsProtoManager::Instance()->LoadOpsProtoPluginSo(path);
  ASSERT_EQ(OpsProtoManager::Instance()->handles_.size(), 0);
  system(("rm -rf " + path).c_str());
}

TEST_F(OpsprotoManagerUt, LoadOpsCustomSo_Path_invalid) {
  std::string path = __FILE__;
  PluginManager manager;
  std::string host_env_os;
  std::string host_env_cpu;
  manager.GetCurEnvPackageOsAndCpuType(host_env_os, host_env_cpu);
  path = path.substr(0, path.rfind("/") + 1) + "opp_test/";
  system(("mkdir -p " + path + "/lib/" + host_env_os + "/" + host_env_cpu).c_str());
  system(("touch " + path + "/lib/"+ host_env_os + "/" + host_env_cpu + "/libopsproto.so").c_str());
  path = path + + "/lib/"+ host_env_os + "/" + host_env_cpu + "/libopsproto.so";
  OpsProtoManager::Instance()->handles_.clear();
  OpsProtoManager::Instance()->LoadOpsProtoPluginSo(path);
  ASSERT_EQ(OpsProtoManager::Instance()->handles_.size(), 1);
  system(("rm -rf " + path).c_str());
}
}  // namespace ge
