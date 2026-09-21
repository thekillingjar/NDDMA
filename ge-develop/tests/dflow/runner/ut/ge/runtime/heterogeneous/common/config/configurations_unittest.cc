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
#include <vector>
#include <string>
#include <map>
#include <stdlib.h>

#include <unistd.h>
#include "nlohmann/json.hpp"
#include "framework/common/debug/ge_log.h"
#include "depends/mmpa/src/mmpa_stub.h"
#include "common/env_path.h"

#include "macro_utils/dt_public_scope.h"
#include "common/config/configurations.h"
#include "macro_utils/dt_public_unscope.h"

using namespace std;
namespace ge {
class UtConfigurations : public testing::Test {
 public:
  UtConfigurations() {}
 protected:
  void SetUp() override {}
  void TearDown() override {
    MmpaStub::GetInstance().Reset();
  }

  static void SetConfigEnv(const char *env_name, const std::string &path) {
    std::string data_path = PathUtils::Join({EnvPath().GetAirBasePath(), "tests/dflow/runner/ut/ge/runtime/data"});
    std::string config_path = PathUtils::Join({data_path, path});
    setenv(env_name, config_path.c_str(), 1);
  }
};

// host
TEST_F(UtConfigurations, parse_server_info_success) {
  class MockMmpaRealPath : public ge::MmpaStubApiGe {
   public:
    int32_t RealPath(const CHAR *path, CHAR *realPath, INT32 realPathLen) override {
      memcpy_s(realPath, realPathLen, path, strlen(path));
      return 0;
    }
  };
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpaRealPath>());
  SetConfigEnv("RESOURCE_CONFIG_PATH", "valid/server/numa_config.json");
  EXPECT_EQ(Configurations().InitInformation(), SUCCESS);
  unsetenv("RESOURCE_CONFIG_PATH");
}

TEST_F(UtConfigurations, GetWorkingDir_host_realpath_failed) {
  class MockMmpaRealPath : public ge::MmpaStubApiGe {
   public:
    int32_t RealPath(const CHAR *path, CHAR *realPath, INT32 realPathLen) override {
      realPath[0] = '\0';
      return 0;
    }
  };
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpaRealPath>());
  Configurations config;
  std::string config_dir;
  EXPECT_NE(config.GetWorkingDir(config_dir), SUCCESS);
}

TEST_F(UtConfigurations, GetWorkingDir_host_path_failed) {
  class MockMmpaRealPath : public ge::MmpaStubApiGe {
   public:
    int32_t RealPath(const CHAR *path, CHAR *realPath, INT32 realPathLen) override {
      memcpy_s(realPath, realPathLen, "../", strlen("../"));
      return 0;
    }
  };
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpaRealPath>());
  Configurations config;
  std::string config_dir;
  EXPECT_NE(config.GetWorkingDir(config_dir), SUCCESS);
}

TEST_F(UtConfigurations, GetWorkingDir_Host_AscendWorkPath_success) {
  ge::char_t current_path[MMPA_MAX_PATH] = {'\0'};
  getcwd(current_path, MMPA_MAX_PATH);
  mmSetEnv("ASCEND_WORK_PATH", current_path, 1);
  Configurations config;
  std::string config_dir;
  EXPECT_EQ(config.GetWorkingDir(config_dir), SUCCESS);

  EXPECT_EQ(config_dir, current_path);
  unsetenv("ASCEND_WORK_PATH");
}
}

