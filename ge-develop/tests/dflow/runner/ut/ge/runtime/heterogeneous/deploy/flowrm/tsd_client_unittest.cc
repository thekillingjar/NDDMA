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
#include "deploy/flowrm/tsd_client.h"
#include "depends/mmpa/src/mmpa_stub.h"

namespace ge {
namespace {
class MockMmpa : public MmpaStubApiGe {
 public:
  void *DlSym(void *handle, const char *func_name) override {
    if (std::string(func_name) == "TsdFileLoad") {
      return (void *) &TsdFileLoad;
    } else if (std::string(func_name) == "TsdFileUnLoad") {
      return (void *) &TsdFileUnLoad;
    } else if (std::string(func_name) == "TsdGetProcListStatus") {
      return (void *) &TsdGetProcListStatus;
    } else if (std::string(func_name) == "TsdProcessOpen") {
      return (void *) &TsdProcessOpen;
    } else if (std::string(func_name) == "ProcessCloseSubProcList") {
      return (void *) &ProcessCloseSubProcList;
    } else if (std::string(func_name) == "TsdCapabilityGet") {
      return (void *) &TsdCapabilityGet;
    } else if (std::string(func_name) == "TsdInitFlowGw") {
      return (void *) &TsdInitFlowGw;
    }
    std::cout << "func name:" << func_name << " not stub\n";
    return (void *) 0xFFFFFFFF;
  }

  void *DlOpen(const char *fileName, int32_t mode) override {
    return (void *) 0xFFFFFFFF;
  }
  int32_t DlClose(void *handle) override {
    return 0L;
  }
};
}

class UtTsdClient : public testing::Test {
 public:
  UtTsdClient() {}
 protected:
  void SetUp() override {
    MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
    TsdClient::GetInstance().Initialize();
  }
  void TearDown() override {
    TsdClient::GetInstance().Finalize();
    MmpaStub::GetInstance().Reset();
  }
};

TEST_F(UtTsdClient, run_GetProcStatus) {
  ProcStatus status = ProcStatus::INVALID;
  auto ret = TsdClient::GetInstance().GetProcStatus(0, 111, status, "NPU");
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(status, ProcStatus::NORMAL);
}

TEST_F(UtTsdClient, run_GetLoadFileCount) {
  auto count = TsdClient::GetLoadFileCount();
  EXPECT_EQ(count, 0UL);
  count = TsdClient::GetLoadFileCount();
  EXPECT_EQ(count, 1UL);
  count = TsdClient::GetLoadFileCount();
  EXPECT_EQ(count, 2UL);
}

TEST_F(UtTsdClient, run_StartFlowGw) {
  pid_t pid = 0;
  auto ret = TsdClient::GetInstance().StartFlowGw(0, "GRP", pid);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtTsdClient, run_ForkSubprocess) {
  pid_t pid = 0;
  SubprocessManager::SubprocessConfig config = {};
  config.envs["env1"] = "val1";
  config.envs["env2"] = "val2";
  config.kv_args["key1"] = "v1";
  config.kv_args["key2"] = "v2";
  config.process_type = "NPU";
  config.args.emplace_back("arg1");
  config.args.emplace_back("arg2");
  auto ret = TsdClient::GetInstance().ForkSubprocess(0, config, pid);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtTsdClient, run_ForkSubprocessWithPath) {
  pid_t pid = 0;
  SubprocessManager::SubprocessConfig config = {};
  config.envs["env1"] = "val1";
  config.envs["env2"] = "val2";
  config.kv_args["key1"] = "v1";
  config.kv_args["key2"] = "v2";
  config.process_type = "UDF";
  config.args.emplace_back("arg1");
  config.args.emplace_back("arg2");
  auto ret = TsdClient::GetInstance().ForkSubprocess(0, config, "/a/b/c", pid);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtTsdClient, run_ShutdownSubprocess) {
  pid_t pid = 0;
  auto ret = TsdClient::GetInstance().ShutdownSubprocess(0, pid, "NPU");
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtTsdClient, run_LoadFile) {
  auto ret = TsdClient::GetInstance().LoadFile(0, "/a/b/c", "s.om");
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtTsdClient, run_UnloadFile) {
  auto ret = TsdClient::GetInstance().UnloadFile(0, "/a/b/c/s.om");
  EXPECT_EQ(ret, SUCCESS);
}
}
