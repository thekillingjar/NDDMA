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
#include <memory>

#include "macro_utils/dt_public_scope.h"
#include "common/config/configurations.h"
#include "deploy/heterogeneous_execution_runtime.h"
#include "macro_utils/dt_public_unscope.h"

#include "dflow/base/exec_runtime/execution_runtime.h"
#include "depends/mmpa/src/mmpa_stub.h"
#include "common/env_path.h"

using namespace std;
using namespace ::testing;

namespace ge {
class HelperExecutionRuntimeTest : public testing::Test {
};

namespace {
class MockMmpa : public MmpaStubApiGe {
 public:
  int32_t RealPath(const CHAR *path, CHAR *realPath, INT32 realPathLen) override {
    (void)strncpy_s(realPath, realPathLen, path, strlen(path));
    return EN_OK;
  }
};
}  // namespace

TEST_F(HelperExecutionRuntimeTest, InitializeAndFinalize) {
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  Configurations::GetInstance().information_ = DeployerConfig{};
  std::string real_path = PathUtils::Join(
      {EnvPath().GetAirBasePath(), "tests/dflow/runner/ut/ge/runtime/data/valid/server/numa_config2.json"});
  setenv("RESOURCE_CONFIG_PATH", real_path.c_str(), 1);
  std::map<std::string, std::string> options;
  EXPECT_EQ(InitializeHeterogeneousRuntime(options), SUCCESS);
  auto const *execution_runtime = ExecutionRuntime::GetInstance();
  EXPECT_NE(execution_runtime, nullptr);
  (void)execution_runtime->GetCompileHostResourceType();
  (void)execution_runtime->GetCompileDeviceInfo();
  ExecutionRuntime::FinalizeExecutionRuntime();
  MmpaStub::GetInstance().Reset();
  unsetenv("RESOURCE_CONFIG_PATH");
}
}  // namespace ge
