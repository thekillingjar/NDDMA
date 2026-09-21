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
#include <memory>
#include "depends/mmpa/src/mmpa_stub.h"

#include "macro_utils/dt_public_scope.h"
#include "dflow/base/exec_runtime/execution_runtime.h"
#include "exec_runtime/execution_runtime_utils.h"
#include "macro_utils/dt_public_unscope.h"

#include "graph/ge_local_context.h"

namespace ge {

namespace {
void *mock_handle = nullptr;
void *mock_method = nullptr;

Status InitializeHeterogeneousRuntime(const std::map<std::string, std::string> &options) {
  return SUCCESS;
}

class MockMmpa : public MmpaStubApiGe {
 public:
  void *DlOpen(const char *file_name, int32_t mode) override {
    return mock_handle;
  }
  void *DlSym(void *handle, const char *func_name) override {
    return mock_method;
  }

  int32_t DlClose(void *handle) override {
    return 0;
  }
};
}  // namespace

class ExecutionRuntimeTest : public testing::Test {
 protected:
  void SetUp() {
    MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  }
  void TearDown() {
    ExecutionRuntime::FinalizeExecutionRuntime();
    MmpaStub::GetInstance().Reset();
    mock_handle = nullptr;
    mock_method = nullptr;
  }
};

TEST_F(ExecutionRuntimeTest, TestLoadHeterogeneousLib) {
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  ASSERT_EQ(ExecutionRuntime::LoadHeterogeneousLib(), FAILED); // error load so
  mock_handle = (void *)0xffffffff;
  ASSERT_EQ(ExecutionRuntime::LoadHeterogeneousLib(), SUCCESS);
}

TEST_F(ExecutionRuntimeTest, TestSetupHeterogeneousRuntime) {
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  const std::map<std::string, std::string> options;
  ASSERT_EQ(ExecutionRuntime::SetupHeterogeneousRuntime(options), FAILED); // error find init func
  ExecutionRuntime::handle_ = (void *)0xffffffff;
  mock_method = (void *)&InitializeHeterogeneousRuntime;
  ASSERT_EQ(ExecutionRuntime::SetupHeterogeneousRuntime(options), SUCCESS);
}

TEST_F(ExecutionRuntimeTest, TestInitAndFinalize) {
  ASSERT_FALSE(ExecutionRuntimeUtils::IsHeterogeneous());
  ExecutionRuntime::FinalizeExecutionRuntime();  // instance not set
  ASSERT_TRUE(ExecutionRuntime::GetInstance() == nullptr);
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());

  mock_handle = (void *)0xffffffff;
  mock_method = (void *)&InitializeHeterogeneousRuntime;
  const std::map<std::string, std::string> options;
  ASSERT_EQ(ExecutionRuntime::InitHeterogeneousRuntime(options), SUCCESS); // error load so
  ASSERT_TRUE(ExecutionRuntime::handle_ != nullptr);
  ExecutionRuntime::FinalizeExecutionRuntime();  // instance not set

  ASSERT_TRUE(ExecutionRuntime::GetInstance() == nullptr);
  ASSERT_TRUE(ExecutionRuntime::handle_ == nullptr);
}
} // namespace ge
