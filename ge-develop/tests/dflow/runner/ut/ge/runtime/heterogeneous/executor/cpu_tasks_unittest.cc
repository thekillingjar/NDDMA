/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <thread>
#include <gtest/gtest.h>

#include "macro_utils/dt_public_scope.h"
#include "executor/cpu_tasks.h"
#include "macro_utils/dt_public_unscope.h"
#include "depends/runtime/src/runtime_stub.h"

using namespace std;

namespace ge {
class CpuTasksTest : public testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};
TEST_F(CpuTasksTest, testExecuteModelEschedPriorityTask) {
  int32_t esched_process_priority = 0;
  int32_t esched_event_priority = 0;
  const auto ret = CpuTasks::ExecuteModelEschedPriorityTask(esched_process_priority, esched_event_priority);
  EXPECT_EQ(ret, SUCCESS);
}
TEST_F(CpuTasksTest, testExecuteModelClearTask) {
  int32_t clear_type = 1;
  const std::vector<uint32_t> davinci_model_runtime_ids = {0, 1};
  auto ret = CpuTasks::ExecuteModelClearTask(clear_type, davinci_model_runtime_ids);
  EXPECT_EQ(ret, SUCCESS);
  clear_type = 2;
  ret = CpuTasks::ExecuteModelClearTask(clear_type, davinci_model_runtime_ids);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(CpuTasksTest, testExceptionNotify) {
  const std::vector<uint32_t> davinci_model_runtime_ids = {0, 1};
  uint32_t type = 0;
  uint64_t trans_id = 1;
  auto ret = CpuTasks::ExceptionNotify(davinci_model_runtime_ids, type, trans_id);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(CpuTasksTest, testCheckExceptionNotify) {
  auto ret = CpuTasks::CheckSupportExceptionNotify();
  EXPECT_EQ(ret, SUCCESS);
  g_runtime_stub_mock = "rtCpuKernelLaunchWithFlag";
  ret = CpuTasks::CheckSupportExceptionNotify();
  EXPECT_NE(ret, SUCCESS);
  g_runtime_stub_mock.clear();
}

TEST_F(CpuTasksTest, testExecuteModeCheckSupported) {
  bool is_supported = false;
  auto ret = CpuTasks::ExecuteCheckSupported("test1", is_supported);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(is_supported, true);
  is_supported = false;
  ret = CpuTasks::ExecuteCheckSupported("gatherDequeue", is_supported);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(is_supported, true);
  g_runtime_stub_mock = "rtCpuKernelLaunchWithFlag";
  is_supported = false;
  ret = CpuTasks::ExecuteCheckSupported("gatherDequeue", is_supported);
  EXPECT_TRUE(is_supported == false);
}
}  // namespace ge
