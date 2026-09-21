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
#include <thread>
#include <atomic>
#include "core/executor/multi_thread_topological/executor/schedule/task/exec_task.h"
#include "core/executor/multi_thread_topological/executor/schedule/queue/cache_utility.h"

using namespace gert;

struct BasicConfigUnitTest : public testing::Test {
 protected:
};

TEST_F(BasicConfigUnitTest, should_lock_free_for_basic_type_and_pointer) {
  ASSERT_TRUE(std::atomic<size_t>{}.is_lock_free());
  ASSERT_TRUE(std::atomic<ExecTask *>{}.is_lock_free());
  ASSERT_TRUE(std::atomic<uint64_t>{}.is_lock_free());
}

TEST_F(BasicConfigUnitTest, should_get_hardward_cache_line_size) {
  ASSERT_TRUE(kHardwareConstructiveInterferenceSize >= 64);
  ASSERT_TRUE(kHardwareDestructiveInterferenceSize >= 64);
}

TEST_F(BasicConfigUnitTest, should_get_hardward_cpu_number) {
  printf("CPU CORE THREAD NUMER = %d\n", std::thread::hardware_concurrency());
  ASSERT_TRUE(std::thread::hardware_concurrency() >= 2);
}
