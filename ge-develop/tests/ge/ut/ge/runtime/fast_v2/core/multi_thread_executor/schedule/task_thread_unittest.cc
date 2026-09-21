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
#include <atomic>
#include "core/executor/multi_thread_topological/executor/schedule/worker/task_thread.h"

using namespace gert;

struct TaskThreadUnitTest : public testing::Test {
 protected:
  std::atomic<uint32_t> counter{0};
};

TEST_F(TaskThreadUnitTest, should_init_and_join_single_thread) {
  constexpr size_t max = 100;

  TaskThread thread("count_thread", TaskThreadMode::MODERATE);
  thread.Start([this, max]() {
    for (size_t i = 0; i < max; i++) {
      counter.fetch_add(1, std::memory_order_relaxed);
    }
  });

  thread.Stop();

  ASSERT_EQ(max, counter.load(std::memory_order_relaxed));
}

TEST_F(TaskThreadUnitTest, should_init_and_join_multiple_thread) {
  constexpr size_t max = 100;

  TaskThread thread1("count_thread_1", TaskThreadMode::URGENT);
  thread1.Start([this, max]() {
    for (size_t i = 0; i < max; i++) {
      counter.fetch_add(1, std::memory_order_relaxed);
    }
  });

  TaskThread thread2("count_thread_2", TaskThreadMode::URGENT);
  thread2.Start([this, max]() {
    for (size_t i = 0; i < max; i++) {
      counter.fetch_add(1, std::memory_order_relaxed);
    }
  });

  thread1.Stop();
  thread2.Stop();

  ASSERT_EQ(max * 2, counter.load(std::memory_order_relaxed));
}

TEST_F(TaskThreadUnitTest, should_yield_and_sleep_in_multiple_thread) {
  constexpr size_t max = 100;

  TaskThread thread1("count_thread_1", TaskThreadMode::MODERATE);
  thread1.Start([this, max, &thread1]() {
    for (size_t i = 0; i < max; i++) {
      counter.fetch_add(1, std::memory_order_relaxed);
      std::this_thread::yield();
    }
  });

  TaskThread thread2("count_thread_2", TaskThreadMode::MODERATE);
  thread2.Start([this, max, &thread2]() {
    for (size_t i = 0; i < max; i++) {
      counter.fetch_add(1, std::memory_order_relaxed);
      std::this_thread::yield();
    }
  });

  thread1.Stop();
  thread2.Stop();

  ASSERT_EQ(max * 2, counter.load(std::memory_order_relaxed));
}
