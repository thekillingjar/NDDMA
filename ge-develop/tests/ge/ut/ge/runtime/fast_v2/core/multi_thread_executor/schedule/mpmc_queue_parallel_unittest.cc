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
#include "core/executor/multi_thread_topological/executor/schedule/queue/lock_free/mpmc_queue.h"

using namespace gert;

struct MpmcQueueParallelUnitTest : public testing::Test {
 private:
  void SetUp() override {
    for (uint32_t i = 0; i < QUEUE_LENGTH; i++) {
      items1[i] = i;
      items2[i] = i;
      values1[i] = nullptr;
      values2[i] = nullptr;
    }
  }

 public:
  void producer(int *items) {
    for (uint32_t i = 0; i < PRODUCE_MAX; i++) {
      queue.Push(&items[i & QUEUE_CAPACITY]);
    }
  }

  void consumer(int **values) {
    for (uint32_t i = 0; i < CONSUME_MAX; i++) {
      queue.Pop(values[i & QUEUE_CAPACITY]);
    }
  }

 protected:
  static constexpr uint32_t SIZE_LOG2 = 4;
  static constexpr uint32_t QUEUE_LENGTH = 1 << SIZE_LOG2;
  static constexpr uint32_t QUEUE_CAPACITY = (1 << SIZE_LOG2) - 1;

  static constexpr uint32_t PRODUCER_NUM = 1;
  static constexpr uint32_t CONSUMER_NUM = 1;

  static constexpr uint32_t PRODUCE_MAX = 10000;
  static constexpr uint32_t CONSUME_MAX = 10000;

 protected:
  int items1[QUEUE_LENGTH];
  int items2[QUEUE_LENGTH];
  int *values1[QUEUE_LENGTH];
  int *values2[QUEUE_LENGTH];

 protected:
  MpmcQueue<int *> queue{SIZE_LOG2};
};

TEST_F(MpmcQueueParallelUnitTest, should_parallel_exec_by_one_producer_and_one_consumer) {
  std::thread pt(std::mem_fn(&MpmcQueueParallelUnitTest::producer), this, items1);
  std::thread ct(std::mem_fn(&MpmcQueueParallelUnitTest::consumer), this, values1);
  pt.join();
  ct.join();
  ASSERT_LE(queue.GetSize(), QUEUE_LENGTH);
}

TEST_F(MpmcQueueParallelUnitTest, should_parallel_exec_by_one_comsumer_and_one_producer) {
  std::thread ct(std::mem_fn(&MpmcQueueParallelUnitTest::consumer), this, values1);
  std::thread pt(std::mem_fn(&MpmcQueueParallelUnitTest::producer), this, items1);
  pt.join();
  ct.join();
  ASSERT_LE(queue.GetSize(), QUEUE_LENGTH);
}

TEST_F(MpmcQueueParallelUnitTest, should_parallel_exec_by_one_producer_and_two_consumer) {
  std::thread pt(std::mem_fn(&MpmcQueueParallelUnitTest::producer), this, items1);
  std::thread ct1(std::mem_fn(&MpmcQueueParallelUnitTest::consumer), this, values1);
  std::thread ct2(std::mem_fn(&MpmcQueueParallelUnitTest::consumer), this, values2);
  pt.join();
  ct1.join();
  ct2.join();
  ASSERT_LE(queue.GetSize(), QUEUE_LENGTH);
}

TEST_F(MpmcQueueParallelUnitTest, should_parallel_exec_by_two_producer_and_two_consumer) {
  std::thread pt1(std::mem_fn(&MpmcQueueParallelUnitTest::producer), this, items1);
  std::thread pt2(std::mem_fn(&MpmcQueueParallelUnitTest::producer), this, items2);
  std::thread ct1(std::mem_fn(&MpmcQueueParallelUnitTest::consumer), this, values1);
  std::thread ct2(std::mem_fn(&MpmcQueueParallelUnitTest::consumer), this, values2);
  pt1.join();
  pt2.join();
  ct1.join();
  ct2.join();
  ASSERT_LE(queue.GetSize(), QUEUE_LENGTH);
}
