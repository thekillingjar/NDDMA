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
#include "core/executor/multi_thread_topological/executor/schedule/queue/lock_free/spsc_queue.h"

using namespace gert;

struct SpscQueueUnitTest : public testing::Test {
 private:
  void SetUp() override {
    for (uint32_t i = 0; i < QUEUE_LENGTH; i++) {
      items[i] = i;
    }
  }

 protected:
  static constexpr uint32_t SIZE_LOG2 = 4;
  static constexpr uint32_t QUEUE_LENGTH = 1 << SIZE_LOG2;
  static constexpr uint32_t QUEUE_CAPACITY = (1 << SIZE_LOG2) - 1;

 protected:
  int items[QUEUE_LENGTH];

 protected:
  SpscQueue<int *> queue{SIZE_LOG2};
};

TEST_F(SpscQueueUnitTest, should_be_empty_when_init) {
  ASSERT_TRUE(queue.IsEmpty());
  ASSERT_FALSE(queue.IsFull());
  ASSERT_EQ(0, queue.GetSize());
  ASSERT_EQ(QUEUE_CAPACITY, queue.GetCapacity());
}

TEST_F(SpscQueueUnitTest, should_push_items) {
  ASSERT_TRUE(queue.Push(&items[0]));

  ASSERT_FALSE(queue.IsEmpty());
  ASSERT_FALSE(queue.IsFull());
  ASSERT_EQ(1, queue.GetSize());

  ASSERT_TRUE(queue.Push(&items[1]));

  ASSERT_FALSE(queue.IsEmpty());
  ASSERT_FALSE(queue.IsFull());
  ASSERT_EQ(2, queue.GetSize());
}

TEST_F(SpscQueueUnitTest, should_pop_items) {
  ASSERT_TRUE(queue.Push(&items[0]));

  int *v;
  ASSERT_TRUE(queue.Pop(v));
  ASSERT_TRUE(queue.IsEmpty());
  ASSERT_EQ(0, *v);
  ASSERT_EQ(0, queue.GetSize());

  ASSERT_TRUE(queue.Push(&items[2]));
  ASSERT_TRUE(queue.Push(&items[3]));

  ASSERT_TRUE(queue.Pop(v));
  ASSERT_FALSE(queue.IsEmpty());
  ASSERT_EQ(2, *v);
  ASSERT_EQ(1, queue.GetSize());
}

TEST_F(SpscQueueUnitTest, should_push_items_full) {
  for (uint32_t i = 0; i < QUEUE_CAPACITY + 3; i++) {
    if (!queue.Push(&items[i])) break;
  }

  ASSERT_FALSE(queue.IsEmpty());
  ASSERT_TRUE(queue.IsFull());
  ASSERT_EQ(QUEUE_CAPACITY, queue.GetSize());
}

TEST_F(SpscQueueUnitTest, should_pop_items_empty) {
  for (uint32_t i = 0; i < QUEUE_CAPACITY; i++) {
    if (!queue.Push(&items[i])) break;
  }
  ASSERT_TRUE(queue.IsFull());

  for (uint32_t i = 0; i < QUEUE_CAPACITY + 3; i++) {
    int *v;
    if (!queue.Pop(v)) break;
  }
  ASSERT_TRUE(queue.IsEmpty());
  ASSERT_TRUE(queue.IsEmpty());
  ASSERT_EQ(0, queue.GetSize());
}

TEST_F(SpscQueueUnitTest, should_push_and_pop_in_ring) {
  for (uint32_t i = 0; i < QUEUE_CAPACITY; i++) {
    ASSERT_TRUE(queue.Push(&items[i]));
  }
  ASSERT_TRUE(queue.IsFull());

  for (int i = 0; i < 3; i++) {
    int *v;
    ASSERT_TRUE(queue.Pop(v));
  }
  ASSERT_FALSE(queue.IsEmpty());
  ASSERT_FALSE(queue.IsFull());

  int push_cnt = 0;
  for (uint32_t i = 0; i < QUEUE_CAPACITY; i++) {
    if (!queue.Push(&items[i])) break;
    push_cnt++;
  }
  ASSERT_TRUE(queue.IsFull());
  ASSERT_EQ(3, push_cnt);

  int pop_cnt = 0;
  for (uint32_t i = 0; i < QUEUE_CAPACITY - 1; i++) {
    int *v;
    if (!queue.Pop(v)) break;
    pop_cnt++;
  }
  ASSERT_FALSE(queue.IsEmpty());
  ASSERT_FALSE(queue.IsFull());
  ASSERT_EQ(1, queue.GetSize());
  ASSERT_EQ(QUEUE_CAPACITY - 1, pop_cnt);
}
