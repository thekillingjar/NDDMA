/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "core/ring_queue_impl.h"
#include <gtest/gtest.h>
#include <cstdint>
#include "core/builder/executor_builder.h"
namespace gert {
class RingQueueUT : public testing::Test {};
TEST_F(RingQueueUT, Push_Ok_OneElement) {
  auto q = std::unique_ptr<RingQueue, void (*)(void *)>(CreateRingQueue(1), &free);
  ASSERT_NE(q, nullptr);
  ASSERT_TRUE(IsQueueEmpty(q.get()));

  int64_t a;
  ASSERT_EQ(PushQueue(q.get(), &a), 0);
  ASSERT_FALSE(IsQueueEmpty(q.get()));
}
TEST_F(RingQueueUT, Push_Ok_MultipleElements) {
  auto q = std::unique_ptr<RingQueue, void (*)(void *)>(CreateRingQueue(5), &free);
  ASSERT_NE(q, nullptr);
  ASSERT_TRUE(IsQueueEmpty(q.get()));

  PriorityQueueElementHead head;
  head.priority = 1;
  ASSERT_EQ(PushQueue(q.get(), &head), kStatusSuccess);
  ASSERT_EQ(PushQueue(q.get(), &head), kStatusSuccess);
  ASSERT_EQ(PushQueue(q.get(), &head), kStatusSuccess);
  ASSERT_EQ(PushQueue(q.get(), &head), kStatusSuccess);
  ASSERT_EQ(PushQueue(q.get(), &head), kStatusSuccess);
}
TEST_F(RingQueueUT, Push_Failed_ExceedsCap) {
  auto q = std::unique_ptr<RingQueue, void (*)(void *)>(CreateRingQueue(3), &free);
  ASSERT_NE(q, nullptr);
  ASSERT_TRUE(IsQueueEmpty(q.get()));

  PriorityQueueElementHead head;
  head.priority = 1;
  ASSERT_EQ(PushQueue(q.get(), &head), kStatusSuccess);
  ASSERT_EQ(PushQueue(q.get(), &head), kStatusSuccess);
  ASSERT_EQ(PushQueue(q.get(), &head), kStatusSuccess);
  ASSERT_EQ(PushQueue(q.get(), &head), kStatusQueueFull);
}
TEST_F(RingQueueUT, Pop_Ok_One) {
  auto q = std::unique_ptr<RingQueue, void (*)(void *)>(CreateRingQueue(1), &free);
  ASSERT_NE(q, nullptr);
  ASSERT_TRUE(IsQueueEmpty(q.get()));

  PriorityQueueElementHead head;
  head.priority = 1;
  ASSERT_EQ(PushQueue(q.get(), &head), 0);
  auto ele = PopQueue(q.get());
  ASSERT_EQ(ele, &head);
}
TEST_F(RingQueueUT, Pop_Nullptr_Empty) {
  auto q = std::unique_ptr<RingQueue, void (*)(void *)>(CreateRingQueue(2), &free);
  ASSERT_NE(q, nullptr);
  ASSERT_TRUE(IsQueueEmpty(q.get()));

  auto ele = PopQueue(q.get());
  ASSERT_EQ(ele, nullptr);

  PriorityQueueElementHead head;
  head.priority = 1;
  PushQueue(q.get(), &head);
  PopQueue(q.get());
  ele = PopQueue(q.get());
  ASSERT_EQ(ele, nullptr);
}
TEST_F(RingQueueUT, PushPop_Ok_Multiple) {
  auto q = std::unique_ptr<RingQueue, void (*)(void *)>(CreateRingQueue(10), &free);
  ASSERT_NE(q, nullptr);
  PriorityQueueElementHead heads[10] = {{0}};
  std::vector<void *> push_elements, pop_elements;
  for (auto &head : heads) {
    push_elements.push_back(&head);
    ASSERT_EQ(PushQueue(q.get(), &head), kStatusSuccess);

    pop_elements.push_back(PopQueue(q.get()));
  }

  ASSERT_EQ(push_elements, pop_elements);
  ASSERT_EQ(pop_elements.size(), 10U);
}
TEST_F(RingQueueUT, PushAfterPop_Ok) {
  auto q = std::unique_ptr<RingQueue, void (*)(void *)>(CreateRingQueue(5), &free);
  ASSERT_NE(q, nullptr);
  ASSERT_TRUE(IsQueueEmpty(q.get()));

  int64_t head[10];
  ASSERT_EQ(PushQueue(q.get(), &head[0]), kStatusSuccess);
  ASSERT_EQ(PushQueue(q.get(), &head[1]), kStatusSuccess);
  ASSERT_EQ(PopQueue(q.get()), &head[0]);
  ASSERT_EQ(PushQueue(q.get(), &head[2]), kStatusSuccess);
  ASSERT_EQ(PushQueue(q.get(), &head[3]), kStatusSuccess);

  ASSERT_EQ(PopQueue(q.get()), &head[1]);
  ASSERT_EQ(PopQueue(q.get()), &head[2]);
  ASSERT_EQ(PopQueue(q.get()), &head[3]);

  ASSERT_EQ(PushQueue(q.get(), &head[4]), kStatusSuccess);
  ASSERT_EQ(PopQueue(q.get()), &head[4]);
  ASSERT_EQ(PopQueue(q.get()), nullptr);
}
}  // namespace gert