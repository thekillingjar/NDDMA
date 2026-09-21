/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "core/priority_queue_impl.h"
#include <gtest/gtest.h>
#include "core/builder/executor_builder.h"
namespace gert {
class PriorityQueueUT : public testing::Test {};
TEST_F(PriorityQueueUT, Push_Ok_OneElement) {
  auto q = std::unique_ptr<PriorityQueue, void (*)(void *)>(CreatePriorityQueue(1), &free);
  ASSERT_NE(q, nullptr);
  ASSERT_EQ(q->size, 0U);

  PriorityQueueElementHead head;
  head.priority = 1;
  ASSERT_EQ(PushQueue(q.get(), &head), 0);
  ASSERT_EQ(q->size, 1U);
}
TEST_F(PriorityQueueUT, Push_Ok_MultipleElements) {
  auto q = std::unique_ptr<PriorityQueue, void (*)(void *)>(CreatePriorityQueue(5), &free);
  ASSERT_NE(q, nullptr);
  ASSERT_EQ(q->size, 0U);

  PriorityQueueElementHead head;
  head.priority = 1;
  ASSERT_EQ(PushQueue(q.get(), &head), kStatusSuccess);
  ASSERT_EQ(PushQueue(q.get(), &head), kStatusSuccess);
  ASSERT_EQ(q->size, 2U);
  ASSERT_EQ(PushQueue(q.get(), &head), kStatusSuccess);
  ASSERT_EQ(PushQueue(q.get(), &head), kStatusSuccess);
  ASSERT_EQ(PushQueue(q.get(), &head), kStatusSuccess);
  ASSERT_EQ(q->size, 5U);
}
TEST_F(PriorityQueueUT, Push_Failed_ExceedsCap) {
  auto q = std::unique_ptr<PriorityQueue, void (*)(void *)>(CreatePriorityQueue(3), &free);
  ASSERT_NE(q, nullptr);
  ASSERT_EQ(q->size, 0U);

  PriorityQueueElementHead head;
  head.priority = 1;
  ASSERT_EQ(PushQueue(q.get(), &head), kStatusSuccess);
  ASSERT_EQ(PushQueue(q.get(), &head), kStatusSuccess);
  ASSERT_EQ(PushQueue(q.get(), &head), kStatusSuccess);
  ASSERT_EQ(q->size, 3U);
  ASSERT_EQ(PushQueue(q.get(), &head), kStatusQueueFull);
}
TEST_F(PriorityQueueUT, Pop_Ok_One) {
  auto q = std::unique_ptr<PriorityQueue, void (*)(void *)>(CreatePriorityQueue(1), &free);
  ASSERT_NE(q, nullptr);
  ASSERT_EQ(q->size, 0U);

  PriorityQueueElementHead head;
  head.priority = 1;
  ASSERT_EQ(PushQueue(q.get(), &head), 0);
  auto ele = PopQueue(q.get());
  ASSERT_NE(ele, nullptr);
  ASSERT_EQ(ele, &head);
}
TEST_F(PriorityQueueUT, Pop_Nullptr_Empty) {
  auto q = std::unique_ptr<PriorityQueue, void (*)(void *)>(CreatePriorityQueue(2), &free);
  ASSERT_NE(q, nullptr);
  ASSERT_EQ(q->size, 0U);

  auto ele = PopQueue(q.get());
  ASSERT_EQ(ele, nullptr);

  PriorityQueueElementHead head;
  head.priority = 1;
  PushQueue(q.get(), &head);
  PopQueue(q.get());
  ele = PopQueue(q.get());
  ASSERT_EQ(ele, nullptr);
}
TEST_F(PriorityQueueUT, PushPop_Ok_Multiple) {
  auto q = std::unique_ptr<PriorityQueue, void (*)(void *)>(CreatePriorityQueue(10), &free);
  ASSERT_NE(q, nullptr);
  PriorityQueueElementHead heads[10] = {{0}};
  std::set<PriorityQueueElementHead *> push_elements, pop_elements;
  for (auto &head : heads) {
    push_elements.insert(&head);
    ASSERT_EQ(PushQueue(q.get(), &head), kStatusSuccess);

    pop_elements.insert(PopQueue(q.get()));
  }

  ASSERT_EQ(push_elements, pop_elements);
  ASSERT_EQ(pop_elements.size(), 10U);
}
TEST_F(PriorityQueueUT, PushPop_PriorityCorrect_Multiple1) {
  auto q = std::unique_ptr<PriorityQueue, void (*)(void *)>(CreatePriorityQueue(20), &free);
  ASSERT_NE(q, nullptr);
  PriorityQueueElementHead heads[20];
  for (int64_t i = 0; i < 20; ++i) {
    heads[i].priority = i;
  }
  for (auto &head : heads) {
    PushQueue(q.get(), &head);
  }

  for (int64_t i = 0; i < 20; ++i) {
    auto ele = PopQueue(q.get());
    ASSERT_EQ(ele->priority, i);
  }
}
TEST_F(PriorityQueueUT, PushPop_PriorityCorrect_Multiple2) {
  auto q = std::unique_ptr<PriorityQueue, void (*)(void *)>(CreatePriorityQueue(20), &free);
  ASSERT_NE(q, nullptr);
  PriorityQueueElementHead heads[20];
  for (int64_t i = 0; i < 20; ++i) {
    heads[i].priority = i;
  }
  for (size_t i = 20U; i > 0U; --i) {
    PushQueue(q.get(), &(heads[i - 1]));
  }

  for (int64_t i = 0; i < 20; ++i) {
    auto ele = PopQueue(q.get());
    ASSERT_EQ(ele->priority, i);
  }
}
TEST_F(PriorityQueueUT, PushPop_PriorityCorrect_Multiple3) {
  auto q = std::unique_ptr<PriorityQueue, void (*)(void *)>(CreatePriorityQueue(20), &free);
  ASSERT_NE(q, nullptr);
  PriorityQueueElementHead heads[20];
  for (int64_t i = 0; i < 20; ++i) {
    heads[i].priority = i * 100;
  }
  for (size_t i = 0U; i < 20U; i += 2U) {
    PushQueue(q.get(), &(heads[i]));
  }
  for (size_t i = 1U; i < 20U; i += 2U) {
    PushQueue(q.get(), &(heads[i]));
  }

  for (int64_t i = 0; i < 20; ++i) {
    auto ele = PopQueue(q.get());
    ASSERT_EQ(ele->priority, i * 100);
  }
}
TEST_F(PriorityQueueUT, PushAfterPop_Ok) {
  auto q = std::unique_ptr<PriorityQueue, void (*)(void *)>(CreatePriorityQueue(5), &free);
  ASSERT_NE(q, nullptr);
  ASSERT_EQ(q->size, 0U);

  PriorityQueueElementHead head;
  head.priority = 1;
  ASSERT_EQ(PushQueue(q.get(), &head), kStatusSuccess);
  ASSERT_EQ(PushQueue(q.get(), &head), kStatusSuccess);
  ASSERT_EQ(q->size, 2U);
  ASSERT_EQ(PopQueue(q.get()), &head);
  ASSERT_EQ(q->size, 1U);
  ASSERT_EQ(PushQueue(q.get(), &head), kStatusSuccess);
  ASSERT_EQ(PushQueue(q.get(), &head), kStatusSuccess);
  ASSERT_EQ(PushQueue(q.get(), &head), kStatusSuccess);
  ASSERT_EQ(q->size, 4U);
}
}  // namespace gert