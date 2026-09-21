/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "runtime/subscriber/executor_subscriber_guarder.h"
#include <gtest/gtest.h>
namespace gert {
namespace {
void TestSubscriberFunc(int32_t type, void *arg, ExecutorEvent event, const void *node, KernelStatus result) {}
}
class ExecutorSubscriberGuarderUT : public testing::Test {};
TEST_F(ExecutorSubscriberGuarderUT, NoLeaks) {
  EXPECT_NO_THROW(ExecutorSubscriberGuarder g1(TestSubscriberFunc, malloc(10), free));
}
TEST_F(ExecutorSubscriberGuarderUT, GetMembersOk) {
  auto arg = malloc(10);
  ExecutorSubscriberGuarder g1(TestSubscriberFunc, arg, free);
  EXPECT_TRUE(g1.GetSubscriber().callback == TestSubscriberFunc);
  EXPECT_EQ(g1.GetSubscriber().arg, arg);
}

TEST_F(ExecutorSubscriberGuarderUT, MoveConstructOk) {
  EXPECT_NO_THROW(
    ExecutorSubscriberGuarder g1(TestSubscriberFunc, malloc(10), free);
    ExecutorSubscriberGuarder g2(std::move(g1));
  );
}
TEST_F(ExecutorSubscriberGuarderUT, MoveAssignmentOk) {
  EXPECT_NO_THROW(
    ExecutorSubscriberGuarder g1(TestSubscriberFunc, malloc(10), free);
    ExecutorSubscriberGuarder g2(nullptr, nullptr, nullptr);
    g2 = std::move(g1);
  );
}
}  // namespace gert
