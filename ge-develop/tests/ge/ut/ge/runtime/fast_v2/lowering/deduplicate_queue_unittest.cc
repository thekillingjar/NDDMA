/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "lowering/pass/utils/deduplicate_queue.h"
#include <gtest/gtest.h>
#include "graph/node.h"
#include "faker/node_faker.h"
namespace gert {
class DeduplicateQueueUT : public testing::Test {};
TEST_F(DeduplicateQueueUT, PushPop_SameValue_Int) {
  bg::DeduplicateQueue<int32_t> que;
  ASSERT_TRUE(que.empty());
  que.push(1000);
  ASSERT_EQ(que.pop(), 1000);
}
TEST_F(DeduplicateQueueUT, IsEmpty_Ok) {
  bg::DeduplicateQueue<int32_t> que;
  ASSERT_TRUE(que.empty());
  que.push(1000);
  ASSERT_FALSE(que.empty());
}
TEST_F(DeduplicateQueueUT, Pop_Fifo_Int) {
  bg::DeduplicateQueue<int32_t> que;
  ASSERT_TRUE(que.empty());
  que.push(1000);
  que.push(1001);
  que.push(1002);
  ASSERT_EQ(que.pop(), 1000);
  ASSERT_EQ(que.pop(), 1001);
  ASSERT_EQ(que.pop(), 1002);
}
TEST_F(DeduplicateQueueUT, PushDuplicateValue_NoEffect_Int) {
  bg::DeduplicateQueue<int32_t> que;
  ASSERT_TRUE(que.empty());
  que.push(1000);
  que.push(1001);
  que.push(1000);
  que.push(1000);
  ASSERT_FALSE(que.empty());
  ASSERT_EQ(que.pop(), 1000);
  ASSERT_EQ(que.pop(), 1001);
  ASSERT_TRUE(que.empty());
}
TEST_F(DeduplicateQueueUT, PushDuplicateValue_NoEffect_NodePtr) {
  auto node1 = ComputeNodeFaker().IoNum(1,1).Build();
  auto node2 = ComputeNodeFaker().IoNum(1,1).Build();
  bg::DeduplicateQueue<ge::NodePtr> que;
  ASSERT_TRUE(que.empty());
  que.push(node1);
  que.push(node2);
  que.push(node1);
  ASSERT_FALSE(que.empty());
  ASSERT_EQ(que.pop(), node1);
  ASSERT_EQ(que.pop(), node2);
  ASSERT_TRUE(que.empty());
}
}  // namespace gert