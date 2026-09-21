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

#include "framework/runtime/event_allocator.h"
#include "depends/runtime/src/runtime_stub.h"
#include "framework/common/ge_inner_error_codes.h"

namespace gert {
class EventAllocatorUT : public testing::Test {};

TEST_F(EventAllocatorUT, Acquire_Success_EventPoolEmpty) {
  size_t num = 22U;
  EventAllocator ea;
  auto events = ea.AcquireEvents(num);
  ASSERT_NE(events, nullptr);
  ASSERT_EQ(events->GetSize(), num);
  for (size_t i = 0U; i < num; ++i) {
    ASSERT_NE(events->GetData()[i], nullptr);
  }
}
TEST_F(EventAllocatorUT, Acquire_Success_Expand) {
  EventAllocator ea;
  ASSERT_NE(ea.AcquireEvents(11), nullptr);

  size_t num = 22U;
  auto events = ea.AcquireEvents(num);
  ASSERT_NE(events, nullptr);
  ASSERT_EQ(events->GetSize(), num);
  for (size_t i = 0U; i < num; ++i) {
    ASSERT_NE(events->GetData()[i], nullptr);
  }
}
TEST_F(EventAllocatorUT, Acquire_Success_Shrink) {
  EventAllocator ea;
  ASSERT_NE(ea.AcquireEvents(33), nullptr);

  size_t num = 22U;
  auto events = ea.AcquireEvents(num);
  ASSERT_NE(events, nullptr);
  ASSERT_EQ(events->GetSize(), 33);
  for (size_t i = 0U; i < num; ++i) {
    ASSERT_NE(events->GetData()[i], nullptr);
  }
}

TEST_F(EventAllocatorUT, AcquireEvents_Fail_NoEnoughEventResource) {
  uint32_t max_event_num = 0;
  ASSERT_EQ(rtGetAvailEventNum(&max_event_num), RT_ERROR_NONE);
  size_t n_events = max_event_num + 8U;
  EventAllocator ea;
  EXPECT_EQ(ea.AcquireEvents(n_events), nullptr);
}
}  // namespace gert