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

#include "framework/runtime/notify_allocator.h"
#include "depends/runtime/src/runtime_stub.h"
#include "framework/common/ge_inner_error_codes.h"

namespace gert {
class NotifyAllocatorUT : public testing::Test {};

TEST_F(NotifyAllocatorUT, Acquire_Success_NotifyPoolEmpty) {
  size_t num = 22U;
  NotifyAllocator na;
  auto notifies = na.AcquireNotifies(0, num);
  ASSERT_NE(notifies, nullptr);
  ASSERT_EQ(notifies->GetSize(), num);
  for (size_t i = 0U; i < num; ++i) {
    ASSERT_NE(notifies->GetData()[i], nullptr);
  }
}
TEST_F(NotifyAllocatorUT, Acquire_Success_Expand) {
  NotifyAllocator na;
  ASSERT_NE(na.AcquireNotifies(0, 11), nullptr);

  size_t num = 22U;
  auto notifies = na.AcquireNotifies(0, num);
  ASSERT_NE(notifies, nullptr);
  ASSERT_EQ(notifies->GetSize(), num);
  for (size_t i = 0U; i < num; ++i) {
    ASSERT_NE(notifies->GetData()[i], nullptr);
  }
}
TEST_F(NotifyAllocatorUT, Acquire_Success_Shrink) {
  NotifyAllocator na;
  ASSERT_NE(na.AcquireNotifies(0, 33), nullptr);

  size_t num = 22U;
  auto notifies = na.AcquireNotifies(0, num);
  ASSERT_NE(notifies, nullptr);
  ASSERT_EQ(notifies->GetSize(), 33);
  for (size_t i = 0U; i < num; ++i) {
    ASSERT_NE(notifies->GetData()[i], nullptr);
  }
}
}  // namespace gert