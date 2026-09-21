/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel/memory/ti_block_allocator.h"
#include <gtest/gtest.h>
#include "faker/fake_value.h"
namespace gert {
namespace memory {
class TiBlockAllocatorUT : public testing::Test {};
TEST_F(TiBlockAllocatorUT, ShareAndFree_Ok) {
  auto holder = TensorFaker().Shape({2, 3, 4}).Build();

  ASSERT_EQ(holder.GetRefCount(), 1U);
  TiBlockAllocator allocator;
  auto block = allocator.ShareFromTensorData(holder.GetTensor()->GetTensorData());
  ASSERT_NE(block, nullptr);
  EXPECT_EQ(block->GetAddr(), holder.GetTensor()->GetAddr());
  EXPECT_EQ(block->GetSize(), holder.GetTensor()->GetSize());
  EXPECT_EQ(block->GetCount(), 1U);
  ASSERT_EQ(holder.GetRefCount(), 2U);

  block->Free();
  ASSERT_EQ(holder.GetRefCount(), 1U);
}
}  // namespace memory
}  // namespace gert