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
#include "kernel/memory/borrow_allocator.h"
#include "faker/multi_stream_allocator_faker.h"

namespace gert {
namespace memory {
class BorrowAllocatorUT : public testing::Test {};
TEST_F(BorrowAllocatorUT, Free_Alloc_Success) {
  auto holder = MultiStreamAllocatorFaker().StreamNum(1).Build();
  size_t size = 100U;
  auto mem_block = holder.l1_allocator->Malloc(size);
  ASSERT_NE(mem_block, nullptr);

  auto msb = MultiStreamMemBlock();
  msb.ReInit(holder.at(0), mem_block, BlockAllocType::kNorm);

  BorrowAllocator borrow_allocator;
  borrow_allocator.Free(&msb);
  auto borrow_msb = borrow_allocator.Alloc(size);
  ASSERT_EQ(borrow_msb, &msb);
  ASSERT_EQ(borrow_msb->GetAddr(), msb.GetAddr());
}

TEST_F(BorrowAllocatorUT, All_Borrow_Block_Small_Than_Existed_Max_Size_Then_Alloc_Failed) {
  BorrowAllocator borrow_allocator;
  size_t size = 1024U;
  auto borrow_block = borrow_allocator.Alloc(size);
  ASSERT_EQ(borrow_block, nullptr);
}

TEST_F(BorrowAllocatorUT, GetAndClearBorrowBlocks_Success) {
  auto holder = MultiStreamAllocatorFaker().StreamNum(1).Build();
  size_t size = 100U;
  auto mem_block = holder.l1_allocator->Malloc(size);
  ASSERT_NE(mem_block, nullptr);

  auto msb = MultiStreamMemBlock();
  msb.ReInit(holder.at(0), mem_block, BlockAllocType::kNorm);

  BorrowAllocator borrow_allocator;
  borrow_allocator.Free(&msb);
  auto migration_blocks = borrow_allocator.GetAndClearBorrowBlocks(0);
  ASSERT_EQ(migration_blocks.size(), 1U);
  ASSERT_EQ(migration_blocks.front(), &msb);
}

TEST_F(BorrowAllocatorUT, GetAndClearBorrowBlocks_Not_Existed_Stream_Id_Failed) {
  BorrowAllocator borrow_allocator;
  auto migration_block = borrow_allocator.GetAndClearBorrowBlocks(0);
  ASSERT_EQ(migration_block.size(), 0U);
}
}  // namespace memory
}  // namespace gert