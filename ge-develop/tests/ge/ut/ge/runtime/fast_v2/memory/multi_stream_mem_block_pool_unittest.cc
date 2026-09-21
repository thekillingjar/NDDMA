/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel/memory/multi_stream_mem_block_pool.h"
#include <gtest/gtest.h>
#include "faker/multi_stream_allocator_faker.h"
namespace gert {
namespace memory {
class MultiStreamMemBlockPoolUT : public testing::Test {};
// Todo: 后续适配
TEST_F(MultiStreamMemBlockPoolUT, AcquireAndRelease_Success) {
  auto holder = MultiStreamAllocatorFaker().StreamNum(2).Build();
  auto mb = holder.l1_allocator->Malloc(100);

  MultiStreamMemBlockPool pool;
  auto block = pool.Acquire(holder.at(0), mb, BlockAllocType::kNorm);
  ASSERT_NE(block, nullptr);
  ASSERT_EQ(block->GetCount(0), 1U);
  ASSERT_EQ(block->GetCount(1), 0U);

  pool.Release(block);
}
TEST_F(MultiStreamMemBlockPoolUT, Acquire_ReuseSuccess) {
  auto holder = MultiStreamAllocatorFaker().StreamNum(2).Build();
  auto mb1 = holder.l1_allocator->Malloc(100);
  auto mb2 = holder.l1_allocator->Malloc(100);

  MultiStreamMemBlockPool pool;
  auto block = pool.Acquire(holder.at(0), mb1, BlockAllocType::kNorm);
  auto backup_block = block;
  pool.Release(block);
  block = pool.Acquire(holder.at(0), mb2, BlockAllocType::kNorm);
  EXPECT_TRUE(block == backup_block);
}

TEST_F(MultiStreamMemBlockPoolUT, AcquireReleaseMultiple_Success) {
  auto holder = MultiStreamAllocatorFaker().StreamNum(2).Build();
  MultiStreamMemBlockPool pool;

  EXPECT_NO_THROW(
    std::vector<MultiStreamMemBlock *> blocks;
    for (int32_t i = 0; i < 10; ++i) {
      blocks.emplace_back(pool.Acquire(holder.at(0), holder.l1_allocator->Malloc(100), BlockAllocType::kNorm));
    }
    for (auto &block : blocks) {
      pool.Release(block);
    }
  );
}
}  // namespace memory
}  // namespace gert
