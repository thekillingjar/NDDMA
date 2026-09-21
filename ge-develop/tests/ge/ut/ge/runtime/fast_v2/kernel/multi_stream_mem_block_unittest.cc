/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel/memory/multi_stream_mem_block.h"
#include <gtest/gtest.h>
#include "exe_graph/runtime/gert_mem_allocator.h"
#include "faker/multi_stream_allocator_faker.h"
namespace gert {
namespace memory {
class MultiStreamMemBlockUT : public testing::Test {};

TEST_F(MultiStreamMemBlockUT, multi_stream_init_success) {
  auto holder = MultiStreamAllocatorFaker().StreamNum(2).Build();
  auto mem_block = holder.l1_allocator->Malloc(100U);

  auto gert_mem_block = MultiStreamMemBlock();
  gert_mem_block.ReInit(holder.at(0), mem_block, BlockAllocType::kNorm);

  ASSERT_EQ(gert_mem_block.GetAddr(), mem_block->GetAddr());
  ASSERT_EQ(gert_mem_block.GetSize(), 100U);
  ASSERT_EQ(gert_mem_block.GetBirthStreamId(), 0);
  ASSERT_EQ(gert_mem_block.GetCount(0), 1);
  ASSERT_EQ(gert_mem_block.GetCount(1), 0);
}

TEST_F(MultiStreamMemBlockUT, multi_stream_add_sub_count_success) {
  auto holder = MultiStreamAllocatorFaker().StreamNum(2).Build();

  auto gert_mem_block = MultiStreamMemBlock();
  gert_mem_block.ReInit(holder.at(0), nullptr, BlockAllocType::kNorm);
  ASSERT_EQ(gert_mem_block.GetCount(0), 1);

  gert_mem_block.SubCount(0);
  ASSERT_EQ(gert_mem_block.GetCount(0), 0);

  ASSERT_EQ(gert_mem_block.GetCount(1), 0);
  gert_mem_block.AddCount(1);
  ASSERT_EQ(gert_mem_block.GetCount(1), 1);
}
}  // namespace memory
}  // namespace gert