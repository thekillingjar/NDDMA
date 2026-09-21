/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel/memory/single_stream_l2_allocator.h"
#include <gtest/gtest.h>
#include "stub/gert_runtime_stub.h"
#include "checker/memory_profiling_log_matcher.h"

namespace gert {
namespace memory {
class SingleStreamL2AllocatorUT : public testing::Test {};

TEST_F(SingleStreamL2AllocatorUT, SingleStreamL2Allocator_malloc_free_success) {
  GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().SetLevelInfo();
  auto caching_mem_allocator = std::make_shared<CachingMemAllocator>(0);
  memory::SingleStreamL2Allocator single_stream_l2_allocator(caching_mem_allocator.get());
  single_stream_l2_allocator.SetL1Allocator(caching_mem_allocator.get());
  ASSERT_EQ(single_stream_l2_allocator.GetL1Allocator(), caching_mem_allocator.get());
  ASSERT_EQ(single_stream_l2_allocator.GetStreamId(), 0);

  auto multi_stream_mem_block = reinterpret_cast<MultiStreamMemBlock *>(single_stream_l2_allocator.Malloc(100U));
  ASSERT_NE(multi_stream_mem_block, nullptr);
  ASSERT_NE(multi_stream_mem_block->GetAddr(), nullptr);

  single_stream_l2_allocator.Free(multi_stream_mem_block);
  ASSERT_TRUE(runtime_stub.GetSlogStub().FindInfoLogRegex(kBirthRecycleRe, {{2, "0"}}) >= 0);
}

TEST_F(SingleStreamL2AllocatorUT, SingleStreamL2Allocator_use_gert_allocator_cache_list_success) {
  auto caching_mem_allocator = std::make_shared<CachingMemAllocator>(0);
  memory::SingleStreamL2Allocator single_stream_l2_allocator(caching_mem_allocator.get());
  single_stream_l2_allocator.SetL1Allocator(caching_mem_allocator.get());

  auto multi_stream_mem_block1 = reinterpret_cast<MultiStreamMemBlock *>(single_stream_l2_allocator.Malloc(100U));
  ASSERT_NE(multi_stream_mem_block1, nullptr);
  ASSERT_NE(multi_stream_mem_block1->GetAddr(), nullptr);

  single_stream_l2_allocator.Free(multi_stream_mem_block1);

  auto multi_stream_mem_block2 = reinterpret_cast<MultiStreamMemBlock *>(single_stream_l2_allocator.Malloc(100U));
  ASSERT_NE(multi_stream_mem_block2, nullptr);
  ASSERT_NE(multi_stream_mem_block2->GetAddr(), nullptr);

  ASSERT_EQ(multi_stream_mem_block1, multi_stream_mem_block2);

  single_stream_l2_allocator.Free(multi_stream_mem_block2);
}

TEST_F(SingleStreamL2AllocatorUT, malloc_tensor_data_success) {
  auto l1_allocator = std::make_shared<CachingMemAllocator>(0);
  memory::SingleStreamL2Allocator single_stream_l2_allocator(l1_allocator.get());

  auto gert_tensor_data = single_stream_l2_allocator.MallocTensorData(100U);
  ASSERT_NE(gert_tensor_data.GetAddr(), nullptr);
  ASSERT_EQ(gert_tensor_data.GetPlacement(), kOnDeviceHbm);
}
}  // namespace memory
}  // namespace gert