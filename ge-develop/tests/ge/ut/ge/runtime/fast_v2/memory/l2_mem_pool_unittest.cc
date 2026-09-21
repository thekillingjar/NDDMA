/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel/memory/l2_mem_pool.h"
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "stub/gert_runtime_stub.h"
#include "checker/memory_profiling_log_matcher.h"
namespace gert {
namespace memory {
class MockL1Allocator : public ge::Allocator {
 public:
  MOCK_METHOD(ge::MemBlock *, Malloc, (size_t size));
  MOCK_METHOD(void, Free, (ge::MemBlock *block));
};
class MultiStreamL1AllocatorUT : public testing::Test {};
TEST_F(MultiStreamL1AllocatorUT, Alloc_Success) {
  MockL1Allocator l1a;
  ge::MemBlock block(l1a, (void *)0x1000, 1024);
  MultiStreamL1Allocator ms_l1a(&l1a, (rtStream_t)0x100);
  EXPECT_CALL(l1a, Malloc(1000)).WillOnce(testing::Return(&block));

  GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().NoConsoleOut().SetLevelInfo();
  ASSERT_EQ(ms_l1a.Alloc(1000), &block);
  ASSERT_TRUE(runtime_stub.GetSlogStub().FindInfoLogRegex(kPoolExpand, {{1, "0x100"}, {5, "0x1000"}, {6, "1000"}}) >= 0);
}
TEST_F(MultiStreamL1AllocatorUT, Alloc_Success_NullStream) {
  MockL1Allocator l1a;
  ge::MemBlock block(l1a, (void *)0x1000, 1024);
  MultiStreamL1Allocator ms_l1a(&l1a, nullptr);
  EXPECT_CALL(l1a, Malloc(1000)).WillOnce(testing::Return(&block));

  GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().NoConsoleOut().SetLevelInfo();
  ASSERT_EQ(ms_l1a.Alloc(1000), &block);
  ASSERT_TRUE(runtime_stub.GetSlogStub().FindInfoLogRegex(kPoolExpand, {{1, "(nil)"}, {5, "0x1000"}, {6, "1000"}}) >= 0);
}
TEST_F(MultiStreamL1AllocatorUT, Free_Success) {
  MockL1Allocator l1a;
  ge::MemBlock block(l1a, (void *)0x1000, 1024);
  MultiStreamL1Allocator ms_l1a(&l1a, (rtStream_t)0x2000);
  EXPECT_CALL(l1a, Free(&block)).Times(1);

  GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().NoConsoleOut().SetLevelInfo();
  ASSERT_TRUE(ms_l1a.Free(&block));
  ASSERT_TRUE(runtime_stub.GetSlogStub().FindInfoLogRegex(kPoolShrink, {{1, "0x2000"}, {5, "0x1000"}}) >= 0);
}
TEST_F(MultiStreamL1AllocatorUT, Free_Success_NullStream) {
  MockL1Allocator l1a;
  ge::MemBlock block(l1a, (void *)0x1000, 1024);
  MultiStreamL1Allocator ms_l1a(&l1a, nullptr);
  EXPECT_CALL(l1a, Free(&block)).Times(1);

  GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().NoConsoleOut().SetLevelInfo();
  ASSERT_TRUE(ms_l1a.Free(&block));
  ASSERT_TRUE(runtime_stub.GetSlogStub().FindInfoLogRegex(kPoolShrink, {{1, "(nil)"}, {5, "0x1000"}}) >= 0);
}
TEST_F(MultiStreamL1AllocatorUT, Free_Failed_NullBlock) {
  MockL1Allocator l1a;
  MultiStreamL1Allocator ms_l1a(&l1a, nullptr);
  EXPECT_CALL(l1a, Free(nullptr)).Times(0);

  GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().SetLevelInfo();
  ASSERT_TRUE(ms_l1a.Free(nullptr));
  ASSERT_TRUE(runtime_stub.GetSlogStub().FindInfoLogRegex(kPoolShrink) < 0);
}
TEST_F(MultiStreamL1AllocatorUT, GetDeviceId_Always_1) {
  MockL1Allocator l1a;
  MultiStreamL1Allocator ms_l1a(&l1a, nullptr);
  ASSERT_EQ(ms_l1a.GetDeviceId(), -1);
}

class L2MemPoolUT : public testing::Test {
 public:
  CachingMemAllocator l1_allocator_{0, RT_MEMORY_HBM};
};
TEST_F(L2MemPoolUT, Alloc_Success) {
  L2MemPool l2_mem_pool{&l1_allocator_, nullptr};
  ASSERT_EQ(l2_mem_pool.GetStream(), nullptr);
  rtStream_t stream = (rtStream_t)0x2000;
  l2_mem_pool.SetStream(stream);
  ASSERT_EQ(l2_mem_pool.GetStream(), stream);
  auto block = l2_mem_pool.Malloc(1024U);
  ASSERT_NE(block, nullptr);
  ASSERT_NE(block->GetAddr(), nullptr);
  block->Free();
}
TEST_F(L2MemPoolUT, Check_Free_Log_Success) {
  GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().SetLevelDebug();
  auto l2_mem_pool = std::make_unique<L2MemPool>(&l1_allocator_, nullptr);
  ASSERT_EQ(l2_mem_pool->GetStream(), nullptr);
  rtStream_t stream = (rtStream_t)0x2000;
  l2_mem_pool->SetStream(stream);
  ASSERT_EQ(l2_mem_pool->GetStream(), stream);
  auto block = l2_mem_pool->Malloc(1024U);
  ASSERT_NE(block, nullptr);
  ASSERT_NE(block->GetAddr(), nullptr);
  block->Free();
  l2_mem_pool.reset(nullptr);
  ASSERT_TRUE(runtime_stub.GetSlogStub().FindInfoLogRegex(kPoolShrink, {{1, "0x2000"}}) >= 0);
  runtime_stub.Clear();
}
}  // namespace memory
}  // namespace gert