/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel/memory/multi_stream_l2_allocator.h"
#include <gtest/gtest.h>
#include <optional>
#include "kernel/memory/multi_stream_mem_block.h"
#include "stub/runtime_stub_impl.h"
#include "core/utils/tensor_utils.h"
#include "faker/multi_stream_allocator_faker.h"
#include "stub/gert_runtime_stub.h"
#include "checker/memory_profiling_log_matcher.h"

namespace gert {
namespace memory {
constexpr uint64_t kMemSize = 4849664UL;

class MultiStreamL2AllocatorsFaker : public MultiStreamAllocatorFaker {
 public:
  struct Holder {
    std::set<MultiStreamMemBlock *> PickBlocks(std::initializer_list<int64_t> ids) {
      std::set<MultiStreamMemBlock *> ret;
      for (auto id : ids) {
        ret.insert(blocks[id].get());
      }
      return ret;
    }
    MultiStreamAllocatorFaker::Holder allocator_holder;
    std::vector<MultiStreamMemBlockPtr> blocks;
  };
  MultiStreamL2AllocatorsFaker() : MultiStreamAllocatorFaker() {
    L1Allocator(std::make_shared<CachingMemAllocator>(0, RT_MEMORY_HBM));
  }

  /**
   * 4条流
   * block 情况：
   * |No.|birth stream|use stream|local recycled on stream|
   * | 1 | 0          | 0, 1     |    1                   |
   * | 2 | 0          | 0, 1     | 0, 1                   |
   * | 3 | 1          | 0, 1, 2  |    1, 2                |
   * | 4 | 1          | 0, 1, 2  | 0,    2                |
   * @return
   */
  Holder Fake1() {
    Holder holder;
    holder.allocator_holder = StreamNum(4).Build();

    holder.blocks.emplace_back(FakeBlock(holder.allocator_holder, 0, {0, 1}, {1}));
    holder.blocks.emplace_back(FakeBlock(holder.allocator_holder, 0, {0, 1}, {0, 1}));
    holder.blocks.emplace_back(FakeBlock(holder.allocator_holder, 1, {0, 1, 2}, {1, 2}));
    holder.blocks.emplace_back(FakeBlock(holder.allocator_holder, 1, {0, 1, 2}, {0, 2}));

    return holder;
  }

 private:
  static MultiStreamMemBlockPtr FakeBlock(MultiStreamAllocatorFaker::Holder &holder, int64_t birth_stream,
                                          const std::vector<int64_t> &use_streams,
                                          const std::vector<int64_t> &local_recycle_streams) {
    auto block = holder.AllocBlock(birth_stream, 1024);
    for (auto use_stream : use_streams) {
      if (use_stream == birth_stream) {
        continue;
      }
      block->NewAccessStream(birth_stream, use_stream);
    }
    for (auto local_recycle_stream : local_recycle_streams) {
      block->Free(local_recycle_stream);
    }
    return block;
  }
};

class MultiStreamL2AllocatorUT : public testing::Test {
 protected:
  template <class T>
  static std::vector<MultiStreamMemBlock *> ConsumeVersionBlocks(T &blocks) {
    std::vector<MultiStreamMemBlock *> ret;
    for (auto iter = blocks.Begin(); iter != blocks.End(); blocks.Next(iter)) {
      ret.emplace_back(iter->version_block.block);
    }
    return ret;
  }
};

TEST_F(MultiStreamL2AllocatorUT, malloc_failed) {
  auto holder = MultiStreamL2AllocatorsFaker().StreamNum(1).Build();

  class MockRuntime : public ge::RuntimeStub {
    rtError_t rtMalloc(void **dev_ptr, uint64_t size, rtMemType_t type, uint16_t moduleId) {
      return -1;
    }
  };
  auto mock_runtime = std::make_shared<MockRuntime>();
  ge::RuntimeStub::SetInstance(mock_runtime);

  auto multi_stream_mem_block_0 = holder.AllocBlock(0, kMemSize);
  ASSERT_EQ(multi_stream_mem_block_0, nullptr);

  ge::RuntimeStub::Reset();
}

TEST_F(MultiStreamL2AllocatorUT, l2_allocator_cached_memory_success) {
  auto holder = MultiStreamL2AllocatorsFaker().StreamNum(1).Build();

  auto multi_stream_mem_block_0 = holder.AllocBlock(0, kMemSize);
  ASSERT_NE(multi_stream_mem_block_0, nullptr);
  ASSERT_EQ(multi_stream_mem_block_0->GetSize(), kMemSize);
  multi_stream_mem_block_0->Free(0);
  ASSERT_EQ(multi_stream_mem_block_0->GetMemBlock(), nullptr);
  multi_stream_mem_block_0.release();

  class MockRuntime : public ge::RuntimeStub {
    rtError_t rtMalloc(void **dev_ptr, uint64_t size, rtMemType_t type, uint16_t moduleId) {
      return -1;
    }
  };
  auto mock_runtime = std::make_shared<MockRuntime>();
  ge::RuntimeStub::SetInstance(mock_runtime);

  multi_stream_mem_block_0 = holder.AllocBlock(0, kMemSize);
  ASSERT_NE(multi_stream_mem_block_0, nullptr);
  ASSERT_EQ(multi_stream_mem_block_0->GetSize(), kMemSize);
  multi_stream_mem_block_0->Free(0);
  ASSERT_EQ(multi_stream_mem_block_0->GetMemBlock(), nullptr);

  ge::RuntimeStub::Reset();
}

TEST_F(MultiStreamL2AllocatorUT, own_malloc_free_success) {
  auto holder = MultiStreamL2AllocatorsFaker().StreamNum(1).Build();

  auto multi_stream_mem_block = holder.AllocBlock(0, kMemSize);
  ASSERT_NE(multi_stream_mem_block, nullptr);
  ASSERT_EQ(multi_stream_mem_block->GetSize(), kMemSize);
  ASSERT_NE(multi_stream_mem_block->GetAddr(), nullptr);
  ASSERT_NE(multi_stream_mem_block->GetMemBlock(), nullptr);
  ASSERT_EQ(multi_stream_mem_block->GetCount(0), 1U);

  multi_stream_mem_block->Free(0);
  ASSERT_EQ(multi_stream_mem_block->GetMemBlock(), nullptr);
}

/**
 * 1.stream 0's l2 allocator malloc block_0
 * 2.block_0 access memory form stream 0 to stream 1
 * 3.block_0 free on stream 0, type is local-recycle
 * 4.send and receive event from stream 0 to stream 1
 * 5.block_0 free on stream_1, type is borrow recycle
 * 6.rtMalloc fake fail, stream 1's l2 allocator malloc block_1 from borrow
 * 7.block_1 free on stream_1, type is borrow recycle
 * 8.migration from stream 1 to stram 0
 */
TEST_F(MultiStreamL2AllocatorUT, own_malloc_failed_but_using_borrow_success) {
  auto holders = MultiStreamL2AllocatorsFaker().StreamNum(2).Build();

  auto multi_stream_mem_block_0 = reinterpret_cast<MultiStreamMemBlock *>(holders.at(0)->Malloc(kMemSize));
  ASSERT_NE(multi_stream_mem_block_0, nullptr);
  const auto mem_block_addr = multi_stream_mem_block_0->GetAddr();
  ASSERT_NE(mem_block_addr, nullptr);
  // stream 0's count is 1
  ASSERT_EQ(multi_stream_mem_block_0->GetCount(0), 1U);

  // Access mem from 0 to 1
  ASSERT_EQ(multi_stream_mem_block_0->NewAccessStream(0, 1), ge::GRAPH_SUCCESS);
  // stream 1's count is 1
  ASSERT_EQ(multi_stream_mem_block_0->GetCount(1), 1U);

  // free on stream 0, stream 0 local-recycle the block, the block in stream 0's local_recycle
  multi_stream_mem_block_0->Free(0);
  ASSERT_NE(multi_stream_mem_block_0->GetMemBlock(), nullptr);
  ASSERT_NE(multi_stream_mem_block_0->GetAddr(), nullptr);
  // stream 0's count is 0
  ASSERT_EQ(multi_stream_mem_block_0->GetCount(0), 0U);
  // stream 0 local-recycle the block, the block in stream 0's local_recycle
  auto local_recycle_blocks_0 = holders.at(0)->GetClearLocalRecycleBlocks(1);

  // send and receive event from stream 0 to stream 1
  auto block = local_recycle_blocks_0.Begin();
  block->version_block.block->SyncLocalRecycleStatus(0, 1);

  // free on stream 1, stream 1 recycle the block, recycle type is borrow
  multi_stream_mem_block_0->Free(1);

  class MockRuntime : public ge::RuntimeStub {
    rtError_t rtMalloc(void **dev_ptr, uint64_t size, rtMemType_t type, uint16_t moduleId) {
      return -1;
    }
  };
  auto mock_runtime = std::make_shared<MockRuntime>();
  ge::RuntimeStub::SetInstance(mock_runtime);

  // stream 1 malloc, using borrow block
  auto multi_stream_mem_block_1 = reinterpret_cast<MultiStreamMemBlock *>(holders.at(1)->Malloc(kMemSize));
  ASSERT_NE(multi_stream_mem_block_1, nullptr);
  ASSERT_EQ(multi_stream_mem_block_1->GetAddr(), mem_block_addr);

  multi_stream_mem_block_1->Free(1);

  // migration from stream 1 to stream 0
  auto borrow_blocks = holders.at(1)->GetAndClearBorrowBlocks(0);
  ASSERT_EQ(borrow_blocks.size(), 1U);
  ASSERT_EQ(borrow_blocks.front(), multi_stream_mem_block_0);
  for (const auto &block : borrow_blocks) {
    holders.at(0)->BirthRecycle(block);
  }

  // gert mem block 's mem block is null
  ASSERT_EQ(multi_stream_mem_block_0->GetMemBlock(), nullptr);

  ge::RuntimeStub::Reset();
}

/**
 * 1.stream 0's l2 allocator malloc block_0
 * 2.block_0 access memory form stream 0 to stream 1
 * 3.block_0 free on stream 0, type is local-recycle
 * 4.block_0 free on stream_1, type is local-recycle
 * 5.send and receive event from stream 0 to stream 1, borrow-recycle
 * 6.migration from stream 1 to stram 0
 */
TEST_F(MultiStreamL2AllocatorUT, own_malloc_free_before_sync_local_status_success) {
  auto holders = MultiStreamL2AllocatorsFaker().StreamNum(2).Build();

  auto multi_stream_mem_block_0 = reinterpret_cast<MultiStreamMemBlock *>(holders.at(0)->Malloc(kMemSize));
  ASSERT_NE(multi_stream_mem_block_0, nullptr);
  const auto mem_block_addr = multi_stream_mem_block_0->GetAddr();
  ASSERT_NE(mem_block_addr, nullptr);
  // stream 0's count is 1
  ASSERT_EQ(multi_stream_mem_block_0->GetCount(0), 1U);

  // Access mem from 0 to 1
  multi_stream_mem_block_0->NewAccessStream(0, 1);
  // stream 1's count is 1
  ASSERT_EQ(multi_stream_mem_block_0->GetCount(1), 1U);

  // free on stream 0, stream 0 local-recycle the block, the block in stream 0's local_recycle
  multi_stream_mem_block_0->Free(0);
  ASSERT_NE(multi_stream_mem_block_0->GetMemBlock(), nullptr);
  ASSERT_NE(multi_stream_mem_block_0->GetAddr(), nullptr);
  // stream 0's count is 0
  ASSERT_EQ(multi_stream_mem_block_0->GetCount(0), 0U);
  // stream 0 local-recycle the block, the block in stream 0's local_recycle
  auto local_recycle_blocks_0 = holders.at(0)->GetClearLocalRecycleBlocks(1);

  // free on stream 1
  multi_stream_mem_block_0->Free(1);
  ASSERT_NE(multi_stream_mem_block_0->GetAddr(), nullptr);

  auto origin_version = multi_stream_mem_block_0->GetVersion();

  // send and receive event from stream 0 to stream 1
  // stream 1 recycle the block, recycle type is borrow
  auto block = local_recycle_blocks_0.Begin();
  block->version_block.block->SyncLocalRecycleStatus(0, 1);

  ASSERT_NE(multi_stream_mem_block_0->GetVersion(), origin_version);

  // migration from stream 1 to stream 0
  auto borrow_blocks = holders.at(1)->GetAndClearBorrowBlocks(0);
  ASSERT_EQ(borrow_blocks.size(), 1U);
  ASSERT_EQ(borrow_blocks.front(), multi_stream_mem_block_0);
  for (const auto &block : borrow_blocks) {
    holders.at(0)->BirthRecycle(block);
  }

  // gert mem block 's mem block is null
  ASSERT_EQ(multi_stream_mem_block_0->GetMemBlock(), nullptr);
}

/**
 * 1.create stream 0's multi stream l2 allocator with l1 allocator 0
 * 2.set new l1 allocator 1 in multi stream l2 allocator
 * 3.l2 allocator malloc block
 * 4.free on stream 0, stream 0 recycle the block
 */
TEST_F(MultiStreamL2AllocatorUT, change_l1_allocator_success) {
  auto l1_allocator_0 = std::make_shared<CachingMemAllocator>(0, RT_MEMORY_HBM);
  auto l1_allocator_1 = std::make_shared<CachingMemAllocator>(1, RT_MEMORY_HBM);
  auto holder = MultiStreamL2AllocatorsFaker().L1Allocator(l1_allocator_0).Build();

  // set new l1 allocator 1 in multi stream l2 allocator
  ASSERT_EQ(holder.at(0)->SetL1Allocator(l1_allocator_1.get()), ge::GRAPH_SUCCESS);
  ASSERT_EQ(l1_allocator_1->GetScalableAllocator()->GetOccupiedMemSize(), 0U);

  auto multi_stream_mem_block_0 = reinterpret_cast<MultiStreamMemBlock *>(holder.at(0)->Malloc(kMemSize));
  ASSERT_EQ(l1_allocator_1->GetScalableAllocator()->GetOccupiedMemSize(), kMemSize);
  ASSERT_NE(multi_stream_mem_block_0, nullptr);
  auto mem_block_addr = multi_stream_mem_block_0->GetAddr();
  ASSERT_NE(mem_block_addr, nullptr);
  ASSERT_EQ(multi_stream_mem_block_0->GetCount(0), 1U);

  // free on stream 0, stream 0 recycle the block
  multi_stream_mem_block_0->Free(0);
  ASSERT_EQ(multi_stream_mem_block_0->GetMemBlock(), nullptr);
}

TEST_F(MultiStreamL2AllocatorUT, Malloc_GertMemBlockReuse) {
  auto holder = MultiStreamL2AllocatorsFaker().Build();

  auto multi_stream_mem_block_1 = holder.at(0)->Malloc(kMemSize);
  ASSERT_NE(multi_stream_mem_block_1, nullptr);
  auto multi_stream_mem_block_2 = holder.at(0)->Malloc(kMemSize);
  ASSERT_NE(multi_stream_mem_block_2, nullptr);
  multi_stream_mem_block_1->Free(0);

  auto multi_stream_mem_block_3 = holder.at(0)->Malloc(kMemSize);
  ASSERT_NE(multi_stream_mem_block_3, nullptr);

  // multi_stream_mem_block_3 use cached multi_stream_mem_block_1
  ASSERT_EQ(multi_stream_mem_block_3, multi_stream_mem_block_1);

  multi_stream_mem_block_2->Free(0);
  multi_stream_mem_block_3->Free(0);
}

TEST_F(MultiStreamL2AllocatorUT, malloc_tensor_data_success) {
  auto holder = MultiStreamL2AllocatorsFaker().Build();

  auto gert_tensor_data = holder.at(0)->MallocTensorData(100U);
  ASSERT_NE(gert_tensor_data.GetAddr(), nullptr);
  ASSERT_EQ(gert_tensor_data.GetPlacement(), kOnDeviceHbm);
}
TEST_F(MultiStreamL2AllocatorUT, Free_LocalRecycle_WhenOtherUses) {
  auto holder = MultiStreamL2AllocatorsFaker().StreamNum(2).Build();
  auto block = holder.AllocBlock(0, 1024);
  ASSERT_NE(block, nullptr);
  auto addr = ToHex(block->GetAddr());
  ASSERT_EQ(block->NewAccessStream(0, 1), ge::GRAPH_SUCCESS);
  GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().NoConsoleOut().SetLevelInfo();
  block->Free(0);
  ASSERT_EQ(block->GetBirthStreamId(), 0);
  ASSERT_EQ(block->GetCount(0), 0);
  ASSERT_EQ(block->GetCount(1), 1);
  ASSERT_TRUE(runtime_stub.GetSlogStub().FindInfoLogRegex(kLocalRecycleRe, {{2, "0"}, {3, addr}}) >= 0);
  block->SyncLocalRecycleStatus(0, 1);
}
TEST_F(MultiStreamL2AllocatorUT, Free_BorrowRecycle_AfterSync) {
  auto holder = MultiStreamL2AllocatorsFaker().StreamNum(2).Build();
  auto block = holder.AllocBlock(0, 1024);
  ASSERT_NE(block, nullptr);
  auto addr = ToHex(block->GetAddr());
  ASSERT_EQ(block->NewAccessStream(0, 1), ge::GRAPH_SUCCESS);
  block->Free(0);
  block->Free(1);
  GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().NoConsoleOut().SetLevelInfo();
  block->SyncLocalRecycleStatus(0, 1);
  ASSERT_TRUE(runtime_stub.GetSlogStub().FindInfoLogRegex(kBorrowRecycleRe, {{2, "1"}, {3, addr}}) >= 0);
}
TEST_F(MultiStreamL2AllocatorUT, Free_BorrowRecycle_AfterFree) {
  auto holder = MultiStreamL2AllocatorsFaker().StreamNum(2).Build();
  auto block = holder.AllocBlock(0, 1024);
  ASSERT_NE(block, nullptr);
  auto addr = ToHex(block->GetAddr());
  ASSERT_EQ(block->NewAccessStream(0, 1), ge::GRAPH_SUCCESS);
  block->Free(0);
  block->SyncLocalRecycleStatus(0, 1);
  GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().NoConsoleOut().SetLevelInfo();
  block->Free(1);
  ASSERT_TRUE(runtime_stub.GetSlogStub().FindInfoLogRegex(kBorrowRecycleRe, {{2, "1"}, {3, addr}}) >= 0);
}
TEST_F(MultiStreamL2AllocatorUT, Free_BirthRecycle_SingleStream) {
  auto holder = MultiStreamL2AllocatorsFaker().StreamNum(2).Build();
  auto block = holder.AllocBlock(0, 1024);
  ASSERT_NE(block, nullptr);
  auto addr = ToHex(block->GetAddr());
  GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().NoConsoleOut().SetLevelInfo();
  block->Free(0);
  ASSERT_TRUE(runtime_stub.GetSlogStub().FindInfoLogRegex(kBirthRecycleRe, {{2, "0"}, {3, addr}}) >= 0);
}
TEST_F(MultiStreamL2AllocatorUT, All_L2MemPool_Recycle_When_L1_Malloc_Fail) {
  static uint64_t total_mem_size = 0;
  static uint8_t rt_malloc_count = 0;
  class MockRuntime : public RuntimeStubImpl {
    rtError_t rtMalloc(void **dev_ptr, uint64_t size, rtMemType_t type, uint16_t moduleId) {
      rt_malloc_count++;
      total_mem_size += size;
      if (total_mem_size > 30 * 1024UL * 1024UL) {
        total_mem_size = (rt_malloc_count == 15) ? 0 : total_mem_size;
        *dev_ptr = nullptr;
        return -1;
      }
      *dev_ptr = new uint8_t[1];
      return RT_ERROR_NONE;
    }
  };

  auto runtime_stub = GertRuntimeStub{std::make_unique<MockRuntime>()};
  runtime_stub.GetSlogStub().SetLevel(1);
  int64_t stream_num = 3;
  auto l1_allocator = std::make_shared<CachingMemAllocator>(0, RT_MEMORY_HBM);
  auto holder = MultiStreamL2AllocatorsFaker().StreamNum(stream_num).L1Allocator(l1_allocator).Build();
  auto origin_occupied_size = l1_allocator->GetScalableAllocator()->GetOccupiedMemSize();
  size_t page_size = 1024UL * 1024UL;
  auto block_0 = holder.at(0)->Malloc(10 * page_size);
  ASSERT_NE(block_0->GetAddr(), nullptr);
  auto block_1 = holder.at(1)->Malloc(10 * page_size);
  ASSERT_NE(block_1->GetAddr(), nullptr);
  auto block_2 = holder.at(2)->Malloc(10 * page_size);
  ASSERT_NE(block_2->GetAddr(), nullptr);
  auto after_malloc_occupied_size = l1_allocator->GetScalableAllocator()->GetOccupiedMemSize() - origin_occupied_size;
  ASSERT_EQ(after_malloc_occupied_size, 30 * page_size);
  block_0->Free(0);
  block_1->Free(1);
  block_2->Free(2);
  auto after_free_occupied_size = l1_allocator->GetScalableAllocator()->GetOccupiedMemSize() - origin_occupied_size;
  ASSERT_EQ(after_free_occupied_size, 30 * page_size);
  auto block = holder.at(1)->Malloc(25 * page_size);
  ASSERT_NE(block, nullptr);
  ASSERT_NE(block->GetAddr(), nullptr);
  auto after_recycle_occupied_size = l1_allocator->GetScalableAllocator()->GetOccupiedMemSize() - origin_occupied_size;
  ASSERT_EQ(after_recycle_occupied_size, 25 * page_size);
  block->Free(1);
  std::string recycle_log = "malloc memory failed, try to free l2 mem pool and malloc again";
  EXPECT_NE(runtime_stub.GetSlogStub().FindInfoLogRegex(recycle_log.c_str()), -1);
  for (int64_t i = 0; i < stream_num; ++i) {
    std::string l1_free_log = "Shrink memory pool at stream 0x" + std::to_string(i + 1);
    EXPECT_NE(runtime_stub.GetSlogStub().FindInfoLogRegex(l1_free_log.c_str()), -1);
  }
  runtime_stub.Clear();
}
TEST_F(MultiStreamL2AllocatorUT, Free_BirthRecycle_AfterSync) {
  auto holder = MultiStreamL2AllocatorsFaker().StreamNum(2).Build();
  auto block = holder.AllocBlock(0, 1024);
  ASSERT_NE(block, nullptr);
  auto addr = ToHex(block->GetAddr());
  ASSERT_EQ(block->NewAccessStream(0, 1), ge::GRAPH_SUCCESS);
  block->Free(1);
  block->Free(0);
  GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().NoConsoleOut().SetLevelInfo();
  block->SyncLocalRecycleStatus(1, 0);
  ASSERT_TRUE(runtime_stub.GetSlogStub().FindInfoLogRegex(kBirthRecycleRe, {{2, "0"}, {3, addr}}) >= 0);
}
TEST_F(MultiStreamL2AllocatorUT, Free_BirthRecycle_AfterFree) {
  auto holder = MultiStreamL2AllocatorsFaker().StreamNum(2).Build();
  auto block = holder.AllocBlock(0, 1024);
  ASSERT_NE(block, nullptr);
  auto addr = ToHex(block->GetAddr());
  ASSERT_EQ(block->NewAccessStream(0, 1), ge::GRAPH_SUCCESS);
  block->Free(1);
  block->SyncLocalRecycleStatus(1, 0);
  GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().NoConsoleOut().SetLevelInfo();
  block->Free(0);
  ASSERT_TRUE(runtime_stub.GetSlogStub().FindInfoLogRegex(kBirthRecycleRe, {{2, "0"}, {3, addr}}) >= 0);
}
TEST_F(MultiStreamL2AllocatorUT, GetClearLocalRecycleBlocks_Success_ByStreamId1) {
  auto holder = MultiStreamL2AllocatorsFaker().Fake1();

  auto version_blocks = holder.allocator_holder.at(1)->GetClearLocalRecycleBlocks(0);
  std::set<MultiStreamMemBlock *> blocks;
  for (auto iter = version_blocks.Begin(); iter != version_blocks.End(); version_blocks.Next(iter)) {
    blocks.insert(iter->version_block.block);
  }
  ASSERT_EQ(blocks, holder.PickBlocks({0, 1, 2}));

  auto version_blocks_2 = holder.allocator_holder.at(1)->GetClearLocalRecycleBlocks(0);
  ASSERT_EQ(version_blocks_2.Begin(), version_blocks_2.End());
}
TEST_F(MultiStreamL2AllocatorUT, GetClearLocalRecycleBlocks_Success_ByStreamId2) {
  auto holder = MultiStreamL2AllocatorsFaker().Fake1();

  holder.allocator_holder.at(1)->GetClearLocalRecycleBlocks(0);
  auto version_blocks = holder.allocator_holder.at(1)->GetClearLocalRecycleBlocks(3);
  std::set<MultiStreamMemBlock *> blocks;
  for (auto iter = version_blocks.Begin(); iter != version_blocks.End(); version_blocks.Next(iter)) {
    blocks.insert(iter->version_block.block);
  }
  ASSERT_EQ(blocks, holder.PickBlocks({0, 1, 2}));
}
TEST_F(MultiStreamL2AllocatorUT, MoveL2ToL1_HasSplited_Success) {
  auto l1_allocator = std::make_shared<memory::CachingMemAllocator>(0, RT_MEMORY_HBM);
  auto holder = MultiStreamL2AllocatorsFaker().StreamNum(1).L1Allocator(l1_allocator).Build();
  auto l2_allocator = holder.at(0);
  auto block_1 = l2_allocator->Malloc(1024UL * 1024UL);
  ASSERT_NE(block_1, nullptr);
  block_1->Free(0);
  auto block_2 = reinterpret_cast<MultiStreamMemBlock *>(l2_allocator->Malloc(1024UL));
  ASSERT_NE(block_2, nullptr);
  auto mem_block = block_2->GetMemBlock();
  auto span = reinterpret_cast<PageSpan *>(mem_block);
  ASSERT_NE(span, nullptr);
  ASSERT_EQ(span->HasSplited(), true);
  ASSERT_EQ(l2_allocator->MoveL2ToL1(block_2), ge::GRAPH_SUCCESS);
  ASSERT_NE(block_2->GetMemBlock(), mem_block);
  block_2->Free(0);
}
TEST_F(MultiStreamL2AllocatorUT, MoveL2ToL1_NotSplited_Success) {
  auto l1_allocator = std::make_shared<memory::CachingMemAllocator>(0, RT_MEMORY_HBM);
  auto holder = MultiStreamL2AllocatorsFaker().StreamNum(1).L1Allocator(l1_allocator).Build();
  auto l2_allocator = holder.at(0);
  auto block_1 = reinterpret_cast<MultiStreamMemBlock *>(l2_allocator->Malloc(1024UL));
  ASSERT_NE(block_1, nullptr);
  auto mem_block = block_1->GetMemBlock();
  auto root_block = reinterpret_cast<PageSpan *>(mem_block)->GetBlockAddr();
  auto span = reinterpret_cast<PageSpan *>(mem_block);
  ASSERT_NE(span, nullptr);
  ASSERT_EQ(span->HasSplited(), false);
  ASSERT_EQ(l2_allocator->MoveL2ToL1(block_1), ge::GRAPH_SUCCESS);
  ASSERT_NE(block_1->GetMemBlock(), mem_block);
  ASSERT_EQ(block_1->GetMemBlock(), root_block);
  block_1->Free(0);
}
TEST_F(MultiStreamL2AllocatorUT, GetClearLocalRecycleBlocks_Success_GetAll1) {
  auto holder = MultiStreamL2AllocatorsFaker().Fake1();

  auto vb1 = holder.allocator_holder.at(1)->GetClearLocalRecycleBlocks(0);
  for (auto iter = vb1.Begin(); iter != vb1.End(); vb1.Next(iter))
    ;
  auto version_blocks = holder.allocator_holder.at(1)->GetClearLocalRecycleBlocks();
  std::set<MultiStreamMemBlock *> blocks;
  for (auto iter = version_blocks.Begin(); iter != version_blocks.End(); version_blocks.Next(iter)) {
    blocks.insert(iter->version_block.block);
  }
  ASSERT_EQ(blocks, holder.PickBlocks({0, 1, 2}));
}
TEST_F(MultiStreamL2AllocatorUT, GetClearLocalRecycleBlocks_Success_GetAll2) {
  auto holder = MultiStreamL2AllocatorsFaker().Fake1();

  auto vb1 = holder.allocator_holder.at(1)->GetClearLocalRecycleBlocks(0);
  ASSERT_EQ(ConsumeVersionBlocks(vb1).size(), 3);
  auto vb2 = holder.allocator_holder.at(1)->GetClearLocalRecycleBlocks(2);
  ASSERT_EQ(ConsumeVersionBlocks(vb2).size(), 3);
  auto vb3 = holder.allocator_holder.at(1)->GetClearLocalRecycleBlocks(3);
  ASSERT_EQ(ConsumeVersionBlocks(vb3).size(), 3);

  auto version_blocks = holder.allocator_holder.at(1)->GetClearLocalRecycleBlocks();
  std::set<MultiStreamMemBlock *> blocks;
  for (auto iter = version_blocks.Begin(); iter != version_blocks.End(); version_blocks.Next(iter)) {
    blocks.insert(iter->version_block.block);
  }
  ASSERT_TRUE(blocks.empty());
}
// TEST_F(MultiStreamL2AllocatorUT, GetClearLocalRecycleBlocks_FiltSuccess_HasOutDatedBlock) {}
}  // namespace memory
}  // namespace gert