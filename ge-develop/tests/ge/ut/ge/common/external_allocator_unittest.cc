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
#include "kernel/memory/external_allocator.h"

namespace ge {
void *NullAllocFunc(void *obj, size_t size) {
  return nullptr;
}
void *NullAllocAdviseFunc(void *obj, size_t size, void *addr) {
  return nullptr;
}
void NullFreeFunc(void *obj, void *block) {
}
void *MyAllocFunc(void *obj, size_t size) {
  return new (std::nothrow) int(size);
}
void *MyAllocAdviseFunc(void *obj, size_t size, void *addr) {
  return new (std::nothrow) int(size);
}
void MyFreeFunc(void *obj, void *block) {
  delete static_cast<int *>(block);
}
void * MyGetAddrFromBlockFunc(void *block) {
  return block;
}
class ExternalAllocatorUT : public testing::Test {
  void SetUp() {
    null_allocator_desc_.alloc_func = NullAllocFunc;
    null_allocator_desc_.alloc_advise_func = NullAllocAdviseFunc;
    null_allocator_desc_.free_func = NullFreeFunc;

    allocator_desc_.alloc_func = MyAllocFunc;
    allocator_desc_.alloc_advise_func = MyAllocAdviseFunc;
    allocator_desc_.free_func = MyFreeFunc;
    allocator_desc_.get_addr_from_block_func = MyGetAddrFromBlockFunc;
  }
  void TearDown() {}

  public:
   AllocatorDesc null_allocator_desc_;
   AllocatorDesc allocator_desc_;
};

TEST_F(ExternalAllocatorUT, MallocFail) {
  gert::ExternalAllocator allocator{null_allocator_desc_};

  allocator.Free(nullptr);
  auto mem_block = allocator.Malloc(1024U);
  EXPECT_EQ(mem_block, nullptr);

  void *addr = (void *)0x10;
  mem_block = allocator.MallocAdvise(1024U, addr);
  EXPECT_EQ(mem_block, nullptr);
}

TEST_F(ExternalAllocatorUT, MallocOk) {
  gert::ExternalAllocator allocator{allocator_desc_};

  auto mem_block1 = allocator.Malloc(1024U);
  EXPECT_NE(mem_block1, nullptr);
  EXPECT_NE(mem_block1->GetAddr(), nullptr);
  EXPECT_EQ(mem_block1->GetSize(), 1024U);
  EXPECT_EQ(mem_block1->GetCount(), 1U);
  mem_block1->Free();

  void *addr = (void *)0x10;
  auto mem_block2 = allocator.MallocAdvise(2048U, addr);
  EXPECT_NE(mem_block2, nullptr);
  EXPECT_NE(mem_block2->GetAddr(), nullptr);
  EXPECT_EQ(mem_block2->GetSize(), 2048U);
  EXPECT_EQ(mem_block2->GetCount(), 1U);
  mem_block2->Free();
}
}  // namespace ge
