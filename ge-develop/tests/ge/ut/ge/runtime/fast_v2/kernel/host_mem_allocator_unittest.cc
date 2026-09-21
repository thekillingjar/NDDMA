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
#include "kernel/memory/host_mem_allocator.h"
#include "runtime/mem.h"
#include <cmath>
#include "stub/gert_runtime_stub.h"
#include "core/utils/tensor_utils.h"

namespace gert {
namespace memory {

struct HostMemAllocatorTest : public testing::Test {};

TEST_F(HostMemAllocatorTest, test_host_allocator_free) {
  HostMemAllocator allocator;
  auto host_mem = allocator.Malloc(1024UL);
  ASSERT_NE(nullptr, host_mem);
  allocator.Free(host_mem);
}

TEST_F(HostMemAllocatorTest, test_host_allocator_to_tensor_data) {
  {
    HostMemAllocator allocator;
    auto host_mem = allocator.Malloc(1024UL);
    ASSERT_NE(nullptr, host_mem);
    {
      auto tensor_data = TensorUtils::ToTensorData(host_mem, 1024, kOnHost);
      tensor_data.SetPlacement(kOnHost);
      tensor_data.SetSize(1024UL);
      ASSERT_EQ(host_mem->GetCount(), 1);
      ASSERT_NE(host_mem, nullptr);
      ASSERT_EQ(tensor_data.GetSize(), 1024UL);
      ASSERT_EQ(tensor_data.GetPlacement(), kOnHost);
      {
        TensorData tensor_data_share1;
        ASSERT_EQ(tensor_data_share1.ShareFrom(tensor_data), ge::GRAPH_SUCCESS);
        ASSERT_EQ(host_mem->GetCount(), 2);
        ASSERT_EQ(tensor_data_share1.GetSize(), 1024UL);
        ASSERT_EQ(tensor_data_share1.GetPlacement(), kOnHost);

        TensorData tensor_data_share2;
        ASSERT_EQ(tensor_data_share2.ShareFrom(tensor_data), ge::GRAPH_SUCCESS);
        ASSERT_EQ(host_mem->GetCount(), 3);
        ASSERT_EQ(tensor_data_share2.GetSize(), 1024UL);
        ASSERT_EQ(tensor_data_share2.GetPlacement(), kOnHost);
      }
      ASSERT_EQ(host_mem->GetCount(), 1);
    }
    ASSERT_NE(host_mem, nullptr);
  }
}

TEST_F(HostMemAllocatorTest, HostGertMemAllocator_malloc_free_success) {
  auto host_mem_allocator = std::make_shared<HostMemAllocator>();
  memory::HostGertMemAllocator host_gert_mem_allocator(host_mem_allocator.get());
  auto multi_stream_mem_block = reinterpret_cast<MultiStreamMemBlock *>(host_gert_mem_allocator.Malloc(100U));
  ASSERT_NE(multi_stream_mem_block, nullptr);
  ASSERT_NE(multi_stream_mem_block->GetMemBlock(), nullptr);
  ASSERT_EQ(multi_stream_mem_block->GetCount(0), 1U);
  ASSERT_EQ(multi_stream_mem_block->AddCount(0), 2U);
  ASSERT_EQ(multi_stream_mem_block->SubCount(0), 1U);
  multi_stream_mem_block->Free(0);
  ASSERT_EQ(multi_stream_mem_block->GetMemBlock(), nullptr);
}

TEST_F(HostMemAllocatorTest, HostGertMemAllocator_use_gert_allocator_cache_list_success) {
  auto host_mem_allocator = std::make_shared<HostMemAllocator>();
  memory::HostGertMemAllocator host_gert_mem_allocator(host_mem_allocator.get());
  auto multi_stream_mem_block1 = reinterpret_cast<MultiStreamMemBlock *>(host_gert_mem_allocator.Malloc(100U));
  ASSERT_NE(multi_stream_mem_block1, nullptr);
  ASSERT_NE(multi_stream_mem_block1->GetMemBlock(), nullptr);
  multi_stream_mem_block1->Free(0);
  ASSERT_EQ(multi_stream_mem_block1->GetMemBlock(), nullptr);

  auto multi_stream_mem_block_2 = reinterpret_cast<MultiStreamMemBlock *>(host_gert_mem_allocator.Malloc(100U));
  ASSERT_NE(multi_stream_mem_block_2, nullptr);
  ASSERT_NE(multi_stream_mem_block_2->GetMemBlock(), nullptr);
  ASSERT_EQ(multi_stream_mem_block_2, multi_stream_mem_block1);
  multi_stream_mem_block_2->Free(0);
  ASSERT_EQ(multi_stream_mem_block_2->GetMemBlock(), nullptr);
}

TEST_F(HostMemAllocatorTest, malloc_tensor_data_success) {
  auto host_mem_allocator = std::make_shared<HostMemAllocator>();
  memory::HostGertMemAllocator host_gert_mem_allocator(host_mem_allocator.get());

  auto gert_tensor_data = host_gert_mem_allocator.MallocTensorData(100U);
  ASSERT_NE(gert_tensor_data.GetAddr(), nullptr);
  ASSERT_EQ(gert_tensor_data.GetPlacement(), kOnHost);
}
}  // namespace memory
}  // namespace gert