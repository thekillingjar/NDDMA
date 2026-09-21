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
#include <memory>

#include "macro_utils/dt_public_scope.h"
#include "graph/manager/rdma_pool_allocator.h"
#include <framework/common/debug/log.h>
#include "framework/common/debug/ge_log.h"
#include "graph/ge_context.h"
#include "runtime/dev.h"
#include "graph/manager/mem_manager.h"
#include "macro_utils/dt_public_unscope.h"

namespace ge {
class UtestRdmaPoolAllocatorTest : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UtestRdmaPoolAllocatorTest, RdmaPoolAllocator) {
  RdmaPoolAllocator allocator(RT_MEMORY_HBM);
  (void)allocator.Initialize();
  (void)allocator.InitMemory(1024*1024);
  uint32_t device_id = 0;
  uint8_t *addr = allocator.Malloc(1024, device_id);
  uint64_t base_addr;
  uint64_t mem_size;
  (void)allocator.GetBaseAddr(base_addr, mem_size);
  EXPECT_EQ(allocator.Free(addr, device_id), SUCCESS);
  allocator.Finalize();
}

TEST_F(UtestRdmaPoolAllocatorTest, malloc_with_reuse) {
  std::vector<rtMemType_t> mem_type;
  mem_type.push_back(RT_MEMORY_HBM);
  EXPECT_EQ(SUCCESS, MemManager::Instance().Initialize(mem_type));
  size_t size = 1024 * 10;
  EXPECT_EQ(SUCCESS, MemManager::Instance().RdmaPoolInstance(RT_MEMORY_HBM).InitMemory(size));
  size_t node_mem = 512;
  auto node1 = MemManager::Instance().RdmaPoolInstance(RT_MEMORY_HBM).Malloc(node_mem);
  EXPECT_NE(nullptr, node1);
  auto node2 = MemManager::Instance().RdmaPoolInstance(RT_MEMORY_HBM).Malloc(node_mem);
  EXPECT_NE(nullptr, node2);
  auto node3 = MemManager::Instance().RdmaPoolInstance(RT_MEMORY_HBM).Malloc(node_mem);
  EXPECT_NE(nullptr, node3);
  auto node4 = MemManager::Instance().RdmaPoolInstance(RT_MEMORY_HBM).Malloc(node_mem);
  EXPECT_NE(nullptr, node4);
  auto node5 = MemManager::Instance().RdmaPoolInstance(RT_MEMORY_HBM).Malloc(node_mem);
  EXPECT_NE(nullptr, node5);

  EXPECT_EQ(SUCCESS, MemManager::Instance().RdmaPoolInstance(RT_MEMORY_HBM).Free(node1));
  EXPECT_EQ(SUCCESS, MemManager::Instance().RdmaPoolInstance(RT_MEMORY_HBM).Free(node3));
  EXPECT_EQ(SUCCESS, MemManager::Instance().RdmaPoolInstance(RT_MEMORY_HBM).Free(node2));
  EXPECT_EQ(SUCCESS, MemManager::Instance().RdmaPoolInstance(RT_MEMORY_HBM).Free(node4));
  EXPECT_EQ(SUCCESS, MemManager::Instance().RdmaPoolInstance(RT_MEMORY_HBM).Free(node5));
  MemManager::Instance().Finalize();
}

TEST_F(UtestRdmaPoolAllocatorTest, initrdma_success) {
  std::vector<rtMemType_t> mem_type;
  mem_type.push_back(RT_MEMORY_HBM);
  EXPECT_EQ(SUCCESS, MemManager::Instance().Initialize(mem_type));
  size_t size = 1024;
  EXPECT_EQ(SUCCESS, MemManager::Instance().RdmaPoolInstance(RT_MEMORY_HBM).InitMemory(size));
  MemManager::Instance().Finalize();
}

TEST_F(UtestRdmaPoolAllocatorTest, repeat_initrdma) {
  std::vector<rtMemType_t> mem_type;
  mem_type.push_back(RT_MEMORY_HBM);
  EXPECT_EQ(SUCCESS, MemManager::Instance().Initialize(mem_type));
  size_t size = 1024;
  EXPECT_EQ(SUCCESS, MemManager::Instance().RdmaPoolInstance(RT_MEMORY_HBM).InitMemory(size));
  EXPECT_NE(SUCCESS, MemManager::Instance().RdmaPoolInstance(RT_MEMORY_HBM).InitMemory(size));
  MemManager::Instance().Finalize();
}

TEST_F(UtestRdmaPoolAllocatorTest, malloc_falled) {
  std::vector<rtMemType_t> mem_type;
  mem_type.push_back(RT_MEMORY_HBM);
  EXPECT_EQ(SUCCESS, MemManager::Instance().Initialize(mem_type));
  size_t size = 1024 * 10;
  EXPECT_EQ(SUCCESS, MemManager::Instance().RdmaPoolInstance(RT_MEMORY_HBM).InitMemory(size));
  EXPECT_EQ(nullptr, MemManager::Instance().RdmaPoolInstance(RT_MEMORY_HBM).Malloc(size * 2));
  MemManager::Instance().Finalize();
}

TEST_F(UtestRdmaPoolAllocatorTest, get_baseaddr) {
  std::vector<rtMemType_t> mem_type;
  mem_type.push_back(RT_MEMORY_HBM);
  EXPECT_EQ(SUCCESS, MemManager::Instance().Initialize(mem_type));
  uint64_t base_addr;
  uint64_t base_size;
  EXPECT_NE(SUCCESS, MemManager::Instance().RdmaPoolInstance(RT_MEMORY_HBM).GetBaseAddr(base_addr, base_size));
  size_t size = 1024;
  EXPECT_EQ(SUCCESS, MemManager::Instance().RdmaPoolInstance(RT_MEMORY_HBM).InitMemory(size));
  EXPECT_EQ(SUCCESS, MemManager::Instance().RdmaPoolInstance(RT_MEMORY_HBM).GetBaseAddr(base_addr, base_size));
  MemManager::Instance().Finalize();
}

} // namespace ge
