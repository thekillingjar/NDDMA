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

#include "graph/anchor.h"
#include "graph/attr_value.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "omg/omg_inner_types.h"

#include "macro_utils/dt_public_scope.h"
#include "graph/manager/mem_manager.h"
#include "macro_utils/dt_public_unscope.h"

using namespace std;
using namespace testing;
using namespace ge;
using domi::GetContext;

class UtestGraphCachingAllocatorTest : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() { GetContext().out_nodes_map.clear(); }
};

TEST_F(UtestGraphCachingAllocatorTest, initialize_success) {
  std::vector<rtMemType_t> mem_type;
  mem_type.push_back(RT_MEMORY_HBM);
  EXPECT_EQ(MemManager::Instance().Initialize(mem_type), SUCCESS);
  MemManager::Instance().Finalize();
}

TEST_F(UtestGraphCachingAllocatorTest, malloc_success) {
  std::vector<rtMemType_t> mem_type;
  mem_type.push_back(RT_MEMORY_HBM);
  EXPECT_EQ(MemManager::Instance().Initialize(mem_type), SUCCESS);
  uint8_t *ptr = MemManager::Instance().CachingInstance(RT_MEMORY_HBM).Malloc(kMByteSize);
  EXPECT_NE(nullptr, ptr);
  MemManager::Instance().Finalize();
}

TEST_F(UtestGraphCachingAllocatorTest, extend_malloc_success) {
  std::vector<rtMemType_t> mem_type;
  mem_type.push_back(RT_MEMORY_HBM);
  EXPECT_EQ(MemManager::Instance().Initialize(mem_type), SUCCESS);
  uint8_t *ptr = MemManager::Instance().CachingInstance(RT_MEMORY_HBM).Malloc(kMByteSize);
  EXPECT_NE(nullptr, ptr);
  ptr = MemManager::Instance().CachingInstance(RT_MEMORY_HBM).Malloc(kBinSizeUnit32*kMByteSize + kRoundBlockSize);
  EXPECT_NE(nullptr, ptr);
  MemManager::Instance().Finalize();
}

TEST_F(UtestGraphCachingAllocatorTest, malloc_same_success) {
  std::vector<rtMemType_t> mem_type;
  mem_type.push_back(RT_MEMORY_HBM);
  EXPECT_EQ(MemManager::Instance().Initialize(mem_type), SUCCESS);
  uint8_t *ptr = MemManager::Instance().CachingInstance(RT_MEMORY_HBM).Malloc(kBinSizeUnit8*kMByteSize + kRoundBlockSize);
  EXPECT_NE(nullptr, ptr);
  uint8_t *ptr1 = MemManager::Instance().CachingInstance(RT_MEMORY_HBM).Malloc(kBinSizeUnit8*kMByteSize + kRoundBlockSize);
  EXPECT_NE(nullptr, ptr1);
  uint8_t *ptr2 = MemManager::Instance().CachingInstance(RT_MEMORY_HBM).Malloc(kBinSizeUnit8*kMByteSize + kRoundBlockSize);
  EXPECT_NE(nullptr, ptr2);
  EXPECT_EQ(MemManager::Instance().CachingInstance(RT_MEMORY_HBM).Free(ptr), SUCCESS);
  EXPECT_EQ(MemManager::Instance().CachingInstance(RT_MEMORY_HBM).Free(ptr1), SUCCESS);
  EXPECT_EQ(MemManager::Instance().CachingInstance(RT_MEMORY_HBM).Free(ptr2), SUCCESS);
  ptr = MemManager::Instance().CachingInstance(RT_MEMORY_HBM).Malloc(kBinSizeUnit8*kMByteSize + kRoundBlockSize, ptr1);
  EXPECT_EQ(ptr, ptr1);
  MemManager::Instance().Finalize();
}

TEST_F(UtestGraphCachingAllocatorTest, malloc_statics) {
  std::vector<rtMemType_t> mem_type;
  mem_type.push_back(RT_MEMORY_HBM);
  EXPECT_EQ(MemManager::Instance().Initialize(mem_type), SUCCESS);
  uint8_t *ptr = MemManager::Instance().CachingInstance(RT_MEMORY_HBM).Malloc(kMByteSize);
  EXPECT_NE(nullptr, ptr);
  uint8_t *ptr1 = MemManager::Instance().CachingInstance(RT_MEMORY_HBM).Malloc(kKByteSize);
  EXPECT_NE(nullptr, ptr);
  EXPECT_EQ(MemManager::Instance().CachingInstance(RT_MEMORY_HBM).Free(ptr), SUCCESS);
  EXPECT_EQ(MemManager::Instance().CachingInstance(RT_MEMORY_HBM).Free(ptr1), SUCCESS);
  MemManager::Instance().CachingInstance(RT_MEMORY_HBM).FreeCachedBlocks();
  MemManager::Instance().Finalize();
}
