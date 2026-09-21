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
#include "omg/omg_inner_types.h"

#include "macro_utils/dt_public_scope.h"
#include "graph/manager/mem_manager.h"

using namespace std;
using namespace testing;
using namespace ge;
using domi::GetContext;

class UtestSessionScopeMemAllocator : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() { GetContext().out_nodes_map.clear(); }
};

TEST_F(UtestSessionScopeMemAllocator, initialize_success) {
  std::vector<rtMemType_t> mem_type;
  mem_type.push_back(RT_MEMORY_HBM);
  EXPECT_EQ(MemManager::Instance().Initialize(mem_type), SUCCESS);
  MemManager::Instance().Finalize();
}

TEST_F(UtestSessionScopeMemAllocator, malloc_success) {
  std::vector<rtMemType_t> mem_type;
  mem_type.push_back(RT_MEMORY_HBM);
  EXPECT_EQ(MemManager::Instance().Initialize(mem_type), SUCCESS);
  uint8_t *ptr = MemManager::Instance().SessionScopeMemInstance(RT_MEMORY_HBM).Malloc(1000, 0);
  EXPECT_NE(nullptr, ptr);
  MemManager::Instance().Finalize();
}

TEST_F(UtestSessionScopeMemAllocator, free_success) {
  std::vector<rtMemType_t> mem_type;
  mem_type.push_back(RT_MEMORY_HBM);
  EXPECT_EQ(MemManager::Instance().Initialize(mem_type), SUCCESS);
  uint8_t *ptr = MemManager::Instance().SessionScopeMemInstance(RT_MEMORY_HBM).Malloc(100, 0);
  EXPECT_NE(nullptr, ptr);
  ptr = MemManager::Instance().SessionScopeMemInstance(RT_MEMORY_HBM).Malloc(100, 0);
  EXPECT_NE(nullptr, ptr);

  EXPECT_EQ(SUCCESS, MemManager::Instance().SessionScopeMemInstance(RT_MEMORY_HBM).Free(0));
  EXPECT_NE(SUCCESS, MemManager::Instance().SessionScopeMemInstance(RT_MEMORY_HBM).Free(0));
  MemManager::Instance().Finalize();
}

TEST_F(UtestSessionScopeMemAllocator, free_success_session) {
  std::vector<rtMemType_t> mem_type;
  mem_type.push_back(RT_MEMORY_HBM);
  mem_type.push_back(RT_MEMORY_P2P_DDR);
  EXPECT_EQ(MemManager::Instance().Initialize(mem_type), SUCCESS);
  uint8_t *ptr = MemManager::Instance().SessionScopeMemInstance(RT_MEMORY_HBM).Malloc(100, 0);
  EXPECT_NE(nullptr, ptr);
  ptr = MemManager::Instance().SessionScopeMemInstance(RT_MEMORY_HBM).Malloc(100, 0);
  EXPECT_NE(nullptr, ptr);
  for (auto memory_type : MemManager::Instance().memory_type_) {
    if (RT_MEMORY_P2P_DDR == memory_type) {
      EXPECT_NE(MemManager::Instance().SessionScopeMemInstance(memory_type).Free(0), SUCCESS);
    } else {
      EXPECT_EQ(MemManager::Instance().SessionScopeMemInstance(memory_type).Free(0), SUCCESS);
    }
  }
  MemManager::Instance().Finalize();
}
