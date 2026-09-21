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
#include <gmock/gmock.h>
#include "graph/load/model_manager/memory_block_manager.h"

#include "stub/gert_runtime_stub.h"
#include "runtime_stub.h"

using namespace std;

namespace ge {
class UtestMemoryBlockManager : public testing::Test {};

TEST_F(UtestMemoryBlockManager, MallocSuccess) {
  class MockRuntime : public ge::RuntimeStub {
   public:
    rtError_t rtMalloc(void **dev_ptr, uint64_t size, rtMemType_t type, uint16_t moduleId) override {
      size_local += size;
      alloc_count++;
      *dev_ptr = new uint8_t[size];
      memset_s(*dev_ptr, size, 0, size);
      return RT_ERROR_NONE;
    }

    rtError_t rtFree(void *dev_ptr) override {
      free_count++;
      delete[](uint8_t *) dev_ptr;
      return RT_ERROR_NONE;
    }

    uint32_t alloc_count = 0U;
    uint32_t size_local = 0U;
    uint32_t free_count = 0U;
  };
  auto mock_runtime = std::make_shared<MockRuntime>();
  ge::RuntimeStub::SetInstance(mock_runtime);

  MemoryBlockManager manager(RT_MEMORY_HBM);
  void *last_ptr = nullptr;
  for (size_t i = 0U; i < 2 * 1024 * 1024 / 512; ++i) {
    auto ptr = manager.Malloc("", 1);
    ASSERT_NE(ptr, nullptr);
    EXPECT_EQ(mock_runtime->alloc_count, 1);

    if (last_ptr != nullptr) {
      EXPECT_EQ(PtrToValue(ptr)- PtrToValue(last_ptr), 512);
    }
    last_ptr = ptr;
  }
  auto ptr = manager.Malloc("", 1);
  ASSERT_NE(ptr, nullptr);
  EXPECT_EQ(mock_runtime->alloc_count, 2);

  manager.Release();
  EXPECT_EQ(mock_runtime->free_count, 2);

  ptr = manager.Malloc("", 2 * 1024 * 1024 + 1);
  ASSERT_NE(ptr, nullptr);
  EXPECT_EQ(mock_runtime->alloc_count, 3);

  ptr = manager.Malloc("", 1025);
  ASSERT_NE(ptr, nullptr);
  EXPECT_EQ(mock_runtime->alloc_count, 3);
  manager.Release();
  EXPECT_EQ(mock_runtime->free_count, 3);

  ptr = manager.Malloc("", 1025);
  ASSERT_NE(ptr, nullptr);
  EXPECT_EQ(mock_runtime->alloc_count, 4);
  manager.Release();
  EXPECT_EQ(mock_runtime->free_count, 4);
  ge::RuntimeStub::Reset();
}
}  // namespace ge
