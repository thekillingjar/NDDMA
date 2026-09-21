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
#include "graph/manager/host_mem_allocator.h"
#include "macro_utils/dt_public_unscope.h"

namespace ge {
class UtestHostMemManagerTest : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UtestHostMemManagerTest, malloc_zero_size) {
  HostMemAllocator allocator(RT_MEMORY_HBM);
  EXPECT_EQ(allocator.allocated_blocks_.size(), 0);
  EXPECT_EQ(allocator.Malloc(nullptr, 0), nullptr);
  EXPECT_EQ(allocator.allocated_blocks_.size(), 1);
  EXPECT_EQ(allocator.Malloc(nullptr, 1), nullptr);
  EXPECT_EQ(allocator.allocated_blocks_.size(), 1);
}
} // namespace ge
