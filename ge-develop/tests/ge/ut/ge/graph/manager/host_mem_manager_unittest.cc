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
#include "graph/manager/host_mem_manager.h"
#include "macro_utils/dt_public_unscope.h"

namespace ge {
class UtestSharedMemAllocatorTest : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UtestSharedMemAllocatorTest, malloc_zero_size) {
  SharedMemAllocator allocator;
  string var_name = "host_params";
  SharedMemInfo info;
  uint8_t tmp(0);
  info.device_address = &tmp;

  std::shared_ptr<AlignedPtr> aligned_ptr = std::make_shared<AlignedPtr>(100, 16);

  info.host_aligned_ptr = aligned_ptr;
  info.fd=0;
  info.mem_size = 100;
  info.op_name = var_name;
  EXPECT_EQ(allocator.Allocate(info), SUCCESS);
  EXPECT_EQ(allocator.DeAllocate(info), SUCCESS);
}
} // namespace ge
