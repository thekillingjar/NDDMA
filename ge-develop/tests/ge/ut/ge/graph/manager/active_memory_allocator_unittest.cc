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
#include "graph/manager/active_memory_allocator.h"
#include "macro_utils/dt_public_unscope.h"

using namespace std;
using namespace testing;
using namespace ge;
using domi::GetContext;

class UtestActiveMemoryAllocatorTest : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(UtestActiveMemoryAllocatorTest, RemoveAllocatorByDeviceId_Success) {
  auto allocator_origin = SessionMemAllocator<ExpandableActiveMemoryAllocator>::Instance().GetMemAllocator(1, 0);
  SessionMemAllocator<ExpandableActiveMemoryAllocator>::Instance().RemoveAllocator(1, 0);
  auto allocator_new = SessionMemAllocator<ExpandableActiveMemoryAllocator>::Instance().GetMemAllocator(1, 0);
  EXPECT_NE(allocator_new, allocator_origin);
}

TEST_F(UtestActiveMemoryAllocatorTest, RemoveAllocatorByDeviceId_DeviceAbsent) {
  auto allocator_origin = SessionMemAllocator<ExpandableActiveMemoryAllocator>::Instance().GetMemAllocator(1, 0);
  SessionMemAllocator<ExpandableActiveMemoryAllocator>::Instance().RemoveAllocator(1, 2);
  auto allocator_new = SessionMemAllocator<ExpandableActiveMemoryAllocator>::Instance().GetMemAllocator(1, 0);
  EXPECT_TRUE(allocator_new == allocator_origin);
}

TEST_F(UtestActiveMemoryAllocatorTest, RemoveAllocatorByDeviceId_SessionIdAbsent) {
  auto allocator_origin = SessionMemAllocator<ExpandableActiveMemoryAllocator>::Instance().GetMemAllocator(1, 0);
  SessionMemAllocator<ExpandableActiveMemoryAllocator>::Instance().RemoveAllocator(2, 2);
  auto allocator_new = SessionMemAllocator<ExpandableActiveMemoryAllocator>::Instance().GetMemAllocator(1, 0);
  EXPECT_TRUE(allocator_new == allocator_origin);
}

TEST_F(UtestActiveMemoryAllocatorTest, RemoveAllocatorBySessionId_Success) {
  auto allocator_origin = SessionMemAllocator<ExpandableActiveMemoryAllocator>::Instance().GetMemAllocator(1, 0);
  SessionMemAllocator<ExpandableActiveMemoryAllocator>::Instance().RemoveAllocator(1);
  auto allocator_new = SessionMemAllocator<ExpandableActiveMemoryAllocator>::Instance().GetMemAllocator(1, 0);
  EXPECT_NE(allocator_new, allocator_origin);
}

TEST_F(UtestActiveMemoryAllocatorTest, RemoveAllocatorBySessionId_SessionIdAbsent) {
  auto allocator_origin = SessionMemAllocator<ExpandableActiveMemoryAllocator>::Instance().GetMemAllocator(1, 0);
  SessionMemAllocator<ExpandableActiveMemoryAllocator>::Instance().RemoveAllocator(2);
  auto allocator_new = SessionMemAllocator<ExpandableActiveMemoryAllocator>::Instance().GetMemAllocator(1, 0);
  EXPECT_TRUE(allocator_new == allocator_origin);
}

TEST_F(UtestActiveMemoryAllocatorTest, FixedBaseExpandableAllocator_AllocFreeSuccess) {
  auto allocator = SessionMemAllocator<FixedBaseExpandableAllocator>::Instance().GetMemAllocator(1226, 0);
  auto block1 = allocator->Malloc(32 * 1024 * 1024);
  auto block2 = allocator->Malloc(50 * 1024 * 1024);
  auto block3 = allocator->Malloc(40 * 1024 * 1024);
  auto block4 = allocator->Malloc(60 * 1024 * 1024);
  ASSERT_NE(block1, nullptr);
  ASSERT_NE(block2, nullptr);
  ASSERT_NE(block3, nullptr);
  ASSERT_NE(block4, nullptr);
  allocator->Free(block1);
  allocator->Free(block2);
  allocator->Free(block3);
  allocator->Free(block4);
  SessionMemAllocator<ExpandableActiveMemoryAllocator>::Instance().RemoveAllocator(1226);
}
