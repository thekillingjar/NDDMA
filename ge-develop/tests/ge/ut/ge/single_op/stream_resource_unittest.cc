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
#include <vector>
#include <mmpa/mmpa_api.h>
#include "runtime/rt.h"

#include "macro_utils/dt_public_scope.h"
#include "single_op/stream_resource.h"
#include "macro_utils/dt_public_unscope.h"
#include "framework/ge_runtime_stub/include/faker/fake_allocator.h"

using namespace std;
using namespace testing;
using namespace ge;

class UtestStreamResource : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}

  rtStream_t stream;
};

TEST_F(UtestStreamResource, test_malloc_memory) {
  StreamResource res((uintptr_t)1);
  res.allocator_ = &res.internal_allocator_;
  string purpose("test");
  ASSERT_NE(res.MallocMemory(purpose, 100, false), nullptr);
  ASSERT_NE(res.MallocMemory(purpose, 100), nullptr);
  ASSERT_NE(res.MallocMemory(purpose, 100, false), nullptr);
}

TEST_F(UtestStreamResource, test_build_op) {
  StreamResource res((uintptr_t)1);
  ModelData model_data;
  SingleOp *single_op = nullptr;
  DynamicSingleOp *dynamic_single_op = nullptr;
  res.op_map_[0].reset(single_op);
  res.dynamic_op_map_[1].reset(dynamic_single_op);

  hybrid::CallbackManager *callback_manager = nullptr;
  EXPECT_EQ(res.GetCallbackManager(&callback_manager), SUCCESS);

  ThreadPool *thread_pool = nullptr;
  EXPECT_EQ(res.GetThreadPool(&thread_pool), SUCCESS);

  EXPECT_EQ(res.GetOperator(0), nullptr);
  EXPECT_EQ(res.GetDynamicOperator(1), nullptr);
  EXPECT_EQ(res.BuildOperator(model_data, &single_op, 0), SUCCESS);
  EXPECT_EQ(res.BuildDynamicOperator(model_data, &dynamic_single_op, 1), SUCCESS);
}

TEST_F(UtestStreamResource, test_malloc_overflow) {
  StreamResource res((uintptr_t)1);
  EXPECT_EQ(res.InitOverflowMemory(), SUCCESS);
}

TEST_F(UtestStreamResource, MallocMemory_Ok_WithoutExternalAllocator) {
  StreamResource res((uintptr_t)1);
  res.allocator_ = &res.internal_allocator_;
  auto mem = (ge::MemBlock *)0x01;
  EXPECT_NE(res.MallocMemory("malloc mem", 10, false, mem), nullptr);
  EXPECT_NE(mem,  (ge::MemBlock *)0x01);
}

TEST_F(UtestStreamResource, MallocMemory_NoReuseAllocated_WithoutExternalAllocator) {
  StreamResource res((uintptr_t)1);
  res.allocator_ = &res.internal_allocator_;
  auto mem = (ge::MemBlock *)0x01;
  EXPECT_NE(res.MallocMemory("malloc mem", 10, false, mem), nullptr);
  EXPECT_NE(mem,  (ge::MemBlock *)0x01);
  EXPECT_EQ(res.internal_allocator_.memory_list_.size(), 1);
  auto last_alloc = mem;
  EXPECT_NE(res.MallocMemory("malloc mem", 20, false, mem), nullptr);
  EXPECT_EQ(res.internal_allocator_.memory_list_.size(), 1);
  EXPECT_NE(last_alloc, mem);
}

TEST_F(UtestStreamResource, MallocMemory_MockRtMallocFail_WithoutExternalAllocator) {
  StreamResource res((uintptr_t)1);
  res.allocator_ = &res.internal_allocator_;
  auto mem = (ge::MemBlock *)0x01;
  mmSetEnv("CONSTANT_FOLDING_PASS_2", "mock_fail", 1);
  EXPECT_EQ(res.MallocMemory("malloc mem", 10, false, mem), nullptr);
  unsetenv("CONSTANT_FOLDING_PASS_2");
}

TEST_F(UtestStreamResource, MallocMemory_MockRtMemsetFail_WithoutExternalAllocator) {
  StreamResource res((uintptr_t)1);
  res.allocator_ = &res.internal_allocator_;
  auto mem = (ge::MemBlock *)0x01;
  EXPECT_EQ(res.MallocMemory("malloc mem", 321, false, mem), nullptr);
}
