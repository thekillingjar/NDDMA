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

#include "runtime/rt.h"
#include "framework/executor/ge_executor.h"

#include "macro_utils/dt_public_scope.h"
#include "single_op/single_op_manager.h"
#include "hybrid/common/npu_memory_allocator.h"
#include "macro_utils/dt_public_unscope.h"

using namespace std;
using namespace testing;
using namespace ge;

class UtestSingleOpManager : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(UtestSingleOpManager, test_get_resource) {
  rtStream_t stream = (rtStream_t)0x01;
  auto &instance = SingleOpManager::GetInstance();
  ASSERT_NE(instance.GetResource(0x01, stream), nullptr);
}

TEST_F(UtestSingleOpManager, test_relesase_resource) {
  auto stream = (rtStream_t)0x99;
  auto &instance = SingleOpManager::GetInstance();

  auto allocator = hybrid::NpuMemoryAllocator::GetAllocator(stream);
  ASSERT_NE(allocator, nullptr);
  ASSERT_EQ(instance.ReleaseResource(stream), SUCCESS);
  instance.GetResource(0x99, stream);
  ASSERT_EQ(instance.ReleaseResource(stream), SUCCESS);
}

TEST_F(UtestSingleOpManager, test_delete_resource) {
  int32_t fake_stream = 0;
  uint64_t op_id = 999;
  auto stream = reinterpret_cast<rtStream_t>(fake_stream);
  auto &instance = SingleOpManager::GetInstance();
  uintptr_t resource_id = 0U;
  ASSERT_EQ(instance.GetResourceId(stream, resource_id), SUCCESS);
  auto res = instance.GetResource(resource_id, stream);
  ASSERT_NE(res, nullptr);
  auto new_op = MakeUnique<SingleOp>(res, &res->stream_mu_, res->stream_);
  res->op_map_[op_id] = std::move(new_op);
  auto new_dynamic_op = MakeUnique<DynamicSingleOp>(&res->tensor_pool_, res->resource_id_, &res->stream_mu_, res->stream_);
  res->dynamic_op_map_[op_id] = std::move(new_dynamic_op);
  ASSERT_EQ(ge::GeExecutor::UnloadSingleOp(op_id), SUCCESS);
  ASSERT_EQ(ge::GeExecutor::UnloadDynamicSingleOp(op_id), SUCCESS);
  ASSERT_EQ(instance.ReleaseResource(stream), SUCCESS);
}

TEST_F(UtestSingleOpManager, SingleOp_SetAllocator) {
  int32_t fake_stream = 0;
  auto stream = reinterpret_cast<rtStream_t>(fake_stream);
  auto &instance = SingleOpManager::GetInstance();
  uintptr_t resource_id = 0U;
  ASSERT_EQ(instance.GetResourceId(stream, resource_id), SUCCESS);
  auto res = instance.GetResource(resource_id, stream);
  ASSERT_NE(res, nullptr);
  SingleOpManager::GetInstance().SetAllocator(stream, (ge::Allocator *)0x01);
  EXPECT_NE(res->allocator_, nullptr);
}
