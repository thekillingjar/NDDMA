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
#include "exe_graph/runtime/atomic_clean_tiling_context.h"

namespace gert {
namespace {
constexpr const size_t kKernelRunContextSize = 48U;
}  // namespace

class AbiCompatibilityForExeGraphUT : public testing::Test {};

TEST_F(AbiCompatibilityForExeGraphUT, AtomicCleanTilingContext_CheckMemLayoutNotChanged) {
  AtomicCleanTilingContext c;
  ASSERT_EQ(sizeof(AtomicCleanTilingContext), kKernelRunContextSize);
  ASSERT_EQ(static_cast<void *>(&c), static_cast<void *>(&c.context_));
  EXPECT_EQ(sizeof(c.context_), kKernelRunContextSize);
}
}  // namespace gert
