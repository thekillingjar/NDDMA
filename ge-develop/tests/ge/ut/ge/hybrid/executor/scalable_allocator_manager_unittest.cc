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

#include "hybrid/executor/runtime_v2/scalable_allocator_manager.h"

namespace ge {
class UtestScalabelAllocatorManager : public testing::Test {
 public:
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(UtestScalabelAllocatorManager, GetAllocator_Success) {
  ScalableAllocatorManager mng;
  auto allocators = mng.GetAllocator("some_graph", 0);
  EXPECT_NE(allocators->GetAllocator(gert::kOnDeviceHbm,
                                     static_cast<size_t>(gert::AllocatorUsage::kAllocNodeOutput)), nullptr);

  allocators = mng.GetAllocator("some_graph", 0);
  for (size_t i = 0U; i < static_cast<size_t>(gert::kTensorPlacementEnd); ++i) {
    for (size_t j = 0U; j < static_cast<size_t>(gert::AllocatorUsage::kAllocNodeOutput); ++j) {
      auto allocator = allocators->GetAllocator(gert::kOnDeviceHbm,
                                                static_cast<size_t>(gert::AllocatorUsage::kAllocNodeOutput));
      EXPECT_NE(allocator, nullptr);
    }
  }
}
}
