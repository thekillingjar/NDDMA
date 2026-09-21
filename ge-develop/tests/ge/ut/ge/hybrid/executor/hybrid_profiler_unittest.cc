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
#include <vector>
#include <iostream>

#include "macro_utils/dt_public_scope.h"
#include "hybrid/executor/hybrid_profiler.h"

using namespace std;
using namespace testing;

namespace ge {
using namespace hybrid;

class UtestHybridProfiling : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UtestHybridProfiling, test_basic_function) {
  HybridProfiler profiler;
  profiler.RecordEvent((HybridProfiler::EventType::GENERAL), "test");
  ASSERT_EQ(profiler.events_.empty(), false);
  ASSERT_EQ(profiler.events_.front().event_type, HybridProfiler::EventType::GENERAL);
  profiler.Dump(std::cout);
  ASSERT_EQ(profiler.counter_, 0);
}

}  // namespace ge
