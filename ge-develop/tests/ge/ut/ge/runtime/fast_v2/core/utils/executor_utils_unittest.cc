/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "core/utils/executor_utils.h"
#include <gtest/gtest.h>
namespace gert {
class ExecutorUtilsUT : public testing::Test {};
TEST_F(ExecutorUtilsUT, CalcIndexes_Ok1) {
  ASSERT_EQ(CalcArgIndex(6U, ExecuteArgIndex::kStream), 2);
  ASSERT_EQ(CalcArgIndex(6U, ExecuteArgIndex::kExternalAllocator), 3);
  ASSERT_EQ(CalcArgIndex(6U, ExecuteArgIndex::kRtEvents), 4);
}
TEST_F(ExecutorUtilsUT, CalcIndexes_Ok2) {
  ASSERT_EQ(CalcArgIndex(4U, ExecuteArgIndex::kStream), 0);
  ASSERT_EQ(CalcArgIndex(4U, ExecuteArgIndex::kExternalAllocator), 1);
  ASSERT_EQ(CalcArgIndex(4U, ExecuteArgIndex::kRtEvents), 2);
}
}  // namespace gert