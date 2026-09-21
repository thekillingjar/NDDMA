/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "exe_graph/runtime/execute_graph_types.h"
#include <gtest/gtest.h>
namespace gert {
class ExecuteGraphTypesUT : public testing::Test {};
TEST_F(ExecuteGraphTypesUT, GetStr_Ok) {
  EXPECT_STREQ(GetExecuteGraphTypeStr(ExecuteGraphType::kInit), "Init");
  EXPECT_STREQ(GetExecuteGraphTypeStr(ExecuteGraphType::kMain), "Main");
  EXPECT_STREQ(GetExecuteGraphTypeStr(ExecuteGraphType::kDeInit), "DeInit");
}
TEST_F(ExecuteGraphTypesUT, GetStr_Nullptr_OutOfRange) {
  EXPECT_EQ(GetExecuteGraphTypeStr(ExecuteGraphType::kNum), nullptr);
}
}  // namespace gert
