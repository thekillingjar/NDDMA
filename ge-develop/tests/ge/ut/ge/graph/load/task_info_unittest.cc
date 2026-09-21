/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/load/model_manager/task_info/task_info.h"
#include <gtest/gtest.h>
namespace ge {
class TaskInfoUT : public testing::Test {};
TEST_F(TaskInfoUT, GetArgsPlacementStr_ReturnUnknown_OutOfRange) {
  ASSERT_STREQ(GetArgsPlacementStr(ArgsPlacement::kEnd), "unknown");
  ASSERT_STREQ(GetArgsPlacementStr(static_cast<ArgsPlacement>(100)), "unknown");
}
}  // namespace ge
