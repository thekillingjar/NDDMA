/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/preload/model/pre_model_partition_utils.h"
#include <gtest/gtest.h>

namespace ge {
class PreModelPartitionUtilsUnittest : public testing::Test {};

TEST_F(PreModelPartitionUtilsUnittest, CheckNanoPartitionType) {
  EXPECT_NE(PreModelPartitionUtils::GetInstance().CheckNanoPartitionType(MAX_TASK), SUCCESS);
}

TEST_F(PreModelPartitionUtilsUnittest, PreparePartitionData) {
  PreModelPartitionUtils::GetInstance().Reset();
  EXPECT_NE(PreModelPartitionUtils::GetInstance().PreparePartitionData(EngineType::kNanoEngine), SUCCESS);
}

}
