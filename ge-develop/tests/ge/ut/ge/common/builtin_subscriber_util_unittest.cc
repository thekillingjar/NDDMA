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
#include "runtime/subscriber/built_in_subscriber_definitions.h"
namespace gert {
class BuiltInSubscriberUtilUT : public testing::Test {};
TEST_F(BuiltInSubscriberUtilUT, BuildProfilingEnableFlagsOk) {
  EXPECT_TRUE(BuiltInSubscriberUtil::BuildEnableFlags<ProfilingType>({ProfilingType::kGeHost}) &
              BuiltInSubscriberUtil::EnableBit<ProfilingType>(ProfilingType::kGeHost));

  EXPECT_FALSE(BuiltInSubscriberUtil::BuildEnableFlags<ProfilingType>({ProfilingType::kDevice}) &
               BuiltInSubscriberUtil::EnableBit<ProfilingType>(ProfilingType::kGeHost));

  EXPECT_TRUE(BuiltInSubscriberUtil::BuildEnableFlags<ProfilingType>({ProfilingType::kAll}) &
              BuiltInSubscriberUtil::EnableBit<ProfilingType>(ProfilingType::kGeHost));

  EXPECT_TRUE(BuiltInSubscriberUtil::BuildEnableFlags<ProfilingType>({ProfilingType::kAll}) &
              BuiltInSubscriberUtil::EnableBit<ProfilingType>(ProfilingType::kDevice));

  EXPECT_TRUE(BuiltInSubscriberUtil::BuildEnableFlags<ProfilingType>({ProfilingType::kAll}) &
              BuiltInSubscriberUtil::EnableBit<ProfilingType>(ProfilingType::kHost));

  EXPECT_TRUE(BuiltInSubscriberUtil::BuildEnableFlags<ProfilingType>({ProfilingType::kAll}) &
              BuiltInSubscriberUtil::EnableBit<ProfilingType>(ProfilingType::kSingleOpReg));
}
}  // namespace gert
