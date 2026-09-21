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
#include <memory>

#include "ge/fusion/pattern_matcher_config.h"

namespace ge {
namespace fusion {
class UtestPatternMatcherConfig : public testing::Test {
 protected:
  void SetUp() {
  }

  void TearDown() {
  }
};
TEST_F(UtestPatternMatcherConfig, CreatePatternMatcherConfig_EnableConstMatch) {
  auto pattern_matcher_config = PatternMatcherConfigBuilder().EnableConstValueMatch().Build();
  EXPECT_NE(pattern_matcher_config, nullptr);

  EXPECT_TRUE(pattern_matcher_config->IsEnableConstValueMatch());
  EXPECT_FALSE(pattern_matcher_config->IsEnableIrAttrMatch());
}

TEST_F(UtestPatternMatcherConfig, CreatePatternMatcherConfig_EnableIrAttrMatch) {
  auto pattern_matcher_config = PatternMatcherConfigBuilder().EnableIrAttrMatch().Build();
  EXPECT_NE(pattern_matcher_config, nullptr);

  EXPECT_FALSE(pattern_matcher_config->IsEnableConstValueMatch());
  EXPECT_TRUE(pattern_matcher_config->IsEnableIrAttrMatch());
}

TEST_F(UtestPatternMatcherConfig, CreatePatternMatcherConfig_EnableIrAttrMatch_EnableConstMatch) {
  auto pattern_matcher_config = PatternMatcherConfigBuilder().EnableIrAttrMatch().EnableConstValueMatch().Build();
  EXPECT_NE(pattern_matcher_config, nullptr);

  EXPECT_TRUE(pattern_matcher_config->IsEnableConstValueMatch());
  EXPECT_TRUE(pattern_matcher_config->IsEnableIrAttrMatch());
}
} // namespace fusion
} // namespace ge

