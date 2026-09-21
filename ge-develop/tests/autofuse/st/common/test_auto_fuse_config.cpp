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
#include <string>
#include <fstream>
#define private public
#include "autofuse_config/auto_fuse_config.h"
using namespace ge;
namespace att
{
class AutoFuseConfigTest: public testing::Test {
 public:
  void SetUp() override {
    att::AutoFuseConfig::MutableAttStrategyConfig().Reset();
  }
  void TearDown() override {
  }
};

TEST_F(AutoFuseConfigTest, SetTilingAlgorithmToAxesReorder) {
  setenv("AUTOFUSE_DFX_FLAGS", "--autofuse_att_algorithm=AxesReorder", 1);
  ASSERT_EQ(att::AutoFuseConfig::Instance().att_strategy_config_.Init(), SUCCESS);
  EXPECT_EQ(att::AutoFuseConfig::Instance().att_strategy_config_.tiling_algorithm, "AxesReorder");
  unsetenv("AUTOFUSE_DFX_FLAGS");
}

TEST_F(AutoFuseConfigTest, SetTilingAlgorithmDefault) {
  setenv("AUTOFUSE_DFX_FLAGS", "--autofuse_att_algorithm=HighPerf1", 1);
  ASSERT_EQ(att::AutoFuseConfig::Instance().att_strategy_config_.Init(), SUCCESS);
  EXPECT_EQ(att::AutoFuseConfig::Instance().att_strategy_config_.tiling_algorithm, "AxesReorder");
  unsetenv("AUTOFUSE_DFX_FLAGS");
}

TEST_F(AutoFuseConfigTest, SetSolutionAccuracyLevelValid) {
  setenv("AUTOFUSE_DFX_FLAGS", "--att_accuracy_level=0", 0);
  ASSERT_EQ(att::AutoFuseConfig::Instance().att_strategy_config_.Init(), SUCCESS);
  EXPECT_EQ(att::AutoFuseConfig::Instance().att_strategy_config_.solution_accuracy_level, 0);
  unsetenv("AUTOFUSE_DFX_FLAGS");
}

TEST_F(AutoFuseConfigTest, SetSolutionAccuracyLevelInvalidDefault) {
  setenv("AUTOFUSE_DFX_FLAGS", "--att_accuracy_level=-1000", 1);
  ASSERT_EQ(att::AutoFuseConfig::Instance().att_strategy_config_.Init(), SUCCESS);
  // default is 100
  EXPECT_EQ(att::AutoFuseConfig::Instance().att_strategy_config_.solution_accuracy_level, 1);
  unsetenv("AUTOFUSE_DFX_FLAGS");
}

TEST_F(AutoFuseConfigTest, SetUbThresholdValid) {
  setenv("AUTOFUSE_DFX_FLAGS", "--att_ub_threshold=50", 1);
  ASSERT_EQ(att::AutoFuseConfig::Instance().att_strategy_config_.Init(), SUCCESS);
  constexpr double kExpectUbThreshold = 50;
  EXPECT_EQ(att::AutoFuseConfig::Instance().att_strategy_config_.ub_threshold, kExpectUbThreshold);
  unsetenv("AUTOFUSE_DFX_FLAGS");
}

TEST_F(AutoFuseConfigTest, SetUbThresholdInvalidDefault) {
  setenv("AUTOFUSE_DFX_FLAGS", "--att_ub_threshold=150", 1);
  ASSERT_EQ(att::AutoFuseConfig::Instance().att_strategy_config_.Init(), SUCCESS);
  // default is 0.2
  EXPECT_EQ(att::AutoFuseConfig::Instance().att_strategy_config_.ub_threshold, 20);
  unsetenv("AUTOFUSE_DFX_FLAGS");
}
TEST_F(AutoFuseConfigTest, SetCorenumThresholdValid) {
  setenv("AUTOFUSE_DFX_FLAGS", "--att_corenum_threshold=50", 1);
  ASSERT_EQ(att::AutoFuseConfig::Instance().att_strategy_config_.Init(), SUCCESS);
  constexpr double kExpectCorenumThreshold = 50;
  EXPECT_EQ(att::AutoFuseConfig::Instance().att_strategy_config_.corenum_threshold, kExpectCorenumThreshold);
  unsetenv("AUTOFUSE_DFX_FLAGS");
}
TEST_F(AutoFuseConfigTest, SetCorenumThresholdInvalidDefault) {
  setenv("AUTOFUSE_DFX_FLAGS", "--att_corenum_threshold=150", 1);
  ASSERT_EQ(att::AutoFuseConfig::Instance().att_strategy_config_.Init(), SUCCESS);
  // default is 0.8
  EXPECT_EQ(att::AutoFuseConfig::Instance().att_strategy_config_.corenum_threshold, 40);
  unsetenv("AUTOFUSE_DFX_FLAGS");
}
} //namespace