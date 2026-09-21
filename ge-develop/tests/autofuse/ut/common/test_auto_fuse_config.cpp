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
#define private public
#include "autofuse_config/auto_fuse_config.h"
#undef private
#include "autofuse_config/auto_fuse_config_utils.h"
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
  setenv("AUTOFUSE_DFX_FLAGS", "--autofuse_att_algorithm=HighPerf", 1);
  ASSERT_EQ(att::AutoFuseConfig::Instance().att_strategy_config_.Init(), SUCCESS);
  EXPECT_EQ(att::AutoFuseConfig::Instance().att_strategy_config_.tiling_algorithm, "HighPerf");
  unsetenv("AUTOFUSE_DFX_FLAGS");
}

TEST_F(AutoFuseConfigTest, SetTilingAlgorithmDefault) {
  setenv("AUTOFUSE_DFX_FLAGS", "--autofuse_att_algorithm=HighPerf1", 1);
  ASSERT_EQ(att::AutoFuseConfig::Instance().att_strategy_config_.Init(), SUCCESS);
  EXPECT_EQ(att::AutoFuseConfig::Instance().att_strategy_config_.tiling_algorithm, "AxesReorder");
  unsetenv("AUTOFUSE_DFX_FLAGS");
}

TEST_F(AutoFuseConfigTest, SetSolutionAccuracyLevelValid) {
  setenv("AUTOFUSE_DFX_FLAGS", "--att_accuracy_level=0", 1);
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
  // default is 0.4
  EXPECT_EQ(att::AutoFuseConfig::Instance().att_strategy_config_.corenum_threshold, 40);
  unsetenv("AUTOFUSE_DFX_FLAGS");
}

TEST_F(AutoFuseConfigTest, SetCorenumThresholdInvalidCharNumDefault) {
  setenv("AUTOFUSE_DFX_FLAGS", "--att_corenum_threshold=true111", 1);
  ASSERT_EQ(att::AutoFuseConfig::Instance().att_strategy_config_.Init(), SUCCESS);
  // default is 0.4
  EXPECT_EQ(att::AutoFuseConfig::Instance().att_strategy_config_.corenum_threshold, 40);
  unsetenv("AUTOFUSE_DFX_FLAGS");
}

TEST_F(AutoFuseConfigTest, ParseForceTilingCase) {
  std::vector<std::string> test_cases = {
      "g0_0_R,g1_0_R,g2_1",  // 有效分组格式 -> Group: {0:0, 1:0, 2:1}
      "2_R",                // 有效单一格式 -> Single: 2
      "g0_0",             // 单组有效格式 -> Group: {0:0}
      "g10_25_R,g3_0",      // 多位数有效 -> Group: {10:25, 3:0}
      "2",                    // 有效单一格式，无sub_tag
      "",                 // 空字符串 -> -1 (错误)
      "g0",               // 缺少下划线和case -> -1 (错误)
      "g_1",              // 缺少组号 -> -1 (错误)
      "g0_",              // 缺少case号 -> -1 (错误)
      "g0_a",             // 非数字case -> -1 (错误)
      "a0_0",             // 首字母不是g -> -1 (错误)
      "0_0",              // 缺少g前缀 -> -1 (错误)
      "g0_0,g0_1",       // 重复组号 -> -1 (错误)
      "2a"               // 含非数字字符 -> -1 (错误)
  };
  std::vector<std::pair<bool, ge::ForceTilingCaseResult>> expect_results = {
    {true, {false, -1, "", {{0, {0, "R"}}, {1, {0, "R"}}, {2, {1, ""}}}}},
    {true, {true, 2, "R"}},
    {true, {false, -1, "", {{0, {0, ""}}}}},
    {true, {false, -1, "", {{10, {25, "R"}}, {3, {0, ""}}}}},
    {true, {true, 2, ""}},
    {false, {}},
    {false, {}},
    {false, {}},
    {false, {}},
    {false, {}},
    {false, {}},
    {false, {true, 0, ""}},
    {false, {}},
    {false, {}},
  };
  ASSERT_EQ(expect_results.size(), test_cases.size());
  for (int32_t id = 0; id < static_cast<int32_t>(test_cases.size()); id++) {
    ge::ForceTilingCaseResult result;
    auto ret = ge::AttStrategyConfigUtils::ParseForceTilingCase(test_cases[id], result);
    EXPECT_EQ(ret == ge::SUCCESS, expect_results[id].first) << "id: " << id;
    if (ret == ge::SUCCESS) {
      EXPECT_EQ(result.Debug(), expect_results[id].second.Debug()) << "id: " << id;
    } else {
      EXPECT_EQ(result.Debug(), expect_results[id].second.Debug()) << "id: " << id;
    }
    if (id == 0) {
      EXPECT_EQ(result.GetTag(0), "R");
      EXPECT_EQ(result.GetTag(1), "R");
      EXPECT_EQ(result.GetTag(2), "");
      EXPECT_EQ(result.GetTag(3), "");
      EXPECT_EQ(result.GetCase(3).first, -1);
    }
  }
}

TEST_F(AutoFuseConfigTest, ClearForceTilingCase) {
  std::vector<ge::ForceTilingCaseResult> test_cases = {
      {false, -1, "", {{0, {0, "R"}}, {1, {0, "R"}}, {2, {1, ""}}}},
      {true, 2, "R"},
      {false, -1, {{0, 0}}},
  };
  ge::ForceTilingCaseResult expect_result;
  for (auto &test_case : test_cases) {
    test_case.Clear();
    EXPECT_EQ(test_case.Debug(), expect_result.Debug());
  }
}

TEST_F(AutoFuseConfigTest, GetForceGroupTilingCase) {
  ge::ForceTilingCaseResult test_case = {
      false,
      -1, "",
      {{0, {0, ""}}, {1, {0, ""}}, {2, {1, ""}}},
  };
  auto group_tiling_case = test_case.GetCase(0);
  std::pair<int32_t, std::string> expect0({0, ""});
  EXPECT_EQ(group_tiling_case, expect0);

  std::pair<int32_t, std::string> expect1({0, ""});
  group_tiling_case = test_case.GetCase(1);
  EXPECT_EQ(group_tiling_case, expect1);

  std::pair<int32_t, std::string> expect2({1, ""});
  group_tiling_case = test_case.GetCase(2);
  EXPECT_EQ(group_tiling_case, expect2);
}

// Test for out_of_range exception in ParseInt64Config when value exceeds int64_t range
TEST_F(AutoFuseConfigTest, SetSolutionAccuracyLevelOutOfRangePositive) {
  // int64_t max is 9223372036854775807, this value exceeds it
  setenv("AUTOFUSE_DFX_FLAGS", "--att_accuracy_level=9999999999999999999999999999999999999999999999999999999999999999", 1);
  ASSERT_EQ(att::AutoFuseConfig::Instance().att_strategy_config_.Init(), SUCCESS);
  // default is 1
  EXPECT_EQ(att::AutoFuseConfig::Instance().att_strategy_config_.solution_accuracy_level, 1);
  unsetenv("AUTOFUSE_DFX_FLAGS");
}

// Test for out_of_range exception in ParseInt64Config when value is below int64_t range
TEST_F(AutoFuseConfigTest, SetSolutionAccuracyLevelOutOfRangeNegative) {
  // int64_t min is -9223372036854775808, this value is below it
  setenv("AUTOFUSE_DFX_FLAGS", "--att_accuracy_level=-9999999999999999999999999999999999999999999999999999999999999999", 1);
  ASSERT_EQ(att::AutoFuseConfig::Instance().att_strategy_config_.Init(), SUCCESS);
  // default is 1
  EXPECT_EQ(att::AutoFuseConfig::Instance().att_strategy_config_.solution_accuracy_level, 1);
  unsetenv("AUTOFUSE_DFX_FLAGS");
}

// Test for out_of_range exception in ParseInt64Config for ub_threshold
TEST_F(AutoFuseConfigTest, SetUbThresholdOutOfRange) {
  setenv("AUTOFUSE_DFX_FLAGS", "--att_ub_threshold=9999999999999999999999999999999999999999999999999999999999999999", 1);
  ASSERT_EQ(att::AutoFuseConfig::Instance().att_strategy_config_.Init(), SUCCESS);
  // default is 20
  EXPECT_EQ(att::AutoFuseConfig::Instance().att_strategy_config_.ub_threshold, 20);
  unsetenv("AUTOFUSE_DFX_FLAGS");
}

// Test for out_of_range exception in ParseInt64Config for corenum_threshold
TEST_F(AutoFuseConfigTest, SetCorenumThresholdOutOfRange) {
  setenv("AUTOFUSE_DFX_FLAGS", "--att_corenum_threshold=9999999999999999999999999999999999999999999999999999999999999999", 1);
  ASSERT_EQ(att::AutoFuseConfig::Instance().att_strategy_config_.Init(), SUCCESS);
  // default is 40
  EXPECT_EQ(att::AutoFuseConfig::Instance().att_strategy_config_.corenum_threshold, 40);
  unsetenv("AUTOFUSE_DFX_FLAGS");
}
} //namespace
