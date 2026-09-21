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

#include "register/optimization_option_registry.h"
#include "ge_common/ge_api_types.h"
#include "framework/common/debug/ge_log.h"
#include "common/ge_common/string_util.h"

namespace {
bool ThresholdCheckerFunc(const std::string &opt_value) {
  std::string tmp_opt_value = opt_value;
  std::stringstream ss(ge::StringUtils::Trim(tmp_opt_value));
  int64_t opt_convert;
  ss >> opt_convert;
  if (ss.fail() || !ss.eof()) {
    return false;
  }
  return true;
}
}  // namespace
namespace ge {
class OptimizationOptRegistryUT : public testing::Test {
 protected:
  void SetUp() override {
    dlog_setlevel(0, 0, 0);
  }
  void TearDown() override {
    dlog_setlevel(0, 3, 0);
  }
  OptionRegistry &opt_registry = OptionRegistry::GetInstance();
  const PassOptionRegistry &pass_opt_registry = PassOptionRegistry::GetInstance();
  const char_t *const kInvalidOptionName = "ge.oo.invalid_option_name";
  const char_t *const kUtOptionName = "ge.ut_test_option";

  std::unordered_map<std::string, OoInfo> GetRegisteredOptionsByLevel(OoLevel level) {
    std::unordered_map<std::string, OoInfo> options;
    for (const auto &opt_info : opt_registry.GetRegisteredOptTable()) {
      if (OoInfoUtils::IsBitSet(opt_info.second.levels, static_cast<uint32_t>(level))) {
        const auto value_str = OoInfoUtils::GetDefaultValue(opt_info.second, level);
        options.emplace(opt_info.first, opt_info.second);
      }
    }
    return options;
  }
};

TEST_F(OptimizationOptRegistryUT, RegisterPassOption_Fail_InvalidParams) {
  std::vector<std::string> option_names;
  // pass name is empty
  REG_PASS_OPTION("").LEVELS(OoLevel::kO1);
  EXPECT_EQ(pass_opt_registry.FindOptionNamesByPassName("", option_names), GRAPH_FAILED);
  // option levels is invalid
  REG_PASS_OPTION("InvalidTestPass").LEVELS(OoLevel::kEnd);
  EXPECT_EQ(pass_opt_registry.FindOptionNamesByPassName("InvalidTestPass", option_names), GRAPH_FAILED);
  // option name is invalid
  REG_PASS_OPTION("InvalidTestPass").LEVELS(OoLevel::kO1).SWITCH_OPT("");
  EXPECT_EQ(pass_opt_registry.FindOptionNamesByPassName("InvalidTestPass", option_names), GRAPH_FAILED);
  // hierarchy is invalid
  REG_PASS_OPTION("InvalidTestPass")
      .LEVELS(OoLevel::kO1)
      .SWITCH_OPT("test_option_name1", OoHierarchy::kH1)
      .SWITCH_OPT("test_option_name", OoHierarchy::kEnd);
  EXPECT_EQ(pass_opt_registry.FindOptionNamesByPassName("InvalidTestPass", option_names), GRAPH_FAILED);
  // invalid switch options
  REG_PASS_OPTION("UtFakePass").SWITCH_OPT("ut.fake_option_secondary", OoHierarchy::kH2);
  EXPECT_EQ(pass_opt_registry.FindOptionNamesByPassName("InvalidTestPass", option_names), GRAPH_FAILED);
  REG_PASS_OPTION("UtFakePass")
      .SWITCH_OPT("ut.fake_option", OoHierarchy::kH2)
      .SWITCH_OPT("ut.fake_option_secondary", OoHierarchy::kH2);
  EXPECT_EQ(pass_opt_registry.FindOptionNamesByPassName("UtFakePass", option_names), GRAPH_FAILED);
  REG_PASS_OPTION("UtFakePass")
      .SWITCH_OPT("ut.fake_option", OoHierarchy::kH1)
      .SWITCH_OPT("ut.fake_option_secondary", OoHierarchy::kH1);
  EXPECT_EQ(pass_opt_registry.FindOptionNamesByPassName("UtFakePass", option_names), GRAPH_SUCCESS);
  EXPECT_EQ(option_names.front(), "ut.fake_option");
}

TEST_F(OptimizationOptRegistryUT, RegisterOption_Fail_InvalidParams) {
  // option name is empty
  REG_OPTION("").LEVELS(OoLevel::kO1);
  EXPECT_EQ(opt_registry.FindOptInfo(""), nullptr);
  // option level is invalid
  REG_OPTION(kUtOptionName).LEVELS(OoLevel::kEnd);
  EXPECT_EQ(opt_registry.FindOptInfo("ut.test_option"), nullptr);
  // option level is not set
  REG_OPTION(kUtOptionName).DEFAULT_VALUES({OoLevel::kO1, "true"});
  EXPECT_EQ(opt_registry.FindOptInfo(kUtOptionName), nullptr);
  // option hierarchy is invalid
  REG_OPTION(kUtOptionName, OoHierarchy::kEnd).LEVELS(OoLevel::kO1);
  EXPECT_EQ(opt_registry.FindOptInfo(kUtOptionName), nullptr);
}

TEST_F(OptimizationOptRegistryUT, RegisterOption_Ok_RegisterOptionMoreThanOnce) {
  // Set OoLevel and more than once will not overwrite
  REG_OPTION("ge.oo.fake_more_than_once1")
      .LEVELS(OoLevel::kO3)
      .LEVELS(OoLevel::kO2)
      .VISIBILITY(OoEntryPoint::kAtc, OoEntryPoint::kSession, OoEntryPoint::kIrBuild)
      .DEFAULT_VALUES({{OoLevel::kO1, "true"}});
  auto opt1 = opt_registry.FindOptInfo("ge.oo.fake_more_than_once1");
  EXPECT_NE(opt1, nullptr);
  EXPECT_EQ(OoInfoUtils::IsBitSet(opt1->levels, static_cast<uint32_t>(OoLevel::kO2)), false);

  EXPECT_EQ(GetRegisteredOptionsByLevel(OoLevel::kO2).count("ge.oo.fake_more_than_once1"), 0UL);
  EXPECT_EQ(GetRegisteredOptionsByLevel(OoLevel::kO3).count("ge.oo.fake_more_than_once1"), 1UL);

  // cannot register same options more than once
  REG_OPTION("ge.oo.fake_more_than_once1").LEVELS(OoLevel::kO2);
  REG_PASS_OPTION("UtMoreThanOncePass").SWITCH_OPT("ge.oo.fake_more_than_once1");
  const auto option_ptr = opt_registry.FindOptInfo("ge.oo.fake_more_than_once2");
  ASSERT_EQ(option_ptr, nullptr);
  EXPECT_EQ(GetRegisteredOptionsByLevel(OoLevel::kO2).count("ge.oo.fake_more_than_once1"), 0UL);
  EXPECT_EQ(GetRegisteredOptionsByLevel(OoLevel::kO3).count("ge.oo.fake_more_than_once1"), 1UL);
  std::vector<std::string> opt_names;
  EXPECT_EQ(pass_opt_registry.FindOptionNamesByPassName("UtMoreThanOncePass", opt_names), GRAPH_SUCCESS);
}

TEST_F(OptimizationOptRegistryUT, RegisterOption_Ok_FunctionalPassWithoutVisibleOption) {
  REG_PASS_OPTION("FakeFunctionalPass0").LEVELS(OoLevel::kO0);
  REG_PASS_OPTION("FakeFunctionalPass1").LEVELS(OoLevel::kO1);
  REG_PASS_OPTION("FakeFunctionalPass2").LEVELS(OoLevel::kO2);
  REG_PASS_OPTION("FakeFunctionalPass3").LEVELS(OoLevel::kO3);
  // repeated registration will not take effect
  REG_PASS_OPTION("FakeFunctionalPass3").LEVELS(OoLevel::kO3);
  // check pass name ot option names
  auto check_pass_option = [this](const std::string &pass_name) {
    std::vector<std::string> opt_names;
    const auto ret = pass_opt_registry.FindOptionNamesByPassName(pass_name, opt_names);
    ASSERT_EQ(ret, GRAPH_SUCCESS);
    ASSERT_EQ(opt_names.size(), 1UL);
    EXPECT_EQ(opt_names[static_cast<uint32_t>(OoHierarchy::kH1)], pass_name);
    ASSERT_NE(opt_registry.FindOptInfo(pass_name), nullptr);
    EXPECT_EQ(opt_registry.FindOptInfo(pass_name)->visibility, 0UL);
  };
  check_pass_option("FakeFunctionalPass0");
  check_pass_option("FakeFunctionalPass1");
  check_pass_option("FakeFunctionalPass2");
  check_pass_option("FakeFunctionalPass3");
}

TEST_F(OptimizationOptRegistryUT, RegisterOption_Ok_MultiHierarchicalOptions) {
  // multiple switch options
  REG_OPTION("ge.oo.fake_graph_fusion")
      .LEVELS(OoLevel::kO3)
      .VISIBILITY(OoEntryPoint::kAtc, OoEntryPoint::kSession, OoEntryPoint::kIrBuild)
      .SHOW_NAME(OoEntryPoint::kAtc, "oo_fake_graph_fusion", OoCategory::kModelTuning)
      .CHECKER(OoInfoUtils::IsSwitchOptValueValid)
      .HELP("The switch of fake graph fusion");
  REG_OPTION("ge.oo.fake_graph_fusion_add_relu", OoHierarchy::kH2)
      .LEVELS(OoLevel::kO3)
      .VISIBILITY(OoEntryPoint::kAtc, OoEntryPoint::kSession, OoEntryPoint::kIrBuild)
      .SHOW_NAME(OoEntryPoint::kAtc, "oo_fake_graph_fusion_add_relu", OoCategory::kModelTuning)
      .CHECKER(OoInfoUtils::IsSwitchOptValueValid)
      .HELP("The secondary switch of fake graph fusion for add-relu");
  REG_PASS_OPTION("GraphFusionAddReluPass")
      .SWITCH_OPT("ge.oo.fake_graph_fusion")
      .SWITCH_OPT("ge.oo.fake_graph_fusion_add_relu", OoHierarchy::kH2);
  const auto option_ptr = opt_registry.FindOptInfo("ge.oo.fake_graph_fusion");
  const auto option_ptr2 = opt_registry.FindOptInfo("ge.oo.fake_graph_fusion_add_relu");
  EXPECT_NE(option_ptr, nullptr);
  EXPECT_NE(option_ptr2, nullptr);
  EXPECT_EQ(option_ptr->show_infos.at(OoEntryPoint::kAtc).show_name, "oo_fake_graph_fusion");
  EXPECT_EQ(option_ptr2->show_infos.at(OoEntryPoint::kAtc).show_name, "oo_fake_graph_fusion_add_relu");
  // bind option to another pass
  REG_OPTION("ge.oo.fake_conv_relu_thres")
      .LEVELS(OoLevel::kO3)
      .DEFAULT_VALUES({{OoLevel::kO3, "800"}})
      .VISIBILITY(OoEntryPoint::kAtc, OoEntryPoint::kSession, OoEntryPoint::kIrBuild)
      .SHOW_NAME(OoEntryPoint::kAtc, "oo_fake_conv_relu_thres", OoCategory::kModelTuning)
      .CHECKER(ThresholdCheckerFunc)
      .HELP("The threshold of conv relu");
  REG_PASS_OPTION("FakeConvReluFusionPass").SWITCH_OPT("ge.oo.fake_graph_fusion");
  const auto option_ptr3 = opt_registry.FindOptInfo("ge.oo.fake_graph_fusion");
  const auto option_ptr4 = opt_registry.FindOptInfo("ge.oo.fake_conv_relu_thres");
  EXPECT_NE(option_ptr3, nullptr);
  EXPECT_NE(option_ptr4, nullptr);
  EXPECT_EQ(option_ptr3->checker("TRUE"), false);
  EXPECT_EQ(option_ptr4->checker("233"), true);
  const auto option_infos = GetRegisteredOptionsByLevel(OoLevel::kO3);
  ASSERT_EQ(option_infos.count("ge.oo.fake_graph_fusion"), 1UL);
  ASSERT_EQ(option_infos.count("ge.oo.fake_conv_relu_thres"), 1UL);
  EXPECT_EQ(OoInfoUtils::GetDefaultValue(option_infos.at("ge.oo.fake_conv_relu_thres"), OoLevel::kO3), "800");
}

TEST_F(OptimizationOptRegistryUT, GetCommandLineOptions_Ok) {
  REG_OPTION("ge.oo.fake_option")
      .LEVELS(OoLevel::kO3)
      .DEFAULT_VALUES({{OoLevel::kO3, "800"}})
      .VISIBILITY(OoEntryPoint::kAtc, OoEntryPoint::kSession, OoEntryPoint::kIrBuild)
      .SHOW_NAME(OoEntryPoint::kAtc, "oo_fake_option", OoCategory::kModelTuning)
      .CHECKER(ThresholdCheckerFunc)
      .HELP("fake option");
  REG_OPTION("ge.oo.fake_option_a", OoHierarchy::kH2)
      .LEVELS(OoLevel::kO3)
      .VISIBILITY(OoEntryPoint::kAtc, OoEntryPoint::kSession, OoEntryPoint::kIrBuild)
      .SHOW_NAME(OoEntryPoint::kAtc, "oo_fake_option_a", OoCategory::kModelTuning)
      .CHECKER(OoInfoUtils::IsSwitchOptValueValid)
      .HELP("The secondary switch of fake graph fusion for add-relu");
  REG_PASS_OPTION("FakePass0").LEVELS(OoLevel::kO0);
  REG_PASS_OPTION("FakePass1").LEVELS(OoLevel::kO1);
  REG_PASS_OPTION("FakePass2").LEVELS(OoLevel::kO2);
  REG_PASS_OPTION("FakePass3").LEVELS(OoLevel::kO3);
  const auto cmd_options = opt_registry.GetVisibleOptions(OoEntryPoint::kAtc);
  for (const auto &cmd : cmd_options) {
    EXPECT_EQ(cmd.first.empty(), false);
    std::cout << cmd.first << " " << cmd.second.help_text << std::endl;
  }
}
}  // namespace ge