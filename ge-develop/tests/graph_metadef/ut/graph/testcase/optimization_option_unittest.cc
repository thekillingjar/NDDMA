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

#include "graph/option/optimization_option.h"
#include "register/optimization_option_registry.h"
#include "register/pass_option_utils.h"
#include "ge_common/ge_api_types.h"
#include "framework/common/debug/ge_log.h"
#include "graph/ge_local_context.h"

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

bool CustomCheckerFunc(const std::string &opt_value) {
  if (opt_value.empty() || (opt_value == "disable") || (opt_value == "enable")) {
    return true;
  }
  return false;
}
}  // namespace
namespace ge {
class OptimizationOptionUT : public testing::Test {
 protected:
  void SetUp() override {
    oopt.Initialize({}, {});
    dlog_setlevel(0, 0, 0);
  }
  void TearDown() override {
    dlog_setlevel(0, 3, 0);
  }

  OptimizationOption &oopt = GetThreadLocalContext().GetOo();

  const std::unordered_map<std::string, OoInfo> &registered_opt_table =
      OptionRegistry::GetInstance().GetRegisteredOptTable();

  static void CheckOptionValue(const OptimizationOption &oo, const std::string &opt_name, const std::string &expect_value) {
    std::string value;
    EXPECT_EQ(oo.GetValue(opt_name, value), GRAPH_SUCCESS);
    EXPECT_EQ(value, expect_value);
  }
  static void CheckNotConfiguredOption(const OptimizationOption &oo, const std::string &opt_name) {
    std::string value;
    EXPECT_NE(oo.GetValue(opt_name, value), GRAPH_SUCCESS);
  }
};

REG_PASS_OPTION("OoUtFunctionalPass1").LEVELS(OoLevel::kO0);
REG_PASS_OPTION("OoUtFunctionalPass2").LEVELS(OoLevel::kO1);
REG_PASS_OPTION("OoUtFunctionalPass3").LEVELS(OoLevel::kO2);
REG_PASS_OPTION("OoUtFunctionalPass4").LEVELS(OoLevel::kO3);

REG_OPTION("ge.oo.test_dead_code_elimination")
    .LEVELS(OoLevel::kO1)
    .CHECKER(OoInfoUtils::IsSwitchOptValueValid)
    .VISIBILITY(OoEntryPoint::kAtc, OoEntryPoint::kSession, OoEntryPoint::kIrBuild)
    .SHOW_NAME(OoEntryPoint::kAtc, "oo_test_dead_code_elimination", OoCategory::kModelTuning);
REG_PASS_OPTION("OoUtDeadCodeEliminationPass").SWITCH_OPT("ge.oo.test_dead_code_elimination");

REG_OPTION("ge.oo.test_constant_folding")
    .LEVELS(OoLevel::kO1)
    .CHECKER(OoInfoUtils::IsSwitchOptValueValid)
    .VISIBILITY(OoEntryPoint::kAtc, OoEntryPoint::kSession, OoEntryPoint::kIrBuild)
    .SHOW_NAME(OoEntryPoint::kAtc, "oo_test_constant_folding", OoCategory::kModelTuning);
REG_OPTION("ge.oo.test_constant_folding_max_expand")
    .LEVELS(OoLevel::kO1)
    .CHECKER(ThresholdCheckerFunc)
    .DEFAULT_VALUES({{OoLevel::kO1, "400"}, {OoLevel::kO2, "600"}, {OoLevel::kO3, "800"}})
    .VISIBILITY(OoEntryPoint::kAtc, OoEntryPoint::kSession, OoEntryPoint::kIrBuild)
    .SHOW_NAME(OoEntryPoint::kAtc, "oo_test_constant_folding_max_expand", OoCategory::kModelTuning);
REG_PASS_OPTION("OoUtConstantFoldingPass").SWITCH_OPT("ge.oo.test_constant_folding");

REG_OPTION("ge.oo.test_graph_fusion")
    .LEVELS(OoLevel::kO3)
    .CHECKER(OoInfoUtils::IsSwitchOptValueValid)
    .VISIBILITY(OoEntryPoint::kAtc, OoEntryPoint::kSession, OoEntryPoint::kIrBuild)
    .SHOW_NAME(OoEntryPoint::kAtc, "oo_test_graph_fusion", OoCategory::kModelTuning);
REG_OPTION("ge.oo.test_graph_fusion_add_relu")
    .LEVELS(OoLevel::kO3)
    .CHECKER(OoInfoUtils::IsSwitchOptValueValid)
    .VISIBILITY(OoEntryPoint::kAtc, OoEntryPoint::kSession, OoEntryPoint::kIrBuild)
    .SHOW_NAME(OoEntryPoint::kAtc, "oo_test_graph_fusion_add_relu", OoCategory::kModelTuning);
REG_PASS_OPTION("OoUtGraphFusionAddReluPass")
    .SWITCH_OPT("ge.oo.test_graph_fusion")
    .SWITCH_OPT("ge.oo.test_graph_fusion_add_relu", OoHierarchy::kH2);

REG_OPTION("ge.oo.test_graph_fusion_conv_relu")
    .LEVELS(OoLevel::kO3)
    .CHECKER(OoInfoUtils::IsSwitchOptValueValid)
    .VISIBILITY(OoEntryPoint::kAtc, OoEntryPoint::kSession, OoEntryPoint::kIrBuild)
    .SHOW_NAME(OoEntryPoint::kAtc, "oo_test_graph_fusion_conv_relu", OoCategory::kModelTuning);
REG_PASS_OPTION("OoUtGraphFusionConvReluPass")
    .SWITCH_OPT("ge.oo.test_graph_fusion")
    .SWITCH_OPT("ge.oo.test_graph_fusion_conv_relu", OoHierarchy::kH2);

REG_OPTION("ge.oo.test_other_type_switch")
    .LEVELS(OoLevel::kO3)
    .DEFAULT_VALUES({{OoLevel::kO3, "enable"}})
    .VISIBILITY(OoEntryPoint::kAtc, OoEntryPoint::kSession, OoEntryPoint::kIrBuild)
    .SHOW_NAME(OoEntryPoint::kAtc, "oo_test_other_type_switch", OoCategory::kModelTuning)
    .CHECKER(CustomCheckerFunc)
    .HELP("The switch of another feature");

TEST_F(OptimizationOptionUT, Initialize_Failed_InvalidOoLevel) {
  std::map<std::string, std::string> ge_options = {{ge::OO_LEVEL, ""}};
  EXPECT_EQ(oopt.Initialize(ge_options, registered_opt_table), GRAPH_PARAM_INVALID);
  std::string opt_value;
  EXPECT_NE(oopt.GetValue("ge.oo.test_other_type_switch", opt_value), GRAPH_SUCCESS);

  ge_options = {{ge::OO_LEVEL, "OvO"}};
  EXPECT_EQ(oopt.Initialize(ge_options, registered_opt_table), GRAPH_PARAM_INVALID);
}

TEST_F(OptimizationOptionUT, Initialize_Failed_InvalidOptionValue) {
  std::map<std::string, std::string> ge_options = {{ge::OO_LEVEL, "O1"}, {"ge.oo.test_graph_fusion_conv_relu", "TRUE"}};
  EXPECT_NE(oopt.Initialize(ge_options, registered_opt_table), GRAPH_SUCCESS);

  ge_options = {{"ge.oo.test_constant_folding_max_expand", "2.33"}};
  EXPECT_NE(oopt.Initialize(ge_options, registered_opt_table), GRAPH_SUCCESS);

  // threshold exceeds the maximum value of int64_t
  ge_options = {{"ge.oo.test_constant_folding_max_expand", "9223372036854775807000"}};
  EXPECT_NE(oopt.Initialize(ge_options, registered_opt_table), GRAPH_SUCCESS);
}

TEST_F(OptimizationOptionUT, Initialize_Ok_OoLevelIsNotSpecified) {
  std::string opt_value;
  EXPECT_EQ(oopt.Initialize({}, registered_opt_table), GRAPH_SUCCESS);
  CheckOptionValue(oopt, "ge.oo.test_other_type_switch", "enable");

  std::map<std::string, std::string> ge_options = {{"ge.oo.test_constant_folding", "false"},
                                                   {"ge.oo.test_constant_folding_max_expand", "233"}};
  EXPECT_EQ(oopt.Initialize(ge_options, registered_opt_table), GRAPH_SUCCESS);
  CheckOptionValue(oopt, "ge.oo.test_graph_fusion", "");
  CheckOptionValue(oopt, "ge.oo.test_graph_fusion_conv_relu", "");
  CheckOptionValue(oopt, "ge.oo.test_graph_fusion_add_relu", "");
  CheckOptionValue(oopt, "ge.oo.test_constant_folding", "false");
  CheckOptionValue(oopt, "ge.oo.test_constant_folding_max_expand", "233");
  CheckOptionValue(oopt, "ge.oo.test_dead_code_elimination", "");
  CheckOptionValue(oopt, "ge.oo.test_other_type_switch", "enable");
}

TEST_F(OptimizationOptionUT, Initialize_Ok_OoLevelIsSpecified) {
  std::map<std::string, std::string> ge_options = {{ge::OO_LEVEL, "O1"},
                                                   {"ge.constLifecycle", "graph"},
                                                   {"ge.oo.test_graph_fusion", "true"},
                                                   {"ge.oo.test_graph_fusion_add_relu", "false"}};
  EXPECT_EQ(oopt.Initialize(ge_options, registered_opt_table), GRAPH_SUCCESS);
  CheckOptionValue(oopt, "ge.oo.test_graph_fusion", "true");
  CheckOptionValue(oopt, "ge.oo.test_constant_folding", "");
  CheckOptionValue(oopt, "ge.oo.test_constant_folding_max_expand", "400");
  CheckOptionValue(oopt, "ge.oo.test_dead_code_elimination", "");
  CheckOptionValue(oopt, "ge.oo.test_graph_fusion_add_relu", "false");
  CheckNotConfiguredOption(oopt, "ge.oo.test_graph_fusion_conv_relu");
  CheckNotConfiguredOption(oopt, "ge.oo.test_other_type_switch");

  bool is_enabled = false;
  EXPECT_EQ(PassOptionUtils::CheckIsPassEnabled("OoUtFunctionalPass1", is_enabled), GRAPH_SUCCESS);
  EXPECT_EQ(is_enabled, true);
  EXPECT_EQ(PassOptionUtils::CheckIsPassEnabled("OoUtFunctionalPass2", is_enabled), GRAPH_SUCCESS);
  EXPECT_EQ(is_enabled, true);
  EXPECT_EQ(PassOptionUtils::CheckIsPassEnabled("OoUtFunctionalPass3", is_enabled), GRAPH_SUCCESS);
  EXPECT_TRUE(is_enabled == false);
  EXPECT_EQ(PassOptionUtils::CheckIsPassEnabled("OoUtFunctionalPass4", is_enabled), GRAPH_SUCCESS);
  EXPECT_TRUE(is_enabled == false);
  EXPECT_EQ(PassOptionUtils::CheckIsPassEnabled("OoUtDeadCodeEliminationPass", is_enabled), GRAPH_SUCCESS);
  EXPECT_EQ(is_enabled, true);
  EXPECT_EQ(PassOptionUtils::CheckIsPassEnabled("OoUtConstantFoldingPass", is_enabled), GRAPH_SUCCESS);
  EXPECT_EQ(is_enabled, true);
  EXPECT_EQ(PassOptionUtils::CheckIsPassEnabled("OoUtGraphFusionAddReluPass", is_enabled), GRAPH_SUCCESS);
  EXPECT_TRUE(is_enabled == false);
  EXPECT_EQ(PassOptionUtils::CheckIsPassEnabled("OoUtGraphFusionConvReluPass", is_enabled), GRAPH_SUCCESS);
  EXPECT_EQ(is_enabled, true);
}

TEST_F(OptimizationOptionUT, IsPassEnable_Failed_PassIsNotRegistered) {
  REG_PASS_OPTION("NoOptionRegisteredPass").SWITCH_OPT("unknown_option");
  std::map<std::string, std::string> ge_options = {{ge::OO_LEVEL, "O1"},
                                                   {"ge.constLifecycle", "graph"},
                                                   {"ge.oo.test_graph_fusion", "true"},
                                                   {"ge.oo.test_graph_fusion_add_relu", "false"}};
  EXPECT_EQ(oopt.Initialize(ge_options, registered_opt_table), GRAPH_SUCCESS);
  bool is_enabled = false;
  EXPECT_EQ(PassOptionUtils::CheckIsPassEnabled("OoUtUnknownPass", is_enabled), GRAPH_FAILED);
  EXPECT_EQ(PassOptionUtils::CheckIsPassEnabled("NoOptionRegisteredPass", is_enabled), GRAPH_FAILED);
}

TEST_F(OptimizationOptionUT, Initialize_With_OptimizationSwitch) {
  std::map<std::string, std::string> ge_options = {{ge::OPTIMIZATION_SWITCH, "pass1:on;pass2;pass3:;:on;pass5:of;pass6:on"}};
  const std::unordered_set<std::string> ge_option_set = {"pass6"};

  EXPECT_EQ(oopt.Initialize(ge_options, registered_opt_table, ge_option_set), GRAPH_SUCCESS);
  std::string opt_value;
  EXPECT_EQ(oopt.GetValue("ge.oo.test_other_type_switch", opt_value), GRAPH_SUCCESS);

  bool is_enabled = false;
  EXPECT_EQ(PassOptionUtils::CheckIsPassEnabled("pass1", is_enabled), GRAPH_SUCCESS);
  EXPECT_EQ(is_enabled, true);

  EXPECT_NE(PassOptionUtils::CheckIsPassEnabled("pass2", is_enabled), GRAPH_SUCCESS);

  EXPECT_NE(PassOptionUtils::CheckIsPassEnabled("pass3", is_enabled), GRAPH_SUCCESS);

  EXPECT_NE(PassOptionUtils::CheckIsPassEnabled("pass5", is_enabled), GRAPH_SUCCESS);

  EXPECT_NE(PassOptionUtils::CheckIsPassEnabled("pass6", is_enabled), GRAPH_SUCCESS);
}

TEST_F(OptimizationOptionUT, Initialize_With_FusionConfigStr_Have_OptimizationSwitch) {
  std::map<std::string, std::string> options_map;
  options_map[ge::OPTIMIZATION_SWITCH] = "pass1:on;pass2;pass3:;:on;pass5:of;pass6:on";
  GetThreadLocalContext().SetGraphOption(options_map);

  std::map<std::string, std::string> ge_options = {{ge::OPTIMIZATION_SWITCH, "pass1:on;pass2;pass3:;:on;pass5:of;pass6:on"}};
  EXPECT_EQ(oopt.Initialize(ge_options, registered_opt_table, {}), GRAPH_SUCCESS);

  std::string fusion_config_str = "pass6:on;pass7:on;pass8;pass9:;:on;pass11:of";

  EXPECT_EQ(oopt.RefreshPassSwitch(fusion_config_str), GRAPH_SUCCESS);

  bool is_enabled = false;

  EXPECT_EQ(PassOptionUtils::CheckIsPassEnabled("pass6", is_enabled), GRAPH_SUCCESS);
  EXPECT_EQ(is_enabled, true);

  EXPECT_EQ(PassOptionUtils::CheckIsPassEnabled("pass7", is_enabled), GRAPH_SUCCESS);
  EXPECT_EQ(is_enabled, true);

  EXPECT_NE(PassOptionUtils::CheckIsPassEnabled("pass8", is_enabled), GRAPH_SUCCESS);

  EXPECT_NE(PassOptionUtils::CheckIsPassEnabled("pass9", is_enabled), GRAPH_SUCCESS);

  EXPECT_NE(PassOptionUtils::CheckIsPassEnabled("pass11", is_enabled), GRAPH_SUCCESS);

  options_map.clear();
  GetThreadLocalContext().SetGraphOption(options_map);
}

TEST_F(OptimizationOptionUT, Initialize_With_FusionConfigStr_No_OptimizationSwitch) {
  EXPECT_EQ(oopt.Initialize({}, registered_opt_table, {}), GRAPH_SUCCESS);

  std::string fusion_config_str = "pass13:on;pass14;pass15:;:on;pass17:of";

  EXPECT_EQ(oopt.RefreshPassSwitch(fusion_config_str), GRAPH_SUCCESS);

  bool is_enabled = false;
  EXPECT_EQ(PassOptionUtils::CheckIsPassEnabled("pass13", is_enabled), GRAPH_SUCCESS);
  EXPECT_EQ(is_enabled, true);

  EXPECT_NE(PassOptionUtils::CheckIsPassEnabled("pass14", is_enabled), GRAPH_SUCCESS);

  EXPECT_NE(PassOptionUtils::CheckIsPassEnabled("pass15", is_enabled), GRAPH_SUCCESS);

  EXPECT_NE(PassOptionUtils::CheckIsPassEnabled("pass17", is_enabled), GRAPH_SUCCESS);
}
}  // namespace ge
