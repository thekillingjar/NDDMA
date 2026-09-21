/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/ge_common/ge_types.h"
#include "common/checker.h"
#include "graph/ge_context.h"
#include "register/optimization_option_registry.h"

#include <base/err_msg.h>

namespace ge {
namespace {
std::string ValidValuesToString(const std::set<std::string> &valid_values) {
  std::string valid_values_str;
  size_t index = 0UL;
  for (const auto &value : valid_values) {
    valid_values_str += value;
    index++;
    if (index < valid_values.size()) {
      valid_values_str += ", ";
    }
  }
  return valid_values_str;
}

bool ExportCompileStatChecker(const std::string &opt_value) {
  static const std::set<std::string> kExportCompileStatValidValues = {"0", "1", "2"};
  if (kExportCompileStatValidValues.count(opt_value) == 0UL) {
    auto reason = "The value must be selected from {" + ValidValuesToString(kExportCompileStatValidValues) + "}.";
    const auto opt_name = "--" + ge::GetContext().GetReadableName(OPTION_EXPORT_COMPILE_STAT);
    (void)REPORT_PREDEFINED_ERR_MSG("E10001", std::vector<const char *>({"parameter", "value", "reason"}),
                              std::vector<const char *>({opt_name.c_str(), opt_value.c_str(), reason.c_str()}));
    GELOGE(ge::PARAM_INVALID, "[Check][Option] option[%s] value[%s] invalid, %s.", opt_name.c_str(), opt_value.c_str(),
           reason.c_str());
    return false;
  }
  return true;
}
}  // namespace
REG_OPTION(OO_CONSTANT_FOLDING)
    .LEVELS(OoLevel::kO1)
    .DEFAULT_VALUES({{OoLevel::kO1, "true"}, {OoLevel::kO3, "true"}})
    .CHECKER(OoInfoUtils::IsSwitchOptValueValid)
    .VISIBILITY({OoEntryPoint::kSession, OoEntryPoint::kIrBuild, OoEntryPoint::kAtc})
    .SHOW_NAME(OoEntryPoint::kAtc, "oo_constant_folding", OoCategory::kModelTuning)
    .HELP("The switch of constant folding, false: disable; true(default): enable.");

REG_OPTION(OO_DEAD_CODE_ELIMINATION)
    .LEVELS(OoLevel::kO1)
    .DEFAULT_VALUES({{OoLevel::kO1, "true"}, {OoLevel::kO3, "true"}})
    .CHECKER(OoInfoUtils::IsSwitchOptValueValid)
    .VISIBILITY({OoEntryPoint::kSession, OoEntryPoint::kIrBuild, OoEntryPoint::kAtc})
    .SHOW_NAME(OoEntryPoint::kAtc, "oo_dead_code_elimination", OoCategory::kModelTuning)
    .HELP("The switch of dead code elimination, false: disable; true(default): enable.");

static bool TopoSortingModeValueChecker(const std::string &opt_value) {
  if (opt_value.empty()) {
    return true;
  }
  std::set<std::string> valid_values = {"0", "1", "2", "3"};
  const auto iter = valid_values.find(opt_value);
  if (iter == valid_values.end()) {
    std::string reason = "The value must be selected from {" + ValidValuesToString(valid_values) + "}.";
    const auto opt_name = "--" + ge::GetContext().GetReadableName(OPTION_EXPORT_COMPILE_STAT);
    (void)REPORT_PREDEFINED_ERR_MSG("E10001", std::vector<const char *>({"parameter", "value", "reason"}),
                                    std::vector<const char *>({"topo_sorting_mode", opt_value.c_str(), reason.c_str()}));
    GELOGE(ge::PARAM_INVALID, "Value of option topo_sorting_mode: %s is invalid, valid range is: [%s]",
           opt_value.c_str(), ValidValuesToString(valid_values).c_str());
    return false;
  }
  return true;
}

REG_OPTION(OPTION_TOPOSORTING_MODE)
    .LEVELS(OoLevel::kO1)
    .DEFAULT_VALUES({{OoLevel::kO1, "3"}, {OoLevel::kO3, ""}})
    .CHECKER(TopoSortingModeValueChecker)
    .VISIBILITY({OoEntryPoint::kSession, OoEntryPoint::kIrBuild, OoEntryPoint::kAtc})
    .SHOW_NAME(OoEntryPoint::kAtc, "topo_sorting_mode", OoCategory::kModelTuning)
    .HELP("The option of graph topological sort, 0: BFS; 1: DFS(default); 2: RDFS; 3: StableRDFS(stable topo).");

REG_OPTION(OPTION_EXPORT_COMPILE_STAT)
    .LEVELS(OoLevel::kO3)
    .DEFAULT_VALUES({{OoLevel::kO1, "1"}, {OoLevel::kO3, "1"}})
    .CHECKER(ExportCompileStatChecker)
    .VISIBILITY({OoEntryPoint::kSession, OoEntryPoint::kIrBuild, OoEntryPoint::kAtc})
    .SHOW_NAME(OoEntryPoint::kAtc, "export_compile_stat", OoCategory::kDebug)
    .HELP(
        "The option of configuring statistics of the graph compiler, 0: Not Generate; 1: Generated when the program "
        "exits(default); 2: Generated when graph compilation complete.");
}  // namespace ge
