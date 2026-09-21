/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_GRAPH_OPTION_OPTIMIZATION_OPTION_INFO_H
#define INC_GRAPH_OPTION_OPTIMIZATION_OPTION_INFO_H
#include <string>
#include <vector>
#include <map>
#include <cassert>
#include <functional>
#include <cstdint>

namespace ge {
// pre-defined set of optimizaion templates
enum class OoLevel : uint32_t {
  kO0 = 0,
  kO1 = 1,
  kO2 = 2,
  kO3 = 3,
  kEnd,
};
static_assert(static_cast<int32_t>(OoLevel::kEnd) < 64, "The number of OoLevels exceeds 64!");

// the hierarchy of optimizaion options
enum class OoHierarchy : uint32_t {
  kH1 = 0,  // Primary options
  kH2 = 1,  // Secondary options
  kEnd,
};

// the entry points to GE
enum class OoEntryPoint : uint32_t {
  kSession = 0,
  kIrBuild = 1,
  kAtc = 2,
  kEnd,
};

enum class OoCategory : uint32_t {
  kGeneral = 0,
  kInput,
  kOutput,
  kTarget,
  kFeature,
  kModelTuning,
  kOperatorTuning,
  kDebug,
  kEnd,
};

struct OoShowInfo {
  OoCategory catagory;
  std::string show_name;
};

struct OoInfo {
  using ValueChecker = bool (*)(const std::string& opt_value);
  // identifies the visibility of the option at different entrances of the program
  uint64_t visibility;
  uint64_t levels;
  OoHierarchy hierarchy;
  ValueChecker checker;
  std::string name;
  std::string help_text;
  // Maps each entry point to its corresponding display option information
  std::map<OoEntryPoint, OoShowInfo> show_infos;
  std::map<OoLevel, std::string> default_values;

  explicit OoInfo(std::string opt_name, OoHierarchy opt_hierarchy = OoHierarchy::kEnd, uint64_t opt_level = 0UL,
                  uint64_t opt_vis = 0UL, std::map<OoLevel, std::string> opt_values = {},
                  ValueChecker opt_checker = nullptr, std::map<OoEntryPoint, OoShowInfo> opt_entry_infos = {},
                  std::string opt_help = "")
      : visibility(opt_vis), levels(opt_level), hierarchy(opt_hierarchy), checker(opt_checker),
        name(std::move(opt_name)), help_text(std::move(opt_help)), show_infos(std::move(opt_entry_infos)),
        default_values(std::move(opt_values)) {}
};

class OoInfoUtils {
 public:
  static bool IsBitSet(const uint64_t bits, const uint32_t pos);
  static uint64_t GenOptVisibilityBits(const std::vector<OoEntryPoint> &entries);
  static uint64_t GenOptLevelBits(const std::vector<OoLevel> &levels);
  static std::string GenOoLevelStr(const uint64_t opt_level);
  static std::string GetDefaultValue(const OoInfo &info, OoLevel target_level);
  static bool IsSwitchOptValueValid(const std::string &opt_value);
};
}  // namespace ge
#endif  // INC_GRAPH_OPTION_OPTIMIZATION_OPTION_INFO_H
