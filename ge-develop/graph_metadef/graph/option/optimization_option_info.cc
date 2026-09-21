/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/option/optimization_option_info.h"
#include <unordered_set>
#include "common/ge_common/debug/ge_log.h"
#include "graph/ge_local_context.h"

namespace {
const std::map<ge::OoLevel, std::string> kOoLevelStr = {{ge::OoLevel::kO1, "O1"}, {ge::OoLevel::kO3, "O3"}};
}  // namespace
namespace ge {
bool OoInfoUtils::IsBitSet(const uint64_t bits, const uint32_t pos) {
  if (pos < sizeof(uint32_t)) {
    return ((bits & (1UL << pos)) != 0UL);
  }
  return false;
}

uint64_t OoInfoUtils::GenOptLevelBits(const std::vector<OoLevel> &levels) {
  uint64_t level_bits = 0;
  for (const auto level : levels) {
    if (level <= OoLevel::kO3) {
      /** O3 > O2 > O1 > O0, 当前认为四个级别之间是子集包含关系 (如果有变, 此处需要修改)
       *  例如, O1级别的选项属于 O1/O2/O3, 对应的三个比特位被置为 1
       */
      for (auto i = static_cast<uint64_t>(level); i <= static_cast<uint64_t>(OoLevel::kO3); ++i) {
        level_bits |= (1 << static_cast<uint64_t>(i));
      }
    } else if (level < OoLevel::kEnd) {
      // 超过 kO3 的优化模板都属于某个独立功能集，不一定是子集包含关系
      level_bits |= (1 << static_cast<uint64_t>(static_cast<uint64_t>(level)));
    }
  }
  return level_bits;
}

uint64_t OoInfoUtils::GenOptVisibilityBits(const std::vector<OoEntryPoint> &entries) {
  uint64_t vis_bits = 0;
  for (const auto entry : entries) {
    if (entry < OoEntryPoint::kEnd) {
      vis_bits |= (1 << static_cast<uint64_t>(static_cast<uint64_t>(entry)));
    }
  }
  return vis_bits;
}

std::string OoInfoUtils::GenOoLevelStr(const uint64_t opt_level) {
  std::string level_str;
  for (const auto &level : kOoLevelStr) {
    if (OoInfoUtils::IsBitSet(opt_level, static_cast<uint32_t>(level.first))) {
      level_str.append(level.second);
      level_str.push_back('/');
    }
  }
  if (level_str.back() == '/') {
    level_str.pop_back();
  }
  return level_str;
}

std::string OoInfoUtils::GetDefaultValue(const ge::OoInfo &info, ge::OoLevel target_level) {
  if (info.default_values.count(target_level) > 0UL) {
    return info.default_values.at(target_level);
  }
  return {};
}

bool OoInfoUtils::IsSwitchOptValueValid(const std::string &opt_value) {
  if (opt_value.empty() || (opt_value == "true") || (opt_value == "false")) {
    return true;
  }
  const auto opt_name = GetThreadLocalContext().GetReadableName(OO_LEVEL);
  (void)REPORT_PREDEFINED_ERR_MSG(
      "E10001", std::vector<const char *>({"parameter", "value", "reason"}),
      std::vector<const char *>({opt_name.c_str(), opt_value.c_str(), "The value must be true or false"}));
  GELOGE(ge::GRAPH_PARAM_INVALID, "Valid switch option value: \"true\" or \"false\" or null");
  return false;
}
}  // namespace ge
