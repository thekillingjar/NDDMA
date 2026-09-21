/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/core_num_utils.h"
#include "ge_common/debug/ge_log.h"
#include "common/checker.h"
#include "graph/ge_local_context.h"
#include "graph/utils/attr_utils.h"
#include "platform/platform_info.h"

namespace ge {
namespace {
graphStatus ConvertStrToInt32(const std::string &str, int32_t &val) {
  try {
    val = std::stoi(str);
  } catch (const std::invalid_argument &) {
    GELOGE(GRAPH_FAILED, "[Parse][Param]Failed, digit str:%s is invalid", str.c_str());
    REPORT_INNER_ERR_MSG("E19999", "Parse param failed, digit str:%s is invalid", str.c_str());
    return GRAPH_FAILED;
  } catch (const std::out_of_range &) {
    GELOGE(GRAPH_FAILED, "[Parse][Param]Failed, digit str:%s cannot change to int", str.c_str());
    REPORT_INNER_ERR_MSG("E19999", "Parse param failed, digit str:%s cannot change to int", str.c_str());
    return GRAPH_FAILED;
  } catch (...) {
    GELOGE(GRAPH_FAILED, "[Parse][Param]Failed, digit str:%s cannot change to int", str.c_str());
    REPORT_INNER_ERR_MSG("E19999", "Parse param failed, digit str:%s cannot change to int", str.c_str());
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

graphStatus ReportParamError(const std::string &param_name, const std::string &param_value_str, const std::string &reason) {
  (void)REPORT_PREDEFINED_ERR_MSG("E10001", std::vector<const char *>({"parameter", "value", "reason"}),
                            std::vector<const char *>({param_name.c_str(), param_value_str.c_str(), reason.c_str()}));
  GELOGE(PARAM_INVALID, "Value %s for parameter %s is invalid. Reason: %s", param_value_str.c_str(), param_name.c_str(), reason.c_str());
  return PARAM_INVALID;
}
}  // namespace
graphStatus CoreNumUtils::ParseAicoreNumFromOption(std::map<std::string, std::string> &options) {
  auto it = options.find(AICORE_NUM);
  if (it == options.end()) {
    return GRAPH_SUCCESS;
  }
  std::string aicore_num_option_str = it->second;
  if (aicore_num_option_str.empty()) {
    return GRAPH_SUCCESS;
  }
  if (aicore_num_option_str.find('|') == std::string::npos) {
    GELOGW("Invalid format for ge.aicoreNum: %s. Expected format: aicore_num|vectorcore_num.", aicore_num_option_str.c_str());
    return GRAPH_SUCCESS;
  }
  GELOGI("origin ge.aicoreNum in options, value: %s.", aicore_num_option_str.c_str());
  const auto delimiter_pos = aicore_num_option_str.find('|');
  std::string aicore_num_str = aicore_num_option_str.substr(0U, delimiter_pos);
  std::string vectorcore_num_str = aicore_num_option_str.substr(delimiter_pos + 1U);

  options[AICORE_NUM] = aicore_num_str;
  options[kVectorCoreNum] = vectorcore_num_str;
  return GRAPH_SUCCESS;
}

graphStatus CoreNumUtils::ParseAndValidateCoreNum(const std::string &param_name, const std::string &param_value_str, int32_t min_value,
                             int32_t max_value, int32_t &parsed_value) {
  if (ConvertStrToInt32(param_value_str, parsed_value) != GRAPH_SUCCESS) {
    return ReportParamError(param_name, param_value_str, "It is not integer.");
  }

  if (std::to_string(parsed_value) != param_value_str) {
    return ReportParamError(param_name, param_value_str,
                            "Invalid integer format: only non-negative decimal digits are allowed "
                            "(no leading zeros, spaces, or mixed characters).");
  }

  if (parsed_value < min_value || parsed_value > max_value) {
    return ReportParamError(param_name, param_value_str,
                            "It is out of range, value must be in range [" + std::to_string(min_value) + ", " +
                                std::to_string(max_value) + "].");
  }

  GELOGD("Parse and validate core num success, param_name: %s, param_value_str: %s, parsed_value: %d",
       param_name.c_str(), param_value_str.c_str(), parsed_value);
  return GRAPH_SUCCESS;
}

graphStatus CoreNumUtils::GetGeDefaultPlatformInfo(const std::string &soc_version, fe::PlatformInfo &platform_info) {
  GE_ASSERT_TRUE(fe::PlatformInfoManager::GeInstance().InitializePlatformInfo() == 0U, "Initialize platform info failed.");

  fe::OptionalInfo optional_info;
  GE_ASSERT_TRUE(fe::PlatformInfoManager::GeInstance().GetPlatformInfo(soc_version, platform_info, optional_info) == 0U, "Unable to get platform info.");
  return GRAPH_SUCCESS;
}

graphStatus CoreNumUtils::UpdateCoreCountWithOpDesc(const std::string &param_name, const std::string &op_core_num_str, int32_t soc_core_num,
                                          const std::string &res_key, std::map<std::string, std::string> &res) {
  int32_t op_core_num = -1;
  GE_ASSERT_SUCCESS(ParseAndValidateCoreNum(param_name, op_core_num_str, 0, soc_core_num, op_core_num));
  if (op_core_num > 0) {
    GELOGD("Change %s from platform %d to op_desc %d.", res_key.c_str(), soc_core_num, op_core_num);
    res[res_key] = std::to_string(op_core_num);
  }

  return GRAPH_SUCCESS;
}

graphStatus CoreNumUtils::UpdatePlatformInfosWithOpDesc(const fe::PlatformInfo &platform_info, const ge::OpDescPtr &op_desc,
                                         fe::PlatFormInfos &platform_infos, bool &is_op_core_num_set) {
  std::map<std::string, std::string> res;
  (void)platform_infos.GetPlatformResWithLock("SoCInfo", res);

  std::string aicore_num_str;
  if (ge::AttrUtils::GetStr(op_desc, kAiCoreNumOp, aicore_num_str)) {
    GELOGD("Attr: %s exists in op_desc, opName: %s", kAiCoreNumOp.c_str(), op_desc->GetName().c_str());
    GE_ASSERT_SUCCESS(UpdateCoreCountWithOpDesc(
        kAiCoreNumOp, aicore_num_str, static_cast<int32_t>(platform_info.soc_info.ai_core_cnt), kAiCoreCntIni, res));
    res[kCubeCoreCntIni] = res[kAiCoreCntIni];
    is_op_core_num_set = true;
  }

  std::string vector_core_num_str;
  if (ge::AttrUtils::GetStr(op_desc, kVectorCoreNumOp, vector_core_num_str)) {
    GELOGD("Attr: %s exists in op_desc, opName: %s", kVectorCoreNumOp.c_str(), op_desc->GetName().c_str());
    GE_ASSERT_SUCCESS(UpdateCoreCountWithOpDesc(
        kVectorCoreNumOp, vector_core_num_str, static_cast<int32_t>(platform_info.soc_info.vector_core_cnt), kVectorCoreCntIni, res));
    is_op_core_num_set = true;
  }

  if (is_op_core_num_set) {
    GELOGI("Need to update core cnt in platform infos.");
    platform_infos.SetPlatformResWithLock(kSocInfo, res);
  }

  return GRAPH_SUCCESS;
}
}  // namespace ge