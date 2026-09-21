/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/fe_context_utils.h"
#include "common/fe_log.h"
#include "common/aicore_util_constants.h"
#include "common/string_utils.h"
#include "common/fe_type_utils.h"
#include "common/platform_utils.h"
#include "common/fe_report_error.h"
#include "graph/ge_context.h"
#include "ge/ge_api_types.h"
#include "graph/tuning_utils.h"
#include "common/platform_utils.h"
#include "platform/platform_info.h"

namespace fe {
namespace {
static const std::unordered_map<std::string, PrecisionMode> kPrecisionModeMap = {
    {ALLOW_MIX_PRECISION, PrecisionMode::ENUM_ALLOW_MIX_PRECISION_FP16},
    {FORCE_FP16, PrecisionMode::ENUM_FORCE_FP16},
    {FORCE_FP32, PrecisionMode::ENUM_FORCE_FP32},
    {ALLOW_FP32_TO_FP16, PrecisionMode::ENUM_ALLOW_FP32_TO_FP16},
    {MUST_KEEP_ORIGIN_DTYPE, PrecisionMode::ENUM_MUST_KEEP_ORIGIN_DTYPE},
    {FORCE_LOWERPRECISION, PrecisionMode::ENUM_FORCE_LOWERPRECISION},
    {ALLOW_FP32_TO_BF16, PrecisionMode::ENUM_ALLOW_FP32_TO_BF16},
    {ALLOW_FP32_TO_LOWPRECISION, PrecisionMode::ENUM_ALLOW_FP32_TO_LOWPRECISION},
    {ALLOW_MIX_PRECISION_FP16, PrecisionMode::ENUM_ALLOW_MIX_PRECISION_FP16},
    {ALLOW_MIX_PRECISION_BF16, PrecisionMode::ENUM_ALLOW_MIX_PRECISION_BF16},
    {CUBE_FP16IN_FP32OUT, PrecisionMode::ENUM_FORCE_FP32},
    {V2_FP16, PrecisionMode::ENUM_FORCE_FP16},
    {V2_MIX_FP16, PrecisionMode::ENUM_ALLOW_MIX_PRECISION_FP16},
    {V2_MIX_BF16, PrecisionMode::ENUM_ALLOW_MIX_PRECISION_BF16},
    {V2_ORIGIN, PrecisionMode::ENUM_MUST_KEEP_ORIGIN_DTYPE},
    {V2_ORIGIN, PrecisionMode::ENUM_MUST_KEEP_ORIGIN_DTYPE},
    {kCubeHif8, PrecisionMode::ENUM_CUBE_HIF8},
    {kMixedHif8, PrecisionMode::ENUM_MIXED_HIF8}
};
const std::unordered_set<std::string> kIntrinsicvconvSet = {
    "f322bf16r", "f322bf16f", "f322bf16c", "f322bf16a", "f322bf16z"
};
const std::map<string, bool> kSwitchMap {{"1", true}, {"0", false}};
const int32_t kBase = 10;
const std::string INTRINSIC_VCONV = "Intrinsic_vconv";
}

std::string FEContextUtils::GetPrecisionMode() {
  std::string precision_mode_str = GetGeContextValue(ge::PRECISION_MODE);
  if (precision_mode_str.empty()) {
    precision_mode_str = GetGeContextValue(ge::PRECISION_MODE_V2);
  }
  return precision_mode_str;
}

bool FEContextUtils::IsTrainMode() {
  std::string run_mode = GetGeContextValue(ge::OPTION_GRAPH_RUN_MODE, "1");
  if (ge::GraphRunMode(std::strtol(run_mode.c_str(), nullptr, kBase)) >= ge::TRAIN) {
    return true;
  }
  return false;
}

void FEContextUtils::SetDefaultPrecisionMode(std::string &precision_mode) {
  if (!IsTrainMode()) {
    precision_mode = FORCE_FP16;
  } else {
    if (PlatformUtils::Instance().IsEnableCubeHighPrecision()) {
      precision_mode = MUST_KEEP_ORIGIN_DTYPE;
    } else {
      precision_mode = ALLOW_FP32_TO_FP16;
    }
  }
  FE_LOGI("The value of [%s] is empty, set to default [%s].", ge::PRECISION_MODE.c_str(), precision_mode.c_str());
}

void FEContextUtils::SetDefaultPrecisionMode(fe::PrecisionMode &precision_mode) {
  if (!IsTrainMode()) {
    precision_mode = PrecisionMode::ENUM_FORCE_FP16;
  } else {
    if (PlatformUtils::Instance().IsEnableCubeHighPrecision()) {
      precision_mode = PrecisionMode::ENUM_MUST_KEEP_ORIGIN_DTYPE;
    } else {
      precision_mode = PrecisionMode::ENUM_ALLOW_FP32_TO_FP16;
    }
  }
  FE_LOGI("The value for [%s] is empty; setting it to the default [%d].", ge::PRECISION_MODE.c_str(), precision_mode);
}

Status FEContextUtils::GetPrecisionMode(std::string &precision_mode) {
  precision_mode = GetPrecisionMode();
  if (precision_mode.empty()) {
    SetDefaultPrecisionMode(precision_mode);
    return SUCCESS;
  }
  const auto &iter = kPrecisionModeMap.find(precision_mode);
  if (iter == kPrecisionModeMap.end()) {
    FE_LOGE("Precision mode value %s is incorrect, please check it.", precision_mode.c_str());
    ErrorMessageDetail err_msg(EM_INPUT_OPTION_INVALID, {precision_mode, ge::PRECISION_MODE,
                               "The current value is not within the valid range"});
    ReportErrorMessage(err_msg);
    return FAILED;
  }
  const auto &precision_mode_enum = iter->second;
  if (precision_mode_enum == PrecisionMode::ENUM_ALLOW_MIX_PRECISION_BF16 ||
      precision_mode_enum == PrecisionMode::ENUM_ALLOW_FP32_TO_BF16) {
    PlatFormInfos platform_infos;
    OptionalInfos optional_infos;
    bool enable_flag = false;
    if (PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_infos, optional_infos) != SUCCESS) {
      FE_LOGE("Getting platform information without SOC version has failed");
      return FAILED;
    }
    std::map<std::string, std::vector<std::string>> intrinsic_map = platform_infos.GetAICoreIntrinsicDtype();
    auto intr_iter = intrinsic_map.find(INTRINSIC_VCONV);
    std::vector<std::string> intrinsic_vec;
    if (intr_iter != intrinsic_map.end()) {
      intrinsic_vec = intr_iter->second;
    }
    for (auto nIterator = intrinsic_vec.cbegin(); nIterator != intrinsic_vec.cend(); ++nIterator) {
      if (kIntrinsicvconvSet.count(*nIterator) > 0) {
        enable_flag = true;
        break;
      }
    }
    if (!enable_flag) {
      FE_LOGE("The AI core doesn't support allow_mix_precision_bf16 or allow_fp32_to_bf16.");
      ErrorMessageDetail err_msg(EM_INPUT_OPTION_INVALID,
                                 {precision_mode, ge::PRECISION_MODE, "Current soc not support dtype of BFloat16"});
      ReportErrorMessage(err_msg);
      return FAILED;
    }
  }

  return SUCCESS;
}

Status FEContextUtils::GetPrecisionMode(fe::PrecisionMode &precision_mode) {
  std::string precision_mode_str = GetPrecisionMode();
  if (precision_mode_str.empty()) {
    SetDefaultPrecisionMode(precision_mode);
    return SUCCESS;
  }
  const auto &iter = kPrecisionModeMap.find(precision_mode_str);
  if (iter == kPrecisionModeMap.end()) {
    FE_LOGE("Precision mode value %s is incorrect, please check it.", precision_mode_str.c_str());
    ErrorMessageDetail err_msg(EM_INPUT_OPTION_INVALID, {precision_mode_str, ge::PRECISION_MODE,
                               "The current value is not within the valid range"});
    ReportErrorMessage(err_msg);
    return FAILED;
  }
  const auto &precision_mode_enum = iter->second;
  if (precision_mode_enum == PrecisionMode::ENUM_ALLOW_MIX_PRECISION_BF16 ||
      precision_mode_enum == PrecisionMode::ENUM_ALLOW_FP32_TO_BF16) {
    PlatFormInfos platform_infos;
    OptionalInfos optional_infos;
    bool enable_flag = false;
    if (PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_infos, optional_infos) != SUCCESS) {
      FE_LOGE("Getting platform information without SOC version has failed");
      return FAILED;
    }
    std::map<std::string, std::vector<std::string>> intrinsic_map = platform_infos.GetAICoreIntrinsicDtype();
    auto intr_iter = intrinsic_map.find(INTRINSIC_VCONV);
    std::vector<std::string> intrinsic_vec;
    if (intr_iter != intrinsic_map.end()) {
      intrinsic_vec = intr_iter->second;
    }
    for (auto nIterator = intrinsic_vec.cbegin(); nIterator != intrinsic_vec.cend(); ++nIterator) {
      if (kIntrinsicvconvSet.count(*nIterator) > 0) {
        enable_flag = true;
        break;
      }
    }
    if (!enable_flag) {
      FE_LOGE("The AI core doesn't support mixed_bfloat16, allow_mix_precision_bf16 or allow_fp32_to_bf16.");
      ErrorMessageDetail err_msg(EM_INPUT_OPTION_INVALID,
          {precision_mode_str, ge::PRECISION_MODE, "Current soc not support dtype of BFloat16"});
      ReportErrorMessage(err_msg);
      return FAILED;
    }
  }
  precision_mode = precision_mode_enum;
  return SUCCESS;
}

std::string FEContextUtils::GetBuildMode() { return GetGeContextValue(ge::BUILD_MODE); }

std::string FEContextUtils::GetBuildStep() { return GetGeContextValue(ge::BUILD_STEP); }

std::string FEContextUtils::GetCoreType() { return GetGeContextValue(ge::CORE_TYPE); }

std::string FEContextUtils::GetFusionSwitchFilePath() { return GetGeContextValue(ge::FUSION_SWITCH_FILE); }

std::string FEContextUtils::GetGeContextValue(const std::string &key) {
  std::string option_value;
  return GetGeContextValue(key, option_value);
}

std::string FEContextUtils::GetGeContextValue(const std::string &key, const std::string &default_value) {
  std::string option_value;
  ge::graphStatus status = ge::GetContext().GetOption(key, option_value);
  if (status != ge::GRAPH_SUCCESS) {
    FE_LOGD("Cannot get option value [%s].", key.c_str());
  } else {
    FE_LOGD("The option value[%s] in ge context is %s.", key.c_str(), option_value.c_str());
    return option_value;
  }
  return default_value;
}

bool FEContextUtils::GetGeContextBoolValue(const std::string &key, const bool &default_value) {
  std::string option_value;
  ge::graphStatus status = ge::GetContext().GetOption(key, option_value);
  if (status != ge::GRAPH_SUCCESS || option_value.empty()) {
    FE_LOGD("Cannot get option value [%s] or the value is empty.", key.c_str());
    return default_value;
  }
  FE_LOGD("The option value[%s] in ge context is [%s].", key.c_str(), option_value.c_str());
  const std::map<std::string, bool>::const_iterator iter = kSwitchMap.find(option_value);
  if (iter == kSwitchMap.end()) {
    FE_LOGD("The value [%s] is neither 0 nor 1.", option_value.c_str());
    return default_value;
  }
  return iter->second;
}

bool FEContextUtils::IsOpTuneMode() {
  std::string build_mode = GetBuildMode();
  return (build_mode == ge::BUILD_MODE_BASELINE || build_mode == ge::BUILD_MODE_TUNING ||
          build_mode == ge::BUILD_MODE_OPAT_RESULT);
}
}
