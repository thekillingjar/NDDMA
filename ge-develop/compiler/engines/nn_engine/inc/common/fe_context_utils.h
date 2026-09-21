/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_INC_COMMON_FE_CONTEXT_UTILS_H_
#define AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_INC_COMMON_FE_CONTEXT_UTILS_H_

#include <string>
#include "register/graph_optimizer/graph_optimize_register_error_codes.h"

namespace fe {
enum class PrecisionMode {
  ENUM_FORCE_FP16 = 0,
  ENUM_ALLOW_FP32_TO_FP16 = 1,
  ENUM_ALLOW_MIX_PRECISION_FP16 = 2,
  ENUM_MUST_KEEP_ORIGIN_DTYPE = 3,
  ENUM_FORCE_FP32 = 4,
  ENUM_FORCE_LOWERPRECISION = 5,
  ENUM_ALLOW_FP32_TO_BF16 = 6,
  ENUM_ALLOW_FP32_TO_LOWPRECISION = 7,
  ENUM_ALLOW_MIX_PRECISION_BF16 = 8,
  ENUM_CUBE_HIF8 = 9,
  ENUM_MIXED_HIF8 = 10,
  ENUM_UNDEFINED = 11
};

class FEContextUtils {
public:
  static std::string GetPrecisionMode();
  static Status GetPrecisionMode(std::string &precision_mode);
  static Status GetPrecisionMode(fe::PrecisionMode &precision_mode);
  static std::string GetBuildMode();
  static std::string GetBuildStep();
  static std::string GetCoreType();
  static std::string GetFusionSwitchFilePath();
  static bool IsTrainMode();
  static bool IsOpTuneMode();
private:
  static std::string GetGeContextValue(const std::string &key);
  static std::string GetGeContextValue(const std::string &key, const std::string &default_value);
  static bool GetGeContextBoolValue(const std::string &key, const bool &default_value);
  static void SetDefaultPrecisionMode(std::string &precision_mode);
  static void SetDefaultPrecisionMode(fe::PrecisionMode &precision_mode);
};
}
#endif  // AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_INC_COMMON_FE_CONTEXT_UTILS_H_
