/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/graph_optimizer/fusion_common/fusion_config_info.h"
#include "mmpa/mmpa_api.h"
#include "framework/common/debug/ge_log.h"

namespace fe {
FusionConfigInfo& FusionConfigInfo::Instance() {
  static FusionConfigInfo fusion_config_info;
  return fusion_config_info;
}

Status FusionConfigInfo::Initialize() {
  if (is_init_) {
    return SUCCESS;
  }

  InitEnvParam();
  is_init_ = true;
  return SUCCESS;
}

void FusionConfigInfo::InitEnvParam() {
  const char *env_value = nullptr;
  MM_SYS_GET_ENV(MM_ENV_ENABLE_NETWORK_ANALYSIS_DEBUG, env_value);
  if (env_value != nullptr) {
    std::string env_str_value = std::string(env_value);
    GELOGD("The value of env[ENABLE_NETWORK_ANALYSIS_DEBUG] is [%s].", env_str_value.c_str());
    is_enable_network_analysis_ = static_cast<bool> (std::stoi(env_str_value.c_str()));
  }
  GELOGD("Enable network analysis is set to [%d].", is_enable_network_analysis_);
}

Status FusionConfigInfo::Finalize() {
  is_init_ = false;
  is_enable_network_analysis_ = false;
  return SUCCESS;
}

bool FusionConfigInfo::IsEnableNetworkAnalysis() const {
  return is_enable_network_analysis_;
}
}
