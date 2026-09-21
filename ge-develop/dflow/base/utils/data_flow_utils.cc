/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "data_flow_utils.h"
#include "mmpa/mmpa_api.h"
#include "framework/common/debug/ge_log.h"

namespace ge {
std::string DataFlowUtils::GetAscendHomePath() {
  const char *env_value = nullptr;
  constexpr const char *kAscendHomePath = "ASCEND_HOME_PATH";
  MM_SYS_GET_ENV(MM_ENV_ASCEND_HOME_PATH, env_value);
  if (env_value != nullptr) {
    std::string file_path(env_value);
    GEEVENT("env:%s value=%s.", kAscendHomePath, file_path.c_str());
    return file_path;
  }
  constexpr const char *kDefaultAscendHomePath = "/usr/local/Ascend/cann";
  GEEVENT("env:%s do not exist, use default path:%s", kAscendHomePath, kDefaultAscendHomePath);
  return kDefaultAscendHomePath;
}
}  // namespace ge
