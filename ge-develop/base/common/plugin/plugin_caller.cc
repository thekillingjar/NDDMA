/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/plugin/plugin_caller.h"
#include "common/checker.h"

namespace ge {
void PluginCaller::ReportInnerError() const {
  const char_t *error = mmDlerror();
  error = (error == nullptr) ? "" : error;
  GELOGW("[Invoke][DlOpen] failed. path = %s, error = %s", lib_name_.c_str(), error);
}

Status PluginCaller::LoadLib() {
  const std::unique_lock<std::mutex> lock(mutex_);
  if (handle_ != nullptr) {
    GELOGD("Do not need load lib.");
    return SUCCESS;
  }
  GELOGI("To invoke dlopen on lib: %s", lib_name_.c_str());
  constexpr uint32_t open_flag = static_cast<uint32_t>(MMPA_RTLD_NOW) | static_cast<uint32_t>(MMPA_RTLD_GLOBAL);
  handle_ = mmDlopen(lib_name_.c_str(), static_cast<int32_t>(open_flag));
  if (handle_ == nullptr) {
    ReportInnerError();
    return NOT_CHANGED;
  }
  return SUCCESS;
}

void PluginCaller::UnloadLib() {
  const std::unique_lock<std::mutex> lock(mutex_);
  if (handle_ == nullptr) {
    GELOGD("Do not need unload lib.");
    return;
  }
  const int32_t ret = mmDlclose(handle_);
  if (ret != 0) {
    ReportInnerError();
  }
  handle_ = nullptr;
}
}  // namespace ge