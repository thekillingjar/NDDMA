/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "execution_runtime_utils.h"
#include "mmpa/mmpa_api.h"
#include "graph/ge_context.h"
#include "graph/ge_local_context.h"
#include "common/ge_common/ge_types.h"
#include "framework/common/debug/ge_log.h"

namespace ge {
thread_local bool ExecutionRuntimeUtils::in_heterogeneous_executor_ = false;
bool ExecutionRuntimeUtils::global_in_heterogeneous_executor_ = false;

void ExecutionRuntimeUtils::UpdateGraphOptions(const std::string &key, const std::string &value) {
  auto graph_options = GetThreadLocalContext().GetAllGraphOptions();
  graph_options[key] = value;
  GetThreadLocalContext().SetGraphOption(graph_options);
}

bool ExecutionRuntimeUtils::IsHeterogeneous() {
  std::string resource_path;
  (void)ge::GetContext().GetOption(RESOURCE_CONFIG_PATH, resource_path);
  if (!resource_path.empty()) {
    GELOGI("Get is heterogenous from option[%s], path = %s", RESOURCE_CONFIG_PATH, resource_path.c_str());
    return true;
  }

  const char_t *file_path = nullptr;
  MM_SYS_GET_ENV(MM_ENV_RESOURCE_CONFIG_PATH, file_path);
  if (file_path != nullptr) {
    GELOGI("Get resource config path success, path = %s", file_path);
    UpdateGraphOptions(RESOURCE_CONFIG_PATH, file_path);
    return true;
  }
  return false;
}

void ExecutionRuntimeUtils::EnableInHeterogeneousExecutor() {
  in_heterogeneous_executor_ = true;
}

void ExecutionRuntimeUtils::EnableGlobalInHeterogeneousExecutor() {
  global_in_heterogeneous_executor_ = true;
}

bool ExecutionRuntimeUtils::IsInHeterogeneousExecutor() {
  return global_in_heterogeneous_executor_ || in_heterogeneous_executor_;
}
}  // namespace ge