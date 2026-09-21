/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "execution_runtime.h"
#include "mmpa/mmpa_api.h"
#include "graph_metadef/common/ge_common/util.h"

namespace ge {
namespace {
constexpr const char_t *kHeterogeneousRuntimeLibName = "libmodel_deployer.so";
}
std::mutex ExecutionRuntime::mu_;
void *ExecutionRuntime::handle_;
std::shared_ptr<ExecutionRuntime> ExecutionRuntime::instance_;

void ExecutionRuntime::SetExecutionRuntime(const std::shared_ptr<ExecutionRuntime> &instance) {
  const std::lock_guard<std::mutex> lk(mu_);
  instance_ = instance;
}

ExecutionRuntime *ExecutionRuntime::GetInstance() {
  const std::lock_guard<std::mutex> lk(mu_);
  return instance_.get();
}

Status ExecutionRuntime::InitHeterogeneousRuntime(const std::map<std::string, std::string> &options) {
  if (handle_ != nullptr) {
    GELOGI("Heterogeneous runtime has been inited.");
    return SUCCESS;
  }
  GE_DISMISSABLE_GUARD(rollback, ([]() { FinalizeExecutionRuntime(); }));
  GE_CHK_STATUS_RET(LoadHeterogeneousLib(), "Failed to load heterogeneous lib.");
  GE_CHK_STATUS_RET(SetupHeterogeneousRuntime(options), "Failed to setup heterogeneous runtime.");
  GE_DISMISS_GUARD(rollback);
  GEEVENT("Heterogeneous runtime init success.");
  return SUCCESS;
}

Status ExecutionRuntime::LoadHeterogeneousLib() {
  const auto open_flag =
      static_cast<int32_t>(static_cast<uint32_t>(MMPA_RTLD_NOW) | static_cast<uint32_t>(MMPA_RTLD_GLOBAL));
  handle_ = mmDlopen(kHeterogeneousRuntimeLibName, open_flag);
  if (handle_ == nullptr) {
    const auto *error_msg = mmDlerror();
    GE_IF_BOOL_EXEC(error_msg == nullptr, error_msg = "unknown error");
    GELOGE(FAILED, "[Dlopen][So] failed, so name = %s, error_msg = %s", kHeterogeneousRuntimeLibName, error_msg);
    return FAILED;
  }
  GELOGD("Open %s succeeded", kHeterogeneousRuntimeLibName);
  return SUCCESS;
}

Status ExecutionRuntime::SetupHeterogeneousRuntime(const std::map<std::string, std::string> &options) {
  using InitFunc = Status(*)(const std::map<std::string, std::string> &);
  const auto init_func = reinterpret_cast<InitFunc>(mmDlsym(handle_, "InitializeHeterogeneousRuntime"));
  if (init_func == nullptr) {
    GELOGE(FAILED, "[Dlsym] failed to find function: InitializeHeterogeneousRuntime");
    return FAILED;
  }
  GE_CHK_STATUS_RET(init_func(options), "Failed to invoke InitializeHeterogeneousRuntime");
  return SUCCESS;
}

void ExecutionRuntime::FinalizeExecutionRuntime() {
  GEEVENT("Execution runtime finalize begin.");
  const auto instance = GetInstance();
  if (instance != nullptr) {
    (void) instance->Finalize();
    instance_ = nullptr;
  }

  if (handle_ != nullptr) {
    GELOGD("close so: %s", kHeterogeneousRuntimeLibName);
    (void) mmDlclose(handle_);
    handle_ = nullptr;
  }
  GEEVENT("Execution runtime finalized.");
}

const std::string &ExecutionRuntime::GetCompileHostResourceType() const {
  static std::string empty_string;
  return empty_string;
}

const std::map<std::string, std::string> &ExecutionRuntime::GetCompileDeviceInfo() const {
  static std::map<std::string, std::string> empty_map;
  return empty_map;
}
}  // namespace ge