/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/ops_kernel_builder_registry.h"
#include "framework/common/debug/ge_log.h"
#include "common/util/sanitizer_options.h"

namespace ge {
OpsKernelBuilderRegistry::~OpsKernelBuilderRegistry() noexcept {
  // disable memory leaks within the scope
  DT_ALLOW_LEAKS_GUARD(OpsKernelBuilderRegistry);
  for (auto &it : kernel_builders_) {
    GELOGW("[Unregister][Destruct] %s was not unregistered", it.first.c_str());
    // to avoid core dump when unregister is not called when so was close
    // this is called only when app is shutting down, so no release would be leaked
    new (std::nothrow) std::shared_ptr<OpsKernelBuilder>(it.second);
  }
}
void OpsKernelBuilderRegistry::Register(const string &lib_name, const OpsKernelBuilderPtr &instance) {
  const auto it = kernel_builders_.emplace(lib_name, instance);
  if (it.second) {
    GELOGI("Register OpsKernelBuilder successfully, kernel lib name = %s", lib_name.c_str());
  } else {
    GELOGW("[Register][Check] OpsKernelBuilder already registered. kernel lib name = %s", lib_name.c_str());
  }
}

void OpsKernelBuilderRegistry::UnregisterAll() {
  kernel_builders_.clear();
  GELOGI("All builders are unregistered");
}

void OpsKernelBuilderRegistry::Unregister(const string &lib_name) {
  (void)kernel_builders_.erase(lib_name);
  GELOGI("OpsKernelBuilder of %s is unregistered", lib_name.c_str());
}

const std::map<std::string, OpsKernelBuilderPtr> &OpsKernelBuilderRegistry::GetAll() const {
  return kernel_builders_;
}
OpsKernelBuilderRegistry &OpsKernelBuilderRegistry::GetInstance() {
  static OpsKernelBuilderRegistry instance;
  return instance;
}

OpsKernelBuilderRegistrar::OpsKernelBuilderRegistrar(const string &kernel_lib_name,
                                                     const CreateFn fn)
    : kernel_lib_name_(kernel_lib_name) {
  GELOGI("Register kernel lib name = %s", kernel_lib_name.c_str());
  std::shared_ptr<OpsKernelBuilder> builder;
  if (fn != nullptr) {
    builder.reset(fn());
    if (builder == nullptr) {
      GELOGE(INTERNAL_ERROR, "[Create][OpsKernelBuilder]kernel lib name = %s", kernel_lib_name.c_str());
    }
  } else {
    GELOGE(INTERNAL_ERROR, "[Check][Param:fn]Creator is nullptr, kernel lib name = %s", kernel_lib_name.c_str());
  }

  // May add empty ptr, so that error can be found afterward
  OpsKernelBuilderRegistry::GetInstance().Register(kernel_lib_name, builder);
}

OpsKernelBuilderRegistrar::~OpsKernelBuilderRegistrar() noexcept {
  GELOGI("Unregister kernel lib name = %s", kernel_lib_name_.c_str());
  OpsKernelBuilderRegistry::GetInstance().Unregister(kernel_lib_name_);
}
}  // namespace ge
