/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "op_config_registry_impl.h"
#include "framework/common/debug/ge_log.h"
#include "op_def_impl.h"

namespace ops {

OpConfigRegistry::OpConfigRegistry() {}

void OpConfigRegistry::RegisterOpAICoreConfig(const char* name, const char* socVersion, OpAICoreConfigFunc func = nullptr) {
  if (name == nullptr) {
    GELOGE(ge::PARAM_INVALID, "Register Op name is null.");
    return;
  }

  if (socVersion == nullptr) {
    GELOGE(ge::PARAM_INVALID, "Register socVersion for op[%s] is null.", name);
    return;
  }

  GELOGD("Add aicore config for op[%s] at socVersion[%s] in OpConfigRegistry.", name, socVersion);
  OpConfigRegistryImpl::GetInstance().AddAICoreConfig(name, socVersion, func);
}

OpConfigRegistryImpl& OpConfigRegistryImpl::GetInstance() {
  static OpConfigRegistryImpl instance;
  return instance;
}

void OpConfigRegistryImpl::AddAICoreConfig(const char* name, const char* socVersion, OpAICoreConfigFunc func) {
  if (name == nullptr) {
    GELOGE(ge::PARAM_INVALID, "AddAICoreConfig Op name is null.");
    return;
  }

  if (socVersion == nullptr) {
    GELOGE(ge::PARAM_INVALID, "AddAICoreConfig socVersion for op is null.");
    return;
  }

  GELOGD("Add aicore config for op[%s] at socVersion[%s] in OpConfigRegistryImpl.", name, socVersion);
  funcData_[ge::AscendString(name)][ge::AscendString(socVersion)] = func;
}

std::map<ge::AscendString, OpAICoreConfigFunc> OpConfigRegistryImpl::GetOpAllAICoreConfig(const char* name) {
  if (name == nullptr) {
    GELOGE(ge::PARAM_INVALID, "GetOpAllAICoreConfig Op name is null.");
  }
  
  GELOGD("Aicore config size: %zu", funcData_.size());
  
  auto iter = funcData_.find(ge::AscendString(name));
  if (iter == funcData_.end()) {
      GELOGD("Can not find aicore config for op[%s].", name);
      return std::map<ge::AscendString, OpAICoreConfigFunc>();
    }
    GELOGD("Found aicore config for op[%s].", name);
    return iter->second;
}

std::map<ge::AscendString, OpAICoreConfigFunc> GetOpAllAICoreConfig(const char* name) {
  if (name == nullptr) {
    GELOGE(ge::PARAM_INVALID, "GetOpAllAICoreConfig Op name is null.");
    return {};
  }
  return OpConfigRegistryImpl::GetInstance().GetOpAllAICoreConfig(name);
}

}