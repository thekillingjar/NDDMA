/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <map>
#include <vector>
#include "register/scope/scope_pass_registry_impl.h"
#include "framework/common/debug/ge_log.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "common/ge_common/ge_inner_error_codes.h"

namespace ge {
struct CreatePassFnPack {
  bool is_enable;
  ScopeFusionPassRegistry::CreateFn create_fn;
};

void ScopeFusionPassRegistry::ScopeFusionPassRegistryImpl::RegisterScopeFusionPass(
    const std::string &pass_name, ScopeFusionPassRegistry::CreateFn create_fn, bool is_general) {
  const std::lock_guard<std::mutex> lock(mu_);
  const auto iter = std::find(pass_names_.begin(), pass_names_.end(), pass_name);
  if (iter != pass_names_.end()) {
    GELOGW("[Register][Check] ScopeFusionPass %s already exists and will not be overwritten",
           pass_name.c_str());
    return;
  }

  CreatePassFnPack create_fn_pack;
  create_fn_pack.is_enable = is_general;
  create_fn_pack.create_fn = create_fn;
  create_fn_packs_[pass_name] = create_fn_pack;
  pass_names_.push_back(pass_name);
  GELOGI("Register pass name = %s, is_enable = %s.", pass_name.c_str(), is_general ? "true" : "false");
}

ScopeFusionPassRegistry::CreateFn ScopeFusionPassRegistry::ScopeFusionPassRegistryImpl::GetCreateFn(
    const std::string &pass_name) {
  const std::lock_guard<std::mutex> lock(mu_);
  const auto it = create_fn_packs_.find(pass_name);
  if (it == create_fn_packs_.end()) {
    GELOGW("[Get][CreateFun] ScopeFusionPass %s not registered", pass_name.c_str());
    return nullptr;
  }

  CreatePassFnPack &create_fn_pack = it->second;
  if (create_fn_pack.is_enable) {
    return create_fn_pack.create_fn;
  } else {
    GELOGW("[Get][CreateFun] ScopeFusionPass %s is disabled", pass_name.c_str());
    return nullptr;
  }
}

std::vector<std::string> ScopeFusionPassRegistry::ScopeFusionPassRegistryImpl::GetAllRegisteredPasses() {
  const std::lock_guard<std::mutex> lock(mu_);
  std::vector<std::string> all_passes;
  for (size_t i = 0U; i < pass_names_.size(); ++i) {
    if (create_fn_packs_[pass_names_[i]].is_enable) {
      all_passes.push_back(pass_names_[i]);
    }
  }

  return all_passes;
}

bool ScopeFusionPassRegistry::ScopeFusionPassRegistryImpl::SetPassEnableFlag(
    const std::string pass_name, const bool flag) {
  const std::lock_guard<std::mutex> lock(mu_);
  const auto it = create_fn_packs_.find(pass_name);
  if (it == create_fn_packs_.end()) {
    GELOGW("[Set][EnableFlag] ScopeFusionPass %s not registered", pass_name.c_str());
    return false;
  }

  CreatePassFnPack &create_fn_pack = it->second;
  create_fn_pack.is_enable = flag;
  GELOGI("enable flag of scope fusion pass:%s is set with %s.", pass_name.c_str(), flag ? "true" : "false");

  return true;
}

std::unique_ptr<ScopeBasePass> ScopeFusionPassRegistry::ScopeFusionPassRegistryImpl::CreateScopeFusionPass(
    const std::string &pass_name) {
  const auto create_fn = GetCreateFn(pass_name);
  if (create_fn == nullptr) {
    GELOGD("Create scope fusion pass failed, pass name = %s.", pass_name.c_str());
    return nullptr;
  }
  GELOGI("Create scope fusion pass, pass name = %s.", pass_name.c_str());
  return std::unique_ptr<ScopeBasePass>(create_fn());
}

ScopeFusionPassRegistry::ScopeFusionPassRegistry() {
  impl_ = ge::ComGraphMakeUnique<ScopeFusionPassRegistryImpl>();
}

ScopeFusionPassRegistry::~ScopeFusionPassRegistry() = default;

ScopeFusionPassRegistry& ScopeFusionPassRegistry::GetInstance() {
  static ScopeFusionPassRegistry instance;
  return instance;
}

void ScopeFusionPassRegistry::RegisterScopeFusionPass(const std::string &pass_name, CreateFn create_fn,
                                                      bool is_general) {
  if (impl_ == nullptr) {
    GELOGE(ge::MEMALLOC_FAILED, "Failed to register %s, ScopeFusionPassRegistry is not properly initialized.",
           pass_name.c_str());
    return;
  }
  impl_->RegisterScopeFusionPass(pass_name, create_fn, is_general);
}

void ScopeFusionPassRegistry::RegisterScopeFusionPass(const char_t *pass_name, CreateFn create_fn,
                                                      bool is_general) {
  if (impl_ == nullptr) {
    GELOGE(ge::MEMALLOC_FAILED, "Failed to register %s, ScopeFusionPassRegistry is not properly initialized.",
           pass_name);
    return;
  }
  std::string str_pass_name;
  if (pass_name != nullptr) {
    str_pass_name = pass_name;
  }
  impl_->RegisterScopeFusionPass(str_pass_name, create_fn, is_general);
}

ScopeFusionPassRegistrar::ScopeFusionPassRegistrar(const char_t *pass_name, ScopeBasePass *(*create_fn)(),
                                                   bool is_general) {
  if (pass_name == nullptr) {
    GELOGE(PARAM_INVALID, "Failed to register scope fusion pass, pass name is null.");
    return;
  }

  ScopeFusionPassRegistry::GetInstance().RegisterScopeFusionPass(pass_name, create_fn, is_general);
}
}  // namespace ge
