/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mutex"
#include "register/prototype_pass_registry.h"

#include "framework/common/debug/ge_log.h"
#include "graph/types.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "common/ge_common/ge_inner_error_codes.h"

namespace ge {
class ProtoTypePassRegistry::ProtoTypePassRegistryImpl {
 public:
  void RegisterProtoTypePass(const std::string &pass_name, ProtoTypePassRegistry::CreateFn create_fn,
                             const domi::FrameworkType fmk_type) {
    const std::lock_guard<std::mutex> lock(mu_);
    if (std::find(pass_names_.begin(), pass_names_.end(), pass_name) != pass_names_.end()) {
      GELOGW("[Register][Check] The prototype pass %s has been registered and will not overwrite the previous one",
             pass_name.c_str());
      return;
    }
    pass_names_.push_back(pass_name);

    const auto iter = create_fns_.find(fmk_type);
    if (iter != create_fns_.end()) {
      create_fns_[fmk_type].push_back(std::make_pair(pass_name, create_fn));
      GELOGD("Register prototype pass, pass name = %s", pass_name.c_str());
      return;
    }

    std::vector<std::pair<std::string, ProtoTypePassRegistry::CreateFn>> create_fn_vector;
    create_fn_vector.push_back(std::make_pair(pass_name, create_fn));
    create_fns_[fmk_type] = create_fn_vector;
    GELOGD("Register prototype pass, pass name = %s", pass_name.c_str());
  }

  std::vector<std::pair<std::string, ProtoTypePassRegistry::CreateFn>> GetCreateFnByType(
    domi::FrameworkType fmk_type) {
    const std::lock_guard<std::mutex> lock(mu_);
    const auto iter = create_fns_.find(fmk_type);
    if (iter == create_fns_.end()) {
      return std::vector<std::pair<std::string, ProtoTypePassRegistry::CreateFn>>{};
    }
    return iter->second;
  }

 private:
  std::mutex mu_;
  std::vector<std::string> pass_names_;
  std::map<domi::FrameworkType, std::vector<std::pair<std::string, ProtoTypePassRegistry::CreateFn>>> create_fns_;
};

ProtoTypePassRegistry::ProtoTypePassRegistry() {
  impl_ = ge::ComGraphMakeUnique<ProtoTypePassRegistryImpl>();
}

ProtoTypePassRegistry::~ProtoTypePassRegistry() = default;

ProtoTypePassRegistry &ProtoTypePassRegistry::GetInstance() {
  static ProtoTypePassRegistry instance;
  return instance;
}

void ProtoTypePassRegistry::RegisterProtoTypePass(const char_t *const pass_name, const CreateFn &create_fn,
                                                  const domi::FrameworkType fmk_type) {
  if (impl_ == nullptr) {
    GELOGE(MEMALLOC_FAILED, "ProtoTypePassRegistry is not properly initialized.");
    return;
  }
  std::string str_pass_name;
  if (pass_name != nullptr) {
    str_pass_name = pass_name;
  }
  impl_->RegisterProtoTypePass(str_pass_name, create_fn, fmk_type);
}

std::vector<std::pair<std::string, ProtoTypePassRegistry::CreateFn>> ProtoTypePassRegistry::GetCreateFnByType(
    const domi::FrameworkType fmk_type) const {
  if (impl_ == nullptr) {
    GELOGE(MEMALLOC_FAILED, "ProtoTypePassRegistry is not properly initialized.");
    return std::vector<std::pair<std::string, ProtoTypePassRegistry::CreateFn>>{};
  }
  return impl_->GetCreateFnByType(fmk_type);
}

ProtoTypePassRegistrar::ProtoTypePassRegistrar(const char_t *const pass_name, ProtoTypeBasePass *(*const create_fn)(),
                                               const domi::FrameworkType fmk_type) {
  if (pass_name == nullptr) {
    GELOGE(PARAM_INVALID, "Failed to register ProtoType pass, pass name is null.");
    return;
  }
  ProtoTypePassRegistry::GetInstance().RegisterProtoTypePass(pass_name, create_fn, fmk_type);
}
}  // namespace ge
