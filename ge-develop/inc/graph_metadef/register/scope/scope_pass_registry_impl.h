/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef REGISTER_SCOPE_SCOPE_PASS_REGISTRY_IMPL_H
#define REGISTER_SCOPE_SCOPE_PASS_REGISTRY_IMPL_H

#include <mutex>
#include "register/scope/scope_fusion_pass_register.h"

namespace ge {
struct CreatePassFnPack;
class ScopeFusionPassRegistry::ScopeFusionPassRegistryImpl {
 public:
  void RegisterScopeFusionPass(const std::string &pass_name, ScopeFusionPassRegistry::CreateFn create_fn,
                               bool is_general);
  ScopeFusionPassRegistry::CreateFn GetCreateFn(const std::string &pass_name);
  std::unique_ptr<ScopeBasePass> CreateScopeFusionPass(const std::string &pass_name);
  std::vector<std::string> GetAllRegisteredPasses();
  bool SetPassEnableFlag(const std::string pass_name, const bool flag);

 private:
  std::mutex mu_;
  std::vector<std::string> pass_names_;  // In the order of user registration
  std::map<std::string, CreatePassFnPack> create_fn_packs_;
};
}  // namespace ge
#endif  // REGISTER_SCOPE_SCOPE_PASS_REGISTRY_IMPL_H
