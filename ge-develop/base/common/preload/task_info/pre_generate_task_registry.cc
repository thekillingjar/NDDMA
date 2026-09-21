/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/preload/task_info/pre_generate_task_registry.h"

namespace ge {
PreGenerateTaskRegistry &PreGenerateTaskRegistry::GetInstance() {
  static PreGenerateTaskRegistry registry;
  return registry;
}
PreGenerateTaskRegistry::PreGenerateTask PreGenerateTaskRegistry::FindPreGenerateTask(const std::string &func_name) {
  const auto iter = names_to_register_task_.find(func_name);
  GE_ASSERT_TRUE(!(iter == names_to_register_task_.end()));
  return iter->second;
}
void PreGenerateTaskRegistry::Register(const std::string &func_name, const PreGenerateTask func) {
  names_to_register_task_[func_name] = func;
}

PreGenerateTaskRegister::PreGenerateTaskRegister(const std::string &func_name,
                                                 const PreGenerateTaskRegistry::PreGenerateTask func) noexcept {
  PreGenerateTaskRegistry::GetInstance().Register(func_name, func);
}
}  // namespace ge