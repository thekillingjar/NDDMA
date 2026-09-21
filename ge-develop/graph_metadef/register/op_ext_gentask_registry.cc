/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_ext_gentask_registry.h"

namespace fe {
OpExtGenTaskRegistry &OpExtGenTaskRegistry::GetInstance() {
  static OpExtGenTaskRegistry registry;
  return registry;
}

OpExtGenTaskFunc OpExtGenTaskRegistry::FindRegisterFunc(const std::string &op_type) const {
  auto iter = names_to_register_func_.find(op_type);
  if (iter == names_to_register_func_.end()) {
    return nullptr;
  }
  return iter->second;
}

SKExtGenTaskFunc OpExtGenTaskRegistry::FindSKRegisterFunc(const std::string &op_type) const {
  auto iter = types_to_sk_register_func_.find(op_type);
  if (iter == types_to_sk_register_func_.end()) {
    return nullptr;
  }
  return iter->second;
}

ExtTaskType OpExtGenTaskRegistry::GetExtTaskType(const std::string &op_type) const {
  if (aicore_ext_task_ops_.count(op_type) > 0) {
    return ExtTaskType::kAicoreTask;
  }
  return ExtTaskType::kFftsPlusTask;
}

void OpExtGenTaskRegistry::Register(const std::string &op_type, OpExtGenTaskFunc const func) {
  names_to_register_func_[op_type] = func;
}

void OpExtGenTaskRegistry::RegisterSKFunc(const std::string &op_type, SKExtGenTaskFunc const func) {
  types_to_sk_register_func_[op_type] = func;
}

void OpExtGenTaskRegistry::RegisterAicoreExtTask(const std::string &op_type) {
  aicore_ext_task_ops_.emplace(op_type);
}

OpExtGenTaskRegister::OpExtGenTaskRegister(const char *op_type, OpExtGenTaskFunc func) noexcept {
  OpExtGenTaskRegistry::GetInstance().Register(op_type, func);
}

SKExtGenTaskRegister::SKExtGenTaskRegister(const char *op_type, SKExtGenTaskFunc func) noexcept {
  OpExtGenTaskRegistry::GetInstance().RegisterSKFunc(op_type, func);
}

ExtTaskTypeRegister::ExtTaskTypeRegister(const char *op_type, ExtTaskType type) noexcept {
  if (type == ExtTaskType::kAicoreTask) {
    OpExtGenTaskRegistry::GetInstance().RegisterAicoreExtTask(op_type);
  }
}
}  // namespace fe
