/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/stream_manage_func_registry.h"

namespace ge {
StreamMngFuncRegistry &StreamMngFuncRegistry::GetInstance() {
  static StreamMngFuncRegistry registry;
  return registry;
}

Status StreamMngFuncRegistry::TryCallStreamMngFunc(const StreamMngFuncType func_type, MngActionType action_type,
                                                   MngResourceHandle handle) {
  StreamMngFunc callback_func = LookUpStreamMngFunc(func_type);
  if (callback_func == nullptr) {
    GELOGI("Stream manage func is not found, FuncType is [%u]", static_cast<uint32_t>(func_type));
    return SUCCESS;
  }
  const uint32_t ret = callback_func(action_type, handle);
  GELOGI("Call stream manage func, ret = %u!", ret);
  return SUCCESS;
}

void StreamMngFuncRegistry::Register(const StreamMngFuncType func_type, StreamMngFunc const manage_func) {
  std::lock_guard<std::mutex> lock(mutex_);
  type_to_func_[func_type] = manage_func;
}

StreamMngFunc StreamMngFuncRegistry::LookUpStreamMngFunc(const StreamMngFuncType func_type) {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto iter = type_to_func_.find(func_type);
  if (iter == type_to_func_.end()) {
    return nullptr;
  }
  return iter->second;
}

StreamMngFuncRegister::StreamMngFuncRegister(const StreamMngFuncType func_type, StreamMngFunc const manage_func) {
  StreamMngFuncRegistry::GetInstance().Register(func_type, manage_func);
}
}  // namespace ge
