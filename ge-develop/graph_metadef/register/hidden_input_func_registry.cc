/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/hidden_input_func_registry.h"
#include <iostream>
#include "framework/common/debug/ge_log.h"
namespace ge {
GetHiddenAddr HiddenInputFuncRegistry::FindHiddenInputFunc(const HiddenInputType input_type) {
  const auto &iter = type_to_funcs_.find(input_type);
  if (iter != type_to_funcs_.end()) {
    return iter->second;
  }
  GELOGW("Hidden input func not found, type:[%d].", static_cast<int32_t>(input_type));
  return nullptr;
}

void HiddenInputFuncRegistry::Register(const HiddenInputType input_type, const GetHiddenAddr func) {
  type_to_funcs_[input_type] = func;
}
HiddenInputFuncRegistry &HiddenInputFuncRegistry::GetInstance() {
  static HiddenInputFuncRegistry registry;
  return registry;
}

HiddenInputFuncRegister::HiddenInputFuncRegister(const HiddenInputType input_type, const GetHiddenAddr func) {
  HiddenInputFuncRegistry::GetInstance().Register(input_type, func);
}
}  // namespace ge
