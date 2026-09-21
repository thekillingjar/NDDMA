/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/hidden_inputs_func_registry.h"
#include <iostream>
#include "framework/common/debug/ge_log.h"
namespace ge {
GetHiddenAddrs HiddenInputsFuncRegistry::FindHiddenInputsFunc(const HiddenInputsType input_type) {
  const auto &iter = type_to_funcs_.find(input_type);
  if (iter != type_to_funcs_.end()) {
    return iter->second;
  }
  GELOGW("Hidden input func not found, type:[%d].", static_cast<int32_t>(input_type));
  return nullptr;
}

void HiddenInputsFuncRegistry::Register(HiddenInputsType const input_type, GetHiddenAddrs const func) {
  GELOGD("Hidden input func reg, type:[%d].", static_cast<int32_t>(input_type));
  type_to_funcs_[input_type] = func;
}
HiddenInputsFuncRegistry &HiddenInputsFuncRegistry::GetInstance() {
  static HiddenInputsFuncRegistry registry;
  return registry;
}

HiddenInputsFuncRegister::HiddenInputsFuncRegister(const HiddenInputsType input_type, const GetHiddenAddrs func) {
  HiddenInputsFuncRegistry::GetInstance().Register(input_type, func);
}
}  // namespace ge
