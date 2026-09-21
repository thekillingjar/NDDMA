/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "lowering/placement/placed_lowering_register.h"
namespace gert {
PlacedLoweringRegistry &PlacedLoweringRegistry::GetInstance() {
  static PlacedLoweringRegistry registry;
  return registry;
}
PlacedLower PlacedLoweringRegistry::FindPlacedLower(const std::string &type) {
  const auto it = type_to_register_data_.find(type);
  if (it == type_to_register_data_.end()) {
    return nullptr;
  }
  return it->second;
}
void PlacedLoweringRegistry::Register(const std::string &type, PlacedLower func) {
  type_to_register_data_[type] = func;
}
PlacedLoweringRegister::PlacedLoweringRegister (const char *type, PlacedLower func) noexcept {
  PlacedLoweringRegistry::GetInstance().Register(type, func);
}
}
