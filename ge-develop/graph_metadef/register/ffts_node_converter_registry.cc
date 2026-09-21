/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/ffts_node_converter_registry.h"
#include "common/hyper_status.h"

namespace gert {
FFTSNodeConverterRegistry &FFTSNodeConverterRegistry::GetInstance() {
  static FFTSNodeConverterRegistry registry;
  return registry;
}

FFTSNodeConverterRegistry::NodeConverter FFTSNodeConverterRegistry::FindNodeConverter(const string &func_name) {
  auto data = FindRegisterData(func_name);
  if (data == nullptr) {
    return nullptr;
  }
  return data->converter;
}
void FFTSNodeConverterRegistry::RegisterNodeConverter(const std::string &func_name, NodeConverter func) {
  names_to_register_data_[func_name] = {func, -1};
}
const FFTSNodeConverterRegistry::ConverterRegisterData *FFTSNodeConverterRegistry::FindRegisterData(
    const string &func_name) const {
  auto iter = names_to_register_data_.find(func_name);
  if (iter == names_to_register_data_.end()) {
    return nullptr;
  }
  return &iter->second;
}
void FFTSNodeConverterRegistry::Register(const string &func_name,
    const FFTSNodeConverterRegistry::ConverterRegisterData &data) {
  names_to_register_data_[func_name] = data;
}
FFTSNodeConverterRegister::FFTSNodeConverterRegister(const char *lower_func_name,
    FFTSNodeConverterRegistry::NodeConverter func) noexcept {
  FFTSNodeConverterRegistry::GetInstance().Register(lower_func_name, {func, -1});
}
FFTSNodeConverterRegister::FFTSNodeConverterRegister(const char *lower_func_name, int32_t require_placement,
    FFTSNodeConverterRegistry::NodeConverter func) noexcept {
  FFTSNodeConverterRegistry::GetInstance().Register(lower_func_name, {func, require_placement});
}
}  // namespace gert
