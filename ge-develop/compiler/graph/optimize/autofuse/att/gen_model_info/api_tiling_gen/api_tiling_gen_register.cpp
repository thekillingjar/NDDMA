/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "api_tiling_gen_register.h"
namespace att {
ApiTilingGenRegistry &ApiTilingGenRegistry::Instance() {
  static ApiTilingGenRegistry instance;
  return instance;
}

void ApiTilingGenRegistry::RegFunc(const std::string &op_type, const AutofuseApiTilingDataGenerator &gen_func_define,
                                   const AutofuseApiTilingDataGenerator &gen_func_call,
                                   const AutofuseApiTilingDataGenerator &gen_head_files) {
  api_tiling_generators_[op_type] = {gen_func_define, gen_func_call, gen_head_files};
}

AutofuseApiTilingDataGenerator ApiTilingGenRegistry::GetApiTilingGenImplFunc(const std::string &op_type) {
  const auto iter = api_tiling_generators_.find(op_type);
  if (iter != api_tiling_generators_.cend()) {
    return iter->second.func_impl_generator;
  }
  return nullptr;
}

AutofuseApiTilingDataGenerator ApiTilingGenRegistry::GetApiTilingGenInvokeFunc(const std::string &op_type) {
  const auto iter = api_tiling_generators_.find(op_type);
  if (iter != api_tiling_generators_.cend()) {
    return iter->second.func_invoke_generator;
  }
  return nullptr;
}

AutofuseApiTilingDataGenerator ApiTilingGenRegistry::GetApiTilingGenHeadFiles(const std::string &op_type) {
  const auto iter = api_tiling_generators_.find(op_type);
  if (iter != api_tiling_generators_.cend()) {
    return iter->second.head_files_generator;
  }
  return nullptr;
}

bool ApiTilingGenRegistry::IsApiTilingRegistered(const std::string &op_type) {
  return api_tiling_generators_.find(op_type) != api_tiling_generators_.end();
}
}  // namespace att
