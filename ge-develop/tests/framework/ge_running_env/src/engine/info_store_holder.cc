/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ge_running_env/info_store_holder.h"
FAKE_NS_BEGIN

namespace {
std::string GenStoreName() {
  static int store_id = 0;
  return "store_" + std::to_string(store_id++);
}
}  // namespace

InfoStoreHolder::InfoStoreHolder(const std::string& kernel_lib_name) : kernel_lib_name_(kernel_lib_name) {}

InfoStoreHolder::InfoStoreHolder() : kernel_lib_name_(GenStoreName()) {}

void InfoStoreHolder::RegistOp(std::string op_type) {
  OpInfo default_op_info = {.engine = engine_name_,
                            .opKernelLib = kernel_lib_name_,
                            .computeCost = 0,
                            .flagPartial = false,
                            .flagAsync = false,
                            .isAtomic = false};

  auto iter = op_info_map_.find(op_type);
  if (iter == op_info_map_.end()) {
    op_info_map_.emplace(op_type, default_op_info);
  }
}

void InfoStoreHolder::EngineName(std::string engine_name) { engine_name_ = engine_name; }

std::string InfoStoreHolder::GetLibName() { return kernel_lib_name_; }

FAKE_NS_END
