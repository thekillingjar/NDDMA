/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "loop_op_overrides.h"
#include "asc_overrides.h"

namespace ge {
namespace loop {
namespace {
thread_local OpOverridesPtr current_overrides = nullptr;
}

// 获取当前线程关联的OpOverrides对象，通过线程变量，实现将Loop转换为不同的OpOverrides对象
OpOverridesPtr Ops() {
  if (current_overrides != nullptr) {
    return current_overrides;
  }
  static OpOverridesPtr default_overrides = std::make_shared<AscOverrides>("default");
  return default_overrides;
}

OpsGuard::OpsGuard(OpOverridesPtr overrides) {
  prior_ = Ops();
  current_overrides = std::move(overrides);
}

OpsGuard::~OpsGuard() {
  current_overrides = prior_;
}
}  // namespace loop
}  // namespace ge