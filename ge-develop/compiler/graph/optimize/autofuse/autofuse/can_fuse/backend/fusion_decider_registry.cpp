/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "fusion_decider_registry.h"
#include "framework/common/debug/ge_log.h"

namespace ge {
FusionDeciderRegistry &FusionDeciderRegistry::Instance() {
  static FusionDeciderRegistry instance;
  return instance;
}

void FusionDeciderRegistry::Register(std::unique_ptr<FusionDecider> decider, AutoFuseFwkType type) {
  decider_[type] = std::move(decider);
}

const std::unique_ptr<FusionDecider> &FusionDeciderRegistry::Get(AutoFuseFwkType type) const {
  const auto it = decider_.find(type);
  if (it != decider_.end()) {
    return it->second;
  }
  GELOGE(FAILED, "FusionDecider type:%d has not been registered.", type);
  return default_decider_;
}
}