/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fusion_strategy_registry.h"

namespace ge {
FusionStrategyRegistry &FusionStrategyRegistry::Instance() {
  static FusionStrategyRegistry instance;
  return instance;
}

void FusionStrategyRegistry::Register(std::unique_ptr<FusionStrategy> decider, loop::FuseType fuse_type) {
  fusion_strategy_[(1UL << static_cast<uint64_t>(fuse_type))] = std::move(decider);
  if (default_fusion_strategy_ == nullptr) {
    default_fusion_strategy_ = std::move(ComGraphMakeUnique<FusionStrategy>());
  }
}

const std::vector<FusionStrategy *> FusionStrategyRegistry::Get(uint64_t fuse_type) const {
  std::vector<FusionStrategy *> fusion_strategies;
  for (const auto &it : fusion_strategy_) {
    if ((it.first & fuse_type) != 0) {
      fusion_strategies.emplace_back(it.second.get());
    }
  }
  fusion_strategies.emplace_back(default_fusion_strategy_.get());
  return fusion_strategies;
}
}  // namespace ge
