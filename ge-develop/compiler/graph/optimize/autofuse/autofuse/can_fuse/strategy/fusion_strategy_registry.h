/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_CAN_FUSE_FUSION_STRATEGY_REGISTRY_H_
#define AUTOFUSE_CAN_FUSE_FUSION_STRATEGY_REGISTRY_H_
#include "fusion_strategy.h"
#include "utils/autofuse_attrs.h"

namespace ge {
class FusionStrategyRegistry {
 public:
  static FusionStrategyRegistry &Instance();
  void Register(std::unique_ptr<FusionStrategy> decider, loop::FuseType fuse_type);
  const std::vector<FusionStrategy *> Get(uint64_t fuse_type) const;

 private:
  FusionStrategyRegistry() = default;
  ~FusionStrategyRegistry() = default;
  FusionStrategyRegistry(const FusionStrategyRegistry &) = delete;
  FusionStrategyRegistry &operator=(const FusionStrategyRegistry &) = delete;

  std::unique_ptr<FusionStrategy> default_fusion_strategy_;
  std::map<uint64_t, std::unique_ptr<FusionStrategy>> fusion_strategy_;
};

#define REGISTER_FUSION_STRATEGY(class_name, fuse_type)                                                          \
  class class_name##Registrar {                                                                                  \
   public:                                                                                                       \
    class_name##Registrar() {                                                                                    \
      FusionStrategyRegistry::Instance().Register(ComGraphMakeUnique<class_name>(), fuse_type);                  \
    }                                                                                                            \
  };                                                                                                             \
  __attribute__((used)) static class_name##Registrar class_name##_registrar_instance
}  // namespace ge

#endif  // AUTOFUSE_CAN_FUSE_FUSION_STRATEGY_REGISTRY_H_
