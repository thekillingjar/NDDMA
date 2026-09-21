/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_CAN_FUSE_BACKEND_FUSION_DECIDER_REGISTRY_H_
#define AUTOFUSE_CAN_FUSE_BACKEND_FUSION_DECIDER_REGISTRY_H_
#include "ge_common/ge_api_types.h"
#include "graph/compute_graph.h"
#include "fusion_decider.h"
#include "common/autofuse_base_type.h"

namespace ge {
class FusionDeciderRegistry {
 public:
  static FusionDeciderRegistry &Instance();

  void Register(std::unique_ptr<FusionDecider> decider, AutoFuseFwkType type = AutoFuseFwkType::kGe);
  const std::unique_ptr<FusionDecider> &Get(AutoFuseFwkType type) const;

 private:
  FusionDeciderRegistry() = default;
  ~FusionDeciderRegistry() = default;
  FusionDeciderRegistry(const FusionDeciderRegistry &) = delete;
  FusionDeciderRegistry &operator=(const FusionDeciderRegistry &) = delete;

  std::unique_ptr<FusionDecider> default_decider_;
  std::map<AutoFuseFwkType, std::unique_ptr<FusionDecider>> decider_;
};

#define REGISTER_FUSION_DECIDER(class_name, type)                                                   \
  class class_name##Registry {                                                                      \
   public:                                                                                          \
    class_name##Registry() {                                                                        \
      FusionDeciderRegistry::Instance().Register(ComGraphMakeUnique<class_name>(), type);           \
    }                                                                                               \
  };                                                                                                \
  __attribute__((used))  static class_name##Registry class_name##_registry_instance
}  // namespace ge
#endif  // AUTOFUSE_CAN_FUSE_BACKEND_FUSION_DECIDER_REGISTRY_H_