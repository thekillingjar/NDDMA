/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CANN_GRAPH_ENGINE_PASS_REGISTRY_H
#define CANN_GRAPH_ENGINE_PASS_REGISTRY_H
#include "ge/fusion/pass/fusion_pass_reg.h"
namespace ge {
namespace fusion {
inline std::string CustomPassStageToString(CustomPassStage stage) {
  static const std::map<CustomPassStage, std::string> kCustomPassStageToStringMap = {
      {CustomPassStage::kBeforeInferShape, "BeforeInferShape"},
      {CustomPassStage::kAfterInferShape, "AfterInferShape"},
      {CustomPassStage::kAfterAssignLogicStream, "AfterAssignLogicStream"},
      {CustomPassStage::kAfterBuiltinFusionPass, "AfterBuiltinFusionPass"},
      {CustomPassStage::kAfterOriginGraphOptimize, "AfterOriginGraphOptimize"},
      {CustomPassStage::kCompatibleInherited, "CompatibleInherited"},
      {CustomPassStage::kInvalid, "InvalidStage"}
  };
  if (stage > CustomPassStage::kInvalid) {
    return "";
  }
  return kCustomPassStageToStringMap.find(stage)->second;
}

class PassRegistry {
 public:
  ~PassRegistry();

  static PassRegistry &GetInstance();

  void RegisterFusionPass(FusionPassRegistrationData &reg_data);

  std::vector<FusionPassRegistrationData> GetFusionPassRegDataByStage(CustomPassStage stage) const;

 private:
  PassRegistry();
  std::map<std::string, FusionPassRegistrationData> name_2_fusion_pass_regs_;
};
}  // namespace fuison
}  // namespace ge


#endif  // CANN_GRAPH_ENGINE_PASS_REGISTRY_H
