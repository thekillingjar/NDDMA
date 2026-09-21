/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "pass_registry.h"
#include "framework/common/debug/ge_log.h"
#include "graph/option/optimization_option_info.h"
#include "optimization_option_registry.h"

namespace ge {
namespace fusion {
namespace {
void RegisterPassOption(const std::string &pass_name) {
  const uint64_t level_bits = ge::OoInfoUtils::GenOptLevelBits({OoLevel::kO3});
  GELOGD("Fusion name [%s] registered with compile level %lu.", pass_name.c_str(), level_bits);
  ge::OoInfo opt{pass_name, ge::OoHierarchy::kH1, level_bits};
  ge::OptionRegistry::GetInstance().Register(opt);
  ge::PassOptionRegistry::GetInstance().Register(pass_name, {{ge::OoHierarchy::kH1, opt.name}});
}
} // namespace
PassRegistry::PassRegistry() = default;
PassRegistry::~PassRegistry() = default;

PassRegistry &PassRegistry::GetInstance() {
  static PassRegistry instance;
  return instance;
}

void PassRegistry::RegisterFusionPass(FusionPassRegistrationData &reg_data) {
  auto pass_name = reg_data.GetPassName().GetString();
  auto iter = name_2_fusion_pass_regs_.find(pass_name);
  if (iter != name_2_fusion_pass_regs_.cend()) {
    GELOGI("Fusion Pass has already registered, detail:[%s]", iter->second.ToString().GetString());
  } else {
    name_2_fusion_pass_regs_.emplace(pass_name, reg_data);
    RegisterPassOption(pass_name);
    GELOGI("Fusion Pass registered success, detail:[%s]", reg_data.ToString().GetString());
  }
}

std::vector<FusionPassRegistrationData> PassRegistry::GetFusionPassRegDataByStage(CustomPassStage stage) const {
  std::vector<FusionPassRegistrationData> fusion_passes;
  for (const auto &name_2_fusion_reg : name_2_fusion_pass_regs_) {
    if (name_2_fusion_reg.second.GetStage() == stage) {
      fusion_passes.emplace_back(name_2_fusion_reg.second);
      GELOGI("Got Fusion pass:%s", name_2_fusion_reg.second.ToString().GetString());
    }
  }
  return fusion_passes;
}
}  // namespace fusion
}  // namespace ge