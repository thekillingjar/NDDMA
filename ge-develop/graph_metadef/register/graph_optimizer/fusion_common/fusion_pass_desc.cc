/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/graph_optimizer/fusion_common/fusion_pass_desc.h"
#include "ge_common/debug/ge_log.h"
#include "register/optimization_option_registry.h"
namespace fe {
bool IsPassAttrTypeOn(PassAttr pass_attr, PassAttrType attr_type) {
  return ((pass_attr >> static_cast<PassAttr>(attr_type)) & PASS_BIT_MASK) == 1;
}
void RegPassCompileLevel(const std::string &pass_name, PassAttr pass_attr) {
  std::vector<ge::OoLevel> levels;
  if (IsPassAttrTypeOn(pass_attr, PassAttrType::COMPILE_O0)) {
    (void)levels.emplace_back(ge::OoLevel::kO0);
  }
  if (IsPassAttrTypeOn(pass_attr, PassAttrType::COMPILE_O1)) {
    (void)levels.emplace_back(ge::OoLevel::kO1);
  }
  if (IsPassAttrTypeOn(pass_attr, PassAttrType::COMPILE_O2)) {
    (void)levels.emplace_back(ge::OoLevel::kO2);
  }
  if (IsPassAttrTypeOn(pass_attr, PassAttrType::COMPILE_O3)) {
    (void)levels.emplace_back(ge::OoLevel::kO3);
  }
  if (levels.empty()) {
    return;
  }
  const uint64_t level_bits = ge::OoInfoUtils::GenOptLevelBits(levels);
  GELOGD("Fusion name [%s] registered with compile level %lu.", pass_name.c_str(), level_bits);
  ge::OoInfo opt{pass_name, ge::OoHierarchy::kH1, level_bits};
  ge::OptionRegistry::GetInstance().Register(opt);
  ge::PassOptionRegistry::GetInstance().Register(pass_name, {{ge::OoHierarchy::kH1, opt.name}});
  return;
}
}
