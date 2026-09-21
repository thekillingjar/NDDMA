/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fusion_config_manager/fusion_attr_manager.h"

namespace fe {
FusionAttrManager::FusionAttrManager(const FusionPriorityMgrPtr &fusion_priority_mgr_ptr) :
    fusion_priority_mgr_ptr_(fusion_priority_mgr_ptr) {}

Status FusionAttrManager::InitAlwaysGeneralizeOps(const FusionPassOrRule &pass_or_rule,
                                                  std::set<std::string> &ops) {
  auto pattern_fusion_base_pass_ptr = std::unique_ptr<PatternFusionBasePass>(
      dynamic_cast<PatternFusionBasePass *>(pass_or_rule.pass_desc.create_fn()));
  if (pattern_fusion_base_pass_ptr == nullptr) {
    REPORT_FE_ERROR("[FusionAttrManager][Initialize] PassType[%d] PassName[%s] is nullptr.",
                    pass_or_rule.type, pass_or_rule.name.c_str());
    return FAILED;
  }
  const auto &patterns = pattern_fusion_base_pass_ptr->GetPatterns();

  for (auto &pattern : patterns) {
    const auto &op_descs = pattern->GetOpDescs();
    for (const auto &op_desc : op_descs) {
      for (const auto &type : op_desc->types) {
        ops.emplace(type);
      }
    }
  }
  return SUCCESS;
}

Status FusionAttrManager::Initialize() {
  if (fusion_priority_mgr_ptr_ == nullptr) {
    return SUCCESS;
  }
  const std::vector<FusionPassOrRule> &sorted_pass_single_op_scene =
      fusion_priority_mgr_ptr_->GetSortedGraphFusionList(true);
  for (const FusionPassOrRule &pass_or_rule : sorted_pass_single_op_scene) {
    if (!IsPassAttrTypeOn(pass_or_rule.pass_desc.attr, PassAttrType::ALWAYS_GENERALIZE_FLAG)) {
      continue;
    }

    Status ret = InitAlwaysGeneralizeOps(pass_or_rule, always_generalize_ops_single_op_scene_);
    if (ret != SUCCESS) {
      return ret;
    }
  }

  const std::vector<FusionPassOrRule> &sorted_pass =
      fusion_priority_mgr_ptr_->GetSortedGraphFusionList(false);
  for (const FusionPassOrRule &pass_or_rule : sorted_pass) {
    if (!IsPassAttrTypeOn(pass_or_rule.pass_desc.attr, PassAttrType::ALWAYS_GENERALIZE_FLAG)) {
      continue;
    }

    Status ret = InitAlwaysGeneralizeOps(pass_or_rule, always_generalize_ops_);
    if (ret != SUCCESS) {
      return ret;
    }
  }
  return SUCCESS;
}

FusionAttrManager::~FusionAttrManager() {}

bool FusionAttrManager::IsAlwaysGeneralize(const string &op_type, bool is_single_op_scene) const {
  if (is_single_op_scene) {
    return always_generalize_ops_single_op_scene_.count(op_type);
  } else {
    return always_generalize_ops_.count(op_type);
  }
}
} // end namespace fe
