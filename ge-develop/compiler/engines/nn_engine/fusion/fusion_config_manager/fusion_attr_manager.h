/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_FUSION_FUSION_CONFIG_MANAGER_FUSION_ATTR_MANAGER_H_
#define AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_FUSION_FUSION_CONFIG_MANAGER_FUSION_ATTR_MANAGER_H_
#include "fusion_config_manager/fusion_priority_manager.h"
#include <set>

namespace fe {
class FusionAttrManager {
 public:
  FusionAttrManager(const FusionPriorityMgrPtr &fusion_priority_mgr_ptr);
  Status Initialize();
  ~FusionAttrManager();

  bool IsAlwaysGeneralize(const string &op_type, bool is_single_op_scene) const;
 private:
  Status InitAlwaysGeneralizeOps(const FusionPassOrRule &pass_or_rule, std::set<std::string> &ops);

  FusionPriorityMgrPtr fusion_priority_mgr_ptr_;
  std::set<std::string> always_generalize_ops_;
  std::set<std::string> always_generalize_ops_single_op_scene_;
};
}
#endif // AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_FUSION_FUSION_CONFIG_MANAGER_FUSION_ATTR_MANAGER_H_
