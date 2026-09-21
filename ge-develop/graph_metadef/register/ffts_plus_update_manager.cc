/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/ffts_plus_update_manager.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "graph_metadef/common/plugin/plugin_manager.h"

namespace ge {
FftsPlusUpdateManager &FftsPlusUpdateManager::Instance() {
  static FftsPlusUpdateManager instance;
  return instance;
}

FftsCtxUpdatePtr FftsPlusUpdateManager::GetUpdater(const std::string &core_type) const {
  const std::map<std::string, FftsCtxUpdateCreatorFun>::const_iterator it = creators_.find(core_type);
  if (it == creators_.cend()) {
    GELOGW("Cannot find creator for core type: %s.", core_type.c_str());
    return nullptr;
  }

  return it->second();
}

void FftsPlusUpdateManager::RegisterCreator(const std::string &core_type, const FftsCtxUpdateCreatorFun &creator) {
  if (creator == nullptr) {
    GELOGW("Register null creator for core type: %s", core_type.c_str());
    return;
  }

  const auto it = creators_.find(core_type);
  if (it != creators_.end()) {
    GELOGW("Creator already exist for core type: %s", core_type.c_str());
    return;
  }

  GELOGI("Register creator for core type: %s", core_type.c_str());
  creators_[core_type] = creator;
}

Status FftsPlusUpdateManager::Initialize() {
  return SUCCESS;
}

FftsPlusUpdateManager::~FftsPlusUpdateManager() {
  creators_.clear();  // clear must be called before `plugin_manager_.reset` which would close so
  plugin_manager_.reset();
}
}  // namespace ge
