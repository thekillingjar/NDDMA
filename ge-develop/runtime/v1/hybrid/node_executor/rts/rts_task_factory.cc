/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hybrid/node_executor/rts/rts_task_factory.h"

namespace ge {
namespace hybrid {
RtsNodeTaskPtr RtsTaskFactory::Create(const std::string &task_type) const {
  const auto it = creators_.find(task_type);
  if (it == creators_.end()) {
    GELOGW("Cannot find task type %s in inner map.", task_type.c_str());
    return nullptr;
  }

  return it->second();
}

void RtsTaskFactory::RegisterCreator(const std::string &task_type, const RtsTaskCreatorFun &creator) {
  if (creator == nullptr) {
    GELOGW("Register %s creator is null", task_type.c_str());
    return;
  }

  const std::map<std::string, RtsTaskCreatorFun>::const_iterator it = creators_.find(task_type);
  if (it != creators_.cend()) {
    GELOGW("Task %s creator already exist", task_type.c_str());
    return;
  }

  creators_[task_type] = creator;
}
}  // namespace hybrid
}  // namespace ge
