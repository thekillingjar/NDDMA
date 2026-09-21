/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "prototype_pass_manager.h"

#include "common/util.h"
#include "framework/common/debug/ge_log.h"

namespace ge {
ProtoTypePassManager &ProtoTypePassManager::Instance() {
  static ProtoTypePassManager instance;
  return instance;
}

Status ProtoTypePassManager::Run(google::protobuf::Message *message, const domi::FrameworkType &fmk_type) const {
  GE_CHECK_NOTNULL(message);
  const auto &pass_vec = ProtoTypePassRegistry::GetInstance().GetCreateFnByType(fmk_type);
  for (const auto &pass_item : pass_vec) {
    std::string pass_name = pass_item.first;
    const auto &func = pass_item.second;
    GE_CHECK_NOTNULL(func);
    std::unique_ptr<ProtoTypeBasePass> pass = std::unique_ptr<ProtoTypeBasePass>(func());
    GE_CHECK_NOTNULL(pass);
    Status ret = pass->Run(message);
    if (ret != SUCCESS) {
      GELOGE(FAILED, "Run ProtoType pass:%s failed", pass_name.c_str());
      return ret;
    }
    GELOGD("Run ProtoType pass:%s success", pass_name.c_str());
  }
  return SUCCESS;
}
}  // namespace ge
