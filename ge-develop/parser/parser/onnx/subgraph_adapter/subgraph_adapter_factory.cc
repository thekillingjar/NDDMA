/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "subgraph_adapter_factory.h"
#include "framework/common/debug/ge_log.h"

namespace ge {
SubgraphAdapterFactory* SubgraphAdapterFactory::Instance() {
  static SubgraphAdapterFactory instance;
  return &instance;
}

std::shared_ptr<SubgraphAdapter> SubgraphAdapterFactory::CreateSubgraphAdapter(
    const std::string &op_type) {
  // First look for CREATOR_FUN based on OpType, then call CREATOR_FUN to create SubgraphAdapter.
  std::map<std::string, CREATOR_FUN>::const_iterator iter = subgraph_adapter_creator_map_.find(op_type);
  if (iter != subgraph_adapter_creator_map_.end()) {
    return iter->second();
  }

  GELOGW("SubgraphAdapterFactory::CreateSubgraphAdapter: Not supported type: %s", op_type.c_str());
  return nullptr;
}

// This function is only called within the constructor of the global SubgraphAdapterRegisterar object,
// and does not involve concurrency, so there is no need to lock it
void SubgraphAdapterFactory::RegisterCreator(const std::string &type, CREATOR_FUN fun) {
  std::map<std::string, CREATOR_FUN> *subgraph_adapter_creator_map = &subgraph_adapter_creator_map_;
  GELOGD("SubgraphAdapterFactory::RegisterCreator: op type:%s.", type.c_str());
  (*subgraph_adapter_creator_map)[type] = fun;
}
}  // namespace ge
