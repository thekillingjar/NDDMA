/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "framework/common/host_resource_center/host_resource_center.h"
#include "common/checker.h"
#include "common/plugin/ge_make_unique_util.h"
#include "graph/compute_graph.h"
#include "common/host_resource_center/weight_manager.h"
#include "common/host_resource_center/tiling_resource_manager.h"
namespace ge {
HostResourceCenter::HostResourceCenter() {
  resource_managers_[static_cast<size_t>(HostResourceType::kWeight)] = MakeUnique<WeightManager>();
  resource_managers_[static_cast<size_t>(HostResourceType::kTilingData)] = MakeUnique<TilingResourceManager>();
}

const HostResourceManager *HostResourceCenter::GetHostResourceMgr(HostResourceType type) const {
  GE_ASSERT_TRUE(static_cast<size_t>(type) < resource_managers_.size());
  return resource_managers_[static_cast<size_t>(type)].get();
}

Status HostResourceCenter::TakeOverHostResources(const ComputeGraphPtr &root_graph) {
  GE_ASSERT_TRUE(!resource_managers_.empty());
  for (const auto &node : root_graph->GetAllNodes()) {
    for (auto &resource_mgr : resource_managers_) {
      GE_ASSERT_NOTNULL(resource_mgr);
      GE_ASSERT_GRAPH_SUCCESS(resource_mgr->TakeoverResources(node->GetOpDesc()));
    }
  }
  return SUCCESS;
}
}  // namespace ge