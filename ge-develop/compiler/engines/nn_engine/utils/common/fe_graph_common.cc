/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/fe_graph_common.h"
#include "common/fe_log.h"
#include "common/aicore_util_attr_define.h"
#include "graph/debug/ge_attr_define.h"

namespace fe {
bool FeGraphCommon::IsNodeOfUnknownRootGraph(const ge::Node &node) {
  if (node.GetType() == "MemSet") {
    ge::NodePtr original_node = nullptr;
    original_node = node.GetOpDesc()->TryGetExtAttr(ATTR_NAME_ORIGINAL_NODE, original_node);
    if (original_node != nullptr) {
      return IsNodeOfUnknownRootGraph(*original_node);
    }
  }
  const auto &owner_graph = node.GetOwnerComputeGraph();
  FE_CHECK(owner_graph == nullptr, FE_LOGW("GetOwnerComputeGraph failed"), return false);
  const auto &src_graph = owner_graph->TryGetExtAttr(kPartSrcGraph, ge::ComputeGraphPtr());
  ge::ComputeGraphPtr root_graph;
  if (src_graph == nullptr) {
    root_graph = ge::GraphUtils::FindRootGraph(owner_graph);
  } else {
    root_graph = ge::GraphUtils::FindRootGraph(src_graph);
  }
  return root_graph->GetGraphUnknownFlag();
}

bool FeGraphCommon::IsNodeOfUnknownGraph(const ge::Node &node) {
  const auto &owner_graph = node.GetOwnerComputeGraph();
  FE_CHECK(owner_graph == nullptr, FE_LOGW("GetOwnerComputeGraph failed"), return false);
  bool is_dynamic = false;
  (void)ge::AttrUtils::GetBool(owner_graph, ge::ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, is_dynamic);
  return (is_dynamic || owner_graph->GetGraphUnknownFlag());
}

bool FeGraphCommon::IsUnknownGraph(const ge::ComputeGraphPtr &graph_ptr) {
  FE_CHECK(graph_ptr == nullptr, FE_LOGW("IsUnknownGraph check for graph_ptr was unsuccessful"), return false);
  auto root_graph = ge::GraphUtils::FindRootGraph(graph_ptr);
  return root_graph->GetGraphUnknownFlag();
}
} // namespace fe
