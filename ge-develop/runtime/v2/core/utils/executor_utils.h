/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_CORE_UTILS_EXECUTOR_UTILS_H_
#define AIR_CXX_RUNTIME_V2_CORE_UTILS_EXECUTOR_UTILS_H_
#include "runtime/model_v2_executor.h"
#include "common/ge_inner_attrs.h"
#include "core/builder/graph_node.h"

namespace gert {
inline size_t CalcArgIndex(size_t total_num, ExecuteArgIndex arg_index) {
  size_t tensor_num = total_num - static_cast<size_t>(ExecuteArgIndex::kNum);
  return tensor_num + static_cast<size_t>(static_cast<int64_t>(arg_index) * -1 - 1);
}

inline void SetPriorityForNode(std::pair<ge::FastNode *, Node *> &graph_to_exe_nodes) {
  int64_t priority = std::numeric_limits<int64_t>::max();
  if (!ge::AttrUtils::GetInt(graph_to_exe_nodes.first->GetOpDescBarePtr(), "priority", priority)) {
    GELOGW("Can not get the priority of node %s, the node will be run at the lowest priority",
           graph_to_exe_nodes.first->GetName().c_str());
  }
  reinterpret_cast<PriorityQueueElementHead *>(graph_to_exe_nodes.second)->priority = priority;
}

// For node whose kernel require unusual runtime extra info
// e.g. SwitchNotify kernel request 'Watcher' at runtime
inline ge::graphStatus AssembleNodeRequestedExtraInfos(const std::pair<ge::FastNode *, Node *> &node_to_exe_node,
                                                       const GraphNode &graph_node) {
  auto &org_node = node_to_exe_node.first;
  auto &exe_node = node_to_exe_node.second;
  bool request_watcher = false;
  const auto op_desc = org_node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  if (ge::AttrUtils::GetBool(op_desc, ge::kRequestWatcher, request_watcher) && request_watcher) {
    GELOGD("Assemble attr '%s' to node %s type %s", ge::kWatcherAddress, org_node->GetNamePtr(),
           org_node->GetTypePtr());
    int64_t address_buffer = 0;
    auto &watcher = graph_node.node_watchers[exe_node->node_id];
    auto ret = memcpy_s(&address_buffer, sizeof(address_buffer), &watcher, sizeof(Watcher *));
    GE_ASSERT_EOK(ret, "Failed copy watcher address to dst buffer, ret:%d", ret);

    GE_ASSERT_TRUE(ge::AttrUtils::SetInt(op_desc, ge::kWatcherAddress, address_buffer),
                   "Failed set watcher for request-watcher node %s", org_node->GetName().c_str());
  }
  return ge::GRAPH_SUCCESS;
}
}
#endif  // AIR_CXX_RUNTIME_V2_CORE_UTILS_EXECUTOR_UTILS_H_
