/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "task_node_map.h"
#include "common/checker.h"
#include "framework/common/debug/ge_log.h"
#include "common/util/mem_utils.h"
#include "graph/compute_graph.h"
namespace ge {
namespace {
std::vector<size_t> g_empty;
NodePtr BuildNoNode() {
  const auto graph = MakeShared<ComputeGraph>("g");
  GE_ASSERT_NOTNULL(graph);
  return graph->AddNode(MakeShared<OpDesc>("op_index_-1_node", "Unknown"));
}
}  // namespace
Status TaskNodeMap::Init(const ComputeGraphPtr &graph, size_t task_num) {
  auto no_node = BuildNoNode();
  GE_ASSERT_NOTNULL(no_node);
  node_ids_to_node_[-1] = std::move(no_node);

  for (const auto &node : graph->GetAllNodes()) {
    node_ids_to_node_[node->GetOpDesc()->GetId()] = node;
    GELOGD("node index %lld, name %s, type %s", node->GetOpDesc()->GetId(), node->GetName().c_str(),
           node->GetType().c_str());
  }

  task_indexes_to_node_.resize(task_num);
  return SUCCESS;
}
Status TaskNodeMap::AddRelation(size_t task_index, int64_t node_id) {
  const auto iter = node_ids_to_node_.find(node_id);
  if (iter == node_ids_to_node_.cend()) {
    GELOGE(PARAM_INVALID, "Failed to add relation between task %zu and node %lld, invalid node id", task_index,
           node_id);
    return PARAM_INVALID;
  }
  if (task_index >= task_indexes_to_node_.size()) {
    GELOGE(PARAM_INVALID, "Failed to add relation between task %zu and node %lld(%s), task index out of range",
           task_index, node_id, iter->second->GetName().c_str());
    return PARAM_INVALID;
  }
  node_ids_to_task_indexes_[node_id].emplace_back(task_index);
  // 调用者保证task_indexes_to_node_总是够长
  task_indexes_to_node_[task_index] = {node_id, iter->second};
  return SUCCESS;
}
const TaskNodeMap::NodeInfo &TaskNodeMap::FindNodeByTaskIndex(size_t task_index) const {
  return task_indexes_to_node_.at(task_index);
}
const std::vector<size_t> &TaskNodeMap::FindTasksByNodeId(int64_t node_id) const {
  const auto iter = node_ids_to_task_indexes_.find(node_id);
  if (iter == node_ids_to_task_indexes_.end()) {
    return g_empty;
  }
  return iter->second;
}
}  // namespace ge
