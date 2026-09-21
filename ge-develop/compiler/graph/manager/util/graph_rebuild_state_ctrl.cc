/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_rebuild_state_ctrl.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/types.h"

namespace ge {
namespace {
inline bool IsVariable(const std::string &node_type) {
  return node_type == ge::VARIABLE || node_type == ge::VARIABLEV2 || node_type == ge::VARHANDLEOP;
}
}
void GraphRebuildStateCtrl::AddGraph(uint32_t graph_id, const ComputeGraphPtr &compute_graph) {
  if (compute_graph == nullptr) {
    GELOGE(PARAM_INVALID, "[Check][Param] Failed to add graph %u, the compute graph is null", graph_id);
    return;
  }
  // add variable in cur graph
  std::set<std::string> tmp_var_names;
  for (auto &node : compute_graph->GetAllNodes()) {
    auto node_type = node->GetType();
    if (IsVariable(node_type)) {
      GELOGD("Add graph %u contains variable op, name: %s", graph_id, node->GetName().c_str());
      tmp_var_names.insert(node->GetName());
    }
  }

  std::lock_guard<std::mutex> lock(mutex_);
  auto &var_names = graph_ids_to_resource_names_[graph_id];
  var_names = std::move(tmp_var_names);

  GELOGD("Add graph %u, var count %zu", graph_id, var_names.size());
}

void GraphRebuildStateCtrl::RemoveGraph(uint32_t graph_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  GELOGD("Remove graph %u", graph_id);
  graph_ids_to_resource_names_.erase(graph_id);
  graph_ids_need_rebuild_.erase(graph_id);
}

bool GraphRebuildStateCtrl::IsGraphNeedRebuild(uint32_t graph_id) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return graph_ids_need_rebuild_.count(graph_id) > 0;
}

void GraphRebuildStateCtrl::SetGraphBuildEnd(uint32_t graph_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  graph_ids_need_rebuild_.erase(graph_id);
  GELOGD("The graph %u has built end, remove it from the rebuild-set", graph_id);
}

void GraphRebuildStateCtrl::AddResourceName(uint32_t graph_id, const std::string &resource_name) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto &resource_keys = graph_ids_to_resource_names_[graph_id];
  resource_keys.insert(resource_name);
  GELOGI("The resource %s of graph %u added to ctrl.", resource_name.c_str(), graph_id);
}

bool GraphRebuildStateCtrl::IsVarPermitToChangeFormats(const std::string &var_name) {
  std::lock_guard<std::mutex> lock(mutex_);
  const std::map<std::string, int32_t>::const_iterator &iter = resource_names_to_change_times_.find(var_name);
  if (iter == resource_names_to_change_times_.end()) {
    return true;
  }
  return iter->second < kMaxVarChangeTimes_;
}

void GraphRebuildStateCtrl::SetStateChanged(const std::string &resource_name) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto times = ++resource_names_to_change_times_[resource_name];
  for (auto &graph_id_to_var_names : graph_ids_to_resource_names_) {
    if (graph_id_to_var_names.second.count(resource_name) > 0) {
      GELOGI("The resource %s has been changed, total changed times %d, "
          "the graph %u contains which should be re-build before next run",
          resource_name.c_str(), times, graph_id_to_var_names.first);
      /// The graph being compiled right now is also added to the rebuild-list
      /// and can be deleted by calling `SetGraphBuildEnd` at the end of compilation.
      graph_ids_need_rebuild_.insert(graph_id_to_var_names.first);
    }
  }
}
}  // namespace ge
