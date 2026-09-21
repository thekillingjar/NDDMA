/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_MANAGER_UTIL_GRAPH_REBUILD_STATE_CTRL_H_
#define GE_GRAPH_MANAGER_UTIL_GRAPH_REBUILD_STATE_CTRL_H_

#include <map>
#include <set>
#include <string>
#include <mutex>
#include "graph/compute_graph.h"

namespace ge {
class GraphRebuildStateCtrl {
 public:
  void AddGraph(uint32_t graph_id, const ComputeGraphPtr &compute_graph);
  void RemoveGraph(uint32_t graph_id);
  bool IsGraphNeedRebuild(uint32_t graph_id) const;
  void SetGraphBuildEnd(uint32_t graph_id);

  /// If resource(variable format/resource_op shape) changed, call this func to tell ctrl
  /// @param resource_name
  void SetStateChanged(const std::string &resource_name);

  /// Add changeable resource name. Changeable resource can be variable or resource_op.
  /// \param graph_id
  /// \param resource_name. resource_op: (graph_id + resource_op_name)
  ///                       variable: (variable_op_name)
  void AddResourceName(uint32_t graph_id, const std::string &resource_name);

  /// In order to prevent the variable format from being repeatedly changed
  /// between different formats, we simply limited the variable format to
  /// only one time of changing
  /// \param var_name
  /// \return
  bool IsVarPermitToChangeFormats(const std::string &var_name);

 private:
  /// The graph id of the graph to be rebuilt. When the format of a variable is
  /// changed, the graph which contains this variable is needs to be rebuilt.
  std::set<uint32_t> graph_ids_need_rebuild_;
  /// the mutable resource(variable/resource op) and graph relationships will construct when `AddGraph`
  std::map<uint32_t, std::set<std::string>> graph_ids_to_resource_names_;
  /// Number of variable names and their format changes.
  /// In order to prevent the variable format from being repeatedly changed
  /// between different formats, we simply limited the variable format to
  /// only one time of changing
  std::map<std::string, int32_t> resource_names_to_change_times_;
  static const int32_t kMaxVarChangeTimes_ = 1;
  mutable std::mutex mutex_;
};
}  // namespace ge

#endif  // GE_GRAPH_MANAGER_UTIL_GRAPH_REBUILD_STATE_CTRL_H_
