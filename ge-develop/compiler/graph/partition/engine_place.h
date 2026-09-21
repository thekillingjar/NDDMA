/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PARTITION_ENGINE_PLACE_H_
#define GE_GRAPH_PARTITION_ENGINE_PLACE_H_

#include <unordered_map>
#include <mutex>

#include "framework/common/ge_inner_error_codes.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include "graph/compute_graph.h"

namespace ge {
using NodeEngineMap = std::unordered_map<ConstNodePtr, std::string>;

class EngineReAssignPass;

///
/// @ingroup graph/partition
/// @brief Assigned individual DNNEngine to each node in the origin graph
/// @author
///
class EnginePlacer {
 public:
  explicit EnginePlacer(const ComputeGraphPtr &graph) : compute_graph_(graph) {}
  EnginePlacer() : compute_graph_(nullptr), node_atomic_engine_map_{}, node_composite_engine_map_{} {};
  EnginePlacer(const EnginePlacer &obj) {
    compute_graph_ = obj.compute_graph_;
    node_atomic_engine_map_ = obj.node_atomic_engine_map_;
    node_composite_engine_map_ = obj.node_composite_engine_map_; 
  }
  EnginePlacer &operator=(const EnginePlacer &obj) {
    if (this == &obj) {
      return *this;
    }
    compute_graph_ = obj.compute_graph_;
    node_atomic_engine_map_ = obj.node_atomic_engine_map_;
    node_composite_engine_map_ = obj.node_composite_engine_map_;
    return *this;
  }
  ~EnginePlacer() = default;

  Status SelectEngine(const NodePtr &node, const std::set<std::string> &exclude_engines,
                      bool &is_check_support_success, OpInfo &matched_op_info);
  Status RunAllSubgraphs();
  Status Run(bool direct_node_flag = true);
  Status AssignCompositeEngine();
  Status ReAssignEngine();
  Status RunHostcpuEngineUpdatePass();

  // Get the unique node-engine map
  const NodeEngineMap &GetNodeEngineMap(bool is_composite_engine_mode) const;

  void SetComputeGraph(const ComputeGraphPtr &compute_graph) { compute_graph_ = compute_graph; }

 private:
  Status Check() const;
  Status RunSinglePass(const std::string &pass_name, const std::shared_ptr<EngineReAssignPass> &pass);
  static void GetOpInfoByOpDesc(const OpDescPtr &op_desc, const std::string &attr_engine_name,
                                const std::string &attr_kernel_name, OpInfo &matched_op_info);
  static void UpdateOpdescWithAttr(const OpDescPtr &op_desc, const std::string &attr_engine_name,
                                   const std::string &attr_kernel_name, OpInfo &matched_op_info);

  ComputeGraphPtr compute_graph_;
  NodeEngineMap node_atomic_engine_map_;
  NodeEngineMap node_composite_engine_map_;
  mutable std::mutex mutex_;
};

class EngineReAssignPass {
 public:
  EngineReAssignPass() = default;
  virtual ~EngineReAssignPass() = default;
  virtual Status Run(const ComputeGraphPtr &graph,
                     NodeEngineMap &node_atomic_engine_map,
                     NodeEngineMap &node_composite_engine_map) = 0;
};
}  // namespace ge

#endif  // GE_GRAPH_PARTITION_ENGINE_PLACE_H_
