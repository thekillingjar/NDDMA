/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PASSES_HOSTCPU_ENGINE_UPDATE_PASS_H_
#define GE_GRAPH_PASSES_HOSTCPU_ENGINE_UPDATE_PASS_H_

#include <unordered_map>
#include "graph/passes/graph_pass.h"
#include "graph/partition/engine_place.h"

namespace ge {
class HostcpuEngineUpdatePass : public EngineReAssignPass {
public:
  Status Run(const ComputeGraphPtr &graph,
             NodeEngineMap &node_atomic_engine_map,
             NodeEngineMap &node_composite_engine_map) override;

private:
  void ClearNodeMapSource();
  bool IsExecOnHost(const NodePtr &node);
  NodePtr GetParentNode(const NodePtr &node);
  bool CheckInputForHostExec(const NodePtr &node, const size_t idx);

  bool CheckAndMarkHostExec(const NodePtr &node, NodeEngineMap &node_atomic_engine_map,
                            NodeEngineMap &node_composite_engine_map);
  Status FindHostDataOfSubgraph(const ComputeGraphPtr &graph, std::deque<NodePtr> &q);
  Status FindHostDataOfPartitionCall(const ComputeGraphPtr &graph, std::deque<NodePtr> &q);
  Status FindHostInputOfControlV2Node(const NodePtr &node, std::deque<NodePtr> &q);
  Status FindAndMarkHostCpuNode(std::deque<NodePtr> &q, NodeEngineMap &node_atomic_engine_map,
                                NodeEngineMap &node_composite_engine_map);
  Status UpdateHostcpuEngine(const ComputeGraphPtr &graph, NodeEngineMap &node_atomic_engine_map,
                             NodeEngineMap &node_composite_engine_map, bool is_partition_call = false);

  std::set<NodePtr> host_exe_ops_;
  std::unordered_map<NodePtr, NodePtr> node_to_parent_node_;
  std::unordered_map<NodePtr, bool> is_node_execute_on_host_;
};
}  // namespace ge
#endif  // GE_GRAPH_PASSES_HOSTCPU_ENGINE_UPDATE_PASS_H_
