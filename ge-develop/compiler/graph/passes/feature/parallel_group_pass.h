/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PASSES_PARALLEL_GROUP_PASS_H
#define GE_GRAPH_PASSES_PARALLEL_GROUP_PASS_H

#include <map>
#include <unordered_set>
#include "graph/graph.h"
#include "graph/passes/graph_pass.h"

namespace ge {
class ParallelGroupPass : public GraphPass {
 public:
  Status Run(ComputeGraphPtr graph) override;
 private:
  Status ProcessGraphGroupNodes(ComputeGraphPtr graph, std::unordered_set<std::string> &parallel_groups) const;
  Status ProcessOnGraph(const ComputeGraphPtr &graph,
                        std::map<ge::NodePtr, std::unordered_set<std::string>> &pnodes_2_parallel_groups) const;

  Status AddCtrlEdge(NodePtr pre_node, NodePtr cur_node) const;

  Status ReplaceWithSwitchAndMerge(NodePtr pre_node, NodePtr cur_node,
      const std::map<NodePtr, std::pair<std::set<NodePtr>, NodePtr>> &node_2_switch_merge) const;

  bool HasSameSwitch(const std::set<NodePtr> &switch_set1, const std::set<NodePtr> &switch_set2) const;

  Status ProcessGroupNodeInSwitch(ComputeGraphPtr graph,
                                  std::map<NodePtr, std::pair<std::set<NodePtr>, NodePtr>> &node_2_switch_merge) const;

  void FindGroupNodeAndMerge(NodePtr stream_switch_node, std::set<NodePtr> &group_nodes,
                             std::vector<NodePtr> &merge_nodes, std::set<std::string> &stream_labels) const;

  Status MappingNodeToSwitchAndMerge(const std::set<NodePtr> &group_nodes, const std::vector<NodePtr> &merge_nodes,
                                     const NodePtr &cast_node, const NodePtr &switch_node,
                                     std::map<NodePtr, std::pair<std::set<NodePtr>,
                                     NodePtr>> &node_2_switch_merge) const;

  bool IsBigSmallLoopStreamSwitch(OpDescPtr switch_op_desc) const;
  bool IsWhileStreamSwitch(OpDescPtr switch_op_desc) const;
  bool IsIndirectConnect(const NodePtr &node_a, const NodePtr &node_b) const;
};
}  // namespace ge
#endif // GE_GRAPH_PASSES_PARALLEL_GROUP_PASS_H
