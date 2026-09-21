/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PASSES_NEXT_ITERATION_PASS_H_
#define GE_GRAPH_PASSES_NEXT_ITERATION_PASS_H_

#include "graph/passes/graph_pass.h"

struct LoopCondGroup {
  ge::NodePtr loop_cond;                                              // LoopCond node
  std::vector<ge::NodePtr> enter_nodes;                               // Enter nodes
  std::vector<std::pair<ge::NodePtr, ge::NodePtr>> merge_next_pairs;  // <Merge, NextIteration>
  std::vector<ge::NodePtr> switch_nodes;                              // Switch nodes
};
using LoopCondGroupPtr = std::shared_ptr<LoopCondGroup>;

namespace ge {
class NextIterationPass : public GraphPass {
 public:
  Status Run(ComputeGraphPtr graph);

  /// @brief Clear Status, used for subgraph pass
  /// @return SUCCESS
  Status ClearStatus() override;

 private:
  /// @brief Group Enter node
  /// @param [in] enter_node
  /// @return Status
  Status GroupEnterNode(const NodePtr &enter_node);

  /// @brief Find while groups
  /// @return Status
  Status FindWhileGroups();

  /// @brief Verify if valid
  /// @return bool
  bool VerifyWhileGroup() const;

  /// @brief Handle while group
  /// @param [in] graph
  /// @return Status
  Status HandleWhileGroup(ComputeGraphPtr &graph);

  /// @brief Create Active Node
  /// @param [in] graph
  /// @param [in] name
  /// @return ge::NodePtr
  NodePtr CreateActiveNode(ComputeGraphPtr &graph, const std::string &name) const;

  /// @brief Break NextIteration Link & add name to merge attr
  /// @param [in] next_node
  /// @param [in] merge_node
  /// @return Status
  Status BreakNextIteration(const NodePtr &next_node, NodePtr &merge_node) const;

  /// @brief find target node
  /// @param [in] node
  /// @param [in] target_type
  /// @param [in] is_input
  /// @param [in] batch_label
  /// @param [out] target_node
  /// @return Status
  Status FindTargetNode(const NodePtr &node, const std::string &target_type, bool is_input, NodePtr &target_node) const;

  /// @brief Mark force unknown for Exit node
  /// @param [in] group of LoopCond
  /// @param [in] index of LoopCond Node
  /// @return void
  void HandleSwitchExitNodes(const LoopCondGroup &loop_group, int64_t group_index) const;

  // std::map<frame_name, LoopCondGroup>
  std::unordered_map<std::string, LoopCondGroupPtr> loop_group_map_;
};
}  // namespace ge
#endif  // GE_GRAPH_PASSES_NEXT_ITERATION_PASS_H_
