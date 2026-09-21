/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PASSES_ATTACH_STREAM_LABEL_PASS_H_
#define GE_GRAPH_PASSES_ATTACH_STREAM_LABEL_PASS_H_

#include <stack>
#include "graph/passes/graph_pass.h"

namespace ge {
class AttachStreamLabelPass : public GraphPass {
 public:
  explicit AttachStreamLabelPass(bool after_optimize_subgraph = false)
      : after_optimize_subgraph_(after_optimize_subgraph) {}
  Status Run(ComputeGraphPtr graph);

 private:
  /// @brief Find StreamSwitch / StreamMerge / Enter node
  /// @param [in] graph
  /// @param [out] need_label_nodes
  /// @param [out] enter_nodes
  /// @param [out] branch_head_nodes
  /// @param [out] cmo_nodes
  /// @return void
  void FindNodes(const ComputeGraphPtr &graph, std::vector<NodePtr> &need_label_nodes,
                 std::vector<NodePtr> &enter_nodes, std::map<NodePtr, NodePtr> &branch_head_nodes,
                 std::vector<NodePtr> &cmo_nodes) const;

  /// @brief update cond branch
  /// @param [in] node
  /// @param [in] branch_head_nodes
  /// @return Status
  Status UpdateCondBranch(const NodePtr &node, const std::map<NodePtr, NodePtr> &branch_head_nodes) const;

  /// @brief attach flag
  /// @param [in] node
  /// @param [out] stream_label
  /// @return Status
  static Status AttachFlag(const NodePtr &node, std::string &stream_label);

  /// @brief Update stream_label for loop_branch
  /// @param [in] enter_nodes
  /// @param [in] stream_label
  /// @return Status
  static Status UpdateLoopBranch(const std::stack<NodePtr> &enter_nodes, const std::string &stream_label);

  /// @brief Update stream_label start with enter nodes
  /// @param [in] enter_nodes
  /// @return Status
  Status UpdateEnterNode(const std::vector<NodePtr> &enter_nodes) const;

  /// @brief Set stream_label for enter_nodes
  /// @param [in] enter_nodes
  /// @param [in] active_node
  /// @return Status
  static Status SetEnterLabel(const std::vector<NodePtr> &enter_nodes, const NodePtr &active_node);

  /// @brief Update stream_label in subgraph start with subgraph name to distinguish same stream_label added in
  ///        subgraph optimization
  /// @param [in] enter_nodes
  /// @return Status
  Status UpdateSubgraphStreamLabel(const ComputeGraphPtr &graph) const;

  Status SetCmoStreamLabel(const std::vector<NodePtr> &cmo_nodes) const;

  bool after_optimize_subgraph_ = false;
};
}  // namespace ge
#endif  // GE_GRAPH_PASSES_ATTACH_STREAM_LABEL_PASS_H_
