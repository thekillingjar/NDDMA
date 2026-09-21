/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PASSES_COND_REMOVE_PASS_H
#define GE_GRAPH_PASSES_COND_REMOVE_PASS_H

#include "graph/passes/base_pass.h"

namespace ge {
class CondRemovePass : public BaseNodePass {
 public:
  Status Run(NodePtr &node) override;

  /// @brief Remove if dead branch, for if
  static Status GetIfChosenBranch(const NodePtr &node, const uint32_t cond, ComputeGraphPtr &compute_graph);

  /// @brief Remove if dead branch, for case
  static Status GetCaseChosenBranch(const NodePtr &node, const uint32_t cond_index, ComputeGraphPtr &compute_graph);

  static int32_t GetCondIndex(const GeTensor *tensor);

  /// @brief Remove dead condition input, for if / case / while
  static Status RemoveDeadCondLink(const int32_t index, const NodePtr &node);

  /// @brief Remove if dead branch, for if
  static Status ReplaceIfCaseNodeWithPartitioncall(const NodePtr &node, const ComputeGraphPtr &save_branch);

 private:
  /// @brief Get cond info for if/case node
  /// @param [in] node: If/Case op
  /// @param [out] graph: owner_graph of if node
  /// @param [out] cond_out_anchor: peer_cond_anchor
  /// @param [out] cond_in_anchor: cond_input of if
  /// @return Status
  Status GetCondInfoForIfCase(const NodePtr &node, ComputeGraphPtr &graph, OutDataAnchorPtr &cond_out_anchor,
                              InDataAnchorPtr &cond_in_anchor) const;

  /// @brief Get cond info for if/case node
  /// @param [in] node: If/Case op
  /// @param [out] graph: owner_graph of if node
  /// @param [out] cond_out_anchor: peer_cond_anchor
  /// @param [out] cond_in_anchor: cond_input of if
  /// @return Status
  Status GetCondInfo(const NodePtr &node, ComputeGraphPtr &graph, OutDataAnchorPtr &cond_out_anchor,
                     InDataAnchorPtr &cond_in_anchor);

  /// @brief Check if condition input is const, for if / case / while
  bool CheckIfCondConstInput(const OutDataAnchorPtr &cond_out_anchor, const InDataAnchorPtr &cond_in_anchor,
                             int32_t &cond_index) const;


  static OpDescPtr CreateSubgraphOpDesc(const NodePtr &node, const std::string &name, size_t input_num,
                                 size_t output_num);
};
}  // namespace ge
#endif  // GE_GRAPH_PASSES_COND_REMOVE_PASS_H
