/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PASSES_MERGE_TO_STREAM_MERGE_PASS_H_
#define GE_GRAPH_PASSES_MERGE_TO_STREAM_MERGE_PASS_H_

#include "graph/passes/graph_pass.h"

namespace ge {
class MergeToStreamMergePass : public GraphPass {
 public:
  Status Run(ComputeGraphPtr graph);

 private:
  /// @brief Replace Merge Op
  /// @param [in] graph
  /// @param [in] merge_node
  /// @return Status
  Status ReplaceMergeNode(const ComputeGraphPtr &graph, const NodePtr &merge_node) const;

  /// @brief Add StreamActive Op as StreamMerge in_node
  /// @param [in] graph
  /// @param [in] node
  /// @return Status
  Status AddActiveNodes(const ComputeGraphPtr &graph, const NodePtr &node) const;

  /// @brief Create Active Op
  /// @param [in] graph
  /// @param [in] node
  /// @return ge::NodePtr
  NodePtr CreateActiveNode(const ComputeGraphPtr &graph, const NodePtr &node) const;

  std::set<NodePtr> bypass_nodes_;
};
}  // namespace ge
#endif  // GE_GRAPH_PASSES_MERGE_TO_STREAM_MERGE_PASS_H_
