/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PASSES_MERGE_ADD_INPUT_MEMCPY_PASS_H_
#define GE_GRAPH_PASSES_MERGE_ADD_INPUT_MEMCPY_PASS_H_

#include "graph/passes/graph_pass.h"

namespace ge {
class MergeInputMemcpyPass : public GraphPass {
 public:
  Status Run(ComputeGraphPtr graph);

 private:
  /// @brief Add MemcpyAsync Op as Merge in_node
  /// @param [in] graph
  /// @param [in] node
  /// @param [in] multi_batch_flag
  /// @return Status
  Status AddMemcpyAsyncNodes(const ComputeGraphPtr &graph, const NodePtr &node, bool multi_batch_flag) const;

  /// @brief Add MemcpyAsync Node
  /// @param [in] graph
  /// @param [in] name
  /// @param [in] out_data_anchor
  /// @param [in] multi_batch_flag
  /// @return ge::NodePtr
  NodePtr CreateMemcpyAsyncNode(const ComputeGraphPtr &graph, const std::string &name,
                                const OutDataAnchorPtr &out_data_anchor, bool multi_batch_flag) const;
};
}  // namespace ge
#endif  // GE_GRAPH_PASSES_MERGE_ADD_INPUT_MEMCPY_PASS_H_
