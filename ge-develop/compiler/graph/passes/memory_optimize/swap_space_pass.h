/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PASSES_SWAP_SPACE_PASS_H_
#define GE_GRAPH_PASSES_SWAP_SPACE_PASS_H_

#include "graph/passes/graph_pass.h"
#include "runtime/mem.h"

#include <vector>

namespace ge {
class SwapSpacePass : public GraphPass {
 public:
  Status Run(ComputeGraphPtr graph) override;

 private:
  struct SwapInfo {
    std::string node;
    std::map<int32_t, std::vector<std::string>> output_to_swaps;
  };
  Status GetAllSwapCandidates(const ComputeGraphPtr &graph, std::map<NodePtr, SwapInfo> &swapping_candidates) const;
  Status SwapOutProcess(const std::pair<NodePtr, SwapInfo> &swapping_candidate,
                        std::vector<NodePtr> &d2h_mem_cpy_nodes) const;
  Status SwapInProcess(const ComputeGraphPtr &graph, const NodePtr &node);
  Status InsertSpaceCopyNodes(const ComputeGraphPtr &graph, const std::map<NodePtr, SwapInfo> &swapping_candidates);
  static Status PrepareForMemAssign(const NodePtr &mem_copy_node, const rtMemcpyKind_t rt_memcpy_kind);
  static Status CheckNodeCouldSwap(const NodePtr &node);
};
}  // namespace ge
#endif  // GE_GRAPH_PASSES_SWAP_SPACE_PASS_H_
