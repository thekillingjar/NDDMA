/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PASSES_MEM_CHECK_PASS_H_
#define GE_GRAPH_PASSES_MEM_CHECK_PASS_H_
#include <map>
#include <vector>
#include <bitset>
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/types.h"
#include "graph/graph.h"
#include "graph/passes/graph_pass.h"
#include "graph/optimize/mem_layout_conflict_optimize/checker/checker.h"

namespace ge {
class MemLayoutConflictOptimizer {
 public:
  MemLayoutConflictOptimizer();

  ~MemLayoutConflictOptimizer();

  Status Run(ge::ComputeGraphPtr graph);

 private:
  std::unique_ptr<Checker> checker_;

  SymbolToAnchors symbol_to_anchors_;

  AnchorToSymbol anchor_to_symbol_;

  Status Process(ge::ComputeGraphPtr &graph);

  void MarkAllAttribute(const ComputeGraphPtr &graph);

  Status SolveConflict(ComputeGraphPtr &graph, AnchorSet &anchors) const;

  Status SolveInAnchorConflict(const ComputeGraphPtr &graph, const InDataAnchorPtr &in_anchor) const;

  Status SolveOutAnchorConflict(const ComputeGraphPtr &graph, const OutDataAnchorPtr &out_anchor) const;
};
}  // namespace ge
#endif  // GE_GRAPH_PASSES_MEM_CHECK_PASS_H_
