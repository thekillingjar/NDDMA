/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PASSES_MARK_FORCE_UNKNOWN_FOR_COND_PASS_H_
#define GE_GRAPH_PASSES_MARK_FORCE_UNKNOWN_FOR_COND_PASS_H_

#include "graph/passes/graph_pass.h"

#include <queue>

namespace ge {
class MarkForceUnknownForCondPass : public GraphPass {
 public:
  Status Run(ComputeGraphPtr graph);

 private:
  /// @brief Deal with Switch node for LoopCond
  /// @param [in] Switch node
  /// @param [in] dest span
  /// @param [out] Search queue
  /// @return true: Switch In while loop / false: Not in while Loop.
  bool DealAsLoopSwitch(const NodePtr &node, uint32_t dst_span,
                        std::queue<std::pair<NodePtr, uint32_t>> &search_queue) const;

  /// @brief Mark force unknown shape for Switch node
  /// @param [in] merge node
  /// @param [out] switch group
  /// @return
  void MarkUnknownForSwitch(const NodePtr &node, std::vector<NodePtr> &switch_group) const;

  /// @brief Mark force unknown shape for Switch node
  /// @param [in] switch groups
  /// @return
  void MarkUnknownForSwitch(const std::map<int64_t, std::vector<NodePtr>> &switch_groups) const;
};
} // namespace ge
#endif  // GE_GRAPH_PASSES_MARK_FORCE_UNKNOWN_FOR_COND_PASS_H_
