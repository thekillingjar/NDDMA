/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PASSES_ASSIGN_REMOVE_PASS_H_
#define GE_GRAPH_PASSES_ASSIGN_REMOVE_PASS_H_

#include "graph/passes/base_pass.h"

namespace ge {
class AssignRemovePass : public BaseNodePass {
 public:
  Status Run(NodePtr &node) override;

 private:
  /// @brief Optimize for assign_node
  /// @param [in] assign_node
  /// @return Status
  Status OptimizedAssignNode(NodePtr &assign_node);

  /// @brief Transform assign_var_name attr
  /// @param [in] node
  /// @return Status
  Status TransformAttr(NodePtr &node);

  /// @brief Check if need optimize for assign_node
  /// @param [in] assign_node
  /// @param [in] peer_data_anchor for ref_input of assign_node
  /// @param [in] peer_data_anchor for value_input of assign_node
  /// @return Status
  static bool IsCondMatch(const NodePtr &node, const OutDataAnchorPtr &ref_peer_anchor,
                          const OutDataAnchorPtr &value_peer_anchor);
};
}  // namespace ge
#endif  // GE_GRAPH_PASSES_ASSIGN_REMOVE_PASS_H_
