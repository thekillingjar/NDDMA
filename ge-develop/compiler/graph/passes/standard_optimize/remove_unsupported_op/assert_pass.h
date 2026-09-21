/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PASSES_ASSERT_PASS_H_
#define GE_GRAPH_PASSES_ASSERT_PASS_H_

#include <vector>

#include "graph/passes/base_pass.h"

namespace ge {
class AssertPass : public BaseNodePass {
 public:
  Status Run(NodePtr& node) override;

 private:
  /// collect assert and other unused ops
  /// @param assert_node assert node
  /// @param nodes_unused nodes to be deleted
  /// @return void
  void CollectUnusedNode(const NodePtr &assert_node, std::vector<ge::NodePtr>& nodes_unused) const;

  /// remove unused nodes from graph
  /// @param graph
  /// @param nodes_unused nodes to be deleted
  /// @return Status
  Status RemoveUnusedNode(std::vector<NodePtr>& nodes_unused);
};
}  // namespace ge
#endif  // GE_GRAPH_PASSES_ASSERT_PASS_H_
