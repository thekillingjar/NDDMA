/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PASSES_CONTROL_TRIGGER_PASS_H_
#define GE_GRAPH_PASSES_CONTROL_TRIGGER_PASS_H_

#include <set>
#include <string>
#include <unordered_map>
#include <utility>

#include "graph/passes/graph_pass.h"

namespace ge {
enum ControlNodeType {
  kNotControlOp,
  kCondSwitch,
  kCondMerge,
  kLoopSwitchT,
  kLoopSwitchF,
  kEnter,
  kInvalidType
};

class ControlTriggerPass : public GraphPass {
 public:
  Status Run(ComputeGraphPtr graph);
  Status ClearStatus() override;

 private:
  Status HandleDynamicCtrlEdges(ComputeGraphPtr &graph, NodePtr &node, NodePtr &in_ctrl_node);
  Status FindSwitchNode(const NodePtr &node, NodePtr &switch_node, bool &branch_flag);
  ControlNodeType TransferNodeType(const NodePtr &node, uint32_t index);
  void GetInNodes(const NodePtr &node, std::set<std::pair<NodePtr, uint32_t>> &in_nodes) const;
  Status InsertOppositeBranch(ComputeGraphPtr &graph, NodePtr &node, NodePtr &in_ctrl_node, NodePtr &switch_node,
                              bool branch_flag);
  NodePtr InsertMergeNode(ComputeGraphPtr &graph, NodePtr &node, NodePtr &in_ctrl_node, const GeTensorDesc &data_desc,
                          const NodePtr &cond_node) const;
  NodePtr InsertConstNode(ComputeGraphPtr &graph, NodePtr &merge_node, const GeTensorDesc &data_desc, bool flag) const;
  NodePtr InsertIdentityNode(ComputeGraphPtr &graph, const std::string &name,
                             const GeTensorDesc &data_desc, NodePtr &switch_node) const;
  Status FindPredInput(const NodePtr &switch_node);

  // <switch_node, pred_node>
  std::unordered_map<NodePtr, NodePtr> switch_cond_map_;
  // <ControlTrigger, <pred_node, {const_f, const_t}>>
  std::unordered_map<NodePtr, std::unordered_map<NodePtr, std::pair<NodePtr, NodePtr>>> control_trigger_map_;
};
}  // namespace ge
#endif  // GE_GRAPH_PASSES_CONTROL_TRIGGER_PASS_H_
