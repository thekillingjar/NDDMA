/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_COMMON_SUBEXPRESSION_MIGRATION_H_
#define GE_COMMON_SUBEXPRESSION_MIGRATION_H_

#include "graph/types.h"
#include "graph/passes/graph_pass.h"

#include <map>
#include <set>
#include <vector>
#include <string>


namespace ge {
namespace {
using OrderedGraphToNodes = std::map<ComputeGraphPtr, std::map<uint32_t, NodePtr>, ComputeGraphCompareKey>;
}
class SubexpressionMigrationPass : public GraphPass {
 public:
  Status Run(ComputeGraphPtr graph) override;

 private:
  /// @ingroup ge
  /// @brief Get all Data nodes for all subgraph.
  /// @param [in] graph: Root compute graph.
  /// @param [in] func_desc: functional OpDesc of Case.
  /// @param [out] graph_nodes: Data groups of subgraph.
  /// @return 0: SUCCESS / others: FAILED
  Status ClassifyDataNodes(const ComputeGraphPtr &graph, const OpDescPtr &func_desc,
                           OrderedGraphToNodes &graph_nodes) const;

  /// @ingroup ge
  /// @brief Get all Data nodes for all subgraph.
  /// @param [in] node: Node Directly to Data.
  /// @param [out] inputs: parent index of Input.
  /// @param [out] outputs: parent index of Output.
  /// @return true: SUCCESS / false: FAILED
  bool GetAssociatedNodes(const NodePtr &node, std::map<uint32_t, uint32_t> &inputs,
                          std::map<uint32_t, uint32_t> &outputs) const;

  /// @ingroup ge
  /// @brief Get all Data nodes for all subgraph.
  /// @param [in] graph_nodes: Data groups of subgraph.
  /// @param [in] base_node: Data Node for migration.
  /// @param [in] node_idx: Parent index of Data node.
  /// @param [in] anchor_idx: Anchor index of node.
  /// @return true: Same / false: not same
  bool IsParallelNodeSame(const OrderedGraphToNodes &graph_nodes,
                          const NodePtr &base_node, uint32_t node_idx, uint32_t anchor_idx) const;

  /// @ingroup ge
  /// @brief Migration subgraph Node to Root
  /// @param [in] graph: Root compute graph.
  /// @param [in] func_node: functional Node of Case.
  /// @param [in] graph_nodes: Data groups of subgraph.
  /// @param [in] data_base: Data Node for migration.
  /// @param [in] data_idx: Data groups of subgraph.
  /// @return 0: SUCCESS / others: FAILED
  Status GraphNodeMigration(const ComputeGraphPtr &graph, const NodePtr &func_node,
                            OrderedGraphToNodes &graph_nodes,
                            const NodePtr &data_base, uint32_t data_idx);

  /// @ingroup ge
  /// @brief Move node to Parent graph.
  /// @param [in] graph: Root compute graph.
  /// @param [in] func_node: functional Node of Case.
  /// @param [in] graph_nodes: Data groups of subgraph.
  /// @param [in] anchor_idx: anchor index of move Node.
  /// @param [in] inputs: Parent index of Node input.
  /// @param [in] outputs: Parent index of Node output.
  /// @return 0: SUCCESS / others: FAILED
  Status MoveNodeToParent(const ComputeGraphPtr &graph, const NodePtr &func_node,
                          const OrderedGraphToNodes &graph_nodes,
                          uint32_t anchor_idx, uint32_t base_index,
                          const std::map<uint32_t, uint32_t> &inputs,
                          const std::map<uint32_t, uint32_t> &outputs) const;

  /// @ingroup ge
  /// @brief Append Input Tensor for functional node.
  /// @param [in] graph_nodes: Data groups of subgraph.
  /// @param [in] func_node: functional Node of Case.
  /// @param [in] outputs: Parent index of Node output.
  /// @return 0: SUCCESS / others: FAILED
  Status AppendParallelNode(OrderedGraphToNodes &graph_nodes,
                            const NodePtr &func_node, std::map<uint32_t, uint32_t> &outputs);

  /// @ingroup ge
  /// @brief Delete Node from all subgraph.
  /// @param [in] graph_nodes: Data groups of subgraph.
  /// @param [in] detach: Node will move to parent.
  /// @param [in] outputs: Parent index of Node output.
  /// @return 0: SUCCESS / others: FAILED
  Status DetachParallelNode(const std::map<uint32_t, NodePtr> &graph_datas, const NodePtr &detach,
                            const std::map<uint32_t, uint32_t> &outputs) const;

  /// @ingroup ge
  /// @brief Move Node to Parent Graph.
  /// @param [in] graph: Parent compute graph.
  /// @param [in] func_node: functional Node of Case.
  /// @param [in] attach: Node will move to parent.
  /// @param [in] inputs: Parent index of Node input.
  /// @param [in] outputs: Parent index of Node output.
  /// @return 0: SUCCESS / others: FAILED
  Status AttachParallelNode(const ComputeGraphPtr &graph, const NodePtr &func_node, const NodePtr &attach,
                            const std::map<uint32_t, uint32_t> &inputs,
                            const std::map<uint32_t, uint32_t> &outputs) const;

  Status SplitIdentityN(ComputeGraphPtr &graph);
  Status ClassifyInputAnchor(NodePtr &identity_node,
                             std::vector<int32_t> &data_input_anchor_index,
                             std::vector<int32_t> &non_data_input_anchor_index) const;
  Status AddNewIdentityN(NodePtr &old_node, std::vector<int32_t> &input_anchor_index,
                         const std::string &new_node_name) const;
  bool migration_append_{false};
};
}  // namespace ge
#endif  // GE_COMMON_SUBEXPRESSION_MIGRATION_H_
