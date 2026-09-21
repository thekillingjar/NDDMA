/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_COMMON_SUBGRAPH_CONST_MIGRATION_H_
#define GE_COMMON_SUBGRAPH_CONST_MIGRATION_H_

#include "graph/types.h"
#include "graph/passes/graph_pass.h"

#include <map>
#include <set>
#include <vector>
#include <string>


namespace ge {
namespace {
using OrderedGraphToConstNodes = std::map<ComputeGraphPtr, std::map<std::string, NodePtr>, ComputeGraphCompareKey>;
using OrderedGraphToDataNodes = std::map<ComputeGraphPtr, std::map<uint32_t, NodePtr>, ComputeGraphCompareKey>;
}
class SubgraphConstMigrationPass : public GraphPass {
 public:
  Status Run(ComputeGraphPtr graph) override;

 private:
  /// @ingroup ge
  /// @brief Get all Const/Data nodes for all subgraph.
  /// @param [in] graph: Root compute graph.
  /// @param [in] func_desc: functional OpDesc of Case.
  /// @param [out] all_const_nodes: Const groups of subgraph.
  /// @param [out] all_data_nodes: Data groups of subgraph.
  /// @return 0: SUCCESS / others: FAILED
  Status ClassifyGraphNodes(const ComputeGraphPtr &graph, const OpDescPtr &func_desc,
                            OrderedGraphToConstNodes &all_const_nodes,
                            OrderedGraphToDataNodes &all_data_nodes) const;

  /// @ingroup ge
  /// @brief Get parent_index for Const node migration.
  /// @param [in] all_data_nodes: Data groups of subgraph.
  /// @param [in] const_node: Const node will process.
  /// @param [out] parent_index: parent index for replace Data.
  /// @return true: SUCCESS / false: FAILED
  bool GetAssociatedNodes(const OrderedGraphToDataNodes &all_data_nodes,
                          const NodePtr &const_node, uint32_t &parent_index) const;

  /// @ingroup ge
  /// @brief Check parallel node is same for all subgraph.
  /// @param [in] all_const_nodes: Const groups of subgraph.
  /// @param [in] const_node: Const Node for migration.
  /// @param [in] node_key: Key of Const node.
  /// @return true: Same / false: not same
  bool IsParallelNodeSame(const OrderedGraphToConstNodes &all_const_nodes,
                          const NodePtr &const_node, const std::string &node_key) const;

  /// @ingroup ge
  /// @brief Migration subgraph Node to Root
  /// @param [in] graph: Root compute graph.
  /// @param [in] func_node: functional Node of Case.
  /// @param [in] all_const_nodes: Const groups of subgraph.
  /// @param [in] all_data_nodes: Data groups of subgraph.
  /// @param [in] const_node: Key of const node and const node pair for migration.
  /// @return 0: SUCCESS / others: FAILED
  Status GraphNodeMigration(const ComputeGraphPtr &graph, const NodePtr &func_node,
                            const OrderedGraphToConstNodes &all_const_nodes,
                            OrderedGraphToDataNodes &all_data_nodes,
                            const std::pair<std::string, NodePtr> const_node) const;

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
                          const OrderedGraphToConstNodes &all_const_nodes,
                          const OrderedGraphToDataNodes &all_data_nodes,
                          const std::string &node_key, uint32_t parent_index) const;

  /// @ingroup ge
  /// @brief Append Input Tensor for functional node.
  /// @param [in] graph_nodes: Const groups of subgraph.
  /// @param [in/out] parent_index: Parent index for migration.
  /// @param [in/out] all_data_nodes: Data groups of subgraph.
  /// @return 0: SUCCESS / others: FAILED
  Status AppendParallelNode(const NodePtr &func_node, uint32_t &parent_index,
                            OrderedGraphToDataNodes &all_data_nodes) const;

  /// @ingroup ge
  /// @brief Delete Node from subgraph.
  /// @param [in] graph: subgraph for process.
  /// @param [in] const_node: Node will move to parent.
  /// @param [in] data_node: Place holder for Const.
  /// @return 0: SUCCESS / others: FAILED
  Status DetachParallelNode(const ComputeGraphPtr &graph, const NodePtr &const_node, const NodePtr &data_node) const;

  /// @ingroup ge
  /// @brief Move Node to Parent Graph.
  /// @param [in] graph: Parent compute graph.
  /// @param [in] func_node: functional Node of Case.
  /// @param [in] const_node: Node will move to parent.
  /// @param [in] parent_index: Parent index of Node input.
  /// @return 0: SUCCESS / others: FAILED
  Status AttachParallelNode(const ComputeGraphPtr &graph, const NodePtr &func_node,
                            const NodePtr &const_node, uint32_t parent_index) const;

  void GetPeerNameList(const NodePtr &node, std::set<std::string> &peer_name_list) const;
};
}  // namespace ge
#endif  // GE_COMMON_SUBGRAPH_CONST_MIGRATION_H_