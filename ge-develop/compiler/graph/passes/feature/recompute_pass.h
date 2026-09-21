/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PASSES_RECOMPUTE_PASS_H_
#define GE_GRAPH_PASSES_RECOMPUTE_PASS_H_

#include <atomic>
#include "graph/passes/graph_pass.h"
/* Pass Explaination:
 *                                                     ————————————————
 *                                                   /                   |
 *             node1 → node1_b                   node1 → node1_b         |
 *               ⬇        ⬆                        ⬇        ⬆            ⬇  (node2_input)
 * (recompute) node2 → node2_b        ==>        node2   node2_b  ←  node2_copy
 *               ⬇        ⬆                        ⬇        ⬆            ⬆
 *             node3 → node3_b                   node3 → node3_b  ------  (control edge)
 */

namespace ge {
struct NodePtrCmp {
  bool operator()(const ge::NodePtr &lhs, const ge::NodePtr &rhs) const {
    return lhs->GetName() < rhs->GetName();
  }
};
using NodeInEdges = std::map<NodePtr, std::vector<int32_t>, NodePtrCmp>;  // key:bakward node, value:in_anchor_index

class RecomputePass : public GraphPass {
 public:
  Status Run(ComputeGraphPtr graph) override;

 private:
  void InheritNodesAttributes(const ComputeGraphPtr &graph) const;
};

class RecomputeRewriting {
 public:
  Status RewriteGraph(const ComputeGraphPtr &graph);

 private:
  Status CopyAndFindNodes(const ComputeGraphPtr &graph, std::vector<NodePtr> &recompute_start_nodes,
                          std::vector<NodePtr> &all_backward_nodes);

  bool AnyInNodeIsRecompute(const NodePtr &node) const;

  Status CopyRecomputeNode(const NodePtr &node, const ComputeGraphPtr &graph);

  Status RecordLastForwardNode(const std::vector<NodePtr> &all_backward_nodes);

  Status HandleAllRecomputeNodeInputOutput(const ComputeGraphPtr &graph, std::vector<NodePtr> &recompute_start_nodes);

  Status DfsRecomputeNodes(const NodePtr &node, NodeInEdges &needed_replaced_bp_edges,
                           NodeInEdges &small_topo_order_bp_edges, bool &is_need_replace_bp_edges);

  Status HandleRecomputeNodeInputs(const NodePtr &node) const;

  void SaveNeededReplacedBpNodes(const AnchorPtr &peer_out_node_in_anchor, std::vector<NodePtr> &backward_nodes,
                                 NodeInEdges &needed_replaced_bp_edges, NodeInEdges &small_topo_order_bp_edges);

  Status ReplaceBackwardNodeEdge(const NodePtr &node, const int32_t in_anchor_index) const;

  Status RecoverBackwardNodeEdge(const NodePtr &node, const int32_t in_anchor_index) const;

  bool IsNodeNeedRecomputed(NodePtr &node) const;

  void FindAndSortBackwardNodes(const NodePtr &node, std::vector<NodePtr> &backward_nodes) const;

  NodePtr GetCopyNode(const std::string &name) const;

  NodePtr GetRecomputeNode(const std::string &name) const;

  std::map<std::string, NodePtr> name_to_copy_nodes_;
  std::map<std::string, NodePtr> name_to_recompute_nodes_;
  NodePtr last_forward_node_{nullptr};
};
}  // namespace ge
#endif  // GE_GRAPH_PASSES_RECOMPUTE_PASS_H_
