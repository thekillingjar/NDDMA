/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PASSES_ATOMIC_ADDR_CLEAN_PASS_H_
#define GE_GRAPH_PASSES_ATOMIC_ADDR_CLEAN_PASS_H_

#include <vector>

#include "graph/graph.h"
#include "graph/passes/graph_pass.h"

namespace ge {
/*
 * Atomic addr clean task fusion
 * Find all atomic op in graph,and insert one AtomicAddrClean op.
 * To clean atomic output and workspace once for all.
 * before iteration starts, empty AtomicAdd output, workspace memory
 *         op1                         op1
 *          |                           |
 *         op2(atomic)       ==>       op2
 *          |                           |  \
 *         op3(atomic)                 op3 -AtomicClean
 */
class AtomicAddrCleanPass : public GraphPass {
 public:
  Status Run(ComputeGraphPtr graph);
  Status ClearStatus() override;
  /**
   * Handle atomic node in unknown graph
   * @param atomic_node_vec: atomic node vector in unknown graph
   * @return
   */
  static Status CallCompileOp(const std::vector<NodePtr> &atomic_node_vec);

 private:
  /**
    * HandleLoopGraph
    * @param graph
    * @return
    */
  Status HandleLoopGraph(ComputeGraphPtr &graph, const std::vector<NodePtr> &atomic_node_vec);
  /**
   * HandleNormalGraph
   * @param graph
   * @return
   */
  Status HandleNormalGraph(ComputeGraphPtr &graph, const std::vector<NodePtr> &atomic_node_vec) const;
  /**
   * Insert atomic clean node to graph
   * @param graph
   * @return
   */
  NodePtr InsertAtomicMemsetNode(ComputeGraphPtr &graph) const;

  /**
   * Link control anchor from atomic clean node to atomic node
   * @param atomic_node
   * @param atomic_clean_node
   * @return
   */
  Status LinkToAtomicNode(const NodePtr &atomic_node, NodePtr &atomic_clean_node) const;

  /**
   * Link atomic clean node to all potential precedence nodes which may execute before atomic clean node
   * @param graph
   * @param atomic_clean_node
   * @return
   */
  Status LinkToPotentialPrecedenceNode(ComputeGraphPtr &graph, NodePtr &atomic_clean_node,
                                       const std::vector<NodePtr> &dispersed_atomic_nodes) const;

  /**
   * Check if this node is atomic op.
   * @param node
   * @return
   */
  bool IsNeedCleanNode(const NodePtr &node, const bool need_clean_separately);

  Status HandleDispersedAtomicNodes(ComputeGraphPtr &graph, const std::vector<NodePtr> &atomic_node_vec,
                                    std::vector<NodePtr> &common_atomic_nodes,
                                    std::vector<NodePtr> &dispersed_atomic_nodes) const;

  bool CheckAtomicFromOpsKernel(const NodePtr &node);

  bool IsAllConnectHcclNode(const NodePtr &node) const;

  bool IsConnectNoTaskNode(const NodePtr &node) const;

  bool RefVariable(const NodePtr &node) const;

  bool IsOutputIndexPeerInputAtomic(const NodePtr &node, int64_t output_index);

  bool CheckSkipInsertInLoopGraph(const NodePtr &node);

  Status LinkToHcomPeerInNode(NodePtr &atomic_clean_node) const;

  void IsNeedCallCompileForUnknownGraph(std::vector<NodePtr> &atomic_node_vec) const;

  static bool CheckAccuracySupported(const OpDescPtr &op_desc);

  bool IsHcomAtomicNode(const ge::NodePtr &node) const;

  std::vector<NodePtr> hcom_node_vec_;
  bool is_loop_graph_ = false;
  std::vector<NodePtr> need_gentask_atomic_node_;
};
}  // namespace ge

#endif  // GE_GRAPH_PASSES_ATOMIC_ADDR_CLEAN_PASS_H_
