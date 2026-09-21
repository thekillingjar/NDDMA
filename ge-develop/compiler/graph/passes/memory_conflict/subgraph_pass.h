/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PASSES_SUBGRAPH_PASS_H_
#define GE_GRAPH_PASSES_SUBGRAPH_PASS_H_

#include "graph/passes/graph_pass.h"

namespace ge {
class SubgraphPass : public GraphPass {
 public:
  /**
   * @ingroup ge
   * @brief Subgraph optimizer.
   * @param [in] graph: Input ComputeGraph
   * @return: 0 for success / others for fail
   */
  Status Run(ComputeGraphPtr graph) override;

 private:
  /**
   * @ingroup ge
   * @brief Check Subgraph Data node.
   * @param [in] graph: ComputeGraph.
   * @param [in] node: NetOutput node in Subgraph.
   * @return: 0 for SUCCESS / others for FAILED
   */
  Status SubgraphInputNode(const ComputeGraphPtr &graph, const NodePtr &node) const;

  /**
   * @ingroup ge
   * @brief Check Subgraph NetOutput node.
   * @param [in] graph: ComputeGraph.
   * @param [in] node: NetOutput node in Subgraph.
   * @return: 0 for SUCCESS / others for FAILED
   */
  Status SubgraphOutputNode(const ComputeGraphPtr &graph, const NodePtr &node) const;

  /**
   * @ingroup ge
   * @brief Check is Input->While and Input link to other nodes
   * @param [in] graph: ComputeGraph.
   * @param [in] node: While node.
   * @return: 0 for SUCCESS / others for FAILED
   */
  Status WhileInputNodes(const NodePtr &node) const;

  /**
   * @ingroup ge
   * @brief Check body subgraph of While op
   * @param [in] graph: ComputeGraph.
   * @param [in] node: While node.
   * @return: 0 for SUCCESS / others for FAILED
   */
  Status WhileBodySubgraph(const ComputeGraphPtr &graph, const NodePtr &node) const;

  /**
   * @ingroup ge
   * @brief Insert input memcpy node in while_body
   * @param [in] graph: while_body
   * @param [in] data_nodes: data_nodes
   * @return: 0 for SUCCESS / others for FAILED
   */
  Status InsertInputMemcpy(const ComputeGraphPtr &graph, const std::vector<NodePtr> &data_nodes) const;

  /**
   * @ingroup ge
   * @brief Insert output memcpy node in while_body
   * @param [in] graph: while_body
   * @param [in] output_node: NetOutput
   * @param [in] bypass_index
   * @return: 0 for SUCCESS / others for FAILED
   */
  Status InsertOutputMemcpy(const ComputeGraphPtr &graph, const NodePtr &output_node,
                            const std::set<uint32_t> &bypass_index) const;

  /**
   * @ingroup ge
   * @brief Check is data->netoutput without change in while body
   * @param [in] node: data node
   * @param [out] bypass_index
   * @return: false for data->netoutput without change in while body / for true for others
   */
  bool CheckInsertInputMemcpy(const NodePtr &node, std::set<uint32_t> &bypass_index) const;

  /**
   * @ingroup ge
   * @brief Check is AtomicOp->NetOutput
   * @param [in] node
   * @param [in] out_index
   * @return: true for AtomicOp->NetOutput / false for others
   */
  bool IsAtomicRequired(const NodePtr &node, int64_t out_index) const;

  /**
   * @ingroup ge
   * @brief Check is OutputContinuesRequiredOp->NetOutput
   * @param [in] node
   * @param [in] out_index
   * @return: true for OutputContinuesRequiredOp->NetOutput / false for others
   */
  bool IsOutputContinuesRequired(const NodePtr &node, int32_t out_index) const;

  /**
   * @ingroup ge
   * @brief Check is InputContinuesRequiredOp->NetOutput
   * @param [in] node
   * @param [in] in_index
   * @return: true for InputContinuesRequiredOp->NetOutput / false for others
   */
  bool IsInputContinuesRequired(const NodePtr &node, int32_t in_index) const;

  /**
   * @ingroup ge
   * @brief Insert memcpy node
   * @param [in] graph
   * @param [in] out_anchor
   * @param [in] in_anchors
   * @param [in] name
   * @return: 0 for success / others for fail
   */
  Status InsertMemcpyNode(const OutDataAnchorPtr &out_anchor,
                          const std::vector<InDataAnchorPtr> &in_anchors, const std::string &name) const;

  /// @brief Insert node: src->insert_node:input_index, insert_node:output_index->dst
  /// @param [in] src
  /// @param [in] dsts
  /// @param [in] insert_node
  /// @param [in] input_index
  /// @param [in] output_index
  /// @return Status
  Status InsertNodeBetween(const OutDataAnchorPtr &src, const std::vector<InDataAnchorPtr> &dsts,
                           const NodePtr &insert_node, uint32_t input_index, uint32_t output_index) const;

  /**
   * @ingroup ge
   * @brief 在静态图中，使用插入identity算子的方式对条件算子子图的输出做地址隔离。
   *        不允许存在子图中某算子单输出多引用给到多个子图输出，这样会导致静态图按symbol分配地址时出现冲突。以if为例：
   *        true graph                false graph                  parent graph
   *        +------------------+      +------------------+
   *        |                  |      |                  |                if
   *        |        op        |      |   op1      op2   |               /  \
   *        |       /  \       |      |    \       /     |           op_a    op_b
   *        |    netoutput     |      |     netoutput    |
   *        |__________________|      |__________________|
   *
   * @param [in] node: Condition node in graph.
   * @return: 0 for SUCCESS / others for FAILED
   */
  Status SeparateOutputsAddrOfCondSubgraph(const NodePtr &node) const;
};
}  // namespace ge
#endif  // GE_GRAPH_PASSES_SUBGRAPH_PASS_H_
