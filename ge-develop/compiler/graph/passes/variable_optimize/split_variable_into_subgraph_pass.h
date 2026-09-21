/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_NODE_PASSES_SPLIT_VARIABLE_INTO_SUBGRAPH_PASS_H
#define AIR_NODE_PASSES_SPLIT_VARIABLE_INTO_SUBGRAPH_PASS_H
#include "graph/passes/base_pass.h"
namespace ge {
/**
 * (1)
 * When variable connect to a node with subgraph(if/case/paritioncall)
 * we need split variable into subgraph to make sure variable function ok in subgraph
 * Before:
 *                               +-------------------+
 *                               |       data        |
 *       variable                |        |          |
 *       /   \                   |       opB         |
 *    opA    PartitionedCall     +-------------------+
 *
 *
 * After:
 *                                  +-------------------+
 *                                  |       variable    |
 *       variable                   |         |         |
 *        /   \                     |        opB        |
 *     opA    PartitionedCall       +-------------------+
 *(2)
 * When variable connect to a while node
 * This pass will copy variable into subgraph, consider while node as normal op, after while_node executed,
 * opB can access variable.
 *
 *      variable
 *        /  \
 *     opA   while
 *            |
 *           opB
 *
 *
 *         variable
 *        /   |    \c
 *     opA    |   while
 *            \    /c
 *             opB
 *
 * Attention:
 *  This pass should be executed after DataPass which mark parent_node_input_index on inner data of subgraph.
 */
class SplitVariableIntoSubgraphPass : public BaseNodePass {
 public:
  Status Run(NodePtr &node) override;

 private:
  Status RefreshMultiDataRefDataNodeShape(const NodePtr &data_node, NodePtr &ref_data_node) const;
  Status CopyVarToSubgraph(const NodePtr &var, int32_t parent_input_index, const ComputeGraphPtr &subgraph) const;
  bool IsMultiBatchSubgraph(const ComputeGraphPtr &subgraph) const;
};
}  // namespace ge
#endif  // AIR_NODE_PASSES_SPLIT_VARIABLE_INTO_SUBGRAPH_PASS_H
