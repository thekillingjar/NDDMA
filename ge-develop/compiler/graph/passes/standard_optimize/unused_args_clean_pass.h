/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_COMMON_CASE_ARGS_CLEAN_H_
#define GE_COMMON_CASE_ARGS_CLEAN_H_

#include "graph/types.h"
#include "graph/passes/graph_pass.h"

#include <map>
#include <set>
#include <vector>
#include <string>


namespace ge {
class UnusedArgsCleanPass : public GraphPass {
 public:
  Status Run(ComputeGraphPtr graph) override;

 private:
  /// @ingroup ge
  /// @brief Create nodes for root graph.
  /// @param [in] graph_nodes: Data groups of subgraph.
  /// @param [in] func_node: functional Node of Case.
  /// @param [in] parent_index: parent index for check.
  /// @return true: unused / false: used
  bool UnusedInputTensor(const std::map<ComputeGraphPtr, std::map<uint32_t, NodePtr>> &graph_nodes,
                         const NodePtr &func_node, uint32_t parent_index) const;

  /// @ingroup ge
  /// @brief Get all Data nodes for all subgraph.
  /// @param [in] graph: Root compute graph.
  /// @param [in] func_desc: functional OpDesc of Case.
  /// @param [out] graph_nodes: Data groups of subgraph.
  /// @return 0: SUCCESS / others: FAILED
  Status ClassifyDataNodes(const ComputeGraphPtr &graph, const OpDescPtr &func_desc,
                           std::map<ComputeGraphPtr, std::map<uint32_t, NodePtr>> &graph_nodes) const;

  /// @ingroup ge
  /// @brief Remove Case input Tensor.
  /// @param [in] graph_nodes: Data groups of subgraph.
  /// @param [in] func_node: functional Node of Case.
  /// @param [in] parent_index: parent index for remove.
  /// @return 0: SUCCESS / others: FAILED
  Status RemoveInputTensor(const std::map<ComputeGraphPtr, std::map<uint32_t, NodePtr>> &graph_nodes,
                           const NodePtr &func_node, uint32_t parent_index) const;

  /// @ingroup ge
  /// @brief Update Case input Tensor.
  /// @param [in] graph_nodes: Data groups of subgraph.
  /// @param [in] func_node: functional Node of Case.
  /// @param [in] parent_index: parent index for update.
  /// @param [in] unused_num: unused args num.
  /// @return 0: SUCCESS / others: FAILED
  Status UpdateInputTensor(const std::map<ComputeGraphPtr, std::map<uint32_t, NodePtr>> &graph_nodes,
                           const NodePtr &func_node, uint32_t parent_index, uint32_t unused_num) const;
};
}  // namespace ge
#endif  // GE_COMMON_CASE_ARGS_CLEAN_H_
