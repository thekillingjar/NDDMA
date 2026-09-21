/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PASSES_CREATE_SUBGRAPH_WITH_SCOPE_PASS_H_
#define GE_GRAPH_PASSES_CREATE_SUBGRAPH_WITH_SCOPE_PASS_H_

#include <map>
#include <string>
#include <vector>

#include "graph/passes/graph_pass.h"

namespace ge {
class CreateSubGraphWithScopePass : public GraphPass {
 public:
  Status Run(ComputeGraphPtr graph);

 private:
  Status ProcessHeterogeneousMultiBatch(const ComputeGraphPtr &graph);
  Status CollectDynamicNodes(const ComputeGraphPtr &graph, std::vector<NodePtr> &dynamic_shape_nodes);
  Status UpdateDynamicConfigAttrs(const std::vector<NodePtr> &dynamic_shape_nodes);
  Status UpdateSubgraphMultiDimsAttr(NodePtr node, const std::string &pre_input_shape,
                                     const std::string &pre_input_dims, const std::string &new_input_shape,
                                     const std::string &new_input_dims) const;
  Status CreateMultiBatchScope(const ComputeGraphPtr &graph);
  bool IsGraphMultiBatchCondition(const ComputeGraphPtr &graph);
  Status CollectScopeNodesByIndex(const ComputeGraphPtr &graph);
  Status ParseMultiDimsAttr(const std::vector<NodePtr> &scope_nodes);
  Status CollectIoNodes(const ComputeGraphPtr &subgraph);
  Status GetSubgraphFromScope(const ComputeGraphPtr &root_graph,
                              const std::pair<const int32_t,
                              std::vector<NodePtr>> &scope,
                              ComputeGraphPtr &subgraph) const;
  Status MergeNodesToSubgraph(const std::vector<NodePtr> &scope_nodes, const ComputeGraphPtr &subgraph);
  Status CheckCtrlAnchorInvalid(const NodePtr &node, const std::vector<NodePtr> &scope_nodes) const;
  Status MergeInputAnchors(const ComputeGraphPtr &subgraph,
                           const NodePtr &node, const std::vector<NodePtr> &scope_nodes);
  Status MergeOutputAnchors(const ComputeGraphPtr &subgraph,
                            const NodePtr &node, const std::vector<NodePtr> &scope_nodes);
  Status ProcPartitionInputAnchor(const ComputeGraphPtr &subgraph, const OutDataAnchorPtr &node_anchor,
                                  const InDataAnchorPtr &partition_anchor);
  Status ProcPartitionOutputAnchor(const OutDataAnchorPtr &partition_anchor,
                                   const InDataAnchorPtr &node_anchor);
  Status SetParentIndexToData(const ComputeGraphPtr &subgraph);
  Status SetParentIndexToNetOutput(const ComputeGraphPtr &subgraph) const;
  Status SetDataDynDimsInfo(const NodePtr &node, const NodePtr &data_node, const int32_t input_index);
  Status CopyPartitionedCallAttrToData(const ComputeGraphPtr &subgraph);
  Status CreateNewPartitionedCall(const ComputeGraphPtr &root_graph, const ComputeGraphPtr &subgraph,
                                  const std::vector<NodePtr> &scope_nodes);
  Status CopyPartitionedCallStaticInput(const NodePtr &src_node, const std::vector<NodePtr> &scope_nodes);
  Status CopyPartitionedCallStaticOutput(const NodePtr &src_node, const std::vector<NodePtr> &scope_nodes);
  Status AddSubgraphToNewPartitionedCall(const ComputeGraphPtr &root_graph, const ComputeGraphPtr &subgraph);
  void SortAnchorByIndex(std::set<std::pair<int32_t, InDataAnchorPtr>> &index_anchor) const;
  Status ParseSubGraphMultiAttrs(const NodePtr &node, const std::string &input_shape, const std::string &multi_dims);
  Status RefreshTensorShape(const NodePtr &node);
  Status UpdateTensorMaxShape(const ComputeGraphPtr &root_graph, const ComputeGraphPtr &subgraph) const;
  Status ParseMultiBatchParams() const;
  std::map<int32_t, std::vector<NodePtr>> scopes_;
  std::map<InDataAnchorPtr, NodePtr> anchor_to_data_nodes_;
  std::map<OutDataAnchorPtr, InDataAnchorPtr> anchor_to_output_;
  std::map<NodePtr, std::map<int64_t, std::vector<int64_t>>> node_to_input_shape_;
  std::map<NodePtr, std::map<int64_t, std::vector<std::vector<int64_t>>>> node_to_multi_dims_;
  std::map<NodePtr, NodePtr> data_node_to_scope_data_node_;
  NodePtr new_partitioned_call_;
  bool is_set_dynamic_config_;
};
}  // namespace ge
#endif  // GE_GRAPH_PASSES_CREATE_SUBGRAPH_WITH_SCOPE_PASS_H_
