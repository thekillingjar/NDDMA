/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PASSES_SUBGRAPH_MULTI_DIMS_CLONE_PASS_H_
#define GE_GRAPH_PASSES_SUBGRAPH_MULTI_DIMS_CLONE_PASS_H_

#include <map>
#include <string>
#include <vector>

#include "graph/passes/graph_pass.h"

namespace ge {
class SubgraphMultiDimsClonePass : public GraphPass {
 public:
  Status Run(ComputeGraphPtr graph);

 private:
  Status MergeDataDynDims();
  Status CreateOriGraph(const ComputeGraphPtr &subgraph);
  Status CollectIoNodes(const ComputeGraphPtr &subgraph);
  Status ParseInputShapeAndDynDims(ComputeGraphPtr &subgraph);
  Status CreateGetShapeNode(const ComputeGraphPtr &subgraph);
  Status CreateConcatNode(const ComputeGraphPtr &subgraph);
  Status CreateMapIndexNode(const ComputeGraphPtr &subgraph);
  Status CreateIndexConstNode(const ComputeGraphPtr &subgraph);
  Status CreateCaseNode(const ComputeGraphPtr &subgraph);
  Status CreateConstNode(const ComputeGraphPtr &subgraph);
  Status CreateInputNode(const ComputeGraphPtr &subgraph);
  Status CreateOutputNode(const ComputeGraphPtr &subgraph);
  Status CreateSubgraphs(const ComputeGraphPtr &root_graph, const ComputeGraphPtr &subgraph,
                         const ComputeGraphPtr &branch);
  Status UpdateSubgraphData(const NodePtr &data, size_t grade_index) const;
  inline std::string GetNodeName(const ComputeGraphPtr &graph, const std::string &name_prefix) const;

  std::vector<NodePtr> all_data_nodes_;
  std::vector<NodePtr> all_const_nodes_;
  NodePtr output_node_;

  std::string input_shapes_;
  std::string input_dynamic_dims_;
  std::string input_mode_;

  std::vector<NodePtr> get_shape_node_;
  NodePtr map_index_node_;
  NodePtr const_node_;
  NodePtr case_node_;
  NodePtr concat_node_;

  std::vector<std::vector<int64_t>> merged_multi_dims_;
  std::vector<std::vector<int64_t>> changed_dims_;
};
}  // namespace ge
#endif  // GE_GRAPH_PASSES_MULTI_BATCH_CLONE_PASS_H_
