/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_CONTROL_NODE_SUBGRAPH_CONFLICT_SOLVE_H_
#define GE_GRAPH_CONTROL_NODE_SUBGRAPH_CONFLICT_SOLVE_H_

#include "graph/graph.h"
#include "graph/passes/graph_pass.h"
#include "framework/common/ge_types.h"

namespace ge {

class CtrlNodeConflict {
 public:
  static Status SolveCtrlNodeSubGraphConflict(const ComputeGraphPtr &graph);
 
 private:
  using SubGraphSolveConflictCall = std::function<Status(const NodePtr &ctrl_node)>;
  static std::map<std::string, SubGraphSolveConflictCall> get_subgraph_solve_call;

  static Status SolveIfConflict(const NodePtr &ctrl_node);

  static Status SolveCaseConflict(const NodePtr &ctrl_node);

  static Status SolveOneSubGraphConflict(const ComputeGraphPtr &sub_graph);

  static Status InsertInputIdentity(const ComputeGraphPtr &graph, const std::vector<NodePtr> &data_nodes);

  static Status InsertOutputIdentity(const ComputeGraphPtr &graph, const NodePtr &output_node,
                                     const std::set<uint32_t> &bypass_index);

  static Status SolveWhileConflict(const NodePtr &ctrl_node);

  static Status SolveNodesConflict(const ComputeGraphPtr &graph,
                                   const std::vector<InDataAnchorPtr> &in_data_anchors);
};

}

#endif  // GE_GRAPH_CONTROL_NODE_SUBGRAPH_CONFLICT_SOLVE_H_
