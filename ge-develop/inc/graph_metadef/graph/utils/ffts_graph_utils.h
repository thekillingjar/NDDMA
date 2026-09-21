/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_GRAPH_UTILS_FFTS_GRAPH_UTILS_H_
#define INC_GRAPH_UTILS_FFTS_GRAPH_UTILS_H_

#include "graph/anchor.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/node.h"

namespace ge {
class FftsGraphUtils {
 public:
  using CalcFunc = std::function<std::vector<uint32_t>(const NodePtr &)>;
  static graphStatus GraphPartition(ComputeGraph &graph, const std::set<NodePtr> &unsupported_nodes);

  static graphStatus GraphPartition(ComputeGraph &graph,
                                    const CalcFunc &calc_func,
                                    const std::vector<uint32_t> &upper_limit);
 private:
  static graphStatus CollectClipNodesAndGraphs(const ComputeGraphPtr &graph,
                                               const std::set<NodePtr> &unsupported_nodes,
                                               std::unordered_set<NodePtr> &nodes_need_clip,
                                               std::unordered_set<ComputeGraphPtr> &graphs_need_split);

  static bool IsGraphNeedSplit(const ComputeGraphPtr &graph, const std::unordered_set<NodePtr> &nodes_need_clip);

  static graphStatus SplitNodesWithCheck(const ComputeGraphPtr &graph,
                                         const std::unordered_set<NodePtr> &nodes_need_clip,
                                         std::vector<std::pair<bool, std::set<NodePtr>>> &split_nodes);

  static void SplitNodes(const std::set<NodePtr> &calc_nodes, const std::function<bool(const NodePtr &)> &is_cur_stage,
                         std::set<NodePtr> &visited_nodes, std::set<NodePtr> &cur_nodes, std::set<NodePtr> &next_nodes);

  static graphStatus SplitSubgraph(const ComputeGraphPtr &subgraph,
                                   const std::vector<std::pair<bool, std::set<NodePtr>>> &split_nodes);

  static graphStatus BuildFftsPlusSubgraphWithAllNodes(const ComputeGraphPtr &subgraph);

  static void CollectCalcNodeInSubgraph(const ComputeGraphPtr &subgraph, std::set<NodePtr> &calc_nodes);

  static void CollectEndNodeInSubgraph(const ComputeGraphPtr &subgraph, const std::set<std::string> &ctrl_goto_types,
                                       std::set<NodePtr> &edge_nodes);

  static ComputeGraphPtr GetFftsPlusGraph(ComputeGraph &graph);

  static graphStatus SetAttrForFftsPlusSubgraph(const ComputeGraphPtr &subgraph);

  static graphStatus Calculate(const ComputeGraphPtr &graph,
                               const CalcFunc &calc_func,
                               std::map<NodePtr, std::vector<uint32_t>> &node_value,
                               std::map<ComputeGraphPtr, std::vector<uint32_t>> &graph_value,
                               const uint32_t recursive_depth = 1U);

  static std::vector<uint32_t> Calculate(const NodePtr &node, const CalcFunc &calc_func,
                                        std::map<NodePtr, std::vector<uint32_t>> &node_value,
                                        std::map<ComputeGraphPtr, std::vector<uint32_t>> &graph_value,
                                        const uint32_t recursive_depth);

  static bool IsValueValid(const ComputeGraphPtr &graph, const std::vector<uint32_t> &upper_limit,
                           const std::map<NodePtr, std::vector<uint32_t>> &node_value,
                           const std::map<ComputeGraphPtr, std::vector<uint32_t>> &graph_value);

  static graphStatus PartitionGraphWithLimit(const ComputeGraphPtr &graph,
                                             std::map<NodePtr, std::vector<uint32_t>> &node_value,
                                             std::map<ComputeGraphPtr, std::vector<uint32_t>> &graph_value,
                                             const std::vector<uint32_t> &upper_limit,
                                             const uint32_t recursive_depth = 1U);

  static graphStatus SplitFuncNode(const std::vector<NodePtr> exceed_single_node,
                                   std::map<NodePtr, std::vector<uint32_t>> &node_value,
                                   std::map<ComputeGraphPtr, std::vector<uint32_t>> &graph_value,
                                   const std::vector<uint32_t> &upper_limit,
                                   const uint32_t recursive_depth);
};
}  // namespace ge
#endif  // INC_GRAPH_UTILS_GRAPH_UTILS_H_
