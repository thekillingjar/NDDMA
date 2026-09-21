/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_LOWERING_PASS_NODESCOLLECTOR_H_
#define AIR_CXX_RUNTIME_V2_LOWERING_PASS_NODESCOLLECTOR_H_
#include <functional>
#include <queue>
#include "common/checker.h"
#include "graph/compute_graph.h"
#include "graph/fast_graph/execute_graph.h"
namespace gert {
namespace bg {
template <typename CNode, typename CNodes>
ge::graphStatus CollectNodes(const ge::ExecuteGraph *const graph, CNodes &collected_nodes,
                             const std::function<bool(ge::FastNode *const, CNode &)> &is_start_node,
                             const std::function<bool(ge::FastNode *const)> &is_stop_node) {
  std::queue<CNode> ready_nodes;
  for (const auto node : graph->GetDirectNode()) {
    CNode c_node;
    if (is_start_node(node, c_node) && !is_stop_node(node)) {
      collected_nodes.Append(c_node);
      ready_nodes.emplace(std::move(c_node));
    }
  }

  auto is_all_nodes_seen = [](const ge::FastNode *const node, const std::unordered_set<ge::FastNode *> &nodes_seen) {
    for (const auto in_data_edge : node->GetAllInDataEdgesRef()) {
      if ((in_data_edge != nullptr) && (nodes_seen.count(in_data_edge->src) == 0)) {
        return false;
      }
    }

    for (const auto in_ctl_edge : node->GetAllInControlEdgesRef()) {
      if ((in_ctl_edge != nullptr) && (nodes_seen.count(in_ctl_edge->src) == 0)) {
        return false;
      }
    }
    return true;
  };

  auto append_node = [&ready_nodes, &collected_nodes, &is_all_nodes_seen, &is_stop_node](ge::FastNode *const node) {
    if (!is_all_nodes_seen(node, collected_nodes.GetAllCollectedSet())) {
      return;
    }
    if (is_stop_node(node)) {
      return;
    }
    if (collected_nodes.Append(node)) {
      ready_nodes.emplace(node);
    }
    return;
  };

  while (!ready_nodes.empty()) {
    auto current_node = std::move(ready_nodes.front());
    ready_nodes.pop();
    for (const auto &out_data_edges : current_node.node->GetAllOutDataEdgesRef()) {
      std::for_each(out_data_edges.begin(), out_data_edges.end(), [append_node](ge::FastEdge *const edge) {
        if (edge != nullptr) {
          append_node(edge->dst);
        }
      });
    }

    for (const auto out_ctl_edge : current_node.node->GetAllOutControlEdgesRef()) {
      if (out_ctl_edge != nullptr) {
        append_node(out_ctl_edge->dst);
      }
    }
  }
  return ge::GRAPH_SUCCESS;
}
}  // namespace bg
}  // namespace gert
#endif  // AIR_CXX_RUNTIME_V2_LOWERING_PASS_NODESCOLLECTOR_H_
