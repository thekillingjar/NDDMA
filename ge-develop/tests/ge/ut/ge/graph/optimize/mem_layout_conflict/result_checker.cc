/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "result_checker.h"
#include "common/mem_conflict_share_graph.h"
#include "graph/compute_graph.h"

namespace ge {
namespace mem_check {
graphStatus ResultChecker::CheckIdentityBefore(const ComputeGraphPtr &graph, const std::vector<NodeIo> &nodes) {
  std::map<std::string, NodePtr> name_to_node;
  for (const auto &node : graph->GetAllNodes()) {
    name_to_node[node->GetName()] = node;
  }
  graphStatus check_result = GRAPH_SUCCESS;
  for (const auto &node_io : nodes) {
    const auto node_iter = name_to_node.find(node_io.name);
    if (node_iter == name_to_node.end()) {
      check_result = GRAPH_FAILED;
      std::cerr << "can not find node " << node_io.name << std::endl;
      continue;
    }
    if (node_io.index >= node_iter->second->GetAllInDataAnchorsSize()) {
      check_result = GRAPH_FAILED;
      std::cerr << "node[" << node_io.name << "] has " << node_iter->second->GetAllInDataAnchorsSize()
                << "in anchors, but in index[" << node_io.index << "] is given." << std::endl;
      continue;
    }
    const auto in_anchor = node_iter->second->GetInDataAnchor(node_io.index);
    if (in_anchor == nullptr) {
      check_result = GRAPH_FAILED;
      std::cerr << "node[" << node_io.name << "] [" << node_io.index << "]'th in anchor is nullptr." << std::endl;
      continue;
    }
    if ((in_anchor->GetPeerOutAnchor() == nullptr) || (in_anchor->GetPeerOutAnchor()->GetOwnerNode() == nullptr)) {
      check_result = GRAPH_FAILED;
      std::cerr << "node[" << node_io.name << "] [" << node_io.index
                << "]'th peer out anchor is null, or peer node is null." << std::endl;
      continue;
    }

    if (in_anchor->GetPeerOutAnchor()->GetOwnerNode()->GetType() != IDENTITY) {
      check_result = GRAPH_FAILED;
      std::cerr << "node[" << node_io.name << "] [" << node_io.index << "]'th input is not identity." << std::endl;
      continue;
    }
  }
  return check_result;
}

graphStatus ResultChecker::CheckIdentityNum(const ComputeGraphPtr &graph, const size_t num) {
  std::vector<NodePtr> identity_nodes;
  for (const auto &node : graph->GetAllNodes()) {
    if (node->GetType() == IDENTITY) {
      identity_nodes.emplace_back(node);
    }
  }
  if (identity_nodes.size() != num) {
    std::cerr << "expect " << num << " identity ndoes, actrual number: " << identity_nodes.size() << std::endl;
    for (const auto &node : identity_nodes) {
      std::cerr << node->GetName() << ", ";
    }
    std::cerr << std::endl;
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

bool ResultChecker::CheckGraphEqual(ge::ComputeGraphPtr graph_a, ge::ComputeGraphPtr graph_b) {
  SymbolToAnchors a_symbol_to_anchors_;
  AnchorToSymbol a_anchor_to_symbol_;
  SymbolToAnchors b_symbol_to_anchors_;
  AnchorToSymbol b_anchor_to_symbol_;

  graph_a->TopologicalSorting();
  GraphUtils::GetRefMapping(graph_a, a_symbol_to_anchors_, a_anchor_to_symbol_);
  graph_b->TopologicalSorting();
  GraphUtils::GetRefMapping(graph_b, b_symbol_to_anchors_, b_anchor_to_symbol_);

  if (a_anchor_to_symbol_ == b_anchor_to_symbol_) {
    return true;
  } else {
    return false;
  }
}

}; // namespace mem_layout_conflict_optimize
} // namespace ge
