/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/standard_optimize/prune_pass.h"
#include <deque>
#include <set>
#include <unordered_set>
#include <vector>
#include "framework/common/debug/log.h"
#include "framework/common/types.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_type_utils.h"

namespace ge {
namespace {
bool IsTargetRefNode(const NodePtr &node) {
  GE_CHECK_NOTNULL(node);
  bool is_ref = false;
  (void)AttrUtils::GetBool(node->GetOpDesc(), ATTR_NAME_REFERENCE, is_ref);

  if (is_ref && node->GetOutNodes().empty()) {
    GELOGD("Node %s is ref node without any output. Consider as target node of graph, skip prune.", node->GetNamePtr());
    return true;
  }
  return false;
}
} // namespace
Status PrunePass::Run(ge::ComputeGraphPtr graph) {
  if (graph == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Param graph is nullptr, check invalid");
    GELOGE(GE_GRAPH_ISNULL, "[Check][Param] input compute graph is NULL.");
    return GE_GRAPH_ISNULL;
  }
  GELOGD("PrunePass Start, graph is [%s]", graph->GetName().c_str());
  std::vector<NodePtr> out_nodes;
  std::vector<NodePtr> force_skip_nodes;
  std::unordered_set<NodePtr> nodes;
  for (NodePtr &node_ptr : graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(node_ptr->GetOpDesc());
    nodes.insert(node_ptr);
    if (node_ptr->GetOpDesc()->GetType() == NETOUTPUT || IsTargetRefNode(node_ptr)) {
      out_nodes.push_back(node_ptr);
      continue;
    }
    bool skip_prune{false};
    (void)AttrUtils::GetBool(node_ptr->GetOpDesc(), ATTR_NAME_SKIP_PRUNE_OPTIMIZE, skip_prune);
    if (skip_prune) {
      force_skip_nodes.push_back(node_ptr);
      GELOGI("[PrunePass] Node [%s] skip to prune", node_ptr->GetNamePtr());
    }
  }
  if (out_nodes.empty()) {
    GELOGW("graph [%s] does not contain NETOUTPUT type node,no return value. Do nothing!", graph->GetName().c_str());
    return ge::SUCCESS;
  }
  out_nodes.insert(out_nodes.cend(), force_skip_nodes.cbegin(), force_skip_nodes.cend());

  std::unordered_set<NodePtr> nodes_seen;
  for (NodePtr &node_ptr : out_nodes) {
    std::deque<NodePtr> queue;
    queue.push_back(node_ptr);
    nodes_seen.insert(node_ptr);
    while (!queue.empty()) {
      NodePtr node = queue.front();
      GE_CHECK_NOTNULL(node->GetOpDesc());
      queue.pop_front();
      for (auto &in_node : node->GetInAllNodes()) {
        if (nodes_seen.insert(in_node).second) {
          queue.push_back(in_node);
        }
      }
    }
  }

  for (auto &node_ptr : nodes) {
    if (nodes_seen.count(node_ptr) != 0) {
      continue;
    }
    if (OpTypeUtils::IsDataNode(node_ptr->GetType())) {
      Status status = ge::GraphUtils::AddEdge(node_ptr->GetOutControlAnchor(), out_nodes[0]->GetInControlAnchor());
      if (status != ge::SUCCESS) {
        REPORT_INNER_ERR_MSG("E19999", "Add control edge between op:%s(%s) and op:%s(%s) failed",
                          node_ptr->GetName().c_str(), node_ptr->GetType().c_str(),
                          out_nodes[0]->GetName().c_str(), out_nodes[0]->GetType().c_str());
        GELOGE(INTERNAL_ERROR, "[add][ControlEdge] failed between DATA node[%s] and NETOUTPUT node[%s]!",
               node_ptr->GetName().c_str(), out_nodes[0]->GetName().c_str());
        return INTERNAL_ERROR;
      }
      GELOGI("[PrunePass] add extra control edge between DATA node[%s] and NETOUTPUT node[%s]!",
             node_ptr->GetName().c_str(), out_nodes[0]->GetName().c_str());
      continue;
    }
    // Remove subgraphs on the node before remove it in graph.
    (void)NodeUtils::RemoveSubgraphsOnNode(node_ptr);
    /// Common function:[RemoveNode] will delete not only input node but its constant input node also will be deleted
    (void)graph->RemoveNode(node_ptr);
    GELOGI("[PrunePass] remove graph node [%s]!", node_ptr->GetOpDesc()->GetName().c_str());
  }
  return ge::SUCCESS;
}

REG_PASS_OPTION("PrunePass").LEVELS(OoLevel::kO3);
}  // namespace ge
