/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "data_flow_graph_prune_pass.h"
#include <deque>
#include <set>
#include <string>
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
const std::string kAttrIsolatedData = "_isolate_data_after_prune";
}
Status DataFlowGraphPrunePass::Run(ge::ComputeGraphPtr graph) {
  if (graph == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Param graph is nullptr, check invalid");
    GELOGE(GE_GRAPH_ISNULL, "[Check][Param] input compute graph is NULL.");
    return GE_GRAPH_ISNULL;
  }
  GELOGD("DatatFlowPrunePass Start, graph is [%s]", graph->GetName().c_str());
  const auto out_nodes = graph->GetOutputNodes();
  if (out_nodes.empty()) {
    GELOGW("graph [%s] does not contain output node,no return value. Do nothing!", graph->GetName().c_str());
    return ge::SUCCESS;
  }
  std::vector<NodePtr> anchor_nodes{out_nodes.begin(), out_nodes.end()};
  std::unordered_set<NodePtr> nodes_seen;
  for (const auto &node_ptr : anchor_nodes) {
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

  for (const auto &node_ptr : graph->GetDirectNode()) {
    if (nodes_seen.count(node_ptr) != 0) {
      continue;
    }
    if (OpTypeUtils::IsDataNode(node_ptr->GetType())) {
      GE_CHK_STATUS_RET(GraphUtils::AddEdge(node_ptr->GetOutControlAnchor(), out_nodes.at(0UL)->GetInControlAnchor()),
                        "[add][ControlEdge] failed between DATA node[%s] and NETOUTPUT node[%s]!",
                        node_ptr->GetName().c_str(), out_nodes.at(0UL)->GetName().c_str());
      (void)AttrUtils::SetBool(node_ptr->GetOpDesc(), kAttrIsolatedData, true);
      GELOGI("[PrunePass] add extra control edge between DATA node[%s] and NETOUTPUT node[%s]!",
             node_ptr->GetName().c_str(), out_nodes.at(0UL)->GetName().c_str());
      continue;
    }
    if (OpTypeUtils::IsGraphOutputNode(node_ptr->GetType())) {
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

REG_PASS_OPTION("DataFlowGraphPrunePass").LEVELS(OoLevel::kO0);
}  // namespace ge
