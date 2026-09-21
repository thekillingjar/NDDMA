/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/utils/cycle_detector.h"
#include "framework/common/debug/ge_log.h"
#include "graph_metadef/graph/debug/ge_util.h"

namespace ge {
namespace {
void PrintAllNodes(const std::vector<NodePtr> &scope_nodes) {
  for (const auto &node : scope_nodes) {
    if (node == nullptr) {
      GELOGD("type: null, name: null");
    } else {
      GELOGD("type: %s, name: %s", node->GetType().c_str(), node->GetName().c_str());
    }
  }
}

bool CheckEachPeerOut(const NodePtr &node, const std::unordered_set<NodePtr> &scope_nodes_set,
                      const std::vector<NodePtr> &scope_nodes,
                      const std::unique_ptr<ConnectionMatrix> &connectivity) {
  for (const auto &peer_out : node->GetOutAllNodes()) {
    if (scope_nodes_set.count(peer_out) > 0) {
      continue;
    }
    for (const auto &node_temp : scope_nodes) {
      if ((node_temp == nullptr) || (node_temp == node)) {
        continue;
      }
      GELOGD("Check %s and %s.", peer_out->GetName().c_str(), node_temp->GetName().c_str());

      if (connectivity->IsConnected(peer_out, node_temp)) {
        GELOGD("There is a path between %s and %s after fusing:", peer_out->GetName().c_str(),
               node_temp->GetName().c_str());
        PrintAllNodes(scope_nodes);
        return true;
      }
    }
  }
  return false;
}

bool DetectOneScope(const std::vector<NodePtr> &scope_nodes,
                    const std::unique_ptr<ConnectionMatrix> &connectivity) {
  /* Create a set for accelerating the searching. */
  const std::unordered_set<NodePtr> scope_nodes_set(scope_nodes.begin(), scope_nodes.end());

  for (const auto &node : scope_nodes) {
    if (node == nullptr) {
      continue;
    }
    if (CheckEachPeerOut(node, scope_nodes_set, scope_nodes, connectivity)) {
      return true;
    }
  }
  return false;
}
}  // namespace
graphStatus CycleDetector::Init(const ComputeGraphPtr &graph) {
  if (connectivity_ == nullptr) {
    connectivity_ = ComGraphMakeUnique<ConnectionMatrix>(graph);
    if (connectivity_ == nullptr) {
      GELOGW("Make shared failed");
      return FAILED;
    }

    const Status ret = connectivity_->Generate(graph);
    if (ret != SUCCESS) {
      GE_LOGE("Cannot generate connection matrix for graph %s.", graph->GetName().c_str());
      return FAILED;
    }
  }
  return SUCCESS;
}

bool CycleDetector::HasDetectedCycle(const std::vector<std::vector<NodePtr>> &fusion_nodes) {
  for (const auto &scope_nodes : fusion_nodes) {
    if (DetectOneScope(scope_nodes, connectivity_)) {
      return true;
    }
  }
  return false;
}

void CycleDetector::Update(const ComputeGraphPtr &graph, const std::vector<NodePtr> &fusion_nodes) {
  if (connectivity_ == nullptr) {
    GELOGW("Connectivity is empty, please call HasDetectedCycle first.");
    return;
  }
  connectivity_->Update(graph, fusion_nodes);
}

void CycleDetector::ExpandAndUpdate(const vector<ge::NodePtr> &fusion_nodes, const std::string &node_name) {
  if (connectivity_ == nullptr) {
    GELOGW("Connectivity is empty, please generate first.");
    return;
  }
  connectivity_->ExpandAndUpdate(fusion_nodes, node_name);
}
}  // namespace ge
