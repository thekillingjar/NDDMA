/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "continues_broadcast_optimization.h"
#include "attr_utils.h"
#include "ascir_ops.h"
#include "ascgraph_info_complete.h"
#include "ascir_ops_utils.h"
#include "node_utils.h"
#include "schedule_utils.h"
#include "graph_utils.h"

using namespace ge::ops;
using namespace ge::ascir_op;
namespace optimize {
Status ContinuesBroadcastOptimizationPass::RunPass(ge::AscGraph &graph) {
  std::set<std::shared_ptr<ge::AscNode>> visited_brc_nodes;
  for (const auto &node : graph.GetAllNodes()) {
    if (visited_brc_nodes.find(node) != visited_brc_nodes.end()) {
      continue;
    }
    GE_CHECK_NOTNULL(node);
    GELOGD("Check node : %s %s, out size: %d, in size: %d", node->GetTypePtr(), node->GetNamePtr(),
           node->GetOutDataNodesSize(), node->GetInDataNodesSize());
    if (!IsOps<Broadcast>(node)) {
      continue;
    }
    GELOGD("Find Broadcast node [%s], start to check", node->GetNamePtr());

    // Find more continuous Broadcast nodes.
    std::vector<ge::AscNodePtr> continues_brc_nodes;
    if (!ScheduleUtils::FindContinuesBroadcastNode(node, continues_brc_nodes) || continues_brc_nodes.size() <= 1) {
      continues_brc_nodes.clear();
      continue;
    }

    // Relink first Broadcast input to last Broadcast input
    GE_CHECK_NOTNULL(node->GetInDataAnchor(0));
    auto pre_node_out_anchor = node->GetInDataAnchor(0)->GetPeerOutAnchor();
    GE_ASSERT_NOTNULL(pre_node_out_anchor);
    const auto &last_brc_node = continues_brc_nodes[continues_brc_nodes.size() - 1UL];
    visited_brc_nodes.insert(last_brc_node);
    continues_brc_nodes.pop_back();
    const auto &pre_last_brc_node = continues_brc_nodes[continues_brc_nodes.size() - 1UL];
    GE_ASSERT_SUCCESS(ge::GraphUtils::ReplaceEdgeSrc(pre_last_brc_node->GetOutDataAnchor(0),
                                                     last_brc_node->GetInDataAnchor(0), pre_node_out_anchor));
    // Remove all broadcast nodes.
    for (const auto &continues_brc_node : continues_brc_nodes) {
      visited_brc_nodes.insert(continues_brc_node);
      GELOGD("Continuous Broadcast node [%s] can be optimized, remove it.", continues_brc_node->GetNamePtr());
      ge::NodeUtils::UnlinkAll(*continues_brc_node);
      GE_CHECK_NOTNULL(continues_brc_node->GetOwnerComputeGraph());
      ge::GraphUtils::RemoveNodeWithoutRelink(continues_brc_node->GetOwnerComputeGraph(), continues_brc_node);
    }
  }
  return ge::SUCCESS;
}
}  // namespace optimize
