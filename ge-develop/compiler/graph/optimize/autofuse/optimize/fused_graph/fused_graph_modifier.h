/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPTIMIZE_FUSED_GRAPH_FUSED_GRAPH_MODIFIER_H_
#define OPTIMIZE_FUSED_GRAPH_FUSED_GRAPH_MODIFIER_H_
#include "ascgen_log.h"
#include "schedule_result.h"

#include <queue>
namespace optimize {
struct OutAnchorAttr {
  int64_t linked_output_ir_idx = -1;
  int64_t depends = -1;
  int64_t used_ws_idx = -1;
};

struct ProcessNodesContext {
  std::map<const ge::Node *, std::map<int64_t, OutAnchorAttr>> &nodes_to_out_anchor_idx_to_attr;
  std::set<int64_t> &free_workspace_id;
  std::set<int64_t> &data_used_ids;
};

class FusedGraphModifier {
 public:
  static Status SubgraphConnectionsToWorkspace(const ge::ComputeGraphPtr &fused_graph,
                                               std::map<ge::Node *, ge::AscGraph> &asc_backend_to_ascgraph);

  static Status ChangeStartingOutputToWorkspace(std::vector<ascir::ScheduleGroup> &schedule_groups);

 private:
  static Status InitAscbcOutAnchorAttr(
      const ge::ComputeGraphPtr &fused_graph,
      std::map<const ge::Node *, std::map<int64_t, OutAnchorAttr>> &nodes_to_out_anchor_idx_to_attr);

  static Status ProcessOutputNodes(const ge::AscNodePtr &sub_out_node, const ge::Node *const ascbc_node,
                                   ge::AscGraph &asc_graph, ProcessNodesContext &context, int64_t &max_workspace_num);

  static Status ProcessDataNodes(const ge::AscNodePtr &sub_data_node, const ge::Node *const ascbc_node,
                                 ge::AscGraph &asc_graph, ProcessNodesContext &context);
  static int64_t ReuseWorkspaceId(std::set<int64_t> &free_ws_ids, const std::set<int64_t> &data_used_ids,
                                  int64_t &max_workspace_num);
};
}  // namespace optimize

#endif  // OPTIMIZE_FUSED_GRAPH_FUSED_GRAPH_MODIFIER_H_
