/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/feature/hccl_tailing_optimization_pass.h"
#include "common/op/transop_util.h"

namespace ge {
Status HcclTailingOptimizationPass::Run(ComputeGraphPtr graph) {
  for (const auto &node : graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(node);
    if (node->GetType() != HCOMALLREDUCE) {
      continue;
    }
    for (auto &out_node : node->GetOutDataNodes()) {
      if (!TransOpUtil::IsTransOp(out_node)) {
        continue;
      }

      GE_CHK_STATUS_RET_NOLOG(CopyControlEdgesForTransOp(out_node));
    }
  }
  return SUCCESS;
}
Status HcclTailingOptimizationPass::CopyControlEdgesForTransOp(NodePtr &first_trans_op) const {
  auto dst_in_ctrl_anchor = first_trans_op->GetInControlAnchor();
  GE_CHECK_NOTNULL(dst_in_ctrl_anchor);
  std::set<OutControlAnchorPtr> src_out_ctrl_anchors;
  std::vector<NodePtr> trans_op_nodes{first_trans_op};

  while (!trans_op_nodes.empty()) {
    auto trans_op_node = trans_op_nodes.back();
    trans_op_nodes.pop_back();

    for (auto &next_node : trans_op_node->GetOutDataNodes()) {
      auto in_ctrl_anchor = next_node->GetInControlAnchor();
      GE_CHECK_NOTNULL(in_ctrl_anchor);

      auto peer_out_ctrl_anchors = in_ctrl_anchor->GetPeerOutControlAnchors();

      for (auto src_ctrl_anchor : peer_out_ctrl_anchors) {
        GE_CHECK_NOTNULL(src_ctrl_anchor->GetOwnerNode());
        src_out_ctrl_anchors.emplace(src_ctrl_anchor);
      }
      if (TransOpUtil::IsTransOp(next_node)) {
        trans_op_nodes.emplace_back(next_node);
      }
    }
  }

  for (auto &src_out_ctrl_anchor : src_out_ctrl_anchors) {
    if (!src_out_ctrl_anchor->IsLinkedWith(dst_in_ctrl_anchor)) {
      GE_CHK_GRAPH_STATUS_RET(GraphUtils::AddEdge(src_out_ctrl_anchor, dst_in_ctrl_anchor),
                              "[Add][Edge] between %s->%s failed",
                              src_out_ctrl_anchor->GetOwnerNode()->GetName().c_str(),
                              first_trans_op->GetName().c_str());
    }
  }

  return SUCCESS;
}

REG_PASS_OPTION("HcclTailingOptimizationPass").LEVELS(OoLevel::kO3);
}  // namespace ge
