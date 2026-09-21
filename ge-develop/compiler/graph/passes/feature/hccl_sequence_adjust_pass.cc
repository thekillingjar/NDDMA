/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_sequence_adjust_pass.h"

#include <stack>

namespace ge {
Status HcclSequenceAdjustPass::Run(ComputeGraphPtr graph) {
  if (graph->GetParentNode() != nullptr) {
    return SUCCESS;
  }
  GELOGD("Start to run HcclSequenceAdjustPass.");

  std::vector<NodePtr> func_nodes;
  GE_CHK_STATUS_RET(GetFunctionNodesWithHcclGroup(graph, func_nodes), "Get function nodes with hccl group failed.");

  if (func_nodes.empty()) {
    GELOGD("Can not find function nodes with hccl group.");
    return SUCCESS;
  }
  GELOGD("Size of function node with hccl group is %zu.", func_nodes.size());

  GE_CHK_STATUS_RET(RebuildHcclControlRelation(graph), "rebuild hccl control relation failed.");

  GELOGD("Success to run HcclSequenceAdjustPass.");
  return SUCCESS;
}

Status HcclSequenceAdjustPass::GetFunctionNodesWithHcclGroup(const ComputeGraphPtr &graph,
                                                             std::vector<NodePtr> &func_nodes) const {
  GELOGD("Start to get function nodes with hccl group.");

  std::string group_id;
  for (const auto &node : graph->GetDirectNode()) {
    const auto &op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    if (!op_desc->GetSubgraphInstanceNames().empty() &&
        AttrUtils::GetStr(op_desc, ATTR_NAME_HCCL_FUSED_GROUP, group_id)) {
      func_nodes.emplace_back(node);
    }
  }

  GELOGD("Success to get function nodes with hccl group.");
  return SUCCESS;
}

bool HcclSequenceAdjustPass::HasRelationPath(const NodePtr &second_hccl, const NodePtr &last_hccl) const {
  std::stack<NodePtr> out_nodes;
  out_nodes.push(second_hccl);
  std::set<NodePtr> node_seen;
  while (!out_nodes.empty()) {
    const auto one_node = out_nodes.top();
    out_nodes.pop();
    if (one_node == last_hccl) {
      return true;
    }
    for (const auto &out_node : one_node->GetOutAllNodes()) {
      if (node_seen.emplace(out_node).second) {
        out_nodes.push(out_node);
      }
    }
  }
  return false;
}

Status HcclSequenceAdjustPass::RebuildHcclControlRelation(const ComputeGraphPtr &graph) const {
  GELOGD("Start to rebulid hccl control relation of graph: %s.", graph->GetName().c_str());

  NodePtr last_hccl = nullptr;
  NodePtr second_hccl = nullptr;
  for (const auto &node : graph->GetDirectNode()) {
    if (node->GetType() == HCOMALLREDUCE) {
      const auto &out_ctrl_nodes = node->GetOutControlNodes();
      if (std::none_of(out_ctrl_nodes.begin(), out_ctrl_nodes.end(),
                       [](const NodePtr &n) { return (n->GetType() == HCOMALLREDUCE); })) {
        const auto &in_ctrl_nodes = node->GetInControlNodes();
        if (std::any_of(in_ctrl_nodes.begin(), in_ctrl_nodes.end(),
                        [](const NodePtr &n) { return (n->GetType() == HCOMALLREDUCE); })) {
          second_hccl = node;
          GELOGD("Find second last hccl node: %s.", node->GetName().c_str());
        } else {
          last_hccl = node;
          GELOGD("Find last hccl node: %s.", node->GetName().c_str());
        }
      }
    }
  }
  if ((last_hccl == nullptr) || (second_hccl == nullptr)) {
    GELOGW("Can not find optimizable HcomAllReduce nodes..");
    return SUCCESS;
  }
  if (HasRelationPath(second_hccl, last_hccl)) {
    GELOGW("Exist path from %s to %s, skip link.", second_hccl->GetName().c_str(), last_hccl->GetName().c_str());
    return SUCCESS;
  }
  const auto &out_ctrl_anchor = last_hccl->GetOutControlAnchor();
  GE_CHECK_NOTNULL(out_ctrl_anchor);
  const auto &in_ctrl_anchor = second_hccl->GetInControlAnchor();
  GE_CHECK_NOTNULL(in_ctrl_anchor);
  GE_CHK_STATUS_RET(out_ctrl_anchor->LinkTo(in_ctrl_anchor),
                    "Add link from %s to %s failed.", last_hccl->GetName().c_str(), second_hccl->GetName().c_str());
  GELOGD("Add control edge from %s to %s.", last_hccl->GetName().c_str(), second_hccl->GetName().c_str());

  GELOGD("Success to rebulid hccl control relation of graph: %s.", graph->GetName().c_str());
  return SUCCESS;
}

REG_PASS_OPTION("HcclSequenceAdjustPass").LEVELS(OoLevel::kO3);
}  // namespace ge
