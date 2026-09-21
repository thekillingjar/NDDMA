/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/control_flow_and_stream/enter_pass.h"

#include "graph/debug/ge_attr_define.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/debug/log.h"
#include "graph/utils/graph_utils.h"
#include "common/checker.h"
#include "graph_metadef/common/ge_common/util.h"

namespace {
const size_t kOutNodesNum = 1;
const size_t kInCtrlNodesNum = 1;
}

namespace ge {
Status EnterPass::Run(NodePtr &node) {
  GE_CHECK_NOTNULL(node);
  if ((node->GetType() != ENTER) && (node->GetType() != REFENTER)) {
    return SUCCESS;
  }

  GELOGD("EnterPass running");
  // enter node has only one input
  GE_ASSERT_TRUE(!node->GetInDataNodes().empty());
  NodePtr in_node = node->GetInDataNodes().at(0);
  GE_CHECK_NOTNULL(in_node);

  if ((in_node->GetType() != CONSTANT) && (in_node->GetType() != CONSTANTOP)) {
    return SUCCESS;
  }

  const bool need_remove_flag = in_node->GetInControlNodes().empty() && node->GetInControlNodes().empty();
  if (!need_remove_flag) {
    return SUCCESS;
  }
  if (node->GetOutDataNodes().empty()) {
    for (auto &out_ctrl_node : node->GetOutControlNodes()) {
      if (out_ctrl_node == nullptr) {
        continue;
      }
      if (out_ctrl_node->GetInNodesSize() == kInCtrlNodesNum) {
        return SUCCESS;
      }
      GELOGI("Remove control edge from %s to %s.", node->GetName().c_str(), out_ctrl_node->GetName().c_str());
      GE_ASSERT_SUCCESS(GraphUtils::RemoveEdge(node->GetOutControlAnchor(), out_ctrl_node->GetInControlAnchor()));
    }
  } else {
    if (OptimizeEnterWithOnlyDataOut(node, in_node) != SUCCESS) {
      GELOGE(FAILED, "[Optimize][EnterNode] [%s] with only out data node failed.", node->GetName().c_str());
      return FAILED;
    }
    if (UnlinkCtrlEdgeBeforeConst(node) != SUCCESS) {
      GELOGE(FAILED, "[Unlink][ControlEdge] before const of node[%s]'s out nodes failed.", node->GetName().c_str());
      return FAILED;
    }
  }

  GELOGD("EnterPass success");
  return SUCCESS;
}

Status EnterPass::OptimizeEnterWithOnlyDataOut(const NodePtr &node, const NodePtr &in_node) {
  if ((in_node->GetOutAllNodes().size() != kOutNodesNum) || !node->GetOutControlNodes().empty()) {
    return SUCCESS;
  }
  bool is_constant_flag = true;
  (void)AttrUtils::GetBool(node->GetOpDesc(), ENTER_ATTR_CONSTANT_FLAG, is_constant_flag);
  if (!is_constant_flag) {
    return SUCCESS;
  }

  GE_CHECK_NOTNULL(in_node->GetOutDataAnchor(0));
  GE_CHK_GRAPH_STATUS_RET(in_node->GetOutDataAnchor(0)->Unlink(node->GetInDataAnchor(0)));
  const auto &out_data_anchor = node->GetOutDataAnchor(0);
  GE_CHECK_NOTNULL(out_data_anchor);
  for (const auto &peer_in_data_anchor : out_data_anchor->GetPeerInDataAnchors()) {
    GE_CHK_GRAPH_STATUS_RET(out_data_anchor->Unlink(peer_in_data_anchor));
    GE_CHK_GRAPH_STATUS_RET(in_node->GetOutDataAnchor(0)->LinkTo(peer_in_data_anchor));
  }
  GE_CHK_GRAPH_STATUS_RET(GraphUtils::RemoveNodeWithoutRelink(node->GetOwnerComputeGraph(), node));
  AddNodeDeleted(node);
  AddRePassNodesWithInOut(in_node);

  return SUCCESS;
}

Status EnterPass::UnlinkCtrlEdgeBeforeConst(const NodePtr &node) const {
  auto out_ctrl_nodes = node->GetOutControlNodes();
  if (out_ctrl_nodes.empty()) {
    return SUCCESS;
  }
  auto out_ctrl_anchor = node->GetOutControlAnchor();
  GE_CHECK_NOTNULL(out_ctrl_anchor);

  for (auto &out_ctrl_node : out_ctrl_nodes) {
    GE_CHECK_NOTNULL(out_ctrl_node);
    if ((out_ctrl_node->GetType() != CONSTANT) && (out_ctrl_node->GetType() != CONSTANTOP)) {
      continue;
    }
    auto in_ctrl_nodes = out_ctrl_node->GetInControlNodes();
    if (in_ctrl_nodes.size() != kInCtrlNodesNum) {
      continue;
    }

    // Skip when has merge out
    bool has_merge_out = false;
    auto out_nodes_of_const = out_ctrl_node->GetOutAllNodes();
    for (const auto &out_node_of_const : out_nodes_of_const) {
      GE_CHECK_NOTNULL(out_node_of_const);
      if (out_node_of_const->GetType() == MERGE || out_node_of_const->GetType() == REFMERGE) {
        has_merge_out = true;
        break;
      }
    }
    if (has_merge_out) {
      continue;
    }

    GELOGI("Unlink control edge from %s to %s.", node->GetName().c_str(), out_ctrl_node->GetName().c_str());
    GE_CHK_GRAPH_STATUS_RET(out_ctrl_anchor->Unlink(out_ctrl_node->GetInControlAnchor()));
    for (auto &out_node_of_const : out_nodes_of_const) {
      if (!out_ctrl_anchor->IsLinkedWith(out_node_of_const->GetInControlAnchor())) {
        GELOGI("Link control edge from %s to %s.", node->GetName().c_str(), out_node_of_const->GetName().c_str());
        GE_CHK_GRAPH_STATUS_RET(out_ctrl_anchor->LinkTo(out_node_of_const->GetInControlAnchor()));
      }
    }
  }
  return SUCCESS;
}

REG_PASS_OPTION("EnterPass").LEVELS(OoLevel::kO3);
}  // namespace ge
