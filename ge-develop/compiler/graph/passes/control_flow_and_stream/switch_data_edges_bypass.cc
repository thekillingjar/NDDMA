/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/control_flow_and_stream/switch_data_edges_bypass.h"

#include <atomic>
#include "framework/common/debug/log.h"
#include "common/plugin/ge_make_unique_util.h"
#include "framework/common/op/ge_op_utils.h"
#include "framework/common/util.h"
#include "graph/utils/node_utils.h"
#include "common/checker.h"

namespace ge {
namespace {
bool IsSwitchInWhileLoop(const NodePtr &node) {
  auto pred_anchor = node->GetInDataAnchor(SWITCH_PRED_INPUT);
  if (pred_anchor == nullptr) {
    GELOGW("The switch node %s does not have a pred in anchor, the node may be invalid", node->GetName().c_str());
    return true;
  }
  auto pred_node_anchor = pred_anchor->GetPeerOutAnchor();
  if (pred_node_anchor == nullptr) {
    GELOGW("The switch node %s does not have a pred in node, the graph may be invalid", node->GetName().c_str());
    return true;
  }
  auto pred_node = pred_node_anchor->GetOwnerNode();
  if (pred_node->GetType() == LOOPCOND) {
    GELOGD("The switch node %s is in a while loop, skip the bypass process", node->GetName().c_str());
    return true;
  }
  return false;
}
std::vector<std::pair<NodePtr, InDataAnchorPtr>> GetOutDataNodesByIndex(const NodePtr &node, int32_t index) {
  auto out_anchor = node->GetOutDataAnchor(index);
  if (out_anchor == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Node:%s(%s) has no index:%d out data anchor, check invalid",
                       node->GetName().c_str(), node->GetType().c_str(), index);
    GELOGE(PARAM_INVALID, "[Get][OutDataNodes] of index %d from node %s failed, the anchor does not exist",
           index, node->GetName().c_str());
    return {};
  }
  std::vector<std::pair<NodePtr, InDataAnchorPtr>> nodes_and_anchors;
  for (const auto &in_anchor : out_anchor->GetPeerInDataAnchors()) {
    auto out_node = in_anchor->GetOwnerNode();
    nodes_and_anchors.emplace_back(out_node, in_anchor);
  }
  return nodes_and_anchors;
}
std::pair<NodePtr, OutDataAnchorPtr> GetInDataNodeByIndex(const NodePtr &node, int32_t index) {
  auto in_anchor = node->GetInDataAnchor(index);
  if (in_anchor == nullptr) {
    GELOGD("Failed to get in data node of index %d from node %s, the anchor does not exist", index,
           node->GetName().c_str());
    return {};
  }
  auto out_anchor = in_anchor->GetPeerOutAnchor();
  if (out_anchor == nullptr) {
    GELOGD("Failed to get in data node of index %d from node %s, the data input does not exist", index,
           node->GetName().c_str());
    return {};
  }
  return {out_anchor->GetOwnerNode(), out_anchor};
}
NodePtr AddIdentityAfterNode(const NodePtr &node, int32_t index) {
  static std::atomic_long atomic_identity_counter(0);
  auto identity_counter = atomic_identity_counter.fetch_add(1);

  auto node_desc = node->GetOpDesc();
  if (node_desc == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "OpDesc in node is nullptr, check invalid");
    GELOGE(INTERNAL_ERROR, "[Get][OpDesc] failed, the op desc is nullptr");
    return nullptr;
  }
  auto tensor = node_desc->GetOutputDescPtr(index);
  if (tensor == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Node:%s(%s) has no index:%d output tensor, check invalid",
                       node_desc->GetName().c_str(), node_desc->GetType().c_str(), index);
    GELOGE(INTERNAL_ERROR, "[Get][OutputDescPtr] failed, Node:%s(%s) has no index:%d output tensor",
           node_desc->GetName().c_str(), node_desc->GetType().c_str(), index);
    return nullptr;
  }
  auto anchor = node->GetOutDataAnchor(index);
  if (anchor == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Node:%s(%s) has no index:%d out data anchor, check invalid",
                       node->GetName().c_str(), node->GetType().c_str(), index);
    GELOGE(OUT_OF_MEMORY, "[Get][OutDataAnchor] failed, Node:%s(%s) has no index:%d out data anchor",
           node->GetName().c_str(), node->GetType().c_str(), index);
    return nullptr;
  }

  auto identity_opdesc =
      MakeShared<OpDesc>("SwitchDataEdgesByPass_Identity_" + std::to_string(identity_counter), IDENTITY);
  if (identity_opdesc == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "New OpDesc failed");
    GELOGE(OUT_OF_MEMORY, "[New][OpDesc] failed");
    return nullptr;
  }
  auto ret1 = identity_opdesc->AddInputDesc("x", *tensor);
  auto ret2 = identity_opdesc->AddOutputDesc("y", *tensor);
  NodePtr identity = node->GetOwnerComputeGraph()->InsertNode(node, identity_opdesc);
  if (ret1 != GRAPH_SUCCESS || ret2 != GRAPH_SUCCESS || identity == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Add input ouput desc to op:%s(%s) failed or add it to graph:%s failed",
                      identity_opdesc->GetName().c_str(), identity_opdesc->GetType().c_str(),
                      node->GetOwnerComputeGraph()->GetName().c_str());
    GELOGE(OUT_OF_MEMORY, "[Check][Param] Add input ouput desc to op:%s(%s) failed or add it to graph:%s failed",
           identity_opdesc->GetName().c_str(), identity_opdesc->GetType().c_str(),
           node->GetOwnerComputeGraph()->GetName().c_str());
    return nullptr;
  }
  (void)anchor->LinkTo(identity->GetInDataAnchor(0));

  return identity;
}
NodePtr AddMemcpyBeforeNode(const NodePtr &node, int32_t index) {
  static std::atomic_long atomic_counter(0);
  auto counter = atomic_counter.fetch_add(1);

  auto node_desc = node->GetOpDesc();
  if (node_desc == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "OpDesc in node is nullptr, check invalid");
    GELOGE(INTERNAL_ERROR, "[Get][OpDesc] failed, OpDesc in node is nullptr");
    return nullptr;
  }
  auto tensor = node_desc->GetInputDescPtr(index);
  if (tensor == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Node:%s(%s) has no index:%d input tensor, check invalid",
                       node_desc->GetName().c_str(), node_desc->GetType().c_str(), index);
    GELOGE(INTERNAL_ERROR, "[Get][InputDescPtr] failed, Node:%s(%s) has no index:%d input tensor",
           node_desc->GetName().c_str(), node_desc->GetType().c_str(), index);
    return nullptr;
  }
  auto anchor = node->GetInDataAnchor(index);
  if (anchor == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Node:%s(%s) has no index:%d in data anchor, check invalid",
                       node->GetName().c_str(), node->GetType().c_str(), index);
    GELOGE(INTERNAL_ERROR, "[Get][InDataAnchor] failed, Node:%s(%s) has no index:%d in data anchor",
           node->GetName().c_str(), node->GetType().c_str(), index);
    return nullptr;
  }

  auto memcpy_opdesc = MakeShared<OpDesc>("SwitchDataEdgesByPass_Memcpy_" + std::to_string(counter), MEMCPYASYNC);
  if (memcpy_opdesc == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "New OpDesc failed");
    GELOGE(OUT_OF_MEMORY, "[New][OpDesc] failed");
    return nullptr;
  }
  auto ret1 = memcpy_opdesc->AddInputDesc(*tensor);
  auto ret2 = memcpy_opdesc->AddOutputDesc(*tensor);
  auto memcpy_node = node->GetOwnerComputeGraph()->AddNode(memcpy_opdesc);
  if (ret1 != GRAPH_SUCCESS || ret2 != GRAPH_SUCCESS || memcpy_node == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Add input ouput desc to op:%s(%s) failed or add it to graph:%s failed",
                      memcpy_opdesc->GetName().c_str(), memcpy_opdesc->GetType().c_str(),
                      node->GetOwnerComputeGraph()->GetName().c_str());
    GELOGE(OUT_OF_MEMORY, "[Check][Param] Add input ouput desc to op:%s(%s) failed or add it to graph:%s failed",
           memcpy_opdesc->GetName().c_str(), memcpy_opdesc->GetType().c_str(),
           node->GetOwnerComputeGraph()->GetName().c_str());
    return nullptr;
  }
  auto out_data_anchor = memcpy_node->GetOutDataAnchor(0);
  GE_ASSERT_NOTNULL(out_data_anchor);
  (void)out_data_anchor->LinkTo(anchor);

  return memcpy_node;
}
Status BypassSwitchOut(const NodePtr &switch_node, int32_t out_index) {
  auto nodes_and_anchors = GetOutDataNodesByIndex(switch_node, out_index);
  if (nodes_and_anchors.empty()) {
    GELOGD("The switch node %s does not has out branch %d, skip the bypass process", switch_node->GetName().c_str(),
           out_index);
    return SUCCESS;
  }

  auto data_node_and_anchor = GetInDataNodeByIndex(switch_node, SWITCH_DATA_INPUT);
  if (data_node_and_anchor.first == nullptr) {
    GELOGW("Can not bypass switch node %s, the node does not has a data input", switch_node->GetName().c_str());
    return SUCCESS;
  }

  auto identity = AddIdentityAfterNode(switch_node, out_index);
  GE_CHECK_NOTNULL(identity);

  std::set<Node *> connected_nodes;
  for (const auto &node_and_anchor : nodes_and_anchors) {
    auto head_anchor = node_and_anchor.second;
    head_anchor->UnlinkAll();

    auto head_node = node_and_anchor.first;
    GE_CHECK_NOTNULL(head_node);
    auto head_node_type = NodeUtils::GetNodeType(*head_node);
    if (head_node_type == MEMCPYASYNC) {
      // if the switch connect to the merge directly, insert memcpy before merge
      auto memcpy_node = AddMemcpyBeforeNode(head_node, head_anchor->GetIdx());
      GE_CHECK_NOTNULL(memcpy_node);
      GELOGD("Add memcpy %s before merge node %s", memcpy_node->GetName().c_str(), head_node->GetName().c_str());
      head_node = memcpy_node;
      head_anchor = memcpy_node->GetInDataAnchor(0);
    }
    (void)data_node_and_anchor.second->LinkTo(head_anchor);
    GE_CHECK_NOTNULL(head_node);
    if (connected_nodes.insert(head_node.get()).second) {
      (void)identity->GetOutControlAnchor()->LinkTo(head_node->GetInControlAnchor());
    }
  }
  GELOGI("Bypass switch %s out index %d success", switch_node->GetName().c_str(), out_index);
  return SUCCESS;
}
}  // namespace
Status SwitchDataEdgesBypass::Run(ComputeGraphPtr graph) {
  for (const auto &node : graph->GetDirectNode()) {
    auto ret = BypassSwitch(node);
    GE_CHK_STATUS_RET(ret, "[Bypass][Switch] node %s failed", node->GetName().c_str());
  }
  return SUCCESS;
}
Status SwitchDataEdgesBypass::BypassSwitch(const NodePtr &node) const {
  auto node_type = NodeUtils::GetNodeType(*node);
  if ((node_type != SWITCH) && (node_type != REFSWITCH)) {
    return SUCCESS;
  }
  if (IsSwitchInWhileLoop(node)) {
    return SUCCESS;
  }

  auto ret = BypassSwitchOut(node, SWITCH_FALSE_OUTPUT);
  GE_CHK_STATUS_RET(ret, "[Bypass][Switch] node %s false output failed", node->GetName().c_str());
  ret = BypassSwitchOut(node, SWITCH_TRUE_OUTPUT);
  GE_CHK_STATUS_RET(ret, "[Bypass][Switch] node %s true output failed", node->GetName().c_str());

  return SUCCESS;
}

REG_PASS_OPTION("SwitchDataEdgesBypass").LEVELS(OoLevel::kO1);
}  // namespace ge
