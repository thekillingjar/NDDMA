/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/variable_optimize/var_is_initialized_op_pass.h"
#include <memory>
#include <utility>
#include "framework/common/debug/ge_log.h"
#include "common/plugin/ge_make_unique_util.h"
#include "graph/anchor.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/node.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "checker.h"

namespace ge {
namespace {
const int32_t kAssignVarRefIndex = 0;
const int32_t kVarIsInitializedIOCnt = 1;
const int32_t kVarIsInitVarInputIndex = 0;
}  // namespace
Status VarIsInitializedOpPass::Run(NodePtr &node) {
  GE_CHECK_NOTNULL(node);
  auto ret = UpdateInitedVars(node);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Call][UpdateInitedVars] for node:%s failed", node->GetName().c_str());
    return ret;
  }

  if ((node->GetType() != VARISINITIALIZEDOP) && (node->GetType() != ISVARIABLEINITIALIZED)) {
    return SUCCESS;
  }

  bool inited = false;
  if (CheckSrcNode(node, inited) != SUCCESS) {
    GELOGE(ret, "[Call][CheckSrcNode] for node:%s failed", node->GetName().c_str());
    return FAILED;
  }
  GELOGI("The variable inited status %s on node %s",
         inited ? "true" : "false", node->GetName().c_str());

  ret = ChangeNodeToConstant(node, inited);
  GELOGI("Change VarIsInitializedOp %s to be Constant %s end.",
         node->GetName().c_str(), inited ? "true" : "false");
  return ret;
}

Status VarIsInitializedOpPass::CheckSrcNode(const NodePtr &node, bool &inited) const {
  GE_CHECK_NOTNULL(node);
  auto input_nodes = node->GetInDataNodes();
  if (input_nodes.size() != kVarIsInitializedIOCnt) {
    REPORT_INNER_ERR_MSG("E19999", "In data node num:%zu of node:%s(%s) not equal to %d, check invalid",
                       input_nodes.size(), node->GetName().c_str(), node->GetType().c_str(), kVarIsInitializedIOCnt);
    GELOGE(FAILED, "[Check][Param] In data node num:%zu of node:%s(%s) not equal to %d.",
           input_nodes.size(), node->GetName().c_str(), node->GetType().c_str(), kVarIsInitializedIOCnt);
    return FAILED;
  }

  auto &input_node = input_nodes.at(kVarIsInitVarInputIndex);
  GE_CHECK_NOTNULL(input_node);
  auto input_node_name = input_node->GetName();
  auto input_node_type = input_node->GetType();
  if (input_node_type != VARIABLE) {
    REPORT_INNER_ERR_MSG("E19999", "Index:%d In data node of node:%s(%s), type:%s not %s, check invalid",
                       kVarIsInitVarInputIndex, node->GetName().c_str(), node->GetType().c_str(),
                       input_node_type.c_str(), VARIABLE);
    GELOGE(FAILED, "[Check][Param] Index:%d In data node of node:%s(%s), type:%s not equal to %s.",
           kVarIsInitVarInputIndex, node->GetName().c_str(), node->GetType().c_str(),
           input_node_type.c_str(), VARIABLE);
    return FAILED;
  }

  // initialized and initialized check graph must not be in the same graph
  ComputeGraphPtr compute_graph = node->GetOwnerComputeGraph();
  auto session_id = compute_graph->GetSessionID();
  const auto &var_manager = VarManager::Instance(session_id);
  GE_ASSERT_NOTNULL(var_manager);
  if (var_manager->IsVarExist(input_node_name)) {
    inited = true;
    return SUCCESS;
  }
  GE_CHECK_NOTNULL(input_node->GetOpDesc());
  inited = IsVarInitedOnTheGraphAndNode(node, input_node->GetOpDesc()->GetId());
  return SUCCESS;
}

Status VarIsInitializedOpPass::CreateConstant(const NodePtr &node, OpDescPtr &op_desc, bool inited) const {
  GE_CHECK_NOTNULL(node);
  // 1. get OpDesc of VarIsInitializedOp
  OpDescPtr original_op_desc = node->GetOpDesc();
  if (original_op_desc == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "OpDesc in node is nullptr, check invalid");
    GELOGE(FAILED, "[Get][OpDesc] failed, Op desc of node must not be null.");
    return FAILED;
  }
  GeTensorDesc original_desc = original_op_desc->GetOutputDesc(0);

  // 2. create Constant OpDesc
  op_desc = MakeShared<OpDesc>(node->GetName().c_str(), CONSTANT);
  if (op_desc == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "New OpDesc failed");
    GELOGE(FAILED, "[New][OpDesc] failed.");
    return FAILED;
  }

  // 3. create attr value of Constant, is a tensor
  bool val = inited;
  GeTensorPtr const_tensor_ptr = MakeShared<GeTensor>(original_desc, reinterpret_cast<uint8_t *>(&val), sizeof(bool));
  if (const_tensor_ptr == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "New GeTensor failed");
    GELOGE(FAILED, "[New][GeTensor] failed.");
    return FAILED;
  }
  if (!AttrUtils::SetTensor(op_desc, ATTR_NAME_WEIGHTS, const_tensor_ptr)) {
    REPORT_INNER_ERR_MSG("E19999", "Set Attr:%s to op:%s(%s) failed", ATTR_NAME_WEIGHTS.c_str(),
                      op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Set][Attr] %s to op:%s(%s) failed", ATTR_NAME_WEIGHTS.c_str(),
           op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return FAILED;
  }

  // 4. set Constant output desc
  GE_CHK_STATUS_RET(op_desc->AddOutputDesc(original_desc),
                    "[Add][OutputDesc] to op:%s(%s) failed",
                    op_desc->GetName().c_str(), op_desc->GetType().c_str());
  return SUCCESS;
}

Status VarIsInitializedOpPass::ProcessInAnchor(NodePtr &node, NodePtr &new_node) const {
  GE_CHECK_NOTNULL(node);
  GE_CHECK_NOTNULL(new_node);
  auto in_anchors = node->GetAllInDataAnchors();
  auto out_anchors = node->GetAllOutDataAnchors();
  if ((in_anchors.size() != kVarIsInitializedIOCnt) ||
      (out_anchors.size() != kVarIsInitializedIOCnt)) {
    REPORT_INNER_ERR_MSG("E19999", "In data anchor num:%zu and out data anchor num:%zu of node:%s(%s), "
                       "must be equal to %d, check invalid", in_anchors.size(), out_anchors.size(),
                       node->GetName().c_str(), node->GetType().c_str(), kVarIsInitializedIOCnt);
    GELOGE(FAILED, "[Check][Param] In data anchor num:%zu and out data anchor num:%zu of node:%s(%s), "
           "must be equal to %d.", in_anchors.size(), out_anchors.size(),
           node->GetName().c_str(), node->GetType().c_str(), kVarIsInitializedIOCnt);
    return FAILED;
  }

  // 1. delete in data anchor of VarIsInitializedOp node
  auto &in_anchor = in_anchors.at(kVarIsInitVarInputIndex);
  GE_CHECK_NOTNULL(in_anchor);
  auto peer_out_anchor = in_anchor->GetPeerOutAnchor();
  GE_CHECK_NOTNULL(peer_out_anchor);
  if (GraphUtils::RemoveEdge(in_anchor, peer_out_anchor) != GRAPH_SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Remove edge between op:%s(%s)(index:%d) and op:%s(%s)(index:%d) failed",
                      in_anchor->GetOwnerNode()->GetName().c_str(), in_anchor->GetOwnerNode()->GetType().c_str(),
                      in_anchor->GetIdx(), peer_out_anchor->GetOwnerNode()->GetName().c_str(),
                      peer_out_anchor->GetOwnerNode()->GetType().c_str(), peer_out_anchor->GetIdx());
    GELOGE(FAILED, "[Remove][Edge] between op:%s(%s)(index:%d) and op:%s(%s)(index:%d) failed",
           in_anchor->GetOwnerNode()->GetName().c_str(), in_anchor->GetOwnerNode()->GetType().c_str(),
           in_anchor->GetIdx(), peer_out_anchor->GetOwnerNode()->GetName().c_str(),
           peer_out_anchor->GetOwnerNode()->GetType().c_str(), peer_out_anchor->GetIdx());
    return FAILED;
  }
  auto src_node = peer_out_anchor->GetOwnerNode();
  if (GraphUtils::AddEdge(src_node->GetOutControlAnchor(), new_node->GetInControlAnchor()) != GRAPH_SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Add control edge between op:%s(%s) and op:%s(%s) failed",
                      src_node->GetName().c_str(), src_node->GetType().c_str(),
                      new_node->GetName().c_str(), new_node->GetType().c_str());
    GELOGE(FAILED, "[Add][ControlEdge] between op:%s(%s) and op:%s(%s) failed",
           src_node->GetName().c_str(), src_node->GetType().c_str(),
           new_node->GetName().c_str(), new_node->GetType().c_str());
    return FAILED;
  }

  if (GraphUtils::MoveInCtrlEdges(node, new_node) != GRAPH_SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Move in control edge from node:%s(%s) to node:%s(%s) failed",
                      node->GetName().c_str(), node->GetType().c_str(),
                      new_node->GetName().c_str(), new_node->GetType().c_str());
    GELOGE(FAILED, "[Move][InCtrlEdges] from node:%s(%s) to node:%s(%s) failed",
           node->GetName().c_str(), node->GetType().c_str(),
           new_node->GetName().c_str(), new_node->GetType().c_str());
    return FAILED;
  }

  if (GraphUtils::MoveOutCtrlEdges(node, new_node) != GRAPH_SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Move out control edge from node:%s(%s) to node:%s(%s) failed",
                      node->GetName().c_str(), node->GetType().c_str(),
                      new_node->GetName().c_str(), new_node->GetType().c_str());
    GELOGE(FAILED, "[Move][OutCtrlEdges] from node:%s(%s) to node:%s(%s) failed",
           node->GetName().c_str(), node->GetType().c_str(),
           new_node->GetName().c_str(), new_node->GetType().c_str());
    return FAILED;
  }

  return SUCCESS;
}

Status VarIsInitializedOpPass::ChangeNodeToConstant(NodePtr &node, bool inited) {
  GE_CHECK_NOTNULL(node);
  ComputeGraphPtr graph = node->GetOwnerComputeGraph();
  OpDescPtr constant_op_desc = nullptr;
  if (CreateConstant(node, constant_op_desc, inited) != SUCCESS) {
    GELOGE(FAILED, "[Create][Constant] failed, node:%s", node->GetName().c_str());
    return FAILED;
  }

  NodePtr const_node = graph->InsertNode(node, constant_op_desc);
  if (const_node == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Add node:%s(%s) to graph:%s front failed",
                      constant_op_desc->GetName().c_str(), constant_op_desc->GetType().c_str(),
                      graph->GetName().c_str());
    GELOGE(FAILED, "[Add][Node] %s(%s) to graph:%s front failed",
           constant_op_desc->GetName().c_str(), constant_op_desc->GetType().c_str(), graph->GetName().c_str());
    return FAILED;
  }

  if (ProcessInAnchor(node, const_node) != SUCCESS) {
    GELOGE(FAILED, "[Process][InAnchor] failed, node:%s", node->GetName().c_str());
    return FAILED;
  }

  if (NodeUtils::MoveOutputEdges(node, const_node) != GRAPH_SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Move out edge from node:%s(%s) to node:%s(%s) failed",
                      node->GetName().c_str(), node->GetType().c_str(),
                      const_node->GetName().c_str(), const_node->GetType().c_str());
    GELOGE(FAILED, "[Move][OutputEdges] from node:%s(%s) to node:%s(%s) failed",
           node->GetName().c_str(), node->GetType().c_str(),
           const_node->GetName().c_str(), const_node->GetType().c_str());
    return FAILED;
  }

  if (GraphUtils::RemoveNodeWithoutRelink(graph, node) != SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Remove node:%s(%s) without relink in graph:%s failed",
                      node->GetName().c_str(), node->GetType().c_str(), graph->GetName().c_str());
    GELOGE(FAILED, "[Remove][Node] %s(%s) without relink in graph:%s failed",
           node->GetName().c_str(), node->GetType().c_str(), graph->GetName().c_str());
    return FAILED;
  }

  AddRePassNodesWithInOut(const_node);
  // delete VarIsInitializedOp node from the graph
  AddNodeDeleted(node);
  return SUCCESS;
}

Status VarIsInitializedOpPass::UpdateInitedVars(const NodePtr &node) {
  GE_CHECK_NOTNULL(node);
  std::set<int64_t> *inited_vars = nullptr;
  bool inited_vars_merged = false;

  bool init_var = false;
  int64_t inited_var_id;
  auto ret = CheckAndSetVarInited(node, init_var, inited_var_id);
  if (ret != SUCCESS) {
    return ret;
  }

  if (init_var) {
    inited_vars = CreateInitedVars();
    if (inited_vars == nullptr) {
      return OUT_OF_MEMORY;
    }
    inited_vars_merged = true;
    inited_vars->insert(inited_var_id);
  }

  for (auto &in_node : node->GetInNodes()) {
    GE_CHECK_NOTNULL(in_node->GetOpDesc());
    const auto iter = nodes_to_inited_vars_.find(in_node->GetOpDesc()->GetId());
    if (iter == nodes_to_inited_vars_.end()) {
      continue;
    }
    if (inited_vars == nullptr) {
      inited_vars = iter->second;
      continue;
    }
    if (inited_vars == iter->second) {
      continue;
    }

    // if there are multiple different inited_vars set, we should merge them to a new one
    if (inited_vars_merged) {
      inited_vars->insert(iter->second->cbegin(), iter->second->cend());
    } else {
      auto origin_inited_vars = inited_vars;
      inited_vars = CreateInitedVars();
      if (inited_vars == nullptr) {
        return OUT_OF_MEMORY;
      }
      inited_vars_merged = true;
      inited_vars->insert(origin_inited_vars->cbegin(), origin_inited_vars->cend());
      inited_vars->insert(iter->second->cbegin(), iter->second->cend());
    }
  }

  if (inited_vars != nullptr) {
    GE_CHECK_NOTNULL(node->GetOpDesc());
    nodes_to_inited_vars_[node->GetOpDesc()->GetId()] = inited_vars;
    GELOGD("Inited vars on this graph when node %s, inited vars count %zu",
           node->GetName().c_str(), inited_vars->size());
  }

  return SUCCESS;
}

std::set<int64_t> *VarIsInitializedOpPass::CreateInitedVars() {
  std::unique_ptr<std::set<int64_t>> inited_vars_keeper(new(std::nothrow) std::set<int64_t>());
  if (inited_vars_keeper == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "New set failed");
    GELOGE(OUT_OF_MEMORY, "[New][Set] failed");
    return nullptr;
  }
  auto inited_vars = inited_vars_keeper.get();
  var_inited_keeper_.emplace_back(std::move(inited_vars_keeper));
  return inited_vars;
}

bool VarIsInitializedOpPass::IsVarInitedOnTheGraphAndNode(const NodePtr &node, int64_t var_id) const {
  if (node == nullptr || node->GetOpDesc() == nullptr) {
    return false;
  }
  auto iter = nodes_to_inited_vars_.find(node->GetOpDesc()->GetId());
  if (iter == nodes_to_inited_vars_.end()) {
    return false;
  }
  return iter->second->count(var_id) > 0;
}

Status VarIsInitializedOpPass::CheckAndSetVarInited(const NodePtr &node, bool &inited, int64_t &inited_var) const {
  GE_CHECK_NOTNULL(node);
  inited = false;
  if (node->GetType() != ASSIGN) {
    return SUCCESS;
  }
  auto ref_in_anchor = node->GetInDataAnchor(kAssignVarRefIndex);
  if (ref_in_anchor == nullptr) {
    GELOGW("Invalid assign node on graph, no ref input. name %s", node->GetName().c_str());
    return PARAM_INVALID;
  }
  auto var_out_anchor = ref_in_anchor->GetPeerOutAnchor();
  if (var_out_anchor == nullptr) {
    GELOGW("Invalid assign node on graph, no variable peer. name %s", node->GetName().c_str());
    return PARAM_INVALID;
  }
  auto var = var_out_anchor->GetOwnerNode();
  if (var == nullptr) {
    GELOGW("Invalid assign node on graph, no variable peer. name %s", node->GetName().c_str());
    return PARAM_INVALID;
  }
  inited = true;
  GE_CHECK_NOTNULL(var->GetOpDesc());
  inited_var = var->GetOpDesc()->GetId();
  return SUCCESS;
}

REG_PASS_OPTION("VarIsInitializedOpPass").LEVELS(OoLevel::kO0);
}  // namespace ge
