/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/feature/recompute_pass.h"

#include "ge/ge_api_types.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_type_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_local_context.h"
#include "graph/ge_context.h"
#include "graph/tuning_utils.h"
#include "mmpa/mmpa_api.h"
#include "runtime/mem.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/util.h"
#include "framework/common/ge_types.h"
#include "framework/memory/memory_api.h"
#include "common/checker.h"

namespace {
const std::string kAttrNameRecompute = "_recompute";
const std::string kAttrNameBackward = "_backward";
const std::string kSuffix = "_recompute_copy";
// should be remove when refresh properties pass done.
const std::string kPrefix = "gradients/";
const std::string kGraphFreeMem = "_graph_free_mem";
bool IsNoNeedRecomputedNodes(const std::string &type) {
  return ge::OpTypeUtils::IsDataNode(type) || (type == ge::CONSTANT) || (type == ge::CONSTANTOP) ||
         ge::OpTypeUtils::IsVarLikeNode(type) || (type == ge::NETOUTPUT);
}
}  // namespace

namespace ge {
Status RecomputePass::Run(ComputeGraphPtr graph) {
  GE_CHECK_NOTNULL(graph);
  if (graph->GetParentGraph() != nullptr) {
    GELOGD("subgraph:%s do not do recompute separately", graph->GetName().c_str());
    return SUCCESS;
  }

  GE_CHK_STATUS_RET(graph->TopologicalSorting(), "[Call][TopologicalSorting] failed, graph:%s.",
                    graph->GetName().c_str());

  InheritNodesAttributes(graph);

  std::string build_mode;
  std::string recompute_mode;
  // first get mode from graph
  (void)AttrUtils::GetStr(graph, ATTR_NAME_RECOMPUTE_MODE, recompute_mode);
  (void)GetContext().GetOption(BUILD_MODE, build_mode);
  // second get mode from option(user option is first priority)
  (void)GetContext().GetOption(RECOMPUTE, recompute_mode);
  GELOGD("Build Mode:%s", build_mode.c_str());
  if (!build_mode.empty() && (recompute_mode == "auto")) {
    const auto graph_options = GetThreadLocalContext().GetAllGraphOptions();
    const std::map<std::string, std::string>::const_iterator iter = graph_options.find(TUNING_PATH);
    if (iter == graph_options.cend()) {
      return SUCCESS;
    }
    const std::string &tuning_path = iter->second;
    if (!graph->GetDirectNode().empty()) {
      uint64_t free_graph_size = 0U;
      GE_CHK_STATUS_RET(GetGraphAvailableMemory(graph, free_graph_size), "Get graph available size failed");
      const auto net_output = *(graph->GetDirectNode().end() - 1);  // get netoutput node, set attr
      GE_CHECK_NOTNULL(net_output);
      GELOGI("Set free graph size:%lu to node:%s", free_graph_size, net_output->GetNamePtr());
      (void)AttrUtils::SetInt(net_output->GetOpDescBarePtr(), kGraphFreeMem, free_graph_size);
    }
    GELOGD("In aoe tuning, tuning path:%s", tuning_path.c_str());
    // This is used as the input of the load graph in the aoe process.
    GraphUtils::DumpGEGraph(graph, "", true, tuning_path);
    return SUCCESS;
  }
  // This operation is for debugging and comparison
  GE_DUMP(graph, "BeforeRecompute");

  if (recompute_mode == "auto") {
    GELOGE(FAILED, "The current branch should not be executed, as this feature has been removed from service.");
    return FAILED;
  }

  RecomputeRewriting recompute_rewriting;
  GE_CHK_STATUS_RET(recompute_rewriting.RewriteGraph(graph), "Recompute rewriting failed.");

  GE_DUMP(graph, "AfterRecompute");
  return SUCCESS;
}

// should be remove when refresh properties pass done.
void RecomputePass::InheritNodesAttributes(const ComputeGraphPtr &graph) const {
  const std::string kCreatedReshapePrefix = "Reshape_ReshapeRecoveryPass_";
  const std::string kInsertIndentityLabel = "_Identity";
  for (const NodePtr &node : graph->GetDirectNode()) {
    if ((node->GetName().find(kCreatedReshapePrefix) == std::string::npos) &&
        (node->GetName().find(kInsertIndentityLabel) == std::string::npos)) {
      continue;
    }
    for (const auto &in_node : node->GetInNodes()) {
      if (in_node == nullptr) {
        continue;
      }
      bool is_in_node_backward = false;
      (void)AttrUtils::GetBool(in_node->GetOpDesc(), kAttrNameBackward, is_in_node_backward);
      if (is_in_node_backward) {
        (void)AttrUtils::SetBool(node->GetOpDesc(), kAttrNameBackward, is_in_node_backward);
        break;
      }
    }
  }
}

Status RecomputeRewriting::RewriteGraph(const ComputeGraphPtr &graph) {
  std::vector<NodePtr> recompute_start_nodes;
  std::vector<NodePtr> all_backward_nodes;
  GE_CHK_STATUS_RET(CopyAndFindNodes(graph, recompute_start_nodes, all_backward_nodes),
                    "Find backward nodes and recompute start nodes failed.");

  GE_CHK_STATUS_RET(RecordLastForwardNode(all_backward_nodes), "Not find last forward node.");

  GE_CHK_STATUS_RET(HandleAllRecomputeNodeInputOutput(graph, recompute_start_nodes),
                    "HandleAllRecomputeNodeInputOutput failed");
  return SUCCESS;
}

Status RecomputeRewriting::CopyAndFindNodes(const ComputeGraphPtr &graph, std::vector<NodePtr> &recompute_start_nodes,
                                            std::vector<NodePtr> &all_backward_nodes) {
  for (NodePtr &node : graph->GetDirectNode()) {
    bool is_node_backward = false;
    (void)AttrUtils::GetBool(node->GetOpDesc(), kAttrNameBackward, is_node_backward);
    if (is_node_backward) {
      all_backward_nodes.emplace_back(node);
      continue;
    }
    bool is_recompute = false;
    (void)AttrUtils::GetBool(node->GetOpDesc(), kAttrNameRecompute, is_recompute);
    if (!is_recompute) {
      continue;
    }
    if (!IsNodeNeedRecomputed(node)) {
      continue;
    }
    GE_CHK_STATUS_RET(CopyRecomputeNode(node, graph), "Copy node:%s failed", node->GetName().c_str());
    if (!AnyInNodeIsRecompute(node)) {
      GELOGD("Start recompute node:%s", node->GetName().c_str());
      recompute_start_nodes.emplace_back(node);
    }
  }
  return SUCCESS;
}

bool RecomputeRewriting::AnyInNodeIsRecompute(const NodePtr &node) const {
  for (const auto &in_anchor : node->GetAllInAnchors()) {
    GE_CHECK_NOTNULL(in_anchor);
    for (const auto peer_out_anchor : in_anchor->GetPeerAnchorsPtr()) {
      if (peer_out_anchor == nullptr) {
        continue;
      }
      const auto in_node = peer_out_anchor->GetOwnerNodeBarePtr();
      GE_CHECK_NOTNULL(in_node);
      bool has_in_node_recompute = false;
      (void)AttrUtils::GetBool(in_node->GetOpDesc(), kAttrNameRecompute, has_in_node_recompute);
      if (has_in_node_recompute) {
        return true;
      }
    }
  }
  return false;
}

Status RecomputeRewriting::RecordLastForwardNode(const std::vector<NodePtr> &all_backward_nodes) {
  std::vector<NodePtr> forward_nodes;
  for (const auto &backward_node : all_backward_nodes) {
    for (const auto &in_node : backward_node->GetInNodes()) {
      if (in_node == nullptr) {
        continue;
      }
      bool is_in_node_backward = false;
      (void)AttrUtils::GetBool(in_node->GetOpDesc(), kAttrNameBackward, is_in_node_backward);
      // Apart from is_in_node_backward, others should be remove when refresh properties pass done.
      bool is_needed_forward_node = (!is_in_node_backward) && (in_node->GetName().find(kPrefix) == std::string::npos) &&
                                    (!(in_node->GetInDataNodes().empty()));
      if (is_needed_forward_node) {
        forward_nodes.emplace_back(in_node);
      }
    }
  }
  if (forward_nodes.empty()) {
    return SUCCESS;
  }
  std::sort(forward_nodes.begin(), forward_nodes.end(),
            [](NodePtr node1, NodePtr node2) { return node1->GetOpDesc()->GetId() > node2->GetOpDesc()->GetId(); });
  GELOGD("The last forward node is:%s", forward_nodes.front()->GetName().c_str());
  last_forward_node_ = forward_nodes.front();
  return SUCCESS;
}

bool RecomputeRewriting::IsNodeNeedRecomputed(NodePtr &node) const {
  if (IsNoNeedRecomputedNodes(node->GetType())) {
    GELOGW("Node:%s is %s, cannot do recompute", node->GetName().c_str(), node->GetType().c_str());
    (void)node->GetOpDesc()->DelAttr(kAttrNameRecompute);
    return false;
  }
  for (const auto &input_desc : node->GetOpDesc()->GetAllInputsDescPtr()) {
    if (input_desc->GetDataType() == DT_RESOURCE) {
      GELOGW("Node:%s input data type is DT_RESOURCE, cannot do recompute", node->GetName().c_str());
      (void)node->GetOpDesc()->DelAttr(kAttrNameRecompute);
      return false;
    }
  }
  for (const auto &output_desc : node->GetOpDesc()->GetAllOutputsDescPtr()) {
    if (output_desc->GetDataType() == DT_RESOURCE) {
      GELOGW("Node:%s output data type is DT_RESOURCE, cannot do recompute", node->GetName().c_str());
      (void)node->GetOpDesc()->DelAttr(kAttrNameRecompute);
      return false;
    }
  }
  return true;
}

Status RecomputeRewriting::HandleAllRecomputeNodeInputOutput(const ComputeGraphPtr &graph,
                                                             std::vector<NodePtr> &recompute_start_nodes) {
  NodeInEdges needed_replaced_bp_edges;
  NodeInEdges small_topo_order_bp_edges;
  for (const NodePtr &node : recompute_start_nodes) {
    GELOGD("Begin dfs from start recompute node:%s", node->GetName().c_str());
    NodeInEdges part_needed_replaced_bp_edges;
    NodeInEdges part_small_topo_order_bp_edges;
    bool is_need_replace_bp_edges = true;
    GE_CHK_STATUS_RET(DfsRecomputeNodes(node, part_needed_replaced_bp_edges, part_small_topo_order_bp_edges,
                                        is_need_replace_bp_edges),
                      "DfsRecomputeNodes failed");
    if (!is_need_replace_bp_edges) {
      continue;
    }

    for (const auto &part_needed_replaced_bp_edge : part_needed_replaced_bp_edges) {
      const auto iter = needed_replaced_bp_edges.find(part_needed_replaced_bp_edge.first);
      if (iter != needed_replaced_bp_edges.cend()) {
        iter->second.insert(iter->second.end(), part_needed_replaced_bp_edge.second.cbegin(),
                            part_needed_replaced_bp_edge.second.cend());
      } else {
        needed_replaced_bp_edges.emplace(part_needed_replaced_bp_edge);
      }
    }

    for (const auto &part_small_topo_order_bp_edge : part_small_topo_order_bp_edges) {
      const auto iter = small_topo_order_bp_edges.find(part_small_topo_order_bp_edge.first);
      if (iter != small_topo_order_bp_edges.cend()) {
        iter->second.insert(iter->second.end(), part_small_topo_order_bp_edge.second.cbegin(),
                            part_small_topo_order_bp_edge.second.cend());
      } else {
        small_topo_order_bp_edges.emplace(part_small_topo_order_bp_edge);
      }
    }
  }

  for (auto &item : needed_replaced_bp_edges) {
    const auto node = item.first;
    for (const int32_t in_anchor_index : item.second) {
      GE_CHK_STATUS_RET(ReplaceBackwardNodeEdge(node, in_anchor_index), "Replace backward node edge failed");
    }
  }
  for (auto &item : small_topo_order_bp_edges) {
    const auto node = item.first;
    for (const int32_t in_anchor_index : item.second) {
      GE_CHK_STATUS_RET(ReplaceBackwardNodeEdge(node, in_anchor_index), "Replace backward node edge failed");
    }
  }

  if (graph->TopologicalSorting() != SUCCESS) {
    for (auto &item : small_topo_order_bp_edges) {
      const auto node = item.first;
      for (const int32_t in_anchor_index : item.second) {
        GE_CHK_STATUS_RET(RecoverBackwardNodeEdge(node, in_anchor_index), "Recover backward node edge failed");
      }
    }
  }
  return SUCCESS;
}

Status RecomputeRewriting::RecoverBackwardNodeEdge(const NodePtr &node, const int32_t in_anchor_index) const {
  GELOGD("Recover edge node:%s, in anchor index:%d", node->GetName().c_str(), in_anchor_index);
  const auto in_anchor = node->GetInAnchor(in_anchor_index);
  GE_ASSERT_NOTNULL(in_anchor);
  for (const auto &peer_out_anchor : in_anchor->GetPeerAnchors()) {
    if (peer_out_anchor == nullptr) {
      continue;
    }
    const auto in_node = peer_out_anchor->GetOwnerNode();
    if (in_node->GetName().find(kSuffix) == std::string::npos) {
      continue;
    }
    const auto cur_recompute_node = GetRecomputeNode(in_node->GetName());
    if (cur_recompute_node == nullptr) {
      GELOGE(PARAM_INVALID, "Node:%s is not copy node, no recompute node", in_node->GetName().c_str());
      return FAILED;
    }
    GE_CHK_BOOL_RET_STATUS((GraphUtils::RemoveEdge(peer_out_anchor, in_anchor) == GRAPH_SUCCESS) &&
                               (GraphUtils::AddEdge(cur_recompute_node->GetOutAnchor(peer_out_anchor->GetIdx()),
                                                    in_anchor) == GRAPH_SUCCESS),
                           FAILED, "[Add][Edge] between op:%s(index:%d) and op:%s(index:%d) failed",
                           cur_recompute_node->GetName().c_str(), peer_out_anchor->GetIdx(), node->GetName().c_str(),
                           in_anchor->GetIdx());
  }
  return SUCCESS;
}

void RecomputeRewriting::SaveNeededReplacedBpNodes(const AnchorPtr &peer_out_node_in_anchor,
                                                   std::vector<NodePtr> &backward_nodes,
                                                   NodeInEdges &needed_replaced_bp_edges,
                                                   NodeInEdges &small_topo_order_bp_edges) {
  const auto &out_node = peer_out_node_in_anchor->GetOwnerNode();
  if ((last_forward_node_ != nullptr) && (out_node->GetOpDesc()->GetId() < last_forward_node_->GetOpDesc()->GetId())) {
    GELOGD("node:%s index:%ld, less than last forward node index:%ld, not do recompute", out_node->GetName().c_str(),
           out_node->GetOpDesc()->GetId(), last_forward_node_->GetOpDesc()->GetId());
    const auto iter = small_topo_order_bp_edges.find(out_node);
    if (iter != small_topo_order_bp_edges.cend()) {
      iter->second.emplace_back(peer_out_node_in_anchor->GetIdx());
    } else {
      std::vector<int32_t> anchors_index{peer_out_node_in_anchor->GetIdx()};
      small_topo_order_bp_edges.emplace(out_node, anchors_index);
    }
    return;
  }
  backward_nodes.emplace_back(out_node);
  const auto iter = needed_replaced_bp_edges.find(out_node);
  if (iter != needed_replaced_bp_edges.cend()) {
    iter->second.emplace_back(peer_out_node_in_anchor->GetIdx());
  } else {
    std::vector<int32_t> anchors_index{peer_out_node_in_anchor->GetIdx()};
    needed_replaced_bp_edges.emplace(out_node, anchors_index);
  }
  return;
}

Status RecomputeRewriting::DfsRecomputeNodes(const NodePtr &node, NodeInEdges &needed_replaced_bp_edges,
                                             NodeInEdges &small_topo_order_bp_edges, bool &is_need_replace_bp_edges) {
  std::vector<NodePtr> backward_nodes;
  std::vector<NodePtr> stack;
  std::set<NodePtr> visited;
  stack.emplace_back(node);
  while (!stack.empty()) {
    NodePtr tmp_node = stack.back();
    stack.pop_back();
    if (!visited.insert(tmp_node).second) {
      continue;
    }
    GE_CHK_STATUS_RET(HandleRecomputeNodeInputs(tmp_node), "Handle node:%s inputs failed.",
                      tmp_node->GetName().c_str());
    for (const auto &out_anchor : tmp_node->GetAllOutAnchors()) {
      for (const auto &peer_out_node_in_anchor : out_anchor->GetPeerAnchors()) {
        if (peer_out_node_in_anchor == nullptr) {
          continue;
        }
        const auto &out_node = peer_out_node_in_anchor->GetOwnerNode();
        GE_CHECK_NOTNULL(out_node);
        bool is_out_node_recompute = false;
        (void)AttrUtils::GetBool(out_node->GetOpDesc(), kAttrNameRecompute, is_out_node_recompute);
        if (is_out_node_recompute) {
          stack.emplace_back(out_node);
          continue;
        }
        bool is_out_node_backward = false;
        (void)AttrUtils::GetBool(out_node->GetOpDesc(), kAttrNameBackward, is_out_node_backward);
        if (!is_out_node_backward) {
          continue;
        }
        GELOGD("In the path, find backward node:%s", out_node->GetName().c_str());
        SaveNeededReplacedBpNodes(peer_out_node_in_anchor, backward_nodes, needed_replaced_bp_edges,
                                  small_topo_order_bp_edges);
      }
    }
  }
  if (backward_nodes.empty()) {
    GELOGW("In the path , no backward node");
    is_need_replace_bp_edges = false;
    return SUCCESS;
  }
  std::sort(backward_nodes.begin(), backward_nodes.end(),
            [](NodePtr node1, NodePtr node2) { return node1->GetOpDesc()->GetId() < node2->GetOpDesc()->GetId(); });
  std::vector<NodePtr> backward_node_in_nodes;
  FindAndSortBackwardNodes(backward_nodes.front(), backward_node_in_nodes);
  if (backward_node_in_nodes.empty()) {
    GELOGW("node:%s not include backward node input", backward_nodes.front()->GetName().c_str());
    is_need_replace_bp_edges = false;
    return SUCCESS;
  }
  auto start_copy_node = GetCopyNode(node->GetName());
  GE_CHECK_NOTNULL(start_copy_node);
  GELOGD("Add control edge from node:%s to node:%s", backward_node_in_nodes.front()->GetName().c_str(),
         start_copy_node->GetName().c_str());
  GE_CHK_GRAPH_STATUS_RET(
      GraphUtils::AddEdge(backward_node_in_nodes.front()->GetOutControlAnchor(), start_copy_node->GetInControlAnchor()),
      "[Add][ControlEdge] between %s and %s failed.", backward_node_in_nodes.front()->GetName().c_str(),
      start_copy_node->GetName().c_str());
  return SUCCESS;
}

Status RecomputeRewriting::ReplaceBackwardNodeEdge(const NodePtr &node, const int32_t in_anchor_index) const {
  GELOGD("Replace edge node:%s, in anchor index:%d", node->GetName().c_str(), in_anchor_index);
  auto in_anchor = node->GetInAnchor(in_anchor_index);
  GE_ASSERT_NOTNULL(in_anchor);
  for (const auto &peer_out_anchor : in_anchor->GetPeerAnchors()) {
    if (peer_out_anchor == nullptr) {
      continue;
    }
    auto in_node = peer_out_anchor->GetOwnerNodeBarePtr();
    auto cur_copy_node = GetCopyNode(in_node->GetName());
    if (cur_copy_node == nullptr) {
      GELOGD("Node:%s is not recompute node, no copy node", in_node->GetName().c_str());
      continue;
    }
    GE_CHK_BOOL_RET_STATUS(
        (GraphUtils::RemoveEdge(peer_out_anchor, in_anchor) == GRAPH_SUCCESS) &&
            (GraphUtils::AddEdge(cur_copy_node->GetOutAnchor(peer_out_anchor->GetIdx()), in_anchor) == GRAPH_SUCCESS),
        FAILED, "[Add][Edge] between op:%s(index:%d) and op:%s(index:%d) failed", cur_copy_node->GetName().c_str(),
        peer_out_anchor->GetIdx(), node->GetName().c_str(), in_anchor->GetIdx());
  }
  return SUCCESS;
}

NodePtr RecomputeRewriting::GetCopyNode(const std::string &name) const {
  auto iter = name_to_copy_nodes_.find(name + kSuffix);
  if (iter == name_to_copy_nodes_.end()) {
    return nullptr;
  }
  return iter->second;
}

NodePtr RecomputeRewriting::GetRecomputeNode(const std::string &name) const {
  auto iter = name_to_recompute_nodes_.find(name.substr(0, name.size() - kSuffix.size()));
  if (iter == name_to_recompute_nodes_.end()) {
    return nullptr;
  }
  return iter->second;
}

Status RecomputeRewriting::CopyRecomputeNode(const NodePtr &node, const ComputeGraphPtr &graph) {
  const OpDescPtr op_desc = OpDescUtils::CopyOpDesc(node->GetOpDesc());
  GE_CHECK_NOTNULL(op_desc);

  GE_CHK_GRAPH_STATUS_RET(GraphUtils::CopyTensorAttrs(op_desc, node), "Copy tensor attr from op:%s failed",
                          node->GetName().c_str());

  op_desc->SetName(node->GetName() + kSuffix);
  (void)op_desc->DelAttr(kAttrNameRecompute);
  NodePtr copy_node = graph->InsertNode(node, op_desc);
  GE_CHECK_NOTNULL(copy_node);
  name_to_copy_nodes_.emplace(copy_node->GetName(), copy_node);
  name_to_recompute_nodes_.emplace(node->GetName(), node);
  GELOGD("Success copy recompute node:%s, copy_node:%s", node->GetName().c_str(), copy_node->GetName().c_str());
  return SUCCESS;
}

Status RecomputeRewriting::HandleRecomputeNodeInputs(const NodePtr &node) const {
  auto cur_copy_node = GetCopyNode(node->GetName());
  GE_CHECK_NOTNULL(cur_copy_node);
  GELOGD("Begin handle copy_node:%s inputs", cur_copy_node->GetName().c_str());
  for (const AnchorPtr &in_anchor : node->GetAllInAnchors()) {
    GE_CHECK_NOTNULL(in_anchor);
    for (const auto &peer_out_anchor : in_anchor->GetPeerAnchors()) {
      if (peer_out_anchor == nullptr) {
        continue;
      }
      auto in_node = peer_out_anchor->GetOwnerNodeBarePtr();
      bool is_in_node_recompute = false;
      (void)AttrUtils::GetBool(in_node->GetOpDesc(), kAttrNameRecompute, is_in_node_recompute);
      if (!is_in_node_recompute) {
        if (!peer_out_anchor->IsLinkedWith(cur_copy_node->GetInAnchor(in_anchor->GetIdx()))) {
          GELOGD("Input node is no need recompute, add edge between node:%s(index:%d) and node:%s(index:%d)",
                 peer_out_anchor->GetOwnerNodeBarePtr()->GetNamePtr(), peer_out_anchor->GetIdx(),
                 cur_copy_node->GetName().c_str(), in_anchor->GetIdx());
          GE_CHK_GRAPH_STATUS_RET(GraphUtils::AddEdge(peer_out_anchor, cur_copy_node->GetInAnchor(in_anchor->GetIdx())),
                                  "[Add][Edge] between op:%s(index:%d) and op:%s(index:%d) failed",
                                  peer_out_anchor->GetOwnerNodeBarePtr()->GetNamePtr(), peer_out_anchor->GetIdx(),
                                  cur_copy_node->GetName().c_str(), in_anchor->GetIdx());
        }
      } else {
        auto pre_copy_node = GetCopyNode(in_node->GetName());
        GE_CHECK_NOTNULL(pre_copy_node);
        auto pre_node_out_anchor = pre_copy_node->GetOutAnchor(peer_out_anchor->GetIdx());
        auto cur_node_in_anchor = cur_copy_node->GetInAnchor(in_anchor->GetIdx());
        GE_ASSERT_NOTNULL(pre_node_out_anchor);
        GE_ASSERT_NOTNULL(cur_node_in_anchor);
        if (!pre_node_out_anchor->IsLinkedWith(cur_node_in_anchor)) {
          GELOGD("Input node is recompute, add edge between node:%s(index:%d) and node:%s(index:%d)",
                 pre_copy_node->GetName().c_str(), peer_out_anchor->GetIdx(), cur_copy_node->GetName().c_str(),
                 in_anchor->GetIdx());
          GE_CHK_GRAPH_STATUS_RET(GraphUtils::AddEdge(pre_node_out_anchor, cur_node_in_anchor),
                                  "[Add][Edge] between op:%s(index:%d) and op:%s(index:%d) failed",
                                  pre_copy_node->GetName().c_str(), peer_out_anchor->GetIdx(),
                                  cur_copy_node->GetName().c_str(), in_anchor->GetIdx());
        }
      }
    }
  }
  return SUCCESS;
}

void RecomputeRewriting::FindAndSortBackwardNodes(const NodePtr &node, std::vector<NodePtr> &backward_nodes) const {
  for (const auto &in_node : node->GetInNodes()) {
    if (in_node == nullptr) {
      continue;
    }
    bool is_in_node_backward = false;
    (void)AttrUtils::GetBool(in_node->GetOpDesc(), kAttrNameBackward, is_in_node_backward);
    if (is_in_node_backward) {
      backward_nodes.emplace_back(in_node);
    }
  }
  if (backward_nodes.empty()) {
    return;
  }
  std::sort(backward_nodes.begin(), backward_nodes.end(),
            [](NodePtr node1, NodePtr node2) { return node1->GetOpDesc()->GetId() > node2->GetOpDesc()->GetId(); });
  return;
}

REG_PASS_OPTION("RecomputePass").LEVELS(OoLevel::kO3);
}  // namespace ge
