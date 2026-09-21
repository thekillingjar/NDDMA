/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_node.h"
#include "graph/utils/execute_graph_utils.h"
#include "graph/utils/fast_node_utils.h"
#include "common/checker.h"
#include "core/utils/executor_utils.h"
#include "utils/utils.h"

namespace gert {
namespace {
bool IsZeroInDegree(const ge::FastNode *const n) {
  const auto op_type = n->GetTypePtr();
  return (IsInnerDataType(op_type) || IsConstType(op_type));
}

bool IsSrcZeroInDegree(const ge::FastEdge *const e) {
  if (e == nullptr) {
    return true;
  }
  return IsZeroInDegree(e->src);
}

// anonymous function to determine weather branch node should be active by branch pivot
bool IsGuardedByPivot(const ge::FastNode *const n) {
  if (IsZeroInDegree(n)) {
    return false;
  }
  const auto &in_data_edges = n->GetAllInDataEdgesRef();
  const auto &in_ctrl_edges = n->GetAllInControlEdgesRef();
  return std::all_of(in_data_edges.begin(), in_data_edges.end(), IsSrcZeroInDegree) &&
         std::all_of(in_ctrl_edges.begin(), in_ctrl_edges.end(), IsSrcZeroInDegree);
}

ge::graphStatus GetCtrlGraphInfo(const ge::FastNode *const node, const ge::FastNode *&start,
                                 const ge::FastNode *&end) {
  if (IsIfOrCaseType(node->GetTypePtr())) {
    const auto cond_graph = ge::FastNodeUtils::GetSubgraphFromNode(node, 0U);
    GE_ASSERT_NOTNULL(cond_graph);
    start = ge::ExecuteGraphUtils::FindFirstNodeMatchType(cond_graph, "SwitchNotify");
    end = ge::ExecuteGraphUtils::FindFirstNodeMatchType(cond_graph, "WaitAnyone");
  } else if (IsWhileType(node->GetTypePtr())) {
    const auto root_graph = ge::ExecuteGraphUtils::FindRootGraph(node->GetExtendInfo()->GetOwnerGraphBarePtr());
    GE_ASSERT_NOTNULL(root_graph);
    const auto op_desc = node->GetOpDescBarePtr();
    GE_ASSERT_NOTNULL(op_desc);
    const auto control_graph = root_graph->GetSubGraph(op_desc->GetSubgraphInstanceName(0U));
    GE_ASSERT_NOTNULL(control_graph);
    start = ge::ExecuteGraphUtils::FindFirstNodeMatchType(control_graph, "Enter");
    end = ge::ExecuteGraphUtils::FindFirstNodeMatchType(control_graph, "Exit");
  } else { // SubGraphCall当前在while控制子图固定结构中，不存在级联场景
    return ge::GRAPH_FAILED;
  }
  GE_ASSERT_NOTNULL(start);
  GE_ASSERT_NOTNULL(end);
  return ge::GRAPH_SUCCESS;
}

bool IsNodeConnected(const ge::FastNode *src, const ge::FastNode *dst) {
  for (const auto &edges : src->GetAllOutDataEdgesRef()) {
    for (const auto &edge : edges) {
      if ((edge != nullptr) && (edge->dst == dst)) {
        return true;
      }
    }
  }

  for (const auto &edge : src->GetAllOutControlEdgesRef()) {
    if ((edge != nullptr) && (edge->dst == dst)) {
      return true;
    }
  }
  return false;
}

std::vector<ge::FastNode *> GetGroupStartNodes(std::unordered_set<ge::FastNode *> &nodes) {
  if (nodes.size() == 1UL) {
    return {*nodes.begin()};
  }
  std::vector<ge::FastNode *> start_nodes;
  for (auto &node : nodes) {
    bool isStart = true;
    for (auto &in_node : node->GetAllInNodes()) {
      if (nodes.count(in_node) > 0UL) {
        isStart = false;
        break;
      }
    }
    if (isStart) {
      start_nodes.push_back(node);
    }
  }
  return start_nodes;
}

std::vector<ge::FastNode *> GetGroupEndNodes(std::unordered_set<ge::FastNode *> &nodes) {
  if (nodes.size() == 1UL) {
    return {*nodes.begin()};
  }
  std::vector<ge::FastNode *> end_nodes;
  for (auto &node : nodes) {
    bool isEnd = true;
    for (auto &out_node : node->GetAllOutNodes()) {
      if (nodes.count(out_node) > 0UL) {
        isEnd = false;
        break;
      }
    }
    if (isEnd) {
      end_nodes.push_back(node);
    }
  }
  return end_nodes;
}
}  // namespace

ge::graphStatus GraphNode::GuardGraphByPivotAndDone(ge::ExecuteGraph *const graph, ge::FastNode *pivot,
                                                    ge::FastNode *done) {
  for (const auto body_node : graph->GetDirectNode()) {
    if (IsGuardedByPivot(body_node)) {
      GE_ASSERT_GRAPH_SUCCESS(AddAdditionalInfo(pivot, body_node));
    }
  }
  const auto net_output = ge::ExecuteGraphUtils::FindFirstNodeMatchType(graph, "InnerNetOutput");
  GE_ASSERT_NOTNULL(net_output);
  for (const auto body_output : net_output->GetAllInNodes()) {
    GE_ASSERT_GRAPH_SUCCESS(AddAdditionalInfo(body_output, done));
  }
  for (const auto node : graph->GetDirectNode()) {
    // 除InnerNetOutput之外的所有叶子节点
    if ((node->GetAllOutNodes().empty()) && (!IsInnerOutput(node->GetTypePtr()))) {
      GELOGD("Add additional info from src node[%s] to dst node[%s]", node->GetNamePtr(), done->GetNamePtr());
      GE_ASSERT_GRAPH_SUCCESS(AddAdditionalInfo(node, done));
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus GraphNode::ReadInIfOrCase(const ge::FastNode *const node) {
  const auto root_graph = ge::ExecuteGraphUtils::FindRootGraph(node->GetExtendInfo()->GetOwnerGraphBarePtr());
  GE_ASSERT_NOTNULL(root_graph, "Root-graph of node %s not found", node->GetNamePtr());
  const auto cond_graph = ge::FastNodeUtils::GetSubgraphFromNode(node, 0U);
  GE_ASSERT_NOTNULL(cond_graph, "Subgraph of %s named %s not found", root_graph->GetName().c_str(), node->GetNamePtr());

  // full connect control edges between if/case and IO nodes
  const auto switch_notify = ge::ExecuteGraphUtils::FindFirstNodeMatchType(cond_graph, "SwitchNotify");
  GE_ASSERT_NOTNULL(switch_notify, "Key node SwitchNotify not found in cond graph of node %s", node->GetNamePtr());
  for (const auto in : node->GetAllInNodes()) {
    GE_ASSERT_GRAPH_SUCCESS(AddAdditionalInfo(in, switch_notify));
  }
  const auto wait_anyone = ge::ExecuteGraphUtils::FindFirstNodeMatchType(cond_graph, "WaitAnyone");
  GE_ASSERT_NOTNULL(wait_anyone, "Key node WaitAnyone not found in cond graph of node %s", node->GetNamePtr());
  for (const auto out : node->GetAllOutNodes()) {
    GE_ASSERT_GRAPH_SUCCESS(AddAdditionalInfo(wait_anyone, out));
  }

  // full connect control edges: BranchPivot --> BranchSubgraphNodes --> BranchDone
  for (const auto cond_node : cond_graph->GetDirectNode()) {
    const auto &op_type = cond_node->GetTypePtr();
    if ((!IsBranchPivot(op_type)) && (!IsBranchDone(op_type))) {
      continue;
    }

    int32_t branch_index = -1;
    GE_ASSERT_TRUE(ge::AttrUtils::GetInt(cond_node->GetOpDescBarePtr(), ge::kRelativeBranch, branch_index),
                   "Failed to get attr 'branch' from node %s", cond_node->GetNamePtr());
    auto branch_graph = ge::FastNodeUtils::GetSubgraphFromNode(node, static_cast<uint32_t>(branch_index));
    GE_ASSERT_NOTNULL(branch_graph, "Failed to find branch graph for node %s, index %d", node->GetNamePtr(),
                      branch_index);

    if (IsBranchDone(op_type)) {
      const auto net_output = ge::ExecuteGraphUtils::FindFirstNodeMatchType(branch_graph, "InnerNetOutput");
      GE_ASSERT_NOTNULL(net_output, "No InnerNetOutput node in subgraph %d from node %s", branch_index,
                        node->GetNamePtr());
      for (const auto in : net_output->GetAllInNodes()) {
        GE_ASSERT_GRAPH_SUCCESS(AddAdditionalInfo(in, cond_node));
      }
      for (const auto tmp_node : branch_graph->GetDirectNode()) {
        // 除InnerNetOutput之外的所有叶子节点
        if (tmp_node->GetAllOutNodes().empty() && (!IsInnerOutput(tmp_node->GetTypePtr()))) {
          GE_ASSERT_GRAPH_SUCCESS(AddAdditionalInfo(tmp_node, cond_node));
        }
      }
    } else {  // BranchPivot
      for (const auto sub_n : branch_graph->GetDirectNode()) {
        if (IsGuardedByPivot(sub_n)) {
          GE_ASSERT_GRAPH_SUCCESS(AddAdditionalInfo(cond_node, sub_n));
        }
      }
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus GraphNode::ReadInWhile(const ge::FastNode *const node) {
  const auto root_graph = ge::ExecuteGraphUtils::FindRootGraph(node->GetExtendInfo()->GetOwnerGraphBarePtr());
  GE_ASSERT_NOTNULL(root_graph);

  const auto op_desc = node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  const auto control_graph = root_graph->GetSubGraph(op_desc->GetSubgraphInstanceName(0U));
  GE_ASSERT_NOTNULL(control_graph);
  const auto enter = ge::ExecuteGraphUtils::FindFirstNodeMatchType(control_graph, "Enter");
  GE_ASSERT_NOTNULL(enter);
  for (const auto in : node->GetAllInNodes()) {
    GE_ASSERT_GRAPH_SUCCESS(AddAdditionalInfo(in, enter));
  }
  const auto exit = ge::ExecuteGraphUtils::FindFirstNodeMatchType(control_graph, "Exit");
  GE_ASSERT_NOTNULL(exit);
  for (const auto out : node->GetAllOutNodes()) {
    GE_ASSERT_GRAPH_SUCCESS(AddAdditionalInfo(exit, out));
  }

  const auto wait_anyone = ge::ExecuteGraphUtils::FindFirstNodeMatchType(control_graph, "WaitAnyone");
  const auto body_pivot = ge::ExecuteGraphUtils::FindFirstNodeMatchType(control_graph, "BranchPivot");
  const auto body_done = ge::ExecuteGraphUtils::FindFirstNodeMatchType(control_graph, "BranchDone");
  GE_ASSERT_NOTNULL(wait_anyone);
  GE_ASSERT_NOTNULL(body_pivot);
  GE_ASSERT_NOTNULL(body_done);
  // Go to next iteration
  GE_ASSERT_GRAPH_SUCCESS(AddAdditionalInfo(body_done, wait_anyone));

  const auto body_graph = root_graph->GetSubGraph(op_desc->GetSubgraphInstanceName(1U));
  GE_ASSERT_NOTNULL(body_graph);

  GE_ASSERT_GRAPH_SUCCESS(GuardGraphByPivotAndDone(body_graph, body_pivot, body_done));

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus GraphNode::ReadInSubgraphCall(const ge::FastNode *const node) {
  const auto root_graph = ge::ExecuteGraphUtils::FindRootGraph(node->GetExtendInfo()->GetOwnerGraphBarePtr());
  GE_ASSERT_NOTNULL(root_graph);

  const auto op_desc = node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  const auto control_graph = root_graph->GetSubGraph(op_desc->GetSubgraphInstanceName(0U));
  GE_ASSERT_NOTNULL(control_graph);
  auto pivot = ge::ExecuteGraphUtils::FindFirstNodeMatchType(control_graph, "BranchPivot");
  GE_ASSERT_NOTNULL(pivot);
  for (const auto in : node->GetAllInNodes()) {
    GE_ASSERT_GRAPH_SUCCESS(AddAdditionalInfo(in, pivot));
  }
  const auto done = ge::ExecuteGraphUtils::FindFirstNodeMatchType(control_graph, "BranchDone");
  GE_ASSERT_NOTNULL(done);
  for (const auto out : node->GetAllOutNodes()) {
    GE_ASSERT_GRAPH_SUCCESS(AddAdditionalInfo(done, out));
  }

  const auto body_graph = root_graph->GetSubGraph(op_desc->GetSubgraphInstanceName(1U));
  GE_ASSERT_NOTNULL(body_graph);

  GE_ASSERT_GRAPH_SUCCESS(GuardGraphByPivotAndDone(body_graph, pivot, done));
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus GraphNode::ReadInNodeHasSubgraph(const ge::FastNode *const node) {
  if (IsIfOrCaseType(node->GetTypePtr())) {
    return ReadInIfOrCase(node);
  } else if (IsWhileType(node->GetTypePtr())) {
    return ReadInWhile(node);
  } else if (IsSubgraphCall(node->GetTypePtr())) {
    return ReadInSubgraphCall(node);
  }
  return ge::GRAPH_FAILED;
}

ge::graphStatus GraphNode::ReadInWatcher(const std::pair<ge::FastNode *, Node *> &node_to_exe_node,
                                         Watcher *&watcher) {
  const auto node = node_to_exe_node.first;
  std::vector<NodeIdentity> watch_nodes;
  NodeIdentity node_id = 0UL;
  for (const auto &out_data_edges : node->GetAllOutDataEdgesRef()) {
    for (const auto out_data_edge : out_data_edges) {
      if ((out_data_edge != nullptr) && IsNodeNeedExec(out_data_edge->dst->GetTypePtr())) {
        GE_ASSERT_GRAPH_SUCCESS(GetExeNodeId(out_data_edge->dst, node_id));
        watch_nodes.push_back(node_id);
      }
    }
  }
  for (const auto out_ctl_edge : node->GetAllOutControlEdgesRef()) {
    if ((out_ctl_edge != nullptr) && IsNodeNeedExec(out_ctl_edge->dst->GetTypePtr())) {
      GE_ASSERT_GRAPH_SUCCESS(GetExeNodeId(out_ctl_edge->dst, node_id));
      watch_nodes.push_back(node_id);
    }
  }
  GE_ASSERT_GRAPH_SUCCESS(UpdateWatcherInfo(node, watch_nodes));
  watcher = CreateWatch(watch_nodes.size(), watch_nodes.data());
  GE_ASSERT_NOTNULL(watcher);
  node_watchers[node_to_exe_node.second->node_id] = watcher;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus GraphNode::AddAdditionalInfo(const ge::FastNode *src_node, const ge::FastNode *dst_node) {
  GELOGD("Add additional info between src node[%s] and dst node[%s]", src_node->GetNamePtr(), dst_node->GetNamePtr());
  return SetAdditionalInfo(src_node, dst_node, true);
}

ge::graphStatus GraphNode::RemoveAdditionalInfo(const ge::FastNode *src_node, const ge::FastNode *dst_node) {
  GELOGD("Remove additional info between src node[%s] and dst node[%s]",
         src_node->GetNamePtr(), dst_node->GetNamePtr());
  return SetAdditionalInfo(src_node, dst_node, false);
}

ge::graphStatus GraphNode::SetAdditionalInfo(const ge::FastNode *src_node,
                                             const ge::FastNode *dst_node, bool isAdd) {
  auto is_no_need_update = [](const char *node_type) {
    return IsGraphInputNode(node_type) || IsGraphOutputNode(node_type) || IsUsrOutputNode(node_type) ||
           IsMemTransferNode(node_type) || IsStroreConstDataNode(node_type);
  };
  if (is_no_need_update(src_node->GetTypePtr()) || is_no_need_update(dst_node->GetTypePtr())) {
    return ge::GRAPH_SUCCESS;
  }

  // 处理控制算子级联的场景
  const ge::FastNode *start = nullptr;
  const ge::FastNode *end = nullptr;
  if (IsHasSubGraphNode(src_node->GetTypePtr())) {
    GE_ASSERT_GRAPH_SUCCESS(GetCtrlGraphInfo(src_node, start, end));
    src_node = end;
  }

  if (IsHasSubGraphNode(dst_node->GetTypePtr())) {
    GE_ASSERT_GRAPH_SUCCESS(GetCtrlGraphInfo(dst_node, start, end));
    dst_node = start;
  }

  if (isAdd) {
    additional_add_info[src_node].emplace_back(dst_node);
    ++additional_indegree_info[dst_node];
  } else {
    additional_del_info[src_node].emplace_back(dst_node);
    --additional_indegree_info[dst_node];
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus GraphNode::ReadInTopoInfo(const std::pair<ge::FastNode *, Node *> &graph_node_to_exe_node,
                                          Watcher *&watcher) {
  ReadInIndegree(graph_node_to_exe_node);
  ReadInStartNode(graph_node_to_exe_node);
  GE_ASSERT_SUCCESS(ReadInWatcher(graph_node_to_exe_node, watcher));
  GE_ASSERT_SUCCESS(AssembleNodeRequestedExtraInfos(graph_node_to_exe_node, *this));
  return ge::GRAPH_SUCCESS;
}

// 多线程执行器要求node执行严格按照计算图topo序，需要在topo序相邻的节点之间增加控制关系
ge::graphStatus GraphNode::EnsureNodeExeInOrderInSubgraph(const ge::ExecuteGraph *sub_exe_graph) {
  std::map<int64_t, std::unordered_set<ge::FastNode *>> priority_to_target_nodes;
  std::map<int64_t, std::unordered_set<ge::FastNode *>> priority_to_use_rt_api_with_addr_nodes;
  for (const auto &node : sub_exe_graph->GetDirectNode()) {
    if (IsLaunchOrHasSubGraphNode(node)) {
      int64_t priority = std::numeric_limits<int64_t>::max();
      (void)ge::AttrUtils::GetInt(node->GetOpDescBarePtr(), "priority", priority);
      (void)priority_to_target_nodes[priority].insert(node);
    }
    auto op_desc = node->GetOpDescBarePtr();
    GE_ASSERT_NOTNULL(op_desc);
    if (ge::AttrUtils::HasAttr(op_desc, "remove_launch_free_edge_alloc")) {
      auto nodes = node->GetOutDataNodes();
      nodes.emplace_back(node);
      for (const auto &target_node : nodes) {
        if (IsCopyAsyncNode(target_node)) {
          int64_t priority = std::numeric_limits<int64_t>::max();
          (void)ge::AttrUtils::GetInt(target_node->GetOpDescBarePtr(), "priority", priority);
          (void)priority_to_use_rt_api_with_addr_nodes[priority].insert(target_node);
        }
      }
    }
  }

  for (auto it = priority_to_target_nodes.begin(); it != priority_to_target_nodes.end(); ++it) {
    auto next_it = std::next(it);
    if (next_it == priority_to_target_nodes.end()) {
      break;
    }
    const auto group_end_nodes = GetGroupEndNodes(it->second);
    std::vector<ge::FastNode *> group_start_nodes = GetGroupStartNodes(next_it->second);
    if (!priority_to_use_rt_api_with_addr_nodes.empty()) {
      auto use_rt_api_with_addr_iter = priority_to_use_rt_api_with_addr_nodes.find(next_it->first);
      if (use_rt_api_with_addr_iter != priority_to_use_rt_api_with_addr_nodes.end()) {
        group_start_nodes = GetGroupStartNodes(use_rt_api_with_addr_iter->second);
      }
    }
    for (auto &end_node : group_end_nodes) {
      for (auto &start_node : group_start_nodes) {
        if (!IsNodeConnected(end_node, start_node)) {
          GE_ASSERT_GRAPH_SUCCESS(AddAdditionalInfo(end_node, start_node));
        }
      }
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus GraphNode::EnsureNodeExeInOrder(ge::ExecuteGraph *exe_graph) {
  const auto root_graph = ge::ExecuteGraphUtils::FindRootGraph(exe_graph);
  GE_ASSERT_NOTNULL(root_graph);
  for (const auto &sub_graph : root_graph->GetAllSubgraphs()) {
    GE_ASSERT_GRAPH_SUCCESS(EnsureNodeExeInOrderInSubgraph(sub_graph));
  }
  return ge::GRAPH_SUCCESS;
}
}  // namespace gert
