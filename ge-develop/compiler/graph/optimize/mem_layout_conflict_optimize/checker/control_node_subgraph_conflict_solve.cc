/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "control_node_subgraph_conflict_solve.h"
#include "graph/optimize/mem_layout_conflict_optimize/mem_layout_conflict_util.h"
#include "common/checker.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_type_utils.h"

namespace ge {

constexpr uint8_t kThenBranchIndex = 0U;
constexpr uint8_t kElseBranchIndex = 1U;
constexpr uint8_t kBodyBranchIndex = 1U;

using SubGraphSolveConflictCall = std::function<Status(const NodePtr &ctrl_node)>;
std::map<std::string, SubGraphSolveConflictCall> CtrlNodeConflict::get_subgraph_solve_call = {
  {IF, CtrlNodeConflict::SolveIfConflict},
  {STATELESSIF, CtrlNodeConflict::SolveIfConflict},
  {CASE, CtrlNodeConflict::SolveCaseConflict},
  {STATELESSCASE, CtrlNodeConflict::SolveCaseConflict},
  {WHILE, CtrlNodeConflict::SolveWhileConflict},
  {STATELESSWHILE, CtrlNodeConflict::SolveWhileConflict}
};

Status CtrlNodeConflict::SolveIfConflict(const NodePtr &ctrl_node) {
  const auto then_graph = NodeUtils::GetSubgraph(*ctrl_node, kThenBranchIndex);
  const auto else_graph = NodeUtils::GetSubgraph(*ctrl_node, kElseBranchIndex);
  GE_CHECK_NOTNULL(then_graph, "node: %s", ctrl_node->GetNamePtr());
  GE_CHECK_NOTNULL(else_graph, "node: %s", ctrl_node->GetNamePtr());

  GE_ASSERT_SUCCESS(SolveOneSubGraphConflict(then_graph),
                    "[Call][SolveOneCtrlNodeConflict] for if node:%s then graph failed.", ctrl_node->GetName().c_str());
  GE_ASSERT_SUCCESS(SolveOneSubGraphConflict(else_graph),
                    "[Call][SolveOneCtrlNodeConflict] for if node:%s else graph failed.", ctrl_node->GetName().c_str());
  return SUCCESS;
}

Status CtrlNodeConflict::SolveCaseConflict(const NodePtr &ctrl_node) {
  const size_t num_subgraphs = ctrl_node->GetOpDesc()->GetSubgraphInstanceNames().size();
  for (size_t i = 0U; i < num_subgraphs; ++i) {
    const auto sub_graph = NodeUtils::GetSubgraph(*ctrl_node, i);
    GE_CHECK_NOTNULL(sub_graph, "node: %s", ctrl_node->GetNamePtr());
    GE_ASSERT_SUCCESS(SolveOneSubGraphConflict(sub_graph),
                      "[Call][SolveOneCtrlNodeConflict] for case node:%s graph idx %d failed.",
                      ctrl_node->GetName().c_str(), i);
  }
  return SUCCESS;
}

Status CtrlNodeConflict::SolveNodesConflict(const ComputeGraphPtr &graph,
                                            const std::vector<InDataAnchorPtr> &in_data_anchors) {
  OpDescPtr identity_op = nullptr;
  GE_ASSERT_SUCCESS(MemLayoutConflictUtil::CreateIdentityOpDesc(in_data_anchors, identity_op));
  auto append_before_anchor = in_data_anchors.back();
  GE_ASSERT_NOTNULL(append_before_anchor);
  auto append_before_node = append_before_anchor->GetOwnerNode();
  auto identity_node = graph->InsertNode(append_before_node, identity_op);
  GE_ASSERT_NOTNULL(identity_node);
  size_t index = 0U;
  for (const auto &in_data_anchor : in_data_anchors) {
    GE_ASSERT_NOTNULL(in_data_anchor);
    GE_ASSERT_NOTNULL(in_data_anchor->GetPeerOutAnchor());
    GE_ASSERT_NOTNULL(in_data_anchor->GetOwnerNode());
    GE_ASSERT_NOTNULL(in_data_anchor->GetPeerOutAnchor()->GetOwnerNode());

    GELOGI("[MemConflict][INSERT][NODE] %s between %s->%s", identity_node->GetNamePtr(),
           in_data_anchor->GetPeerOutAnchor()->GetOwnerNode()->GetNamePtr(),
           in_data_anchor->GetOwnerNode()->GetNamePtr());
    GE_ASSERT_SUCCESS(GraphUtils::InsertNodeBefore(in_data_anchor, identity_node, index, index));
    ++index;
  }
  GE_ASSERT_SUCCESS(MemLayoutConflictUtil::UpdateIsInputConstForNetoutput(in_data_anchors, identity_node),
                    "[call][Update] failed. identity_node: %s.", identity_node->GetName().c_str());
  return SUCCESS;
}

Status CtrlNodeConflict::SolveOneSubGraphConflict(const ComputeGraphPtr &sub_graph) {
  std::vector<InDataAnchorPtr> in_data_anchors;
  GE_ASSERT_SUCCESS(MemLayoutConflictUtil::CheckOneSubGraphConflict(sub_graph, in_data_anchors),
                    "[Call][CheckOneSubGraphConflict] faild.");
  if (!in_data_anchors.empty()) {
    GE_ASSERT_SUCCESS(SolveNodesConflict(sub_graph, in_data_anchors), "[Call][SolveOneNodeConflict] faild.");
    GELOGD("[MemConflict][Conflict] Insert identity node at input of subgraph[%s].", sub_graph->GetName().c_str());
    return SUCCESS;
  }
  return SUCCESS;
}

Status CtrlNodeConflict::InsertInputIdentity(const ComputeGraphPtr &graph, const std::vector<NodePtr> &data_nodes) {
  if (data_nodes.empty()) {
    return SUCCESS;
  }

  std::vector<InDataAnchorPtr> in_data_anchors;
  for (const auto &data_node : data_nodes) {
    GE_CHECK_NOTNULL(data_node);
    // Data node has and only has one output
    OutDataAnchorPtr out_data_anchor = data_node->GetOutDataAnchor(0);
    GE_CHECK_NOTNULL(out_data_anchor);
    InDataAnchorPtr in_data_anchor = out_data_anchor->GetPeerInDataAnchors().at(0U);
    GE_CHECK_NOTNULL(in_data_anchor);
    in_data_anchors.emplace_back(in_data_anchor);
  }

  GE_ASSERT_SUCCESS(SolveNodesConflict(graph, in_data_anchors), "[Call][SolveOneNodeConflict] faild.");
  GELOGI("[MemConflict][Conflict] Data to netoutput with exchange order, inserted identity node"
         " at input of while_body [%s].", graph->GetName().c_str());
  return SUCCESS;
}

Status CtrlNodeConflict::InsertOutputIdentity(const ComputeGraphPtr &graph, const NodePtr &output_node,
                                              const std::set<uint32_t> &bypass_index) {
  std::vector<InDataAnchorPtr> in_data_anchors;
  for (size_t i = 0U; i < output_node->GetAllInDataAnchorsSize(); i++) {
    if (bypass_index.count(i) != 0U) {
      continue;
    }

    InDataAnchorPtr in_data_anchor = output_node->GetInDataAnchor(i);
    GE_CHECK_NOTNULL(in_data_anchor);
    if (MemLayoutConflictUtil::IsSkipInsert(in_data_anchor)) {
      continue;
    }
    
    in_data_anchors.emplace_back(in_data_anchor);
  }

  if (in_data_anchors.empty()) {
    GELOGD("[MemConflict] No need to insert output identity node in while_body %s, output_size equal to bypass_num.",
           graph->GetName().c_str());
    return SUCCESS;
  }

  OpDescPtr identity_op = nullptr;
  GE_ASSERT_SUCCESS(MemLayoutConflictUtil::CreateIdentityOpDesc(in_data_anchors, identity_op));
  auto append_before_anchor = in_data_anchors.back();
  GE_ASSERT_NOTNULL(append_before_anchor);
  auto append_before_node = append_before_anchor->GetOwnerNode();
  GE_ASSERT_NOTNULL(append_before_node);
  auto identity_node = graph->InsertNode(append_before_node, identity_op);
  GE_ASSERT_NOTNULL(identity_node);
  size_t index = 0U;
  for (const auto &in_data_anchor : in_data_anchors) {
    GELOGI("[MemConflict][INSERT][NODE] %s between %s->%s, index: %zu", identity_node->GetNamePtr(),
           in_data_anchor->GetPeerOutAnchor()->GetOwnerNode()->GetNamePtr(),
           in_data_anchor->GetOwnerNode()->GetNamePtr(), index);
    GE_ASSERT_SUCCESS(GraphUtils::InsertNodeBefore(in_data_anchor, identity_node, index, index));
    ++index;
  }

  GELOGI("[MemConflict][Conflict] To ensure execute order to avoid overwriting input,"
         " inserted identity node[%s] at output of while_body[%s].",
         identity_node->GetNamePtr(), graph->GetName().c_str());
  return SUCCESS;
}

Status CtrlNodeConflict::SolveWhileConflict(const NodePtr &ctrl_node) {
  const auto while_body = NodeUtils::GetSubgraph(*ctrl_node, kBodyBranchIndex);
  GE_CHECK_NOTNULL(while_body, "node: %s", ctrl_node->GetNamePtr());

  std::vector<NodePtr> data_nodes_change;
  std::set<uint32_t> bypass_index_no_change;
  GE_ASSERT_SUCCESS(MemLayoutConflictUtil::GetWhileBodyDataToNetoutputNodes(
                      while_body, data_nodes_change, bypass_index_no_change),
                      "[Call][GetWhileBodyDataToNetoutputNodes] faild.");

  const auto netoutput = while_body->GetOrUpdateNetOutputNode();
  GE_CHECK_NOTNULL(netoutput);

  bool is_need_insert = true;
  GE_ASSERT_SUCCESS(MemLayoutConflictUtil::IsWhileNeedInsertIdentityAtOutput(while_body, is_need_insert));
  if (is_need_insert) {
    GE_ASSERT_SUCCESS(InsertOutputIdentity(while_body, netoutput, bypass_index_no_change),
                      "[Call][InsertOutputIdentity] faild.");
  }

  // 先插输出的identity，后插输入的identity。插输出的时候会判断前一个如果是identity就不插了
  GE_ASSERT_SUCCESS(InsertInputIdentity(while_body, data_nodes_change), "[Call][InsertInputIdentity] faild.");

  return SUCCESS;
}

Status CtrlNodeConflict::SolveCtrlNodeSubGraphConflict(const ComputeGraphPtr &graph) {
  GELOGI("[MemConflict] Start to check memory for control node. static graph[%s].", graph->GetName().c_str());
  for (const auto &node : graph->GetAllNodes()) {
    const auto node_type = node->GetType();
    if (get_subgraph_solve_call.find(node_type) != get_subgraph_solve_call.end()) {
      GE_ASSERT_SUCCESS(get_subgraph_solve_call[node_type](node),
                        "[Call]subgraph_solve_call faild, node name: %s, node type: %s.",
                        node->GetName().c_str(), node->GetType().c_str());
    }
  }

  return SUCCESS;
}
} // namespace ge
