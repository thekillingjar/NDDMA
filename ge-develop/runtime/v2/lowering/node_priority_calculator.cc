/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "node_priority_calculator.h"
#include <map>
#include <vector>
#include <queue>
#include "common/checker.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/fast_node_utils.h"
#include "exe_graph/lowering/exe_graph_attrs.h"
#include "exe_graph/runtime/continuous_buffer.h"
#include "exe_graph/runtime/compute_node_info.h"
#include "exe_graph/runtime/continuous_vector.h"
#include "core/builder/node_types.h"

namespace gert {
namespace bg {
namespace {
constexpr static int64_t kInitPriority = std::numeric_limits<int64_t>::max();
const int64_t kPriorityExpansion = 10;
const int64_t kPriorityDecrease = 5;
struct NodePriorities {
  explicit NodePriorities(size_t node_num) {
    node_ids_to_priority.resize(node_num, kInitPriority);
    seen_nodes.resize(node_num, false);
  }
  void Set(int64_t node_id, int64_t priority) {
    node_ids_to_priority[node_id] = priority;
    seen_nodes[node_id] = true;
  }
  int64_t Get(int64_t node_id) const {
    return node_ids_to_priority[node_id];
  }
  bool HasSet(int64_t node_id) {
    return seen_nodes[node_id];
  }
  std::vector<int64_t> node_ids_to_priority;
  std::vector<bool> seen_nodes;
};
void PushQueue(ge::FastNode *const node, std::queue<ge::FastNode *> &queue, std::set<ge::FastNode *> &seen) {
  if (seen.insert(node).second) {
    queue.push(node);
  }
}

void PushQueueForAncestors(const ge::FastNode *const priority_node, std::queue<ge::FastNode *> &nodes,
                           std::set<ge::FastNode *> &seen) {
  for (const auto in_data_edge : priority_node->GetAllInDataEdgesRef()) {
    if (in_data_edge != nullptr) {
      PushQueue(in_data_edge->src, nodes, seen);
    }
  }
  for (const auto in_ctrl_edge : priority_node->GetAllInControlEdgesRef()) {
    if (in_ctrl_edge != nullptr) {
      PushQueue(in_ctrl_edge->src, nodes, seen);
    }
  }
}

void MarkAncestorsPriorities(const ge::FastNode *const priority_node, int64_t priority,
                             NodePriorities &node_ids_to_priority) {
  std::queue<ge::FastNode *> nodes;
  std::set<ge::FastNode *> seen;
  PushQueueForAncestors(priority_node, nodes, seen);

  GELOGD("Set priority %" PRId64 " to priority node %s", priority, priority_node->GetNamePtr());
  node_ids_to_priority.Set(priority_node->GetOpDescBarePtr()->GetId(), priority);
  while (!nodes.empty()) {
    auto node = nodes.front();
    nodes.pop();
    const int64_t node_id = node->GetOpDescBarePtr()->GetId();
    if (node_ids_to_priority.HasSet(node_id)) {
      continue;
    }
    GELOGD("Set priority %" PRId64 " to node %s for priority node %s", priority, node->GetNamePtr(),
           priority_node->GetNamePtr());
    node_ids_to_priority.Set(node_id, priority);
    PushQueueForAncestors(node, nodes, seen);
  }
}
void MarkPriorityByPreNode(ge::FastNode *const priority_node, NodePriorities &node_ids_to_priority) {
  std::queue<ge::FastNode *> nodes;
  std::set<ge::FastNode *> seen;
  PushQueueForAncestors(priority_node, nodes, seen);

  int64_t lowest_priority = kInitPriority;
  std::map<int64_t, ge::FastNode *> unset_ancestors_to_node;
  unset_ancestors_to_node[priority_node->GetOpDescBarePtr()->GetId()] = priority_node;
  while (!nodes.empty()) {
    auto node = nodes.front();
    nodes.pop();

    const int64_t node_id = node->GetOpDescBarePtr()->GetId();
    if (node_ids_to_priority.HasSet(node_id)) {
      if (lowest_priority == kInitPriority) {
        lowest_priority = node_ids_to_priority.Get(node_id);
      } else {
        lowest_priority = std::max(node_ids_to_priority.Get(node_id), lowest_priority);
      }
    } else {
      PushQueueForAncestors(node, nodes, seen);
      unset_ancestors_to_node[node_id] = node;
    }
  }

  for (auto &node_id_to_node : unset_ancestors_to_node) {
    int64_t priority = lowest_priority;
    if (IsSendEventsNode(node_id_to_node.second->GetTypePtr())) {
      priority = lowest_priority + 1;
    }
    GELOGD("Set priority %" PRId64 " to node %s for priority node %s", priority, node_id_to_node.second->GetNamePtr(),
           priority_node->GetNamePtr());
    node_ids_to_priority.Set(node_id_to_node.first, priority);
  }
}
void MarkIfSubGraphPriorities(const ge::FastNode *parent_node, NodePriorities &node_ids_to_priority) {
  const auto op_desc = parent_node->GetOpDescBarePtr();
  int64_t priority = node_ids_to_priority.Get(op_desc->GetId());
  int64_t min_priority = priority;
  int64_t max_priority = priority;
  const auto &sub_graph_indexes = op_desc->GetSubgraphNameIndexes();
  std::vector<const ge::FastNode *> before_branch_execute_nodes;
  std::vector<const ge::FastNode *> after_branch_execute_nodes;
  std::vector<ge::FastNode *> other_unset_nodes;
  for (const auto &index: sub_graph_indexes) {
    const auto &graph = ge::FastNodeUtils::GetSubgraphFromNode(parent_node, index.second);
    for (const auto node : graph->GetAllNodes()) {
      const auto &node_type = node->GetTypePtr();
      auto current_priority = node_ids_to_priority.Get(node->GetOpDescBarePtr()->GetId());

      if (IsBranchDone(node_type) || IsWaitAnyone(node_type)) {
        after_branch_execute_nodes.push_back(node);
      } else if (IsSwitchNotifyNode(node_type)) {
        before_branch_execute_nodes.push_back(node);
      } else {
        if (current_priority == kInitPriority) {
          other_unset_nodes.emplace_back(node);
        }
      }

      if (current_priority != kInitPriority) {
        if (current_priority > min_priority) {
          min_priority = current_priority;
        }
        if (current_priority < max_priority) {
          max_priority = current_priority;
        }
      }
    }
  }
  for (const auto node : before_branch_execute_nodes) {
    node_ids_to_priority.Set(node->GetOpDescBarePtr()->GetId(), max_priority);
    GELOGD("control node %s set priority %lld", node->GetNamePtr(), max_priority);
  }
  for (const auto node : after_branch_execute_nodes) {
    node_ids_to_priority.Set(node->GetOpDescBarePtr()->GetId(), min_priority);
    GELOGD("control node %s set priority %lld", node->GetNamePtr(), min_priority);
  }
  for (const auto node : other_unset_nodes) {
    MarkPriorityByPreNode(node, node_ids_to_priority);
  }
}
bool CheckNodeIdValid(ge::FastNode *const node, size_t count) {
  const auto op_desc = node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  GE_ASSERT_TRUE(static_cast<size_t>(op_desc->GetId()) < count, "The node %s topo id %" PRId64 " exceeds max count %zu",
                 op_desc->GetNamePtr(), op_desc->GetId(), count);
  return true;
}
ge::graphStatus SetPriorityToParentNode(ge::FastNode *node, NodePriorities &priorities) {
  std::set<int64_t> unset;
  int64_t priority = kInitPriority;
  while (node != nullptr) {
    const auto op_desc = node->GetOpDescBarePtr();
    GE_ASSERT_NOTNULL(op_desc);
    priority = priorities.Get(op_desc->GetId());
    if (priority != kInitPriority) {
      break;
    }

    // found circle
    GE_ASSERT_TRUE(unset.insert(op_desc->GetId()).second, "Circle found when iter parent nodes for node %s",
                   node->GetNamePtr());
    const auto extend_info = node->GetExtendInfo();
    GE_ASSERT_NOTNULL(extend_info);
    const auto graph = extend_info->GetOwnerGraphBarePtr();
    GE_ASSERT_NOTNULL(graph);
    node = graph->GetParentNodeBarePtr();
  }
  if (priority != kInitPriority) {
    for (const auto &node_id : unset) {
      priorities.Set(node_id, priority);
    }
  }
  return ge::GRAPH_SUCCESS;
}
}  // namespace
NodePriorityCalculator::NodePriorityCalculator(const GraphFrame &frame) : frame_(frame) {}

ge::graphStatus NodePriorityCalculator::CalcNodeExecutionPriorities(const std::vector<ge::FastNode *> &main_graph_nodes,
                                                                    const size_t root_all_nodes_cnt) {
  std::vector<int64_t> index_2_compute_node_id;
  for (const auto &compute_node : frame_.GetIndexesToNode()) {
    const auto op_desc = compute_node->GetOpDescBarePtr();
    GE_ASSERT_NOTNULL(op_desc);
    index_2_compute_node_id.emplace_back(op_desc->GetId());
    GELOGD("Get compute node info, node name: %s, id: %" PRId64 "", compute_node->GetNamePtr(), op_desc->GetId());
  }

  NodePriorities node_ids_to_priority(root_all_nodes_cnt);
  vector<ge::FastNode *> pending_mark_nodes;
  std::vector<ge::FastNode *> if_sub_graph_nodes;
  std::multimap<int64_t, ge::FastNode *> launch_priority_to_launch_node;
  int64_t launch_priority = 0;
  for (const auto node : main_graph_nodes) {
    GE_ASSERT_TRUE(CheckNodeIdValid(node, root_all_nodes_cnt));
    if (IsLaunchOrHasSubGraphNode(node)) {
      // 只收集if case
      if (IsIfOrCaseType(node->GetTypePtr())) {
        if_sub_graph_nodes.push_back(node);
      }
      int64_t compute_node_index = 0;
      // 获取launch节点对应计算节点的topo序，并计算优先级
      if (ge::AttrUtils::GetInt(node->GetOpDescBarePtr(), kComputeNodeIndex, compute_node_index)) {
        GE_ASSERT_TRUE(static_cast<size_t>(compute_node_index) < index_2_compute_node_id.size());
        auto compute_node_id = index_2_compute_node_id[compute_node_index];

        // 将launch的优先级设置为node_id * 10 + 9，即该区间里最低的优先级
        launch_priority = (compute_node_id * kPriorityExpansion) + (kPriorityExpansion - 1);
        if (IsAtomicLaunchNode(node->GetTypePtr())) {
          launch_priority -= kPriorityDecrease;
        }
      }
      (void)launch_priority_to_launch_node.emplace(launch_priority++, node);
    } else if (IsFreeNode(node->GetTypePtr()) || IsSendEventsNode(node->GetTypePtr())) {
      pending_mark_nodes.emplace_back(node);
    } else {
      // do nothing
    }
  }
  for (const auto &node_info : launch_priority_to_launch_node) {
    MarkAncestorsPriorities(node_info.second, node_info.first, node_ids_to_priority);
  }

  for (const auto node : pending_mark_nodes) {
    MarkPriorityByPreNode(node, node_ids_to_priority);
  }

  for (auto node : if_sub_graph_nodes) {
    MarkIfSubGraphPriorities(node, node_ids_to_priority);
  }

  // when a subgraph has no target or free nodes, nodes in the subgraph should have the same priority with parent node
  // e.g. the ctrl-frame subgraph of if/case/while
  for (const auto node : main_graph_nodes) {
    if (node_ids_to_priority.Get(node->GetOpDescBarePtr()->GetId()) == kInitPriority) {
      GE_ASSERT_SUCCESS(SetPriorityToParentNode(node, node_ids_to_priority));
    }
  }

  for (const auto node : main_graph_nodes) {
    const auto op_desc = node->GetOpDescBarePtr();
    GE_ASSERT_NOTNULL(op_desc);
    GE_ASSERT_TRUE(ge::AttrUtils::SetInt(op_desc, "priority", node_ids_to_priority.Get(op_desc->GetId())));
  }

  return ge::GRAPH_SUCCESS;
}
}  // namespace bg
}  // namespace gert
