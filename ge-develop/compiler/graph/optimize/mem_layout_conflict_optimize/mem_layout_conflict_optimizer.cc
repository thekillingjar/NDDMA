/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mem_layout_conflict_optimizer.h"
#include "graph/optimize/graph_optimize.h"
#include "mem_layout_conflict_util.h"
#include "graph/optimize/mem_layout_conflict_optimize/checker/checker.h"
#include "runtime/rt.h"
#include "common/checker.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_type_utils.h"
#include "graph/optimize/mem_layout_conflict_optimize/checker/checker_log.h"
#include "graph/optimize/mem_layout_conflict_optimize/checker/control_node_subgraph_conflict_solve.h"
#include "graph/manager/graph_var_manager.h"

namespace ge {
namespace {
std::map<std::string, AnchorPtr> GetOrderedAnchors(const AnchorSet &anchors) {
  std::map<std::string, AnchorPtr> name_to_anchor;
  for (const auto &anchor : anchors) {
    const auto io_type = anchor->IsTypeIdOf<ge::InDataAnchor>() ? IOType::kIn : IOType::kOut;
    NodeIndexIO node_index_io(anchor->GetOwnerNode(), anchor->GetIdx(), io_type);
    name_to_anchor[node_index_io.ToString()] = anchor;
  }
  return name_to_anchor;
}
}
Status GraphOptimize::HandleMemoryLayoutConflict(ComputeGraphPtr &compute_graph) const {
  MemLayoutConflictOptimizer mem_layout_conflict_optimizer;
  return mem_layout_conflict_optimizer.Run(compute_graph);
}

Status MemLayoutConflictOptimizer::Run(ge::ComputeGraphPtr graph) {
  GE_CHECK_NOTNULL(graph);
  GELOGI("[MemConflict] Start to check memory of root graph %s.", graph->GetName().c_str());

  std::vector<ge::ComputeGraphPtr> top_static_graphs;
  if (MemLayoutConflictUtil::IsStaticGraph(graph)) {
    top_static_graphs.emplace_back(graph);
  } else {
    for (const auto &it : graph->GetAllSubgraphs()) {
      if (MemLayoutConflictUtil::IsStaticGraph(it) && (it->GetParentNode() != nullptr) &&
          !MemLayoutConflictUtil::IsStaticGraph(it->GetParentNode()->GetOwnerComputeGraph())) {
        top_static_graphs.emplace_back(it);
      }
    }
  }
  if (top_static_graphs.empty()) {
    GELOGI("[MemConflict] Graph %s does not have any static subgraph, skip memory check pass.",
           graph->GetName().c_str());
    return SUCCESS;
  }

  // 只有对根图调用拓扑排序，才能遍历到所有的子图。子图上的子图也是挂在根图上的。
  GE_ASSERT_SUCCESS(graph->TopologicalSorting(), "[Call][TopologicalSorting] for graph:%s failed.",
                    graph->GetName().c_str());
  for (auto &static_graph : top_static_graphs) {
    GE_ASSERT_SUCCESS(CtrlNodeConflict::SolveCtrlNodeSubGraphConflict(static_graph),
                      "[Call][SolveCtrlNodeSubGraphConflict] for static_graph:%s failed.",
                      static_graph->GetName().c_str());
  }

  GE_ASSERT_SUCCESS(graph->TopologicalSorting(), "[Call][TopologicalSorting] for graph:%s failed.",
                    graph->GetName().c_str());
  for (auto &static_graph : top_static_graphs) {
    GE_ASSERT_SUCCESS(Process(static_graph), "static_graph: %s", static_graph->GetName().c_str());
  }
  return SUCCESS;
}

Status MemLayoutConflictOptimizer::Process(ge::ComputeGraphPtr &graph) {
  symbol_to_anchors_.clear();
  anchor_to_symbol_.clear();
  checker_->ClearTypes();

  GraphInfo graph_info;
  const auto root_graph = GraphUtils::FindRootGraph(graph);
  graph_info.is_root_graph_static = MemLayoutConflictUtil::IsStaticGraph(root_graph);
  graph_info.is_feature_map_refreshable = MemLayoutConflictUtil::IsGraphFeatureMapRefreshable(graph);
  graph_info.is_support_user_input_nopadding_continuous_output =
      MemLayoutConflictUtil::IsSupportUserInputNopaddingContinuousOutput(graph);
  graph_info.is_physical_memory_refreshable = VarManager::IsGeUseExtendSizeMemoryFull();
  GE_ASSERT_SUCCESS(GraphUtils::GetRefMapping(graph, symbol_to_anchors_, anchor_to_symbol_),
                    "[Call][GetRefMapping] for graph:%s failed.", graph->GetName().c_str());
  GELOGI("[MemConflict] Start to check memory of static graph %s, is_root_graph_static: %d,"
      " is_feature_map_refreshable: %d, support_user_input_nopadding_continuous_output: %d,"
      " is_physical_memory_refreshable: %d",
      graph->GetName().c_str(), graph_info.is_root_graph_static,
      graph_info.is_feature_map_refreshable, graph_info.is_support_user_input_nopadding_continuous_output,
      graph_info.is_physical_memory_refreshable);

  MarkAllAttribute(graph);

  const std::map<std::string, std::list<NodeIndexIO>> ordered_symbol_to_anchors(symbol_to_anchors_.begin(),
                                                                                symbol_to_anchors_.end());
  for (const auto &anchor_iter : ordered_symbol_to_anchors) {
    NodeIndexIOVector all_nodes(anchor_iter.second.cbegin(), anchor_iter.second.cend());
    for (const auto &node_index_io : all_nodes) {
      GE_ASSERT_NOTNULL(MemLayoutConflictUtil::GetAnchorFromIndexIo(node_index_io), "node_index_io, %s",
                        node_index_io.ToString().c_str());
    }

    AnchorSet conflict_set;
    GE_ASSERT_SUCCESS(MemLayoutConflictUtil::FindConflictNodes(all_nodes, conflict_set, graph_info, checker_),
                      "[Call][FindConflictNodes] for graph:%s, symbol: %s, is_root_graph_static: %d, "
                      "is_feature_map_refreshable: %d, is_physical_memory_refreshable: %d",
                      graph->GetName().c_str(), anchor_iter.first.c_str(), graph_info.is_root_graph_static,
                      graph_info.is_feature_map_refreshable, graph_info.is_physical_memory_refreshable);

    GE_ASSERT_SUCCESS(SolveConflict(graph, conflict_set),
                      "[Call][SolveConflict] for graph:%s, symbol: %s, is_root_graph_static: %d,"
                      " is_feature_map_refreshable: %d, is_physical_memory_refreshable: %d",
                      graph->GetName().c_str(), anchor_iter.first.c_str(), graph_info.is_root_graph_static,
                      graph_info.is_feature_map_refreshable, graph_info.is_physical_memory_refreshable);
  }

  return SUCCESS;
}

void MemLayoutConflictOptimizer::MarkAllAttribute(const ComputeGraphPtr &graph) {
  for (const auto &node : graph->GetAllNodes()) {
    MemLayoutConflictUtil::MarkInTypes(node, graph, checker_);
    MemLayoutConflictUtil::MarkOutTypes(node, graph, symbol_to_anchors_, anchor_to_symbol_, checker_);
  }
}

Status MemLayoutConflictOptimizer::SolveConflict(ComputeGraphPtr &graph, AnchorSet &anchors) const {
  if (anchors.empty()) {
    return SUCCESS;
  }
  const auto name_to_anchors = GetOrderedAnchors(anchors);
  for (const auto &iter : name_to_anchors) {
    const ge::AnchorPtr &anchor = iter.second;
    if (anchor->IsTypeIdOf<ge::InDataAnchor>()) {
      auto in_data_anchor = std::static_pointer_cast<InDataAnchor>(anchor);
      if (MemLayoutConflictUtil::IsSkipInsert(in_data_anchor)) {
        continue;
      }
      GE_ASSERT_SUCCESS(SolveInAnchorConflict(graph, in_data_anchor));
    }

    if (anchor->IsTypeIdOf<ge::OutDataAnchor>()) {
      auto out_data_anchor = std::static_pointer_cast<OutDataAnchor>(anchor);
      if (MemLayoutConflictUtil::IsSkipInsert(out_data_anchor)) {
        continue;
      }
      GE_ASSERT_SUCCESS(SolveOutAnchorConflict(graph, out_data_anchor));
    }
  }

  return SUCCESS;
}

Status MemLayoutConflictOptimizer::SolveInAnchorConflict(const ComputeGraphPtr &graph,
                                                         const InDataAnchorPtr &in_anchor) const {
  (void)graph;
  OpDescPtr identity_op = nullptr;
  GE_ASSERT_SUCCESS(MemLayoutConflictUtil::CreateIdentityOpDesc({in_anchor}, identity_op));
  GELOGI("[MemConflict][INSERT][NODE] %s before %s", identity_op->GetNamePtr(),
         in_anchor->GetOwnerNode()->GetNamePtr());
  auto identity_node = GraphUtils::InsertNodeBefore(in_anchor, identity_op);
  GE_ASSERT_NOTNULL(identity_node);
  GE_ASSERT_SUCCESS(MemLayoutConflictUtil::UpdateIsInputConstForNetoutput({in_anchor}, identity_node),
                    "[call][UpdateIsInputConstForNetoutput] failed. identity: %s, dst node: %s %dth input failed.",
                    identity_node->GetName().c_str(), in_anchor->GetOwnerNode()->GetName().c_str(),
                    in_anchor->GetIdx());
  return SUCCESS;
}

Status MemLayoutConflictOptimizer::SolveOutAnchorConflict(const ComputeGraphPtr &graph,
                                                          const OutDataAnchorPtr &out_anchor) const {
  (void)graph;
  const auto peer_in_anchors = out_anchor->GetPeerInDataAnchors();
  if (!peer_in_anchors.empty()) {
    const auto peer_in_anchors_vec = std::vector<ge::InDataAnchorPtr>(peer_in_anchors.begin(), peer_in_anchors.end());
    const uint32_t input_index = 0U;  // use the first input anchor index, because all in anchor need insert
    const uint32_t output_index = out_anchor->GetIdx();

    OpDescPtr insert_op = nullptr;
    GE_ASSERT_SUCCESS(MemLayoutConflictUtil::CreateIdentityOpDesc({peer_in_anchors_vec[input_index]}, insert_op));
    GELOGI("[MemConflict][INSERT][NODE] %s after %s, index: %u", insert_op->GetNamePtr(),
           out_anchor->GetOwnerNode()->GetNamePtr(), output_index);
    auto insert_node = GraphUtils::InsertNodeAfter(out_anchor, peer_in_anchors_vec, insert_op);
    GE_ASSERT_NOTNULL(insert_node);
    GE_ASSERT_SUCCESS(MemLayoutConflictUtil::UpdateIsInputConstForNetoutput(peer_in_anchors_vec, insert_node),
                      "[call][UpdateIsInputConstForNetoutput] failed. identity: %s", insert_node->GetName().c_str());
  }
  return SUCCESS;
}

MemLayoutConflictOptimizer::MemLayoutConflictOptimizer() {
  checker_.reset(new Checker);
}

MemLayoutConflictOptimizer::~MemLayoutConflictOptimizer() = default;
}  // namespace ge
