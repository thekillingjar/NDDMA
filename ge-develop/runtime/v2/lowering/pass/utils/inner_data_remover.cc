/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "inner_data_remover.h"

#include "graph/utils/fast_node_utils.h"
#include "graph/utils/execute_graph_utils.h"
namespace gert {
namespace bg {
SubgraphNodeInnerDataRemover::SubgraphNodeInnerDataRemover(ge::FastNode *const parent_node)
    : parent_fast_node_(parent_node) {}

ge::graphStatus SubgraphNodeInnerDataRemover::RemoveUnusedInnerDataNodes(ge::ExecuteGraph *const subgraph,
                                                                         bool &changed) {
  auto iter = exe_subg_to_indexed_inner_data_nodes_.find(subgraph);
  if (iter == exe_subg_to_indexed_inner_data_nodes_.end()) {
    GELOGE(ge::FAILED, "The subgraph %s does not belongs to node %s, or the `Init` method has not been called",
           subgraph->GetName().c_str(), parent_fast_node_->GetNamePtr());
    return ge::GRAPH_FAILED;
  }
  auto &inner_data_nodes = iter->second;
  for (size_t i = 0U; i < inner_data_nodes.size(); ++i) {
    auto node = inner_data_nodes[i];
    if (node == nullptr) {
      continue;
    }
    if (node->GetAllOutDataEdgesSize() != 0U) {
      continue;
    }
    GE_ASSERT_SUCCESS(ge::ExecuteGraphUtils::IsolateNode(node, {}));
    GE_ASSERT_SUCCESS(ge::ExecuteGraphUtils::RemoveNodeWithoutRelink(subgraph, node));
    inner_data_nodes[i] = nullptr;
    GE_ASSERT_TRUE(i < input_indexes_to_used_count_.size());
    GE_ASSERT_TRUE(input_indexes_to_used_count_[i]-- > 0U);
    changed = true;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SubgraphNodeInnerDataRemover::RemoveUnusedInputs() {
  std::vector<int32_t> old_indexes_to_new(input_indexes_to_used_count_.size(), -1);
  std::vector<size_t> new_input_indexes_to_used_count;
  new_input_indexes_to_used_count.reserve(input_indexes_to_used_count_.size());
  int32_t new_index = 0;

  GE_ASSERT_TRUE(ge::IntegerChecker<int32_t>::Compat(input_indexes_to_used_count_.size()));
  auto owner_graph = parent_fast_node_->GetExtendInfo()->GetOwnerGraphBarePtr();
  GE_ASSERT_NOTNULL(owner_graph);
  for (int32_t old_index = 0U; old_index < static_cast<int32_t>(input_indexes_to_used_count_.size()); ++old_index) {
    /** todo 暂时不支持while的可变输入的移除，移除while的可变输入时，需要同时移除对应的输出、NetOutput，
     *  对应的，可能产生新的Body中剪枝机会。
     *  当前不可移除输入的场景：
     *      1. while的可变输入不可移除
     *      2. 子图中还存在对应InnerData的不可移除
     */
    if (static_cast<size_t>(old_index) < variant_num_ || input_indexes_to_used_count_[old_index] > 0U) {
      if (old_index != new_index) {
        auto old_in_data_edge = parent_fast_node_->GetInDataEdgeByIndex(old_index);
        GE_ASSERT_NOTNULL(old_in_data_edge);
        auto src_node = old_in_data_edge->src;
        auto src_index = old_in_data_edge->src_output;
        GE_ASSERT_GRAPH_SUCCESS(owner_graph->RemoveEdge(old_in_data_edge));
        GE_ASSERT_NOTNULL(owner_graph->AddEdge(src_node, src_index, parent_fast_node_, new_index));
      }
      new_input_indexes_to_used_count.push_back(input_indexes_to_used_count_[old_index]);
      old_indexes_to_new[old_index] = new_index++;
    } else {
      GELOGD("Remove useless input index %d from node %s", old_index, parent_fast_node_->GetNamePtr());
      auto old_in_data_edge = parent_fast_node_->GetInDataEdgeByIndex(old_index);
      GE_ASSERT_GRAPH_SUCCESS(owner_graph->RemoveEdge(old_in_data_edge));
    }
  }

  if (new_index == static_cast<int32_t>(old_indexes_to_new.size())) {
    // not changed
    return ge::GRAPH_SUCCESS;
  }

  GE_ASSERT_SUCCESS(ge::FastNodeUtils::RemoveInputEdgeInfo(parent_fast_node_, new_index));
  input_indexes_to_used_count_ = std::move(new_input_indexes_to_used_count);
  return UpdateInnerDataIndex(old_indexes_to_new);
}

ge::graphStatus SubgraphNodeInnerDataRemover::UpdateInnerDataIndex(const std::vector<int32_t> &old_indexes_to_new) {
  for (auto old_index_iter = static_cast<int32_t>(old_indexes_to_new.size()); old_index_iter > 0; --old_index_iter) {
    auto old_index = old_index_iter - 1;
    auto new_index = old_indexes_to_new[old_index];
    if (old_index == new_index) {
      continue;
    }
    if (new_index == -1) {
      for (auto &subgraph_and_inner_data_nodes : exe_subg_to_indexed_inner_data_nodes_) {
        auto &inner_data_nodes = subgraph_and_inner_data_nodes.second;
        inner_data_nodes.erase(inner_data_nodes.cbegin() + old_index);
      }
    } else {
      for (const auto &subgraph_and_inner_data_nodes : exe_subg_to_indexed_inner_data_nodes_) {
        GE_ASSERT_TRUE(static_cast<size_t>(old_index) < subgraph_and_inner_data_nodes.second.size());
        auto &inner_data = subgraph_and_inner_data_nodes.second[old_index];
        if (inner_data == nullptr) {
          continue;
        }
        GE_ASSERT_TRUE(ge::AttrUtils::SetInt(inner_data->GetOpDescBarePtr(), "index", new_index));
      }
    }
  }

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SubgraphNodeInnerDataRemover::Init() {
  GE_ASSERT_NOTNULL(parent_fast_node_);
  if (IsWhileType(parent_fast_node_->GetTypePtr())) {
    variant_num_ = parent_fast_node_->GetDataOutNum();
  }
  auto input_num = parent_fast_node_->GetDataInNum();
  if (input_num == 0U) {
    return ge::GRAPH_SUCCESS;
  }
  input_indexes_to_used_count_.resize(input_num, 0);

  for (size_t i = 0U; i < parent_fast_node_->GetOpDescBarePtr()->GetSubgraphInstanceNames().size(); ++i) {
    auto subgraph = ge::FastNodeUtils::GetSubgraphFromNode(parent_fast_node_, i);
    GE_ASSERT_NOTNULL(subgraph);
    auto ret = exe_subg_to_indexed_inner_data_nodes_.emplace(subgraph, std::vector<ge::FastNode *>(input_num, nullptr));
    GE_ASSERT_TRUE(ret.second);
    auto &inner_data_nodes = ret.first->second;
    for (const auto node : subgraph->GetDirectNode()) {
      if (!IsTypeInnerData(node->GetTypePtr())) {
        continue;
      }
      int32_t index;
      GE_ASSERT_TRUE(ge::AttrUtils::GetInt(node->GetOpDescBarePtr(), "index", index));
      GE_ASSERT_TRUE(static_cast<size_t>(index) < input_num,
                     "Invalid index %d of InnerData %s in subgraph %s, exceeds the input num %zu of node %s", index,
                     node->GetNamePtr(), subgraph->GetName().c_str(), input_indexes_to_used_count_.size(),
                     parent_fast_node_->GetNamePtr());
      input_indexes_to_used_count_[index]++;
      inner_data_nodes[index] = node;
    }
  }
  return ge::GRAPH_SUCCESS;
}

InnerDataRemoverForSubgraphs::InnerDataRemoverForSubgraphs(std::vector<ge::ExecuteGraph *> graphs)
    : init_exe_subgraphs_(std::move(graphs)) {}

bool InnerDataRemoverForSubgraphs::HasUndoneExeGraphs() const {
  return !exe_graphs_to_be_done_.empty();
}

ge::graphStatus InnerDataRemoverForSubgraphs::AddGraph(ge::ExecuteGraph *const graph) {
  auto parent_node = graph->GetParentNodeBarePtr();
  if (parent_node == nullptr) {
    // the graph is root graph
    return ge::GRAPH_SUCCESS;
  }
  FastNodeToBeDone *ctrl;
  auto iter = cached_fast_nodes_.find(parent_node);
  if (iter == cached_fast_nodes_.end()) {
    auto result =
        cached_fast_nodes_.emplace(parent_node, FastNodeToBeDone{{}, SubgraphNodeInnerDataRemover(parent_node)});
    GE_ASSERT_TRUE(result.second);
    ctrl = &result.first->second;
    GE_ASSERT_SUCCESS(ctrl->inputs_stat.Init());
  } else {
    ctrl = &(iter->second);
  }

  if (exe_graphs_to_be_done_set_.insert(graph).second) {
    exe_subgraphs_to_cached_node_[graph] = ctrl;
    exe_graphs_to_be_done_.push(graph);
    ctrl->unpass_subgraphs.insert(graph);
  }

  return ge::GRAPH_SUCCESS;
}

ge::ExecuteGraph *InnerDataRemoverForSubgraphs::GetNextExeGraph() {
  auto graph = std::move(exe_graphs_to_be_done_.front());
  exe_graphs_to_be_done_.pop();
  exe_graphs_to_be_done_set_.erase(graph);
  return graph;
}

// remove unused inner data nodes in graph, then remove unused inputs from the parent node if all subgraphs of it done
ge::graphStatus InnerDataRemoverForSubgraphs::RemoveAllInnerDataNodes(ge::ExecuteGraph *const graph, bool &changed) {
  GE_ASSERT_TRUE(exe_subgraphs_to_cached_node_.find(graph) != exe_subgraphs_to_cached_node_.end());
  auto node = exe_subgraphs_to_cached_node_.at(graph);
  GE_ASSERT_SUCCESS(node->inputs_stat.RemoveUnusedInnerDataNodes(graph, changed));
  node->unpass_subgraphs.erase(graph);
  if (node->unpass_subgraphs.empty()) {
    GE_ASSERT_SUCCESS(node->inputs_stat.RemoveUnusedInputs());
  }
  return ge::GRAPH_SUCCESS;
}

void InnerDataRemoverForSubgraphs::Clear() {
  init_exe_subgraphs_.clear();
  exe_graphs_to_be_done_set_.clear();
  cached_fast_nodes_.clear();
  exe_subgraphs_to_cached_node_.clear();
}

ge::graphStatus InnerDataRemoverForSubgraphs::RemoveUnusedInnerDataNodes(bool &changed) {
  for (auto graph : init_exe_subgraphs_) {
    GE_ASSERT_SUCCESS(AddGraph(graph));
  }

  while (HasUndoneExeGraphs()) {
    auto graph = GetNextExeGraph();
    bool tmp_changed = false;
    GE_ASSERT_SUCCESS(RemoveAllInnerDataNodes(graph, tmp_changed));

    if (tmp_changed) {
      changed = true;
      const auto parent_node = graph->GetParentNodeBarePtr();
      GE_ASSERT_NOTNULL(parent_node);
      auto parent_graph = parent_node->GetExtendInfo()->GetOwnerGraphBarePtr();
      GE_ASSERT_NOTNULL(parent_graph);
      GE_ASSERT_SUCCESS(AddGraph(parent_graph));
    }
  }

  Clear();

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus InnerDataRemoverForSubgraphs::RemoveUnusedInnerDataNodes() {
  bool changed;
  return RemoveUnusedInnerDataNodes(changed);
}
}  // namespace bg
}  // namespace gert
