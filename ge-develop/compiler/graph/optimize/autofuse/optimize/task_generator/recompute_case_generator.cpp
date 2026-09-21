/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "recompute_case_generator.h"
#include <unordered_set>
#include <queue>
#include "schedule_utils.h"
#include "ascgraph_info_complete.h"
#include "graph/utils/graph_utils.h"

namespace {
constexpr size_t kContinuousOneAxisNum = 2UL;
constexpr size_t kRecomputeNumThreshold = 5UL;
constexpr size_t kRecomputeGapThreshold = 5UL;
const ge::Symbol kContinuousOneAxisSizeThreshold = ge::Symbol(256);

bool IsSupportedComputeType(ge::ComputeType compute_type) {
  static std::unordered_set<ge::ComputeType> supported_types{
      ge::ComputeType::kComputeElewise, ge::ComputeType::kComputeBroadcast, ge::ComputeType::kComputeLoad,
      ge::ComputeType::kComputeStore};
  return supported_types.count(compute_type) > 0UL;
}

ge::Status CopyRecomputeNode(ge::Node *ori_node, ge::AscGraph &graph, ge::AscNodePtr &new_node) {
  const auto &op_desc = ge::GraphUtils::CopyOpDesc(ori_node->GetOpDesc(), nullptr);
  GE_CHECK_NOTNULL(op_desc);
  op_desc->SetName(ori_node->GetName() + "_recompute");
  ge::Operator op = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  new_node = graph.AddNode(op);
  GE_ASSERT_NOTNULL(new_node);
  auto src_asc_node = std::dynamic_pointer_cast<ge::AscNode>(ori_node->shared_from_this());
  GE_ASSERT_NOTNULL(src_asc_node);
  GE_ASSERT_TRUE(ge::AscGraph::CopyAscNodeTensorAttr(src_asc_node, new_node),
                 "DoCopyAscNodeTensorAttr failed, node = %s[%s]", ori_node->GetNamePtr(), ori_node->GetTypePtr());
  return ge::SUCCESS;
}

size_t CountComputeNodes(const std::unordered_set<ge::Node *> &nodes) {
  size_t count = 0UL;
  for (const auto &node : nodes) {
    auto node_attr = node->GetOpDescBarePtr()->GetOrCreateAttrsGroup<ge::AscNodeAttr>();
    if (node_attr != nullptr && node_attr->api.type == ge::ApiType::kAPITypeCompute) {
      ++count;
    }
  }
  return count;
}
}  // namespace

namespace optimize {
Status RecomputeCaseGenerator::GeneratorTask(ascir::HintGraph &hint_graph, std::vector<ScheduleTask> &tasks,
                                             const OptimizerOptions &options) {
  (void)options;                                            
  auto graph_axis = hint_graph.GetAllAxis();
  is_static_graph_ = std::all_of(graph_axis.begin(), graph_axis.end(),
                                 [](const ge::AxisPtr &axis) { return axis->size.IsConstExpr(); });

  std::vector<ge::AscNodePtr> potential_store;
  bool has_unsupported_type{false};
  for (const auto &node : hint_graph.GetAllNodes()) {
    if (ScheduleUtils::IsBuffer(node)) {
      continue;
    }
    if (!IsSupportedComputeType(node->attr.api.compute_type)) {
      has_unsupported_type = true;
      break;
    }
    if (ScheduleUtils::IsStore(node) && IsRecomputableNode(hint_graph, node.get())) {
      GELOGD("Store node [%s] may be recomputable.", node->GetNamePtr());
      potential_store.push_back(node);
    }
  }

  if (has_unsupported_type || potential_store.empty()) {
    return GroupPartitionAndGenTasks(hint_graph, tasks);
  }

  for (const auto &store_node : potential_store) {
    GE_ASSERT_SUCCESS(AnalyzeSplittablePath(hint_graph, store_node));
  }
  MergeAllPaths();

  if (result_.merged_path_nodes.empty()) {
    return GroupPartitionAndGenTasks(hint_graph, tasks);
  }

  GE_ASSERT_SUCCESS(DoTaskGenerator(hint_graph, tasks));
  return ge::SUCCESS;
}

Status RecomputeCaseGenerator::AnalyzeSplittablePath(ascir::HintGraph &hint_graph,
                                                     const ge::AscNodePtr &potential_store) {
  SplitResultInOnePath single_path_results;
  auto out_nodes = potential_store->GetOutDataNodes();
  GE_ASSERT_TRUE(out_nodes.size() == 1UL);
  auto output_node = out_nodes.at(0UL).get();
  single_path_results.valid_path.emplace(output_node);
  single_path_results.output_nodes.emplace(output_node);

  std::set<ge::Node *> visited_nodes;
  visited_nodes.emplace(potential_store.get());
  std::queue<ge::AscNode *> head_nodes;
  head_nodes.emplace(potential_store.get());
  while (!head_nodes.empty()) {
    const auto top_node = head_nodes.front();
    head_nodes.pop();
    GE_ASSERT_NOTNULL(top_node);
    bool current_valid = IsRecomputableNode(hint_graph, top_node);
    if (!current_valid) {
      GELOGD("Store node [%s] cannot be split for node:[%s].", top_node->GetNamePtr(), potential_store->GetNamePtr());
      return ge::SUCCESS;
    }

    GELOGD("Add path node [%s] in path of split node:[%s].", potential_store->GetNamePtr(), top_node->GetNamePtr());
    single_path_results.valid_path.emplace(top_node);
    bool is_all_in_path = true;
    for (const auto &out_node : top_node->GetOutDataNodes()) {
      if (single_path_results.valid_path.count(out_node.get()) == 0UL ||
          single_path_results.recompute_nodes.count(out_node.get()) > 0UL) {
        is_all_in_path = false;
        break;
      }
    }

    if (!is_all_in_path) {
      GELOGD("Add recompute node [%s] in path of split node:[%s].", potential_store->GetNamePtr(),
             top_node->GetNamePtr());
      single_path_results.recompute_nodes.emplace(top_node);
    }
    // add inputs
    for (auto &in_node : top_node->GetInDataNodes()) {
      auto asc_node = std::dynamic_pointer_cast<ge::AscNode>(in_node);
      GE_ASSERT_NOTNULL(asc_node);
      if (visited_nodes.insert(asc_node.get()).second) {
        head_nodes.push(asc_node.get());
      }
    }
  }
  result_.valid_path_results.emplace_back(std::move(single_path_results));

  return ge::SUCCESS;
}

void RecomputeCaseGenerator::MergeAllPaths() {
  // 预留动态评估单个path被拆分的代价
  for (const auto &path : result_.valid_path_results) {
    result_.merged_output_nodes.insert(path.output_nodes.begin(), path.output_nodes.end());
    result_.merged_recompute_nodes.insert(path.recompute_nodes.begin(), path.recompute_nodes.end());
    result_.merged_path_nodes.insert(path.valid_path.begin(), path.valid_path.end());
  }
  auto it = result_.merged_recompute_nodes.begin();
  while (it != result_.merged_recompute_nodes.end()) {
    bool is_all_in_path = true;
    for (const auto &out_node : (*it)->GetOutDataNodes()) {
      if (result_.merged_path_nodes.count(out_node.get()) == 0 ||
          result_.merged_recompute_nodes.count(out_node.get()) > 0UL) {
        is_all_in_path = false;
        break;
      }
    }

    if (is_all_in_path) {
      GELOGD("Node [%s]'s outputs are all in path, no need to do recompute.", (*it)->GetNamePtr());
      it = result_.merged_recompute_nodes.erase(it);
    } else {
      ++it;
    }
  }
  GELOGD("After merging path, [%zu] path nodes, [%zu] recompute_nodes [%zu] output nodes remaining.",
         result_.merged_path_nodes.size(), result_.merged_recompute_nodes.size(), result_.merged_output_nodes.size());
}

Status RecomputeCaseGenerator::DoTaskGenerator(ascir::ImplGraph &impl_graph, std::vector<ScheduleTask> &tasks) const {
  if (!IsRecomputableAlwaysBetter()) {
    ascir::ImplGraph origin_graph(impl_graph.GetName().c_str());
    GE_ASSERT_TRUE(origin_graph.CopyFrom(impl_graph), "Failed to copy graph from [%s].", impl_graph.GetName().c_str());
    GE_ASSERT_SUCCESS(GroupPartitionAndGenTasks(origin_graph, tasks));
  }

  // Gen recompute task
  auto cg = ge::AscGraphUtils::GetComputeGraph(impl_graph);
  GE_ASSERT_NOTNULL(cg);
  cg->SetName(cg->GetName() + "_recompute_case");
  GE_ASSERT_SUCCESS(DoGraphSplit(impl_graph), "Failed to split recompute graph from [%s].",
                    impl_graph.GetName().c_str());
  GE_ASSERT_SUCCESS(GroupPartitionAndGenTasks(impl_graph, tasks));

  return ge::SUCCESS;
}

Status RecomputeCaseGenerator::DoGraphSplit(ge::AscGraph &graph) const {
  // copy recompute_nodes
  std::unordered_map<ge::Node *, ge::Node *> recompute_to_new;
  for (auto recompute_node : result_.merged_recompute_nodes) {
    ge::AscNodePtr new_node;
    GE_ASSERT_SUCCESS(CopyRecomputeNode(recompute_node, graph, new_node), "Failed to copy recompute node [%s].",
                      recompute_node->GetNamePtr());
    recompute_to_new[recompute_node] = new_node.get();
  }

  std::queue<ge::Node *> bfs_queue;
  std::unordered_set<ge::Node *> visited;
  for (auto &out_node : result_.merged_output_nodes) {
    bfs_queue.emplace(out_node);
    visited.emplace(out_node);
  }

  while (!bfs_queue.empty()) {
    ge::Node *current = bfs_queue.front();
    bfs_queue.pop();
    auto cur_iter = recompute_to_new.find(current);
    bool is_cur_recompute = cur_iter != recompute_to_new.end();
    for (const auto &in_anchor : current->GetAllInDataAnchors()) {
      GE_ASSERT_NOTNULL(in_anchor);
      auto peer_out = in_anchor->GetPeerOutAnchor();
      if (peer_out == nullptr) {
        continue;
      }
      auto input_node = peer_out->GetOwnerNodeBarePtr();
      GE_ASSERT_NOTNULL(input_node);
      auto in_iter = recompute_to_new.find(input_node);
      bool is_in_recompute = in_iter != recompute_to_new.end();
      if (is_cur_recompute) {
        GE_ASSERT_TRUE(is_in_recompute);
        GE_ASSERT_SUCCESS(ge::GraphUtils::AddEdge(in_iter->second->GetOutDataAnchor(peer_out->GetIdx()),
                                                  cur_iter->second->GetInDataAnchor(in_anchor->GetIdx())));
      } else {
        if (is_in_recompute) {
          GE_ASSERT_SUCCESS(ge::GraphUtils::RemoveEdge(peer_out, in_anchor));
          GE_ASSERT_SUCCESS(ge::GraphUtils::AddEdge(in_iter->second->GetOutDataAnchor(peer_out->GetIdx()), in_anchor));
        }
      }

      if (visited.count(input_node) == 0UL) {
        visited.insert(input_node);
        bfs_queue.push(input_node);
      }
    }
  }

  return ge::SUCCESS;
}

bool RecomputeCaseGenerator::IsRecomputableNode(ascir::HintGraph &hint_graph, ge::AscNode *node) const {
  if (node->attr.api.type == ge::ApiType::kAPITypeBuffer) {
    return true;
  }
  auto check_static_tensor = [&hint_graph](ge::AscTensorAttr &tensor_attr) -> bool {
    ge::Expression axis_size_product = ge::sym::kSymbolOne;
    const auto &repeats = tensor_attr.repeats;
    const auto &axes = tensor_attr.axis;

    for (size_t i = 0UL; i < repeats.size() && i < axes.size(); ++i) {
      if (ge::SymbolicUtils::StaticCheckNe(repeats[i], ge::sym::kSymbolOne) == ge::TriBool::kTrue) {
        break;
      }
      auto axis = hint_graph.FindAxis(axes[i]);
      if (axis == nullptr || ge::SymbolicUtils::StaticCheckEq(axis->size, ge::sym::kSymbolOne) == ge::TriBool::kTrue) {
        break;
      }
      auto last_threshold = kContinuousOneAxisSizeThreshold / axis->size;
      if (ge::SymbolicUtils::StaticCheckGt(axis_size_product, last_threshold) == ge::TriBool::kTrue) {
        axis_size_product = kContinuousOneAxisSizeThreshold + ge::sym::kSymbolOne;
        break;
      }
      axis_size_product = axis_size_product * axis->size;
    }
    return ge::SymbolicUtils::StaticCheckGt(axis_size_product, kContinuousOneAxisSizeThreshold) == ge::TriBool::kTrue;
  };

  auto check_dynamic_tensor = [](ge::AscTensorAttr &tensor_attr) -> bool {
    size_t continuous_ones_count = 0UL;
    for (const auto &repeat : tensor_attr.repeats) {
      if (ge::SymbolicUtils::StaticCheckEq(repeat, ge::sym::kSymbolOne) == ge::TriBool::kTrue) {
        ++continuous_ones_count;
      } else {
        break;
      }
    }
    return continuous_ones_count > kContinuousOneAxisNum;
  };

  auto output_tensors = node->outputs();
  if (is_static_graph_) {
    return std::all_of(output_tensors.begin(), output_tensors.end(),
                       [&](ge::AscTensor *&tensor) { return check_static_tensor(tensor->attr); });
  }
  return std::all_of(output_tensors.begin(), output_tensors.end(),
                     [&](ge::AscTensor *&tensor) { return check_dynamic_tensor(tensor->attr); });
}

bool RecomputeCaseGenerator::IsRecomputableAlwaysBetter() const {
  if (!is_static_graph_) {
    return false;
  }

  const size_t recompute_valid_size = CountComputeNodes(result_.merged_recompute_nodes);
  if (recompute_valid_size >= kRecomputeNumThreshold) {
    return false;
  }

  const size_t path_valid_size = CountComputeNodes(result_.merged_path_nodes);
  const bool is_second_condition_met = (recompute_valid_size * kRecomputeGapThreshold) < path_valid_size;
  return is_second_condition_met;
}
}  // namespace optimize