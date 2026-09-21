/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_BG_CONDITION_H_
#define AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_BG_CONDITION_H_
#include <functional>
#include <vector>

#include "common/checker.h"
#include "common/debug/ge_log.h"
#include "common/ge_inner_attrs.h"
#include "common/util/mem_utils.h"
#include "core/builder/node_types.h"
#include "exe_graph/lowering/value_holder.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph_builder/value_holder_generator.h"
#include "graph/fast_graph/execute_graph.h"

namespace gert {
namespace bg {
using SubgraphBuilder = std::function<std::vector<ValueHolderPtr>()>;
ge::graphStatus CreateBranchGraphs(
    const ValueHolderPtr &cond_holder, const std::vector<SubgraphBuilder> &builders,
    const std::vector<std::string> &subgraph_names,
    std::unordered_map<std::string, std::vector<ValueHolderPtr>> &subgraph_names_to_outputs);
ge::graphStatus CreateControlFrame(const ValueHolderPtr &cond_holder, size_t branch_num);

template <typename T>
ge::graphStatus UpdatePlacement(
    const std::unordered_map<std::string, std::vector<ValueHolderPtr>> &subgraph_names_to_outputs,
    const std::vector<std::shared_ptr<T>> &outputs) {
  auto output_count = outputs.size();
  std::vector<std::set<int32_t>> indexes_to_placements(output_count);
  for (const auto &subgraph_name_and_outputs : subgraph_names_to_outputs) {
    size_t i = 0U;
    for (const auto &output : subgraph_name_and_outputs.second) {
      GE_ASSERT_TRUE(i < indexes_to_placements.size());
      auto placement = output->GetPlacement();
      if (placement == kFollowing) {
        placement = kOnHost;
      }
      indexes_to_placements[i++].insert(placement);
    }
  }
  for (size_t i = 0U; i < indexes_to_placements.size(); ++i) {
    if (indexes_to_placements[i].size() > 1U) {
      outputs[i]->SetPlacement(kTensorPlacementEnd);
    } else {
      outputs[i]->SetPlacement(*(indexes_to_placements[i].begin()));
    }
  }
  return ge::GRAPH_SUCCESS;
}

template <typename T>
ge::graphStatus AddGuardersToParentNode(const std::set<std::pair<int32_t, std::string>> &all_guarder_indexes_types,
                                        const std::vector<std::shared_ptr<T>> &outputs) {
  for (const auto &guarder_index_type : all_guarder_indexes_types) {
    ValueHolder::CreateVoidGuarder(guarder_index_type.second.c_str(), outputs[guarder_index_type.first], {});
  }
  return ge::GRAPH_SUCCESS;
}
struct PlusRefGuarderInfo {
  ge::FastNode *output_node;
  int32_t output_index;
  ge::EdgeDstEndpoint netoutput_in;
};
struct OutputsGuarderInfo {
  std::vector<ge::FastNode *> to_be_removed_guarders;
  std::vector<PlusRefGuarderInfo> plus_refs;
};

ge::graphStatus AddPlusCount(const std::string &name, const std::vector<PlusRefGuarderInfo> &plus_refs);
ge::graphStatus AddPointFromForConflicts(const std::string &name, const std::set<int32_t> &conflict_indexes,
                                         const std::vector<ge::FastNode *> &netoutput_nodes);
ge::graphStatus CalcSubgraphGuardersPolicy(const ge::FastNode * const parent_node,
                                           std::set<std::pair<int32_t, std::string>> &all_guarder_indexes_types,
                                           std::vector<OutputsGuarderInfo> &subgraph_indexes_to_policy);
ge::graphStatus CalcChainConflictSolvePolicy(const ge::FastNode * const node,
                                             std::vector<ge::FastNode *> &subgraph_indexes_to_netoutput,
                                             std::set<int32_t> &conflict_indexes, std::set<int32_t> &all_same_indexes);
ge::graphStatus RemoveGuarders(const std::vector<ge::FastNode *> &remove_guarders);
ge::graphStatus SortAllSubgraphs(const ValueHolderPtr &holder);
std::vector<ValueHolderPtr> EmptySubgraphBuilder();

template <typename T>
ge::graphStatus EnsureResourceLifeCycleExtended(const std::vector<std::shared_ptr<T>> &outputs) {
  auto node = outputs[0]->GetFastNode();
  std::vector<OutputsGuarderInfo> subgraph_indexes_to_policy;
  std::set<std::pair<int32_t, std::string>> all_guarder_indexes_types;
  GE_ASSERT_SUCCESS(CalcSubgraphGuardersPolicy(node, all_guarder_indexes_types, subgraph_indexes_to_policy));
  if (all_guarder_indexes_types.empty()) {
    return ge::GRAPH_SUCCESS;
  }
  GE_ASSERT_SUCCESS(AddGuardersToParentNode<T>(all_guarder_indexes_types, outputs));
  for (size_t i = 0U; i < subgraph_indexes_to_policy.size(); ++i) {
    if (!subgraph_indexes_to_policy[i].to_be_removed_guarders.empty()) {
      GE_ASSERT_SUCCESS(RemoveGuarders(subgraph_indexes_to_policy[i].to_be_removed_guarders));
    }
    if (!subgraph_indexes_to_policy[i].plus_refs.empty()) {
      GE_ASSERT_SUCCESS(AddPlusCount(node->GetName() + "_Subgraph_" + std::to_string(i) + "_AddResRef_",
                                     subgraph_indexes_to_policy[i].plus_refs));
    }
  }

  return ge::GRAPH_SUCCESS;
}

template <typename T>
ge::graphStatus EnsureChainConflictSolved(const std::vector<std::shared_ptr<T>> &outputs) {
  auto node = outputs[0]->GetFastNode();
  const auto op_desc = node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  const auto &subgraph_names = op_desc->GetSubgraphInstanceNames();
  GE_ASSERT_TRUE(!subgraph_names.empty());
  const size_t subgraph_num = subgraph_names.size() - 1U;
  std::vector<ge::FastNode *> subgraph_indexes_to_netoutput(subgraph_num);
  std::set<int32_t> conflict_indexes, all_same_indexes;
  GE_ASSERT_SUCCESS(
      CalcChainConflictSolvePolicy(node, subgraph_indexes_to_netoutput, conflict_indexes, all_same_indexes));
  if (!conflict_indexes.empty()) {
    GE_ASSERT_SUCCESS(
        AddPointFromForConflicts(node->GetName() + "_PointFrom", conflict_indexes, subgraph_indexes_to_netoutput));
  }
  // todo: by pass if/case node for `all_same_indexes`

  return ge::GRAPH_SUCCESS;
}

// A If/Case node has one CondGraph and some branch graphs
//                         +--------------+
//                         |  Cond Node   +-----------------------------------------+
//                         +------+-------+                                         |
//                                |                                                 |
//                                v                                                 |
// +-----------------------------------------------------------------------------+  |   +----------------------+
// |                            CondGraph                                        |  |   |                      |
// |                                                                             |  |   |                      |
// |                            +----------------+                               |  +-> |     one or more      |
// |                            | InnerData(0)   |                               |      |     branch graphs    |
// |                            +----------------+                               |      |                      |
// |                                    |                                        |      +----------------------+
// |                                    v                                        |
// |                            +----------------+                               |
// |   +----------------------- |  SwitchNotify  | ------+                       |
// |   |                        +----------------+       |                       |
// |   |                          |                      |                       |
// |   v                          v                      v                       |
// | +--------------------+     +----------------+     +----------------+        |
// | | WatcherPlaceholder |     | BranchPivot(0) |     | BranchPivot(n) |        |
// | +--------------------+     +----------------+     +----------------+        |
// |                              :                      :                       |
// |                              v                      v                       |
// |                            +----------------+     +----------------+        |
// |                            | BranchDone(0)  |     | BranchDone(n)  |        |
// |                            +----------------+     +----------------+        |
// |                              :                      :                       |
// |                              v                      :                       |
// |                            +----------------+       :                       |
// |                            |   WaitAnyone   | <......                       |
// |                            +----------------+                               |
// +-----------------------------------------------------------------------------+
template <typename T, typename... Args>
std::vector<std::shared_ptr<T>> Cond(const std::vector<bg::ValueHolderPtr> &index, const char *node_type,
                                     const std::vector<std::string> &subgraph_names,
                                     const std::vector<SubgraphBuilder> &builders, Args... args) {
  GE_ASSERT_TRUE(!subgraph_names.empty(), "Failed to create %s node, no subgraph inputs", node_type);
  auto cond_holder = ValueHolder::CreateVoid<T>(node_type, index, args...);
  if (cond_holder == nullptr) {
    return {};
  }

  GE_ASSERT_SUCCESS(CreateControlFrame(cond_holder, subgraph_names.size()));
  std::unordered_map<std::string, std::vector<bg::ValueHolderPtr>> subgraph_names_to_outputs;
  GE_ASSERT_SUCCESS(CreateBranchGraphs(cond_holder, builders, subgraph_names, subgraph_names_to_outputs));

  auto out_count = subgraph_names_to_outputs.begin()->second.size();
  std::vector<std::shared_ptr<T>> outputs;
  if (out_count == 0UL) {
    outputs.emplace_back(cond_holder);
  } else {
    ValueHolderPtr tmp_holder = cond_holder;
    for (const auto &holder : tmp_holder->AppendOutputs<T>(out_count, args...)) {
      outputs.emplace_back(holder);
    }
    GE_ASSERT_TRUE(!outputs.empty() && (outputs[0] != nullptr));
    GE_ASSERT_SUCCESS(EnsureResourceLifeCycleExtended<T>(outputs));
    GE_ASSERT_SUCCESS(EnsureChainConflictSolved<T>(outputs));
    GE_ASSERT_SUCCESS(UpdatePlacement<T>(subgraph_names_to_outputs, outputs));
  }

  GE_ASSERT_SUCCESS(SortAllSubgraphs(outputs[0]));

  return outputs;
}

// A If node has three subgraphs: CondGraph, Then Graph, Else Graph
//         +--------------+
//         |  IF Node     +-----------------------+---------------------------+
//         +------+-------+                       |                           |
//                |                               |                           |
//                v                               v                           v
//    +----------------------+      +----------------------+     +----------------------+
//    |                      |      |                      |     |                      |
//    |                      |      |                      |     |                      |
//    |     CondGraph        |      |     Then Graph       |     |     Else Graph       |
//    |                      |      |                      |     |                      |
//    |                      |      |                      |     |                      |
//    +----------------------+      +----------------------+     +----------------------+

template <typename T, typename... Args>
std::vector<std::shared_ptr<T>> If(const ValueHolderPtr &cond, SubgraphBuilder then_branch, SubgraphBuilder else_branch,
                                   Args... args) {
  if (cond == nullptr) {
    GELOGE(ge::PARAM_INVALID, "Failed to create if node, the input cond is nullptr");
    return {};
  }

  if (then_branch == nullptr && else_branch == nullptr) {
    GELOGE(ge::PARAM_INVALID, "Failed to create if node, then and else branch builders are nullptr");
    return {};
  }

  std::vector<SubgraphBuilder> builders;
  if (then_branch != nullptr) {
    builders.emplace_back(std::move(then_branch));
  } else {
    builders.emplace_back(EmptySubgraphBuilder);
  }
  if (else_branch != nullptr) {
    builders.emplace_back(std::move(else_branch));
  } else {
    builders.emplace_back(EmptySubgraphBuilder);
  }

  return Cond<T>({cond}, "If", {ge::kThenGraph, ge::kElseGraph}, builders, args...);
}

// A Case node has n + 1 subgraphs: CondGraph, <branch-n-graphs>
//      +--------------+
//      |  Case Node   +--------------------+-------------------------+--------------------------+
//      +------+-------+                    |                         |                          |
//             |                            |                         |                          |
//             v                            v                         v                          v
// +----------------------+   +----------------------+   +----------------------+   +----------------------+
// |                      |   |                      |   |                      |   |                      |
// |                      |   |                      |   |                      |   |                      |
// |     CondGraph        |   |   Branch 0 Graph     |   |   Branch 1 Graph     |   |   Branch .. Graphs   |
// |                      |   |                      |   |                      |   |                      |
// |                      |   |                      |   |                      |   |                      |
// +----------------------+   +----------------------+   +----------------------+   +----------------------+
template <typename T, typename... Args>
std::vector<std::shared_ptr<T>> Case(const ValueHolderPtr &index, const std::vector<SubgraphBuilder> &builders,
                                     Args... args) {
  if (index == nullptr) {
    GELOGE(ge::PARAM_INVALID, "Failed to create Case node, the input index is nullptr");
    return {};
  }
  std::vector<std::string> subgraph_names;
  for (size_t i = 0UL; i < builders.size(); ++i) {
    if (builders[i] == nullptr) {
      GELOGE(ge::PARAM_INVALID, "Failed to create Case node, the %zu th builder is nullptr", i);
      return {};
    }
    std::stringstream ss;
    ss << ge::kRelativeBranch << '_' << i;
    subgraph_names.emplace_back(ss.str());
  }

  return Cond<T>({index}, "Case", subgraph_names, builders, args...);
}
}  // namespace bg
}  // namespace gert
#endif  // AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_BG_CONDITION_H_
