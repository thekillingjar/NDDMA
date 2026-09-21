/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bg_condition.h"

#include "common/checker.h"
#include "common/debug/ge_log.h"
#include "common/ge_inner_attrs.h"
#include "common/util/mem_utils.h"
#include "core/builder/node_types.h"
#include "exe_graph/lowering/exe_graph_attrs.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/fast_graph/execute_graph.h"
#include "exe_graph/lowering/value_holder_utils.h"
#include "graph/utils/execute_graph_utils.h"

namespace gert {
namespace bg {
namespace {
struct GuarderInfo {
  bool has_guarder;
  ge::FastNode *in_graph_guarder;
  std::string guarder_node_type;
};

ValueHolderPtr CreateInnerData(int32_t index) {
  auto data = ValueHolder::CreateSingleDataOutput("InnerData", {});
  GE_ASSERT_NOTNULL(data);
  auto desc = ValueHolderUtils::GetNodeOpDescBarePtr(data);
  GE_ASSERT_TRUE(ge::AttrUtils::SetInt(desc, "index", index));
  return data;
}

std::string GetGuarderOutside(const ge::FastNode *node) {
  if (!IsInnerDataType(node->GetTypePtr())) {
    return "";
  }
  auto outside_guarder_type = ge::AttrUtils::GetStr(node->GetOpDescBarePtr(), kNodeWithGuarderOutside);
  return outside_guarder_type != nullptr ? *outside_guarder_type : "";
}

GuarderInfo FindGuarder(const ge::EdgeSrcEndpoint &edge_src_point) {
  // 如果与InnerNetoutput直连的节点是子图输入，且子图输入的父节点对应输入节点带有guarder，需要考虑延长生命周期：
  // InnerData与InnerNetoutput之间插入IdentityAddr，以增加引用计数，子图外部创建对应Guarder节点。
  auto src_node = edge_src_point.node;
  const auto guarder_type = GetGuarderOutside(src_node);
  if (!guarder_type.empty()) {
    if (!IsFreeNode(guarder_type.c_str())) {
      GELOGE(ge::PARAM_INVALID, "get guarder type of node %s[%s] failed, guarder type = %s",
             src_node->GetNamePtr(), src_node->GetTypePtr(),
             guarder_type.c_str());
      return {false, nullptr, ""};
    }
    return {true, nullptr, guarder_type};
  }

  for (const auto &guarder_edge : src_node->GetOutEdgesRefByIndex(edge_src_point.index)) {
    if (guarder_edge == nullptr) {
      continue;
    }
    auto guarder_node = guarder_edge->dst;
    int32_t guarder_index = 0;
    if (!ge::AttrUtils::GetInt(guarder_node->GetOpDescBarePtr(), kReleaseResourceIndex, guarder_index)) {
      continue;
    }
    if (guarder_index == guarder_edge->dst_input) {
      return {true, guarder_node, guarder_node->GetType()};
    }
  }
  return {false, nullptr, ""};
}
}  // namespace

std::vector<ValueHolderPtr> EmptySubgraphBuilder() { return {}; }

ge::graphStatus CreateBranchGraphs(
    const ValueHolderPtr &cond_holder, const std::vector<SubgraphBuilder> &builders,
    const std::vector<std::string> &subgraph_names,
    std::unordered_map<std::string, std::vector<ValueHolderPtr>> &subgraph_names_to_outputs) {
  GE_ASSERT_EQ(builders.size(), subgraph_names.size());
  size_t output_count = 0UL;
  for (size_t i = 0UL; i < builders.size(); ++i) {
    GE_ASSERT_NOTNULL(ValueHolder::PushGraphFrame(cond_holder, subgraph_names[i].c_str()));
    auto outputs = builders[i]();

    if (i == 0UL) {
      output_count = outputs.size();
    } else {
      if (output_count != outputs.size()) {
        GELOGE(ge::PARAM_INVALID,
               "Failed to create branch graphs, %zu th(%s) outputs num are different with first(%s) %zu/%zu", i,
               subgraph_names[i].c_str(), subgraph_names[0U].c_str(), outputs.size(), output_count);
        return ge::GRAPH_PARAM_INVALID;
      }
    }

    auto frame = ValueHolder::PopGraphFrame(outputs, {});
    if (frame == nullptr) {
      GELOGE(ge::PARAM_INVALID, "Failed to construct %s graph", subgraph_names[i].c_str());
      return ge::PARAM_INVALID;
    }
    subgraph_names_to_outputs[subgraph_names[i]] = std::move(outputs);
  }
  return ge::GRAPH_SUCCESS;
}

/*
 *
 *                       WaitAnyone(Only One)
 *                          :
 *                      BranchDone(0..subgraph_names)
 *                          :
 * WatcherPlaceholder   BranchPivot(0..subgraph_names)
 *                 \       |
 *                  SwitchNotify
 *                      |
 *             InnerData(switch-index)
 */
ge::graphStatus CreateControlFrame(const ValueHolderPtr &cond_holder, size_t branch_num) {
  GE_ASSERT_NOTNULL(ValueHolder::PushGraphFrame(cond_holder, ge::kConditionGraph));
  auto cond_data = CreateInnerData(0);
  auto outputs = ValueHolder::CreateDataOutput("SwitchNotify", {cond_data}, branch_num + 1U);
  GE_ASSERT_NOTNULL(outputs[0U]);
  auto output_desc = ValueHolderUtils::GetNodeOpDescBarePtr(outputs[0U]);
  GE_ASSERT_TRUE(ge::AttrUtils::SetBool(output_desc, ge::kRequestWatcher, true));

  GE_ASSERT_NOTNULL(ValueHolder::CreateVoid<bg::ValueHolder>("WatcherPlaceholder", {outputs[0U]}));
  auto wait_anyone = ValueHolder::CreateVoid<bg::ValueHolder>("WaitAnyone", {});

  for (size_t i = 0U; i < branch_num; ++i) {
    auto pivot = ValueHolder::CreateVoid<bg::ValueHolder>("BranchPivot", {outputs[i + 1U]});
    auto done = ValueHolder::CreateVoid<bg::ValueHolder>("BranchDone", {});

    GE_ASSERT_NOTNULL(pivot);
    GE_ASSERT_NOTNULL(done);

    auto pivot_desc = ValueHolderUtils::GetNodeOpDescBarePtr(pivot);
    GE_ASSERT_TRUE(ge::AttrUtils::SetInt(pivot_desc, ge::kRelativeBranch, i + 1U));
    auto done_desc = ValueHolderUtils::GetNodeOpDescBarePtr(done);
    GE_ASSERT_TRUE(ge::AttrUtils::SetInt(done_desc, ge::kRelativeBranch, i + 1U));

    GE_ASSERT_HYPER_SUCCESS(ValueHolder::AddDependency(pivot, done));
    GE_ASSERT_HYPER_SUCCESS(ValueHolder::AddDependency(done, wait_anyone));
  }

  ValueHolder::PopGraphFrame();
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus CalcSubgraphGuardersPolicy(const ge::FastNode * const parent_node,
                                           std::set<std::pair<int32_t, std::string>> &all_guarder_indexes_types,
                                           std::vector<OutputsGuarderInfo> &subgraph_indexes_to_policy) {
  const auto op_desc = parent_node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  const auto &subgraph_name = op_desc->GetSubgraphInstanceNames();
  GE_ASSERT_TRUE(!subgraph_name.empty());
  const size_t subgraph_num = subgraph_name.size() - 1U;
  subgraph_indexes_to_policy.resize(subgraph_num);

  std::vector<std::set<int32_t>> subgraph_indexes_to_guarder_indexes(subgraph_num);
  std::vector<std::set<ge::EdgeSrcEndpoint>> subgraph_indexes_to_guarded_endpoint(subgraph_num);
  std::vector<std::map<int32_t, PlusRefGuarderInfo>> subgraph_indexes_to_parent_indexes_to_no_guarder(subgraph_num);

  for (size_t i = 0U; i < subgraph_num; ++i) {
    auto subgraph = ge::FastNodeUtils::GetSubgraphFromNode(parent_node, i + 1U);
    GE_ASSERT_NOTNULL(subgraph);
    auto netoutput_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(subgraph, "InnerNetOutput");
    GE_ASSERT_NOTNULL(netoutput_node);
    for (const auto edge : netoutput_node->GetAllInDataEdgesRef()) {
      if (edge == nullptr) {
        continue;
      }
      GE_ASSERT_NOTNULL(edge->src);
      GE_ASSERT_NOTNULL(edge->dst);
      ge::EdgeSrcEndpoint src_endpoint{edge->src, edge->src_output};
      const GuarderInfo guarder_info = FindGuarder(src_endpoint);
      if (guarder_info.has_guarder) {
        all_guarder_indexes_types.insert({edge->dst_input, guarder_info.guarder_node_type});
        subgraph_indexes_to_guarder_indexes[i].insert(edge->dst_input);
        if (guarder_info.in_graph_guarder != nullptr) {
          // 如果未插入成功，说明遇到了同一个guarded输出连接到多个子图外的场景，那么从第二个开始加加引用计数，并在子图外部新增guarder
          if (!subgraph_indexes_to_guarded_endpoint[i].insert(src_endpoint).second) {
            subgraph_indexes_to_policy[i].plus_refs.push_back(
                  {edge->src, edge->src_output, ge::EdgeDstEndpoint{edge->dst, edge->dst_input}});
          }
          subgraph_indexes_to_policy[i].to_be_removed_guarders.push_back(guarder_info.in_graph_guarder);
        } else {
          // 如果子图外guarder通过子图的输入透传到子图的输出，那么在父图上来看，就多了一个需要管理生命周期的tensor，
          // 此时一种正确的做法是为新增的这个tensor也加上guarder。因此此处为父节点对应输入增加guarder，在子图内部增加引用计数。
          // 因此，这也是这里加加计数，外部加guarder的来历。
          subgraph_indexes_to_policy[i].plus_refs.push_back(
                {edge->src, edge->src_output, ge::EdgeDstEndpoint{edge->dst, edge->dst_input}});
        }
      } else {
        subgraph_indexes_to_parent_indexes_to_no_guarder[i][edge->dst_input] = {edge->src, edge->src_output,
                                                                                {edge->dst, edge->dst_input}};
      }
    }
  }

  for (auto guarder_index_type : all_guarder_indexes_types) {
    for (size_t i = 0U; i < subgraph_num; ++i) {
      if (subgraph_indexes_to_guarder_indexes[i].count(guarder_index_type.first) > 0) {
        continue;
      }
      subgraph_indexes_to_policy[i].plus_refs.emplace_back(
          std::move(subgraph_indexes_to_parent_indexes_to_no_guarder[i][guarder_index_type.first]));
    }
  }

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddGuardersToParentNode(const std::set<std::pair<int32_t, std::string>> &all_guarder_indexes_types,
                                        const std::vector<ValueHolderPtr> &outputs) {
  for (const auto &guarder_index_type : all_guarder_indexes_types) {
    ValueHolder::CreateVoidGuarder(guarder_index_type.second.c_str(), outputs[guarder_index_type.first], {});
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus RemoveGuarders(const std::vector<ge::FastNode *> &remove_guarders) {
  for (const auto &guarder : remove_guarders) {
    ge::FastNodeUtils::UnlinkAll(guarder);
    ge::ExecuteGraphUtils::RemoveNodeWithoutRelink(guarder->GetExtendInfo()->GetOwnerGraphBarePtr(), guarder);
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddPlusCount(const std::string &name, const std::vector<PlusRefGuarderInfo> &plus_refs) {
  auto op_desc = ge::MakeShared<ge::OpDesc>(name, "IdentityAddr");
  GE_ASSERT_NOTNULL(op_desc);
  for (size_t i = 0U; i < plus_refs.size(); ++i) {
    GE_ASSERT_SUCCESS(op_desc->AddInputDesc(ge::GeTensorDesc()));
    GE_ASSERT_SUCCESS(op_desc->AddOutputDesc(ge::GeTensorDesc()));
  }
  const auto node = plus_refs[0].output_node->GetExtendInfo()->GetOwnerGraphBarePtr()->AddNode(op_desc);
  GE_ASSERT_NOTNULL(node);
  for (size_t i = 0U; i < plus_refs.size(); ++i) {
    ge::ExecuteGraphUtils::InsertNodeAfter({plus_refs[i].output_node, plus_refs[i].output_index},
                                           {plus_refs[i].netoutput_in}, node, i, i);
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus CalcChainConflictSolvePolicy(const ge::FastNode * const node,
                                             std::vector<ge::FastNode *> &subgraph_indexes_to_netoutput,
                                             std::set<int32_t> &conflict_indexes, std::set<int32_t> &all_same_indexes) {
  const auto op_desc = node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  const auto &subgraph_name = op_desc->GetSubgraphInstanceNames();
  GE_ASSERT_TRUE(!subgraph_name.empty());
  const size_t subgraph_num = subgraph_name.size() - 1U;
  std::vector<std::set<int32_t>> output_indexes_to_input_indexes(node->GetDataOutNum());
  subgraph_indexes_to_netoutput.resize(subgraph_num);
  for (size_t subgraph_index = 0U; subgraph_index < subgraph_num; ++subgraph_index) {
    auto subgraph = ge::FastNodeUtils::GetSubgraphFromNode(node, subgraph_index + 1U);
    GE_ASSERT_NOTNULL(subgraph);
    auto netoutput = ge::ExecuteGraphUtils::FindFirstNodeMatchType(subgraph, "InnerNetOutput");
    GE_ASSERT_NOTNULL(netoutput);
    subgraph_indexes_to_netoutput[subgraph_index] = netoutput;
    size_t output_index = 0U;
    for (const auto &in_node : netoutput->GetInDataNodes()) {
      GE_ASSERT_TRUE(output_index < output_indexes_to_input_indexes.size());
      if (IsInnerDataType(in_node->GetTypePtr())) {
        int32_t data_index;
        GE_ASSERT_TRUE(ge::AttrUtils::GetInt(in_node->GetOpDescBarePtr(), kFeedIndex, data_index));
        output_indexes_to_input_indexes[output_index++].insert(data_index);
      } else {
        output_indexes_to_input_indexes[output_index++].insert(-1);
      }
    }
  }
  for (size_t i = 0U; i < output_indexes_to_input_indexes.size(); ++i) {
    if (output_indexes_to_input_indexes[i].size() > 1U) {
      conflict_indexes.insert(static_cast<int32_t>(i));
      continue;
    }
    if (*output_indexes_to_input_indexes[i].begin() != -1) {
      all_same_indexes.insert(static_cast<int32_t>(i));
      continue;
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddPointFromForConflicts(const std::string &name, const std::set<int32_t> &conflict_indexes,
                                         const std::vector<ge::FastNode *> &netoutput_nodes) {
  size_t subgraph_index = 0U;
  for (const auto &netoutput_node : netoutput_nodes) {
    auto op_desc = ge::MakeShared<ge::OpDesc>(name + "_" + std::to_string(subgraph_index++), "PointFromInputs");
    GE_ASSERT_NOTNULL(op_desc);
    for (size_t i = 0U; i < conflict_indexes.size(); ++i) {
      GE_ASSERT_SUCCESS(op_desc->AddInputDesc(ge::GeTensorDesc()));
      GE_ASSERT_SUCCESS(op_desc->AddOutputDesc(ge::GeTensorDesc()));
    }
    const auto node = netoutput_node->GetExtendInfo()->GetOwnerGraphBarePtr()->AddNode(op_desc);
    GE_ASSERT_NOTNULL(node);
    uint32_t pfi_index = 0U;
    for (auto conflict_index : conflict_indexes) {
      GE_ASSERT_SUCCESS(ge::ExecuteGraphUtils::InsertNodeBefore({netoutput_node, conflict_index}, node,
                                                                pfi_index, pfi_index));
      ++pfi_index;
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SortAllSubgraphs(const ValueHolderPtr &holder) {
  auto node = holder->GetFastNode();
  for (size_t i = 1U; i < node->GetOpDescBarePtr()->GetSubgraphInstanceNames().size(); ++i) {
    const auto subgraph = ge::FastNodeUtils::GetSubgraphFromNode(node, i);
    GE_ASSERT_NOTNULL(subgraph);
    GE_ASSERT_SUCCESS(subgraph->TopologicalSorting(), "Failed to topo sort on %zu graph , node %s type %s", i,
                      node->GetNamePtr(), node->GetTypePtr());
  }
  return ge::GRAPH_SUCCESS;
}
}  // namespace bg
}  // namespace gert
