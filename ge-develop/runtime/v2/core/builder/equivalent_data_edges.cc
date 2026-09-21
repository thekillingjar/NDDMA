/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "equivalent_data_edges.h"
#include "common/checker.h"
#include "exe_graph/lowering/exe_graph_attrs.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/execute_graph_utils.h"
#include "graph/utils/fast_node_utils.h"
#include "core/builder/node_types.h"

namespace gert {
namespace {
constexpr uint64_t kInputEnd = 0U;
constexpr uint64_t kOutputEnd = 1U;
}  // namespace

ge::graphStatus EquivalentDataEdges::UpdateEquivalentEdges(ge::ExecuteGraph *const exe_graph) {
  // exe_graph需要保证已完成topo排序，此时GetAllNodes获取到的节点的顺序和topo序一致，数组下标即为node id
  const auto &all_nodes = exe_graph->GetAllNodes();
  GE_ASSERT_GRAPH_SUCCESS(ConstructIoEquivalent(all_nodes));
  GE_ASSERT_GRAPH_SUCCESS(ConstructInnerDataOutputEquivalent(all_nodes));
  GE_ASSERT_GRAPH_SUCCESS(ConstructRefOutputEquivalent(exe_graph, all_nodes));
  GE_ASSERT_GRAPH_SUCCESS(SetInputOutputSymbols(all_nodes));
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus EquivalentDataEdges::ConstructIoEquivalent(const std::vector<ge::FastNode *> &all_nodes) {
  std::vector<bool> node_id_states;
  node_id_states.resize(all_nodes.size(), false);
  for (const auto node : all_nodes) {
    GE_ASSERT_NOTNULL(node->GetOpDescBarePtr());
    const auto node_id = static_cast<size_t>(node->GetOpDescBarePtr()->GetId());
    GE_ASSERT_TRUE((node_id < node_id_states.size() && !node_id_states[node_id]),
                   "the id[%ld] of node[%s] must be unique and in range less than %zu", node_id, node->GetNamePtr(),
                   node_id_states.size());
    node_id_states[node_id] = true;

    for (size_t index = 0U; index < node->GetDataOutNum(); ++index) {
      const auto out_endpoint =
          Encode({static_cast<uint64_t>(node_id), static_cast<uint64_t>(index), kOutputEnd});
      Add(out_endpoint);

      for (const auto edge : node->GetOutEdgesRefByIndex(static_cast<int32_t>(index))) {
        if (edge == nullptr) {
          continue;
        }
        Merge(out_endpoint, Encode({static_cast<uint64_t>(edge->dst->GetOpDescBarePtr()->GetId()),
                                    static_cast<uint64_t>(edge->dst_input), kInputEnd}));
      }
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus EquivalentDataEdges::ConstructInnerDataOutputEquivalent(const std::vector<ge::FastNode *> &all_nodes) {
  for (const auto node : all_nodes) {
    if (IsInnerDataType(node->GetTypePtr())) {
      GE_ASSERT_GRAPH_SUCCESS(ConstructInnerDataEquivalent(node));
    }
    if (IsInnerOutput(node->GetTypePtr())) {
      GE_ASSERT_GRAPH_SUCCESS(ConstructInnerOutputEquivalent(node));
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus EquivalentDataEdges::ConstructInnerDataEquivalent(const ge::FastNode *const node) {
  int32_t index;
  GE_ASSERT_TRUE(ge::AttrUtils::GetInt(node->GetOpDescBarePtr(), ge::ATTR_NAME_INDEX, index));
  GE_ASSERT_NOTNULL(node->GetExtendInfo());
  GE_ASSERT_NOTNULL(node->GetExtendInfo()->GetOwnerGraphBarePtr());
  const auto parent_node = node->GetExtendInfo()->GetOwnerGraphBarePtr()->GetParentNodeBarePtr();
  GE_ASSERT_NOTNULL(parent_node);
  const auto parent_node_in_endpoint = Encode(
      {static_cast<uint64_t>(parent_node->GetOpDescBarePtr()->GetId()), static_cast<uint64_t>(index), kInputEnd});
  const auto data_out_endpoint = Encode({static_cast<uint64_t>(node->GetOpDescBarePtr()->GetId()), 0UL, kOutputEnd});
  GELOGD("Start Merge node[%s][idx=%d] and node[%s][idx=%d]", node->GetNamePtr(), 0, parent_node->GetNamePtr(), index);
  Merge(parent_node_in_endpoint, data_out_endpoint);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus EquivalentDataEdges::ConstructInnerOutputEquivalent(const ge::FastNode *const node) {
  GE_ASSERT_NOTNULL(node->GetExtendInfo());
  GE_ASSERT_NOTNULL(node->GetExtendInfo()->GetOwnerGraphBarePtr());
  const auto parent_node = node->GetExtendInfo()->GetOwnerGraphBarePtr()->GetParentNodeBarePtr();
  GE_ASSERT_NOTNULL(parent_node);
  const auto inner_netoutput_node_id = static_cast<uint64_t>(node->GetOpDescBarePtr()->GetId());
  const auto parent_node_id = static_cast<uint64_t>(parent_node->GetOpDescBarePtr()->GetId());
  for (const auto output_in_edge : node->GetAllInDataEdgesRef()) {
    if (output_in_edge != nullptr) {
      const auto inner_netoutput_in_endpoint =
          Encode({inner_netoutput_node_id, static_cast<uint64_t>(output_in_edge->dst_input), kInputEnd});
      const auto parent_out_endpoint =
          Encode({parent_node_id, static_cast<uint64_t>(output_in_edge->dst_input), kOutputEnd});
      GELOGD("Start Merge node[%s][idx=%d] and node[%s][idx=%d]", node->GetNamePtr(), output_in_edge->dst_input,
             parent_node->GetNamePtr(), output_in_edge->dst_input);
      Merge(inner_netoutput_in_endpoint, parent_out_endpoint);
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus EquivalentDataEdges::ConstructRefOutputEquivalent(ge::ExecuteGraph *const exe_graph,
                                                                  const std::vector<ge::FastNode *> &all_nodes) {
  const auto &node_names_to_node = ge::ExecuteGraphUtils::GetNodeMapFromAllNodes(exe_graph);
  for (const auto node : all_nodes) {
    const auto op_desc = node->GetOpDescBarePtr();
    for (size_t i = 0U; i < node->GetDataOutNum(); ++i) {
      const std::string *src_node_name =
          ge::AttrUtils::GetStr(op_desc->GetOutputDesc(static_cast<uint32_t>(i)), kRefFromNode);
      if (src_node_name == nullptr) {
        continue;
      }
      int32_t src_index = 0;
      GE_ASSERT_TRUE(ge::AttrUtils::GetInt(op_desc->GetOutputDesc(i), kRefFromIndex, src_index));

      const auto iter = node_names_to_node.find(*src_node_name);
      GE_ASSERT_TRUE(iter != node_names_to_node.end());
      const auto src_ref_endpoint = Encode({static_cast<uint64_t>(iter->second->GetOpDescBarePtr()->GetId()),
                                            static_cast<uint64_t>(src_index), kOutputEnd});
      const auto out_data_endpoint =
          Encode({static_cast<uint64_t>(op_desc->GetId()), static_cast<uint64_t>(i), kOutputEnd});
      GELOGD("Start Merge node[%s][idx=%zu] and node[%s][idx=%d]", node->GetNamePtr(), i, iter->second->GetNamePtr(),
             src_index);
      Merge(out_data_endpoint, src_ref_endpoint);
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus EquivalentDataEdges::SetInputOutputSymbols(const std::vector<ge::FastNode *> &all_nodes) {
  for (const auto &iter : symbols_to_endpoints_) {
    const auto symbol = iter.first;
    for (const auto endpoint : iter.second) {
      const auto end_point = Decode(endpoint);
      const auto node_id = static_cast<uint64_t>(end_point.id);
      const auto index = static_cast<uint64_t>(end_point.index);
      GE_ASSERT_TRUE(node_id < all_nodes.size(), "node id is %lu while all node size is %zu", node_id,
                     all_nodes.size());
      // exe_graph已完成topo排序，且之后图没有变化，数组下标即为node id
      const auto node = all_nodes[node_id];
      if (end_point.direction == kInputEnd) {
        GE_ASSERT_GRAPH_SUCCESS(node->GetExtendInfo()->SetInputSymbol(index, symbol),
                                "Set symbol[%lx] for node [%lu][%s][%s] input_%lu failed", symbol, node_id,
                                node->GetNamePtr(), node->GetTypePtr(), index);
        GELOGD("[SetSymbol] set symbol[%lx] for node [%lu][%s][%s] input_%lu", symbol, node_id, node->GetNamePtr(),
               node->GetTypePtr(), index);
      } else {
        GE_ASSERT_GRAPH_SUCCESS(node->GetExtendInfo()->SetOutputSymbol(index, symbol),
                                "Set symbol[%lx] for node [%lu][%s][%s] output_%lu failed", symbol, node_id,
                                node->GetNamePtr(), node->GetTypePtr(), index);
        GELOGD("[SetSymbol] set symbol[%lx] for node [%lu][%s][%s] output_%lu", symbol, node_id, node->GetNamePtr(),
               node->GetTypePtr(), index);
      }
    }
  }
  return ge::GRAPH_SUCCESS;
}

void EquivalentDataEdges::Add(uint64_t endpoint) {
  symbols_to_endpoints_[endpoint] = {endpoint};
  endpoints_to_symbol_[endpoint] = endpoint;
  GELOGD("[AddSymbol] add %s as symbol[%lx]", Decode(endpoint).ToString().c_str(), endpoint);
}

void EquivalentDataEdges::Merge(uint64_t endpoint1, uint64_t endpoint2) {
  const auto iter1 = endpoints_to_symbol_.find(endpoint1);
  const auto iter2 = endpoints_to_symbol_.find(endpoint2);
  if ((iter1 == endpoints_to_symbol_.end()) && (iter2 == endpoints_to_symbol_.end())) {
    symbols_to_endpoints_[endpoint1] = {endpoint1, endpoint2};
    endpoints_to_symbol_[endpoint1] = endpoint1;
    endpoints_to_symbol_[endpoint2] = endpoint1;
  } else if (iter1 == endpoints_to_symbol_.end()) {
    (void)symbols_to_endpoints_[iter2->second].emplace(endpoint1);
    endpoints_to_symbol_[endpoint1] = iter2->second;
    GELOGD("[MergeSymbol] symbol1 is null, merge endpoint1[%s] to symbol2[%lx]", Decode(endpoint1).ToString().c_str(),
           iter2->second);
  } else if (iter2 == endpoints_to_symbol_.end()) {
    (void)symbols_to_endpoints_[iter1->second].emplace(endpoint2);
    endpoints_to_symbol_[endpoint2] = iter1->second;
    GELOGD("[MergeSymbol] symbol2 is null, merge endpoint2[%s] to symbol1[%lx]", Decode(endpoint2).ToString().c_str(),
           iter1->second);
  } else if (iter1->second == iter2->second) {
    GELOGD("[MergeSymbol] symbol1[%lx] is equal to symbol2[%lx], no need to merge", iter1->second, iter2->second);
  } else {
    const auto symbol1 = iter1->second;
    const auto symbol2 = iter2->second;
    auto *from_endpoints = &symbols_to_endpoints_[symbol1];
    auto *to_endpoints = &symbols_to_endpoints_[symbol2];
    auto from_symbol = symbol1;
    auto to_symbol = symbol2;

    if (from_endpoints->size() > to_endpoints->size()) {
      GELOGD("from_endpoints size[%zu] is greater than to_endpoints size[%zu], swap them.", from_endpoints->size(),
             to_endpoints->size());
      std::swap(from_endpoints, to_endpoints);
      std::swap(from_symbol, to_symbol);
    }
    GELOGD("symbol1 and symbol2 not null either, merge from_symbol[%lx] to to_symbol[%lx]", from_symbol, to_symbol);
    for (auto from_endpoint : *from_endpoints) {
      endpoints_to_symbol_[from_endpoint] = to_symbol;
      (void)symbols_to_endpoints_[to_symbol].emplace(from_endpoint);
      GELOGD("[MergeSymbol] merge endpoint[%s] to symbol[%lx]", Decode(from_endpoint).ToString().c_str(), to_symbol);
    }
    symbols_to_endpoints_.erase(from_symbol);
  }
}

uint64_t EquivalentDataEdges::Encode(const EquivalentDataEdges::EndPoint end_point) const {
  return (static_cast<uint64_t>(end_point.id) | static_cast<uint64_t>(end_point.index) << 32UL |
          static_cast<uint64_t>(end_point.direction) << 63UL);
}

EquivalentDataEdges::EndPoint EquivalentDataEdges::Decode(uint64_t endpoint) const {
  return {endpoint & 0xffffffffUL, (endpoint >> 32UL) & 0x7fffffffUL, endpoint >> 63UL};
}
}  // namespace gert
