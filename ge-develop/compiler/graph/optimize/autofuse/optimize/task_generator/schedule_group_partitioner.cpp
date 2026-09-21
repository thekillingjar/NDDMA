/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "schedule_group_partitioner.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/ascendc_ir/utils/asc_graph_utils.h"
#include "ascendc_ir.h"
#include "ascir_ops.h"
#include "schedule_utils.h"
#include "util/mem_utils.h"

using namespace ge;

namespace optimize {
Status ScheduleGroupGraphPartitioner::PartitionByConnectivity(const ::ascir::ImplGraph &optimize_graph,
                                                              std::vector<::ascir::ImplGraph> &sub_optimize_graphs,
                                                              std::vector<AscNodePtr> node_order) {
  std::map<std::string, ::ascir::NodeView> name_to_output_nodes;
  size_t num_nodes = 0;
  for (const auto &node : optimize_graph.GetAllNodes()) {
    GE_ASSERT_NOTNULL(node);
    if (node->GetOutDataNodes().empty()) {
      name_to_output_nodes.emplace(node->GetName(), node);
      GELOGI("Output node: %s[%s]", node->GetName().c_str(), node->GetType().c_str());
    }
    ++num_nodes;
  }
  if (node_order.empty()) {
    for (const auto &name_and_node : name_to_output_nodes) {
      node_order.emplace_back(name_and_node.second);
    }
  }
  int32_t index = 0;
  std::set<ge::NodePtr> visited;
  for (const auto &node : node_order) {
    const auto &root_node = node;
    if (visited.find(root_node) == visited.cend()) {
      ::ascir::ImplGraph sub_graph((optimize_graph.GetName() + "_" + std::to_string(index)).c_str());
      GE_ASSERT_TRUE(sub_graph.CopyAttrFrom(optimize_graph));
      GE_CHK_STATUS_RET(AddConnectedNodes(root_node, sub_graph, visited),
                        "Failed to add connected nodes, root_node = %s", root_node->GetNamePtr());
      GE_ASSERT_GRAPH_SUCCESS(ScheduleUtils::TopologicalSorting(sub_graph));
      sub_optimize_graphs.emplace_back(sub_graph);
      ++index;
    }
  }
  if (sub_optimize_graphs.size() > 1U) {  // 切图后会产生冗余轴信息
    for (auto &sub_optimize_graph : sub_optimize_graphs) {
      GE_CHK_STATUS_RET(ScheduleUtils::RemoveUnusedAxes(sub_optimize_graph), "Failed to remove unused axes");
    }
  }
  if (visited.size() != num_nodes) {
    for (const auto &node : optimize_graph.GetAllNodes()) {
      if (visited.find(node) == visited.cend()) {
        GELOGE(ge::FAILED, "node: %s[%s] not visited", node->GetName().c_str(), node->GetType().c_str());
      }
    }
    GELOGE(ge::FAILED, "number of visited nodes = %zu, num_nodes = %zu", visited.size(), num_nodes);
    return ge::FAILED;
  }

  GELOGI("Partition success, subgraph number = %zu", sub_optimize_graphs.size());
  return ge::SUCCESS;
}

Status ScheduleGroupGraphPartitioner::NeedRefreshAxisSize(const ::ascir::ImplGraph &optimize_graph,
                                                          bool &need_refresh) {
  const auto concat_node = ScheduleUtils::FindFirstNodeOfType<ge::ascir_op::Concat>(optimize_graph);
  GE_CHK_BOOL_RET_SPECIAL_STATUS(concat_node == nullptr, ge::SUCCESS, "no Concat node was found");
  need_refresh = true;
  return ge::SUCCESS;
}

Status ScheduleGroupGraphPartitioner::CollectConnectedNodes(const ge::AscNodePtr &root_node,
                                                            std::set<ge::NodePtr> &visited,
                                                            std::vector<ge::AscNodePtr> &asc_nodes) {
  std::list<ge::NodePtr> next_nodes{root_node};
  visited.emplace(root_node);
  while (!next_nodes.empty()) {
    auto node = next_nodes.front();
    next_nodes.pop_front();
    auto asc_node = std::dynamic_pointer_cast<ge::AscNode>(node);
    GE_CHECK_NOTNULL(asc_node);
    asc_nodes.emplace_back(std::move(asc_node));
    for (auto &in_node : node->GetInNodes()) {
      if (visited.find(in_node) == visited.cend()) {
        next_nodes.emplace_back(in_node);
        visited.emplace(in_node);
      }
    }
    for (auto &out_node : node->GetOutNodes()) {
      if (visited.find(out_node) == visited.cend()) {
        next_nodes.emplace_back(out_node);
        visited.emplace(out_node);
      }
    }
  }
  return ge::SUCCESS;
}

Status ScheduleGroupGraphPartitioner::AddConnectedNodes(const ge::AscNodePtr &root_node, ::ascir::ImplGraph &sub_graph,
                                                        std::set<ge::NodePtr> &all_visited) {
  GELOGI("AddConnectedNodes in, root_node = %s[%s]", root_node->GetName().c_str(), root_node->GetType().c_str());
  std::unordered_map<std::string, ge::NodePtr> all_new_nodes;
  std::set<ge::NodePtr> visited;
  std::vector<ge::AscNodePtr> asc_nodes;
  GE_ASSERT_SUCCESS(CollectConnectedNodes(root_node, visited, asc_nodes));
  std::sort(asc_nodes.begin(), asc_nodes.end(), CompareByNodeId);
  for (const auto &asc_node : asc_nodes) {
    const auto &op_desc = ge::GraphUtils::CopyOpDesc(asc_node->GetOpDesc(), nullptr);
    GE_CHECK_NOTNULL(op_desc);
    op_desc->SetName(asc_node->GetName());
    ge::Operator op = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc);
    auto dst_new_node = sub_graph.AddNode(op);
    all_new_nodes[dst_new_node->GetName()] = dst_new_node;
    GE_ASSERT_TRUE(AscGraph::CopyAscNodeTensorAttr(asc_node, dst_new_node),
                   "DoCopyAscNodeTensorAttr failed, node = %s[%s]", asc_node->GetNamePtr(), asc_node->GetTypePtr());
  }

  for (const auto &src_node : visited) {
    GE_CHK_STATUS_RET(ge::GraphUtils::RelinkGraphEdges(src_node, "", all_new_nodes), "RelinkGraphEdges failed");
  }
  all_visited.insert(visited.cbegin(), visited.cend());
  return ge::SUCCESS;
}

bool ScheduleGroupGraphPartitioner::CompareByNodeId(const AscNodePtr &lhs, const AscNodePtr &rhs) {
  return lhs->GetOpDesc()->GetId() < rhs->GetOpDesc()->GetId();
}

Status ScheduleGroupGraphPartitioner::RecordAxisSizes(const std::vector<ge::Expression> &repeats,
                                                      const std::vector<int64_t> &axis_ids,
                                                      std::map<ge::AxisId, ge::Expression> &axis_id_to_size) {
  for (size_t j = 0UL; j < repeats.size(); ++j) {
    if (ge::SymbolicUtils::StaticCheckEq(repeats[j], ops::One) != ge::TriBool::kTrue) {
      const auto axis_id = axis_ids[j];
      if (!axis_id_to_size.emplace(axis_id, repeats[j]).second) {
        GE_LOGW_IF(ge::SymbolicUtils::StaticCheckEq(repeats[j], axis_id_to_size[axis_id]) != TriBool::kTrue,
                   "inconsistent axis size, id = %ld, size0 = %s, size1 = %s", axis_id,
                   SymbolicUtils::ToString(repeats[j]).c_str(),
                   SymbolicUtils::ToString(axis_id_to_size[axis_id]).c_str());
      }
    }
  }
  return SUCCESS;
}

Status ScheduleGroupGraphPartitioner::RefreshAxisSize(const ::ascir::ImplGraph &sub_graph) {
  GELOGI("RefreshAxisSize start, graph_name = %s", sub_graph.GetName().c_str());
  const auto graph_attr = ge::AscGraphUtils::GetComputeGraph(sub_graph)->GetOrCreateAttrsGroup<ge::AscGraphAttr>();
  GE_ASSERT_NOTNULL(graph_attr);
  std::map<ge::AxisId, ge::Expression> axis_id_to_size;
  for (const auto &node : sub_graph.GetAllNodes()) {
    if (ScheduleUtils::IsBuffer(node)) {
      continue;
    }
    for (uint32_t i = 0U; i < node->GetAllOutDataAnchorsSize(); ++i) {
      const auto &repeats = node->outputs[i].attr.repeats;
      const auto &axis_ids = node->outputs[i].attr.axis;
      GE_ASSERT_TRUE(repeats.size() == axis_ids.size(), "node: %s:%u repeats.size() = %zu, but axis_ids.size() = %zu",
                     node->GetNamePtr(), i, repeats.size(), axis_ids.size());
      GE_ASSERT_SUCCESS(RecordAxisSizes(repeats, axis_ids, axis_id_to_size));
    }
  }
  for (const auto &axis : graph_attr->axis) {
    GE_ASSERT_NOTNULL(axis);
    const auto it = axis_id_to_size.find(axis->id);
    if ((it != axis_id_to_size.cend()) && (SymbolicUtils::StaticCheckEq(axis->size, it->second) != TriBool::kTrue)) {
      GELOGI("update axis, id = %ld, name = %s, from %s to %s", axis->id, axis->name.c_str(),
             SymbolicUtils::ToString(axis->size).c_str(), SymbolicUtils::ToString(it->second).c_str());
      axis->size = it->second;
    }
  }
  GELOGI("RefreshAxisSize end, graph_name = %s", sub_graph.GetName().c_str());
  return SUCCESS;
}
}  // namespace optimize