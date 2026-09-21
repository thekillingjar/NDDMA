/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "remove_launch_free_edge.h"
#include "common/compile_profiling/ge_call_wrapper.h"
#include "core/builder/node_types.h"
#include "graph/utils/graph_dump_utils.h"
#include "kernel/memory/memory_kernel.h"
#include "graph/utils/op_desc_utils_ex.h"
namespace gert {
namespace bg {
namespace {
bool IsLaunchTargetNode(const ge::FastNode *const node) {
  const auto &node_type = node->GetTypePtr();
  // ExecuteOpFunc inner use allocator, so skip, in case workspace addr same with output addr
  return IsExecuteOplaunchNode(node_type);
}
bool IsFreeTargetNode(const ge::FastNode *const node) {
  const auto &node_type = node->GetTypePtr();
  static std::vector<const char *> kFreeKernels = {"FreeMemory", "FreeMemHbm", "FreeBatchHbm", "FreeTensorMemory"};
  auto func = [&node_type](const char *const type) { return (strcmp(node_type, type) == 0); };
  return std::any_of(kFreeKernels.begin(), kFreeKernels.end(), func);
}

ge::graphStatus ReplaceFreeNode(const ge::FastNode *free_node) {
  if (strcmp(free_node->GetTypePtr(), "FreeTensorMemory") == 0) {
    return ge::GRAPH_SUCCESS;
  }
  static std::map<std::string, std::string> origin_free_to_new_free_types = {
      {kernel::kFreeMemory, kernel::kFreeMemoryHoldAddr},
      {kernel::kFreeMemHbm, kernel::kFreeMemHbmHoldAddr},
      {kernel::kFreeBatchHbm, kernel::kFreeBatchHbmHoldAddr}};
  auto op_desc = free_node->GetOpDescPtr();
  GE_ASSERT_NOTNULL(op_desc);
  const auto iter = origin_free_to_new_free_types.find(free_node->GetType());
  GE_ASSERT_TRUE(iter != origin_free_to_new_free_types.end(), "free node %s type %s is invalid",
                 free_node->GetNamePtr(), free_node->GetTypePtr());
  ge::OpDescUtilsEx::SetType(op_desc, iter->second);
  op_desc->SetName(iter->second + "_" + free_node->GetName());
  return ge::GRAPH_SUCCESS;
}
}
ge::graphStatus RemoveLaunchFreeEdge::Run(ge::ExecuteGraph *const graph, bool &changed) {
  GE_TIMESTAMP_START(RemoveLaunchFreeEdge);
  const auto launch_nodes = graph->GetAllNodes(IsLaunchTargetNode);
  std::unordered_set<ge::FastNode *> target_free_nodes;
  for (const auto launch_node : launch_nodes) {
    std::vector<ge::FastNode *> alloc_nodes;
    std::vector<ge::FastNode *> free_nodes;
    std::vector<ge::Edge<ge::FastNode> *> launch_free_ctrl_edges;
    for (const auto out_ctrl_edge : launch_node->GetAllOutControlEdges()) {
      const auto node = out_ctrl_edge->dst;
      if (IsFreeTargetNode(node)) {
        bool is_for_host = false;
        for (const auto in_node : node->GetInDataNodes()) {
          if (IsAllocHostNode(in_node->GetTypePtr())) {
            is_for_host = true;
            break;
          }
          alloc_nodes.push_back(in_node);
        }
        if (!is_for_host) {
          free_nodes.push_back(node);
          target_free_nodes.insert(node);
          launch_free_ctrl_edges.push_back(out_ctrl_edge);
        }
      }
    }

    for (const auto alloc_node : alloc_nodes) {
      auto alloc_op_desc = alloc_node->GetOpDescBarePtr();
      GE_ASSERT_NOTNULL(alloc_op_desc);
      (void) ge::AttrUtils::SetInt(alloc_op_desc, "remove_launch_free_edge_alloc", 1);
      for (const auto free_node : free_nodes) {
        bool has_edge = false;
        for (const auto in_node : free_node->GetAllInNodes()) {
          if (in_node == alloc_node) {
            has_edge = true;
            break;
          }
        }
        if (has_edge) {
          continue;
        }
        auto current_graph = alloc_node->GetExtendInfo()->GetOwnerGraphBarePtr();
        GE_ASSERT_NOTNULL(current_graph);
        current_graph->AddEdge(alloc_node, ge::kControlEdgeIndex, free_node, ge::kControlEdgeIndex);
        GELOGD("add control edge from %s to %s", alloc_node->GetNamePtr(), free_node->GetNamePtr());
      }
    }
    for (const auto edge : launch_free_ctrl_edges) {
      const auto launch_node_local = edge->src;
      GE_ASSERT_TRUE(IsLaunchTargetNode(launch_node_local));
      const auto current_graph = launch_node_local->GetExtendInfo()->GetOwnerGraphBarePtr();
      GE_ASSERT_NOTNULL(current_graph);
      GELOGD("remove ctrl edge from %s to %s", launch_node_local->GetNamePtr(), edge->dst->GetNamePtr());
      GE_ASSERT_GRAPH_SUCCESS(current_graph->RemoveEdge(edge));
    }
  }
  for (auto free_node : target_free_nodes) {
    GE_ASSERT_SUCCESS(ReplaceFreeNode(free_node));
  }
  if (launch_nodes.size() > 0) {
    changed = true;
    ge::DumpGraph(graph, "RemoveLaunchFreeEdgeAfter");
    GE_TIMESTAMP_EVENT_END(RemoveLaunchFreeEdge, "Pass::RemoveLaunchFreeEdge");
  }
  return ge::GRAPH_SUCCESS;
}
}  // namespace bg
}  // namespace gert
