/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "binary_partitioner.h"
#include "framework/common/debug/ge_log.h"
#include "common/checker.h"
#include "graph/utils/op_type_utils.h"

namespace ge {
BinaryGraphBuilder BinaryPartitioner::graph_builder_;

Status BinaryPartitioner::Partition(const ComputeGraphPtr &graph, 
                                    const std::vector<NodePtr> &infered_nodes,
                                    PartionResult &p_ret) {
  GE_ASSERT_NOTNULL(graph);
  if (infered_nodes.empty()) {
    GELOGE(ge::FAILED, "Partition failed! param is invalid. graph name:%s", graph->GetName().c_str());
    return GRAPH_FAILED;
  }

  auto uninfer_nodes = GetRemainingNodes(graph, infered_nodes);
  if (uninfer_nodes.empty()) {
    GELOGI("Partition completed! remaining nodes is empty. graph name:%s", graph->GetName().c_str());
    p_ret.sliced_graph = graph;
    return GRAPH_SUCCESS;
  }

  if (CheckNodesContainsCycle(infered_nodes, uninfer_nodes)) {
    GELOGE(ge::FAILED, "Partition failed! nodes contain cycle. graph name:%s", graph->GetName().c_str());
    return GRAPH_PARAM_INVALID;
  }

  auto compute_graph = graph_builder_.BuildGraph(infered_nodes, SLICED_GRAPH_NAME);
  GE_ASSERT_NOTNULL(compute_graph, "BuildGraph failed! graph name:%s", graph->GetName().c_str());
  p_ret.sliced_graph = compute_graph;

  compute_graph = graph_builder_.BuildGraph(uninfer_nodes, REMAINING_GRAPH_NAME);
  GE_ASSERT_NOTNULL(compute_graph, "BuildGraph failed! graph name:%s", graph->GetName().c_str());
  p_ret.remaining_graph = compute_graph;

  BinaryGraphIOLinkage io_link;
  io_link.sliced_graph = p_ret.sliced_graph;
  io_link.remaining_graph = p_ret.remaining_graph;
  io_link.infered_nodes = infered_nodes;
  io_link.uninfer_nodes = uninfer_nodes;

  GE_ASSERT_SUCCESS(graph_builder_.GetIOMapping(io_link));
  GE_ASSERT_SUCCESS(graph_builder_.ReplaceInputNode(io_link));
  GE_ASSERT_SUCCESS(graph_builder_.MergeSameInputNode(io_link));
  const auto ret = graph_builder_.SetInputNodeDesc(io_link);
  p_ret.out_idx_2_in_idxs = io_link.out_idx_2_in_idxs;
  return ret;
}

std::vector<NodePtr> BinaryPartitioner::GetRemainingNodes(const ComputeGraphPtr &graph, 
                                                          const std::vector<NodePtr> &infered_nodes) {
  std::vector<NodePtr> uninfer_nodes;
  for (const auto &node : graph->GetDirectNode()) {
    const auto num = std::count(infered_nodes.begin(), infered_nodes.end(), node);
    if (num == 0) {
      GE_ASSERT_TRUE(!ge::OpTypeUtils::IsDataNode(node->GetType()), "graph:%s data node:%s is not symbolize",
                     graph->GetName().c_str(), node->GetName().c_str());
      uninfer_nodes.push_back(node);
      GELOGD("uninfer node:%s", node->GetName().c_str());
    }
  }
  GELOGI("GetRemainingNodes completed. infer size:%zu, uninfer size:%zu ", infered_nodes.size(), uninfer_nodes.size());
  return uninfer_nodes;
}

bool BinaryPartitioner::CheckNodesContainsCycle(const std::vector<NodePtr> &infered_nodes,
                                                const std::vector<NodePtr> &uninfer_nodes) {
  for (const auto &node : infered_nodes) {
    GELOGD("infered node:%s", node->GetName().c_str());
    for (auto &out_data_node : node->GetInDataNodes()) {
      if (std::count(uninfer_nodes.begin(), uninfer_nodes.end(), out_data_node) != 0) {
        GELOGE(ge::FAILED, "CheckNodesContainsCycle node:%s is uninfered, but it is infered node:%s input",
               out_data_node->GetName().c_str(), node->GetName().c_str());
        return true;
      }
    }
    for (auto &out_control_node : node->GetInControlNodes()) {
      if (std::count(uninfer_nodes.begin(), uninfer_nodes.end(), out_control_node) != 0) {
        GELOGE(ge::FAILED, "CheckNodesContainsCycle node:%s is uninfered, but it is infered node:%s input",
               out_control_node->GetName().c_str(), node->GetName().c_str());
        return true;
      }
    }
  }
  return false;
}
}  // namespace ge