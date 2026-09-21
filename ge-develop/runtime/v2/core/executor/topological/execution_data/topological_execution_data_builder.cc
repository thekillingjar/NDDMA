/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "topological_execution_data_builder.h"
#include "core/executor/sequential/execution_data/sequential_execution_data.h"
#include "core/utils/executor_utils.h"

namespace gert {
TopologicalExecutionDataBuilder::TopologicalExecutionDataBuilder(GraphExecutorBuilder &executor_builder)
    : ExecutionDataBuilder(executor_builder), base_ed_builder_(executor_builder) {}

ResourceGuardPtr TopologicalExecutionDataBuilder::Build() {
  GraphNode graph_nodes;
  return Build(graph_nodes);
}

ResourceGuardPtr TopologicalExecutionDataBuilder::Build(GraphNode &graph_nodes) {
  auto resource_guard = ge::MakeUnique<TopologicalResourceGuard>();
  GE_ASSERT_NOTNULL(resource_guard);
  auto execution_data_holder = ge::MakeUnique<uint8_t[]>(sizeof(TopologicalExecutionData));
  GE_ASSERT_NOTNULL(execution_data_holder);

  auto execution_data = reinterpret_cast<TopologicalExecutionData *>(execution_data_holder.get());
  GE_ASSERT_GRAPH_SUCCESS(Build(execution_data, resource_guard.get(), graph_nodes));
  resource_guard->ResetExecutionData(std::move(execution_data_holder));
  GE_ASSERT_SUCCESS(CreateKernelOutputs(base_ed_builder_.GetOrderedGraphToExeNodes()));
  return resource_guard;
}

ge::graphStatus TopologicalExecutionDataBuilder::Build(TopologicalExecutionData *execution_data,
                                                       ResourceGuard *resource_guard, GraphNode &graph_nodes) {
  GE_ASSERT_SUCCESS(base_ed_builder_.ReOrderByPriority(false).Build(&(execution_data->base_ed), resource_guard,
                                                                    graph_nodes));

  auto &graph_to_exe_nodes = base_ed_builder_.GetOrderedGraphToExeNodes();
  auto exe_nodes_size = graph_to_exe_nodes.size();
  graph_nodes.node_indegrees.resize(exe_nodes_size);
  graph_nodes.node_watchers.resize(exe_nodes_size);
  const auto topo_resource_guard = reinterpret_cast<TopologicalResourceGuard *>(resource_guard);
  for (auto &graph_node_to_exe_node : graph_to_exe_nodes) {
    Watcher *watcher = nullptr;
    GE_ASSERT_SUCCESS(graph_nodes.ReadInTopoInfo(graph_node_to_exe_node, watcher));
    topo_resource_guard->PushWatcher(watcher);
    if (priority_execution_) {
      SetPriorityForNode(graph_node_to_exe_node);
    }
  }

  GE_ASSERT_SUCCESS(CreateExecutionData(graph_nodes, execution_data, resource_guard));
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus TopologicalExecutionDataBuilder::CreateExecutionData(GraphNode &graph_node,
                                                                     TopologicalExecutionData *execution_data,
                                                                     ResourceGuard *resource_guard) const {
  const auto topo_resource_guard = reinterpret_cast<TopologicalResourceGuard *>(resource_guard);
  execution_data->start_num = graph_node.start_nodes.size();
  execution_data->start_nodes =
      reinterpret_cast<Node **>(topo_resource_guard->ResetStartNodesArray(CreateCArray(graph_node.start_nodes)));

  execution_data->node_indegrees =
      reinterpret_cast<int64_t *>(topo_resource_guard->ResetNodesIndgreeArray(CreateCArray(graph_node.node_indegrees)));
  execution_data->node_indegrees_backup = reinterpret_cast<int64_t *>(
      topo_resource_guard->ResetNodesWaitIndgreeArray(CreateCArray(graph_node.node_indegrees)));
  execution_data->node_watchers =
      reinterpret_cast<Watcher **>(topo_resource_guard->ResetWatchersArray(CreateCArray(graph_node.node_watchers)));

  if (priority_execution_) {
    execution_data->ready_queue =
        topo_resource_guard->ResetReadyQueue(CreatePriorityQueue(graph_node.nodes.size() + 1U));
  } else {
    execution_data->ready_queue = topo_resource_guard->ResetReadyQueue(CreateRingQueue(graph_node.nodes.size()));
  }

  return ge::GRAPH_SUCCESS;
}
TopologicalExecutionDataBuilder &TopologicalExecutionDataBuilder::PriorityExecution(bool flag) {
  priority_execution_ = flag;
  return *this;
}
}  // namespace gert
