/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "multi_thread_execution_data_builder.h"
#include "multi_thread_execution_data.h"
#include "core/executor/multi_thread_topological/executor/schedule/scheduler/task_scheduler_factory.h"
#include "core/executor/multi_thread_topological/executor/schedule/producer/task_producer_factory.h"
#include "core/utils/executor_utils.h"
#include "multi_thread_exe_graph_resource_guard.h"
#include "framework/runtime/executor_option/multi_thread_executor_option.h"

namespace gert {
MultiThreadExecutionDataBuilder::MultiThreadExecutionDataBuilder(GraphExecutorBuilder &executor_builder)
    : ExecutionDataBuilder(executor_builder), base_ed_builder_(executor_builder) {}

ResourceGuardPtr MultiThreadExecutionDataBuilder::Build() {
  GraphNode graph_nodes;
  return Build(graph_nodes);
}

ResourceGuardPtr MultiThreadExecutionDataBuilder::Build(GraphNode &graph_nodes) {
  auto resource_guard = ge::MakeUnique<MultiThreadResourceGuard>();
  auto execution_data_holder = ge::MakeUnique<uint8_t[]>(sizeof(MultiThreadExecutionData));
  GE_ASSERT_NOTNULL(execution_data_holder);
  auto execution_data = reinterpret_cast<MultiThreadExecutionData *>(execution_data_holder.get());
  base_ed_builder_.ReOrderByPriority(false).Build(&(execution_data->topo_ed.base_ed),
                                                  resource_guard.get(), graph_nodes);

  GE_ASSERT_SUCCESS(graph_nodes.EnsureNodeExeInOrder(GetExecutorBuilder().GetExeGraph().get()));

  auto &graph_to_exe_nodes = base_ed_builder_.GetOrderedGraphToExeNodes();
  auto exe_nodes_size = graph_to_exe_nodes.size();
  graph_nodes.node_indegrees.resize(exe_nodes_size);
  graph_nodes.node_watchers.resize(exe_nodes_size);
  for (auto &graph_node_to_exe_node : graph_to_exe_nodes) {
    Watcher *watcher = nullptr;
    GE_ASSERT_SUCCESS(graph_nodes.ReadInTopoInfo(graph_node_to_exe_node, watcher));
    resource_guard->PushWatcher(watcher);
    SetPriorityForNode(graph_node_to_exe_node);
  }
  GE_ASSERT_SUCCESS(CreateExecutionData(graph_nodes, &(execution_data->topo_ed), resource_guard.get()));

  GE_ASSERT_SUCCESS(CreateKernelOutputs(base_ed_builder_.GetOrderedGraphToExeNodes()));
  auto multi_thread_executor_option =
      reinterpret_cast<MultiThreadExecutorOption *>(GetExecutorBuilder().GetExecutorOption());
  auto thread_num = multi_thread_executor_option->GetThreadNum();
  GE_ASSERT_TRUE(thread_num >= kLeastThreadNumber, "new thread num is less than least %u", kLeastThreadNumber);
  GELOGD("multi thread executor will create %zu threads", thread_num);

  TaskSchedulerConfig cfg;
  cfg.producer_cfg.type = TaskProducerFactory::GetInstance().GetProducerType();
  cfg.AddWorkers(1, ExecTaskType::MEMORY, TaskThreadMode::URGENT, 1);
  if (thread_num > kLeastThreadNumber) {
    cfg.AddWorkers(1, ExecTaskType::LAUNCH, TaskThreadMode::URGENT, 1);
    thread_num--;
  }
  cfg.AddWorkers(1, ExecTaskType::NORMAL, TaskThreadMode::URGENT, thread_num - 1); // one for memory worker

  execution_data->scheduler = resource_guard->ResetTaskScheduler(
      std::unique_ptr<TaskScheduler>(TaskSchedulerFactory::GetInstance().Create(cfg)));
  GE_ASSERT_SUCCESS(execution_data->scheduler->Prepare(TaskScheduler::ScheduleData(&(execution_data->topo_ed))));

  resource_guard->ResetExecutionData(std::move(execution_data_holder));
  return resource_guard;
}

ge::graphStatus MultiThreadExecutionDataBuilder::CreateExecutionData(GraphNode &graph_node,
                                                                     TopologicalExecutionData *topo_execution_data,
                                                                     ResourceGuard *resource_guard) const {
  const auto topo_resource_guard = reinterpret_cast<TopologicalResourceGuard *>(resource_guard);
  topo_execution_data->start_num = graph_node.start_nodes.size();
  topo_execution_data->start_nodes =
      reinterpret_cast<Node **>(topo_resource_guard->ResetStartNodesArray(CreateCArray(graph_node.start_nodes)));

  topo_execution_data->node_indegrees =
      reinterpret_cast<int64_t *>(topo_resource_guard->ResetNodesIndgreeArray(CreateCArray(graph_node.node_indegrees)));
  topo_execution_data->node_indegrees_backup = reinterpret_cast<int64_t *>(
      topo_resource_guard->ResetNodesWaitIndgreeArray(CreateCArray(graph_node.node_indegrees)));
  topo_execution_data->node_watchers =
      reinterpret_cast<Watcher **>(topo_resource_guard->ResetWatchersArray(CreateCArray(graph_node.node_watchers)));

  topo_execution_data->ready_queue =
      topo_resource_guard->ResetReadyQueue(CreatePriorityQueue(graph_node.nodes.size() + 1U));
  return ge::GRAPH_SUCCESS;
}
}  // namespace gert