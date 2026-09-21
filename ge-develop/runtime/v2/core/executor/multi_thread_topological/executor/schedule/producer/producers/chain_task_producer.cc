/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "chain_task_producer.h"
#include "core/executor/multi_thread_topological/executor/schedule/task/task_allocator.h"
#include "core/executor/multi_thread_topological/executor/schedule/task/task_package.h"
#include "common/checker.h"
#include "core/execution_data.h"

namespace gert {
ChainTaskProducer::ChainTaskProducer(size_t task_capacity) : task_allocator_(task_capacity) {}

ge::Status ChainTaskProducer::Prepare(const void *execution_data) {
  GE_ASSERT_TRUE(execution_data != nullptr);
  execution_data_ = execution_data;
  tags_.Reset(reinterpret_cast<const ExecutionData *>(execution_data_)->base_ed.node_num);
  return ge::SUCCESS;
}

ge::Status ChainTaskProducer::StartUp() {
  auto execution_data = reinterpret_cast<const ExecutionData *>(execution_data_);
  for (size_t i = 0U; i < execution_data->start_num; ++i) {
    auto exe_task = task_allocator_.New();
    if (exe_task != nullptr) {
      CollectChain(*exe_task, execution_data->start_nodes[i]);
      tasks_.push_back(*exe_task);
    }
  }
  return ge::SUCCESS;
}

bool ChainTaskProducer::AddKernel(ExecTask &task, Node *node) {
  auto kernel_tag = tags_.GetByNode(node);
  if (kernel_tag == ExecTaskType::NORMAL) {
    task.AddKernel(node);
    return true;
  }
  task.SetType(kernel_tag);
  task.AddKernel(node);
  return true;
}

void ChainTaskProducer::CollectChain(ExecTask &task, Node *node) {
  auto execution_data = reinterpret_cast<const ExecutionData *>(execution_data_);
  AddKernel(task, node);
  auto current_kernel_type = reinterpret_cast<const ExtendedKernelContext *>(&node->context)->GetKernelType();
  if ((current_kernel_type != nullptr) &&
      (strcmp(current_kernel_type, "SwitchNotify") == 0 || strcmp(current_kernel_type, "CondSwitchNotify") == 0)) {
    return;
  }
  Watcher *watchers = execution_data->node_watchers[node->node_id];
  for (size_t i = 0U; i < watchers->watch_num; ++i) {
    NodeIdentity dst_id = watchers->node_ids[i];
    if (execution_data->node_indegrees_backup[dst_id] == 1 && --execution_data->node_indegrees[dst_id] == 0) {
      CollectChain(task, execution_data->base_ed.nodes[dst_id]);
    }
  }
}

ge::Status ChainTaskProducer::KernelDone(const Node *node) {
  auto execution_data = reinterpret_cast<const ExecutionData *>(execution_data_);
  Watcher *watchers = execution_data->node_watchers[node->node_id];
  for (size_t i = 0U; i < watchers->watch_num; ++i) {
    NodeIdentity node_id = watchers->node_ids[i];
    if (execution_data->node_indegrees[node_id] == 0) {
      execution_data->node_indegrees[node_id] = execution_data->node_indegrees_backup[node_id];
      continue;
    }
    if (--execution_data->node_indegrees[node_id] == 0) {
      auto exe_task = task_allocator_.New();
      if (exe_task != nullptr) {
        CollectChain(*exe_task, execution_data->base_ed.nodes[node_id]);
        tasks_.push_back(*exe_task);
        execution_data->node_indegrees[node_id] = execution_data->node_indegrees_backup[node_id];
      }
    }
  }
  return ge::SUCCESS;
}

ge::Status ChainTaskProducer::DoRecycleTask(ExecTask &task) {
  auto kernel_ret = task.GetKernelRet();
  if (kernel_ret != ge::GRAPH_SUCCESS) {
    task_allocator_.Free(task);
    return kernel_ret;
  }
  for (auto &iter : task.GetTaskNodeList()) {
    KernelDone(iter);
  }
  task_allocator_.Free(task);
  return ge::SUCCESS;
}

ge::Status ChainTaskProducer::FreeAllTask(TaskPackage &package) {
  while (auto task = package.pop_front()) {
    if (task) {
      task_allocator_.Free(*task);
    }
  }
  return ge::SUCCESS;
}

void ChainTaskProducer::Dump() const {
  GEEVENT("|-- Producer [type : chain]");
}
}  // namespace gert
