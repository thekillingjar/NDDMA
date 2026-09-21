/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "kernel_task_producer.h"
#include "common/checker.h"
#include "core/execution_data.h"
#include "core/builder/node_types.h"

namespace gert {
namespace {
const size_t kMaxGenerateTaskNum = 60;
bool IsNeedGenWatcherAfterExecute(const Node *node) {
  auto node_type = reinterpret_cast<const ExtendedKernelContext *>(&node->context)->GetKernelType();
  if ((node_type != nullptr) &&
      ((strcmp(node_type, "SwitchNotify") == 0 || strcmp(node_type, "CondSwitchNotify") == 0))) {
    return true;
  }
  return false;
}
bool IsStopGenWatcherNode(const Node *node) {
  const auto extended_kernel_info = reinterpret_cast<const ExtendedKernelContext *>(&node->context);
  const auto &node_type = extended_kernel_info->GetKernelType();
  // while没做优先级的修复，所以BranchDone和WaitAnyOne要停一下
  // while的WaitAnyOne的不能用recovery indegree，因为实际执行时可能会变
  if ((node_type != nullptr) && (strcmp(node_type, "WaitAnyone") == 0)) {
    if ((extended_kernel_info->GetComputeNodeInfo() != nullptr)) {
      const auto &compute_node_type = extended_kernel_info->GetComputeNodeInfo()->GetNodeType();
      if (IsWhileType(compute_node_type)) {
        return true;
      }
    }
  }
  return false;
}
}  // namespace

KernelTaskProducer::KernelTaskProducer(size_t task_capacity, size_t thread_num)
    : task_allocator_(task_capacity), thread_num_(thread_num) {}

void KernelTaskProducer::RecoverNodeInDegrees(const ExecutionData *execution_data) {
  size_t node_num = execution_data->base_ed.node_num;
  for (size_t index = 0U; index < node_num; ++index) {
    auto indegree = all_tasks_[index]->GetRecoveryIndegree();
    execution_data->node_indegrees[index] = indegree;
    all_tasks_[index]->ReInit();
  }
}

ge::Status KernelTaskProducer::Prepare(const void *execution_data) {
  GE_ASSERT_TRUE(execution_data != nullptr);
  execution_data_ = execution_data;
  const auto ed = reinterpret_cast<const ExecutionData *>(execution_data_);
  auto node_size = ed->base_ed.node_num;
  tags_.Reset(node_size, thread_num_);
  all_tasks_.resize(node_size);
  for (size_t i = 0U; i < node_size; ++i) {
    auto exe_task = task_allocator_.New();
    if (exe_task != nullptr) {
      AddKernel(*exe_task, ed->base_ed.nodes[i]);
      exe_task->SetRecoveryIndegree(ed->node_indegrees_backup[i]);
      all_tasks_[i] = exe_task;
    }
  }
  for (size_t i = 0U; i < node_size; i++) {
    auto watchers = ed->node_watchers[i];
    auto &task = all_tasks_[i];
    task->SetIndegree(ed->node_indegrees_backup[i]);

    if (IsNeedGenWatcherAfterExecute(ed->base_ed.nodes[i])) {
      task->SetGenWatcherAfterExecute();
      continue;
    }

    std::priority_queue<ExecTask *, std::vector<ExecTask*>, ExecTaskCompare> priority_task_queue;
    for (size_t j = 0U; j < watchers->watch_num; j++) {
      auto watch_id = watchers->node_ids[j];
      priority_task_queue.push(all_tasks_[watch_id]);
      if (IsStopGenWatcherNode(ed->base_ed.nodes[watch_id])) {
        task->SetStopGenWatcher(true);
      }
    }
    while (!priority_task_queue.empty()) {
      auto watcher_task = priority_task_queue.top();
      task->AddWatcher(watcher_task);
      if (task->GetPriority() > watcher_task->GetPriority()) {
        task->SetStopGenWatcher(true);
        GELOGI("pre node %s priority %lld greater than watcher node %s priority %lld, pre node stop gen watcher",
               reinterpret_cast<const ExtendedKernelContext *>(&task->GetTaskNodeList()[0]->context)->GetKernelName(),
               task->GetPriority(),
               reinterpret_cast<const ExtendedKernelContext *>(&watcher_task->GetTaskNodeList()[0]->context)
                   ->GetKernelName(),
               watcher_task->GetPriority());
      }
      priority_task_queue.pop();
    }
  }

  ordered_task_id_ = 0U;
  // 根据假设完成，收集到第一个SwitchNotify之前的所有task
  std::map<size_t, int64_t> node_id_to_recovery_indegree;
  std::priority_queue<ExecTask *, std::vector<ExecTask *>, ExecTaskCompare> priority_start_task_queue;
  for (size_t i = 0U; i < ed->start_num; ++i) {
    CollectStartTasks(ed->start_nodes[i], node_id_to_recovery_indegree, priority_start_task_queue);
  }
  while (!priority_start_task_queue.empty()) {
    auto task = priority_start_task_queue.top();
    start_tasks_.push_back(task);
    GELOGD("start tasks push node %s",
           reinterpret_cast<const ExtendedKernelContext *>(&task->GetTaskNodeList()[0]->context)->GetKernelName());
    priority_start_task_queue.pop();
  }
  RecoveryPriorityQueue();
  GELOGI("start tasks size: %zu, total kernel num: %zu", start_tasks_.size(), ed->base_ed.node_num);
  return ge::SUCCESS;
}

bool KernelTaskProducer::AddKernel(ExecTask &task, Node *node) {
  auto kernel_tag = tags_.GetByNode(node);
  if (kernel_tag == ExecTaskType::NORMAL) {
    task.AddKernel(node);
    return true;
  }
  task.SetType(kernel_tag);
  task.AddKernel(node);
  return true;
}

ge::Status KernelTaskProducer::ReInit() {
  GE_ASSERT_TRUE(priority_task_queue_.size() == start_tasks_.size());
  while (!produced_tasks_.empty()) {
    produced_tasks_.pop();
  }
  GE_ASSERT_TRUE(ordered_task_id_ == start_tasks_.size());
  has_generate_task_num_ = 0U;
  return ge::SUCCESS;
}

ge::Status KernelTaskProducer::StartUp() {
  GE_ASSERT_SUCCESS(ReInit());
  return ge::SUCCESS;
}

void KernelTaskProducer::RecoveryPriorityQueue() {
  for (auto task : start_tasks_) {
    task->StartTaskReInit();
  }
  priority_task_queue_ = std::priority_queue<ExecTask *, std::vector<ExecTask *>, ExecTaskCompare>(start_tasks_.begin(),
                                                                                                   start_tasks_.end());
  ordered_task_id_ = priority_task_queue_.size();
}

ge::Status KernelTaskProducer::EndUp() {
  RecoveryPriorityQueue();
  return ge::SUCCESS;
}

KernelTaskProducer::~KernelTaskProducer() noexcept {
  for (auto &task : all_tasks_) {
    task_allocator_.Free(*task);
  }
}

TaskPackage KernelTaskProducer::Produce() {
  if (latest_target_task_ && !latest_target_task_->IsGenWatcher()) {
    GELOGD("latest_target_task %s not execute",
           reinterpret_cast<const ExtendedKernelContext *>(&latest_target_task_->GetTaskNodeList()[0]->context)
               ->GetKernelName());
    return std::move(tasks_);
  }
  latest_target_task_ = nullptr;
  while (!priority_task_queue_.empty() && (tasks_.size() < thread_num_)) {
    auto task = priority_task_queue_.top();
    tasks_.push_back(*task);
    GELOGD("produce kernel %s",
           reinterpret_cast<const ExtendedKernelContext *>(&task->GetTaskNodeList()[0]->context)->GetKernelName());
    priority_task_queue_.pop();
    if (task->IsGenWatcherAfterExecute()) {
      latest_target_task_ = task;
      break;
    }
    produced_tasks_.push(task);
  }
  return std::move(tasks_);
}

ge::Status KernelTaskProducer::Recycle(TaskPackage &package) {
  while (auto task = package.pop_front()) {
    if (task->IsGenWatcherAfterExecute()) {
      produced_tasks_.push(task);
    }
    auto kernel_ret = task->GetKernelRet();
    if (kernel_ret != ge::GRAPH_SUCCESS) {
      auto execution_data = reinterpret_cast<const ExecutionData *>(execution_data_);
      RecoverNodeInDegrees(execution_data);
      RecoveryPriorityQueue();
      return kernel_ret;
    }
  }
  while (!produced_tasks_.empty()) {
    auto task = produced_tasks_.top();
    produced_tasks_.pop();
    auto ret = DoRecycleTask(*task);
    if (ret != ge::SUCCESS) {
      return ret;
    }
  }
  has_generate_task_num_ = 0U;
  return ge::SUCCESS;
}

void KernelTaskProducer::CollectStartTasks(
    const Node *node, std::map<size_t, int64_t> &node_id_to_recovery_indegree,
    std::priority_queue<ExecTask *, std::vector<ExecTask *>, ExecTaskCompare> &priority_start_task_queue) {
  auto execution_data = reinterpret_cast<const ExecutionData *>(execution_data_);
  auto node_id = node->node_id;
  auto &task = all_tasks_[node_id];
  task->SetTaskId(ordered_task_id_++);
  priority_start_task_queue.push(task);

  if (task->IsGenWatcherAfterExecute()) {
    task->SetIsGenWatcherBackup(false);
    return;
  }
  if (task->IsStopGenWatcher()) {
    GELOGD("node %s need stop gen watcher, skip",
           reinterpret_cast<const ExtendedKernelContext *>(&node->context)->GetKernelName());
    return;
  }
  Watcher *watchers = execution_data->node_watchers[node->node_id];
  task->SetIsGenWatcherBackup(true);
  for (size_t i = 0U; i < watchers->watch_num; ++i) {
    NodeIdentity dst_id = watchers->node_ids[i];
    auto watcher_indegree = --execution_data->node_indegrees[dst_id];
    const auto &iter = node_id_to_recovery_indegree.find(dst_id);
    if ((iter == node_id_to_recovery_indegree.end()) ||
        ((iter != node_id_to_recovery_indegree.end()) && (watcher_indegree < iter->second))) {
      node_id_to_recovery_indegree[dst_id] = watcher_indegree;
      // 设置watcher的recovery indegree
      all_tasks_[dst_id]->SetRecoveryIndegree(watcher_indegree);
      GELOGD("pre node %s, set watcher node %s recovery indegree %zu ",
             reinterpret_cast<const ExtendedKernelContext *>(&node->context)->GetKernelName(),
             reinterpret_cast<const ExtendedKernelContext *>(&execution_data->base_ed.nodes[dst_id]->context)
                 ->GetKernelName(),
             watcher_indegree);
    }
    if (watcher_indegree == 0) {
      CollectStartTasks(execution_data->base_ed.nodes[dst_id], node_id_to_recovery_indegree, priority_start_task_queue);
      execution_data->node_indegrees[dst_id] = all_tasks_[dst_id]->GetRecoveryIndegree();
      GELOGD("recovery node %s indegree %zu",
             reinterpret_cast<const ExtendedKernelContext *>(&execution_data->base_ed.nodes[dst_id]->context)
                 ->GetKernelName(),
             all_tasks_[dst_id]->GetRecoveryIndegree());
    }
  }
}

void KernelTaskProducer::CollectChain(const Node *node) {
  auto execution_data = reinterpret_cast<const ExecutionData *>(execution_data_);
  auto node_id = node->node_id;
  auto &task = all_tasks_[node_id];
  task->SetTaskId(ordered_task_id_++);
  priority_task_queue_.push(task);
  GELOGD("priority q push node: %s",
         reinterpret_cast<const ExtendedKernelContext *>(&node->context)->GetKernelName());
  has_generate_task_num_++;

  if (task->IsGenWatcherAfterExecute()) {
    task->SetIsGenWatcher(false);
    return;
  }
  if (task->IsStopGenWatcher()) {
    GELOGD("node %s need stop gen watcher, skip",
           reinterpret_cast<const ExtendedKernelContext *>(&node->context)->GetKernelName());
    return;
  }

  if (has_generate_task_num_ >= kMaxGenerateTaskNum) {
    GELOGD("generated task num has %zu", has_generate_task_num_);
    return;
  }
  Watcher *watchers = execution_data->node_watchers[node->node_id];
  task->SetIsGenWatcher(true);
  for (size_t i = 0U; i < watchers->watch_num; ++i) {
    NodeIdentity dst_id = watchers->node_ids[i];
    GELOGD("pre node %s, watcher node %s indegree %zu -1",
           reinterpret_cast<const ExtendedKernelContext *>(&node->context)->GetKernelName(),
           reinterpret_cast<const ExtendedKernelContext *>(&execution_data->base_ed.nodes[dst_id]->context)
               ->GetKernelName(),
           execution_data->node_indegrees[dst_id]);
    if (--execution_data->node_indegrees[dst_id] == 0) {
      CollectChain(execution_data->base_ed.nodes[dst_id]);
      execution_data->node_indegrees[dst_id] = all_tasks_[dst_id]->GetRecoveryIndegree();
      GELOGD("recovery node %s indegree %zu",
             reinterpret_cast<const ExtendedKernelContext *>(&execution_data->base_ed.nodes[dst_id]->context)
                 ->GetKernelName(),
             all_tasks_[dst_id]->GetRecoveryIndegree());
    }
  }
}

void KernelTaskProducer::KernelDone(const Node *node, bool is_gen_watcher) {
  const auto &cur_task = all_tasks_[node->node_id];
  if (is_gen_watcher) {
    cur_task->SetIsGenWatcher(false);
    GELOGD("node %s is gen watcher, skip KernelDone",
           reinterpret_cast<const ExtendedKernelContext *>(&node->context)->GetKernelName());
    return;
  }
  auto execution_data = reinterpret_cast<const ExecutionData *>(execution_data_);
  Watcher *watchers = execution_data->node_watchers[node->node_id];
  if (cur_task->IsGenWatcherAfterExecute()) {
    // 由于SwitchNOtify没有设置Watcher task，所以他的后继节点的入度在这里减掉
    for (size_t i = 0U; i < watchers->watch_num; ++i) {
      all_tasks_[watchers->node_ids[i]]->SubIndegree();
    }
    cur_task->SetIsGenWatcher(true);
  }
  for (size_t i = 0U; i < watchers->watch_num; ++i) {
    NodeIdentity node_id = watchers->node_ids[i];
    const auto &cur_node = execution_data->base_ed.nodes[node_id];

    GELOGD("pre node %s, watcher node %s indegree %zu -1",
           reinterpret_cast<const ExtendedKernelContext *>(&node->context)->GetKernelName(),
           reinterpret_cast<const ExtendedKernelContext *>(&cur_node->context)->GetKernelName(),
           execution_data->node_indegrees[node_id]);
    if (--execution_data->node_indegrees[node_id] == 0) {
      CollectChain(cur_node);
      execution_data->node_indegrees[node_id] = all_tasks_[node_id]->GetRecoveryIndegree();
      GELOGD("recovery node %s indegree %zu",
             reinterpret_cast<const ExtendedKernelContext *>(&execution_data->base_ed.nodes[node_id]->context)
                 ->GetKernelName(),
             execution_data->node_indegrees_backup[node_id]);
    }
  }
}

ge::Status KernelTaskProducer::DoRecycleTask(ExecTask &task) {
  for (auto &iter : task.GetTaskNodeList()) {
    KernelDone(iter, task.IsGenWatcher());
  }
  return ge::SUCCESS;
}

ge::Status KernelTaskProducer::FreeAllTask(TaskPackage &package) {
  while (auto task = package.pop_front()) {
    if (task) {
      task_allocator_.Free(*task);
    }
  }
  return ge::SUCCESS;
}

void KernelTaskProducer::Dump() const {
  GEEVENT("|-- Producer [type : kernel]");
}
}  // namespace gert
