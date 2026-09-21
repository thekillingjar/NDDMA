/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "op_task_producer.h"
#include "common/checker.h"
#include "exe_graph/runtime/extended_kernel_context.h"
#include "core/execution_data.h"

namespace gert {
constexpr uint32_t kInvalidTaskId = 100000U;
constexpr uint32_t kInvalidWatcherPtr = 20000U;

OpTaskProducer::OpTaskProducer(size_t task_capacity) {
  (void)task_capacity;
}

ge::Status OpTaskProducer::Prepare(const void *execution_data) {
  GE_ASSERT_TRUE(execution_data != nullptr);
  execution_data_ = reinterpret_cast<const ExecutionData *>(execution_data);
  GE_ASSERT_SUCCESS(Init());
  return ge::SUCCESS;
}

ge::Status OpTaskProducer::Init() {
  tags_.Reset(execution_data_->base_ed.node_num);
  kernel_id_to_task_id_.resize(execution_data_->base_ed.node_num + 1, kInvalidWatcherPtr);

  // todo task size is large
  all_tasks_.reserve(execution_data_->base_ed.node_num);

  in_degrees_.reserve(execution_data_->base_ed.node_num);
  watchers_.reserve(execution_data_->base_ed.node_num);
  wait_in_degrees_.reserve(execution_data_->base_ed.node_num);
  std::map<std::string, uint32_t> op_to_task_id;
  for (size_t i = 0U; i < execution_data_->start_num; ++i) {
    TravelNodes(execution_data_->start_nodes[i], op_to_task_id);
  }
  for (size_t i = 0U; i < execution_data_->start_num; ++i) {
    GE_ASSERT_SUCCESS(UpdateInDegrees(execution_data_->start_nodes[i]));
  }
  return ge::SUCCESS;
}

namespace {
bool OpNameEqual(const Node *left, const Node *right) {
  auto extend_context_left = reinterpret_cast<const ExtendedKernelContext *>(&left->context);
  auto extend_context_right = reinterpret_cast<const ExtendedKernelContext *>(&right->context);
  if (extend_context_left->GetNodeName() != nullptr && extend_context_right->GetNodeName()!= nullptr) {
    return strcmp(extend_context_left->GetNodeName(), extend_context_right->GetNodeName()) == 0;
  }
  return false;
}

std::string ToOriOpName(const char *op_name) {
  std::string ori_op_name = op_name;
  auto pos = ori_op_name.find("_AtomicClean");
  if (pos != ori_op_name.npos) {
    return ori_op_name.substr(0, pos);
  }
  return ori_op_name;
}
}  // namespace

void OpTaskProducer::TravelNodes(Node *node, std::map<std::string, uint32_t> &op_to_task_id) {
  UpdateExecTask(node, op_to_task_id);
  Watcher *watchers = execution_data_->node_watchers[node->node_id];
  for (size_t i = 0U; i < watchers->watch_num; ++i) {
    NodeIdentity node_id = watchers->node_ids[i];
    Node *dst_node = execution_data_->base_ed.nodes[node_id];
    if (--execution_data_->node_indegrees[node_id] == 0) {
      if (execution_data_->node_indegrees_backup[node_id] == 1 && OpNameEqual(node, dst_node)) {
        AppendExeTask(node, dst_node);
      }
      TravelNodes(dst_node, op_to_task_id);
      execution_data_->node_indegrees[node_id] = execution_data_->node_indegrees_backup[node_id];
    }
  }
}

void OpTaskProducer::UpdateExecTask(Node *node, std::map<std::string, uint32_t> &op_to_task_id) {
  if (kernel_id_to_task_id_[node->node_id] != kInvalidWatcherPtr) {
    return;
  }
  ExecTaskType type = tags_.GetByNode(node);
  if (type == ExecTaskType::MEMORY) {
    InsertExeTask(node, type);
    return;
  }
  auto extend_context_node = reinterpret_cast<const ExtendedKernelContext *>(&node->context);
  if (extend_context_node->GetNodeName() == nullptr) {
    InsertExeTask(node, ExecTaskType::NORMAL);
    return;
  }

  auto ori_op_name = ToOriOpName(extend_context_node->GetNodeName());
  auto iter = op_to_task_id.find(ori_op_name);
  if (iter == op_to_task_id.end()) {
    InsertExeTask(node, ExecTaskType::NORMAL);
    op_to_task_id.emplace(ori_op_name, all_tasks_.size() - 1);
  } else {
    // add to existed op's task
    auto task_id = iter->second;
    // check does node rely on the task's node's watcher_nodes which is in other task
    if (HasLoopBetweenNodeAndTask(node, task_id)) {
      InsertExeTask(node, ExecTaskType::NORMAL);
      op_to_task_id[ori_op_name] = all_tasks_.size() - 1;
      return;
    }
    all_tasks_[task_id].AddKernel(node);
    kernel_id_to_task_id_[node->node_id] = task_id;
    GELOGD("add to existed op's task, task id: %u, task type: %u, kernel name: %s, kernel type: %s", task_id,
           static_cast<size_t>(type), reinterpret_cast<const ExtendedKernelContext *>(&node->context)->GetKernelName(),
           reinterpret_cast<const ExtendedKernelContext *>(&node->context)->GetKernelType());
  }
}

void OpTaskProducer::Dump() const {
  GEEVENT("|-- Producer [type : op]");
  GEEVENT("    |-- task size %ld, watcher size %ld", all_tasks_.size(), watchers_.size());
  for (auto &task : all_tasks_) {
    auto watcher_addr = task.GetTaskId();
    while (watchers_[watcher_addr].task_id_ != kInvalidTaskId) {
      if (watchers_[watcher_addr].next_ == kInvalidWatcherPtr) {
        break;
      }
      watcher_addr = watchers_[watcher_addr].next_;
    }
    task.Dump();
  }
}

void OpTaskProducer::InsertExeTask(Node *node, ExecTaskType type) {
  auto task_id = all_tasks_.size();
  all_tasks_.emplace_back(task_id, type, node);
  watchers_.push_back({kInvalidTaskId, kInvalidWatcherPtr});
  in_degrees_.emplace_back(0U);
  wait_in_degrees_.emplace_back(0U);
  kernel_id_to_task_id_[node->node_id] = all_tasks_.size() - 1;
  GELOGD("new task, task id: %u, task type: %u, kernel name: %s, kernel type: %s", task_id, static_cast<size_t>(type),
         reinterpret_cast<const ExtendedKernelContext *>(&node->context)->GetKernelName(),
         reinterpret_cast<const ExtendedKernelContext *>(&node->context)->GetKernelType());
}

void OpTaskProducer::AppendExeTask(Node *prev, Node *current) {
  auto task_id = GetTaskId(prev);
  all_tasks_[task_id].AddKernel(current);
  kernel_id_to_task_id_[current->node_id] = task_id;
  GELOGD("add to existed op's task, task id: %u, task type: %u, kernel name: %s, kernel type: %s", task_id,
         static_cast<size_t>(all_tasks_[task_id].GetType()),
         reinterpret_cast<const ExtendedKernelContext *>(&current->context)->GetKernelName(),
         reinterpret_cast<const ExtendedKernelContext *>(&current->context)->GetKernelType());
}

uint32_t OpTaskProducer::GetTaskId(const Node *node) const {
  return kernel_id_to_task_id_[node->node_id];
}

ge::Status OpTaskProducer::UpdateInDegrees(Node *node) {
  Watcher *watchers = execution_data_->node_watchers[node->node_id];
  for (size_t i = 0U; i < watchers->watch_num; ++i) {
    NodeIdentity dst_id = watchers->node_ids[i];
    GE_ASSERT_SUCCESS(LinkTask(node->node_id, dst_id));
    GE_ASSERT_SUCCESS(UpdateInDegrees(execution_data_->base_ed.nodes[dst_id]));
  }
  return ge::SUCCESS;
}

bool OpTaskProducer::HasLoopBetweenNodeAndTask(Node *node, uint32_t task_id) {
  for (auto &task_node : all_tasks_[task_id].GetTaskNodeList()) {
    Watcher *watchers = execution_data_->node_watchers[task_node->node_id];
    for (size_t i = 0U; i < watchers->watch_num; ++i) {
      NodeIdentity node_id = watchers->node_ids[i];
      Node *dst_node = execution_data_->base_ed.nodes[node_id];
      if (kernel_id_to_task_id_[node_id] != kInvalidWatcherPtr && task_id != kernel_id_to_task_id_[node_id] &&
          HasLoopBetweenNodes(dst_node, node)) {
        return true;
      }
    }
  }
  return false;
}

bool OpTaskProducer::HasLoopBetweenNodes(const Node *src_node, const Node *dst_node) {
  if (src_node->node_id == dst_node->node_id) {
    return true;
  }
  Watcher *watchers = execution_data_->node_watchers[src_node->node_id];
  for (size_t i = 0U; i < watchers->watch_num; ++i) {
    NodeIdentity node_id = watchers->node_ids[i];
    Node *watcher_node = execution_data_->base_ed.nodes[node_id];
    if (execution_data_->node_indegrees[node_id] == 0) {
      return HasLoopBetweenNodes(watcher_node, dst_node);
    }
  }
  return false;
}

bool OpTaskProducer::HasLoopBetweenTasks(uint32_t src_task_id, uint32_t dst_task_id) {
  if (watchers_[dst_task_id].task_id_ == src_task_id) {
    return true;
  }
  if (watchers_[dst_task_id].task_id_ == kInvalidTaskId || watchers_[dst_task_id].next_ == kInvalidWatcherPtr) {
    return false;
  }
  return HasLoopBetweenTasks(src_task_id, watchers_[dst_task_id].task_id_) ||
         HasLoopBetweenTasks(src_task_id, watchers_[dst_task_id].next_);
}

ge::Status OpTaskProducer::LinkTask(uint32_t src_kernel_id, uint32_t dst_kernel_id) {
  auto src_task_id = kernel_id_to_task_id_[src_kernel_id];
  auto dst_task_id = kernel_id_to_task_id_[dst_kernel_id];
  if (src_task_id == dst_task_id) {
    return ge::SUCCESS;
  }
  GELOGD("src kernel: %s, dst kernel: %s, src task: %u, dst task: %u",
         reinterpret_cast<ExtendedKernelContext *>(&execution_data_->base_ed.nodes[src_kernel_id]->context)
             ->GetKernelName(),
         reinterpret_cast<ExtendedKernelContext *>(&execution_data_->base_ed.nodes[dst_kernel_id]->context)
             ->GetKernelName(),
         src_task_id, dst_task_id);
  // todo: 生成task的时候已经考虑了是否成环，这里再校验一次是否成环属于冗余操作，可以考虑删除
  GE_ASSERT_TRUE(!HasLoopBetweenTasks(src_task_id, dst_task_id));
  auto watcher_addr = src_task_id;
  while (watchers_[watcher_addr].task_id_ != kInvalidTaskId) {
    if (watchers_[watcher_addr].task_id_ == dst_task_id) {
      return ge::SUCCESS;
    }
    if (watchers_[watcher_addr].next_ == kInvalidWatcherPtr) {
      break;
    }
    watcher_addr = watchers_[watcher_addr].next_;
  }
  watchers_[watcher_addr].task_id_ = dst_task_id;
  watchers_[watcher_addr].next_ = watchers_.size();
  in_degrees_[dst_task_id]++;
  wait_in_degrees_[dst_task_id]++;
  watchers_.push_back({kInvalidTaskId, kInvalidWatcherPtr});
  return ge::SUCCESS;
}

ge::Status OpTaskProducer::StartUp() {
  for (size_t i = 0; i < execution_data_->start_num; ++i) {
    auto task_id = GetTaskId(execution_data_->start_nodes[i]);
    if (in_degrees_[task_id] == 0) {
      tasks_.push_back(all_tasks_[task_id]);
    }
  }
  return ge::SUCCESS;
}

ge::Status OpTaskProducer::DoRecycleTask(ExecTask &exec_task) {
  auto watch_addr = exec_task.GetTaskId();
  while (watchers_[watch_addr].task_id_ != kInvalidTaskId) {
    auto dst_task_id = watchers_[watch_addr].task_id_;
    if (--in_degrees_[dst_task_id] == 0) {
      tasks_.push_back(all_tasks_[dst_task_id]);
      in_degrees_[dst_task_id] = wait_in_degrees_[dst_task_id];
    }
    watch_addr = watchers_[watch_addr].next_;
  }
  return ge::SUCCESS;
}
}  // namespace gert
