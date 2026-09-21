/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_EXEC_TASK_H
#define AIR_CXX_RUNTIME_V2_EXEC_TASK_H

#include <vector>
#include <limits>
#include <atomic>
#include "exec_task_type.h"
#include "ge/ge_api_types.h"
#include "runtime/subscriber/executor_subscriber_c.h"
#include "exe_graph/runtime/extended_kernel_context.h"
#include "kernel/memory/util/link_node.h"
#include "core/execution_data.h"
#include "framework/common/debug/ge_log.h"

namespace gert {
class ExecTask : public LinkNode<ExecTask> {
 public:
  ExecTask() = default;

  explicit ExecTask(ExecTaskType type) : type_(type) {
    task_node_list_.reserve(10U);
  }

  ExecTask(size_t task_id, ExecTaskType type, Node *kernel);

  ExecTask(const ExecTask &other) : indegree_(other.indegree_.load()) {}
  ExecTask &operator=(const ExecTask &other) {
    if (this != &other) {
      indegree_.store(other.indegree_.load());
    }
    return *this;
  }
  ExecTask(ExecTask &&other) : indegree_(other.indegree_.load()) {}
  ExecTask &operator=(ExecTask &&other) {
    if (this != &other) {
      indegree_.store(other.indegree_.load());
    }
    return *this;
  }

  size_t GetTaskId() const {
    return task_id_;
  }
  void AddKernel(Node *kernel);
  const std::vector<Node *> &GetTaskNodeList() const {
    return task_node_list_;
  }
  ExecTaskType GetType() const {
    return type_;
  }
  long long GetPriority() const {
    return priority_;
  }
  void SetType(ExecTaskType type) {
    this->type_ = type;
  }
  void SetTaskId(size_t id) {
    task_id_ = id;
  }
  void SetIsGenWatcher(bool flag);
  bool IsGenWatcher() const {
    return is_gen_watcher_;
  }

  void SetIsGenWatcherBackup(bool flag);
  void Dump() const;

  ge::graphStatus GetKernelRet() const;

  ge::Status Execute();
  ge::Status Execute(int sub_graph_type, ExecutorSubscriber *es);
  ~ExecTask();

  void AddWatcher(ExecTask *task) {
    watcher_tasks_.emplace_back(task);
  }

  void SetIndegree(int64_t indegree) {
    indegree_backup_ = indegree;
    indegree_.store(indegree, std::memory_order_release);
  }
  void SubIndegree();
  bool IsReady();
  void ReInit() {
    is_gen_watcher_ = false;
    indegree_.store(indegree_backup_);
  }

  void StartTaskReInit() {
    is_gen_watcher_ = is_gen_watcher_backup_;
  }

  void SetStopGenWatcher(bool flag) {
    stop_gen_watcher_ = flag;
  }

  bool IsStopGenWatcher() const {
    return stop_gen_watcher_;
  }

  void SetRecoveryIndegree(int64_t indegree) {
    recovery_indegree_ = indegree;
  }

  int64_t GetRecoveryIndegree() const {
    return recovery_indegree_;
  }
  void SetForceQuit(bool *force_quit) {
    force_quit_ = force_quit;
  }
  void SetGenWatcherAfterExecute() {
    is_gen_watcher_after_execute_ = true;
  }

  bool IsGenWatcherAfterExecute() const {
    return is_gen_watcher_after_execute_;
  }

 private:
  size_t task_id_{0U};
  ExecTaskType type_{ExecTaskType::NORMAL};
  ge::graphStatus kernel_ret_{ge::GRAPH_FAILED};
  std::vector<Node *> task_node_list_;
  std::vector<ExecTask *> watcher_tasks_;
  // 实际执行时的入度
  std::atomic<int64_t> indegree_{0U};
  // 判断是否退出
  bool *force_quit_{nullptr};
  // 执行结束用来恢复的入度
  int64_t indegree_backup_{0U};
  // 主线程调度时用来恢复的入度
  int64_t recovery_indegree_{0U};
  long long priority_{0U};
  // 记录是否遍历过后继watcher， 用于start_task
  bool is_gen_watcher_{false};
  bool is_gen_watcher_backup_{false};
  // 标记 不遍历后继watcher
  bool stop_gen_watcher_{false};
  bool is_gen_watcher_after_execute_{false};
};
}  // namespace gert

#endif // AIR_CXX_RUNTIME_V2_EXEC_TASK_H
