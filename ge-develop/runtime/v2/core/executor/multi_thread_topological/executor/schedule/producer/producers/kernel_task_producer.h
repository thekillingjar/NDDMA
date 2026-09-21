/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef AIR_CXX_RUNTIME_V2_KERNEL_TASK_PRODUCER_H
#define AIR_CXX_RUNTIME_V2_KERNEL_TASK_PRODUCER_H

#include <queue>
#include "tagged_task_producer.h"
#include "core/executor/multi_thread_topological/executor/schedule/task/task_package.h"
#include "core/executor/multi_thread_topological/executor/schedule/task/exec_task.h"
#include "core/priority_queue.h"

namespace gert {
class KernelTaskProducer : public TaggedTaskProducer {
 public:
  explicit KernelTaskProducer(size_t task_capacity, size_t thread_num);
  ~KernelTaskProducer() noexcept override;
  TaskPackage Produce() override;
  ge::Status Recycle(TaskPackage &package) override;
  ge::Status EndUp() override;

 public:
  ge::Status Prepare(const void *execution_data) override;
  ge::Status StartUp() override;
  ge::Status DoRecycleTask(ExecTask &task) override;
  ge::Status FreeAllTask(TaskPackage &package) override;
  void Dump() const override;

 private:
  struct ExecTaskCompare{
    bool operator() (const ExecTask *task1, const ExecTask *task2) const {
      if (task1->GetPriority() == task2->GetPriority()) {
        return task1->GetTaskId() > task2->GetTaskId();
      }
      return task1->GetPriority() > task2->GetPriority();
    }
  };

 private:
  void KernelDone(const Node *node, bool is_gen_watcher);
  bool AddKernel(ExecTask &task, Node *node);
  void CollectChain(const Node *node);
  void CollectStartTasks(
      const Node *node, std::map<size_t, int64_t> &node_id_to_recovery_indegree,
      std::priority_queue<ExecTask *, std::vector<ExecTask *>, ExecTaskCompare> &priority_start_task_queue);
  void RecoverNodeInDegrees(const ExecutionData *execution_data);
  ge::Status ReInit();
  void RecoveryPriorityQueue();

  std::priority_queue<ExecTask *, std::vector<ExecTask*>, ExecTaskCompare> priority_task_queue_;
  TaskAllocator task_allocator_;
  const void *execution_data_{nullptr};
  std::priority_queue<ExecTask *, std::vector<ExecTask*>, ExecTaskCompare> produced_tasks_;

  size_t thread_num_{3U};

  size_t has_generate_task_num_{0U};
  std::vector<ExecTask *> all_tasks_;

  // 记录第一个SwtichNotify之前的任务
  std::vector<ExecTask *> start_tasks_;

  // 记录下发的顺序
  size_t ordered_task_id_{0U};

  // 用来记录最新的SwitchNotify task, 只有当其真正执行完才会继续Produce
  ExecTask *latest_target_task_{nullptr};
};
}  // namespace gert

#endif // AIR_CXX_RUNTIME_V2_KERNEL_TASK_PRODUCER_H
