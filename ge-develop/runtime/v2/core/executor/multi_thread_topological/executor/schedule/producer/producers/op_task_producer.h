/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_TASK_PRODUCER_H
#define AIR_CXX_RUNTIME_V2_OP_TASK_PRODUCER_H

#include <memory>
#include "tagged_task_producer.h"
#include "core/executor/multi_thread_topological/executor/schedule/task/task_allocator.h"
#include "core/executor/multi_thread_topological/executor/schedule/task/exec_task.h"

namespace gert {
class OpTaskProducer : public TaggedTaskProducer {
 public:
  explicit OpTaskProducer(size_t task_capacity);

 private:
  ge::Status Prepare(const void *execution_data) override;
  ge::Status StartUp() override;
  ge::Status DoRecycleTask(ExecTask &exec_task) override;
  void Dump() const override;

 private:
  ge::Status Init();
  ge::Status UpdateInDegrees(Node *node);
  ge::Status LinkTask(uint32_t src_kernel_id, uint32_t dst_kernel_id);
  uint32_t GetTaskId(const Node *node) const;
  void TravelNodes(Node *node, std::map<std::string, uint32_t> &op_to_task_id);
  void UpdateExecTask(Node *node, std::map<std::string, uint32_t> &op_to_task_id);
  void InsertExeTask(Node *node, ExecTaskType type);
  void AppendExeTask(Node *prev, Node *current);
  bool HasLoopBetweenTasks(uint32_t src_task_id, uint32_t dst_task_id);
  bool HasLoopBetweenNodeAndTask(Node *node, uint32_t task_id);
  bool HasLoopBetweenNodes(const Node *src_node, const Node *dst_node);

 private:
  struct TaskWatcher {
    uint32_t task_id_;
    uint32_t next_;
  };

 private:
  const ExecutionData *execution_data_{nullptr};
  std::vector<uint32_t> kernel_id_to_task_id_;
  std::vector<ExecTask> all_tasks_;
  std::vector<uint32_t> in_degrees_;
  std::vector<uint32_t> wait_in_degrees_;
  std::vector<TaskWatcher> watchers_;
};
}  // namespace gert

#endif // AIR_CXX_RUNTIME_V2_OP_TASK_PRODUCER_H
