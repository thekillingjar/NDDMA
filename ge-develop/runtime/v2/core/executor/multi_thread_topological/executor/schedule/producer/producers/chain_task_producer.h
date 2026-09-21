/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_CHAIN_TASK_PRODUCER_H
#define AIR_CXX_RUNTIME_V2_CHAIN_TASK_PRODUCER_H

#include "tagged_task_producer.h"
#include "core/executor/multi_thread_topological/executor/schedule/task/task_allocator.h"
#include "core/executor/multi_thread_topological/executor/schedule/task/exec_task.h"

namespace gert {
class ChainTaskProducer : public TaggedTaskProducer {
 public:
  explicit ChainTaskProducer(size_t task_capacity);

 private:
  ge::Status Prepare(const void *execution_data) override;
  ge::Status StartUp() override;
  ge::Status DoRecycleTask(ExecTask &task) override;
  ge::Status FreeAllTask(TaskPackage &package) override;
  void Dump() const override;

 private:
  ge::Status KernelDone(const Node *node);
  void CollectChain(ExecTask &task, Node *node);
  bool AddKernel(ExecTask &task, Node *node);

 private:
  TaskAllocator task_allocator_;
  const void *execution_data_{nullptr};
};
}  // namespace gert

#endif // AIR_CXX_RUNTIME_V2_CHAIN_TASK_PRODUCER_H
