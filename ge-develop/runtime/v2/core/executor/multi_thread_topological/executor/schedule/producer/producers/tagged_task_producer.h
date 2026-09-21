/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_TAGGED_TASK_PRODUCER_H
#define AIR_CXX_RUNTIME_V2_TAGGED_TASK_PRODUCER_H

#include <queue>
#include "core/executor/multi_thread_topological/executor/schedule/producer/task_producer.h"
#include "core/executor/multi_thread_topological/executor/schedule/producer/producers/kernel_tags/kernel_tags.h"
#include "core/executor/multi_thread_topological/executor/schedule/task/exec_task.h"
#include "core/executor/multi_thread_topological/executor/schedule/task/task_allocator.h"
#include "exe_graph/runtime/extended_kernel_context.h"
#include "common/checker.h"

namespace gert {
class TaggedTaskProducer : public TaskProducer {
 public:
  TaskPackage Produce() override {
    return std::move(tasks_);
  }

  ge::Status Recycle(TaskPackage &package) override {
    while (auto task = package.pop_front()) {
      if (task) {
        auto ret = DoRecycleTask(*task);
        if (ret != ge::SUCCESS) {
          GE_ASSERT_SUCCESS(FreeAllTask(package));
          return ret;
        }
      }
    }
    return ge::SUCCESS;
  }

  ge::Status EndUp() override {
    return ge::SUCCESS;
  }

 private:
  virtual ge::Status DoRecycleTask(ExecTask &) = 0;

 protected:
  TaskPackage tasks_;
  KernelTags tags_;
};
}  // namespace gert
#endif // AIR_CXX_RUNTIME_V2_TAGGED_TASK_PRODUCER_H
