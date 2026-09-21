/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef F3F33961A4FB4F42A4237A1B6423A442_H
#define F3F33961A4FB4F42A4237A1B6423A442_H

#include "graph/compute_graph.h"
#include "proto/task.pb.h"

namespace gert {
class TaskDefFaker {
 public:
  enum TaskType {
    kWithHandle,     // ModelTaskType::MODEL_TASK_ALL_KERNEL
    kWithoutHandle,  // ModelTaskType::MODEL_TASK_KERNEL
    kTfAicpu,        // ModelTaskType::MODEL_TASK_KERNEL_EX
    kCCAicpu,        // ModelTaskType::MODEL_TASK_KERNEL
    kDvppTask,       // ModelTaskType::MODEL_TASK_KERNEL
    kDsaRandomNormal, // ModelTaskType::MODEL_TASK_DSA_TASK
    kHccl, // ModelTaskType::MODEL_TASK_HCCL
    kRts, // ModelTaskType::MODEL_TASK_MEMCPY_ASYNC
    kLabelSwitch, // ModelTaskType::MODEL_TASK_STREAM_LABEL_SWITCH_BY_INDEX
    kEvent, // ModelTaskType::MODEL_TASK_EVENT_RECORD
    kTaskTypeEnd
  };

  enum KernelType {
    kTE_AiCore,
    kTF_AiCpu,
    kCC_AiCpu,
    kCustAiCpu,
    kDvppKernel,
    kDsa_RandomNormal,
    kHccl_Kernel,
    kKernelTypeEnd,
  };

  struct TaskTypeInfo {
    TaskType task_type;
    KernelType kernel_type;
    uint64_t bin_data;
  };

  TaskDefFaker& AddTask(const TaskTypeInfo& taskInfo);

 public:
  virtual std::vector<domi::TaskDef> CreateTaskDef(uint64_t op_index = 0);
  virtual ~TaskDefFaker() = default;
  virtual std::unique_ptr<TaskDefFaker> Clone() const;

 private:
  std::vector<TaskTypeInfo> task_infos;
};
}  // namespace gert

#endif
