/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_TASK_QUEUE_H
#define AIR_CXX_RUNTIME_V2_TASK_QUEUE_H

#include <cstddef>
#include "core/executor/multi_thread_topological/executor/schedule/task/exec_task.h"

namespace gert {
class TaskQueue {
 public:
  virtual bool Push(ExecTask &task) = 0;
  virtual ExecTask *Pop() = 0;
  virtual bool IsEmpty() const = 0;
  virtual bool IsFull() const = 0;
  virtual size_t GetSize() const = 0;
  virtual size_t GetCapacity() const = 0;
  virtual ~TaskQueue() = default;
};
}  // namespace gert

#endif // AIR_CXX_RUNTIME_V2_TASK_QUEUE_H
