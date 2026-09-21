/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_TASK_WORKER_CONFIG_H
#define AIR_CXX_RUNTIME_V2_TASK_WORKER_CONFIG_H

#include "default_config.h"

namespace gert {
struct TaskWorkerConfig {
  std::string name{kTaskWorkerNameDefault};
  ExecTaskType bind_task_type{kTaskWorkerBindToTypeDefault};
  TaskThreadMode thread_mode{kTaskThreadModeDefault};
  size_t thread_count{kTaskWorkerThreadCountDefault};
  size_t pending_queue_size_log2{kPendingTaskQueueSizeLog2Default};
  size_t completed_queue_size_log2{kCompletedTaskQueueSizeLog2Default};
};
}  // namespace gert

#endif // AIR_CXX_RUNTIME_V2_TASK_WORKER_CONFIG_H