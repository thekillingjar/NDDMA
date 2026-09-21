/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_DEFAULT_CONFIG_H
#define AIR_CXX_RUNTIME_V2_DEFAULT_CONFIG_H

#include "core/executor/multi_thread_topological/executor/schedule/producer/task_producer_type.h"
#include "core/executor/multi_thread_topological/executor/schedule/task/exec_task_type.h"
#include "core/executor/multi_thread_topological/executor/schedule/worker/task_thread_mode.h"

#include <string>

namespace gert {
// default config for task producer:
constexpr TaskProducerType kTaskProducerTypeDefault = TaskProducerType::KERNEL;
constexpr size_t kTaskProducerPreparedTaskDefault = 1024;

// default config for task worker:
constexpr const char *kTaskWorkerNameDefault = "task_worker";
constexpr ExecTaskType kTaskWorkerBindToTypeDefault = ExecTaskType::NORMAL;
constexpr TaskThreadMode kTaskThreadModeDefault = TaskThreadMode::MODERATE;
constexpr size_t kTaskWorkerThreadCountDefault = 1;
constexpr size_t kPendingTaskQueueSizeLog2Default = 20;    // 2 ^ 20 = 1048576
constexpr size_t kCompletedTaskQueueSizeLog2Default = 20;  // 2 ^ 20 = 1048576
}  // namespace gert

#endif // AIR_CXX_RUNTIME_V2_DEFAULT_CONFIG_H