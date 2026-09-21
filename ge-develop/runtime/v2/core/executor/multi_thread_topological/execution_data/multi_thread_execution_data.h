/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_MULTI_THREAD_EXECUTION_DATA_H
#define AIR_CXX_RUNTIME_V2_MULTI_THREAD_EXECUTION_DATA_H
#include "core/executor/topological/execution_data/topological_execution_data.h"
#include "core/executor/multi_thread_topological/executor/schedule/scheduler/task_scheduler.h"

namespace {
struct MultiThreadExecutionData {
  TopologicalExecutionData topo_ed;
  gert::TaskScheduler *scheduler;
};
}  // namespace

#endif  // AIR_CXX_RUNTIME_V2_MULTI_THREAD_EXECUTION_DATA_H
