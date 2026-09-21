/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_MULTI_THREAD_EXE_GRAPH_RESOURCE_GUARD_H
#define AIR_CXX_RUNTIME_V2_MULTI_THREAD_EXE_GRAPH_RESOURCE_GUARD_H
#include "framework/runtime/exe_graph_resource_guard.h"
#include "core/executor/multi_thread_topological/executor/schedule/scheduler/task_scheduler.h"

namespace gert {
class VISIBILITY_EXPORT MultiThreadResourceGuard : public TopologicalResourceGuard {
 public:
  TaskScheduler *ResetTaskScheduler(std::unique_ptr<TaskScheduler> scheduler);

 private:
  std::unique_ptr<TaskScheduler> task_scheduler_guarder_{nullptr};
};
}

#endif  // AIR_CXX_RUNTIME_V2_MULTI_THREAD_EXE_GRAPH_RESOURCE_GUARD_H
