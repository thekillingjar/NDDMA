/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "multi_thread_topological_executor.h"
#include "core/executor/multi_thread_topological/execution_data/multi_thread_execution_data.h"
#include "common/checker.h"

KernelStatus MultiThreadTopologicalExecute(void *arg) {
  MultiThreadExecutionData *execution_data = (MultiThreadExecutionData *)arg;
  auto scheduler = execution_data->scheduler;
  GE_ASSERT_NOTNULL(scheduler);
  return scheduler->Schedule();
}

KernelStatus MultiThreadTopologicalExecuteWithCallback(int sub_graph_type, void *arg, ExecutorSubscriber *es) {
  MultiThreadExecutionData *execution_data = (MultiThreadExecutionData *)arg;
  auto scheduler = execution_data->scheduler;
  GE_ASSERT_NOTNULL(scheduler);
  return scheduler->Schedule(sub_graph_type, es);
}