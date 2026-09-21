/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_CORE_EXECUTOR_TOPOLOGICAL_EXECUTION_DATA_H_
#define AIR_CXX_RUNTIME_V2_CORE_EXECUTOR_TOPOLOGICAL_EXECUTION_DATA_H_
#include <stdlib.h>
#include <stdint.h>
#include "core/executor/executor_base_def.h"
#include "core/executor/sequential/execution_data/sequential_execution_data.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
  size_t watch_num;
  NodeIdentity node_ids[0];
} Watcher;
typedef struct {
  SequentialExecutionData base_ed;
  void *ready_queue;

  size_t start_num;
  Node **start_nodes;

  // the num of watchers and indegrees are equal to `base_ed.node_num`
  Watcher **node_watchers;
  int64_t *node_indegrees;
  int64_t *node_indegrees_backup;
} TopologicalExecutionData;
#ifdef __cplusplus
}
#endif
#endif  // AIR_CXX_RUNTIME_V2_CORE_EXECUTOR_TOPOLOGICAL_EXECUTION_DATA_H_
