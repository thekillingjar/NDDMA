/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_CORE_2P1_EXECUTOR_INNER_H_
#define AIR_CXX_RUNTIME_V2_CORE_2P1_EXECUTOR_INNER_H_

#include <stdlib.h>
#include "ring_queue.h"
#include "exe_graph/runtime/kernel_run_context.h"
#include "executor/executor_base_def.h"
#include "executor/topological/execution_data/topological_execution_data.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef TopologicalExecutionData ExecutionData;
#ifdef __cplusplus
}
#endif
#endif  // AIR_CXX_RUNTIME_V2_CORE_2P1_EXECUTOR_INNER_H_
