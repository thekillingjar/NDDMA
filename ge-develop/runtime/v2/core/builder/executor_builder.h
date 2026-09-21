/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_CORE_2P1_EXECUTOR_BUILDER_H_
#define AIR_CXX_RUNTIME_V2_CORE_2P1_EXECUTOR_BUILDER_H_
#include "core/execution_data.h"
#include "core/ring_queue.h"
#include "core/priority_queue.h"
#include "core/executor/topological/execution_data/topological_execution_data.h"
#ifdef __cplusplus
extern "C" {
#endif

RingQueue *CreateRingQueue(size_t cap);
PriorityQueue *CreatePriorityQueue(size_t cap);
Node *CreateNode(NodeIdentity node_id, size_t input_count, AsyncAnyValue **inputs, size_t output_count,
                 AsyncAnyValue **outputs);
Watcher *CreateWatch(size_t count, NodeIdentity *node_ids) ;

#ifdef __cplusplus
}
#endif
#endif  // AIR_CXX_RUNTIME_V2_CORE_2P1_EXECUTOR_BUILDER_H_
