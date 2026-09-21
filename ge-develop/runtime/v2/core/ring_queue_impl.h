/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_CORE_RING_QUEUE_IMPL_H_
#define AIR_CXX_RUNTIME_V2_CORE_RING_QUEUE_IMPL_H_
#include "ring_queue.h"
#include "exe_graph/runtime/base_type.h"
#include "execution_data.h"
#include "core/executor_error_code.h"
#ifdef __cplusplus
extern "C" {
#endif
static int IsQueueEmpty(RingQueue *que) {
  return que->front == que->back;
}

static void *PopQueue(RingQueue *que) {
  if (IsQueueEmpty(que)) {
    return NULL;
  }
  void *ret = que->data[que->front++];
  if (que->front == que->cap) {
    que->front = 0;
  }
  return ret;
}

static KernelStatus PushQueue(RingQueue *que, void *element) {
  size_t next_back = que->back + 1;
  if (next_back == que->cap) {
    next_back = 0;
  }
  if (next_back == que->front) {
    return kStatusQueueFull;
  }
  que->data[que->back] = element;
  que->back = next_back;
  return kStatusSuccess;
}

#ifdef __cplusplus
}
#endif
#endif  // AIR_CXX_RUNTIME_V2_CORE_RING_QUEUE_IMPL_H_
