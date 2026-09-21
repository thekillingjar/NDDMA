/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_CORE_PRIORITY_QUEUE_IMPL_H_
#define AIR_CXX_RUNTIME_V2_CORE_PRIORITY_QUEUE_IMPL_H_
#include "priority_queue.h"
#include "exe_graph/runtime/base_type.h"
#include "executor_error_code.h"

#ifdef __cplusplus
extern "C" {
#endif
static int IsQueueEmpty(const PriorityQueue *que) {
  return que->size == 0U;
}

static void Swap(PriorityQueue *que, size_t pos1, size_t pos2) {
  PriorityQueueElementHead *tmp = que->data[pos1];
  que->data[pos1] = que->data[pos2];
  que->data[pos2] = tmp;
}

static PriorityQueueElementHead *PopQueue(PriorityQueue *que) {
  if (IsQueueEmpty(que)) {
    return NULL;
  }
  PriorityQueueElementHead *result = que->data[1U];
  que->data[1U] = que->data[que->size--];

  size_t element_pos = 1U;
  size_t child_pos;
  while ((child_pos = element_pos * 2U) <= que->size) {
    if ((child_pos < que->size) && (que->data[child_pos + 1U]->priority < que->data[child_pos]->priority)) {
      child_pos += 1;
    }
    if (que->data[element_pos]->priority <= que->data[child_pos]->priority) {
      break;
    }
    Swap(que, element_pos, child_pos);
    element_pos = child_pos;
  }

  return result;
}

static KernelStatus PushQueue(PriorityQueue *que, PriorityQueueElementHead *element) {
  if (que->size >= que->cap) {
    return kStatusQueueFull;
  }
  size_t element_pos = ++que->size;
  que->data[element_pos] = element;

  while (element_pos > 1) {
    size_t parent_pos = element_pos / 2U;
    if (que->data[parent_pos]->priority <= que->data[element_pos]->priority) {
      break;
    }
    Swap(que, element_pos, parent_pos);
    element_pos = parent_pos;
  }

  return kStatusSuccess;
}
#ifdef __cplusplus
}
#endif
#endif  // AIR_CXX_RUNTIME_V2_CORE_PRIORITY_QUEUE_IMPL_H_
