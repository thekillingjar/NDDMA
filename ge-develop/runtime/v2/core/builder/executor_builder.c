/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "executor_builder.h"
#include <malloc.h>
#include <string.h>
#include <securec.h>
#include "core/execution_data.h"

#ifdef __cplusplus
extern "C" {
#endif
RingQueue *CreateRingQueue(size_t cap) {
  cap += 1U;
  RingQueue *que = (RingQueue *) malloc(sizeof(RingQueue) + sizeof(void *) * cap);
  if (que == NULL) {
    return NULL;
  }
  que->cap = cap;
  que->front = que->back = 0U;
  if (memset_s(que->data, sizeof(void *) * cap, 0, sizeof(void *) * cap) != EOK) {
    free(que);
    return NULL;
  }
  return que;
}

PriorityQueue *CreatePriorityQueue(size_t capacity) {
  PriorityQueue *que = (PriorityQueue *)malloc(sizeof(PriorityQueueElementHead *) * capacity + sizeof(PriorityQueue));
  if (que != NULL) {
    que->cap = capacity;
    que->size = 0U;
  }
  return que;
}

AsyncAnyValue **GetInputs(Node *node) {
  return &(node->context.values[0]);
}

AsyncAnyValue **GetOutputs(Node *node) {
  return &(node->context.values[node->context.input_size]);
}

Node *CreateNode(NodeIdentity node_id, size_t input_count, AsyncAnyValue **inputs, size_t output_count,
                 AsyncAnyValue **outputs) {
  size_t size = sizeof(Node) + sizeof(AsyncAnyValue *) * (input_count + output_count);
  Node *node = (Node *) malloc(size);
  if (node != NULL) {
    if (memset_s(node, size, 0, size) != EOK) {
      free(node);
      return NULL;
    }
    node->node_id = node_id;
    node->context.input_size = input_count;
    node->context.output_size = output_count;
    size_t input_size = sizeof(AsyncAnyValue *) * input_count;
    size_t output_size = sizeof(AsyncAnyValue *) * output_count;
    if ((input_size > 0U) && (memcpy_s(GetInputs(node), input_size + output_size, inputs, input_size) != EOK)) {
      free(node);
      return NULL;
    }
    if ((output_size > 0U) && (memcpy_s(GetOutputs(node), output_size, outputs, output_size) != EOK)) {
      free(node);
      return NULL;
    }
  }
  return node;
}

Watcher *CreateWatch(size_t count, NodeIdentity *node_ids) {
  size_t size = sizeof(Watcher) + sizeof(Node *) * count;
  Watcher *watcher = (Watcher *) malloc(size);
  if (watcher != NULL) {
    watcher->watch_num = count;
    if (count > 0U && memcpy_s(watcher->node_ids, sizeof(Node *) * count, node_ids, sizeof(Node *) * count) != EOK) {
      free(watcher);
      return NULL;
    }
  }
  return watcher;
}
#ifdef __cplusplus
}
#endif