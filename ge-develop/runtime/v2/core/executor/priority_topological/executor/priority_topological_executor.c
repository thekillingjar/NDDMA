/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "priority_topological_executor.h"
#include "core/executor/topological/execution_data/topological_execution_data.h"
#include "core/priority_queue_impl.h"
#include "core/executor_error_code.h"
#ifdef __cplusplus
extern "C" {
#endif
static size_t Clear(TopologicalExecutionData *execution_data) {
  size_t pop_count = 0U;
  while (PopQueue(execution_data->ready_queue)) {
    ++pop_count;
  }
  return pop_count;
}

static void RecoverNodeInDegrees(TopologicalExecutionData *execution_data) {
  size_t node_num = execution_data->base_ed.node_num;
  for (size_t index = 0U; index < node_num; ++index) {
    execution_data->node_indegrees[index] = execution_data->node_indegrees_backup[index];
  }
}

KernelStatus PriorityTopologicalExecute(void *arg) {
  TopologicalExecutionData *execution_data = (TopologicalExecutionData *)arg;
  // todo warning if the ready queue is not empty
  Clear(execution_data);
  for (size_t i = 0; i < execution_data->start_num; ++i) {
    PushQueue(execution_data->ready_queue, (PriorityQueueElementHead *)(execution_data->start_nodes[i]));
  }

  Node *node;
  while ((node = (Node *) (PopQueue(execution_data->ready_queue))) != NULL) {
    KernelStatus ret = node->func(&node->context);
    if (ret != 0) {
      RecoverNodeInDegrees(execution_data);
      return ret;
    }
    Watcher *watchers = execution_data->node_watchers[node->node_id];
    for (size_t i = 0; i < watchers->watch_num; ++i) {
      NodeIdentity node_id = watchers->node_ids[i];
      if (--execution_data->node_indegrees[node_id] == 0) {
        PushQueue(execution_data->ready_queue, (PriorityQueueElementHead *)(execution_data->base_ed.nodes[node_id]));
        execution_data->node_indegrees[node_id] = execution_data->node_indegrees_backup[node_id];
      }
    }
  }
  return kStatusSuccess;
}

KernelStatus PriorityTopologicalExecuteWithCallback(int sub_graph_type, void *arg, ExecutorSubscriber *es) {
  TopologicalExecutionData *execution_data = (TopologicalExecutionData *)arg;
  if ((es == NULL) || (es->callback == NULL)) {
    return kStatusFailed;
  }
  // todo warning if the ready queue is not empty
  Clear(execution_data);
  for (size_t i = 0U; i < execution_data->start_num; ++i) {
    PushQueue(execution_data->ready_queue, (PriorityQueueElementHead *)(execution_data->start_nodes[i]));
  }

  Node *node;
  es->callback(sub_graph_type, es->arg, kModelStart, NULL, kStatusSuccess);
  while ((node = (Node *)(PopQueue(execution_data->ready_queue))) != NULL) {
    es->callback(sub_graph_type, es->arg, kExecuteStart, node, kStatusSuccess);
    KernelStatus ret = node->func(&node->context);
    es->callback(sub_graph_type, es->arg, kExecuteEnd, node, ret);
    if (ret != kStatusSuccess) {
      RecoverNodeInDegrees(execution_data);
      return ret;
    }

    Watcher *watchers = execution_data->node_watchers[node->node_id];
    for (size_t i = 0U; i < watchers->watch_num; ++i) {
      NodeIdentity node_id = watchers->node_ids[i];
      if (--execution_data->node_indegrees[node_id] == 0) {
        PushQueue(execution_data->ready_queue, (PriorityQueueElementHead *)(execution_data->base_ed.nodes[node_id]));
        execution_data->node_indegrees[node_id] = execution_data->node_indegrees_backup[node_id];
      }
    }
  }
  es->callback(sub_graph_type, es->arg, kModelEnd, NULL, kStatusSuccess);
  return kStatusSuccess;
}
#ifdef __cplusplus
}
#endif