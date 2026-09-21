/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sequential_executor.h"
#include "core/executor/sequential/execution_data/sequential_execution_data.h"
#include "core/executor_error_code.h"
#ifdef __cplusplus
extern "C" {
#endif
KernelStatus SequentialExecute(void *arg) {
  SequentialExecutionData *execution_data = (SequentialExecutionData *)arg;

  for (size_t i = 0U; i < execution_data->node_num; ++i) {
    Node *node = execution_data->nodes[i];
    KernelStatus ret = node->func(&(node->context));
    if (ret != kStatusSuccess) {
      return ret;
    }
  }

  return kStatusSuccess;
}

KernelStatus SequentialExecuteWithCallback(int sub_graph_type, void *arg, ExecutorSubscriber *es) {
  SequentialExecutionData *execution_data = (SequentialExecutionData *)arg;
  if (es == NULL || es->callback == NULL) {
    return kStatusFailed;
  }
  es->callback(sub_graph_type, es->arg, kModelStart, NULL, kStatusSuccess);
  for (size_t i = 0U; i < execution_data->node_num; ++i) {
    Node *node = execution_data->nodes[i];
    es->callback(sub_graph_type, es->arg, kExecuteStart, node, kStatusSuccess);
    KernelStatus ret = node->func(&(node->context));
    es->callback(sub_graph_type, es->arg, kExecuteEnd, node, ret);
    if (ret != kStatusSuccess) {
      return ret;
    }
  }
  es->callback(sub_graph_type, es->arg, kModelEnd, NULL, kStatusSuccess);

  return kStatusSuccess;
}
#ifdef __cplusplus
}
#endif