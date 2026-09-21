/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_INC_FRAMEWORK_RUNTIME_EXECUTOR_SUBSCRIBER_C_H_
#define AIR_CXX_INC_FRAMEWORK_RUNTIME_EXECUTOR_SUBSCRIBER_C_H_
#include "exe_graph/runtime/base_type.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef enum {
  kExecuteStart,
  kExecuteEnd,
  kModelStart,
  kModelEnd,
  kExecuteEventEnd
} ExecutorEvent;

typedef void (*SubscriberFunc)(int type, void *arg, ExecutorEvent event, const void *node, KernelStatus result);
typedef struct {
  SubscriberFunc callback;
  void *arg;
} ExecutorSubscriber;
#ifdef __cplusplus
}
#endif
#endif  // AIR_CXX_INC_FRAMEWORK_RUNTIME_EXECUTOR_SUBSCRIBER_C_H_
