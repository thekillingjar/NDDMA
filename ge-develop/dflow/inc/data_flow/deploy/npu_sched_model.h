/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_FRAMEWORK_HETEROGENEOUS_NPU_SCHED_MODEL_H_
#define INC_FRAMEWORK_HETEROGENEOUS_NPU_SCHED_MODEL_H_
#include <stdlib.h>
#include "common/ge_common/ge_types.h"
#ifdef __cplusplus
extern "C" {
#endif

using NpuSchedModelHandler = void *;

typedef struct {
  uint32_t model_id;
  int32_t device_id;
  ge::ModelQueueParam model_queue_param;
  uint32_t req_msg_queue_id;
  uint32_t resp_msg_queue_id;
} NpuSchedLoadParam;

uint32_t InitializeNpuSched(int32_t device_id);
NpuSchedModelHandler CreateNpuSchedModelHandler();
uint32_t LoadNpuSchedModel(NpuSchedModelHandler handler, NpuSchedLoadParam *load_param);
uint32_t UnloadNpuSchedModel(NpuSchedModelHandler handler);
uint32_t DestroyNpuSchedModelHandler(NpuSchedModelHandler handler);
uint32_t FinalizeNpuSched(int32_t device_id);

#ifdef __cplusplus
}
#endif

#endif  // INC_FRAMEWORK_HETEROGENEOUS_NPU_SCHED_MODEL_H_
