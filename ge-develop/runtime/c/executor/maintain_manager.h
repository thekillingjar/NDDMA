/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_EXECUTOR_C_MAINTAIN_MANAGER_H_
#define GE_EXECUTOR_C_MAINTAIN_MANAGER_H_
#include "framework/executor_c/ge_executor_types.h"
#include "framework/executor_c/ge_executor.h"
#include "framework/executor_c/types.h"
#include "runtime/rt_model.h"
#ifdef __cplusplus
extern "C" {
#endif
void *CreatModelDbgHandle(void);
Status LoadModelPreProcess(rtMdlLoad_t *mdlLoad, size_t taskDescSize, char *modelName,
                           void *dbgHandle);
void StepIdConuterPlus(void *dbgHandle);
void DataFreeLoadDumpInfo(void *dbgHandle);
bool GetProfEnable(void);
Status LoadModelPostProcess(uint32_t modelId, char *modelName, uint64_t *stepIdAddr, void *dbgHandle);
#ifdef __cplusplus
}
#endif
#endif  // GE_EXECUTOR_C_MAINTAIN_MANAGER_H_
