/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_FRAMEWORK_EXECUTOR_GE_C_EXECUTOR_H_
#define INC_FRAMEWORK_EXECUTOR_GE_C_EXECUTOR_H_

#include <stdint.h>
#include "framework/executor_c/ge_executor_types.h"
#include "framework/executor_c/types.h"
#if defined(__cplusplus)
extern "C" {
#endif

Status GeInitialize(void);
Status GeFinalize(void);
Status GetModelDescInfo(uint32_t modelId, ModelInOutInfo *info);

Status GetMemAndWeightSize(const char *fileName, size_t *workSize, size_t *weightSize);
Status GetPartitionSize(const char *fileName, GePartitionSize *mdlPartitionSize);
Status ExecModel(uint32_t modelId, ExecHandleDesc *execDesc, bool sync, const InputData *inputData,
  OutputData *outputData);
Status GeLoadModelFromData(uint32_t *modelId, const ModelData *data);
Status LoadDataFromFile(const char *modelPath, ModelData *data);
void FreeModelData(ModelData *data);
Status UnloadModel(uint32_t modelId);
Status GetModelDescInfoFromMem(const ModelData *modelData, ModelInOutInfo *info);
void DestoryModelInOutInfo(ModelInOutInfo *info);
void DestroyModelInOutInfo(ModelInOutInfo *info);
Status GeDbgInit(const char *configPath);
Status GeDbgDeInit(void);
Status GeNofifySetDevice(uint32_t chipId, uint32_t deviceId);
#if defined(__cplusplus)
}
#endif

#endif  // INC_FRAMEWORK_EXECUTOR_GE_C_EXECUTOR_H_
