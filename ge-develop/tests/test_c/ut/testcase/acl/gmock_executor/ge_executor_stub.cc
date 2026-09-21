/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <securec.h>
#include "ge_executor_stub.h"

Status GetPartitionSize(const char *fileName, GePartitionSize *mdlPartitionSize) {
  return GeExecutorStubMock::GetInstance().GetPartitionSize(fileName, mdlPartitionSize);
}

Status GetMemAndWeightSize(const char *fileName, size_t *workSize, size_t *weightSize) {
  return GeExecutorStubMock::GetInstance().GetMemAndWeightSize(fileName, workSize, weightSize);
}
Status ExecModel(uint32_t modelId, ExecHandleDesc *execDesc, bool sync, const InputData *inputData,
                 OutputData *outputData) {
  return GeExecutorStubMock::GetInstance().ExecModel(modelId, execDesc, sync, inputData, outputData);
}
Status GeLoadModelFromData(uint32_t *modelId, const ModelData *modelData) {
  return GeExecutorStubMock::GetInstance().GeLoadModelFromData(modelId, modelData);
}
Status UnloadModel(uint32_t modelId) {
  return GeExecutorStubMock::GetInstance().UnloadModel(modelId);
}
Status GeInitialize() {
  return GeExecutorStubMock::GetInstance().GeInitialize();
}
Status GeFinalize() {
  return GeExecutorStubMock::GetInstance().GeFinalize();
}
Status GeDbgInit(const char *configPath) {
  return GeExecutorStubMock::GetInstance().GeDbgInit(configPath);
}
Status GeDbgDeInit(void) {
  return GeExecutorStubMock::GetInstance().GeDbgDeInit();
}
Status GeNofifySetDevice(uint32_t chipId, uint32_t deviceId) {
  return GeExecutorStubMock::GetInstance().GeNofifySetDevice(chipId, deviceId);
}
