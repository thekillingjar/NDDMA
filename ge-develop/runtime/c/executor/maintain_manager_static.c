/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "framework/executor_c/ge_log.h"
#include "ge/ge_error_codes.h"
#include "framework/executor_c/ge_executor.h"
#include "maintain_manager.h"
#include "dbg_main.h"
#include "profiling.h"
#include "dump.h"

Status GeDbgInit(const char *configPath) {
  return DbgInit(configPath);
}

Status GeDbgDeInit(void) {
  return DbgDeInit();
}

Status GeNofifySetDevice(uint32_t chipId, uint32_t deviceId) {
  return DbgNotifySetDevice(chipId, deviceId);
}

bool GetProfEnable(void) {
  return DbgGetprofEnable();
}

Status LoadModelPostProcess(uint32_t modelId, char *modelName, uint64_t *stepIdAddr, void *dbgHandle) {
  return DbgLoadModelPostProcess(modelId, modelName, stepIdAddr, dbgHandle);
}

Status LoadModelPreProcess(rtMdlLoad_t *mdlLoad, size_t taskDescSize, char *modelName,
                           void *dbgHandle) {
  return DbgDumpPreProcess(mdlLoad, taskDescSize, modelName, dbgHandle);
}

void StepIdConuterPlus(void *dbgHandle) {
  DbgStepIdCounterPlus(dbgHandle);
}

void DataFreeLoadDumpInfo(void *dbgHandle) {
  DbgFreeLoadDumpInfo(dbgHandle);
}

void *CreatModelDbgHandle(void) {
  return DbgCreateModelHandle();
}