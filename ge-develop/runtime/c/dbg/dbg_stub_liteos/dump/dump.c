/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dump.h"
Status DbgDumpInit(const char *cfg) {
  (void)cfg;
  return SUCCESS;
}

void *DbgCreateModelHandle(void) {
  return NULL;
}

Status DbgDumpPreProcess(rtMdlLoad_t *mdlLoad, size_t taskDescSize, char *modelName, void *dbgHandle) {
  (void)mdlLoad;
  (void)taskDescSize;
  (void)modelName;
  (void)dbgHandle;
  return SUCCESS;
}

Status DbgDumpPostProcess(uint32_t modelId, uint64_t *stepIdAddr, void *dbgHandle) {
  (void)modelId;
  (void)stepIdAddr;
  (void)dbgHandle;
  return SUCCESS;
}

void DbgStepIdCounterPlus(void *dbgHandle) {
  (void)dbgHandle;
  return;
}

void DbgFreeLoadDumpInfo(void *dbgHandle) {
  (void)dbgHandle;
  return;
}

Status DbgDumpDeInit(void) {
  return SUCCESS;
}