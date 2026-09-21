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
#include "profiling.h"
#include "toolchain/prof_api.h"
#include "dump_thread_manager.h"
#include "dbg_stub.h"
rtError_t rtDumpInit() {
  return DbgStubMock::GetInstance().rtDumpInit();
}

uint32_t InitDataDump() {
  return 0;
}

uint32_t StopDataDump() {
  return 0;
}

int32_t MsprofInit(uint32_t dataType, void *data, uint32_t dataLen) {
  return DbgStubMock::GetInstance().MsprofInit(dataType, data, dataLen);
}

int32_t MsprofRegisterCallback(uint32_t moduleId, ProfCommandHandle handle) {
  return DbgStubMock::GetInstance().MsprofRegisterCallback(moduleId, handle);
}

uint64_t MsprofGetHashId(const char *hashInfo, size_t length) {
  (void)hashInfo;
  (void)length;
  return 0;
}

uint64_t MsprofSysCycleTime() {
  return 0;
}

int32_t MsprofReportAdditionalInfo(uint32_t agingFlag, const VOID_PTR data, uint32_t length) {
  return DbgStubMock::GetInstance().MsprofReportAdditionalInfo(agingFlag, data, length);
}

int32_t MsprofFinalize() {
  return DbgStubMock::GetInstance().MsprofFinalize();
}

int32_t MsprofNotifySetDevice(uint32_t chipId, uint32_t deviceId, bool isOpen) {
  return DbgStubMock::GetInstance().MsprofNotifySetDevice(chipId, deviceId, isOpen);
}



