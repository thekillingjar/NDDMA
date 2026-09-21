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
#include "ascend_rt_stub.h"
rtError_t rtMalloc_Normal_Invoke(void **devPtr, uint64_t size, rtMemType_t type, const uint16_t moduleId) {
  (void)type;
  (void)moduleId;
  if ((size == MALLOC_SIZE_INVALID) || (size == MALLOC_SIZE_ABNORMAL) || (size == 0)) {
    return -1;
  }
  *devPtr = malloc(size);
  return RT_ERROR_NONE;
}

rtError_t rtMalloc_Abnormal_Invoke(void **devPtr, uint64_t size, rtMemType_t type, const uint16_t moduleId) {
  (void)type;
  (void)moduleId;
  if ((size == MALLOC_SIZE_INVALID) || (size == MALLOC_SIZE_ABNORMAL) || (size == 0)) {
    return -1;
  }
  *devPtr = nullptr;
  return -1;
}

rtError_t rtMalloc(void **devPtr, uint64_t size, rtMemType_t type, const uint16_t moduleId) {
  return RtStubMock::GetInstance().rtMalloc(devPtr, size, type, moduleId);
}

rtError_t rtMemcpy_Normal_Invoke(void *dst, uint64_t destMax, const void* src, uint64_t cnt, rtMemcpyKind_t kind) {
  (void)dst;
  (void)src;
  (void)kind;
  if ((destMax == MEMCPY_SIZE_INVALID) || (cnt == MEMCPY_SIZE_INVALID)) {
    return -1;
  }
  return RT_ERROR_NONE;
}

rtError_t rtMemcpy_Abnormal_Invoke(void *dst, uint64_t destMax, const void* src, uint64_t cnt, rtMemcpyKind_t kind) {
  (void)dst;
  (void)src;
  (void)kind;
  if ((destMax == MEMCPY_SIZE_INVALID) || (cnt == MEMCPY_SIZE_INVALID)) {
    return -1;
  }
  return -1;
}

rtError_t rtMemcpy(void *dst, uint64_t destMax, const void* src, uint64_t cnt, rtMemcpyKind_t kind) {
  return RtStubMock::GetInstance().rtMemcpy(dst, destMax, src, cnt, kind);
}

rtError_t rtFree(void *devPtr) {
  free(devPtr);
  return RT_ERROR_NONE;
}

rtError_t rtGetDevice(int32_t *devId) {
  *devId = 0;
  return RT_ERROR_NONE;
}

rtError_t rtStreamGetSqid(rtStream_t stream, uint32_t *sqid) {
  return RtStubMock::GetInstance().rtStreamGetSqid(stream, sqid);
}
rtError_t rtStreamSetLastMeid(rtStream_t stream, const uint64_t meid) {
  return RtStubMock::GetInstance().rtStreamSetLastMeid(stream, meid);
}

rtError_t rtNanoModelLoad(rtMdlLoad_t *modelLoad, uint32_t *phyModelId) {
  return RtStubMock::GetInstance().rtNanoModelLoad(modelLoad, phyModelId);
}

rtError_t rtNanoModelDestroy(uint32_t phyMdlId) {
  return RtStubMock::GetInstance().rtNanoModelDestroy(phyMdlId);
}

rtError_t rtNanoModelExecute(rtMdlExecute_t *modelExec) {
  return RtStubMock::GetInstance().rtNanoModelExecute(modelExec);
}

rtError_t rtDumpInit() {
  return RtStubMock::GetInstance().rtDumpInit();
}

rtError_t rtDumpDeInit() {
  return RT_ERROR_NONE;
}

rtError_t rtMsgSend(uint32_t tid, uint32_t sendTid, int32_t timeout, void *sendInfo, uint32_t size) {
  return RtStubMock::GetInstance().rtMsgSend(tid, sendTid, timeout, sendInfo, size);
}

rtError_t rtSetTaskDescDumpFlag(void *taskDescBaseAddr, size_t taskDescSize, uint32_t taskId) {
  return RtStubMock::GetInstance().rtSetTaskDescDumpFlag(taskDescBaseAddr, taskDescSize, taskId);
}

void *dlopen(const char *filename, int flag) {
  return RtStubMock::GetInstance().dlopen(filename, flag);
}

void *dlsym(void *handle, const char *symbol) {
  return RtStubMock::GetInstance().dlsym(handle, symbol);
}

int dlclose(void *handle) {
  return RtStubMock::GetInstance().dlclose(handle);
}

char *dlerror(void) {
  return RtStubMock::GetInstance().dlerror();
}

int access(const char *name, int type) {
  return RtStubMock::GetInstance().access(name, type);
}
