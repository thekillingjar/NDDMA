/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __INC_LLT_DRIVER_STUB_H
#define __INC_LLT_DRIVER_STUB_H
#include <vector>
#include <dlfcn.h>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "runtime/stream.h"
#include "runtime/rt_model.h"
#include "runtime/mem.h"
#include "runtime/dev.h"

#define MALLOC_SIZE_INVALID 666
#define MALLOC_SIZE_ABNORMAL 24
#define MEMCPY_SIZE_INVALID 333

class RtStubMock {
public:
  static RtStubMock& GetInstance() {
    static RtStubMock mock;
    return mock;
  }
  rtError_t rtGetDevice(int32_t *devId);
  rtError_t rtFree(void *devPtr) ;
  rtError_t rtDumpDeInit();
  MOCK_METHOD0(rtDumpInit, rtError_t());
  MOCK_METHOD4(rtMalloc, rtError_t(void **devPtr, uint64_t size, rtMemType_t type, const uint16_t moduleId));
  MOCK_METHOD5(rtMemcpy, rtError_t(void *dst, uint64_t destMax, const void* src, uint64_t cnt, rtMemcpyKind_t kind));
  MOCK_METHOD2(rtStreamGetSqid, rtError_t(rtStream_t stream, uint32_t *sqid));
  MOCK_METHOD2(rtStreamSetLastMeid, rtError_t(rtStream_t stream, const uint64_t meid));
  MOCK_METHOD2(rtNanoModelLoad, rtError_t(rtMdlLoad_t *modelLoad, uint32_t *phyModelId));
  MOCK_METHOD1(rtNanoModelDestroy, rtError_t(uint32_t phyMdlId));
  MOCK_METHOD1(rtNanoModelExecute, rtError_t(rtMdlExecute_t *modelExec));
  MOCK_METHOD5(rtMsgSend, rtError_t(uint32_t tid, uint32_t sendTid, int32_t timeout, void *sendInfo, uint32_t size));
  MOCK_METHOD3(rtSetTaskDescDumpFlag, rtError_t(void *taskDescBaseAddr, size_t taskDescSize, uint32_t taskId));
  MOCK_METHOD2(dlopen, void *(const char *filename, int flag));
  MOCK_METHOD2(dlsym, void *(void *handle, const char *symbol));
  MOCK_METHOD1(dlclose, int(void *handle));
  MOCK_METHOD0(dlerror, char *());
  MOCK_METHOD2(access, int(const char *name, int type));
};
rtError_t rtMalloc_Normal_Invoke(void **devPtr, uint64_t size, rtMemType_t type, const uint16_t moduleId);
rtError_t rtMalloc_Abnormal_Invoke(void **devPtr, uint64_t size, rtMemType_t type, const uint16_t moduleId);
rtError_t rtMemcpy_Normal_Invoke(void *dst, uint64_t destMax, const void* src, uint64_t cnt, rtMemcpyKind_t kind);
rtError_t rtMemcpy_Abnormal_Invoke(void *dst, uint64_t destMax, const void* src, uint64_t cnt, rtMemcpyKind_t kind);

#endif
