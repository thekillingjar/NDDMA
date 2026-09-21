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
#include "runtime/base.h"
#include "runtime/context.h"
#include "runtime/mem.h"
#include "runtime/kernel.h"
#include "runtime/dev.h"
#include "runtime/rt_model.h"

#define EVENT_LENTH 10

rtError_t rtCtxSetCurrent(rtContext_t ctx)
{
    return RT_ERROR_NONE;
}

rtError_t rtGetStreamId(rtStream_t stream, int32_t *streamId)
{
    *streamId = 0;
    return RT_ERROR_NONE;
}

rtError_t rtCtxGetCurrent(rtContext_t *ctx)
{
    int x = 1;
    *ctx = (void *)x;
    return RT_ERROR_NONE;
}

rtError_t rtCtxSetDryRun (rtContext_t ctx, rtDryRunFlag_t enable, uint32_t flag)
{
    return RT_ERROR_NONE;
}

rtError_t rtEventGetTimeStamp(uint64_t *time, rtEvent_t event)
{
    *time = 12345;
    return RT_ERROR_NONE;
}

rtError_t rtEventCreate(rtEvent_t *event)
{
    *event = new int[EVENT_LENTH];
    return RT_ERROR_NONE;
}
rtError_t rtEventRecord(rtEvent_t event, rtStream_t stream)
{
    return RT_ERROR_NONE;
}

rtError_t rtEventSynchronize(rtEvent_t event)
{
    return RT_ERROR_NONE;
}

rtError_t rtEventDestroy(rtEvent_t event)
{
    delete [] (int*)event;
    return RT_ERROR_NONE;
}

rtError_t rtMalloc(void **devPtr, uint64_t size, rtMemType_t type, uint16_t moduleId)
{
    *devPtr = new uint8_t[size];
    return RT_ERROR_NONE;
}

rtError_t rtMemset(void *devPtr, uint64_t destMax, uint32_t value, uint64_t count)
{
    return RT_ERROR_NONE;
}

rtError_t rtFree(void *devPtr)
{
    delete[] (uint8_t *)devPtr;
    return RT_ERROR_NONE;
}

rtError_t rtMallocHost(void **hostPtr,  uint64_t  size, uint16_t moduleId)
{
    *hostPtr = new uint8_t[size];
    return RT_ERROR_NONE;
}

rtError_t rtFreeHost(void *hostPtr)
{
    delete[] (uint8_t*)hostPtr;
    return RT_ERROR_NONE;
}

rtError_t rtStreamCreate(rtStream_t *stream, int32_t priority)
{
    *stream = new uint32_t;
    return RT_ERROR_NONE;
}

rtError_t rtStreamDestroy(rtStream_t stream)
{
    delete (uint32_t*)stream;
    return RT_ERROR_NONE;
}

rtError_t rtSetDevice(int32_t device)
{
    return RT_ERROR_NONE;
}

rtError_t rtStreamSynchronize(rtStream_t stream)
{
    return RT_ERROR_NONE;
}

rtError_t rtMemcpy(void *dst, uint64_t destMax, const void *src, uint64_t count, rtMemcpyKind_t kind)
{
    #ifdef OTQT_UT
    if (destMax == 12 && count == 12) { // UTEST_kernelinfo_manager.all_success特殊处理
        memcpy_s(dst, destMax, src, count);
    }
    #endif
    return RT_ERROR_NONE;
}

rtError_t rtMemcpyEx(void *dst, uint64_t destMax, const void *src, uint64_t count, rtMemcpyKind_t kind)
{
  return rtMemcpy(dst, destMax, src, count, kind);
}

rtError_t rtMemcpyAsync(void *dst, uint64_t destMax, const void *src, uint64_t count, rtMemcpyKind_t kind, rtStream_t stream)
{
    return RT_ERROR_NONE;
}

rtError_t rtMemcpyAsyncWithoutCheckKind(void *dst, uint64_t destMax, const void *src, uint64_t count,
                                        rtMemcpyKind_t kind, rtStream_t stream)
{
  return rtMemcpyAsync(dst, destMax, src, count, kind, stream);
}

rtError_t rtStreamWaitEvent(rtStream_t stream, rtEvent_t event)
{
    return RT_ERROR_NONE;
}

rtError_t rtGetDeviceCount(int32_t* count)
{
    *count = 1;
    return RT_ERROR_NONE;
}

rtError_t rtDeviceReset(int32_t device)
{
    return RT_ERROR_NONE;
}

rtError_t rtEventElapsedTime(float *time, rtEvent_t start, rtEvent_t end)
{
    *time = 10.0f;
    return RT_ERROR_NONE;
}
rtError_t rtFunctionRegister(void *binHandle,
    const void *stubFunc,
    const char *stubName,
    const void *devFunc)
{
    return RT_ERROR_NONE;
}

rtError_t rtFunctionRegister(void *binHandle,
     const void *stubFunc,
     const char *stubName,
     const void *devFunc,
     uint32_t funcMode)
{
    return RT_ERROR_NONE;
}

rtError_t rtDevBinaryRegister(const rtDevBinary_t *bin, void **handle)
{
    return RT_ERROR_NONE;
}

rtError_t rtKernelConfigTransArg(const void *ptr,
    uint64_t size,
    uint32_t flag,
    void** arg)
{
    return RT_ERROR_NONE;
}

rtError_t rtKernelLaunch(const void *stubFunc,
    uint32_t blockDim,
    void *args,
    uint32_t argsSize,
    rtSmDesc_t *smDesc,
    rtStream_t stream)
{
    return RT_ERROR_NONE;
}
rtError_t rtSetupArgument(const void *arg, uint32_t size, uint32_t offset)
{
    return RT_ERROR_NONE;
}
rtError_t rtLaunch(const void *stubFunc)
{
    return RT_ERROR_NONE;
}
rtError_t rtDevBinaryUnRegister(void *handle)
{
    return RT_ERROR_NONE;
}
rtError_t rtConfigureCall(uint32_t numBlocks, rtSmDesc_t *smDesc, rtStream_t stream)
{
    return RT_ERROR_NONE;
}

rtError_t rtSetProfDir(char *profdir)
{
    return RT_ERROR_NONE;
}

rtError_t rtAiCoreMemorySizes(rtAiCoreMemorySize_t *aiCoreMemorySize)
{
    return RT_ERROR_NONE;
}

rtError_t rtMemAdvise(void *ptr, uint64_t size, uint32_t advise)
{
    return RT_ERROR_NONE;
}

/**
 * @ingroup rt_kernel
 * @brief start fusion kernels.
 * @param [in] stream   stream for fusion kernels
 * @return RT_ERROR_NONE for ok, errno for failed
 */
rtError_t rtKernelFusionStart(rtStream_t stream)
{
    return RT_ERROR_NONE;
}

/**
 * @ingroup rt_kernel
 * @brief end fusion kernels.
 * @param [in] stream   stream for fusion kernels
 * @return RT_ERROR_NONE for ok, errno for failed
 */
rtError_t rtKernelFusionEnd(rtStream_t stream)
{
    return RT_ERROR_NONE;
}
rtError_t rtMemGetInfo(size_t* free, size_t* total)
{
    *free = 512UL * 1024UL * 1024UL;
    *total = 1024UL * 1024UL * 1024UL;
    return RT_ERROR_NONE;
}

rtError_t rtMemAllocManaged(void **ptr, uint64_t size, uint32_t flag, uint16_t moduleId)
{
    return RT_ERROR_NONE;
}

rtError_t rtMemFreeManaged(void *ptr)
{
    return RT_ERROR_NONE;
}

rtError_t rtMetadataRegister(void *handle, const char *metadata)
{
	return RT_ERROR_NONE;
}
rtError_t rtSetTaskGenCallback(rtTaskGenCallback callback)
{
    return RT_ERROR_NONE;
}

rtError_t rtModelCreate(rtModel_t *model, uint32_t flag)
{
    *model = new uint32_t;
    return RT_ERROR_NONE;
}

rtError_t rtSetModelName(rtModel_t model, const char_t *mdlName)
{
    return RT_ERROR_NONE;
}

rtError_t rtModelDestroy(rtModel_t model)
{
    delete model;
    return RT_ERROR_NONE;
}

rtError_t rtModelBindStream(rtModel_t model, rtStream_t stream, uint32_t flag)
{
    return RT_ERROR_NONE;
}
rtError_t rtModelUnbindStream(rtModel_t model, rtStream_t stream)
{
    return RT_ERROR_NONE;
}
rtError_t rtModelExecute(rtModel_t model, rtStream_t stream, uint32_t flag)
{
    return RT_ERROR_NONE;
}

rtError_t rtGetFunctionByName(const char *stubName, void **stubFunc)
{
    *(char**)stubFunc =  "func";
    return RT_ERROR_NONE;
}

rtError_t rtQueryFunctionRegistered(const char *stubName)
{
    return RT_ERROR_NONE;
}

rtError_t rtCtxCreate(rtContext_t *ctx, uint32_t flags, int32_t device)
{
    return RT_ERROR_NONE;
}

rtError_t rtKernelLaunchEx(void *args, uint32_t argsSize, uint32_t flags, rtStream_t stream_)
{
    return RT_ERROR_NONE;
}

rtError_t rtModelGetTaskId(void* handle, uint32_t *taskid)
{
    *taskid = 0;
    return RT_ERROR_NONE;
}
rtError_t rtProfilerStop(void)
{
    return RT_ERROR_NONE;
}

rtError_t rtCtxDestroy(rtContext_t ctx)
{
	return RT_ERROR_NONE;
}

rtError_t rtProfilerStart(void)
{
    return RT_ERROR_NONE;
}

rtError_t rtLabelCreate(rtLabel_t *label)
{
    return RT_ERROR_NONE;
}

rtError_t rtLabelDestroy(rtLabel_t label)
{
    return RT_ERROR_NONE;
}

rtError_t rtLabelSet(rtLabel_t label, rtStream_t stream)
{
    return RT_ERROR_NONE;
}

rtError_t rtInvalidCache(uint64_t base, uint32_t len)
{
    return RT_ERROR_NONE;
}

rtError_t rtModelLoadComplete(rtModel_t model)
{
    return RT_ERROR_NONE;
}

rtError_t rtStreamCreateWithFlags(rtStream_t *stream, int32_t priority, uint32_t flags)
{
    *stream = new uint32_t;
    return RT_ERROR_NONE;
}

rtError_t rtFlushCache(uint64_t base, uint32_t len)
{
    return RT_ERROR_NONE;
}

rtError_t rtMemSetRC(const void *devPtr, uint64_t size, uint32_t readCount)
{
    return RT_ERROR_NONE;
}

rtError_t rtStreamActive(rtStream_t active_stream, rtStream_t stream)
{
    return RT_ERROR_NONE;
}

rtError_t rtGetAiCpuCount(uint32_t *aiCpuCnt)
{
  return RT_ERROR_NONE;
}

rtError_t rtGetDevice(int32_t *device)
{
  *device = 0;
  return RT_ERROR_NONE;
}

rtError_t rtGetDeviceInfo(uint32_t deviceId, int32_t moduleType, int32_t infoType, int64_t *value)
{
  *value = 2000;
  return RT_ERROR_NONE;
}