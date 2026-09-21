/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <securec.h>
#include "runtime/dev.h"
#include "runtime/base.h"
#include "runtime/context.h"
#include "runtime/mem.h"
#include "runtime/kernel.h"
#include "runtime/rt_model.h"
#ifdef __cplusplus
extern "C" {
#endif
#define EVENT_LENTH 10

rtError_t rtCtxSetCurrent(rtContext_t ctx) {
  return RT_ERROR_NONE;
}

rtError_t rtGetStreamId(rtStream_t stream, int32_t *stream_id) {
  *stream_id = 0;
  return RT_ERROR_NONE;
}

rtError_t rtCtxGetCurrent(rtContext_t *ctx) {
  uintptr_t x = 1;
  *ctx = (rtContext_t *)x;
  return RT_ERROR_NONE;
}

rtError_t rtCtxSetDryRun(rtContext_t ctx, rtDryRunFlag_t enable, uint32_t flag) {
  return RT_ERROR_NONE;
}

rtError_t rtEventGetTimeStamp(uint64_t *time, rtEvent_t event) {
  *time = 12345;
  return RT_ERROR_NONE;
}

rtError_t rtEventCreate(rtEvent_t *event) {
  *event = new int[EVENT_LENTH];
  return RT_ERROR_NONE;
}

rtError_t rtEventCreateWithFlag(rtEvent_t *event, uint32_t flag) {
  return rtEventCreate(event);
}

rtError_t rtEventRecord(rtEvent_t event, rtStream_t stream) {
  return RT_ERROR_NONE;
}

rtError_t rtEventSynchronize(rtEvent_t event) {
  return RT_ERROR_NONE;
}

rtError_t rtEventDestroy(rtEvent_t event) {
  delete[](int *) event;
  return RT_ERROR_NONE;
}

rtError_t rtMalloc(void **dev_ptr, uint64_t size, rtMemType_t type, const uint16_t moduleid) {
  *dev_ptr = new uint8_t[size];
  return RT_ERROR_NONE;
}

rtError_t rtMemset(void *dev_ptr, uint64_t dest_max, uint32_t value, uint64_t count) {
  return RT_ERROR_NONE;
}

rtError_t rtFree(void *dev_ptr) {
  delete[](uint8_t *) dev_ptr;
  return RT_ERROR_NONE;
}

rtError_t rtMallocHost(void **host_ptr, uint64_t size, const uint16_t moduleid) {
  *host_ptr = new uint8_t[size];
  return RT_ERROR_NONE;
}

rtError_t rtFreeHost(void *host_ptr) {
  delete[](uint8_t *) host_ptr;
  return RT_ERROR_NONE;
}

rtError_t rtStreamCreate(rtStream_t *stream, int32_t priority) {
  *stream = new uint32_t;
  return RT_ERROR_NONE;
}

rtError_t rtStreamDestroy(rtStream_t stream) {
  if (stream != nullptr) {
    delete (uint32_t *)stream;
  }
  return RT_ERROR_NONE;
}

rtError_t rtSetDevice(int32_t device) {
  return RT_ERROR_NONE;
}

rtError_t rtStreamSynchronize(rtStream_t stream) {
  return RT_ERROR_NONE;
}

rtError_t rtMemcpy(void *dst, uint64_t dest_max, const void *src, uint64_t count, rtMemcpyKind_t kind) {
  if (dst != nullptr && src != nullptr) {
    memcpy_s(dst, dest_max, src, count);
  }
  return RT_ERROR_NONE;
}
rtError_t rtMemcpyAsync(void *dst, uint64_t dest_max, const void *src, uint64_t count, rtMemcpyKind_t kind,
                        rtStream_t stream) {
  if (dst != nullptr && src != nullptr) {
    memcpy_s(dst, dest_max, src, count);
  }
  return RT_ERROR_NONE;
}

rtError_t rtMemcpyAsyncPtr(void *memcpyAddrInfo, uint64_t dest_max, uint64_t count, rtMemcpyKind_t kind,
                           rtStream_t stream, uint32_t qosCfg) {
  if (memcpyAddrInfo != nullptr) {
    uint32_t srcPtrOffset = 16U;
    uint32_t dstPtrOffset = 24U;
    memcpy_s(reinterpret_cast<void *>(reinterpret_cast<rtMemcpyAddrInfo *>(memcpyAddrInfo) + dstPtrOffset), dest_max,
             reinterpret_cast<void *>(reinterpret_cast<rtMemcpyAddrInfo *>(memcpyAddrInfo) + srcPtrOffset), count);
  }
  return RT_ERROR_NONE;
}

rtError_t rtStreamWaitEvent(rtStream_t stream, rtEvent_t event) {
  return RT_ERROR_NONE;
}
rtError_t rtNotifyWait(rtNotify_t notify_, rtStream_t stream_) {
  return RT_ERROR_NONE;
}
rtError_t rtGetNotifyID(rtNotify_t notify_, uint32_t *notify_id) {
  return RT_ERROR_NONE;
}
rtError_t rtNotifyRecord(rtNotify_t notify_, rtStream_t stream_) {
  return RT_ERROR_NONE;
}
rtError_t rtSetTSDevice(uint32_t tsId) {
  return RT_ERROR_NONE;
}

rtError_t rtGetDeviceCount(int32_t *count) {
  *count = 1;
  return RT_ERROR_NONE;
}

rtError_t rtDeviceReset(int32_t device) {
  return RT_ERROR_NONE;
}

rtError_t rtEventElapsedTime(float *time, rtEvent_t start, rtEvent_t end) {
  *time = 10.0f;
  return RT_ERROR_NONE;
}

rtError_t rtFunctionRegister(void *bin_handle, const void *stub_func, const char *stub_name, const void *dev_func,
                             uint32_t func_mode) {
  return RT_ERROR_NONE;
}

rtError_t rtDevBinaryRegister(const rtDevBinary_t *bin, void **handle) {
  return RT_ERROR_NONE;
}

rtError_t rtRegisterAllKernel(const rtDevBinary_t *bin, void **handle) {
  return RT_ERROR_NONE;
}

rtError_t rtKernelConfigTransArg(const void *ptr, uint64_t size, uint32_t flag, void **arg) {
  return RT_ERROR_NONE;
}

rtError_t rtKernelLaunchWithHandle(void *handle, const uint64_t tilingKey, uint32_t blockDim, rtArgsEx_t *argsInfo,
                                   rtSmDesc_t *smDesc, rtStream_t stream, const void *kernelInfo) {
  return RT_ERROR_NONE;
}

rtError_t rtKernelLaunch(const void *stub_func, uint32_t block_dim, void *args, uint32_t args_size, rtSmDesc_t *sm_desc,
                         rtStream_t stream) {
  return RT_ERROR_NONE;
}
rtError_t rtSetupArgument(const void *arg, uint32_t size, uint32_t offset) {
  return RT_ERROR_NONE;
}
rtError_t rtLaunch(const void *stub_func) {
  return RT_ERROR_NONE;
}
rtError_t rtDevBinaryUnRegister(void *handle) {
  return RT_ERROR_NONE;
}
rtError_t rtConfigureCall(uint32_t num_blocks, rtSmDesc_t *sm_desc, rtStream_t stream) {
  return RT_ERROR_NONE;
}

rtError_t rtSetProfDir(char *prof_dir) {
  return RT_ERROR_NONE;
}

rtError_t rtAiCoreMemorySizes(rtAiCoreMemorySize_t *aicore_memory_size) {
  return RT_ERROR_NONE;
}

rtError_t rtMemAdvise(void *ptr, uint64_t size, uint32_t advise) {
  return RT_ERROR_NONE;
}

rtError_t rtKernelFusionStart(rtStream_t stream) {
  return RT_ERROR_NONE;
}

rtError_t rtKernelFusionEnd(rtStream_t stream) {
  return RT_ERROR_NONE;
}
rtError_t rtMemGetInfo(size_t *free, size_t *total) {
  *free = 512UL * 1024UL * 1024UL;
  *total = 1024UL * 1024UL * 1024UL;
  return RT_ERROR_NONE;
}

rtError_t rtMemAllocManaged(void **ptr, uint64_t size, uint32_t flag, const uint16_t moduleid) {
  return RT_ERROR_NONE;
}

rtError_t rtMemFreeManaged(void *ptr) {
  return RT_ERROR_NONE;
}

rtError_t rtMetadataRegister(void *handle, const char *meta_data) {
  return RT_ERROR_NONE;
}
rtError_t rtSetTaskGenCallback(rtTaskGenCallback callback) {
  return RT_ERROR_NONE;
}

rtError_t rtModelCreate(rtModel_t *model, uint32_t flag) {
  *model = new uint32_t;
  return RT_ERROR_NONE;
}

rtError_t rtModelDestroy(rtModel_t model) {
  uint32_t *stub = static_cast<uint32_t *>(model);
  delete stub;
  return RT_ERROR_NONE;
}

rtError_t rtModelBindStream(rtModel_t model, rtStream_t stream, uint32_t flag) {
  return RT_ERROR_NONE;
}
rtError_t rtModelUnbindStream(rtModel_t model, rtStream_t stream) {
  return RT_ERROR_NONE;
}
rtError_t rtModelExecute(rtModel_t model, rtStream_t stream, uint32_t flag) {
  return RT_ERROR_NONE;
}

rtError_t rtGetFunctionByName(const char *stub_name, void **stub_func) {
  *(char **)stub_func = "func";
  return RT_ERROR_NONE;
}
rtError_t rtGetAddrByFun(const void *stubFunc, void **addr) {
  *(char **)addr = "dev_func";
  return RT_ERROR_NONE;
}
rtError_t rtQueryFunctionRegistered(const char *stub_name) {
  return RT_ERROR_NONE;
}

rtError_t rtCtxCreate(rtContext_t *ctx, uint32_t flags, int32_t device) {
  return RT_ERROR_NONE;
}

rtError_t rtKernelLaunchEx(void *args, uint32_t args_size, uint32_t flags, rtStream_t stream_) {
  return RT_ERROR_NONE;
}

rtError_t rtModelGetTaskId(void *handle, uint32_t *task_id, uint32_t *stream_id) {
  *task_id = 0;
  *stream_id = 0;
  return RT_ERROR_NONE;
}
rtError_t rtEndGraph(rtModel_t model, rtStream_t stream) {
  return RT_ERROR_NONE;
}
rtError_t rtEndGraphEx(rtModel_t model, rtStream_t stream, uint32_t flags) {
  return RT_ERROR_NONE;
}

rtError_t rtCtxDestroy(rtContext_t ctx) {
  return RT_ERROR_NONE;
}

rtError_t rtLabelCreate(rtLabel_t *label) {
  *label = new uint64_t;
  return RT_ERROR_NONE;
}

rtError_t rtLabelCreateEx(rtLabel_t *label, rtStream_t stream) {
  *label = new uint64_t;
  return RT_ERROR_NONE;
}

rtError_t rtLabelCreateV2(rtLabel_t *label, rtModel_t model) {
  *label = new uint64_t;
  return RT_ERROR_NONE;
}

rtError_t rtLabelCreateExV2(rtLabel_t *label, rtModel_t model, rtStream_t stream) {
  *label = new uint64_t;
  return RT_ERROR_NONE;
}

rtError_t rtLabelListCpy(rtLabel_t *label, uint32_t labelNumber, void *dst, uint32_t dstMax) {
  return RT_ERROR_NONE;
}

rtError_t rtLabelDestroy(rtLabel_t label) {
  uint64_t *stub = static_cast<uint64_t *>(label);
  delete stub;
  return RT_ERROR_NONE;
}

rtError_t rtLabelSet(rtLabel_t label, rtStream_t stream) {
  return RT_ERROR_NONE;
}

rtError_t rtLabelSwitchByIndex(void *ptr, uint32_t max, void *labelInfoPtr, rtStream_t stream) {
  return RT_ERROR_NONE;
}

rtError_t rtCmoAddrTaskLaunch(void *cmoAddrInfo, uint64_t destMax, rtCmoOpCode_t cmoOpCode, rtStream_t stm,
                              uint32_t flag) {
  return RT_ERROR_NONE;
}

rtError_t rtInvalidCache(void *base, size_t len) {
  return RT_ERROR_NONE;
}

rtError_t rtModelLoadComplete(rtModel_t model) {
  return RT_ERROR_NONE;
}

rtError_t rtStreamCreateWithFlags(rtStream_t *stream, int32_t priority, uint32_t flags) {
  *stream = new uint32_t;
  return RT_ERROR_NONE;
}

rtError_t rtFlushCache(void *base, size_t len) {
  return RT_ERROR_NONE;
}

rtError_t rtProfilerTraceEx(uint64_t id, uint64_t modelId, uint16_t tagId, rtStream_t stream) {
  return RT_ERROR_NONE;
}

rtError_t rtMemSetRC(const void *dev_ptr, uint64_t size, uint32_t read_count) {
  return RT_ERROR_NONE;
}

rtError_t rtStreamSwitchEx(void *ptr, rtCondition_t condition, void *value_ptr, rtStream_t true_stream,
                           rtStream_t stream, rtSwitchDataType_t data_type) {
  return RT_ERROR_NONE;
}

rtError_t rtStreamActive(rtStream_t active_stream, rtStream_t stream) {
  return RT_ERROR_NONE;
}

rtError_t rtEventReset(rtEvent_t event, rtStream_t stream) {
  return RT_ERROR_NONE;
}

rtError_t rtGetDevice(int32_t *device) {
  return RT_ERROR_NONE;
}

rtError_t rtDatadumpInfoLoad(const void *dump_info, uint32_t length) {
  return RT_ERROR_NONE;
}

rtError_t rtKernelLaunchWithFlag(const void *stub_func, uint32_t block_dim, rtArgsEx_t *argsInfo, rtSmDesc_t *sm_desc,
                                 rtStream_t stream_, uint32_t flags) {
  return RT_ERROR_NONE;
}

rtError_t rtCpuKernelLaunchWithFlag(const void *so_name, const void *kernel_name, uint32_t core_dim,
                                    const rtArgsEx_t *argsInfo, rtL2Ctrl_t *l2ctrl, rtStream_t stream_,
                                    uint32_t flags) {
  return RT_ERROR_NONE;
}

rtError_t rtModelGetId(rtModel_t model, uint32_t *modelId) {
  return RT_ERROR_NONE;
}

rtError_t rtModelBindQueue(rtModel_t model, uint32_t queueId, rtModelQueueFlag_t flag) {
  return RT_ERROR_NONE;
}

rtError_t rtSetSocVersion(const char *version) {
  return RT_ERROR_NONE;
}

rtError_t rtGetSocVersion(char *version, const uint32_t maxLen) {
  return RT_ERROR_NONE;
}

rtError_t rtGetAiCoreCount(uint32_t *aiCoreCnt) {
  return RT_ERROR_NONE;
}

rtError_t rtSetTaskFailCallback(rtTaskFailCallback callback) {
  return RT_ERROR_NONE;
}

rtError_t rtMallocHostSharedMemory(rtMallocHostSharedMemoryIn *in, rtMallocHostSharedMemoryOut *out) {
  out->ptr = new uint8_t[in->size];
  out->devPtr = new uint8_t[in->size];
  return RT_ERROR_NONE;
}

rtError_t rtFreeHostSharedMemory(rtFreeHostSharedMemoryIn *in) {
  delete[](uint8_t *) in->ptr;
  delete[](uint8_t *) in->devPtr;
  return RT_ERROR_NONE;
}

rtError_t rtGetAicpuDeploy(rtAicpuDeployType_t *deplyType) {
  return RT_ERROR_NONE;
}

rtError_t rtDebugRegister(rtModel_t model, uint32_t flag, const void *addr, uint32_t *streamId, uint32_t *taskId) {
  return RT_ERROR_NONE;
}

rtError_t rtDebugUnRegister(rtModel_t model) {
  return RT_ERROR_NONE;
}

rtError_t rtDumpAddrSet(rtModel_t model, void *addr, uint32_t dumpSize, uint32_t flag) {
  return RT_ERROR_NONE;
}

rtError_t rtSetCtxINFMode(bool mode) {
  return RT_ERROR_NONE;
}

rtError_t rtGetRtCapability(rtFeatureType_t featureType, int32_t featureInfo, int64_t *value) {
  return RT_ERROR_NONE;
}

rtError_t rtGetMaxStreamAndTask(uint32_t streamType, uint32_t *maxStrCount, uint32_t *maxTaskCount) {
  *maxStrCount = 1024;
  *maxTaskCount = 1024;
  return RT_ERROR_NONE;
}

rtError_t rtModelExit(rtModel_t model, rtStream_t stream) {
  return RT_ERROR_NONE;
}

rtError_t rtGetTaskIdAndStreamID(uint32_t *taskId, uint32_t *streamId) {
  return RT_ERROR_NONE;
}

rtError_t rtDebugRegisterForStream(rtStream_t stream, uint32_t flag, const void *addr, uint32_t *streamId,
                                   uint32_t *taskId) {
  return RT_ERROR_NONE;
}

rtError_t rtDebugUnRegisterForStream(rtStream_t stream) {
  return RT_ERROR_NONE;
}

rtError_t rtStreamSwitchN(void *ptr, uint32_t size, void *valuePtr, rtStream_t *trueStreamPtr, uint32_t elementSize,
                          void *stream, rtSwitchDataType_t dataType) {
  return RT_ERROR_NONE;
}

rtError_t rtGetEventID(rtEvent_t event, uint32_t *eventId) {
  return RT_ERROR_NONE;
}

rtError_t rtGetIsHeterogenous(int32_t *heterogenous) {
  return RT_ERROR_NONE;
}

rtError_t rtNpuGetFloatStatus(void *outputAddr, uint64_t outputSize, uint32_t checkMode, rtStream_t stm) {
  return RT_ERROR_NONE;
}

rtError_t rtNpuClearFloatStatus(uint32_t checkMode, rtStream_t stm) {
  return RT_ERROR_NONE;
}

rtError_t rtNpuGetFloatDebugStatus(void *outputAddr, uint64_t outputSize, uint32_t checkMode, rtStream_t stm) {
  return RT_ERROR_NONE;
}

rtError_t rtNpuClearFloatDebugStatus(uint32_t checkMode, rtStream_t stm) {
  return RT_ERROR_NONE;
}

rtError_t rtGeneralCtrl(uintptr_t *ctl, uint32_t num, uint32_t type) {
  return RT_ERROR_NONE;
}

#ifdef __cplusplus
}
#endif
