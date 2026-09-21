/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "offline_build_config_parse.h"
#include "hcom_acl_adapter.h"
#include "graph/ge_local_context.h"
#include "ge/ge_api_types.h"  // ge对内options
#include <cstdlib>

HcclResult MemcpyKindTranslate(HcclRtMemcpyKind kind, aclrtMemcpyKind *rtKind) {
  switch (kind) {
    case HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_HOST:
      *rtKind = ACL_MEMCPY_HOST_TO_HOST;
      return HCCL_SUCCESS;

    case HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE:
      *rtKind = ACL_MEMCPY_HOST_TO_DEVICE;
      return HCCL_SUCCESS;

    case HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_HOST:
      *rtKind = ACL_MEMCPY_DEVICE_TO_HOST;
      return HCCL_SUCCESS;

    case HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_DEVICE:
      *rtKind = ACL_MEMCPY_DEVICE_TO_DEVICE;
      return HCCL_SUCCESS;

    case HcclRtMemcpyKind::HCCL_RT_MEMCPY_ADDR_DEVICE_TO_DEVICE:
      *rtKind = ACL_MEMCPY_INNER_DEVICE_TO_DEVICE;
      return HCCL_SUCCESS;

    default: {
      HCCL_ERROR("[MemcpyKindTranslate]Not support the memory copy type[%d].", kind);
      return HCCL_E_PARA;
    }
  }
}

HcclResult hrtMemAsyncCopy(void *dst, uint64_t destMax, const void *src, uint64_t count, HcclRtMemcpyKind kind,
                           rtStream_t stream) {
#ifndef HCCD
  CHK_PTR_NULL(dst);
  CHK_PTR_NULL(src);
  CHK_PTR_NULL(stream);
  CHK_PRT_RET(count == 0, HCCL_WARNING("[hrtMemAsyncCopy] count is zero"), HCCL_SUCCESS);

  aclrtMemcpyKind rtKind;
  CHK_RET(MemcpyKindTranslate(kind, &rtKind));
  if (rtKind == ACL_MEMCPY_INNER_DEVICE_TO_DEVICE) {
    // runtime提供的acl接口不支持910A，暂时使用rtMemcpyAsync接口
    rtError_t ret = rtMemcpyAsync(dst, destMax, src, count, RT_MEMCPY_ADDR_DEVICE_TO_DEVICE, stream);

    HCCL_DEBUG(
        "Call hrtMemAsyncCopy, return value[%d], dstAddr[%p], destMax[%llu], "
        "srcAddr[%p], count[%llu], rtKind[%d]",
        ret, dst, destMax, src, count, RT_MEMCPY_ADDR_DEVICE_TO_DEVICE);

    CHK_PRT_RET(
        ret != RT_ERROR_NONE,
        HCCL_ERROR("[AsyncCopy][Mem] rt memory async copy failed, "
                   "return[%d], para: dstAddr[%p], destMax[%llu], srcAddr[%p], count[%llu], rtKind[%d], stream[%p].",
                   ret, dst, destMax, src, count, RT_MEMCPY_ADDR_DEVICE_TO_DEVICE, stream),
        HCCL_E_RUNTIME);

    return HCCL_SUCCESS;
  } else {
    aclError ret = aclrtMemcpyAsync(dst, destMax, src, count, rtKind, stream);

    HCCL_DEBUG(
        "Call hrtMemAsyncCopy, return value[%d], dstAddr[%p], destMax[%llu], "
        "srcAddr[%p], count[%llu], rtKind[%d]",
        ret, dst, destMax, src, count, rtKind);

    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR("[AsyncCopy][Mem] rt memory async copy failed, "
                   "return[%d], para: dstAddr[%p], destMax[%llu], srcAddr[%p], count[%llu], rtKind[%d], stream[%p].",
                   ret, dst, destMax, src, count, rtKind, stream),
        HCCL_E_RUNTIME);

    return HCCL_SUCCESS;
  }
#else
  HCCL_ERROR("[hrtMemAsyncCopy]Does not support this interface.");
  return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult hrtMemSyncCopy(void *dst, uint64_t destMax, const void *src, uint64_t count, HcclRtMemcpyKind kind) {
#ifndef HCCD
  CHK_PTR_NULL(dst);
  CHK_PTR_NULL(src);
  CHK_PRT_RET(count == 0, HCCL_WARNING("[hrtMemSyncCopy] count is zero"), HCCL_SUCCESS);

  aclrtMemcpyKind rtKind;
  CHK_RET(MemcpyKindTranslate(kind, &rtKind));
  aclError ret = aclrtMemcpy(dst, destMax, src, count, rtKind);
  HCCL_DEBUG("Call aclrtMemcpy, return[%d], para: dstAddr[%p], destMax[%llu], srcAddr[%p], count[%llu], rtKind[%d]",
             ret, dst, destMax, src, count, rtKind);
  CHK_PRT_RET(ret != ACL_SUCCESS,
              HCCL_ERROR("[SyncCopy][Mem] aclrtMemcpy failed, "
                         "return[%d], para: dstAddr[%p], destMax[%llu], srcAddr[%p], count[%llu], rtKind[%d].",
                         ret, dst, destMax, src, count, rtKind),
              HCCL_E_RUNTIME);
  return HCCL_SUCCESS;
#else
  HCCL_ERROR("[hrtMemSyncCopy]Does not support this interface.");
  return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult hrtStreamGetMode(rtStream_t const stream, uint64_t *const stmMode) {
#ifndef HCCD
  CHK_PTR_NULL(stream);

  s32 streamId = -1;
  rtError_t ret = aclrtStreamGetId(stream, &streamId);
  HCCL_DEBUG("Call rtGetStreamId, return value[%d].", ret);

  aclrtStreamAttrValue value;
  ret = aclrtGetStreamAttribute(stream, ACL_STREAM_ATTR_FAILURE_MODE, &value);
  CHK_PRT_RET(ret != ACL_SUCCESS,
              HCCL_ERROR("[Stream][GetMode] aclrtGetStreamAttribute error, "
                         "rtRet[%d]",
                         ret),
              HCCL_E_RUNTIME);
  *stmMode = value.failureMode;
  HCCL_DEBUG("Call aclrtGetStreamAttribute return value[%d]. stmMode[%llu], streamId[%d].", ret, *stmMode, streamId);
  return HCCL_SUCCESS;
#else
  HCCL_ERROR("[hrtMemAsyncCopy]Does not support this interface.");
  return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult hrtFree(void *devPtr) {
#ifndef HCCD
  CHK_PTR_NULL(devPtr);

  aclError ret = aclrtFree(devPtr);
  HCCL_DEBUG("Call aclrtFree, return value[%d], para: dev_ptr[%p].", ret, devPtr);
  CHK_PRT_RET((ret != ACL_SUCCESS),
              HCCL_ERROR("[Free][Mem] aclrtFree failed, "
                         "return[%d], para: devPtrAddr[%p].",
                         ret, devPtr),
              HCCL_E_RUNTIME);
  return HCCL_SUCCESS;
#else
  CHK_PTR_NULL(devPtr);
  free(devPtr);
  return HCCL_SUCCESS;
#endif
}

HcclResult hrtGetDeviceRefresh(s32 *deviceLogicId) {
#ifndef HCCD
  CHK_PTR_NULL(deviceLogicId);

  aclError ret = 0;
  ret = aclrtGetDevice(deviceLogicId);
  CHK_PRT_RET(ret != ACL_SUCCESS,
              HCCL_ERROR("[Get][DeviceRefresh] rtGet device fail, "
                         "please make sure that device is set. return[%d], para:deviceLogicId[%d]",
                         ret, *deviceLogicId),
              HCCL_E_RUNTIME);
  HCCL_INFO("[hrtGetDeviceRefresh]deviceLogicId[%d]", *deviceLogicId);
  return HCCL_SUCCESS;
#else
  HCCL_WARNING("[hrtGetDeviceRefresh]Does not support this interface.");
  return HCCL_E_NOT_SUPPORT;
#endif
}

/* 设定当前线程操作的目标设备编号 */
HcclResult hrtSetDevice(s32 deviceLogicId) {
#ifndef HCCD
  aclError ret = aclrtSetDevice(deviceLogicId);
  HCCL_DEBUG("Call aclrtSetDevice, return value[%d], para: device_id[%d].", ret, deviceLogicId);
  CHK_PRT_RET(ret != ACL_SUCCESS,
              HCCL_ERROR("[Set][Device]errNo[0x%016llx] rtSet device fail, return[%d], "
                         "para:deviceLogicId[%d]",
                         HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret, deviceLogicId),
              HCCL_E_RUNTIME);
  return HCCL_SUCCESS;
#else
  HCCL_WARNING("[hrtSetDevice]Does not support this interface.");
  return HCCL_E_NOT_SUPPORT;
#endif
}

/* 释放当前线程操作的目标设备编号,释放前必须先释放设备资源 */
HcclResult hrtResetDevice(s32 deviceLogicId) {
#ifndef HCCD
  aclError ret = aclrtResetDevice(deviceLogicId);
  HCCL_DEBUG("Call aclrtResetDevice, return value[%d], para: device_id[%d].", ret, deviceLogicId);
  CHK_PRT_RET(ret != ACL_SUCCESS,
              HCCL_ERROR("[Reset][Device]errNo[0x%016llx] rtReset device fail, return[%d], "
                         "para: deviceLogicId[%d]",
                         HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret, deviceLogicId),
              HCCL_E_RUNTIME);
  return HCCL_SUCCESS;
#else
  HCCL_WARNING("[hrtResetDevice]Does not support this interface.");
  return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult hrtGetSocVer(std::string &socName) {
#ifndef HCCD
  const char *socNamePtr = aclrtGetSocName();
  CHK_PRT_RET((socNamePtr == nullptr),
              HCCL_ERROR("[Get][SocVer]errNo[0x%016llx] aclrtGet socName failed", HCCL_ERROR_CODE(HCCL_E_RUNTIME)),
              HCCL_E_RUNTIME);

  socName = socNamePtr;
  return HCCL_SUCCESS;
#else
  HCCL_ERROR("[hrtGetSocVer]Does not support this interface.");
  return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult hrtMalloc(void **devPtr, u64 size, bool level2Address) {
#ifndef HCCD
  CHK_PTR_NULL(devPtr);
  std::string socVersion;
  CHK_RET(hrtGetSocVer(socVersion));

  u32 policy;
  bool isTsMem;
  std::string sGroup = "HCCL_WORLD_GROUP";
  HcomGetMemType(sGroup.c_str(), socVersion.c_str(), true, &policy, &isTsMem, false, level2Address);

  aclrtMallocAttrValue moduleIdValue;
  moduleIdValue.moduleId = HCCL;
  aclrtMallocAttribute attrs{.attr = ACL_RT_MEM_ATTR_MODULE_ID, .value = moduleIdValue};
  aclrtMallocConfig cfg{.attrs = &attrs, .numAttrs = 1};

  aclError ret = ACL_SUCCESS;
  if (UNLIKELY(isTsMem)) {
    ret = aclrtMallocForTaskScheduler(devPtr, size, static_cast<aclrtMemMallocPolicy>(policy), &cfg);
  } else {
    ret = aclrtMallocWithCfg(devPtr, size, static_cast<aclrtMemMallocPolicy>(policy), &cfg);
  }
  if (ret != ACL_SUCCESS) {
    REPORT_PREDEFINED_ERR_MSG(
        "EI0007", std::vector<const char *>({"resource_type", "resource_info"}),
        std::vector<const char *>({"DeviceMemory", (std::string("size:") + std::to_string(size)).c_str()}));
    HCCL_ERROR(
        "[Malloc][Mem] rtMalloc failed, "
        "return[%d], para: devPtrAddr[%p], size[%llu Byte], isTsMem[%d].",
        ret, *devPtr, size, isTsMem);
    return HCCL_E_RUNTIME;
  }

  s32 deviceId = 0;
  CHK_RET(hrtGetDeviceRefresh(&deviceId));
  HCCL_DEBUG(
      "Malloc DevMem para: deviceId[%d] policy[0x%x] devPtr[%p] size[%llu Byte] "
      "level2Address[%u] isTsMem[%d].",
      deviceId, policy, *devPtr, size, level2Address, isTsMem);
  return HCCL_SUCCESS;
#else
  CHK_PTR_NULL(devPtr);
  *devPtr = malloc(size);
  CHK_PTR_NULL(*devPtr);
  return HCCL_SUCCESS;
#endif
}

HcclResult hrtMemcpyAddrAsync(void *dst, uint64_t destMax, uint64_t destOffset, const void *src, uint64_t count,
                              uint64_t srcOffset, rtStream_t stream) {
#ifndef HCCD
  // 参数有效性检查
  CHK_PTR_NULL(dst);
  CHK_PTR_NULL(src);
  CHK_PTR_NULL(stream);

  if (count == 0) {
    HCCL_WARNING("Call hrtMemcpyAddrAsync, count [%d]", count);
    return HCCL_SUCCESS;
  }

  aclError ret = aclrtMemcpyAsyncWithOffset(reinterpret_cast<void **>(dst), destMax, destOffset,
                                            reinterpret_cast<const void **>(const_cast<void *>(src)), count, srcOffset,
                                            ACL_MEMCPY_INNER_DEVICE_TO_DEVICE, stream);

  HCCL_DEBUG(
      "Call hrtMemcpyAddrAsync, return value[%d], dstAddr[%p], destMax[%llu], destOffset[%llu], "
      "srcAddr[%p], count[%llu], srcOffset[%llu]",
      ret, dst, destMax, destOffset, src, count, srcOffset);

  if (ret == ACL_ERROR_RT_FEATURE_NOT_SUPPORT) {
    HCCL_WARNING("hrtMemcpyAddrAsync is not supported.", ret);
    return HCCL_E_NOT_SUPPORT;
  }

  CHK_PRT_RET(
      ret != ACL_SUCCESS,
      HCCL_ERROR(
          "[AsyncCopy][Mem]errNo[0x%016llx] rt memory async copy failed, "
          "return[%d], para: dstAddr[%p], destMax[%llu], destOffset[%llu], srcAddr[%p], count[%llu], srcOffset[%llu], "
          "stream[%p].",
          HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret, dst, destMax, destOffset, src, count, srcOffset, stream),
      HCCL_E_RUNTIME);

  return HCCL_SUCCESS;
#else
  HCCL_ERROR("[hrtMemcpyAddrAsync]Does not support this interface.");
  return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult hrtMallocHost(void **hostPtr, u64 size) {
  CHK_PTR_NULL(hostPtr);
  aclrtMallocAttrValue moduleIdValue;
  moduleIdValue.moduleId = HCCL;
  aclrtMallocAttribute attrs{.attr = ACL_RT_MEM_ATTR_MODULE_ID, .value = moduleIdValue};
  aclrtMallocConfig cfg{.attrs = &attrs, .numAttrs = 1};

  aclError ret = aclrtMallocHostWithCfg(hostPtr, size, &cfg);

  if (ret != ACL_SUCCESS) {
    REPORT_PREDEFINED_ERR_MSG(
        "EI0007", std::vector<const char *>({"resource_type", "resource_info"}),
        std::vector<const char *>({"HostMemory", (std::string("size:") + std::to_string(size)).c_str()}));
    HCCL_ERROR(
        "[Malloc][Host]errNo rt malloc host fail. return[%d], "
        "para: hostPtr[%p], size[%llu Byte].",
        ret, *hostPtr, size);
    return HCCL_E_RUNTIME;
  }
  HCCL_DEBUG("Malloc HostMem para: hostPtr[%p], size[%llu Byte]", *hostPtr, size);
  return HCCL_SUCCESS;
}

HcclResult hrtFreeHost(void *hostPtr) {
  CHK_PTR_NULL(hostPtr);

  aclError ret = aclrtFreeHost(hostPtr);

  if (ret != ACL_SUCCESS) {
    HCCL_ERROR(
        "[Free][Host] rt free host fail. return[%d], "
        "para: hostPtr[%p].",
        ret, hostPtr);
    return HCCL_E_RUNTIME;
  }
  HCCL_DEBUG("Call aclrtFreeHost, return value[%d].", ret);
  return HCCL_SUCCESS;
}

HcclResult hcclStreamSynchronize(void *stream) {
#ifndef HCCD
  CHK_PTR_NULL(stream);
  aclError ret = aclrtSynchronizeStream(stream);
  CHK_PRT_RET(ret != ACL_SUCCESS,
              HCCL_ERROR("[Synchronize][Stream]errNo[0x%016llx] rt "
                         "aclrtSynchronizeStream fail. return[%d].",
                         HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret),
              HCCL_E_RUNTIME);
  return HCCL_SUCCESS;
#else
  HCCL_ERROR("[hcclStreamSynchronize]Does not support this interface.");
  return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult hrtEventCreate(aclrtEvent *event) {
#ifndef HCCD
  CHK_PTR_NULL(event);
  aclError ret = aclrtCreateEvent(event);
  if (ret != ACL_SUCCESS) {
    REPORT_PREDEFINED_ERR_MSG("EI0007", std::vector<const char *>({"resource_type", "resource_info"}),
                              std::vector<const char *>({"event", "null"}));
    HCCL_ERROR(
        "[Create][Event]errNo[0x%016llx] rt event create, return[%d], "
        "event[%p]",
        HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret, event);
    return HCCL_E_RUNTIME;
  }
  HCCL_DEBUG("Call aclrtCreateEvent, return value[%d], para: event[%p]", ret, *event);
  return HCCL_SUCCESS;
#else
  HCCL_ERROR("[hrtEventCreate]Does not support this interface.");
  return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult hrtEventRecord(aclrtEvent event, aclrtStream stream) {
#ifndef HCCD
  CHK_PTR_NULL(event);
  CHK_PTR_NULL(stream);
  aclError ret = aclrtRecordEvent(event, stream);
  HCCL_DEBUG("Call aclrtRecordEvent, return value[%d], para: event[%p]", ret, event);
  CHK_PRT_RET(ret != ACL_SUCCESS,
              HCCL_ERROR("[Record][Event]errNo[0x%016llx] rt event record, return[%d], "
                         "event[%p]",
                         HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret, event),
              HCCL_E_RUNTIME);
  return HCCL_SUCCESS;
#else
  HCCL_ERROR("[hrtEventRecord]Does not support this interface.");
  return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult hrtEventQuery(aclrtEvent event) {
#ifndef HCCD
  CHK_PTR_NULL(event);
  aclrtEventRecordedStatus status = ACL_EVENT_RECORDED_STATUS_NOT_READY;
  aclError ret = aclrtQueryEventStatus(event, &status);
  HCCL_DEBUG("Call aclrtQueryEventStatus, return value[%d], status[%d], para: event[%p]", ret, status, event);
  CHK_PRT_RET(ret != ACL_SUCCESS,
              HCCL_ERROR("[aclrtQueryEventStatus] query event status failed, event:%p, return value[%d], status[%d].",
                         event, ret, status),
              HCCL_E_RUNTIME);
  return (status == ACL_EVENT_RECORDED_STATUS_COMPLETE) ? HCCL_SUCCESS : HCCL_E_RUNTIME;
#else
  HCCL_ERROR("[hrtEventQuery]Does not support this interface.");
  return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult hrtEventDestroy(aclrtEvent event) {
#ifndef HCCD
  CHK_PTR_NULL(event);
  aclError ret = aclrtDestroyEvent(event);
  HCCL_DEBUG("Call aclrtDestroyEvent, return value[%d], para: event[%p]", ret, event);
  CHK_PRT_RET(ret != ACL_SUCCESS,
              HCCL_ERROR("[Destroy][Event]errNo[0x%016llx] rt event destroy, return[%d], "
                         "event[%p]",
                         HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret, event),
              HCCL_E_RUNTIME);
  return HCCL_SUCCESS;
#else
  HCCL_ERROR("[hrtEventDestroy]Does not support this interface.");
  return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult hrtStreamCreateWithFlags(aclrtStream *stream, int32_t priority, uint32_t flags) {
#ifndef HCCD
  CHK_PTR_NULL(stream);

  aclError ret = aclrtCreateStreamWithConfig(stream, static_cast<uint32_t>(priority), flags);
  CHK_PRT_RET(ret != ACL_SUCCESS,
              HCCL_ERROR("[Stream][CreateWithFlags]errNo[0x%016llx] aclrtCreateStreamWithConfig "
                         "error, rtRet[%d], flags[%u]",
                         HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret, flags),
              HCCL_E_RUNTIME);
  return HCCL_SUCCESS;
#else
  HCCL_ERROR("[hrtStreamCreateWithFlags]Does not support this interface.");
  return HCCL_E_NOT_SUPPORT;
#endif
}