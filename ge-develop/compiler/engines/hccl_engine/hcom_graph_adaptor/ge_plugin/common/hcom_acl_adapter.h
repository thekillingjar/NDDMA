/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOM_ACL_ADAPTER_H
#define HCOM_ACL_ADAPTER_H

#include "runtime/rt.h"
#include "acl/acl_rt.h"
#include "hcom_log.h"

enum class HcclRtMemcpyKind {
  HCCL_RT_MEMCPY_KIND_HOST_TO_HOST = 0, /**< host to host */
  HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE,   /**< host to device */
  HCCL_RT_MEMCPY_KIND_DEVICE_TO_HOST,   /**< device to host */
  HCCL_RT_MEMCPY_KIND_DEVICE_TO_DEVICE, /**< device to device */
  HCCL_RT_MEMCPY_ADDR_DEVICE_TO_DEVICE, /**< Level-2 address copy, device to device */
  HCCL_RT_MEMCPY_KIND_RESERVED,
};

HcclResult hrtGetSocVer(std::string &socName);

HcclResult MemcpyKindTranslate(HcclRtMemcpyKind kind, aclrtMemcpyKind *rtKind);

HcclResult hrtMemAsyncCopy(void *dst, uint64_t destMax, const void *src, uint64_t count, HcclRtMemcpyKind kind,
                           rtStream_t stream);

HcclResult hrtMemSyncCopy(void *dst, uint64_t destMax, const void *src, uint64_t count, HcclRtMemcpyKind kind);

HcclResult hrtStreamGetMode(rtStream_t const stream, uint64_t *const stmMode);

HcclResult hrtFree(void *devPtr);

HcclResult hrtGetDeviceRefresh(s32 *deviceLogicId);
HcclResult hrtSetDevice(s32 deviceLogicId);
HcclResult hrtResetDevice(s32 deviceLogicId);

HcclResult hrtMalloc(void **devPtr, u64 size, bool level2Address = false);

HcclResult hrtMemcpyAddrAsync(void *dst, uint64_t destMax, uint64_t destOffset, const void *src, uint64_t count,
                              uint64_t srcOffset, rtStream_t stream);

HcclResult hrtMallocHost(void **hostPtr, u64 size);

HcclResult hrtFreeHost(void *hostPtr);

HcclResult hcclStreamSynchronize(void *stream);

HcclResult hrtEventCreate(aclrtEvent *event);
HcclResult hrtEventRecord(aclrtEvent event, aclrtStream stream);
HcclResult hrtEventQuery(aclrtEvent event);
HcclResult hrtEventDestroy(aclrtEvent event);

HcclResult hrtStreamCreateWithFlags(aclrtStream *stream, int32_t priority, uint32_t flags);

extern rtError_t rtMemcpyAsync(void *dst, uint64_t dest_max, const void *src, uint64_t count, rtMemcpyKind_t kind,
                               rtStream_t stream);

#endif
