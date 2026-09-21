/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_BASE_COMMON_RUNTIME_API_WRAPPER_H_
#define AIR_CXX_BASE_COMMON_RUNTIME_API_WRAPPER_H_
#include "runtime/rt_ffts_plus.h"
#include "runtime/rt_stars.h"
#include "runtime/mem.h"

namespace ge {
rtError_t rtFftsPlusTaskLaunchWithFlag(const rtFftsPlusTaskInfo_t *const fftsPlusTaskInfo, const void *const stm,
                                       const uint32_t flag);

rtError_t rtFftsPlusTaskLaunch(const rtFftsPlusTaskInfo_t *const fftsPlusTaskInfo, const void *const stm);

rtError_t rtNpuGetFloatStatus(const void *const outputAddr, const uint64_t outputSize, const uint32_t checkMode,
                              const void *const stm);

rtError_t rtNpuClearFloatStatus(const uint32_t checkMode, const void *const stm);

rtError_t rtNpuGetFloatDebugStatus(const void *const outputAddr, const uint64_t outputSize, const uint32_t checkMode,
                                   const void *const stm);

rtError_t rtNpuClearFloatDebugStatus(const uint32_t checkMode, const void *const stm);

rtError_t rtStarsTaskLaunchWithFlag(const void *const taskSqe, const uint32_t sqeLen, const void *const stm,
                                    const uint32_t flag);

rtError_t rtStarsTaskLaunch(const void *const taskSqe, const uint32_t sqeLen, const void *const stm);

rtError_t rtCmoTaskLaunch(const rtCmoTaskInfo_t *const taskInfo, const void *const stm, const uint32_t flag);

rtError_t rtBarrierTaskLaunch(const rtBarrierTaskInfo_t *const taskInfo, const void *const stm, const uint32_t flag);

struct MemcpyAsyncMemInfo {
  void *dst;
  uint64_t destMax;
  const void *src;
  uint64_t cnt;
};
rtError_t rtMemcpyAsyncWithCfg(const MemcpyAsyncMemInfo &meminfo, const rtMemcpyKind_t kind, const void *const stm,
                               const uint32_t qosCfg);

rtError_t rtSetStreamTag(const void *const stm, const uint32_t tag);
}  // namespace ge
#endif  // AIR_CXX_BASE_COMMON_RUNTIME_API_WRAPPER_H_
