/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "runtime_api_wrapper.h"
#include <array>
#include "graph/def_types.h"

namespace ge {
rtError_t rtFftsPlusTaskLaunchWithFlag(const rtFftsPlusTaskInfo_t *const fftsPlusTaskInfo, const void *const stm,
                                       const uint32_t flag) {
  constexpr size_t size = 3U;
  std::array<uintptr_t, size> inputs = {
      static_cast<uintptr_t>(PtrToValue(PtrToPtr<rtFftsPlusTaskInfo_t, void>(fftsPlusTaskInfo))),
      static_cast<uintptr_t>(PtrToValue(stm)), static_cast<uintptr_t>(flag)};
  return rtGeneralCtrl(&(inputs[0U]), static_cast<uint32_t>(size), RT_GNL_CTRL_TYPE_FFTS_PLUS_FLAG);
}
rtError_t rtFftsPlusTaskLaunch(const rtFftsPlusTaskInfo_t *const fftsPlusTaskInfo, const void *const stm) {
  constexpr size_t size = 2U;
  std::array<uintptr_t, size> inputs = {
      static_cast<uintptr_t>(PtrToValue(PtrToPtr<rtFftsPlusTaskInfo_t, void>(fftsPlusTaskInfo))),
      static_cast<uintptr_t>(PtrToValue(stm))};
  return rtGeneralCtrl(&(inputs[0U]), static_cast<uint32_t>(size), RT_GNL_CTRL_TYPE_FFTS_PLUS);
}
rtError_t rtNpuGetFloatStatus(const void *const outputAddr, const uint64_t outputSize, const uint32_t checkMode,
                              const void *const stm) {
  constexpr size_t size = 4U;
  std::array<uintptr_t, size> inputs = {static_cast<uintptr_t>(PtrToValue(outputAddr)),
                                        static_cast<uintptr_t>(outputSize), static_cast<uintptr_t>(checkMode),
                                        static_cast<uintptr_t>(PtrToValue(stm))};
  return rtGeneralCtrl(&(inputs[0U]), static_cast<uint32_t>(size), RT_GNL_CTRL_TYPE_NPU_GET_FLOAT_STATUS);
}
rtError_t rtNpuClearFloatStatus(const uint32_t checkMode, const void *const stm) {
  constexpr size_t size = 2U;
  std::array<uintptr_t, size> inputs = {static_cast<uintptr_t>(checkMode), static_cast<uintptr_t>(PtrToValue(stm))};
  return rtGeneralCtrl(&(inputs[0U]), static_cast<uint32_t>(size), RT_GNL_CTRL_TYPE_NPU_CLEAR_FLOAT_STATUS);
}
rtError_t rtNpuGetFloatDebugStatus(const void *const outputAddr, const uint64_t outputSize, const uint32_t checkMode,
                                   const void *const stm) {
  constexpr size_t size = 4U;
  std::array<uintptr_t, size> inputs = {static_cast<uintptr_t>(PtrToValue(outputAddr)),
                                        static_cast<uintptr_t>(outputSize), static_cast<uintptr_t>(checkMode),
                                        static_cast<uintptr_t>(PtrToValue(stm))};
  return rtGeneralCtrl(&(inputs[0U]), static_cast<uint32_t>(size), RT_GNL_CTRL_TYPE_NPU_GET_FLOAT_DEBUG_STATUS);
}
rtError_t rtNpuClearFloatDebugStatus(const uint32_t checkMode, const void *const stm) {
  constexpr size_t size = 2U;
  std::array<uintptr_t, size> inputs = {static_cast<uintptr_t>(checkMode), static_cast<uintptr_t>(PtrToValue(stm))};
  return rtGeneralCtrl(&(inputs[0U]), static_cast<uint32_t>(size), RT_GNL_CTRL_TYPE_NPU_CLEAR_FLOAT_DEBUG_STATUS);
}
rtError_t rtStarsTaskLaunch(const void *const taskSqe, const uint32_t sqeLen, const void *const stm) {
  constexpr size_t size = 3U;
  std::array<uintptr_t, size> inputs = {static_cast<uintptr_t>(PtrToValue(taskSqe)), static_cast<uintptr_t>(sqeLen),
                                        static_cast<uintptr_t>(PtrToValue(stm))};
  return rtGeneralCtrl(&(inputs[0U]), static_cast<uint32_t>(size), RT_GNL_CTRL_TYPE_STARS_TSK);
}
rtError_t rtStarsTaskLaunchWithFlag(const void *const taskSqe, const uint32_t sqeLen, const void *const stm,
                                    const uint32_t flag) {
  constexpr size_t size = 4U;
  std::array<uintptr_t, size> inputs = {static_cast<uintptr_t>(PtrToValue(taskSqe)), static_cast<uintptr_t>(sqeLen),
                                        static_cast<uintptr_t>(PtrToValue(stm)), static_cast<uintptr_t>(flag)};
  return rtGeneralCtrl(&(inputs[0U]), static_cast<uint32_t>(size), RT_GNL_CTRL_TYPE_STARS_TSK_FLAG);
}
rtError_t rtCmoTaskLaunch(const rtCmoTaskInfo_t *const taskInfo, const void *const stm, const uint32_t flag) {
  constexpr size_t size = 3U;
  std::array<uintptr_t, size> inputs = {static_cast<uintptr_t>(PtrToValue(PtrToPtr<rtCmoTaskInfo_t, void>(taskInfo))),
                                        static_cast<uintptr_t>(PtrToValue(stm)), static_cast<uintptr_t>(flag)};
  return rtGeneralCtrl(&(inputs[0U]), static_cast<uint32_t>(size), RT_GNL_CTRL_TYPE_CMO_TSK);
}
rtError_t rtBarrierTaskLaunch(const rtBarrierTaskInfo_t *const taskInfo, const void *const stm, const uint32_t flag) {
  constexpr size_t size = 3U;
  std::array<uintptr_t, size> inputs = {
      static_cast<uintptr_t>(PtrToValue(PtrToPtr<rtBarrierTaskInfo_t, void>(taskInfo))),
      static_cast<uintptr_t>(PtrToValue(stm)), static_cast<uintptr_t>(flag)};
  return rtGeneralCtrl(&(inputs[0U]), static_cast<uint32_t>(size), RT_GNL_CTRL_TYPE_BARRIER_TSK);
}

rtError_t rtMemcpyAsyncWithCfg(const MemcpyAsyncMemInfo &meminfo, const rtMemcpyKind_t kind, const void *const stm,
                               const uint32_t qosCfg) {
  constexpr size_t size = 7U;
  std::array<uintptr_t, size> inputs = {reinterpret_cast<uintptr_t>(meminfo.dst),
                                        static_cast<uintptr_t>(meminfo.destMax),
                                        reinterpret_cast<uintptr_t>(meminfo.src),
                                        static_cast<uintptr_t>(meminfo.cnt),
                                        static_cast<uintptr_t>(kind),
                                        reinterpret_cast<uintptr_t>(stm),
                                        static_cast<uintptr_t>(qosCfg)};
  return rtGeneralCtrl(&(inputs[0U]), static_cast<uint32_t>(size), RT_GNL_CTRL_TYPE_MEMCPY_ASYNC_CFG);
}

rtError_t rtSetStreamTag(const void *const stm, const uint32_t tag) {
  constexpr size_t size = 2U;
  std::array<uintptr_t, size> inputs = {static_cast<uintptr_t>(PtrToValue(stm)), static_cast<uintptr_t>(tag)};
  return rtGeneralCtrl(&(inputs[0U]), static_cast<uint32_t>(size),
                       static_cast<uint32_t>(RT_GNL_CTRL_TYPE_STARS_TSK_FLAG) + 1U);
}
}  // namespace ge
