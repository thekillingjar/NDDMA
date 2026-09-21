/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_TESTS_ST_STUBS_UTILS_MOCK_RUNTIME_H_
#define AIR_TESTS_ST_STUBS_UTILS_MOCK_RUNTIME_H_

#include <memory>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "runtime/kernel.h"
#include "runtime/rt.h"
#include "ge_running_env/fake_ns.h"
#include "tests/depends/runtime/src/runtime_stub.h"

namespace ge {
class MockRuntime : public RuntimeStub {
 public:
  MOCK_METHOD4(rtKernelLaunchEx, int32_t(void *, uint32_t, uint32_t, rtStream_t));
  MOCK_METHOD6(rtKernelLaunchWithFlag, int32_t(const void *stubFunc, uint32_t blockDim, rtArgsEx_t *arg_ex,
                                               rtSmDesc_t *smDesc, rtStream_t stream, uint32_t flags));

  MOCK_METHOD7(rtCpuKernelLaunchWithFlag,
               int32_t(const void *so_name, const void *kernel_name, uint32_t core_dim, const rtArgsEx_t *args,
                       rtSmDesc_t *smDesc, rtStream_t stream_, uint32_t flags));
  MOCK_METHOD6(rtAicpuKernelLaunchWithFlag, int32_t(const rtKernelLaunchNames_t *, uint32_t,
                                                    const rtArgsEx_t *, rtSmDesc_t *, rtStream_t, uint32_t));

  MOCK_METHOD7(rtKernelLaunchWithHandle, int32_t(void *, const uint64_t, uint32_t, rtArgsEx_t *,
                                                 rtSmDesc_t *, rtStream_t, const void *));

  MOCK_METHOD5(rtMemcpy, int32_t(void *, uint64_t, const void *, uint64_t, rtMemcpyKind_t));
  MOCK_METHOD5(rtKernelGetAddrAndPrefCntV2, int32_t(void *handle, const uint64_t tilingKey, const void *const stubFunc,
                                                    const uint32_t flag, rtKernelDetailInfo_t *kernelInfo));
  MOCK_METHOD3(rtsBinaryLoadFromFile, int32_t(const char * const binPath, const rtLoadBinaryConfig_t *const optionalCfg,
                                             rtBinHandle *binHandle));
  MOCK_METHOD5(rtsLaunchCpuKernel, int32_t(const rtFuncHandle funcHandle, const uint32_t blockDim, rtStream_t st,
                                           const rtKernelLaunchCfg_t *cfg, rtCpuKernelArgs_t *argsInfo));

  MOCK_METHOD4(rtsBinaryLoadFromData, int32_t(const void * const data, const uint64_t length, const rtLoadBinaryConfig_t * const optionalCfg,
                                              rtBinHandle *handle));
  MOCK_METHOD4(rtsRegisterCpuFunc, int32_t(const rtBinHandle binHandle, const char_t * const funcName,
                                           const char_t * const kernelName, rtFuncHandle *funcHandle));
  MOCK_METHOD8(rtsLaunchKernelWithHostArgs, int32_t(rtFuncHandle funcHandle, uint32_t blockDim, rtStream_t stm,
                                                    rtKernelLaunchCfg_t *cfg, void *hostArgs, uint32_t argsSize,
                                                    rtPlaceHolderInfo_t *placeHolderArray, uint32_t placeHolderNum));
};

std::shared_ptr<MockRuntime> MockForKernelLaunchWithHostMemInput();
std::shared_ptr<MockRuntime> MockForKernelLaunchForStaticShapeUseBinary();
std::shared_ptr<MockRuntime> MockForKernelGetAddrAndPrefCntV2();
std::shared_ptr<MockRuntime> MockForKernelLaunchExFailed();
rtError_t MockRtKernelLaunchEx(void *args, uint32_t args_size, uint32_t flags, rtStream_t stream_);
rtError_t MockRtKernelLaunchExFailed(void *args, uint32_t args_size, uint32_t flags, rtStream_t stream_);
rtError_t MockRtMemcpy(void *dst, uint64_t dest_max, const void *src, uint64_t count, rtMemcpyKind_t kind);
rtError_t MockRtKernelGetAddrAndPrefCntV2(void *handle, const uint64_t tilingKey, const void *const stubFunc,
                                          const uint32_t flag, rtKernelDetailInfo_t *kernelInfo);
constexpr uint8_t kHostMemInputValue = 110U;
}
#endif  // AIR_TESTS_ST_STUBS_UTILS_MOCK_RUNTIME_H_
