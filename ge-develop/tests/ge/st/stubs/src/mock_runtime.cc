/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "utils/mock_runtime.h"
#include "securec.h"
#include "aicpu_task_struct.h"
#include "graph/def_types.h"
#include "fwk_adpt_struct.h"

namespace ge {

rtError_t MockRtKernelLaunchEx(void *args, uint32_t args_size, uint32_t flags, rtStream_t stream_) {
  return RT_ERROR_NONE;
}

rtError_t MockRtKernelLaunchExFailed(void *args, uint32_t args_size, uint32_t flags, rtStream_t stream_) {
  return -1;
}

rtError_t MockRtKernelLaunchExForHostMemAicpuTfKernel(void *args, uint32_t args_size, uint32_t flags,
                                                      rtStream_t stream_) {
  EXPECT_GE(args_size, sizeof(aicpu::AicpuParamHead));
  return RT_ERROR_NONE;
}

int32_t CheckArgs(const rtArgsEx_t *const args) {
  if (args->hostInputInfoNum == 0U) {
    return 0;
  }
  EXPECT_NE(args->hostInputInfoPtr, nullptr);
  // check offset
  for (size_t i = 0; i < args->hostInputInfoNum; i++) {
    rtHostInputInfo_t &host_info = args->hostInputInfoPtr[i];
    uint64_t data_ptr = PtrToValue(args->args) + host_info.dataOffset;

    uint64_t *value_ptr = PtrToPtr<void, uint64_t>(ValueToPtr(data_ptr));
    EXPECT_EQ(*value_ptr, kHostMemInputValue);

    uint64_t *addr_ptr = reinterpret_cast<uint64_t *>(reinterpret_cast<uint8_t *>(args->args) + host_info.addrOffset);
    EXPECT_EQ(*addr_ptr, PtrToValue(value_ptr));
  }
  return 0;
}

int32_t CheckStaticShapeUseBinaryArgs(const rtArgsEx_t *const args) {

  const std::string tiling_data_str = "666";
  EXPECT_EQ(args->hasTiling, true);
  uint64_t data_ptr = PtrToValue(args->args) + args->tilingDataOffset;
  char *value_ptr = PtrToPtr<void, char>(ValueToPtr(data_ptr));
  std::string tmp(value_ptr, tiling_data_str.size());
  EXPECT_EQ(tmp, tiling_data_str);

  return 0;
}

rtError_t MockRtKernelLaunchWithFlagForHostMem(const void *stubFunc, uint32_t blockDim, rtArgsEx_t *args,
                                               rtSmDesc_t *smDesc, rtStream_t stream, uint32_t flags) {
  // check offset
  EXPECT_EQ(CheckArgs(args), 0);
  return RT_ERROR_NONE;
}

rtError_t MockRtCpuKernelLaunchWithFlagForHostMem(const void *so_name, const void *kernel_name, uint32_t core_dim,
                                                  const rtArgsEx_t *args, rtSmDesc_t *smDesc, rtStream_t stream_,
                                                  uint32_t flags) {
  // check offset
  EXPECT_EQ(CheckArgs(args), 0);
  return RT_ERROR_NONE;
}

rtError_t MockRtAicpuKernelLaunchWithFlagForHostMem(const rtKernelLaunchNames_t *launchNames, uint32_t blockDim,
                                                    const rtArgsEx_t *args, rtSmDesc_t *smDesc, rtStream_t stream,
                                                    uint32_t flags) {
  // check offset
  EXPECT_EQ(CheckArgs(args), 0);
  return RT_ERROR_NONE;
}

rtError_t MockRtKernelLaunchWithHandle(void *handle, const uint64_t tilingkey, uint32_t blockDim, rtArgsEx_t *args,
                                   rtSmDesc_t *smDesc, rtStream_t stream, const void *kernelInfo) {
  // check offset
  EXPECT_EQ(CheckStaticShapeUseBinaryArgs(args), 0);
  return RT_ERROR_NONE;
}

rtError_t MockRtKernelGetAddrAndPrefCntV2(void *handle, const uint64_t tilingKey, const void *const stubFunc,
                                          const uint32_t flag, rtKernelDetailInfo_t *kernelInfo) {
  kernelInfo->functionInfoNum = 2;
  kernelInfo->functionInfo[0].pcAddr = (void *)(0x1245);
  kernelInfo->functionInfo[0].prefetchCnt = 1;
  kernelInfo->functionInfo[1].pcAddr = (void *)(0x1234);
  kernelInfo->functionInfo[1].prefetchCnt = 2;
  return RT_ERROR_NONE;
}

rtError_t MockRtMemcpy(void *dst, uint64_t dest_max, const void *src, uint64_t count, rtMemcpyKind_t kind) {
  if (count == 0) {
    return RT_ERROR_NONE;
  }
  if (count == sizeof(aicpu::FWKAdapter::ResultSummary) && kind == RT_MEMCPY_DEVICE_TO_HOST) {
    aicpu::FWKAdapter::ResultSummary summary{};
    summary.shape_data_size = 0;
    summary.raw_data_size = 0;
    return memcpy_s(dst, dest_max, &summary, count);
  } else {
    return memcpy_s(dst, dest_max, src, count);
  }
}

std::shared_ptr<MockRuntime> MockForKernelLaunchWithHostMemInput() {
  auto runtime_stub = std::make_shared<MockRuntime>();
  RuntimeStub::SetInstance(runtime_stub);
  EXPECT_CALL(*runtime_stub, rtKernelLaunchEx)
      .WillRepeatedly(testing::Invoke(MockRtKernelLaunchExForHostMemAicpuTfKernel));
  EXPECT_CALL(*runtime_stub, rtKernelLaunchWithFlag)
      .WillRepeatedly(testing::Invoke(MockRtKernelLaunchWithFlagForHostMem));
  EXPECT_CALL(*runtime_stub, rtCpuKernelLaunchWithFlag)
      .WillRepeatedly(testing::Invoke(MockRtCpuKernelLaunchWithFlagForHostMem));
  EXPECT_CALL(*runtime_stub, rtAicpuKernelLaunchWithFlag)
      .WillRepeatedly(testing::Invoke(MockRtAicpuKernelLaunchWithFlagForHostMem));

  EXPECT_CALL(*runtime_stub, rtMemcpy).WillRepeatedly(testing::Invoke(MockRtMemcpy));
  return runtime_stub;
}

std::shared_ptr<MockRuntime> MockForKernelLaunchForStaticShapeUseBinary() {
  auto runtime_stub = std::make_shared<MockRuntime>();
  RuntimeStub::SetInstance(runtime_stub);
  EXPECT_CALL(*runtime_stub, rtKernelLaunchWithHandle)
      .WillRepeatedly(testing::Invoke(MockRtKernelLaunchWithHandle));
  EXPECT_CALL(*runtime_stub, rtMemcpy).WillRepeatedly(testing::Invoke(MockRtMemcpy));
  return runtime_stub;
}

std::shared_ptr<MockRuntime> MockForKernelGetAddrAndPrefCntV2() {
  auto runtime_stub = std::make_shared<MockRuntime>();
  RuntimeStub::SetInstance(runtime_stub);
  EXPECT_CALL(*runtime_stub,
              rtKernelGetAddrAndPrefCntV2).WillRepeatedly(testing::Invoke(MockRtKernelGetAddrAndPrefCntV2));
  return runtime_stub;
}

std::shared_ptr<MockRuntime> MockForKernelLaunchExFailed() {
  auto runtime_stub = std::make_shared<MockRuntime>();
  RuntimeStub::SetInstance(runtime_stub);
  EXPECT_CALL(*runtime_stub, rtKernelLaunchEx).WillRepeatedly(testing::Invoke(MockRtKernelLaunchExFailed));
  return runtime_stub;
}
}
