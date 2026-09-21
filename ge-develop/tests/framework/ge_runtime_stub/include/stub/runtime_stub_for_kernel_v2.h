/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_SRC_RUNTIME_STUB_FOR_KERNEL_V2_H_
#define AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_SRC_RUNTIME_STUB_FOR_KERNEL_V2_H_

#include <set>
#include "runtime_stub_impl.h"

namespace gert {
class RuntimeStubForKernelV2 : public RuntimeStubImpl {
 public:
  rtError_t rtsBinaryLoadFromFile(const char * const binPath, const rtLoadBinaryConfig_t *const optionalCfg,
      rtBinHandle *binHandle) override;
  rtError_t rtsBinaryLoadFromData(const void * const data, const uint64_t length,
      const rtLoadBinaryConfig_t * const optionalCfg, rtBinHandle *handle) override;

  rtError_t rtsFuncGetByEntry(const rtBinHandle binHandle, const uint64_t funcEntry,
      rtFuncHandle *funcHandle) override;

  rtError_t rtsRegisterCpuFunc(rtBinHandle binHandle, const char_t * const funcName,
      const char_t * const kernelName, rtFuncHandle *funcHandle) override;

  rtError_t rtsBinaryUnload(const rtBinHandle binHandle) override;

  rtError_t rtsLaunchKernelWithDevArgs(rtFuncHandle funcHandle, uint32_t blockDim, rtStream_t stm,
      rtKernelLaunchCfg_t *cfg, const void *args, uint32_t argsSize, void *reserve) override {
    return RT_ERROR_NONE;
  }
  rtError_t rtsFuncGetByName(const rtBinHandle binHandle, const char *kernelName,
      rtFuncHandle *funcHandle) override;
  rtError_t rtsGetHardwareSyncAddr(void **addr) override {
    *addr = reinterpret_cast<void *>(static_cast<uintptr_t>(ffts_addr_));
    return RT_ERROR_NONE;
  };
 private:
  int64_t ffts_addr_{0xfff3243};
  uint64_t stub_bin_addr_{0x1200};
  uint64_t stub_func_addr_{0x1600};
  std::set<rtBinHandle> bin_handle_store_;
};
}
#endif // AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_SRC_RUNTIME_STUB_FOR_KERNEL_V2_H_