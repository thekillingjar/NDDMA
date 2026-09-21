/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "stub/runtime_stub_for_kernel_v2.h"

namespace gert {
rtError_t RuntimeStubForKernelV2::rtsBinaryLoadFromFile(const char * const binPath,
    const rtLoadBinaryConfig_t *const optionalCfg, rtBinHandle *binHandle) {
  *binHandle = reinterpret_cast<void *>(static_cast<uintptr_t>(stub_bin_addr_));
  bin_handle_store_.emplace(*binHandle);
  stub_bin_addr_++;
  return RT_ERROR_NONE;
}

rtError_t RuntimeStubForKernelV2::rtsBinaryLoadFromData(const void * const data, const uint64_t length,
    const rtLoadBinaryConfig_t * const optionalCfg, rtBinHandle *handle) {
  *handle = reinterpret_cast<void *>(static_cast<uintptr_t>(stub_bin_addr_));
  bin_handle_store_.emplace(*handle);
  stub_bin_addr_++;
  return RT_ERROR_NONE;
}

rtError_t RuntimeStubForKernelV2::rtsFuncGetByEntry(const rtBinHandle binHandle, const uint64_t funcEntry,
    rtFuncHandle *funcHandle) {
  if (bin_handle_store_.find(binHandle) != bin_handle_store_.end()) {
    *funcHandle = reinterpret_cast<void *>(static_cast<uintptr_t>(stub_func_addr_));
    stub_func_addr_++;
    return RT_ERROR_NONE;
  }
  return -1;
}

rtError_t RuntimeStubForKernelV2::rtsRegisterCpuFunc(rtBinHandle binHandle, const char_t * const funcName,
    const char_t * const kernelName, rtFuncHandle *funcHandle) {
  if (bin_handle_store_.find(binHandle) != bin_handle_store_.end()) {
    *funcHandle = reinterpret_cast<void *>(static_cast<uintptr_t>(stub_func_addr_));
    stub_func_addr_++;
    return RT_ERROR_NONE;
  }
  return -1;
}

rtError_t RuntimeStubForKernelV2::rtsBinaryUnload(const rtBinHandle binHandle) {
  bin_handle_store_.erase(binHandle);
  return RT_ERROR_NONE;
}

rtError_t RuntimeStubForKernelV2::rtsFuncGetByName(const rtBinHandle binHandle, const char *kernelName,
    rtFuncHandle *funcHandle) {
  if (bin_handle_store_.find(binHandle) != bin_handle_store_.end()) {
    *funcHandle = reinterpret_cast<void *>(static_cast<uintptr_t>(stub_func_addr_));
    stub_func_addr_++;
    return RT_ERROR_NONE;
  }
  return -1;
}
}