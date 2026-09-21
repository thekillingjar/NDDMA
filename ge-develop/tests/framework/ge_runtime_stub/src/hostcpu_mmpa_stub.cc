/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "stub/hostcpu_mmpa_stub.h"
#include "aicpu_task_struct.h"

namespace gert {
namespace {
uint32_t RunHostCpuFuncStub(void *args) {
  auto *arg_base = reinterpret_cast<uint8_t *>(args);
  auto io_addrs = reinterpret_cast<uintptr_t *>(arg_base + sizeof(aicpu::AicpuParamHead));
  auto *input_0 = reinterpret_cast<int32_t *>(io_addrs[0]);
  auto *input_1 = reinterpret_cast<int32_t *>(io_addrs[1]);
  auto *output = reinterpret_cast<int32_t *>(io_addrs[2]);
  return 0;
}
} // namespace

void *HostcpuMockMmpa::DlSym(void *handle, const char *func_name) {
  if (std::string(func_name) == "RunHostCpuKernel") {
    return (void *) &RunHostCpuFuncStub;
  }
  return nullptr;
}

int32_t HostcpuMockMmpa::RealPath(const CHAR *path, CHAR *realPath, INT32 realPathLen) {
  strncpy(realPath, path, realPathLen);
  return EN_OK;
}

void *HostcpuMockMmpa::DlOpen(const char *file_name, int32_t mode) {
  return (void *)0x8888;
}
}  // namespace gert
