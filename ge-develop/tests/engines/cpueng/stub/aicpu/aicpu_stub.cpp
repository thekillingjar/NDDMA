/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cpu_context.h"
#include "cpu_kernel_register.h"
#include "cpu_kernel.h"

namespace aicpu {
CpuKernelRegister &CpuKernelRegister::Instance() {
  static CpuKernelRegister instance;
  return instance;
}

class TestCpuKernel : public CpuKernel {
 public:
  TestCpuKernel() = default;
  ~TestCpuKernel() = default;
  uint32_t Compute(CpuKernelContext &ctx) override {
    return 0;
  };
};

std::shared_ptr<CpuKernel> CpuKernelRegister::GetCpuKernel(const std::string &opType) {
  std::shared_ptr<CpuKernel> testCpuKernel = MakeShared<TestCpuKernel>();
  return testCpuKernel;
}

uint32_t CpuKernelRegister::RunCpuKernel(CpuKernelContext &ctx) {
  return 0;
}

std::vector<std::string> CpuKernelRegister::GetAllRegisteredOpTypes() const {
  std::vector<std::string> ret;
  ret.push_back("testop");
  return ret;
}

CpuKernelContext::CpuKernelContext(DeviceType type) {
}

uint32_t CpuKernelContext::Init(void *nodeDef) {
  return 0;
}
}