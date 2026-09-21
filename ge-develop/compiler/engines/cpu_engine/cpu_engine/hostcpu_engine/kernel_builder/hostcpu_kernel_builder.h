/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_HOST_CPU_KERNEL_BUILDER_H_
#define AICPU_HOST_CPU_KERNEL_BUILDER_H_

#include "common/kernel_builder/cpu_kernel_builder.h"

namespace aicpu {
using KernelBuilderPtr = std::shared_ptr<KernelBuilder>;

class HostCpuKernelBuilder : public CpuKernelBuilder {
 public:
  /**
   * Singleton access method, Thread safe
   * @return return unique instance in global scope
   */
  static KernelBuilderPtr Instance();

  /**
   * Destructor
   */
  virtual ~HostCpuKernelBuilder() = default;

 private:
  /**
   * Constructor
   */
  HostCpuKernelBuilder() = default;

  // Copy prohibited
  HostCpuKernelBuilder(const HostCpuKernelBuilder &hostcpu_kernel_builder) = delete;

  // Move prohibited
  HostCpuKernelBuilder(const HostCpuKernelBuilder &&hostcpu_kernel_builder) = delete;

  // Copy prohibited
  HostCpuKernelBuilder &operator=(
      const HostCpuKernelBuilder &hostcpu_kernel_builder) = delete;

  // Move prohibited
  HostCpuKernelBuilder &operator=(
      HostCpuKernelBuilder &&hostcpu_kernel_builder) = delete;

 private:
  // singleton instance
  static KernelBuilderPtr instance_;
};
}  // namespace aicpu
#endif  // AICPU_HOST_CPU_KERNEL_BUILDER_H_
