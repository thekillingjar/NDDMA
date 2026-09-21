/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_HOST_CPU_OPS_KERNEL_BUILDER_H_
#define AICPU_HOST_CPU_OPS_KERNEL_BUILDER_H_

#include <memory>
#include <vector>
#include "common/aicpu_ops_kernel_builder/kernel_builder.h"
#include "common/opskernel/ops_kernel_builder.h"
#include "common/util/constant.h"

namespace aicpu {
using KernelBuilderPtr = std::shared_ptr<KernelBuilder>;

class HostCpuOpsKernelBuilder : public ge::OpsKernelBuilder {
 public:
  /**
   * Contructor
   */
  explicit HostCpuOpsKernelBuilder() : engine_name_(kHostCpuEngine) {}

  /**
   * Destructor
   */
  virtual ~HostCpuOpsKernelBuilder() = default;

  /**
   * Initialize related resources of the aicpu kernelinfo store
   * @return status whether this operation success
   */
  ge::Status Initialize(
      const std::map<std::string, std::string> &options) override;

  /**
   * Release related resources of the aicpu kernelinfo store
   * @return status whether this operation success
   */
  ge::Status Finalize() override;

  /**
   * Calc the running size of Operator,then GE will alloc the memsize from
   * runtime
   * @param node Node information, return task_memsize in node's attr
   * @return status whether this operation success
   */
  ge::Status CalcOpRunningParam(ge::Node &node) override;

  /**
   * Copy the data from host to device, then address is allocated by GE, then
   * invoked the runtime's interface to generate the task
   * @param node Node information
   * @param runContext Run context from Ge
   * @return task memsize in node
   */
  ge::Status GenerateTask(const ge::Node &ge_node, ge::RunContext &context,
                          std::vector<domi::TaskDef> &tasks) override;

  ge::Status UpdateTask(const ge::Node &node, std::vector<domi::TaskDef> &tasks) override;

  // Copy prohibited
  HostCpuOpsKernelBuilder(const HostCpuOpsKernelBuilder
                          &hostcpu_ops_kernel_builder) = delete;

  // Move prohibited
  HostCpuOpsKernelBuilder(const HostCpuOpsKernelBuilder
                          &&hostcpu_ops_kernel_builder) = delete;

  // Copy prohibited
  HostCpuOpsKernelBuilder &operator=(
      const HostCpuOpsKernelBuilder &hostcpu_ops_kernel_builder) = delete;

  // Move prohibited
  HostCpuOpsKernelBuilder &operator=(
      HostCpuOpsKernelBuilder &&hostcpu_ops_kernel_builder) = delete;
 private:
  std::string engine_name_;

  // kernel_util map
  std::map<std::string, KernelBuilderPtr> kernel_builder_map_;
};
}  // namespace aicpu

#endif  // AICPU_HOST_CPU_OPS_KERNEL_BUILDER_H_
