/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_AICPU_OPS_KERNEL_BUILDER_H_
#define AICPU_AICPU_OPS_KERNEL_BUILDER_H_

#include <memory>
#include <vector>

#include "aicpu_ops_kernel_info_store/op_struct.h"
#include "common/opskernel/ops_kernel_builder.h"

namespace aicpu {
class KernelBuilder;
using KernelBuilderPtr = std::shared_ptr<KernelBuilder>;

class AicpuOpsKernelBuilder : public ge::OpsKernelBuilder {
 public:
  /**
   * Contructor
   */
  explicit AicpuOpsKernelBuilder(const std::string &engine_name) : engine_name_(engine_name) {}

  /**
   * Destructor
   */
  virtual ~AicpuOpsKernelBuilder() = default;

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
  ge::Status GenerateTask(const ge::Node &node, ge::RunContext &context,
                          std::vector<domi::TaskDef> &tasks) override;
  ge::Status UpdateTask(const ge::Node &node, std::vector<domi::TaskDef> &tasks) override;
  /**
   * Generate the task
   * @param node Node information
   * @param task[out]
   * @param task_info[out]
   * @return status whether this operation success
   */
  ge::Status GenSingleOpRunTask(const ge::NodePtr &node,
                                STR_FWK_OP_KERNEL &task,
                                string &task_info) override;

  /**
   * Generate the task
   * @param count the memcopy times
   * @param task[out]
   * @param task_info[out]
   * @return status whether this operation success
   */
  ge::Status GenMemCopyTask(uint64_t count, STR_FWK_OP_KERNEL &task,
                            string &task_info) override;

  // Copy prohibited
  AicpuOpsKernelBuilder(const AicpuOpsKernelBuilder &aicpu_ops_kernel_builder) = delete;

  // Move prohibited
  AicpuOpsKernelBuilder(const AicpuOpsKernelBuilder &&aicpu_ops_kernel_builder) = delete;

  // Copy prohibited
  AicpuOpsKernelBuilder &operator=(
      const AicpuOpsKernelBuilder &aicpu_ops_kernel_builder) = delete;

  // Move prohibited
  AicpuOpsKernelBuilder &operator=(
      AicpuOpsKernelBuilder &&aicpu_ops_kernel_builder) = delete;

 private:
  // Get internal kernel builder by op type
  KernelBuilderPtr GetKernelBuilderByName(const std::string &kernel_name);

  /**
   * Get op info
   * @param all_op_info store op information
   * @return status whether this operation success
   */
  ge::Status GetOpsInfo(
      std::map<std::string, aicpu::OpFullInfo> &all_op_info) const;

 private:
  std::string engine_name_;

  // kernel_util map
  std::map<std::string, KernelBuilderPtr> kernel_builder_map_;
};
}  // namespace aicpu

#endif  // AICPU_AICPU_OPS_KERNEL_BUILDER_H_
