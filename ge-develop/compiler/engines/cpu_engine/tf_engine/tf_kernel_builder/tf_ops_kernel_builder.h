/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_TF_OPS_KERNEL_BUILDER_H_
#define AICPU_TF_OPS_KERNEL_BUILDER_H_
#include <vector>
#include <memory>
#include "common/util/constant.h"
#include "common/opskernel/ops_kernel_builder.h"
#include "common/aicpu_ops_kernel_builder/kernel_builder.h"

namespace aicpu {
using KernelBuilderPtr = std::shared_ptr<KernelBuilder>;

class AicpuTfOpsKernelBuilder : public ge::OpsKernelBuilder {
 public:
  /**
   * Contructor
   */
  explicit AicpuTfOpsKernelBuilder() = default;

  /**
   * Destructor
   */
  virtual ~AicpuTfOpsKernelBuilder() = default;

  /**
   * Initialize related resources of the aicpu kernelinfo store
   * @return status whether this operation success
   */
  ge::Status Initialize(const std::map<std::string, std::string> &options) override;

  /**
   * Release related resources of the aicpu kernelinfo store
   * @return status whether this operation success
   */
  ge::Status Finalize() override;

  /**
   * Calc the running size of Operator,then GE will alloc the memsize from runtime
   * @param node Node information, return task_memsize in node's attr
   * @return status whether this operation success
   */
  ge::Status CalcOpRunningParam(ge::Node &node) override;

  /**
   * Copy the data from host to device, then address is allocated by GE, then invoked
   * the runtime's interface to generate the task
   * @param ge_node Node information
   * @param context Run context from Ge
   * @param tasks task vector
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
  ge::Status GenSingleOpRunTask(const ge::NodePtr &node, STR_FWK_OP_KERNEL &task, string &task_info) override;

  /**
  * Generate the task
  * @param count the memcopy times
  * @param task[out]
  * @param task_info[out]
  * @return status whether this operation success
  */
  ge::Status GenMemCopyTask(uint64_t count, STR_FWK_OP_KERNEL &task, string &task_info) override;

  // Copy prohibited
  AicpuTfOpsKernelBuilder(const AicpuTfOpsKernelBuilder &aicpu_tf_ops_kernel_builder) = delete;

  // Move prohibited
  AicpuTfOpsKernelBuilder(const AicpuTfOpsKernelBuilder &&aicpu_tf_ops_kernel_builder) = delete;

  // Copy prohibited
  AicpuTfOpsKernelBuilder &operator=(const AicpuTfOpsKernelBuilder &aicpu_tf_ops_kernel_builder) = delete;

  // Move prohibited
  AicpuTfOpsKernelBuilder &operator=(AicpuTfOpsKernelBuilder &&aicpu_tf_ops_kernel_builder) = delete;

 private:
  std::string engine_name_ = kTfEngine;
  // kernel_util map
  std::map<std::string, KernelBuilderPtr> kernel_builder_map_;
};
} // namespace aicpu

#endif // AICPU_OPS_KERNEL_BUILDER_H_