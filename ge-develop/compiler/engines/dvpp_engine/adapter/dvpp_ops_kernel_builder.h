/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DVPP_ENGINE_ADAPTER_DVPP_OPS_KERNEL_BUILDER_H_
#define DVPP_ENGINE_ADAPTER_DVPP_OPS_KERNEL_BUILDER_H_

#include "common/dvpp_builder.h"

namespace dvpp {
class DvppOpsKernelBuilder : public ge::OpsKernelBuilder {
 public:
  /**
   * @brief Construct
   */
  DvppOpsKernelBuilder() = default;

  /**
   * @brief Destructor
   */
  virtual ~DvppOpsKernelBuilder() = default;

  // Copy constructor prohibited
  DvppOpsKernelBuilder(
      const DvppOpsKernelBuilder& dvpp_ops_kernel_builder) = delete;

  // Move constructor prohibited
  DvppOpsKernelBuilder(
      const DvppOpsKernelBuilder&& dvpp_ops_kernel_builder) = delete;

  // Copy assignment prohibited
  DvppOpsKernelBuilder& operator=(
      const DvppOpsKernelBuilder& dvpp_ops_kernel_builder) = delete;

  // Move assignment prohibited
  DvppOpsKernelBuilder& operator=(
      DvppOpsKernelBuilder&& dvpp_ops_kernel_builder) = delete;

  /**
   * @brief initialize dvpp ops kernel builder
   * @param options initial options
   * @return status whether success
   */
  ge::Status Initialize(
      const std::map<std::string, std::string>& options) override;

  /**
   * @brief close dvpp ops kernel builder
   * @return status whether success
   */
  ge::Status Finalize() override;

  /**
   * @brief calculate running size of op
   *        then GE will alloc the memory from runtime
   * @param node node info, return task memory size in node attr
   * @return status whether success
   */
  ge::Status CalcOpRunningParam(ge::Node& node) override;

  /**
   * @brief make the task info details
   *        then GE will alloc the address and send task by runtime
   * @param node node info
   * @param context run context from GE
   * @param tasks make the task return to GE
   * @return status whether success
   */
  ge::Status GenerateTask(const ge::Node& node, ge::RunContext& context,
                          std::vector<domi::TaskDef>& tasks) override;

 private:
  std::shared_ptr<DvppBuilder> dvpp_builder_{nullptr};
}; // class DvppOpsKernelBuilder
} // namespace dvpp

#endif // DVPP_ENGINE_ADAPTER_DVPP_OPS_KERNEL_BUILDER_H_