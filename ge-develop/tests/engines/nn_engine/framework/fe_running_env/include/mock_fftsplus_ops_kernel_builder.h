/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_TEST_ENGINES_NNENG_FRAMEWORK_FE_RUNNING_ENV_INCLUDE_MOCK_FFTSPLUS_OPS_KERNEL_BUILDER_H_
#define AIR_TEST_ENGINES_NNENG_FRAMEWORK_FE_RUNNING_ENV_INCLUDE_MOCK_FFTSPLUS_OPS_KERNEL_BUILDER_H_

#include "task_builder/fftsplus_ops_kernel_builder.h"

namespace ffts {
using FFTSPlusOpsKernelBuilderPtr = std::shared_ptr<FFTSPlusOpsKernelBuilder>;

class MockFFTSPlusOpsKernelBuilder : public ge::OpsKernelBuilder {
 public:
  /**
   * Constructor for AICoreOpsKernelBuilder
   */
  MockFFTSPlusOpsKernelBuilder();

  /**
   * Deconstruction for AICoreOpsKernelBuilder
   */
  ~MockFFTSPlusOpsKernelBuilder() override;

  /**
   * Initialization
   * @param options
   * @return Status SUCCESS / FAILED or others
   */
  Status Initialize(const std::map<std::string, std::string> &options) override;

  /**
   * Finalization
   * @return Status SUCCESS / FAILED or others
   */
  Status Finalize() override;

  /**
   * Calculate the running parameters for node
   * @param node node object
   * @return Status SUCCESS / FAILED or others
   */
  Status CalcOpRunningParam(ge::Node &node) override;

  /**
   * Generate task for node
   * @param node node object
   * @param context context object
   * @param tasks Task list
   * @return Status SUCCESS / FAILED or others
   */
  Status GenerateTask(const ge::Node &node, ge::RunContext &context, std::vector<domi::TaskDef> &task_defs) override;

private:
  FFTSPlusOpsKernelBuilderPtr fftsplus_ops_kernel_build_ptr_;
};
} // namespace
#endif // AIR_TEST_ENGINES_NNENG_FRAMEWORK_FE_RUNNING_ENV_INCLUDE_MOCK_FFTSPLUS_OPS_KERNEL_BUILDER_H_
