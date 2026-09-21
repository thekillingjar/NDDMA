/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_OPS_KERNEL_BUILDER_AICORE_OPS_KERNEL_BUILDER_H_
#define AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_OPS_KERNEL_BUILDER_AICORE_OPS_KERNEL_BUILDER_H_

#include "graph_optimizer/graph_optimize_register_error_codes.h"
#include "common/opskernel/ops_kernel_builder.h"
#include "adapter/adapter_itf/task_builder_adapter.h"

namespace fe {

void ProcDfxBufferSize(const ge::OpDescPtr op_desc);
Status GenerateExtTask(const ge::Node &node, ge::RunContext &context, std::vector<domi::TaskDef> &task_defs);

class AICoreOpsKernelBuilder : public ge::OpsKernelBuilder {
 public:
 using ge::OpsKernelBuilder::GenerateTask;
  /**
   * Constructor for AICoreOpsKernelBuilder
   */
  AICoreOpsKernelBuilder();

  /**
   * Deconstruction for AICoreOpsKernelBuilder
   */
  ~AICoreOpsKernelBuilder() override;

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
  Status GenerateTask(const ge::Node &node, ge::RunContext &context, std::vector<domi::TaskDef> &tasks) override;

  private:
  static Status CalcExtOpRunningParam(ge::Node &node);
  static Status CalcTilingSinkRunningParam(const bool is_tiling_sink, ge::Node &node, bool &proc_flag);
};
}  // namespace fe
#endif  // AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_OPS_KERNEL_BUILDER_AICORE_OPS_KERNEL_BUILDER_H_
