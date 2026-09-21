/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "framework/fe_running_env/include/fe_running_env.h"
#include "ops_kernel_builder/aicore_ops_kernel_builder.h"

namespace fe {
class AICoreOpsKernelSubBuilder : public AICoreOpsKernelBuilder {
 public:
  /**
   * Constructor for AICoreOpsKernelSubBuilder
   */
  AICoreOpsKernelSubBuilder();

  /**
   * Deconstruction for AICoreOpsKernelSubBuilder
   */
  ~AICoreOpsKernelSubBuilder() override;
  Status GenerateTask(const ge::Node &node, ge::RunContext &context, std::vector<domi::TaskDef> &tasks) override;
};
}
