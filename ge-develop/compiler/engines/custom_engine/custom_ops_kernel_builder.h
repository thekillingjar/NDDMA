/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CUSTOM_ENGINE_CUSTOM_OPS_KERNEL_UTILS_H_
#define CUSTOM_ENGINE_CUSTOM_OPS_KERNEL_UTILS_H_

#include "ge/ge_api_error_codes.h"
#include "common/opskernel/ops_kernel_builder.h"

namespace ge {
namespace custom {
class GE_FUNC_VISIBILITY CustomOpsKernelBuilder : public OpsKernelBuilder {
 public:
  CustomOpsKernelBuilder() = default;
  ~CustomOpsKernelBuilder() override;
  Status Initialize(const std::map<std::string, std::string> &options) override;

  Status Finalize() override;

  Status CalcOpRunningParam(Node &node) override;

  Status GenerateTask(const Node &node, RunContext &context, std::vector<domi::TaskDef> &tasks) override;
};
}  // namespace custom
}  // namespace ge

#endif  // CUSTOM_ENGINE_CUSTOM_OPS_KERNEL_UTILS_H_
