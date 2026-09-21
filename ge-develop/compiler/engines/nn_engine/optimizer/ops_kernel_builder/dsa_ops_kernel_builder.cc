/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ops_kernel_builder/dsa_ops_kernel_builder.h"

#include <map>
#include <string>
#include <vector>

#include "common/fe_log.h"
#include "common/op_tensor_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "param_calculate/tensorsize_calculator.h"
#include "ops_kernel_builder/task_builder/dsa_task_builder.h"
#include "register/ops_kernel_builder_registry.h"

namespace fe {
REGISTER_OPS_KERNEL_BUILDER(kDsaCoreName, DsaOpsKernelBuilder);

Status DsaOpsKernelBuilder::Initialize(const std::map<std::string, std::string> &options) {
  (void) options;
  return SUCCESS;
}

Status DsaOpsKernelBuilder::Finalize() {
  return SUCCESS;
}

Status DsaOpsKernelBuilder::CalcOpRunningParam(ge::Node &node) {
  (void) node;
  return SUCCESS;
}

Status DsaOpsKernelBuilder::GenerateTask(const ge::Node &node, ge::RunContext &context,
                                         std::vector<domi::TaskDef> &tasks) {
  DsaTaskBuilder task_builder;
  Status status = task_builder.GenerateTask(node, context, tasks);
  return status;
}
}  // namespace fe
