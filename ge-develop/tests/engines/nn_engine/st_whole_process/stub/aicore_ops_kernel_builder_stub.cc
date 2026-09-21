/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicore_ops_kernel_builder_stub.h"
#include "common/fe_log.h"
#include "common/constants_define.h"
#include "common/op_tensor_utils.h"
#include "graph/utils/attr_utils.h"
#include "common/aicore_util_attr_define.h"
#include "graph/debug/ge_attr_define.h"
#include "common/aicore_util_constants.h"
#include "param_calculate/tensorsize_calculator.h"
#include "ops_kernel_builder/task_builder/task_builder.h"
#include "register/ops_kernel_builder_registry.h"
#include "adapter/tbe_adapter/tbe_task_builder_adapter.h"
#include "common/op_tensor_utils.h"
#include "inc/ffts_param_calculator.h"

namespace fe {
AICoreOpsKernelSubBuilder::AICoreOpsKernelSubBuilder() {}

AICoreOpsKernelSubBuilder::~AICoreOpsKernelSubBuilder() {}

Status AICoreOpsKernelSubBuilder::GenerateTask(const ge::Node &node, ge::RunContext &context, std::vector<domi::TaskDef> &tasks) {
  AICoreOpsKernelBuilder::GenerateTask(node, context, tasks);
  fe_env::FeRunningEnv::tasks_map.insert(std::make_pair(node.GetName(), tasks));
  return SUCCESS;
}
}
