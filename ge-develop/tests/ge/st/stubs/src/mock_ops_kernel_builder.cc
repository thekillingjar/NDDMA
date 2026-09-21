/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "utils/mock_ops_kernel_builder.h"
#include "common/taskdown_common.h"
#include "common/opskernel/ops_kernel_info_types.h"

namespace ge {
MockOpsKernelBuilderPtr MockForGenerateTask(const std::string &engine_name, const GenerateTaskFun &func,
                                            MockOpsKernelBuilderPtr builder) {
  if (builder == nullptr) {
    builder = std::make_shared<MockOpsKernelBuilder>();
  }
  OpsKernelBuilderRegistry::GetInstance().kernel_builders_[engine_name] = builder;
  EXPECT_CALL(*builder, GenerateTask).WillRepeatedly(testing::Invoke(func));
  return builder;
}

void TearDownForGenerateTask(const std::string &engine_name) {
  OpsKernelBuilderRegistry::GetInstance().Unregister(engine_name);
}
MockOpsKernelBuilderPtr MockOnceForOnceSkipGenerateTask(const std::string &name, const GenerateTaskFun &func1,
                                                        const GenerateTaskFun &func2, MockOpsKernelBuilderPtr builder) {
  if (builder == nullptr) {
    builder = std::make_shared<MockOpsKernelBuilder>();
  }
  OpsKernelBuilderRegistry::GetInstance().kernel_builders_[name] = builder;
  EXPECT_CALL(*builder, GenerateTask).WillOnce(testing::Invoke(func1)).WillRepeatedly(testing::Invoke(func2));
  return builder;
}
MockOpsKernelBuilderPtr MockForAiCoreGenerateTask() {
  auto aicore_func = [](const ge::Node &node, RunContext &context, std::vector<domi::TaskDef> &tasks) -> Status {
    auto op_desc = node.GetOpDesc();
    op_desc->SetOpKernelLibName("AIcoreEngine");
    size_t arg_size = 100;
    std::vector<uint8_t> args(arg_size, 0);
    domi::TaskDef task_def;
    task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
    auto kernel_info = task_def.mutable_kernel();
    kernel_info->set_args(args.data(), args.size());
    kernel_info->set_args_size(arg_size);
    kernel_info->mutable_context()->set_kernel_type(static_cast<uint32_t>(ccKernelType::TE));
    kernel_info->set_kernel_name(node.GetName());
    kernel_info->set_block_dim(1);
    uint16_t args_offset[2] = {0};
    kernel_info->mutable_context()->set_args_offset(args_offset, 2 * sizeof(uint16_t));
    kernel_info->mutable_context()->set_op_index(node.GetOpDesc()->GetId());

    tasks.emplace_back(task_def);
    return SUCCESS;
  };
  return MockForGenerateTask("AIcoreEngine", aicore_func);
}
void TearDownForAiCoreGenerateTask() {
  TearDownForGenerateTask("AIcoreEngine");
}
}  // namespace ge
