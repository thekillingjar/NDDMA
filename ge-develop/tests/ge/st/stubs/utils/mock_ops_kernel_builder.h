/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_TESTS_ST_STUBS_UTILS_MOCK_OPS_KERNEL_BUILDER_H_
#define AIR_CXX_TESTS_ST_STUBS_UTILS_MOCK_OPS_KERNEL_BUILDER_H_
#include <gmock/gmock.h>
#include "macro_utils/dt_public_scope.h"
#include "ge_running_env/fake_ops_kernel_builder.h"
#include "engines/manager/opskernel_manager/ops_kernel_builder_manager.h"
#include "macro_utils/dt_public_unscope.h"

namespace ge {
class MockOpsKernelBuilder : public FakeOpsKernelBuilder {
 public:
  MOCK_METHOD3(GenerateTask, Status(const Node &, RunContext &, vector<domi::TaskDef> &));
};
using GenerateTaskFun = std::function<Status(const Node &node, RunContext &context, std::vector<domi::TaskDef> &tasks)>;
using MockOpsKernelBuilderPtr = std::shared_ptr<MockOpsKernelBuilder>;
MockOpsKernelBuilderPtr MockForGenerateTask(const std::string &engine_name, const GenerateTaskFun &func,
                                            MockOpsKernelBuilderPtr builder = nullptr);
void TearDownForGenerateTask(const std::string &engine_name);
MockOpsKernelBuilderPtr MockOnceForOnceSkipGenerateTask(const std::string &name, const GenerateTaskFun &func1,
                                                        const GenerateTaskFun &func2,
                                                        MockOpsKernelBuilderPtr builder = nullptr);

MockOpsKernelBuilderPtr MockForAiCoreGenerateTask();
void TearDownForAiCoreGenerateTask();
}  // namespace ge

#endif  // AIR_CXX_TESTS_ST_STUBS_UTILS_MOCK_OPS_KERNEL_BUILDER_H_
