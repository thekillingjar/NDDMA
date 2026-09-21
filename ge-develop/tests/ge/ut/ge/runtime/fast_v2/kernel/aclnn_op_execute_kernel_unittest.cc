/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "base/registry/op_impl_space_registry_v2.h"
#include "faker/kernel_run_context_facker.h"
#include "stub/gert_runtime_stub.h"

namespace gert {
using namespace ge;

class AclNNOpExecuteKernelUT : public testing::Test {
 public:
  KernelRegistry &registry = KernelRegistry::GetInstance();
};

ge::graphStatus OpExecuteDoNothing(OpExecuteContext *) {
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus OpExecutePrepareDoNothing(OpExecutePrepareContext *) {
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus OpExecuteLaunchDoNothing(OpExecuteLaunchContext *) {
  return ge::GRAPH_SUCCESS;
}

TEST_F(AclNNOpExecuteKernelUT, AclNNOpExecuteKernelUT_GetSpaceRegistryV2_success) {
  const std::string node_type = "FOO_OP_EXEC";

  auto space_registry_bak = DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto space_registry = ge::MakeShared<OpImplSpaceRegistryV2>();
  auto funcs = space_registry->CreateOrGetOpImpl("FOO_OP_EXEC");
  funcs->op_execute_func = OpExecuteDoNothing;
  DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(space_registry);

  auto run_context = BuildKernelRunContext(3, 1);

  ASSERT_NE(nullptr, space_registry);
  run_context.value_holder[0].Set(const_cast<char *>(node_type.c_str()), nullptr);
  run_context.value_holder[1].Set(space_registry.get(), nullptr);

  auto find_func = registry.FindKernelFuncs("FindOpExeFunc");
  ASSERT_NE(find_func, nullptr);
  ASSERT_EQ(find_func->run_func(run_context), ge::SUCCESS);
  DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(space_registry_bak);
}

TEST_F(AclNNOpExecuteKernelUT, AclNNOpExecuteKernelUT_TwoStages_GetSpaceRegistryV2_success) {
  const std::string node_type = "FOO_OP_EXEC";
  auto space_registry_bak = DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto space_registry = ge::MakeShared<OpImplSpaceRegistryV2>();
  auto funcs = space_registry->CreateOrGetOpImpl("FOO_OP_EXEC");
  funcs->op_execute_prepare_func = OpExecutePrepareDoNothing;
  funcs->op_execute_launch_func = OpExecuteLaunchDoNothing;
  DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(space_registry);

  auto run_context = BuildKernelRunContext(3, 2);

  ASSERT_NE(nullptr, space_registry);
  run_context.value_holder[0].Set(const_cast<char *>(node_type.c_str()), nullptr);
  run_context.value_holder[1].Set(space_registry.get(), nullptr);

  auto find_func = registry.FindKernelFuncs("FindOpExe2PhaseFunc");
  ASSERT_NE(find_func, nullptr);
  ASSERT_EQ(find_func->run_func(run_context), ge::SUCCESS);

  auto failed_run_context1 = BuildKernelRunContext(3, 2);
  failed_run_context1.value_holder[0].Set(nullptr, nullptr);
  failed_run_context1.value_holder[1].Set(space_registry.get(), nullptr);
  ASSERT_NE(find_func->run_func(failed_run_context1), ge::SUCCESS);

  auto failed_run_context2 = BuildKernelRunContext(3, 2);
  failed_run_context2.value_holder[0].Set(const_cast<char *>(node_type.c_str()), nullptr);
  failed_run_context2.value_holder[1].Set(nullptr, nullptr);
  ASSERT_NE(find_func->run_func(failed_run_context2), ge::SUCCESS);
  
  DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(space_registry_bak);
}
}  // namespace gert