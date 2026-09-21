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

#include "faker/kernel_run_context_facker.h"
#include "register/kernel_registry.h"
#include "common/ge_inner_attrs.h"
#include "core/execution_data.h"
#include "graph/compute_graph.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/utils/tensor_utils.h"
#include "exe_graph/runtime/gert_tensor_data.h"
#include "kernel/common_kernel_impl/condition_calc.h"
#include "graph/utils/execute_graph_utils.h"

namespace gert {
TEST(ConditionCalc, ConditionCalc) {
  const KernelRegistry::KernelFuncs *kernel = KernelRegistry::GetInstance().FindKernelFuncs("ConditionCalc");
  ASSERT_NE(kernel, nullptr);
  auto graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("input1", "Data")->NODE("condition_calc", "ConditionCalc")->NODE("NetOutput", "NetOutput"));
      CHAIN(NODE("input2", "Data")->EDGE(0, 1)->NODE("condition_calc", "ConditionCalc"));
      CHAIN(NODE("input3", "Data")->EDGE(0, 2)->NODE("condition_calc", "ConditionCalc"));
    };
    return ge::ToExecuteGraph(g);
  }();
  auto node = ge::ExecuteGraphUtils::FindNodeFromAllNodes(graph.get(), "condition_calc");
  ASSERT_NE(node, nullptr);

  auto run_context = BuildKernelRunContext(0, 1);
  auto kernel_context = run_context.GetContext<KernelContext>();
  EXPECT_NE(kernel->outputs_creator, nullptr);
  EXPECT_NE(kernel->run_func(kernel_context), ge::GRAPH_SUCCESS);  // output num mismatch

  const static size_t kInputNum = 1U;
  run_context = BuildKernelRunContext(kInputNum, 1);
  kernel_context = run_context.GetContext<KernelContext>();
  *const_cast<KernelRegistry::KernelFunc *>(kernel_context->GetInputPointer<KernelRegistry::KernelFunc>(
      kInputNum - 1)) = [](KernelContext *context) { return ge::GRAPH_FAILED; };
  EXPECT_NE(kernel->outputs_creator, nullptr);
  EXPECT_NE(kernel->run_func(kernel_context), ge::GRAPH_SUCCESS);  // kernel func error

  run_context = BuildKernelRunContext(kInputNum, 1);
  kernel_context = run_context.GetContext<KernelContext>();
  *const_cast<KernelRegistry::KernelFunc *>(kernel_context->GetInputPointer<KernelRegistry::KernelFunc>(
      kInputNum - 1)) = [](KernelContext *context) { return ge::GRAPH_SUCCESS; };
  EXPECT_NE(kernel->outputs_creator, nullptr);
  EXPECT_EQ(kernel->run_func(kernel_context), ge::GRAPH_SUCCESS);
}
TEST(ConditionCalc, FindKernelFunc) {
  const KernelRegistry::KernelFuncs *kernel = KernelRegistry::GetInstance().FindKernelFuncs("FindKernelFunc");
  ASSERT_NE(kernel, nullptr);
  auto graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("input1", "Data")->NODE("find_kernel_func", "FindKernelFunc")->NODE("NetOutput", "NetOutput"));
    };
    return ge::ToExecuteGraph(g);
  }();
  auto node = ge::ExecuteGraphUtils::FindNodeFromAllNodes(graph.get(), "find_kernel_func");
  ASSERT_NE(node, nullptr);

  auto run_context = BuildKernelRunContext(1, 1);
  auto kernel_context = run_context.GetContext<KernelContext>();
  const_cast<Chain *>(kernel_context->GetInput(0))->Set(nullptr, nullptr);
  EXPECT_NE(kernel->run_func(kernel_context), ge::GRAPH_SUCCESS);  // kernel name is nullptr

  constexpr char *kernel_name = (char *)"foo";
  const_cast<Chain *>(kernel_context->GetInput(0))->Set(kernel_name, nullptr);
  EXPECT_NE(kernel->run_func(kernel_context), ge::GRAPH_SUCCESS);  // kernel not found

  KernelRegistry::GetInstance().RegisterKernel(kernel_name,
                                               {{.run_func = [](KernelContext *) { return ge::GRAPH_SUCCESS; }}, ""});
  EXPECT_EQ(kernel->run_func(kernel_context), ge::GRAPH_SUCCESS);
  auto found_kernel = *kernel_context->GetOutputPointer<KernelRegistry::KernelFunc>(0);
  auto kernel_funcs = KernelRegistry::GetInstance().FindKernelFuncs(kernel_name);
  EXPECT_EQ(kernel_funcs->run_func, found_kernel);
}
TEST(ConditionCalc, BuildUnmanagedTensorData) {
  const KernelRegistry::KernelFuncs *kernel = KernelRegistry::GetInstance().FindKernelFuncs("BuildUnmanagedTensorData");
  ASSERT_NE(kernel, nullptr);
  auto graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("input1", "Data")->NODE("build_unmanaged_tensor_data", "BuildUnmanagedTensorData")->NODE("NetOutput", "NetOutput"));
    };
    return ge::ToExecuteGraph(g);
  }();
  auto node = ge::ExecuteGraphUtils::FindNodeFromAllNodes(graph.get(), "build_unmanaged_tensor_data");
  ASSERT_NE(node, nullptr);
  int64_t logic_stream_id = 0;
  auto kernel_holder = KernelRunContextFaker()
                           .KernelIONum(static_cast<size_t>(kernel::BuildUnmanagedTensorDataInputs::kEnd),
                                        static_cast<size_t>(kernel::BuildUnmanagedTensorDataOutputs::kEnd))
                           .Inputs({nullptr, &logic_stream_id})
                           .Build();
  auto run_context = kernel_holder.GetContext<KernelContext>();
  EXPECT_EQ(kernel->outputs_creator(node, run_context), ge::GRAPH_SUCCESS);
  EXPECT_EQ(kernel->run_func(run_context), ge::GRAPH_SUCCESS);
  auto device_gtd = run_context->GetOutputPointer<GertTensorData>(0U);
  ASSERT_EQ(device_gtd->GetPlacement(), kTensorPlacementEnd);
}
}  // namespace gert
