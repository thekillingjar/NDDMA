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
#include "faker/kernel_outputs_faker.h"
#include "faker/node_faker.h"
#include "register/kernel_registry.h"
#include "kernel/tensor_attr.h"
#include "exe_graph/runtime/gert_tensor_data.h"
#include "faker/fake_value.h"

namespace gert {
namespace kernel {
struct NpuGetFloatStatusArgsHolder {
  void *output_holder{nullptr};
  size_t output_size{64U};
};
}  // namespace kernel

struct OverFlowDetectionKernelUt : public testing::Test {
  KernelRegistry &registry = KernelRegistry::GetInstance();
};

TEST_F(OverFlowDetectionKernelUt, NpuClearFloatStatusOk) {
  auto funcs = registry.FindKernelFuncs("NpuClearFloatStatus");
  ASSERT_NE(funcs, nullptr);
  void *stream;
  auto context_holder = KernelRunContextFaker().KernelIONum(1U, 0U).Inputs({&stream}).Build();
  auto context = context_holder.GetContext<KernelContext>();
  EXPECT_EQ(funcs->run_func(context), ge::GRAPH_SUCCESS);
  context_holder.FreeAll();
}

TEST_F(OverFlowDetectionKernelUt, NpuGetFloatStatusArgsOk) {
  auto funcs = registry.FindKernelFuncs("NpuGetFloatStatusArgs");
  ASSERT_NE(funcs, nullptr);
  uint8_t tmp1[64] = {0U};
  GertTensorData data = {tmp1, 0U, kTensorPlacementEnd, -1};
  auto context_holder = KernelRunContextFaker().KernelIONum(1U, 1U).Inputs({&data}).Build();
  auto node = FastNodeFaker().Build();
  auto context = context_holder.GetContext<KernelContext>();
  EXPECT_EQ(funcs->outputs_creator(node, context), ge::GRAPH_SUCCESS);
  EXPECT_EQ(funcs->run_func(context), ge::GRAPH_SUCCESS);

  auto holder = context->GetOutputPointer<gert::kernel::NpuGetFloatStatusArgsHolder>(0);
  EXPECT_NE(holder, nullptr);
  EXPECT_EQ(holder->output_holder, tmp1);
  context_holder.FreeAll();
}

TEST_F(OverFlowDetectionKernelUt, NpuGetFloatStatusOk) {
  auto funcs = registry.FindKernelFuncs("NpuGetFloatStatus");
  ASSERT_NE(funcs, nullptr);
  uint8_t tmp1[64] = {0};
  uint8_t tmp2[64] = {0};
  GertTensorData data = {tmp1, 0U, kTensorPlacementEnd, -1};
  gert::kernel::NpuGetFloatStatusArgsHolder args_holder;
  args_holder.output_holder = tmp2;
  void *stream;
  auto context_holder = KernelRunContextFaker()
                            .KernelIONum(3U, 1U)
                            .Inputs({&data, &args_holder, &stream})
                            .Build();
  auto context = context_holder.GetContext<KernelContext>();
  EXPECT_EQ(funcs->run_func(context), ge::GRAPH_SUCCESS);
  context_holder.FreeAll();
}

TEST_F(OverFlowDetectionKernelUt, NpuClearFloatDebugStatusOk) {
  auto funcs = registry.FindKernelFuncs("NpuClearFloatDebugStatus");
  ASSERT_NE(funcs, nullptr);
  void *stream;
  auto context_holder = KernelRunContextFaker().KernelIONum(1U, 0U).Inputs({&stream}).Build();
  auto context = context_holder.GetContext<KernelContext>();
  EXPECT_EQ(funcs->run_func(context), ge::GRAPH_SUCCESS);
  context_holder.FreeAll();
}

TEST_F(OverFlowDetectionKernelUt, NpuGetFloatDebugStatusOk) {
  auto funcs = registry.FindKernelFuncs("NpuGetFloatDebugStatus");
  ASSERT_NE(funcs, nullptr);
  uint8_t tmp1[64] = {0};
  uint8_t tmp2[64] = {0};
  GertTensorData data = {tmp1, 0U, kTensorPlacementEnd, -1};
  gert::kernel::NpuGetFloatStatusArgsHolder args_holder;
  args_holder.output_holder = tmp2;
  void *stream;
  auto context_holder = KernelRunContextFaker().KernelIONum(3U, 1U).Inputs({&data, &args_holder, &stream}).Build();
  auto context = context_holder.GetContext<KernelContext>();
  EXPECT_EQ(funcs->run_func(context), ge::GRAPH_SUCCESS);
  context_holder.FreeAll();
}
}  // namespace gert
