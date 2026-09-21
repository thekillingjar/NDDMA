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
#include "kernel/ge_local_kernel/size_kernel.h"

namespace gert {
namespace {
const std::string kKernelName = "GetShapeSizeKernel";
}
struct SizeKernelUt : public testing::Test {
  KernelRegistry &registry = KernelRegistry::GetInstance();
};

TEST_F(SizeKernelUt, SizeKernelInt64Ok) {
  StorageShape shape = {{2,2,3}, {2,2,3}};
  auto context_holder = KernelRunContextFaker()
                            .KernelIONum(static_cast<size_t>(kernel::SizeKernelInputs::kEnd),
                                         static_cast<size_t>(kernel::SizeKernelOutputs::kEnd))
                            .Inputs({&shape})
                            .NodeAttrs({{"dtype", ge::AnyValue::CreateFrom<int64_t>(ge::DataType::DT_INT64)}})
                            .Build();
  auto context = context_holder.GetContext<KernelContext>();

  auto funcs = registry.FindKernelFuncs(kKernelName);
  ASSERT_NE(funcs, nullptr);
  FastNodeFaker faker;
  auto node = faker.Build();
  (void)ge::AttrUtils::SetInt(node->GetOpDescBarePtr(), "dtype", ge::DT_INT64);
  ASSERT_EQ(funcs->outputs_creator(node, context), ge::GRAPH_SUCCESS);
  EXPECT_EQ(funcs->run_func(context), ge::GRAPH_SUCCESS);

  auto td = context->GetOutputPointer<GertTensorData>(static_cast<size_t>(0));
  ASSERT_NE(td, nullptr);
  EXPECT_EQ(td->GetPlacement(), kOnHost);
  EXPECT_EQ(td->GetSize(), 8);
  auto output_data = static_cast<const int64_t *>(td->GetAddr());
  EXPECT_EQ(output_data[0], 12);

  context_holder.FreeAll();
}

TEST_F(SizeKernelUt, SizeKernelInt32Ok) {
  StorageShape shape = {{2,2,3}, {2,2,3}};
  auto context_holder = KernelRunContextFaker()
                            .KernelIONum(static_cast<size_t>(kernel::SizeKernelInputs::kEnd),
                                         static_cast<size_t>(kernel::SizeKernelOutputs::kEnd))
                            .Inputs({&shape})
                            .NodeAttrs({{"dtype", ge::AnyValue::CreateFrom<int64_t>(ge::DataType::DT_INT32)}})
                            .Build();
  auto context = context_holder.GetContext<KernelContext>();

  auto funcs = registry.FindKernelFuncs(kKernelName);
  ASSERT_NE(funcs, nullptr);
  FastNodeFaker faker;
  auto node = faker.Build();
  (void)ge::AttrUtils::SetInt(node->GetOpDescBarePtr(), "dtype", ge::DT_INT32);
  ASSERT_EQ(funcs->outputs_creator(node, context), ge::GRAPH_SUCCESS);
  EXPECT_EQ(funcs->run_func(context), ge::GRAPH_SUCCESS);

  auto td = context->GetOutputPointer<GertTensorData>(static_cast<size_t>(0));
  ASSERT_NE(td, nullptr);
  EXPECT_EQ(td->GetPlacement(), kOnHost);
  EXPECT_EQ(td->GetSize(), 4);
  auto output_data = static_cast<const int32_t *>(td->GetAddr());
  EXPECT_EQ(output_data[0], 12);

  context_holder.FreeAll();
}

TEST_F(SizeKernelUt, SizeKernelFloatNotSupportFailed) {
  StorageShape shape = {{2, 2, 3}, {2, 2, 3}};
  auto context_holder = KernelRunContextFaker()
                            .KernelIONum(static_cast<size_t>(kernel::SizeKernelInputs::kEnd),
                                         static_cast<size_t>(kernel::SizeKernelOutputs::kEnd))
                            .Inputs({&shape})
                            .NodeAttrs({{"dtype", ge::AnyValue::CreateFrom<int64_t>(ge::DataType::DT_FLOAT)}})
                            .Build();
  auto context = context_holder.GetContext<KernelContext>();

  auto funcs = registry.FindKernelFuncs(kKernelName);
  ASSERT_NE(funcs, nullptr);
  FastNodeFaker faker;
  auto node = faker.Build();
  (void)ge::AttrUtils::SetInt(node->GetOpDescBarePtr(), "dtype", ge::DT_FLOAT);
  ASSERT_NE(funcs->outputs_creator(node, context), ge::GRAPH_SUCCESS);
  context_holder.FreeAll();
}

TEST_F(SizeKernelUt, IrInputDTypeLostFailed) {
  StorageShape shape = {{2, 2, 3}, {2, 2, 3}};
  auto context_holder = KernelRunContextFaker()
                            .KernelIONum(static_cast<size_t>(kernel::SizeKernelInputs::kEnd),
                                         static_cast<size_t>(kernel::SizeKernelOutputs::kEnd))
                            .Inputs({&shape})
                            .Build();
  auto context = context_holder.GetContext<KernelContext>();

  auto funcs = registry.FindKernelFuncs(kKernelName);
  ASSERT_NE(funcs, nullptr);
  auto node = ComputeNodeFaker().Build();
  (void)ge::AttrUtils::SetInt(node->GetOpDesc(), "dtype", ge::DT_INT32);
  EXPECT_NE(funcs->run_func(context), ge::GRAPH_SUCCESS);
  context_holder.FreeAll();
}
}  // namespace gert