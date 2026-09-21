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
#include "kernel/ge_local_kernel/gather_shapes_kernel.h"

namespace gert {
namespace {
const std::vector<std::vector<int64_t>> kAxes{{1, 2}, {0, 1}};
const std::string kKernelName = "GatherShapesKernel";
}
struct GatherShapesKernelUt : public testing::Test {
  KernelRegistry &registry = KernelRegistry::GetInstance();
};

TEST_F(GatherShapesKernelUt, GatherShapesKernelInt64Ok) {
  StorageShape shape = {{10,20,30}, {10,20,30}};
  StorageShape shape2 = {{10,20,-1}, {10,20,30}};
  Shape kOutputData{-1, 20};
  auto context_holder = KernelRunContextFaker()
      .KernelIONum(2U, static_cast<size_t>(kernel::GatherShapesKernelOutputs::kEnd))
      .Inputs({&shape, &shape2})
      .NodeAttrs({{"axes", ge::AnyValue::CreateFrom<std::vector<std::vector<int64_t>>>(kAxes)},
                  {"dtype", ge::AnyValue::CreateFrom<int64_t>(ge::DataType::DT_INT64)}})
      .Build();
  auto context = context_holder.GetContext<KernelContext>();

  auto funcs = registry.FindKernelFuncs(kKernelName);
  ASSERT_NE(funcs, nullptr);
  FastNodeFaker fast_node_faker;
  auto node = fast_node_faker.Build();
  (void)ge::AttrUtils::SetInt(node->GetOpDescBarePtr(), "dtype", ge::DT_INT64);
  ASSERT_EQ(funcs->outputs_creator(node, context), ge::GRAPH_SUCCESS);
  EXPECT_EQ(funcs->run_func(context), ge::GRAPH_SUCCESS);

  auto td = context->GetOutputPointer<GertTensorData>(static_cast<size_t>(0));
  ASSERT_NE(td, nullptr);
  EXPECT_EQ(td->GetPlacement(), kOnHost);
  EXPECT_EQ(td->GetSize(), Shape::kMaxDimNum * sizeof(int64_t));
  auto shape_data = static_cast<const int64_t *>(td->GetAddr());
  EXPECT_EQ(shape_data[0], kOutputData[0]);
  EXPECT_EQ(shape_data[1], kOutputData[1]);

  context_holder.FreeAll();
}

TEST_F(GatherShapesKernelUt, GatherShapesKernelInt32Ok) {
  StorageShape shape = {{10,20,30}, {10,20,30}};
  StorageShape shape2 = {{10,20,30}, {10,20,30}};
  Shape kOutputData{30, 20};
  auto context_holder = KernelRunContextFaker()
      .KernelIONum(2U, static_cast<size_t>(kernel::GatherShapesKernelOutputs::kEnd))
      .Inputs({&shape, &shape2})
      .NodeAttrs({{"axes", ge::AnyValue::CreateFrom<std::vector<std::vector<int64_t>>>(kAxes)},
                  {"dtype", ge::AnyValue::CreateFrom<int64_t>(ge::DataType::DT_INT32)}})
      .Build();
  auto context = context_holder.GetContext<KernelContext>();

  auto funcs = registry.FindKernelFuncs(kKernelName);
  ASSERT_NE(funcs, nullptr);
  FastNodeFaker fast_node_faker;
  auto node = fast_node_faker.Build();
  (void)ge::AttrUtils::SetInt(node->GetOpDescBarePtr(), "dtype", ge::DT_INT32);
  ASSERT_EQ(funcs->outputs_creator(node, context), ge::GRAPH_SUCCESS);
  EXPECT_EQ(funcs->run_func(context), ge::GRAPH_SUCCESS);

  auto td = context->GetOutputPointer<GertTensorData>(static_cast<size_t>(0));
  ASSERT_NE(td, nullptr);
  EXPECT_EQ(td->GetPlacement(), kOnHost);
  EXPECT_EQ(td->GetSize(), Shape::kMaxDimNum * sizeof(int32_t));
  auto shape_data = static_cast<const int32_t *>(td->GetAddr());
  EXPECT_EQ(shape_data[0], kOutputData[0]);
  EXPECT_EQ(shape_data[1], kOutputData[1]);

  context_holder.FreeAll();
}

TEST_F(GatherShapesKernelUt, GatherShapesKernelFloatNotSupportFailed) {
  StorageShape shape = {{10,20,30}, {10,20,30}};
  StorageShape shape2 = {{10,20,30}, {10,20,30}};
  auto context_holder = KernelRunContextFaker()
      .KernelIONum(2U, static_cast<size_t>(kernel::GatherShapesKernelOutputs::kEnd))
      .Inputs({&shape, &shape2})
      .NodeAttrs({{"axes", ge::AnyValue::CreateFrom<std::vector<std::vector<int64_t>>>(kAxes)},
                  {"dtype", ge::AnyValue::CreateFrom<int64_t>(ge::DataType::DT_FLOAT)}})
      .Build();
  auto context = context_holder.GetContext<KernelContext>();

  auto funcs = registry.FindKernelFuncs(kKernelName);
  ASSERT_NE(funcs, nullptr);
  FastNodeFaker fast_node_faker;
  auto node = fast_node_faker.Build();
  (void)ge::AttrUtils::SetInt(node->GetOpDescBarePtr(), "dtype", ge::DT_FLOAT);
  ASSERT_NE(funcs->outputs_creator(node, context), ge::GRAPH_SUCCESS);
  context_holder.FreeAll();
}

TEST_F(GatherShapesKernelUt, IrInputAxesLostFailed) {
  StorageShape shape = {{10,20,30}, {10,20,30}};
  StorageShape shape2 = {{10,20,30}, {10,20,30}};
  auto context_holder = KernelRunContextFaker()
      .KernelIONum(2U, static_cast<size_t>(kernel::GatherShapesKernelOutputs::kEnd))
      .Inputs({&shape, &shape2})
      .NodeAttrs({})
      .Build();
  auto context = context_holder.GetContext<KernelContext>();

  auto funcs = registry.FindKernelFuncs(kKernelName);
  ASSERT_NE(funcs, nullptr);
  FastNodeFaker fast_node_faker;
  auto node = fast_node_faker.Build();
  (void)ge::AttrUtils::SetInt(node->GetOpDescBarePtr(), "dtype", ge::DT_INT32);
  ASSERT_EQ(funcs->outputs_creator(node, context), ge::GRAPH_SUCCESS);
  EXPECT_NE(funcs->run_func(context), ge::GRAPH_SUCCESS);
  context_holder.FreeAll();
}

TEST_F(GatherShapesKernelUt, IrInputDTypeLostFailed) {
  StorageShape shape = {{10,20,30}, {10,20,30}};
  StorageShape shape2 = {{10,20,30}, {10,20,30}};
  auto context_holder = KernelRunContextFaker()
      .KernelIONum(2U, static_cast<size_t>(kernel::GatherShapesKernelOutputs::kEnd))
      .Inputs({&shape, &shape2})
      .NodeAttrs({{"axes", ge::AnyValue::CreateFrom<std::vector<std::vector<int64_t>>>(kAxes)}})
      .Build();
  auto context = context_holder.GetContext<KernelContext>();

  auto funcs = registry.FindKernelFuncs(kKernelName);
  ASSERT_NE(funcs, nullptr);
  FastNodeFaker fast_node_faker;
  auto node = fast_node_faker.Build();
  (void)ge::AttrUtils::SetInt(node->GetOpDescBarePtr(), "dtype", ge::DT_INT32);
  ASSERT_EQ(funcs->outputs_creator(node, context), ge::GRAPH_SUCCESS);
  EXPECT_NE(funcs->run_func(context), ge::GRAPH_SUCCESS);
  context_holder.FreeAll();
}

TEST_F(GatherShapesKernelUt, AxesInputIndexOutOfBoundsFailed) {
  StorageShape shape = {{10,20,30}, {10,20,30}};
  StorageShape shape2 = {{10,20,30}, {10,20,30}};
  Shape kOutputData{30, 20};
  const std::vector<std::vector<int64_t>> axes{{3, 2}, {0, 1}};
  auto context_holder = KernelRunContextFaker()
      .KernelIONum(2U, static_cast<size_t>(kernel::GatherShapesKernelOutputs::kEnd))
      .Inputs({&shape, &shape2})
      .NodeAttrs({{"axes", ge::AnyValue::CreateFrom<std::vector<std::vector<int64_t>>>(axes)},
                  {"dtype", ge::AnyValue::CreateFrom<int64_t>(ge::DataType::DT_INT64)}})
      .Build();
  auto context = context_holder.GetContext<KernelContext>();

  auto funcs = registry.FindKernelFuncs(kKernelName);
  ASSERT_NE(funcs, nullptr);
  FastNodeFaker fast_node_faker;
  auto node = fast_node_faker.Build();
  (void)ge::AttrUtils::SetInt(node->GetOpDescBarePtr(), "dtype", ge::DT_INT64);
  ASSERT_EQ(funcs->outputs_creator(node, context), ge::GRAPH_SUCCESS);
  EXPECT_NE(funcs->run_func(context), ge::GRAPH_SUCCESS);
  context_holder.FreeAll();
}

TEST_F(GatherShapesKernelUt, AxesDimIndexOutOfBoundsFailed) {
  StorageShape shape = {{10,20,30}, {10,20,30}};
  StorageShape shape2 = {{10,20,30}, {10,20,30}};
  Shape kOutputData{30, 20};
  const std::vector<std::vector<int64_t>> axes{{1, 2}, {0, 3}};
  auto context_holder = KernelRunContextFaker()
      .KernelIONum(2U, static_cast<size_t>(kernel::GatherShapesKernelOutputs::kEnd))
      .Inputs({&shape, &shape2})
      .NodeAttrs({{"axes", ge::AnyValue::CreateFrom<std::vector<std::vector<int64_t>>>(axes)},
                  {"dtype", ge::AnyValue::CreateFrom<int64_t>(ge::DataType::DT_INT64)}})
      .Build();
  auto context = context_holder.GetContext<KernelContext>();

  auto funcs = registry.FindKernelFuncs(kKernelName);
  ASSERT_NE(funcs, nullptr);
  FastNodeFaker fast_node_faker;
  auto node = fast_node_faker.Build();
  (void)ge::AttrUtils::SetInt(node->GetOpDescBarePtr(), "dtype", ge::DT_INT64);
  ASSERT_EQ(funcs->outputs_creator(node, context), ge::GRAPH_SUCCESS);
  EXPECT_NE(funcs->run_func(context), ge::GRAPH_SUCCESS);
  context_holder.FreeAll();
}
}  // namespace gert