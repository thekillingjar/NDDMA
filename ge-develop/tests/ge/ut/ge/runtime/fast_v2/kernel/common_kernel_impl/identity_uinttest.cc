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
#include "core/utils/tensor_utils.h"
#include "exe_graph/runtime/gert_tensor_data.h"

namespace gert {
class IdentityKernelUT : public testing::Test {
 public:
  KernelRegistry &registry = KernelRegistry::GetInstance();
};

TEST_F(IdentityKernelUT, test_identity_shape_and_addr_succeed) {
  auto run_context = KernelRunContextFaker().KernelIONum(2, 2).NodeIoNum(2, 0).IrInputNum(2).Build();
  auto kernel_funcs = registry.FindKernelFuncs("IdentityShapeAndAddr");
  ASSERT_NE(kernel_funcs, nullptr);
  ASSERT_EQ(kernel_funcs->outputs_creator(nullptr, run_context), ge::GRAPH_SUCCESS);
  Tensor shape_tensor;
  shape_tensor.MutableOriginShape().SetDimNum(1);
  shape_tensor.MutableOriginShape().SetDim(0, 8);
  shape_tensor.MutableStorageShape().SetDimNum(2);
  shape_tensor.MutableStorageShape().SetDim(0, 2);
  shape_tensor.MutableStorageShape().SetDim(1, 3);
  void *address = reinterpret_cast<void *>(0xABC);
  GertTensorData data = {address, 64U, TensorPlacement::kFollowing, -1};
  run_context.value_holder[0].Set(&shape_tensor, nullptr);
  run_context.value_holder[1].Set(&data, nullptr);
  ASSERT_EQ(kernel_funcs->run_func(run_context), ge::GRAPH_SUCCESS);
  auto output_shape = run_context.GetContext<KernelContext>()->GetOutputPointer<StorageShape>(0);
  auto output_tensor = run_context.GetContext<KernelContext>()->GetOutputPointer<Tensor>(0);
  auto output_data = run_context.GetContext<KernelContext>()->GetOutputPointer<TensorData>(1);
  ASSERT_EQ(output_shape->GetStorageShape(), shape_tensor.GetStorageShape());
  ASSERT_EQ(output_shape->GetOriginShape(), shape_tensor.GetOriginShape());
  ASSERT_EQ(output_tensor->GetStorageShape(), shape_tensor.GetStorageShape());
  ASSERT_EQ(output_tensor->GetOriginShape(), shape_tensor.GetOriginShape());
  ASSERT_EQ(output_tensor->GetAddr(), nullptr);
  ASSERT_EQ(output_data->GetAddr(), data.GetAddr());
  ASSERT_EQ(output_data->GetPlacement(), data.GetPlacement());
  ASSERT_EQ(output_data->GetSize(), data.GetSize());
  run_context.FreeAll();
}

TEST_F(IdentityKernelUT, test_identity_shape_succeed) {
  auto run_context = KernelRunContextFaker().KernelIONum(1, 1).NodeIoNum(1, 0).IrInputNum(1).Build();
  auto kernel_funcs = registry.FindKernelFuncs("IdentityShape");
  ASSERT_NE(kernel_funcs, nullptr);
  ASSERT_EQ(kernel_funcs->outputs_creator(nullptr, run_context), ge::GRAPH_SUCCESS);
  Tensor shape_tensor;
  shape_tensor.MutableOriginShape().SetDimNum(1);
  shape_tensor.MutableOriginShape().SetDim(0, 8);
  shape_tensor.MutableStorageShape().SetDimNum(2);
  shape_tensor.MutableStorageShape().SetDim(0, 2);
  shape_tensor.MutableStorageShape().SetDim(1, 3);
  run_context.value_holder[0].Set(&shape_tensor, nullptr);
  ASSERT_EQ(kernel_funcs->run_func(run_context), ge::GRAPH_SUCCESS);
  auto output_shape = run_context.GetContext<KernelContext>()->GetOutputPointer<StorageShape>(0);
  ASSERT_EQ(output_shape->GetStorageShape(), shape_tensor.GetStorageShape());
  ASSERT_EQ(output_shape->GetOriginShape(), shape_tensor.GetOriginShape());
  auto output_tensor = run_context.GetContext<KernelContext>()->GetOutputPointer<Tensor>(0);
  ASSERT_EQ(output_tensor->GetStorageShape(), shape_tensor.GetStorageShape());
  ASSERT_EQ(output_tensor->GetOriginShape(), shape_tensor.GetOriginShape());
  ASSERT_EQ(output_tensor->GetAddr(), nullptr);
  run_context.FreeAll();
}
}  // namespace gert
