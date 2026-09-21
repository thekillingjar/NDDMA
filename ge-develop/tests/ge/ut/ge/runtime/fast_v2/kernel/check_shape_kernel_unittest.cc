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

class CheckShapeKernelUT : public testing::Test {
 public:
  KernelRegistry &registry = KernelRegistry::GetInstance();
};

TEST_F(CheckShapeKernelUT, CheckShapeKernelUT_CheckOutputShapesEmpty_EmptyTensor_success) {
  StorageShape shape1 = {{0, 1}, {0, 1}};
  StorageShape shape2 = {{0, 2}, {0, 2}};

  auto run_context =
      KernelRunContextFaker()
          .NodeIoNum(1UL, 2UL)
          .IrInputNum(2UL)
          .KernelIONum(2UL, 1UL)
          .Inputs({&shape1, &shape2})
          .Build();

  auto find_func = registry.FindKernelFuncs("CheckOutputShapesEmpty");
  ASSERT_NE(find_func, nullptr);
  ASSERT_EQ(find_func->run_func(run_context), ge::SUCCESS);
  auto cond = run_context.value_holder[2].GetPointer<uint32_t>();
  ASSERT_EQ(*cond, 0U);
}

TEST_F(CheckShapeKernelUT, CheckShapeKernelUT_CheckOutputShapesEmpty_NotAllEmptyTensor_success) {
  StorageShape shape1 = {{1, 0}, {1, 0}};
  StorageShape shape2 = {{2, 2}, {2, 2}};

  auto run_context =
      KernelRunContextFaker()
          .NodeIoNum(1UL, 2UL)
          .IrInputNum(2UL)
          .KernelIONum(2UL, 1UL)
          .Inputs({&shape1, &shape2})
          .Build();

  auto find_func = registry.FindKernelFuncs("CheckOutputShapesEmpty");
  ASSERT_NE(find_func, nullptr);
  ASSERT_EQ(find_func->run_func(run_context), ge::SUCCESS);
  auto cond = run_context.value_holder[2].GetPointer<uint32_t>();
  ASSERT_EQ(*cond, 1U);
}
} // namespace gert