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
#include "faker/continuous_vector_builder.h"
#include "runtime/base.h"
#include "depends/mmpa/src/mmpa_stub.h"
#include "common/ge_inner_error_codes.h"
#include "register/kernel_registry.h"
#include "checker/memory_profiling_log_matcher.h"
#include "stub/gert_runtime_stub.h"

namespace gert {
class NotifyKernelUT : public testing::Test {
 public:
  KernelRegistry &registry = KernelRegistry::GetInstance();
};

TEST_F(NotifyKernelUT, TEST_NOTIFY_SUCCESS) {
  auto streams = ContinuousVectorBuilder::Create<rtNotify_t>(
      {(rtNotify_t)0x100, (rtNotify_t)0x200, (rtNotify_t)0x300, (rtNotify_t)0x400});
  auto context =
      KernelRunContextFaker().Inputs({streams.get(), (void *)4}).Outputs({nullptr, nullptr, nullptr, nullptr}).Build();
  ASSERT_EQ(registry.FindKernelFuncs("CreateNotifies")->run_func(context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(*context.GetContext<KernelContext>()->GetOutputPointer<rtNotify_t>(2), (rtNotify_t)0x300);
}

TEST_F(NotifyKernelUT, TEST_NOTIFY_OUT_OF_RAGED) {
  auto streams = ContinuousVectorBuilder::Create<rtNotify_t>(
      {(rtNotify_t)0x100, (rtNotify_t)0x200, (rtNotify_t)0x300, (rtNotify_t)0x400});
  auto context =
      KernelRunContextFaker().Inputs({streams.get(), (void *)5}).Outputs({nullptr, nullptr, nullptr, nullptr}).Build();
  ASSERT_NE(registry.FindKernelFuncs("CreateNotifies")->run_func(context), ge::GRAPH_SUCCESS);
}

TEST_F(NotifyKernelUT, TEST_NOTIFY_NULLPTR) {
  auto context =
      KernelRunContextFaker().Inputs({nullptr, (void *)4}).Outputs({nullptr, nullptr, nullptr, nullptr}).Build();
  ASSERT_NE(registry.FindKernelFuncs("CreateNotifies")->run_func(context), ge::GRAPH_SUCCESS);
}
}  // namespace gert