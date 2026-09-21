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
#include "stub/gert_runtime_stub.h"
#include "faker/kernel_run_context_facker.h"
#include "register/kernel_registry.h"
#include "graph/op_desc.h"
#include "mmpa/mmpa_api.h"
#include "core/debug/kernel_tracing.h"

namespace gert {
class StarsKernelLaunchST : public testing::Test {
 public:
  KernelRegistry &registry = KernelRegistry::GetInstance();
};

TEST_F(StarsKernelLaunchST, STARS_KernelLaunch) {
  auto context = BuildKernelRunContext(3, 0);
  void *addr = reinterpret_cast<void *>(0x10);
  size_t len = 0x11;
  rtStream_t stream = reinterpret_cast<void *>(0x12);

  context.value_holder[0].Set(reinterpret_cast<void *>(addr), nullptr);
  context.value_holder[1].Set(reinterpret_cast<void *>(len), nullptr);
  context.value_holder[2].Set(stream, nullptr);

  ASSERT_EQ(registry.FindKernelFuncs("StarsTaskLaunchKernel")->run_func(context), ge::GRAPH_SUCCESS);
  EXPECT_FALSE(registry.FindKernelFuncs("StarsTaskLaunchKernel")->trace_printer(context).empty());
}
}  // namespace gert