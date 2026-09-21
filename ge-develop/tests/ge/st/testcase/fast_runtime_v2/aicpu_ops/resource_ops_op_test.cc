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
#include "register/kernel_registry_impl.h"
#include "faker/kernel_run_context_facker.h"
#include <iostream>
#include "aicpu/converter/sequence/resource_mgr.h"

namespace gert {
class TEST_RESOURCE_OPS_UT : public testing::Test {};

TEST_F(TEST_RESOURCE_OPS_UT, TestCreateSession) {
  uint64_t session_id = 31;
  uint64_t output_session_id = 0;
  auto kernelHolder = gert::KernelRunContextFaker()
                          .KernelIONum(1, 1)
                          .Inputs({reinterpret_cast<void*>(session_id)})
                          .Outputs({reinterpret_cast<void*>(output_session_id)})
                          .Build();
  auto context = kernelHolder.GetContext<KernelContext>();
  auto ret = KernelRegistry::GetInstance().FindKernelFuncs("CreateSession")->run_func(context);
  ASSERT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(TEST_RESOURCE_OPS_UT, TestClearStepContainer) {
  uint64_t session_id = 31;
  uint64_t container_id = 32;
  auto kernelHolder = gert::KernelRunContextFaker()
                          .KernelIONum(2, 0)
                          .Inputs({reinterpret_cast<void*>(session_id), reinterpret_cast<void*>(container_id)})
                          .Build();
  auto context = kernelHolder.GetContext<KernelContext>();
  auto ret = KernelRegistry::GetInstance().FindKernelFuncs("ClearContainer")->run_func(context);
  ASSERT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(TEST_RESOURCE_OPS_UT, TestDestroySession) {
  uint64_t session_id = 31;
  auto kernelHolder = gert::KernelRunContextFaker()
                          .KernelIONum(1, 0)
                          .Inputs({reinterpret_cast<void*>(session_id)})
                          .Build();
  auto context = kernelHolder.GetContext<KernelContext>();
  auto ret = KernelRegistry::GetInstance().FindKernelFuncs("DestroySession")->run_func(context);
  ASSERT_EQ(ret, ge::GRAPH_SUCCESS);
}
}  // namespace gert