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

#include "macro_utils/dt_public_scope.h"
#include "compiler/engines/manager/opskernel_manager/ops_kernel_manager.h"
#include "ge_running_env/ge_running_env_faker.h"
#include "macro_utils/dt_public_unscope.h"

namespace ge {
class OpsKernelManagerSt : public testing::Test {};

TEST_F(OpsKernelManagerSt, register_test_ok) {
  GeRunningEnvFaker ge_env;
  ge_env.InstallDefault();
  EXPECT_NO_THROW(OpsKernelManager::GetInstance().InitGraphOptimizerPriority());
}
}
