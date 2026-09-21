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
#include "hostcpu_engine/engine/hostcpu_engine.h"

#include "util/util.h"
#include "config/config_file.h"
#include "stub.h"

#include "ge/ge_api_types.h"

using namespace aicpu;
using namespace ge;
using namespace std;

TEST(HostCpuKernelInfo, Initialize_SUCCESS)
{
    map<string, string> options;
    options[SOC_VERSION] = "Ascend910";
    ASSERT_EQ(Initialize(options), SUCCESS);
    map<string, OpsKernelInfoStorePtr> opsKernelInfoStores;
    GetOpsKernelInfoStores(opsKernelInfoStores);
    ASSERT_NE(opsKernelInfoStores["DNN_VM_HOST_CPU_OP_STORE"], nullptr);
    ASSERT_EQ(opsKernelInfoStores["DNN_VM_HOST_CPU_OP_STORE"]->Initialize(options), SUCCESS);
}

TEST(HostCpuKernelInfo, NewOpp_Initialize_SUCCESS)
{
  char *build_path = getenv("BUILD_PATH");
  std::string opp_path = std::string(build_path);
  setenv("ASCEND_OPP_PATH", opp_path.c_str(), 1);

  map<string, string> options;
  options[SOC_VERSION] = "Ascend910";
  ASSERT_EQ(Initialize(options), SUCCESS);
  map<string, OpsKernelInfoStorePtr> opsKernelInfoStores;
  GetOpsKernelInfoStores(opsKernelInfoStores);
  ASSERT_NE(opsKernelInfoStores["DNN_VM_HOST_CPU_OP_STORE"], nullptr);
  ASSERT_EQ(opsKernelInfoStores["DNN_VM_HOST_CPU_OP_STORE"]->Initialize(options), SUCCESS);
  unsetenv("ASCEND_OPP_PATH");
}
