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

#include <vector>
#include "rt_error_codes.h"
#include "framework/common/debug/ge_log.h"
#include "depends/mmpa/src/mmpa_stub.h"
#include "depends/runtime/src/runtime_stub.h"
#include "depends/helper_runtime/src/caas_dataflow_auth_stub.h"

#include "macro_utils/dt_public_scope.h"
#include "deploy/deployer/deployer.h"
#include "deploy/resource/resource_manager.h"
#include "daemon/daemon_service.h"
#include "deploy/deployer/deployer_authentication.h"
#include "macro_utils/dt_public_unscope.h"
#include "deploy/abnormal_status_handler/device_abnormal_status_handler.h"

using namespace std;
namespace ge {
class DeployerTest : public testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(DeployerTest, TestProcessFail) {
  NodeConfig node_config;
  RemoteDeployer remote_deployer(node_config);
  deployer::DeployerRequest request;
  deployer::DeployerResponse response;
  ASSERT_NE(remote_deployer.Process(request, response), SUCCESS);
}
}  // namespace ge
