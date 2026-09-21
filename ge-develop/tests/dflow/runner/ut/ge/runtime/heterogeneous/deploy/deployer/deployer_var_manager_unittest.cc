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
#include <gmock/gmock.h>
#include <vector>
#include <string>
#include <map>
#include "framework/common/debug/ge_log.h"
#include "macro_utils/dt_public_scope.h"
#include "deploy/deployer/deployer_var_manager.h"
#include "common/config/configurations.h"
#include "macro_utils/dt_public_unscope.h"
#include "depends/runtime/src/runtime_stub.h"

namespace ge {
class DeployerVarManagerTest : public testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(DeployerVarManagerTest, TestInitilizeFinalize) {
  DeployerVarManager deployer_var_manager;
  deployer::VarManagerInfo var_info;
  var_info.set_use_max_mem_size(12800);
  var_info.set_session_id(1);
  ASSERT_EQ(deployer_var_manager.Initialize(var_info), SUCCESS);

  rtMbufPtr_t mbuf;
  rtMbufAlloc(&mbuf, 1);
  deployer_var_manager.var_mbuf_vec_.emplace_back(mbuf);
  // Finalize will call rtMbufFree
  deployer_var_manager.Finalize();
}

TEST_F(DeployerVarManagerTest, TestInitilizeFinalizeRemoteDevice) {
  DeployerVarManager deployer_var_manager;
  deployer::VarManagerInfo var_info;
  var_info.set_use_max_mem_size(12800);
  var_info.set_session_id(1);
  ASSERT_EQ(deployer_var_manager.Initialize(var_info), SUCCESS);

  rtMbufPtr_t mbuf;
  rtMbufAlloc(&mbuf, 1);
  deployer_var_manager.var_mbuf_vec_.emplace_back(mbuf);
  // Finalize will call ProxyEventManager::FreeMbuf
  deployer_var_manager.Finalize();
  rtMbufFree(mbuf);
}

TEST_F(DeployerVarManagerTest, TestProcessVarManager) {
  DeployerVarManager deployer_var_manager;
  deployer::VarManagerInfo var_info;
  var_info.set_use_max_mem_size(12800);
  var_info.set_session_id(1);
  ASSERT_EQ(deployer_var_manager.Initialize(var_info), SUCCESS);

  deployer::SharedContentDescription shared_content_desc;
  shared_content_desc.set_session_id(1);
  shared_content_desc.set_node_name("node");
  shared_content_desc.set_om_content("hello");
  shared_content_desc.set_total_length(4);
  shared_content_desc.set_current_offset(0);
  ASSERT_EQ(deployer_var_manager.ProcessSharedContent(shared_content_desc, 4, 0, 0), SUCCESS);

  deployer_var_manager.Finalize();
}
}  // namespace ge
