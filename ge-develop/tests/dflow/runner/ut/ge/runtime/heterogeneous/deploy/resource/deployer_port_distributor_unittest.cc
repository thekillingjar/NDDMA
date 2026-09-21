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
#include "framework/common/debug/ge_log.h"

#include "macro_utils/dt_public_scope.h"
#include "deploy/resource/deployer_port_distributor.h"
#include "macro_utils/dt_public_unscope.h"

using namespace std;
namespace ge {
class DeployerPortDistributorTest : public testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(DeployerPortDistributorTest, TestAllocatePort) {
  DeployerPortDistributor distributor;
  int32_t port = -1;
  auto ret = distributor.AllocatePort("10.1.1.1", "1024~1026", port);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(port, 1024);
  ret = distributor.AllocatePort("10.1.1.1", "1024~ 1026", port);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(port, 1025);
  ret = distributor.AllocatePort("10.1.1.1", "1024 ~ 1026", port);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(port, 1026);
  // out of range
  ret = distributor.AllocatePort("10.1.1.1", "1024~1026", port);
  EXPECT_NE(ret, SUCCESS);
  distributor.Finalize();
  ret = distributor.AllocatePort("10.1.1.1", "1024~1026", port);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(port, 1024);
  distributor.Finalize();
}

TEST_F(DeployerPortDistributorTest, TestAllocatePortFailed) {
  DeployerPortDistributor distributor;
  int32_t port = -1;
  auto ret = distributor.AllocatePort("10.1.1.1", "1026~1024", port);
  EXPECT_NE(ret, SUCCESS);

  ret = distributor.AllocatePort("10.1.1.1", "xxx ~ 1026", port);
  EXPECT_NE(ret, SUCCESS);

  ret = distributor.AllocatePort("10.1.1.1", "1024 - 1026", port);
  EXPECT_NE(ret, SUCCESS);

  // range failed, not in [1024, 32767]
  ret = distributor.AllocatePort("10.1.1.1", "0 ~ 1026", port);
  EXPECT_NE(ret, SUCCESS);

  ret = distributor.AllocatePort("10.1.1.1", "1025 ~ 65536", port);
  EXPECT_NE(ret, SUCCESS);

  distributor.Finalize();
}
}  // namespace ge
