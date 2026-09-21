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
#include "deploy/deployer/deployer_proxy.h"
#include "deploy/resource/resource_manager.h"
#include "macro_utils/dt_public_unscope.h"

using namespace std;
namespace ge {
class ResourceManagerTest : public testing::Test {
 protected:
  void SetUp() override {
    cpu_info_ = DeviceInfo(0, CPU, 0);
    cpu_info_.SetDgwPort(6666);
    npu_info_ = DeviceInfo(1, NPU, 0);
    npu_info_.SetDgwPort(16666);
    npu2_info_ = DeviceInfo(1, NPU, 1);
    npu2_info_.SetDgwPort(16667);
    DeployerProxy::GetInstance().Finalize();
  }
  void TearDown() override {
    DeployerProxy::GetInstance().Finalize();
  }

  DeviceInfo cpu_info_;
  DeviceInfo npu_info_;
  DeviceInfo npu2_info_;
};

TEST_F(ResourceManagerTest, TestResourceManager) {
  DeployerProxy deployer_proxy;
  {
    auto deployer_1 = MakeUnique<LocalDeployer>();
    deployer_1->node_info_.AddDeviceInfo(cpu_info_);
    DeployerProxy::GetInstance().deployers_.emplace_back(std::move(deployer_1));
  }
  {
    auto deployer_2 = MakeUnique<LocalDeployer>();
    deployer_2->node_info_.AddDeviceInfo(npu_info_);
    DeployerProxy::GetInstance().deployers_.emplace_back(std::move(deployer_2));
  }
  ResourceManager resource_manager;
  ASSERT_EQ(resource_manager.Initialize(), SUCCESS);
  ASSERT_EQ(resource_manager.GetNpuDeviceInfoList().size(), 1);
  ASSERT_EQ(resource_manager.GetDeviceInfoList().size(), 2);
  ASSERT_TRUE(resource_manager.GetDeviceInfo(0, -1, CPU) == nullptr);
  ASSERT_TRUE(resource_manager.GetDeviceInfo(0, 0, CPU) != nullptr);
  ASSERT_TRUE(resource_manager.GetDeviceInfo(1, 0, NPU) != nullptr);
  ASSERT_TRUE(resource_manager.GetDeviceInfo(1, -1, CPU) == nullptr);
  ASSERT_TRUE(resource_manager.GetDeviceInfo(1, 1, NPU) == nullptr);
}

TEST_F(ResourceManagerTest, TestResourceManager_2PG) {
  DeployerProxy deployer_proxy;
  {
    auto deployer_1 = MakeUnique<LocalDeployer>();
    deployer_1->node_info_.AddDeviceInfo(cpu_info_);
    DeployerProxy::GetInstance().deployers_.emplace_back(std::move(deployer_1));
  }
  {
    auto deployer_2 = MakeUnique<LocalDeployer>();
    deployer_2->node_info_.AddDeviceInfo(npu_info_);
    deployer_2->node_info_.AddDeviceInfo(npu2_info_);
    DeployerProxy::GetInstance().deployers_.emplace_back(std::move(deployer_2));
  }
  ResourceManager resource_manager;
  ASSERT_EQ(resource_manager.Initialize(), SUCCESS);
  ASSERT_EQ(resource_manager.GetNpuDeviceInfoList().size(), 2);
  ASSERT_EQ(resource_manager.GetDeviceInfoList().size(), 3);
  ASSERT_TRUE(resource_manager.GetDeviceInfo(0, 0, CPU) != nullptr);
  ASSERT_EQ(resource_manager.GetDeviceInfo(0, 0, CPU)->GetDgwPort(), 6666);

  ASSERT_TRUE(resource_manager.GetDeviceInfo(1, 0, NPU) != nullptr);
  ASSERT_EQ(resource_manager.GetDeviceInfo(1, 0, NPU)->GetDgwPort(), 16666);

  ASSERT_TRUE(resource_manager.GetDeviceInfo(1, 1, NPU) != nullptr);
  ASSERT_EQ(resource_manager.GetDeviceInfo(1, 1, NPU)->GetDgwPort(), 16667);

  ASSERT_TRUE(resource_manager.GetDeviceInfo(1, -1, CPU) == nullptr);
  ASSERT_TRUE(resource_manager.GetDeviceInfo(1, 2, NPU) == nullptr);
}

TEST_F(ResourceManagerTest, TestResourceManager_Server) {
  DeployerProxy deployer_proxy;
  {
    auto deployer_1 = MakeUnique<LocalDeployer>();
    deployer_1->node_info_.AddDeviceInfo(DeviceInfo(0, CPU, 0));
    deployer_1->node_info_.AddDeviceInfo(DeviceInfo(0, NPU, 0));
    deployer_1->node_info_.AddDeviceInfo(DeviceInfo(0, NPU, 1));
    deployer_1->node_info_.AddDeviceInfo(DeviceInfo(1, NPU, 0));
    DeployerProxy::GetInstance().deployers_.emplace_back(std::move(deployer_1));
  }
  ResourceManager resource_manager;
  ASSERT_EQ(resource_manager.Initialize(), SUCCESS);
  ASSERT_EQ(resource_manager.GetNpuDeviceInfoList().size(), 3);
  ASSERT_EQ(resource_manager.GetDeviceInfoList().size(), 4);
  ASSERT_TRUE(resource_manager.GetDeviceInfo(0, 0, NPU) != nullptr);
  ASSERT_TRUE(resource_manager.GetDeviceInfo(0, 0, CPU) != nullptr);
  ASSERT_TRUE(resource_manager.GetDeviceInfo(1, 0, CPU) == nullptr);
  ASSERT_TRUE(resource_manager.GetDeviceInfo(2, 0, NPU) == nullptr);
  ASSERT_TRUE(resource_manager.GetDeviceInfo(0, 5, NPU) == nullptr);
}

TEST_F(ResourceManagerTest, TestNodeInfo) {
  NodeInfo node_info(true);
  node_info.AddDeviceInfo(cpu_info_);
  ASSERT_EQ(node_info.GetDeviceList().size(), 1);
  ASSERT_EQ(node_info.IsLocal(), true);

  NodeInfo node2_info(false);
  node2_info.AddDeviceInfo(cpu_info_);
  ASSERT_EQ(node2_info.GetDeviceList().size(), 1);
  ASSERT_EQ(node2_info.IsLocal(), false);
}
}  // namespace ge
