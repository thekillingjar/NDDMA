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
#include "depends/runtime/src/runtime_stub.h"
#include "graph/ge_local_context.h"

#include "macro_utils/dt_public_scope.h"
#include "common/mem_grp/memory_group_manager.h"
#include "macro_utils/dt_public_unscope.h"

using namespace std;
using namespace ::testing;

namespace ge {
class UtMemoryGroupManager : public testing::Test {
 public:
  UtMemoryGroupManager() {}
 protected:
  void SetUp() override {}
  void TearDown() override {
    ge::MemoryGroupManager::GetInstance().Finalize();
  }
};

class MockRuntime : public RuntimeStub {
 public:
  MOCK_METHOD3(rtMemGrpCacheAlloc, int32_t(const char *, int32_t, const rtMemGrpCacheAllocPara *));
};


TEST_F(UtMemoryGroupManager, run_MemGrpCreate) {
  std::string group_name = "DM_QS_GROUP";
  EXPECT_EQ(ge::MemoryGroupManager::GetInstance().MemGrpCreate(group_name), ge::SUCCESS);
}

TEST_F(UtMemoryGroupManager, run_MemGrpCreateFailed) {
  std::string group_name = "DN_QS_GROUP";
  EXPECT_NE(ge::MemoryGroupManager::GetInstance().MemGrpCreate(group_name), ge::SUCCESS);
}

TEST_F(UtMemoryGroupManager, run_MemGrpAddProc) {
  std::string group_name = "DM_QS_GROUP";
  EXPECT_EQ(ge::MemoryGroupManager::GetInstance().MemGrpAddProc(group_name, 0, true, true), ge::SUCCESS);
}

TEST_F(UtMemoryGroupManager, run_MemGrpAddProcError) {
  std::string group_name = "DM_QS_GROUP";
  EXPECT_NE(ge::MemoryGroupManager::GetInstance().MemGrpAddProc(group_name, -1, true, true), ge::SUCCESS);
}

TEST_F(UtMemoryGroupManager, run_MemGrpAddProcFailed) {
  std::string group_name = "DN_QS_GROUP";
  EXPECT_EQ(ge::MemoryGroupManager::GetInstance().MemGrpAddProc(group_name, 0, true, true), ge::SUCCESS);
}

TEST_F(UtMemoryGroupManager, run_cpu_InitializeSuccess) {
  NodeConfig node_config = {};
  DeviceConfig device_config = {};
  device_config.device_type = CPU;
  device_config.device_id = 0;
  node_config.device_list.emplace_back(device_config);
  auto ret = ge::MemoryGroupManager::GetInstance().MemGroupInit(node_config, "DM_QS_GROUP");
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtMemoryGroupManager, run_npu_InitializeSuccess) {
  NodeConfig node_config = {};
  DeviceConfig device_config = {};
  device_config.device_type = NPU;
  device_config.device_id = 0;
  node_config.device_list.emplace_back(device_config);
  device_config.device_id = 1;
  node_config.device_list.emplace_back(device_config);
  auto ret = ge::MemoryGroupManager::GetInstance().MemGroupInit(node_config, "DM_QS_GROUP");
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtMemoryGroupManager, run_RemoteMemGroupSuccess) {
  ge::MemoryGroupManager group_manager;
  auto group_name = group_manager.GetRemoteMemGroupName(0);
  auto ret = group_manager.RemoteMemGrpAddProc(0, group_name, 111, false, true);
  ASSERT_EQ(ret, SUCCESS);
  ret = group_manager.SetRemoteGroupCacheConfig("");
  ASSERT_EQ(ret, SUCCESS);

  std::string remote_group_cache_config = std::to_string(UINT32_MAX);
  ret = group_manager.SetRemoteGroupCacheConfig(remote_group_cache_config);
  ASSERT_EQ(ret, SUCCESS);

  remote_group_cache_config = "99999999999999999999";
  ret = group_manager.SetRemoteGroupCacheConfig(remote_group_cache_config);
  ASSERT_NE(ret, SUCCESS);
}

TEST_F(UtMemoryGroupManager, run_RemoteMemGroupMultiPoolTest) {
  ge::MemoryGroupManager group_manager;
  // one pool with alloc limit
  std::string remote_group_cache_config =
      std::to_string(15 * 1024 * 1024 * 1024UL) + ":" + std::to_string(10 * 1024 * 1024);
  auto ret = group_manager.SetRemoteGroupCacheConfig(remote_group_cache_config);
  ASSERT_EQ(ret, SUCCESS);

  // one pool with alloc limit 0
  remote_group_cache_config = std::to_string(15 * 1024 * 1024 * 1024UL) + ":0";
  ret = group_manager.SetRemoteGroupCacheConfig(remote_group_cache_config);
  ASSERT_EQ(ret, SUCCESS);

  // one pool with alloc limit too small(less than 1024)
  remote_group_cache_config = std::to_string(15 * 1024 * 1024 * 1024UL) + ":1023";
  ret = group_manager.SetRemoteGroupCacheConfig(remote_group_cache_config);
  ASSERT_NE(ret, SUCCESS);

  // 2 pools both with alloc limit
  remote_group_cache_config = std::to_string(15 * 1024 * 1024 * 1024UL) + ":" + std::to_string(10 * 1024 * 1024) + "," +
                              std::to_string(2 * 1024 * 1024 * 1024UL) + ":" + std::to_string(2 * 1024 * 1024);
  ret = group_manager.SetRemoteGroupCacheConfig(remote_group_cache_config);
  ASSERT_EQ(ret, SUCCESS);

  // 2 pools one with alloc limit
  remote_group_cache_config = std::to_string(15 * 1024 * 1024 * 1024UL) + "," +
                              std::to_string(2 * 1024 * 1024 * 1024UL) + ":" + std::to_string(2 * 1024 * 1024);
  ret = group_manager.SetRemoteGroupCacheConfig(remote_group_cache_config);
  ASSERT_EQ(ret, SUCCESS);

  // 2 over max memory
  remote_group_cache_config = std::to_string(200 * 1024 * 1024 * 1024UL) + "," +
                              std::to_string(200 * 1024 * 1024 * 1024UL) + ":" + std::to_string(20 * 1024 * 1024);
  ret = group_manager.SetRemoteGroupCacheConfig(remote_group_cache_config);
  ASSERT_NE(ret, SUCCESS);
}
}
