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
#include "common/data_flow/event/proxy_event_manager.h"

namespace ge {
class UtProxyEventManager : public testing::Test {
 public:
  UtProxyEventManager() {}
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(UtProxyEventManager, run_GetProxyPid) {
  pid_t pid = 0;
  auto ret = ProxyEventManager::GetProxyPid(0, pid);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtProxyEventManager, run_CreateGroup) {
  auto ret = ProxyEventManager::CreateGroup(0, "GRP", 10);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtProxyEventManager, run_AddGroup) {
  auto ret = ProxyEventManager::AddGroup(0, "GRP", 123, false, true);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtProxyEventManager, run_CreateGroupSinglePool) {
  uint64_t mem_size = 2 * 1024 * 1024 * 1024UL;
  std::vector<std::pair<uint64_t, uint32_t>> pool_list = {{mem_size, 0}};
  auto ret = ProxyEventManager::CreateGroup(0, "GRP", mem_size, pool_list);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtProxyEventManager, run_CreateGroupSinglePoolWithLimit) {
  uint64_t mem_size = 2 * 1024 * 1024 * 1024UL;
  std::vector<std::pair<uint64_t, uint32_t>> pool_list = {{mem_size, 2048}};
  auto ret = ProxyEventManager::CreateGroup(0, "GRP", mem_size, pool_list);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtProxyEventManager, run_CreateGroupMiltiPool) {
  uint64_t mem_size = 2 * 1024 * 1024 * 1024UL;
  std::vector<std::pair<uint64_t, uint32_t>> pool_list = {{mem_size, 0}, {mem_size, 2048}};
  auto ret = ProxyEventManager::CreateGroup(0, "GRP", mem_size, pool_list);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtProxyEventManager, run_Mbuf) {
  rtMbufPtr_t mbuf = nullptr;
  void *data_ptr = nullptr;
  auto ret = ProxyEventManager::AllocMbuf(0, 10, &mbuf, &data_ptr);
  EXPECT_EQ(ret, SUCCESS);
  ret = ProxyEventManager::FreeMbuf(0, mbuf);
  EXPECT_EQ(ret, SUCCESS);
  ret = ProxyEventManager::CopyQMbuf(0, 123456, 10, 0);
  EXPECT_EQ(ret, SUCCESS);
}
}
