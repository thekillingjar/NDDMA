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
#include "macro_utils/dt_public_scope.h"
#include "deploy/flowrm/flowgw_client_manager.h"
#include "macro_utils/dt_public_unscope.h"
#include "depends/mmpa/src/mmpa_stub.h"

namespace ge {
namespace {
class MockMmpa : public MmpaStubApiGe {
 public:
  int32_t Sleep(UINT32 microSecond) override {
    return 0;
  }
};

class MockFlowGwClientManager : public FlowGwClientManager {
 public:
  MOCK_CONST_METHOD4(CreateClient, FlowGwClient *(int32_t,
                                                  int32_t,
                                                  const std::vector<int32_t> &,
                                                  bool));
};

class MockFlowGwClient : public FlowGwClient {
 public:
  MockFlowGwClient(uint32_t device_id, int32_t device_type, const std::vector<int32_t> res_ids, bool is_proxy)
      : FlowGwClient(device_id, device_type, res_ids, is_proxy) {}
  virtual ~MockFlowGwClient() = default;
 public:
  MOCK_METHOD0(Initialize, Status());
  MOCK_CONST_METHOD0(GetSubProcStat, ProcStatus());
};
}

class UtFlowGwClientManager : public testing::Test {
 public:
  UtFlowGwClientManager() {}
 protected:
  void SetUp() override {
  }
  void TearDown() override {
  }
};

TEST_F(UtFlowGwClientManager, GetOrCreateClient_ReturnNull) {
  MockFlowGwClientManager client_manager;
  EXPECT_CALL(client_manager, CreateClient).WillOnce(testing::Return(nullptr));
  auto client = client_manager.GetOrCreateClient(0, 0, {0}, false);
  EXPECT_EQ(client, nullptr);
}

TEST_F(UtFlowGwClientManager, GetOrCreateClient_InitFailed) {
  auto mock_client = new (std::nothrow)MockFlowGwClient(0, 0, {0}, false);
  MockFlowGwClientManager client_manager;
  EXPECT_CALL(client_manager, CreateClient).WillRepeatedly(testing::Return(mock_client));
  EXPECT_CALL(*mock_client, Initialize).WillRepeatedly(testing::Return(FAILED));
  auto client = client_manager.GetOrCreateClient(0, 0, {0}, false);
  EXPECT_EQ(client, nullptr);
  client_manager.Finalize();
}

TEST_F(UtFlowGwClientManager, GetOrCreateClient) {
  auto mock_client = new (std::nothrow)MockFlowGwClient(0, 0, {0}, false);
  MockFlowGwClientManager client_manager;
  EXPECT_CALL(client_manager, CreateClient).WillRepeatedly(testing::Return(mock_client));
  EXPECT_CALL(*mock_client, Initialize).WillRepeatedly(testing::Return(SUCCESS));
  auto client = client_manager.GetOrCreateClient(0, 0, {0}, false);
  EXPECT_NE(client, nullptr);
  // test already created
  client = client_manager.GetOrCreateClient(0, 0, {0}, false);
  EXPECT_NE(client, nullptr);
  client_manager.Finalize();
}

TEST_F(UtFlowGwClientManager, GetFlowGwStatus) {
  auto mock_client = new (std::nothrow)MockFlowGwClient(0, 0, {0}, false);
  MockFlowGwClientManager client_manager;
  EXPECT_CALL(client_manager, CreateClient).WillRepeatedly(testing::Return(mock_client));
  EXPECT_CALL(*mock_client, GetSubProcStat).WillOnce(testing::Return(ProcStatus::NORMAL))
                                           .WillOnce(testing::Return(ProcStatus::EXITED))
                                           .WillOnce(testing::Return(ProcStatus::STOPPED));
  EXPECT_CALL(*mock_client, Initialize).WillRepeatedly(testing::Return(SUCCESS));
  auto client = client_manager.GetOrCreateClient(0, 0, {0}, false);
  EXPECT_NE(client, nullptr);
  deployer::DeployerResponse response1;
  client_manager.GetFlowGwStatus(response1);
  deployer::DeployerResponse response2;
  client_manager.GetFlowGwStatus(response2);
  deployer::DeployerResponse response3;
  client_manager.GetFlowGwStatus(response3);
  client_manager.Finalize();
}

TEST_F(UtFlowGwClientManager, BindQueuesDiffDeviceType) {
  auto mock_client = new (std::nothrow)MockFlowGwClient(0, 1, {0}, false);
  MockFlowGwClientManager client_manager;
  EXPECT_CALL(client_manager, CreateClient).WillRepeatedly(testing::Return(mock_client));
  EXPECT_CALL(*mock_client, Initialize).WillRepeatedly(testing::Return(SUCCESS));
  auto client = client_manager.GetOrCreateClient(0, 1, {0}, false);
  EXPECT_NE(client, nullptr);

  std::vector<std::pair<const ExchangeEndpoint *,
                        const ExchangeEndpoint *>> queue_routes;
  ExchangeEndpoint src = {};
  ExchangeEndpoint dst = {};
  src.type = ExchangeEndpointType::kEndpointTypeQueue;
  dst.type = ExchangeEndpointType::kEndpointTypeQueue;
  src.device_id = 0;
  dst.device_id = 0;
  src.device_type = 0;
  dst.device_type = 1;
  src.id = 1;
  dst.id = 2;
  queue_routes.emplace_back(std::make_pair(&src, &dst));
  auto ret = client_manager.BindQueues(queue_routes);
  EXPECT_EQ(ret, SUCCESS);
  src.device_type = 1;
  dst.device_type = 0;
  queue_routes.clear();
  queue_routes.emplace_back(std::make_pair(&src, &dst));
  EXPECT_EQ(ret, SUCCESS);
  client_manager.Finalize();
}
}
