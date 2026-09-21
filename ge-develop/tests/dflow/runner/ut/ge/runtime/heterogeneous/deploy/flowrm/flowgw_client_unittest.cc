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
#include "deploy/flowrm/flowgw_client.h"
#include "deploy/flowrm/network_manager.h"
#include "macro_utils/dt_public_unscope.h"

#include "depends/mmpa/src/mmpa_stub.h"
#include "common/config/configurations.h"
#include "deploy/flowrm/tsd_client.h"
#include "common/env_path.h"

namespace ge {
namespace {
class MockMmpa : public MmpaStubApiGe {
 public:
  void *DlSym(void *handle, const char *func_name) override {
    std::cout << "func name:" << func_name << " begin to stub\n";
    if (std::string(func_name) == "TsdInitFlowGw") {
      std::cout << "func name:" << func_name << " stub success, addr = " << &TsdInitFlowGw << std::endl;
      return (void *) &TsdInitFlowGw;
    } else if (std::string(func_name) == "TsdProcessOpen") {
      return (void *) &TsdProcessOpen;
    } else if (std::string(func_name) == "ProcessCloseSubProcList") {
      return (void *) &ProcessCloseSubProcList;
    } else if (std::string(func_name) == "TsdFileLoad") {
      return (void *) &TsdFileLoad;
    }
    std::cout << "func name:" << func_name << " not stub\n";
    return (void *) 0xFFFFFFFF;
  }

  void *DlOpen(const char *fileName, int32_t mode) override {
    return (void *) 0xFFFFFFFF;
  }
  int32_t DlClose(void *handle) override {
    return 0L;
  }

  int32_t Sleep(UINT32 microSecond) override {
    return 0;
  }
  int32_t RealPath(const CHAR *path, CHAR *realPath, INT32 realPathLen) override {
    memcpy_s(realPath, realPathLen, path, strlen(path));
    return 0;
  }
};

class MockFlowGwClient : public FlowGwClient {
 public:
  MockFlowGwClient(uint32_t device_id, int32_t device_type, const std::vector<int32_t> res_ids, bool is_proxy)
      : FlowGwClient(device_id, device_type, res_ids, is_proxy) {};
  MOCK_METHOD2(KillProcess, int32_t(pid_t, int32_t));
};
}

class UtFlowGwClient : public testing::Test {
 public:
  UtFlowGwClient() {}
 protected:
  void SetUp() override {
    MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  }
  void TearDown() override {
    MmpaStub::GetInstance().Reset();
    unsetenv("RESOURCE_CONFIG_PATH");
  }

  static void SetConfigEnv(const std::string &path) {
    std::string config_path = PathUtils::Join({EnvPath().GetAirBasePath(), "tests/dflow/runner/ut/ge/runtime/data", path});
    setenv("RESOURCE_CONFIG_PATH", config_path.c_str(), 1);
    EXPECT_EQ(Configurations::GetInstance().InitInformation(), SUCCESS);
  }
};

TEST_F(UtFlowGwClient, run_InitializeAndFinalize) {
  MockFlowGwClient flowgw_client(0, 0, {0}, false);
  EXPECT_CALL(flowgw_client, KillProcess).WillRepeatedly(testing::Return(1));
  EXPECT_EQ(flowgw_client.Initialize(), SUCCESS);
  EXPECT_EQ(flowgw_client.GetSubProcStat(), ProcStatus::NORMAL);
  flowgw_client.Finalize();

  FlowGwClient flowgw_proxy_client(0, 0, {0}, true);
  EXPECT_EQ(flowgw_proxy_client.Initialize(), SUCCESS);
  EXPECT_EQ(flowgw_client.GetSubProcStat(), ProcStatus::INVALID);
  flowgw_proxy_client.Finalize();
}

TEST_F(UtFlowGwClient, run_Exception) {
  MockFlowGwClient flowgw_client(0, 0, {0}, false);
  EXPECT_CALL(flowgw_client, KillProcess).WillRepeatedly(testing::Return(1));
  EXPECT_EQ(flowgw_client.Initialize(), SUCCESS);
  flowgw_client.SetExceptionFlag();
  const std::set<uint32_t> model_ids;
  EXPECT_EQ(flowgw_client.ClearFlowgwModelData(model_ids, 1),
      SUCCESS);
  flowgw_client.Finalize();
}

TEST_F(UtFlowGwClient, run_HostFlowgwInitialize) {
  MockFlowGwClient flowgw_client(0, 0, {0}, false);
  EXPECT_CALL(flowgw_client, KillProcess).WillRepeatedly(testing::Return(1));
  auto &node_config = Configurations::GetInstance().information_.node_config;
  auto node_bk = node_config;
  node_config.proxy_device_ids = {0, 1};
  EXPECT_EQ(flowgw_client.Initialize(), SUCCESS);
  EXPECT_EQ(flowgw_client.GetSubProcStat(), ProcStatus::NORMAL);
  flowgw_client.Finalize();
  // restore
  node_config = node_bk;
}

TEST_F(UtFlowGwClient, DataGwProcStatusChange) {
  FlowGwClient flowgw_client(0, 0, {0}, false);
  SubprocessManager::GetInstance().excpt_handle_callbacks_.clear();
  EXPECT_EQ(flowgw_client.Initialize(), SUCCESS);
  for(auto callback : SubprocessManager::GetInstance().excpt_handle_callbacks_){
    callback.second(ProcStatus::NORMAL);
  }
  SubprocessManager::GetInstance().Finalize();
}

TEST_F(UtFlowGwClient, run_GrantQueueForRoute) {
  FlowGwClient flowgw_client(0, 0, {0}, false);
  std::pair<const ExchangeEndpoint *,
            const ExchangeEndpoint *> queue_route;
  ExchangeEndpoint src = {};
  ExchangeEndpoint dst = {};
  src.type = ExchangeEndpointType::kEndpointTypeQueue;
  dst.type = ExchangeEndpointType::kEndpointTypeQueue;
  src.device_id = 0;
  dst.device_id = 0;
  src.device_type = 0;
  dst.device_type = 0;
  src.id = 1;
  dst.id = 2;
  queue_route = std::make_pair(&src, &dst);
  auto ret = flowgw_client.GrantQueueForRoute(queue_route);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtFlowGwClient, run_BindQueues) {
  FlowGwClient flowgw_client(0, 0, {0}, false);
  std::vector<std::pair<const ExchangeEndpoint *,
                        const ExchangeEndpoint *>> queue_routes;
  ExchangeEndpoint src = {};
  ExchangeEndpoint dst = {};
  src.type = ExchangeEndpointType::kEndpointTypeQueue;
  dst.type = ExchangeEndpointType::kEndpointTypeQueue;
  src.device_id = 0;
  dst.device_id = 0;
  src.device_type = 0;
  dst.device_type = 0;
  src.id = 1;
  dst.id = 2;
  queue_routes.emplace_back(std::make_pair(&src, &dst));
  auto ret = flowgw_client.BindQueues(queue_routes);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtFlowGwClient, run_GrantQueue) {
  uint32_t device_id = 0;
  uint32_t qid = 1;
  pid_t pid = 100;
  ge::GrantType grant_type = GrantType::kReadOnly;
  auto ret = FlowGwClient::GrantQueue(device_id, qid, pid, grant_type);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtFlowGwClient, run_UnbindQueues) {
  FlowGwClient flowgw_client(0, 0, {0}, false);
  std::vector<std::pair<const ExchangeEndpoint *,
                        const ExchangeEndpoint *>> queue_routes;
  ExchangeEndpoint src = {};
  ExchangeEndpoint dst = {};
  src.type = ExchangeEndpointType::kEndpointTypeQueue;
  dst.type = ExchangeEndpointType::kEndpointTypeQueue;
  src.device_id = 0;
  dst.device_id = 0;
  src.device_type = 0;
  dst.device_type = 0;
  src.id = 1;
  dst.id = 2;
  auto ret = flowgw_client.UnbindQueues(queue_routes);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtFlowGwClient, ToFlowGwEndpoint_InvalidType) {
  FlowGwClient flowgw_client(0, 0, {0}, false);
  ExchangeEndpoint ge_endpoint;
  ge_endpoint.type = ExchangeEndpointType::kEndpointTypeNone;
  ge_endpoint.id = 1;
  auto bqs_endpoint = flowgw_client.ToFlowGwEndpoint(ge_endpoint);
  EXPECT_NE(bqs_endpoint.type, bqs::EndpointType::MEM_QUEUE);
}

TEST_F(UtFlowGwClient, ToFlowGwEndpoint_Queue) {
  FlowGwClient flowgw_client(0, 0, {0}, false);
  ExchangeEndpoint ge_endpoint;
  ge_endpoint.id = 1;
  ge_endpoint.type = ExchangeEndpointType::kEndpointTypeQueue;
  auto bqs_endpoint = flowgw_client.ToFlowGwEndpoint(ge_endpoint);
  EXPECT_EQ(bqs_endpoint.type, bqs::EndpointType::MEM_QUEUE);
  EXPECT_EQ(bqs_endpoint.attr.memQueueAttr.queueId, 1);
}

TEST_F(UtFlowGwClient, ToFlowGwEndpoint_ExternalQueue) {
  FlowGwClient flowgw_client(0, 0, {0}, false);
  ExchangeEndpoint ge_endpoint;
  ge_endpoint.id = 1;
  ge_endpoint.type = ExchangeEndpointType::kEndpointTypeExternalQueue;
  auto bqs_endpoint = flowgw_client.ToFlowGwEndpoint(ge_endpoint);
  EXPECT_EQ(bqs_endpoint.type, bqs::EndpointType::MEM_QUEUE);
  EXPECT_EQ(bqs_endpoint.attr.memQueueAttr.queueId, 1);
}

TEST_F(UtFlowGwClient, ToFlowGwEndpoint_Group) {
  FlowGwClient flowgw_client(0, 0, {0}, false);
  ExchangeEndpoint ge_endpoint;
  ge_endpoint.id = 1;
  ge_endpoint.type = ExchangeEndpointType::kEndpointTypeGroup;
  auto bqs_endpoint = flowgw_client.ToFlowGwEndpoint(ge_endpoint);
  EXPECT_EQ(bqs_endpoint.type, bqs::EndpointType::GROUP);
  EXPECT_EQ(bqs_endpoint.attr.memQueueAttr.queueId, 1);
}

TEST_F(UtFlowGwClient, ToFlowGwEndpoint_Tag) {
  FlowGwClient flowgw_client(0, 0, {0}, false);
  ExchangeEndpoint ge_endpoint;
  ge_endpoint.type = ExchangeEndpointType::kEndpointTypeTag;
  ge_endpoint.rank_id = 2;
  ge_endpoint.peer_rank_id = 3;
  ge_endpoint.tag_id = 4;
  ge_endpoint.peer_tag_id = 40;
  ge_endpoint.hcom_handle = 6666;
  auto bqs_endpoint = flowgw_client.ToFlowGwEndpoint(ge_endpoint);
  EXPECT_EQ(bqs_endpoint.type, bqs::EndpointType::COMM_CHANNEL);
  EXPECT_EQ(bqs_endpoint.attr.channelAttr.handle, 6666);
  EXPECT_EQ(bqs_endpoint.attr.channelAttr.localTagId, 4);
  EXPECT_EQ(bqs_endpoint.attr.channelAttr.peerTagId, 40);
  EXPECT_EQ(bqs_endpoint.attr.channelAttr.localRankId, 2);
  EXPECT_EQ(bqs_endpoint.attr.channelAttr.peerRankId, 3);
}

TEST_F(UtFlowGwClient, run_DestroyInvalidHandle) {
  FlowGwClient flowgw_client(0, 0, {0}, false);
  auto ret = flowgw_client.DestroyHcomHandle();
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtFlowGwClient, run_DestroyValidHandle) {
  FlowGwClient flowgw_client(0, 0, {0}, false);
  flowgw_client.SetHcomInfo("test", 0U);
  auto ret = flowgw_client.CreateHcomHandle();
  EXPECT_EQ(ret, SUCCESS);
  ret = flowgw_client.DestroyHcomHandle();
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtFlowGwClient, run_GetHostPort) {
  SetConfigEnv("valid/server/numa_config.json");
  ge::NetworkManager::GetInstance().Initialize();
  int32_t port = -1;
  auto ret = ge::NetworkManager::GetInstance().GetDataPanelPort(port);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtFlowGwClient, run_GetHostIp) {
  SetConfigEnv("valid/server/numa_config.json");
  std::string ip;
  EXPECT_EQ(ge::NetworkManager::GetInstance().GetDataPanelIp(ip), SUCCESS);
}


TEST_F(UtFlowGwClient, run_CreateDataGwGroup) {
  FlowGwClient flowgw_client(0, 0, {0}, false);
  ExchangeEndpoint endpoint = {};
  endpoint.type = ExchangeEndpointType::kEndpointTypeQueue;
  std::vector<const ExchangeEndpoint *> endpoint_list;
  endpoint_list.emplace_back(&endpoint);
  int32_t group_id = 0;
  auto ret = flowgw_client.CreateFlowGwGroup(endpoint_list, group_id);
  ASSERT_EQ(ret, ge::SUCCESS);
}

TEST_F(UtFlowGwClient, run_CreateDataGwGroup_Empty) {
  FlowGwClient flowgw_client(0, 0, {0}, false);
  std::vector<const ExchangeEndpoint *> endpoint_list;
  int32_t group_id = 0;
  auto ret = flowgw_client.CreateFlowGwGroup(endpoint_list, group_id);
  ASSERT_NE(ret, ge::SUCCESS);
}

TEST_F(UtFlowGwClient, run_GrantQueue_WriteOnly) {
  uint32_t device_id = 0;
  uint32_t qid = 1;
  pid_t pid = 100;
  ge::GrantType grant_type = GrantType::kWriteOnly;
  auto ret = FlowGwClient::GrantQueue(device_id, qid, pid, grant_type);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtFlowGwClient, run_GrantQueue_ReadAndWrite) {
  uint32_t device_id = 0;
  uint32_t qid = 1;
  pid_t pid = 100;
  ge::GrantType grant_type = GrantType::kReadAndWrite;
  auto ret = FlowGwClient::GrantQueue(device_id, qid, pid, grant_type);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtFlowGwClient, run_GrantQueue_GRANT_INVALID) {
  uint32_t device_id = 0;
  uint32_t qid = 1;
  pid_t pid = 100;
  ge::GrantType grant_type = GrantType::kInvalid;
  auto ret = FlowGwClient::GrantQueue(device_id, qid, pid, grant_type);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(UtFlowGwClient, run_BindMainPortError) {
  SetConfigEnv("invalid/numa_config_error_port.json");
  auto ret = ge::NetworkManager::GetInstance().BindMainPort();
  EXPECT_NE(ret, SUCCESS);
}

TEST_F(UtFlowGwClient, run_BindMainPortError2) {
  SetConfigEnv("invalid/numa_config_zero_port.json");
  auto ret = ge::NetworkManager::GetInstance().BindMainPort();
  EXPECT_NE(ret, SUCCESS);
}

TEST_F(UtFlowGwClient, run_init_and_finalize) {
  EXPECT_EQ(ge::NetworkManager::GetInstance().Initialize(), SUCCESS);
  EXPECT_EQ(ge::NetworkManager::GetInstance().Finalize(), SUCCESS);
}


TEST_F(UtFlowGwClient, run_get_data_gw_status) {
  FlowGwClient flowgw_client(0, 0, {0}, false);
  EXPECT_EQ(flowgw_client.Initialize(), SUCCESS);
  EXPECT_EQ(flowgw_client.GetSubProcStat(), ProcStatus::NORMAL);
  flowgw_client.Finalize();
}

TEST_F(UtFlowGwClient, run_bind_main_port_failed) {
  NetworkManager manager;
  NetworkInfo networkInfo;
  networkInfo.available_ports = "1test~200";
  HostInfo hostInfo;
  hostInfo.data_panel = networkInfo;
  DeployerConfig deployerConfig;
  deployerConfig.host_info = hostInfo;
  Configurations::GetInstance().information_ = deployerConfig;
  ASSERT_EQ(manager.BindMainPort(), FAILED);
  Configurations::GetInstance().information_ = {};
}

TEST_F(UtFlowGwClient, run_config_sched_info_to_flowgw) {
  FlowGwClient flowgw_client(0, 0, {0}, false);
  auto ret = flowgw_client.ConfigSchedInfoToDataGw(0, 1, 2, 3, 0, true);
  ASSERT_EQ(ret, SUCCESS);
}
}
