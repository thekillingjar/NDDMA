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
#include <common/plugin/ge_make_unique_util.h>
#include "framework/common/debug/ge_log.h"
#include "hccl/hccl_types.h"
#include "common/data_flow/queue/heterogeneous_exchange_service.h"
#include "depends/mmpa/src/mmpa_stub.h"
#include "depends/runtime/src/runtime_stub.h"
#include "macro_utils/dt_public_scope.h"
#include "daemon/daemon_client_manager.h"
#include "macro_utils/dt_public_unscope.h"

using namespace std;
namespace ge {
namespace {
class MockMmpa : public MmpaStubApiGe {
 public:
  INT32 Open2(const CHAR *path_name, INT32 flags, MODE mode) override {
    return 0;
  }
};

class MockRuntime : public RuntimeStub {
 public:
  rtError_t rtMemQueueDeQueue(int32_t device, uint32_t qid, void **mbuf) override {
    return 0;
  }

  rtError_t rtMbufGetBuffAddr(rtMbufPtr_t mbuf, void **databuf) override {
    *databuf = data_;
    return 0;
  }

  rtError_t rtMbufGetBuffSize(rtMbufPtr_t mbuf, uint64_t *size) override {
    *size = 0;
    return 0;
  }

  rtError_t rtMbufFree(rtMbufPtr_t mbuf) override {
    // 由MockRuntimeNoLeaks统一释放
    return RT_ERROR_NONE;
  }
  rtError_t rtMbufAlloc(rtMbufPtr_t *mbuf, uint64_t size) override {
    // 此处打桩记录所有申请的Mbuf,此UT不会Dequeue和Free而造成泄漏,因此在MockRuntime析构时统一释放
    RuntimeStub::rtMbufAlloc(mbuf, size);
    std::lock_guard<std::mutex> lk(mu_);
    mem_bufs_.emplace_back(*mbuf);
    return 0;
  }

  ~MockRuntime() {
    for (auto &mbuf : mem_bufs_) {
      RuntimeStub::rtMbufFree(mbuf);
    }
    mem_bufs_.clear();
  }

 private:
  std::mutex mu_;
  uint8_t data_[128] = {};
  std::vector<void *> mem_bufs_;
};
}

// client manager
class DaemonClientManagerTest : public testing::Test {
 protected:
  void SetUp() override {
    RuntimeStub::SetInstance(std::make_shared<MockRuntime>());
    MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  }
  void TearDown() override {
    HeterogeneousExchangeService::GetInstance().Finalize();
    RuntimeStub::Reset();
    MmpaStub::GetInstance().Reset();
  }
};

TEST_F(DaemonClientManagerTest, run_initialize_and_finalize) {
  DaemonClientManager client_manager;
  ASSERT_EQ(client_manager.Initialize(), SUCCESS);
  client_manager.Finalize();
}

TEST_F(DaemonClientManagerTest, run_create_client_success) {
  SubprocessManager::GetInstance().executable_paths_["deployer_daemon"] = "deployer_daemon";
  DaemonClientManager client_manager;
  int64_t client_id = 0;
  std::map<std::string, std::string> deployer_envs;
  deployer_envs.emplace("FAKE_ENV", "1");
  auto ret = client_manager.CreateAndInitClient("ipv4:192.168.1.1:12345", deployer_envs, client_id);
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(DaemonClientManagerTest, run_close_and_get_client_failed) {
  DaemonClientManager client_manager;
  int64_t client_id = 0;
  ASSERT_NE(client_manager.CloseClient(client_id), SUCCESS);
  ASSERT_EQ(client_manager.GetClient(client_id), nullptr);
}

TEST_F(DaemonClientManagerTest, TestEvict) {
  DaemonClientManager client_manager;
  int64_t client_id = 0;
  client_manager.clients_[0] = MakeUnique<DeployerDaemonClient>(client_id);
  client_manager.clients_[0]->last_heartbeat_ts_ = {};
  client_manager.clients_[0]->is_executing_ = true;
  client_manager.EvictExpiredClients();
  ASSERT_EQ(client_manager.clients_.size(), 1);
  client_manager.clients_[0]->is_executing_ = false;
  client_manager.EvictExpiredClients();
  ASSERT_EQ(client_manager.clients_.size(), 0);
}

// client
class DeployerDaemonClientTest : public testing::Test {
 protected:
  void SetUp() override {
    RuntimeStub::SetInstance(std::make_shared<MockRuntime>());
    MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  }
  void TearDown() override {
    HeterogeneousExchangeService::GetInstance().Finalize();
    RuntimeStub::Reset();
    MmpaStub::GetInstance().Reset();
  }
};

TEST_F(DeployerDaemonClientTest, run_initialize_and_finalize) {
  RuntimeStub::SetInstance(std::make_shared<MockRuntime>());
  SubprocessManager::GetInstance().executable_paths_["deployer_daemon"] = "deployer_daemon";
  DeployerDaemonClient client(1);
  std::map<std::string, std::string> deployer_envs = {};
  ASSERT_EQ(client.Initialize(deployer_envs), SUCCESS);
  ASSERT_EQ(client.Finalize(), SUCCESS);
  HeterogeneousExchangeService::GetInstance().Finalize();
  RuntimeStub::Reset();
}

TEST_F(DeployerDaemonClientTest, TestIsExpired) {
  DeployerDaemonClient client(1);
  client.last_heartbeat_ts_ = {};
  ASSERT_TRUE(client.IsExpired());
  client.OnHeartbeat();
  ASSERT_FALSE(client.IsExpired());
}

TEST_F(DeployerDaemonClientTest, TestIsExecuting) {
  DeployerDaemonClient client(1);
  ASSERT_FALSE(client.IsExecuting());
  client.SetIsExecuting(true);
  ASSERT_TRUE(client.IsExecuting());
}

TEST_F(DeployerDaemonClientTest, TestProcessHeartBeatRequest_Abnormal) {
  DeployerDaemonClient client(1);
  client.deployer_msg_client_ = std::make_shared<DeployerMessageClient>(0, true);
  ASSERT_NE(client.deployer_msg_client_, nullptr);
  deployer::DeployerRequest request;
  deployer::DeployerResponse response;
  client.sub_deployer_proc_stat_ = ProcStatus::EXITED;
  ASSERT_NE(client.ProcessHeartbeatRequest(request, response), SUCCESS);
  client.sub_deployer_proc_stat_ = ProcStatus::STOPPED;
  ASSERT_NE(client.ProcessHeartbeatRequest(request, response), SUCCESS);
}

TEST_F(DeployerDaemonClientTest, TestDeployerProcStatusChange) {
  RuntimeStub::SetInstance(std::make_shared<MockRuntime>());
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  SubprocessManager::GetInstance().executable_paths_["deployer_daemon"] = "deployer_daemon";

  SubprocessManager::GetInstance().excpt_handle_callbacks_.clear();
  DeployerDaemonClient client(1);
  std::map<std::string, std::string> deployer_envs = {};
  ASSERT_EQ(client.Initialize(deployer_envs), SUCCESS);
  for (auto callback : SubprocessManager::GetInstance().excpt_handle_callbacks_) {
    callback.second(ProcStatus::NORMAL);
  }
  ASSERT_EQ(client.Finalize(), SUCCESS);

  HeterogeneousExchangeService::GetInstance().Finalize();
  RuntimeStub::Reset();
  MmpaStub::GetInstance().Reset();
}
}
