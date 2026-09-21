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
#include <string>
#include "deploy/rpc/deployer_client.h"
#include "framework/common/debug/ge_log.h"
#include "deploy/rpc/deployer_server.h"
#include "depends/mmpa/src/mmpa_stub.h"

#include "macro_utils/dt_public_scope.h"
#include "daemon/daemon_service.h"
#include "depends/runtime/src/runtime_stub.h"
#include "common/data_flow/queue/heterogeneous_exchange_service.h"
#include "deploy/deployer/deployer_service_impl.h"
#include "macro_utils/dt_public_unscope.h"

using namespace std;
namespace ge {
namespace {
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

class GrpcServer {
  class MockDeployerMessageClient : public DeployerMessageClient {
   public:
    MockDeployerMessageClient(int32_t device_id) : DeployerMessageClient(device_id, true) {}

    Status WaitForProcessInitialized() override {
      return SUCCESS;
    }
  };

  class MockDeployerDaemonClient : public DeployerDaemonClient {
   public:
    explicit MockDeployerDaemonClient(int64_t client_id) : DeployerDaemonClient(client_id) {
      std::string ctx_name = std::string("client_") + std::to_string(client_id);
      context_.SetName(ctx_name);
      context_.SetDeployerPid(client_id);
      context_.Initialize();
    }

    std::shared_ptr<DeployerMessageClient> CreateMessageClient() override {
      return MakeShared<MockDeployerMessageClient>(0);
    }

    Status ProcessDeployRequest(const deployer::DeployerRequest &request, deployer::DeployerResponse &response) override {
      GE_CHK_STATUS_RET_NOLOG(DeployerServiceImpl::GetInstance().Process(context_, request, response));
      return SUCCESS;
    }

    Status ProcessHeartbeatRequest(const deployer::DeployerRequest &request,
                                   deployer::DeployerResponse &response) override {
      GE_CHK_STATUS_RET_NOLOG(DeployerServiceImpl::GetInstance().Process(context_, request, response));
      return SUCCESS;
    }

   private:
    DeployContext context_;
  };

  class MockDaemonClientManager : public DaemonClientManager {
   public:
    std::unique_ptr<DeployerDaemonClient> CreateClient(int64_t client_id) override {
      return MakeUnique<MockDeployerDaemonClient>(client_id);
    }
  };

  class MockDaemonService : public DaemonService {
   public:
    Status InitClientManager() override {
      client_manager_ = MakeUnique<MockDaemonClientManager>();
      GE_CHECK_NOTNULL(client_manager_);
      GE_CHK_STATUS_RET(client_manager_->Initialize(), "Failed to initialize ClientManager");
      return SUCCESS;
    }
  };

  class MockDeployerDaemonService : public DeployerDaemonService {
   public:
    Status Initialize() override {
      daemon_service_ = MakeUnique<MockDaemonService>();
      GE_CHECK_NOTNULL(daemon_service_);
      return daemon_service_->Initialize();
    }
  };

 public:
  Status Run() {
    grpc_server_.SetServiceProvider(std::unique_ptr<MockDeployerDaemonService>(new MockDeployerDaemonService()));
    return grpc_server_.Run();
  }

  void Finalize() {
    grpc_server_.Finalize();
  }

 private:
  ge::DeployerServer grpc_server_;
};
}
class UtDeployerClient : public testing::Test {
 public:
  UtDeployerClient() {}
 protected:
  void SetUp() override {
    RuntimeStub::SetInstance(std::make_shared<MockRuntime>());
  }
  void TearDown() override {
    HeterogeneousExchangeService::GetInstance().Finalize();
    RuntimeStub::Reset();
  }
};

TEST_F(UtDeployerClient, run_init) {
  std::string ip("127.0.0.1:1000");
  ge::DeployerClient client;
  auto ret = client.Initialize(ip);
  ASSERT_EQ(ret, ge::SUCCESS);
}
}

