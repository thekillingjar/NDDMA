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

#include "macro_utils/dt_public_scope.h"
#include "daemon/daemon_service.h"
#include "common/config/configurations.h"
#include "common/data_flow/queue/heterogeneous_exchange_service.h"
#include "macro_utils/dt_public_unscope.h"

#include "depends/mmpa/src/mmpa_stub.h"
#include "depends/runtime/src/runtime_stub.h"
#include "deploy/deployer/deployer_service_impl.h"
#include "depends/helper_runtime/src/caas_dataflow_auth_stub.h"
#include "deploy/deployer/deployer_authentication.h"
#include "common/env_path.h"

using namespace std;

namespace ge {
namespace {
class MockMmpa : public MmpaStubApiGe {
 public:
  int32_t RealPath(const char *path, char *realPath, int32_t realPathLen) override {
    (void)strncpy_s(realPath, realPathLen, path, strlen(path));
    return 0;
  }

  INT32 Open2(const CHAR *path_name, INT32 flags, MODE mode) override {
    return 0;
  }

  void *DlSym(void *handle, const char *func_name) override {
    if (std::string(func_name) == "NewSignResult") {
      return (void *) &NewSignResult;
    } else if (std::string(func_name) == "DeleteSignResult") {
      return (void *) &DeleteSignResult;
    } else if (std::string(func_name) == "GetSignLength") {
      return (void *) &GetSignLength;
    } else if (std::string(func_name) == "GetSignData") {
      return (void *) &GetSignData;
    } else if (std::string(func_name) == "DataFlowAuthMasterInit") {
      return (void *) &DataFlowAuthMasterInit;
    } else if (std::string(func_name) == "DataFlowAuthSign") {
      return (void *) &DataFlowAuthSign;
    } else if (std::string(func_name) == "DataFlowAuthVerify") {
      return (void *) &DataFlowAuthVerify;
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
}

class DaemonServiceUnittest : public testing::Test {
 protected:
  void SetUp() override {
    RuntimeStub::SetInstance(std::make_shared<MockRuntime>());
    MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  }

  void TearDown() override {
    HeterogeneousExchangeService::GetInstance().Finalize();
    RuntimeStub::Reset();
    MmpaStub::GetInstance().Reset();
    unsetenv("RESOURCE_CONFIG_PATH");
  }

  static void SetNumaConfigEnv(const std::string &path) {
    std::string data_path = PathUtils::Join({EnvPath().GetAirBasePath(), "tests/dflow/runner/ut/ge/runtime/data"});
    std::string config_path = PathUtils::Join({data_path, path});
    setenv("RESOURCE_CONFIG_PATH", config_path.c_str(), 1);
    EXPECT_EQ(Configurations::GetInstance().InitInformation(), SUCCESS);
  }
};

TEST_F(DaemonServiceUnittest, TestInitializeAndFinalize) {
  ge::DaemonService daemon_service;
  SetNumaConfigEnv("valid/server/numa_config2.json");
  EXPECT_EQ(daemon_service.Initialize(), SUCCESS);
  daemon_service.Finalize();
}

TEST_F(DaemonServiceUnittest, TestProcessInitRequest) {
  SubprocessManager::GetInstance().executable_paths_["deployer_daemon"] = "deployer_daemon";
  ge::DaemonService daemon_service;
  EXPECT_EQ(daemon_service.Initialize(), SUCCESS);
  SetNumaConfigEnv("valid/server/numa_config2.json");
  deployer::DeployerRequest request;
  deployer::DeployerResponse response;
  deployer::InitRequest init_request_;
  *request.mutable_init_request() = init_request_;
  daemon_service.ProcessInitRequest("xxx:10.216.56.15:8080", request, response);
  ASSERT_EQ(response.error_code(), SUCCESS);
}

TEST_F(DaemonServiceUnittest, TestProcessInitServerRequest) {
  auto information = Configurations::GetInstance().information_;
  GE_MAKE_GUARD(recover, [information]() {
    Configurations::GetInstance().information_ = information;
  });
  SetNumaConfigEnv("valid/server/numa_config2_with_auth.json");
  SubprocessManager::GetInstance().executable_paths_["deployer_daemon"] = "deployer_daemon";
  ge::DaemonService daemon_service;
  EXPECT_EQ(daemon_service.Initialize(), SUCCESS);
  deployer::DeployerRequest request;
  deployer::DeployerResponse response;
  deployer::InitRequest init_request;
  init_request.set_sign_data("stub_sign");
  *request.mutable_init_request() = init_request;
  daemon_service.ProcessInitRequest("xxx:10.216.56.15:8080", request, response);
  ASSERT_EQ(response.error_code(), SUCCESS);

  daemon_service.ProcessInitRequest("xxx", request, response);
  ASSERT_NE(response.error_code(), SUCCESS);
  daemon_service.Finalize();
}

TEST_F(DaemonServiceUnittest, TestHeartbeatRequest) {
  SubprocessManager::GetInstance().executable_paths_["deployer_daemon"] = "deployer_daemon";
  DaemonService daemon_service;
  SetNumaConfigEnv("valid/server/numa_config2.json");
  EXPECT_EQ(daemon_service.Initialize(), SUCCESS);
  deployer::DeployerRequest request;
  deployer::DeployerResponse response;
  deployer::InitRequest init_request_;
  request.set_client_id(0);
  *request.mutable_init_request() = init_request_;
  daemon_service.ProcessInitRequest("xxx:10.216.56.15:8080", request, response);
  request.set_type(deployer::kHeartbeat);
  daemon_service.Process("xxx:10.216.56.15:8080", request, response);
  ASSERT_EQ(response.error_code(), SUCCESS);
}

TEST_F(DaemonServiceUnittest, TestDisconnectRequest) {
  DaemonService daemon_service;
  SetNumaConfigEnv("valid/server/numa_config2.json");
  EXPECT_EQ(daemon_service.Initialize(), SUCCESS);
  deployer::DeployerRequest request;
  request.set_type(deployer::kDisconnect);
  deployer::DeployerResponse response;
  daemon_service.Process("xxx:10.216.56.15:8080", request, response);
  daemon_service.Finalize();
}

TEST_F(DaemonServiceUnittest, TestInitAndDisconnectRequest) {
  DaemonService daemon_service;
  EXPECT_EQ(daemon_service.Initialize(), SUCCESS);
  SetNumaConfigEnv("valid/server/numa_config2.json");
  deployer::DeployerRequest request;
  deployer::DeployerResponse response;
  deployer::InitRequest init_request_;
  *request.mutable_init_request() = init_request_;
  daemon_service.ProcessInitRequest("xxx:10.216.56.15:8080", request, response);
  request.set_client_id(0);
  daemon_service.ProcessDisconnectRequest("xxx:10.216.56.15:8080", request, response);
}

TEST_F(DaemonServiceUnittest, TestDeployRequest) {
  SubprocessManager::GetInstance().executable_paths_["deployer_daemon"] = "deployer_daemon";
  DaemonService daemon_service;
  SetNumaConfigEnv("valid/server/numa_config2.json");
  EXPECT_EQ(daemon_service.Initialize(), SUCCESS);
  auto client = std::make_unique<DeployerDaemonClient>(1);
  std::map<std::string, std::string> deployer_envs = {};
  ASSERT_EQ(client->Initialize(deployer_envs), SUCCESS);
  daemon_service.client_manager_->clients_[1] = std::move(client);
  deployer::DeployerRequest request;
  request.set_client_id(1);
  request.set_type(deployer::kLoadModel);
  deployer::DeployerResponse response;
  daemon_service.Process("xxx:10.216.56.15:8080", request, response);
  ASSERT_EQ(response.error_code(), SUCCESS);
  daemon_service.Finalize();
}

TEST_F(DaemonServiceUnittest, TestProcessInitRequestVerifyToolSuccess) {
  SubprocessManager::GetInstance().executable_paths_["deployer_daemon"] = "deployer_daemon";
  ge::DaemonService daemon_service;
  SetNumaConfigEnv("valid/server/numa_config2.json");
  EXPECT_EQ(daemon_service.Initialize(), SUCCESS);
  deployer::DeployerRequest request;
  deployer::DeployerResponse response;
  deployer::InitRequest init_request_;
  *request.mutable_init_request() = init_request_;
  daemon_service.ProcessInitRequest("xxx:10.216.56.15:8080", request, response);
  ASSERT_EQ(response.error_code(), SUCCESS);
}
} // namespace ge
