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
#include "rt_error_codes.h"
#include "framework/common/debug/ge_log.h"
#include "depends/mmpa/src/mmpa_stub.h"
#include "depends/runtime/src/runtime_stub.h"
#include "depends/helper_runtime/src/caas_dataflow_auth_stub.h"

#include "macro_utils/dt_public_scope.h"
#include "deploy/deployer/deployer_proxy.h"
#include "deploy/resource/resource_manager.h"
#include "daemon/daemon_service.h"
#include "deploy/deployer/deployer_authentication.h"
#include "macro_utils/dt_public_unscope.h"
#include "deploy/abnormal_status_handler/device_abnormal_status_handler.h"

using namespace std;
namespace ge {
class DeployerProxyTest : public testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

namespace {
class MockDeployer : public Deployer {
 public:
  MockDeployer() = default;
  const NodeInfo &GetNodeInfo() const override {
    return node_info_;
  }
  Status Process(deployer::DeployerRequest &request, deployer::DeployerResponse &response) override {
    return SUCCESS;
  }
  Status Initialize() override {
    return SUCCESS;
  }
  Status Finalize() override {
    return SUCCESS;
  }
  Status GetDevStat() override {
    return SUCCESS;
  }
 private:
  NodeInfo node_info_;
};

class MockDeployerClient : public DeployerClient {
 public:
  MockDeployerClient() = default;
  Status SendRequest(const deployer::DeployerRequest &request,
                     deployer::DeployerResponse &response, const int64_t timeout_sec = -1) override {
    if (request.type() == deployer::kInitRequest) {
      response.mutable_init_response()->set_client_id(0);
      response.mutable_init_response()->set_dev_count(2);
      response.mutable_init_response()->set_dgw_port_offset(0);
    }
    return SUCCESS;
  }

  Status SendRequestWithRetry(const deployer::DeployerRequest &request,
                              deployer::DeployerResponse &response,
                              const int64_t timeout_in_sec,
                              const int32_t retry_times) override {
    if (request.type() == deployer::kInitRequest) {
      response.mutable_init_response()->set_client_id(0);
      response.mutable_init_response()->set_dev_count(2);
      response.mutable_init_response()->set_dgw_port_offset(0);
    }
    return SUCCESS;
  }
};

class MockRemoteDeployer : public RemoteDeployer {
 public:
  explicit MockRemoteDeployer(NodeConfig node_config) : RemoteDeployer(node_config) {};
  std::unique_ptr<DeployerClient> CreateClient() override {
    return MakeUnique<MockDeployerClient>();
  }
};

class MockMmpa : public MmpaStubApiGe {
 public:
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

class CallbackManager {
 public:
  static CallbackManager &GetInstance() {
    static CallbackManager callbackManager;
    return callbackManager;
  }

  rtError_t Register(const char *moduleName, rtTaskFailCallback callback) {
    if (return_failed) {
      return ACL_ERROR_RT_PARAM_INVALID;
    }
    if (callback == nullptr) {
      callbacks_.erase(moduleName);
    } else {
      callbacks_.emplace(moduleName, callback);
    }
    return RT_ERROR_NONE;
  }

  void Call(const char *moduleName, rtExceptionInfo *excpt_info) {
    const std::string name = moduleName;
    auto iter = callbacks_.find(name);
    if (iter != callbacks_.cend()) {
      rtTaskFailCallback callback = iter->second;
      callback(excpt_info);
    }
  }

  void SetReturnFailed() {
    return_failed = true;
  }

  void Reset() {
    callbacks_.clear();
    return_failed = false;
  }

  std::map<std::string, rtTaskFailCallback> callbacks_;
  bool return_failed = false;
};

class MockRuntimeCallbck : public RuntimeStub {
 public:
  rtError_t rtRegTaskFailCallbackByModule(const char *moduleName, rtTaskFailCallback callback) override {
    CallbackManager::GetInstance().Register(moduleName, callback);
    return RT_ERROR_NONE;
  }
};
}

TEST_F(DeployerProxyTest, TestInitializeAndAutoFinalize) {
  NodeConfig local_node_config;
  DeviceConfig device_config;
  device_config.device_id = -1;
  local_node_config.device_list.emplace_back(device_config);
  local_node_config.is_local = true;
  std::vector<NodeConfig> node_config_list{local_node_config};
  DeployerProxy deployer_proxy;
  ASSERT_EQ(deployer_proxy.Initialize(node_config_list), SUCCESS);
  ASSERT_EQ(deployer_proxy.NumNodes(), node_config_list.size());
}

TEST_F(DeployerProxyTest, TestCreateDeployer) {
  NodeConfig local_node_config;
  DeviceConfig device_config;
  device_config.device_id = -1;
  local_node_config.device_list.emplace_back(device_config);
  ASSERT_TRUE(DeployerProxy::CreateDeployer(local_node_config) != nullptr);
  NodeConfig remote_node_config;
  device_config.device_id = 0;
  remote_node_config.device_list.emplace_back(device_config);
  ASSERT_TRUE(DeployerProxy::CreateDeployer(remote_node_config) != nullptr);
}

TEST_F(DeployerProxyTest, TestInitializeAndFinalize_empty) {
  std::vector<NodeConfig> node_config_list{};
  DeployerProxy deployer_proxy;
  auto ret = deployer_proxy.Initialize(node_config_list);
  ASSERT_EQ(ret, ge::SUCCESS);
  deployer_proxy.Finalize();
}

TEST_F(DeployerProxyTest, TestSendRequest) {
  NodeConfig local_node_config;
  DeviceConfig device_config;
  device_config.device_id = -1;
  local_node_config.device_list.emplace_back(device_config);
  NodeConfig remote_node_config;
  device_config.device_id = 0;
  remote_node_config.device_list.emplace_back(device_config);
  std::vector<NodeConfig> node_config_list{local_node_config, remote_node_config};
  DeployerProxy deployer_proxy;
  deployer_proxy.deployers_.emplace_back(MakeUnique<MockDeployer>());
  deployer_proxy.deployers_.emplace_back(MakeUnique<MockDeployer>());
  ASSERT_TRUE(deployer_proxy.GetNodeInfo(0) != nullptr);
  ASSERT_TRUE(deployer_proxy.GetNodeInfo(1) != nullptr);
  ASSERT_TRUE(deployer_proxy.GetNodeInfo(2) == nullptr);
  ASSERT_TRUE(deployer_proxy.GetNodeInfo(-1) == nullptr);
  deployer::DeployerRequest request;
  deployer::DeployerResponse response;
  ASSERT_EQ(deployer_proxy.SendRequest(0, request, response), SUCCESS);
  ASSERT_EQ(deployer_proxy.SendRequest(1, request, response), SUCCESS);
  ASSERT_EQ(deployer_proxy.SendRequest(2, request, response), PARAM_INVALID);
  ASSERT_EQ(deployer_proxy.SendRequest(-1, request, response), PARAM_INVALID);
}

TEST_F(DeployerProxyTest, TestLocalDeployer) {
  DeployerConfig deployer_config = {};
  NodeConfig node_config = {};
  node_config.ipaddr = "192.168.10.120";
  DeviceConfig dev_config = {};
  node_config.device_list.push_back(dev_config);
  node_config.available_ports = "61000~65535";
  deployer_config.node_config = node_config;
  Configurations::GetInstance().information_ = deployer_config;
  LocalDeployer deployer;
  ASSERT_EQ(deployer.Initialize(), SUCCESS);
  deployer::DeployerRequest request;
  request.set_type(deployer::kHeartbeat);
  deployer::DeployerResponse response;
  ASSERT_EQ(deployer.Process(request, response), SUCCESS);
  deployer.Finalize();
}

TEST_F(DeployerProxyTest, TestLocalDeployerGetDevStatWithDeviceOom) {
  auto mock_runtime = std::make_shared<MockRuntimeCallbck>();
  RuntimeStub::SetInstance(mock_runtime);
  DeviceAbnormalStatusHandler::Instance().Initialize();
  rtExceptionInfo excpt_info{};
  excpt_info.deviceid = 1;
  excpt_info.retcode = ACL_ERROR_RT_DEVICE_OOM;
  CallbackManager::GetInstance().Call("deployer_dev_abnormal", &excpt_info);

  DeployerConfig deployer_config = {};
  NodeConfig node_config = {};
  node_config.ipaddr = "192.168.10.120";
  DeviceConfig dev_config = {};
  node_config.device_list.push_back(dev_config);
  node_config.available_ports = "61000~65535";
  deployer_config.node_config = node_config;
  Configurations::GetInstance().information_ = deployer_config;
  std::unique_ptr<Deployer> deployer = nullptr;
  deployer.reset((Deployer*)(new LocalDeployer()));
  ASSERT_EQ(deployer->Initialize(), SUCCESS);
  EXPECT_EQ(deployer->GetDevStat(), SUCCESS);
  DeployerProxy::GetInstance().deployers_.emplace_back(std::move(deployer));
  EXPECT_EQ(DeployerProxy::GetInstance().GetDeviceAbnormalCode(), ACL_ERROR_RT_DEVICE_OOM);
  RuntimeStub::Reset();
}

TEST_F(DeployerProxyTest, TestLocalDeployerWithParseResponse) {
  DeployerConfig deployer_config = {};
  NodeConfig node_config = {};
  node_config.ipaddr = "192.168.10.120";
  DeviceConfig dev_config = {};
  node_config.device_list.push_back(dev_config);
  node_config.available_ports = "61000~65535";
  deployer_config.node_config = node_config;
  Configurations::GetInstance().information_ = deployer_config;
  LocalDeployer deployer;
  ASSERT_EQ(deployer.Initialize(), SUCCESS);
  deployer::DeployerResponse response;

  dlog_setlevel(0, 0, 0);
  auto abnormal_submodel_instance_name =
      response.mutable_heartbeat_response()->mutable_abnormal_submodel_instance_name();
  deployer::SubmodelInstanceName submodel_instance_name_rsp;
  (*(submodel_instance_name_rsp.mutable_submodel_instance_name()))["model1"] = false;
  ((*abnormal_submodel_instance_name)[0]) = submodel_instance_name_rsp;
  response.mutable_heartbeat_response()->set_abnormal_type(kAbnormalTypeModelInstance);
  deployer.ParseRsponse(response);

  auto dev_status1 = response.mutable_heartbeat_response()->add_device_status();
  response.mutable_heartbeat_response()->set_abnormal_type(kAbnormalTypeDevice);
  dev_status1->set_device_id(0);
  dev_status1->set_device_type(NPU);
  deployer.ParseRsponse(response);

  (void)response.mutable_heartbeat_response()->add_device_status();
  response.mutable_heartbeat_response()->set_abnormal_type(kAbnormalTypeNode);
  deployer.ParseRsponse(response);
  dlog_setlevel(0, 3, 0);
  deployer.Finalize();
}

TEST_F(DeployerProxyTest, TestRemoteDeployer) {
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  setenv("ASCEND_GLOBAL_LOG_LEVEL", "3", 1);
  GE_MAKE_GUARD(recover, []() {
    MmpaStub::GetInstance().Reset();
    unsetenv("ASCEND_GLOBAL_LOG_LEVEL");
  });
  NodeConfig remote_node_config;
  DeviceConfig device_config;
  device_config.device_id = 0;
  remote_node_config.device_list.emplace_back(device_config);
  remote_node_config.ipaddr = "127.0.0.1";
  remote_node_config.port = 2509;
  remote_node_config.available_ports = "61000~65535";
  remote_node_config.lazy_connect = false;
  remote_node_config.auth_lib_path = "/a/b/c/libdataflow_auth.so";
  ASSERT_EQ(DeployerAuthentication::GetInstance().Initialize("/a/b/c/libdataflow_auth.so", true), SUCCESS);
  MockRemoteDeployer deployer(remote_node_config);
  ASSERT_EQ(deployer.Initialize(), SUCCESS);
  std::thread keepalive_thread = std::thread([&deployer]() {
    deployer.Keepalive();
  });
  deployer.SendHeartbeat(1);
  deployer.Finalize();
  if (keepalive_thread.joinable()) {
    keepalive_thread.join();
  }
  DeployerAuthentication::GetInstance().Finalize();
}

TEST_F(DeployerProxyTest, TestRemoteDeployerWithParseRsponse) {
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  GE_MAKE_GUARD(recover, []() { MmpaStub::GetInstance().Reset(); });
  NodeConfig remote_node_config;
  DeviceConfig device_config;
  device_config.device_id = 0;
  remote_node_config.device_list.emplace_back(device_config);
  remote_node_config.ipaddr = "127.0.0.1";
  remote_node_config.port = 2509;
  remote_node_config.available_ports = "61000~65535";
  remote_node_config.lazy_connect = false;
  remote_node_config.auth_lib_path = "/a/b/c/libdataflow_auth.so";
  ASSERT_EQ(DeployerAuthentication::GetInstance().Initialize("/a/b/c/libdataflow_auth.so", true), SUCCESS);
  MockRemoteDeployer deployer(remote_node_config);
  ASSERT_EQ(deployer.Initialize(), SUCCESS);
  std::thread keepalive_thread = std::thread([&deployer]() {
    deployer.Keepalive();
  });

  deployer::DeployerResponse response;

  dlog_setlevel(0, 0, 0);
  auto abnormal_submodel_instance_name =
      response.mutable_heartbeat_response()->mutable_abnormal_submodel_instance_name();
  deployer::SubmodelInstanceName submodel_instance_name_rsp;
  (*(submodel_instance_name_rsp.mutable_submodel_instance_name()))["model1"] = false;
  ((*abnormal_submodel_instance_name)[0]) = submodel_instance_name_rsp;
  response.mutable_heartbeat_response()->set_abnormal_type(kAbnormalTypeModelInstance);
  deployer.ParseRsponse(response);

  auto dev_status1 = response.mutable_heartbeat_response()->add_device_status();
  response.mutable_heartbeat_response()->set_abnormal_type(kAbnormalTypeDevice);
  dev_status1->set_device_id(0);
  dev_status1->set_device_type(NPU);
  deployer.ParseRsponse(response);

  (void)response.mutable_heartbeat_response()->add_device_status();
  response.mutable_heartbeat_response()->set_abnormal_type(kAbnormalTypeNode);
  deployer.ParseRsponse(response);
  dlog_setlevel(0, 3, 0);
  deployer.Finalize();
  if (keepalive_thread.joinable()) {
    keepalive_thread.join();
  }
  DeployerAuthentication::GetInstance().Finalize();
}

TEST_F(DeployerProxyTest, TestGetNodeStat) {
  std::unique_ptr<Deployer> deployer = nullptr;
  deployer.reset((Deployer*)(new LocalDeployer()));
  DeployerProxy::GetInstance().deployers_.emplace_back(std::move(deployer));
  EXPECT_EQ(DeployerProxy::GetInstance().GetNodeStat(), ge::SUCCESS);
  DeployerProxy::GetInstance().Finalize();
}
}  // namespace ge
