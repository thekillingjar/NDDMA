/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <memory>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <cstdlib>

#include "macro_utils/dt_public_scope.h"
#include "deploy/deployer/master_model_deployer.h"
#include "deploy/deployer/deployer_service_impl.h"
#include "deploy/flowrm/network_manager.h"
#include "deploy/resource/resource_manager.h"
#include "deploy/model_send/flow_model_sender.h"
#include "dflow/base/exec_runtime/execution_runtime.h"
#include "macro_utils/dt_public_unscope.h"

#include "proto/deployer.pb.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_local_context.h"
#include "graph/utils/graph_utils.h"
#include "graph/manager/graph_var_manager.h"
#include "common/config/configurations.h"
#include "deploy/heterogeneous_execution_runtime.h"
#include "deploy/flowrm/tsd_client.h"
#include "framework/common/types.h"
#include "deploy/flowrm/flow_route_manager.h"
#include "hccl/hccl_types.h"
#include "depends/mmpa/src/mmpa_stub.h"
#include "stub_models.h"
#include "stub/heterogeneous_stub_env.h"
#include "common/data_flow/queue/heterogeneous_exchange_service.h"
#include "depends/runtime/src/runtime_stub.h"
#include "common/env_path.h"

using namespace std;
using namespace ::testing;

namespace ge {
namespace {
void *mock_handle = nullptr;
class MockMmpa : public MmpaStubApiGe {
 public:
  void *DlSym(void *handle, const char *func_name) override {
    if (std::string(func_name) == "TsdFileLoad") {
      return (void *) &TsdFileLoad;
    } else if (std::string(func_name) == "TsdFileUnLoad") {
      return (void *) &TsdFileUnLoad;
    } else if (std::string(func_name) == "TsdGetProcListStatus") {
      return (void *) &TsdGetProcListStatus;
    } else if (std::string(func_name) == "TsdProcessOpen") {
      return (void *) &TsdProcessOpen;
    } else if (std::string(func_name) == "ProcessCloseSubProcList") {
      return (void *) &ProcessCloseSubProcList;
    } else if (std::string(func_name) == "TsdCapabilityGet") {
      return (void *) &TsdCapabilityGet;
    } else if (std::string(func_name) == "TsdInitFlowGw") {
      return (void *) &TsdInitFlowGw;
    }
    std::cout << "func name:" << func_name << " not stub\n";
    return dlsym(handle, func_name);
  }

  int32_t RealPath(const CHAR *path, CHAR *realPath, INT32 realPathLen) override {
    (void)strncpy_s(realPath, realPathLen, path, strlen(path));
    return EN_OK;
  }

  void *DlOpen(const char *file_name, int32_t mode) override {
    if (std::string("libtsdclient.so") == file_name) {
      return mock_handle;
    }
    return dlopen(file_name, mode);
  }
  int32_t DlClose(void *handle) override {
    if (mock_handle == handle) {
      return 0L;
    }
    return dlclose(handle);
  }

  int32_t Sleep(UINT32 microSecond) override {
    return 0;
  }
};

class MockRuntime : public RuntimeStub {
 public:
  ~MockRuntime() {
    // 用于释放TransferWeightWithQ中创建的Mbuf，调用rtMemQueueEnQueue后不会再访问Mbuf地址，可以释放
    // 此处打桩只在三个特定用例中被使用
    for (auto &m_buf : mem_bufs) {
      rtMbufFree(m_buf);
    }
    mem_bufs.clear();
  }
  rtError_t rtMemQueueEnQueue(int32_t dev_id, uint32_t qid, void *mem_buf) override {
    mem_bufs.emplace_back(mem_buf);
    return RT_ERROR_NONE;
  }
 private:
  std::vector<void *> mem_bufs;
};

class MockRuntimeNoLeaks : public RuntimeStub {
 public:
  rtError_t rtMbufFree(rtMbufPtr_t mbuf) {
    // 由MockRuntimeNoLeaks统一释放
    return RT_ERROR_NONE;
  }
  rtError_t rtMbufAlloc(rtMbufPtr_t *mbuf, uint64_t size) {
    // 此处打桩记录所有申请的Mbuf,此UT不会Dequeue和Free而造成泄漏,因此在MockRuntime析构时统一释放
    RuntimeStub::rtMbufAlloc(mbuf, size);
    std::lock_guard<std::mutex> lk(mu_);
    mem_bufs_.emplace_back(*mbuf);
    return 0;
  }
  ~MockRuntimeNoLeaks() {
    for (auto &mbuf : mem_bufs_) {
      RuntimeStub::rtMbufFree(mbuf);
    }
    mem_bufs_.clear();
  }

 private:
  std::mutex mu_;
  std::vector<void *> mem_bufs_;
};

class MockMasterModelDeployer : public MasterModelDeployer {
 public:
  MOCK_CONST_METHOD2(DeployLocalExchangePlan, Status(const deployer::FlowRoutePlan &, ExchangeRoute &));
  MOCK_METHOD3(DeployRemoteVarManager, Status(const DeployPlan &,
      const std::map<int32_t, std::vector<ConstSubmodelInfoPtr>> &,
      MasterModelDeployer::DeployedModel &));
};
}  // namespace

class MasterModelDeployerTest : public testing::Test {
 protected:
  void SetUp() override {
    mock_handle = (void *) 0x8888;
    MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
    // 用例依赖下面的状态，本文件中未初始化SubprocessManager，依赖前面SubprocessManager的UT
    SubprocessManager::GetInstance().executable_paths_.emplace("queue_schedule", "/var/queue_schedule");
    HeterogeneousStubEnv::SetupAIServerEnv();
  }

  void TearDown() override {
    HeterogeneousStubEnv::ClearEnv();
    MmpaStub::GetInstance().Reset();
    unsetenv("RESOURCE_CONFIG_PATH");
  }
};

/**
 *     NetOutput
 *         |
 *       PC_3
 *      /   \
 *    PC_1  PC2
 *    |      |
 *  data1  data2
 */
TEST_F(MasterModelDeployerTest, TestDeployModel_Success) {
  mock_handle = (void *) 0x8888;
  EXPECT_EQ(VarManager::Instance(0)->Init(0, 0, 1, 0), SUCCESS);
  auto &exchange_service =
      reinterpret_cast<stub::MockExchangeService &>(ExecutionRuntime::GetInstance()->GetExchangeService());
  EXPECT_CALL(exchange_service, DoCreateQueue).WillRepeatedly(Return(SUCCESS));
  EXPECT_CALL(exchange_service, DestroyQueue).WillRepeatedly(Return(SUCCESS));

  map<std::string, std::string> sess_options = ge::GetThreadLocalContext().GetAllSessionOptions();
  GE_MAKE_GUARD(recover_sess_cfg, [&sess_options](){
    GetThreadLocalContext().SetSessionOption(sess_options);
  });

  map<std::string, std::string> options{{"TestSessionOption", "TestSessionOptionValue"}};
  GetThreadLocalContext().SetSessionOption(options);

  DeployContext deploy_context;
  deploy_context.Initialize();
  deploy_context.SetName("RemoteContext");
  auto stub_func = [&deploy_context](deployer::DeployerRequest &request, deployer::DeployerResponse &response) -> Status{
    return DeployerServiceImpl::GetInstance().Process(deploy_context, request, response);
  };
  auto &remote_device =
      reinterpret_cast<stub::MockRemoteDeployer &>(*DeployerProxy::GetInstance().deployers_[1]);
  EXPECT_CALL(remote_device, Process)
      .WillRepeatedly(Invoke(stub_func));

  auto flow_model = StubModels::BuildFlowModel(StubModels::BuildGraphWithoutNeedForBindingQueues());
  AttrUtils::SetInt(flow_model->GetRootGraph(), "_inputs_align_max_cache_num", 100);
  AttrUtils::SetInt(flow_model->GetRootGraph(), "_inputs_align_timeout", 30 * 1000);
  AttrUtils::SetBool(flow_model->GetRootGraph(), "_inputs_align_dropout", true);

  MasterModelDeployer::DeployedModel deployed_model;
  MockMasterModelDeployer model_deployer;
  DeployResult deploy_result;
  auto mock_runtime = std::make_shared<MockRuntimeNoLeaks>();
  RuntimeStub::SetInstance(mock_runtime);
  ASSERT_EQ(model_deployer.DeployModel(flow_model, deploy_result), SUCCESS);
  ASSERT_EQ(model_deployer.deployed_models_.size(), 1);
  ASSERT_EQ(model_deployer.UpdateProfilingInfo(true), SUCCESS);
  ASSERT_EQ(model_deployer.Undeploy(deploy_result.model_id), SUCCESS);
  ASSERT_EQ(model_deployer.deployed_models_.size(), 0);
  ASSERT_NE(deploy_result.dev_abnormal_callback, nullptr);
  ASSERT_EQ(deploy_result.dev_abnormal_callback(), SUCCESS);
  RuntimeStub::Reset();
  mock_handle = nullptr;
}

TEST_F(MasterModelDeployerTest, SetTrimmingModelInstanceNames) {
  MockMasterModelDeployer model_deployer;
  std::map<std::string, std::vector<std::string>> org_model_instance_names;
  org_model_instance_names["name1"] = {"name2", "name3"};
  org_model_instance_names["name2"] = {"name4"};
  org_model_instance_names["name3"] = {"name4"};
  org_model_instance_names["name5"] = {"name6", "name7"};
  org_model_instance_names["name7"] = {"name8"};
  std::vector<std::unordered_set<std::string>> processed_model_instance_names;
  model_deployer.SetTrimmingModelInstanceNames(org_model_instance_names, processed_model_instance_names);
  EXPECT_EQ(processed_model_instance_names.size(), 2);
  EXPECT_EQ(processed_model_instance_names[0].size(), 4);
  EXPECT_EQ(processed_model_instance_names[1].size(), 4);
}

TEST_F(MasterModelDeployerTest, TestInitinalize) {
  DeployerProxy::GetInstance().Finalize();
  ResourceManager::GetInstance().Finalize();
  MasterModelDeployer model_deployer;
  HeterogeneousStubEnv::LoadAIServerHostConfig("valid/server/numa_config2.json");
  ASSERT_EQ(model_deployer.Initialize({}), SUCCESS);
  ASSERT_EQ(model_deployer.Finalize(), SUCCESS);
}

TEST_F(MasterModelDeployerTest, TestFinalize) {
  auto flow_model = StubModels::BuildFlowModel(StubModels::BuildGraphWithoutNeedForBindingQueues());
  MasterModelDeployer::DeployedModel deployed_model;
  MockMasterModelDeployer model_deployer;
  DeployResult deploy_result;
  ASSERT_EQ(model_deployer.DeployModel(flow_model, deploy_result), SUCCESS);
  ASSERT_EQ(model_deployer.deployed_models_.size(), 1);
  ASSERT_EQ(model_deployer.Finalize(), SUCCESS);
  ASSERT_TRUE(model_deployer.deployed_models_.empty());
}

TEST_F(MasterModelDeployerTest, TestWithErrorHostCompileRes) {
  auto flow_model = StubModels::BuildFlowModel(StubModels::BuildGraphWithoutNeedForBindingQueues());
  ASSERT_TRUE(flow_model != nullptr);
  // with host resource type while without device resource
  const auto compile_resource = MakeShared<ModelCompileResource>();
  compile_resource->host_resource_type = "stub_host_resource";
  flow_model->SetCompileResource(compile_resource);

  MasterModelDeployer::DeployedModel deployed_model;
  MockMasterModelDeployer model_deployer;
  DeployResult deploy_result;
  ASSERT_EQ(model_deployer.DeployModel(flow_model, deploy_result), FAILED);
}

TEST_F(MasterModelDeployerTest, TestGetDeviceMeshIndex) {
  Configurations::GetInstance().information_.node_config.node_id = 0;
  DeviceInfo local_device(0, NPU, 0);
  local_device.SetNodeMeshIndex({0, 0});
  ResourceManager::GetInstance().device_info_map_[0][0][NPU] = &local_device;
  MockMasterModelDeployer model_deployer;
  std::vector<int32_t> device_mesh_index;
  ASSERT_EQ(model_deployer.GetDeviceMeshIndex(0, device_mesh_index), SUCCESS);
  ASSERT_EQ(device_mesh_index.size(), 4);
  ASSERT_EQ(model_deployer.GetDeviceMeshIndex(1, device_mesh_index), SUCCESS);
  ASSERT_EQ(model_deployer.GetDeviceMeshIndex(2, device_mesh_index), FAILED);
}

TEST_F(MasterModelDeployerTest, TestGetValidLogicDeviceId) {
  MockMasterModelDeployer model_deployer;
  std::string logic_device_id = "0:0:0:0";
  ASSERT_EQ(model_deployer.GetValidLogicDeviceId(logic_device_id), SUCCESS);
}

TEST_F(MasterModelDeployerTest, TestWithDeviceCompileRes) {
  auto flow_model = StubModels::BuildFlowModel(StubModels::BuildGraphWithoutNeedForBindingQueues());
  ASSERT_TRUE(flow_model != nullptr);
  // with host resource type while without device resource
  const auto compile_resource = MakeShared<ModelCompileResource>();
  compile_resource->host_resource_type = "X86";
  compile_resource->logic_dev_id_to_res_type["0"] = "Ascend";
  flow_model->SetCompileResource(compile_resource);
  ResourceManager::GetInstance().compile_resource_.host_resource_type = "X86";
  ResourceManager::GetInstance().compile_resource_.logic_dev_id_to_res_type["0"] = "Arrch";

  MasterModelDeployer::DeployedModel deployed_model;
  MockMasterModelDeployer model_deployer;
  DeployResult deploy_result;
  ASSERT_EQ(model_deployer.DeployModel(flow_model, deploy_result), FAILED);

  flow_model->GetCompileResource()->logic_dev_id_to_res_type.clear();
  ResourceManager::GetInstance().compile_resource_.logic_dev_id_to_res_type["0"] = "Ascend";
  ASSERT_EQ(model_deployer.DeployModel(flow_model, deploy_result), FAILED);

  ResourceManager::GetInstance().compile_resource_.host_resource_type.clear();
  ResourceManager::GetInstance().compile_resource_.logic_dev_id_to_res_type.clear();
}

TEST_F(MasterModelDeployerTest, TestWithDeviceCompileResMoreErr) {
  auto flow_model = StubModels::BuildFlowModel(StubModels::BuildGraphWithoutNeedForBindingQueues());
  ASSERT_TRUE(flow_model != nullptr);
  // with host resource type while without device resource
  const auto compile_resource = MakeShared<ModelCompileResource>();
  compile_resource->host_resource_type = "X86";
  compile_resource->logic_dev_id_to_res_type["0"] = "Ascend";
  compile_resource->logic_dev_id_to_res_type["1"] = "Ascend";
  flow_model->SetCompileResource(compile_resource);
  ResourceManager::GetInstance().compile_resource_.host_resource_type = "X86";
  ResourceManager::GetInstance().compile_resource_.logic_dev_id_to_res_type["0"] = "Ascend";

  MasterModelDeployer::DeployedModel deployed_model;
  MockMasterModelDeployer model_deployer;
  DeployResult deploy_result;
  ASSERT_EQ(model_deployer.DeployModel(flow_model, deploy_result), FAILED);

  ResourceManager::GetInstance().compile_resource_.host_resource_type.clear();
  ResourceManager::GetInstance().compile_resource_.logic_dev_id_to_res_type.clear();
}

TEST_F(MasterModelDeployerTest, TestUndeployModel_InvalidModelId) {
  MasterModelDeployer model_deployer;
  ASSERT_EQ(model_deployer.Undeploy(0), PARAM_INVALID);
}

TEST_F(MasterModelDeployerTest, TestDeployRemoteVarManager) {
  EXPECT_EQ(VarManager::Instance(0)->Init(0, 0, 1, 0), SUCCESS);
  auto deploy_plan = StubModels::BuildSimpleDeployPlan();
  auto &remote_device =
      reinterpret_cast<stub::MockRemoteDeployer &>(*DeployerProxy::GetInstance().deployers_[1]);
  EXPECT_CALL(remote_device, Process)
      .WillRepeatedly(Return(SUCCESS));
  DeployState deploy_state;
  deploy_state.SetDeployPlan(std::move(deploy_plan));
  FlowModelSender flow_model_sender;
  ASSERT_EQ(flow_model_sender.DeployRemoteVarManager(deploy_state), SUCCESS);
}

TEST_F(MasterModelDeployerTest, TestDeployRemoteVarManagerWithFileConstant) {
  auto mock_runtime = std::make_shared<MockRuntime>();
  RuntimeStub::SetInstance(mock_runtime);
  auto &client_manager = DeployContext::LocalContext().GetFlowGwClientManager();
  client_manager.GetOrCreateClient(0, 0, {0}, false);
  (void)system("echo 1 > hello.bin");
  auto deploy_plan = StubModels::BuildSingleModelWithFileConstDeployPlan("hello.bin");
  auto &remote_device =
      reinterpret_cast<stub::MockRemoteDeployer &>(*DeployerProxy::GetInstance().deployers_[1]);
  EXPECT_CALL(remote_device, Process)
      .WillRepeatedly(Return(SUCCESS));
  DeployState deploy_state;
  deploy_state.SetDeployPlan(std::move(deploy_plan));
  FlowModelSender flow_model_sender;
  // TransferWeightWithQ中申请的Mbuf未释放
  ASSERT_EQ(flow_model_sender.DeployRemoteVarManager(deploy_state), SUCCESS);
  (void)system("rm -f hello.bin");
  client_manager.Finalize();
  RuntimeStub::Reset();
}

TEST_F(MasterModelDeployerTest, TestTransferFileConstants) {
  auto mock_runtime = std::make_shared<MockRuntime>();
  RuntimeStub::SetInstance(mock_runtime);
  auto &client_manager = DeployContext::LocalContext().GetFlowGwClientManager();
  client_manager.GetOrCreateClient(0, 0, {0}, false);
  auto &remote_device =
      reinterpret_cast<stub::MockRemoteDeployer &>(*DeployerProxy::GetInstance().deployers_[1]);
  EXPECT_CALL(remote_device, Process)
      .WillRepeatedly(Return(SUCCESS));
  (void)system("echo 1 > hello.bin");
  auto op_desc = make_shared<OpDesc>("var_name", FILECONSTANT);
  AttrUtils::SetStr(op_desc, ATTR_NAME_LOCATION, "hello.bin");
  AttrUtils::SetInt(op_desc, ATTR_NAME_OFFSET, 0);
  AttrUtils::SetInt(op_desc, ATTR_NAME_LENGTH, 2);
  GeShape shape({16, 16});
  GeTensorDesc tensor_desc(shape, FORMAT_ND, DT_INT16);
  op_desc->AddOutputDesc(tensor_desc);
  std::vector<int16_t> tensor(16 * 16);
  auto size = 16 * 16 * 2;
  EXPECT_EQ(VarManager::Instance(0)->AssignVarMem("var_name", nullptr, tensor_desc, RT_MEMORY_HBM), SUCCESS);
  std::string buffer(reinterpret_cast<char *>(tensor.data()), size);
  FlowModelSender flow_model_sender;
  EXPECT_EQ(flow_model_sender.DeployDevCfg(0, DeviceDebugConfig::ConfigType::kConfigTypeEnd), SUCCESS);
  std::map<int32_t, std::set<int32_t>> node_2_device_ids;
  std::set<int32_t> device_ids{0};
  (void)node_2_device_ids.emplace(1, device_ids);
  std::map<int32_t, std::map<uint64_t, std::map<OpDescPtr, std::set<int32_t>>>> node_need_transfer_memory;
  std::map<uint64_t, std::map<OpDescPtr, std::set<int32_t>>> session_2_desc;
  session_2_desc[0][op_desc].emplace(0);
  (void)node_need_transfer_memory.emplace(1, session_2_desc);
  // 释放TransferWeightWithQ中申请的Mbuf
  EXPECT_EQ(flow_model_sender.TransferFileConstants(node_2_device_ids, node_need_transfer_memory), SUCCESS);
  HeterogeneousExchangeService::GetInstance().Finalize();
  (void)system("rm -f hello.bin");
  client_manager.Finalize();
  RuntimeStub::Reset();
}

TEST_F(MasterModelDeployerTest, TestCopyOneWeightToTransfer) {
  auto mock_runtime = std::make_shared<MockRuntime>();
  RuntimeStub::SetInstance(mock_runtime);
  auto &client_manager = DeployContext::LocalContext().GetFlowGwClientManager();
  client_manager.GetOrCreateClient(0, 0, {0}, false);
  auto &remote_device =
      reinterpret_cast<stub::MockRemoteDeployer &>(*DeployerProxy::GetInstance().deployers_[1]);
  EXPECT_CALL(remote_device, Process)
      .WillRepeatedly(Return(SUCCESS));
  auto op_desc = make_shared<OpDesc>("var_name", FILECONSTANT);
  GeShape shape({16, 16});
  GeTensorDesc tensor_desc(shape, FORMAT_ND, DT_INT16);
  op_desc->AddOutputDesc(tensor_desc);
  std::vector<int16_t> tensor(16 * 16);
  auto size = 16 * 16 * 2;
  EXPECT_EQ(VarManager::Instance(0)->AssignVarMem("var_name", nullptr, tensor_desc, RT_MEMORY_HBM), SUCCESS);
  std::string buffer(reinterpret_cast<char *>(tensor.data()), size);
  std::stringstream ss(buffer);
  FlowModelSender::SendInfo send_info{};
  send_info.device_ids = {0};
  send_info.node_id = 1;
  FlowModelSender flow_model_sender;
  EXPECT_EQ(flow_model_sender.DeployDevCfg(0, DeviceDebugConfig::ConfigType::kConfigTypeEnd), SUCCESS);
  // TransferWeightWithQ中申请的Mbuf未释放
  EXPECT_EQ(flow_model_sender.CopyOneWeightToTransfer(send_info, ss, size, op_desc, {0}), SUCCESS);
  HeterogeneousExchangeService::GetInstance().Finalize();
  client_manager.Finalize();
  RuntimeStub::Reset();
}

TEST_F(MasterModelDeployerTest, TestInitFlowGwInfoSuccess) {
  uint32_t old_port = NetworkManager::GetInstance().main_port_;
  MasterModelDeployer model_deployer;
  ASSERT_EQ(model_deployer.InitFlowGwInfo(), SUCCESS);
  NetworkManager::GetInstance().main_port_ = old_port;
}

/**
 *     NetOutput
 *         |
 *       PC_3
 *      /   \
 *    PC_1  PC2
 *    |      |
 *  data1  data2
 */
TEST_F(MasterModelDeployerTest, TestDynamicSchedDeployModel_Success) {
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  mock_handle = (void *) 0x8888;
  EXPECT_EQ(VarManager::Instance(0)->Init(0, 0, 1, 0), SUCCESS);
  auto &exchange_service =
      reinterpret_cast<stub::MockExchangeService &>(ExecutionRuntime::GetInstance()->GetExchangeService());
  EXPECT_CALL(exchange_service, DoCreateQueue).WillRepeatedly(Return(SUCCESS));
  EXPECT_CALL(exchange_service, DestroyQueue).WillRepeatedly(Return(SUCCESS));

  map<std::string, std::string> sess_options = ge::GetThreadLocalContext().GetAllSessionOptions();
  GE_MAKE_GUARD(recover_sess_cfg, [&sess_options](){
    GetThreadLocalContext().SetSessionOption(sess_options);
  });

  map<std::string, std::string> options{{"TestSessionOption", "TestSessionOptionValue"}};
  GetThreadLocalContext().SetSessionOption(options);

  DeployContext deploy_context;
  deploy_context.Initialize();
  deploy_context.SetName("RemoteContext");
  auto stub_func = [&deploy_context](deployer::DeployerRequest &request, deployer::DeployerResponse &response) -> Status{
    return DeployerServiceImpl::GetInstance().Process(deploy_context, request, response);
  };
  
  auto &remote_device =
      reinterpret_cast<stub::MockRemoteDeployer &>(*DeployerProxy::GetInstance().deployers_[1]);
  EXPECT_CALL(remote_device, Process)
      .WillRepeatedly(Invoke(stub_func));

  auto flow_model = StubModels::BuildFlowModel(StubModels::BuildGraphWithoutNeedForBindingQueues());
  MasterModelDeployer::DeployedModel deployed_model;
  MockMasterModelDeployer model_deployer;
  DeployResult deploy_result;
  (void)AttrUtils::SetBool(flow_model->GetRootGraph(), "dynamic_schedule_enable", true);
  auto mock_runtime = std::make_shared<MockRuntimeNoLeaks>();
  RuntimeStub::SetInstance(mock_runtime);
  ASSERT_EQ(model_deployer.DeployModel(flow_model, deploy_result), SUCCESS);
  ASSERT_EQ(model_deployer.deployed_models_.size(), 1);
  ASSERT_EQ(model_deployer.UpdateProfilingInfo(true), SUCCESS);
  ASSERT_EQ(model_deployer.Undeploy(deploy_result.model_id), SUCCESS);
  ASSERT_EQ(model_deployer.deployed_models_.size(), 0);
  RuntimeStub::Reset();
}

TEST_F(MasterModelDeployerTest, TestExceptionClearDeployModel_Success) {
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  mock_handle = (void *) 0x8888;
  EXPECT_EQ(VarManager::Instance(0)->Init(0, 0, 1, 0), SUCCESS);
  auto &exchange_service =
      reinterpret_cast<stub::MockExchangeService &>(ExecutionRuntime::GetInstance()->GetExchangeService());
  EXPECT_CALL(exchange_service, DoCreateQueue).WillRepeatedly(Return(SUCCESS));
  EXPECT_CALL(exchange_service, DestroyQueue).WillRepeatedly(Return(SUCCESS));

  map<std::string, std::string> sess_options = ge::GetThreadLocalContext().GetAllSessionOptions();
  GE_MAKE_GUARD(recover_sess_cfg, [&sess_options](){
    GetThreadLocalContext().SetSessionOption(sess_options);
  });

  map<std::string, std::string> options{{"TestSessionOption", "TestSessionOptionValue"}};
  GetThreadLocalContext().SetSessionOption(options);

  DeployContext deploy_context;
  deploy_context.Initialize();
  deploy_context.SetName("RemoteContext");
  auto stub_func = [&deploy_context](deployer::DeployerRequest &request, deployer::DeployerResponse &response) -> Status{
    return SUCCESS;
  };
  auto &remote_device =
      reinterpret_cast<stub::MockRemoteDeployer &>(*DeployerProxy::GetInstance().deployers_[1]);
  EXPECT_CALL(remote_device, Process)
      .WillRepeatedly(Invoke(stub_func));

  auto flow_model = StubModels::BuildFlowModel(StubModels::BuildGraphWithoutNeedForBindingQueues());
  MasterModelDeployer::DeployedModel deployed_model;
  MockMasterModelDeployer model_deployer;
  DeployResult deploy_result;
  (void)AttrUtils::SetBool(flow_model->GetRootGraph(), "dynamic_schedule_enable", true);
  ASSERT_EQ(model_deployer.DeployModel(flow_model, deploy_result), SUCCESS);

  ASSERT_EQ(model_deployer.deployed_models_.size(), 1);
  std::vector<DeployPlan::DeviceInfo> device_infos;
  for (auto node : model_deployer.deployed_models_.begin()->second.deployed_remote_nodes) {
    DeployPlan::DeviceInfo device_info(0, node, 0);
    device_infos.push_back(device_info);
  }
  ASSERT_EQ(model_deployer.abnormal_status_handler_.ClearModelExceptionData(0, device_infos), SUCCESS);
  ASSERT_EQ(model_deployer.UpdateProfilingInfo(true), SUCCESS);
  ASSERT_EQ(model_deployer.Undeploy(deploy_result.model_id), SUCCESS);
  ASSERT_EQ(model_deployer.deployed_models_.size(), 0);
}

TEST_F(MasterModelDeployerTest, TestExceptionClearDeployModelOneModel_Success) {
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  mock_handle = (void *) 0x8888;
  EXPECT_EQ(VarManager::Instance(0)->Init(0, 0, 1, 0), SUCCESS);
  auto &exchange_service =
      reinterpret_cast<stub::MockExchangeService &>(ExecutionRuntime::GetInstance()->GetExchangeService());
  EXPECT_CALL(exchange_service, DoCreateQueue).WillRepeatedly(Return(SUCCESS));
  EXPECT_CALL(exchange_service, DestroyQueue).WillRepeatedly(Return(SUCCESS));

  map<std::string, std::string> sess_options = ge::GetThreadLocalContext().GetAllSessionOptions();
  GE_MAKE_GUARD(recover_sess_cfg, [&sess_options](){
    GetThreadLocalContext().SetSessionOption(sess_options);
  });

  map<std::string, std::string> options{{"TestSessionOption", "TestSessionOptionValue"}};
  GetThreadLocalContext().SetSessionOption(options);

  DeployContext deploy_context;
  deploy_context.Initialize();
  deploy_context.SetName("RemoteContext");
  auto stub_func = [&deploy_context](deployer::DeployerRequest &request, deployer::DeployerResponse &response) -> Status{
    return SUCCESS;
  };
  auto &remote_device =
      reinterpret_cast<stub::MockRemoteDeployer &>(*DeployerProxy::GetInstance().deployers_[1]);
  EXPECT_CALL(remote_device, Process)
      .WillRepeatedly(Invoke(stub_func));

  auto flow_model = StubModels::BuildFlowModel(StubModels::BuildGraphWithoutNeedForBindingQueues());
  MasterModelDeployer::DeployedModel deployed_model;
  MockMasterModelDeployer model_deployer;
  DeployResult deploy_result;
  (void)AttrUtils::SetBool(flow_model->GetRootGraph(), "dynamic_schedule_enable", true);
  ASSERT_EQ(model_deployer.DeployModel(flow_model, deploy_result), SUCCESS);

  ASSERT_EQ(model_deployer.deployed_models_.size(), 1);
  std::vector<DeployPlan::DeviceInfo> device_infos;
  for (auto node : model_deployer.deployed_models_.begin()->second.deployed_remote_nodes) {
    DeployPlan::DeviceInfo device_info(0, node, 0);
    device_infos.push_back(device_info);
    break;
  }
  ASSERT_EQ(model_deployer.abnormal_status_handler_.ClearModelExceptionData(0, device_infos), SUCCESS);
  ASSERT_EQ(model_deployer.UpdateProfilingInfo(true), SUCCESS);
  ASSERT_EQ(model_deployer.Undeploy(deploy_result.model_id), SUCCESS);
  ASSERT_EQ(model_deployer.deployed_models_.size(), 0);
}

TEST_F(MasterModelDeployerTest, ClearModelExceptionData_mul_node) {
  MasterModelDeployer model_deployer;
  AbnormalStatusHandler::DeployedModel deploy_model;
  deploy_model.deployed_remote_nodes.insert(0);
  deploy_model.deployed_remote_nodes.insert(1);
  model_deployer.abnormal_status_handler_.deployed_models_[0] = deploy_model;
  std::vector<DeployPlan::DeviceInfo> device_infos;
  DeployPlan::DeviceInfo device_info(0, 0, 0);
  device_infos.push_back(device_info);
  ASSERT_EQ(model_deployer.abnormal_status_handler_.ClearModelExceptionData(0, device_infos), SUCCESS);
}

void DeleteLines(const char* real_path, const std::vector<int>& lineNumbers) {
  std::ifstream inputFile(real_path);
  if (!inputFile.is_open()) {
    std::cerr << "Failed to open input file: " << real_path << std::endl;
    return;
  }
  std::vector<std::string> fileContent;
  std::string line;
  while (std::getline(inputFile, line)) {
    fileContent.push_back(line);
  }
  inputFile.close();
  std::stringstream ss;
  ss << real_path << "_bk";
  std::ofstream outputFile(ss.str());
  if (!outputFile.is_open()) {
    std::cerr << "Failed to open output file: " << ss.str() << std::endl;
    return;
  }
  for (const auto& line : fileContent) {
    outputFile << line << std::endl;
  }
  outputFile.close();

  std::vector<std::string> newContent;
  int lineNumber = 1;
  for (const auto& line : fileContent) {
    if (std::find(lineNumbers.begin(), lineNumbers.end(), lineNumber) == lineNumbers.end()) {
      newContent.push_back(line);
    }
    lineNumber++;
  }
  std::ofstream inputFile2(real_path, std::ios::trunc);
  if (!inputFile2.is_open()) {
    std::cerr << "Failed to open input file: " << real_path << std::endl;
    return;
  }
  for (const auto& line : newContent) {
    inputFile2 << line << std::endl;
  }
  inputFile2.close();
}

void CreateRedeployFile(const char* real_path) {
  std::string parent_path = real_path;
  parent_path = parent_path.substr(0, parent_path.find_last_of("/\\") + 1);
  std::ofstream redeploy_file(parent_path + "redeploy");
  if (redeploy_file.is_open()) {
    std::cout << "redeploy file created!" << std::endl;
    redeploy_file.close();
  }
  else {
    std::cerr << "Error creating redeploy file!" << std::endl;
  }
}

void RestoreFile(const char* path) {
  std::string str_path(path);
  size_t pos = str_path.find_last_of("/\\");
  if (pos == std::string::npos) {
    std::cerr << "Invalid path: " << path << std::endl;
    return;
  }
  std::string dir_path = str_path.substr(0, pos);
  std::string file_name = str_path.substr(pos + 1);
  std::string bk_file_name = file_name + "_bk";
  std::string bk_file_path = dir_path + "/" + bk_file_name;
  std::string new_file_path = dir_path + "/" + file_name;
  std::string redeploy_file_path = dir_path + "/" + "redeploy";
  std::string result_file_path = dir_path + "/" + "redeploy.done";
  std::string result_error_file_path = dir_path + "/" + "redeploy.error";

  std::ifstream redeploy_file(redeploy_file_path);
  if (redeploy_file) {
    std::cout << "redeploy.done file created!" << std::endl;
    redeploy_file.close();
    std::remove(redeploy_file_path.c_str());
  }
  std::ifstream redeploy_done_file(result_file_path);
  if (redeploy_done_file) {
    std::cout << "redeploy.done file created!" << std::endl;
    redeploy_done_file.close();
    std::remove(result_file_path.c_str());
  }
  std::ifstream redeploy_error_file(result_error_file_path);
  if (redeploy_error_file) {
    std::cout << "redeploy.error file created!" << std::endl;
    redeploy_error_file.close();
    std::remove(result_error_file_path.c_str());
  }

  std::remove(new_file_path.c_str());
  std::rename(bk_file_path.c_str(), new_file_path.c_str());
}

TEST_F(MasterModelDeployerTest, TestFileMonitorWithResourceConfigPath_Success) {
  DeployerProxy::GetInstance().Finalize();
  ResourceManager::GetInstance().Finalize();
  MasterModelDeployer model_deployer;
  HeterogeneousStubEnv::LoadAIServerHostConfig("redeploy/server/numa_config.json");
  ASSERT_EQ(model_deployer.Initialize({}), SUCCESS);
  model_deployer.abnormal_status_handler_.is_dynamic_sched_ = true;
  DeployPlan::AbnormalStatusCallbackInfo abnormal_status_callback_info;
  {
    DeployPlan::AbnormalStatusCallback abnormal_status_clear_callback = [this](
        uint32_t abnormal_status_operation_type, RootModelId2SubmodelName &abnormal_submodel_instances_name)
        -> Status {
      (void) abnormal_submodel_instances_name;
      (void) abnormal_status_operation_type;
      return SUCCESS;
    };
    std::lock_guard<std::mutex> lk(model_deployer.abnormal_status_handler_.abnormal_status_callback_info_.mu);
    (model_deployer.abnormal_status_handler_.abnormal_status_callback_info_.callback_list)[0U] = abnormal_status_clear_callback;
  }
  AbnormalStatusHandler::DeployedModel deploy_model;
  DeployPlan::DeviceInfo local_device1(NPU, 0, 0);
  DeployPlan::DeviceInfo local_device2(NPU, 0, 1);
  DeployPlan::DeviceInfo local_device3(NPU, 0, 2);
  DeployPlan::DeviceInfo local_device4(NPU, 0, 3);
  DeployPlan::DeviceInfo local_device5(NPU, 0, 4);
  DeployPlan::DeviceInfo local_device6(NPU, 0, 5);
  DeployPlan::DeviceInfo local_device7(NPU, 0, 6);
  DeployPlan::DeviceInfo local_device8(NPU, 0, 7);
  deploy_model.model_deploy_infos["model1"]["model1_0_0_0"] = {local_device1};
  deploy_model.model_deploy_infos["model1"]["model1_0_0_1"] = {local_device2};
  deploy_model.model_deploy_infos["model1"]["model1_0_0_2"] = {local_device3};
  deploy_model.model_deploy_infos["model1"]["model1_0_0_3"] = {local_device4};
  deploy_model.model_deploy_infos["model1"]["model1_0_0_4"] = {local_device5};
  deploy_model.model_deploy_infos["model1"]["model1_0_0_5"] = {local_device6};
  deploy_model.model_deploy_infos["model1"]["model1_0_0_6"] = {local_device7};
  deploy_model.model_deploy_infos["model1"]["model1_0_0_7"] = {local_device8};
  model_deployer.abnormal_status_handler_.deployed_models_[0] = deploy_model;
  // 更改numa_config.json
  std::string config_path = PathUtils::Join(
    {EnvPath().GetAirBasePath(), "tests/dflow/runner/ut/ge/runtime/data/redeploy/server/numa_config.json"});
  char real_path[200];
  realpath(config_path.c_str(), real_path);
  std::vector<int> lineNumbers = {45, 46, 47, 48, 49};
  DeleteLines(real_path, lineNumbers);
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  CreateRedeployFile(real_path);
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  ASSERT_EQ(model_deployer.Finalize(), SUCCESS);
  RestoreFile(real_path);
}

TEST_F(MasterModelDeployerTest, TestFileMonitorWithMulRootModel_Success) {
  DeployerProxy::GetInstance().Finalize();
  ResourceManager::GetInstance().Finalize();
  MasterModelDeployer model_deployer;
  HeterogeneousStubEnv::LoadAIServerHostConfig("redeploy/server/numa_config.json");
  ASSERT_EQ(model_deployer.Initialize({}), SUCCESS);
  model_deployer.abnormal_status_handler_.is_dynamic_sched_ = true;
  DeployPlan::AbnormalStatusCallbackInfo abnormal_status_callback_info;
  {
    DeployPlan::AbnormalStatusCallback abnormal_status_clear_callback = [this](
        uint32_t abnormal_status_operation_type, RootModelId2SubmodelName &abnormal_submodel_instances_name)
        -> Status {
      (void) abnormal_submodel_instances_name;
      (void) abnormal_status_operation_type;
      return SUCCESS;
    };
    std::lock_guard<std::mutex> lk(model_deployer.abnormal_status_handler_.abnormal_status_callback_info_.mu);
    (model_deployer.abnormal_status_handler_.abnormal_status_callback_info_.callback_list)[0U] = abnormal_status_clear_callback;
    (model_deployer.abnormal_status_handler_.abnormal_status_callback_info_.callback_list)[1U] = abnormal_status_clear_callback;
  }
  AbnormalStatusHandler::DeployedModel deploy_model;
  DeployPlan::DeviceInfo local_device1(NPU, 0, 0);
  DeployPlan::DeviceInfo local_device2(NPU, 0, 1);
  DeployPlan::DeviceInfo local_device3(NPU, 0, 2);
  DeployPlan::DeviceInfo local_device4(NPU, 0, 3);
  DeployPlan::DeviceInfo local_device5(NPU, 0, 4);
  DeployPlan::DeviceInfo local_device6(NPU, 0, 5);
  DeployPlan::DeviceInfo local_device7(NPU, 0, 6);
  DeployPlan::DeviceInfo local_device8(NPU, 0, 7);
  deploy_model.model_deploy_infos["model1"]["model1_0_0_0"] = {local_device1};
  deploy_model.model_deploy_infos["model1"]["model1_0_0_1"] = {local_device2};
  deploy_model.model_deploy_infos["model1"]["model1_0_0_2"] = {local_device3};
  deploy_model.model_deploy_infos["model1"]["model1_0_0_3"] = {local_device4};
  deploy_model.model_deploy_infos["model1"]["model1_0_0_4"] = {local_device5};
  deploy_model.model_deploy_infos["model1"]["model1_0_0_5"] = {local_device6};
  deploy_model.model_deploy_infos["model1"]["model1_0_0_6"] = {local_device7};
  deploy_model.model_deploy_infos["model1"]["model1_0_0_7"] = {local_device8};
  model_deployer.abnormal_status_handler_.deployed_models_[0] = deploy_model;

  AbnormalStatusHandler::DeployedModel deploy_model1;
  deploy_model1.model_deploy_infos["model1"]["model1_0_0_0"] = {local_device1};
  deploy_model1.model_deploy_infos["model1"]["model1_0_0_1"] = {local_device2};
  deploy_model1.model_deploy_infos["model1"]["model1_0_0_2"] = {local_device3};
  deploy_model1.model_deploy_infos["model1"]["model1_0_0_3"] = {local_device4};
  deploy_model1.model_deploy_infos["model1"]["model1_0_0_4"] = {local_device5};
  deploy_model1.model_deploy_infos["model1"]["model1_0_0_5"] = {local_device6};
  deploy_model1.model_deploy_infos["model1"]["model1_0_0_6"] = {local_device7};
  deploy_model1.model_deploy_infos["model1"]["model1_0_0_7"] = {local_device8};
  model_deployer.abnormal_status_handler_.deployed_models_[1] = deploy_model1;

  // 更改numa_config.json
  std::string config_path = PathUtils::Join(
    {EnvPath().GetAirBasePath(), "tests/dflow/runner/ut/ge/runtime/data/redeploy/server/numa_config.json"});
  char real_path[200];
  realpath(config_path.c_str(), real_path);
  std::vector<int> lineNumbers = {45, 46, 47, 48, 49};
  DeleteLines(real_path, lineNumbers);
  CreateRedeployFile(real_path);
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  ASSERT_EQ(model_deployer.Finalize(), SUCCESS);
  RestoreFile(real_path);
}

TEST_F(MasterModelDeployerTest, TestFileMonitorWithResourceConfigPath_FAILED) {
  DeployerProxy::GetInstance().Finalize();
  ResourceManager::GetInstance().Finalize();
  MasterModelDeployer model_deployer;
  HeterogeneousStubEnv::LoadAIServerHostConfig("redeploy/server/numa_config.json");
  ASSERT_EQ(model_deployer.Initialize({}), SUCCESS);
  model_deployer.abnormal_status_handler_.is_dynamic_sched_ = false;
  {
    DeployPlan::AbnormalStatusCallback abnormal_status_clear_callback = [this](
        uint32_t abnormal_status_operation_type, RootModelId2SubmodelName &abnormal_submodel_instances_name)
        -> Status {
      (void) abnormal_submodel_instances_name;
      (void) abnormal_status_operation_type;
      return SUCCESS;
    };
    std::lock_guard<std::mutex> lk(model_deployer.abnormal_status_handler_.abnormal_status_callback_info_.mu);
    (model_deployer.abnormal_status_handler_.abnormal_status_callback_info_.callback_list)[0U] =
        abnormal_status_clear_callback;
  }
  AbnormalStatusHandler::DeployedModel deploy_model;
  DeployPlan::DeviceInfo local_device(NPU, 0, 6);
  deploy_model.model_deploy_infos["model1"]["model1_1_0_0"] = {local_device};
  model_deployer.abnormal_status_handler_.deployed_models_[0U] = deploy_model;
  // 更改numa_config.json
  std::string config_path = PathUtils::Join(
    {EnvPath().GetAirBasePath(), "tests/dflow/runner/ut/ge/runtime/data/redeploy/server/numa_config.json"});
  char real_path[200];
  (void)realpath(config_path.c_str(), real_path);
  std::vector<int> lineNumbers = {45, 46, 47, 48, 49};
  DeleteLines(real_path, lineNumbers);
  CreateRedeployFile(real_path);
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  ASSERT_EQ(model_deployer.Finalize(), SUCCESS);
  RestoreFile(real_path);
}

TEST_F(MasterModelDeployerTest, TestHeartbeatMonitor_Success) {
  DeployerProxy::GetInstance().Finalize();
  ResourceManager::GetInstance().Finalize();
  MasterModelDeployer model_deployer;
  HeterogeneousStubEnv::LoadAIServerHostConfig("redeploy/server/numa_config.json");
  ASSERT_EQ(model_deployer.Initialize({}), SUCCESS);
  model_deployer.abnormal_status_handler_.is_dynamic_sched_ = true;
  DeployPlan::AbnormalStatusCallbackInfo abnormal_status_callback_info;
  {
    DeployPlan::AbnormalStatusCallback abnormal_status_clear_callback = [](
        uint32_t abnormal_status_operation_type, RootModelId2SubmodelName &abnormal_submodel_instances_name)
        -> Status {
      (void) abnormal_submodel_instances_name;
      (void) abnormal_status_operation_type;
      return SUCCESS;
    };
    std::lock_guard<std::mutex> lk(model_deployer.abnormal_status_handler_.abnormal_status_callback_info_.mu);
    (model_deployer.abnormal_status_handler_.abnormal_status_callback_info_.callback_list)[0U] = abnormal_status_clear_callback;
  }
  AbnormalStatusHandler::DeployedModel deploy_model;
  deploy_model.model_id = 0;
  DeployPlan::DeviceInfo local_device(NPU, 0, 0);
  deploy_model.model_deploy_infos["model1"]["model1_1_0_0_pid1"] = {local_device};
  deploy_model.model_deploy_infos["model1"]["model1_1_0_0_pid2"] = {local_device};
  model_deployer.abnormal_status_handler_.deployed_models_[0] = deploy_model;
  // 触发进程异常
  auto &deploy_context = DeployContext::LocalContext();
  {
    std::lock_guard<std::mutex> lk(deploy_context.GetAbnormalHeartbeatInfoMu());
    deploy_context.AddAbnormalSubmodelInstanceName(0,"model1_1_0_0_pid1");
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  ASSERT_EQ(model_deployer.Finalize(), SUCCESS);
}

TEST_F(MasterModelDeployerTest, TestHeartbeatMonitorFailedWithNode0CpuAbnormal) {
  DeployerProxy::GetInstance().Finalize();
  ResourceManager::GetInstance().Finalize();
  MasterModelDeployer model_deployer;
  HeterogeneousStubEnv::LoadAIServerHostConfig("redeploy/server/numa_config.json");
  ASSERT_EQ(model_deployer.Initialize({}), SUCCESS);
  model_deployer.abnormal_status_handler_.is_dynamic_sched_ = true;
  DeployPlan::AbnormalStatusCallbackInfo abnormal_status_callback_info;
  {
    DeployPlan::AbnormalStatusCallback abnormal_status_clear_callback = [](
        uint32_t abnormal_status_operation_type, RootModelId2SubmodelName &abnormal_submodel_instances_name)
        -> Status {
      (void) abnormal_submodel_instances_name;
      (void) abnormal_status_operation_type;
      return SUCCESS;
    };
    std::lock_guard<std::mutex> lk(model_deployer.abnormal_status_handler_.abnormal_status_callback_info_.mu);
    (model_deployer.abnormal_status_handler_.abnormal_status_callback_info_.callback_list)[0U] = abnormal_status_clear_callback;
  }
  AbnormalStatusHandler::DeployedModel deploy_model;
  deploy_model.model_id = 0;
  DeployPlan::DeviceInfo local_device(NPU, 0, 0);
  deploy_model.model_deploy_infos["model1"]["model1_1_0_0"] = {local_device};
  model_deployer.abnormal_status_handler_.deployed_models_[0] = deploy_model;
  // 触发进程异常
  auto &deploy_context = DeployContext::LocalContext();
  {
    std::lock_guard<std::mutex> lk(deploy_context.GetAbnormalHeartbeatInfoMu());
    deploy_context.AddAbnormalSubmodelInstanceName(0,"model1_1_0_0");
    deploy_context.AddAbnormalSubmodelInstanceName(0,"model2");
    deploy_context.AddAbnormalSubmodelInstanceName(0,"model3");
    NodeConfig node_config0;
    node_config0.node_id = 0;
    DeviceConfig device_info0;
    device_info0.device_type = CPU;
    device_info0.device_id = 0;
    node_config0.device_list.push_back(device_info0);
    NodeConfig node_config1;
    node_config1.node_id = 1;
    DeviceConfig device_info1;
    device_info1.device_type = NPU;
    device_info1.device_id = 1;
    node_config1.device_list.push_back(device_info1);
    deploy_context.AddAbnormalNodeConfig(node_config0);
    deploy_context.AddAbnormalNodeConfig(node_config1);
    DeployPlan::DeviceInfo device_info(0, 0, 1);
    deploy_context.AddAbnormalDeviceInfo(device_info);
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  ASSERT_EQ(model_deployer.Finalize(), SUCCESS);
}

TEST_F(MasterModelDeployerTest, TestHeartbeatMonitorFailedWithDevice0) {
  DeployerProxy::GetInstance().Finalize();
  ResourceManager::GetInstance().Finalize();
  MasterModelDeployer model_deployer;
  HeterogeneousStubEnv::LoadAIServerHostConfig("redeploy/server/numa_config.json");
  ASSERT_EQ(model_deployer.Initialize({}), SUCCESS);
  model_deployer.abnormal_status_handler_.is_dynamic_sched_ = true;
  DeployPlan::AbnormalStatusCallbackInfo abnormal_status_callback_info;
  {
    DeployPlan::AbnormalStatusCallback abnormal_status_clear_callback = [](
        uint32_t abnormal_status_operation_type, RootModelId2SubmodelName &abnormal_submodel_instances_name)
        -> Status {
      (void) abnormal_submodel_instances_name;
      (void) abnormal_status_operation_type;
      return SUCCESS;
    };
    std::lock_guard<std::mutex> lk(model_deployer.abnormal_status_handler_.abnormal_status_callback_info_.mu);
    (model_deployer.abnormal_status_handler_.abnormal_status_callback_info_.callback_list)[0U] = abnormal_status_clear_callback;
  }
  AbnormalStatusHandler::DeployedModel deploy_model;
  deploy_model.model_id = 0;
  DeployPlan::DeviceInfo local_device(NPU, 0, 0);
  deploy_model.model_deploy_infos["model1"]["model1_1_0_0"] = {local_device};
  model_deployer.abnormal_status_handler_.deployed_models_[0] = deploy_model;
  // 触发进程异常
  auto &deploy_context = DeployContext::LocalContext();
  {
    std::lock_guard<std::mutex> lk(deploy_context.GetAbnormalHeartbeatInfoMu());
    DeployPlan::DeviceInfo device_info(0, 0, 0);
    deploy_context.AddAbnormalDeviceInfo(device_info);
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  ASSERT_EQ(model_deployer.Finalize(), SUCCESS);
}

TEST_F(MasterModelDeployerTest, TestHeartbeatMonitorWithNoNewAbnormal) {
  DeployerProxy::GetInstance().Finalize();
  ResourceManager::GetInstance().Finalize();
  MasterModelDeployer model_deployer;
  HeterogeneousStubEnv::LoadAIServerHostConfig("redeploy/server/numa_config.json");
  ASSERT_EQ(model_deployer.Initialize({}), SUCCESS);
  model_deployer.abnormal_status_handler_.is_dynamic_sched_ = true;
  DeployPlan::AbnormalStatusCallbackInfo abnormal_status_callback_info;
  {
    DeployPlan::AbnormalStatusCallback abnormal_status_clear_callback = [](
        uint32_t abnormal_status_operation_type, RootModelId2SubmodelName &abnormal_submodel_instances_name)
        -> Status {
      (void) abnormal_submodel_instances_name;
      (void) abnormal_status_operation_type;
      return SUCCESS;
    };
    std::lock_guard<std::mutex> lk(model_deployer.abnormal_status_handler_.abnormal_status_callback_info_.mu);
    (model_deployer.abnormal_status_handler_.abnormal_status_callback_info_.callback_list)[0U] = abnormal_status_clear_callback;
  }
  AbnormalStatusHandler::DeployedModel deploy_model;
  deploy_model.model_id = 0;
  DeployPlan::DeviceInfo local_device(NPU, 0, 0);
  deploy_model.model_deploy_infos["model1"]["model1_1_0_0"] = {local_device};
  model_deployer.abnormal_status_handler_.deployed_models_[0] = deploy_model;
  // 触发进程异常
  auto &deploy_context = DeployContext::LocalContext();
  {
    std::lock_guard<std::mutex> lk(deploy_context.GetAbnormalHeartbeatInfoMu());
    deploy_context.AddAbnormalSubmodelInstanceName(0,"model0");
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  ASSERT_EQ(model_deployer.Finalize(), SUCCESS);
}

TEST_F(MasterModelDeployerTest, TestHeartbeatMonitorWithMulRootModel_Success) {
  DeployerProxy::GetInstance().Finalize();
  ResourceManager::GetInstance().Finalize();
  MasterModelDeployer model_deployer;
  HeterogeneousStubEnv::LoadAIServerHostConfig("redeploy/server/numa_config.json");
  ASSERT_EQ(model_deployer.Initialize({}), SUCCESS);
  model_deployer.abnormal_status_handler_.is_dynamic_sched_ = true;
  DeployPlan::AbnormalStatusCallbackInfo abnormal_status_callback_info;
  {
    DeployPlan::AbnormalStatusCallback abnormal_status_clear_callback = [](
        uint32_t abnormal_status_operation_type, RootModelId2SubmodelName &abnormal_submodel_instances_name)
        -> Status {
      (void) abnormal_submodel_instances_name;
      (void) abnormal_status_operation_type;
      return SUCCESS;
    };
    std::lock_guard<std::mutex> lk(model_deployer.abnormal_status_handler_.abnormal_status_callback_info_.mu);
    (model_deployer.abnormal_status_handler_.abnormal_status_callback_info_.callback_list)[0U] = abnormal_status_clear_callback;
    (model_deployer.abnormal_status_handler_.abnormal_status_callback_info_.callback_list)[1U] = abnormal_status_clear_callback;
  }
  AbnormalStatusHandler::DeployedModel deploy_model;
  deploy_model.model_id = 0;
  DeployPlan::DeviceInfo local_device(NPU, 0, 0);
  deploy_model.model_deploy_infos["model1"]["model1_1_0_0"] = {local_device};
  model_deployer.abnormal_status_handler_.deployed_models_[0] = deploy_model;

  AbnormalStatusHandler::DeployedModel deploy_model1;
  deploy_model1.model_id = 1;
  deploy_model1.model_deploy_infos["model1"]["model1_1_0_0"] = {local_device};
  model_deployer.abnormal_status_handler_.deployed_models_[1] = deploy_model1;
  // 触发进程异常
  auto &deploy_context = DeployContext::LocalContext();
  {
    std::lock_guard<std::mutex> lk(deploy_context.GetAbnormalHeartbeatInfoMu());
    deploy_context.AddAbnormalSubmodelInstanceName(0, "model1");
    deploy_context.AddAbnormalSubmodelInstanceName(0, "model2");
    deploy_context.AddAbnormalSubmodelInstanceName(0, "model3");
    deploy_context.AddAbnormalSubmodelInstanceName(1, "model1");
    deploy_context.AddAbnormalSubmodelInstanceName(1, "model2");
    deploy_context.AddAbnormalSubmodelInstanceName(1, "model3");
    NodeConfig node_config0;
    node_config0.node_id = 0;
    DeviceConfig device_info0;
    device_info0.device_type = CPU;
    device_info0.device_id = 0;
    node_config0.device_list.push_back(device_info0);
    NodeConfig node_config1;
    node_config1.node_id = 1;
    DeviceConfig device_info1;
    device_info1.device_type = NPU;
    device_info1.device_id = 1;
    node_config1.device_list.push_back(device_info1);
    deploy_context.AddAbnormalNodeConfig(node_config0);
    deploy_context.AddAbnormalNodeConfig(node_config1);
    DeployPlan::DeviceInfo device_info(0, 0, 1);
    deploy_context.AddAbnormalDeviceInfo(device_info);
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  ASSERT_EQ(model_deployer.Finalize(), SUCCESS);
}

TEST_F(MasterModelDeployerTest, TestFindAbnormalDeviceOnServer) {
  DeployerProxy::GetInstance().Finalize();
  ResourceManager::GetInstance().Finalize();
  MasterModelDeployer model_deployer;
  HeterogeneousStubEnv::LoadAIServerHostConfig("redeploy/server/numa_config.json");
  ASSERT_EQ(model_deployer.Initialize({}), SUCCESS);
  DeployPlan::DeviceStateList device_state_list;
  DeployerConfig information_new;
  DeployerConfig information_old;
  information_old.node_config.ipaddr = "127.0.0.2";
  NodeConfig node_config1;
  node_config1.node_id = 0;
  node_config1.ipaddr = "127.0.0.3";
  node_config1.node_mesh_index = {0, 0, 1};
  NodeConfig node_config2;
  node_config2.node_id = 0;
  node_config2.ipaddr = "127.0.0.4";
  node_config2.node_mesh_index = {0, 0, 2};
  NodeConfig node_config3;
  node_config3.node_id = 0;
  node_config3.ipaddr = "127.0.0.5";
  node_config3.node_mesh_index = {0, 0, 3};
  DeviceConfig device0;
  device0.device_id = 0;
  node_config3.device_list.push_back(device0);
  information_new.remote_node_config_list.push_back(node_config1);
  information_new.remote_node_config_list.push_back(node_config2);
  information_old.remote_node_config_list.push_back(node_config1);
  information_old.remote_node_config_list.push_back(node_config2);
  information_old.remote_node_config_list.push_back(node_config3);
  ASSERT_EQ(model_deployer.abnormal_status_handler_.FindAbnormalDeviceOnServer(device_state_list,
      information_new, information_old), SUCCESS);
  ASSERT_EQ(model_deployer.Finalize(), SUCCESS);
}

TEST_F(MasterModelDeployerTest, TestPreHandleAbnormalInfo) {
  DeployerProxy::GetInstance().Finalize();
  ResourceManager::GetInstance().Finalize();
  MasterModelDeployer model_deployer;
  HeterogeneousStubEnv::LoadAIServerHostConfig("redeploy/server/numa_config.json");
  ASSERT_EQ(model_deployer.Initialize({}), SUCCESS);

  AbnormalStatusHandler::DeployedModel deployed_model;
  model_deployer.abnormal_status_handler_.deployed_models_[1U] = deployed_model;
  std::thread thread_run;
  thread_run = std::thread([&model_deployer]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(100)); // 等待0.1秒进行下一次读取
    DeployPlan::AbnormalStatusCallback abnormal_status_clear_callback = [&model_deployer](
        uint32_t abnormal_status_operation_type, RootModelId2SubmodelName &abnormal_submodel_instances_name)
        -> Status {
      (void) abnormal_status_operation_type;
      (void) abnormal_submodel_instances_name;
      return SUCCESS;
    };
    std::lock_guard<std::mutex> lk(model_deployer.abnormal_status_handler_.abnormal_status_callback_info_.mu);
    model_deployer.abnormal_status_handler_.abnormal_status_callback_info_.callback_list[1U] =
        abnormal_status_clear_callback;
  });
  model_deployer.abnormal_status_handler_.PreHandleAbnormalInfo();
  ASSERT_EQ(model_deployer.abnormal_status_handler_.RedeployStart(1U), SUCCESS);
  if (thread_run.joinable()) {
    thread_run.join();
  }
  ASSERT_EQ(model_deployer.Finalize(), SUCCESS);
}

TEST_F(MasterModelDeployerTest, TestIsReployFileGeneratedThenRemove) {
  MasterModelDeployer model_deployer;
  std::string file_path1 = "test.json";
  ASSERT_EQ(model_deployer.abnormal_status_handler_.IsReployFileGeneratedThenRemove(file_path1), false);
  std::string file_path2 = "this_is_a_not_exist_dir/redeploy/server/";
  ASSERT_EQ(model_deployer.abnormal_status_handler_.IsReployFileGeneratedThenRemove(file_path2), false);
  ASSERT_EQ(model_deployer.Finalize(), SUCCESS);
}

TEST_F(MasterModelDeployerTest, TestIsSupportDynamicSched) {
  DeployerProxy::GetInstance().Finalize();
  ResourceManager::GetInstance().Finalize();
  MasterModelDeployer model_deployer;
  HeterogeneousStubEnv::LoadAIServerHostConfig("redeploy/server/numa_config.json");
  ASSERT_EQ(model_deployer.Initialize({}), SUCCESS);
  DeployPlan::DeviceStateList device_state_list;
  DeployPlan::DeviceInfo device_info = DeployPlan::DeviceInfo(NPU, 1, 0);
  device_state_list[device_info] = false;
  model_deployer.abnormal_status_handler_.is_dynamic_sched_ = false;
  EXPECT_EQ(model_deployer.abnormal_status_handler_.IsSupportDynamicSchedRecover(0), false);
  model_deployer.abnormal_status_handler_.is_dynamic_sched_ = true;
  EXPECT_EQ(model_deployer.abnormal_status_handler_.IsSupportDynamicSchedRecover(0), true);
  device_state_list.clear();
  DeployPlan::DeviceInfo device_info1 = DeployPlan::DeviceInfo(NPU, 1, 1);
  device_state_list[device_info1] = false;
  AbnormalStatusHandler::DeployedModel deploy_model;
  deploy_model.model_id = 0;
  deploy_model.model_deploy_infos["1"]["1_0_0_0"] = {device_info1};
  model_deployer.abnormal_status_handler_.deployed_models_[0] = deploy_model;
  model_deployer.abnormal_status_handler_.abnormal_submodel_instances_name_[0]["1_0_0_0"] = false;
  EXPECT_EQ(model_deployer.abnormal_status_handler_.IsSupportDynamicSchedRecover(0), false);
  ASSERT_EQ(model_deployer.Finalize(), SUCCESS);
}

TEST_F(MasterModelDeployerTest, TestIsInDeviceList) {
  DeployerProxy::GetInstance().Finalize();
  ResourceManager::GetInstance().Finalize();
  MasterModelDeployer model_deployer;
  HeterogeneousStubEnv::LoadAIServerHostConfig("redeploy/server/numa_config.json");
  ASSERT_EQ(model_deployer.Initialize({}), SUCCESS);
  DeployPlan::DeviceStateList device_state_list;
  DeployPlan::DeviceInfo device_info = DeployPlan::DeviceInfo(NPU, 0, 1);
  device_state_list[device_info] = false;
  std::set<DeployPlan::DeviceInfo> device_info_proxy;
  device_info_proxy.emplace(DeployPlan::DeviceInfo(CPU, 0, 0, 1));
  ASSERT_EQ(model_deployer.abnormal_status_handler_.IsInDeviceList(device_info_proxy, device_state_list), false);
  ASSERT_EQ(model_deployer.Finalize(), SUCCESS);
}

TEST_F(MasterModelDeployerTest, TestNotifyException) {
  uint32_t root_model_id = 1;
  MasterModelDeployer model_deployer;
  model_deployer.deployed_models_[root_model_id].deployed_remote_nodes = {};
  UserExceptionNotify notify{};
  auto ret = model_deployer.NotifyException(root_model_id, notify);
  EXPECT_EQ(ret, SUCCESS);
  ret = model_deployer.NotifyException(999, notify);
  EXPECT_NE(ret, SUCCESS);
}

TEST_F(MasterModelDeployerTest, TestUpdateProf_OtherRank) {
  std::map<std::string, std::string> graph_options = GetThreadLocalContext().GetAllGraphOptions();
  graph_options[OPTION_EXEC_RANK_ID] = "10";
  GetThreadLocalContext().SetGraphOption(graph_options);
  MasterModelDeployer model_deployer;
  ASSERT_EQ(model_deployer.UpdateProfilingInfo(true), SUCCESS);
  ASSERT_EQ(model_deployer.Finalize(), SUCCESS);
  graph_options.clear();
  GetThreadLocalContext().SetGraphOption(graph_options);
}

TEST_F(MasterModelDeployerTest, TestGetRemoteGroupCacheConfig) {
  MasterModelDeployer model_deployer;
  std::string remote_group_cache_config;
  std::map<std::string, std::string> graph_options = GetThreadLocalContext().GetAllGraphOptions();
  graph_options.erase(OPTION_FLOW_GRAPH_MEMORY_MAX_SIZE);
  EXPECT_EQ(model_deployer.GetRemoteGroupCacheConfig(remote_group_cache_config), SUCCESS);
  EXPECT_EQ(remote_group_cache_config, "");

  graph_options[OPTION_FLOW_GRAPH_MEMORY_MAX_SIZE] = "";
  GetThreadLocalContext().SetGraphOption(graph_options);
  EXPECT_NE(model_deployer.GetRemoteGroupCacheConfig(remote_group_cache_config), SUCCESS);

  std::string expect_config_value = "99999999";
  graph_options[OPTION_FLOW_GRAPH_MEMORY_MAX_SIZE] = expect_config_value;
  GetThreadLocalContext().SetGraphOption(graph_options);
  EXPECT_EQ(model_deployer.GetRemoteGroupCacheConfig(remote_group_cache_config), SUCCESS);
  EXPECT_EQ(remote_group_cache_config, expect_config_value);
  graph_options.clear();
  GetThreadLocalContext().SetGraphOption(graph_options);
}
}  // namespace ge
