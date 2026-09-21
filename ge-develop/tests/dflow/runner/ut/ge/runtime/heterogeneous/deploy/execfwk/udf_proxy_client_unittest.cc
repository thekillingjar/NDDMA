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
#include <runtime_stub.h>
#include "depends/mmpa/src/mmpa_stub.h"
#include "fstream"
#include "common/util.h"
#include "framework/common/debug/ge_log.h"
#include "hccl/hccl_types.h"
#include "udf_def.pb.h"
#include "macro_utils/dt_public_scope.h"
#include "deploy/execfwk/udf_proxy_client.h"
#include "macro_utils/dt_public_unscope.h"
#include "deploy/flowrm/tsd_client.h"
#include "common/profiling/profiling_properties.h"
#include "common/config/device_debug_config.h"
#include "common/dump/dump_manager.h"

namespace ge {
namespace {
class MockMmpaUdfClient : public ge::MmpaStubApiGe {
 public:
  int32_t RealPath(const CHAR *path, CHAR *realPath, INT32 realPathLen) override {
    memcpy_s(realPath, realPathLen, path, strlen(path));
    return 0;
  }

  int32_t WaitPid(mmProcess pid, INT32 *status, INT32 options) override {
    return waitpid(pid, status, options);
  }

  void *DlSym(void *handle, const char *func_name) override {
    std::cout << "func name:" << func_name << " begin to stub\n";
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
uint32_t TsdGetProcStatusExited(const uint32_t device_id, ProcStatusParam *status, uint32_t num) {
  if (num >= 1) {
    status[num - 1].curStat = SUB_PROCESS_STATUS_EXITED;
  }
  return 0U;
}

uint32_t TsdGetProcStatusFailed(const uint32_t device_id, ProcStatusParam *status, uint32_t num) {
  return 100U;
}

class MockMmpaSubProcStatFailed : public MockMmpaUdfClient {
 public:
  void *DlSym(void *handle, const char *func_name) override {
    std::cout << "func name:" << func_name << " begin to stub\n";
    if (std::string(func_name) == "TsdFileLoad") {
      return (void *)&TsdFileLoad;
    } else if (std::string(func_name) == "TsdFileUnLoad") {
      return (void *)&TsdFileUnLoad;
    } else if (std::string(func_name) == "TsdGetProcListStatus") {
      return get_proc_status_func_;
    } else if (std::string(func_name) == "TsdProcessOpen") {
      return (void *)&TsdProcessOpen;
    } else if (std::string(func_name) == "ProcessCloseSubProcList") {
      return (void *)&ProcessCloseSubProcList;
    } else if (std::string(func_name) == "TsdCapabilityGet") {
      return (void *)&TsdCapabilityGet;
    }
    std::cout << "func name:" << func_name << " not stub\n";
    return (void *)0xFFFFFFFF;
  }
  void *get_proc_status_func_ = (void *)&TsdGetProcStatusFailed;
};
class MockExecutorMessageClient : public ExecutorMessageClient {
 public:
  MockExecutorMessageClient() : ExecutorMessageClient(0) {}
  Status SendRequest(const deployer::ExecutorRequest &request, deployer::ExecutorResponse &resp, int64_t timeout) override {
    return SUCCESS;
  }
};

class MockUdfProxyClient : public UdfProxyClient {
 public:
  explicit MockUdfProxyClient(int32_t device_id) : UdfProxyClient(device_id) {}
 private:
  Status CreateAndInitMessageClient(const uint32_t model_id, const pid_t child_pid, const uint32_t req_msg_queue_id,
                                    const uint32_t rsp_msg_queue_id) {
    (void)req_msg_queue_id;
    (void)rsp_msg_queue_id;
    pid_to_message_client_[child_pid] = MakeUnique<MockExecutorMessageClient>();
    model_id_to_pids_[model_id].emplace_back(child_pid);
    return SUCCESS;
  }
};
}  // namespace
class UdfProxyClientTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    system(
        "mkdir ./ut_udf_client && cd ./ut_udf_client && echo test > ./udf_stub.so && "
        "tar -zcf ./release.tar.gz ./udf_stub.so");
  }
  static void TearDownTestSuite() {
    system("rm -fr ut_udf_client");
  }
  void SetUp() override {
    MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpaUdfClient>());
    TsdClient::GetInstance().Initialize();
    system("mkdir ./ut_udf_models");
  }
  void TearDown() override {
    TsdClient::GetInstance().Finalize();
    system("rm -fr ./ut_udf_models");
    MmpaStub::GetInstance().Reset();
    RuntimeStub::Reset();
  }

  void WriteProto(google::protobuf::MessageLite &protoMsg, const std::string &path) {
    std::ofstream outStream;
    outStream.open(path.c_str(), std::ios::out | std::ios::trunc | std::ios::binary);
    protoMsg.SerializeToOstream(&outStream);
    outStream.close();
  }

  void CreateUdfModel(udf::UdfModelDef &model_def, const std::string &flowFuncName, bool is_built_in = false) {
    auto udfDef = model_def.add_udf_def();
    udfDef->set_name(flowFuncName);
    udfDef->set_bin("so bin");
    udfDef->set_func_name(flowFuncName);
    if (!is_built_in) {
      udfDef->set_bin_name(flowFuncName + ".so");
    } else {
      udfDef->set_bin_name("libbuilt_in_flowfunc.so");
    }
  }
};

TEST_F(UdfProxyClientTest, InitAndFinalize) {
  UdfProxyClient client(0);
  EXPECT_EQ(client.Initialize(), SUCCESS);
  EXPECT_EQ(client.Finalize(), SUCCESS);
}

TEST_F(UdfProxyClientTest, GetSubProcStat) {
  UdfProxyClient client(0);
  client.model_id_to_pids_ = {{1, {9001, 9002}}};
  EXPECT_EQ(client.GetSubProcStat(), ProcStatus::NORMAL);
}

TEST_F(UdfProxyClientTest, GetSubProcStatExited) {
  UdfProxyClient client(0);
  client.model_id_to_pids_ = {{1, {9001, 9002}}};
  auto stub = std::make_shared<MockMmpaSubProcStatFailed>();
  stub->get_proc_status_func_ = (void *)TsdGetProcStatusExited;
  MmpaStub::GetInstance().SetImpl(stub);
  EXPECT_EQ(client.GetSubProcStat(), ProcStatus::EXITED);
}

TEST_F(UdfProxyClientTest, GetSubProcStatFailed) {
  UdfProxyClient client(0);
  client.model_id_to_pids_ = {{1, {9001, 9002}}};
  auto stub = std::make_shared<MockMmpaSubProcStatFailed>();
  stub->get_proc_status_func_ = (void *)TsdGetProcStatusFailed;
  MmpaStub::GetInstance().SetImpl(stub);
  EXPECT_EQ(client.GetSubProcStat(), ProcStatus::EXITED);
}

TEST_F(UdfProxyClientTest, builtin_udf) {
  UdfProxyClient client(0);
  deployer::ExecutorRequest_BatchLoadModelMessage load_model_desc;
  auto model = load_model_desc.add_models();
  if (model == nullptr) {
    return;
  }
  udf::UdfModelDef model_def;
  CreateUdfModel(model_def, "udf", true);
  WriteProto(model_def, "./ut_udf_models/test.om");
  model->set_model_path("./ut_udf_models/test.om");
  model->set_saved_model_file_path("xxx");
  model->set_is_builtin_udf(true);
  auto *input_desc = model->mutable_model_queues_attrs()->add_input_queues_attrs();
  input_desc->set_device_id(0);
  input_desc->set_device_type(NPU);
  input_desc->set_device_id(0);
  auto *output_desc = model->mutable_model_queues_attrs()->add_output_queues_attrs();
  output_desc->set_device_id(1);
  output_desc->set_device_type(NPU);
  output_desc->set_device_id(0);
  EXPECT_EQ(client.LoadModel(load_model_desc), SUCCESS);
  EXPECT_EQ(client.UnloadModel(0), SUCCESS);
  EXPECT_EQ(client.GetSubProcStat(), ProcStatus::NORMAL);
  system("rm -f ./ut_udf_models/test.om");
}

TEST_F(UdfProxyClientTest, LoadModelAndUnload_success) {
  MockUdfProxyClient client(0);
  deployer::ExecutorRequest_BatchLoadModelMessage load_model_desc;
  auto model = load_model_desc.add_models();
  if (model == nullptr) {
    return;
  }
  udf::UdfModelDef model_def;
  CreateUdfModel(model_def, "udf");
  WriteProto(model_def, "./ut_udf_models/test.om");
  model->set_model_path("./ut_udf_models/test.om");
  model->set_saved_model_file_path("./ut_udf_models/test.om");
  auto *input_desc = model->mutable_model_queues_attrs()->add_input_queues_attrs();
  input_desc->set_device_id(0);
  input_desc->set_device_type(NPU);
  input_desc->set_device_id(0);
  auto *output_desc = model->mutable_model_queues_attrs()->add_output_queues_attrs();
  output_desc->set_device_id(1);
  output_desc->set_device_type(NPU);
  output_desc->set_device_id(0);
  EXPECT_EQ(client.LoadModel(load_model_desc), SUCCESS);
  EXPECT_EQ(client.LoadModel(load_model_desc), SUCCESS);
  EXPECT_EQ(client.UnloadModel(0), SUCCESS);
  EXPECT_EQ(client.GetSubProcStat(), ProcStatus::NORMAL);
  system("rm -f ./ut_udf_models/test.om");
}

TEST_F(UdfProxyClientTest, UnLoadModel) {
  MockUdfProxyClient client(0);
  client.model_id_to_pids_[0].emplace_back(111111111111);
  EXPECT_EQ(client.UnloadModel(0), SUCCESS);
  EXPECT_EQ(client.model_id_to_pids_.size(), 0U);
}

TEST_F(UdfProxyClientTest, ForkProcess_with_args) {
  UdfProxyClient client(0);
  // init profiling config
  DeviceMaintenanceMasterCfg dev_cfg;
  ProfilingProperties::Instance().SetExecuteProfiling(true);
  ge::DumpProperties dump_properties;
  dump_properties.SetDumpStep("0|2-5|10");
  ge::DumpManager::GetInstance().AddDumpProperties(ge::kInferSessionId, dump_properties);
  dev_cfg.InitDevMaintenanceConfigs();
  std::string data;
  auto ret = dev_cfg.GetJsonDataByType(DeviceDebugConfig::ConfigType::kProfilingConfigType, data);
  ret = dev_cfg.GetJsonDataByType(DeviceDebugConfig::ConfigType::kDumpConfigType, data);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(data.size() > 0, true);
  // load json data
  DeviceMaintenanceClientCfg dev_client_cfg;
  ret = dev_client_cfg.LoadJsonData(data);
  EXPECT_EQ(ret, SUCCESS);
  client.context_.dev_maintenance_cfg = &dev_client_cfg;
  pid_t child_pid = 0;
  SubprocessManager::GetInstance().excpt_handle_callbacks_.clear();
  deployer::ExecutorRequest_LoadModelRequest model_req;
  UdfExecutorClient::SubProcessParams id_params;
  id_params.is_builtin = false;
  id_params.requeset_queue_id = UINT32_MAX;
  id_params.response_queue_id = UINT32_MAX;
  ret = client.ForkChildProcess(model_req, "", "1", id_params, child_pid);
  for (auto callback : SubprocessManager::GetInstance().excpt_handle_callbacks_) {
    callback.second(ProcStatus::NORMAL);
  }
  EXPECT_EQ(ret, SUCCESS);
  ge::DumpManager::GetInstance().RemoveDumpProperties(ge::kInferSessionId);
}
}  // namespace ge
