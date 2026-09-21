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
#include "common/subprocess/subprocess_manager.h"
#include "common/data_flow/queue/heterogeneous_exchange_service.h"
#include "common/config/configurations.h"
#include "deploy/execfwk/udf_executor_client.h"
#include "macro_utils/dt_public_unscope.h"
#include "deploy/flowrm/tsd_client.h"

namespace ge {
namespace {
class MockMmpaUdfClient : public ge::MmpaStubApiGe {
 public:
  int32_t RealPath(const CHAR *path, CHAR *realPath, INT32 realPathLen) override {
    memcpy_s(realPath, realPathLen, path, strlen(path));
    return 0;
  }

  void *DlSym(void *handle, const char *func_name) override {
    std::cout << "func name:" << func_name << " begin to stub\n";
    if (std::string(func_name) == "TsdFileLoad") {
      return (void *)&TsdFileLoad;
    } else if (std::string(func_name) == "TsdFileUnLoad") {
      return (void *) &TsdFileUnLoad;
    } else if (std::string(func_name) == "TsdGetProcListStatus") {
      return get_proc_status_func_;
    } else if (std::string(func_name) == "TsdProcessOpen") {
      return (void *) &TsdProcessOpen;
    } else if (std::string(func_name) == "ProcessCloseSubProcList") {
      return (void *) &ProcessCloseSubProcList;
    } else if (std::string(func_name) == "TsdCapabilityGet") {
      return get_tsd_capability_func_;
    }
    std::cout << "func name:" << func_name << " not stub\n";
    return (void *)0xFFFFFFFF;
  }

  void *DlOpen(const char *fileName, int32_t mode) override {
    return (void *)0xFFFFFFFF;
  }

  int32_t DlClose(void *handle) override {
    return 0L;
  }

  void *get_proc_status_func_ = (void *)&TsdGetProcListStatus;
  void *get_tsd_capability_func_ = (void *)&TsdCapabilityGet;
};

class MockExecutorMessageClient : public ExecutorMessageClient {
 public:
  MockExecutorMessageClient(bool response_flag = true) : ExecutorMessageClient(0), response_flag_(response_flag) {}

  Status WaitResponse(deployer::ExecutorResponse &response, int64_t timeout = -1) override {
    if (response_flag_) {
      response.set_error_code(0);
      response.set_error_message("success !!");
    } else {
      ExecutorMessageClient::get_stat_func_ = []() { return SUCCESS; };
      (void)ExecutorMessageClient::WaitResponse(response, 1);
      response.set_error_code(2);
      response.set_error_message("failed !!");
    }
    return SUCCESS;
  }
  bool response_flag_ = true;
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

 private:
  uint8_t data_[128] = {};
};

class MockRuntime2 : public RuntimeStub {
 public:
  rtError_t rtMemQueueDeQueue(int32_t device, uint32_t qid, void **mbuf) override {
    if (dequeue_count_ < 3000U) {
      ++dequeue_count_;
      return 207013;
    }
    return 3;
  }

 private:
  uint32_t dequeue_count_ = 0U;
};

class MockUdfExecutorClient : public UdfExecutorClient {
 public:
  explicit MockUdfExecutorClient(int32_t device_id) : UdfExecutorClient(device_id) {}
 protected:
  Status CreateAndInitMessageClient(const uint32_t model_id, const pid_t child_pid, const uint32_t req_msg_queue_id,
                                    const uint32_t rsp_msg_queue_id) {
    (void)req_msg_queue_id;
    (void)rsp_msg_queue_id;
    pid_to_message_client_[child_pid] = MakeUnique<MockExecutorMessageClient>();
    model_id_to_pids_[model_id].emplace_back(child_pid);
    return SUCCESS;
  }
};

class MockUdfExecutorClientError : public UdfExecutorClient {
 public:
  explicit MockUdfExecutorClientError(int32_t device_id) : UdfExecutorClient(device_id) {}
 protected:
  Status ForkChildProcess(const deployer::ExecutorRequest_LoadModelRequest &model_req,
      const std::string &file_path, const std::string &group_name,
      const SubProcessParams &params, pid_t &child_pid) {
    return FAILED;
  }
};
}  // namespace
class UdfExecutorClientTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    system(
        "mkdir ./ut_udf_client && cd ./ut_udf_client && echo test > ./udf_stub.so && "
        "tar -zcf ./release.tar.gz ./udf_stub.so");
    // clear other not finished process callback
    SubprocessManager::GetInstance().Finalize();
  }
  static void TearDownTestSuite() {
    system("rm -fr ut_udf_client");
  }
  void SetUp() override {
    MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpaUdfClient>());
    SubprocessManager::GetInstance().Initialize();
    auto mock_runtime = std::make_shared<MockRuntime>();
    RuntimeStub::SetInstance(mock_runtime);
    system("mkdir ./ut_udf_models");
  }
  void TearDown() override {
    RuntimeStub::Reset();
    SubprocessManager::GetInstance().Finalize();
    system("rm -fr ./ut_udf_models");
    MmpaStub::GetInstance().Reset();
    HeterogeneousExchangeService::GetInstance().Finalize();
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

TEST_F(UdfExecutorClientTest, InitAndFinalize) {
  UdfExecutorClient client(0);
  client.model_id_to_pids_[0].emplace_back(1111111111111);
  client.model_id_to_pids_[1].emplace_back(2222222222222);
  EXPECT_EQ(client.Initialize(), SUCCESS);
  EXPECT_EQ(client.Finalize(), SUCCESS);
  EXPECT_EQ(client.model_id_to_pids_.size(), 0U);
}

TEST_F(UdfExecutorClientTest, SyncVarManager) {
  UdfExecutorClient client(0);
  deployer::ExecutorRequest_SyncVarManageRequest sync_var_manage_desc;
  EXPECT_EQ(client.SyncVarManager(sync_var_manage_desc), SUCCESS);
}

TEST_F(UdfExecutorClientTest, LoadModel_model_file_not_exist) {
  MockUdfExecutorClient client(0);
  deployer::ExecutorRequest_BatchLoadModelMessage load_model_desc;
  auto model = load_model_desc.add_models();
  if (model == nullptr) {
    return;
  }
  model->set_model_path("");
  model->set_saved_model_file_path("");
  auto *input_desc = model->mutable_model_queues_attrs()->add_input_queues_attrs();
  input_desc->set_queue_id(0);
  input_desc->set_device_type(NPU);
  input_desc->set_device_id(0);
  auto *output_desc = model->mutable_model_queues_attrs()->add_output_queues_attrs();
  output_desc->set_queue_id(1);
  output_desc->set_device_type(NPU);
  output_desc->set_device_id(0);
  SubprocessManager::GetInstance().executable_paths_["UDF"] = "";
  EXPECT_NE(client.LoadModel(load_model_desc), SUCCESS);
}

TEST_F(UdfExecutorClientTest, LoadModel_no_model) {
  MockUdfExecutorClient client(0);
  deployer::ExecutorRequest_BatchLoadModelMessage load_model_desc;
  SubprocessManager::GetInstance().executable_paths_["UDF"] = "";
  EXPECT_NE(client.LoadModel(load_model_desc), SUCCESS);
}

TEST_F(UdfExecutorClientTest, LoadModel_success_host) {
  MockUdfExecutorClient client(0);
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
  input_desc->set_queue_id(0);
  input_desc->set_device_type(CPU);
  input_desc->set_device_id(0);
  auto *output_desc = model->mutable_model_queues_attrs()->add_output_queues_attrs();
  output_desc->set_queue_id(1);
  output_desc->set_device_type(CPU);
  output_desc->set_device_id(0);
  SubprocessManager::GetInstance().executable_paths_["UDF"] = "";
  EXPECT_EQ(client.LoadModel(load_model_desc), SUCCESS);
  EXPECT_EQ(client.LoadModel(load_model_desc), SUCCESS);
  EXPECT_EQ(client.GetSubProcStat(), ProcStatus::NORMAL);
  system("rm -f ./ut_udf_models/test.om");
}

TEST_F(UdfExecutorClientTest, LoadModel_success_host_failed) {
  MockUdfExecutorClientError client(0);
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
  input_desc->set_queue_id(0);
  input_desc->set_device_type(CPU);
  input_desc->set_device_id(0);
  auto *output_desc = model->mutable_model_queues_attrs()->add_output_queues_attrs();
  output_desc->set_queue_id(1);
  output_desc->set_device_type(CPU);
  output_desc->set_device_id(0);
  EXPECT_EQ(client.LoadModel(load_model_desc), FAILED);
  system("rm -f ./ut_udf_models/test.om");
}

TEST_F(UdfExecutorClientTest, LoadModel_success_host_heavy_load) {
  MockUdfExecutorClient client(0);
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
  input_desc->set_queue_id(0);
  input_desc->set_device_type(CPU);
  input_desc->set_device_id(0);
  auto *output_desc = model->mutable_model_queues_attrs()->add_output_queues_attrs();
  output_desc->set_queue_id(1);
  output_desc->set_device_type(CPU);
  output_desc->set_device_id(0);
  SubprocessManager::GetInstance().executable_paths_["UDF"] = "";
  EXPECT_EQ(client.LoadModel(load_model_desc), SUCCESS);
  EXPECT_EQ(client.GetSubProcStat(), ProcStatus::NORMAL);
  system("rm -f ./ut_udf_models/test.om");
}

TEST_F(UdfExecutorClientTest, LoadModel_success_device) {
  MockUdfExecutorClient client(0);
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
  input_desc->set_queue_id(0);
  input_desc->set_device_type(NPU);
  input_desc->set_device_id(0);
  auto *output_desc = model->mutable_model_queues_attrs()->add_output_queues_attrs();
  output_desc->set_queue_id(1);
  output_desc->set_device_type(NPU);
  output_desc->set_device_id(0);
  SubprocessManager::GetInstance().executable_paths_["UDF"] = "";
  EXPECT_EQ(client.LoadModel(load_model_desc), SUCCESS);
  EXPECT_EQ(client.LoadModel(load_model_desc), SUCCESS);
  EXPECT_EQ(client.GetSubProcStat(), ProcStatus::NORMAL);
  system("rm -f ./ut_udf_models/test.om");
}

TEST_F(UdfExecutorClientTest, LoadModel_success_device_npu_sched) {
  MockUdfExecutorClient client(0);
  deployer::ExecutorRequest_BatchLoadModelMessage load_model_desc;
  auto model = load_model_desc.add_models();
  if (model == nullptr) {
    return;
  }
  model->mutable_attrs()->operator[]("_npu_sched_model") = "1";
  udf::UdfModelDef model_def;
  CreateUdfModel(model_def, "udf");
  WriteProto(model_def, "./ut_udf_models/test.om");
  model->set_model_path("./ut_udf_models/test.om");
  model->set_saved_model_file_path("./ut_udf_models/test.om");
  auto *input_desc = model->mutable_model_queues_attrs()->add_input_queues_attrs();
  input_desc->set_queue_id(0);
  input_desc->set_device_type(NPU);
  input_desc->set_device_id(0);
  auto *output_desc = model->mutable_model_queues_attrs()->add_output_queues_attrs();
  output_desc->set_queue_id(1);
  output_desc->set_device_type(NPU);
  output_desc->set_device_id(0);
  SubprocessManager::GetInstance().executable_paths_["UDF"] = "";
  EXPECT_EQ(client.LoadModel(load_model_desc), SUCCESS);
  EXPECT_EQ(client.GetSubProcStat(), ProcStatus::NORMAL);
  system("rm -f ./ut_udf_models/test.om");
}

TEST_F(UdfExecutorClientTest, LoadModel_repeat_load_success) {
  MockUdfExecutorClient client(0);
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
  input_desc->set_queue_id(0);
  input_desc->set_device_type(NPU);
  input_desc->set_device_id(0);
  auto *output_desc = model->mutable_model_queues_attrs()->add_output_queues_attrs();
  output_desc->set_queue_id(1);
  output_desc->set_device_type(NPU);
  output_desc->set_device_id(0);
  SubprocessManager::GetInstance().executable_paths_["UDF"] = "";
  EXPECT_EQ(client.LoadModel(load_model_desc), SUCCESS);
  EXPECT_EQ(client.LoadModel(load_model_desc), SUCCESS);
  EXPECT_EQ(client.GetSubProcStat(), ProcStatus::NORMAL);
  system("rm -f ./ut_udf_models/test.om");
}

TEST_F(UdfExecutorClientTest, LoadModel_builtin_udf) {
  UdfExecutorClient client(0);
  deployer::ExecutorRequest_BatchLoadModelMessage load_model_desc;
  auto model = load_model_desc.add_models();
  if (model == nullptr) {
    return;
  }
  udf::UdfModelDef model_def;
  CreateUdfModel(model_def, "udf", true);
  WriteProto(model_def, "./ut_udf_models/test.om");
  model->set_model_path("./ut_udf_models/test.om");
  model->set_saved_model_file_path("./ut_udf_models/test.om");
  auto *input_desc = model->mutable_model_queues_attrs()->add_input_queues_attrs();
  input_desc->set_queue_id(0);
  input_desc->set_device_type(NPU);
  input_desc->set_device_id(0);
  auto *output_desc = model->mutable_model_queues_attrs()->add_output_queues_attrs();
  output_desc->set_queue_id(1);
  output_desc->set_device_type(NPU);
  output_desc->set_device_id(0);
  SubprocessManager::GetInstance().executable_paths_["UDF"] = "";
  EXPECT_EQ(client.LoadModel(load_model_desc), SUCCESS);
  EXPECT_EQ(client.LoadModel(load_model_desc), SUCCESS);
  EXPECT_EQ(client.GetSubProcStat(), ProcStatus::NORMAL);
  system("rm -f ./ut_udf_models/test.om");
}

TEST_F(UdfExecutorClientTest, batch_untar_success_with_all_builtin_udf) {
  std::vector<deployer::SubmodelDesc> model_descs;
  std::string base_dir = "./local_context_999/0/1/builtin.om";
  UdfExecutorClient client(0);
  EXPECT_EQ(client.PreProcess(model_descs, base_dir), SUCCESS);
  std::set<std::string> local_udf_saved_path = {};
  EXPECT_EQ(UdfExecutorClient::PreprocessUdfTarPackage(local_udf_saved_path, base_dir), SUCCESS);
  (void)system("rm -rf ./local_context_999");
}

TEST_F(UdfExecutorClientTest, batch_untar_success) {
 std::string untar_cmd = R"(
mkdir -p ./ut_udf_models/batch_untar_succes
cd ./ut_udf_models/batch_untar_succes
touch test1.om
touch test1.so
echo "hello" > test1.om
echo "world" > test1.so
tar -czvf test1_release.tar.gz test1.om test1.so
cd ../..
)";
  int32_t cmd_ret = system(untar_cmd.c_str());
  EXPECT_EQ(cmd_ret, 0);
  std::set<std::string> local_udf_saved_path = {"./ut_udf_models/batch_untar_succes/test1_release.tar.gz"};
  // all tar is 0 size
  EXPECT_EQ(UdfExecutorClient::PreprocessUdfTarPackage(local_udf_saved_path, "./ut_udf_models/batch_untar_result/"), SUCCESS);
  system("rm -rf ./ut_udf_models");
}

TEST_F(UdfExecutorClientTest, LoadModel_tar_file_write_success_by_multi_times) {
  class MockMmpaWrite : public MockMmpaUdfClient {
   public:
    mmSsize_t Write(INT32 fd, VOID *mm_buf, UINT32 mm_count) override {
      if (is_first_time_) {
        is_first_time_ = false;
        return write(fd, mm_buf, 1);
      } else {
        return write(fd, mm_buf, mm_count);
      }
    }

   private:
    bool is_first_time_ = true;
  };
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpaWrite>());
  MockUdfExecutorClient client(0);
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
  input_desc->set_queue_id(0);
  input_desc->set_device_type(NPU);
  input_desc->set_device_id(0);
  auto *output_desc = model->mutable_model_queues_attrs()->add_output_queues_attrs();
  output_desc->set_queue_id(1);
  output_desc->set_device_type(NPU);
  output_desc->set_device_id(0);
  SubprocessManager::GetInstance().executable_paths_["UDF"] = "";
  EXPECT_EQ(client.LoadModel(load_model_desc), SUCCESS);
  EXPECT_EQ(client.GetSubProcStat(), ProcStatus::NORMAL);
  system("rm -f ./ut_udf_models/test.om");
}

TEST_F(UdfExecutorClientTest, UnLoadModel) {
  UdfExecutorClient client(0);
  client.model_id_to_pids_[0].emplace_back(111111111111);
  EXPECT_EQ(client.UnloadModel(0), SUCCESS);
  EXPECT_EQ(client.model_id_to_pids_.size(), 0U);
}

TEST_F(UdfExecutorClientTest, GetSubProcStat) {
  UdfExecutorClient client(0);
  EXPECT_EQ(client.GetSubProcStat(), ProcStatus::NORMAL);
}

TEST_F(UdfExecutorClientTest, UpdateModelInsNameByPid) {
  UdfExecutorClient client(0);
  client.pid_to_model_id_[11] = 1;
  client.pid_to_model_id_[21] = 2;
  client.pid_to_model_instances_name_[11].push_back("model_1");
  client.pid_to_model_instances_name_[11].push_back("model_2");
  client.pid_to_model_instances_name_[21].push_back("model_3");
  client.UpdateModelInsNameByPid(11);
  EXPECT_EQ(client.abnormal_model_instances_name_.size(), 1);
  EXPECT_EQ(client.abnormal_model_instances_name_[1][0], "model_1");
  EXPECT_EQ(client.abnormal_model_instances_name_[1][1], "model_2");
}

TEST_F(UdfExecutorClientTest, GetAbnormalModelInsName) {
  UdfExecutorClient client(0);
  client.pid_to_model_id_[11] = 1;
  client.pid_to_model_instances_name_[11].push_back("model");
  client.abnormal_model_instances_name_[1].push_back("model_1");
  client.abnormal_model_instances_name_[1].push_back("model_2");
  client.abnormal_model_instances_name_[2].push_back("model2_0");
  std::map<uint32_t, std::vector<std::string>> abnormal_model_instances_name;
  abnormal_model_instances_name[1].push_back("model_0");
  client.GetAbnormalModelInsName(abnormal_model_instances_name);
  EXPECT_EQ(abnormal_model_instances_name[1][0], "model_0");
  EXPECT_EQ(abnormal_model_instances_name[1][1], "model_1");
  EXPECT_EQ(abnormal_model_instances_name[1][2], "model_2");
  EXPECT_EQ(abnormal_model_instances_name[2][0], "model2_0");
}

TEST_F(UdfExecutorClientTest, CheckDevPidStatusStartByFork) {
  UdfExecutorClient client(0);
  EXPECT_EQ(client.CheckDevPidStatus(0, 0), SUCCESS);
  client.sub_proc_stat_flag_[100] = ProcStatus::NORMAL;
  EXPECT_EQ(client.CheckDevPidStatus(0, 100), SUCCESS);
}

TEST_F(UdfExecutorClientTest, ForkProcess_with_args) {
  UdfExecutorClient client(0);
  DeviceMaintenanceClientCfg cfg;
  client.context_.dev_maintenance_cfg = &cfg;
  pid_t child_pid = 0;
  UdfExecutorClient::SubProcessParams id_params;
  id_params.is_builtin = false;
  id_params.requeset_queue_id = UINT32_MAX;
  id_params.response_queue_id = UINT32_MAX;
  id_params.npu_sched_model = "1";
  SubprocessManager::GetInstance().excpt_handle_callbacks_.clear();
  deployer::ExecutorRequest_LoadModelRequest model_req;
  Status ret =client.ForkChildProcess(model_req, "", "1", id_params, child_pid);
  for (auto callback : SubprocessManager::GetInstance().excpt_handle_callbacks_) {
    callback.second(ProcStatus::NORMAL);
  }
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UdfExecutorClientTest, LoadModel_dynamic_sched_repeat_load_success) {
  MockUdfExecutorClient client(0);
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
  input_desc->set_queue_id(0);
  input_desc->set_device_type(NPU);
  input_desc->set_device_id(0);
  auto *output_desc = model->mutable_model_queues_attrs()->add_output_queues_attrs();
  output_desc->set_queue_id(1);
  output_desc->set_device_type(NPU);
  output_desc->set_device_id(0);
  auto *input_status_desc = model->mutable_status_queues()->add_input_queues_attrs();
  input_status_desc->set_queue_id(1);
  input_status_desc->set_device_type(NPU);
  input_status_desc->set_device_id(0);
  auto *output_status_desc = model->mutable_status_queues()->add_output_queues_attrs();
  output_status_desc->set_queue_id(1);
  output_status_desc->set_device_type(NPU);
  output_status_desc->set_device_id(0);
  SubprocessManager::GetInstance().executable_paths_["UDF"] = "";
  EXPECT_EQ(client.LoadModel(load_model_desc), SUCCESS);
  EXPECT_EQ(client.LoadModel(load_model_desc), SUCCESS);
  EXPECT_EQ(client.GetSubProcStat(), ProcStatus::NORMAL);
  system("rm -f ./ut_udf_models/test.om");
}

TEST_F(UdfExecutorClientTest, Send_clear_data_request_failed_not_exist) {
  MockUdfExecutorClient client(0);
  EXPECT_EQ(client.ClearModelRunningData(0, 1, {}), FAILED);
  client.model_id_to_pids_[0].emplace_back(100);
  client.model_id_to_pids_[0].emplace_back(101);
  EXPECT_EQ(client.ClearModelRunningData(0, 1, {}), FAILED);
}

class MockRuntimeNoLeaks : public RuntimeStub {
 public:
  ~MockRuntimeNoLeaks() {
    // 用于释放SendRequestAsync中创建的Mbuf，调用rtMemQueueEnQueue后不会再访问Mbuf地址，可以释放
    // 此处打桩只在Send_clear_data_request、Send_clear_data_request_succ中使用
    for (auto &m_buf : mem_bufs_) {
      rtMbufFree(m_buf);
    }
    mem_bufs_.clear();
  }
  rtError_t rtMemQueueEnQueue(int32_t dev_id, uint32_t qid, void *mem_buf) override {
    std::lock_guard<std::mutex> lk(mu_);
    mem_bufs_.emplace_back(mem_buf);
    return RT_ERROR_NONE;
  }
 private:
  std::mutex mu_;
  std::vector<void *> mem_bufs_;
};

TEST_F(UdfExecutorClientTest, Send_clear_data_request) {
  MockUdfExecutorClient client(0);
  client.model_id_to_pids_[0].emplace_back(100);
  client.model_id_to_pids_[0].emplace_back(101);
  client.pid_to_message_client_[100] =  MakeUnique<MockExecutorMessageClient>();
  client.pid_to_message_client_[101] =  MakeUnique<MockExecutorMessageClient>(false);
  auto mock_runtime = std::make_shared<MockRuntimeNoLeaks>();
  RuntimeStub::SetInstance(mock_runtime);
  EXPECT_EQ(client.ClearModelRunningData(0, 1, {}), FAILED);
  RuntimeStub::Reset();
}

TEST_F(UdfExecutorClientTest, Send_notify_exception) {
  MockUdfExecutorClient client(0);
  client.model_id_to_pids_[0].emplace_back(100);
  client.model_id_to_pids_[0].emplace_back(101);
  client.pid_to_message_client_[100] =  MakeUnique<MockExecutorMessageClient>();
  client.pid_to_message_client_[101] =  MakeUnique<MockExecutorMessageClient>(false);
  auto mock_runtime = std::make_shared<MockRuntimeNoLeaks>();
  RuntimeStub::SetInstance(mock_runtime);
  deployer::DataFlowExceptionNotifyRequest req_body;
  req_body.set_root_model_id(0);
  EXPECT_NE(client.DataFlowExceptionNotify(req_body), SUCCESS);
  RuntimeStub::Reset();
}

TEST_F(UdfExecutorClientTest, Send_notify_exception_get_client_failed) {
  MockUdfExecutorClient client(0);
  client.model_id_to_pids_[0].emplace_back(100);
  client.model_id_to_pids_[0].emplace_back(101);
  deployer::DataFlowExceptionNotifyRequest req_body;
  req_body.set_root_model_id(999);
  EXPECT_NE(client.DataFlowExceptionNotify(req_body), SUCCESS);
  req_body.set_root_model_id(0);
  EXPECT_NE(client.DataFlowExceptionNotify(req_body), SUCCESS);
}

TEST_F(UdfExecutorClientTest, Send_clear_data_request_not_related_abnormal_devids) {
  MockUdfExecutorClient client(0);
  client.model_id_to_pids_[0].emplace_back(100);
  client.model_id_to_pids_[0].emplace_back(101);
  client.pid_to_message_client_[100] =  MakeUnique<MockExecutorMessageClient>();
  client.pid_to_message_client_[101] =  MakeUnique<MockExecutorMessageClient>();
  auto mock_runtime = std::make_shared<MockRuntimeNoLeaks>();
  RuntimeStub::SetInstance(mock_runtime);
  std::set<int32_t> device_ids = {2, 3};
  EXPECT_EQ(client.ClearModelRunningData(0, 1, device_ids), SUCCESS);
  RuntimeStub::Reset();
}

TEST_F(UdfExecutorClientTest, Send_clear_data_request_related_abnormal_devids) {
  MockUdfExecutorClient client(0);
  client.model_id_to_pids_[0].emplace_back(100);
  client.model_id_to_pids_[0].emplace_back(101);
  client.pid_to_message_client_[100] =  MakeUnique<MockExecutorMessageClient>();
  client.pid_to_message_client_[101] =  MakeUnique<MockExecutorMessageClient>();
  client.pid_to_message_client_[103] =  MakeUnique<MockExecutorMessageClient>();
  auto mock_runtime = std::make_shared<MockRuntimeNoLeaks>();
  client.npu_device_id_related_pids_[2].insert(100);
  client.npu_device_id_related_pids_[4].insert(103);
  client.npu_device_id_related_pids_[5].insert(101);
  RuntimeStub::SetInstance(mock_runtime);
  std::set<int32_t> device_ids = {2, 3};
  EXPECT_EQ(client.ClearModelRunningData(0, 1, device_ids), SUCCESS);
  RuntimeStub::Reset();
}

TEST_F(UdfExecutorClientTest, Wait_rsp_failed) {
  auto mock_runtime = std::make_shared<MockRuntime2>();
  RuntimeStub::SetInstance(mock_runtime);
  MockExecutorMessageClient handler(0);
  handler.pid_ = 101;
  handler.req_msg_queue_id_ = 100;
  handler.rsp_msg_queue_id_ = 101;
  handler.get_stat_func_ = []() -> Status {
    return SUCCESS;
  };
  deployer::ExecutorResponse executor_response;
  EXPECT_EQ(handler.WaitResponse(executor_response, 1), SUCCESS);
  RuntimeStub::Reset();
}

TEST_F(UdfExecutorClientTest, Bind_hostpid_success) {
  MockUdfExecutorClient client(0);
  EXPECT_EQ(client.DoBindHostPid(1234), SUCCESS);
}
}  // namespace ge
