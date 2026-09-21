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
#include "framework/common/debug/ge_log.h"
#include "hccl/hccl_types.h"
#include "dflow/inc/data_flow/model/pne_model.h"
#include "depends/mmpa/src/mmpa_stub.h"
#include "macro_utils/dt_public_scope.h"
#include "common/subprocess/subprocess_manager.h"
#include "common/data_flow/queue/heterogeneous_exchange_service.h"
#include "deploy/execfwk/builtin_executor_client.h"
#include "deploy/flowrm/tsd_client.h"

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

 private:
  uint8_t data_[128] = {};
};

class MockMmpa : public MmpaStubApiGe {
 public:
  INT32 WaitPid(mmProcess pid, INT32 *status, INT32 options) override {
    std::cout << "mock wait pid stub begin\n";
    auto ret = waitpid(pid, status, options);
    if (ret != 0) {
      // always wait success
      *status = 0;
      return ret;
    }
    return 0;
  }

  void *DlSym(void *handle, const char *func_name) override {
    if (std::string(func_name) == "TsdFileLoad") {
      return (void *) &TsdFileLoad;
    } else if (std::string(func_name) == "TsdFileUnLoad") {
      return (void *) &TsdFileUnLoad;
    } else if (std::string(func_name) == "TsdProcessOpen") {
      return (void *) &TsdProcessOpen;
    } else if (std::string(func_name) == "ProcessCloseSubProcList") {
      return (void *) &ProcessCloseSubProcList;
    }
    std::cout << "func name:" << func_name << " not stub\n";
    return (void *) 0xFFFFFFFF;
  }

  void *DlOpen(const char *file_name, int32_t mode) override {
    return (void *) 0xFFFFFFFF;
  }
  int32_t DlClose(void *handle) override {
    return 0L;
  }
};
}  // namespace
class BuiltinExecutorClientTest : public testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {
    RuntimeStub::Reset();
    MmpaStub::GetInstance().Reset();
    HeterogeneousExchangeService::GetInstance().Finalize();
  }
};

TEST_F(BuiltinExecutorClientTest, TestInitAndFinalize) {
  SubprocessManager::GetInstance().executable_paths_[PNE_ID_NPU] = "npu_executor";
  auto mock_runtime = std::make_shared<MockRuntime>();
  RuntimeStub::SetInstance(mock_runtime);
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());

  BuiltinExecutorClient client(0);
  EXPECT_EQ(client.Initialize(), SUCCESS);
  EXPECT_EQ(client.Finalize(), SUCCESS);
}

TEST_F(BuiltinExecutorClientTest, TestNpuOnHostInitAndFinalize) {
  SubprocessManager::GetInstance().executable_paths_[PNE_ID_NPU] = "npu_executor";
  auto mock_runtime = std::make_shared<MockRuntime>();
  RuntimeStub::SetInstance(mock_runtime);
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());

  PneExecutorClient::ClientContext context = {};
  context.device_id = 0;
  context.device_type = static_cast<int32_t>(NPU);
  BuiltinExecutorClient client(0);
  EXPECT_EQ(client.Initialize(), SUCCESS);
  EXPECT_EQ(client.Finalize(), SUCCESS);
}

TEST_F(BuiltinExecutorClientTest, BindHostSuccess) {
  BuiltinExecutorClient client(0, true);
  client.pid_ = 1;
  PneExecutorClient::ClientContext context;
  context.device_type = CPU;
  client.SetContext(context);
  deployer::ExecutorRequest_ModelQueuesAttrs model_queue_attrs;
  auto *const input_queue_def = model_queue_attrs.mutable_input_queues_attrs()->Add();
  input_queue_def->set_queue_id(0);
  input_queue_def->set_device_type(NPU);
  input_queue_def->set_device_id(1);
  int32_t use_queue_process_device_type = context.device_type;
  ASSERT_EQ(client.GrantQueuesForProcess(100, use_queue_process_device_type, model_queue_attrs), SUCCESS);
}

TEST_F(BuiltinExecutorClientTest, TestInitializeFailed) {
  auto executable_paths = std::move(SubprocessManager::GetInstance().executable_paths_);
  BuiltinExecutorClient client(0);
  EXPECT_EQ(client.Initialize(), FAILED);
  SubprocessManager::GetInstance().executable_paths_ = std::move(executable_paths);
}

TEST_F(BuiltinExecutorClientTest, TestExecutorProcStatusChange) {
  SubprocessManager::GetInstance().executable_paths_[PNE_ID_NPU] = "npu_executor";
  auto mock_runtime = std::make_shared<MockRuntime>();
  RuntimeStub::SetInstance(mock_runtime);
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());

  SubprocessManager::GetInstance().excpt_handle_callbacks_.clear();
  BuiltinExecutorClient client(0);
  EXPECT_EQ(client.Initialize(), SUCCESS);
  for (auto callback : SubprocessManager::GetInstance().excpt_handle_callbacks_) {
    callback.second(ProcStatus::NORMAL);
  }
  EXPECT_EQ(client.Finalize(), SUCCESS);
}

TEST_F(BuiltinExecutorClientTest, TestGenerateKvArgs) {
  BuiltinExecutorClient client(0);
  std::map<std::string, std::string> kv_args;
  PneExecutorClient::ClientContext context = {};
  context.options[OPTION_EXEC_PROFILING_MODE] = "1";
  context.options[OPTION_EXEC_PROFILING_OPTIONS] = "some_val";
  client.SetContext(context);
  EXPECT_EQ(client.GenerateKvArgs(kv_args), SUCCESS);
}

TEST_F(BuiltinExecutorClientTest, TestExceptionNotify) {
  uint32_t ret_code = 0;
  class MockExecutorMessageClient : public ExecutorMessageClient {
   public:
    MockExecutorMessageClient(std::function<uint32_t()> get_ret_code)
        : ExecutorMessageClient(0), get_ret_code_(get_ret_code) {}
    Status WaitResponse(deployer::ExecutorResponse &response, int64_t timeout = -1) override {
      response.set_error_code(get_ret_code_());
      return SUCCESS;
    }

   private:
    std::function<uint32_t()> get_ret_code_;
  };
  BuiltinExecutorClient client(0);
  client.message_client_.reset(new MockExecutorMessageClient([&ret_code]() { return ret_code; }));
  client.is_valid_ = true;
  deployer::DataFlowExceptionNotifyRequest req_body;
  req_body.set_root_model_id(0);
  EXPECT_EQ(client.DataFlowExceptionNotify(req_body), SUCCESS);
  ret_code = 999;
  EXPECT_NE(client.DataFlowExceptionNotify(req_body), SUCCESS);
}

TEST_F(BuiltinExecutorClientTest, TestUpdateProf) {
  uint32_t ret_code = 0;
  class MockExecutorMessageClient : public ExecutorMessageClient {
   public:
    MockExecutorMessageClient(std::function<uint32_t()> get_ret_code)
        : ExecutorMessageClient(0), get_ret_code_(get_ret_code) {}
    Status WaitResponse(deployer::ExecutorResponse &response, int64_t timeout = -1) override {
      response.set_error_code(get_ret_code_());
      return SUCCESS;
    }

   private:
    std::function<uint32_t()> get_ret_code_;
  };
  BuiltinExecutorClient client(0);
  client.message_client_.reset(new MockExecutorMessageClient([&ret_code]() { return ret_code; }));
  client.is_host_ = true;
  deployer::ExecutorRequest_UpdateProfRequest req_body;
  req_body.set_is_prof_start(1);
  req_body.set_prof_data("test");
  EXPECT_EQ(client.UpdateProfilingFromExecutor(req_body), SUCCESS);
}
}  // namespace ge

