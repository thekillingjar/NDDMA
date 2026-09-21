/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef D_AIR_TESTS_UT_GE_RUNTIME_HETEROGENEOUS_STUB_HETEROGENEOUS_STUB_ENV_H_
#define D_AIR_TESTS_UT_GE_RUNTIME_HETEROGENEOUS_STUB_HETEROGENEOUS_STUB_ENV_H_

#include <gmock/gmock.h>
#include <nlohmann/json.hpp>

#include "deploy/flowrm/flow_route_planner.h"
#include "deploy/deployer/master_model_deployer.h"

#include "dflow/base/exec_runtime/execution_runtime.h"

#include "deploy/execfwk/builtin_executor_client.h"

namespace ge {
namespace stub {
class MockExchangeService : public ExchangeService {
 public:
  Status CreateQueue(int32_t device_id,
                     const string &name,
                     const MemQueueAttr &mem_queue_attr,
                     uint32_t &queue_id) override {
    queue_id = queue_id_gen_++;
    return DoCreateQueue();
  }

  MOCK_METHOD0(DoCreateQueue, Status());
  MOCK_METHOD2(DestroyQueue, Status(int32_t, uint32_t));

  Status Enqueue(const int32_t device_id, const uint32_t queue_id, const void *const data, const size_t size,
                 const ControlInfo &control_info) override {
    return SUCCESS;
  }

  Status Enqueue(int32_t device_id, uint32_t queue_id, size_t size, rtMbufPtr_t m_buf,
                 const ControlInfo &control_info) override {
    return SUCCESS;
  }

  Status Enqueue(const int32_t device_id, const uint32_t queue_id, const size_t size, const FillFunc &fill_func,
                 const ControlInfo &control_info) override {
    return SUCCESS;
  }

  Status Enqueue(const int32_t device_id, const uint32_t queue_id, const std::vector<BuffInfo> &buffs,
                 const ControlInfo &control_info) override {
    return SUCCESS;
  }

  Status EnqueueMbuf(int32_t device_id, uint32_t queue_id, rtMbufPtr_t m_buf, int32_t timeout) override {
    return SUCCESS;
  }
  Status Dequeue(const int32_t device_id,
                 const uint32_t queue_id,
                 void *const data,
                 const size_t size,
                 ControlInfo &control_info) override {
    return SUCCESS;
  }

  Status DequeueMbufTensor(const int32_t device_id, const uint32_t queue_id, std::shared_ptr<AlignedPtr> &aligned_ptr,
                           const size_t size, ControlInfo &control_info) {
    return SUCCESS;
  }

  Status DequeueTensor(const int32_t device_id,
                       const uint32_t queue_id,
                       GeTensor &tensor,
                       ControlInfo &control_info) override {
    return 0;
  }
  Status DequeueMbuf(int32_t device_id, uint32_t queue_id, rtMbufPtr_t *m_buf, int32_t timeout) override {
    return 0;
  }
  void ResetQueueInfo(const int32_t device_id, const uint32_t queue_id) override {
    return;
  }

  int queue_id_gen_ = 100;
};

class MockExecutionRuntime : public ExecutionRuntime {
 public:
  Status Initialize(const map<std::string, std::string> &options) override {
    return SUCCESS;
  }
  Status Finalize() override {
    return SUCCESS;
  }
  ModelDeployer &GetModelDeployer() override {
    return model_deployer_;
  }
  ExchangeService &GetExchangeService() override {
    return exchange_service_;
  }

 private:
  MasterModelDeployer model_deployer_;
  MockExchangeService exchange_service_;
};

class MockRemoteDeployer : public RemoteDeployer {
 public:
  explicit MockRemoteDeployer(const NodeConfig &node_config) : RemoteDeployer(node_config) {}
  MOCK_METHOD2(Process, Status(deployer::DeployerRequest & , deployer::DeployerResponse & ));
};

class MockExecutorMessageClient : public ExecutorMessageClient {
 public:
  MockExecutorMessageClient() : ExecutorMessageClient(0) {}
  Status SendRequest(const deployer::ExecutorRequest &request, deployer::ExecutorResponse &resp, int64_t timeout) override {
    return SUCCESS;
  }
};

class MockPneExecutorClient : public BuiltinExecutorClient {
 public:
  explicit MockPneExecutorClient(int32_t device_id) : BuiltinExecutorClient(device_id) {}

 protected:
  Status ForAndInit(int32_t device_id, unique_ptr<ExecutorMessageClient> &executor_process) override {
    executor_process = MakeUnique<MockExecutorMessageClient>();
    return SUCCESS;
  }
};
}  // namespace

class HeterogeneousStubEnv {
 public:
  static void ClearEnv();
  static void SetupAIServerEnv();
  static void LoadAIServerHostConfig(const string &path);
};
}  // namespace ge

#endif //D_AIR_TESTS_UT_GE_RUNTIME_HETEROGENEOUS_STUB_HETEROGENEOUS_STUB_ENV_H_
