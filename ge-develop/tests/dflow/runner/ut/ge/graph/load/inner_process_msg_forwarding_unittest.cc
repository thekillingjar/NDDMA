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
#include <numeric>

#include "macro_utils/dt_public_scope.h"
#include "common/blocking_queue.h"
#include "dflow/executor/inner_process_msg_forwarding.h"
#include "graph/load/model_manager/davinci_model.h"
#include "graph/load/model_manager/model_manager.h"
#include "graph/utils/tensor_utils.h"
#include "dflow/base/exec_runtime/execution_runtime.h"
#include "framework/common/runtime_tensor_desc.h"
#include "macro_utils/dt_public_unscope.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/utils/tensor_adapter.h"
#include "depends/runtime/src/runtime_stub.h"
#include "stub_models.h"

using namespace std;
using namespace testing;

namespace ge {
namespace {
Status DynamicSchedDequeueMbufStub(int32_t device_id, uint32_t queue_id,
                                   rtMbufPtr_t *m_buf, int32_t timeout) {
  if (queue_id == 100) { // status
    domi::SubmodelStatus submodel_status;
    auto queue_status = submodel_status.add_queue_statuses();
    domi::QueueAttrs *queue_attrs = queue_status->mutable_queue_attrs();
    queue_attrs->set_queue_id(0);
    queue_attrs->set_logic_id(0);
    queue_status->set_input_consume_num(10);
    queue_status->set_queue_depth(0);
    rtMbufPtr_t req_msg_mbuf = nullptr;
    void *input_buffer = nullptr;
    uint64_t input_buffer_size = 0;
    auto req_msg_mbuf_size = submodel_status.ByteSizeLong();
    EXPECT_EQ(rtMbufAlloc(&req_msg_mbuf, req_msg_mbuf_size), RT_ERROR_NONE);
    EXPECT_EQ(rtMbufSetDataLen(req_msg_mbuf, req_msg_mbuf_size), RT_ERROR_NONE);
    EXPECT_EQ(rtMbufGetBuffAddr(req_msg_mbuf, &input_buffer), RT_ERROR_NONE);
    EXPECT_EQ(rtMbufGetBuffSize(req_msg_mbuf, &input_buffer_size), RT_ERROR_NONE);
    EXPECT_EQ(submodel_status.SerializeToArray(input_buffer, static_cast<int32_t>(req_msg_mbuf_size)), true);
    *m_buf = req_msg_mbuf;
    return SUCCESS;
  } else if (queue_id == 102) { // exception
    domi::SubmodelStatus submodel_status;
    submodel_status.set_msg_type(1);
    auto *exception_info = submodel_status.mutable_exception();
    exception_info->set_exception_code(-1);
    exception_info->set_trans_id(11);
    exception_info->set_user_context_id(100);
    rtMbufPtr_t req_msg_mbuf = nullptr;
    void *input_buffer = nullptr;
    uint64_t input_buffer_size = 0;
    auto req_msg_mbuf_size = submodel_status.ByteSizeLong();
    EXPECT_EQ(rtMbufAlloc(&req_msg_mbuf, req_msg_mbuf_size), RT_ERROR_NONE);
    EXPECT_EQ(rtMbufSetDataLen(req_msg_mbuf, req_msg_mbuf_size), RT_ERROR_NONE);
    EXPECT_EQ(rtMbufGetBuffAddr(req_msg_mbuf, &input_buffer), RT_ERROR_NONE);
    EXPECT_EQ(rtMbufGetBuffSize(req_msg_mbuf, &input_buffer_size), RT_ERROR_NONE);
    EXPECT_EQ(submodel_status.SerializeToArray(input_buffer, static_cast<int32_t>(req_msg_mbuf_size)), true);
    *m_buf = req_msg_mbuf;
    return SUCCESS;
  }
  return SUCCESS;
}

class MockRuntimeFailed : public RuntimeStub {
 public:
  rtError_t rtMemQueueDeQueue(int32_t device, uint32_t qid, void **mbuf) override {
    return 0;
  }

  rtError_t rtMbufGetBuffAddr(rtMbufPtr_t mbuf, void **databuf) override {
    if (get_buffer_error) {
      return 100;
    }
    *databuf = data_;
    return 0;
  }

  rtError_t rtMbufGetBuffSize(rtMbufPtr_t mbuf, uint64_t *size) override {
    if (get_size_error) {
      return 100;
    }
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

  ~MockRuntimeFailed() {
    for (auto &mbuf : mem_bufs_) {
      RuntimeStub::rtMbufFree(mbuf);
    }
    mem_bufs_.clear();
  }

 public :
  bool get_buffer_error = false;
  bool get_size_error = false;
 private:
  std::mutex mu_;
  uint8_t data_[128] = {};
  std::vector<void *> mem_bufs_;
};
}
class InnerProcessMsgForwardingTest : public testing::Test {
 protected:
  class MockExchangeService : public ExchangeService {
   public:
    Status CreateQueue(int32_t device_id,
                       const string &name,
                       const MemQueueAttr &mem_queue_attr,
                       uint32_t &queue_id) override {
      queue_id = queue_id_gen_++;
      return SUCCESS;
    }
    MOCK_METHOD5(Enqueue, Status(int32_t, uint32_t, const void *, size_t, const ExchangeService::ControlInfo &));
    MOCK_METHOD5(Enqueue, Status(int32_t, uint32_t, size_t, const ExchangeService::FillFunc &, const ExchangeService::ControlInfo &));
    MOCK_METHOD5(Dequeue, Status(int32_t, uint32_t, void * , size_t, ExchangeService::ControlInfo &));
    MOCK_METHOD5(DequeueMbufTensor, Status(int32_t, uint32_t, std::shared_ptr<AlignedPtr> &, size_t, ExchangeService::ControlInfo &));
    MOCK_METHOD4(DequeueTensor, Status(int32_t, uint32_t, GeTensor & , ExchangeService::ControlInfo &));
    MOCK_METHOD4(DequeueMbuf, Status(int32_t, uint32_t, rtMbufPtr_t *, int32_t));

    void ResetQueueInfo(const int32_t device_id, const uint32_t queue_id) override {
      return;
    }

    Status Enqueue(int32_t device_id, uint32_t queue_id, size_t size, rtMbufPtr_t m_buf,
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
    Status DestroyQueue(int32_t device_id, uint32_t queue_id) override {
      return SUCCESS;
    }

    int queue_id_gen_ = 100;
  };

  class MockModelDeployer : public ModelDeployer {
   public:
    Status DeployModel(const FlowModelPtr &flow_model,
                       DeployResult &deploy_result) override {
      return SUCCESS;
    }
    Status Undeploy(uint32_t model_id) override {
      return SUCCESS;
    }
    Status GetDeviceMeshIndex(const int32_t device_id, std::vector<int32_t> &node_mesh_index) override {
      node_mesh_index = {0, 0, device_id, 0};
      return SUCCESS;
    }
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
    MockModelDeployer model_deployer_;
    MockExchangeService exchange_service_;
  };
  void SetUp() {
    ExecutionRuntime::SetExecutionRuntime(make_shared<MockExecutionRuntime>());
    status_messages_queue_.Restart();
  }

  void TearDown() {
    status_messages_queue_.Clear();
    status_messages_queue_.Stop();
    RuntimeStub::Reset();
  }
 public:
  BlockingQueue<domi::SubmodelStatus> status_messages_queue_;
};

TEST_F(InnerProcessMsgForwardingTest, InitAndFinalize) {
  InnerProcessMsgForwarding forwarding;
  EXPECT_EQ(forwarding.Initialize({}), FAILED);
  DeployQueueAttr queue_attr;
  queue_attr.queue_id = 1;
  auto &exchange_service = (MockExchangeService &) ExecutionRuntime::GetInstance()->GetExchangeService();
  EXPECT_CALL(exchange_service, DequeueMbuf).WillRepeatedly(testing::Invoke(DynamicSchedDequeueMbufStub));
  EXPECT_EQ(forwarding.Initialize({queue_attr}), SUCCESS);
  forwarding.Finalize();
}

TEST_F(InnerProcessMsgForwardingTest, RegAndRunSuccess) {
  InnerProcessMsgForwarding forwarding;
  DeployQueueAttr queue_attr;
  queue_attr.queue_id = 100;
  auto &exchange_service = (MockExchangeService &) ExecutionRuntime::GetInstance()->GetExchangeService();
  EXPECT_CALL(exchange_service, DequeueMbuf).WillRepeatedly(testing::Invoke(DynamicSchedDequeueMbufStub));
  EXPECT_EQ(forwarding.Initialize({queue_attr}), SUCCESS);
  forwarding.Start();
  auto func = [this](const domi::SubmodelStatus& request) {
    GE_CHK_BOOL_RET_STATUS(status_messages_queue_.Push(std::move(request)), INTERNAL_ERROR, "Failed to enqueue input");
    return SUCCESS;
  };
  forwarding.RegisterCallBackFunc(StatusQueueMsgType::STATUS, func);
  domi::SubmodelStatus submodel_status;
  while(true) {
    EXPECT_EQ(status_messages_queue_.Pop(submodel_status), true);
    forwarding.Finalize();
    status_messages_queue_.Clear();
    status_messages_queue_.Stop();
    break;
  }
  EXPECT_EQ(submodel_status.msg_type(), 0);
  EXPECT_EQ(submodel_status.queue_statuses().size(), 1);
  EXPECT_EQ(submodel_status.queue_statuses(0).input_consume_num(), 10);
}

TEST_F(InnerProcessMsgForwardingTest, RunRegNotFound) {
  InnerProcessMsgForwarding forwarding;
  DeployQueueAttr queue_attr;
  queue_attr.queue_id = 102;
  auto &exchange_service = (MockExchangeService &) ExecutionRuntime::GetInstance()->GetExchangeService();
  EXPECT_CALL(exchange_service, DequeueMbuf).WillRepeatedly(testing::Invoke(DynamicSchedDequeueMbufStub));
  EXPECT_EQ(forwarding.Initialize({queue_attr}), SUCCESS);
  forwarding.Start();
  auto func = [this](const domi::SubmodelStatus& request) {
    GE_CHK_BOOL_RET_STATUS(status_messages_queue_.Push(std::move(request)), INTERNAL_ERROR, "Failed to enqueue input");
    return SUCCESS;
  };
  EXPECT_EQ(forwarding.RegisterCallBackFunc(StatusQueueMsgType::STATUS, func), SUCCESS);
  EXPECT_EQ(forwarding.RegisterCallBackFunc(StatusQueueMsgType::STATUS, func), FAILED);
  usleep(2000);
  domi::SubmodelStatus submodel_status;
  EXPECT_EQ(status_messages_queue_.Size(), 0);
  forwarding.Finalize();
}

TEST_F(InnerProcessMsgForwardingTest, RunRegGetBufferError) {
  RuntimeStub::SetInstance(std::make_shared<MockRuntimeFailed>());
  (static_cast<MockRuntimeFailed *>(RuntimeStub::GetInstance()))->get_buffer_error = true;
  InnerProcessMsgForwarding forwarding;
  DeployQueueAttr queue_attr;
  queue_attr.queue_id = 1;
  auto &exchange_service = (MockExchangeService &) ExecutionRuntime::GetInstance()->GetExchangeService();
  EXPECT_CALL(exchange_service, DequeueMbuf).WillRepeatedly(testing::Invoke(DynamicSchedDequeueMbufStub));
  EXPECT_EQ(forwarding.Initialize({queue_attr}), SUCCESS);
  forwarding.Start();
  usleep(2000);
  (static_cast<MockRuntimeFailed *>(RuntimeStub::GetInstance()))->get_buffer_error = false;
  forwarding.Finalize();
}

TEST_F(InnerProcessMsgForwardingTest, RunRegGetSizeError) {
  RuntimeStub::SetInstance(std::make_shared<MockRuntimeFailed>());
  (static_cast<MockRuntimeFailed *>(RuntimeStub::GetInstance()))->get_size_error = true;
  InnerProcessMsgForwarding forwarding;
  DeployQueueAttr queue_attr;
  queue_attr.queue_id = 1;
  auto &exchange_service = (MockExchangeService &) ExecutionRuntime::GetInstance()->GetExchangeService();
  EXPECT_CALL(exchange_service, DequeueMbuf).WillRepeatedly(testing::Invoke(DynamicSchedDequeueMbufStub));
  EXPECT_EQ(forwarding.Initialize({queue_attr}), SUCCESS);
  forwarding.Start();
  usleep(2000);
  (static_cast<MockRuntimeFailed *>(RuntimeStub::GetInstance()))->get_size_error = false;
  forwarding.Finalize();
}

TEST_F(InnerProcessMsgForwardingTest, RunRegExecuteFuncError) {
  InnerProcessMsgForwarding forwarding;
  DeployQueueAttr queue_attr;
  queue_attr.queue_id = 102;
  auto &exchange_service = (MockExchangeService &) ExecutionRuntime::GetInstance()->GetExchangeService();
  EXPECT_CALL(exchange_service, DequeueMbuf).WillRepeatedly(testing::Invoke(DynamicSchedDequeueMbufStub));
  EXPECT_EQ(forwarding.Initialize({queue_attr}), SUCCESS);
  forwarding.Start();
  auto func = [this](const domi::SubmodelStatus& request) {
    status_messages_queue_.Stop();
    GE_CHK_BOOL_RET_STATUS(status_messages_queue_.Push(std::move(request)), INTERNAL_ERROR, "Failed to enqueue input");
    return SUCCESS;
  };
  forwarding.RegisterCallBackFunc(StatusQueueMsgType::EXCEPTION, func);
  usleep(2000);
  domi::SubmodelStatus submodel_status;
  EXPECT_EQ(status_messages_queue_.Size(), 0);
  forwarding.Finalize();
}
}  // namespace ge
