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
#include "dflow/executor/heterogeneous_model_executor.h"
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
#include "acl/acl.h"

using namespace std;
using namespace testing;

namespace ge {
namespace {
Status DevStatCallBackFail() {
  return FAILED;
}
} // namespace

class HeterogeneousModelExecutorTest : public testing::Test {
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
    MOCK_METHOD4(EnqueueMbuf, Status(int32_t device_id, uint32_t queue_id, rtMbufPtr_t m_buf, int32_t timeout));

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

  void SetUp() override {
    default_root_model_ = StubModels::BuildRootModel(StubModels::BuildGraphWithoutNeedForBindingQueues());
    default_executor_ = DefaultExecutor();
    default_executor_->SetModelId(666);
    ExecutionRuntime::SetExecutionRuntime(make_shared<MockExecutionRuntime>());
  }

  void TearDown() override {
    RuntimeStub::Reset();
  }

  unique_ptr<HeterogeneousModelExecutor> DefaultExecutor() {
    DeployResult deploy_result;
    deploy_result.model_id = 777;
    deploy_result.input_queue_attrs = {{1, 0, 0}, {2, 0, 0}};
    deploy_result.broadcast_input_queue_attrs= {{{1, 0, 0}}};
    deploy_result.output_queue_attrs = {{5, 0, 0}};
    deploy_result.input_model_name = default_root_model_->GetModelName();
    deploy_result.dev_abnormal_callback = []() -> Status { return SUCCESS; };
    auto flow_model = std::make_shared<FlowModel>();
    flow_model->AddSubModel(default_root_model_);
    return unique_ptr<HeterogeneousModelExecutor>(new HeterogeneousModelExecutor(flow_model, deploy_result));
  }

  PneModelPtr default_root_model_;
  unique_ptr<HeterogeneousModelExecutor> default_executor_;
};

MATCHER_P(Enqueue3VoidMatcher, data, "") {
  return true;
}

MATCHER_P(Enqueue3SizeMatcher, size, "") {
  return true;
}

namespace {
Status DequeueNoTilingStub(int32_t device_id, uint32_t queue_id, void *data, size_t size,
                           ExchangeService::ControlInfo &control_info) {
  RuntimeTensorDesc mbuf_tensor_desc;
  mbuf_tensor_desc.shape[0] = 4;
  mbuf_tensor_desc.shape[1] = 1;
  mbuf_tensor_desc.shape[2] = 1;
  mbuf_tensor_desc.shape[3] = 224;
  mbuf_tensor_desc.shape[4] = 224;
  mbuf_tensor_desc.dtype = static_cast<int64_t>(DT_INT64);
  mbuf_tensor_desc.data_addr = static_cast<int64_t>(reinterpret_cast<intptr_t>(data));
  if (memcpy_s(data, sizeof(RuntimeTensorDesc), &mbuf_tensor_desc, sizeof(RuntimeTensorDesc)) != EOK) {
    printf("Failed to copy mbuf data, dst addr:%p dst size:%zu, src size:%zu\n", data, size, sizeof(RuntimeTensorDesc));
    return FAILED;
  }
  return 0;
}

Status DequeueNoTilingStubWithEOS(int32_t device_id, uint32_t queue_id, GeTensor &tensor,
                                  ExchangeService::ControlInfo &control_info) {
  control_info.end_of_sequence_flag = true;
  return 0;
}

Status DequeueTensorFailed(int32_t device_id, uint32_t queue_id, GeTensor &tensor,
                                  ExchangeService::ControlInfo &control_info) {
  return ACL_ERROR_RT_QUEUE_EMPTY;
}
}

/**
 *     NetOutput
 *         |
 *       PC_3
 *      /   \
 *    PC_1  PC_2
 *    |      |
 *  data1  data2
 */
TEST_F(HeterogeneousModelExecutorTest, TestInitializeSuccess) {
  ASSERT_EQ(default_executor_->input_queue_attrs_[0].queue_id, 1);
  ASSERT_EQ(default_executor_->input_queue_attrs_[1].queue_id, 2);
  ASSERT_EQ(default_executor_->output_queue_attrs_[0].queue_id, 5);
  ASSERT_EQ(default_executor_->GetDeployedModelId(), 777);
  ASSERT_EQ(default_executor_->model_id_, 666);

  ASSERT_EQ(default_executor_->Initialize(), SUCCESS);
  ASSERT_EQ(default_executor_->input_tensor_desc_.size(), 2);
  ASSERT_EQ(default_executor_->output_tensor_desc_.size(), 1);
  ASSERT_EQ(default_executor_->input_tensor_sizes_.size(), 2);
  ASSERT_EQ(default_executor_->output_tensor_sizes_.size(), 1);
}

TEST_F(HeterogeneousModelExecutorTest, TestInitializeWithSingleModel) {
  auto model = StubModels::BuildRootModel(StubModels::BuildGraphWithoutNeedForBindingQueues());
  auto &submodels = model->GetSubmodels();
  auto it = submodels.find("subgraph-1");
  auto single_model = it->second;
  DeployResult deploy_result = {};
  deploy_result.model_id = 777;
  deploy_result.input_queue_attrs = {{1, 0, 0}};
  deploy_result.output_queue_attrs = {{2, 0, 0}};
  auto flow_model = std::make_shared<FlowModel>();
  flow_model->AddSubModel(model);
  HeterogeneousModelExecutor executor(flow_model, deploy_result);
  ASSERT_EQ(executor.Initialize(), SUCCESS);
  ASSERT_EQ(executor.flow_model_->GetSubmodels().size(), 1);
}

TEST_F(HeterogeneousModelExecutorTest, TestBuildTensorDescMappingFailed) {
  auto &model_relation = *default_root_model_->model_relation_;
  std::map<std::string, GeTensorDescPtr> mapping;

  default_executor_->model_relation_ = default_root_model_->model_relation_;

  // test input queue size mismatches
  model_relation.submodel_endpoint_infos["subgraph-1"].input_endpoint_names.emplace_back("queue_0");
  ASSERT_EQ(default_executor_->BuildInputTensorDescMapping(mapping), PARAM_INVALID);

  // test output queue size mismatches
  model_relation.submodel_endpoint_infos["subgraph-1"].output_endpoint_names.clear();
  ASSERT_EQ(default_executor_->BuildOutputTensorDescMapping(mapping), PARAM_INVALID);
}

TEST_F(HeterogeneousModelExecutorTest, TestSetTensorInfoFailed) {
  std::map<std::string, GeTensorDescPtr> input_mapping;
  GeShape ge_shape{vector<int64_t>({1, 2, 3, 4})};
  GeTensorDescPtr input = std::make_shared<GeTensorDesc>();
  input->SetShape(ge_shape);
  input->SetFormat(FORMAT_NCHW);
  input->SetDataType(DT_FLOAT);
  input_mapping["input"] = input;
  TensorUtils::SetSize(*input, -1);

  DeployResult deploy_result = {};
  deploy_result.model_id = 777;
  deploy_result.input_queue_attrs = {{1, 0, 0}};
  deploy_result.output_queue_attrs = {{2, 0 ,0}};
  auto flow_model = std::make_shared<FlowModel>();
  HeterogeneousModelExecutor executor(flow_model, deploy_result);
  ASSERT_EQ(executor.SetTensorInfo(input_mapping, {"input"}, true), UNSUPPORTED);
}

TEST_F(HeterogeneousModelExecutorTest, TestExecuteModelSuccess) {
  ASSERT_EQ(default_executor_->Initialize(), SUCCESS);
  std::vector<GeTensor> input_tensors;
  for (size_t i = 0; i < default_executor_->input_queue_attrs_.size(); ++i) {
    auto tensor_desc = default_executor_->input_tensor_desc_[i];
    auto tensor_size = default_executor_->input_tensor_sizes_[i];
    GeTensor tensor(*tensor_desc, std::vector<uint8_t>(tensor_size));
    input_tensors.emplace_back(tensor);
  }

  auto &exchange_service = (MockExchangeService &) ExecutionRuntime::GetInstance()->GetExchangeService();
  void *data = nullptr;
  EXPECT_CALL(exchange_service, Enqueue(_, _, Matcher<const void*>(Enqueue3VoidMatcher(data)), _, _))
      .WillRepeatedly(Return(SUCCESS));
  EXPECT_CALL(exchange_service, Dequeue).WillRepeatedly(Return(SUCCESS));
  std::vector<GeTensor> output_tensors;
  ASSERT_EQ(default_executor_->Execute(input_tensors, output_tensors), SUCCESS);
}

TEST_F(HeterogeneousModelExecutorTest, TestExecuteModelFailed) {
  ASSERT_EQ(default_executor_->Initialize(), SUCCESS);
  std::vector<GeTensor> input_tensors;
  for (size_t i = 0; i < default_executor_->input_queue_attrs_.size(); ++i) {
    auto tensor_desc = default_executor_->input_tensor_desc_[i];
    auto tensor_size = default_executor_->input_tensor_sizes_[i];
    GeTensor tensor(*tensor_desc, std::vector<uint8_t>(tensor_size));
    input_tensors.emplace_back(tensor);
  }

  std::map<std::string, std::string> options_info;
  options_info.insert(std::pair<std::string, std::string>("ge.exec.graphExecTimeout", "2000"));
  ge::GetThreadLocalContext().SetGraphOption(options_info);

  auto &exchange_service = (MockExchangeService &) ExecutionRuntime::GetInstance()->GetExchangeService();
  void *data = nullptr;
  EXPECT_CALL(exchange_service, Enqueue(_, _, Matcher<const void*>(Enqueue3VoidMatcher(data)), _, _))
      .WillRepeatedly(Return(SUCCESS));
  EXPECT_CALL(exchange_service, Dequeue).WillRepeatedly(Return((Status)12345U));
  std::vector<GeTensor> output_tensors;
  ASSERT_EQ(default_executor_->Execute(input_tensors, output_tensors), (Status)12345U);
  // check callback state return failed
  default_executor_->dev_abnormal_callback_ = DevStatCallBackFail;
  EXPECT_CALL(exchange_service, Dequeue).WillRepeatedly(Return((Status)RT_ERROR_TO_GE_STATUS(ACL_ERROR_RT_QUEUE_EMPTY)));
  ASSERT_NE(default_executor_->Execute(input_tensors, output_tensors), SUCCESS);
}

TEST_F(HeterogeneousModelExecutorTest, TestExecuteModelNoTilingSuccess) {
  ASSERT_EQ(default_executor_->Initialize(), SUCCESS);
  std::vector<GeTensor> input_tensors;
  for (size_t i = 0; i < default_executor_->input_queue_attrs_.size(); ++i) {
    auto tensor_desc = default_executor_->input_tensor_desc_[i];
    auto tensor_size = default_executor_->input_tensor_sizes_[i];
    default_executor_->input_is_no_tiling_[i] = true;
    GeTensor tensor(*tensor_desc, std::vector<uint8_t>(tensor_size));
    input_tensors.emplace_back(tensor);
  }

  for (size_t i = 0; i < default_executor_->output_queue_attrs_.size(); ++i) {
    default_executor_->output_is_no_tiling_[i] = true;
    default_executor_->output_tensor_sizes_[i] = 1 * 1 * 224 * 224 * 10;
  }

  auto &exchange_service = (MockExchangeService &) ExecutionRuntime::GetInstance()->GetExchangeService();
  void *data = nullptr;
  EXPECT_CALL(exchange_service, Enqueue(_, _, Matcher<const void*>(Enqueue3VoidMatcher(data)), _, _))
      .WillRepeatedly(Return(SUCCESS));
  EXPECT_CALL(exchange_service, Dequeue).WillRepeatedly(testing::Invoke(DequeueNoTilingStub));
  EXPECT_CALL(exchange_service, DequeueTensor).WillRepeatedly(testing::Return(SUCCESS));
  std::vector<GeTensor> output_tensors;
  ASSERT_EQ(default_executor_->Execute(input_tensors, output_tensors), SUCCESS);
}

TEST_F(HeterogeneousModelExecutorTest, TestInputSizeMismatches) {
  ASSERT_EQ(default_executor_->Initialize(), SUCCESS);
  std::vector<GeTensor> input_tensors;
  ASSERT_EQ(default_executor_->EnqueueInputTensors(input_tensors), PARAM_INVALID);
}

TEST_F(HeterogeneousModelExecutorTest, TestEnqueueFailed) {
  ASSERT_EQ(default_executor_->Initialize(), SUCCESS);
  default_executor_->control_input_queue_attrs_ = {{6, 0, 0}};
  std::vector<GeTensor> input_tensors;
  for (size_t i = 0; i < default_executor_->input_queue_attrs_.size(); ++i) {
    auto tensor_desc = default_executor_->input_tensor_desc_[i];
    auto tensor_size = default_executor_->input_tensor_sizes_[i];
    GeTensor tensor(*tensor_desc, std::vector<uint8_t>(tensor_size));
    input_tensors.emplace_back(tensor);
  }

  auto &exchange_service = (MockExchangeService &) ExecutionRuntime::GetInstance()->GetExchangeService();
  void *data = nullptr;
  EXPECT_CALL(exchange_service, Enqueue(_, _, Matcher<const void*>(Enqueue3VoidMatcher(data)), _, _))
      .WillRepeatedly(Return(RT_FAILED));
  ASSERT_EQ(default_executor_->EnqueueInputTensors(input_tensors), RT_FAILED);
}

TEST_F(HeterogeneousModelExecutorTest, TestExecuteModelWithEOS) {
  ASSERT_EQ(default_executor_->Initialize(), SUCCESS);
  std::vector<GeTensor> input_tensors;
  for (size_t i = 0; i < default_executor_->input_queue_attrs_.size(); ++i) {
  auto tensor_desc = default_executor_->input_tensor_desc_[i];
  auto tensor_size = default_executor_->input_tensor_sizes_[i];
  default_executor_->input_is_no_tiling_[i] = true;
  GeTensor tensor(*tensor_desc, std::vector<uint8_t>(tensor_size));
  input_tensors.emplace_back(tensor);
  }

  for (size_t i = 0; i < default_executor_->output_queue_attrs_.size(); ++i) {
  default_executor_->output_is_no_tiling_[i] = true;
  default_executor_->output_tensor_sizes_[i] = 1 * 1 * 224 * 224 * 10;
  }

  auto &exchange_service = (MockExchangeService &) ExecutionRuntime::GetInstance()->GetExchangeService();
  void *data = nullptr;
  EXPECT_CALL(exchange_service, Enqueue(_, _, Matcher<const void*>(Enqueue3VoidMatcher(data)), _, _))
      .WillRepeatedly(Return(SUCCESS));
  EXPECT_CALL(exchange_service, DequeueTensor).WillRepeatedly(testing::Invoke(DequeueNoTilingStubWithEOS));
  std::vector<GeTensor> output_tensors;
  ASSERT_EQ(default_executor_->Execute(input_tensors, output_tensors), END_OF_SEQUENCE);
}

TEST_F(HeterogeneousModelExecutorTest, TestFeedDataSuccess) {
  default_executor_->contains_n_mapping_node_ = true;
  ASSERT_EQ(default_executor_->Initialize(), SUCCESS);
  std::vector<GeTensor> input_tensors;
  for (size_t i = 0; i < default_executor_->input_queue_attrs_.size(); ++i) {
    auto tensor_desc = default_executor_->input_tensor_desc_[i];
    auto tensor_size = default_executor_->input_tensor_sizes_[i];
    default_executor_->input_is_no_tiling_[i] = true;
    GeTensor tensor(*tensor_desc, std::vector<uint8_t>(tensor_size));
    input_tensors.emplace_back(tensor);
  }

  auto &exchange_service = (MockExchangeService &) ExecutionRuntime::GetInstance()->GetExchangeService();
  void *data = nullptr;
  EXPECT_CALL(exchange_service, Enqueue(_, _, Matcher<const void*>(Enqueue3VoidMatcher(data)), _, _))
      .WillRepeatedly(Return(SUCCESS));
  size_t size = 0;
  EXPECT_CALL(exchange_service, Enqueue(_, _, Matcher<size_t>(Enqueue3SizeMatcher(size)), _, _))
      .WillRepeatedly(Return(SUCCESS));

  DataFlowInfo info;
  info.SetStartTime(1UL);
  info.SetEndTime(2UL);
  info.SetFlowFlags(0U);
  info.SetTransactionId(1000);

  auto original_handle = ExecutionRuntime::handle_;
  ExecutionRuntime::handle_ = (void*)0x12345678;
  auto options_bk = GetThreadLocalContext().GetAllSessionOptions();
  auto options = options_bk;
  options.emplace(RESOURCE_CONFIG_PATH, "xxx");
  GetThreadLocalContext().SetSessionOption(options);
  GE_MAKE_GUARD(recover, [options_bk]() { GetThreadLocalContext().SetSessionOption(options_bk); });

  default_executor_->is_fusion_input_[1] = true;
  default_executor_->input_queue_attrs_[0].queue_id = UINT32_MAX;
  Status ret = default_executor_->FeedData({}, input_tensors, info, 1000);
  ASSERT_EQ(ret, SUCCESS);
  info.SetFlowFlags(1U);
  // empty input used for feed eos
  ret = default_executor_->FeedData({}, {}, info, 1000);
  ExecutionRuntime::handle_ = original_handle;
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(HeterogeneousModelExecutorTest, TestFeedRawDataSuccess) {
  ASSERT_EQ(default_executor_->Initialize(), SUCCESS);
  std::vector<GeTensor> input_tensors;
  for (size_t i = 0; i < default_executor_->input_queue_attrs_.size(); ++i) {
    auto tensor_desc = default_executor_->input_tensor_desc_[i];
    auto tensor_size = default_executor_->input_tensor_sizes_[i];
    default_executor_->input_is_no_tiling_[i] = true;
    GeTensor tensor(*tensor_desc, std::vector<uint8_t>(tensor_size));
    input_tensors.emplace_back(tensor);
  }

  auto &exchange_service = (MockExchangeService &) ExecutionRuntime::GetInstance()->GetExchangeService();
  void *data = nullptr;
  EXPECT_CALL(exchange_service, Enqueue(_, _, Matcher<const void*>(Enqueue3VoidMatcher(data)), _, _))
      .WillRepeatedly(Return(SUCCESS));
  DataFlowInfo info;
  info.SetStartTime(1UL);
  info.SetEndTime(2UL);
  info.SetFlowFlags(0U);
  info.SetTransactionId(1024);

  auto original_handle = ExecutionRuntime::handle_;
  ExecutionRuntime::handle_ = (void*)0x12345678;
  auto options_bk = GetThreadLocalContext().GetAllSessionOptions();
  auto options = options_bk;
  options.emplace(RESOURCE_CONFIG_PATH, "xxx");
  GetThreadLocalContext().SetSessionOption(options);
  GE_MAKE_GUARD(recover, [options_bk]() { GetThreadLocalContext().SetSessionOption(options_bk); });

  std::vector<uint32_t> feed_indexes;
  feed_indexes.resize(input_tensors.size());
  // genertate feed indexes from 0
  std::iota(feed_indexes.begin(), feed_indexes.end(), 0U);
  default_executor_->is_fusion_input_[1] = true;
  default_executor_->input_queue_attrs_[0].queue_id = UINT32_MAX;
  Status ret = default_executor_->FeedRawData({}, 0, info, 1000);
  ASSERT_EQ(ret, FAILED);
  uint64_t sample_data = 100;
  RawData raw_data = {.addr = reinterpret_cast<void *>(&sample_data), .len = sizeof(uint64_t)};
  ret = default_executor_->FeedRawData({raw_data}, 0, info, 1000);
  ASSERT_EQ(ret, SUCCESS);
  ret = default_executor_->FeedRawData({raw_data}, 1, info, 1000);
  ASSERT_EQ(ret, SUCCESS);
  info.SetFlowFlags(1U);
  ret = default_executor_->FeedData({}, {}, info, 1000);
  ExecutionRuntime::handle_ = original_handle;
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(HeterogeneousModelExecutorTest, TestFeedDataWithSingleInputSuccess) {
  auto model = StubModels::BuildRootModel(StubModels::BuildGraphWithoutNeedForBindingQueues());
  auto &submodels = model->GetSubmodels();
  auto it = submodels.find("subgraph-1");
  auto single_model = it->second;
  DeployResult deploy_result = {};
  deploy_result.model_id = 777;
  deploy_result.input_queue_attrs = {{1, 0, 0}};
  deploy_result.output_queue_attrs = {{2, 0, 0}};
  auto flow_model = std::make_shared<FlowModel>();
  flow_model->AddSubModel(model);
  HeterogeneousModelExecutor executor(flow_model, deploy_result);
  ASSERT_EQ(executor.Initialize(), SUCCESS);

  std::vector<GeTensor> input_tensors;
  for (size_t i = 0; i < executor.input_queue_attrs_.size(); ++i) {
    auto tensor_desc = executor.input_tensor_desc_[i];
    auto tensor_size = executor.input_tensor_sizes_[i];
    GeTensor tensor(*tensor_desc, std::vector<uint8_t>(tensor_size));
    input_tensors.emplace_back(tensor);
  }

  auto &exchange_service = (MockExchangeService &) ExecutionRuntime::GetInstance()->GetExchangeService();
  void *data = nullptr;
  EXPECT_CALL(exchange_service, Enqueue(_, _, Matcher<const void*>(Enqueue3VoidMatcher(data)), _, _))
      .WillRepeatedly(Return(SUCCESS));
  DataFlowInfo info;
  info.SetStartTime(1UL);
  info.SetEndTime(2UL);
  info.SetFlowFlags(0U);

  auto original_handle = ExecutionRuntime::handle_;
  ExecutionRuntime::handle_ = (void*)0x12345678;
  auto options_bk = GetThreadLocalContext().GetAllSessionOptions();
  auto options = options_bk;
  options.emplace(RESOURCE_CONFIG_PATH, "xxx");
  GetThreadLocalContext().SetSessionOption(options);
  GE_MAKE_GUARD(recover, [options_bk]() { GetThreadLocalContext().SetSessionOption(options_bk); });
  std::vector<uint32_t> feed_indexes;
  feed_indexes.resize(input_tensors.size());
  // genertate feed indexes from 0
  std::iota(feed_indexes.begin(), feed_indexes.end(), 0U);
  Status ret = executor.FeedData(feed_indexes, input_tensors, info, 1000);
  ASSERT_EQ(ret, SUCCESS);
  ExecutionRuntime::handle_ = original_handle;
}

TEST_F(HeterogeneousModelExecutorTest, TestFetchDataSameElement) {
  ASSERT_EQ(default_executor_->Initialize(), SUCCESS);
  for (size_t i = 0; i < default_executor_->output_queue_attrs_.size(); ++i) {
    default_executor_->output_is_no_tiling_[i] = true;
    default_executor_->output_tensor_sizes_[i] = 1 * 1 * 224 * 224 * 10;
  }

  auto original_handle = ExecutionRuntime::handle_;
  ExecutionRuntime::handle_ = (void*)0x12345678;
  auto options_bk = GetThreadLocalContext().GetAllSessionOptions();
  auto options = options_bk;
  options.emplace(RESOURCE_CONFIG_PATH, "xxx");
  GetThreadLocalContext().SetSessionOption(options);
  GE_MAKE_GUARD(recover, [options_bk]() { GetThreadLocalContext().SetSessionOption(options_bk); });
  DataFlowInfo info;
  std::vector<GeTensor> outputs;
  Status ret = default_executor_->FetchData({0, 0}, outputs, info, 1000U);
  ExecutionRuntime::handle_ = original_handle;
  ASSERT_NE(ret, SUCCESS);
}

TEST_F(HeterogeneousModelExecutorTest, TestFetchDataSuccess) {
  ASSERT_EQ(default_executor_->Initialize(), SUCCESS);
  for (size_t i = 0; i < default_executor_->output_queue_attrs_.size(); ++i) {
    default_executor_->output_is_no_tiling_[i] = true;
    default_executor_->output_tensor_sizes_[i] = 1 * 1 * 224 * 224 * 10;
  }
  auto &exchange_service = (MockExchangeService &)ExecutionRuntime::GetInstance()->GetExchangeService();
  EXPECT_CALL(exchange_service, Dequeue).WillRepeatedly(Return(SUCCESS));
  EXPECT_CALL(exchange_service, DequeueTensor).WillRepeatedly(testing::Return(SUCCESS));

  DataFlowInfo info;
  info.SetStartTime(1UL);
  info.SetEndTime(2UL);
  info.SetFlowFlags(0U);
  std::vector<GeTensor> outputs;
  auto original_handle = ExecutionRuntime::handle_;
  ExecutionRuntime::handle_ = (void*)0x12345678;
  auto options_bk = GetThreadLocalContext().GetAllSessionOptions();
  auto options = options_bk;
  options.emplace(RESOURCE_CONFIG_PATH, "xxx");
  GetThreadLocalContext().SetSessionOption(options);
  GE_MAKE_GUARD(recover, [options_bk]() { GetThreadLocalContext().SetSessionOption(options_bk); });
  Status ret = default_executor_->FetchData({}, outputs, info, 1000U);
  ExecutionRuntime::handle_ = original_handle;
  ASSERT_EQ(ret, SUCCESS);
  EXPECT_EQ(outputs.size(), default_executor_->output_queue_attrs_.size());
}

TEST_F(HeterogeneousModelExecutorTest, TestFetchDataEOS) {
  ASSERT_EQ(default_executor_->Initialize(), SUCCESS);
  for (size_t i = 0; i < default_executor_->output_queue_attrs_.size(); ++i) {
    default_executor_->output_is_no_tiling_[i] = true;
    default_executor_->output_tensor_sizes_[i] = 1 * 1 * 224 * 224 * 10;
  }
  auto &exchange_service = (MockExchangeService &)ExecutionRuntime::GetInstance()->GetExchangeService();
  EXPECT_CALL(exchange_service, DequeueTensor).WillRepeatedly(testing::Invoke(DequeueNoTilingStubWithEOS));

  DataFlowInfo info;
  info.SetStartTime(1UL);
  info.SetEndTime(2UL);
  info.SetFlowFlags(0U);
  std::vector<GeTensor> outputs;
  auto original_handle = ExecutionRuntime::handle_;
  ExecutionRuntime::handle_ = (void *)0x12345678;
  auto options_bk = GetThreadLocalContext().GetAllSessionOptions();
  auto options = options_bk;
  options.emplace(RESOURCE_CONFIG_PATH, "xxx");
  GetThreadLocalContext().SetSessionOption(options);
  GE_MAKE_GUARD(recover, [options_bk]() { GetThreadLocalContext().SetSessionOption(options_bk); });
  Status ret = default_executor_->FetchData({}, outputs, info, 1000U);
  ExecutionRuntime::handle_ = original_handle;
  ASSERT_EQ(ret, END_OF_SEQUENCE);
  EXPECT_EQ(outputs.size(), 0UL);
}

TEST_F(HeterogeneousModelExecutorTest, TestTensorsExecuteFusion) {
  default_executor_->input_queue_attrs_ = {{1, 0, 0}, {1, 0, 0}}; // queue ids are same;
  ASSERT_EQ(default_executor_->Initialize(), SUCCESS);
  std::vector<GeTensor> input_tensors;
  for (size_t i = 0; i < default_executor_->input_queue_attrs_.size(); ++i) {
    auto tensor_desc = default_executor_->input_tensor_desc_[i];
    auto tensor_size = default_executor_->input_tensor_sizes_[i];
    GeTensor tensor(*tensor_desc, std::vector<uint8_t>(tensor_size));
    input_tensors.emplace_back(tensor);
  }

  auto &exchange_service = (MockExchangeService &) ExecutionRuntime::GetInstance()->GetExchangeService();
  void *data = nullptr;
  EXPECT_CALL(exchange_service, Enqueue(_, _, Matcher<const void*>(Enqueue3VoidMatcher(data)), _, _))
      .WillRepeatedly(Return(SUCCESS));
  size_t size = 0;
  EXPECT_CALL(exchange_service, Enqueue(_, _, Matcher<size_t>(Enqueue3SizeMatcher(size)), _, _))
      .WillRepeatedly(Return(SUCCESS));
  EXPECT_CALL(exchange_service, Dequeue).WillRepeatedly(Return(SUCCESS));
  std::vector<GeTensor> output_tensors;
  ASSERT_EQ(default_executor_->Execute(input_tensors, output_tensors), SUCCESS);
}

TEST_F(HeterogeneousModelExecutorTest, BuildInputTensorDescMapping_DataCtrLinkToOut) {
  DEF_GRAPH(flow_graph) {
    auto data0 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0);
    auto data1 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 1);
    auto node0 = OP_CFG("Neg").InCnt(1).OutCnt(1);
    auto netoutput = OP_CFG("NetOutput").InCnt(1).OutCnt(1);
    CHAIN(NODE("data0", data0)->EDGE(0, 0)->NODE("node0", node0)->NODE("netoutput", netoutput));
    CTRL_CHAIN(NODE("data1", data1)->NODE("netoutput", netoutput));
  };

  auto root_graph = ToComputeGraph(flow_graph);

  auto flow_model = std::make_shared<FlowModel>();
  auto model_relation = std::make_shared<ModelRelation>();
  model_relation->endpoints.emplace_back("data0", EndpointType::kDummyQueue);
  model_relation->endpoints.emplace_back("data1", EndpointType::kDummyQueue);
  model_relation->endpoints.emplace_back("node0", EndpointType::kQueue);
  model_relation->root_model_endpoint_info.input_endpoint_names = {"data0", "data1"};
  flow_model->SetModelRelation(model_relation);
  flow_model->SetRootGraph(root_graph);

  DeployResult deploy_result;
  deploy_result.model_id = 777;
  deploy_result.input_queue_attrs = {{1, 0, 0}, {2, 0, 0}};

  HeterogeneousModelExecutor executor(flow_model, deploy_result);
  executor.model_relation_ = model_relation;
  std::map<std::string, GeTensorDescPtr> mapping;
  EXPECT_EQ(executor.BuildInputTensorDescMapping(mapping), SUCCESS);
  ASSERT_EQ(mapping.size(), 2U);
}

uint32_t g_sched_cnt;
Status DynamicSchedDequeueMbufStub(int32_t device_id, uint32_t queue_id,
                                   rtMbufPtr_t *m_buf, int32_t timeout) {
  if (g_sched_cnt > 0 && queue_id == 105) {
    domi::SubmodelStatus submodel_status;
    auto queue_status = submodel_status.add_queue_statuses();
    domi::QueueAttrs *queue_attrs = queue_status->mutable_queue_attrs();
    queue_attrs->set_queue_id(0);
    queue_attrs->set_logic_id(0);
    queue_status->set_input_consume_num(0);
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
  } else if (queue_id == 102) {
    domi::FlowgwRequest flowgw_request;
    flowgw_request.set_input_index(102);
    auto queue_infos = flowgw_request.add_queue_infos();
    domi::QueueAttrs *queue_attrs = queue_infos->mutable_queue_attrs();
    queue_attrs->set_queue_id(1);
    queue_attrs->set_logic_id(1);
    queue_infos->set_model_uuid(1);
    queue_infos->set_logic_group_id(1);
    queue_infos->set_trans_id(0);
    queue_infos->set_route_label(0);
    queue_infos->set_choose_logic_id(0);
    rtMbufPtr_t req_msg_mbuf = nullptr;
    void *input_buffer = nullptr;
    uint64_t input_buffer_size = 0;
    auto req_msg_mbuf_size = flowgw_request.ByteSizeLong();
    EXPECT_EQ(rtMbufAlloc(&req_msg_mbuf, req_msg_mbuf_size), RT_ERROR_NONE);
    EXPECT_EQ(rtMbufSetDataLen(req_msg_mbuf, req_msg_mbuf_size), RT_ERROR_NONE);
    EXPECT_EQ(rtMbufGetBuffAddr(req_msg_mbuf, &input_buffer), RT_ERROR_NONE);
    EXPECT_EQ(rtMbufGetBuffSize(req_msg_mbuf, &input_buffer_size), RT_ERROR_NONE);
    EXPECT_EQ(flowgw_request.SerializeToArray(input_buffer, static_cast<int32_t>(req_msg_mbuf_size)), true);
    *m_buf = req_msg_mbuf;
    return SUCCESS;
  }
  return RT_ERROR_TO_GE_STATUS(ACL_ERROR_RT_QUEUE_EMPTY);
}

bool g_dynamic_sched_by_cache = false;
Status DynamicSchedEnqueueStub(int32_t device_id, uint32_t queue_id, size_t size, const ExchangeService::FillFunc &fill_func,
                   const ExchangeService::ControlInfo &control_info) {
  if (queue_id == 101) {
    rtMbufPtr_t req_msg_mbuf = nullptr;
    void *input_buffer = nullptr;
    EXPECT_EQ(rtMbufAlloc(&req_msg_mbuf, size), RT_ERROR_NONE);
    EXPECT_EQ(rtMbufSetDataLen(req_msg_mbuf, size), RT_ERROR_NONE);
    EXPECT_EQ(rtMbufGetBuffAddr(req_msg_mbuf, &input_buffer), RT_ERROR_NONE);
    EXPECT_EQ(fill_func(input_buffer, size), SUCCESS);
    google::protobuf::io::ArrayInputStream stream(input_buffer, static_cast<int32_t>(size));
    domi::FlowgwResponse flowgw_response;
    EXPECT_EQ(flowgw_response.ParseFromZeroCopyStream(&stream), true);
    g_sched_cnt++;
    rtMbufFree(req_msg_mbuf);
  }
  return SUCCESS;
}

TEST_F(HeterogeneousModelExecutorTest, TestDynamicSchedFindGroupIndexByDefaultStatus) {
  default_executor_->is_dynamic_sched_ = true;
  DeployQueueAttr queue_attr;
  queue_attr.queue_id = 105;
  default_executor_->status_input_queue_attrs_.push_back(queue_attr);
  queue_attr.queue_id = 101;
  queue_attr.global_logic_id = 1;
  default_executor_->sched_input_queue_attrs_.push_back(queue_attr);
  queue_attr.queue_id = 102;
  default_executor_->sched_output_queue_attrs_.push_back(queue_attr);

  DeployPlan::DeviceInfo device_info =
      DeployPlan::DeviceInfo(static_cast<int32_t>(CPU), 0, 0);
  DeployPlan::ExtendedIndexInfo index_info;
  index_info.device_info = device_info;
  index_info.submodel_instance_name = "model1";
  index_info.is_normal = true;
  DeployPlan::DynamicGroupRouteInfo route1 = {0, 0, index_info, false};
  DeployPlan::DynamicGroupRouteInfo route2 = {1, 1, index_info, false};
  DeployPlan::DynamicGroupRouteInfo route3 = {2, 2, index_info, false};
  DeployPlan::DstGroupInfo group_info {3, {route1, route2, route3}};
  default_executor_->model_index_info_ = {{1, {{1,
      {{index_info}, {{1, group_info}}}
  }}}};
  default_executor_->datagw_request_bindings_ = {{1, 102}};
  for (size_t i = 1; i <= 1025; ++i) {
    default_executor_->cached_trans_ids_[i] = {0};
  }
  default_executor_->routelabel_cache_info_ = {
    {{1, 0}, {{3, std::make_pair(1, "")}}},
    {{7, 8}, {{9, std::make_pair(10, "")}}}
  };
  auto &exchange_service = (MockExchangeService &) ExecutionRuntime::GetInstance()->GetExchangeService();
  EXPECT_CALL(exchange_service, DequeueMbuf).WillRepeatedly(testing::Invoke(DynamicSchedDequeueMbufStub));
  ASSERT_EQ(default_executor_->Initialize(), SUCCESS);
  g_sched_cnt = 0;

  EXPECT_CALL(exchange_service, Dequeue).WillRepeatedly(Return(SUCCESS));
  size_t size = 0;
  EXPECT_CALL(exchange_service, Enqueue(_, _, Matcher<size_t>(Enqueue3SizeMatcher(size)), _, _))
      .WillRepeatedly(testing::Invoke(DynamicSchedEnqueueStub));

  ASSERT_EQ(default_executor_->ModelRunStart(), SUCCESS);

  // wait sched
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  ASSERT_EQ(default_executor_->ModelRunStop(), SUCCESS);
  ASSERT_EQ(default_executor_->ModelRunStop(), SUCCESS);  // not started
  EXPECT_EQ(default_executor_->cached_trans_ids_.size(), 1024);
  default_executor_->is_dynamic_sched_ = false;
  default_executor_->status_input_queue_attrs_.clear();
  default_executor_->sched_input_queue_attrs_.clear();
  default_executor_->sched_output_queue_attrs_.clear();
  default_executor_->model_index_info_.clear();
  default_executor_->datagw_request_bindings_.clear();
  default_executor_->cached_trans_ids_.clear();
}

TEST_F(HeterogeneousModelExecutorTest, TestDynamicSchedFindGroupIndexBySched) {
  default_executor_->is_dynamic_sched_ = true;
  DeployQueueAttr queue_attr;
  queue_attr.queue_id = 105;
  default_executor_->status_input_queue_attrs_.push_back(queue_attr);
  queue_attr.queue_id = 101;
  queue_attr.global_logic_id = 1;
  default_executor_->sched_input_queue_attrs_.push_back(queue_attr);
  queue_attr.queue_id = 102;
  default_executor_->sched_output_queue_attrs_.push_back(queue_attr);

  DeployPlan::DeviceInfo device_info =
      DeployPlan::DeviceInfo(static_cast<int32_t>(CPU), 0, 0);
  DeployPlan::ExtendedIndexInfo index_info;
  index_info.device_info = device_info;
  index_info.submodel_instance_name = "model1";
  index_info.is_normal = true;
  DeployPlan::DynamicGroupRouteInfo route1 = {0, 0, index_info, false};
  DeployPlan::DynamicGroupRouteInfo route2 = {1, 1, index_info, false};
  DeployPlan::DynamicGroupRouteInfo route3 = {2, 2, index_info, false};
  DeployPlan::DstGroupInfo group_info {1, {route1, route2, route3}};
  default_executor_->model_index_info_ = {{1, {{1,
      {{index_info}, {{1, group_info}}}
  }}}};
  default_executor_->datagw_request_bindings_ = {{1, 102}};
  for (size_t i = 1; i <= 1025; ++i) {
    default_executor_->cached_trans_ids_[i] = {0};
  }
  HeterogeneousModelExecutor::QueueStatus status;
  status.queue_depth = 0;
  status.device_id = 0;
  status.device_type = 0;
  default_executor_->queue_status_info_[1].first = status;
  default_executor_->queue_status_info_[1].second = 0;
  auto &exchange_service = (MockExchangeService &) ExecutionRuntime::GetInstance()->GetExchangeService();
  EXPECT_CALL(exchange_service, DequeueMbuf).WillRepeatedly(testing::Invoke(DynamicSchedDequeueMbufStub));
  ASSERT_EQ(default_executor_->Initialize(), SUCCESS);
  g_sched_cnt = 0;

  EXPECT_CALL(exchange_service, Dequeue).WillRepeatedly(Return(SUCCESS));
  size_t size = 0;
  EXPECT_CALL(exchange_service, Enqueue(_, _, Matcher<size_t>(Enqueue3SizeMatcher(size)), _, _))
      .WillRepeatedly(testing::Invoke(DynamicSchedEnqueueStub));

  ASSERT_EQ(default_executor_->ModelRunStart(), SUCCESS);

  // wait sched
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  ASSERT_EQ(default_executor_->ModelRunStop(), SUCCESS);
  ASSERT_EQ(default_executor_->ModelRunStop(), SUCCESS);  // not started
  default_executor_->is_dynamic_sched_ = false;
  default_executor_->status_input_queue_attrs_.clear();
  default_executor_->sched_input_queue_attrs_.clear();
  default_executor_->sched_output_queue_attrs_.clear();
  default_executor_->model_index_info_.clear();
  default_executor_->datagw_request_bindings_.clear();
  default_executor_->cached_trans_ids_.clear();
}

TEST_F(HeterogeneousModelExecutorTest, TestDynamicSchedFindGroupIndexInSingleInstance) {
  default_executor_->is_dynamic_sched_ = true;
  DeployQueueAttr queue_attr;
  queue_attr.queue_id = 105;
  default_executor_->status_input_queue_attrs_.push_back(queue_attr);
  queue_attr.queue_id = 101;
  queue_attr.global_logic_id = 1;
  default_executor_->sched_input_queue_attrs_.push_back(queue_attr);
  queue_attr.queue_id = 102;
  default_executor_->sched_output_queue_attrs_.push_back(queue_attr);

  DeployPlan::DeviceInfo device_info =
      DeployPlan::DeviceInfo(static_cast<int32_t>(CPU), 0, 0);
  DeployPlan::ExtendedIndexInfo index_info;
  index_info.device_info = device_info;
  index_info.submodel_instance_name = "model1";
  index_info.is_normal = true;
  DeployPlan::DynamicGroupRouteInfo route1 = {0, 0, index_info, false};
  DeployPlan::DynamicGroupRouteInfo route2 = {1, 1, index_info, true};
  DeployPlan::DynamicGroupRouteInfo route3 = {2, 2, index_info, true};
  DeployPlan::DstGroupInfo group_info {1, {route1, route2, route3}};
  default_executor_->model_index_info_ = {{1, {{1,
      {{index_info}, {{1, group_info}}}
  }}}};
  default_executor_->datagw_request_bindings_ = {{1, 102}};
  for (size_t i = 1; i <= 1025; ++i) {
    default_executor_->cached_trans_ids_[i] = {0};
  }
  HeterogeneousModelExecutor::QueueStatus status;
  status.queue_depth = 0;
  status.device_id = 0;
  status.device_type = 0;
  default_executor_->queue_status_info_[1].first = status;
  default_executor_->queue_status_info_[1].second = 0;
  auto &exchange_service = (MockExchangeService &) ExecutionRuntime::GetInstance()->GetExchangeService();
  EXPECT_CALL(exchange_service, DequeueMbuf).WillRepeatedly(testing::Invoke(DynamicSchedDequeueMbufStub));
  ASSERT_EQ(default_executor_->Initialize(), SUCCESS);
  g_sched_cnt = 0;

  EXPECT_CALL(exchange_service, Dequeue).WillRepeatedly(Return(SUCCESS));
  size_t size = 0;
  EXPECT_CALL(exchange_service, Enqueue(_, _, Matcher<size_t>(Enqueue3SizeMatcher(size)), _, _))
      .WillRepeatedly(testing::Invoke(DynamicSchedEnqueueStub));

  ASSERT_EQ(default_executor_->ModelRunStart(), SUCCESS);

  // wait sched
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  ASSERT_EQ(default_executor_->ModelRunStop(), SUCCESS);
  ASSERT_EQ(default_executor_->ModelRunStop(), SUCCESS);  // not started
  default_executor_->is_dynamic_sched_ = false;
  default_executor_->status_input_queue_attrs_.clear();
  default_executor_->sched_input_queue_attrs_.clear();
  default_executor_->sched_output_queue_attrs_.clear();
  default_executor_->model_index_info_.clear();
  default_executor_->datagw_request_bindings_.clear();
  default_executor_->cached_trans_ids_.clear();
}

TEST_F(HeterogeneousModelExecutorTest, TestDynamicSchedResponseEnqueueFailed) {
  default_executor_->is_dynamic_sched_ = true;
  DeployQueueAttr queue_attr;
  queue_attr.queue_id = 105;
  default_executor_->status_input_queue_attrs_.push_back(queue_attr);
  queue_attr.queue_id = 101;
  queue_attr.global_logic_id = 1;
  default_executor_->sched_input_queue_attrs_.push_back(queue_attr);
  queue_attr.queue_id = 102;
  default_executor_->sched_output_queue_attrs_.push_back(queue_attr);

  DeployPlan::DeviceInfo device_info =
      DeployPlan::DeviceInfo(static_cast<int32_t>(CPU), 0, 0);
  DeployPlan::ExtendedIndexInfo index_info;
  index_info.device_info = device_info;
  index_info.submodel_instance_name = "model1";
  index_info.is_normal = true;
  DeployPlan::ExtendedIndexInfo index_info2;
  index_info2.device_info = device_info;
  index_info2.submodel_instance_name = "model1";
  index_info2.is_normal = false;
  DeployPlan::DynamicGroupRouteInfo route1 = {0, 0, index_info, false};
  DeployPlan::DynamicGroupRouteInfo route2 = {1, 1, index_info, false};
  DeployPlan::DynamicGroupRouteInfo route3 = {2, 2, index_info, false};
  DeployPlan::DynamicGroupRouteInfo route4 = {2, 2, index_info2, false};
  DeployPlan::DstGroupInfo group_info {1, {route1, route2, route3, route4}};
  default_executor_->model_index_info_ = {{1, {{1,
      {{index_info}, {{1, group_info}}}
  }}}};
  default_executor_->datagw_request_bindings_ = {{1, 102}};
  for (size_t i = 1; i <= 1025; ++i) {
    default_executor_->cached_trans_ids_[i] = {0};
  }
  HeterogeneousModelExecutor::QueueStatus status;
  status.queue_depth = 0;
  status.device_id = 0;
  status.device_type = 0;
  default_executor_->queue_status_info_[1].first = status;
  default_executor_->queue_status_info_[1].second = 0;
  auto &exchange_service = (MockExchangeService &) ExecutionRuntime::GetInstance()->GetExchangeService();
  EXPECT_CALL(exchange_service, DequeueMbuf).WillRepeatedly(testing::Invoke(DynamicSchedDequeueMbufStub));
  ASSERT_EQ(default_executor_->Initialize(), SUCCESS);
  g_sched_cnt = 0;

  EXPECT_CALL(exchange_service, Dequeue).WillRepeatedly(Return(SUCCESS));
  size_t size = 0;
  EXPECT_CALL(exchange_service, Enqueue(_, _, Matcher<size_t>(Enqueue3SizeMatcher(size)), _, _))
      .WillRepeatedly(testing::Invoke(DynamicSchedEnqueueStub));

  ASSERT_EQ(default_executor_->ModelRunStart(), SUCCESS);

  // wait sched
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  ASSERT_EQ(default_executor_->ModelRunStop(), SUCCESS);
  ASSERT_EQ(default_executor_->ModelRunStop(), SUCCESS);  // not started
  default_executor_->is_dynamic_sched_ = false;
  default_executor_->status_input_queue_attrs_.clear();
  default_executor_->sched_input_queue_attrs_.clear();
  default_executor_->sched_output_queue_attrs_.clear();
  default_executor_->model_index_info_.clear();
  default_executor_->datagw_request_bindings_.clear();
  default_executor_->cached_trans_ids_.clear();
}

TEST_F(HeterogeneousModelExecutorTest, TestDynamicSchedFindGroupIndexByCache) {
  default_executor_->is_dynamic_sched_ = true;
  DeployQueueAttr queue_attr;
  queue_attr.queue_id = 105;
  default_executor_->status_input_queue_attrs_.push_back(queue_attr);
  queue_attr.queue_id = 101;
  queue_attr.global_logic_id = 1;
  default_executor_->sched_input_queue_attrs_.push_back(queue_attr);
  queue_attr.queue_id = 102;
  default_executor_->sched_output_queue_attrs_.push_back(queue_attr);

  DeployPlan::DeviceInfo device_info =
      DeployPlan::DeviceInfo(static_cast<int32_t>(CPU), 0, 0);
  DeployPlan::ExtendedIndexInfo index_info;
  index_info.device_info = device_info;
  index_info.submodel_instance_name = "model1";
  index_info.is_normal = true;
  DeployPlan::DynamicGroupRouteInfo route1 = {0, 0, index_info, false};
  DeployPlan::DynamicGroupRouteInfo route2 = {1, 1, index_info, false};
  DeployPlan::DynamicGroupRouteInfo route3 = {2, 2, index_info, false};
  DeployPlan::DstGroupInfo group_info {3, {route1, route2, route3}};
  default_executor_->model_index_info_ = {{1, {{1,
      {{index_info}, {{1, group_info}}}
  }}}};
  default_executor_->datagw_request_bindings_ = {{1, 102}};
  for (size_t i = 1; i <= 1025; ++i) {
    default_executor_->cached_trans_ids_[i] = {0};
  }
  default_executor_->routelabel_cache_info_ = {
    {{1, 0}, {{3, std::make_pair(1, "")}}},
    {{7, 8}, {{9, std::make_pair(10, "")}}}
  };
  g_dynamic_sched_by_cache = true;
  auto &exchange_service = (MockExchangeService &) ExecutionRuntime::GetInstance()->GetExchangeService();
  EXPECT_CALL(exchange_service, DequeueMbuf).WillRepeatedly(testing::Invoke(DynamicSchedDequeueMbufStub));
  ASSERT_EQ(default_executor_->Initialize(), SUCCESS);
  g_sched_cnt = 0;

  EXPECT_CALL(exchange_service, Dequeue).WillRepeatedly(Return(SUCCESS));
  size_t size = 0;
  EXPECT_CALL(exchange_service, Enqueue(_, _, Matcher<size_t>(Enqueue3SizeMatcher(size)), _, _))
      .WillRepeatedly(testing::Invoke(DynamicSchedEnqueueStub));

  ASSERT_EQ(default_executor_->ModelRunStart(), SUCCESS);

  // wait sched
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  ASSERT_EQ(default_executor_->ModelRunStop(), SUCCESS);
  ASSERT_EQ(default_executor_->ModelRunStop(), SUCCESS);  // not started
  default_executor_->is_dynamic_sched_ = false;
  default_executor_->status_input_queue_attrs_.clear();
  default_executor_->sched_input_queue_attrs_.clear();
  default_executor_->sched_output_queue_attrs_.clear();
  default_executor_->model_index_info_.clear();
  default_executor_->datagw_request_bindings_.clear();
  default_executor_->cached_trans_ids_.clear();
  default_executor_->routelabel_cache_info_.clear();
  g_dynamic_sched_by_cache = false;
}

TEST_F(HeterogeneousModelExecutorTest, TestDynamicSchedInitFailedWithWrongBindings) {
  default_executor_->is_dynamic_sched_ = true;
  DeployQueueAttr queue_attr;
  queue_attr.queue_id = 105;
  default_executor_->status_input_queue_attrs_.push_back(queue_attr);
  queue_attr.queue_id = 101;
  queue_attr.global_logic_id = 1;
  default_executor_->sched_input_queue_attrs_.push_back(queue_attr);
  queue_attr.queue_id = 102;
  default_executor_->sched_output_queue_attrs_.push_back(queue_attr);

  DeployPlan::DeviceInfo device_info =
      DeployPlan::DeviceInfo(static_cast<int32_t>(CPU), 0, 0);
  DeployPlan::ExtendedIndexInfo index_info;
  index_info.device_info = device_info;
  index_info.submodel_instance_name = "model1";
  index_info.is_normal = true;
  DeployPlan::DynamicGroupRouteInfo route1 = {0, 0, index_info, false};
  DeployPlan::DynamicGroupRouteInfo route2 = {1, 1, index_info, false};
  DeployPlan::DynamicGroupRouteInfo route3 = {2, 2, index_info, false};
  DeployPlan::DstGroupInfo group_info {1, {route1, route2, route3}};
  default_executor_->model_index_info_ = {{1, {{1,
      {{index_info}, {{1, group_info}}}
  }}}};
  default_executor_->datagw_request_bindings_ = {{3, 102}};
  auto &exchange_service = (MockExchangeService &) ExecutionRuntime::GetInstance()->GetExchangeService();
  EXPECT_CALL(exchange_service, DequeueMbuf).WillRepeatedly(testing::Invoke(DynamicSchedDequeueMbufStub));
  ASSERT_EQ(default_executor_->Initialize(), FAILED);
  default_executor_->is_dynamic_sched_ = false;
  default_executor_->status_input_queue_attrs_.clear();
  default_executor_->sched_input_queue_attrs_.clear();
  default_executor_->sched_output_queue_attrs_.clear();
  default_executor_->model_index_info_.clear();
  default_executor_->datagw_request_bindings_.clear();
  default_executor_->routelabel_cache_info_.clear();
}

TEST_F(HeterogeneousModelExecutorTest, TestDynamicSchedFetchDataSuccess) {
  default_executor_->is_dynamic_sched_ = true;
  DeployQueueAttr queue_attr;
  queue_attr.queue_id = 105;
  default_executor_->status_input_queue_attrs_.push_back(queue_attr);
  queue_attr.queue_id = 101;
  queue_attr.global_logic_id = 1;
  default_executor_->sched_input_queue_attrs_.push_back(queue_attr);
  queue_attr.queue_id = 102;
  default_executor_->sched_output_queue_attrs_.push_back(queue_attr);

  DeployPlan::DeviceInfo device_info =
      DeployPlan::DeviceInfo(static_cast<int32_t>(CPU), 0, 0);
  DeployPlan::ExtendedIndexInfo index_info;
  index_info.device_info = device_info;
  index_info.submodel_instance_name = "model1";
  index_info.is_normal = true;
  DeployPlan::DynamicGroupRouteInfo route1 = {0, 0, index_info, false};
  DeployPlan::DynamicGroupRouteInfo route2 = {1, 1, index_info, false};
  DeployPlan::DynamicGroupRouteInfo route3 = {2, 2, index_info, false};
  DeployPlan::DstGroupInfo group_info {1, {route1, route2, route3}};
  default_executor_->model_index_info_ = {{1, {{1,
      {{index_info}, {{1, group_info}}}
  }}}};
  default_executor_->datagw_request_bindings_ = {{1, 102}};
  auto &exchange_service = (MockExchangeService &)ExecutionRuntime::GetInstance()->GetExchangeService();
  EXPECT_CALL(exchange_service, DequeueMbuf).WillRepeatedly(testing::Invoke(DynamicSchedDequeueMbufStub));
  EXPECT_CALL(exchange_service, Dequeue).WillRepeatedly(Return(SUCCESS));
  EXPECT_CALL(exchange_service, DequeueTensor).WillRepeatedly(testing::Return(SUCCESS));
  ASSERT_EQ(default_executor_->Initialize(), SUCCESS);
  for (size_t i = 0; i < default_executor_->output_queue_attrs_.size(); ++i) {
    default_executor_->output_is_no_tiling_[i] = true;
    default_executor_->output_tensor_sizes_[i] = 1 * 1 * 224 * 224 * 10;
  }

  DataFlowInfo info;
  info.SetStartTime(1UL);
  info.SetEndTime(2UL);
  info.SetFlowFlags(0U);
  std::vector<GeTensor> outputs;
  auto original_handle = ExecutionRuntime::handle_;
  ExecutionRuntime::handle_ = (void*)0x12345678;
  auto options_bk = GetThreadLocalContext().GetAllSessionOptions();
  auto options = options_bk;
  options.emplace(RESOURCE_CONFIG_PATH, "xxx");
  GetThreadLocalContext().SetSessionOption(options);
  GE_MAKE_GUARD(recover, [options_bk]() { GetThreadLocalContext().SetSessionOption(options_bk); });
  Status ret = default_executor_->FetchData({}, outputs, info, 1000U);
  ExecutionRuntime::handle_ = original_handle;
  ASSERT_EQ(ret, SUCCESS);
  EXPECT_EQ(outputs.size(), default_executor_->output_queue_attrs_.size());
  default_executor_->is_dynamic_sched_ = false;
  default_executor_->status_input_queue_attrs_.clear();
  default_executor_->sched_input_queue_attrs_.clear();
  default_executor_->sched_output_queue_attrs_.clear();
  default_executor_->model_index_info_.clear();
  default_executor_->datagw_request_bindings_.clear();
}

TEST_F(HeterogeneousModelExecutorTest, TestDynamicSchedModelIndexInfoUpdateSuccess) {
  default_executor_->is_dynamic_sched_ = true;
  DeployPlan::AbnormalStatusCallbackInfo abnormal_status_callback_info;
  default_executor_->abnormal_status_callback_info_ = &abnormal_status_callback_info;
  default_executor_->deploy_state_ = 0U;

  DeployQueueAttr queue_attr;
  queue_attr.queue_id = 105;
  default_executor_->status_input_queue_attrs_.push_back(queue_attr);
  queue_attr.queue_id = 101;
  queue_attr.global_logic_id = 1;
  default_executor_->sched_input_queue_attrs_.push_back(queue_attr);
  queue_attr.queue_id = 102;
  default_executor_->sched_output_queue_attrs_.push_back(queue_attr);

  DeployPlan::DeviceInfo device_info =
      DeployPlan::DeviceInfo(static_cast<int32_t>(CPU), 0, 0);
  DeployPlan::ExtendedIndexInfo index_info;
  index_info.device_info = device_info;
  index_info.submodel_instance_name = "model1";
  index_info.is_normal = true;
  DeployPlan::DynamicGroupRouteInfo route1 = {0, 0, index_info, false};
  DeployPlan::DynamicGroupRouteInfo route2 = {1, 1, index_info, false};
  DeployPlan::DynamicGroupRouteInfo route3 = {2, 2, index_info, false};
  DeployPlan::DstGroupInfo group_info {1, {route1, route2, route3}};
  default_executor_->model_index_info_ = {{1, {{1,
      {{index_info}, {{1, group_info}}}
  }}}};
  default_executor_->datagw_request_bindings_ = {{1, 102}};
  DeployQueueAttr queue_attr2 = {};
  queue_attr2.device_id = 0;
  queue_attr2.queue_id = 0U;
  default_executor_->input_queue_attrs_.emplace_back(queue_attr2);
  default_executor_->output_queue_attrs_.emplace_back(queue_attr2);
  default_executor_->control_input_queue_attrs_.emplace_back(queue_attr2);
  default_executor_->control_output_queue_attrs_.emplace_back(queue_attr2);
  GeShape ge_shape{vector<int64_t>({1, 2, 3, 4})};
  GeTensorDescPtr tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(ge_shape);
  tensor_desc->SetFormat(FORMAT_NCHW);
  tensor_desc->SetDataType(DT_FLOAT);
  default_executor_->output_tensor_desc_.emplace_back(std::move(tensor_desc));

  auto &exchange_service = (MockExchangeService &)ExecutionRuntime::GetInstance()->GetExchangeService();
  EXPECT_CALL(exchange_service, DequeueTensor).WillRepeatedly(testing::Invoke(DequeueTensorFailed));
  EXPECT_CALL(exchange_service, Dequeue).WillRepeatedly(Return(ACL_ERROR_RT_QUEUE_EMPTY));
  EXPECT_CALL(exchange_service, DequeueMbuf).WillRepeatedly(Return(ACL_ERROR_RT_QUEUE_EMPTY));
  printf("start AbnormalStatusCallbackInit\n");
  ASSERT_EQ(default_executor_->Initialize(), SUCCESS);
  printf("success AbnormalStatusCallbackInit size=%zu\n",
      default_executor_->abnormal_status_callback_info_->callback_list.size());
  uint32_t dynamic_sched = kCallbackStartRedeploy;
  RootModelId2SubmodelName par2;
  par2[0]["model1_0_0_1"] = false;
  par2[0]["model2_0_0_1"] = false;
  for (auto &iter : abnormal_status_callback_info.callback_list) {
     ASSERT_EQ((iter.second(dynamic_sched, par2)), SUCCESS);
   }
  dynamic_sched = kCallbackDynamicSched;
  for (auto &iter : abnormal_status_callback_info.callback_list) {
     ASSERT_EQ((iter.second(dynamic_sched, par2)), SUCCESS);
   }
  dynamic_sched = kCallbackFailedRedeploy;
  for (auto &iter : abnormal_status_callback_info.callback_list) {
     ASSERT_EQ((iter.second(dynamic_sched, par2)), SUCCESS);
   }
  std::unordered_set<std::string> trimming_names = {"test_model"};
  default_executor_->model_trimming_edges_model_instances_.emplace_back(trimming_names);
  dynamic_sched = kCallbackStartRedeploy;
  for (auto &iter : abnormal_status_callback_info.callback_list) {
    ASSERT_EQ((iter.second(dynamic_sched, par2)), SUCCESS);
  }
  default_executor_->abnormal_submodel_instances_name_[default_executor_->GetDeployedModelId()] = {{"test_model", false}};
  for (auto &iter : abnormal_status_callback_info.callback_list) {
    ASSERT_EQ((iter.second(dynamic_sched, par2)), SUCCESS);
  }
  default_executor_->is_dynamic_sched_ = false;
  default_executor_->status_input_queue_attrs_.clear();
  default_executor_->sched_input_queue_attrs_.clear();
  default_executor_->sched_output_queue_attrs_.clear();
  default_executor_->model_index_info_.clear();
  default_executor_->datagw_request_bindings_.clear();
  default_executor_->deploy_state_ = 0U;
  default_executor_->abnormal_status_callback_info_ = nullptr;
  default_executor_->input_queue_attrs_.clear();
  default_executor_->output_queue_attrs_.clear();
  default_executor_->control_input_queue_attrs_.clear();
  default_executor_->control_output_queue_attrs_.clear();
  default_executor_->model_trimming_edges_model_instances_.clear();
  default_executor_->abnormal_submodel_instances_name_.clear();
}

TEST_F(HeterogeneousModelExecutorTest, UpdateAbnormalNamesWithTrimmingModels) {
  std::unordered_set<std::string> trimming_names = {"test_model", "test_model2"};
  default_executor_->model_trimming_edges_model_instances_.emplace_back(trimming_names);
  RootModelId2SubmodelName abnormal_submodel_instances_name;
  abnormal_submodel_instances_name[default_executor_->GetDeployedModelId()] = {{"test_model", true}};
  default_executor_->UpdateAbnormalInstanceList(abnormal_submodel_instances_name);
  ASSERT_EQ(default_executor_->abnormal_submodel_instances_name_[default_executor_->GetDeployedModelId()].size(), 2);
  ASSERT_EQ(default_executor_->abnormal_submodel_instances_name_[default_executor_->GetDeployedModelId()]["test_model2"], true);
  default_executor_->model_trimming_edges_model_instances_.clear();
  default_executor_->abnormal_submodel_instances_name_.clear();

  default_executor_->model_trimming_edges_model_instances_.emplace_back(trimming_names);
  abnormal_submodel_instances_name[default_executor_->GetDeployedModelId()] = {{"test_model", false}};
  default_executor_->UpdateAbnormalInstanceList(abnormal_submodel_instances_name);
  ASSERT_EQ(default_executor_->abnormal_submodel_instances_name_[default_executor_->GetDeployedModelId()].size(), 1);
  ASSERT_EQ(default_executor_->abnormal_submodel_instances_name_[default_executor_->GetDeployedModelId()]["test_model"], false);
  default_executor_->model_trimming_edges_model_instances_.clear();
  default_executor_->abnormal_submodel_instances_name_.clear();
}

TEST_F(HeterogeneousModelExecutorTest, TestDynamicSchedFeedFetchRedeploying) {
  default_executor_->is_dynamic_sched_ = true;
  DeployPlan::AbnormalStatusCallbackInfo abnormal_status_callback_info;
  default_executor_->abnormal_status_callback_info_ = &abnormal_status_callback_info;
  default_executor_->deploy_state_ = 0U;

  DeployQueueAttr queue_attr;
  queue_attr.queue_id = 105;
  default_executor_->status_input_queue_attrs_.push_back(queue_attr);
  queue_attr.queue_id = 101;
  queue_attr.global_logic_id = 1;
  default_executor_->sched_input_queue_attrs_.push_back(queue_attr);
  queue_attr.queue_id = 102;
  default_executor_->sched_output_queue_attrs_.push_back(queue_attr);

  DeployPlan::DeviceInfo device_info =
      DeployPlan::DeviceInfo(static_cast<int32_t>(CPU), 0, 0);
  DeployPlan::ExtendedIndexInfo index_info;
  index_info.device_info = device_info;
  index_info.submodel_instance_name = "model1";
  index_info.is_normal = true;
  DeployPlan::DynamicGroupRouteInfo route1 = {0, 0, index_info, false};
  DeployPlan::DynamicGroupRouteInfo route2 = {1, 1, index_info, false};
  DeployPlan::DynamicGroupRouteInfo route3 = {2, 2, index_info, false};
  DeployPlan::DstGroupInfo group_info {1, {route1, route2, route3}};
  default_executor_->model_index_info_ = {{1, {{1,
      {{index_info}, {{1, group_info}}}
  }}}};
  default_executor_->datagw_request_bindings_ = {{1, 102}};
  auto &exchange_service = (MockExchangeService &) ExecutionRuntime::GetInstance()->GetExchangeService();
  EXPECT_CALL(exchange_service, DequeueMbuf).WillRepeatedly(testing::Invoke(DynamicSchedDequeueMbufStub));

  auto original_handle = ExecutionRuntime::handle_;
  ExecutionRuntime::handle_ = (void*)0x12345678;
  auto options_bk = GetThreadLocalContext().GetAllSessionOptions();
  auto options = options_bk;
  options.emplace(RESOURCE_CONFIG_PATH, "xxx");
  GetThreadLocalContext().SetSessionOption(options);
  GE_MAKE_GUARD(recover, [options_bk]() { GetThreadLocalContext().SetSessionOption(options_bk); });

  ASSERT_EQ(default_executor_->Initialize(), SUCCESS);
  default_executor_->deploy_state_ = 1U;
  {
    GeTensor output_tensor;
    std::shared_ptr<AlignedPtr> aligned_ptr;
    ExchangeService::ControlInfo control_info;
    size_t output_index = 0U;
    ASSERT_EQ(default_executor_->DoDequeue(output_tensor, aligned_ptr, control_info, output_index), ACL_ERROR_GE_REDEPLOYING);
  }
  {
    std::vector<uint32_t> indexes;
    std::vector<GeTensor> inputs;
    DataFlowInfo info;
    int32_t timeout = 0;
    ASSERT_EQ(default_executor_->FeedData(indexes, inputs, info, timeout), ACL_ERROR_GE_REDEPLOYING);
  }
  {
    std::vector<uint32_t> indexes;
    std::vector<GeTensor> outputs;
    DataFlowInfo info;
    int32_t timeout = 0;
    ASSERT_EQ(default_executor_->FetchData(indexes, outputs, info, timeout), ACL_ERROR_GE_REDEPLOYING);
  }

  default_executor_->is_dynamic_sched_ = false;
  default_executor_->status_input_queue_attrs_.clear();
  default_executor_->sched_input_queue_attrs_.clear();
  default_executor_->sched_output_queue_attrs_.clear();
  default_executor_->model_index_info_.clear();
  default_executor_->datagw_request_bindings_.clear();
  default_executor_->deploy_state_ = 0U;
  default_executor_->abnormal_status_callback_info_ = nullptr;
  ExecutionRuntime::handle_ = original_handle;
}

TEST_F(HeterogeneousModelExecutorTest, TestFeedDataSubhealthyState) {
  default_executor_->deploy_state_ = 0U;
  ASSERT_EQ(default_executor_->Initialize(), SUCCESS);
  std::vector<GeTensor> input_tensors;
  for (size_t i = 0; i < default_executor_->input_queue_attrs_.size(); ++i) {
    auto tensor_desc = default_executor_->input_tensor_desc_[i];
    auto tensor_size = default_executor_->input_tensor_sizes_[i];
    default_executor_->input_is_no_tiling_[i] = true;
    GeTensor tensor(*tensor_desc, std::vector<uint8_t>(tensor_size));
    input_tensors.emplace_back(tensor);
  }

  auto &exchange_service = (MockExchangeService &) ExecutionRuntime::GetInstance()->GetExchangeService();
  void *data = nullptr;
  EXPECT_CALL(exchange_service, Enqueue(_, _, Matcher<const void *>(Enqueue3VoidMatcher(data)), _, _))
      .WillRepeatedly(Return(SUCCESS));
  size_t size = 0;
  EXPECT_CALL(exchange_service, Enqueue(_, _, Matcher<size_t>(Enqueue3SizeMatcher(size)), _, _))
      .WillRepeatedly(Return(SUCCESS));
  DataFlowInfo info;
  info.SetStartTime(1UL);
  info.SetEndTime(2UL);
  info.SetFlowFlags(0U);

  auto original_handle = ExecutionRuntime::handle_;
  ExecutionRuntime::handle_ = (void*)0x12345678;
  auto options_bk = GetThreadLocalContext().GetAllSessionOptions();
  auto options = options_bk;
  options.emplace(RESOURCE_CONFIG_PATH, "xxx");
  GetThreadLocalContext().SetSessionOption(options);
  GE_MAKE_GUARD(recover, [options_bk]() { GetThreadLocalContext().SetSessionOption(options_bk); });
  std::vector<uint32_t> feed_indexes;
  feed_indexes.resize(input_tensors.size());
  // genertate feed indexes from 0
  std::iota(feed_indexes.begin(), feed_indexes.end(), 0U);
  default_executor_->is_fusion_input_[1] = true;
  default_executor_->input_queue_attrs_[0].queue_id = UINT32_MAX;
  default_executor_->deploy_state_ = kCallbackRedeployDone;
  Status ret = default_executor_->FeedData(feed_indexes, input_tensors, info, 1000);
  ASSERT_EQ(ret, ACL_ERROR_GE_SUBHEALTHY);
  default_executor_->deploy_state_ = 0U;
  ExecutionRuntime::handle_ = original_handle;
}

TEST_F(HeterogeneousModelExecutorTest, TestFetchDataSubhealthyState) {
  default_executor_->deploy_state_ = 0U;
  ASSERT_EQ(default_executor_->Initialize(), SUCCESS);
  for (size_t i = 0; i < default_executor_->output_queue_attrs_.size(); ++i) {
    default_executor_->output_is_no_tiling_[i] = true;
    default_executor_->output_tensor_sizes_[i] = 1 * 1 * 224 * 224 * 10;
  }
  auto &exchange_service = (MockExchangeService &)ExecutionRuntime::GetInstance()->GetExchangeService();
  EXPECT_CALL(exchange_service, Dequeue).WillRepeatedly(Return(SUCCESS));
  EXPECT_CALL(exchange_service, DequeueTensor).WillRepeatedly(testing::Return(SUCCESS));

  DataFlowInfo info;
  info.SetStartTime(1UL);
  info.SetEndTime(2UL);
  info.SetFlowFlags(0U);
  std::vector<GeTensor> outputs;
  auto original_handle = ExecutionRuntime::handle_;
  ExecutionRuntime::handle_ = (void*)0x12345678;
  auto options_bk = GetThreadLocalContext().GetAllSessionOptions();
  auto options = options_bk;
  options.emplace(RESOURCE_CONFIG_PATH, "xxx");
  GetThreadLocalContext().SetSessionOption(options);
  GE_MAKE_GUARD(recover, [options_bk]() { GetThreadLocalContext().SetSessionOption(options_bk); });
  default_executor_->deploy_state_ = kCallbackRedeployDone;
  Status ret = default_executor_->FetchData({}, outputs, info, 1000U);
  ExecutionRuntime::handle_ = original_handle;
  ASSERT_EQ(ret, ACL_ERROR_GE_SUBHEALTHY);
  EXPECT_EQ(outputs.size(), default_executor_->output_queue_attrs_.size());
  default_executor_->deploy_state_ = 0U;
}

TEST_F(HeterogeneousModelExecutorTest, TestModelIndexInfoUpdate) {
  DeployPlan::DeviceInfo device_info1 =
      DeployPlan::DeviceInfo(static_cast<int32_t>(CPU), 0, 0);
  DeployPlan::DeviceInfo device_info2 =
      DeployPlan::DeviceInfo(static_cast<int32_t>(NPU), 1, 0);
  DeployPlan::DeviceInfo device_info3 =
      DeployPlan::DeviceInfo(static_cast<int32_t>(NPU), 1, 1);

  DeployPlan::ExtendedIndexInfo index_info1;
  index_info1.device_info = device_info1;
  index_info1.submodel_instance_name = "model1";
  index_info1.is_normal = true;
  DeployPlan::ExtendedIndexInfo index_info1_false;
  index_info1_false.device_info = device_info1;
  index_info1_false.submodel_instance_name = "model1";
  index_info1_false.is_normal = false;
  DeployPlan::ExtendedIndexInfo index_info2;
  index_info2.device_info = device_info2;
  index_info2.submodel_instance_name = "model2";
  index_info2.is_normal = true;
  DeployPlan::ExtendedIndexInfo index_info3;
  index_info3.device_info = device_info3;
  index_info3.submodel_instance_name = "model3";
  index_info3.is_normal = true;
  DeployPlan::DynamicGroupRouteInfo route1 = {0, 0, index_info2, false};
  DeployPlan::DynamicGroupRouteInfo route2 = {1, 1, index_info3, false};
  DeployPlan::DynamicGroupRouteInfo route3 = {2, 2, index_info3, false};
  DeployPlan::DstGroupInfo group_info {1, {route1, route2, route3}};
  dlog_setlevel(0, 0, 0);
  default_executor_->model_index_info_ = {{1, {{1,
      {index_info1, {{1, group_info}}}
  }}}};
  default_executor_->abnormal_submodel_instances_name_[default_executor_->GetDeployedModelId()] = {{"model1", false}};
  default_executor_->ModelIndexInfoUpdate();
  EXPECT_EQ(default_executor_->model_index_info_[1][1].first.is_normal, false);
  default_executor_->model_index_info_.clear();
  default_executor_->abnormal_submodel_instances_name_.clear();
  default_executor_->model_index_info_ = {{1, {{1,
      {index_info1, {{1, group_info}}}
  }}}};
  default_executor_->abnormal_submodel_instances_name_[default_executor_->GetDeployedModelId()] = {{"model2", false}};
  default_executor_->ModelIndexInfoUpdate();
  EXPECT_EQ(default_executor_->model_index_info_[1][1].second[1].routes[0].extended_info.is_normal, false);
  EXPECT_EQ(default_executor_->model_index_info_[1][1].second[1].routes[1].extended_info.is_normal, true);
  dlog_setlevel(0, 3, 0);
  default_executor_->model_index_info_.clear();
  default_executor_->abnormal_submodel_instances_name_.clear();
}

TEST_F(HeterogeneousModelExecutorTest, TestDynamicSchedFeedDataSuccess) {
  default_executor_->is_dynamic_sched_ = true;
  DeployQueueAttr queue_attr;
  queue_attr.queue_id = 105;
  default_executor_->status_input_queue_attrs_.push_back(queue_attr);
  queue_attr.queue_id = 101;
  queue_attr.global_logic_id = 1;
  default_executor_->sched_input_queue_attrs_.push_back(queue_attr);
  queue_attr.queue_id = 102;
  default_executor_->sched_output_queue_attrs_.push_back(queue_attr);

  DeployPlan::DeviceInfo device_info =
      DeployPlan::DeviceInfo(static_cast<int32_t>(CPU), 0, 0);
  DeployPlan::ExtendedIndexInfo index_info;
  index_info.device_info = device_info;
  index_info.submodel_instance_name = "model1";
  index_info.is_normal = true;
  DeployPlan::DynamicGroupRouteInfo route1 = {0, 0, index_info, false};
  DeployPlan::DynamicGroupRouteInfo route2 = {1, 1, index_info, false};
  DeployPlan::DynamicGroupRouteInfo route3 = {2, 2, index_info, false};
  DeployPlan::DstGroupInfo group_info {1, {route1, route2, route3}};
  default_executor_->model_index_info_ = {{1, {{1,
      {index_info, {{1, group_info}}}
  }}}};
  default_executor_->datagw_request_bindings_ = {{1, 102}};
  auto &exchange_service = (MockExchangeService &) ExecutionRuntime::GetInstance()->GetExchangeService();
  EXPECT_CALL(exchange_service, DequeueMbuf).WillRepeatedly(testing::Invoke(DynamicSchedDequeueMbufStub));

  ASSERT_EQ(default_executor_->Initialize(), SUCCESS);
  std::vector<GeTensor> input_tensors;
  for (size_t i = 0; i < default_executor_->input_queue_attrs_.size(); ++i) {
    auto tensor_desc = default_executor_->input_tensor_desc_[i];
    auto tensor_size = default_executor_->input_tensor_sizes_[i];
    default_executor_->input_is_no_tiling_[i] = true;
    GeTensor tensor(*tensor_desc, std::vector<uint8_t>(tensor_size));
    input_tensors.emplace_back(tensor);
  }

  void *data = nullptr;
  EXPECT_CALL(exchange_service, Enqueue(_, _, Matcher<const void*>(Enqueue3VoidMatcher(data)), _, _))
      .WillRepeatedly(Return(SUCCESS));
  size_t size = 0;
  EXPECT_CALL(exchange_service, Enqueue(_, _, Matcher<size_t>(Enqueue3SizeMatcher(size)), _, _))
      .WillRepeatedly(Return(SUCCESS));

  DataFlowInfo info;
  info.SetStartTime(1UL);
  info.SetEndTime(2UL);
  info.SetFlowFlags(0U);

  auto original_handle = ExecutionRuntime::handle_;
  ExecutionRuntime::handle_ = (void*)0x12345678;
  auto options_bk = GetThreadLocalContext().GetAllSessionOptions();
  auto options = options_bk;
  options.emplace(RESOURCE_CONFIG_PATH, "xxx");
  GetThreadLocalContext().SetSessionOption(options);
  GE_MAKE_GUARD(recover, [options_bk]() { GetThreadLocalContext().SetSessionOption(options_bk); });
  std::vector<uint32_t> feed_indexes;
  feed_indexes.resize(input_tensors.size());
  // genertate feed indexes from 0
  std::iota(feed_indexes.begin(), feed_indexes.end(), 0U);
  default_executor_->is_fusion_input_[1] = true;
  Status ret = default_executor_->FeedData(feed_indexes, input_tensors, info, 1000);
  ASSERT_EQ(ret, SUCCESS);
  info.SetFlowFlags(1U);
  ret = default_executor_->FeedData({}, {}, info, 1000);
  ExecutionRuntime::handle_ = original_handle;
  ASSERT_EQ(ret, SUCCESS);
  default_executor_->is_dynamic_sched_ = false;
  default_executor_->status_input_queue_attrs_.clear();
  default_executor_->sched_input_queue_attrs_.clear();
  default_executor_->sched_output_queue_attrs_.clear();
  default_executor_->model_index_info_.clear();
  default_executor_->datagw_request_bindings_.clear();
}

TEST_F(HeterogeneousModelExecutorTest, TestFillFusionInput) {
  DeployResult deploy_result = {};
  deploy_result.model_id = 777;
  deploy_result.input_queue_attrs = {{1, 0, 0}};
  deploy_result.output_queue_attrs = {{2, 0, 0}};
  auto flow_model = std::make_shared<FlowModel>();
  HeterogeneousModelExecutor executor(flow_model, deploy_result);
  auto tensor_data1 = "qwertyui";
  GeTensor tensor1(GeTensorDesc(GeShape({2, 2, 2}), FORMAT_NCHW, DT_INT8), (uint8_t *)tensor_data1, 8);
  auto buff =std::unique_ptr<uint8_t[]>(new uint8_t[1500]);
  std::vector<GeTensor> inputs;
  inputs.emplace_back(tensor1);
  EXPECT_EQ(executor.FillFusionInput(inputs, reinterpret_cast<void*>(buff.get()), 2048), SUCCESS);
}

TEST_F(HeterogeneousModelExecutorTest, TestAlignFetchDataSuccess) {
  default_executor_->align_attrs_.align_max_cache_num = 10;
  default_executor_->align_attrs_.drop_when_not_align = false;
  default_executor_->align_attrs_.align_timeout = -1;
  ASSERT_EQ(default_executor_->Initialize(), SUCCESS);
  auto &exchange_service = (MockExchangeService &)ExecutionRuntime::GetInstance()->GetExchangeService();
  EXPECT_CALL(exchange_service, Dequeue).WillRepeatedly(Return(SUCCESS));
  EXPECT_CALL(exchange_service, DequeueTensor).WillRepeatedly(testing::Return(SUCCESS));

  DataFlowInfo info;
  info.SetStartTime(1UL);
  info.SetEndTime(2UL);
  info.SetFlowFlags(0U);
  std::vector<GeTensor> outputs;
  auto original_handle = ExecutionRuntime::handle_;
  ExecutionRuntime::handle_ = (void*)0x12345678;
  auto options_bk = GetThreadLocalContext().GetAllSessionOptions();
  auto options = options_bk;
  options.emplace(RESOURCE_CONFIG_PATH, "xxx");
  GetThreadLocalContext().SetSessionOption(options);
  GE_MAKE_GUARD(recover, [options_bk]() { GetThreadLocalContext().SetSessionOption(options_bk); });
  Status ret = default_executor_->FetchData({}, outputs, info, 1000U);
  ExecutionRuntime::handle_ = original_handle;
  ASSERT_EQ(ret, SUCCESS);
  EXPECT_EQ(outputs.size(), default_executor_->output_queue_attrs_.size());
}

TEST_F(HeterogeneousModelExecutorTest, TestAlignFetchDataTimeout) {
  default_executor_->align_attrs_.align_max_cache_num = 10;
  default_executor_->align_attrs_.drop_when_not_align = false;
  default_executor_->align_attrs_.align_timeout = -1;
  ASSERT_EQ(default_executor_->Initialize(), SUCCESS);
  auto &exchange_service = (MockExchangeService &)ExecutionRuntime::GetInstance()->GetExchangeService();
  EXPECT_CALL(exchange_service, DequeueTensor).WillRepeatedly(Return(ACL_ERROR_RT_QUEUE_EMPTY));

  DataFlowInfo info;
  std::vector<GeTensor> outputs;
  auto original_handle = ExecutionRuntime::handle_;
  ExecutionRuntime::handle_ = (void*)0x12345678;
  auto options_bk = GetThreadLocalContext().GetAllSessionOptions();
  auto options = options_bk;
  options.emplace(RESOURCE_CONFIG_PATH, "xxx");
  GetThreadLocalContext().SetSessionOption(options);
  GE_MAKE_GUARD(recover, [options_bk]() { GetThreadLocalContext().SetSessionOption(options_bk); });
  Status ret = default_executor_->FetchData({}, outputs, info, 1100U);
  ExecutionRuntime::handle_ = original_handle;
  ASSERT_EQ(ret, ACL_ERROR_RT_QUEUE_EMPTY);
  EXPECT_EQ(outputs.size(), 0);
}

TEST_F(HeterogeneousModelExecutorTest, TestAlignFetchDataFailed) {
  default_executor_->align_attrs_.align_max_cache_num = 10;
  default_executor_->align_attrs_.drop_when_not_align = false;
  default_executor_->align_attrs_.align_timeout = -1;
  ASSERT_EQ(default_executor_->Initialize(), SUCCESS);
  auto &exchange_service = (MockExchangeService &)ExecutionRuntime::GetInstance()->GetExchangeService();
  EXPECT_CALL(exchange_service, DequeueTensor).WillRepeatedly(Return(ACL_ERROR_RT_AICPU_EXCEPTION));

  DataFlowInfo info;
  std::vector<GeTensor> outputs;
  auto original_handle = ExecutionRuntime::handle_;
  ExecutionRuntime::handle_ = (void*)0x12345678;
  auto options_bk = GetThreadLocalContext().GetAllSessionOptions();
  auto options = options_bk;
  options.emplace(RESOURCE_CONFIG_PATH, "xxx");
  GetThreadLocalContext().SetSessionOption(options);
  GE_MAKE_GUARD(recover, [options_bk]() { GetThreadLocalContext().SetSessionOption(options_bk); });
  Status ret = default_executor_->FetchData({}, outputs, info, 1100U);
  ExecutionRuntime::handle_ = original_handle;
  ASSERT_EQ(ret, ACL_ERROR_RT_AICPU_EXCEPTION);
  EXPECT_EQ(outputs.size(), 0);
}

TEST_F(HeterogeneousModelExecutorTest, TestAlignFetchDataOverLimit) {
  DeployResult deploy_result;
  deploy_result.model_id = 777;
  deploy_result.input_queue_attrs = {{1, 0, 0}, {2, 0, 0}};
  deploy_result.status_output_queue_attrs = {{105, 0, 0}};
  deploy_result.broadcast_input_queue_attrs = {{{1, 0, 0}}};
  deploy_result.output_queue_attrs = {{5, 0, 0}, {7, 0, 0}};
  deploy_result.control_input_queue_attrs = {{6, 0, 0}};
  deploy_result.input_model_name = default_root_model_->GetModelName();
  deploy_result.dev_abnormal_callback = []() -> Status { return SUCCESS; };
  deploy_result.input_align_attrs.align_timeout = -1;
  deploy_result.input_align_attrs.drop_when_not_align = false;
  deploy_result.input_align_attrs.align_max_cache_num = 2;
  deploy_result.is_exception_catch = false;

  auto flow_model = std::make_shared<FlowModel>();
  flow_model->AddSubModel(default_root_model_);
  default_executor_ = unique_ptr<HeterogeneousModelExecutor>(new HeterogeneousModelExecutor(flow_model, deploy_result));
  ASSERT_EQ(default_executor_->Initialize(), SUCCESS);
  auto &exchange_service = (MockExchangeService &)ExecutionRuntime::GetInstance()->GetExchangeService();
  EXPECT_CALL(exchange_service, DequeueTensor).WillRepeatedly(Return(SUCCESS));

  GeShape ge_shape{vector<int64_t>({1, 2, 3, 4})};
  GeTensorDescPtr tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(ge_shape);
  tensor_desc->SetFormat(FORMAT_NCHW);
  tensor_desc->SetDataType(DT_FLOAT);
  // default model just have one output, fork to two output.
  default_executor_->output_tensor_desc_.emplace_back(std::move(tensor_desc));
  default_executor_->output_tensor_raw_sizes_.emplace_back(1024);
  default_executor_->output_is_no_tiling_.emplace_back(true);

  DataFlowInfo info;
  std::vector<GeTensor> outputs;
  auto original_handle = ExecutionRuntime::handle_;
  ExecutionRuntime::handle_ = (void*)0x12345678;
  auto options_bk = GetThreadLocalContext().GetAllSessionOptions();
  auto options = options_bk;
  options.emplace(RESOURCE_CONFIG_PATH, "xxx");
  GetThreadLocalContext().SetSessionOption(options);
  GE_MAKE_GUARD(recover, [options_bk]() { GetThreadLocalContext().SetSessionOption(options_bk); });
  auto ret = default_executor_->FetchData({}, outputs, info, 1000U);
  ASSERT_EQ(ret, SUCCESS);
  EXPECT_EQ(outputs.size(), default_executor_->output_queue_attrs_.size());
  outputs.clear();
  ASSERT_EQ(default_executor_->data_aligner_map_.size(), 1);
  auto aligner = default_executor_->data_aligner_map_.begin()->second;
  ExchangeService::MsgInfo msg_info{};
  msg_info.start_time = 123;
  msg_info.ret_code = 999;
  msg_info.trans_id = 100;
  GeTensor tensor;
  TensorWithHeader tensor_with_header{};
  tensor_with_header.tensor = tensor;
  tensor_with_header.msg_info = msg_info;
  bool is_aligned = false;
  EXPECT_EQ(aligner->PushAndAlignData(0, tensor_with_header, outputs, info, is_aligned), SUCCESS);
  msg_info.trans_id = 101;
  tensor_with_header.msg_info = msg_info;
  EXPECT_EQ(aligner->PushAndAlignData(1, tensor_with_header, outputs, info, is_aligned), SUCCESS);
  msg_info.trans_id = 103;
  tensor_with_header.msg_info = msg_info;
  EXPECT_EQ(aligner->PushAndAlignData(0, tensor_with_header, outputs, info, is_aligned), SUCCESS);

  ret = default_executor_->FetchData({}, outputs, info, 1000U);
  ASSERT_EQ(ret, 999);
  EXPECT_EQ(outputs.size(), 0);
  EXPECT_EQ(info.GetStartTime(), 123);

  ExecutionRuntime::handle_ = original_handle;
}

TEST_F(HeterogeneousModelExecutorTest, TestAlignFetchDataWithException) {
  DeployResult deploy_result;
  deploy_result.model_id = 777;
  deploy_result.input_queue_attrs = {{1, 0, 0}, {2, 0, 0}};
  deploy_result.status_output_queue_attrs = {{105, 0, 0}};
  deploy_result.broadcast_input_queue_attrs = {{{1, 0, 0}}};
  deploy_result.output_queue_attrs = {{5, 0, 0}, {7, 0, 0}};
  deploy_result.control_input_queue_attrs = {{6, 0, 0}};
  deploy_result.input_model_name = default_root_model_->GetModelName();
  deploy_result.dev_abnormal_callback = []() -> Status { return SUCCESS; };
  deploy_result.input_align_attrs.align_timeout = -1;
  deploy_result.input_align_attrs.drop_when_not_align = false;
  deploy_result.input_align_attrs.align_max_cache_num = 10;
  deploy_result.is_exception_catch = true;

  std::mutex mt;
  std::condition_variable condition;
  std::set<uint64_t> exception_trans_id;
  deploy_result.exception_notify_callback = [&mt, &condition, &exception_trans_id](const UserExceptionNotify &notify) {
    std::unique_lock<std::mutex> lk(mt);
    exception_trans_id.emplace(notify.trans_id);
    condition.notify_all();
  };
  auto flow_model = std::make_shared<FlowModel>();
  flow_model->AddSubModel(default_root_model_);
  default_executor_ = unique_ptr<HeterogeneousModelExecutor>(new HeterogeneousModelExecutor(flow_model, deploy_result));
  ASSERT_EQ(default_executor_->Initialize(), SUCCESS);
  auto &exchange_service = (MockExchangeService &)ExecutionRuntime::GetInstance()->GetExchangeService();
  EXPECT_CALL(exchange_service, DequeueTensor).WillRepeatedly(Return(SUCCESS));

  GeShape ge_shape{vector<int64_t>({1, 2, 3, 4})};
  GeTensorDescPtr tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(ge_shape);
  tensor_desc->SetFormat(FORMAT_NCHW);
  tensor_desc->SetDataType(DT_FLOAT);
  // default model just have one output, fork to two output.
  default_executor_->output_tensor_desc_.emplace_back(std::move(tensor_desc));
  default_executor_->output_tensor_raw_sizes_.emplace_back(1024);
  default_executor_->output_is_no_tiling_.emplace_back(true);

  auto original_handle = ExecutionRuntime::handle_;
  ExecutionRuntime::handle_ = (void*)0x12345678;
  auto options_bk = GetThreadLocalContext().GetAllSessionOptions();
  auto options = options_bk;
  options.emplace(RESOURCE_CONFIG_PATH, "xxx");
  GetThreadLocalContext().SetSessionOption(options);
  GE_MAKE_GUARD(recover, [options_bk]() { GetThreadLocalContext().SetSessionOption(options_bk); });
  DataFlowInfo info;
  std::vector<GeTensor> outputs;
  auto ret = default_executor_->FetchData({}, outputs, info, 1000U);
  ASSERT_EQ(ret, SUCCESS);
  EXPECT_EQ(outputs.size(), default_executor_->output_queue_attrs_.size());
  outputs.clear();

  domi::DataFlowException data_flow_exception;
  data_flow_exception.set_trans_id(100);
  uint8_t head[256] = {};
  ExchangeService::MsgInfo *msg_info = reinterpret_cast<ExchangeService::MsgInfo *>(head + (256 - 64));
  msg_info->start_time = 123;
  msg_info->trans_id = 100;
  data_flow_exception.set_exception_context(head, sizeof(head));
  default_executor_->exception_handler_.NotifyException(data_flow_exception);
  {
    std::unique_lock<std::mutex> lk(mt);
    condition.wait_for(lk, std::chrono::milliseconds(1000),
                       [&exception_trans_id]() { return !exception_trans_id.empty(); });
  }

  ret = default_executor_->FetchData({}, outputs, info, 1000U);
  ASSERT_EQ(ret, ACL_ERROR_GE_USER_RAISE_EXCEPTION);
  EXPECT_EQ(info.GetStartTime(), 123);
  EXPECT_EQ(outputs.size(), 0);
  ret = default_executor_->FetchData({}, outputs, info, 1000U);
  ASSERT_EQ(ret, SUCCESS);
  EXPECT_EQ(outputs.size(), default_executor_->output_queue_attrs_.size());
  ExecutionRuntime::handle_ = original_handle;
}

TEST_F(HeterogeneousModelExecutorTest, TestFeedFlowMsgSuccess) {
  ASSERT_EQ(default_executor_->Initialize(), SUCCESS);
  std::vector<FlowMsgPtr> input_flow_msgs;
  for (size_t i = 0; i < default_executor_->input_queue_attrs_.size(); ++i) {
    auto tensor_desc = default_executor_->input_tensor_desc_[i];
    auto tensor_size = default_executor_->input_tensor_sizes_[i];
    default_executor_->input_is_no_tiling_[i] = true;
    GeTensor tensor(*tensor_desc, std::vector<uint8_t>(tensor_size));
    auto msg = FlowBufferFactory::ToFlowMsg(TensorAdapter::AsTensor(tensor));
    input_flow_msgs.emplace_back(msg);
  }

  auto &exchange_service = (MockExchangeService &) ExecutionRuntime::GetInstance()->GetExchangeService();
  EXPECT_CALL(exchange_service, EnqueueMbuf).WillRepeatedly(Return(SUCCESS));

  auto original_handle = ExecutionRuntime::handle_;
  ExecutionRuntime::handle_ = (void*)0x12345678;
  auto options_bk = GetThreadLocalContext().GetAllSessionOptions();
  auto options = options_bk;
  options.emplace(RESOURCE_CONFIG_PATH, "xxx");
  GetThreadLocalContext().SetSessionOption(options);
  GE_MAKE_GUARD(recover, [options_bk]() { GetThreadLocalContext().SetSessionOption(options_bk); });

  default_executor_->input_queue_attrs_[0].queue_id = UINT32_MAX;
  Status ret = default_executor_->FeedFlowMsg({}, input_flow_msgs, 1000);
  ASSERT_EQ(ret, SUCCESS);
  ExecutionRuntime::handle_ = original_handle;
}

TEST_F(HeterogeneousModelExecutorTest, TestFetchTensorFlowMsgSuccess) {
  ASSERT_EQ(default_executor_->Initialize(), SUCCESS);
  auto &exchange_service = (MockExchangeService &)ExecutionRuntime::GetInstance()->GetExchangeService();
  auto tensor = FlowBufferFactory::AllocTensorMsg({1, 1}, DT_INT32);
  auto mock_dequeue_mbuf =
    [tensor](int32_t device_id, uint32_t queue_id, rtMbufPtr_t *m_buf, int32_t timeout) -> Status {
    auto input_msg = std::dynamic_pointer_cast<FlowMsgBase>(tensor);
    *m_buf = input_msg->MbufCopyRef();
    return SUCCESS;
  };

  EXPECT_CALL(exchange_service, DequeueMbuf).WillRepeatedly(testing::Invoke(mock_dequeue_mbuf));

  std::vector<FlowMsgPtr> outputs;
  auto original_handle = ExecutionRuntime::handle_;
  ExecutionRuntime::handle_ = (void*)0x12345678;
  auto options_bk = GetThreadLocalContext().GetAllSessionOptions();
  auto options = options_bk;
  options.emplace(RESOURCE_CONFIG_PATH, "xxx");
  GetThreadLocalContext().SetSessionOption(options);
  GE_MAKE_GUARD(recover, [options_bk]() { GetThreadLocalContext().SetSessionOption(options_bk); });
  Status ret = default_executor_->FetchFlowMsg({}, outputs, 1000U);
  ExecutionRuntime::handle_ = original_handle;
  ASSERT_EQ(ret, SUCCESS);
  EXPECT_EQ(outputs.size(), default_executor_->output_queue_attrs_.size());
}

TEST_F(HeterogeneousModelExecutorTest, TestFetchRawDataFlowMsgSuccess) {
  ASSERT_EQ(default_executor_->Initialize(), SUCCESS);
  auto &exchange_service = (MockExchangeService &)ExecutionRuntime::GetInstance()->GetExchangeService();
  auto raw_data = FlowBufferFactory::AllocRawDataMsg(5);
  auto mock_dequeue_mbuf =
    [raw_data](int32_t device_id, uint32_t queue_id, rtMbufPtr_t *m_buf, int32_t timeout) -> Status {
    auto input_msg = std::dynamic_pointer_cast<FlowMsgBase>(raw_data);
    *m_buf = input_msg->MbufCopyRef();
    return SUCCESS;
  };

  EXPECT_CALL(exchange_service, DequeueMbuf).WillRepeatedly(testing::Invoke(mock_dequeue_mbuf));

  std::vector<FlowMsgPtr> outputs;
  auto original_handle = ExecutionRuntime::handle_;
  ExecutionRuntime::handle_ = (void*)0x12345678;
  auto options_bk = GetThreadLocalContext().GetAllSessionOptions();
  auto options = options_bk;
  options.emplace(RESOURCE_CONFIG_PATH, "xxx");
  GetThreadLocalContext().SetSessionOption(options);
  GE_MAKE_GUARD(recover, [options_bk]() { GetThreadLocalContext().SetSessionOption(options_bk); });
  Status ret = default_executor_->FetchFlowMsg({0}, outputs, 1000U);
  ExecutionRuntime::handle_ = original_handle;
  ASSERT_EQ(ret, SUCCESS);
  EXPECT_EQ(outputs.size(), default_executor_->output_queue_attrs_.size());
}

TEST_F(HeterogeneousModelExecutorTest, TestFetchEmptyDataFlowMsgEndOfSequence) {
  ASSERT_EQ(default_executor_->Initialize(), SUCCESS);
  auto &exchange_service = (MockExchangeService &)ExecutionRuntime::GetInstance()->GetExchangeService();
  auto empty = FlowBufferFactory::AllocEmptyDataMsg(MsgType::MSG_TYPE_TENSOR_DATA);
  auto mock_dequeue_mbuf =
    [empty](int32_t device_id, uint32_t queue_id, rtMbufPtr_t *m_buf, int32_t timeout) -> Status {
    auto input_msg = std::dynamic_pointer_cast<FlowMsgBase>(empty);
    *m_buf = input_msg->MbufCopyRef();
    // set end of sequence
    void *head_buf = nullptr;
    uint64_t head_size = 0U;
    rtMbufGetPrivInfo(*m_buf, &head_buf, &head_size);
    (reinterpret_cast<uint8_t *>(head_buf) + 128)[0] = 0x5A;
    return SUCCESS;
  };

  EXPECT_CALL(exchange_service, DequeueMbuf).WillRepeatedly(testing::Invoke(mock_dequeue_mbuf));

  std::vector<FlowMsgPtr> outputs;
  auto original_handle = ExecutionRuntime::handle_;
  ExecutionRuntime::handle_ = (void*)0x12345678;
  auto options_bk = GetThreadLocalContext().GetAllSessionOptions();
  auto options = options_bk;
  options.emplace(RESOURCE_CONFIG_PATH, "xxx");
  GetThreadLocalContext().SetSessionOption(options);
  GE_MAKE_GUARD(recover, [options_bk]() { GetThreadLocalContext().SetSessionOption(options_bk); });
  Status ret = default_executor_->FetchFlowMsg({0}, outputs, 1000U);
  ExecutionRuntime::handle_ = original_handle;
  ASSERT_EQ(ret, END_OF_SEQUENCE);
  EXPECT_EQ(outputs.size(), default_executor_->output_queue_attrs_.size());
}

TEST_F(HeterogeneousModelExecutorTest, TestFetchFlowMsgTimeout) {
  ASSERT_EQ(default_executor_->Initialize(), SUCCESS);
  auto &exchange_service = (MockExchangeService &)ExecutionRuntime::GetInstance()->GetExchangeService();
  auto mock_dequeue_mbuf =
    [](int32_t device_id, uint32_t queue_id, rtMbufPtr_t *m_buf, int32_t timeout) -> Status {
    return RT_ERROR_TO_GE_STATUS(ACL_ERROR_RT_QUEUE_EMPTY);
  };

  EXPECT_CALL(exchange_service, DequeueMbuf).WillRepeatedly(testing::Invoke(mock_dequeue_mbuf));

  std::vector<FlowMsgPtr> outputs;
  auto original_handle = ExecutionRuntime::handle_;
  ExecutionRuntime::handle_ = (void*)0x12345678;
  auto options_bk = GetThreadLocalContext().GetAllSessionOptions();
  auto options = options_bk;
  options.emplace(RESOURCE_CONFIG_PATH, "xxx");
  GetThreadLocalContext().SetSessionOption(options);
  GE_MAKE_GUARD(recover, [options_bk]() { GetThreadLocalContext().SetSessionOption(options_bk); });
  Status ret = default_executor_->FetchFlowMsg({0}, outputs, 1001U);
  ExecutionRuntime::handle_ = original_handle;
  ASSERT_EQ(ret, RT_ERROR_TO_GE_STATUS(ACL_ERROR_RT_QUEUE_EMPTY));
}
}  // namespace ge
