/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <thread>
#include <gtest/gtest.h>

#include "graph/build/graph_builder.h"
#include "macro_utils/dt_public_scope.h"
#include "executor/proxy_dynamic_model_executor.h"
#include "hybrid/model/hybrid_model_builder.h"
#include "graph/load/model_manager/model_manager.h"
#include "macro_utils/dt_public_unscope.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "generator/ge_generator.h"
#include "ge/ge_api.h"
#include "depends/mmpa/src/mmpa_stub.h"
#include "runtime_stub.h"
#include "dflow/base/deploy/exchange_service.h"
#include "dflow/inc/data_flow/model/pne_model.h"
#include "dflow/inc/data_flow/model/flow_model_helper.h"
#include "dflow/inc/data_flow/model/graph_model.h"
#include "stub_models.h"


namespace ge {
class MockProxyDynamicModelExecutor : public ProxyDynamicModelExecutor {
 public:
  MockProxyDynamicModelExecutor() = default;
  ~MockProxyDynamicModelExecutor() override = default;
  Status DoLoadModel(const ModelData &model_data, const ComputeGraphPtr &root_graph) override {
    // (void) DynamicModelExecutor::DoLoadModel(root_model);
    model_id_ = 0;
    aicpu_model_id_ = 1023;
    return SUCCESS;
  }

  Status DoExecuteModel(const std::vector<DataBuffer> &inputs, std::vector<DataBuffer> &outputs) override {
    GELOGD("outputs size is %zu.", outputs.size());
    DataBuffer &data_buffer = outputs[0];
    if (data_buffer.data == nullptr) {
      data_buffer.data = output_buffer_;
    }
    data_buffer.length = 8;
    auto output_values = reinterpret_cast<int32_t *>(data_buffer.data);
    output_values[0] = 222;
    output_values[1] = 666;
    GeTensorDesc tensor_desc;
    output_tensor_descs_ = {tensor_desc};
    return SUCCESS;
  }

  void UnloadModel() {
    Stop();
    (void) UnloadFromAicpuSd();
  }

 protected:
  void Dispatcher() override {
    return;
  }
 private:
  uint8_t output_buffer_[8];
};

class MockFailedProxyDynamicModelExecutor : public MockProxyDynamicModelExecutor {
 public:
  MockFailedProxyDynamicModelExecutor() = default;
  ~MockFailedProxyDynamicModelExecutor() override = default;
  Status DoExecuteModel(const std::vector<DataBuffer> &inputs, std::vector<DataBuffer> &outputs) override {
    GELOGE(FAILED, "Fail to execute model.");
    return FAILED;
  }
  Status OnInputsReady(rtMbufPtr_t req_msg_mbuf, rtMbufPtr_t resp_msg_mbuf) override {
    GELOGE(FAILED, "Fail to call OnInputsReady func.");
    return FAILED;
  }
};

class ProxyDynamicModelExecutorTest : public testing::Test {
 protected:
  void SetUp() override {
  }
  void TearDown() override {
  }

  PneModelPtr BuildRootModel(const std::vector<int64_t> &shape, bool add_align_attr = false, bool with_max_size = true) {
    vector<std::string> engine_list = {"AIcoreEngine"};
    auto data = OP_CFG(DATA).InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, shape).Attr(ATTR_NAME_INDEX, 0);
    auto neg = OP_CFG(NEG).InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, shape);
    auto netoutput = OP_CFG(NETOUTPUT).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_INT32, shape);
    DEF_GRAPH(graph) {
      CHAIN(NODE("Node_data_1", data)->EDGE(0, 0)->NODE("Neg", neg)->NODE("Node_output_1", netoutput));
    };
    auto root_graph = ToComputeGraph(graph);
    if (std::find(shape.begin(), shape.end(), -1) != shape.end()) {
      (void)AttrUtils::SetBool(root_graph, ATTR_NAME_GRAPH_UNKNOWN_FLAG, true);
    }
    if (add_align_attr) {
      auto data_node = root_graph->FindNode("Node_data_1");
      NamedAttrs align_attr;
      AttrUtils::SetInt(align_attr, ATTR_NAME_INPUTS_ALIGN_INTERVAL, 10U);
      AttrUtils::SetInt(align_attr, ATTR_NAME_INPUTS_ALIGN_OFFSET, 5U);
      AttrUtils::SetNamedAttrs(data_node->GetOpDesc(), ATTR_NAME_INPUTS_ALIGN_ATTR, align_attr);
    }
    auto netoutput_node = root_graph->FindNode("Node_output_1");
    const auto &tensor_desc = netoutput_node->GetOpDesc()->MutableInputDesc(0U);
    netoutput_node->GetOpDesc()->SetSrcIndex({0});
    netoutput_node->GetOpDesc()->SetSrcName({"neg"});
    if (with_max_size) {
      AttrUtils::SetInt(*tensor_desc, "_graph_output_max_size", 8);
    }
    PneModelPtr pne_model = StubModels::BuildRootModel(root_graph, false);
    EXPECT_NE(pne_model, nullptr);
    EXPECT_NE(pne_model->GetRootGraph(), nullptr);
    return pne_model;
  }

  GeRootModelPtr BuildRootModelWithDummy(const std::vector<int64_t> &shape, bool add_align_attr = false) {
    vector<std::string> engine_list = {"AIcoreEngine"};
    auto data = OP_CFG(DATA).InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, shape).Attr(ATTR_NAME_INDEX, 0);
    auto neg = OP_CFG(NEG).InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, shape);
    auto netoutput = OP_CFG(NETOUTPUT).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_INT32, shape);
    DEF_GRAPH(graph) {
      CHAIN(NODE("Node_data_1", data)->EDGE(0, 0)->NODE("Neg", neg)->NODE("Node_output_1", netoutput));
    };
    auto root_model = std::make_shared<GeRootModel>();
    auto root_graph = ToComputeGraph(graph);
    root_model->SetRootGraph(root_graph);
    if (add_align_attr) {
      auto data_node = root_graph->FindNode("Node_data_1");
      NamedAttrs align_attr;
      AttrUtils::SetInt(align_attr, ATTR_NAME_INPUTS_ALIGN_INTERVAL, 10U);
      AttrUtils::SetInt(align_attr, ATTR_NAME_INPUTS_ALIGN_OFFSET, 5U);
      AttrUtils::SetNamedAttrs(data_node->GetOpDesc(), ATTR_NAME_INPUTS_ALIGN_ATTR, align_attr);
    }
    auto netoutput_node = root_graph->FindNode("Node_output_1");
    const auto &tensor_desc = netoutput_node->GetOpDesc()->MutableInputDesc(0U);
    AttrUtils::SetInt(*tensor_desc, "_graph_output_max_size", 8);
    return root_model;
  }
};

TEST_F(ProxyDynamicModelExecutorTest, TestProxyDynamicModel_NotExecute) {
  auto root_model = BuildRootModel({-1});
  ModelQueueParam model_queue_param{};
  model_queue_param.input_queues = {1};
  model_queue_param.output_queues = {2};
  QueueAttrs in_queue_0 = {.queue_id = 1, .device_type = NPU, .device_id = 0};
  QueueAttrs out_queue_0 = {.queue_id = 2, .device_type = NPU, .device_id = 0};
  model_queue_param.input_queues_attrs = {in_queue_0};
  model_queue_param.output_queues_attrs = {out_queue_0};

  MockProxyDynamicModelExecutor executor;
  executor.Initialize();
  auto graph_model_ptr = std::dynamic_pointer_cast<GraphModel>(root_model);
  EXPECT_EQ(executor.LoadModel(graph_model_ptr->GetModelData(), root_model->GetRootGraph(), model_queue_param), SUCCESS);
  executor.UnloadModel();
  executor.Finalize();
}

TEST_F(ProxyDynamicModelExecutorTest, TestProxyDynamicModel_NotExecute_WithInputAlignAttrs) {
  vector<std::string> engine_list = {"AIcoreEngine"};
  auto data0 = OP_CFG(DATA).InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {-1}).Attr(ATTR_NAME_INDEX, 0);
  auto data1 = OP_CFG(DATA).InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {-1}).Attr(ATTR_NAME_INDEX, 1);
  auto neg = OP_CFG(NEG).InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {-1});
  auto netoutput = OP_CFG(NETOUTPUT).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_INT32, {-1});
  DEF_GRAPH(graph) {
    CHAIN(NODE("Node_data_1", data0)->EDGE(0, 0)->NODE("Neg", neg)->NODE("Node_output_1", netoutput));
    CHAIN(NODE("Node_data_2", data1)->EDGE(0, 1)->NODE("Neg", neg));
  };
  auto root_model = std::make_shared<GeRootModel>();
  auto root_graph = ToComputeGraph(graph);
  root_model->SetRootGraph(root_graph);
  ModelData model_data{};
  ModelBufferData model_buffer_data;
  StubModels::SaveGeRootModelToModelData(root_model, model_data, model_buffer_data);
  auto netoutput_node = root_graph->FindNode("Node_output_1");
  const auto &tensor_desc = netoutput_node->GetOpDesc()->MutableInputDesc(0U);
  AttrUtils::SetInt(*tensor_desc, "_graph_output_max_size", 8);
  ModelQueueParam model_queue_param{};
  model_queue_param.input_queues = {1, 3};
  model_queue_param.output_queues = {2};
  QueueAttrs in_queue_0 = {.queue_id = 1, .device_type = NPU, .device_id = 0};
  QueueAttrs in_queue_1 = {.queue_id = 3, .device_type = NPU, .device_id = 0};
  QueueAttrs out_queue_0 = {.queue_id = 2, .device_type = NPU, .device_id = 0};
  model_queue_param.input_queues_attrs = {in_queue_0, in_queue_1};
  model_queue_param.output_queues_attrs = {out_queue_0};
  model_queue_param.input_align_attrs = {.align_max_cache_num = 4,
                                         .align_timeout = 200,
                                         .drop_when_not_align = true};
  MockProxyDynamicModelExecutor executor;
  executor.Initialize();
  EXPECT_EQ(executor.LoadModel(model_data, root_graph, model_queue_param), SUCCESS);
  executor.UnloadModel();
  executor.Finalize();
}

TEST_F(ProxyDynamicModelExecutorTest, TestProxyDynamicModel_Execute_Success) {
  auto root_model = BuildRootModel({-1}, true);
  ModelQueueParam model_queue_param{};
  model_queue_param.input_queues = {1};
  model_queue_param.output_queues = {2};
  QueueAttrs in_queue_0 = {.queue_id = 1, .device_type = NPU, .device_id = 0};
  QueueAttrs out_queue_0 = {.queue_id = 2, .device_type = NPU, .device_id = 0};
  model_queue_param.input_queues_attrs = {in_queue_0};
  model_queue_param.output_queues_attrs = {out_queue_0};
  MockProxyDynamicModelExecutor executor;
  executor.Initialize();
  auto graph_model_ptr = std::dynamic_pointer_cast<GraphModel>(root_model);
  EXPECT_EQ(executor.LoadModel(graph_model_ptr->GetModelData(), root_model->GetRootGraph(), model_queue_param), SUCCESS);
  // prepare req_msg_mbuf
  uint64_t input_data = 0UL;
  rtMbufPtr_t req_msg_mbuf = nullptr;
  RuntimeTensorDesc input_runtime_tensor_desc{};
  input_runtime_tensor_desc.shape[0] = 1;
  input_runtime_tensor_desc.shape[1] = 2;
  input_runtime_tensor_desc.original_shape[0] = 1;
  input_runtime_tensor_desc.original_shape[1] = 2;
  input_runtime_tensor_desc.dtype = DT_FLOAT;
  input_runtime_tensor_desc.format = FORMAT_ND;
  input_runtime_tensor_desc.data_addr = reinterpret_cast<uint64_t>(&input_data);
  uint64_t output_data = 0UL;
  const size_t req_mbuf_size = sizeof(RuntimeTensorDesc) + sizeof(uint64_t);
  EXPECT_EQ(rtMbufAlloc(&req_msg_mbuf, req_mbuf_size), RT_ERROR_NONE);
  void *input_buffer = nullptr;
  EXPECT_EQ(rtMbufGetBuffAddr(req_msg_mbuf, &input_buffer), RT_ERROR_NONE);
  memcpy(input_buffer, &input_runtime_tensor_desc, sizeof(input_runtime_tensor_desc));
  uint64_t output_addr = reinterpret_cast<uintptr_t>(&output_data);
  void *output_buffer = reinterpret_cast<void *>(
      reinterpret_cast<uintptr_t>(input_buffer) + sizeof(RuntimeTensorDesc));
  memcpy(output_buffer, &output_addr, sizeof(uint64_t));
  // prepare resp_msg_mbuf
  rtMbufPtr_t resp_msg_mbuf = nullptr;
  const size_t resp_mbuf_size = sizeof(RuntimeTensorDesc);
  EXPECT_EQ(rtMbufAlloc(&resp_msg_mbuf, resp_mbuf_size), RT_ERROR_NONE);
  executor.OnInputsReady(req_msg_mbuf, resp_msg_mbuf);
  executor.UnloadModel();
  EXPECT_EQ(executor.data_ret_code_, 0);
  executor.Finalize();
}

TEST_F(ProxyDynamicModelExecutorTest, TestProxyDynamicModel_Withoumaxsize_Execute_Success) {
  auto root_model = BuildRootModel({-1}, true, false);
  ModelQueueParam model_queue_param{};
  model_queue_param.input_queues = {1};
  model_queue_param.output_queues = {2};
  QueueAttrs in_queue_0 = {.queue_id = 1, .device_type = NPU, .device_id = 0};
  QueueAttrs out_queue_0 = {.queue_id = 2, .device_type = NPU, .device_id = 0};
  model_queue_param.input_queues_attrs = {in_queue_0};
  model_queue_param.output_queues_attrs = {out_queue_0};
  MockProxyDynamicModelExecutor executor;
  executor.Initialize();
  auto graph_model_ptr = std::dynamic_pointer_cast<GraphModel>(root_model);
  EXPECT_EQ(executor.LoadModel(graph_model_ptr->GetModelData(), root_model->GetRootGraph(), model_queue_param), SUCCESS);
  // prepare req_msg_mbuf
  uint64_t input_data = 0UL;
  rtMbufPtr_t req_msg_mbuf = nullptr;
  RuntimeTensorDesc input_runtime_tensor_desc{};
  input_runtime_tensor_desc.shape[0] = 1;
  input_runtime_tensor_desc.shape[1] = 2;
  input_runtime_tensor_desc.original_shape[0] = 1;
  input_runtime_tensor_desc.original_shape[1] = 2;
  input_runtime_tensor_desc.dtype = DT_FLOAT;
  input_runtime_tensor_desc.format = FORMAT_ND;
  input_runtime_tensor_desc.data_addr = reinterpret_cast<uint64_t>(&input_data);
  uint64_t output_data = 0UL;
  const size_t req_mbuf_size = sizeof(RuntimeTensorDesc) + sizeof(uint64_t);
  EXPECT_EQ(rtMbufAlloc(&req_msg_mbuf, req_mbuf_size), RT_ERROR_NONE);
  void *input_buffer = nullptr;
  EXPECT_EQ(rtMbufGetBuffAddr(req_msg_mbuf, &input_buffer), RT_ERROR_NONE);
  memcpy(input_buffer, &input_runtime_tensor_desc, sizeof(input_runtime_tensor_desc));
  uint64_t output_addr = reinterpret_cast<uintptr_t>(&output_data);
  void *output_buffer = reinterpret_cast<void *>(
      reinterpret_cast<uintptr_t>(input_buffer) + sizeof(RuntimeTensorDesc));
  memcpy(output_buffer, &output_addr, sizeof(uint64_t));
  // prepare resp_msg_mbuf
  rtMbufPtr_t resp_msg_mbuf = nullptr;
  const size_t resp_mbuf_size = sizeof(RuntimeTensorDesc);
  EXPECT_EQ(rtMbufAlloc(&resp_msg_mbuf, resp_mbuf_size), RT_ERROR_NONE);
  executor.OnInputsReady(req_msg_mbuf, resp_msg_mbuf);
  executor.UnloadModel();
  EXPECT_EQ(executor.data_ret_code_, 0);
  executor.Finalize();
}

TEST_F(ProxyDynamicModelExecutorTest, TestProxyDynamicModel_Withoumaxsize_Execute_Failed) {
  auto root_model = BuildRootModel({-1}, true, false);
  ModelQueueParam model_queue_param{};
  model_queue_param.input_queues = {1};
  model_queue_param.output_queues = {2};
  QueueAttrs in_queue_0 = {.queue_id = 1, .device_type = NPU, .device_id = 0};
  QueueAttrs out_queue_0 = {.queue_id = 2, .device_type = NPU, .device_id = 0};
  model_queue_param.input_queues_attrs = {in_queue_0};
  model_queue_param.output_queues_attrs = {out_queue_0};
  MockProxyDynamicModelExecutor executor;
  executor.Initialize();
  g_runtime_stub_mock = "rtCpuKernelLaunchWithFlag";
  auto graph_model_ptr = std::dynamic_pointer_cast<GraphModel>(root_model);
  EXPECT_EQ(executor.LoadModel(graph_model_ptr->GetModelData(), root_model->GetRootGraph(), model_queue_param), FAILED);
  executor.Finalize();
  g_runtime_stub_mock = "";
}

TEST_F(ProxyDynamicModelExecutorTest, TestProxyDynamicModel_retcode) {
  auto root_model = BuildRootModel({-1}, true);
  ModelQueueParam model_queue_param{};
  model_queue_param.input_queues = {1};
  model_queue_param.output_queues = {2};
  QueueAttrs in_queue_0 = {.queue_id = 1, .device_type = NPU, .device_id = 0};
  QueueAttrs out_queue_0 = {.queue_id = 2, .device_type = NPU, .device_id = 0};
  model_queue_param.input_queues_attrs = {in_queue_0};
  model_queue_param.output_queues_attrs = {out_queue_0};
  MockProxyDynamicModelExecutor executor;
  executor.Initialize();
  auto graph_model_ptr = std::dynamic_pointer_cast<GraphModel>(root_model);
  EXPECT_EQ(executor.LoadModel(graph_model_ptr->GetModelData(), root_model->GetRootGraph(), model_queue_param), SUCCESS);
  // prepare req_msg_mbuf
  uint64_t input_data = 0UL;
  rtMbufPtr_t req_msg_mbuf = nullptr;
  RuntimeTensorDesc input_runtime_tensor_desc{};
  input_runtime_tensor_desc.shape[0] = 1;
  input_runtime_tensor_desc.shape[1] = 2;
  input_runtime_tensor_desc.original_shape[0] = 1;
  input_runtime_tensor_desc.original_shape[1] = 2;
  input_runtime_tensor_desc.dtype = DT_FLOAT;
  input_runtime_tensor_desc.format = FORMAT_ND;
  input_runtime_tensor_desc.data_addr = reinterpret_cast<uint64_t>(&input_data);
  uint64_t output_data = 0UL;
  const size_t req_mbuf_size = sizeof(RuntimeTensorDesc) + sizeof(uint64_t);
  EXPECT_EQ(rtMbufAlloc(&req_msg_mbuf, req_mbuf_size), RT_ERROR_NONE);
  void *input_buffer = nullptr;
  EXPECT_EQ(rtMbufGetBuffAddr(req_msg_mbuf, &input_buffer), RT_ERROR_NONE);
  memcpy(input_buffer, &input_runtime_tensor_desc, sizeof(input_runtime_tensor_desc));
  void *head_buf = nullptr;
  uint64_t head_size = 0;
  EXPECT_EQ(rtMbufGetPrivInfo(req_msg_mbuf, &head_buf, &head_size), RT_ERROR_NONE);
  if ((head_buf != nullptr) && (head_size >= sizeof(ExchangeService::MsgInfo))) {
    ExchangeService::MsgInfo *msg_info = reinterpret_cast<ExchangeService::MsgInfo *>(
        static_cast<char *>(head_buf) + head_size - sizeof(ExchangeService::MsgInfo));
    msg_info->ret_code = 9999;
  }
  uint64_t output_addr = reinterpret_cast<uintptr_t>(&output_data);
  void *output_buffer = reinterpret_cast<void *>(
      reinterpret_cast<uintptr_t>(input_buffer) + sizeof(RuntimeTensorDesc));
  memcpy(output_buffer, &output_addr, sizeof(uint64_t));
  // prepare resp_msg_mbuf
  rtMbufPtr_t resp_msg_mbuf = nullptr;
  const size_t resp_mbuf_size = sizeof(RuntimeTensorDesc);
  EXPECT_EQ(rtMbufAlloc(&resp_msg_mbuf, resp_mbuf_size), RT_ERROR_NONE);
  executor.OnInputsReady(req_msg_mbuf, resp_msg_mbuf);
  executor.UnloadModel();
  EXPECT_FALSE(executor.is_need_execute_model_);
  EXPECT_EQ(executor.data_ret_code_, 9999);
  executor.Finalize();
}

TEST_F(ProxyDynamicModelExecutorTest, TestProxyDynamicModel_null_data) {
  auto root_model = BuildRootModel({-1}, true);
  ModelQueueParam model_queue_param{};
  model_queue_param.input_queues = {1};
  model_queue_param.output_queues = {2};
  QueueAttrs in_queue_0 = {.queue_id = 1, .device_type = NPU, .device_id = 0};
  QueueAttrs out_queue_0 = {.queue_id = 2, .device_type = NPU, .device_id = 0};
  model_queue_param.input_queues_attrs = {in_queue_0};
  model_queue_param.output_queues_attrs = {out_queue_0};
  MockProxyDynamicModelExecutor executor;
  executor.Initialize();
  auto graph_model_ptr = std::dynamic_pointer_cast<GraphModel>(root_model);
  EXPECT_EQ(executor.LoadModel(graph_model_ptr->GetModelData(), root_model->GetRootGraph(), model_queue_param), SUCCESS);
  // prepare req_msg_mbuf
  uint64_t input_data = 0UL;
  rtMbufPtr_t req_msg_mbuf = nullptr;
  RuntimeTensorDesc input_runtime_tensor_desc{};
  input_runtime_tensor_desc.shape[0] = 1;
  input_runtime_tensor_desc.shape[1] = 2;
  input_runtime_tensor_desc.original_shape[0] = 1;
  input_runtime_tensor_desc.original_shape[1] = 2;
  input_runtime_tensor_desc.dtype = DT_FLOAT;
  input_runtime_tensor_desc.format = FORMAT_ND;
  input_runtime_tensor_desc.data_addr = reinterpret_cast<uint64_t>(&input_data);
  uint64_t output_data = 0UL;
  const size_t req_mbuf_size = sizeof(RuntimeTensorDesc) + sizeof(uint64_t);
  EXPECT_EQ(rtMbufAlloc(&req_msg_mbuf, req_mbuf_size), RT_ERROR_NONE);
  void *input_buffer = nullptr;
  EXPECT_EQ(rtMbufGetBuffAddr(req_msg_mbuf, &input_buffer), RT_ERROR_NONE);
  memcpy(input_buffer, &input_runtime_tensor_desc, sizeof(input_runtime_tensor_desc));
  void *head_buf = nullptr;
  uint64_t head_size = 0;
  EXPECT_EQ(rtMbufGetPrivInfo(req_msg_mbuf, &head_buf, &head_size), RT_ERROR_NONE);
  if ((head_buf != nullptr) && (head_size >= sizeof(ExchangeService::MsgInfo))) {
    ExchangeService::MsgInfo *msg_info = reinterpret_cast<ExchangeService::MsgInfo *>(
        static_cast<char *>(head_buf) + head_size - sizeof(ExchangeService::MsgInfo));
    msg_info->data_flag = 1;
  }
  uint64_t output_addr = reinterpret_cast<uintptr_t>(&output_data);
  void *output_buffer = reinterpret_cast<void *>(
      reinterpret_cast<uintptr_t>(input_buffer) + sizeof(RuntimeTensorDesc));
  memcpy(output_buffer, &output_addr, sizeof(uint64_t));
  // prepare resp_msg_mbuf
  rtMbufPtr_t resp_msg_mbuf = nullptr;
  const size_t resp_mbuf_size = sizeof(RuntimeTensorDesc);
  EXPECT_EQ(rtMbufAlloc(&resp_msg_mbuf, resp_mbuf_size), RT_ERROR_NONE);
  executor.OnInputsReady(req_msg_mbuf, resp_msg_mbuf);
  executor.UnloadModel();
  EXPECT_FALSE(executor.is_need_execute_model_);
  executor.Finalize();
}

TEST_F(ProxyDynamicModelExecutorTest, TestProxyDynamicModel_Execute_Dummy) {
  auto root_model = BuildRootModel({-1}, true);
  ModelQueueParam model_queue_param{};
  model_queue_param.input_queues = {1};
  model_queue_param.output_queues = {UINT32_MAX};
  QueueAttrs in_queue_0 = {.queue_id = 1, .device_type = NPU, .device_id = 0};
  QueueAttrs out_queue_0 = {.queue_id = UINT32_MAX, .device_type = NPU, .device_id = 0};
  model_queue_param.input_queues_attrs = {in_queue_0};
  model_queue_param.output_queues_attrs = {out_queue_0};
  MockProxyDynamicModelExecutor executor;
  executor.Initialize();
  auto graph_model_ptr = std::dynamic_pointer_cast<GraphModel>(root_model);
  EXPECT_EQ(executor.LoadModel(graph_model_ptr->GetModelData(), root_model->GetRootGraph(), model_queue_param), SUCCESS);
  // prepare req_msg_mbuf
  uint64_t input_data = 0UL;
  rtMbufPtr_t req_msg_mbuf = nullptr;
  RuntimeTensorDesc input_runtime_tensor_desc{};
  input_runtime_tensor_desc.shape[0] = 1;
  input_runtime_tensor_desc.shape[1] = 2;
  input_runtime_tensor_desc.original_shape[0] = 1;
  input_runtime_tensor_desc.original_shape[1] = 2;
  input_runtime_tensor_desc.dtype = DT_FLOAT;
  input_runtime_tensor_desc.format = FORMAT_ND;
  input_runtime_tensor_desc.data_addr = reinterpret_cast<uint64_t>(&input_data);
  const size_t req_mbuf_size = sizeof(RuntimeTensorDesc);
  EXPECT_EQ(rtMbufAlloc(&req_msg_mbuf, req_mbuf_size), RT_ERROR_NONE);
  void *input_buffer = nullptr;
  EXPECT_EQ(rtMbufGetBuffAddr(req_msg_mbuf, &input_buffer), RT_ERROR_NONE);
  memcpy(input_buffer, &input_runtime_tensor_desc, sizeof(input_runtime_tensor_desc));
  // prepare resp_msg_mbuf
  rtMbufPtr_t resp_msg_mbuf = nullptr;
  const size_t resp_mbuf_size = sizeof(RuntimeTensorDesc);
  EXPECT_EQ(rtMbufAlloc(&resp_msg_mbuf, resp_mbuf_size), RT_ERROR_NONE);
  executor.OnInputsReady(req_msg_mbuf, resp_msg_mbuf);
  executor.UnloadModel();
  EXPECT_EQ(executor.data_ret_code_, 0);
  executor.Finalize();
}

TEST_F(ProxyDynamicModelExecutorTest, TestProxyDynamicModelWithStaticOut_Execute_Success) {
  auto root_model = BuildRootModel({2}, true);
  ModelQueueParam model_queue_param{};
  model_queue_param.input_queues = {1};
  model_queue_param.output_queues = {2};
  QueueAttrs in_queue_0 = {.queue_id = 1, .device_type = NPU, .device_id = 0};
  QueueAttrs out_queue_0 = {.queue_id = 2, .device_type = NPU, .device_id = 0};
  model_queue_param.input_queues_attrs = {in_queue_0};
  model_queue_param.output_queues_attrs = {out_queue_0};
  MockProxyDynamicModelExecutor executor;
  executor.Initialize();
  auto graph_model_ptr = std::dynamic_pointer_cast<GraphModel>(root_model);
  EXPECT_EQ(executor.LoadModel(graph_model_ptr->GetModelData(), root_model->GetRootGraph(), model_queue_param), SUCCESS);
  executor.UnloadModel();
  executor.Finalize();
}

TEST_F(ProxyDynamicModelExecutorTest, TestProxyDynamicModel_Execute_Failed) {
  auto root_model = BuildRootModel({-1});
  ModelQueueParam model_queue_param{};
  model_queue_param.input_queues = {1};
  model_queue_param.output_queues = {2};
  QueueAttrs in_queue_0 = {.queue_id = 1, .device_type = NPU, .device_id = 0};
  QueueAttrs out_queue_0 = {.queue_id = 2, .device_type = NPU, .device_id = 0};
  model_queue_param.input_queues_attrs = {in_queue_0};
  model_queue_param.output_queues_attrs = {out_queue_0};
  MockFailedProxyDynamicModelExecutor executor;
  executor.Initialize();
  auto graph_model_ptr = std::dynamic_pointer_cast<GraphModel>(root_model);
  EXPECT_EQ(executor.LoadModel(graph_model_ptr->GetModelData(), root_model->GetRootGraph(), model_queue_param), SUCCESS);
  // prepare req_msg_mbuf
  uint64_t input_data = 0UL;
  rtMbufPtr_t req_msg_mbuf = nullptr;
  RuntimeTensorDesc input_runtime_tensor_desc{};
  input_runtime_tensor_desc.shape[0] = 1;
  input_runtime_tensor_desc.shape[1] = 2;
  input_runtime_tensor_desc.original_shape[0] = 1;
  input_runtime_tensor_desc.original_shape[1] = 2;
  input_runtime_tensor_desc.dtype = DT_FLOAT;
  input_runtime_tensor_desc.format = FORMAT_ND;
  input_runtime_tensor_desc.data_addr = reinterpret_cast<uint64_t>(&input_data);
  uint64_t output_data = 0UL;
  const size_t req_mbuf_size = sizeof(RuntimeTensorDesc) + sizeof(uint64_t);
  EXPECT_EQ(rtMbufAlloc(&req_msg_mbuf, req_mbuf_size), RT_ERROR_NONE);
  void *input_buffer = nullptr;
  EXPECT_EQ(rtMbufGetBuffAddr(req_msg_mbuf, &input_buffer), RT_ERROR_NONE);
  memcpy(input_buffer, &input_runtime_tensor_desc, sizeof(input_runtime_tensor_desc));
  uint64_t output_addr = reinterpret_cast<uintptr_t>(&output_data);
  void *output_buffer = reinterpret_cast<void *>(
      reinterpret_cast<uintptr_t>(input_buffer) + sizeof(RuntimeTensorDesc));
  memcpy(output_buffer, &output_addr, sizeof(uint64_t));
  // prepare resp_msg_mbuf
  rtMbufPtr_t resp_msg_mbuf = nullptr;
  const size_t resp_mbuf_size = sizeof(RuntimeTensorDesc);
  EXPECT_EQ(rtMbufAlloc(&resp_msg_mbuf, resp_mbuf_size), RT_ERROR_NONE);
  executor.ProxyDynamicModelExecutor::OnInputsReady(req_msg_mbuf, resp_msg_mbuf);
  executor.UnloadModel();
  EXPECT_NE(executor.data_ret_code_, 0);
  executor.Finalize();
}

TEST_F(ProxyDynamicModelExecutorTest, TestPrepareMsgMbuf_Success) {
  MockProxyDynamicModelExecutor executor;
  executor.dispatcher_running_flag_ = true;
  void *req_msg_mbuf = nullptr;
  void *resp_msg_mbuf = nullptr;
  EXPECT_EQ(executor.PrepareMsgMbuf(req_msg_mbuf, resp_msg_mbuf), SUCCESS);
  EXPECT_NE(req_msg_mbuf, nullptr);
  EXPECT_NE(resp_msg_mbuf, nullptr);
  rtMbufFree(req_msg_mbuf);
  rtMbufFree(resp_msg_mbuf);
}

TEST_F(ProxyDynamicModelExecutorTest, TestDispatcher_Failed) {
  MockFailedProxyDynamicModelExecutor executor;
  executor.dispatcher_running_flag_ = true;
  executor.ProxyDynamicModelExecutor::Dispatcher();
  EXPECT_EQ(executor.dispatcher_running_flag_, false);
  EXPECT_EQ(executor.task_queue_.Size(), 0);
}

class TestPeekFailedMockRuntime : public RuntimeStub {
public:
  rtError_t rtMemQueuePeek(int32_t device, uint32_t qid, size_t *bufLen, int32_t timeout) override {
    *bufLen = 0;
    return ACL_ERROR_RT_PARAM_INVALID;
  }
};

TEST_F(ProxyDynamicModelExecutorTest, TestPrepareMsgMbuf_failed) {
  TestPeekFailedMockRuntime mock_runtime;
  RuntimeStub::Install(&mock_runtime);
  MockProxyDynamicModelExecutor executor;
  executor.dispatcher_running_flag_ = true;
  void *req_msg_mbuf = nullptr;
  void *resp_msg_mbuf = nullptr;
  EXPECT_NE(executor.PrepareMsgMbuf(req_msg_mbuf, resp_msg_mbuf), SUCCESS);
  EXPECT_EQ(req_msg_mbuf, nullptr);
  EXPECT_EQ(resp_msg_mbuf, nullptr);
  RuntimeStub::UnInstall(&mock_runtime);
}

TEST_F(ProxyDynamicModelExecutorTest, ExceptionNotify) {
  MockProxyDynamicModelExecutor executor;
  // proxy dynamic model executor no need notify exception.
  EXPECT_EQ(executor.ExceptionNotify(0, 100), SUCCESS);
}
}
