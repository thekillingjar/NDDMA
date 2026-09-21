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
#include <iostream>
#include <gtest/gtest.h>

#include "ge_common/ge_api_types.h"
#include "macro_utils/dt_public_scope.h"
#include "executor/dynamic_model_executor.h"
#include "executor/cpu_sched_event_dispatcher.h"
#include "executor/cpu_sched_model_builder.h"
#include "executor/executor_context.h"
#include "common/dump/dump_manager.h"
#include "graph/load/model_manager/model_manager.h"
#include "hybrid/model/hybrid_model_builder.h"
#include "macro_utils/dt_public_unscope.h"

#include "graph/build/graph_builder.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "generator/ge_generator.h"
#include "ge/ge_api.h"
#include "depends/mmpa/src/mmpa_stub.h"
#include "depends/runtime/src/runtime_stub.h"
#include "common/data_flow/queue/heterogeneous_exchange_service.h"
#include "dflow/inc/data_flow/model/flow_model_helper.h"
#include "dflow/inc/data_flow/model/graph_model.h"
#include "stub_models.h"

using namespace std;

namespace ge {
class MockMemQueueEnQueue : public RuntimeStub {
 public:
  rtError_t rtMemQueueEnQueue(int32_t device, uint32_t qid, void *mbuf) {
    return ret_;
  }

  void SetErrorCode(rtError_t ret) {
    ret_ = ret;
  }

private:
  rtError_t ret_ = RT_ERROR_NONE;
};

class RuntimeMock : public RuntimeStub {
 public:
  rtError_t rtModelCreate(rtModel_t *model, uint32_t flag) {
    *model = new uint32_t;
    return RT_ERROR_NONE;
  }
};

class MockDynamicModelExecutor : public DynamicModelExecutor {
 public:
  MockDynamicModelExecutor(bool is_host) : DynamicModelExecutor(is_host) {}
  Status DoLoadModel(const ModelData &model_data, const ComputeGraphPtr &root_graph) override {
    (void) DynamicModelExecutor::DoLoadModel(model_data, root_graph);
    model_id_ = 0;
    aicpu_model_id_ = 1023;
    return SUCCESS;
  }

  Status DoExecuteModel(const std::vector<DataBuffer> &inputs, std::vector<DataBuffer> &outputs) override {
    (void) DynamicModelExecutor::DoExecuteModel(inputs, outputs);
    DataBuffer &data_buffer = outputs[0];
    if (data_buffer.data == nullptr) {	
      data_buffer.data = output_buffer_;	
    }
    data_buffer.length = 8;
    auto output_values = reinterpret_cast<int32_t *>(data_buffer.data);
    output_values[0] = 222;
    output_values[1] = 666;
    GeTensorDesc tensor_desc;
    tensor_desc.SetShape(GeShape({2}));
    output_tensor_descs_ = {tensor_desc};
    return SUCCESS;
  }

 private:
  uint8_t output_buffer_[8];
};

namespace {
int32_t AicpuLoadModelWithQStub(void *ptr) {
  (void) ptr;
  return 0;
}

int32_t AICPUModelDestroyStub(uint32_t modelId) {
  (void) modelId;
  return 0;
}

int32_t InitCpuSchedulerStub(const CpuSchedInitParam *const initParam) {
  (void) initParam;
  return 0;
}

int32_t StopCPUSchedulerStub(const uint32_t deviceId, const pid_t hostPid) {
  (void) deviceId;
  (void) hostPid;
  return 0;
}

int32_t AicpuLoadModelStub(void *ptr) {
  (void) ptr;
  return 0;
}

int32_t AICPUModelStopStub(const ReDeployConfig *const reDeployConfig) {
  (void) reDeployConfig;
  return 0;
}

int32_t AICPUModelClearInputAndRestartStub(const ReDeployConfig *const reDeployConfig) {
  (void) reDeployConfig;
  return 0;
}
namespace {
  int32_t gkernel_support = 0;
  std::string ginterface_not_support;
}
int32_t AICPUModelCheckKernelSupportedStub(const CheckKernelSupportedConfig * const cfgPtr) {
  if (gkernel_support == 0) { // supported
    *(reinterpret_cast<int32_t *>(reinterpret_cast<uintptr_t>(cfgPtr->checkResultAddr))) = 0;
  } else if (gkernel_support == 1) { // not supported
    *(reinterpret_cast<int32_t *>(reinterpret_cast<uintptr_t>(cfgPtr->checkResultAddr))) = 100;
  } else if (gkernel_support == 2) { // interface error
    return -1;
  } else {
    return 0;
  }
  return 0;
}

int32_t AICPUModelProcessDataExceptionStub(const DataFlowExceptionNotify *const notify) {
  (void) notify;
  return 0;
}

class MockMmpa : public MmpaStubApiGe {
 public:
  void *DlOpen(const char *file_name, int32_t mode) {
    if (std::string(file_name) == "libaicpu_scheduler.so" ||
        std::string(file_name) == "libhost_aicpu_scheduler.so") {
      return (void *) 0x12345678;
    }
    return dlopen(file_name, mode);
  }

  void *DlSym(void *handle, const char *func_name) override {
    if (ginterface_not_support == std::string(func_name)) {
      return nullptr;
    }
    if (std::string(func_name) == "AicpuLoadModelWithQ") {
      return (void *) &AicpuLoadModelWithQStub;
    } else if (std::string(func_name) == "AICPUModelDestroy") {
      return (void *) &AICPUModelDestroyStub;
    } else if (std::string(func_name) == "InitCpuScheduler") {
      return (void *) &InitCpuSchedulerStub;
    } else if (std::string(func_name) == "AicpuLoadModel") {
      return (void *) &AicpuLoadModelStub;
    } else if (std::string(func_name) == "StopCPUScheduler") {
      return (void *) &StopCPUSchedulerStub;
    } else if (std::string(func_name) == "AICPUModelStop") {
      return (void *) &AICPUModelStopStub;
    } else if (std::string(func_name) == "AICPUModelClearInputAndRestart") {
      return (void *) &AICPUModelClearInputAndRestartStub;
    } else if (std::string(func_name) == "CheckKernelSupported") {
      return (void *) &AICPUModelCheckKernelSupportedStub;
    } else if (std::string(func_name) == "AICPUModelProcessDataException") {
      return (void *) &AICPUModelProcessDataExceptionStub;
    }

    return dlsym(handle, func_name);
  }

  int32_t DlClose(void *handle) override {
    if (handle == (void *) 0x12345678) {
      return 0;
    }
    return dlclose(handle);
  }
};
}

class DynamicModelExecutorTest : public testing::Test {
 protected:
  void SetUp() override {
  }
  void TearDown() override {
  }

  PneModelPtr BuildRootModelWithFileConstant(const std::vector<int64_t> &shape, bool add_align_attr = false) {
    vector<std::string> engine_list = {"AIcoreEngine"};
    auto data1 = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, shape)
        .Attr(ATTR_NAME_INDEX, 0);
    (void) system("echo 1 > hello.bin");
    auto neg = OP_CFG(FILECONSTANT).InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, shape)
                          .Attr(ATTR_NAME_INDEX, 0)
                          .Attr(ATTR_NAME_LOCATION, "hello.bin")
                          .Attr(ATTR_NAME_OFFSET, 0)
                          .Attr(ATTR_NAME_LENGTH, 2);
    auto netoutput = OP_CFG(NETOUTPUT).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_INT32, shape);
    DEF_GRAPH(g1) {
      CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("neg", neg)->NODE("Node_Output", netoutput));
    };
    auto root_graph = ToComputeGraph(g1);
    if (std::find(shape.begin(), shape.end(), -1) != shape.end()) {
      (void)AttrUtils::SetBool(root_graph, ATTR_NAME_GRAPH_UNKNOWN_FLAG, true);
    }
    auto output_node = root_graph->FindNode("Node_Output");
    output_node->GetOpDesc()->SetSrcIndex({0});
    output_node->GetOpDesc()->SetSrcName({"neg"});

    if (add_align_attr) {
      auto data_node = root_graph->FindNode("data_1");
      NamedAttrs align_attr;
      AttrUtils::SetInt(align_attr, ATTR_NAME_INPUTS_ALIGN_INTERVAL, 10U);
      AttrUtils::SetInt(align_attr, ATTR_NAME_INPUTS_ALIGN_OFFSET, 5U);
      AttrUtils::SetNamedAttrs(data_node->GetOpDesc(), ATTR_NAME_INPUTS_ALIGN_ATTR, align_attr);
    }
    PneModelPtr pne_model = StubModels::BuildRootModel(root_graph, false);
    return pne_model;
  }

  PneModelPtr BuildRootModel(const std::vector<int64_t> &shape, bool add_align_attr = false) {
    vector<std::string> engine_list = {"AIcoreEngine"};
    auto data1 = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, shape)
        .Attr(ATTR_NAME_INDEX, 0);

    auto neg = OP_CFG(NEG).InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, shape);
    auto netoutput = OP_CFG(NETOUTPUT).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_INT32, shape);
    DEF_GRAPH(g1) {
      CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("neg", neg)->NODE("Node_Output", netoutput));
    };
    auto root_graph = ToComputeGraph(g1);
    if (std::find(shape.begin(), shape.end(), -1) != shape.end()) {
      (void)AttrUtils::SetBool(root_graph, ATTR_NAME_GRAPH_UNKNOWN_FLAG, true);
    }
    auto output_node = root_graph->FindNode("Node_Output");
    output_node->GetOpDesc()->SetSrcIndex({0});
    output_node->GetOpDesc()->SetSrcName({"neg"});

    if (add_align_attr) {
      auto data_node = root_graph->FindNode("data_1");
      NamedAttrs align_attr;
      AttrUtils::SetInt(align_attr, ATTR_NAME_INPUTS_ALIGN_INTERVAL, 10U);
      AttrUtils::SetInt(align_attr, ATTR_NAME_INPUTS_ALIGN_OFFSET, 5U);
      AttrUtils::SetNamedAttrs(data_node->GetOpDesc(), ATTR_NAME_INPUTS_ALIGN_ATTR, align_attr);
    }
    PneModelPtr pne_model = StubModels::BuildRootModel(root_graph, false);
    return pne_model;
  }
};

TEST_F(DynamicModelExecutorTest, TestDynamicModelWithFileConstantWithDummyQ) {
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  auto &dispatcher = CpuSchedEventDispatcher::GetInstance();
  EXPECT_EQ(dispatcher.Initialize(0, false), SUCCESS);

  rtMbufPtr_t input_mbuf = nullptr;
  RuntimeTensorDesc input_runtime_tensor_desc{};
  input_runtime_tensor_desc.shape[0] = 1;
  input_runtime_tensor_desc.shape[1] = 2;
  input_runtime_tensor_desc.original_shape[0] = 1;
  input_runtime_tensor_desc.original_shape[1] = 2;
  input_runtime_tensor_desc.dtype = DT_FLOAT;
  input_runtime_tensor_desc.format = FORMAT_ND;
  rtMbufAlloc(&input_mbuf, sizeof(input_runtime_tensor_desc) + 8);
  void *input_buffer = nullptr;
  rtMbufGetBuffAddr(input_mbuf, &input_buffer);
  memcpy(input_buffer, &input_runtime_tensor_desc, sizeof(input_runtime_tensor_desc));
  {
    auto root_model = BuildRootModelWithFileConstant({-1});
    ModelQueueParam model_queue_param{};
    model_queue_param.input_queues = {1};
    model_queue_param.output_queues = {UINT32_MAX};
    QueueAttrs in_queue_0 = {.queue_id = 1, .device_type = NPU, .device_id = 0};
    QueueAttrs out_queue_0 = {.queue_id = UINT32_MAX, .device_type = NPU, .device_id = 0};
    model_queue_param.input_queues_attrs = {in_queue_0};
    model_queue_param.output_queues_attrs = {out_queue_0};
    model_queue_param.is_dynamic_sched = true;
    model_queue_param.need_report_status = true;
    DumpProperties dump_properties;
    dump_properties.enable_dump_ = "1";
    dump_properties.dump_step_ = "0|2-4|6";
    DumpManager::GetInstance().AddDumpProperties(0, dump_properties);
    EXPECT_EQ(DumpManager::GetInstance().GetDumpProperties(0).IsDumpOpen(), true);
    MockDynamicModelExecutor executor(false);
    executor.Initialize();
    auto graph_model_ptr = std::dynamic_pointer_cast<GraphModel>(root_model);
    EXPECT_NE(graph_model_ptr, nullptr);
    EXPECT_NE(graph_model_ptr->GetRootGraph(), nullptr);
    EXPECT_EQ(executor.LoadModel(graph_model_ptr->GetModelData(), root_model->GetRootGraph(), model_queue_param), SUCCESS);
    
    AICPUSubEventInfo event_info{};
    event_info.modelId = 1023;
    rtEschedEventSummary_t event_summary;
    event_summary.msg = (char *)&event_info;
    event_summary.msgLen = sizeof(event_info);
    executor.input_mbuf_addresses_[0] = input_mbuf;
    dispatcher.OnInputsReady(event_summary);
    executor.UnloadModel();

    EXPECT_EQ(executor.output_mbuf_addresses_.size(), 1);
    EXPECT_EQ(executor.output_mbuf_addresses_[0], nullptr);
    executor.Finalize();
    rtMbufPtr_t m_buf;
    HeterogeneousExchangeService::GetInstance().DequeueMbuf(executor.device_id_, executor.status_output_queue_id_,
                                                            &m_buf, 3000);
    rtMbufFree(m_buf);
  }
  dispatcher.Finalize();
  rtMbufFree(input_mbuf);
  MmpaStub::GetInstance().Reset();
}

TEST_F(DynamicModelExecutorTest, TestDynamicModelWithDummyQ) {
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  auto &dispatcher = CpuSchedEventDispatcher::GetInstance();
  EXPECT_EQ(dispatcher.Initialize(0, false), SUCCESS);

  rtMbufPtr_t input_mbuf = nullptr;
  RuntimeTensorDesc input_runtime_tensor_desc{};
  input_runtime_tensor_desc.shape[0] = 1;
  input_runtime_tensor_desc.shape[1] = 2;
  input_runtime_tensor_desc.original_shape[0] = 1;
  input_runtime_tensor_desc.original_shape[1] = 2;
  input_runtime_tensor_desc.dtype = DT_FLOAT;
  input_runtime_tensor_desc.format = FORMAT_ND;
  rtMbufAlloc(&input_mbuf, sizeof(input_runtime_tensor_desc) + 8);
  void *input_buffer = nullptr;
  rtMbufGetBuffAddr(input_mbuf, &input_buffer);
  memcpy(input_buffer, &input_runtime_tensor_desc, sizeof(input_runtime_tensor_desc));
  {
    auto root_model = BuildRootModel({-1});
    ModelQueueParam model_queue_param{};
    model_queue_param.input_queues = {1};
    model_queue_param.output_queues = {UINT32_MAX};
    QueueAttrs in_queue_0 = {.queue_id = 1, .device_type = NPU, .device_id = 0};
    QueueAttrs out_queue_0 = {.queue_id = UINT32_MAX, .device_type = NPU, .device_id = 0};
    model_queue_param.input_queues_attrs = {in_queue_0};
    model_queue_param.output_queues_attrs = {out_queue_0};
    model_queue_param.is_dynamic_sched = true;
    model_queue_param.need_report_status = true;
    DumpProperties dump_properties;
    dump_properties.enable_dump_ = "1";
    dump_properties.dump_step_ = "0|2-4|6";
    DumpManager::GetInstance().AddDumpProperties(0, dump_properties);
    EXPECT_EQ(DumpManager::GetInstance().GetDumpProperties(0).IsDumpOpen(), true);
    MockDynamicModelExecutor executor(false);
    executor.Initialize();
    auto graph_model_ptr = std::dynamic_pointer_cast<GraphModel>(root_model);
    EXPECT_NE(graph_model_ptr, nullptr);
    EXPECT_NE(graph_model_ptr->GetRootGraph(), nullptr);
    EXPECT_EQ(executor.LoadModel(graph_model_ptr->GetModelData(), root_model->GetRootGraph(), model_queue_param), SUCCESS);
    
    AICPUSubEventInfo event_info{};
    event_info.modelId = 1023;
    rtEschedEventSummary_t event_summary;
    event_summary.msg = (char *)&event_info;
    event_summary.msgLen = sizeof(event_info);
    executor.input_mbuf_addresses_[0] = input_mbuf;
    dispatcher.OnInputsReady(event_summary);
    executor.UnloadModel();

    EXPECT_EQ(executor.output_mbuf_addresses_.size(), 1);
    EXPECT_EQ(executor.output_mbuf_addresses_[0], nullptr);
    executor.Finalize();
    rtMbufPtr_t m_buf;
    HeterogeneousExchangeService::GetInstance().DequeueMbuf(executor.device_id_, executor.status_output_queue_id_,
                                                            &m_buf, 3000);
    rtMbufFree(m_buf);
  }
  dispatcher.Finalize();
  rtMbufFree(input_mbuf);
  MmpaStub::GetInstance().Reset();
}

TEST_F(DynamicModelExecutorTest, TestDynamicModelClearModel) {
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());

  rtMbufPtr_t input_mbuf = nullptr;
  RuntimeTensorDesc input_runtime_tensor_desc{};
  input_runtime_tensor_desc.shape[0] = 1;
  input_runtime_tensor_desc.shape[1] = 2;
  input_runtime_tensor_desc.original_shape[0] = 1;
  input_runtime_tensor_desc.original_shape[1] = 2;
  input_runtime_tensor_desc.dtype = DT_FLOAT;
  input_runtime_tensor_desc.format = FORMAT_ND;
  rtMbufAlloc(&input_mbuf, sizeof(input_runtime_tensor_desc) + 8);
  void *input_buffer = nullptr;
  rtMbufGetBuffAddr(input_mbuf, &input_buffer);
  memcpy(input_buffer, &input_runtime_tensor_desc, sizeof(input_runtime_tensor_desc));
  {
    auto root_model = BuildRootModel({-1});
    ModelQueueParam model_queue_param{};
    model_queue_param.input_queues = {1};
    model_queue_param.output_queues = {UINT32_MAX};
    QueueAttrs in_queue_0 = {.queue_id = 1, .device_type = NPU, .device_id = 0};
    QueueAttrs out_queue_0 = {.queue_id = UINT32_MAX, .device_type = NPU, .device_id = 0};
    model_queue_param.input_queues_attrs = {in_queue_0};
    model_queue_param.output_queues_attrs = {out_queue_0};
    model_queue_param.is_dynamic_sched = true;
    model_queue_param.need_report_status = true;
    DumpProperties dump_properties;
    dump_properties.enable_dump_ = "1";
    dump_properties.dump_step_ = "0|2-4|6";
    DumpManager::GetInstance().AddDumpProperties(0, dump_properties);
    EXPECT_EQ(DumpManager::GetInstance().GetDumpProperties(0).IsDumpOpen(), true);
    MockDynamicModelExecutor executor(false);
    executor.Initialize();
    auto graph_model_ptr = std::dynamic_pointer_cast<GraphModel>(root_model);
    EXPECT_EQ(executor.LoadModel(graph_model_ptr->GetModelData(), root_model->GetRootGraph(), model_queue_param), SUCCESS);

    EXPECT_EQ(executor.stop_schedule_flag_.load(), false);
    EXPECT_EQ(executor.has_stop_schedule_.load(), false);
    EXPECT_EQ(executor.ClearModel(1), SUCCESS);
    EXPECT_EQ(executor.stop_schedule_flag_.load(), true);
    EXPECT_EQ(executor.has_stop_schedule_.load(), true);
    EXPECT_EQ(executor.ClearModel(2), SUCCESS);
    EXPECT_EQ(executor.stop_schedule_flag_.load(), false);
    EXPECT_EQ(executor.has_stop_schedule_.load(), false);
  
    executor.UnloadModel();

    EXPECT_EQ(executor.output_mbuf_addresses_.size(), 1);
    EXPECT_EQ(executor.output_mbuf_addresses_[0], nullptr);
    executor.Finalize();
  }
  rtMbufFree(input_mbuf);
  MmpaStub::GetInstance().Reset();
}

TEST_F(DynamicModelExecutorTest, TestDynamicModelExceptionNotify) {
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  MockDynamicModelExecutor executor(true);
  executor.Initialize();
  Status ret = executor.CheckLocalAicpuSupportExceptionNotify();
  EXPECT_EQ(ret, SUCCESS);
  ret = executor.ExceptionNotify(0, 100);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(DynamicModelExecutorTest, TestDynamicModel) {
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  auto &dispatcher = CpuSchedEventDispatcher::GetInstance();
  EXPECT_EQ(dispatcher.Initialize(0, false), SUCCESS);

  rtMbufPtr_t input_mbuf = nullptr;
  RuntimeTensorDesc input_runtime_tensor_desc{};
  input_runtime_tensor_desc.shape[0] = 1;
  input_runtime_tensor_desc.shape[1] = 2;
  input_runtime_tensor_desc.original_shape[0] = 1;
  input_runtime_tensor_desc.original_shape[1] = 2;
  input_runtime_tensor_desc.dtype = DT_FLOAT;
  input_runtime_tensor_desc.format = FORMAT_ND;
  rtMbufAlloc(&input_mbuf, sizeof(input_runtime_tensor_desc) + 8);
  void *input_buffer = nullptr;
  rtMbufGetBuffAddr(input_mbuf, &input_buffer);
  memcpy(input_buffer, &input_runtime_tensor_desc, sizeof(input_runtime_tensor_desc));
  {
    auto root_model = BuildRootModel({-1});
    ModelQueueParam model_queue_param{};
    model_queue_param.input_queues = {1};
    model_queue_param.output_queues = {2};
    QueueAttrs in_queue_0 = {.queue_id = 1, .device_type = NPU, .device_id = 0};
    QueueAttrs out_queue_0 = {.queue_id = 2, .device_type = NPU, .device_id = 0};
    model_queue_param.input_queues_attrs = {in_queue_0};
    model_queue_param.output_queues_attrs = {out_queue_0};
    DumpProperties dump_properties;
    dump_properties.enable_dump_ = "1";
    dump_properties.dump_step_ = "0|2-4|6";
    DumpManager::GetInstance().AddDumpProperties(0, dump_properties);
    DumpManager::GetInstance().dump_flag_ = true;
    EXPECT_EQ(DumpManager::GetInstance().GetDumpProperties(0).IsDumpOpen(), true);
    DumpManager::GetInstance().RemoveDumpProperties(0);
    std::map<uint64_t, DumpProperties> infer_dump_properties_map;
    infer_dump_properties_map[kInferSessionId] = dump_properties;
    DumpManager::GetInstance().infer_dump_properties_map_ = infer_dump_properties_map;
    MockDynamicModelExecutor executor(false);
    std::map<std::string, std::string> graph_options;
    graph_options[OPTION_EXEC_DYNAMIC_GRAPH_PARALLEL_MODE] = "1";
    GetThreadLocalContext().SetGraphOption(graph_options);
    executor.Initialize();
    auto graph_model_ptr = std::dynamic_pointer_cast<GraphModel>(root_model);
    EXPECT_EQ(executor.LoadModel(graph_model_ptr->GetModelData(), root_model->GetRootGraph(), model_queue_param), SUCCESS);

    AICPUSubEventInfo event_info{};
    event_info.modelId = 1023;
    rtEschedEventSummary_t event_summary;
    event_summary.msg = (char *)&event_info;
    event_summary.msgLen = sizeof(event_info);
    executor.input_mbuf_addresses_[0] = input_mbuf;
    dispatcher.OnInputsReady(event_summary);
    executor.UnloadModel();

    uint64_t output_buffer_size = 0;
    EXPECT_NE(executor.output_mbuf_addresses_[0], nullptr);
    rtMbufGetBuffSize(executor.output_mbuf_addresses_[0], &output_buffer_size);
    EXPECT_EQ(output_buffer_size, sizeof(RuntimeTensorDesc) + 8);
    void *output_buffer = nullptr;
    rtMbufGetBuffAddr(executor.output_mbuf_addresses_[0], &output_buffer);
    auto output_desc = reinterpret_cast<RuntimeTensorDesc *>(output_buffer);

    EXPECT_EQ(output_desc->shape[0], 1);
    EXPECT_EQ(output_desc->shape[1], 2);
    auto output_values = reinterpret_cast<int32_t *>(PtrToValue(output_buffer) + sizeof(RuntimeTensorDesc));
    EXPECT_EQ(output_values[0], 222);
    EXPECT_EQ(output_values[1], 666);
    rtMbufFree(executor.output_mbuf_addresses_[0]);
    executor.Finalize();
  }
  dispatcher.Finalize();
  rtMbufFree(input_mbuf);
  MmpaStub::GetInstance().Reset();
}

TEST_F(DynamicModelExecutorTest, TestDynamicModelWithInputAlign) {
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  auto &dispatcher = CpuSchedEventDispatcher::GetInstance();
  EXPECT_EQ(dispatcher.Initialize(0, false), SUCCESS);

  {
    vector<std::string> engine_list = {"AIcoreEngine"};
    auto data0 = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {-1})
        .Attr(ATTR_NAME_INDEX, 0);
    auto data1 = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {-1})
        .Attr(ATTR_NAME_INDEX, 1);
    auto add = OP_CFG("Add").InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {-1});
    auto netoutput = OP_CFG(NETOUTPUT).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_INT32, {-1});
    DEF_GRAPH(g1) {
      CHAIN(NODE("data_0", data0)->EDGE(0, 0)->NODE("add", add)->NODE("Node_Output", netoutput));
      CHAIN(NODE("data_1", data1)->EDGE(0, 1)->NODE("add", add));
    };

    auto root_model = std::make_shared<GeRootModel>();
    auto root_graph = ToComputeGraph(g1);
    root_model->SetRootGraph(root_graph);
    ModelData model_data{};
    ModelBufferData model_buffer_data;
    StubModels::SaveGeRootModelToModelData(root_model, model_data, model_buffer_data);

    ModelQueueParam model_queue_param{};
    model_queue_param.input_queues = {1, 3};
    model_queue_param.output_queues = {2};
    QueueAttrs in_queue_0 = {.queue_id = 1, .device_type = NPU, .device_id = 0};
    QueueAttrs in_queue_1 = {.queue_id = 3, .device_type = CPU, .device_id = 0};
    QueueAttrs out_queue_0 = {.queue_id = 2, .device_type = NPU, .device_id = 0};
    model_queue_param.input_queues_attrs = {in_queue_0, in_queue_1};
    model_queue_param.output_queues_attrs = {out_queue_0};
    model_queue_param.input_align_attrs = {.align_max_cache_num = 4,
                                            .align_timeout = 200,
                                            .drop_when_not_align = true};
    DumpProperties dump_properties;
    dump_properties.enable_dump_ = "1";
    dump_properties.dump_step_ = "0|2-4|6";
    DumpManager::GetInstance().AddDumpProperties(0, dump_properties);
    EXPECT_EQ(DumpManager::GetInstance().GetDumpProperties(0).IsDumpOpen(), true);
    MockDynamicModelExecutor executor(true);
    executor.Initialize();
    EXPECT_EQ(executor.LoadModel(model_data, root_graph, model_queue_param), SUCCESS);

    executor.UnloadModel();

    executor.Finalize();
  }

  dispatcher.Finalize();
  MmpaStub::GetInstance().Reset();
}

TEST_F(DynamicModelExecutorTest, TestDynamicModelWithInputAlignFaile_NotSupported) {
  gkernel_support = 1;
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  auto &dispatcher = CpuSchedEventDispatcher::GetInstance();
  EXPECT_EQ(dispatcher.Initialize(0, false), SUCCESS);
  {
    auto root_model = BuildRootModel({-1}, true);
    ModelQueueParam model_queue_param{};
    model_queue_param.input_queues = {1};
    model_queue_param.output_queues = {2};
    QueueAttrs in_queue_0 = {.queue_id = 1, .device_type = NPU, .device_id = 0};
    QueueAttrs out_queue_0 = {.queue_id = 2, .device_type = NPU, .device_id = 0};
    model_queue_param.input_queues_attrs = {in_queue_0};
    model_queue_param.output_queues_attrs = {out_queue_0};
    model_queue_param.input_align_attrs = {.align_max_cache_num = 4,
                                            .align_timeout = 200,
                                            .drop_when_not_align = true};
    DumpProperties dump_properties;
    dump_properties.enable_dump_ = "1";
    dump_properties.dump_step_ = "0|2-4|6";
    DumpManager::GetInstance().AddDumpProperties(0, dump_properties);
    EXPECT_EQ(DumpManager::GetInstance().GetDumpProperties(0).IsDumpOpen(), true);
    MockDynamicModelExecutor executor(false);
    executor.Initialize();
    auto graph_model_ptr = std::dynamic_pointer_cast<GraphModel>(root_model);
    EXPECT_EQ(executor.LoadModel(graph_model_ptr->GetModelData(), root_model->GetRootGraph(), model_queue_param), PARAM_INVALID);

    executor.UnloadModel();
    executor.Finalize();
  }

  dispatcher.Finalize();
  MmpaStub::GetInstance().Reset();
  gkernel_support = 0;
}

TEST_F(DynamicModelExecutorTest, TestDynamicModelWithInputAlignFaile_InterfaceErr) {
  gkernel_support = 2;
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  auto &dispatcher = CpuSchedEventDispatcher::GetInstance();
  EXPECT_EQ(dispatcher.Initialize(0, false), SUCCESS);
  {
    auto root_model = BuildRootModel({-1}, true);
    ModelQueueParam model_queue_param{};
    model_queue_param.input_queues = {1};
    model_queue_param.output_queues = {2};
    QueueAttrs in_queue_0 = {.queue_id = 1, .device_type = NPU, .device_id = 0};
    QueueAttrs out_queue_0 = {.queue_id = 2, .device_type = NPU, .device_id = 0};
    model_queue_param.input_queues_attrs = {in_queue_0};
    model_queue_param.output_queues_attrs = {out_queue_0};
    model_queue_param.input_align_attrs = {.align_max_cache_num = 4,
                                            .align_timeout = 200,
                                            .drop_when_not_align = true};
    DumpProperties dump_properties;
    dump_properties.enable_dump_ = "1";
    dump_properties.dump_step_ = "0|2-4|6";
    DumpManager::GetInstance().AddDumpProperties(0, dump_properties);
    EXPECT_EQ(DumpManager::GetInstance().GetDumpProperties(0).IsDumpOpen(), true);
    MockDynamicModelExecutor executor(false);
    executor.Initialize();
    auto graph_model_ptr = std::dynamic_pointer_cast<GraphModel>(root_model);
    EXPECT_EQ(executor.LoadModel(graph_model_ptr->GetModelData(), root_model->GetRootGraph(), model_queue_param), FAILED);

    executor.UnloadModel();
    executor.Finalize();
  }

  dispatcher.Finalize();
  MmpaStub::GetInstance().Reset();
  gkernel_support = 0;
}


TEST_F(DynamicModelExecutorTest, TestDynamicModelWithInputAlignFaile_WithoutInterface) {
  ginterface_not_support = "CheckKernelSupported";
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  auto &dispatcher = CpuSchedEventDispatcher::GetInstance();
  EXPECT_EQ(dispatcher.Initialize(0, false), SUCCESS);
  {
    auto root_model = BuildRootModel({-1}, true);
    ModelQueueParam model_queue_param{};
    model_queue_param.input_queues = {1};
    model_queue_param.output_queues = {2};
    QueueAttrs in_queue_0 = {.queue_id = 1, .device_type = NPU, .device_id = 0};
    QueueAttrs out_queue_0 = {.queue_id = 2, .device_type = NPU, .device_id = 0};
    model_queue_param.input_queues_attrs = {in_queue_0};
    model_queue_param.output_queues_attrs = {out_queue_0};
    model_queue_param.input_align_attrs = {.align_max_cache_num = 4,
                                            .align_timeout = 200,
                                            .drop_when_not_align = true};
    DumpProperties dump_properties;
    dump_properties.enable_dump_ = "1";
    dump_properties.dump_step_ = "0|2-4|6";
    DumpManager::GetInstance().AddDumpProperties(0, dump_properties);
    EXPECT_EQ(DumpManager::GetInstance().GetDumpProperties(0).IsDumpOpen(), true);
    MockDynamicModelExecutor executor(false);
    executor.Initialize();
    auto graph_model_ptr = std::dynamic_pointer_cast<GraphModel>(root_model);
    EXPECT_EQ(executor.LoadModel(graph_model_ptr->GetModelData(), root_model->GetRootGraph(), model_queue_param), PARAM_INVALID);

    executor.UnloadModel();
    executor.Finalize();
  }

  dispatcher.Finalize();
  MmpaStub::GetInstance().Reset();
  ginterface_not_support.clear();
}

TEST_F(DynamicModelExecutorTest, TestDynamicModelInHostCpuEngineWithClientQFailed) {
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  auto &dispatcher = CpuSchedEventDispatcher::GetInstance();
  EXPECT_EQ(dispatcher.Initialize(0, false), SUCCESS);
  {
    auto root_model = BuildRootModel({-1});
    ModelQueueParam model_queue_param{};
    model_queue_param.input_queues = {1};
    model_queue_param.output_queues = {2};
    QueueAttrs in_queue_0 = {.queue_id = 1, .device_type = NPU, .device_id = 0};
    QueueAttrs out_queue_0 = {.queue_id = 2, .device_type = NPU, .device_id = 0};
    QueueAttrs status_queue_0 = {.queue_id = 3, .device_type = NPU, .device_id = 0};
    model_queue_param.input_queues_attrs = {in_queue_0};
    model_queue_param.output_queues_attrs = {out_queue_0};
    model_queue_param.status_output_queue = {status_queue_0};
    DumpProperties dump_properties;
    dump_properties.enable_dump_ = "1";
    dump_properties.dump_step_ = "0|2-4|6";
    DumpManager::GetInstance().AddDumpProperties(0, dump_properties);
    EXPECT_EQ(DumpManager::GetInstance().GetDumpProperties(0).IsDumpOpen(), true);
    MockDynamicModelExecutor executor(true);
    executor.Initialize();
    auto graph_model_ptr = std::dynamic_pointer_cast<GraphModel>(root_model);
    EXPECT_EQ(executor.LoadModel(graph_model_ptr->GetModelData(), root_model->GetRootGraph(), model_queue_param), SUCCESS);
    executor.UnloadModel();
    executor.Finalize();
  }

  dispatcher.Finalize();
  MmpaStub::GetInstance().Reset();
}

TEST_F(DynamicModelExecutorTest, TestDynamicModelInHostCpuEngineWithClientQAlignFailed) {
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  auto &dispatcher = CpuSchedEventDispatcher::GetInstance();
  EXPECT_EQ(dispatcher.Initialize(0, false), SUCCESS);
  {
    auto root_model = BuildRootModel({-1}, true);
    ModelQueueParam model_queue_param{};
    model_queue_param.input_queues = {1};
    model_queue_param.output_queues = {2};
    QueueAttrs in_queue_0 = {.queue_id = 1, .device_type = NPU, .device_id = 0};
    QueueAttrs out_queue_0 = {.queue_id = 2, .device_type = NPU, .device_id = 0};
    model_queue_param.input_queues_attrs = {in_queue_0};
    model_queue_param.output_queues_attrs = {out_queue_0};
    DumpProperties dump_properties;
    dump_properties.enable_dump_ = "1";
    dump_properties.dump_step_ = "0|2-4|6";
    DumpManager::GetInstance().AddDumpProperties(0, dump_properties);
    EXPECT_EQ(DumpManager::GetInstance().GetDumpProperties(0).IsDumpOpen(), true);
    MockDynamicModelExecutor executor(true);
    executor.Initialize();
    auto graph_model_ptr = std::dynamic_pointer_cast<GraphModel>(root_model);
    EXPECT_EQ(executor.LoadModel(graph_model_ptr->GetModelData(), root_model->GetRootGraph(), model_queue_param), SUCCESS);
    executor.UnloadModel();
    executor.Finalize();
  }

  dispatcher.Finalize();
  MmpaStub::GetInstance().Reset();
}

TEST_F(DynamicModelExecutorTest, TestDynamicModelInHostCpuEngineWithHostQFailed) {
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  auto &dispatcher = CpuSchedEventDispatcher::GetInstance();
  EXPECT_EQ(dispatcher.Initialize(0, false), SUCCESS);
  {
    auto root_model = BuildRootModel({-1});
    ModelQueueParam model_queue_param{};
    model_queue_param.input_queues = {1};
    model_queue_param.output_queues = {2};
    QueueAttrs in_queue_0 = {.queue_id = 1, .device_type = CPU, .device_id = 0};
    QueueAttrs out_queue_0 = {.queue_id = 2, .device_type = CPU, .device_id = 0};
    model_queue_param.input_queues_attrs = {in_queue_0};
    model_queue_param.output_queues_attrs = {out_queue_0};
    DumpProperties dump_properties;
    dump_properties.enable_dump_ = "1";
    dump_properties.dump_step_ = "0|2-4|6";
    DumpManager::GetInstance().AddDumpProperties(0, dump_properties);
    EXPECT_EQ(DumpManager::GetInstance().GetDumpProperties(0).IsDumpOpen(), true);
    MockDynamicModelExecutor executor(true);
    executor.Initialize();
    auto graph_model_ptr = std::dynamic_pointer_cast<GraphModel>(root_model);
    EXPECT_EQ(executor.LoadModel(graph_model_ptr->GetModelData(), root_model->GetRootGraph(), model_queue_param), SUCCESS);
    executor.UnloadModel();
    executor.Finalize();
  }

  dispatcher.Finalize();
  MmpaStub::GetInstance().Reset();
}

TEST_F(DynamicModelExecutorTest, TestDynamicModel_BatchDequeue) {
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  auto &dispatcher = CpuSchedEventDispatcher::GetInstance();
  EXPECT_EQ(dispatcher.Initialize(0, false), SUCCESS);

  rtMbufPtr_t input_mbuf = nullptr;
  RuntimeTensorDesc input_runtime_tensor_desc{};
  input_runtime_tensor_desc.shape[0] = 1;
  input_runtime_tensor_desc.shape[1] = 2;
  input_runtime_tensor_desc.original_shape[0] = 1;
  input_runtime_tensor_desc.original_shape[1] = 2;
  input_runtime_tensor_desc.dtype = DT_FLOAT;
  input_runtime_tensor_desc.format = FORMAT_ND;
  rtMbufAlloc(&input_mbuf, sizeof(input_runtime_tensor_desc) + 8);
  void *input_buffer = nullptr;
  rtMbufGetBuffAddr(input_mbuf, &input_buffer);
  memcpy(input_buffer, &input_runtime_tensor_desc, sizeof(input_runtime_tensor_desc));
  {
    auto root_model = BuildRootModel({-1}, true);
    ModelQueueParam model_queue_param{};
    model_queue_param.input_queues = {1};
    model_queue_param.output_queues = {2};
    QueueAttrs in_queue_0 = {.queue_id = 1, .device_type = NPU, .device_id = 0};
    QueueAttrs out_queue_0 = {.queue_id = 2, .device_type = NPU, .device_id = 0};
    model_queue_param.input_queues_attrs = {in_queue_0};
    model_queue_param.output_queues_attrs = {out_queue_0};
    DumpProperties dump_properties;
    dump_properties.enable_dump_ = "1";
    dump_properties.dump_step_ = "0|2-4|6";
    DumpManager::GetInstance().AddDumpProperties(0, dump_properties);
    EXPECT_EQ(DumpManager::GetInstance().GetDumpProperties(0).IsDumpOpen(), true);
    MockDynamicModelExecutor executor(false);
    executor.Initialize();
    auto graph_model_ptr = std::dynamic_pointer_cast<GraphModel>(root_model);
    EXPECT_EQ(executor.LoadModel(graph_model_ptr->GetModelData(), root_model->GetRootGraph(), model_queue_param), SUCCESS);

    AICPUSubEventInfo event_info{};
    event_info.modelId = 1023;
    rtEschedEventSummary_t event_summary;
    event_summary.msg = (char *)&event_info;
    event_summary.msgLen = sizeof(event_info);
    executor.input_mbuf_addresses_[0] = input_mbuf;
    dispatcher.OnInputsReady(event_summary);
    executor.UnloadModel();

    uint64_t output_buffer_size = 0;
    rtMbufGetBuffSize(executor.output_mbuf_addresses_[0], &output_buffer_size);
    EXPECT_EQ(output_buffer_size, sizeof(RuntimeTensorDesc) + 8);
    void *output_buffer = nullptr;
    rtMbufGetBuffAddr(executor.output_mbuf_addresses_[0], &output_buffer);
    auto output_desc = reinterpret_cast<RuntimeTensorDesc *>(output_buffer);

    EXPECT_EQ(output_desc->shape[0], 1);
    EXPECT_EQ(output_desc->shape[1], 2);
    auto output_values = reinterpret_cast<int32_t *>(PtrToValue(output_buffer) + sizeof(RuntimeTensorDesc));
    EXPECT_EQ(output_values[0], 222);
    EXPECT_EQ(output_values[1], 666);
    rtMbufFree(executor.output_mbuf_addresses_[0]);
    executor.Finalize();
  }

  dispatcher.Finalize();
  rtMbufFree(input_mbuf);
  MmpaStub::GetInstance().Reset();
}

TEST_F(DynamicModelExecutorTest, TestDynamicModel_ReportStatusFail) {
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  auto &dispatcher = CpuSchedEventDispatcher::GetInstance();
  EXPECT_EQ(dispatcher.Initialize(0, false), SUCCESS);

  rtMbufPtr_t input_mbuf = nullptr;
  RuntimeTensorDesc input_runtime_tensor_desc{};
  input_runtime_tensor_desc.shape[0] = 1;
  input_runtime_tensor_desc.shape[1] = 2;
  input_runtime_tensor_desc.original_shape[0] = 1;
  input_runtime_tensor_desc.original_shape[1] = 2;
  input_runtime_tensor_desc.dtype = DT_FLOAT;
  input_runtime_tensor_desc.format = FORMAT_ND;
  rtMbufAlloc(&input_mbuf, sizeof(input_runtime_tensor_desc) + 8);
  void *input_buffer = nullptr;
  rtMbufGetBuffAddr(input_mbuf, &input_buffer);
  memcpy(input_buffer, &input_runtime_tensor_desc, sizeof(input_runtime_tensor_desc));
  {
    auto root_model = BuildRootModel({-1}, true);
    ModelQueueParam model_queue_param{};
    model_queue_param.input_queues = {1};
    model_queue_param.output_queues = {2};
    QueueAttrs in_queue_0 = {.queue_id = 1, .device_type = NPU, .device_id = 0};
    QueueAttrs out_queue_0 = {.queue_id = 2, .device_type = NPU, .device_id = 0};
    model_queue_param.input_queues_attrs = {in_queue_0};
    model_queue_param.output_queues_attrs = {out_queue_0};
    model_queue_param.is_dynamic_sched = true;
    model_queue_param.need_report_status = true;
    DumpProperties dump_properties;
    dump_properties.enable_dump_ = "1";
    dump_properties.dump_step_ = "0|2-4|6";
    DumpManager::GetInstance().AddDumpProperties(0, dump_properties);
    EXPECT_EQ(DumpManager::GetInstance().GetDumpProperties(0).IsDumpOpen(), true);
    MockDynamicModelExecutor executor(false);
    executor.Initialize();
    auto mem_queue_enqueue_mock = std::make_shared<MockMemQueueEnQueue>();
    mem_queue_enqueue_mock->SetErrorCode(100);
    RuntimeStub::SetInstance(mem_queue_enqueue_mock);
    auto graph_model_ptr = std::dynamic_pointer_cast<GraphModel>(root_model);
    EXPECT_EQ(executor.LoadModel(graph_model_ptr->GetModelData(), root_model->GetRootGraph(), model_queue_param), SUCCESS);

    AICPUSubEventInfo event_info{};
    event_info.modelId = 1023;
    rtEschedEventSummary_t event_summary;
    event_summary.msg = (char *)&event_info;
    event_summary.msgLen = sizeof(event_info);
    executor.input_mbuf_addresses_[0] = input_mbuf;
    dispatcher.OnInputsReady(event_summary);
    executor.UnloadModel();
    executor.Finalize();
    rtMbufFree(executor.output_mbuf_addresses_[0]);
  }

  dispatcher.Finalize();
  rtMbufFree(input_mbuf);
  RuntimeStub::Reset();
  MmpaStub::GetInstance().Reset();
}


TEST_F(DynamicModelExecutorTest, UpdateOutputs_EmptyTensor) {
  DynamicModelExecutor executor(false);
  executor.num_outputs_ = 2;
  executor.is_output_dynamic_ = {false, true};
  executor.output_tensor_descs_.resize(2);
  executor.input_mbuf_addresses_.resize(1);
  rtMbufPtr_t input_mbuf = nullptr;
  rtMbufAlloc(&input_mbuf, 8 + sizeof(RuntimeTensorDesc));
  executor.input_mbuf_addresses_[0] = input_mbuf;
  executor.output_mbuf_addresses_.resize(2);
  executor.output_runtime_tensor_descs_.resize(2);
  std::vector<DataBuffer> model_outputs;
  model_outputs.resize(2);
  model_outputs[1].data = nullptr;
  model_outputs[1].length = 0;

  EXPECT_EQ(executor.UpdateOutputs(model_outputs), SUCCESS);
  executor.FreeOutputs();
  executor.ClearOutputs();
  rtMbufFree(input_mbuf);
}

TEST_F(DynamicModelExecutorTest, PublishErrorOutput_Success) {
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  auto &dispatcher = CpuSchedEventDispatcher::GetInstance();
  EXPECT_EQ(dispatcher.Initialize(0, true), SUCCESS);

  rtMbufPtr_t input_mbuf = nullptr;
  rtMbufAlloc(&input_mbuf, 8 + sizeof(RuntimeTensorDesc));
  {
    auto root_model = BuildRootModel({2});
    ModelQueueParam model_queue_param{};
    model_queue_param.input_queues = {1};
    model_queue_param.output_queues = {2};
    QueueAttrs in_queue_0 = {.queue_id = 1, .device_type = NPU, .device_id = 0};
    QueueAttrs out_queue_0 = {.queue_id = 2, .device_type = NPU, .device_id = 0};
    model_queue_param.input_queues_attrs = {in_queue_0};
    model_queue_param.output_queues_attrs = {out_queue_0};
    MockDynamicModelExecutor executor(false);
    executor.Initialize();
    auto graph_model_ptr = std::dynamic_pointer_cast<GraphModel>(root_model);
    EXPECT_EQ(executor.LoadModel(graph_model_ptr->GetModelData(), root_model->GetRootGraph(), model_queue_param), SUCCESS);

    AICPUSubEventInfo event_info{};
    event_info.modelId = 1023;
    rtEschedEventSummary_t event_summary;
    event_summary.msg = (char *)&event_info;
    event_summary.msgLen = sizeof(event_info);
    executor.input_mbuf_addresses_[0] = input_mbuf;
    dispatcher.OnInputsReady(event_summary);
    executor.UnloadModel();

    executor.is_need_alloc_output_mbuf_ = true;
    Status ret_param = SUCCESS;
    executor.PublishErrorOutput(ret_param);
    executor.Finalize();
    rtMbufFree(executor.output_mbuf_addresses_[0]);
  }

  dispatcher.Finalize();
  rtMbufFree(input_mbuf);
  MmpaStub::GetInstance().Reset();
}

TEST_F(DynamicModelExecutorTest, TestFusionInputs) {
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  auto &dispatcher = CpuSchedEventDispatcher::GetInstance();
  EXPECT_EQ(dispatcher.Initialize(0, true), SUCCESS);

  RuntimeTensorDesc input_runtime_tensor_desc{};
  input_runtime_tensor_desc.shape[0] = 1;
  input_runtime_tensor_desc.shape[1] = 2;
  input_runtime_tensor_desc.original_shape[0] = 1;
  input_runtime_tensor_desc.original_shape[1] = 2;
  input_runtime_tensor_desc.dtype = DT_FLOAT;
  input_runtime_tensor_desc.format = FORMAT_ND;
  rtMbufPtr_t input_mbuf0 = nullptr;
  rtMbufAlloc(&input_mbuf0, 2 * sizeof(input_runtime_tensor_desc) + 2 * 8);
  void *input_buffer0 = nullptr;
  rtMbufGetBuffAddr(input_mbuf0, &input_buffer0);
  memcpy(input_buffer0, &input_runtime_tensor_desc, sizeof(input_runtime_tensor_desc));
  auto input_buffer1 = ((uint8_t *)input_buffer0) + sizeof(input_runtime_tensor_desc) + 8;
  memcpy(input_buffer1, &input_runtime_tensor_desc, sizeof(input_runtime_tensor_desc));
  {
    auto data1 = OP_CFG(DATA).InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {-1}).Attr(ATTR_NAME_INDEX, 0);

    auto data2 = OP_CFG(DATA).InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {-1}).Attr(ATTR_NAME_INDEX, 1);

    auto add = OP_CFG("Add").InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {-1});
    auto netoutput = OP_CFG(NETOUTPUT).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_INT32, {-1});
    DEF_GRAPH(g1) {
      CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("Add", add)->NODE("Node_Output", netoutput));
      CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("Add", add));
    };

    auto root_model = std::make_shared<GeRootModel>();
    auto root_graph = ToComputeGraph(g1);
    root_model->SetRootGraph(root_graph);
    ModelData model_data{};
    ModelBufferData model_buffer_data;
    StubModels::SaveGeRootModelToModelData(root_model, model_data, model_buffer_data);
    ModelQueueParam model_queue_param{};
    model_queue_param.input_queues = {1, 1};
    model_queue_param.input_fusion_offsets = {0, 1};
    model_queue_param.output_queues = {2};
    QueueAttrs in_queue_0 = {.queue_id = 1, .device_type = NPU, .device_id = 0};
    QueueAttrs out_queue_0 = {.queue_id = 2, .device_type = NPU, .device_id = 0};
    model_queue_param.input_queues_attrs = {in_queue_0, in_queue_0};
    model_queue_param.output_queues_attrs = {out_queue_0};
    MockDynamicModelExecutor executor(false);
    executor.Initialize();
    EXPECT_EQ(executor.LoadModel(model_data, root_graph, model_queue_param), SUCCESS);

    AICPUSubEventInfo event_info{};
    event_info.modelId = 1023;
    rtEschedEventSummary_t event_summary;
    event_summary.msg = (char *)&event_info;
    event_summary.msgLen = sizeof(event_info);
    executor.input_mbuf_addresses_[0] = input_mbuf0;
    executor.input_mbuf_addresses_[1] = input_mbuf0;
    dispatcher.OnInputsReady(event_summary);
    executor.UnloadModel();

    uint64_t output_buffer_size = 0;
    rtMbufGetBuffSize(executor.output_mbuf_addresses_[0], &output_buffer_size);
    EXPECT_EQ(output_buffer_size, 8 + sizeof(RuntimeTensorDesc));
    int32_t *output_buffer = nullptr;
    rtMbufGetBuffAddr(executor.output_mbuf_addresses_[0], (void **)&output_buffer);
    output_buffer = reinterpret_cast<int32_t *>(PtrToValue(output_buffer) + sizeof(RuntimeTensorDesc));
    EXPECT_EQ(output_buffer[0], 222);
    EXPECT_EQ(output_buffer[1], 666);
    rtMbufFree(executor.output_mbuf_addresses_[0]);
    executor.Finalize();
  }

  dispatcher.Finalize();
  rtMbufFree(input_mbuf0);
  MmpaStub::GetInstance().Reset();
}

TEST_F(DynamicModelExecutorTest, PublishOutputWithoutExecuteEventContinue) {
  DynamicModelExecutor executor(true);
  executor.num_outputs_ = 1U;
  executor.output_events_num_ = 1U;
  EXPECT_EQ(executor.PublishOutputWithoutExecute(), SUCCESS);
}

TEST_F(DynamicModelExecutorTest, PrepareInputsIsEventInput) {
  void *host_input_buf = nullptr;
  rtMallocHost(&host_input_buf, 16, 0);
  ASSERT_NE(host_input_buf, nullptr);
  DynamicModelExecutor executor(true);
  executor.num_inputs_ = 1;
  executor.input_events_num_ = 1;
  GeTensorDesc ge_tensor_desc;
  TensorUtils::SetSize(ge_tensor_desc, 0x16);
  executor.input_tensor_descs_.emplace_back(ge_tensor_desc);
  executor.input_buf_addresses_.emplace_back((void *)host_input_buf);
  std::vector<DataBuffer> model_inputs;
  executor.PrepareInputs(model_inputs);
  ASSERT_EQ(model_inputs.size(), 1);
  EXPECT_EQ(model_inputs[0].data, host_input_buf);
  EXPECT_EQ(model_inputs[0].placement, kPlacementHost);
  EXPECT_EQ(model_inputs[0].length, 512);
  rtFreeHost(host_input_buf);
}
TEST_F(DynamicModelExecutorTest, PrepareOutputsOfDummyQ) {
  DynamicModelExecutor executor(true);
  executor.num_outputs_ = 1;
  executor.output_queues_num_ = 1;
  executor.output_tensor_sizes_.emplace_back(16);
  executor.output_buf_addresses_.resize(1);
  QueueAttrs attr;
  attr.queue_id = UINT32_MAX;
  executor.output_queue_attrs_.emplace_back(attr);
  std::vector<DataBuffer> model_outputs;
  executor.PrepareOutputs(model_outputs);
  ASSERT_EQ(model_outputs.size(), 1);
}

TEST_F(DynamicModelExecutorTest, TestCreateFakeAicpuModelAndStreamSuccess) {
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  DynamicModelExecutor executor(true);
  executor.model_id_ = 1U;
  ASSERT_EQ(executor.CreateFakeAicpuModelAndStream(), SUCCESS);
  // test npu
  executor.is_host_ = false;
  ASSERT_EQ(executor.CreateFakeAicpuModelAndStream(), SUCCESS);
  executor.Finalize();
  RuntimeStub::Reset();
}
}  // namespace ge
