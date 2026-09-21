/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <atomic>
#include <list>
#include <chrono>
#include <common/scope_guard.h>
#include "model/attr_value_impl.h"
#include "securec.h"
#include "gtest/gtest.h"
#include "mockcpp/mockcpp.hpp"
#include "common/common_define.h"
#define private public

#include "execute/flow_func_executor.h"
#include "config/global_config.h"
#include "model/flow_func_model.h"
#include "ff_udf_def.pb.h"

#undef private
#include "flow_func/meta_flow_func.h"
#include "flow_func/meta_multi_func.h"
#include "flow_func/flow_func_manager.h"
#include "common/inner_error_codes.h"
#include "utils/udf_test_helper.h"
#include "reader_writer/mbuf_writer.h"
#include "reader_writer/queue_wrapper.h"

namespace FlowFunc {

namespace {
void *halMbufGetBuffAddrFakeAddr = nullptr;
uint64_t halMbufGetBuffAddrFakeLen = 0UL;
struct MbufImpl {
  uint32_t mbuf_size;
  uint8_t reserve_head[256];
  RuntimeTensorDesc tensor_desc;
  uint8_t data[1024];
};
int halMbufAllocStub(uint64_t size, Mbuf **mbuf) {
  MbufImpl *mbuf_impl = new (std::nothrow) MbufImpl();
  memset_s(mbuf_impl, sizeof(MbufImpl), 0, sizeof(MbufImpl));
  mbuf_impl->mbuf_size = size;
  *mbuf = reinterpret_cast<Mbuf *>(mbuf_impl);
  return DRV_ERROR_NONE;
}

int halMbufAllocStubSimple(uint64_t size, Mbuf **mbuf) {
  *mbuf = (Mbuf *)100;
  return DRV_ERROR_NONE;
}

int halMbufFreeStub(Mbuf *mbuf) {
  MbufImpl *mbuf_impl = reinterpret_cast<MbufImpl *>(mbuf);
  delete mbuf_impl;
  return DRV_ERROR_NONE;
}
int halMbufGetBuffAddrStub(Mbuf *mbuf, void **buf) {
  MbufImpl *mbuf_impl = reinterpret_cast<MbufImpl *>(mbuf);
  *buf = &(mbuf_impl->tensor_desc);
  return DRV_ERROR_NONE;
}

int32_t halMbufGetBuffAddrStubSimple(Mbuf *mbuf, void **buf) {
  *buf = halMbufGetBuffAddrFakeAddr;
  return DRV_ERROR_NONE;
}

int halMbufGetBuffSizeStub(Mbuf *mbuf, uint64_t *total_size) {
  MbufImpl *mbuf_impl = reinterpret_cast<MbufImpl *>(mbuf);
  *total_size = mbuf_impl->mbuf_size;
  return DRV_ERROR_NONE;
}

int halMbufGetBuffSizeStubSimple(Mbuf *mbuf, uint64_t *total_size) {
  *total_size = halMbufGetBuffAddrFakeLen;
  return DRV_ERROR_NONE;
}

int halMbufSetDataLenStub(Mbuf *mbuf, uint64_t len) {
  MbufImpl *mbuf_impl = reinterpret_cast<MbufImpl *>(mbuf);
  mbuf_impl->mbuf_size = len;
  return DRV_ERROR_NONE;
}
int halMbufGetPrivInfoStub(Mbuf *mbuf, void **priv, unsigned int *size) {
  MbufImpl *mbuf_impl = reinterpret_cast<MbufImpl *>(mbuf);
  // start from first byte.
  *priv = &mbuf_impl->reserve_head;
  *size = 256;
  return DRV_ERROR_NONE;
}

int halMbufGetDataLenStub(Mbuf *mbuf, uint64_t *len) {
  MbufImpl *mbuf_impl = reinterpret_cast<MbufImpl *>(mbuf);
  *len = mbuf_impl->mbuf_size;
  return DRV_ERROR_NONE;
}

int halMbufAllocExStub(uint64_t size, unsigned int align, unsigned long flag, int grp_id, Mbuf **mbuf) {
  MbufImpl *mbuf_impl = new (std::nothrow) MbufImpl();
  memset_s(mbuf_impl, sizeof(MbufImpl), 0, sizeof(MbufImpl));
  mbuf_impl->mbuf_size = size;
  *mbuf = reinterpret_cast<Mbuf *>(mbuf_impl);
  return DRV_ERROR_NONE;
}

drvError_t halQueueDeQueueStub(unsigned int dev_id, unsigned int qid, void **mbuf) {
  Mbuf *buf = nullptr;
  (void)halMbufAllocExStub(sizeof(RuntimeTensorDesc) + 1024, 0, 0, 0, &buf);
  *mbuf = buf;
  void *data_buf = nullptr;
  halMbufGetBuffAddrStub(buf, &data_buf);
  auto tensor_desc = static_cast<RuntimeTensorDesc *>(data_buf);
  tensor_desc->dtype = static_cast<int64_t>(TensorDataType::DT_INT8);
  tensor_desc->shape[0] = 1;
  tensor_desc->shape[1] = 1024;
  return DRV_ERROR_NONE;
}

drvError_t halQueueEnQueueStub(unsigned int dev_id, unsigned int qid, void *mbuf) {
  halMbufFreeStub(static_cast<Mbuf *>(mbuf));
  return DRV_ERROR_NONE;
}

class FlowFuncStub : public MetaFlowFunc {
 public:
  FlowFuncStub() = default;

  ~FlowFuncStub() override = default;

  int32_t Init() override {
    return FLOW_FUNC_SUCCESS;
  }

  int32_t Proc(const std::vector<std::shared_ptr<FlowMsg>> &input_tensors) override {
    return FLOW_FUNC_SUCCESS;
  }
};

class MultiFlowFuncStub : public MetaMultiFunc {
 public:
  MultiFlowFuncStub() = default;
  ~MultiFlowFuncStub() override = default;

  int32_t Proc1(const std::shared_ptr<MetaRunContext> &run_context,
                const std::vector<std::shared_ptr<FlowMsg>> &input_msgs) {
    return FLOW_FUNC_SUCCESS;
  }

  int32_t Proc2(const std::shared_ptr<MetaRunContext> &run_context,
                const std::vector<std::shared_ptr<FlowMsg>> &input_msgs) {
    return FLOW_FUNC_SUCCESS;
  }
};

REGISTER_FLOW_FUNC("flow_func_executor_test", FlowFuncStub);
FLOW_FUNC_REGISTRAR(MultiFlowFuncStub)
    .RegProcFunc("multi_func_executor_test1", &MultiFlowFuncStub::Proc1)
    .RegProcFunc("multi_func_executor_test2", &MultiFlowFuncStub::Proc2);

class FlowFuncErrorStub : public MetaFlowFunc {
 public:
  FlowFuncErrorStub() = default;

  ~FlowFuncErrorStub() override = default;

  int32_t Init() override {
    return FLOW_FUNC_FAILED;
  }

  int32_t Proc(const std::vector<std::shared_ptr<FlowMsg>> &input_tensors) override {
    return FLOW_FUNC_SUCCESS;
  }
};

REGISTER_FLOW_FUNC("flow_func_executor_test_init_failed", FlowFuncErrorStub);
}  // namespace

class FlowFuncExecutorUTest : public testing::Test {
 protected:
  virtual void SetUp() {
    // reset device info
    GlobalConfig::Instance().SetPhyDeviceId(-1);
    GlobalConfig::Instance().SetDeviceId(0);
    back_is_on_device_ = GlobalConfig::Instance().IsOnDevice();
  }

  virtual void TearDown() {
    GlobalMockObject::verify();

    // reset device info
    GlobalConfig::Instance().SetPhyDeviceId(-1);
    GlobalConfig::Instance().SetDeviceId(0);
    GlobalConfig::on_device_ = back_is_on_device_;
  }

 private:
  bool back_is_on_device_ = false;
};

TEST_F(FlowFuncExecutorUTest, init_test) {
  GlobalConfig::Instance().SetMessageQueueIds(100, 101);
  MOCKER(halMbufFree).defaults().will(invoke(halMbufFreeStub));
  MOCKER(halMbufGetBuffAddr).defaults().will(invoke(halMbufGetBuffAddrStub));
  MOCKER(halMbufGetBuffSize).defaults().will(invoke(halMbufGetBuffSizeStub));
  MOCKER(halMbufSetDataLen).defaults().will(invoke(halMbufSetDataLenStub));
  MOCKER(halMbufAlloc).defaults().will(invoke(halMbufAllocStub));
  MOCKER(halQueueEnQueue).defaults().will(invoke(halQueueEnQueueStub));
  MOCKER(halQueueDeQueue).defaults().will(returnValue(DRV_ERROR_QUEUE_EMPTY));
  MOCKER(halEschedSubmitEventToThread).stubs().will(returnValue(DRV_ERROR_NO_SUBSCRIBE_THREAD));
  FlowFuncModelParam flow_func_model_param = {"./test_flow_func.proto",
                                              "model_name0",
                                              UdfTestHelper::CreateQueueDevInfos({1, 2}),
                                              UdfTestHelper::CreateQueueDevInfos({UINT32_MAX, 4}),
                                              UdfTestHelper::CreateQueueDevInfo(0),
                                              0U,
                                              false};
  std::unique_ptr<FlowFuncModel> model(new (std::nothrow) FlowFuncModel(flow_func_model_param));
  model->lib_path_ = "./models/flow_func.so";
  model->flow_func_name_ = "test_flow_func_not_found";
  model->name_ = "model0";
  model->data_flow_scope_ = "scope/";
  model->invoked_model_queue_infos_["scope/model_key"].feed_queue_infos =
      UdfTestHelper::CreateQueueDevInfos({1001, 1002});
  model->invoked_model_queue_infos_["scope/model_key"].fetch_queue_infos =
      UdfTestHelper::CreateQueueDevInfos({1003, 1004});
  model->input_align_attrs_ = {.align_max_cache_num = 10, .align_timeout = -1, .drop_when_not_align = false};
  std::vector<std::unique_ptr<FlowFuncModel>> models;
  models.emplace_back(std::move(model));
  FlowFuncModelParam flow_func_model_param2 = {"./test_flow_func.proto",
                                               "model_name1",
                                               UdfTestHelper::CreateQueueDevInfos({5, 6}),
                                               UdfTestHelper::CreateQueueDevInfos({7, 8}),
                                               UdfTestHelper::CreateQueueDevInfo(0),
                                               0U,
                                               true};
  std::unique_ptr<FlowFuncModel> model1(new (std::nothrow) FlowFuncModel(flow_func_model_param2));
  model1->lib_path_ = "./models/flow_func.so";
  model1->flow_func_name_ = "test_flow_func_not_found";
  model1->name_ = "model1";
  models.emplace_back(std::move(model1));
  FlowFuncExecutor executor;
  auto ret = executor.Init(models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = executor.Start();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  // as flow func is not found, it will stop auto.
  executor.WaitForStop();
  GlobalConfig::Instance().SetMessageQueueIds(UINT32_MAX, UINT32_MAX);
}

TEST_F(FlowFuncExecutorUTest, init_queue_test) {
  MOCKER_CPP(&FlowFuncExecutor::SendMessageByResponseQueue).stubs().will(returnValue(0));
  FlowFuncModelParam flow_func_model_param = {"./test_flow_func.proto",
                                              "model_name0",
                                              UdfTestHelper::CreateQueueDevInfos({1, 2}),
                                              UdfTestHelper::CreateQueueDevInfos({3, 4}),
                                              UdfTestHelper::CreateQueueDevInfo(0),
                                              0U,
                                              false};
  std::unique_ptr<FlowFuncModel> model(new (std::nothrow) FlowFuncModel(flow_func_model_param));
  model->model_esched_process_priority_ = 2;
  model->model_esched_event_priority_ = 2;
  model->lib_path_ = "./models/flow_func.so";
  model->flow_func_name_ = "test_flow_func_not_found";
  std::vector<std::unique_ptr<FlowFuncModel>> models;
  models.emplace_back(std::move(model));

  MOCKER(halQueueInit).stubs().will(returnValue(DRV_ERROR_INNER_ERR)).then(returnValue(DRV_ERROR_NONE));
  MOCKER(halEschedSetPidPriority).stubs().will(returnValue(DRV_ERROR_INNER_ERR)).then(returnValue(DRV_ERROR_NONE));
  MOCKER(halEschedSetEventPriority).stubs().will(returnValue(DRV_ERROR_INNER_ERR)).then(returnValue(DRV_ERROR_NONE));
  MOCKER(halMbufFree).defaults().will(invoke(halMbufFreeStub));
  MOCKER(halMbufGetBuffAddr).defaults().will(invoke(halMbufGetBuffAddrStub));
  MOCKER(halMbufGetBuffSize).defaults().will(invoke(halMbufGetBuffSizeStub));
  MOCKER(halMbufSetDataLen).defaults().will(invoke(halMbufSetDataLenStub));
  MOCKER(halMbufAlloc).defaults().will(invoke(halMbufAllocStub));
  MOCKER(halQueueEnQueue).defaults().will(invoke(halQueueEnQueueStub));
  MOCKER(halQueueAttach).stubs().will(returnValue(DRV_ERROR_NONE));

  FlowFuncExecutor executor5;
  int32_t ret = executor5.Init(models);
  // halQueueInit error
  EXPECT_NE(ret, FLOW_FUNC_SUCCESS);

  FlowFuncExecutor executor6;
  ret = executor6.Init(models);
  // halEschedSetPidPriority error
  EXPECT_NE(ret, FLOW_FUNC_SUCCESS);

  FlowFuncExecutor executor7;
  ret = executor7.Init(models);
  // halEschedSetEventPriority error
  EXPECT_NE(ret, FLOW_FUNC_SUCCESS);

  FlowFuncExecutor executor8;
  ret = executor8.Init(models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
}

TEST_F(FlowFuncExecutorUTest, init_drv_test_with_proxy_queue) {
  GlobalConfig::Instance().SetMessageQueueIds(100, 101);
  MOCKER(halQueueDeQueue).defaults().will(returnValue(DRV_ERROR_QUEUE_EMPTY));
  const std::vector<QueueDevInfo> input_queues = {UdfTestHelper::CreateQueueDevInfo(1, Common::kDeviceTypeNpu, 1, true),
                                                  UdfTestHelper::CreateQueueDevInfo(2)};
  const std::vector<QueueDevInfo> output_queues = {
      UdfTestHelper::CreateQueueDevInfo(3, Common::kDeviceTypeNpu, 2, true), UdfTestHelper::CreateQueueDevInfo(4)};
  FlowFuncModelParam flow_func_model_param = {"./test_flow_func.proto",
                                              "model_name",
                                              input_queues,
                                              output_queues,
                                              UdfTestHelper::CreateQueueDevInfo(0),
                                              0U,
                                              false};
  std::unique_ptr<FlowFuncModel> model(new (std::nothrow) FlowFuncModel(flow_func_model_param));
  model->model_esched_process_priority_ = 2;
  model->model_esched_event_priority_ = 2;
  model->lib_path_ = "./models/flow_func.so";
  model->flow_func_name_ = "test_flow_func_not_found";
  std::vector<std::unique_ptr<FlowFuncModel>> models;
  models.emplace_back(std::move(model));

  MOCKER(drvQueryProcessHostPid)
      .stubs()
      .will(repeat(DRV_ERROR_INNER_ERR, 2))
      .then(repeat(DRV_ERROR_NO_PROCESS, 10))
      .then(returnValue(DRV_ERROR_NONE));
  MOCKER(halQueueInit).stubs().will(returnValue(DRV_ERROR_INNER_ERR)).then(returnValue(DRV_ERROR_NONE));
  MOCKER(halMbufFree).defaults().will(invoke(halMbufFreeStub));
  MOCKER(halMbufGetBuffAddr).defaults().will(invoke(halMbufGetBuffAddrStub));
  MOCKER(halMbufGetBuffSize).defaults().will(invoke(halMbufGetBuffSizeStub));
  MOCKER(halMbufSetDataLen).defaults().will(invoke(halMbufSetDataLenStub));
  MOCKER(halMbufAlloc).defaults().will(invoke(halMbufAllocStub));
  MOCKER(halQueueEnQueue).defaults().will(invoke(halQueueEnQueueStub));
  MOCKER(halQueueAttach).stubs().will(returnValue(DRV_ERROR_NONE));
  FlowFuncExecutor executor1;
  auto ret = executor1.Init(models);
  // drvQueryProcessHostPid and  halQueueInit error
  EXPECT_NE(ret, FLOW_FUNC_SUCCESS);

  FlowFuncExecutor executor2;
  ret = executor2.Init(models);
  // drvQueryProcessHostPid error and  halQueueInit success
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);

  FlowFuncExecutor executor3;
  ret = executor3.Init(models);
  // drvQueryProcessHostPid DRV_ERROR_NO_PROCESS 10 times and halQueueInit success
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  GlobalConfig::Instance().SetMessageQueueIds(UINT32_MAX, UINT32_MAX);
}

TEST_F(FlowFuncExecutorUTest, Invalid_input_output_maps_test) {
  MOCKER_CPP(&FlowFuncExecutor::SendMessageByResponseQueue).stubs().will(returnValue(0));
  FlowFuncModelParam flow_func_model_param = {"./test_flow_func.proto",
                                              "model_name",
                                              UdfTestHelper::CreateQueueDevInfos({1, 2}),
                                              UdfTestHelper::CreateQueueDevInfos({3, 4}),
                                              UdfTestHelper::CreateQueueDevInfo(0),
                                              0U,
                                              false};
  std::unique_ptr<FlowFuncModel> model(new (std::nothrow) FlowFuncModel(flow_func_model_param));
  model->lib_path_ = "./models/flow_func.so";
  model->flow_func_name_ = "flow_func_executor_test";
  std::vector<std::unique_ptr<FlowFuncModel>> models;
  models.emplace_back(std::move(model));
  // for multi func
  FlowFuncModelParam flow_func_model_param2 = {"./test_flow_func.proto",
                                               "model_name2",
                                               UdfTestHelper::CreateQueueDevInfos({5, 6, 7}),
                                               UdfTestHelper::CreateQueueDevInfos({8, 9}),
                                               UdfTestHelper::CreateQueueDevInfo(0),
                                               0U,
                                               false};
  std::unique_ptr<FlowFuncModel> model1(new (std::nothrow) FlowFuncModel(flow_func_model_param2));
  std::map<std::string, std::vector<uint32_t>> valid_input_maps;
  std::map<std::string, std::vector<uint32_t>> valid_output_maps;
  valid_input_maps["multi_func_executor_test1"] = {{0, 1}};
  valid_input_maps["multi_func_executor_test2"] = {{2}};
  valid_output_maps["multi_func_executor_test1"] = {{0, 1}};
  valid_output_maps["multi_func_executor_test2"] = {{0, 1}};
  std::map<std::string, std::vector<uint32_t>> invalid_input_maps;
  std::map<std::string, std::vector<uint32_t>> invalid_output_maps;
  invalid_input_maps["multi_func_executor_test1"] = {{0, 1}};
  invalid_input_maps["multi_func_executor_test2"] = {{3}};
  invalid_output_maps["multi_func_executor_test1"] = {{0, 2}};
  invalid_output_maps["multi_func_executor_test2"] = {{0, 2}};
  model1->lib_path_ = "./models/flow_func.so";
  model1->name_ = "model1";
  model1->multi_func_input_maps_.insert(invalid_input_maps.begin(), invalid_input_maps.end());
  model1->multi_func_output_maps_.insert(valid_output_maps.begin(), valid_output_maps.end());
  models.emplace_back(std::move(model1));
  FlowFuncExecutor executor;
  auto ret = executor.Init(models);
  EXPECT_NE(ret, FLOW_FUNC_SUCCESS);
  // invalid output maps:
  FlowFuncModelParam flow_func_model_param3 = {"./test_flow_func.proto",
                                               "model_name3",
                                               UdfTestHelper::CreateQueueDevInfos({5, 6, 7}),
                                               UdfTestHelper::CreateQueueDevInfos({8, 9}),
                                               UdfTestHelper::CreateQueueDevInfo(0),
                                               0U,
                                               false};
  std::unique_ptr<FlowFuncModel> model2(new (std::nothrow) FlowFuncModel(flow_func_model_param3));
  model2->multi_func_input_maps_.insert(valid_input_maps.begin(), valid_input_maps.end());
  model2->multi_func_output_maps_.insert(invalid_output_maps.begin(), invalid_output_maps.end());
  models.clear();
  models.emplace_back(std::move(model2));
  ret = executor.Init(models);
  EXPECT_NE(ret, FLOW_FUNC_SUCCESS);
}

TEST_F(FlowFuncExecutorUTest, ProcessFlowFuncInitEvent_test) {
  MOCKER_CPP(&FlowFuncExecutor::SendMessageByResponseQueue).stubs().will(returnValue(0));
  FlowFuncModelParam flow_func_model_param = {"./test_flow_func.proto",
                                              "model_name",
                                              UdfTestHelper::CreateQueueDevInfos({1, 2}),
                                              UdfTestHelper::CreateQueueDevInfos({3, 4}),
                                              UdfTestHelper::CreateQueueDevInfo(0),
                                              0U,
                                              false};
  std::unique_ptr<FlowFuncModel> model(new (std::nothrow) FlowFuncModel(flow_func_model_param));
  model->lib_path_ = "./models/flow_func.so";
  model->flow_func_name_ = "flow_func_executor_test";
  std::vector<std::unique_ptr<FlowFuncModel>> models;
  models.emplace_back(std::move(model));
  // for multi func
  FlowFuncModelParam flow_func_model_param2 = {"./test_flow_func.proto",
                                               "model_name2",
                                               UdfTestHelper::CreateQueueDevInfos({5, 6, 7}),
                                               UdfTestHelper::CreateQueueDevInfos({8, 9}),
                                               UdfTestHelper::CreateQueueDevInfo(0),
                                               0U,
                                               false};
  std::unique_ptr<FlowFuncModel> model1(new (std::nothrow) FlowFuncModel(flow_func_model_param2));
  std::map<std::string, std::vector<uint32_t>> multi_func_input_maps;
  std::map<std::string, std::vector<uint32_t>> multi_func_output_maps;
  multi_func_input_maps["multi_func_executor_test1"] = {{0, 1}};
  multi_func_input_maps["multi_func_executor_test2"] = {{2}};
  multi_func_output_maps["multi_func_executor_test1"] = {{0, 1}};
  multi_func_output_maps["multi_func_executor_test2"] = {{0, 1}};
  model1->lib_path_ = "./models/flow_func.so";
  model1->name_ = "model1";
  model1->multi_func_input_maps_.insert(multi_func_input_maps.begin(), multi_func_input_maps.end());
  model1->multi_func_output_maps_.insert(multi_func_output_maps.begin(), multi_func_output_maps.end());
  models.emplace_back(std::move(model1));
  FlowFuncExecutor executor;
  auto ret = executor.Init(models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  event_info event;
  event.comm.event_id = static_cast<EVENT_ID>(UdfEvent::kEventIdProcessorInit);
  event.comm.subevent_id = 0;
  MOCKER(halEschedSubmitEvent).expects(exactly(1)).will(returnValue(DRV_ERROR_NONE));
  executor.ProcessEvent(event, 0);
  executor.Stop();
  executor.WaitForStop();
}

TEST_F(FlowFuncExecutorUTest, ProcessFlowFuncInitEvent_failed_with_req) {
  MOCKER_CPP(&FlowFuncExecutor::SendMessageByResponseQueue).stubs().will(returnValue(0));
  FlowFuncModelParam flow_func_model_param = {"./test_flow_func.proto",
                                              "model_name",
                                              UdfTestHelper::CreateQueueDevInfos({1, 2}),
                                              UdfTestHelper::CreateQueueDevInfos({3, 4}),
                                              UdfTestHelper::CreateQueueDevInfo(0),
                                              0U,
                                              false};
  std::unique_ptr<FlowFuncModel> model(new (std::nothrow) FlowFuncModel(flow_func_model_param));
  model->lib_path_ = "./models/flow_func.so";
  model->flow_func_name_ = "flow_func_executor_test";
  std::vector<std::unique_ptr<FlowFuncModel>> models;
  models.emplace_back(std::move(model));
  // for multi func
  FlowFuncModelParam flow_func_model_param2 = {"./test_flow_func.proto",
                                               "model_name2",
                                               UdfTestHelper::CreateQueueDevInfos({5, 6, 7}),
                                               UdfTestHelper::CreateQueueDevInfos({8, 9}),
                                               UdfTestHelper::CreateQueueDevInfo(0),
                                               0U,
                                               false};
  std::unique_ptr<FlowFuncModel> model1(new (std::nothrow) FlowFuncModel(flow_func_model_param2));
  std::map<std::string, std::vector<uint32_t>> multi_func_input_maps;
  std::map<std::string, std::vector<uint32_t>> multi_func_output_maps;
  multi_func_input_maps["multi_func_executor_test1"] = {{0, 1}};
  multi_func_input_maps["multi_func_executor_test2"] = {{2}};
  multi_func_output_maps["multi_func_executor_test1"] = {{0, 1}};
  multi_func_output_maps["multi_func_executor_test2"] = {{0, 1}};
  model1->lib_path_ = "./models/flow_func.so";
  model1->name_ = "model1";
  model1->multi_func_input_maps_.insert(multi_func_input_maps.begin(), multi_func_input_maps.end());
  model1->multi_func_output_maps_.insert(multi_func_output_maps.begin(), multi_func_output_maps.end());
  models.emplace_back(std::move(model1));
  FlowFuncExecutor executor;
  auto ret = executor.Init(models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  event_info event;
  event.comm.event_id = static_cast<EVENT_ID>(UdfEvent::kEventIdFlowFuncInit);
  event.comm.subevent_id = 0;
  MOCKER(halEschedSubmitEvent).stubs().will(returnValue(DRV_ERROR_NONE));
  executor.request_queue_wrapper_.reset(new QueueWrapper(0, 100));
  MOCKER(halQueueDeQueue).stubs().will(returnValue(100));
  executor.ProcessEvent(event, 0);
  EXPECT_FALSE(executor.running_);
  executor.Stop();
  executor.WaitForStop();
}

TEST_F(FlowFuncExecutorUTest, ProcessFlowFuncInitEvent_init_failed) {
  MOCKER_CPP(&FlowFuncExecutor::SendMessageByResponseQueue).stubs().will(returnValue(0));
  FlowFuncModelParam flow_func_model_param = {"./test_flow_func.proto",
                                              "model_name",
                                              UdfTestHelper::CreateQueueDevInfos({1, 2}),
                                              UdfTestHelper::CreateQueueDevInfos({3, 4}),
                                              UdfTestHelper::CreateQueueDevInfo(0),
                                              0U,
                                              false};
  std::unique_ptr<FlowFuncModel> model(new (std::nothrow) FlowFuncModel(flow_func_model_param));
  model->lib_path_ = "./models/flow_func.so";
  model->flow_func_name_ = "flow_func_executor_test_init_failed";
  std::vector<std::unique_ptr<FlowFuncModel>> models;
  models.emplace_back(std::move(model));
  // for multi func
  FlowFuncModelParam flow_func_model_param2 = {"./test_flow_func.proto",
                                               "model_name2",
                                               UdfTestHelper::CreateQueueDevInfos({5, 6, 7}),
                                               UdfTestHelper::CreateQueueDevInfos({8, 9}),
                                               UdfTestHelper::CreateQueueDevInfo(0),
                                               0U,
                                               false};
  std::unique_ptr<FlowFuncModel> model1(new (std::nothrow) FlowFuncModel(flow_func_model_param2));
  std::map<std::string, std::vector<uint32_t>> multi_func_input_maps;
  std::map<std::string, std::vector<uint32_t>> multi_func_output_maps;
  multi_func_input_maps["multi_func_executor_test1"] = {{0, 1}};
  multi_func_input_maps["multi_func_executor_test2"] = {{2}};
  multi_func_output_maps["multi_func_executor_test1"] = {{0, 1}};
  multi_func_output_maps["multi_func_executor_test2"] = {{0, 1}};
  model1->lib_path_ = "./models/flow_func.so";
  model1->name_ = "model1";
  model1->multi_func_input_maps_.insert(multi_func_input_maps.begin(), multi_func_input_maps.end());
  model1->multi_func_output_maps_.insert(multi_func_output_maps.begin(), multi_func_output_maps.end());
  models.emplace_back(std::move(model1));
  FlowFuncExecutor executor;
  auto ret = executor.Init(models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  event_info event;
  event.comm.event_id = static_cast<EVENT_ID>(UdfEvent::kEventIdProcessorInit);
  event.comm.subevent_id = 0;
  MOCKER(halEschedSubmitEvent).expects(once()).will(returnValue(DRV_ERROR_NONE));
  executor.ProcessEvent(event, 0);

  event_info event1;
  event1.comm.event_id = static_cast<EVENT_ID>(UdfEvent::kEventIdFlowFuncInit);
  event1.comm.subevent_id = 0;
  // flow func init failed, will never submit execute event.
  MOCKER(halEschedSubmitEvent).expects(never());
  executor.ProcessEvent(event1, 0);
  executor.Stop();
  executor.WaitForStop();
}

bool ScheduleProcFailedStub(FlowFuncProcessor *processor) {
  processor->status_ = FlowFuncProcessorStatus::kProcError;
  return false;
}

TEST_F(FlowFuncExecutorUTest, ProcessFlowFuncInitEvent_re_init) {
  MOCKER_CPP(&FlowFuncExecutor::SendMessageByResponseQueue).stubs().will(returnValue(0));
  FlowFuncModelParam flow_func_model_param = {"./test_flow_func.proto",
                                              "model_name",
                                              UdfTestHelper::CreateQueueDevInfos({1, 2}),
                                              UdfTestHelper::CreateQueueDevInfos({3, 4}),
                                              UdfTestHelper::CreateQueueDevInfo(0),
                                              0U,
                                              false};
  std::unique_ptr<FlowFuncModel> model(new (std::nothrow) FlowFuncModel(flow_func_model_param));
  model->lib_path_ = "./models/flow_func.so";
  model->flow_func_name_ = "flow_func_executor_test";
  std::vector<std::unique_ptr<FlowFuncModel>> models;
  models.emplace_back(std::move(model));
  // for multi func
  FlowFuncModelParam flow_func_model_param2 = {"./test_flow_func.proto",
                                               "model_name2",
                                               UdfTestHelper::CreateQueueDevInfos({5, 6, 7}),
                                               UdfTestHelper::CreateQueueDevInfos({8, 9}),
                                               UdfTestHelper::CreateQueueDevInfo(0),
                                               0U,
                                               false};
  std::unique_ptr<FlowFuncModel> model1(new (std::nothrow) FlowFuncModel(flow_func_model_param2));
  std::map<std::string, std::vector<uint32_t>> multi_func_input_maps;
  std::map<std::string, std::vector<uint32_t>> multi_func_output_maps;
  multi_func_input_maps["multi_func_executor_test1"] = {{0, 1}};
  multi_func_input_maps["multi_func_executor_test2"] = {{2}};
  multi_func_output_maps["multi_func_executor_test1"] = {{0, 1}};
  multi_func_output_maps["multi_func_executor_test2"] = {{0, 1}};
  model1->lib_path_ = "./models/flow_func.so";
  model1->name_ = "model1";
  model1->multi_func_input_maps_.insert(multi_func_input_maps.begin(), multi_func_input_maps.end());
  model1->multi_func_output_maps_.insert(multi_func_output_maps.begin(), multi_func_output_maps.end());
  ff::udf::AttrValue re_init_attr;
  re_init_attr.set_b(true);
  auto attr_impl = std::make_shared<AttrValueImpl>(re_init_attr);
  model1->attr_map_.emplace("need_re_init_attr", attr_impl);
  models.emplace_back(std::move(model1));
  FlowFuncExecutor executor;
  auto ret = executor.Init(models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  event_info event;
  event.comm.event_id = static_cast<EVENT_ID>(UdfEvent::kEventIdFlowFuncInit);
  event.comm.subevent_id = 0;
  MOCKER_CPP(&FlowFuncProcessor::InitFlowFunc).stubs().will(returnValue(FLOW_FUNC_ERR_INIT_AGAIN));
  MOCKER(halEschedSubmitEvent).expects(once()).will(returnValue(DRV_ERROR_NONE));
  executor.ProcessEvent(event, 0);
  executor.Stop();
  executor.WaitForStop();
}

TEST_F(FlowFuncExecutorUTest, ProcessSignleFlowFuncInitEvent_re_init) {
  MOCKER_CPP(&FlowFuncExecutor::SendMessageByResponseQueue).stubs().will(returnValue(0));
  FlowFuncModelParam flow_func_model_param = {"./test_flow_func.proto",
                                              "model_name",
                                              UdfTestHelper::CreateQueueDevInfos({1, 2}),
                                              UdfTestHelper::CreateQueueDevInfos({3, 4}),
                                              UdfTestHelper::CreateQueueDevInfo(0),
                                              0U,
                                              false};
  std::unique_ptr<FlowFuncModel> model(new (std::nothrow) FlowFuncModel(flow_func_model_param));
  model->lib_path_ = "./models/flow_func.so";
  model->flow_func_name_ = "flow_func_executor_test";
  std::vector<std::unique_ptr<FlowFuncModel>> models;
  models.emplace_back(std::move(model));
  // for multi func
  FlowFuncModelParam flow_func_model_param2 = {"./test_flow_func.proto",
                                               "model_name2",
                                               UdfTestHelper::CreateQueueDevInfos({5, 6, 7}),
                                               UdfTestHelper::CreateQueueDevInfos({8, 9}),
                                               UdfTestHelper::CreateQueueDevInfo(0),
                                               0U,
                                               false};
  std::unique_ptr<FlowFuncModel> model1(new (std::nothrow) FlowFuncModel(flow_func_model_param2));
  std::map<std::string, std::vector<uint32_t>> multi_func_input_maps;
  std::map<std::string, std::vector<uint32_t>> multi_func_output_maps;
  multi_func_input_maps["multi_func_executor_test1"] = {{0, 1}};
  multi_func_input_maps["multi_func_executor_test2"] = {{2}};
  multi_func_output_maps["multi_func_executor_test1"] = {{0, 1}};
  multi_func_output_maps["multi_func_executor_test2"] = {{0, 1}};
  model1->lib_path_ = "./models/flow_func.so";
  model1->name_ = "model1";
  model1->multi_func_input_maps_.insert(multi_func_input_maps.begin(), multi_func_input_maps.end());
  model1->multi_func_output_maps_.insert(multi_func_output_maps.begin(), multi_func_output_maps.end());
  ff::udf::AttrValue re_init_attr;
  re_init_attr.set_b(true);
  auto attr_impl = std::make_shared<AttrValueImpl>(re_init_attr);
  model1->attr_map_.emplace("need_re_init_attr", attr_impl);
  models.emplace_back(std::move(model1));
  FlowFuncExecutor executor;
  auto ret = executor.Init(models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  event_info event;
  event.comm.event_id = static_cast<EVENT_ID>(UdfEvent::kEventIdSingleFlowFuncInit);
  event.comm.subevent_id = 0;
  MOCKER_CPP(&FlowFuncProcessor::InitFlowFunc)
      .stubs()
      .will(returnValue(FLOW_FUNC_ERR_INIT_AGAIN))
      .then(returnValue(FLOW_FUNC_SUCCESS));
  // expect 2 times, 1:submit init event, 2:submit schedule event.
  MOCKER(halEschedSubmitEvent).expects(exactly(2)).will(returnValue(DRV_ERROR_NONE));
  executor.ProcessEvent(event, 0);
  executor.ProcessEvent(event, 0);
  executor.Stop();
  executor.WaitForStop();
}

TEST_F(FlowFuncExecutorUTest, ProcessFlowFuncExecuteEvent_proc_error) {
  MOCKER_CPP(&FlowFuncExecutor::SendMessageByResponseQueue).stubs().will(returnValue(0));
  FlowFuncModelParam flow_func_model_param = {"./test_flow_func.proto",
                                              "model_name",
                                              UdfTestHelper::CreateQueueDevInfos({1, 2}),
                                              UdfTestHelper::CreateQueueDevInfos({3, 4}),
                                              UdfTestHelper::CreateQueueDevInfo(0),
                                              0U,
                                              false};
  std::unique_ptr<FlowFuncModel> model(new (std::nothrow) FlowFuncModel(flow_func_model_param));
  model->lib_path_ = "./models/flow_func.so";
  model->flow_func_name_ = "flow_func_executor_test";
  std::vector<std::unique_ptr<FlowFuncModel>> models;
  models.emplace_back(std::move(model));
  // for multi func
  FlowFuncModelParam flow_func_model_param2 = {"./test_flow_func.proto",
                                               "model_name2",
                                               UdfTestHelper::CreateQueueDevInfos({5, 6, 7}),
                                               UdfTestHelper::CreateQueueDevInfos({8, 9}),
                                               UdfTestHelper::CreateQueueDevInfo(0),
                                               0U,
                                               false};
  std::unique_ptr<FlowFuncModel> model1(new (std::nothrow) FlowFuncModel(flow_func_model_param2));
  std::map<std::string, std::vector<uint32_t>> multi_func_input_maps;
  std::map<std::string, std::vector<uint32_t>> multi_func_output_maps;
  multi_func_input_maps["multi_func_executor_test1"] = {{0, 1}};
  multi_func_input_maps["multi_func_executor_test2"] = {{2}};
  multi_func_output_maps["multi_func_executor_test1"] = {{0, 1}};
  multi_func_output_maps["multi_func_executor_test2"] = {{0, 1}};
  model1->lib_path_ = "./models/flow_func.so";
  model1->name_ = "model1";
  model1->multi_func_input_maps_.insert(multi_func_input_maps.begin(), multi_func_input_maps.end());
  model1->multi_func_output_maps_.insert(multi_func_output_maps.begin(), multi_func_output_maps.end());
  models.emplace_back(std::move(model1));
  FlowFuncExecutor executor;
  auto ret = executor.Init(models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  event_info event;
  event.comm.event_id = static_cast<EVENT_ID>(UdfEvent::kEventIdFlowFuncExecute);
  event.comm.subevent_id = 0;
  MOCKER_CPP(&FlowFuncProcessor::Schedule).stubs().will(invoke(ScheduleProcFailedStub));
  MOCKER(halEschedSubmitEvent).expects(never());
  executor.ProcessEvent(event, 0);
  // proc failed, executor will exit auto
  EXPECT_FALSE(executor.running_);
  executor.WaitForStop();
}

TEST_F(FlowFuncExecutorUTest, ProcessFlowFuncExecuteEvent_no_need_reschedule) {
  MOCKER_CPP(&FlowFuncExecutor::SendMessageByResponseQueue).stubs().will(returnValue(0));
  FlowFuncModelParam flow_func_model_param = {"./test_flow_func.proto",
                                              "model_name",
                                              UdfTestHelper::CreateQueueDevInfos({1, 2}),
                                              UdfTestHelper::CreateQueueDevInfos({3, 4}),
                                              UdfTestHelper::CreateQueueDevInfo(0),
                                              0U,
                                              false};
  std::unique_ptr<FlowFuncModel> model(new (std::nothrow) FlowFuncModel(flow_func_model_param));
  model->lib_path_ = "./models/flow_func.so";
  model->flow_func_name_ = "flow_func_executor_test";
  std::vector<std::unique_ptr<FlowFuncModel>> models;
  models.emplace_back(std::move(model));
  // for multi func
  FlowFuncModelParam flow_func_model_param2 = {"./test_flow_func.proto",
                                               "model_name2",
                                               UdfTestHelper::CreateQueueDevInfos({5, 6, 7}),
                                               UdfTestHelper::CreateQueueDevInfos({8, 9}),
                                               UdfTestHelper::CreateQueueDevInfo(0),
                                               0U,
                                               false};
  std::unique_ptr<FlowFuncModel> model1(new (std::nothrow) FlowFuncModel(flow_func_model_param2));
  std::map<std::string, std::vector<uint32_t>> multi_func_input_maps;
  std::map<std::string, std::vector<uint32_t>> multi_func_output_maps;
  multi_func_input_maps["multi_func_executor_test1"] = {{0, 1}};
  multi_func_input_maps["multi_func_executor_test2"] = {{2}};
  multi_func_output_maps["multi_func_executor_test1"] = {{0, 1}};
  multi_func_output_maps["multi_func_executor_test2"] = {{0, 1}};
  model1->lib_path_ = "./models/flow_func.so";
  model1->name_ = "model1";
  model1->multi_func_input_maps_.insert(multi_func_input_maps.begin(), multi_func_input_maps.end());
  model1->multi_func_output_maps_.insert(multi_func_output_maps.begin(), multi_func_output_maps.end());
  models.emplace_back(std::move(model1));
  FlowFuncExecutor executor;
  auto ret = executor.Init(models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  event_info event;
  event.comm.event_id = static_cast<EVENT_ID>(UdfEvent::kEventIdFlowFuncExecute);
  event.comm.subevent_id = 0;
  MOCKER_CPP(&FlowFuncProcessor::Schedule).stubs().will(returnValue(false));
  MOCKER(halEschedSubmitEvent).expects(never());
  executor.ProcessEvent(event, 0);

  executor.Stop();
  executor.WaitForStop();
}

TEST_F(FlowFuncExecutorUTest, ProcessFlowFuncExecuteEvent_need_reschedule) {
  MOCKER_CPP(&FlowFuncExecutor::SendMessageByResponseQueue).stubs().will(returnValue(0));
  FlowFuncModelParam flow_func_model_param = {"./test_flow_func.proto",
                                              "model_name",
                                              UdfTestHelper::CreateQueueDevInfos({1, 2}),
                                              UdfTestHelper::CreateQueueDevInfos({3, 4}),
                                              UdfTestHelper::CreateQueueDevInfo(0),
                                              0U,
                                              false};
  std::unique_ptr<FlowFuncModel> model(new (std::nothrow) FlowFuncModel(flow_func_model_param));
  model->lib_path_ = "./models/flow_func.so";
  model->flow_func_name_ = "flow_func_executor_test";
  std::vector<std::unique_ptr<FlowFuncModel>> models;
  models.emplace_back(std::move(model));
  // for multi func
  FlowFuncModelParam flow_func_model_param2 = {"./test_flow_func.proto",
                                               "model_name2",
                                               UdfTestHelper::CreateQueueDevInfos({5, 6, 7}),
                                               UdfTestHelper::CreateQueueDevInfos({8, 9}),
                                               UdfTestHelper::CreateQueueDevInfo(0),
                                               0U,
                                               false};
  std::unique_ptr<FlowFuncModel> model1(new (std::nothrow) FlowFuncModel(flow_func_model_param2));
  std::map<std::string, std::vector<uint32_t>> multi_func_input_maps;
  std::map<std::string, std::vector<uint32_t>> multi_func_output_maps;
  multi_func_input_maps["multi_func_executor_test1"] = {{0, 1}};
  multi_func_input_maps["multi_func_executor_test2"] = {{2}};
  multi_func_output_maps["multi_func_executor_test1"] = {{0, 1}};
  multi_func_output_maps["multi_func_executor_test2"] = {{0, 1}};
  model1->lib_path_ = "./models/flow_func.so";
  model1->name_ = "model1";
  model1->multi_func_input_maps_.insert(multi_func_input_maps.begin(), multi_func_input_maps.end());
  model1->multi_func_output_maps_.insert(multi_func_output_maps.begin(), multi_func_output_maps.end());
  models.emplace_back(std::move(model1));
  FlowFuncExecutor executor;
  auto ret = executor.Init(models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  event_info event;
  event.comm.event_id = static_cast<EVENT_ID>(UdfEvent::kEventIdFlowFuncExecute);
  event.comm.subevent_id = 0;
  MOCKER_CPP(&FlowFuncProcessor::Schedule).stubs().will(returnValue(true));
  MOCKER(halEschedSubmitEvent).expects(once()).will(returnValue(DRV_ERROR_NONE));
  executor.ProcessEvent(event, 0);
  executor.Stop();
  executor.WaitForStop();
}

TEST_F(FlowFuncExecutorUTest, ProcessFlowFuncEmptyToNotEmptyEvent_need_reschedule) {
  MOCKER_CPP(&FlowFuncExecutor::SendMessageByResponseQueue).stubs().will(returnValue(0));
  FlowFuncModelParam flow_func_model_param = {"./test_flow_func.proto",
                                              "model_name",
                                              UdfTestHelper::CreateQueueDevInfos({1, 2}),
                                              UdfTestHelper::CreateQueueDevInfos({3, 4}),
                                              UdfTestHelper::CreateQueueDevInfo(0),
                                              0U,
                                              false};
  std::unique_ptr<FlowFuncModel> model(new (std::nothrow) FlowFuncModel(flow_func_model_param));
  model->lib_path_ = "./models/flow_func.so";
  model->flow_func_name_ = "flow_func_executor_test";
  std::vector<std::unique_ptr<FlowFuncModel>> models;
  models.emplace_back(std::move(model));
  FlowFuncExecutor executor;
  auto ret = executor.Init(models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  event_info event;
  event.comm.event_id = EVENT_QUEUE_EMPTY_TO_NOT_EMPTY;
  event.comm.subevent_id = 2;  // subeventid must one of input queue ids
  MOCKER_CPP(&FlowFuncProcessor::EmptyToNotEmpty).stubs().will(returnValue(true));
  MOCKER(halEschedSubmitEvent).expects(once()).will(returnValue(DRV_ERROR_NONE));
  executor.ProcessEvent(event, 0);
  executor.Stop();
  executor.WaitForStop();
}

TEST_F(FlowFuncExecutorUTest, ProcessFlowFuncEmptyToNotEmptyEvent_no_need_reschedule) {
  MOCKER_CPP(&FlowFuncExecutor::SendMessageByResponseQueue).stubs().will(returnValue(0));
  FlowFuncModelParam flow_func_model_param = {"./test_flow_func.proto",
                                              "model_name",
                                              UdfTestHelper::CreateQueueDevInfos({1, 2}),
                                              UdfTestHelper::CreateQueueDevInfos({3, 4}),
                                              UdfTestHelper::CreateQueueDevInfo(0),
                                              0U,
                                              false};
  std::unique_ptr<FlowFuncModel> model(new (std::nothrow) FlowFuncModel(flow_func_model_param));
  model->lib_path_ = "./models/flow_func.so";
  model->flow_func_name_ = "flow_func_executor_test";
  std::vector<std::unique_ptr<FlowFuncModel>> models;
  models.emplace_back(std::move(model));
  FlowFuncExecutor executor;
  auto ret = executor.Init(models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  event_info event;
  event.comm.event_id = EVENT_QUEUE_EMPTY_TO_NOT_EMPTY;
  event.comm.subevent_id = 2;  // subeventid must one of input queue ids
  MOCKER_CPP(&FlowFuncProcessor::EmptyToNotEmpty).stubs().will(returnValue(false));
  MOCKER(halEschedSubmitEvent).expects(never()).will(returnValue(DRV_ERROR_NONE));
  executor.ProcessEvent(event, 0);
  executor.Stop();
  executor.WaitForStop();
}

TEST_F(FlowFuncExecutorUTest, ProcessFlowFuncEmptyToNotEmptyEvent_not_input_queue) {
  MOCKER_CPP(&FlowFuncExecutor::SendMessageByResponseQueue).stubs().will(returnValue(0));
  FlowFuncModelParam flow_func_model_param = {"./test_flow_func.proto",
                                              "model_name",
                                              UdfTestHelper::CreateQueueDevInfos({1, 2}),
                                              UdfTestHelper::CreateQueueDevInfos({3, 4}),
                                              UdfTestHelper::CreateQueueDevInfo(0),
                                              0U,
                                              false};
  std::unique_ptr<FlowFuncModel> model(new (std::nothrow) FlowFuncModel(flow_func_model_param));
  model->lib_path_ = "./models/flow_func.so";
  model->flow_func_name_ = "flow_func_executor_test";
  std::vector<std::unique_ptr<FlowFuncModel>> models;
  models.emplace_back(std::move(model));
  // for multi func
  FlowFuncModelParam flow_func_model_param2 = {"./test_flow_func.proto",
                                               "model_name2",
                                               UdfTestHelper::CreateQueueDevInfos({5, 6, 7}),
                                               UdfTestHelper::CreateQueueDevInfos({8, 9}),
                                               UdfTestHelper::CreateQueueDevInfo(0),
                                               0U,
                                               false};
  std::unique_ptr<FlowFuncModel> model1(new (std::nothrow) FlowFuncModel(flow_func_model_param2));
  std::map<std::string, std::vector<uint32_t>> multi_func_input_maps;
  std::map<std::string, std::vector<uint32_t>> multi_func_output_maps;
  multi_func_input_maps["multi_func_executor_test1"] = {{0, 1}};
  multi_func_input_maps["multi_func_executor_test2"] = {{2}};
  multi_func_output_maps["multi_func_executor_test1"] = {{0, 1}};
  multi_func_output_maps["multi_func_executor_test2"] = {{0, 1}};
  model1->lib_path_ = "./models/flow_func.so";
  // model1->flow_func_name_ = "multi_func_executor_test1";
  model1->name_ = "model1";
  model1->multi_func_input_maps_.insert(multi_func_input_maps.begin(), multi_func_input_maps.end());
  model1->multi_func_output_maps_.insert(multi_func_output_maps.begin(), multi_func_output_maps.end());
  models.emplace_back(std::move(model1));
  FlowFuncExecutor executor;
  auto ret = executor.Init(models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  event_info event;
  event.comm.event_id = EVENT_QUEUE_EMPTY_TO_NOT_EMPTY;
  event.comm.subevent_id = 4;  // subeventid is not one of input queue ids
  MOCKER(halEschedSubmitEvent).expects(never()).will(returnValue(DRV_ERROR_NONE));
  executor.ProcessEvent(event, 0);
  executor.Stop();
  executor.WaitForStop();
}

TEST_F(FlowFuncExecutorUTest, ProcessFlowFuncF2NFEvent_need_reschedule) {
  MOCKER_CPP(&FlowFuncExecutor::SendMessageByResponseQueue).stubs().will(returnValue(0));
  FlowFuncModelParam flow_func_model_param = {"./test_flow_func.proto",
                                              "model_name",
                                              UdfTestHelper::CreateQueueDevInfos({1, 2}),
                                              UdfTestHelper::CreateQueueDevInfos({3, 4}),
                                              UdfTestHelper::CreateQueueDevInfo(0),
                                              0U,
                                              false};
  std::unique_ptr<FlowFuncModel> model(new (std::nothrow) FlowFuncModel(flow_func_model_param));
  model->lib_path_ = "./models/flow_func.so";
  model->flow_func_name_ = "flow_func_executor_test";
  std::vector<std::unique_ptr<FlowFuncModel>> models;
  models.emplace_back(std::move(model));
  // for multi func
  FlowFuncModelParam flow_func_model_param2 = {"./test_flow_func.proto",
                                               "model_name2",
                                               UdfTestHelper::CreateQueueDevInfos({5, 6, 7}),
                                               UdfTestHelper::CreateQueueDevInfos({8, 9}),
                                               UdfTestHelper::CreateQueueDevInfo(0),
                                               0U,
                                               false};
  std::unique_ptr<FlowFuncModel> model1(new (std::nothrow) FlowFuncModel(flow_func_model_param2));
  std::map<std::string, std::vector<uint32_t>> multi_func_input_maps;
  std::map<std::string, std::vector<uint32_t>> multi_func_output_maps;
  multi_func_input_maps["multi_func_executor_test1"] = {{0, 1}};
  multi_func_input_maps["multi_func_executor_test2"] = {{2}};
  multi_func_output_maps["multi_func_executor_test1"] = {{0, 1}};
  multi_func_output_maps["multi_func_executor_test2"] = {{0, 1}};
  model1->lib_path_ = "./models/flow_func.so";
  model1->name_ = "model1";
  model1->multi_func_input_maps_.insert(multi_func_input_maps.begin(), multi_func_input_maps.end());
  model1->multi_func_output_maps_.insert(multi_func_output_maps.begin(), multi_func_output_maps.end());
  models.emplace_back(std::move(model1));
  FlowFuncExecutor executor;
  auto ret = executor.Init(models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  event_info event;
  event.comm.event_id = EVENT_QUEUE_FULL_TO_NOT_FULL;
  event.comm.subevent_id = 4;  // subeventid must one of input queue ids
  MOCKER_CPP(&FlowFuncProcessor::FullToNotFull).stubs().will(returnValue(true));
  MOCKER(halEschedSubmitEvent).expects(once()).will(returnValue(DRV_ERROR_NONE));
  executor.ProcessEvent(event, 0);
  executor.Stop();
  executor.WaitForStop();
}

TEST_F(FlowFuncExecutorUTest, ProcessFlowFuncF2NFEvent_no_need_reschedule) {
  MOCKER_CPP(&FlowFuncExecutor::SendMessageByResponseQueue).stubs().will(returnValue(0));
  FlowFuncModelParam flow_func_model_param = {"./test_flow_func.proto",
                                              "model_name",
                                              UdfTestHelper::CreateQueueDevInfos({1, 2}),
                                              UdfTestHelper::CreateQueueDevInfos({3, 4}),
                                              UdfTestHelper::CreateQueueDevInfo(0),
                                              0U,
                                              false};
  std::unique_ptr<FlowFuncModel> model(new (std::nothrow) FlowFuncModel(flow_func_model_param));
  model->lib_path_ = "./models/flow_func.so";
  model->flow_func_name_ = "flow_func_executor_test";
  std::vector<std::unique_ptr<FlowFuncModel>> models;
  models.emplace_back(std::move(model));
  // for multi func
  FlowFuncModelParam flow_func_model_param2 = {"./test_flow_func.proto",
                                               "model_name2",
                                               UdfTestHelper::CreateQueueDevInfos({5, 6, 7}),
                                               UdfTestHelper::CreateQueueDevInfos({8, 9}),
                                               UdfTestHelper::CreateQueueDevInfo(0),
                                               0U,
                                               false};
  std::unique_ptr<FlowFuncModel> model1(new (std::nothrow) FlowFuncModel(flow_func_model_param2));
  std::map<std::string, std::vector<uint32_t>> multi_func_input_maps;
  std::map<std::string, std::vector<uint32_t>> multi_func_output_maps;
  multi_func_input_maps["multi_func_executor_test1"] = {{0, 1}};
  multi_func_input_maps["multi_func_executor_test2"] = {{2}};
  multi_func_output_maps["multi_func_executor_test1"] = {{0, 1}};
  multi_func_output_maps["multi_func_executor_test2"] = {{0, 1}};
  model1->lib_path_ = "./models/flow_func.so";
  model1->name_ = "model1";
  model1->multi_func_input_maps_.insert(multi_func_input_maps.begin(), multi_func_input_maps.end());
  model1->multi_func_output_maps_.insert(multi_func_output_maps.begin(), multi_func_output_maps.end());
  models.emplace_back(std::move(model1));
  FlowFuncExecutor executor;
  auto ret = executor.Init(models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  event_info event;
  event.comm.event_id = EVENT_QUEUE_FULL_TO_NOT_FULL;
  event.comm.subevent_id = 3;  // subeventid must one of input queue ids
  MOCKER_CPP(&FlowFuncProcessor::FullToNotFull).stubs().will(returnValue(false));
  MOCKER(halEschedSubmitEvent).expects(never()).will(returnValue(DRV_ERROR_NONE));
  executor.ProcessEvent(event, 0);
  executor.Stop();
  executor.WaitForStop();
}

TEST_F(FlowFuncExecutorUTest, ProcessFlowFuncF2NFEvent_not_output_queue) {
  MOCKER_CPP(&FlowFuncExecutor::SendMessageByResponseQueue).stubs().will(returnValue(0));
  FlowFuncModelParam flow_func_model_param = {"./test_flow_func.proto",
                                              "model_name",
                                              UdfTestHelper::CreateQueueDevInfos({1, 2}),
                                              UdfTestHelper::CreateQueueDevInfos({3, 4}),
                                              UdfTestHelper::CreateQueueDevInfo(0),
                                              0U,
                                              false};
  std::unique_ptr<FlowFuncModel> model(new (std::nothrow) FlowFuncModel(flow_func_model_param));
  model->lib_path_ = "./models/flow_func.so";
  model->flow_func_name_ = "flow_func_executor_test";
  std::vector<std::unique_ptr<FlowFuncModel>> models;
  models.emplace_back(std::move(model));
  // for multi func
  FlowFuncModelParam flow_func_model_param2 = {"./test_flow_func.proto",
                                               "model_name2",
                                               UdfTestHelper::CreateQueueDevInfos({5, 6, 7}),
                                               UdfTestHelper::CreateQueueDevInfos({8, 9}),
                                               UdfTestHelper::CreateQueueDevInfo(0),
                                               0U,
                                               false};
  std::unique_ptr<FlowFuncModel> model1(new (std::nothrow) FlowFuncModel(flow_func_model_param2));
  std::map<std::string, std::vector<uint32_t>> multi_func_input_maps;
  std::map<std::string, std::vector<uint32_t>> multi_func_output_maps;
  multi_func_input_maps["multi_func_executor_test1"] = {{0, 1}};
  multi_func_input_maps["multi_func_executor_test2"] = {{2}};
  multi_func_output_maps["multi_func_executor_test1"] = {{0, 1}};
  multi_func_output_maps["multi_func_executor_test2"] = {{0, 1}};
  model1->lib_path_ = "./models/flow_func.so";
  model1->name_ = "model1";
  model1->multi_func_input_maps_.insert(multi_func_input_maps.begin(), multi_func_input_maps.end());
  model1->multi_func_output_maps_.insert(multi_func_output_maps.begin(), multi_func_output_maps.end());
  models.emplace_back(std::move(model1));
  FlowFuncExecutor executor;
  auto ret = executor.Init(models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  event_info event;
  event.comm.event_id = EVENT_QUEUE_FULL_TO_NOT_FULL;
  event.comm.subevent_id = 1;  // subeventid is not one of input queue ids
  MOCKER(halEschedSubmitEvent).expects(never()).will(returnValue(DRV_ERROR_NONE));
  executor.ProcessEvent(event, 0);
  executor.Stop();
  executor.WaitForStop();
}

TEST_F(FlowFuncExecutorUTest, FlowFuncExecutor_Start_Failed) {
  FlowFuncExecutor executor;
  GlobalConfig::Instance().SetRunOnAiCpu(true);
  MOCKER(halGetDeviceInfo).stubs().will(returnValue(DRV_ERROR_NOT_SUPPORT)).then(returnValue(DRV_ERROR_NONE));
  auto ret = executor.Start();
  EXPECT_NE(ret, FLOW_FUNC_SUCCESS);
  GlobalConfig::Instance().SetRunOnAiCpu(false);

  MOCKER(halEschedSubmitEvent).stubs().will(returnValue(DRV_ERROR_NO_SUBSCRIBE_THREAD));
  ret = executor.Start();
  EXPECT_EQ(ret, FLOW_FUNC_ERR_DRV_ERROR);
  executor.Stop(true);
  executor.WaitForStop();
}

TEST_F(FlowFuncExecutorUTest, ProcessFlowFuncCountBatchTimeout_test) {
  MOCKER_CPP(&FlowFuncExecutor::SendMessageByResponseQueue).stubs().will(returnValue(0));
  FlowFuncModelParam flow_func_model_param = {"./test_flow_func.proto",
                                              "model_name",
                                              UdfTestHelper::CreateQueueDevInfos({1, 2}),
                                              UdfTestHelper::CreateQueueDevInfos({3, 4}),
                                              UdfTestHelper::CreateQueueDevInfo(0),
                                              0U,
                                              false};
  std::unique_ptr<FlowFuncModel> model(new (std::nothrow) FlowFuncModel(flow_func_model_param));
  model->lib_path_ = "./models/flow_func.so";
  model->flow_func_name_ = "flow_func_executor_test";

  std::vector<std::unique_ptr<FlowFuncModel>> models;
  models.emplace_back(std::move(model));
  // for multi func
  FlowFuncModelParam flow_func_model_param2 = {"./test_flow_func.proto",
                                               "model_name2",
                                               UdfTestHelper::CreateQueueDevInfos({5, 6, 7}),
                                               UdfTestHelper::CreateQueueDevInfos({8, 9}),
                                               UdfTestHelper::CreateQueueDevInfo(0),
                                               0U,
                                               false};
  std::unique_ptr<FlowFuncModel> model1(new (std::nothrow) FlowFuncModel(flow_func_model_param2));
  std::map<std::string, std::vector<uint32_t>> multi_func_input_maps;
  std::map<std::string, std::vector<uint32_t>> multi_func_output_maps;
  multi_func_input_maps["multi_func_executor_test1"] = {{0, 1}};
  multi_func_input_maps["multi_func_executor_test2"] = {{2}};
  multi_func_output_maps["multi_func_executor_test1"] = {{0, 1}};
  multi_func_output_maps["multi_func_executor_test2"] = {{0, 1}};
  model1->lib_path_ = "./models/flow_func.so";
  model1->name_ = "model1";
  model1->multi_func_input_maps_.insert(multi_func_input_maps.begin(), multi_func_input_maps.end());
  model1->multi_func_output_maps_.insert(multi_func_output_maps.begin(), multi_func_output_maps.end());
  models.emplace_back(std::move(model1));
  FlowFuncExecutor executor;
  auto ret = executor.Init(models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  event_info event;
  event.comm.event_id = static_cast<EVENT_ID>(UdfEvent::kEventIdFlowFuncExecute);
  event.comm.subevent_id = 0;
  executor.ProcessEvent(event, 0);
  executor.Stop();
  executor.WaitForStop();
}

TEST_F(FlowFuncExecutorUTest, ThreadLoop_SubscribeInvokeModelEvent_failed) {
  MOCKER(halEschedSubscribeEvent).stubs().will(returnValue(DRV_ERROR_INNER_ERR));
  FlowFuncExecutor executor;
  executor.running_ = true;
  executor.ThreadLoop(0);
  EXPECT_FALSE(executor.running_);
}

TEST_F(FlowFuncExecutorUTest, ThreadLoop_InvokeModel_halEschedWaitEventSubscribe_failed) {
  MOCKER(halEschedWaitEvent).stubs().will(returnValue(DRV_ERROR_INNER_ERR));
  FlowFuncExecutor executor;
  executor.running_ = true;
  executor.ThreadLoop(0);
  EXPECT_FALSE(executor.running_);
}

TEST_F(FlowFuncExecutorUTest, ThreadLoop_SubscribeFlowMsgQueueEvent_failed) {
  MOCKER(halEschedSubscribeEvent).stubs().will(returnValue(DRV_ERROR_NONE)).then(returnValue(DRV_ERROR_INNER_ERR));
  FlowFuncExecutor executor;
  executor.running_ = true;
  executor.ThreadLoop(0);
  EXPECT_FALSE(executor.running_);
}

TEST_F(FlowFuncExecutorUTest, ThreadLoop_FlowMsgQueue_halEschedWaitEventSubscribe_failed) {
  MOCKER(halEschedWaitEvent).stubs().will(returnValue(DRV_ERROR_NONE)).then(returnValue(DRV_ERROR_INNER_ERR));
  FlowFuncExecutor executor;
  executor.running_ = true;
  executor.ThreadLoop(0);
  EXPECT_FALSE(executor.running_);
}

TEST_F(FlowFuncExecutorUTest, ThreadLoop_SubscribeWaitEventMask_failed) {
  MOCKER(halEschedSubscribeEvent)
      .stubs()
      .will(returnValue(DRV_ERROR_NONE))
      .then(returnValue(DRV_ERROR_NONE))
      .then(returnValue(DRV_ERROR_INNER_ERR));
  FlowFuncExecutor executor;
  executor.running_ = true;
  executor.ThreadLoop(0);
  EXPECT_FALSE(executor.running_);
}

TEST_F(FlowFuncExecutorUTest, ThreadLoop_halEschedSubscribeEvent_failed) {
  MOCKER(halEschedWaitEvent)
      .stubs()
      .will(returnValue(DRV_ERROR_NONE))
      .then(returnValue(DRV_ERROR_NONE))
      .then(returnValue(DRV_ERROR_INNER_ERR));
  FlowFuncExecutor executor;
  executor.running_ = true;
  executor.ThreadLoop(0);
  EXPECT_FALSE(executor.running_);
}

TEST_F(FlowFuncExecutorUTest, ThreadLoop_seccomp_failed) {
  MOCKER_CPP(&FlowFuncThreadPool::ThreadSecureCompute).stubs().will(returnValue(FLOW_FUNC_FAILED));
  FlowFuncExecutor executor;
  executor.running_ = true;
  executor.ThreadLoop(0);
  EXPECT_FALSE(executor.running_);
}

TEST_F(FlowFuncExecutorUTest, MonitorParentExit) {
  MOCKER_CPP(&FlowFuncExecutor::SendMessageByResponseQueue).stubs().will(returnValue(0));
  MOCKER(system).stubs().will(returnValue(0));
  GlobalConfig::on_device_ = true;
  MOCKER(getppid).stubs().will(repeat(1111, 2)).then(returnValue(-1));
  FlowFuncModelParam flow_func_model_param = {"./test_flow_func.proto",
                                              "model_name",
                                              UdfTestHelper::CreateQueueDevInfos({1, 2}),
                                              UdfTestHelper::CreateQueueDevInfos({3, 4}),
                                              UdfTestHelper::CreateQueueDevInfo(0),
                                              0U,
                                              false};
  std::unique_ptr<FlowFuncModel> model(new (std::nothrow) FlowFuncModel(flow_func_model_param));
  model->lib_path_ = "./models/flow_func.so";
  model->flow_func_name_ = "flow_func_executor_test";
  model->name_ = "model0";
  model->invoked_model_queue_infos_["model_key"].feed_queue_infos = UdfTestHelper::CreateQueueDevInfos({1001, 1002});
  model->invoked_model_queue_infos_["model_key"].fetch_queue_infos = UdfTestHelper::CreateQueueDevInfos({1003, 1004});
  std::vector<std::unique_ptr<FlowFuncModel>> models;
  models.emplace_back(std::move(model));
  FlowFuncExecutor executor;
  auto ret = executor.Init(models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = executor.Start();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  // as getppid failed so it will stop auto.
  executor.WaitForStop();
}

TEST_F(FlowFuncExecutorUTest, ProcessProcessorInitEvent) {
  MOCKER(halEschedSubmitEvent).stubs().will(returnValue(DRV_ERROR_INNER_ERR));
  FlowFuncExecutor executor;
  executor.running_ = true;
  executor.output_to_flow_func_processor_idx_[1] = {0, 1};
  executor.input_to_flow_func_processor_idx_[2] = 0;
  struct event_info event {};
  executor.ProcessProcessorInitEvent(event, 0);
  EXPECT_FALSE(executor.running_);
}

TEST_F(FlowFuncExecutorUTest, ProcessProcessorInitEvent_subscribeOutputQueue) {
  MOCKER(halQueueAttach).stubs().will(returnValue(DRV_ERROR_NONE)).then(returnValue(DRV_ERROR_INNER_ERR));
  FlowFuncExecutor executor;
  executor.running_ = true;
  executor.dev_output_queue_map_[0][0U] = false;
  struct event_info event {};
  executor.ProcessProcessorInitEvent(event, 0);
  EXPECT_TRUE(executor.running_);
  executor.ProcessProcessorInitEvent(event, 0);
  EXPECT_FALSE(executor.running_);
}

TEST_F(FlowFuncExecutorUTest, ProcessProcessorInitEvent_subscribeStatusOutputQueue) {
  MOCKER(halQueueAttach).stubs().will(returnValue(DRV_ERROR_INNER_ERR));
  FlowFuncExecutor executor;
  executor.running_ = true;
  executor.status_output_queue_map_[0] = {0U};
  struct event_info event {};
  executor.ProcessProcessorInitEvent(event, 0);
  EXPECT_FALSE(executor.running_);
}

TEST_F(FlowFuncExecutorUTest, ProcessProcessorInitEvent_subscribeInputQueue) {
  MOCKER(halQueueAttach).stubs().will(returnValue(DRV_ERROR_INNER_ERR)).then(returnValue(DRV_ERROR_NONE));
  FlowFuncExecutor executor;
  executor.running_ = true;
  executor.dev_input_queue_map_[0][0U] = false;
  executor.dev_input_queue_map_[0][1U] = false;
  executor.flow_msg_queues_ = {1U};
  struct event_info event {};
  executor.ProcessProcessorInitEvent(event, 0);
  EXPECT_FALSE(executor.running_);
  executor.running_ = true;
  executor.ProcessProcessorInitEvent(event, 0);
  EXPECT_TRUE(executor.running_);
  executor.UnsubscribeInputQueue();
}

TEST_F(FlowFuncExecutorUTest, ProcessReportStatusEvent_success1) {
  MOCKER_CPP(&FlowFuncManager::LoadLib).stubs().will(returnValue(0));
  MOCKER(halQueueEnQueue).stubs().will(returnValue(DRV_ERROR_NONE));
  halMbufGetBuffAddrFakeAddr = malloc(200);
  MOCKER(halMbufAlloc).stubs().will(invoke(halMbufAllocStubSimple));
  MOCKER(halMbufGetBuffAddr).stubs().will(invoke(halMbufGetBuffAddrStubSimple));
  std::shared_ptr<FlowFuncParams> param(new FlowFuncParams("test", 1U, 0, 0, 0));
  param->SetNeedReportStatusFlag(true);
  param->SetInstanceName("test@0_1_1@0@0");
  QueueDevInfo queue_dev_info = {0, 0, 0, 0, false};
  param->SetStatusOutputQueue(queue_dev_info);
  auto ret = param->Init();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  std::shared_ptr<FlowFuncProcessor> processor(new FlowFuncProcessor(
      param, "test", UdfTestHelper::CreateQueueDevInfos({1}), UdfTestHelper::CreateQueueDevInfos({1}), {1U}));
  processor->CreateStatusWriter();
  FlowFuncExecutor executor;
  executor.running_ = true;
  executor.func_processors_.emplace_back(processor);
  struct event_info event {};
  event.comm.subevent_id = 0U;
  event.comm.event_id = static_cast<EVENT_ID>(UdfEvent::kEventIdFlowFuncReportStatus);
  executor.ProcessReportStatusEvent(event, 0U);
  free(halMbufGetBuffAddrFakeAddr);
  halMbufGetBuffAddrFakeAddr = nullptr;
  EXPECT_TRUE(executor.running_);
}

TEST_F(FlowFuncExecutorUTest, ProcessReportStatusEvent_success2) {
  MOCKER_CPP(&FlowFuncManager::LoadLib).stubs().will(returnValue(0));
  MOCKER(halQueueEnQueue).stubs().will(returnValue(DRV_ERROR_NONE));
  MOCKER(halQueueQueryInfo).stubs().will(returnValue(3));
  halMbufGetBuffAddrFakeAddr = malloc(200);
  MOCKER(halMbufAlloc).stubs().will(invoke(halMbufAllocStubSimple));
  MOCKER(halMbufGetBuffAddr).stubs().will(invoke(halMbufGetBuffAddrStubSimple));
  std::shared_ptr<FlowFuncParams> param(new FlowFuncParams("test", 1U, 0, 0, 0));
  param->SetInstanceName("test@0_1_1@0@0");
  param->SetNeedReportStatusFlag(true);
  QueueDevInfo queue_dev_info = {0, 0, 0, 0, false};
  param->SetStatusOutputQueue(queue_dev_info);
  auto ret = param->Init();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  std::shared_ptr<FlowFuncProcessor> processor(new FlowFuncProcessor(
      param, "test", UdfTestHelper::CreateQueueDevInfos({1}), UdfTestHelper::CreateQueueDevInfos({1}), {1U}));
  processor->CreateStatusWriter();
  FlowFuncExecutor executor;
  executor.running_ = true;
  executor.func_processors_.emplace_back(processor);
  struct event_info event {};
  event.comm.subevent_id = 0U;
  event.comm.event_id = static_cast<EVENT_ID>(UdfEvent::kEventIdFlowFuncReportStatus);
  executor.ProcessReportStatusEvent(event, 0U);
  free(halMbufGetBuffAddrFakeAddr);
  halMbufGetBuffAddrFakeAddr = nullptr;
  EXPECT_TRUE(executor.running_);
}

TEST_F(FlowFuncExecutorUTest, ProcessReportStatusEvent_success3) {
  MOCKER_CPP(&FlowFuncManager::LoadLib).stubs().will(returnValue(0));
  MOCKER(halQueueEnQueue).stubs().will(returnValue(DRV_ERROR_QUEUE_FULL));
  halMbufGetBuffAddrFakeAddr = malloc(200);
  MOCKER(halMbufAlloc).stubs().will(invoke(halMbufAllocStubSimple));
  MOCKER(halMbufGetBuffAddr).stubs().will(invoke(halMbufGetBuffAddrStubSimple));
  std::shared_ptr<FlowFuncParams> param(new FlowFuncParams("test", 1U, 0, 0, 0));
  param->SetInstanceName("test@0_1_1@0@0");
  param->SetNeedReportStatusFlag(true);
  QueueDevInfo queue_dev_info = {0, 0, 0, 0, false};
  param->SetStatusOutputQueue(queue_dev_info);
  auto ret = param->Init();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  std::shared_ptr<FlowFuncProcessor> processor(new FlowFuncProcessor(
      param, "test", UdfTestHelper::CreateQueueDevInfos({1}), UdfTestHelper::CreateQueueDevInfos({1}), {1U}));
  processor->CreateStatusWriter();
  FlowFuncExecutor executor;
  executor.running_ = true;
  executor.func_processors_.emplace_back(processor);
  struct event_info event {};
  event.comm.subevent_id = 0U;
  event.comm.event_id = static_cast<EVENT_ID>(UdfEvent::kEventIdFlowFuncReportStatus);
  executor.ProcessReportStatusEvent(event, 0U);
  free(halMbufGetBuffAddrFakeAddr);
  halMbufGetBuffAddrFakeAddr = nullptr;
  EXPECT_TRUE(executor.running_);
}

TEST_F(FlowFuncExecutorUTest, ProcessReportStatusEvent_failed1) {
  FlowFuncExecutor executor;
  executor.running_ = true;
  struct event_info event {};
  event.comm.subevent_id = 0U;
  event.comm.event_id = static_cast<EVENT_ID>(UdfEvent::kEventIdFlowFuncReportStatus);
  executor.ProcessReportStatusEvent(event, 0U);
  EXPECT_FALSE(executor.running_);
}

TEST_F(FlowFuncExecutorUTest, ProcessReportStatusEvent_failed2) {
  MOCKER_CPP(&FlowFuncManager::LoadLib).stubs().will(returnValue(0));
  MOCKER(halMbufAlloc).stubs().will(returnValue(3));
  std::shared_ptr<FlowFuncParams> param(new FlowFuncParams("test", 1U, 0, 0, 0));
  param->SetNeedReportStatusFlag(true);
  param->SetInstanceName("test@0_1_1@0@0");
  QueueDevInfo queue_dev_info = {0, 0, 0, 0, false};
  param->SetStatusOutputQueue(queue_dev_info);
  auto ret = param->Init();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  std::shared_ptr<FlowFuncProcessor> processor(new FlowFuncProcessor(
      param, "test", UdfTestHelper::CreateQueueDevInfos({1}), UdfTestHelper::CreateQueueDevInfos({1}), {1U}));
  processor->CreateStatusWriter();
  FlowFuncExecutor executor;
  executor.running_ = true;
  executor.func_processors_.emplace_back(processor);
  struct event_info event {};
  event.comm.subevent_id = 0U;
  event.comm.event_id = static_cast<EVENT_ID>(UdfEvent::kEventIdFlowFuncReportStatus);
  executor.ProcessReportStatusEvent(event, 0U);
  EXPECT_FALSE(executor.running_);
}

TEST_F(FlowFuncExecutorUTest, ProcessReportStatusEvent_failed3) {
  MOCKER_CPP(&FlowFuncManager::LoadLib).stubs().will(returnValue(0));
  MOCKER(halMbufSetDataLen).stubs().will(returnValue(3));
  std::shared_ptr<FlowFuncParams> param(new FlowFuncParams("test", 1U, 0, 0, 0));
  param->SetNeedReportStatusFlag(true);
  param->SetInstanceName("test@0_1_1@0@0");
  QueueDevInfo queue_dev_info = {0, 0, 0, 0, false};
  param->SetStatusOutputQueue(queue_dev_info);
  auto ret = param->Init();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  std::shared_ptr<FlowFuncProcessor> processor(new FlowFuncProcessor(
      param, "test", UdfTestHelper::CreateQueueDevInfos({1}), UdfTestHelper::CreateQueueDevInfos({1}), {1U}));
  processor->CreateStatusWriter();
  FlowFuncExecutor executor;
  executor.running_ = true;
  executor.func_processors_.emplace_back(processor);
  struct event_info event {};
  event.comm.subevent_id = 0U;
  event.comm.event_id = static_cast<EVENT_ID>(UdfEvent::kEventIdFlowFuncReportStatus);
  executor.ProcessReportStatusEvent(event, 0U);
  EXPECT_FALSE(executor.running_);
}

TEST_F(FlowFuncExecutorUTest, ProcessReportStatusEvent_failed4) {
  MOCKER_CPP(&FlowFuncManager::LoadLib).stubs().will(returnValue(0));
  MOCKER(halMbufGetBuffAddr).stubs().will(returnValue(3));
  std::shared_ptr<FlowFuncParams> param(new FlowFuncParams("test", 1U, 0, 0, 0));
  param->SetNeedReportStatusFlag(true);
  param->SetInstanceName("test@0_1_1@0@0");
  QueueDevInfo queue_dev_info = {0, 0, 0, 0, false};
  param->SetStatusOutputQueue(queue_dev_info);
  auto ret = param->Init();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  std::shared_ptr<FlowFuncProcessor> processor(new FlowFuncProcessor(
      param, "test", UdfTestHelper::CreateQueueDevInfos({1}), UdfTestHelper::CreateQueueDevInfos({1}), {1U}));
  processor->CreateStatusWriter();
  FlowFuncExecutor executor;
  executor.running_ = true;
  executor.func_processors_.emplace_back(processor);
  struct event_info event {};
  event.comm.subevent_id = 0U;
  event.comm.event_id = static_cast<EVENT_ID>(UdfEvent::kEventIdFlowFuncReportStatus);
  executor.ProcessReportStatusEvent(event, 0U);
  EXPECT_FALSE(executor.running_);
}

TEST_F(FlowFuncExecutorUTest, ProcessReportStatusEvent_failed5) {
  MOCKER_CPP(&FlowFuncManager::LoadLib).stubs().will(returnValue(0));
  MOCKER(halQueueEnQueue).stubs().will(returnValue(3));
  halMbufGetBuffAddrFakeAddr = malloc(200);
  MOCKER(halMbufGetBuffAddr).stubs().will(invoke(halMbufGetBuffAddrStubSimple));
  std::shared_ptr<FlowFuncParams> param(new FlowFuncParams("test", 1U, 0, 0, 0));
  param->SetNeedReportStatusFlag(true);
  param->SetInstanceName("test@0_1_1@0@0");
  QueueDevInfo queue_dev_info = {0, 0, 0, 0, false};
  param->SetStatusOutputQueue(queue_dev_info);
  auto ret = param->Init();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  std::shared_ptr<FlowFuncProcessor> processor(new FlowFuncProcessor(
      param, "test", UdfTestHelper::CreateQueueDevInfos({1}), UdfTestHelper::CreateQueueDevInfos({1}), {1U}));
  processor->CreateStatusWriter();
  FlowFuncExecutor executor;
  executor.running_ = true;
  executor.func_processors_.emplace_back(processor);
  struct event_info event {};
  event.comm.subevent_id = 0U;
  event.comm.event_id = static_cast<EVENT_ID>(UdfEvent::kEventIdFlowFuncReportStatus);
  executor.ProcessReportStatusEvent(event, 0U);
  free(halMbufGetBuffAddrFakeAddr);
  halMbufGetBuffAddrFakeAddr = nullptr;
  EXPECT_FALSE(executor.running_);
}

TEST_F(FlowFuncExecutorUTest, CheckProcessorEventFaieldWithInvalidParams) {
  MOCKER_CPP(&FlowFuncManager::LoadLib).stubs().will(returnValue(0));
  std::shared_ptr<FlowFuncParams> param(new FlowFuncParams("test", 1U, 0, 0, 0));
  param->SetNeedReportStatusFlag(true);
  param->SetInstanceName("test@0_1_1@0@0");
  QueueDevInfo queue_dev_info = {0, 0, 0, 0, false};
  param->SetStatusOutputQueue(queue_dev_info);
  auto ret = param->Init();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  std::shared_ptr<FlowFuncProcessor> processor(new FlowFuncProcessor(
      param, "test", UdfTestHelper::CreateQueueDevInfos({1}), UdfTestHelper::CreateQueueDevInfos({1}), {1U}));
  processor->CreateStatusWriter();
  FlowFuncExecutor executor;
  executor.running_ = true;
  executor.func_processors_.emplace_back(processor);
  struct event_info event {};
  // proccessor id greater than processor number
  event.comm.subevent_id = 1U;
  EXPECT_EQ(executor.CheckProcessorEventParams(event), FLOW_FUNC_PROCESSOR_PARAM_ERROR);
  // message length is invalid
  event.comm.subevent_id = 0U;
  event.priv.msg_len = 100U;
  EXPECT_EQ(executor.CheckProcessorEventParams(event), FLOW_FUNC_PROCESSOR_PARAM_ERROR);
  event.priv.msg_len = 4U;
  int32_t evt_ret = FLOW_FUNC_SUSPEND_FAILED;
  (void)memcpy_s(event.priv.msg, sizeof(int32_t), &evt_ret, sizeof(int32_t));
  EXPECT_EQ(executor.CheckProcessorEventParams(event), evt_ret);
}

namespace {
void ConstructModelVector(std::vector<std::unique_ptr<FlowFuncModel>> &models) {
  FlowFuncModelParam flow_func_model_param = {"./test_flow_func.proto",
                                              "model_name",
                                              UdfTestHelper::CreateQueueDevInfos({1, 2}),
                                              UdfTestHelper::CreateQueueDevInfos({UINT32_MAX, 4}),
                                              UdfTestHelper::CreateQueueDevInfo(0),
                                              0U,
                                              false};
  std::unique_ptr<FlowFuncModel> model(new (std::nothrow) FlowFuncModel(flow_func_model_param));
  model->lib_path_ = "./models/flow_func.so";
  model->flow_func_name_ = "test_flow_func_not_found";
  model->name_ = "model0";
  model->invoked_model_queue_infos_["model_key"].feed_queue_infos = UdfTestHelper::CreateQueueDevInfos({1001, 1002});
  model->invoked_model_queue_infos_["model_key"].fetch_queue_infos = UdfTestHelper::CreateQueueDevInfos({1003, 1004});
  model->invoked_scopes_.emplace_back("scope/model_key");
  models.emplace_back(std::move(model));
  FlowFuncModelParam flow_func_model_param2 = {"./test_flow_func.proto",
                                               "model_name2",
                                               UdfTestHelper::CreateQueueDevInfos({5, 6}),
                                               UdfTestHelper::CreateQueueDevInfos({7, 8}),
                                               UdfTestHelper::CreateQueueDevInfo(0),
                                               0U,
                                               true};
  std::unique_ptr<FlowFuncModel> model1(new (std::nothrow) FlowFuncModel(flow_func_model_param2));
  model1->lib_path_ = "./models/flow_func.so";
  model1->flow_func_name_ = "test_flow_func_not_found";
  model1->name_ = "model1";
  models.emplace_back(std::move(model1));
}
}  // namespace

TEST_F(FlowFuncExecutorUTest, ProcessSuspendControlMessage) {
  GlobalConfig::Instance().SetMessageQueueIds(100, 101);
  MOCKER(halQueueDeQueue).defaults().will(returnValue(0)).then(returnValue(0)).then(returnValue(DRV_ERROR_QUEUE_EMPTY));
  MOCKER_CPP(&FlowFuncManager::LoadLib).stubs().will(returnValue(0));
  MOCKER_CPP(&FlowFuncExecutor::SendMessageByResponseQueue).stubs().will(returnValue(0));
  std::vector<std::unique_ptr<FlowFuncModel>> models;
  ConstructModelVector(models);
  FlowFuncExecutor executor;
  auto ret = executor.Init(models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  executor.running_ = true;
  struct event_info event {};
  event.comm.subevent_id = GlobalConfig::Instance().GetReqQueueId();
  executor.ProcessEmptyToNotEmptyEvent(event, 0);
  EXPECT_EQ(executor.running_, false);

  executor.running_ = true;
  ff::deployer::ExecutorRequest executor_request;
  auto control_msg = executor_request.mutable_clear_model_message();
  ASSERT_NE(control_msg, nullptr);
  control_msg->set_clear_msg_type(1);
  control_msg->set_model_id(0);
  halMbufGetBuffAddrFakeLen = executor_request.ByteSizeLong();
  halMbufGetBuffAddrFakeAddr = malloc(halMbufGetBuffAddrFakeLen);
  EXPECT_EQ(executor_request.SerializeToArray(halMbufGetBuffAddrFakeAddr, halMbufGetBuffAddrFakeLen), true);
  MOCKER(halMbufGetBuffAddr).defaults().will(invoke(halMbufGetBuffAddrStubSimple));
  MOCKER(halMbufGetDataLen).defaults().will(invoke(halMbufGetBuffSizeStubSimple));
  executor.ProcessEmptyToNotEmptyEvent(event, 0);
  EXPECT_EQ(executor.suspend_process_ids_.size(), 2);
  free(halMbufGetBuffAddrFakeAddr);
  halMbufGetBuffAddrFakeAddr = nullptr;
  GlobalConfig::Instance().SetMessageQueueIds(UINT32_MAX, UINT32_MAX);
}

TEST_F(FlowFuncExecutorUTest, ProcessExceptionMessage) {
  GlobalConfig::Instance().SetMessageQueueIds(100, 101);
  MOCKER_CPP(&FlowFuncExecutor::SendMessageByResponseQueue).stubs().will(returnValue(0));
  FlowFuncModelParam flow_func_model_param = {"./test_flow_func.proto",
                                              "model_name",
                                              UdfTestHelper::CreateQueueDevInfos({1, 2}),
                                              UdfTestHelper::CreateQueueDevInfos({3, 4}),
                                              UdfTestHelper::CreateQueueDevInfo(0),
                                              0U,
                                              false};
  std::unique_ptr<FlowFuncModel> model(new (std::nothrow) FlowFuncModel(flow_func_model_param));
  model->lib_path_ = "./models/flow_func.so";
  model->flow_func_name_ = "flow_func_executor_test";
  std::vector<std::unique_ptr<FlowFuncModel>> models;
  models.emplace_back(std::move(model));
  // for multi func
  FlowFuncModelParam flow_func_model_param2 = {"./test_flow_func.proto",
                                               "model_name2",
                                               UdfTestHelper::CreateQueueDevInfos({5, 6, 7}),
                                               UdfTestHelper::CreateQueueDevInfos({8, 9}),
                                               UdfTestHelper::CreateQueueDevInfo(0),
                                               0U,
                                               false};
  std::unique_ptr<FlowFuncModel> model1(new (std::nothrow) FlowFuncModel(flow_func_model_param2));
  std::map<std::string, std::vector<uint32_t>> multi_func_input_maps;
  std::map<std::string, std::vector<uint32_t>> multi_func_output_maps;
  multi_func_input_maps["multi_func_executor_test1"] = {{0, 1}};
  multi_func_input_maps["multi_func_executor_test2"] = {{2}};
  multi_func_output_maps["multi_func_executor_test1"] = {{0, 1}};
  multi_func_output_maps["multi_func_executor_test2"] = {{0, 1}};
  model1->lib_path_ = "./models/flow_func.so";
  model1->name_ = "model1";
  model1->multi_func_input_maps_.insert(multi_func_input_maps.begin(), multi_func_input_maps.end());
  model1->multi_func_output_maps_.insert(multi_func_output_maps.begin(), multi_func_output_maps.end());
  models.emplace_back(std::move(model1));
  FlowFuncExecutor executor;
  auto ret = executor.Init(models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  event_info event;
  event.comm.event_id = static_cast<EVENT_ID>(UdfEvent::kEventIdProcessorInit);
  event.comm.subevent_id = 0;
  MOCKER(halEschedSubmitEvent).defaults().will(returnValue(DRV_ERROR_NONE));
  executor.ProcessEvent(event, 0);

  event.comm.subevent_id = GlobalConfig::Instance().GetReqQueueId();
  const uint32_t priv_size = 256;
  uint8_t priv_info[priv_size];
  MbufHeadMsg *head_msg = reinterpret_cast<MbufHeadMsg *>(priv_info + priv_size - sizeof(MbufHeadMsg));
  head_msg->trans_id = 100;
  ff::deployer::ExecutorRequest executor_request;
  auto exp_notify = executor_request.mutable_exception_request();
  auto exp_request = exp_notify->mutable_exception_notify();
  ASSERT_NE(exp_request, nullptr);
  exp_request->set_type(0);
  exp_request->set_exception_code(-1);
  exp_request->set_trans_id(10);
  exp_request->set_scope("");
  exp_request->set_user_context_id(101);
  exp_request->set_exception_context(&priv_info[0], priv_size);
  halMbufGetBuffAddrFakeLen = executor_request.ByteSizeLong();
  halMbufGetBuffAddrFakeAddr = malloc(halMbufGetBuffAddrFakeLen);
  EXPECT_EQ(executor_request.SerializeToArray(halMbufGetBuffAddrFakeAddr, halMbufGetBuffAddrFakeLen), true);
  MOCKER(halMbufGetBuffAddr).defaults().will(invoke(halMbufGetBuffAddrStubSimple));
  MOCKER(halMbufGetDataLen).defaults().will(invoke(halMbufGetBuffSizeStubSimple));
  MOCKER(halQueueDeQueue)
      .defaults()
      .will(returnValue(0))
      .then(returnValue(0))
      .then(returnValue(0))
      .then(returnValue(DRV_ERROR_QUEUE_EMPTY));
  executor.ProcessEmptyToNotEmptyEvent(event, 0);
  free(halMbufGetBuffAddrFakeAddr);
  halMbufGetBuffAddrFakeAddr = nullptr;
  GlobalConfig::Instance().SetMessageQueueIds(UINT32_MAX, UINT32_MAX);
  executor.Stop();
  executor.WaitForStop();
}

TEST_F(FlowFuncExecutorUTest, ProcessExceptionMessageForInvoked) {
  GlobalConfig::Instance().SetMessageQueueIds(100, 101);
  MOCKER_CPP(&FlowFuncExecutor::SendMessageByResponseQueue).stubs().will(returnValue(0));
  std::vector<std::unique_ptr<FlowFuncModel>> models;
  ConstructModelVector(models);
  FlowFuncExecutor executor;
  auto ret = executor.Init(models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  event_info event;
  event.comm.event_id = static_cast<EVENT_ID>(UdfEvent::kEventIdProcessorInit);
  event.comm.subevent_id = 0;
  MOCKER(halEschedSubmitEvent).defaults().will(returnValue(DRV_ERROR_NONE));
  executor.ProcessEvent(event, 0);

  event.comm.subevent_id = GlobalConfig::Instance().GetReqQueueId();
  const uint32_t priv_size = 256;
  uint8_t priv_info[priv_size];
  MbufHeadMsg *head_msg = reinterpret_cast<MbufHeadMsg *>(priv_info + priv_size - sizeof(MbufHeadMsg));
  head_msg->trans_id = 100;
  ff::deployer::ExecutorRequest executor_request;
  auto exp_notify = executor_request.mutable_exception_request();
  auto exp_request = exp_notify->mutable_exception_notify();
  ASSERT_NE(exp_request, nullptr);
  exp_request->set_type(0);
  exp_request->set_exception_code(-1);
  exp_request->set_trans_id(10);
  exp_request->set_scope("scope/");
  exp_request->set_user_context_id(101);
  exp_request->set_exception_context(&priv_info[0], priv_size);
  halMbufGetBuffAddrFakeLen = executor_request.ByteSizeLong();
  halMbufGetBuffAddrFakeAddr = malloc(halMbufGetBuffAddrFakeLen);
  EXPECT_EQ(executor_request.SerializeToArray(halMbufGetBuffAddrFakeAddr, halMbufGetBuffAddrFakeLen), true);
  MOCKER(halMbufGetBuffAddr).defaults().will(invoke(halMbufGetBuffAddrStubSimple));
  MOCKER(halMbufGetDataLen).defaults().will(invoke(halMbufGetBuffSizeStubSimple));
  MOCKER(halQueueDeQueue)
      .defaults()
      .will(returnValue(0))
      .then(returnValue(0))
      .then(returnValue(0))
      .then(returnValue(DRV_ERROR_QUEUE_EMPTY));
  executor.ProcessEmptyToNotEmptyEvent(event, 0);
  free(halMbufGetBuffAddrFakeAddr);
  halMbufGetBuffAddrFakeAddr = nullptr;
  GlobalConfig::Instance().SetMessageQueueIds(UINT32_MAX, UINT32_MAX);
  executor.Stop();
  executor.WaitForStop();
}

TEST_F(FlowFuncExecutorUTest, ProcessDelExceptionMessage) {
  GlobalConfig::Instance().SetMessageQueueIds(100, 101);
  MOCKER_CPP(&FlowFuncExecutor::SendMessageByResponseQueue).stubs().will(returnValue(0));
  FlowFuncModelParam flow_func_model_param = {"./test_flow_func.proto",
                                              "model_name",
                                              UdfTestHelper::CreateQueueDevInfos({1, 2}),
                                              UdfTestHelper::CreateQueueDevInfos({3, 4}),
                                              UdfTestHelper::CreateQueueDevInfo(0),
                                              0U,
                                              false};
  std::unique_ptr<FlowFuncModel> model(new (std::nothrow) FlowFuncModel(flow_func_model_param));
  model->lib_path_ = "./models/flow_func.so";
  model->flow_func_name_ = "flow_func_executor_test";
  std::vector<std::unique_ptr<FlowFuncModel>> models;
  models.emplace_back(std::move(model));
  // for multi func
  FlowFuncModelParam flow_func_model_param2 = {"./test_flow_func.proto",
                                               "model_name2",
                                               UdfTestHelper::CreateQueueDevInfos({5, 6, 7}),
                                               UdfTestHelper::CreateQueueDevInfos({8, 9}),
                                               UdfTestHelper::CreateQueueDevInfo(0),
                                               0U,
                                               false};
  std::unique_ptr<FlowFuncModel> model1(new (std::nothrow) FlowFuncModel(flow_func_model_param2));
  std::map<std::string, std::vector<uint32_t>> multi_func_input_maps;
  std::map<std::string, std::vector<uint32_t>> multi_func_output_maps;
  multi_func_input_maps["multi_func_executor_test1"] = {{0, 1}};
  multi_func_input_maps["multi_func_executor_test2"] = {{2}};
  multi_func_output_maps["multi_func_executor_test1"] = {{0, 1}};
  multi_func_output_maps["multi_func_executor_test2"] = {{0, 1}};
  model1->lib_path_ = "./models/flow_func.so";
  model1->name_ = "model1";
  model1->multi_func_input_maps_.insert(multi_func_input_maps.begin(), multi_func_input_maps.end());
  model1->multi_func_output_maps_.insert(multi_func_output_maps.begin(), multi_func_output_maps.end());
  models.emplace_back(std::move(model1));
  FlowFuncExecutor executor;
  auto ret = executor.Init(models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  event_info event;
  event.comm.event_id = static_cast<EVENT_ID>(UdfEvent::kEventIdProcessorInit);
  event.comm.subevent_id = 0;
  MOCKER(halEschedSubmitEvent).defaults().will(returnValue(DRV_ERROR_NONE));
  executor.ProcessEvent(event, 0);

  event.comm.subevent_id = GlobalConfig::Instance().GetReqQueueId();
  const uint32_t priv_size = 256;
  uint8_t priv_info[priv_size];
  MbufHeadMsg *head_msg = reinterpret_cast<MbufHeadMsg *>(priv_info + priv_size - sizeof(MbufHeadMsg));
  head_msg->trans_id = 100;
  ff::deployer::ExecutorRequest executor_request;
  auto exp_notify = executor_request.mutable_exception_request();
  auto exp_request = exp_notify->mutable_exception_notify();
  ASSERT_NE(exp_request, nullptr);
  exp_request->set_type(1);
  exp_request->set_exception_code(-1);
  exp_request->set_trans_id(10);
  exp_request->set_scope("");
  exp_request->set_user_context_id(101);
  exp_request->set_exception_context(&priv_info[0], priv_size);
  halMbufGetBuffAddrFakeLen = executor_request.ByteSizeLong();
  halMbufGetBuffAddrFakeAddr = malloc(halMbufGetBuffAddrFakeLen);
  EXPECT_EQ(executor_request.SerializeToArray(halMbufGetBuffAddrFakeAddr, halMbufGetBuffAddrFakeLen), true);
  MOCKER(halMbufGetBuffAddr).defaults().will(invoke(halMbufGetBuffAddrStubSimple));
  MOCKER(halMbufGetDataLen).defaults().will(invoke(halMbufGetBuffSizeStubSimple));
  MOCKER(halQueueDeQueue)
      .defaults()
      .will(returnValue(0))
      .then(returnValue(0))
      .then(returnValue(0))
      .then(returnValue(DRV_ERROR_QUEUE_EMPTY));
  executor.ProcessEmptyToNotEmptyEvent(event, 0);
  free(halMbufGetBuffAddrFakeAddr);
  halMbufGetBuffAddrFakeAddr = nullptr;
  GlobalConfig::Instance().SetMessageQueueIds(UINT32_MAX, UINT32_MAX);
  executor.Stop();
  executor.WaitForStop();
}

TEST_F(FlowFuncExecutorUTest, ProcessExceptionMessageNotSameScope) {
  GlobalConfig::Instance().SetMessageQueueIds(100, 101);
  MOCKER_CPP(&FlowFuncExecutor::SendMessageByResponseQueue).stubs().will(returnValue(0));
  FlowFuncModelParam flow_func_model_param = {"./test_flow_func.proto",
                                              "model_name",
                                              UdfTestHelper::CreateQueueDevInfos({1, 2}),
                                              UdfTestHelper::CreateQueueDevInfos({3, 4}),
                                              UdfTestHelper::CreateQueueDevInfo(0),
                                              0U,
                                              false};
  std::unique_ptr<FlowFuncModel> model(new (std::nothrow) FlowFuncModel(flow_func_model_param));
  model->lib_path_ = "./models/flow_func.so";
  model->flow_func_name_ = "flow_func_executor_test";
  std::vector<std::unique_ptr<FlowFuncModel>> models;
  models.emplace_back(std::move(model));
  // for multi func
  FlowFuncModelParam flow_func_model_param2 = {"./test_flow_func.proto",
                                               "model_name2",
                                               UdfTestHelper::CreateQueueDevInfos({5, 6, 7}),
                                               UdfTestHelper::CreateQueueDevInfos({8, 9}),
                                               UdfTestHelper::CreateQueueDevInfo(0),
                                               0U,
                                               false};
  std::unique_ptr<FlowFuncModel> model1(new (std::nothrow) FlowFuncModel(flow_func_model_param2));
  std::map<std::string, std::vector<uint32_t>> multi_func_input_maps;
  std::map<std::string, std::vector<uint32_t>> multi_func_output_maps;
  multi_func_input_maps["multi_func_executor_test1"] = {{0, 1}};
  multi_func_input_maps["multi_func_executor_test2"] = {{2}};
  multi_func_output_maps["multi_func_executor_test1"] = {{0, 1}};
  multi_func_output_maps["multi_func_executor_test2"] = {{0, 1}};
  model1->lib_path_ = "./models/flow_func.so";
  model1->name_ = "model1";
  model1->multi_func_input_maps_.insert(multi_func_input_maps.begin(), multi_func_input_maps.end());
  model1->multi_func_output_maps_.insert(multi_func_output_maps.begin(), multi_func_output_maps.end());
  models.emplace_back(std::move(model1));
  FlowFuncExecutor executor;
  auto ret = executor.Init(models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  event_info event;
  event.comm.event_id = static_cast<EVENT_ID>(UdfEvent::kEventIdProcessorInit);
  event.comm.subevent_id = 0;
  MOCKER(halEschedSubmitEvent).defaults().will(returnValue(DRV_ERROR_NONE));
  executor.ProcessEvent(event, 0);

  event.comm.subevent_id = GlobalConfig::Instance().GetReqQueueId();
  const uint32_t priv_size = 256;
  uint8_t priv_info[priv_size];
  MbufHeadMsg *head_msg = reinterpret_cast<MbufHeadMsg *>(priv_info + priv_size - sizeof(MbufHeadMsg));
  head_msg->trans_id = 100;
  ff::deployer::ExecutorRequest executor_request;
  auto exp_notify = executor_request.mutable_exception_request();
  auto exp_request = exp_notify->mutable_exception_notify();
  ASSERT_NE(exp_request, nullptr);
  exp_request->set_type(0);
  exp_request->set_exception_code(-1);
  exp_request->set_trans_id(10);
  exp_request->set_scope("scope1");
  exp_request->set_user_context_id(101);
  exp_request->set_exception_context(&priv_info[0], priv_size);
  halMbufGetBuffAddrFakeLen = executor_request.ByteSizeLong();
  halMbufGetBuffAddrFakeAddr = malloc(halMbufGetBuffAddrFakeLen);
  EXPECT_EQ(executor_request.SerializeToArray(halMbufGetBuffAddrFakeAddr, halMbufGetBuffAddrFakeLen), true);
  MOCKER(halMbufGetBuffAddr).defaults().will(invoke(halMbufGetBuffAddrStubSimple));
  MOCKER(halMbufGetDataLen).defaults().will(invoke(halMbufGetBuffSizeStubSimple));
  MOCKER(halQueueDeQueue)
      .defaults()
      .will(returnValue(0))
      .then(returnValue(0))
      .then(returnValue(0))
      .then(returnValue(DRV_ERROR_QUEUE_EMPTY));
  executor.ProcessEmptyToNotEmptyEvent(event, 0);
  free(halMbufGetBuffAddrFakeAddr);
  halMbufGetBuffAddrFakeAddr = nullptr;

  exp_request->set_type(1);
  halMbufGetBuffAddrFakeLen = executor_request.ByteSizeLong();
  halMbufGetBuffAddrFakeAddr = malloc(halMbufGetBuffAddrFakeLen);
  EXPECT_EQ(executor_request.SerializeToArray(halMbufGetBuffAddrFakeAddr, halMbufGetBuffAddrFakeLen), true);
  executor.ProcessEmptyToNotEmptyEvent(event, 0);
  free(halMbufGetBuffAddrFakeAddr);
  halMbufGetBuffAddrFakeAddr = nullptr;
  GlobalConfig::Instance().SetMessageQueueIds(UINT32_MAX, UINT32_MAX);
  executor.Stop();
  executor.WaitForStop();
}

TEST_F(FlowFuncExecutorUTest, ProcessDelExceptionMessageNotSameScope) {
  GlobalConfig::Instance().SetMessageQueueIds(100, 101);
  MOCKER_CPP(&FlowFuncExecutor::SendMessageByResponseQueue).stubs().will(returnValue(0));
  FlowFuncModelParam flow_func_model_param = {"./test_flow_func.proto",
                                              "model_name",
                                              UdfTestHelper::CreateQueueDevInfos({1, 2}),
                                              UdfTestHelper::CreateQueueDevInfos({3, 4}),
                                              UdfTestHelper::CreateQueueDevInfo(0),
                                              0U,
                                              false};
  std::unique_ptr<FlowFuncModel> model(new (std::nothrow) FlowFuncModel(flow_func_model_param));
  model->lib_path_ = "./models/flow_func.so";
  model->flow_func_name_ = "flow_func_executor_test";
  std::vector<std::unique_ptr<FlowFuncModel>> models;
  models.emplace_back(std::move(model));
  // for multi func
  FlowFuncModelParam flow_func_model_param2 = {"./test_flow_func.proto",
                                               "model_name2",
                                               UdfTestHelper::CreateQueueDevInfos({5, 6, 7}),
                                               UdfTestHelper::CreateQueueDevInfos({8, 9}),
                                               UdfTestHelper::CreateQueueDevInfo(0),
                                               0U,
                                               false};
  std::unique_ptr<FlowFuncModel> model1(new (std::nothrow) FlowFuncModel(flow_func_model_param2));
  std::map<std::string, std::vector<uint32_t>> multi_func_input_maps;
  std::map<std::string, std::vector<uint32_t>> multi_func_output_maps;
  multi_func_input_maps["multi_func_executor_test1"] = {{0, 1}};
  multi_func_input_maps["multi_func_executor_test2"] = {{2}};
  multi_func_output_maps["multi_func_executor_test1"] = {{0, 1}};
  multi_func_output_maps["multi_func_executor_test2"] = {{0, 1}};
  model1->lib_path_ = "./models/flow_func.so";
  model1->name_ = "model1";
  model1->multi_func_input_maps_.insert(multi_func_input_maps.begin(), multi_func_input_maps.end());
  model1->multi_func_output_maps_.insert(multi_func_output_maps.begin(), multi_func_output_maps.end());
  models.emplace_back(std::move(model1));
  FlowFuncExecutor executor;
  auto ret = executor.Init(models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  event_info event;
  event.comm.event_id = static_cast<EVENT_ID>(UdfEvent::kEventIdProcessorInit);
  event.comm.subevent_id = 0;
  MOCKER(halEschedSubmitEvent).defaults().will(returnValue(DRV_ERROR_NONE));
  executor.ProcessEvent(event, 0);

  event.comm.subevent_id = GlobalConfig::Instance().GetReqQueueId();
  const uint32_t priv_size = 256;
  uint8_t priv_info[priv_size];
  MbufHeadMsg *head_msg = reinterpret_cast<MbufHeadMsg *>(priv_info + priv_size - sizeof(MbufHeadMsg));
  head_msg->trans_id = 100;
  ff::deployer::ExecutorRequest executor_request;
  auto exp_notify = executor_request.mutable_exception_request();
  auto exp_request = exp_notify->mutable_exception_notify();
  ASSERT_NE(exp_request, nullptr);
  exp_request->set_type(1);
  exp_request->set_exception_code(-1);
  exp_request->set_trans_id(10);
  exp_request->set_scope("scope1");
  exp_request->set_user_context_id(101);
  exp_request->set_exception_context(&priv_info[0], priv_size);
  halMbufGetBuffAddrFakeLen = executor_request.ByteSizeLong();
  halMbufGetBuffAddrFakeAddr = malloc(halMbufGetBuffAddrFakeLen);
  EXPECT_EQ(executor_request.SerializeToArray(halMbufGetBuffAddrFakeAddr, halMbufGetBuffAddrFakeLen), true);
  MOCKER(halMbufGetBuffAddr).defaults().will(invoke(halMbufGetBuffAddrStubSimple));
  MOCKER(halMbufGetDataLen).defaults().will(invoke(halMbufGetBuffSizeStubSimple));
  MOCKER(halQueueDeQueue)
      .defaults()
      .will(returnValue(0))
      .then(returnValue(0))
      .then(returnValue(0))
      .then(returnValue(DRV_ERROR_QUEUE_EMPTY));
  executor.ProcessEmptyToNotEmptyEvent(event, 0);
  free(halMbufGetBuffAddrFakeAddr);
  halMbufGetBuffAddrFakeAddr = nullptr;

  GlobalConfig::Instance().SetMessageQueueIds(UINT32_MAX, UINT32_MAX);
  executor.Stop();
  executor.WaitForStop();
}

TEST_F(FlowFuncExecutorUTest, ProcessRaiseExceptionEvent) {
  MOCKER_CPP(&FlowFuncExecutor::SendMessageByResponseQueue).stubs().will(returnValue(0));
  FlowFuncModelParam flow_func_model_param = {"./test_flow_func.proto",
                                              "model_name",
                                              UdfTestHelper::CreateQueueDevInfos({1, 2}),
                                              UdfTestHelper::CreateQueueDevInfos({3, 4}),
                                              UdfTestHelper::CreateQueueDevInfo(0),
                                              0U,
                                              false};
  std::unique_ptr<FlowFuncModel> model(new (std::nothrow) FlowFuncModel(flow_func_model_param));
  model->lib_path_ = "./models/flow_func.so";
  model->flow_func_name_ = "flow_func_executor_test";
  model->model_param_.exception_catch = true;
  model->input_align_attrs_ = {.align_max_cache_num = 4, .align_timeout = -1, .drop_when_not_align = false};
  std::vector<std::unique_ptr<FlowFuncModel>> models;
  models.emplace_back(std::move(model));
  // for multi func
  FlowFuncModelParam flow_func_model_param2 = {"./test_flow_func.proto",
                                               "model_name2",
                                               UdfTestHelper::CreateQueueDevInfos({5, 6, 7}),
                                               UdfTestHelper::CreateQueueDevInfos({8, 9}),
                                               UdfTestHelper::CreateQueueDevInfo(0),
                                               0U,
                                               false};
  std::unique_ptr<FlowFuncModel> model1(new (std::nothrow) FlowFuncModel(flow_func_model_param2));
  std::map<std::string, std::vector<uint32_t>> multi_func_input_maps;
  std::map<std::string, std::vector<uint32_t>> multi_func_output_maps;
  multi_func_input_maps["multi_func_executor_test1"] = {{0, 1}};
  multi_func_input_maps["multi_func_executor_test2"] = {{2}};
  multi_func_output_maps["multi_func_executor_test1"] = {{0, 1}};
  multi_func_output_maps["multi_func_executor_test2"] = {{0, 1}};
  model1->lib_path_ = "./models/flow_func.so";
  model1->name_ = "model1";
  model1->model_param_.exception_catch = true;
  model1->multi_func_input_maps_.insert(multi_func_input_maps.begin(), multi_func_input_maps.end());
  model1->multi_func_output_maps_.insert(multi_func_output_maps.begin(), multi_func_output_maps.end());
  models.emplace_back(std::move(model1));
  FlowFuncExecutor executor;
  auto ret = executor.Init(models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  event_info event;
  event.comm.event_id = static_cast<EVENT_ID>(UdfEvent::kEventIdProcessorInit);
  event.comm.subevent_id = 0;
  MOCKER(halEschedSubmitEvent).defaults().will(returnValue(DRV_ERROR_NONE));
  executor.ProcessEvent(event, 0);

  UdfExceptionInfo exception_info{};
  exception_info.exp_code = -1;
  exception_info.trans_id = 1;
  exception_info.user_context_id = 1000;
  executor.func_processors_[0]->run_context_->trans_id_to_exception_raised_[1] = exception_info;
  event.comm.subevent_id = 0;
  uint64_t *trans_id = reinterpret_cast<uint64_t *>(event.priv.msg);
  *trans_id = 1;
  event.priv.msg_len = sizeof(uint64_t);
  MOCKER(halMbufFree).defaults().will(invoke(halMbufFreeStub));
  MOCKER(halMbufGetBuffAddr).defaults().will(invoke(halMbufGetBuffAddrStub));
  MOCKER(halMbufGetBuffSize).defaults().will(invoke(halMbufGetBuffSizeStub));
  MOCKER(halMbufSetDataLen).defaults().will(invoke(halMbufSetDataLenStub));
  MOCKER(halMbufAlloc).defaults().will(invoke(halMbufAllocStub));
  MOCKER(halQueueEnQueue).defaults().will(invoke(halQueueEnQueueStub));
  MOCKER(halQueueDeQueue).defaults().will(returnValue(DRV_ERROR_QUEUE_EMPTY));
  executor.ProcessRaiseExceptionEvent(event, 0);

  executor.Stop();
  executor.WaitForStop();
}

TEST_F(FlowFuncExecutorUTest, ProcessRaiseExceptionEventInvalidMsg) {
  MOCKER_CPP(&FlowFuncManager::LoadLib).stubs().will(returnValue(0));
  MOCKER_CPP(&FlowFuncExecutor::SendMessageByResponseQueue).stubs().will(returnValue(0));
  MOCKER(halQueueDeQueue)
      .defaults()
      .will(returnValue(DRV_ERROR_QUEUE_EMPTY))
      .then(returnValue(0))
      .then(returnValue(DRV_ERROR_QUEUE_EMPTY));
  std::vector<std::unique_ptr<FlowFuncModel>> models;
  FlowFuncExecutor executor;
  auto ret = executor.Init(models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  event_info event;
  event.comm.event_id = static_cast<EVENT_ID>(UdfEvent::kEventIdProcessorInit);
  event.comm.subevent_id = 0;
  MOCKER(halEschedSubmitEvent).defaults().will(returnValue(DRV_ERROR_NONE));
  executor.ProcessEvent(event, 0);

  event.comm.subevent_id = 10000;
  uint64_t *trans_id = reinterpret_cast<uint64_t *>(event.priv.msg);
  *trans_id = 1;
  event.priv.msg_len = sizeof(uint64_t);
  executor.running_ = true;
  executor.ProcessRaiseExceptionEvent(event, 0);
  EXPECT_EQ(executor.running_, false);

  std::shared_ptr<FlowFuncParams> param(new FlowFuncParams("test", 1U, 0, 0, 0));
  param->SetNeedReportStatusFlag(true);
  param->SetInstanceName("test@0_1_1@0@0");
  QueueDevInfo queue_dev_info = {0, 0, 0, 0, false};
  param->SetStatusOutputQueue(queue_dev_info);
  ret = param->Init();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  std::shared_ptr<FlowFuncProcessor> processor(new FlowFuncProcessor(
      param, "test", UdfTestHelper::CreateQueueDevInfos({1}), UdfTestHelper::CreateQueueDevInfos({1}), {1U}));
  executor.func_processors_.emplace_back(processor);
  event.comm.subevent_id = 0;
  event.priv.msg_len = sizeof(uint32_t);
  executor.running_ = true;
  executor.ProcessRaiseExceptionEvent(event, 0);
  EXPECT_EQ(executor.running_, false);
}

TEST_F(FlowFuncExecutorUTest, ProcessRecoverControlMessage) {
  GlobalConfig::Instance().SetMessageQueueIds(100, 101);
  MOCKER_CPP(&FlowFuncManager::LoadLib).stubs().will(returnValue(0));
  MOCKER_CPP(&FlowFuncExecutor::SendMessageByResponseQueue).stubs().will(returnValue(0));
  MOCKER(halQueueDeQueue).defaults().will(returnValue(0)).then(returnValue(DRV_ERROR_QUEUE_EMPTY));
  std::vector<std::unique_ptr<FlowFuncModel>> models;
  ConstructModelVector(models);
  FlowFuncExecutor executor;
  auto ret = executor.Init(models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  executor.running_ = true;
  struct event_info event {};
  event.comm.subevent_id = GlobalConfig::Instance().GetReqQueueId();

  executor.running_ = true;
  ff::deployer::ExecutorRequest executor_request;
  auto control_msg = executor_request.mutable_clear_model_message();
  ASSERT_NE(control_msg, nullptr);
  control_msg->set_clear_msg_type(2);
  control_msg->set_model_id(0);
  halMbufGetBuffAddrFakeLen = executor_request.ByteSizeLong();
  halMbufGetBuffAddrFakeAddr = malloc(halMbufGetBuffAddrFakeLen);
  EXPECT_EQ(executor_request.SerializeToArray(halMbufGetBuffAddrFakeAddr, halMbufGetBuffAddrFakeLen), true);
  MOCKER(halMbufGetBuffAddr).defaults().will(invoke(halMbufGetBuffAddrStubSimple));
  MOCKER(halMbufGetDataLen).defaults().will(invoke(halMbufGetBuffSizeStubSimple));
  executor.ProcessEmptyToNotEmptyEvent(event, 0);
  EXPECT_EQ(executor.recover_process_ids_.size(), 2);
  free(halMbufGetBuffAddrFakeAddr);
  halMbufGetBuffAddrFakeAddr = nullptr;
  GlobalConfig::Instance().SetMessageQueueIds(UINT32_MAX, UINT32_MAX);
}

TEST_F(FlowFuncExecutorUTest, ProcessInvalidControlMessage) {
  GlobalConfig::Instance().SetMessageQueueIds(100, 101);
  MOCKER_CPP(&FlowFuncManager::LoadLib).stubs().will(returnValue(0));
  MOCKER_CPP(&FlowFuncExecutor::SendMessageByResponseQueue).stubs().will(returnValue(0));
  MOCKER(halQueueDeQueue).stubs().will(returnValue(0)).then(returnValue(DRV_ERROR_QUEUE_EMPTY));
  std::vector<std::unique_ptr<FlowFuncModel>> models;
  ConstructModelVector(models);
  FlowFuncExecutor executor;
  auto ret = executor.Init(models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  executor.running_ = true;
  struct event_info event {};
  event.comm.subevent_id = GlobalConfig::Instance().GetReqQueueId();

  executor.running_ = true;
  ff::deployer::ExecutorRequest_ClearModelRequest control_msg;
  control_msg.set_clear_msg_type(100);
  control_msg.set_model_id(0);
  halMbufGetBuffAddrFakeLen = control_msg.ByteSizeLong();
  halMbufGetBuffAddrFakeAddr = malloc(halMbufGetBuffAddrFakeLen);
  EXPECT_EQ(control_msg.SerializeToArray(halMbufGetBuffAddrFakeAddr, halMbufGetBuffAddrFakeLen), true);
  MOCKER(halMbufGetBuffAddr).defaults().will(invoke(halMbufGetBuffAddrStubSimple));
  MOCKER(halMbufGetBuffSize).defaults().will(invoke(halMbufGetBuffSizeStubSimple));
  executor.ProcessEmptyToNotEmptyEvent(event, 0);
  EXPECT_EQ(executor.running_, false);
  EXPECT_EQ(executor.suspend_process_ids_.size(), 0);
  free(halMbufGetBuffAddrFakeAddr);
  halMbufGetBuffAddrFakeAddr = nullptr;
  GlobalConfig::Instance().SetMessageQueueIds(UINT32_MAX, UINT32_MAX);
}

TEST_F(FlowFuncExecutorUTest, ProcessReportSuspendEvent) {
  MOCKER_CPP(&FlowFuncManager::LoadLib).stubs().will(returnValue(0));
  MOCKER_CPP(&FlowFuncExecutor::SendMessageByResponseQueue).stubs().will(returnValue(0));
  std::vector<std::unique_ptr<FlowFuncModel>> models;
  ConstructModelVector(models);
  FlowFuncExecutor executor;
  auto ret = executor.Init(models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  executor.running_ = true;
  executor.suspend_process_ids_.insert(0);
  executor.suspend_process_ids_.insert(1);
  struct event_info event {};
  event.comm.subevent_id = 0;
  event.priv.msg_len = 4U;
  int32_t evt_ret = FLOW_FUNC_SUSPEND_FAILED;
  (void)memcpy_s(event.priv.msg, sizeof(int32_t), &evt_ret, sizeof(int32_t));
  executor.ProcessReportSuspendEvent(event, 0);
  EXPECT_EQ(executor.running_, false);

  evt_ret = FLOW_FUNC_SUCCESS;
  (void)memcpy_s(event.priv.msg, sizeof(int32_t), &evt_ret, sizeof(int32_t));
  executor.ProcessReportSuspendEvent(event, 0);
  event.comm.subevent_id = 1;
  EXPECT_EQ(executor.suspend_process_ids_.size(), 1UL);
  executor.ProcessReportSuspendEvent(event, 1);
  EXPECT_EQ(executor.suspend_process_ids_.empty(), true);
}

TEST_F(FlowFuncExecutorUTest, ProcessReportRecoverEvent) {
  MOCKER_CPP(&FlowFuncManager::LoadLib).stubs().will(returnValue(0));
  MOCKER_CPP(&FlowFuncExecutor::SendMessageByResponseQueue).stubs().will(returnValue(FLOW_FUNC_FAILED));
  std::vector<std::unique_ptr<FlowFuncModel>> models;
  ConstructModelVector(models);
  FlowFuncExecutor executor;
  auto ret = executor.Init(models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  executor.running_ = true;
  executor.recover_process_ids_.insert(0);
  executor.recover_process_ids_.insert(1);
  struct event_info event {};
  event.comm.subevent_id = 0;
  event.priv.msg_len = 4U;
  int32_t evt_ret = FLOW_FUNC_SUSPEND_FAILED;
  (void)memcpy_s(event.priv.msg, sizeof(int32_t), &evt_ret, sizeof(int32_t));
  executor.ProcessReportRecoverEvent(event, 0);
  EXPECT_EQ(executor.running_, false);

  evt_ret = FLOW_FUNC_SUCCESS;
  (void)memcpy_s(event.priv.msg, sizeof(int32_t), &evt_ret, sizeof(int32_t));
  executor.ProcessReportRecoverEvent(event, 0);
  EXPECT_EQ(executor.recover_process_ids_.size(), 1UL);
  event.comm.subevent_id = 1;
  executor.ProcessReportRecoverEvent(event, 1);
  EXPECT_EQ(executor.recover_process_ids_.empty(), true);
}

TEST_F(FlowFuncExecutorUTest, InitNpuSchedProcessorFaild) {
  bool bak_value = GlobalConfig::Instance().IsNpuSched();
  GlobalConfig::Instance().SetNpuSched(true);
  MOCKER_CPP(&NpuSchedProcessor::Initialize).stubs().will(returnValue(FLOW_FUNC_ERR_PARAM_INVALID));
  MOCKER_CPP(&FlowFuncExecutor::SendMessageByResponseQueue).stubs().will(returnValue(0));
  FlowFuncExecutor executor;
  auto ret = executor.InitNpuSchedProcessor();
  EXPECT_EQ(ret, FLOW_FUNC_ERR_PARAM_INVALID);
  GlobalConfig::Instance().SetNpuSched(bak_value);
}

TEST_F(FlowFuncExecutorUTest, InitNpuSchedProcessorDequeueFailed) {
  bool bak_value = GlobalConfig::Instance().IsNpuSched();
  GlobalConfig::Instance().SetNpuSched(true);
  MOCKER_CPP(&NpuSchedProcessor::Initialize).stubs().will(returnValue(0));
  MOCKER_CPP(&FlowFuncExecutor::SendMessageByResponseQueue).stubs().will(returnValue(0));
  MOCKER(halQueueGetStatus).stubs().will(returnValue(DRV_ERROR_NONE)).then(returnValue(DRV_ERROR_INNER_ERR));
  FlowFuncExecutor executor;
  executor.request_queue_wrapper_.reset(new QueueWrapper(0, 100));
  auto ret = executor.InitNpuSchedProcessor();
  EXPECT_NE(ret, FLOW_FUNC_SUCCESS);
  GlobalConfig::Instance().SetNpuSched(bak_value);
}

TEST_F(FlowFuncExecutorUTest, InitNpuSchedProcessorSuccess) {
  MOCKER(halMbufAllocEx).defaults().will(invoke(halMbufAllocExStub));
  MOCKER(halMbufFree).defaults().will(invoke(halMbufFreeStub));
  MOCKER(halQueueDeQueue).defaults().will(invoke(halQueueDeQueueStub));
  MOCKER(halQueueEnQueue).defaults().will(invoke(halQueueEnQueueStub));
  MOCKER(halMbufAlloc).defaults().will(invoke(halMbufAllocStub));

  bool bak_value = GlobalConfig::Instance().IsNpuSched();
  GlobalConfig::Instance().SetNpuSched(true);
  MOCKER_CPP(&NpuSchedProcessor::Initialize).stubs().will(returnValue(0));
  MOCKER_CPP(&FlowFuncExecutor::SendMessageByResponseQueue).stubs().will(returnValue(0));

  ff::deployer::ExecutorRequest executor_request;
  executor_request.set_type(ff::deployer::ExecutorRequestType::kNotify);

  halMbufGetBuffAddrFakeLen = executor_request.ByteSizeLong();
  halMbufGetBuffAddrFakeAddr = malloc(halMbufGetBuffAddrFakeLen);
  ScopeGuard guard([]() {
    if (halMbufGetBuffAddrFakeAddr != nullptr) {
      free(halMbufGetBuffAddrFakeAddr);
      halMbufGetBuffAddrFakeAddr = nullptr;
    }
  });

  EXPECT_EQ(executor_request.SerializeToArray(halMbufGetBuffAddrFakeAddr, halMbufGetBuffAddrFakeLen), true);
  MOCKER(halMbufGetBuffAddr).defaults().will(invoke(halMbufGetBuffAddrStubSimple));
  MOCKER(halMbufGetDataLen).defaults().will(invoke(halMbufGetBuffSizeStubSimple));

  FlowFuncExecutor executor;
  executor.request_queue_wrapper_.reset(new QueueWrapper(0, 100));
  auto ret = executor.InitNpuSchedProcessor();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  GlobalConfig::Instance().SetNpuSched(bak_value);
}
}  // namespace FlowFunc