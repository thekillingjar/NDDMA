/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "mockcpp/mockcpp.hpp"
#include "mmpa/mmpa_api.h"
#include "execute/npu_sched_processor.h"
namespace FlowFunc {
namespace {
uint32_t MockFuncInitializeNpuSched(int32_t device_id) {
  return 0;
}

NpuSchedModelHandler MockFuncCreateNpuSchedModelHandler() {
  return new int64_t;
}

uint32_t MockFuncLoadNpuSchedModel(NpuSchedModelHandler handler, NpuSchedLoadParam *load_param) {
  if (!load_param->model_queue_param.input_queues.empty()) {
    load_param->req_msg_queue_id = 100 + load_param->model_queue_param.input_queues[0];
  }
  if (!load_param->model_queue_param.output_queues.empty()) {
    load_param->resp_msg_queue_id = 100 + load_param->model_queue_param.output_queues[0];
  }
  return 0;
}

uint32_t MockFuncUnloadNpuSchedModel(NpuSchedModelHandler handler) {
  return 0;
}

uint32_t MockFuncDestroyNpuSchedModelHandler(NpuSchedModelHandler handler) {
  delete (int64_t *)handler;
  return 0;
}

uint32_t MockFuncFinalizeNpuSched(int32_t device_id) {
  return 0;
}

VOID *MockmmDlopen(const CHAR *file_name, INT32 mode) {
  return new int32_t;
}

INT32 MockmmDlclose(VOID *handle) {
  delete (int32_t *)handle;
  return 0;
}

constexpr const char *FUNC_NAME_INITIALIZE_NPU_SCHED = "InitializeNpuSched";
constexpr const char *FUNC_NAME_CREATE_NPU_SCHED_MODEL_HANDLER = "CreateNpuSchedModelHandler";
constexpr const char *FUNC_NAME_LOAD_NPU_SCHED_MODEL = "LoadNpuSchedModel";
constexpr const char *FUNC_NAME_UNLOAD_NPU_SCHED_MODEL = "UnloadNpuSchedModel";
constexpr const char *FUNC_NAME_DESTROY_NPU_SCHED_MODEL_HANDLER = "DestroyNpuSchedModelHandler";
constexpr const char *FUNC_NAME_FINALIZE_NPU_SCHED = "FinalizeNpuSched";

VOID *MockmmDlsym(VOID *handle, const CHAR *func_name) {
  if (std::string(FUNC_NAME_INITIALIZE_NPU_SCHED) == func_name) {
    return MockFuncInitializeNpuSched;
  } else if (std::string(FUNC_NAME_CREATE_NPU_SCHED_MODEL_HANDLER) == func_name) {
    return MockFuncCreateNpuSchedModelHandler;
  } else if (std::string(FUNC_NAME_LOAD_NPU_SCHED_MODEL) == func_name) {
    return MockFuncLoadNpuSchedModel;
  } else if (std::string(FUNC_NAME_DESTROY_NPU_SCHED_MODEL_HANDLER) == func_name) {
    return MockFuncDestroyNpuSchedModelHandler;
  } else if (std::string(FUNC_NAME_FINALIZE_NPU_SCHED) == func_name) {
    return MockFuncFinalizeNpuSched;
  } else if (std::string(FUNC_NAME_UNLOAD_NPU_SCHED_MODEL) == func_name) {
    return MockFuncUnloadNpuSchedModel;
  }
  return nullptr;
}
}  // namespace
class NpuSchedProcessorUTest : public testing::Test {
 protected:
  virtual void SetUp() {
    MOCKER(mmDlopen).defaults().will(invoke(MockmmDlopen));
    MOCKER(mmDlclose).defaults().will(invoke(MockmmDlclose));
    MOCKER(mmDlsym).defaults().will(invoke(MockmmDlsym));
  }

  virtual void TearDown() {
    GlobalMockObject::verify();
  }
};

TEST_F(NpuSchedProcessorUTest, InitializeAndFinalize) {
  void *handle = nullptr;
  MOCKER(mmDlopen).stubs().will(returnValue(handle)).then(invoke(&MockmmDlopen));
  NpuSchedProcessor npu_sched_processor;
  int32_t init_ret = npu_sched_processor.Initialize(1);
  EXPECT_EQ(init_ret, FLOW_FUNC_FAILED);
  void *stub_not_found = nullptr;
  MOCKER(mmDlsym).stubs().will(returnValue(stub_not_found)).then(invoke(MockmmDlsym));
  init_ret = npu_sched_processor.Initialize(1);
  EXPECT_EQ(init_ret, FLOW_FUNC_FAILED);
  npu_sched_processor.Finalize();

  init_ret = npu_sched_processor.Initialize(1);
  EXPECT_EQ(init_ret, FLOW_FUNC_SUCCESS);
  npu_sched_processor.Finalize();
}

TEST_F(NpuSchedProcessorUTest, LoadAndUnload) {
  NpuSchedProcessor npu_sched_processor;
  int32_t init_ret = npu_sched_processor.Initialize(1);
  EXPECT_EQ(init_ret, FLOW_FUNC_SUCCESS);
  FlowFuncModelParam flow_func_model_param{};
  QueueDevInfo tmp_queue{};
  tmp_queue.queue_id = 1;
  tmp_queue.device_type = 0;
  tmp_queue.device_id = 1;
  tmp_queue.logic_queue_id = 10;
  tmp_queue.is_proxy_queue = true;
  flow_func_model_param.input_queues.emplace_back(tmp_queue);
  tmp_queue.queue_id = 2;
  flow_func_model_param.output_queues.emplace_back(tmp_queue);
  FlowFuncModel flow_model(flow_func_model_param);
  ModelQueueInfos model_queue_infos;

  int32_t load_ret = npu_sched_processor.LoadNpuSchedModel(flow_model, model_queue_infos);
  EXPECT_EQ(load_ret, FLOW_FUNC_SUCCESS);
  ASSERT_EQ(model_queue_infos.input_queues.size(), 1);
  EXPECT_EQ(model_queue_infos.input_queues[0].queue_id, 100 + 1);
  ASSERT_EQ(model_queue_infos.output_queues.size(), 1);
  EXPECT_EQ(model_queue_infos.output_queues[0].queue_id, 100 + 2);
  npu_sched_processor.Finalize();
}
}  // namespace FlowFunc