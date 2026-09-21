/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <list>
#include <condition_variable>
#include "gtest/gtest.h"
#include "mockcpp/mockcpp.hpp"
#include "utils/udf_utils.h"
#include "stub/udf_stub.h"
#include "mmpa/mmpa_api.h"
#include "common/data_utils.h"
#include "common/scope_guard.h"
#include "config/global_config.h"
#include "execute/flow_func_executor.h"
#include "model/flow_func_model.h"
#include "config/global_config.h"
#include "execute/npu_sched_processor.h"
#include "flow_func/flow_func_config_manager.h"

namespace FlowFunc {
namespace {

uint32_t MockFuncInitializeNpuSched(int32_t device_id) {
  return 0;
}

NpuSchedModelHandler MockFuncCreateNpuSchedModelHandler() {
  return new int64_t;
}

uint32_t MockFuncLoadNpuSchedModel(NpuSchedModelHandler handler, NpuSchedLoadParam *load_param) {
  load_param->req_msg_queue_id = 111;
  load_param->resp_msg_queue_id = 222;
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
class FlowFuncExecutorWithNpuSchedModelSTest : public testing::Test {
protected:
  static void SetUpTestSuite() {
    FlowFuncConfigManager::SetConfig(
        std::shared_ptr<FlowFuncConfig>(&GlobalConfig::Instance(), [](FlowFuncConfig *) {}));
  }
  virtual void SetUp() {
    MOCKER(mmDlsym).defaults().will(invoke(MockmmDlsym));
    ClearStubEschedEvents();
    CreateModelDir();
    req_queue_id = CreateQueue();
    rsp_queue_id = CreateQueue();
    GlobalConfig::Instance().SetMessageQueueIds(req_queue_id, rsp_queue_id);
    GlobalConfig::Instance().SetNpuSched(true);
  }

  virtual void TearDown() {
    DeleteModelDir();
    for (auto qid : record_queue_ids) {
      DestroyQueue(qid);
    }
    record_queue_ids.clear();
    DestroyQueue(req_queue_id);
    DestroyQueue(rsp_queue_id);
    GlobalMockObject::verify();
    GlobalConfig::Instance().SetMessageQueueIds(UINT32_MAX, UINT32_MAX);
    GlobalConfig::Instance().SetNpuSched(false);
  }

  std::string CreateOnlyOneBatchModel(uint32_t &input_qid, uint32_t &output_qid,
                                      const std::map<std::string, std::string> &attrs = {}) {
    ff::udf::UdfModelDef model_def;
    CreateUdfModel(model_def, "FlowFuncSt", __FILE__, {});
    auto udf_def = model_def.mutable_udf_def(0);

    auto proto_attrs = udf_def->mutable_attrs();
    ff::udf::AttrValue value;
    value.set_type((uint32_t)TensorDataType::DT_INT64);
    proto_attrs->insert({"out_type", value});
    ff::udf::AttrValue re_init_attr;
    re_init_attr.set_b(true);
    proto_attrs->insert({"need_re_init_attr", re_init_attr});
    ff::udf::AttrValue cpu_numvalue;
    cpu_numvalue.set_i(2);
    proto_attrs->insert({std::string("__cpu_num"), cpu_numvalue});
    auto proto_path = WriteProto(model_def, "FlowFuncSt.pb");
    ff::deployer::ExecutorRequest_BatchLoadModelMessage batch_load_model_req;
    input_qid = CreateQueue();
    record_queue_ids.emplace_back(input_qid);
    output_qid = CreateQueue();
    record_queue_ids.emplace_back(output_qid);
    AddModelToBatchModel(batch_load_model_req, proto_path, {input_qid}, {output_qid}, attrs, Common::kDeviceTypeNpu);
    return WriteProto(batch_load_model_req, "batchModels.pb");
  }

 protected:
  std::vector<uint32_t> record_queue_ids;
  uint32_t req_queue_id;
  uint32_t rsp_queue_id;
};

TEST_F(FlowFuncExecutorWithNpuSchedModelSTest, basic_test) {
  MOCKER(drvQueryProcessHostPid).stubs().will(repeat(DRV_ERROR_NO_PROCESS, 10)).then(returnValue(DRV_ERROR_NONE));
  ff::deployer::ExecutorRequest executor_request;
  executor_request.set_type(ff::deployer::ExecutorRequestType::kNotify);
  EnqueueControlMsg(req_queue_id, executor_request);

  uint32_t input_qid = 0;
  uint32_t output_qid = 0;
  std::string batch_model_path = CreateOnlyOneBatchModel(input_qid, output_qid);
  auto batch_models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(batch_models.size(), 1);
  FlowFuncExecutor executor;
  ScopeGuard executor_guard([&executor]() {
    executor.Stop(true);
    executor.WaitForStop();
    executor.Destroy();
  });
  auto ret = executor.Init(batch_models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = executor.Start();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
}
}  // namespace FlowFunc
