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
#include "gtest/gtest.h"
#include "securec.h"
#include "utils/udf_utils.h"
#include "stub/udf_stub.h"
#include "common/data_utils.h"
#include "execute/flow_func_executor.h"
#include "model/flow_func_model.h"
#include "flow_func/meta_multi_func.h"
#include "config/global_config.h"
#include "flow_func/flow_func_config_manager.h"
#include "flow_func/flow_func_log.h"

namespace FlowFunc {
namespace {
constexpr uint64_t kMaxWaitInMs = 10 * 1000UL;
constexpr uint64_t kWaitInMsPerTime = 10;

std::mutex g_sync_mutex;
std::condition_variable_any g_cv_;

class EmptyInputFlowFuncStub : public MetaMultiFunc {
 public:
  EmptyInputFlowFuncStub() = default;

  ~EmptyInputFlowFuncStub() override = default;

  int32_t EmptyInputProc1(const std::shared_ptr<MetaRunContext> &run_context,
                          const std::vector<std::shared_ptr<FlowMsg>> &input_flow_msgs) {
    ++empty_proc_count;
    g_cv_.notify_all();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    return FLOW_FUNC_SUCCESS;
  }

  int32_t EmptyInputProc2(const std::shared_ptr<MetaRunContext> &run_context,
                          const std::vector<std::shared_ptr<FlowMsg>> &input_flow_msgs) {
    EXPECT_EQ(input_flow_msgs[0]->GetTensor()->GetDataSize(), sizeof(int32_t));
    auto data = static_cast<int32_t *>(input_flow_msgs[0]->GetTensor()->GetData());
    if (data != nullptr) {
      *data = empty_proc_count.load();
    }
    return run_context->SetOutput(0, input_flow_msgs[0]);
  }

 private:
  std::atomic_int32_t empty_proc_count{0};
};
FLOW_FUNC_REGISTRAR(EmptyInputFlowFuncStub)
    .RegProcFunc("empty_input_proc1", &EmptyInputFlowFuncStub::EmptyInputProc1)
    .RegProcFunc("empty_input_proc2", &EmptyInputFlowFuncStub::EmptyInputProc2);
}  // namespace
class EmptyInputSTest : public testing::Test {
protected:
  static void SetUpTestSuite() {
    FlowFuncConfigManager::SetConfig(
        std::shared_ptr<FlowFuncConfig>(&GlobalConfig::Instance(), [](FlowFuncConfig *) {}));
  }
  virtual void SetUp() {
    ClearStubEschedEvents();
    CreateModelDir();
    req_queue_id = CreateQueue();
    rsp_queue_id = CreateQueue();
    GlobalConfig::Instance().SetMessageQueueIds(req_queue_id, rsp_queue_id);
  }

  virtual void TearDown() {
    DeleteModelDir();
    for (auto qid : record_queue_ids) {
      DestroyQueue(qid);
    }
    record_queue_ids.clear();
    DestroyQueue(req_queue_id);
    DestroyQueue(rsp_queue_id);
    GlobalConfig::Instance().SetMessageQueueIds(UINT32_MAX, UINT32_MAX);
  }

  std::string CreateEmptyInputModel(std::vector<uint32_t> &input_queues, std::vector<uint32_t> &output_queues,
                                    uint32_t queue_depth = UDF_ST_QUEUE_MAX_DEPTH) {
    ff::udf::UdfModelDef model_def;
    const std::map<std::string, std::vector<uint32_t>> func_input_maps = {{"empty_input_proc1", {}},
                                                                          {"empty_input_proc2", {0}}};
    CreateUdfModel(model_def, "EmptyInputSTest", __FILE__, {}, false, func_input_maps);
    auto proto_path = WriteProto(model_def, "empty_input.pb");
    ff::deployer::ExecutorRequest_BatchLoadModelMessage batch_load_model_req;

    auto input_qid = CreateQueue(queue_depth);
    auto output_qid = CreateQueue(queue_depth);
    record_queue_ids.emplace_back(input_qid);
    record_queue_ids.emplace_back(output_qid);
    input_queues.emplace_back(input_qid);
    output_queues.emplace_back(output_qid);

    AddModelToBatchModel(batch_load_model_req, proto_path, input_queues, output_queues);
    return WriteProto(batch_load_model_req, "empty_input_batch_model.pb");
  }

 protected:
  std::vector<uint32_t> record_queue_ids;
  uint32_t req_queue_id;
  uint32_t rsp_queue_id;
};

TEST_F(EmptyInputSTest, basic_test) {
  std::vector<uint32_t> input_queues;
  std::vector<uint32_t> output_queues;
  std::string batch_model_path = CreateEmptyInputModel(input_queues, output_queues);

  auto batch_models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(batch_models.size(), 1);
  FlowFuncExecutor executor;
  auto ret = executor.Init(batch_models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = executor.Start();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  {
    std::unique_lock<std::mutex> lock(g_sync_mutex);
    // wait empty input notify
    g_cv_.wait_for(lock, std::chrono::seconds(1));
  }
  std::vector<int64_t> shape = {};
  int32_t input_value = 0;
  for (size_t i = 0; i < input_queues.size(); ++i) {
    DataEnqueue(input_queues[i], shape, TensorDataType::DT_INT32, input_value);
  }

  for (size_t i = 0; i < output_queues.size(); i++) {
    void *out_mbuf_ptr = nullptr;
    uint64_t wait_in_ms = 0;
    while (wait_in_ms < kMaxWaitInMs) {
      auto drv_ret = halQueueDeQueue(0, output_queues[i], &out_mbuf_ptr);
      if (drv_ret == DRV_ERROR_NONE) {
        break;
      } else if (drv_ret == DRV_ERROR_QUEUE_EMPTY) {
        std::this_thread::sleep_for(std::chrono::milliseconds(kWaitInMsPerTime));
        wait_in_ms += kWaitInMsPerTime;
        continue;
      } else {
        break;
      }
    }
    ASSERT_NE(out_mbuf_ptr, nullptr);
    Mbuf *out_mbuf = (Mbuf *)out_mbuf_ptr;
    const int32_t *out_data = nullptr;
    uint32_t data_num = 0;
    GetMbufDataAndLen(out_mbuf, out_data, data_num);
    EXPECT_NE(out_data, nullptr);
    EXPECT_EQ(data_num, 1);
    if (data_num == 1) {
      EXPECT_NE(out_data[0], 0);
    }
    halMbufFree(out_mbuf);
  }

  executor.Stop(true);
  executor.WaitForStop();
  executor.Destroy();
}
}  // namespace FlowFunc
