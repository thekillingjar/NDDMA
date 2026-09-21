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
#include "common/data_utils.h"
#include "execute/flow_func_executor.h"
#include "model/flow_func_model.h"
#define private public
#include "config/global_config.h"
#include "flow_func/flow_func_config_manager.h"
#undef private

namespace FlowFunc {
namespace {
constexpr uint64_t kWaitInMsPerTime = 10;
}
class RunFlowModelSTest : public testing::Test {
protected:
  static void SetUpTestSuite() {
    FlowFuncConfigManager::SetConfig(
        std::shared_ptr<FlowFuncConfig>(&GlobalConfig::Instance(), [](FlowFuncConfig *) {}));
  }
  virtual void SetUp() {
    ClearStubEschedEvents();
    CreateModelDir();
    back_is_on_device_ = GlobalConfig::Instance().IsOnDevice();
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
    GlobalMockObject::verify();
    GlobalConfig::on_device_ = back_is_on_device_;
    GlobalConfig::Instance().SetMessageQueueIds(UINT32_MAX, UINT32_MAX);
  }

  void SetQueueAttr(::ff::deployer::ExecutorRequest_QueueAttrs *queue_attr, uint32_t qid,
                    int32_t device_type = Common::kDeviceTypeCpu, int32_t device_id = 0) {
    queue_attr->set_queue_id(qid);
    queue_attr->set_device_type(device_type);
    queue_attr->set_device_id(device_id);
  }

  ff::deployer::ExecutorRequest_ModelQueuesAttrs CreateModelQueues(bool link_model_out_to_in, int32_t device_type) {
    ff::deployer::ExecutorRequest_ModelQueuesAttrs model_queues_attrs;
    auto input_qid1 = CreateQueue();
    auto input_qid2 = CreateQueue();
    auto output_qid1 = CreateQueue();
    auto output_qid2 = CreateQueue();
    record_queue_ids.emplace_back(input_qid1);
    record_queue_ids.emplace_back(input_qid2);
    record_queue_ids.emplace_back(output_qid1);
    record_queue_ids.emplace_back(output_qid2);
    if (link_model_out_to_in) {
      LinkQueueInTest(output_qid1, input_qid1);
      LinkQueueInTest(output_qid2, input_qid2);
    }

    for (auto input_qid : {input_qid1, input_qid2}) {
      auto input_queue_attr = model_queues_attrs.add_input_queues_attrs();
      SetQueueAttr(input_queue_attr, input_qid, device_type);
    }

    for (auto output_qid : {output_qid1, output_qid2}) {
      auto output_queue_attr = model_queues_attrs.add_output_queues_attrs();
      SetQueueAttr(output_queue_attr, output_qid, device_type);
    }
    return model_queues_attrs;
  }

  std::string CreateModelForRunFlowModel(std::vector<uint32_t> &input_queues, std::vector<uint32_t> &output_queues,
                                         bool link_model_out_to_in, int32_t device_type = Common::kDeviceTypeCpu,
                                         int64_t timeout = 1000) {
    ff::udf::UdfModelDef model_def;
    std::map<AscendString, ff::udf::AttrValue> attr_map;
    ff::udf::AttrValue timeout_attr;
    timeout_attr.set_i(timeout);
    attr_map["_TEST_ATTR_TIMEOUT"] = timeout_attr;
    CreateUdfModel(model_def, "ST_FlowFuncCallNn", __FILE__, attr_map);

    auto proto_path = WriteProto(model_def, "st_call_nn.pb");
    ff::deployer::ExecutorRequest_BatchLoadModelMessage batch_load_model_req;

    for (size_t idx = 0; idx < 2; ++idx) {
      auto input_qid = CreateQueue();
      auto output_qid = CreateQueue();
      record_queue_ids.emplace_back(input_qid);
      record_queue_ids.emplace_back(output_qid);
      input_queues.emplace_back(input_qid);
      output_queues.emplace_back(output_qid);
    }

    AddModelToBatchModel(batch_load_model_req, proto_path, input_queues, output_queues, {}, device_type);
    auto invoked_model_queues_attrs = batch_load_model_req.mutable_models(0)->mutable_invoked_model_queues_attrs();
    (*invoked_model_queues_attrs)["float_model_key"] = CreateModelQueues(link_model_out_to_in, device_type);
    (*invoked_model_queues_attrs)["other_model_key"] = CreateModelQueues(link_model_out_to_in, device_type);
    batch_load_model_req.mutable_models(0)->mutable_input_align_attrs()->set_align_max_cache_num(4);
    batch_load_model_req.mutable_models(0)->mutable_input_align_attrs()->set_align_timeout(-1);
    batch_load_model_req.mutable_models(0)->mutable_input_align_attrs()->set_drop_when_not_align(false);

    return WriteProto(batch_load_model_req, "exchange_queues_batch_model.pb");
  }

 protected:
  std::vector<uint32_t> record_queue_ids;
  bool back_is_on_device_ = false;
  uint32_t req_queue_id;
  uint32_t rsp_queue_id;
};

TEST_F(RunFlowModelSTest, basic_test) {
  std::vector<uint32_t> input_queues;
  std::vector<uint32_t> output_queues;
  std::string batch_model_path = CreateModelForRunFlowModel(input_queues, output_queues, true);

  auto batch_models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(batch_models.size(), 1);
  FlowFuncExecutor executor;
  auto ret = executor.Init(batch_models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = executor.Start();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);

  std::vector<int64_t> shape = {1, 2, 3, 4};
  std::vector<float> float_value = {0, 1.1, 2.2, 3.3, 4.4};
  for (size_t i = 0; i < input_queues.size(); ++i) {
    DataEnqueue(input_queues[i], shape, TensorDataType::DT_FLOAT, float_value[i]);
  }

  constexpr uint64_t kMaxWaitInMs = 2 * 1000UL;
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
    std::vector<float> expect_output(CalcElementCnt(shape), float_value[i]);
    CheckMbufData(out_mbuf, shape, TensorDataType::DT_FLOAT, expect_output.data(), expect_output.size());
    halMbufFree(out_mbuf);
  }

  executor.Stop(true);
  executor.WaitForStop();
}

TEST_F(RunFlowModelSTest, basic_test_timeout_zero) {
  std::vector<uint32_t> input_queues;
  std::vector<uint32_t> output_queues;
  int64_t timeout = 0;
  // timeout=0
  std::string batch_model_path =
      CreateModelForRunFlowModel(input_queues, output_queues, true, Common::kDeviceTypeCpu, timeout);

  auto batch_models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(batch_models.size(), 1);
  FlowFuncExecutor executor;
  auto ret = executor.Init(batch_models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = executor.Start();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);

  std::vector<int64_t> shape = {1, 2, 3, 4};
  std::vector<float> float_value = {0, 1.1, 2.2, 3.3, 4.4};
  for (size_t i = 0; i < input_queues.size(); ++i) {
    DataEnqueue(input_queues[i], shape, TensorDataType::DT_FLOAT, float_value[i]);
  }

  constexpr uint64_t kMaxWaitInMs = 2 * 1000UL;
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
    std::vector<float> expect_output(CalcElementCnt(shape), float_value[i]);
    CheckMbufData(out_mbuf, shape, TensorDataType::DT_FLOAT, expect_output.data(), expect_output.size());
    halMbufFree(out_mbuf);
  }
  executor.Stop(true);
  executor.WaitForStop();
}

TEST_F(RunFlowModelSTest, run_flow_model_timeout) {
  GlobalConfig::on_device_ = true;
  MOCKER(system).stubs().will(returnValue(0));
  std::vector<uint32_t> input_queues;
  std::vector<uint32_t> output_queues;
  std::string batch_model_path =
      CreateModelForRunFlowModel(input_queues, output_queues, false, Common::kDeviceTypeNpu, 500);

  auto batch_models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(batch_models.size(), 1);
  FlowFuncExecutor executor;
  auto ret = executor.Init(batch_models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = executor.Start();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);

  std::vector<int64_t> shape = {1, 2, 3, 4};
  std::vector<int32_t> int_value = {0, 1, 2, 3, 4};
  for (size_t i = 0; i < input_queues.size(); ++i) {
    DataEnqueue(input_queues[i], shape, TensorDataType::DT_INT32, int_value[i]);
  }

  constexpr uint64_t kMaxWaitInMs = 1000UL;
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
    ASSERT_EQ(out_mbuf_ptr, nullptr);
  }
  // call nn timeout proc return failed, will auto stop.
  executor.WaitForStop();
}
}  // namespace FlowFunc