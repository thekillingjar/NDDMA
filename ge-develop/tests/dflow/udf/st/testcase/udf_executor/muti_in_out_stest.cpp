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
#include "securec.h"
#include "utils/udf_utils.h"
#include "stub/udf_stub.h"
#include "common/data_utils.h"
#include "execute/flow_func_executor.h"
#include "model/flow_func_model.h"
#include "config/global_config.h"
#include "flow_func/flow_func_config_manager.h"

namespace FlowFunc {
namespace {
constexpr uint64_t kMaxWaitInMs = 120 * 1000UL;
constexpr uint64_t kWaitInMsPerTime = 10;
}
class MultiInOutSTest : public testing::Test {
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

  std::string CreateModel(std::vector<uint32_t> &input_queues, std::vector<uint32_t> &output_queues,
                          const std::vector<uint32_t> &in_to_out_idx_list,
                          uint32_t queue_depth = UDF_ST_QUEUE_MAX_DEPTH) {
    ff::udf::UdfModelDef model_def;
    CreateUdfModel(model_def, "UdfExchangeQueues", __FILE__, {});
    auto udf_def = model_def.mutable_udf_def(0);

    auto proto_attrs = udf_def->mutable_attrs();
    ff::udf::AttrValue value;
    auto array = value.mutable_array();
    for (uint32_t in_to_out_idx : in_to_out_idx_list) {
      array->add_i(in_to_out_idx);
    }
    proto_attrs->insert({"in_to_out_idx_list", value});
    auto proto_path = WriteProto(model_def, "exchange_queues.pb");
    ff::deployer::ExecutorRequest_BatchLoadModelMessage batch_load_model_req;

    for (size_t idx = 0; idx < in_to_out_idx_list.size(); ++idx) {
      auto input_qid = CreateQueue(queue_depth);
      auto output_qid = CreateQueue(queue_depth);
      record_queue_ids.emplace_back(input_qid);
      record_queue_ids.emplace_back(output_qid);
      input_queues.emplace_back(input_qid);
      output_queues.emplace_back(output_qid);
    }
    AddModelToBatchModel(batch_load_model_req, proto_path, input_queues, output_queues);
    auto *input_align_attrs = batch_load_model_req.mutable_models(0)->mutable_input_align_attrs();
    input_align_attrs->set_align_max_cache_num(100);
    input_align_attrs->set_align_timeout(10000);
    input_align_attrs->set_drop_when_not_align(false);
    return WriteProto(batch_load_model_req, "exchange_queues_batch_model.pb");
  }

 protected:
  std::vector<uint32_t> record_queue_ids;
  uint32_t req_queue_id;
  uint32_t rsp_queue_id;
};

TEST_F(MultiInOutSTest, basic_test) {
  std::vector<uint32_t> input_queues;
  std::vector<uint32_t> output_queues;
  std::vector<uint32_t> in_to_out_idx_list = {3, 2, 1, 0};
  std::string batch_model_path = CreateModel(input_queues, output_queues, in_to_out_idx_list);

  auto batch_models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(batch_models.size(), 1);
  FlowFuncExecutor executor;
  auto ret = executor.Init(batch_models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = executor.Start();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);

  std::vector<int64_t> shape = {1, 2, 3, 4};
  std::vector<float> float_value = {0, 1.1, 2.2, 3.3, 4.4};
  std::vector<float> expect_value(float_value.size());
  for (size_t i = 0; i < input_queues.size(); ++i) {
    DataEnqueue(input_queues[i], shape, TensorDataType::DT_FLOAT, float_value[i]);
    expect_value[in_to_out_idx_list[i]] = float_value[i];
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
    std::vector<float> expect_output(CalcElementCnt(shape), expect_value[i]);
    CheckMbufData(out_mbuf, shape, TensorDataType::DT_FLOAT, expect_output.data(), expect_output.size());
    halMbufFree(out_mbuf);
  }

  executor.Stop(true);
  executor.WaitForStop();
  executor.Destroy();
}

TEST_F(MultiInOutSTest, basic_test_need_align) {
  std::vector<uint32_t> input_queues;
  std::vector<uint32_t> output_queues;
  std::vector<uint32_t> in_to_out_idx_list = {3, 2, 1, 0};
  std::string batch_model_path = CreateModel(input_queues, output_queues, in_to_out_idx_list);

  auto batch_models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(batch_models.size(), 1);
  FlowFuncExecutor executor;
  auto ret = executor.Init(batch_models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = executor.Start();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);

  std::vector<int64_t> shape = {1, 2, 3, 4};
  std::vector<float> float_value = {0, 1.1, 2.2, 3.3, 4.4};
  std::vector<float> expect_value(float_value.size());
  for (size_t i = 0; i < input_queues.size(); ++i) {
    expect_value[in_to_out_idx_list[i]] = float_value[i];
  }

  std::set<uint64_t> trans_id_set;
  constexpr uint64_t TEST_COUNT = 5;
  for (uint64_t trans_id = 0; trans_id < TEST_COUNT; ++trans_id) {
    trans_id_set.emplace(trans_id);
    DataEnqueue(input_queues[0], shape, TensorDataType::DT_FLOAT, float_value[0], trans_id, 1);
    DataEnqueue(input_queues[1], shape, TensorDataType::DT_FLOAT, float_value[1], (trans_id + 1) % TEST_COUNT, 1);
    DataEnqueue(input_queues[2], shape, TensorDataType::DT_FLOAT, float_value[2], (trans_id + 2) % TEST_COUNT, 1);
    DataEnqueue(input_queues[3], shape, TensorDataType::DT_FLOAT, float_value[3], (trans_id + 3) % TEST_COUNT, 1);
  }

  std::set<uint64_t> all_out_trans_id_set;
  for (uint64_t test_idx = 0; test_idx < TEST_COUNT; ++test_idx) {
    std::set<uint64_t> trans_id_per_out;
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
      std::vector<float> expect_output(CalcElementCnt(shape), expect_value[i]);
      CheckMbufData(out_mbuf, shape, TensorDataType::DT_FLOAT, expect_output.data(), expect_output.size());
      trans_id_per_out.emplace(GetMbufTransId(out_mbuf));
      halMbufFree(out_mbuf);
    }
    EXPECT_EQ(trans_id_per_out.size(), 1);
    all_out_trans_id_set.insert(trans_id_per_out.cbegin(), trans_id_per_out.cend());
  }
  EXPECT_EQ(trans_id_set, all_out_trans_id_set);
  executor.Stop(true);
  executor.WaitForStop();
  executor.Destroy();
}

TEST_F(MultiInOutSTest, basic_test_align_pad_empty_tensor) {
  std::vector<uint32_t> input_queues;
  std::vector<uint32_t> output_queues;
  std::vector<uint32_t> in_to_out_idx_list = {3, 2, 1, 0};
  std::string batch_model_path = CreateModel(input_queues, output_queues, in_to_out_idx_list, 256);

  auto batch_models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(batch_models.size(), 1);
  FlowFuncExecutor executor;
  auto ret = executor.Init(batch_models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = executor.Start();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);

  std::vector<int64_t> shape = {1, 2, 3, 4};
  std::vector<float> float_value = {0, 1.1, 2.2, 3.3, 4.4};
  std::vector<float> expect_value(float_value.size());
  for (size_t i = 0; i < input_queues.size(); ++i) {
    expect_value[in_to_out_idx_list[i]] = float_value[i];
  }

  constexpr uint64_t TEST_COUNT = 100;
  for (uint64_t trans_id = 0; trans_id < TEST_COUNT; ++trans_id) {
    DataEnqueue(input_queues[0], shape, TensorDataType::DT_FLOAT, float_value[0], trans_id, 1);
    DataEnqueue(input_queues[1], shape, TensorDataType::DT_FLOAT, float_value[1], trans_id, 1);
    DataEnqueue(input_queues[2], shape, TensorDataType::DT_FLOAT, float_value[2], 1000 + trans_id, 1);
    DataEnqueue(input_queues[3], shape, TensorDataType::DT_FLOAT, float_value[3], trans_id, 1);
  }

  constexpr uint32_t max_wait_second = 120;
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
    if (i != in_to_out_idx_list[2]) {
      std::vector<float> expect_output(CalcElementCnt(shape), expect_value[i]);
      CheckMbufData(out_mbuf, shape, TensorDataType::DT_FLOAT, expect_output.data(), expect_output.size());
    } else {
      const auto *head_msg = GetMbufHeadMsg(out_mbuf);
      EXPECT_EQ(head_msg->ret_code, FLOW_FUNC_ERR_DATA_ALIGN_FAILED);
    }
    halMbufFree(out_mbuf);
  }
  executor.Stop(true);
  executor.WaitForStop();
  executor.Destroy();
}

TEST_F(MultiInOutSTest, basic_test_input_queue_not_exist) {
  std::vector<uint32_t> input_queues;
  std::vector<uint32_t> output_queues;
  std::vector<uint32_t> in_to_out_idx_list = {3, 2, 1, 0};
  std::string batch_model_path = CreateModel(input_queues, output_queues, in_to_out_idx_list);

  auto batch_models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(batch_models.size(), 1);
  FlowFuncExecutor executor;
  auto ret = executor.Init(batch_models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = executor.Start();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);

  // enqueue to submit not empty event.
  std::vector<int64_t> shape = {1, 2, 3, 4};
  std::vector<float> float_value = {0, 1.1, 2.2, 3.3, 4.4};
  for (size_t i = 0; i < input_queues.size(); ++i) {
    DataEnqueue(input_queues[i], shape, TensorDataType::DT_FLOAT, float_value[i]);
    if (i == 0) {
      // wait for flow func init.
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
      // Destroy one InputQueue
      DestroyQueue(input_queues.back());
    }
  }

  // as some input queue error, executor will auto exit
  executor.WaitForStop();
  executor.Destroy();
}

TEST_F(MultiInOutSTest, basic_test_output_queue_not_exist) {
  std::vector<uint32_t> input_queues;
  std::vector<uint32_t> output_queues;
  std::vector<uint32_t> in_to_out_idx_list = {3, 2, 1, 0};
  std::string batch_model_path = CreateModel(input_queues, output_queues, in_to_out_idx_list);

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
    if (i == 0) {
      // wait for flow func init.
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
      // Destroy one outputQueue
      DestroyQueue(output_queues.back());
    }
  }

  executor.WaitForStop();
  executor.Destroy();
}

TEST_F(MultiInOutSTest, raw_data_basic_test) {
  std::vector<uint32_t> input_queues;
  std::vector<uint32_t> output_queues;
  std::vector<uint32_t> in_to_out_idx_list = {1, 0};
  std::string batch_model_path = CreateModel(input_queues, output_queues, in_to_out_idx_list);

  auto batch_models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(batch_models.size(), 1);
  FlowFuncExecutor executor;
  auto ret = executor.Init(batch_models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = executor.Start();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);

  float data1[4] = {1.1, 2.1, 3.1, 4.1};
  uint16_t data2[4] = {1, 2, 3, 4};
  auto mbuf_msg1 = MbufFlowMsg::AllocRawDataMsg(sizeof(data1), 0);
  EXPECT_NE(mbuf_msg1, nullptr);
  const auto &mbuf_msg_mbuf_info1 = mbuf_msg1->GetMbufInfo();
  auto mbuf_msg2 = MbufFlowMsg::AllocRawDataMsg(sizeof(data2), 0);
  EXPECT_NE(mbuf_msg2, nullptr);
  const auto &mbuf_msg_mbuf_info2 = mbuf_msg2->GetMbufInfo();
  EXPECT_NE(mbuf_msg_mbuf_info1.mbuf_addr, nullptr);
  EXPECT_GE(mbuf_msg_mbuf_info1.mbuf_len, sizeof(data1));
  memcpy_s(mbuf_msg_mbuf_info1.mbuf_addr, mbuf_msg_mbuf_info1.mbuf_len, data1, sizeof(data1));
  EXPECT_NE(mbuf_msg_mbuf_info2.mbuf_addr, nullptr);
  EXPECT_GE(mbuf_msg_mbuf_info2.mbuf_len, sizeof(data2));
  memcpy_s(mbuf_msg_mbuf_info2.mbuf_addr, mbuf_msg_mbuf_info2.mbuf_len, data2, sizeof(data2));

  Mbuf *mbuf1_ref = nullptr;
  halMbufCopyRef(mbuf_msg1->GetMbuf(), &mbuf1_ref);
  Mbuf *mbuf2_ref = nullptr;
  halMbufCopyRef(mbuf_msg2->GetMbuf(), &mbuf2_ref);

  halQueueEnQueue(0, input_queues[0], mbuf1_ref);
  halQueueEnQueue(0, input_queues[1], mbuf2_ref);

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
    if (i == 0) {
      CheckMbufData(out_mbuf, data2, sizeof(data2) / sizeof(data2[0]));
    } else {
      CheckMbufData(out_mbuf, data1, sizeof(data1) / sizeof(data1[0]));
    }
    halMbufFree(out_mbuf);
  }

  executor.Stop(true);
  executor.WaitForStop();
  executor.Destroy();
}

TEST_F(MultiInOutSTest, multi_loop_test_part_input) {
  std::vector<uint32_t> input_queues;
  std::vector<uint32_t> output_queues;
  std::vector<uint32_t> in_to_out_idx_list = {3, 2, 1, 0};
  std::string batch_model_path = CreateModel(input_queues, output_queues, in_to_out_idx_list);

  auto batch_models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(batch_models.size(), 1);
  FlowFuncExecutor executor;
  auto ret = executor.Init(batch_models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = executor.Start();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  std::vector<int64_t> shape = {1, 2, 3, 4};
  std::vector<float> float_value = {0, 1.1, 2.2, 3.3, 4.4};
  std::vector<float> expect_value(float_value.size());
  for (size_t i = 0; i < input_queues.size(); ++i) {
    DataEnqueue(input_queues[i], shape, TensorDataType::DT_FLOAT, float_value[i]);
    expect_value[in_to_out_idx_list[i]] = float_value[i];
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
    std::vector<float> expect_output(CalcElementCnt(shape), expect_value[i]);
    CheckMbufData(out_mbuf, shape, TensorDataType::DT_FLOAT, expect_output.data(), expect_output.size());
    halMbufFree(out_mbuf);
  }
  executor.Stop(true);
  executor.WaitForStop();
  executor.Destroy();
}

TEST_F(MultiInOutSTest, multi_loop_test_one_by_one) {
  std::vector<uint32_t> input_queues;
  std::vector<uint32_t> output_queues;
  std::vector<uint32_t> in_to_out_idx_list = {3, 2, 1, 0};
  std::string batch_model_path = CreateModel(input_queues, output_queues, in_to_out_idx_list);

  auto batch_models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(batch_models.size(), 1);
  FlowFuncExecutor executor;
  auto ret = executor.Init(batch_models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = executor.Start();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  int32_t loop_num = 20;
  while ((loop_num--) > 0) {
    std::vector<int64_t> shape = {1, 2, 3, 4};
    std::vector<float> float_value = {0, 1.1, 2.2, 3.3, 4.4};
    std::vector<float> expect_value(float_value.size());
    for (size_t i = 0; i < input_queues.size(); ++i) {
      DataEnqueue(input_queues[i], shape, TensorDataType::DT_FLOAT, float_value[i]);
      expect_value[in_to_out_idx_list[i]] = float_value[i];
    }

    constexpr uint64_t kMaxWaitInMs = 120 * 1000UL;
    for (size_t i = 0; i < output_queues.size(); i++) {
      void *out_mbuf_ptr = nullptr;
      uint64_t wait_in_ms = 0;
      while (wait_in_ms < kMaxWaitInMs) {
        auto drv_ret = halQueueDeQueue(0, output_queues[i], &out_mbuf_ptr);
        if (drv_ret == DRV_ERROR_QUEUE_EMPTY) {
          constexpr uint64_t kWaitInMsPerTime = 10;
          std::this_thread::sleep_for(std::chrono::milliseconds(kWaitInMsPerTime));
          wait_in_ms += kWaitInMsPerTime;
        } else {
          break;
        }
      }
      ASSERT_NE(out_mbuf_ptr, nullptr);
      Mbuf *out_mbuf = (Mbuf *)out_mbuf_ptr;
      std::vector<float> expect_output(CalcElementCnt(shape), expect_value[i]);
      CheckMbufData(out_mbuf, shape, TensorDataType::DT_FLOAT, expect_output.data(), expect_output.size());
      halMbufFree(out_mbuf);
    }
  }
  executor.Stop(true);
  executor.WaitForStop();
  executor.Destroy();
}
}  // namespace FlowFunc
