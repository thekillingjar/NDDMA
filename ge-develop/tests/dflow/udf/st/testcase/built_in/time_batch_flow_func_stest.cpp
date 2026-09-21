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
#include "utils/udf_utils.h"
#include "stub/udf_stub.h"
#include "common/data_utils.h"
#include "execute/flow_func_executor.h"
#include "model/flow_func_model.h"
#include "config/global_config.h"

namespace FlowFunc {
namespace {
constexpr uint64_t kWaitMsPerTime = 10;
}
class TimeBatchFlowFuncSTest : public testing::Test {
 protected:
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
    DestroyQueue(req_queue_id);
    DestroyQueue(rsp_queue_id);
    record_queue_ids.clear();
    GlobalConfig::Instance().SetMessageQueueIds(UINT32_MAX, UINT32_MAX);
    ClearStubEschedEvents();
  }

  std::string CreateOnlyOneBatchModel(std::vector<uint32_t> &inputs_qid, std::vector<uint32_t> &outputs_qid,
                                      const std::map<std::string, ff::udf::AttrValue> &attrs) {
    ff::udf::UdfModelDef model_def;
    CreateUdfModel(model_def, "_BuiltIn_TimeBatch", __FILE__, {});
    auto udf_def = model_def.mutable_udf_def(0);

    auto proto_attrs = udf_def->mutable_attrs();
    for (const auto &attr : attrs) {
      proto_attrs->insert({attr.first, attr.second});
    }
    auto proto_path = WriteProto(model_def, "TimeBatchFlowFuncSt.pb");
    ff::deployer::ExecutorRequest_BatchLoadModelMessage batch_load_model_req;
    for (size_t i = 0; i < inputs_qid.size(); ++i) {
      inputs_qid[i] = CreateQueue();
    }
    record_queue_ids.insert(record_queue_ids.end(), inputs_qid.begin(), inputs_qid.end());
    for (size_t i = 0; i < outputs_qid.size(); ++i) {
      outputs_qid[i] = CreateQueue();
    }
    record_queue_ids.insert(record_queue_ids.end(), outputs_qid.begin(), outputs_qid.end());
    AddModelToBatchModel(batch_load_model_req, proto_path, inputs_qid, outputs_qid);
    return WriteProto(batch_load_model_req, "batch_models.pb");
  }

 protected:
  std::vector<uint32_t> record_queue_ids;
  uint32_t req_queue_id;
  uint32_t rsp_queue_id;
};

// window = -1, batch_dim = -1, drop_remainder = false;
TEST_F(TimeBatchFlowFuncSTest, timebatch_with_flag_and_default_attr_value) {
  int32_t in_out_num = 2;
  std::vector<uint32_t> inputs_qid(in_out_num, 0);
  std::vector<uint32_t> outputs_qid(in_out_num, 0);
  std::map<std::string, ff::udf::AttrValue> attrs;
  ff::udf::AttrValue window;
  window.set_i(-1);
  attrs["window"] = window;
  ff::udf::AttrValue batch_dim;
  batch_dim.set_i(-1);
  attrs["batch_dim"] = batch_dim;
  ff::udf::AttrValue drop_remainder;
  drop_remainder.set_b(false);
  attrs["drop_remainder"] = drop_remainder;
  std::string batch_model_path = CreateOnlyOneBatchModel(inputs_qid, outputs_qid, attrs);
  auto batch_models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(batch_models.size(), 1);
  FlowFuncExecutor executor;
  auto ret = executor.Init(batch_models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = executor.Start();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);

  MbufHeadMsg head_msg{0};
  std::vector<int64_t> shape = {2, 3, 4};
  int32_t input_data[] = {4, 7, 5, 5, 1, 0, 9, 6, 7, 8, 6, 0, 6, 4, 6, 3, 7, 5, 5, 8, 5, 6, 7, 3};
  for (const auto qid : inputs_qid) {
    DataEnqueue(qid, shape, TensorDataType::DT_INT32, head_msg, input_data);
  }
  std::vector<void *> outs_mbuf_ptr(in_out_num, nullptr);
  std::vector<bool> get_out(in_out_num, false);
  constexpr uint64_t kMaxWaitInMs = 500;
  uint64_t wait_in_ms = 0;
  int32_t get_out_num = 0;
  while (wait_in_ms < kMaxWaitInMs) {
    for (size_t i = 0; i < outputs_qid.size(); ++i) {
      if (!get_out[i]) {
        auto drv_ret = halQueueDeQueue(0, outputs_qid[i], &(outs_mbuf_ptr[i]));
        if (drv_ret == DRV_ERROR_NONE) {
          get_out[i] = true;
          get_out_num++;
        }
      }
    }
    if (get_out_num == in_out_num) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(kWaitMsPerTime));
    wait_in_ms += kWaitMsPerTime;
  }
  for (size_t i = 0; i < outs_mbuf_ptr.size(); ++i) {
    EXPECT_EQ(outs_mbuf_ptr[i], nullptr);
    EXPECT_FALSE(get_out[i]);
  }
  EXPECT_EQ(get_out_num, 0);
  head_msg.flags = 1;
  for (const auto qid : inputs_qid) {
    DataEnqueue(qid, shape, TensorDataType::DT_INT32, head_msg, input_data);
  }
  wait_in_ms = 0;
  while (wait_in_ms < kMaxWaitInMs) {
    for (size_t i = 0; i < outputs_qid.size(); ++i) {
      if (!get_out[i]) {
        auto drv_ret = halQueueDeQueue(0, outputs_qid[i], &(outs_mbuf_ptr[i]));
        if (drv_ret == DRV_ERROR_NONE) {
          get_out[i] = true;
          get_out_num++;
        }
      }
    }
    if (get_out_num == in_out_num) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(kWaitMsPerTime));
    wait_in_ms += kWaitMsPerTime;
  }
  int32_t expect_out[] = {4, 7, 5, 5, 1, 0, 9, 6, 7, 8, 6, 0, 6, 4, 6, 3, 7, 5, 5, 8, 5, 6, 7, 3,
                          4, 7, 5, 5, 1, 0, 9, 6, 7, 8, 6, 0, 6, 4, 6, 3, 7, 5, 5, 8, 5, 6, 7, 3};
  shape.insert(shape.begin(), in_out_num);
  for (size_t i = 0; i < outs_mbuf_ptr.size(); ++i) {
    EXPECT_NE(outs_mbuf_ptr[i], nullptr);
    if (outs_mbuf_ptr[i] != nullptr) {
      Mbuf *out_mbuf = (Mbuf *)outs_mbuf_ptr[i];
      CheckMbufData(out_mbuf, shape, TensorDataType::DT_INT32, expect_out, CalcElementCnt(shape));
      halMbufFree(out_mbuf);
    }
  }
  executor.Stop(true);
  executor.WaitForStop();
}

// window = -1, batch_dim = 0, drop_remainder = false;
TEST_F(TimeBatchFlowFuncSTest, timebatch_with_flag_and_batch_dim_0_drop_remainder_false) {
  int32_t in_out_num = 2;
  std::vector<uint32_t> inputs_qid(in_out_num, 0);
  std::vector<uint32_t> outputs_qid(in_out_num, 0);
  std::map<std::string, ff::udf::AttrValue> attrs;
  ff::udf::AttrValue window;
  window.set_i(-1);
  attrs["window"] = window;
  ff::udf::AttrValue batch_dim;
  batch_dim.set_i(0);
  attrs["batch_dim"] = batch_dim;
  ff::udf::AttrValue drop_remainder;
  drop_remainder.set_b(false);
  attrs["drop_remainder"] = drop_remainder;
  std::string batch_model_path = CreateOnlyOneBatchModel(inputs_qid, outputs_qid, attrs);
  auto batch_models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(batch_models.size(), 1);
  FlowFuncExecutor executor;
  auto ret = executor.Init(batch_models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = executor.Start();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);

  MbufHeadMsg head_msg{0};
  std::vector<int64_t> shape = {2, 3, 4};
  int32_t input_data[] = {4, 7, 5, 5, 1, 0, 9, 6, 7, 8, 6, 0, 6, 4, 6, 3, 7, 5, 5, 8, 5, 6, 7, 3};
  for (const auto qid : inputs_qid) {
    DataEnqueue(qid, shape, TensorDataType::DT_INT32, head_msg, input_data);
  }
  std::vector<void *> outs_mbuf_ptr(in_out_num, nullptr);
  std::vector<bool> get_out(in_out_num, false);
  constexpr uint64_t kMaxWaitInMs = 500;
  uint64_t wait_in_ms = 0;
  int32_t get_out_num = 0;
  while (wait_in_ms < kMaxWaitInMs) {
    for (size_t i = 0; i < outputs_qid.size(); ++i) {
      if (!get_out[i]) {
        auto drv_ret = halQueueDeQueue(0, outputs_qid[i], &(outs_mbuf_ptr[i]));
        if (drv_ret == DRV_ERROR_NONE) {
          get_out[i] = true;
          get_out_num++;
        }
      }
    }
    if (get_out_num == in_out_num) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(kWaitMsPerTime));
    wait_in_ms += kWaitMsPerTime;
  }
  for (size_t i = 0; i < outs_mbuf_ptr.size(); ++i) {
    EXPECT_EQ(outs_mbuf_ptr[i], nullptr);
    EXPECT_FALSE(get_out[i]);
  }
  EXPECT_EQ(get_out_num, 0);

  head_msg.flags = 1;
  for (const auto qid : inputs_qid) {
    DataEnqueue(qid, shape, TensorDataType::DT_INT32, head_msg, input_data);
  }
  wait_in_ms = 0;
  while (wait_in_ms < kMaxWaitInMs) {
    for (size_t i = 0; i < outputs_qid.size(); ++i) {
      if (!get_out[i]) {
        auto drv_ret = halQueueDeQueue(0, outputs_qid[i], &(outs_mbuf_ptr[i]));
        if (drv_ret == DRV_ERROR_NONE) {
          get_out[i] = true;
          get_out_num++;
        }
      }
    }
    if (get_out_num == in_out_num) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(kWaitMsPerTime));
    wait_in_ms += kWaitMsPerTime;
  }
  int32_t expect_out[] = {4, 7, 5, 5, 1, 0, 9, 6, 7, 8, 6, 0, 6, 4, 6, 3, 7, 5, 5, 8, 5, 6, 7, 3,
                          4, 7, 5, 5, 1, 0, 9, 6, 7, 8, 6, 0, 6, 4, 6, 3, 7, 5, 5, 8, 5, 6, 7, 3};
  shape[batch_dim.i()] = shape[batch_dim.i()] * 2;
  for (size_t i = 0; i < outs_mbuf_ptr.size(); ++i) {
    EXPECT_NE(outs_mbuf_ptr[i], nullptr);
    if (outs_mbuf_ptr[i] != nullptr) {
      Mbuf *out_mbuf = (Mbuf *)outs_mbuf_ptr[i];
      CheckMbufData(out_mbuf, shape, TensorDataType::DT_INT32, expect_out, CalcElementCnt(shape));
      halMbufFree(out_mbuf);
    }
  }
  executor.Stop(true);
  executor.WaitForStop();
}

// window = -1, batch_dim = 1, drop_remainder = false;
TEST_F(TimeBatchFlowFuncSTest, timebatch_with_flag_and_batch_dim_1_drop_remainder_false) {
  int32_t in_out_num = 2;
  std::vector<uint32_t> inputs_qid(in_out_num, 0);
  std::vector<uint32_t> outputs_qid(in_out_num, 0);
  std::map<std::string, ff::udf::AttrValue> attrs;
  ff::udf::AttrValue window;
  window.set_i(-1);
  attrs["window"] = window;
  ff::udf::AttrValue batch_dim;
  batch_dim.set_i(1);
  attrs["batch_dim"] = batch_dim;
  ff::udf::AttrValue drop_remainder;
  drop_remainder.set_b(false);
  attrs["drop_remainder"] = drop_remainder;
  std::string batch_model_path = CreateOnlyOneBatchModel(inputs_qid, outputs_qid, attrs);
  auto batch_models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(batch_models.size(), 1);
  FlowFuncExecutor executor;
  auto ret = executor.Init(batch_models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = executor.Start();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);

  MbufHeadMsg head_msg{0};
  std::vector<int64_t> shape = {2, 3, 4};
  int32_t input_data[] = {4, 7, 5, 5, 1, 0, 9, 6, 7, 8, 6, 0, 6, 4, 6, 3, 7, 5, 5, 8, 5, 6, 7, 3};
  for (const auto qid : inputs_qid) {
    DataEnqueue(qid, shape, TensorDataType::DT_INT32, head_msg, input_data);
  }
  std::vector<void *> outs_mbuf_ptr(in_out_num, nullptr);
  std::vector<bool> get_out(in_out_num, false);
  constexpr uint64_t kMaxWaitInMs = 500;
  uint64_t wait_in_ms = 0;
  int32_t get_out_num = 0;
  while (wait_in_ms < kMaxWaitInMs) {
    for (size_t i = 0; i < outputs_qid.size(); ++i) {
      if (!get_out[i]) {
        auto drv_ret = halQueueDeQueue(0, outputs_qid[i], &(outs_mbuf_ptr[i]));
        if (drv_ret == DRV_ERROR_NONE) {
          get_out[i] = true;
          get_out_num++;
        }
      }
    }
    if (get_out_num == in_out_num) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(kWaitMsPerTime));
    wait_in_ms += kWaitMsPerTime;
  }
  for (size_t i = 0; i < outs_mbuf_ptr.size(); ++i) {
    EXPECT_EQ(outs_mbuf_ptr[i], nullptr);
    EXPECT_FALSE(get_out[i]);
  }
  EXPECT_EQ(get_out_num, 0);

  head_msg.flags = 1;
  for (const auto qid : inputs_qid) {
    DataEnqueue(qid, shape, TensorDataType::DT_INT32, head_msg, input_data);
  }
  wait_in_ms = 0;
  while (wait_in_ms < kMaxWaitInMs) {
    for (size_t i = 0; i < outputs_qid.size(); ++i) {
      if (!get_out[i]) {
        auto drv_ret = halQueueDeQueue(0, outputs_qid[i], &(outs_mbuf_ptr[i]));
        if (drv_ret == DRV_ERROR_NONE) {
          get_out[i] = true;
          get_out_num++;
        }
      }
    }
    if (get_out_num == in_out_num) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(kWaitMsPerTime));
    wait_in_ms += kWaitMsPerTime;
  }
  int32_t expect_out[] = {4, 7, 5, 5, 1, 0, 9, 6, 7, 8, 6, 0, 4, 7, 5, 5, 1, 0, 9, 6, 7, 8, 6, 0,
                          6, 4, 6, 3, 7, 5, 5, 8, 5, 6, 7, 3, 6, 4, 6, 3, 7, 5, 5, 8, 5, 6, 7, 3};
  shape[batch_dim.i()] = shape[batch_dim.i()] * 2;
  for (size_t i = 0; i < outs_mbuf_ptr.size(); ++i) {
    EXPECT_NE(outs_mbuf_ptr[i], nullptr);
    if (outs_mbuf_ptr[i] != nullptr) {
      Mbuf *out_mbuf = (Mbuf *)outs_mbuf_ptr[i];
      CheckMbufData(out_mbuf, shape, TensorDataType::DT_INT32, expect_out, CalcElementCnt(shape));
      halMbufFree(out_mbuf);
    }
  }
  executor.Stop(true);
  executor.WaitForStop();
}

// window = -1, batch_dim = 2, drop_remainder = false;
TEST_F(TimeBatchFlowFuncSTest, timebatch_with_flag_and_batch_dim_2_drop_remainder_false) {
  int32_t in_out_num = 2;
  std::vector<uint32_t> inputs_qid(in_out_num, 0);
  std::vector<uint32_t> outputs_qid(in_out_num, 0);
  std::map<std::string, ff::udf::AttrValue> attrs;
  ff::udf::AttrValue window;
  window.set_i(-1);
  attrs["window"] = window;
  ff::udf::AttrValue batch_dim;
  batch_dim.set_i(2);
  attrs["batch_dim"] = batch_dim;
  ff::udf::AttrValue drop_remainder;
  drop_remainder.set_b(false);
  attrs["drop_remainder"] = drop_remainder;
  std::string batch_model_path = CreateOnlyOneBatchModel(inputs_qid, outputs_qid, attrs);
  auto batch_models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(batch_models.size(), 1);
  FlowFuncExecutor executor;
  auto ret = executor.Init(batch_models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = executor.Start();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);

  MbufHeadMsg head_msg{0};
  std::vector<int64_t> shape = {2, 3, 4};
  int32_t input_data[] = {4, 7, 5, 5, 1, 0, 9, 6, 7, 8, 6, 0, 6, 4, 6, 3, 7, 5, 5, 8, 5, 6, 7, 3};
  for (const auto qid : inputs_qid) {
    DataEnqueue(qid, shape, TensorDataType::DT_INT32, head_msg, input_data);
  }
  std::vector<void *> outs_mbuf_ptr(in_out_num, nullptr);
  std::vector<bool> get_out(in_out_num, false);
  constexpr uint64_t kMaxWaitInMs = 500;
  uint64_t wait_in_ms = 0;
  int32_t get_out_num = 0;
  while (wait_in_ms < kMaxWaitInMs) {
    for (size_t i = 0; i < outputs_qid.size(); ++i) {
      if (!get_out[i]) {
        auto drv_ret = halQueueDeQueue(0, outputs_qid[i], &(outs_mbuf_ptr[i]));
        if (drv_ret == DRV_ERROR_NONE) {
          get_out[i] = true;
          get_out_num++;
        }
      }
    }
    if (get_out_num == in_out_num) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(kWaitMsPerTime));
    wait_in_ms += kWaitMsPerTime;
  }
  for (size_t i = 0; i < outs_mbuf_ptr.size(); ++i) {
    EXPECT_EQ(outs_mbuf_ptr[i], nullptr);
    EXPECT_FALSE(get_out[i]);
  }
  EXPECT_EQ(get_out_num, 0);

  head_msg.flags = 1;
  for (const auto qid : inputs_qid) {
    DataEnqueue(qid, shape, TensorDataType::DT_INT32, head_msg, input_data);
  }
  wait_in_ms = 0;
  while (wait_in_ms < kMaxWaitInMs) {
    for (size_t i = 0; i < outputs_qid.size(); ++i) {
      if (!get_out[i]) {
        auto drv_ret = halQueueDeQueue(0, outputs_qid[i], &(outs_mbuf_ptr[i]));
        if (drv_ret == DRV_ERROR_NONE) {
          get_out[i] = true;
          get_out_num++;
        }
      }
    }
    if (get_out_num == in_out_num) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(kWaitMsPerTime));
    wait_in_ms += kWaitMsPerTime;
  }
  int32_t expect_out[] = {4, 7, 5, 5, 4, 7, 5, 5, 1, 0, 9, 6, 1, 0, 9, 6, 7, 8, 6, 0, 7, 8, 6, 0,
                          6, 4, 6, 3, 6, 4, 6, 3, 7, 5, 5, 8, 7, 5, 5, 8, 5, 6, 7, 3, 5, 6, 7, 3};
  shape[batch_dim.i()] = shape[batch_dim.i()] * 2;
  for (size_t i = 0; i < outs_mbuf_ptr.size(); ++i) {
    EXPECT_NE(outs_mbuf_ptr[i], nullptr);
    if (outs_mbuf_ptr[i] != nullptr) {
      Mbuf *out_mbuf = (Mbuf *)outs_mbuf_ptr[i];
      CheckMbufData(out_mbuf, shape, TensorDataType::DT_INT32, expect_out, CalcElementCnt(shape));
      halMbufFree(out_mbuf);
    }
  }
  executor.Stop(true);
  executor.WaitForStop();
}

// window = -1, batch_dim = 2, drop_remainder = false;
TEST_F(TimeBatchFlowFuncSTest, timebatch_with_flag_and_batch_dim_3_drop_remainder_false) {
  int32_t in_out_num = 2;
  std::vector<uint32_t> inputs_qid(in_out_num, 0);
  std::vector<uint32_t> outputs_qid(in_out_num, 0);
  std::map<std::string, ff::udf::AttrValue> attrs;
  ff::udf::AttrValue window;
  window.set_i(-1);
  attrs["window"] = window;
  ff::udf::AttrValue batch_dim;
  batch_dim.set_i(2);
  attrs["batch_dim"] = batch_dim;
  ff::udf::AttrValue drop_remainder;
  drop_remainder.set_b(false);
  attrs["drop_remainder"] = drop_remainder;
  std::string batch_model_path = CreateOnlyOneBatchModel(inputs_qid, outputs_qid, attrs);
  auto batch_models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(batch_models.size(), 1);
  FlowFuncExecutor executor;
  auto ret = executor.Init(batch_models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = executor.Start();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);

  MbufHeadMsg head_msg{0};
  head_msg.ret_code = -1;
  std::vector<int64_t> shape = {2, 3, 4};
  int32_t input_data[] = {4, 7, 5, 5, 1, 0, 9, 6, 7, 8, 6, 0, 6, 4, 6, 3, 7, 5, 5, 8, 5, 6, 7, 3};
  for (const auto qid : inputs_qid) {
    DataEnqueue(qid, shape, TensorDataType::DT_INT32, head_msg, input_data);
  }
  std::vector<void *> outs_mbuf_ptr(in_out_num, nullptr);
  std::vector<bool> get_out(in_out_num, false);
  constexpr uint64_t kMaxWaitInMs = 500;
  uint64_t wait_in_ms = 0;
  int32_t get_out_num = 0;
  while (wait_in_ms < kMaxWaitInMs) {
    for (size_t i = 0; i < outputs_qid.size(); ++i) {
      if (!get_out[i]) {
        auto drv_ret = halQueueDeQueue(0, outputs_qid[i], &(outs_mbuf_ptr[i]));
        if (drv_ret == DRV_ERROR_NONE) {
          get_out[i] = true;
          get_out_num++;
        }
      }
    }
    if (get_out_num == in_out_num) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(kWaitMsPerTime));
    wait_in_ms += kWaitMsPerTime;
  }
  for (size_t i = 0; i < outs_mbuf_ptr.size(); ++i) {
    EXPECT_NE(outs_mbuf_ptr[i], nullptr);
    if (outs_mbuf_ptr[i] != nullptr) {
      Mbuf *out_mbuf = (Mbuf *)outs_mbuf_ptr[i];
      MbufStImpl *impl = (MbufStImpl *)out_mbuf;
      EXPECT_EQ(impl->head_msg.ret_code, -1);
      halMbufFree(out_mbuf);
    }
  }
  executor.Stop(true);
  executor.WaitForStop();
}

// window = -1, batch_dim = -1, drop_remainder = true;
TEST_F(TimeBatchFlowFuncSTest, timebatch_all_empty_eos_msg) {
  int32_t in_out_num = 2;
  std::vector<uint32_t> inputs_qid(in_out_num, 0);
  std::vector<uint32_t> outputs_qid(in_out_num, 0);
  std::map<std::string, ff::udf::AttrValue> attrs;
  ff::udf::AttrValue window;
  window.set_i(-1);
  attrs["window"] = window;
  ff::udf::AttrValue batch_dim;
  batch_dim.set_i(-1);
  attrs["batch_dim"] = batch_dim;
  ff::udf::AttrValue drop_remainder;
  drop_remainder.set_b(true);
  attrs["drop_remainder"] = drop_remainder;
  std::string batch_model_path = CreateOnlyOneBatchModel(inputs_qid, outputs_qid, attrs);
  auto batch_models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(batch_models.size(), 1);
  FlowFuncExecutor executor;
  auto ret = executor.Init(batch_models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = executor.Start();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);

  MbufHeadMsg head_msg{0};
  head_msg.flags = static_cast<uint32_t>(FlowFlag::FLOW_FLAG_EOS);
  head_msg.data_flag = 1U;
  std::vector<int64_t> shape = {0};
  int32_t input_data[0];
  for (const auto qid : inputs_qid) {
    DataEnqueue(qid, shape, TensorDataType::DT_INT32, head_msg, input_data);
  }
  std::vector<void *> outs_mbuf_ptr(in_out_num, nullptr);
  std::vector<bool> get_out(in_out_num, false);
  constexpr uint64_t kMaxWaitInMs = 5 * 1000;
  uint64_t wait_in_ms = 0;
  int32_t get_out_num = 0;
  while (wait_in_ms < kMaxWaitInMs) {
    for (size_t i = 0; i < outputs_qid.size(); ++i) {
      if (!get_out[i]) {
        auto drv_ret = halQueueDeQueue(0, outputs_qid[i], &(outs_mbuf_ptr[i]));
        if (drv_ret == DRV_ERROR_NONE) {
          get_out[i] = true;
          get_out_num++;
        }
      }
    }
    if (get_out_num == in_out_num) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(kWaitMsPerTime));
    wait_in_ms += kWaitMsPerTime;
  }
  for (size_t i = 0; i < outs_mbuf_ptr.size(); ++i) {
    EXPECT_NE(outs_mbuf_ptr[i], nullptr);
    if (outs_mbuf_ptr[i] != nullptr) {
      Mbuf *out_mbuf = (Mbuf *)outs_mbuf_ptr[i];
      MbufStImpl *impl = (MbufStImpl *)out_mbuf;
      EXPECT_EQ(impl->head_msg.flags, 1U);
      EXPECT_EQ(impl->head_msg.data_flag, 1U);
      halMbufFree(out_mbuf);
    }
  }
  executor.Stop(true);
  executor.WaitForStop();
}

// window = 5, batch_dim = 2, drop_remainder = true;
TEST_F(TimeBatchFlowFuncSTest, timebatch_publish_empty_eos_msg) {
  int32_t in_out_num = 2;
  std::vector<uint32_t> inputs_qid(in_out_num, 0);
  std::vector<uint32_t> outputs_qid(in_out_num, 0);
  std::map<std::string, ff::udf::AttrValue> attrs;
  ff::udf::AttrValue window;
  window.set_i(5);
  attrs["window"] = window;
  ff::udf::AttrValue batch_dim;
  batch_dim.set_i(2);
  attrs["batch_dim"] = batch_dim;
  ff::udf::AttrValue drop_remainder;
  drop_remainder.set_b(true);
  attrs["drop_remainder"] = drop_remainder;
  std::string batch_model_path = CreateOnlyOneBatchModel(inputs_qid, outputs_qid, attrs);
  auto batch_models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(batch_models.size(), 1);
  FlowFuncExecutor executor;
  auto ret = executor.Init(batch_models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = executor.Start();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);

  MbufHeadMsg head_msg{0};
  std::vector<int64_t> shape = {2, 3, 4};
  int32_t input_data[] = {4, 7, 5, 5, 1, 0, 9, 6, 7, 8, 6, 0, 6, 4, 6, 3, 7, 5, 5, 8, 5, 6, 7, 3};
  std::vector<void *> outs_mbuf_ptr(in_out_num, nullptr);
  std::vector<bool> get_out(in_out_num, false);
  constexpr uint64_t kMaxWaitInMs = 5 * 1000;
  uint64_t wait_in_ms = 0;
  int32_t get_out_num = 0;
  head_msg.flags = 1;
  for (const auto qid : inputs_qid) {
    DataEnqueue(qid, shape, TensorDataType::DT_INT32, head_msg, input_data);
  }
  while (wait_in_ms < kMaxWaitInMs) {
    for (size_t i = 0; i < outputs_qid.size(); ++i) {
      if (!get_out[i]) {
        auto drv_ret = halQueueDeQueue(0, outputs_qid[i], &(outs_mbuf_ptr[i]));
        if (drv_ret == DRV_ERROR_NONE) {
          get_out[i] = true;
          get_out_num++;
        }
      }
    }
    if (get_out_num == in_out_num) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(kWaitMsPerTime));
    wait_in_ms += kWaitMsPerTime;
  }
  for (size_t i = 0; i < outs_mbuf_ptr.size(); ++i) {
    EXPECT_NE(outs_mbuf_ptr[i], nullptr);
    if (outs_mbuf_ptr[i] != nullptr) {
      Mbuf *out_mbuf = (Mbuf *)outs_mbuf_ptr[i];
      MbufStImpl *impl = (MbufStImpl *)out_mbuf;
      EXPECT_EQ(impl->head_msg.flags, 1U);
      EXPECT_EQ(impl->head_msg.data_flag, 1U);
      halMbufFree(out_mbuf);
    }
  }
  executor.Stop(true);
  executor.WaitForStop();
}

// window = 5, batch_dim = 2, drop_remainder = false;
TEST_F(TimeBatchFlowFuncSTest, timebatch_publish_cache_msg) {
  int32_t in_out_num = 2;
  std::vector<uint32_t> inputs_qid(in_out_num, 0);
  std::vector<uint32_t> outputs_qid(in_out_num, 0);
  std::map<std::string, ff::udf::AttrValue> attrs;
  ff::udf::AttrValue window;
  window.set_i(5);
  attrs["window"] = window;
  ff::udf::AttrValue batch_dim;
  batch_dim.set_i(2);
  attrs["batch_dim"] = batch_dim;
  ff::udf::AttrValue drop_remainder;
  drop_remainder.set_b(false);
  attrs["drop_remainder"] = drop_remainder;
  std::string batch_model_path = CreateOnlyOneBatchModel(inputs_qid, outputs_qid, attrs);
  auto batch_models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(batch_models.size(), 1);
  FlowFuncExecutor executor;
  auto ret = executor.Init(batch_models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = executor.Start();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);

  MbufHeadMsg head_msg{0};
  std::vector<int64_t> shape = {2, 3, 4};
  int32_t input_data[] = {4, 7, 5, 5, 1, 0, 9, 6, 7, 8, 6, 0, 6, 4, 6, 3, 7, 5, 5, 8, 5, 6, 7, 3};
  std::vector<void *> outs_mbuf_ptr(in_out_num, nullptr);
  std::vector<bool> get_out(in_out_num, false);

  constexpr uint64_t kMaxWaitInMs = 5 * 1000;
  uint64_t wait_in_ms = 0;
  int32_t get_out_num = 0;
  for (const auto qid : inputs_qid) {
    DataEnqueue(qid, shape, TensorDataType::DT_INT32, head_msg, input_data);
  }
  head_msg.flags = 1;
  head_msg.data_flag |= 1U;
  for (const auto qid : inputs_qid) {
    DataEnqueue(qid, {0}, TensorDataType::DT_INT32, head_msg, input_data);
  }
  while (wait_in_ms < kMaxWaitInMs) {
    for (size_t i = 0; i < outputs_qid.size(); ++i) {
      if (!get_out[i]) {
        auto drv_ret = halQueueDeQueue(0, outputs_qid[i], &(outs_mbuf_ptr[i]));
        if (drv_ret == DRV_ERROR_NONE) {
          get_out[i] = true;
          get_out_num++;
        }
      }
    }
    if (get_out_num == in_out_num) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(kWaitMsPerTime));
    wait_in_ms += kWaitMsPerTime;
  }
  for (size_t i = 0; i < outs_mbuf_ptr.size(); ++i) {
    EXPECT_NE(outs_mbuf_ptr[i], nullptr);
    if (outs_mbuf_ptr[i] != nullptr) {
      Mbuf *out_mbuf = (Mbuf *)outs_mbuf_ptr[i];
      MbufStImpl *impl = (MbufStImpl *)out_mbuf;
      EXPECT_EQ(impl->head_msg.flags, 1U);
      EXPECT_EQ(impl->head_msg.data_flag, 0U);
      halMbufFree(out_mbuf);
    }
  }
  executor.Stop(true);
  executor.WaitForStop();
}
}  // namespace FlowFunc