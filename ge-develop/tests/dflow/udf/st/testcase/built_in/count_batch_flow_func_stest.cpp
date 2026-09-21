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
#include "securec.h"
#include "config/global_config.h"

namespace FlowFunc {
namespace {
// 每次等10ms
constexpr uint64_t kWaitMsPerTime = 10;
}
class CountBatchFlowFuncSTest : public testing::Test {
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
    record_queue_ids.clear();
    DestroyQueue(req_queue_id);
    DestroyQueue(rsp_queue_id);
    GlobalMockObject::verify();
    GlobalConfig::Instance().SetMessageQueueIds(UINT32_MAX, UINT32_MAX);
    ClearStubEschedEvents();
  }

  std::string CreateOnlyOneBatchModel(std::vector<uint32_t> &inputs_qid, std::vector<uint32_t> &outputs_qid,
                                      const std::map<std::string, ff::udf::AttrValue> &attrs) {
    ff::udf::UdfModelDef model_def;
    CreateUdfModel(model_def, "_BuiltIn_CountBatch", __FILE__, {});
    auto udf_def = model_def.mutable_udf_def(0);

    auto proto_attrs = udf_def->mutable_attrs();
    for (const auto &attr : attrs) {
      proto_attrs->insert({attr.first, attr.second});
    }
    auto proto_path = WriteProto(model_def, "CountBatchFlowFuncSt.pb");
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
  void CreateCountBatchAttr(int64_t batch_size, int64_t timeout, bool padding, int64_t slide_stride,
                            std::map<std::string, ff::udf::AttrValue> &attrs) {
    ff::udf::AttrValue batch_size_val;
    batch_size_val.set_i(batch_size);
    attrs["batch_size"] = batch_size_val;
    if (timeout >= -1) {
      ff::udf::AttrValue timeout_val;
      timeout_val.set_i(timeout);
      attrs["timeout"] = timeout_val;
    }

    ff::udf::AttrValue padding_val;
    padding_val.set_b(padding);
    attrs["padding"] = padding_val;
    if (slide_stride >= 0) {
      ff::udf::AttrValue slide_stride_val;
      slide_stride_val.set_i(slide_stride);
      attrs["slide_stride"] = slide_stride_val;
    }
    return;
  }

 protected:
  std::vector<uint32_t> record_queue_ids;
  uint32_t req_queue_id = UINT32_MAX;
  uint32_t rsp_queue_id = UINT32_MAX;;
};

TEST_F(CountBatchFlowFuncSTest, parse_and_init_model) {
  int32_t in_out_num = 2;
  std::vector<uint32_t> inputs_qid(in_out_num, 0);
  std::vector<uint32_t> outputs_qid(in_out_num, 0);
  std::map<std::string, ff::udf::AttrValue> attrs;
  std::string batch_model_path = CreateOnlyOneBatchModel(inputs_qid, outputs_qid, attrs);
  auto batch_models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(batch_models.size(), 1);
  FlowFuncExecutor executor;
  auto ret = executor.Init(batch_models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = executor.Start();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  executor.Stop(true);
  executor.WaitForStop();
  executor.Destroy();
}

TEST_F(CountBatchFlowFuncSTest, abnormal_test_with_invalid_timeout_attr) {
  int32_t in_out_num = 2;
  std::vector<uint32_t> inputs_qid(in_out_num, 0);
  std::vector<uint32_t> outputs_qid(in_out_num, 0);
  std::map<std::string, ff::udf::AttrValue> attrs;
  ff::udf::AttrValue batch_size_val;
  batch_size_val.set_i(100);
  attrs["batch_size"] = batch_size_val;
  ff::udf::AttrValue timeout_val;
  timeout_val.set_i(20);
  attrs["timeout"] = timeout_val;
  std::string batch_model_path = CreateOnlyOneBatchModel(inputs_qid, outputs_qid, attrs);
  auto batch_models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(batch_models.size(), 1);
  FlowFuncExecutor executor;
  auto ret = executor.Init(batch_models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = executor.Start();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  executor.Stop(true);
  executor.WaitForStop();
  executor.Destroy();
}

TEST_F(CountBatchFlowFuncSTest, abnormal_test_with_no_timeout) {
  int32_t in_out_num = 2;
  std::vector<uint32_t> inputs_qid(in_out_num, 0);
  std::vector<uint32_t> outputs_qid(in_out_num, 0);
  std::map<std::string, ff::udf::AttrValue> attrs;
  ff::udf::AttrValue batch_size_val;
  batch_size_val.set_i(200);
  attrs["batch_size"] = batch_size_val;
  std::string batch_model_path = CreateOnlyOneBatchModel(inputs_qid, outputs_qid, attrs);
  auto batch_models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(batch_models.size(), 1);
  FlowFuncExecutor executor;
  auto ret = executor.Init(batch_models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = executor.Start();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  executor.Stop(true);
  executor.WaitForStop();
  executor.Destroy();
}

TEST_F(CountBatchFlowFuncSTest, abnormal_test_invalid_timeout) {
  int32_t in_out_num = 2;
  std::vector<uint32_t> inputs_qid(in_out_num, 0);
  std::vector<uint32_t> outputs_qid(in_out_num, 0);
  FlowFuncExecutor executor;
  std::map<std::string, ff::udf::AttrValue> attrs;
  // invalid timeout
  CreateCountBatchAttr(3, -1, false, 0, attrs);
  std::string batch_model_path = CreateOnlyOneBatchModel(inputs_qid, outputs_qid, attrs);
  auto batch_models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(batch_models.size(), 1);
  auto ret = executor.Init(batch_models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = executor.Start();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  executor.Stop(true);
  executor.WaitForStop();
  executor.Destroy();
}

TEST_F(CountBatchFlowFuncSTest, abnormal_test_no_slide_stride_attr) {
  int32_t in_out_num = 2;
  std::vector<uint32_t> inputs_qid(in_out_num, 0);
  std::vector<uint32_t> outputs_qid(in_out_num, 0);
  FlowFuncExecutor executor;
  // no slide_stride attr
  std::map<std::string, ff::udf::AttrValue> attrs;
  CreateCountBatchAttr(3, 3, false, -2, attrs);
  std::string batch_model_path = CreateOnlyOneBatchModel(inputs_qid, outputs_qid, attrs);
  auto batch_models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(batch_models.size(), 1);
  auto ret = executor.Init(batch_models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = executor.Start();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  executor.Stop(true);
  executor.WaitForStop();
  executor.Destroy();
}

TEST_F(CountBatchFlowFuncSTest, abnormal_test_with_input_null) {
  int32_t in_out_num = 2;
  std::vector<uint32_t> inputs_qid(in_out_num, 0);
  std::vector<uint32_t> outputs_qid(in_out_num, 0);
  std::map<std::string, ff::udf::AttrValue> attrs;
  CreateCountBatchAttr(3, 0, false, 0, attrs);
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
  std::vector<int64_t> origin_shape = {2, 3};
  int32_t input_data[] = {4, 7, 5, 5, 1, 0};
  for (const auto qid : inputs_qid) {
    DataEnqueue(qid, origin_shape, TensorDataType::DT_INT32, head_msg, input_data);
  }
  std::vector<void *> outs_mbuf_ptr(in_out_num, nullptr);
  std::vector<bool> get_out(in_out_num, false);
  // 最长等5秒
  constexpr uint64_t kMaxWaitTimeInMs = 5 * 1000UL;
  uint64_t wait_time_in_ms = 0;
  int32_t get_out_num = 0;
  while (wait_time_in_ms < kMaxWaitTimeInMs) {
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
    usleep(kWaitMsPerTime * 1000);
    wait_time_in_ms += kWaitMsPerTime;
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
  executor.Destroy();
}

// batch_size = 3, timeout = 0, slide_stride = 0
TEST_F(CountBatchFlowFuncSTest, basic_test_only_batchSize) {
  int32_t in_out_num = 2;
  std::vector<uint32_t> inputs_qid(in_out_num, 0);
  std::vector<uint32_t> outputs_qid(in_out_num, 0);
  std::map<std::string, ff::udf::AttrValue> attrs;
  int64_t size = 3;
  CreateCountBatchAttr(3, 0, false, 0, attrs);
  std::string batch_model_path = CreateOnlyOneBatchModel(inputs_qid, outputs_qid, attrs);
  auto batch_models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(batch_models.size(), 1);
  FlowFuncExecutor executor;
  auto ret = executor.Init(batch_models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = executor.Start();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);

  MbufHeadMsg head_msg{0};
  std::vector<int64_t> shape = {2, 3};
  int32_t input_data[] = {4, 7, 5, 5, 1, 0};
  std::vector<void *> outs_mbuf_ptr(in_out_num, nullptr);
  std::vector<bool> get_out(in_out_num, false);
  // 最长等0.5秒
  constexpr uint64_t kMaxWaitTimeInMs = 500;
  uint64_t wait_time_in_ms = 0;
  int32_t get_out_num = 0;
  for (int64_t loop_index = 0; loop_index < size; ++loop_index) {
    wait_time_in_ms = 0;
    get_out_num = 0;
    outs_mbuf_ptr.resize(in_out_num, nullptr);
    get_out.resize(in_out_num, false);
    for (const auto qid : inputs_qid) {
      DataEnqueue(qid, shape, TensorDataType::DT_INT32, head_msg, input_data);
    }
    while (wait_time_in_ms < kMaxWaitTimeInMs) {
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
      usleep(kWaitMsPerTime * 1000);
      wait_time_in_ms += kWaitMsPerTime;
    }
    if (loop_index < size - 1) {
      for (size_t i = 0; i < outs_mbuf_ptr.size(); ++i) {
        EXPECT_EQ(outs_mbuf_ptr[i], nullptr);
      }
    } else {
      int32_t expect_out[] = {4, 7, 5, 5, 1, 0, 4, 7, 5, 5, 1, 0, 4, 7, 5, 5, 1, 0, 4, 7, 5, 5, 1, 0, 4, 7, 5, 5, 1, 0};
      shape = {size, 2, 3};
      for (size_t i = 0; i < outs_mbuf_ptr.size(); ++i) {
        EXPECT_NE(outs_mbuf_ptr[i], nullptr);
        if (outs_mbuf_ptr[i] != nullptr) {
          Mbuf *out_mbuf = (Mbuf *)outs_mbuf_ptr[i];
          CheckMbufData(out_mbuf, shape, TensorDataType::DT_INT32, expect_out, CalcElementCnt(shape));
          halMbufFree(out_mbuf);
        }
      }
    }
  }
  executor.Stop(true);
  executor.WaitForStop();
  executor.Destroy();
}

// batch_size = 3, timeout = 0, slide_stride = 1
TEST_F(CountBatchFlowFuncSTest, basic_test_with_slide_stride) {
  int32_t in_out_num = 2;
  std::vector<uint32_t> inputs_qid(in_out_num, 0);
  std::vector<uint32_t> outputs_qid(in_out_num, 0);
  std::map<std::string, ff::udf::AttrValue> attrs;
  int64_t size = 3;
  CreateCountBatchAttr(3, 0, false, 1, attrs);
  std::string batch_model_path = CreateOnlyOneBatchModel(inputs_qid, outputs_qid, attrs);
  auto batch_models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(batch_models.size(), 1);
  FlowFuncExecutor executor;
  auto ret = executor.Init(batch_models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = executor.Start();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);

  MbufHeadMsg head_msg{0};
  std::vector<int64_t> origin_shape = {2, 3};
  std::vector<int64_t> new_shape = {2, 3};
  int32_t input_data[] = {4, 7, 5, 5, 1, 0};
  std::vector<void *> outs_mbuf_ptr(in_out_num, nullptr);
  std::vector<bool> get_out(in_out_num, false);
  // 最长等1秒
  constexpr uint64_t kMaxWaitTimeInMs = 1000;
  uint64_t wait_time_in_ms = 0;
  int32_t get_out_num = 0;
  for (int64_t loop_index = 0; loop_index < size; ++loop_index) {
    wait_time_in_ms = 0;
    get_out_num = 0;
    for (int32_t num_index = 0; num_index < in_out_num; ++num_index) {
      get_out[num_index] = false;
      outs_mbuf_ptr[num_index] = nullptr;
    }
    get_out.resize(in_out_num, false);
    for (const auto qid : inputs_qid) {
      DataEnqueue(qid, origin_shape, TensorDataType::DT_INT32, head_msg, input_data);
    }
    while (wait_time_in_ms < kMaxWaitTimeInMs) {
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
      usleep(kWaitMsPerTime * 1000);
      wait_time_in_ms += kMaxWaitTimeInMs;
    }
    if (loop_index < size - 1) {
      for (size_t i = 0; i < outs_mbuf_ptr.size(); ++i) {
        EXPECT_EQ(outs_mbuf_ptr[i], nullptr);
      }
    } else {
      int32_t expect_out[] = {4, 7, 5, 5, 1, 0, 4, 7, 5, 5, 1, 0, 4, 7, 5, 5, 1, 0};
      new_shape = {size, 2, 3};
      for (size_t i = 0; i < outs_mbuf_ptr.size(); ++i) {
        if (outs_mbuf_ptr[i] != nullptr) {
          Mbuf *out_mbuf = (Mbuf *)outs_mbuf_ptr[i];
          CheckMbufData(out_mbuf, new_shape, TensorDataType::DT_INT32, expect_out, CalcElementCnt(new_shape));
          halMbufFree(out_mbuf);
        }
      }
    }
  }
  executor.Stop(true);
  executor.WaitForStop();
  executor.Destroy();
}

// batch_size = 0, timeout = 100, padding = true, slide_stride = 0
TEST_F(CountBatchFlowFuncSTest, abnormal_test_with_no_batchSize) {
  int32_t in_out_num = 2;
  std::vector<uint32_t> inputs_qid(in_out_num, 0);
  std::vector<uint32_t> outputs_qid(in_out_num, 0);
  std::map<std::string, ff::udf::AttrValue> attrs;
  ff::udf::AttrValue timeout_val;
  timeout_val.set_i(100);
  attrs["timeout"] = timeout_val;
  ff::udf::AttrValue padding;
  padding.set_b(true);
  attrs["padding"] = padding;
  std::string batch_model_path = CreateOnlyOneBatchModel(inputs_qid, outputs_qid, attrs);
  auto batch_models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(batch_models.size(), 1);
  FlowFuncExecutor executor;
  auto ret = executor.Init(batch_models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = executor.Start();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);

  MbufHeadMsg head_msg{0};
  std::vector<int64_t> origin_shape = {2, 3};
  std::vector<int64_t> new_shape = {2, 3};
  int32_t input_data[] = {4, 7, 5, 5, 1, 0};
  std::vector<void *> outs_mbuf_ptr(in_out_num, nullptr);
  std::vector<bool> get_out(in_out_num, false);

  // 最长等1秒
  constexpr uint64_t kMaxWaitTimeInMs = 1000;
  uint64_t wait_time_in_ms = 0;
  int32_t get_out_num = 0;
  for (const auto qid : inputs_qid) {
    DataEnqueue(qid, origin_shape, TensorDataType::DT_INT32, head_msg, input_data);
    DataEnqueue(qid, origin_shape, TensorDataType::DT_INT32, head_msg, input_data);
  }
  while (wait_time_in_ms < kMaxWaitTimeInMs) {
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
    usleep(kWaitMsPerTime * 1000);
    wait_time_in_ms += kWaitMsPerTime;
  }
  for (size_t i = 0; i < outs_mbuf_ptr.size(); ++i) {
    EXPECT_EQ(outs_mbuf_ptr[i], nullptr);
  }
  executor.Stop(true);
  executor.WaitForStop();
  executor.Destroy();
}

// batch_size = 3, timeout = 100, padding = false, slide_stride = 0
TEST_F(CountBatchFlowFuncSTest, timeout_test_with_no_padding) {
  int32_t in_out_num = 2;
  std::vector<uint32_t> inputs_qid(in_out_num, 0);
  std::vector<uint32_t> outputs_qid(in_out_num, 0);
  std::map<std::string, ff::udf::AttrValue> attrs;
  int64_t size = 3;
  CreateCountBatchAttr(3, 100, false, 0, attrs);
  std::string batch_model_path = CreateOnlyOneBatchModel(inputs_qid, outputs_qid, attrs);
  auto batch_models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(batch_models.size(), 1);
  FlowFuncExecutor executor;
  auto ret = executor.Init(batch_models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = executor.Start();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);

  MbufHeadMsg head_msg{0};
  std::vector<int64_t> origin_shape = {2, 3};
  std::vector<int64_t> new_shape = {2, 3};
  int32_t input_data[] = {4, 7, 5, 5, 1, 0};
  std::vector<void *> outs_mbuf_ptr(in_out_num, nullptr);
  std::vector<bool> get_out(in_out_num, false);
  // 最长等1秒
  constexpr uint64_t kMaxWaitTimeInMs = 1000;
  uint64_t wait_time_in_ms = 0;
  int32_t get_out_num = 0;
  for (const auto qid : inputs_qid) {
    DataEnqueue(qid, origin_shape, TensorDataType::DT_INT32, head_msg, input_data);
    DataEnqueue(qid, origin_shape, TensorDataType::DT_INT32, head_msg, input_data);
  }
  while (wait_time_in_ms < kMaxWaitTimeInMs) {
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
    usleep(kWaitMsPerTime * 1000);
    wait_time_in_ms += kWaitMsPerTime;
  }
  int32_t expect_out[] = {4, 7, 5, 5, 1, 0, 4, 7, 5, 5, 1, 0};
  new_shape = {2, 2, 3};
  for (size_t i = 0; i < outs_mbuf_ptr.size(); ++i) {
    EXPECT_NE(outs_mbuf_ptr[i], nullptr);
    if (outs_mbuf_ptr[i] != nullptr) {
      Mbuf *out_mbuf = (Mbuf *)outs_mbuf_ptr[i];
      CheckMbufData(out_mbuf, new_shape, TensorDataType::DT_INT32, expect_out, CalcElementCnt(new_shape));
      halMbufFree(out_mbuf);
    }
  }
  executor.Stop(true);
  executor.WaitForStop();
  executor.Destroy();
}

TEST_F(CountBatchFlowFuncSTest, input_data_type_error) {
  int32_t in_out_num = 2;
  std::vector<uint32_t> inputs_qid(in_out_num, 0);
  std::vector<uint32_t> outputs_qid(in_out_num, 0);
  std::map<std::string, ff::udf::AttrValue> attrs;
  int64_t size = 3;
  CreateCountBatchAttr(3, 100, true, 0, attrs);
  std::string batch_model_path = CreateOnlyOneBatchModel(inputs_qid, outputs_qid, attrs);
  auto batch_models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(batch_models.size(), 1);
  FlowFuncExecutor executor;
  auto ret = executor.Init(batch_models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = executor.Start();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);

  MbufHeadMsg head_msg{0};
  std::vector<int64_t> origin_shape = {2, 3};
  std::vector<int64_t> new_shape = {2, 3};
  int32_t input_data1[] = {4, 7, 5, 5, 1, 0};
  int64_t input_data2[] = {4, 7, 5, 5, 1, 0};
  std::vector<void *> outs_mbuf_ptr(in_out_num, nullptr);
  std::vector<bool> get_out(in_out_num, false);
  // 最长等1秒
  constexpr uint64_t kMaxWaitTimeInMs = 1000;
  uint64_t wait_time_in_ms = 0;
  int32_t get_out_num = 0;
  for (const auto qid : inputs_qid) {
    DataEnqueue(qid, origin_shape, TensorDataType::DT_INT32, head_msg, input_data1);
    DataEnqueue(qid, origin_shape, TensorDataType::DT_INT64, head_msg, input_data2);
  }
  while (wait_time_in_ms < kMaxWaitTimeInMs) {
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
    usleep(kWaitMsPerTime * 1000);
    wait_time_in_ms += kWaitMsPerTime;
  }
  for (size_t i = 0; i < outs_mbuf_ptr.size(); ++i) {
    EXPECT_NE(outs_mbuf_ptr[i], nullptr);
    if (outs_mbuf_ptr[i] != nullptr) {
      Mbuf *out_mbuf = (Mbuf *)outs_mbuf_ptr[i];
      halMbufFree(out_mbuf);
    }
  }

  executor.Stop(true);
  executor.WaitForStop();
  executor.Destroy();
}

TEST_F(CountBatchFlowFuncSTest, input_shape_error) {
  int32_t in_out_num = 2;
  std::vector<uint32_t> inputs_qid(in_out_num, 0);
  std::vector<uint32_t> outputs_qid(in_out_num, 0);
  std::map<std::string, ff::udf::AttrValue> attrs;
  int64_t size = 3;
  CreateCountBatchAttr(3, 100, true, 0, attrs);
  std::string batch_model_path = CreateOnlyOneBatchModel(inputs_qid, outputs_qid, attrs);
  auto batch_models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(batch_models.size(), 1);
  FlowFuncExecutor executor;
  auto ret = executor.Init(batch_models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = executor.Start();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);

  MbufHeadMsg head_msg{0};
  int32_t input_data1[] = {4, 7, 5, 5, 1, 0};
  int32_t input_data2[] = {4, 7, 5, 5};
  std::vector<void *> outs_mbuf_ptr(in_out_num, nullptr);
  std::vector<bool> get_out(in_out_num, false);
  // 最长等1秒
  constexpr uint64_t kMaxWaitTimeInMs = 1000;
  uint64_t wait_time_in_ms = 0;
  int32_t get_out_num = 0;
  for (const auto qid : inputs_qid) {
    DataEnqueue(qid, {2, 3}, TensorDataType::DT_INT32, head_msg, input_data1);
    DataEnqueue(qid, {2, 2}, TensorDataType::DT_INT32, head_msg, input_data2);
  }
  while (wait_time_in_ms < kMaxWaitTimeInMs) {
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
    usleep(kWaitMsPerTime * 1000);
    wait_time_in_ms += kWaitMsPerTime;
  }
  for (size_t i = 0; i < outs_mbuf_ptr.size(); ++i) {
    EXPECT_NE(outs_mbuf_ptr[i], nullptr);
    if (outs_mbuf_ptr[i] != nullptr) {
      Mbuf *out_mbuf = (Mbuf *)outs_mbuf_ptr[i];
      halMbufFree(out_mbuf);
    }
  }

  executor.Stop(true);
  executor.WaitForStop();
  executor.Destroy();
}

// batch_size = 3, timeout = 100, padding = true, slide_stride = 0
TEST_F(CountBatchFlowFuncSTest, timeout_test_with_padding) {
  int32_t in_out_num = 2;
  std::vector<uint32_t> inputs_qid(in_out_num, 0);
  std::vector<uint32_t> outputs_qid(in_out_num, 0);
  std::map<std::string, ff::udf::AttrValue> attrs;
  int64_t size = 3;
  CreateCountBatchAttr(3, 100, true, 0, attrs);
  std::string batch_model_path = CreateOnlyOneBatchModel(inputs_qid, outputs_qid, attrs);
  auto batch_models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(batch_models.size(), 1);
  FlowFuncExecutor executor;
  auto ret = executor.Init(batch_models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = executor.Start();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);

  MbufHeadMsg head_msg{0};
  std::vector<int64_t> origin_shape = {2, 3};
  std::vector<int64_t> new_shape = {2, 3};
  int32_t input_data[] = {4, 7, 5, 5, 1, 0};
  std::vector<void *> outs_mbuf_ptr(in_out_num, nullptr);
  std::vector<bool> get_out(in_out_num, false);
  // 最长等5秒
  constexpr uint64_t kMaxWaitTimeInMs = 1000;
  uint64_t wait_time_in_ms = 0;
  int32_t get_out_num = 0;
  for (const auto qid : inputs_qid) {
    DataEnqueue(qid, origin_shape, TensorDataType::DT_INT32, head_msg, input_data);
    DataEnqueue(qid, origin_shape, TensorDataType::DT_INT32, head_msg, input_data);
  }
  while (wait_time_in_ms < kMaxWaitTimeInMs) {
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
    usleep(kWaitMsPerTime * 1000);
    wait_time_in_ms += kWaitMsPerTime;
  }
  int32_t expect_out[] = {4, 7, 5, 5, 1, 0, 4, 7, 5, 5, 1, 0, 0, 0, 0, 0, 0, 0};
  new_shape = {size, 2, 3};
  for (size_t i = 0; i < outs_mbuf_ptr.size(); ++i) {
    EXPECT_NE(outs_mbuf_ptr[i], nullptr);
    if (outs_mbuf_ptr[i] != nullptr) {
      Mbuf *out_mbuf = (Mbuf *)outs_mbuf_ptr[i];
      CheckMbufData(out_mbuf, new_shape, TensorDataType::DT_INT32, expect_out, CalcElementCnt(new_shape));
      halMbufFree(out_mbuf);
    }
  }
  executor.Stop(true);
  executor.WaitForStop();
  executor.Destroy();
}

// batch_size = 3, timeout = 100, padding = true, slide_stride = 3
TEST_F(CountBatchFlowFuncSTest, basic_test_all) {
  int32_t in_out_num = 2;
  std::vector<uint32_t> inputs_qid(in_out_num, 0);
  std::vector<uint32_t> outputs_qid(in_out_num, 0);
  std::map<std::string, ff::udf::AttrValue> attrs;
  int64_t size = 5;
  CreateCountBatchAttr(5, 300, true, 2, attrs);
  std::string batch_model_path = CreateOnlyOneBatchModel(inputs_qid, outputs_qid, attrs);
  auto batch_models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(batch_models.size(), 1);
  FlowFuncExecutor executor;
  auto ret = executor.Init(batch_models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = executor.Start();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  MbufHeadMsg head_msg{0};
  std::vector<int64_t> origin_shape = {2, 2};
  std::vector<int64_t> new_shape = {2, 2};
  int32_t input_data[] = {4, 7, 5, 5};
  std::vector<void *> outs_mbuf_ptr(in_out_num, nullptr);
  std::vector<bool> get_out(in_out_num, false);
  // 最长等5秒
  constexpr uint64_t kMaxWaitTimeInMs = 5 * 1000;
  uint64_t wait_time_in_ms = 0;
  int32_t get_out_num = 0;
  for (const auto qid : inputs_qid) {
    DataEnqueue(qid, origin_shape, TensorDataType::DT_INT32, head_msg, input_data);
    DataEnqueue(qid, origin_shape, TensorDataType::DT_INT32, head_msg, input_data);
    DataEnqueue(qid, origin_shape, TensorDataType::DT_INT32, head_msg, input_data);
    DataEnqueue(qid, origin_shape, TensorDataType::DT_INT32, head_msg, input_data);
    DataEnqueue(qid, origin_shape, TensorDataType::DT_INT32, head_msg, input_data);
    DataEnqueue(qid, origin_shape, TensorDataType::DT_INT32, head_msg, input_data);
    DataEnqueue(qid, origin_shape, TensorDataType::DT_INT32, head_msg, input_data);
  }
  while (wait_time_in_ms < kMaxWaitTimeInMs) {
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
    usleep(kWaitMsPerTime * 1000);
    wait_time_in_ms += kWaitMsPerTime;
  }
  int32_t expect_out[] = {4, 7, 5, 5, 4, 7, 5, 5, 4, 7, 5, 5, 4, 7, 5, 5, 4, 7, 5, 5};
  new_shape = {size, 2, 2};
  for (size_t i = 0; i < outs_mbuf_ptr.size(); ++i) {
    EXPECT_NE(outs_mbuf_ptr[i], nullptr);
    if (outs_mbuf_ptr[i] != nullptr) {
      Mbuf *out_mbuf = (Mbuf *)outs_mbuf_ptr[i];
      CheckMbufData(out_mbuf, new_shape, TensorDataType::DT_INT32, expect_out, CalcElementCnt(new_shape));
      halMbufFree(out_mbuf);
    }
  }
  get_out_num = 0;
  wait_time_in_ms = 0;
  for (int32_t index = 0; index < in_out_num; ++index) {
    get_out[index] = false;
    outs_mbuf_ptr[index] = nullptr;
  }
  while (wait_time_in_ms < kMaxWaitTimeInMs) {
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
    usleep(kWaitMsPerTime * 1000);
    wait_time_in_ms += kWaitMsPerTime;
  }
  int expect_out2[] = {4, 7, 5, 5, 4, 7, 5, 5, 4, 7, 5, 5, 4, 7, 5, 5, 4, 7, 5, 5};
  new_shape = {size, 2, 2};
  for (size_t i = 0; i < outs_mbuf_ptr.size(); ++i) {
    EXPECT_NE(outs_mbuf_ptr[i], nullptr);
    if (outs_mbuf_ptr[i] != nullptr) {
      Mbuf *out_mbuf = (Mbuf *)outs_mbuf_ptr[i];
      CheckMbufData(out_mbuf, new_shape, TensorDataType::DT_INT32, expect_out2, CalcElementCnt(new_shape));
      halMbufFree(out_mbuf);
    }
  }

  wait_time_in_ms = 0;
  get_out_num = 0;
  for (int32_t index = 0; index < in_out_num; ++index) {
    get_out[index] = false;
    outs_mbuf_ptr[index] = nullptr;
  }
  while (wait_time_in_ms < kMaxWaitTimeInMs) {
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
    usleep(kWaitMsPerTime * 1000);
    wait_time_in_ms += kWaitMsPerTime;
  }
  int32_t expect_out3[] = {4, 7, 5, 5, 4, 7, 5, 5, 4, 7, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0};
  new_shape = {5, 2, 2};
  for (size_t i = 0; i < outs_mbuf_ptr.size(); ++i) {
    EXPECT_NE(outs_mbuf_ptr[i], nullptr);
    if (outs_mbuf_ptr[i] != nullptr) {
      Mbuf *out_mbuf = (Mbuf *)outs_mbuf_ptr[i];
      CheckMbufData(out_mbuf, new_shape, TensorDataType::DT_INT32, expect_out3, CalcElementCnt(new_shape));
      halMbufFree(out_mbuf);
    }
  }
  // forth
  wait_time_in_ms = 0;
  get_out_num = 0;
  for (int32_t index = 0; index < in_out_num; ++index) {
    get_out[index] = false;
    outs_mbuf_ptr[index] = nullptr;
  }
  while (wait_time_in_ms < kMaxWaitTimeInMs) {
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
    usleep(kWaitMsPerTime * 1000);
    wait_time_in_ms += kWaitMsPerTime;
  }
  int32_t expect_out4[] = {4, 7, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  new_shape = {5, 2, 2};
  for (size_t i = 0; i < outs_mbuf_ptr.size(); ++i) {
    EXPECT_NE(outs_mbuf_ptr[i], nullptr);
    if (outs_mbuf_ptr[i] != nullptr) {
      Mbuf *out_mbuf = (Mbuf *)outs_mbuf_ptr[i];
      CheckMbufData(out_mbuf, new_shape, TensorDataType::DT_INT32, expect_out4, CalcElementCnt(new_shape));
      halMbufFree(out_mbuf);
    }
  }
  executor.Stop(true);
  executor.WaitForStop();
  executor.Destroy();
}
}  // namespace FlowFunc
