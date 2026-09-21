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
#include "flow_func/meta_multi_func.h"
#include "flow_func/flow_func_log.h"
#include "config/global_config.h"
#include "flow_func/flow_func_config_manager.h"

namespace FlowFunc {
namespace {
constexpr uint64_t kMaxWaitInMs = 60 * 1000UL;
constexpr uint64_t kWaitInMsPerTime = 10;

class BalanceFlowFunc : public MetaMultiFunc {
 public:
  virtual int32_t Init(const std::shared_ptr<MetaParams> &params) {
    int64_t row_num = 0;
    if (params->GetAttr("_TEST_ATTR_ROW_NUM", row_num) == FLOW_FUNC_SUCCESS) {
      FLOW_FUNC_LOG_INFO("_TEST_ATTR_ROW_NUM value=%ld", row_num);
      row_num_ = static_cast<int32_t>(row_num);
    }
    int64_t col_num = 0;
    if (params->GetAttr("_TEST_ATTR_COL_NUM", col_num) == FLOW_FUNC_SUCCESS) {
      FLOW_FUNC_LOG_INFO("_TEST_ATTR_COL_NUM value=%ld", col_num);
      col_num_ = static_cast<int32_t>(col_num);
    }
    int64_t policy = 0;
    if (params->GetAttr("_TEST_ATTR_POLICY", policy) == FLOW_FUNC_SUCCESS) {
      FLOW_FUNC_LOG_INFO("_TEST_ATTR_POLICY value=%ld", policy);
      policy_ = static_cast<AffinityPolicy>(policy);
    }
    return FLOW_FUNC_SUCCESS;
  }

  int32_t Balance(const std::shared_ptr<MetaRunContext> &run_context,
                  const std::vector<std::shared_ptr<FlowMsg>> &input_flow_msgs) {
    int32_t return_ret = FLOW_FUNC_SUCCESS;
    for (size_t i = 0; i < input_flow_msgs.size(); ++i) {
      const auto tensor = input_flow_msgs[i]->GetTensor();
      if (tensor == nullptr) {
        FLOW_FUNC_LOG_ERROR("GetTensor nullptr.");
        return FLOW_FUNC_FAILED;
      }
      if (tensor->GetDataSize() != sizeof(int32_t) * 2) {
        FLOW_FUNC_LOG_ERROR("data size is invalid, data size=%zu.", tensor->GetDataSize());
        return FLOW_FUNC_FAILED;
      }
      int32_t *data_pos = reinterpret_cast<int32_t *>(tensor->GetData());
      OutOptions options;
      auto balance_config = options.MutableBalanceConfig();
      balance_config->SetAffinityPolicy(policy_);
      balance_config->SetDataPos({{data_pos[0], data_pos[1]}});
      BalanceWeight weight;
      weight.rowNum = row_num_;
      weight.colNum = col_num_;
      balance_config->SetBalanceWeight(weight);
      int32_t ret = run_context->SetOutput(i, input_flow_msgs[i], options);
      if (ret != FLOW_FUNC_SUCCESS) {
        return_ret = ret;
        FLOW_FUNC_LOG_ERROR("set output failed, ret=%d.", ret);
      }
    }
    return return_ret;
  }

 private:
  int32_t row_num_ = 0;
  int32_t col_num_ = 0;
  AffinityPolicy policy_ = AffinityPolicy::NO_AFFINITY;
};

class AssignFlowFunc : public MetaMultiFunc {
 public:
  int32_t Init() {
    return FLOW_FUNC_SUCCESS;
  }

  int32_t Proc(const std::shared_ptr<MetaRunContext> &run_context,
               const std::vector<std::shared_ptr<FlowMsg>> &input_msgs) {
    OutOptions options;
    for (size_t i = 0; i < input_msgs.size(); ++i) {
      run_context->SetOutput(i, input_msgs[i], options);
    }
    return FLOW_FUNC_SUCCESS;
  }
};
FLOW_FUNC_REGISTRAR(BalanceFlowFunc).RegProcFunc("BalanceFlowFunc", &BalanceFlowFunc::Balance);
FLOW_FUNC_REGISTRAR(AssignFlowFunc).RegProcFunc("AssignFlowFunc", &AssignFlowFunc::Proc);
}  // namespace

class BalanceAlignSTest : public testing::Test {
protected:
  static void SetUpTestSuite() {
    FlowFuncConfigManager::SetConfig(
        std::shared_ptr<FlowFuncConfig>(&GlobalConfig::Instance(), [](FlowFuncConfig *) {}));
  }
  virtual void SetUp() {
    ClearStubEschedEvents();
    for (int32_t i = 0; i < 9; ++i) {
      all_queue_ids.emplace_back(CreateQueue());
    }
    CreateModelDir();
    req_queue_id = CreateQueue();
    rsp_queue_id = CreateQueue();
    GlobalConfig::Instance().SetMessageQueueIds(req_queue_id, rsp_queue_id);
  }

  virtual void TearDown() {
    DeleteModelDir();
    for (auto qid : all_queue_ids) {
      DestroyQueue(qid);
    }
    all_queue_ids.clear();
    DestroyQueue(req_queue_id);
    DestroyQueue(rsp_queue_id);
    GlobalConfig::Instance().SetMessageQueueIds(UINT32_MAX, UINT32_MAX);
  }

  void CreateBalanceModel(ff::deployer::ExecutorRequest_BatchLoadModelMessage &batch_load_model_req,
                          const std::vector<uint32_t> &input_queues, const std::vector<uint32_t> &output_queues,
                          int32_t row_num, int32_t col_num, AffinityPolicy policy, const std::string &pp_name) {
    ff::udf::UdfModelDef model_def;
    CreateUdfModel(model_def, "BalanceFlowFunc", pp_name, {}, true);
    auto udf_def = model_def.mutable_udf_def(0);

    auto proto_attrs = udf_def->mutable_attrs();
    ff::udf::AttrValue row_value;
    row_value.set_i(row_num);
    proto_attrs->insert({"_TEST_ATTR_ROW_NUM", row_value});
    ff::udf::AttrValue col_value;
    col_value.set_i(col_num);
    proto_attrs->insert({"_TEST_ATTR_COL_NUM", col_value});
    ff::udf::AttrValue policy_value;
    policy_value.set_i(static_cast<int32_t>(policy));
    proto_attrs->insert({"_TEST_ATTR_POLICY", policy_value});

    ff::udf::AttrValue balance_value;
    balance_value.set_b(true);
    if (policy == AffinityPolicy::NO_AFFINITY) {
      proto_attrs->insert({"_balance_scatter", balance_value});
    } else {
      proto_attrs->insert({"_balance_gather", balance_value});
    }

    auto proto_path = WriteProto(model_def, "Balance.pb");
    AddModelToBatchModel(batch_load_model_req, proto_path, input_queues, output_queues);
    auto *input_align_attrs = batch_load_model_req.mutable_models(0)->mutable_input_align_attrs();
    input_align_attrs->set_align_max_cache_num(100);
    input_align_attrs->set_align_timeout(10000);
    input_align_attrs->set_drop_when_not_align(false);
  }

  void CreateAssignModel(ff::deployer::ExecutorRequest_BatchLoadModelMessage &batch_load_model_req,
                         const std::vector<uint32_t> &input_queues, const std::vector<uint32_t> &output_queues,
                         const std::string &pp_name) {
    ff::udf::UdfModelDef model_def;
    CreateUdfModel(model_def, "AssignFlowFunc", pp_name, {}, true);
    auto proto_path = WriteProto(model_def, "assign.pb");
    AddModelToBatchModel(batch_load_model_req, proto_path, input_queues, output_queues);
    auto *input_align_attrs =
        batch_load_model_req.mutable_models(batch_load_model_req.models_size() - 1)->mutable_input_align_attrs();
    input_align_attrs->set_align_max_cache_num(100);
    input_align_attrs->set_align_timeout(10000);
    input_align_attrs->set_drop_when_not_align(false);
  }

  std::string CreateModel(int32_t row_num, int32_t col_num, AffinityPolicy policy, const std::string &pp_name) {
    ff::deployer::ExecutorRequest_BatchLoadModelMessage batch_load_model_req;
    CreateBalanceModel(batch_load_model_req, {all_queue_ids[0], all_queue_ids[1], all_queue_ids[2]},
                       {all_queue_ids[3], all_queue_ids[4], all_queue_ids[5]}, row_num, col_num, policy,
                       pp_name + "_balance");

    CreateAssignModel(batch_load_model_req, {all_queue_ids[3], all_queue_ids[4], all_queue_ids[5]},
                      {all_queue_ids[6], all_queue_ids[7], all_queue_ids[8]}, pp_name + "_assign");
    return WriteProto(batch_load_model_req, "balance_align.pb");
  }

 protected:
  std::vector<uint32_t> all_queue_ids;
  uint32_t req_queue_id;
  uint32_t rsp_queue_id;
};

TEST_F(BalanceAlignSTest, test_scatter) {
  int32_t row_num = 4;
  int32_t col_num = 5;
  AffinityPolicy policy = AffinityPolicy::NO_AFFINITY;
  std::string batch_model_path = CreateModel(row_num, col_num, policy, "test_scatter" + std::to_string(__LINE__));

  auto batch_models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(batch_models.size(), 2);
  FlowFuncExecutor executor;
  auto ret = executor.Init(batch_models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = executor.Start();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);

  std::vector<int64_t> shape = {2};
  std::vector<std::pair<int32_t, int32_t>> queue_data1 = {{0, 0}, {0, 1}, {1, 0}, {1, 2}};
  std::vector<std::pair<int32_t, int32_t>> queue_data2 = {{0, 0}, {1, 0}, {0, 1}, {1, 2}};
  std::vector<std::pair<int32_t, int32_t>> queue_data3 = {{0, 0}, {1, 2}, {1, 0}, {0, 1}};
  size_t data_group_num = queue_data1.size();
  MbufHeadMsg head_msg = {};
  for (size_t i = 0; i < data_group_num; ++i) {
    int32_t buf1[] = {queue_data1[i].first, queue_data1[i].second};
    DataEnqueue(all_queue_ids[0], shape, TensorDataType::DT_INT32, head_msg, buf1);
    int32_t buf2[] = {queue_data2[i].first, queue_data2[i].second};
    DataEnqueue(all_queue_ids[1], shape, TensorDataType::DT_INT32, head_msg, buf2);
    int32_t buf3[] = {queue_data3[i].first, queue_data3[i].second};
    DataEnqueue(all_queue_ids[2], shape, TensorDataType::DT_INT32, head_msg, buf3);
  }

  std::vector<uint32_t> output_queues = {all_queue_ids[6], all_queue_ids[7], all_queue_ids[8]};

  for (int j = 0; j < data_group_num; ++j) {
    std::set<std::pair<int32_t, int32_t>> group_values;
    std::set<std::pair<uint64_t, uint32_t>> trans_id_and_data_labels;
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
      const int32_t *data = nullptr;
      uint32_t data_num = 0;
      GetMbufDataAndLen(out_mbuf, data, data_num);
      EXPECT_EQ(data_num, 2);
      group_values.emplace(data[0], data[1]);
      const MbufHeadMsg *head_msg = GetMbufHeadMsg(out_mbuf);
      trans_id_and_data_labels.emplace(head_msg->trans_id, head_msg->data_label);
      halMbufFree(out_mbuf);
    }
    // data check align
    EXPECT_EQ(group_values.size(), 1);
    // data label align
    EXPECT_EQ(trans_id_and_data_labels.size(), 1);
  }

  executor.Stop(true);
  executor.WaitForStop();
}

TEST_F(BalanceAlignSTest, test_gather_row) {
  int32_t row_num = 4;
  int32_t col_num = 5;
  AffinityPolicy policy = AffinityPolicy::ROW_AFFINITY;
  std::string batch_model_path = CreateModel(row_num, col_num, policy, "test_gather_row" + std::to_string(__LINE__));

  auto batch_models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(batch_models.size(), 2);
  FlowFuncExecutor executor;
  auto ret = executor.Init(batch_models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = executor.Start();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);

  std::vector<int64_t> shape = {2};
  std::vector<std::pair<int32_t, int32_t>> queue_data1 = {{0, 0}, {0, 1}, {1, 0}, {1, 2}};
  std::vector<std::pair<int32_t, int32_t>> queue_data2 = {{0, 0}, {1, 0}, {0, 1}, {1, 2}};
  std::vector<std::pair<int32_t, int32_t>> queue_data3 = {{0, 0}, {1, 2}, {1, 0}, {0, 1}};
  size_t data_group_num = queue_data1.size();
  MbufHeadMsg head_msg = {};
  for (size_t i = 0; i < data_group_num; ++i) {
    int32_t buf1[] = {queue_data1[i].first, queue_data1[i].second};
    DataEnqueue(all_queue_ids[0], shape, TensorDataType::DT_INT32, head_msg, buf1);
    int32_t buf2[] = {queue_data2[i].first, queue_data2[i].second};
    DataEnqueue(all_queue_ids[1], shape, TensorDataType::DT_INT32, head_msg, buf2);
    int32_t buf3[] = {queue_data3[i].first, queue_data3[i].second};
    DataEnqueue(all_queue_ids[2], shape, TensorDataType::DT_INT32, head_msg, buf3);
  }

  std::vector<uint32_t> output_queues = {all_queue_ids[6], all_queue_ids[7], all_queue_ids[8]};

  for (int j = 0; j < data_group_num; ++j) {
    std::set<std::pair<int32_t, int32_t>> group_values;
    std::set<std::pair<uint64_t, uint32_t>> trans_id_and_data_labels;
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
      const int32_t *data = nullptr;
      uint32_t data_num = 0;
      GetMbufDataAndLen(out_mbuf, data, data_num);
      EXPECT_EQ(data_num, 2);
      group_values.emplace(data[0], data[1]);
      const MbufHeadMsg *head_msg = GetMbufHeadMsg(out_mbuf);
      trans_id_and_data_labels.emplace(head_msg->trans_id, head_msg->data_label);
      halMbufFree(out_mbuf);
    }
    // data check align
    EXPECT_EQ(group_values.size(), 1);
    // data label align
    EXPECT_EQ(trans_id_and_data_labels.size(), 1);
  }

  executor.Stop(true);
  executor.WaitForStop();
}

TEST_F(BalanceAlignSTest, test_gather_col) {
  int32_t row_num = 4;
  int32_t col_num = 5;
  AffinityPolicy policy = AffinityPolicy::COL_AFFINITY;
  std::string batch_model_path = CreateModel(row_num, col_num, policy, "test_gather_col" + std::to_string(__LINE__));

  auto batch_models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(batch_models.size(), 2);
  FlowFuncExecutor executor;
  auto ret = executor.Init(batch_models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = executor.Start();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);

  std::vector<int64_t> shape = {2};
  std::vector<std::pair<int32_t, int32_t>> queue_data1 = {{0, 0}, {0, 1}, {1, 0}, {1, 2}};
  std::vector<std::pair<int32_t, int32_t>> queue_data2 = {{0, 0}, {1, 0}, {0, 1}, {1, 2}};
  std::vector<std::pair<int32_t, int32_t>> queue_data3 = {{0, 0}, {1, 2}, {1, 0}, {0, 1}};
  size_t data_group_num = queue_data1.size();
  MbufHeadMsg head_msg = {};
  for (size_t i = 0; i < data_group_num; ++i) {
    int32_t buf1[] = {queue_data1[i].first, queue_data1[i].second};
    DataEnqueue(all_queue_ids[0], shape, TensorDataType::DT_INT32, head_msg, buf1);
    int32_t buf2[] = {queue_data2[i].first, queue_data2[i].second};
    DataEnqueue(all_queue_ids[1], shape, TensorDataType::DT_INT32, head_msg, buf2);
    int32_t buf3[] = {queue_data3[i].first, queue_data3[i].second};
    DataEnqueue(all_queue_ids[2], shape, TensorDataType::DT_INT32, head_msg, buf3);
  }

  std::vector<uint32_t> output_queues = {all_queue_ids[6], all_queue_ids[7], all_queue_ids[8]};

  for (int j = 0; j < data_group_num; ++j) {
    std::set<std::pair<int32_t, int32_t>> group_values;
    std::set<std::pair<uint64_t, uint32_t>> trans_id_and_data_labels;
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
      const int32_t *data = nullptr;
      uint32_t data_num = 0;
      GetMbufDataAndLen(out_mbuf, data, data_num);
      EXPECT_EQ(data_num, 2);
      group_values.emplace(data[0], data[1]);
      const MbufHeadMsg *head_msg = GetMbufHeadMsg(out_mbuf);
      trans_id_and_data_labels.emplace(head_msg->trans_id, head_msg->data_label);
      halMbufFree(out_mbuf);
    }
    // data check align
    EXPECT_EQ(group_values.size(), 1);
    // data label align
    EXPECT_EQ(trans_id_and_data_labels.size(), 1);
  }

  executor.Stop(true);
  executor.WaitForStop();
}
}  // namespace FlowFunc
