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
#include "common/scope_guard.h"
#define private public
#include "execute/flow_func_executor.h"
#include "model/flow_func_model.h"
#include "mockcpp/mockcpp.hpp"
#include "config/global_config.h"
#undef private
#include "flow_func/flow_func_config_manager.h"

namespace FlowFunc {
namespace {
constexpr uint64_t kWaitInMsPerTime = 10;
bool back_is_on_device_ = false;
}
class MultiFlowFuncSTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    FlowFuncConfigManager::SetConfig(
        std::shared_ptr<FlowFuncConfig>(&GlobalConfig::Instance(), [](FlowFuncConfig *) {}));
    back_is_on_device_ = GlobalConfig::Instance().IsOnDevice();
    GlobalConfig::on_device_ = false;
  }
  static void TearDownTestSuite() {
    GlobalConfig::on_device_ = back_is_on_device_;
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
    GlobalMockObject::verify();
    GlobalConfig::Instance().SetMessageQueueIds(UINT32_MAX, UINT32_MAX);
  }

  ff::deployer::ExecutorRequest_ModelQueuesAttrs CreateModelQueueIds(bool link_model_out_to_in) {
    ff::deployer::ExecutorRequest_ModelQueuesAttrs model_queue_ids;
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

    auto input_queue_attr1 = model_queue_ids.add_input_queues_attrs();
    input_queue_attr1->set_device_id(0);
    input_queue_attr1->set_device_type(0);
    input_queue_attr1->set_queue_id(input_qid1);

    auto input_queue_attr2 = model_queue_ids.add_input_queues_attrs();
    input_queue_attr2->set_device_id(0);
    input_queue_attr2->set_device_type(0);
    input_queue_attr2->set_queue_id(input_qid2);

    auto output_queue_attr1 = model_queue_ids.add_output_queues_attrs();
    output_queue_attr1->set_device_id(0);
    output_queue_attr1->set_device_type(0);
    output_queue_attr1->set_queue_id(output_qid1);

    auto output_queue_attr2 = model_queue_ids.add_output_queues_attrs();
    output_queue_attr2->set_device_id(0);
    output_queue_attr2->set_device_type(0);
    output_queue_attr2->set_queue_id(output_qid2);
    return model_queue_ids;
  }

  std::string CreateBatchModel(std::vector<uint32_t> &multi_inputs_qid, std::vector<uint32_t> &multi_outputs_qid,
                               std::vector<uint32_t> &single_inputs_qid, std::vector<uint32_t> &single_outputs_qid,
                               const std::map<AscendString, ff::udf::AttrValue> &attrs = {},
                               const bool report_status = false, const bool catch_exceptions = false,
                               std::vector<uint32_t> status_output_queues = {},
                               std::vector<uint32_t> logic_input_queues = {}) {
    ff::udf::UdfModelDef multi_func_model_def;
    for (size_t i = 0; i < multi_inputs_qid.size(); ++i) {
      multi_inputs_qid[i] = CreateQueue();
    }
    record_queue_ids.insert(record_queue_ids.end(), multi_inputs_qid.begin(), multi_inputs_qid.end());
    for (size_t i = 0; i < multi_outputs_qid.size(); ++i) {
      multi_outputs_qid[i] = CreateQueue(UDF_ST_QUEUE_MAX_DEPTH * 2);
    }
    record_queue_ids.insert(record_queue_ids.end(), multi_outputs_qid.begin(), multi_outputs_qid.end());
    for (size_t i = 0; i < status_output_queues.size(); ++i) {
      status_output_queues[i] = CreateQueue();
    }
    record_queue_ids.insert(record_queue_ids.end(), status_output_queues.begin(), status_output_queues.end());
    std::map<std::string, std::vector<uint32_t>> input_maps;
    // 4 inputs:
    input_maps["Proc1"] = {0, 1};
    input_maps["Proc2"] = {2, 3};

    std::map<std::string, std::vector<uint32_t>> output_maps;
    // 2 outputs:
    output_maps["Proc1"] = {0, 1};
    output_maps["Proc2"] = {0, 1};

    ff::udf::AttrValue data_type;
    data_type.set_type((uint32_t)TensorDataType::DT_UINT32);
    attrs["out_type"] = data_type;

    ff::udf::AttrValue re_init_attr;
    re_init_attr.set_b(true);
    attrs["need_re_init_attr"] = re_init_attr;

    CreateUdfModel(multi_func_model_def, "Proc1", "Inst_Proc1", attrs, false, input_maps, output_maps);

    auto multi_func_proto_path = WriteProto(multi_func_model_def, "MultiFlowFuncSt.pb");
    ff::deployer::ExecutorRequest_BatchLoadModelMessage batch_load_model_req;
    AddModelToBatchModel(batch_load_model_req, multi_func_proto_path, multi_inputs_qid, multi_outputs_qid, {},
                         Common::kDeviceTypeCpu, report_status, catch_exceptions, status_output_queues,
                         logic_input_queues);

    // multifunc register single func
    if (!single_inputs_qid.empty()) {
      ff::udf::UdfModelDef single_model_def;
      for (size_t i = 0; i < single_inputs_qid.size(); ++i) {
        single_inputs_qid[i] = CreateQueue();
      }

      record_queue_ids.insert(record_queue_ids.end(), single_inputs_qid.begin(), single_inputs_qid.end());
      for (size_t i = 0; i < single_outputs_qid.size(); ++i) {
        single_outputs_qid[i] = CreateQueue();
      }
      record_queue_ids.insert(record_queue_ids.end(), single_outputs_qid.begin(), single_outputs_qid.end());
      for (size_t i = 0; i < status_output_queues.size(); ++i) {
        status_output_queues[i] = CreateQueue();
      }
      record_queue_ids.insert(record_queue_ids.end(), status_output_queues.begin(), status_output_queues.end());
      CreateUdfModel(single_model_def, "SingleProc", "Inst_SingleProc", attrs, false);
      auto single_proto_path = WriteProto(single_model_def, "SingleFlowFuncSt.pb");
      AddModelToBatchModel(batch_load_model_req, single_proto_path, single_inputs_qid, single_outputs_qid, {},
                           Common::kDeviceTypeCpu, report_status, catch_exceptions, status_output_queues,
                           logic_input_queues);
    }
    return WriteProto(batch_load_model_req, "batchModels.pb");
  }

  std::string CreateSingleFuncBatchModel(std::vector<uint32_t> &inputs_qid, std::vector<uint32_t> &outputs_qid,
                                         const std::map<AscendString, ff::udf::AttrValue> &attrs,
                                         const bool report_status, std::vector<uint32_t> &status_output_queues,
                                         std::vector<uint32_t> &logic_input_queues) {
    ff::udf::UdfModelDef model_def;
    for (size_t i = 0; i < inputs_qid.size(); ++i) {
      inputs_qid[i] = CreateQueue();
    }
    record_queue_ids.insert(record_queue_ids.end(), inputs_qid.begin(), inputs_qid.end());
    for (size_t i = 0; i < outputs_qid.size(); ++i) {
      outputs_qid[i] = CreateQueue();
    }
    record_queue_ids.insert(record_queue_ids.end(), outputs_qid.begin(), outputs_qid.end());

    for (size_t i = 0; i < status_output_queues.size(); ++i) {
      status_output_queues[i] = CreateQueue();
    }
    record_queue_ids.insert(record_queue_ids.end(), status_output_queues.begin(), status_output_queues.end());

    ff::udf::AttrValue data_type;
    data_type.set_type((uint32_t)TensorDataType::DT_UINT32);
    attrs["out_type"] = data_type;
    CreateUdfModel(model_def, "SingleProc", __FILE__, attrs, false);
    auto proto_path = WriteProto(model_def, "FlowFuncSt.pb");
    ff::deployer::ExecutorRequest_BatchLoadModelMessage batch_load_model_req;
    AddModelToBatchModel(batch_load_model_req, proto_path, inputs_qid, outputs_qid, {}, Common::kDeviceTypeCpu,
                         report_status, false, status_output_queues, logic_input_queues);
    return WriteProto(batch_load_model_req, "batchModels.pb");
  }

  std::string CreateStreamInputBatchModel(std::vector<uint32_t> &inputs_qid, std::vector<uint32_t> &outputs_qid,
                                          const std::map<AscendString, ff::udf::AttrValue> &attrs) {
    ff::udf::UdfModelDef model_def;
    for (size_t i = 0; i < inputs_qid.size(); ++i) {
      inputs_qid[i] = CreateQueue();
    }
    record_queue_ids.insert(record_queue_ids.end(), inputs_qid.begin(), inputs_qid.end());
    for (size_t i = 0; i < outputs_qid.size(); ++i) {
      outputs_qid[i] = CreateQueue();
    }
    record_queue_ids.insert(record_queue_ids.end(), outputs_qid.begin(), outputs_qid.end());

    ff::udf::AttrValue data_type;
    data_type.set_type((uint32_t)TensorDataType::DT_UINT32);
    attrs["out_type"] = data_type;
    CreateUdfModel(model_def, "StreamInputProc", __FILE__, attrs, false, {}, {}, {}, true);
    auto proto_path = WriteProto(model_def, "StreamInputProc.pb");
    ff::deployer::ExecutorRequest_BatchLoadModelMessage batch_load_model_req;
    AddModelToBatchModel(batch_load_model_req, proto_path, inputs_qid, outputs_qid, {}, Common::kDeviceTypeNpu);
    return WriteProto(batch_load_model_req, "batchModels.pb");
  }

  std::string CreateCallNnFlowModel(std::vector<uint32_t> &input_queues, std::vector<uint32_t> &output_queues,
                                    bool link_model_out_to_in) {
    ff::udf::UdfModelDef model_def;
    // 2 inputs:
    std::map<std::string, std::vector<uint32_t>> input_maps;
    input_maps["CallNnProc1"] = {0, 1};

    std::map<std::string, std::vector<uint32_t>> output_maps;
    // 2 outputs:
    output_maps["CallNnProc1"] = {0, 1};

    CreateUdfModel(model_def, "CallNnProc1", __FILE__, {}, false, input_maps, output_maps);

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

    AddModelToBatchModel(batch_load_model_req, proto_path, input_queues, output_queues);
    auto invoked_model_queues = batch_load_model_req.mutable_models(0)->mutable_invoked_model_queues_attrs();
    (*invoked_model_queues)["float_model_key"] = CreateModelQueueIds(link_model_out_to_in);
    (*invoked_model_queues)["other_model_key"] = CreateModelQueueIds(link_model_out_to_in);

    return WriteProto(batch_load_model_req, "exchange_queues_batch_model.pb");
  }

  std::string CreateAbnormalBatchModel(std::vector<uint32_t> &multi_inputs_qid,
                                       std::vector<uint32_t> &multi_outputs_qid, uint32_t inputs_num,
                                       uint32_t outputs_num,
                                       const std::map<AscendString, ff::udf::AttrValue> &attrs = {}) {
    ff::udf::UdfModelDef multi_func_model_def;
    for (size_t i = 0; i < multi_inputs_qid.size(); ++i) {
      multi_inputs_qid[i] = CreateQueue();
    }
    record_queue_ids.insert(record_queue_ids.end(), multi_inputs_qid.begin(), multi_inputs_qid.end());
    for (size_t i = 0; i < multi_outputs_qid.size(); ++i) {
      multi_outputs_qid[i] = CreateQueue();
    }
    record_queue_ids.insert(record_queue_ids.end(), multi_outputs_qid.begin(), multi_outputs_qid.end());
    std::map<std::string, std::vector<uint32_t>> input_maps;
    // 4 inputs:
    input_maps["Proc1"] = {0, 1};
    input_maps["Proc2"] = {2, inputs_num};

    std::map<std::string, std::vector<uint32_t>> output_maps;
    // 2 outputs:
    output_maps["Proc1"] = {outputs_num};
    output_maps["Proc2"] = {outputs_num};

    ff::udf::AttrValue data_type;
    data_type.set_type((uint32_t)TensorDataType::DT_UINT32);
    attrs["out_type"] = data_type;
    CreateUdfModel(multi_func_model_def, "Proc1", __FILE__, attrs, false, input_maps, output_maps);

    auto multi_func_proto_path = WriteProto(multi_func_model_def, "MultiFlowFuncSt.pb");
    ff::deployer::ExecutorRequest_BatchLoadModelMessage batch_load_model_req;
    AddModelToBatchModel(batch_load_model_req, multi_func_proto_path, multi_inputs_qid, multi_outputs_qid);
    return WriteProto(batch_load_model_req, "batchModels.pb");
  }

  std::string CreateRawDataBatchModel(std::vector<uint32_t> &input_queues, std::vector<uint32_t> &output_queues) {
    ff::udf::UdfModelDef model_def;
    // 2 inputs:
    std::map<std::string, std::vector<uint32_t>> input_maps;
    input_maps["RawDataProc"] = {0, 1};

    std::map<std::string, std::vector<uint32_t>> output_maps;
    // 2 outputs:
    output_maps["RawDataProc"] = {0, 1};

    CreateUdfModel(model_def, "RawDataProc", __FILE__, {}, false, input_maps, output_maps);

    auto proto_path = WriteProto(model_def, "RawDataFlowFuncSt.pb");
    ff::deployer::ExecutorRequest_BatchLoadModelMessage batch_load_model_req;

    for (size_t idx = 0; idx < 2; ++idx) {
      auto input_qid = CreateQueue();
      auto output_qid = CreateQueue();
      record_queue_ids.emplace_back(input_qid);
      record_queue_ids.emplace_back(output_qid);
      input_queues.emplace_back(input_qid);
      output_queues.emplace_back(output_qid);
    }

    AddModelToBatchModel(batch_load_model_req, proto_path, input_queues, output_queues);
    return WriteProto(batch_load_model_req, "batchModels.pb");
  }

  std::string CreateInvalidFuncNameBatchModel(std::vector<uint32_t> &multi_inputs_qid,
                                              std::vector<uint32_t> &multi_outputs_qid, uint32_t inputs_num,
                                              uint32_t outputs_num,
                                              const std::map<AscendString, ff::udf::AttrValue> &attrs = {}) {
    ff::udf::UdfModelDef multi_func_model_def;
    for (size_t i = 0; i < multi_inputs_qid.size(); ++i) {
      multi_inputs_qid[i] = CreateQueue();
    }
    record_queue_ids.insert(record_queue_ids.end(), multi_inputs_qid.begin(), multi_inputs_qid.end());
    for (size_t i = 0; i < multi_outputs_qid.size(); ++i) {
      multi_outputs_qid[i] = CreateQueue();
    }
    record_queue_ids.insert(record_queue_ids.end(), multi_outputs_qid.begin(), multi_outputs_qid.end());
    std::map<std::string, std::vector<uint32_t>> input_maps;
    // 4 inputs:
    input_maps["Invalid1"] = {0, 1};
    input_maps["Invalid2"] = {2};

    std::map<std::string, std::vector<uint32_t>> output_maps;
    // 2 outputs:
    output_maps["Invalid1"] = {0};
    output_maps["Invalid2"] = {0};

    ff::udf::AttrValue data_type;
    data_type.set_type((uint32_t)TensorDataType::DT_UINT32);
    attrs["out_type"] = data_type;
    CreateUdfModel(multi_func_model_def, "Invalid1", __FILE__, attrs, false, input_maps, output_maps);

    auto multi_func_proto_path = WriteProto(multi_func_model_def, "MultiFlowFuncSt.pb");
    ff::deployer::ExecutorRequest_BatchLoadModelMessage batch_load_model_req;
    AddModelToBatchModel(batch_load_model_req, multi_func_proto_path, multi_inputs_qid, multi_outputs_qid);
    return WriteProto(batch_load_model_req, "batchModels.pb");
  }

 protected:
  std::vector<uint32_t> record_queue_ids;
  uint32_t req_queue_id;
  uint32_t rsp_queue_id;
};

TEST_F(MultiFlowFuncSTest, register_single_func) {
  int32_t in_num = 2;
  int32_t out_num = 1;
  std::vector<uint32_t> input_qids(in_num, 0);
  std::vector<uint32_t> output_qids(out_num, 0);
  std::vector<uint32_t> status_output_queues(1, 0);
  std::vector<uint32_t> logic_input_queues(in_num, 0);
  std::map<AscendString, ff::udf::AttrValue> attrs;
  ff::udf::AttrValue re_init_attr;
  re_init_attr.set_b(true);
  attrs["need_re_init_attr"] = re_init_attr;
  std::string batch_model_path =
      CreateSingleFuncBatchModel(input_qids, output_qids, attrs, true, status_output_queues, logic_input_queues);
  auto batch_models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(batch_models.size(), 1);
  FlowFuncExecutor executor;
  auto ret = executor.Init(batch_models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = executor.Start();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);

  MbufHeadMsg head_msg{0};
  std::vector<int64_t> shape = {1, 2};
  uint32_t input_value[] = {11, 22};
  for (auto i = 0; i < in_num; i++) {
    DataEnqueue(input_qids[i], shape, TensorDataType::DT_UINT32, head_msg, input_value);
  }

  for (size_t i = 0; i < output_qids.size(); i++) {
    void *out_mbuf_ptr = nullptr;
    constexpr uint64_t kMaxWaitInMs = 120 * 1000UL;
    uint64_t wait_in_ms = 0;
    while (wait_in_ms < kMaxWaitInMs) {
      auto drv_ret = halQueueDeQueue(0, output_qids[i], &out_mbuf_ptr);
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
    std::vector<uint32_t> expect_output = {22, 44};
    CheckMbufData(out_mbuf, shape, TensorDataType::DT_UINT32, expect_output.data(), expect_output.size());
    halMbufFree(out_mbuf);
  }

  for (size_t i = 0; i < status_output_queues.size(); i++) {
    void *out_mbuf_ptr = nullptr;
    constexpr uint64_t kMaxWaitInMs = 120 * 1000UL;
    uint64_t wait_in_ms = 0;
    while (wait_in_ms < kMaxWaitInMs) {
      auto drv_ret = halQueueDeQueue(0, status_output_queues[i], &out_mbuf_ptr);
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
    halMbufFree(out_mbuf);
  }

  executor.Stop(true);
  executor.WaitForStop();
  executor.Destroy();
}

TEST_F(MultiFlowFuncSTest, register_stream_input_func) {
  GlobalConfig::Instance().SetAbnormalStatus(false);
  GlobalConfig::Instance().SetExitFlag(false);
  int32_t in_num = 2;
  int32_t out_num = 1;
  std::vector<uint32_t> input_qids(in_num, 0);
  std::vector<uint32_t> output_qids(out_num, 0);
  std::map<AscendString, ff::udf::AttrValue> attrs;
  ff::udf::AttrValue re_init_attr;
  re_init_attr.set_b(true);
  attrs["need_re_init_attr"] = re_init_attr;
  std::string batch_model_path = CreateStreamInputBatchModel(input_qids, output_qids, attrs);
  auto batch_models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(batch_models.size(), 1);
  FlowFuncExecutor executor;
  auto ret = executor.Init(batch_models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = executor.Start();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ScopeGuard executor_guard([&executor]() {
    executor.Stop(true);
    executor.WaitForStop();
    executor.Destroy();
  });
  auto &processor = executor.func_processors_[0];
  uint64_t wait_init_in_ms = 0;
  constexpr uint64_t kMaxWaitInitInMs = 5 * 1000UL;
  while (processor->status_ < FlowFuncProcessorStatus::kInitFlowFunc) {
    if (wait_init_in_ms > kMaxWaitInitInMs) {
      FlowFuncProcessorStatus current_status = processor->status_;
      ASSERT_GE(current_status, FlowFuncProcessorStatus::kInitFlowFunc);
    }
    // wait processor init
    std::this_thread::sleep_for(std::chrono::milliseconds(kWaitInMsPerTime));
    wait_init_in_ms += kWaitInMsPerTime;
  }
  EXPECT_EQ(processor->flow_msg_queues_.size(), 2);
  auto &input_flow_msg_queue = processor->flow_msg_queues_[0];
  auto input_mbuf_flow_msg_queue = std::dynamic_pointer_cast<MbufFlowMsgQueue>(input_flow_msg_queue);
  auto q_status = input_mbuf_flow_msg_queue->StatusOk();
  EXPECT_TRUE(q_status);
  auto q_depth = input_mbuf_flow_msg_queue->Depth();
  EXPECT_EQ(q_depth, UDF_ST_QUEUE_MAX_DEPTH);
  auto q_size = input_mbuf_flow_msg_queue->Size();
  EXPECT_EQ(q_size, 0);
  MbufHeadMsg head_msg{0};
  std::vector<int64_t> shape = {1, 2};
  uint32_t input_value[] = {11, 22};

  for (auto i = 0; i < in_num; i++) {
    DataEnqueue(input_qids[i], shape, TensorDataType::DT_UINT32, head_msg, input_value);
  }
  for (size_t i = 0; i < output_qids.size(); i++) {
    void *out_mbuf_ptr = nullptr;
    constexpr uint64_t kMaxWaitInMs = 120 * 1000UL;
    uint64_t wait_in_ms = 0;
    while (wait_in_ms < kMaxWaitInMs) {
      auto drv_ret = halQueueDeQueue(0, output_qids[i], &out_mbuf_ptr);
      if (drv_ret == DRV_ERROR_QUEUE_EMPTY) {
        std::this_thread::sleep_for(std::chrono::milliseconds(kWaitInMsPerTime));
        wait_in_ms += kWaitInMsPerTime;
      } else {
        break;
      }
    }
    ASSERT_NE(out_mbuf_ptr, nullptr);
    Mbuf *out_mbuf = (Mbuf *)out_mbuf_ptr;
    std::vector<uint32_t> expect_output = {22, 44};
    CheckMbufData(out_mbuf, shape, TensorDataType::DT_UINT32, expect_output.data(), expect_output.size());
    halMbufFree(out_mbuf);
  }
}

TEST_F(MultiFlowFuncSTest, register_multi_func_call_nn_test) {
  std::vector<uint32_t> input_queues;
  std::vector<uint32_t> output_queues;
  std::string batch_model_path = CreateCallNnFlowModel(input_queues, output_queues, true);

  auto batch_models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(batch_models.size(), 1);
  FlowFuncExecutor executor;
  auto ret = executor.Init(batch_models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = executor.Start();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);

  std::vector<int64_t> shape = {1, 2, 3, 4};
  std::vector<float> float_value = {0, 1.1, 2.2, 3.3, 4.4};
  for (int64_t j = 0; j < UDF_ST_QUEUE_MAX_DEPTH; ++j) {
    for (size_t i = 0; i < input_queues.size(); ++i) {
      DataEnqueue(input_queues[i], shape, TensorDataType::DT_FLOAT, float_value[i]);
    }
  }

  constexpr uint64_t kMaxWaitInMs = 2 * 1000UL;
  for (int64_t j = 0; j < UDF_ST_QUEUE_MAX_DEPTH; ++j) {
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
  }

  executor.Stop(true);
  executor.WaitForStop();
  executor.Destroy();
}

TEST_F(MultiFlowFuncSTest, register_multi_func_raw_data_test) {
  std::vector<uint32_t> input_queues;
  std::vector<uint32_t> output_queues;
  std::string batch_model_path = CreateRawDataBatchModel(input_queues, output_queues);

  auto batch_models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(batch_models.size(), 1);
  FlowFuncExecutor executor;
  auto ret = executor.Init(batch_models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = executor.Start();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);

  std::vector<int64_t> shape = {1, 2, 3, 4};
  std::vector<float> float_value = {0, 1.1, 2.2, 3.3, 4.4};
  for (int64_t j = 0; j < UDF_ST_QUEUE_MAX_DEPTH; ++j) {
    for (size_t i = 0; i < input_queues.size(); ++i) {
      DataEnqueue(input_queues[i], shape, TensorDataType::DT_FLOAT, float_value[i]);
    }
  }

  constexpr uint64_t kMaxWaitInMs = 2 * 1000UL;
  for (int64_t j = 0; j < UDF_ST_QUEUE_MAX_DEPTH; ++j) {
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
      CheckMbufData(out_mbuf, expect_output.data(), expect_output.size());
      halMbufFree(out_mbuf);
    }
  }

  executor.Stop(true);
  executor.WaitForStop();
  executor.Destroy();
}

TEST_F(MultiFlowFuncSTest, multi_func_with_invalid_inputMaps) {
  int32_t multi_func_in_num = 4;
  int32_t multi_func_out_num = 2;
  std::vector<uint32_t> multi_input_qids(multi_func_in_num, 0);
  std::vector<uint32_t> multi_output_qids(multi_func_out_num, 0);
  std::map<AscendString, ff::udf::AttrValue> attrs;
  std::string batch_model_path1 =
      CreateAbnormalBatchModel(multi_input_qids, multi_output_qids, multi_func_in_num, multi_func_out_num, attrs);
  auto batch_models = FlowFuncModel::ParseModels(batch_model_path1);
  EXPECT_EQ(batch_models.size(), 1);
  FlowFuncExecutor executor;
  auto ret = executor.Init(batch_models);
  EXPECT_NE(ret, FLOW_FUNC_SUCCESS);
  std::string batch_model_path2 =
      CreateAbnormalBatchModel(multi_input_qids, multi_output_qids, multi_func_in_num - 1, multi_func_out_num, attrs);
  batch_models = FlowFuncModel::ParseModels(batch_model_path2);
  EXPECT_EQ(batch_models.size(), 1);
  ret = executor.Init(batch_models);
  EXPECT_NE(ret, FLOW_FUNC_SUCCESS);
  executor.WaitForStop();
}

TEST_F(MultiFlowFuncSTest, multi_func_with_invalid_func_name) {
  int32_t multi_func_in_num = 4;
  int32_t multi_func_out_num = 2;
  std::vector<uint32_t> multi_input_qids(multi_func_in_num, 0);
  std::vector<uint32_t> multi_output_qids(multi_func_out_num, 0);
  std::map<AscendString, ff::udf::AttrValue> attrs;
  std::string batch_model_path1 = CreateInvalidFuncNameBatchModel(multi_input_qids, multi_output_qids,
                                                                  multi_func_in_num, multi_func_out_num, attrs);
  auto batch_models = FlowFuncModel::ParseModels(batch_model_path1);
  EXPECT_EQ(batch_models.size(), 1);
  FlowFuncExecutor executor;
  auto ret = executor.Init(batch_models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = executor.Start();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  MbufHeadMsg head_msg{0};
  std::vector<int64_t> shape = {1, 2};
  uint32_t input_value[] = {11, 22};
  for (auto i = 0; i < multi_func_in_num; i++) {
    DataEnqueue(multi_input_qids[i], shape, TensorDataType::DT_UINT32, head_msg, input_value);
  }

  for (size_t i = 0; i < multi_func_out_num; i++) {
    void *out_mbuf_ptr = nullptr;

    uint64_t wait_in_ms = 0;
    // 最大等1s
    while (wait_in_ms < 1000) {
      auto drv_ret = halQueueDeQueue(0, multi_output_qids[i], &out_mbuf_ptr);
      if (drv_ret == DRV_ERROR_QUEUE_EMPTY) {
        std::this_thread::sleep_for(std::chrono::milliseconds(kWaitInMsPerTime));
        wait_in_ms += kWaitInMsPerTime;
      } else {
        break;
      }
    }
    ASSERT_EQ(out_mbuf_ptr, nullptr);
  }

  executor.Stop(true);
  executor.WaitForStop();
  executor.Destroy();
}

TEST_F(MultiFlowFuncSTest, multi_func_basic_test) {
  int32_t multi_func_in_num = 4;
  int32_t multi_func_out_num = 2;
  int32_t single_func_in_num = 2;
  int32_t single_func_out_num = 1;
  std::vector<uint32_t> multi_input_qids(multi_func_in_num, 0);
  std::vector<uint32_t> multi_output_qids(multi_func_out_num, 0);
  std::vector<uint32_t> single_input_qids(single_func_in_num, 0);
  std::vector<uint32_t> single_output_qids(single_func_out_num, 0);
  std::map<AscendString, ff::udf::AttrValue> attrs;
  std::string batch_model_path =
      CreateBatchModel(multi_input_qids, multi_output_qids, single_input_qids, single_output_qids, attrs);
  auto batch_models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(batch_models.size(), 2);
  FlowFuncExecutor executor;
  auto ret = executor.Init(batch_models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = executor.Start();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);

  MbufHeadMsg head_msg{0};
  std::vector<int64_t> shape = {1, 2};
  uint32_t input_value[] = {11, 22};
  for (auto i = 0; i < multi_func_in_num; i++) {
    DataEnqueue(multi_input_qids[i], shape, TensorDataType::DT_UINT32, head_msg, input_value);
  }

  for (auto i = 0; i < single_func_in_num; i++) {
    DataEnqueue(single_input_qids[i], shape, TensorDataType::DT_UINT32, head_msg, input_value);
  }

  for (size_t i = 0; i < multi_func_out_num; i++) {
    void *out_mbuf_ptr = nullptr;
    constexpr uint64_t kMaxWaitInMs = 60 * 1000UL;
    uint64_t wait_in_ms = 0;
    while (wait_in_ms < kMaxWaitInMs) {
      auto drv_ret = halQueueDeQueue(0, multi_output_qids[i], &out_mbuf_ptr);
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
      std::vector<uint32_t> expect_output = {22, 44};
      CheckMbufData(out_mbuf, shape, TensorDataType::DT_UINT32, expect_output.data(), expect_output.size());
    } else {
      std::vector<uint32_t> expect_output = {33, 66};
      CheckMbufData(out_mbuf, shape, TensorDataType::DT_UINT32, expect_output.data(), expect_output.size());
    }

    halMbufFree(out_mbuf);
  }

  for (size_t i = 0; i < single_output_qids.size(); i++) {
    void *out_mbuf_ptr = nullptr;
    constexpr uint64_t kMaxWaitInMs = 60 * 1000UL;
    uint64_t wait_in_ms = 0;
    while (wait_in_ms < kMaxWaitInMs) {
      auto drv_ret = halQueueDeQueue(0, single_output_qids[i], &out_mbuf_ptr);
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
    std::vector<uint32_t> expect_output = {22, 44};
    CheckMbufData(out_mbuf, shape, TensorDataType::DT_UINT32, expect_output.data(), expect_output.size());
    halMbufFree(out_mbuf);
  }

  executor.Stop(true);
  executor.WaitForStop();
  executor.Destroy();
}

TEST_F(MultiFlowFuncSTest, multi_func_raise_exception) {
  int32_t multi_func_in_num = 4;
  int32_t multi_func_out_num = 2;
  int32_t single_func_in_num = 2;
  int32_t single_func_out_num = 1;
  std::vector<uint32_t> multi_input_qids(multi_func_in_num, 0);
  std::vector<uint32_t> multi_output_qids(multi_func_out_num, 0);
  std::vector<uint32_t> single_input_qids(single_func_in_num, 0);
  std::vector<uint32_t> single_output_qids(single_func_out_num, 0);
  std::vector<uint32_t> status_output_queues(1, 0);
  std::map<AscendString, ff::udf::AttrValue> attrs;
  std::string batch_model_path = CreateBatchModel(multi_input_qids, multi_output_qids, single_input_qids,
                                                  single_output_qids, attrs, false, true, status_output_queues, {});
  auto batch_models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(batch_models.size(), 2);
  for (auto &model : batch_models) {
    model->input_align_attrs_ = {.align_max_cache_num = 4, .align_timeout = -1, .drop_when_not_align = false};
  }
  FlowFuncExecutor executor;
  auto ret = executor.Init(batch_models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = executor.Start();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);

  MbufHeadMsg head_msg{0};
  head_msg.trans_id = 1;
  std::vector<int64_t> shape = {1, 2};
  uint32_t input_value_to_raise[] = {1000, 1000};

  DataEnqueue(multi_input_qids[0], shape, TensorDataType::DT_UINT32, head_msg, input_value_to_raise);
  DataEnqueue(multi_input_qids[1], shape, TensorDataType::DT_UINT32, head_msg, input_value_to_raise);

  constexpr uint64_t kMaxWaitInMs = 20 * 1000;
  uint64_t wait_in_ms = 0;
  bool executor_exit = false;

  // 等待异常退出
  while (wait_in_ms < kMaxWaitInMs) {
    if (executor.running_) {
      executor_exit = true;
      break;
    }
    // 每次检查间隔一段时间，避免CPU空转
    std::this_thread::sleep_for(std::chrono::milliseconds(kWaitInMsPerTime));
    wait_in_ms += kWaitInMsPerTime;
  }

  EXPECT_TRUE(executor_exit);
  executor.Stop(true);
  executor.WaitForStop();
  executor.Destroy();
}

TEST_F(MultiFlowFuncSTest, full_to_not_full) {
  int32_t multi_func_in_num = 4;
  int32_t multi_func_out_num = 2;
  int32_t single_func_in_num = 2;
  int32_t single_func_out_num = 1;
  std::vector<uint32_t> multi_input_qids(multi_func_in_num, 0);
  std::vector<uint32_t> multi_output_qids(multi_func_out_num, 0);
  std::vector<uint32_t> single_input_qids(single_func_in_num, 0);
  std::vector<uint32_t> single_output_qids(single_func_out_num, 0);
  std::map<AscendString, ff::udf::AttrValue> attrs;
  std::string batch_model_path =
      CreateBatchModel(multi_input_qids, multi_output_qids, single_input_qids, single_output_qids, attrs);
  auto batch_models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(batch_models.size(), 2);
  FlowFuncExecutor executor;
  auto ret = executor.Init(batch_models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = executor.Start();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);

  MbufHeadMsg head_msg{0};
  std::vector<int64_t> shape = {1, 2};
  uint32_t input_value[] = {11, 22};
  for (int64_t i = 0; i < UDF_ST_QUEUE_MAX_DEPTH + 2; ++i) {
    for (auto j = 0; j < multi_func_in_num; j++) {
      DataEnqueue(multi_input_qids[j], shape, TensorDataType::DT_UINT32, head_msg, input_value);
    }
  }
  for (int64_t i = 0; i < UDF_ST_QUEUE_MAX_DEPTH + 2; ++i) {
    for (auto j = 0; j < single_func_in_num; j++) {
      DataEnqueue(single_input_qids[j], shape, TensorDataType::DT_UINT32, head_msg, input_value);
    }
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  for (int64_t i = 0; i < UDF_ST_QUEUE_MAX_DEPTH + 2; i++) {
    for (size_t j = 0; j < multi_output_qids.size(); j++) {
      void *out_mbuf_ptr = nullptr;
      constexpr uint64_t kMaxWaitInMs = 20 * 1000;
      uint64_t wait_in_ms = 0;
      while (wait_in_ms < kMaxWaitInMs) {
        auto drv_ret = halQueueDeQueue(0, multi_output_qids[j], &out_mbuf_ptr);
        if (drv_ret == DRV_ERROR_QUEUE_EMPTY) {
          std::this_thread::sleep_for(std::chrono::milliseconds(kWaitInMsPerTime));
          wait_in_ms += kWaitInMsPerTime;
        } else {
          break;
        }
      }
      ASSERT_NE(out_mbuf_ptr, nullptr);
      Mbuf *out_mbuf = (Mbuf *)out_mbuf_ptr;
      if (j == 0) {
        std::vector<uint32_t> expect_output = {22, 44};
        CheckMbufData(out_mbuf, shape, TensorDataType::DT_UINT32, expect_output.data(), expect_output.size());
      } else {
        std::vector<uint32_t> expect_output = {33, 66};
        CheckMbufData(out_mbuf, shape, TensorDataType::DT_UINT32, expect_output.data(), expect_output.size());
      }
      halMbufFree(out_mbuf);
    }
  }

  for (int64_t i = 0; i < UDF_ST_QUEUE_MAX_DEPTH + 2; i++) {
    for (size_t j = 0; j < single_output_qids.size(); j++) {
      void *out_mbuf_ptr = nullptr;
      constexpr uint64_t kMaxWaitInMs = 20 * 1000;
      uint64_t wait_in_ms = 0;
      while (wait_in_ms < kMaxWaitInMs) {
        auto drv_ret = halQueueDeQueue(0, single_output_qids[j], &out_mbuf_ptr);
        if (drv_ret == DRV_ERROR_QUEUE_EMPTY) {
          std::this_thread::sleep_for(std::chrono::milliseconds(kWaitInMsPerTime));
          wait_in_ms += kWaitInMsPerTime;
        } else {
          break;
        }
      }
      ASSERT_NE(out_mbuf_ptr, nullptr);
      Mbuf *out_mbuf = (Mbuf *)out_mbuf_ptr;
      std::vector<uint32_t> expect_output = {22, 44};
      CheckMbufData(out_mbuf, shape, TensorDataType::DT_UINT32, expect_output.data(), expect_output.size());
      halMbufFree(out_mbuf);
    }
  }

  executor.Stop(true);
  executor.WaitForStop();
  executor.Destroy();
}
}  // namespace FlowFunc
