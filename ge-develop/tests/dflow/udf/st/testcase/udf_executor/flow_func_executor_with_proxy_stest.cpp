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
#include "common/scope_guard.h"
#include "config/global_config.h"
#include "execute/flow_func_executor.h"
#include "model/flow_func_model.h"
#include "config/global_config.h"
#include "flow_func/flow_func_config_manager.h"

namespace FlowFunc {
class FlowFuncExecutorWithProxySTest : public testing::Test {
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
    GlobalMockObject::verify();
    GlobalConfig::Instance().SetMessageQueueIds(UINT32_MAX, UINT32_MAX);
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

TEST_F(FlowFuncExecutorWithProxySTest, basic_test) {
  MOCKER(drvQueryProcessHostPid).stubs().will(repeat(DRV_ERROR_NO_PROCESS, 10)).then(returnValue(DRV_ERROR_NONE));
  uint32_t input_qid = 0;
  uint32_t output_qid = 0;
  std::string batch_model_path = CreateOnlyOneBatchModel(input_qid, output_qid);
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

  std::vector<int64_t> shape = {1, 2, 3, 4};
  float float_value = 123.1;
  DataEnqueue(input_qid, shape, TensorDataType::DT_FLOAT, float_value);
  void *out_mbuf_ptr = nullptr;

  constexpr uint64_t kMaxWaitInMs = 120 * 1000UL;
  constexpr uint64_t kWaitInMsPerTime = 10;
  uint64_t wait_in_ms = 0;
  while (wait_in_ms < kMaxWaitInMs) {
    auto drv_ret = halQueueDeQueue(0, output_qid, &out_mbuf_ptr);
    if (drv_ret == DRV_ERROR_NONE) {
      break;
    } else if (drv_ret == DRV_ERROR_QUEUE_EMPTY) {
      std::this_thread::sleep_for(std::chrono::milliseconds(kWaitInMsPerTime));
      wait_in_ms += kWaitInMsPerTime;
      continue;
    } else {
      ADD_FAILURE() << "drv_ret=" << drv_ret;
      break;
    }
  }
  ASSERT_NE(out_mbuf_ptr, nullptr) << "wait_in_ms=" << wait_in_ms;
  Mbuf *out_mbuf = (Mbuf *)out_mbuf_ptr;
  std::vector<int64_t> expect_output(CalcElementCnt(shape), (int64_t)float_value);
  CheckMbufData(out_mbuf, shape, TensorDataType::DT_INT64, expect_output.data(), expect_output.size());
  halMbufFree(out_mbuf);
}
}  // namespace FlowFunc
