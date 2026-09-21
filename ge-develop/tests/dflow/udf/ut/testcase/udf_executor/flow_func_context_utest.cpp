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
#include "securec.h"
#include "ascend_hal.h"

#define private public

#include "flow_func/flow_func_context.h"
#include "flow_func/flow_func_run_context.h"
#undef private
#include "flow_func/flow_func_params.h"
#include "flow_func/flow_func_run_context.h"

#include "mmpa/mmpa_api.h"
#include "mockcpp/mockcpp.hpp"
#include "execute/flow_model_impl.h"
#include "model/attr_value_impl.h"
#include "utils/udf_test_helper.h"
#include "dlog_pub.h"

namespace FlowFunc {
namespace {
struct MbufImpl {
  uint32_t mbuf_size;
  uint8_t reserve_head[256];
  RuntimeTensorDesc tensor_desc;
  uint8_t data[1024];
};

int halMbufAllocExStub(uint64_t size, unsigned int align, unsigned long flag, int grp_id, Mbuf **mbuf) {
  MbufImpl *mbuf_impl = new (std::nothrow) MbufImpl();
  memset_s(mbuf_impl, sizeof(MbufImpl), 0, sizeof(MbufImpl));
  mbuf_impl->mbuf_size = size;
  *mbuf = reinterpret_cast<Mbuf *>(mbuf_impl);
  return DRV_ERROR_NONE;
}

int halMbufFreeStub(Mbuf *mbuf) {
  MbufImpl *mbuf_impl = reinterpret_cast<MbufImpl *>(mbuf);
  delete mbuf_impl;
  return DRV_ERROR_NONE;
}

int halMbufCopyRefStub(Mbuf *mbuf, Mbuf **new_mbuf) {
  MbufImpl *src_mbuf_impl = reinterpret_cast<MbufImpl *>(mbuf);
  MbufImpl *new_mbuf_impl = new (std::nothrow) MbufImpl();
  memcpy_s(new_mbuf_impl, sizeof(MbufImpl), src_mbuf_impl, sizeof(MbufImpl));
  *new_mbuf = reinterpret_cast<Mbuf *>(new_mbuf_impl);
  return DRV_ERROR_NONE;
}

int halMbufGetBuffAddrStub(Mbuf *mbuf, void **buf) {
  MbufImpl *mbuf_impl = reinterpret_cast<MbufImpl *>(mbuf);
  *buf = &(mbuf_impl->tensor_desc);
  return DRV_ERROR_NONE;
}

int halMbufGetBuffSizeStub(Mbuf *mbuf, uint64_t *total_size) {
  MbufImpl *mbuf_impl = reinterpret_cast<MbufImpl *>(mbuf);
  *total_size = mbuf_impl->mbuf_size;
  return DRV_ERROR_NONE;
}

int halMbufSetDataLenStub(Mbuf *mbuf, uint64_t len) {
  MbufImpl *mbuf_impl = reinterpret_cast<MbufImpl *>(mbuf);
  mbuf_impl->mbuf_size = len;
  return DRV_ERROR_NONE;
}

int halMbufGetPrivInfoStub(Mbuf *mbuf, void **priv, unsigned int *size) {
  MbufImpl *mbuf_impl = reinterpret_cast<MbufImpl *>(mbuf);
  // start from first byte.
  *priv = &mbuf_impl->reserve_head;
  *size = 256;
  return DRV_ERROR_NONE;
}

int halMbufGetDataLenStub(Mbuf *mbuf, uint64_t *len) {
  MbufImpl *mbuf_impl = reinterpret_cast<MbufImpl *>(mbuf);
  *len = mbuf_impl->mbuf_size;
  return DRV_ERROR_NONE;
}

class DefaultMetaRunContext : public MetaRunContext {
 public:
  std::shared_ptr<FlowMsg> AllocTensorMsg(const std::vector<int64_t> &shape, TensorDataType data_type) override {
    return std::shared_ptr<FlowMsg>(nullptr);
  }

  std::shared_ptr<FlowMsg> AllocEmptyDataMsg(MsgType msg_type) override {
    return std::shared_ptr<FlowMsg>(nullptr);
  }

  int32_t SetOutput(uint32_t out_idx, std::shared_ptr<FlowMsg> out_msg) override {
    return 0;
  }

  int32_t RunFlowModel(const char *model_key, const std::vector<std::shared_ptr<FlowMsg>> &input_msgs,
                       std::vector<std::shared_ptr<FlowMsg>> &output_msgs, int32_t timeout) override {
    return 0;
  }

  int32_t GetUserData(void *data, size_t size, size_t offset = 0U) const override {
    return 0;
  }

  int32_t SetOutput(uint32_t out_idx, std::shared_ptr<FlowMsg> out_msg, const OutOptions &options) override {
    return 0;
  }

  int32_t SetMultiOutputs(uint32_t out_idx, const std::vector<std::shared_ptr<FlowMsg>> &out_msgs,
                          const OutOptions &options) override {
    return 0;
  }
};

class DefaultMetaParams : public MetaParams {
 public:
  const char *GetName() const override {
    return nullptr;
  }

  std::shared_ptr<const AttrValue> GetAttr(const char *attr_name) const override {
    return std::shared_ptr<const AttrValue>(nullptr);
  }

  size_t GetInputNum() const override {
    return 0;
  }

  size_t GetOutputNum() const override {
    return 0;
  }

  const char *GetWorkPath() const override {
    return nullptr;
  }

  int32_t GetRunningDeviceId() const override {
    return 0;
  }
};
}  // namespace
class FlowFuncContextUTest : public testing::Test {
 protected:
  virtual void SetUp() {
    MOCKER(halMbufAllocEx).defaults().will(invoke(halMbufAllocExStub));
    MOCKER(halMbufFree).defaults().will(invoke(halMbufFreeStub));
    MOCKER(halMbufGetBuffAddr).defaults().will(invoke(halMbufGetBuffAddrStub));
    MOCKER(halMbufGetBuffSize).defaults().will(invoke(halMbufGetBuffSizeStub));
    MOCKER(halMbufSetDataLen).defaults().will(invoke(halMbufSetDataLenStub));
    MOCKER(halMbufGetPrivInfo).defaults().will(invoke(halMbufGetPrivInfoStub));
    MOCKER(halMbufGetDataLen).defaults().will(invoke(halMbufGetDataLenStub));
    MOCKER(halMbufCopyRef).defaults().will(invoke(halMbufCopyRefStub));
  }

  virtual void TearDown() {
    GlobalMockObject::verify();
  }
};

TEST_F(FlowFuncContextUTest, DefaultTest) {
  DefaultMetaRunContext default_context;
  // void method, for cover
  auto ret = default_context.AllocRawDataMsg(1, 1);
  EXPECT_EQ(ret, nullptr);
  DefaultMetaParams default_params;
  auto ins_id = default_params.GetRunningInstanceId();
  EXPECT_EQ(ins_id, -1);
  auto ins_num = default_params.GetRunningInstanceNum();
  EXPECT_EQ(ins_num, -1);
}

TEST_F(FlowFuncContextUTest, flow_func_normal) {
  std::string lib_path = "/xxx";
  std::string work_path = "workpath0";
  std::vector<uint32_t> input_queue_ids = {1, 2};
  std::vector<uint32_t> output_queue_ids = {3, 4};
  std::shared_ptr<FlowFuncParams> flow_func_param(
      new (std::nothrow) FlowFuncParams("pp1", input_queue_ids.size(), output_queue_ids.size(), 0, 0));
  flow_func_param->SetLibPath(lib_path);
  flow_func_param->SetWorkPath(work_path);

  std::shared_ptr<FlowFuncRunContext> flow_func_run_context(new (std::nothrow)
                                                                FlowFuncRunContext(0, flow_func_param, nullptr));
  FlowFuncContext context(flow_func_param, flow_func_run_context);
  EXPECT_EQ(context.GetInputNum(), 2);
  EXPECT_EQ(context.GetOutputNum(), 2);
  EXPECT_EQ(context.GetWorkPath(), work_path);
}

TEST_F(FlowFuncContextUTest, with_invoked_model) {
  std::string flow_func_name = "flow_func_for_ut_normal";
  std::string lib_path = "/xxx";
  auto input_queue_ids = UdfTestHelper::CreateQueueDevInfos({1});
  auto output_queue_ids = UdfTestHelper::CreateQueueDevInfos({2});
  std::unique_ptr<FlowModelImpl> impl(new (std::nothrow) FlowModelImpl(input_queue_ids, output_queue_ids));
  std::shared_ptr<FlowFuncParams> flow_func_param(new (std::nothrow) FlowFuncParams("pp1", 1, 1, 0, 0));
  flow_func_param->AddFlowModel("model_key", std::move(impl));

  std::shared_ptr<FlowFuncRunContext> flow_func_run_context(new (std::nothrow)
                                                                FlowFuncRunContext(0, flow_func_param, nullptr));
  FlowFuncContext context(flow_func_param, flow_func_run_context);

  std::vector<std::shared_ptr<FlowMsg>> output_msgs;
  auto ret = context.RunFlowModel(nullptr, {}, output_msgs, 1000);
  EXPECT_NE(ret, FLOW_FUNC_SUCCESS);
  ret = context.RunFlowModel("ss", {}, output_msgs, 1000);
  EXPECT_NE(ret, FLOW_FUNC_SUCCESS);
  ret = context.RunFlowModel("model_key", {}, output_msgs, 1000);
  EXPECT_NE(ret, FLOW_FUNC_SUCCESS);
}

TEST_F(FlowFuncContextUTest, GetRunningDeviceId) {
  std::shared_ptr<FlowFuncParams> flow_func_param(new (std::nothrow) FlowFuncParams("pp1", 1, 1, 0, 0));
  std::shared_ptr<FlowFuncRunContext> flow_func_run_context(new (std::nothrow)
                                                                FlowFuncRunContext(0, flow_func_param, nullptr));
  FlowFuncContext context(flow_func_param, flow_func_run_context);
  EXPECT_EQ(context.GetRunningDeviceId(), 0);
  std::shared_ptr<FlowFuncParams> flow_func_param1(new (std::nothrow) FlowFuncParams("pp1", 1, 1, 10, 0));
  FlowFuncContext context1(flow_func_param1, flow_func_run_context);
  EXPECT_EQ(context1.GetRunningDeviceId(), 10);
}

TEST_F(FlowFuncContextUTest, FlowFuncParamInitBalanceScatterAttr) {
  std::shared_ptr<FlowFuncParams> flow_func_param(new (std::nothrow) FlowFuncParams("pp1", 1, 1, 0, 0));
  ff::udf::AttrValue balance_attr;
  balance_attr.set_b(true);
  auto attr_impl = std::make_shared<AttrValueImpl>(balance_attr);
  std::map<std::string, std::shared_ptr<const AttrValue>> attr_map;
  attr_map.emplace("_balance_scatter", attr_impl);
  flow_func_param->SetAttrMap(attr_map);
  int32_t ret = flow_func_param->Init();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  EXPECT_TRUE(flow_func_param->IsBalanceScatter());
  EXPECT_FALSE(flow_func_param->IsBalanceGather());
}

TEST_F(FlowFuncContextUTest, RaiseExceptionSucc) {
  const uint32_t priv_size = 256;
  uint8_t priv_info[priv_size];
  MbufHeadMsg *head_msg = reinterpret_cast<MbufHeadMsg *>(priv_info + priv_size - sizeof(MbufHeadMsg));
  head_msg->trans_id = 100;
  MbufInfo info;
  info.head_buf = reinterpret_cast<void *>(priv_info);
  info.head_buf_len = priv_size;
  std::vector<uint32_t> input_queue_ids = {1, 2};
  std::vector<uint32_t> output_queue_ids = {3, 4};
  std::shared_ptr<FlowFuncParams> flow_func_param(
      new (std::nothrow) FlowFuncParams("pp1", input_queue_ids.size(), output_queue_ids.size(), 0, 0));
  std::shared_ptr<FlowFuncRunContext> flow_func_run_context(new (std::nothrow)
                                                                FlowFuncRunContext(0, flow_func_param, nullptr, 0));
  flow_func_run_context->SetInputMbufHead(info);
  FlowFuncContext context(flow_func_param, flow_func_run_context);
  MOCKER(halEschedSubmitEvent).stubs().will(returnValue(DRV_ERROR_NONE));
  context.RaiseException(1000, 2);
  EXPECT_EQ(static_cast<FlowFuncRunContext *>(context.run_context_.get())->trans_id_to_exception_raised_.size(), 1);
}

TEST_F(FlowFuncContextUTest, GetExceptionEmpty) {
  const uint32_t priv_size = 256;
  uint8_t priv_info[priv_size];
  MbufHeadMsg *head_msg = reinterpret_cast<MbufHeadMsg *>(priv_info + priv_size - sizeof(MbufHeadMsg));
  head_msg->trans_id = 100;
  MbufInfo info;
  info.head_buf = reinterpret_cast<void *>(priv_info);
  info.head_buf_len = priv_size;
  std::vector<uint32_t> input_queue_ids = {1, 2};
  std::vector<uint32_t> output_queue_ids = {3, 4};
  std::shared_ptr<FlowFuncParams> flow_func_param(
      new (std::nothrow) FlowFuncParams("pp1", input_queue_ids.size(), output_queue_ids.size(), 0, 0));
  std::shared_ptr<FlowFuncRunContext> flow_func_run_context(new (std::nothrow)
                                                                FlowFuncRunContext(0, flow_func_param, nullptr, 0));
  flow_func_run_context->SetInputMbufHead(info);
  FlowFuncContext context(flow_func_param, flow_func_run_context);
  int32_t exp_code;
  uint64_t user_context_id;
  EXPECT_EQ(context.GetException(exp_code, user_context_id), false);
}

TEST_F(FlowFuncContextUTest, FlowFuncParamInitBalanceGatherAttr) {
  std::shared_ptr<FlowFuncParams> flow_func_param(new (std::nothrow) FlowFuncParams("pp1", 1, 1, 0, 0));
  ff::udf::AttrValue balance_attr;
  balance_attr.set_b(true);
  auto attr_impl = std::make_shared<AttrValueImpl>(balance_attr);
  std::map<std::string, std::shared_ptr<const AttrValue>> attr_map;
  attr_map.emplace("_balance_gather", attr_impl);
  flow_func_param->SetAttrMap(attr_map);
  int32_t ret = flow_func_param->Init();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  EXPECT_FALSE(flow_func_param->IsBalanceScatter());
  EXPECT_TRUE(flow_func_param->IsBalanceGather());
}

TEST_F(FlowFuncContextUTest, FlowFuncParamInitBalanceAttrBothSet) {
  std::shared_ptr<FlowFuncParams> flow_func_param(new (std::nothrow) FlowFuncParams("pp1", 1, 1, 0, 0));
  ff::udf::AttrValue balance_scatter_attr;
  balance_scatter_attr.set_b(true);
  auto attr_impl = std::make_shared<AttrValueImpl>(balance_scatter_attr);
  std::map<std::string, std::shared_ptr<const AttrValue>> attr_map;
  attr_map.emplace("_balance_scatter", attr_impl);
  attr_map.emplace("_balance_gather", attr_impl);
  flow_func_param->SetAttrMap(attr_map);
  int32_t ret = flow_func_param->Init();
  EXPECT_NE(ret, FLOW_FUNC_SUCCESS);
}

TEST_F(FlowFuncContextUTest, SetOutputWithBalanceOptionsScatter) {
  std::shared_ptr<FlowFuncParams> flow_func_param(new (std::nothrow) FlowFuncParams("pp", 1, 1, 0, 0));
  ASSERT_NE(flow_func_param, nullptr);
  ff::udf::AttrValue balance_attr;
  balance_attr.set_b(true);
  auto attr_impl = std::make_shared<AttrValueImpl>(balance_attr);
  std::map<std::string, std::shared_ptr<const AttrValue>> attr_map;
  attr_map.emplace("_balance_scatter", attr_impl);
  flow_func_param->SetAttrMap(attr_map);
  flow_func_param->Init();
  std::shared_ptr<FlowMsg> out_msg;
  auto callback_func = [&out_msg](uint32_t out_idx, const std::shared_ptr<FlowMsg> &flow_msg) {
    out_msg = flow_msg;
    return FLOW_FUNC_SUCCESS;
  };
  std::shared_ptr<FlowFuncRunContext> flow_func_run_context(new (std::nothrow)
                                                                FlowFuncRunContext(0, flow_func_param, callback_func));
  FlowFuncContext context(flow_func_param, flow_func_run_context);
  auto msg = context.AllocTensorMsg({2, 2}, TensorDataType::DT_INT32);

  OutOptions options;
  auto balance_config = options.MutableBalanceConfig();
  balance_config->SetAffinityPolicy(AffinityPolicy::NO_AFFINITY);
  int32_t pos_row = 1;
  int32_t pos_col = 2;
  balance_config->SetDataPos({{pos_row, pos_col}});
  BalanceWeight weight;
  weight.rowNum = 3;
  weight.colNum = 4;
  balance_config->SetBalanceWeight(weight);

  int32_t ret = context.SetOutput(0, msg, options);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  auto mbuf_msg = std::dynamic_pointer_cast<MbufFlowMsg>(out_msg);
  const auto &mbuf_info = mbuf_msg->GetMbufInfo();
  auto head_buf = static_cast<const char *>(mbuf_info.head_buf) + mbuf_info.head_buf_len - sizeof(MbufHeadMsg);
  const auto *head_msg = reinterpret_cast<const MbufHeadMsg *>(head_buf);
  EXPECT_EQ(head_msg->data_label, pos_row * weight.colNum + pos_col + 1);
  EXPECT_EQ(head_msg->route_label, pos_row * weight.colNum + pos_col + 1);
}

TEST_F(FlowFuncContextUTest, SetMultiOutput) {
  std::shared_ptr<FlowFuncParams> flow_func_param(new (std::nothrow) FlowFuncParams("pp", 1, 1, 0, 0));
  ASSERT_NE(flow_func_param, nullptr);
  ff::udf::AttrValue balance_attr;
  balance_attr.set_b(true);
  auto attr_impl = std::make_shared<AttrValueImpl>(balance_attr);
  std::map<std::string, std::shared_ptr<const AttrValue>> attr_map;
  attr_map.emplace("_balance_gather", attr_impl);
  flow_func_param->SetAttrMap(attr_map);
  flow_func_param->Init();
  std::vector<std::shared_ptr<FlowMsg>> out_msgs;
  auto callback_func = [&out_msgs](uint32_t out_idx, const std::shared_ptr<FlowMsg> &flow_msg) {
    out_msgs.emplace_back(flow_msg);
    return FLOW_FUNC_SUCCESS;
  };
  std::shared_ptr<FlowFuncRunContext> flow_func_run_context(new (std::nothrow)
                                                                FlowFuncRunContext(0, flow_func_param, callback_func));
  FlowFuncContext context(flow_func_param, flow_func_run_context);
  auto msg1 = context.AllocTensorMsg({2, 2}, TensorDataType::DT_INT32);
  auto msg2 = context.AllocTensorMsg({2, 2}, TensorDataType::DT_INT32);

  OutOptions options;
  auto balance_config = options.MutableBalanceConfig();
  balance_config->SetAffinityPolicy(AffinityPolicy::ROW_AFFINITY);
  int32_t pos_row1 = 1;
  int32_t pos_col1 = 2;
  int32_t pos_row2 = 3;
  int32_t pos_col2 = 4;
  balance_config->SetDataPos({{pos_row1, pos_col1}, {pos_row2, pos_col2}});
  BalanceWeight weight;
  weight.rowNum = 5;
  weight.colNum = 6;
  balance_config->SetBalanceWeight(weight);

  int32_t ret = context.SetMultiOutputs(0, {msg1, msg2}, options);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ASSERT_EQ(out_msgs.size(), 2);
  auto mbuf_msg1 = std::dynamic_pointer_cast<MbufFlowMsg>(out_msgs[0]);
  auto mbuf_msg2 = std::dynamic_pointer_cast<MbufFlowMsg>(out_msgs[1]);
  const auto &mbuf_info1 = mbuf_msg1->GetMbufInfo();
  const auto &mbuf_info2 = mbuf_msg2->GetMbufInfo();
  auto head_buf1 = static_cast<const char *>(mbuf_info1.head_buf) + mbuf_info1.head_buf_len - sizeof(MbufHeadMsg);
  auto head_buf2 = static_cast<const char *>(mbuf_info2.head_buf) + mbuf_info2.head_buf_len - sizeof(MbufHeadMsg);
  const auto *head_msg1 = reinterpret_cast<const MbufHeadMsg *>(head_buf1);
  const auto *head_msg2 = reinterpret_cast<const MbufHeadMsg *>(head_buf2);
  EXPECT_EQ(head_msg1->data_label, pos_row1 * weight.colNum + pos_col1 + 1);
  EXPECT_EQ(head_msg1->route_label, pos_row1 + 1);

  EXPECT_EQ(head_msg2->data_label, pos_row2 * weight.colNum + pos_col2 + 1);
  EXPECT_EQ(head_msg2->route_label, pos_row2 + 1);
}

TEST_F(FlowFuncContextUTest, AllocTensorMsgWithAlign_RANGE_CHECK) {
  std::vector<uint32_t> input_queue_ids = {1, 2};
  std::vector<uint32_t> output_queue_ids = {3, 4};
  std::shared_ptr<FlowFuncParams> flow_func_param(
      new (std::nothrow) FlowFuncParams("pp1", input_queue_ids.size(), output_queue_ids.size(), 0, 0));
  std::shared_ptr<FlowFuncRunContext> flow_func_run_context(new (std::nothrow)
                                                                FlowFuncRunContext(0, flow_func_param, nullptr));
  FlowFuncContext context(flow_func_param, flow_func_run_context);
  EXPECT_EQ(flow_func_run_context->AllocTensorMsgWithAlign({1, 2, 3}, TensorDataType::DT_FLOAT, 16), nullptr);
  EXPECT_NE(flow_func_run_context->AllocTensorMsgWithAlign({1, 2, 3}, TensorDataType::DT_FLOAT, 256), nullptr);
}
}  // namespace FlowFunc