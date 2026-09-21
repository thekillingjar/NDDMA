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
#include "gtest/gtest.h"
#include "securec.h"
#include "ascend_hal.h"
#include "mockcpp/mockcpp.hpp"
#define private public
#include "built_in/time_batch_flow_func.h"
#include "flow_func/flow_func_context.h"
#include "model/attr_value_impl.h"
#include "flow_func/mbuf_flow_msg.h"
#undef private
#include "flow_func/flow_func_params.h"
#include "flow_func/flow_func_run_context.h"

namespace FlowFunc {
namespace {
struct MbufImpl {
  uint32_t mbuf_size;
  uint8_t reserve_head[256 - sizeof(MbufHeadMsg)];
  MbufHeadMsg head_msg;
  RuntimeTensorDesc tensor_desc;
  uint8_t data[2048];
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

int halMbufGetBuffAddrStub(Mbuf *mbuf, void **buf) {
  MbufImpl *mbuf_impl = reinterpret_cast<MbufImpl *>(mbuf);
  *buf = mbuf_impl->data;
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
}  // namespace

class TimeBatchFlowFuncUTest : public testing::Test {
 protected:
  virtual void SetUp() {
    MOCKER(halMbufAllocEx).defaults().will(invoke(halMbufAllocExStub));
    MOCKER(halMbufFree).defaults().will(invoke(halMbufFreeStub));
    MOCKER(halMbufGetBuffAddr).defaults().will(invoke(halMbufGetBuffAddrStub));
    MOCKER(halMbufGetBuffSize).defaults().will(invoke(halMbufGetBuffSizeStub));
    MOCKER(halMbufSetDataLen).defaults().will(invoke(halMbufSetDataLenStub));
    MOCKER(halMbufGetPrivInfo).defaults().will(invoke(halMbufGetPrivInfoStub));
    MOCKER(halMbufGetDataLen).defaults().will(invoke(halMbufGetDataLenStub));
  }

  virtual void TearDown() {
    GlobalMockObject::verify();
  }
};

std::map<std::string, std::shared_ptr<const AttrValue>> CreateAttrs(int64_t window_value, int64_t batch_dim_value,
                                                                    bool drop_remainder_value) {
  std::map<std::string, std::shared_ptr<const AttrValue>> attrs;
  ff::udf::AttrValue window;
  window.set_i(window_value);
  auto window_attr = std::make_shared<AttrValueImpl>(window);
  attrs["window"] = window_attr;
  ff::udf::AttrValue batch_dim;
  batch_dim.set_i(batch_dim_value);
  auto batch_dim_attr = std::make_shared<AttrValueImpl>(batch_dim);
  attrs["batch_dim"] = batch_dim_attr;
  ff::udf::AttrValue drop_remainder;
  drop_remainder.set_b(drop_remainder_value);
  auto dropRemainderAttr = std::make_shared<AttrValueImpl>(drop_remainder);
  attrs["drop_remainder"] = dropRemainderAttr;
  return attrs;
}

FlowFuncContext CreateTimeBatchFlowFuncContext(std::map<std::string, std::shared_ptr<const AttrValue>> &attrs) {
  std::shared_ptr<FlowFuncParams> flow_func_param(new (std::nothrow) FlowFuncParams("TimeBatch", 2, 2, 0, 0));
  flow_func_param->SetAttrMap(attrs);
  std::shared_ptr<FlowFuncRunContext> flow_func_run_context(new (std::nothrow)
                                                                FlowFuncRunContext(0, flow_func_param, nullptr));
  FlowFuncContext flow_func_context(flow_func_param, flow_func_run_context);
  return flow_func_context;
}

TEST_F(TimeBatchFlowFuncUTest, init_success) {
  TimeBatchFlowFunc time_batch;
  std::map<std::string, std::shared_ptr<const AttrValue>> attrs = CreateAttrs(1, 1, true);
  FlowFuncContext flow_func_context = CreateTimeBatchFlowFuncContext(attrs);
  time_batch.SetContext(&flow_func_context);
  EXPECT_EQ(time_batch.Init(), FLOW_FUNC_SUCCESS);
  EXPECT_EQ(time_batch.window_, 1);
  EXPECT_EQ(time_batch.batch_dim_, 1);
  EXPECT_EQ(time_batch.drop_remainder_, true);
  EXPECT_EQ(time_batch.input_cache_.size(), 0);
  EXPECT_EQ(time_batch.start_time_, 0U);
  EXPECT_EQ(time_batch.end_time_, 0U);
  EXPECT_EQ(time_batch.is_time_batch_ok_, false);
  EXPECT_EQ(time_batch.published_out_num_, 0U);
  EXPECT_EQ(time_batch.output_num_, flow_func_context.GetOutputNum());
}

TEST_F(TimeBatchFlowFuncUTest, init_get_attr_window_failed) {
  TimeBatchFlowFunc time_batch;
  std::map<std::string, std::shared_ptr<const AttrValue>> empty_attr;
  FlowFuncContext flow_func_context = CreateTimeBatchFlowFuncContext(empty_attr);
  time_batch.SetContext(&flow_func_context);
  EXPECT_EQ(time_batch.Init(), FLOW_FUNC_ERR_ATTR_NOT_EXITS);
}

TEST_F(TimeBatchFlowFuncUTest, init_attr_window_value_invalid_failed) {
  TimeBatchFlowFunc time_batch;
  std::map<std::string, std::shared_ptr<const AttrValue>> attrs;
  ff::udf::AttrValue window;
  window.set_i(-3);
  auto window_attr = std::make_shared<AttrValueImpl>(window);
  attrs["window"] = window_attr;
  FlowFuncContext flow_func_context = CreateTimeBatchFlowFuncContext(attrs);
  time_batch.SetContext(&flow_func_context);
  EXPECT_EQ(time_batch.Init(), FLOW_FUNC_ERR_PARAM_INVALID);
}

TEST_F(TimeBatchFlowFuncUTest, init_get_attr_batch_dim_failed) {
  TimeBatchFlowFunc time_batch;
  std::map<std::string, std::shared_ptr<const AttrValue>> attrs;
  ff::udf::AttrValue window;
  window.set_i(-1);
  auto window_attr = std::make_shared<AttrValueImpl>(window);
  attrs["window"] = window_attr;
  FlowFuncContext flow_func_context = CreateTimeBatchFlowFuncContext(attrs);
  time_batch.SetContext(&flow_func_context);
  EXPECT_EQ(time_batch.Init(), FLOW_FUNC_ERR_ATTR_NOT_EXITS);
}

TEST_F(TimeBatchFlowFuncUTest, init_attr_batch_dim_value_invalid_failed) {
  TimeBatchFlowFunc time_batch;
  std::map<std::string, std::shared_ptr<const AttrValue>> attrs;
  ff::udf::AttrValue window;
  window.set_i(1);
  auto window_attr = std::make_shared<AttrValueImpl>(window);
  attrs["window"] = window_attr;
  ff::udf::AttrValue batch_dim;
  batch_dim.set_i(-2);
  auto batch_dim_attr = std::make_shared<AttrValueImpl>(batch_dim);
  attrs["batch_dim"] = batch_dim_attr;
  FlowFuncContext flow_func_context = CreateTimeBatchFlowFuncContext(attrs);
  time_batch.SetContext(&flow_func_context);
  EXPECT_EQ(time_batch.Init(), FLOW_FUNC_ERR_PARAM_INVALID);
}

TEST_F(TimeBatchFlowFuncUTest, init_get_attr_drop_remainder_failed) {
  TimeBatchFlowFunc time_batch;
  std::map<std::string, std::shared_ptr<const AttrValue>> attrs;
  ff::udf::AttrValue window;
  window.set_i(1);
  auto window_attr = std::make_shared<AttrValueImpl>(window);
  attrs["window"] = window_attr;
  ff::udf::AttrValue batch_dim;
  batch_dim.set_i(-1);
  auto batch_dim_attr = std::make_shared<AttrValueImpl>(batch_dim);
  attrs["batch_dim"] = batch_dim_attr;
  FlowFuncContext flow_func_context = CreateTimeBatchFlowFuncContext(attrs);
  time_batch.SetContext(&flow_func_context);
  EXPECT_EQ(time_batch.Init(), FLOW_FUNC_ERR_ATTR_NOT_EXITS);
}

TEST_F(TimeBatchFlowFuncUTest, init_output_num_invalid) {
  TimeBatchFlowFunc time_batch;
  uint32_t output_num = 0;
  std::map<std::string, std::shared_ptr<const AttrValue>> attrs = CreateAttrs(1, 1, true);
  std::shared_ptr<FlowFuncParams> flow_func_param(new (std::nothrow) FlowFuncParams("TimeBatch", 1, output_num, 0, 0));
  flow_func_param->SetAttrMap(attrs);
  std::shared_ptr<FlowFuncRunContext> flow_func_run_context(new (std::nothrow)
                                                                FlowFuncRunContext(0, flow_func_param, nullptr));
  FlowFuncContext flow_func_context(flow_func_param, flow_func_run_context);

  time_batch.SetContext(&flow_func_context);
  EXPECT_EQ(time_batch.Init(), FLOW_FUNC_ERR_PARAM_INVALID);
}

TEST_F(TimeBatchFlowFuncUTest, proc_input_empty) {
  TimeBatchFlowFunc time_batch;
  std::map<std::string, std::shared_ptr<const AttrValue>> attrs = CreateAttrs(1, 1, true);
  FlowFuncContext flow_func_context = CreateTimeBatchFlowFuncContext(attrs);
  time_batch.SetContext(&flow_func_context);
  EXPECT_EQ(time_batch.Init(), FLOW_FUNC_SUCCESS);
  std::vector<std::shared_ptr<FlowMsg>> input_msgs;
  MOCKER_CPP_VIRTUAL(flow_func_context, &FlowFuncContext::SetOutput,
                     int32_t(FlowFuncContext::*)(uint32_t, std::shared_ptr<FlowMsg>))
      .stubs()
      .will(returnValue(FLOW_FUNC_SUCCESS));
  EXPECT_EQ(time_batch.Proc(input_msgs), FLOW_FUNC_SUCCESS);
}

TEST_F(TimeBatchFlowFuncUTest, proc_input_error) {
  TimeBatchFlowFunc time_batch;
  std::map<std::string, std::shared_ptr<const AttrValue>> attrs = CreateAttrs(1, 1, true);
  FlowFuncContext flow_func_context = CreateTimeBatchFlowFuncContext(attrs);
  time_batch.SetContext(&flow_func_context);
  EXPECT_EQ(time_batch.Init(), FLOW_FUNC_SUCCESS);
  std::vector<std::shared_ptr<FlowMsg>> input_msgs;
  std::shared_ptr<MbufFlowMsg> input_msg = std::make_shared<MbufFlowMsg>(nullptr);
  MbufHeadMsg head_msg;
  head_msg.ret_code = -1;
  input_msg->head_msg_ = &head_msg;
  input_msgs.emplace_back(input_msg);
  MOCKER_CPP_VIRTUAL(flow_func_context, &FlowFuncContext::SetOutput,
                     int32_t(FlowFuncContext::*)(uint32_t, std::shared_ptr<FlowMsg>))
      .stubs()
      .will(returnValue(FLOW_FUNC_SUCCESS));
  EXPECT_EQ(time_batch.Proc(input_msgs), FLOW_FUNC_SUCCESS);
}

TEST_F(TimeBatchFlowFuncUTest, proc_input_nullptr) {
  TimeBatchFlowFunc time_batch;
  std::map<std::string, std::shared_ptr<const AttrValue>> attrs = CreateAttrs(1, 1, true);
  FlowFuncContext flow_func_context = CreateTimeBatchFlowFuncContext(attrs);
  time_batch.SetContext(&flow_func_context);
  EXPECT_EQ(time_batch.Init(), FLOW_FUNC_SUCCESS);
  std::vector<std::shared_ptr<FlowMsg>> input_msgs;
  input_msgs.emplace_back(nullptr);
  MOCKER_CPP_VIRTUAL(flow_func_context, &FlowFuncContext::SetOutput,
                     int32_t(FlowFuncContext::*)(uint32_t, std::shared_ptr<FlowMsg>))
      .stubs()
      .will(returnValue(FLOW_FUNC_SUCCESS));
  EXPECT_EQ(time_batch.Proc(input_msgs), FLOW_FUNC_SUCCESS);
}

TEST_F(TimeBatchFlowFuncUTest, proc_all_input_tensor_null_eos) {
  TimeBatchFlowFunc time_batch;
  std::map<std::string, std::shared_ptr<const AttrValue>> attrs = CreateAttrs(1, 1, true);
  FlowFuncContext flow_func_context = CreateTimeBatchFlowFuncContext(attrs);
  time_batch.SetContext(&flow_func_context);
  EXPECT_EQ(time_batch.Init(), FLOW_FUNC_SUCCESS);
  std::vector<std::shared_ptr<FlowMsg>> input_msgs;
  std::shared_ptr<MbufFlowMsg> input_msg = std::make_shared<MbufFlowMsg>(nullptr);
  MbufHeadMsg head_msg{0};
  head_msg.flags = static_cast<uint32_t>(FlowFlag::FLOW_FLAG_EOS);
  input_msg->head_msg_ = &head_msg;
  input_msgs.emplace_back(input_msg);
  MOCKER_CPP_VIRTUAL(flow_func_context, &FlowFuncContext::SetOutput,
                     int32_t(FlowFuncContext::*)(uint32_t, std::shared_ptr<FlowMsg>))
      .stubs()
      .will(returnValue(FLOW_FUNC_SUCCESS));
  EXPECT_EQ(time_batch.Proc(input_msgs), FLOW_FUNC_SUCCESS);
}

TEST_F(TimeBatchFlowFuncUTest, proc_all_input_tensor_null_eos_publish_empty_output_error) {
  TimeBatchFlowFunc time_batch;
  std::map<std::string, std::shared_ptr<const AttrValue>> attrs = CreateAttrs(1, 1, true);
  FlowFuncContext flow_func_context = CreateTimeBatchFlowFuncContext(attrs);
  time_batch.SetContext(&flow_func_context);
  EXPECT_EQ(time_batch.Init(), FLOW_FUNC_SUCCESS);
  std::vector<std::shared_ptr<FlowMsg>> input_msgs;
  std::shared_ptr<MbufFlowMsg> input_msg = std::make_shared<MbufFlowMsg>(nullptr);
  MbufHeadMsg head_msg{0};
  head_msg.flags = static_cast<uint32_t>(FlowFlag::FLOW_FLAG_EOS);
  input_msg->head_msg_ = &head_msg;
  input_msgs.emplace_back(input_msg);
  MOCKER_CPP_VIRTUAL(flow_func_context, &FlowFuncContext::SetOutput,
                     int32_t(FlowFuncContext::*)(uint32_t, std::shared_ptr<FlowMsg>))
      .stubs()
      .will(returnValue(FLOW_FUNC_FAILED));
  EXPECT_EQ(time_batch.Proc(input_msgs), FLOW_FUNC_FAILED);
}

TEST_F(TimeBatchFlowFuncUTest, proc_one_input_tensor_null) {
  TimeBatchFlowFunc time_batch;
  std::map<std::string, std::shared_ptr<const AttrValue>> attrs = CreateAttrs(1, 1, true);
  FlowFuncContext flow_func_context = CreateTimeBatchFlowFuncContext(attrs);
  time_batch.SetContext(&flow_func_context);
  EXPECT_EQ(time_batch.Init(), FLOW_FUNC_SUCCESS);
  std::vector<std::shared_ptr<FlowMsg>> input_msgs;
  std::shared_ptr<MbufFlowMsg> input_msg = std::make_shared<MbufFlowMsg>(nullptr);
  MbufHeadMsg head_msg{0};
  input_msg->head_msg_ = &head_msg;
  input_msgs.emplace_back(input_msg);
  std::vector<int64_t> shape = {1, 2, 3};
  TensorDataType data_type = TensorDataType::DT_INT32;
  MbufHead mbuf_head;
  std::shared_ptr<MbufFlowMsg> mbuf_flow_msg = MbufFlowMsg::AllocTensorMsg(shape, data_type, 0, mbuf_head);
  input_msgs.emplace_back(mbuf_flow_msg);
  MOCKER_CPP_VIRTUAL(flow_func_context, &FlowFuncContext::SetOutput,
                     int32_t(FlowFuncContext::*)(uint32_t, std::shared_ptr<FlowMsg>))
      .stubs()
      .will(returnValue(FLOW_FUNC_SUCCESS));
  EXPECT_EQ(time_batch.Proc(input_msgs), FLOW_FUNC_SUCCESS);
}

TEST_F(TimeBatchFlowFuncUTest, proc_success_no_output) {
  TimeBatchFlowFunc time_batch;
  std::map<std::string, std::shared_ptr<const AttrValue>> attrs = CreateAttrs(1, 1, true);
  FlowFuncContext flow_func_context = CreateTimeBatchFlowFuncContext(attrs);
  time_batch.SetContext(&flow_func_context);
  EXPECT_EQ(time_batch.Init(), FLOW_FUNC_SUCCESS);
  std::vector<std::shared_ptr<FlowMsg>> input_msgs;
  std::vector<int64_t> shape = {1, 2, 3};
  TensorDataType data_type = TensorDataType::DT_INT32;
  MbufHead mbuf_head;
  std::shared_ptr<MbufFlowMsg> mbuf_flow_msg = MbufFlowMsg::AllocTensorMsg(shape, data_type, 0, mbuf_head);
  input_msgs.emplace_back(mbuf_flow_msg);
  input_msgs.emplace_back(mbuf_flow_msg);
  EXPECT_EQ(time_batch.Proc(input_msgs), FLOW_FUNC_SUCCESS);
  EXPECT_EQ(time_batch.input_cache_.size(), 2);
}

TEST_F(TimeBatchFlowFuncUTest, proc_success_has_output) {
  TimeBatchFlowFunc time_batch;
  std::map<std::string, std::shared_ptr<const AttrValue>> attrs = CreateAttrs(1, 1, false);
  FlowFuncContext flow_func_context = CreateTimeBatchFlowFuncContext(attrs);
  time_batch.SetContext(&flow_func_context);
  EXPECT_EQ(time_batch.Init(), FLOW_FUNC_SUCCESS);
  std::vector<std::shared_ptr<FlowMsg>> input_msgs;
  std::vector<int64_t> shape = {1, 2, 3};
  TensorDataType data_type = TensorDataType::DT_INT32;
  MbufHead mbuf_head;
  std::shared_ptr<MbufFlowMsg> mbuf_flow_msg = MbufFlowMsg::AllocTensorMsg(shape, data_type, 0, mbuf_head);
  input_msgs.emplace_back(mbuf_flow_msg);
  input_msgs.emplace_back(mbuf_flow_msg);
  EXPECT_EQ(time_batch.Proc(input_msgs), FLOW_FUNC_SUCCESS);
  EXPECT_EQ(time_batch.input_cache_.size(), 2);
  std::vector<std::shared_ptr<FlowMsg>> input_msgs2;
  std::shared_ptr<MbufFlowMsg> mbuf_flow_msg2 = MbufFlowMsg::AllocTensorMsg(shape, data_type, 0, mbuf_head);
  mbuf_flow_msg2->SetFlowFlags(1);
  input_msgs2.emplace_back(mbuf_flow_msg2);
  input_msgs2.emplace_back(mbuf_flow_msg2);
  MOCKER_CPP_VIRTUAL(flow_func_context, &FlowFuncContext::SetOutput,
                     int32_t(FlowFuncContext::*)(uint32_t, std::shared_ptr<FlowMsg>))
      .stubs()
      .will(returnValue(FLOW_FUNC_SUCCESS));
  EXPECT_EQ(time_batch.Proc(input_msgs2), FLOW_FUNC_SUCCESS);
  EXPECT_EQ(time_batch.input_cache_.size(), 0);
}

TEST_F(TimeBatchFlowFuncUTest, proc_success_has_output_default_attr) {
  TimeBatchFlowFunc time_batch;
  std::map<std::string, std::shared_ptr<const AttrValue>> attrs = CreateAttrs(1, -1, false);
  FlowFuncContext flow_func_context = CreateTimeBatchFlowFuncContext(attrs);
  time_batch.SetContext(&flow_func_context);
  EXPECT_EQ(time_batch.Init(), FLOW_FUNC_SUCCESS);
  std::vector<std::shared_ptr<FlowMsg>> input_msgs;
  std::vector<int64_t> shape = {1, 2, 3};
  TensorDataType data_type = TensorDataType::DT_INT32;
  MbufHead mbuf_head;
  std::shared_ptr<MbufFlowMsg> mbuf_flow_msg = MbufFlowMsg::AllocTensorMsg(shape, data_type, 0, mbuf_head);
  input_msgs.emplace_back(mbuf_flow_msg);
  input_msgs.emplace_back(mbuf_flow_msg);
  EXPECT_EQ(time_batch.Proc(input_msgs), FLOW_FUNC_SUCCESS);
  EXPECT_EQ(time_batch.input_cache_.size(), 2);
  std::vector<std::shared_ptr<FlowMsg>> input_msgs2;
  std::shared_ptr<MbufFlowMsg> mbuf_flow_msg2 = MbufFlowMsg::AllocTensorMsg(shape, data_type, 0, mbuf_head);
  mbuf_flow_msg2->SetFlowFlags(1);
  input_msgs2.emplace_back(mbuf_flow_msg2);
  input_msgs2.emplace_back(mbuf_flow_msg2);
  MOCKER_CPP_VIRTUAL(flow_func_context, &FlowFuncContext::SetOutput,
                     int32_t(FlowFuncContext::*)(uint32_t, std::shared_ptr<FlowMsg>))
      .stubs()
      .will(returnValue(FLOW_FUNC_SUCCESS));
  EXPECT_EQ(time_batch.Proc(input_msgs2), FLOW_FUNC_SUCCESS);
  EXPECT_EQ(time_batch.input_cache_.size(), 0);
}

TEST_F(TimeBatchFlowFuncUTest, proc_success_output_drop) {
  TimeBatchFlowFunc time_batch;
  std::map<std::string, std::shared_ptr<const AttrValue>> attrs = CreateAttrs(1, -1, true);
  FlowFuncContext flow_func_context = CreateTimeBatchFlowFuncContext(attrs);
  time_batch.SetContext(&flow_func_context);
  EXPECT_EQ(time_batch.Init(), FLOW_FUNC_SUCCESS);
  std::vector<std::shared_ptr<FlowMsg>> input_msgs;
  std::vector<int64_t> shape = {1, 2, 3};
  TensorDataType data_type = TensorDataType::DT_INT32;
  MbufHead mbuf_head;
  std::shared_ptr<MbufFlowMsg> mbuf_flow_msg = MbufFlowMsg::AllocTensorMsg(shape, data_type, 0, mbuf_head);
  input_msgs.emplace_back(mbuf_flow_msg);
  input_msgs.emplace_back(mbuf_flow_msg);
  EXPECT_EQ(time_batch.Proc(input_msgs), FLOW_FUNC_SUCCESS);
  EXPECT_EQ(time_batch.input_cache_.size(), 2);
  std::vector<std::shared_ptr<FlowMsg>> input_msgs2;
  std::shared_ptr<MbufFlowMsg> mbuf_flow_msg2 = MbufFlowMsg::AllocTensorMsg(shape, data_type, 0, mbuf_head);
  mbuf_flow_msg2->SetFlowFlags(1);
  input_msgs2.emplace_back(mbuf_flow_msg2);
  input_msgs2.emplace_back(mbuf_flow_msg2);
  MOCKER_CPP_VIRTUAL(flow_func_context, &FlowFuncContext::SetOutput,
                     int32_t(FlowFuncContext::*)(uint32_t, std::shared_ptr<FlowMsg>))
      .stubs()
      .will(returnValue(FLOW_FUNC_SUCCESS));
  EXPECT_EQ(time_batch.Proc(input_msgs2), FLOW_FUNC_SUCCESS);
  EXPECT_EQ(time_batch.input_cache_.size(), 0);
}
}  // namespace FlowFunc