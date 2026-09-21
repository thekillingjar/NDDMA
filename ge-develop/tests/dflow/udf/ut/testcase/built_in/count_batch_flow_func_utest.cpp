/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include "gtest/gtest.h"
#include "securec.h"
#include "ascend_hal.h"
#include "mockcpp/mockcpp.hpp"
#define private public
#include "built_in/count_batch_flow_func.h"
#include "flow_func/flow_func_timer.h"
#include "flow_func/flow_func_context.h"
#include "model/attr_value_impl.h"
#include "flow_func/mbuf_flow_msg.h"
#undef private
#include "flow_func/flow_func_params.h"
#include "flow_func/flow_func_run_context.h"

using namespace std;
namespace FlowFunc {
namespace {
struct MbufImpl {
  uint32_t mbuf_size;
  uint8_t reserve_head[256 - sizeof(MbufHeadMsg)];
  MbufHeadMsg head_msg;
  RuntimeTensorDesc tensor_desc;
  uint8_t data[2048];
};
std::vector<std::shared_ptr<FlowMsg>> output_tensors;
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

class COUNT_BATCH_FLOW_FUNC_UTEST : public testing::Test {
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

std::map<std::string, std::shared_ptr<const AttrValue>> CreateAttrs(int64_t batch_size_value, int64_t timeout_value,
                                                                    bool padding_flag, int64_t slide_stride_value,
                                                                    bool has_padding_attr = true) {
  std::map<std::string, std::shared_ptr<const AttrValue>> attrs;
  if (batch_size_value > 0) {
    ff::udf::AttrValue batch_size;
    batch_size.set_i(batch_size_value);
    auto batch_size_attr = std::make_shared<AttrValueImpl>(batch_size);
    attrs["batch_size"] = batch_size_attr;
  }
  if (timeout_value >= -1) {
    ff::udf::AttrValue timeout;
    timeout.set_i(timeout_value);
    auto timeout_attr = std::make_shared<AttrValueImpl>(timeout);
    attrs["timeout"] = timeout_attr;
  }
  if (has_padding_attr) {
    ff::udf::AttrValue padding;
    padding.set_b(padding_flag);
    auto padding_attr = std::make_shared<AttrValueImpl>(padding);
    attrs["padding"] = padding_attr;
  }
  if (slide_stride_value >= 0) {
    ff::udf::AttrValue slide_stride;
    slide_stride.set_i(slide_stride_value);
    auto slide_stride_attr = std::make_shared<AttrValueImpl>(slide_stride);
    attrs["slide_stride"] = slide_stride_attr;
  }
  return attrs;
}

FlowFuncContext CreateFlowFuncContext(std::map<std::string, std::shared_ptr<const AttrValue>> &attrs) {
  uint32_t input_num = 1;
  uint32_t output_num = 1;
  std::shared_ptr<FlowFuncParams> flow_func_param(new (std::nothrow)
                                                      FlowFuncParams("CountBatch", input_num, output_num, 0, 0));
  flow_func_param->SetAttrMap(attrs);
  std::shared_ptr<FlowFuncRunContext> flow_func_run_context(new (std::nothrow)
                                                                FlowFuncRunContext(0, flow_func_param, nullptr));
  FlowFuncContext flow_func_context(flow_func_param, flow_func_run_context);
  return flow_func_context;
}

TEST_F(COUNT_BATCH_FLOW_FUNC_UTEST, init_success) {
  int64_t batch_size = 5;
  int64_t timeout = 0;
  int64_t slide_stride = 2;
  std::map<std::string, std::shared_ptr<const AttrValue>> attrs = CreateAttrs(batch_size, 0, false, slide_stride);
  FlowFuncContext flow_func_context = CreateFlowFuncContext(attrs);
  CountBatchFlowFunc count_batch;
  count_batch.SetContext(&flow_func_context);
  EXPECT_EQ(count_batch.Init(), FLOW_FUNC_SUCCESS);
  EXPECT_EQ(count_batch.batch_size_, batch_size);
  EXPECT_EQ(count_batch.timeout_, timeout);
  EXPECT_EQ(count_batch.padding_, false);
  EXPECT_EQ(count_batch.slide_stride_, slide_stride);
}

TEST_F(COUNT_BATCH_FLOW_FUNC_UTEST, get_attr_failed) {
  CountBatchFlowFunc count_batch;
  // empty attr
  std::map<std::string, std::shared_ptr<const AttrValue>> empty_attr;
  FlowFuncContext flow_func_context = CreateFlowFuncContext(empty_attr);
  count_batch.SetContext(&flow_func_context);
  EXPECT_EQ(count_batch.Init(), FLOW_FUNC_ERR_ATTR_NOT_EXITS);
  // no batch size attr
  std::map<std::string, std::shared_ptr<const AttrValue>> attrs1 = CreateAttrs(0, -1, false, 2);
  FlowFuncContext flow_func_context1 = CreateFlowFuncContext(attrs1);
  count_batch.SetContext(&flow_func_context1);
  EXPECT_NE(count_batch.Init(), FLOW_FUNC_SUCCESS);
  // no timeout attr
  std::map<std::string, std::shared_ptr<const AttrValue>> attrs2 = CreateAttrs(5, -2, false, 2);
  FlowFuncContext flow_func_context2 = CreateFlowFuncContext(attrs2);
  count_batch.SetContext(&flow_func_context2);
  EXPECT_NE(count_batch.Init(), FLOW_FUNC_SUCCESS);
  // invalid timeout
  std::map<std::string, std::shared_ptr<const AttrValue>> attrs3 = CreateAttrs(5, -1, false, 2);
  FlowFuncContext flow_func_context3 = CreateFlowFuncContext(attrs3);
  count_batch.SetContext(&flow_func_context3);
  EXPECT_NE(count_batch.Init(), FLOW_FUNC_SUCCESS);
  // no padding attr
  std::map<std::string, std::shared_ptr<const AttrValue>> attrs4 = CreateAttrs(5, 3, false, 2, false);
  FlowFuncContext flow_func_context4 = CreateFlowFuncContext(attrs4);
  count_batch.SetContext(&flow_func_context4);
  EXPECT_NE(count_batch.Init(), FLOW_FUNC_SUCCESS);
  // no slide_stride attr
  std::map<std::string, std::shared_ptr<const AttrValue>> attrs5 = CreateAttrs(5, 3, false, -1);
  FlowFuncContext flow_func_context5 = CreateFlowFuncContext(attrs5);
  count_batch.SetContext(&flow_func_context5);
  EXPECT_NE(count_batch.Init(), FLOW_FUNC_SUCCESS);
}

TEST_F(COUNT_BATCH_FLOW_FUNC_UTEST, proc_input_empty) {
  std::map<std::string, std::shared_ptr<const AttrValue>> attrs = CreateAttrs(5, 0, false, 2);
  FlowFuncContext flow_func_context = CreateFlowFuncContext(attrs);
  CountBatchFlowFunc count_batch;
  count_batch.SetContext(&flow_func_context);
  EXPECT_EQ(count_batch.Init(), FLOW_FUNC_SUCCESS);
  std::vector<std::shared_ptr<FlowMsg>> input_msgs;
  EXPECT_EQ(count_batch.Proc(input_msgs), FLOW_FUNC_SUCCESS);
  input_msgs.resize(1);
  EXPECT_EQ(count_batch.Proc(input_msgs), FLOW_FUNC_SUCCESS);
  TensorDataType data_type = TensorDataType::DT_UINT32;
  vector<int64_t> shape = {3, 2};
  MbufHead mbuf_head;
  std::shared_ptr<MbufFlowMsg> mbuf_flow_msg = MbufFlowMsg::AllocTensorMsg({0}, data_type, 0, mbuf_head, true);
  std::vector<std::shared_ptr<FlowMsg>> input_msgs2;
  input_msgs2.emplace_back(mbuf_flow_msg);
  EXPECT_EQ(count_batch.Proc(input_msgs2), FLOW_FUNC_SUCCESS);
}

TEST_F(COUNT_BATCH_FLOW_FUNC_UTEST, proc_input_error_code) {
  std::map<std::string, std::shared_ptr<const AttrValue>> attrs = CreateAttrs(5, 0, false, 2);
  FlowFuncContext flow_func_context = CreateFlowFuncContext(attrs);
  CountBatchFlowFunc count_batch;
  count_batch.SetContext(&flow_func_context);
  EXPECT_EQ(count_batch.Init(), FLOW_FUNC_SUCCESS);
  std::vector<std::shared_ptr<FlowMsg>> input_msgs;
  std::shared_ptr<MbufFlowMsg> input_msg = std::make_shared<MbufFlowMsg>(nullptr);
  MbufHeadMsg head_msg;
  head_msg.ret_code = -1;
  input_msg->head_msg_ = &head_msg;
  input_msgs.emplace_back(input_msg);
  EXPECT_EQ(count_batch.Proc(input_msgs), FLOW_FUNC_SUCCESS);
}

TEST_F(COUNT_BATCH_FLOW_FUNC_UTEST, proc_input_tensor_shape_error) {
  std::map<std::string, std::shared_ptr<const AttrValue>> attrs = CreateAttrs(5, 0, false, 0);
  FlowFuncContext flow_func_context = CreateFlowFuncContext(attrs);
  CountBatchFlowFunc count_batch;
  count_batch.SetContext(&flow_func_context);
  EXPECT_EQ(count_batch.Init(), FLOW_FUNC_SUCCESS);
  std::vector<std::shared_ptr<FlowMsg>> input_msgs;
  TensorDataType data_type = TensorDataType::DT_UINT32;
  vector<int64_t> shape = {3, 2};
  MbufHead mbuf_head;
  std::shared_ptr<MbufFlowMsg> mbuf_flow_msg1 = MbufFlowMsg::AllocTensorMsg(shape, data_type, 0, mbuf_head);
  input_msgs.emplace_back(mbuf_flow_msg1);
  EXPECT_EQ(count_batch.Proc(input_msgs), FLOW_FUNC_SUCCESS);
  std::shared_ptr<MbufFlowMsg> mbuf_flow_msg2 = MbufFlowMsg::AllocTensorMsg({2, 3, 4}, data_type, 0, mbuf_head);
  input_msgs.clear();
  input_msgs.emplace_back(mbuf_flow_msg2);
  EXPECT_EQ(count_batch.Proc(input_msgs), FLOW_FUNC_SUCCESS);
  input_msgs.clear();
  input_msgs.emplace_back(mbuf_flow_msg1);
  EXPECT_EQ(count_batch.Proc(input_msgs), FLOW_FUNC_SUCCESS);
  std::shared_ptr<MbufFlowMsg> mbuf_flow_msg3 = MbufFlowMsg::AllocTensorMsg({2, 3}, data_type, 0, mbuf_head);
  input_msgs.clear();
  input_msgs.emplace_back(mbuf_flow_msg3);
  EXPECT_EQ(count_batch.Proc(input_msgs), FLOW_FUNC_SUCCESS);
}

TEST_F(COUNT_BATCH_FLOW_FUNC_UTEST, proc_input_tensor_dType_error) {
  std::map<std::string, std::shared_ptr<const AttrValue>> attrs = CreateAttrs(5, 0, false, 0);
  FlowFuncContext flow_func_context = CreateFlowFuncContext(attrs);
  CountBatchFlowFunc count_batch;
  count_batch.SetContext(&flow_func_context);
  EXPECT_EQ(count_batch.Init(), FLOW_FUNC_SUCCESS);
  std::vector<std::shared_ptr<FlowMsg>> input_msgs;
  TensorDataType data_type = TensorDataType::DT_UINT32;
  TensorDataType data_type2 = TensorDataType::DT_FLOAT;
  vector<int64_t> shape = {3, 2};
  MbufHead mbuf_head;
  std::shared_ptr<MbufFlowMsg> mbuf_flow_msg1 = MbufFlowMsg::AllocTensorMsg(shape, data_type, 0, mbuf_head);
  input_msgs.emplace_back(mbuf_flow_msg1);
  EXPECT_EQ(count_batch.Proc(input_msgs), FLOW_FUNC_SUCCESS);
  std::shared_ptr<MbufFlowMsg> mbuf_flow_msg2 = MbufFlowMsg::AllocTensorMsg({3, 2}, data_type2, 0, mbuf_head);
  input_msgs.clear();
  input_msgs.emplace_back(mbuf_flow_msg2);
  EXPECT_EQ(count_batch.Proc(input_msgs), FLOW_FUNC_SUCCESS);
}

TEST_F(COUNT_BATCH_FLOW_FUNC_UTEST, proc_input_size_error) {
  std::map<std::string, std::shared_ptr<const AttrValue>> attrs = CreateAttrs(5, 0, false, 0);
  FlowFuncContext flow_func_context = CreateFlowFuncContext(attrs);
  CountBatchFlowFunc count_batch;
  count_batch.SetContext(&flow_func_context);
  EXPECT_EQ(count_batch.Init(), FLOW_FUNC_SUCCESS);
  std::vector<std::shared_ptr<FlowMsg>> input_msgs;
  TensorDataType data_type = TensorDataType::DT_UINT32;
  vector<int64_t> shape = {3, 2};
  MbufHead mbuf_head;
  std::shared_ptr<MbufFlowMsg> mbuf_flow_msg = MbufFlowMsg::AllocTensorMsg(shape, data_type, 0, mbuf_head);
  input_msgs.emplace_back(mbuf_flow_msg);
  EXPECT_EQ(count_batch.Proc(input_msgs), FLOW_FUNC_SUCCESS);
  input_msgs.emplace_back(mbuf_flow_msg);
  input_msgs.emplace_back(mbuf_flow_msg);
  EXPECT_EQ(count_batch.Proc(input_msgs), FLOW_FUNC_SUCCESS);
}

TEST_F(COUNT_BATCH_FLOW_FUNC_UTEST, proc_only_batchsize) {
  int64_t batch_size = 3;
  int64_t timeout = 0;
  int64_t slide_stride = 0;
  std::map<std::string, std::shared_ptr<const AttrValue>> attrs = CreateAttrs(batch_size, timeout, false, slide_stride);
  FlowFuncContext flow_func_context = CreateFlowFuncContext(attrs);
  CountBatchFlowFunc count_batch;
  count_batch.SetContext(&flow_func_context);
  EXPECT_EQ(count_batch.Init(), FLOW_FUNC_SUCCESS);
  std::vector<std::shared_ptr<FlowMsg>> input_msgs;
  TensorDataType data_type = TensorDataType::DT_UINT32;
  vector<int64_t> shape = {3, 2};
  MbufHead mbuf_head;
  std::shared_ptr<MbufFlowMsg> mbuf_flow_msg = MbufFlowMsg::AllocTensorMsg(shape, data_type, 0, mbuf_head);
  input_msgs.emplace_back(mbuf_flow_msg);

  std::vector<int64_t> expert_shape = {batch_size, 3, 2};
  MOCKER_CPP_VIRTUAL(flow_func_context, &FlowFuncContext::SetOutput,
                     int32_t(FlowFuncContext::*)(uint32_t, std::shared_ptr<FlowMsg>))
      .stubs()
      .will(returnValue(FLOW_FUNC_SUCCESS));
  for (uint32_t i = 0; i < batch_size; i++) {
    EXPECT_EQ(count_batch.Proc(input_msgs), FLOW_FUNC_SUCCESS);
    if (i != batch_size - 1) {
      EXPECT_EQ(count_batch.batch_flow_msg_[0].size(), i + 1);
    } else {
      EXPECT_EQ(count_batch.batch_flow_msg_[0].size(), 0);
    }
  }
}

TEST_F(COUNT_BATCH_FLOW_FUNC_UTEST, proc_batchsize_slide_stride) {
  int64_t batch_size = 3;
  int64_t timeout = 0;
  int64_t slide_stride = 1;
  vector<int64_t> shape = {3, 2};
  MbufHead mbuf_head;
  std::map<std::string, std::shared_ptr<const AttrValue>> attrs = CreateAttrs(batch_size, timeout, false, slide_stride);
  FlowFuncContext flow_func_context = CreateFlowFuncContext(attrs);
  CountBatchFlowFunc count_batch;
  count_batch.SetContext(&flow_func_context);
  EXPECT_EQ(count_batch.Init(), FLOW_FUNC_SUCCESS);
  std::vector<std::shared_ptr<FlowMsg>> input_msgs;
  TensorDataType data_type = TensorDataType::DT_UINT32;
  std::shared_ptr<MbufFlowMsg> mbuf_flow_msg = MbufFlowMsg::AllocTensorMsg(shape, data_type, 0, mbuf_head);
  input_msgs.emplace_back(mbuf_flow_msg);
  std::vector<int64_t> expert_shape = {batch_size, 3, 2};
  MOCKER_CPP_VIRTUAL(flow_func_context, &FlowFuncContext::SetOutput,
                     int32_t(FlowFuncContext::*)(uint32_t, std::shared_ptr<FlowMsg>))
      .stubs()
      .will(returnValue(FLOW_FUNC_SUCCESS));
  for (uint32_t i = 0; i < batch_size * 2; ++i) {
    EXPECT_EQ(count_batch.Proc(input_msgs), FLOW_FUNC_SUCCESS);
    if (i < batch_size - 1) {
      EXPECT_EQ(count_batch.batch_flow_msg_[0].size(), i + 1);
    } else {
      EXPECT_EQ(count_batch.batch_flow_msg_[0].size(), 2);
    }
  }
}

TEST_F(COUNT_BATCH_FLOW_FUNC_UTEST, set_output_failed) {
  int64_t batch_size = 5;
  int64_t timeout = 100;
  int64_t slide_stride = 2;
  vector<int64_t> shape = {3, 2};
  MbufHead mbuf_head;
  std::map<std::string, std::shared_ptr<const AttrValue>> attrs = CreateAttrs(batch_size, timeout, true, slide_stride);
  FlowFuncContext flow_func_context = CreateFlowFuncContext(attrs);
  CountBatchFlowFunc count_batch;
  count_batch.SetContext(&flow_func_context);
  EXPECT_EQ(count_batch.Init(), FLOW_FUNC_SUCCESS);
  std::vector<std::shared_ptr<FlowMsg>> input_msgs;
  TensorDataType data_type = TensorDataType::DT_UINT32;
  std::shared_ptr<MbufFlowMsg> mbuf_flow_msg = MbufFlowMsg::AllocTensorMsg(shape, data_type, 0, mbuf_head);
  input_msgs.emplace_back(mbuf_flow_msg);

  MOCKER_CPP_VIRTUAL(flow_func_context, &FlowFuncContext::SetOutput,
                     int32_t(FlowFuncContext::*)(uint32_t, std::shared_ptr<FlowMsg>))
      .stubs()
      .will(returnValue(FLOW_FUNC_FAILED));
  for (uint32_t i = 0; i < batch_size + 2; ++i) {
    EXPECT_EQ(count_batch.Proc(input_msgs), FLOW_FUNC_SUCCESS);
  }
  count_batch.start_time_ = 0UL;
  TimerInfo *timer_info = (TimerInfo *)(count_batch.timer_handle_);
  timer_info->timer_callback();
  usleep(200 * 1000UL);
  timer_info->timer_callback();
  usleep(200 * 1000UL);
  timer_info->timer_callback();
}

TEST_F(COUNT_BATCH_FLOW_FUNC_UTEST, proc_batchsize_timeout) {
  int64_t batch_size = 5;
  int64_t timeout = 100;
  int64_t slide_stride = 2;
  vector<int64_t> shape = {3, 2};
  MbufHead mbuf_head;
  std::map<std::string, std::shared_ptr<const AttrValue>> attrs = CreateAttrs(batch_size, timeout, true, slide_stride);
  FlowFuncContext flow_func_context = CreateFlowFuncContext(attrs);
  CountBatchFlowFunc count_batch;
  count_batch.SetContext(&flow_func_context);
  EXPECT_EQ(count_batch.Init(), FLOW_FUNC_SUCCESS);
  std::vector<std::shared_ptr<FlowMsg>> input_msgs;
  TensorDataType data_type = TensorDataType::DT_UINT32;
  std::shared_ptr<MbufFlowMsg> mbuf_flow_msg = MbufFlowMsg::AllocTensorMsg(shape, data_type, 0, mbuf_head);
  input_msgs.emplace_back(mbuf_flow_msg);

  MOCKER_CPP_VIRTUAL(flow_func_context, &FlowFuncContext::SetOutput,
                     int32_t(FlowFuncContext::*)(uint32_t, std::shared_ptr<FlowMsg>))
      .stubs()
      .will(returnValue(FLOW_FUNC_SUCCESS));
  for (uint32_t i = 0; i < batch_size + 2; ++i) {
    EXPECT_EQ(count_batch.Proc(input_msgs), FLOW_FUNC_SUCCESS);
  }
  count_batch.start_time_ = 0UL;
  TimerInfo *timer_info = (TimerInfo *)(count_batch.timer_handle_);
  timer_info->timer_callback();
  usleep(200 * 1000UL);
  timer_info->timer_callback();
  usleep(200 * 1000UL);
  timer_info->timer_callback();
}
}  // namespace FlowFunc