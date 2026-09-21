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
#define private public

#include "flow_func/flow_func_run_context.h"

#undef private
#include "mockcpp/mockcpp.hpp"
#include "securec.h"
#include "model/attr_value_impl.h"
#include "dlog_pub.h"

namespace FlowFunc {
namespace {
struct MbufImpl {
  uint32_t mbuf_size;
  uint8_t reserve_head[256];
  RuntimeTensorDesc tensor_desc;
  uint8_t data[3072];
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
}  // namespace
class FlowFuncRunContextUTest : public testing::Test {
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

TEST_F(FlowFuncRunContextUTest, SetInputMbufHead_failed) {
  std::shared_ptr<FlowFuncParams> flow_func_param(new (std::nothrow) FlowFuncParams("pp1", 1, 1, 0, 0));
  ASSERT_NE(flow_func_param, nullptr);
  FlowFuncRunContext flow_func_run_context{0, flow_func_param, nullptr};
  MbufInfo mbuf_info;
  EXPECT_EQ(flow_func_run_context.SetInputMbufHead(mbuf_info), FLOW_FUNC_ERR_PARAM_INVALID);
  uint8_t head[257];
  mbuf_info.head_buf = head;
  mbuf_info.head_buf_len = 257;
  EXPECT_EQ(flow_func_run_context.SetInputMbufHead(mbuf_info), FLOW_FUNC_ERR_PARAM_INVALID);
  mbuf_info.head_buf_len = 256;
  MOCKER(memcpy_s).stubs().will(returnValue(-1));
  EXPECT_EQ(flow_func_run_context.SetInputMbufHead(mbuf_info), FLOW_FUNC_FAILED);
}

TEST_F(FlowFuncRunContextUTest, GetUserData_success) {
  std::shared_ptr<FlowFuncParams> flow_func_param(new (std::nothrow) FlowFuncParams("pp", 1, 1, 0, 0));
  ASSERT_NE(flow_func_param, nullptr);
  FlowFuncRunContext flow_func_run_context{0, flow_func_param, nullptr};
  MbufInfo mbuf_info;
  int64_t head[32] = {};
  int64_t user_data = 8;
  head[0] = user_data;
  head[1] = user_data;
  mbuf_info.head_buf = head;
  mbuf_info.head_buf_len = 256;
  EXPECT_EQ(flow_func_run_context.SetInputMbufHead(mbuf_info), FLOW_FUNC_SUCCESS);
  int64_t b = 0;
  EXPECT_EQ(flow_func_run_context.GetUserData(&b, sizeof(int64_t)), FLOW_FUNC_SUCCESS);
  EXPECT_EQ(b, user_data);
  b = 0;
  EXPECT_EQ(flow_func_run_context.GetUserData(&b, sizeof(int64_t), sizeof(int64_t)), FLOW_FUNC_SUCCESS);
  EXPECT_EQ(b, user_data);
}

TEST_F(FlowFuncRunContextUTest, GetUserData_failed) {
  std::shared_ptr<FlowFuncParams> flow_func_param(new (std::nothrow) FlowFuncParams("pp", 1, 1, 0, 0));
  ASSERT_NE(flow_func_param, nullptr);
  FlowFuncRunContext flow_func_run_context{0, flow_func_param, nullptr};
  int32_t b = 0;
  EXPECT_EQ(flow_func_run_context.GetUserData(&b, sizeof(int32_t)), FLOW_FUNC_FAILED);
  MbufInfo mbuf_info;
  uint8_t head[256] = {};
  mbuf_info.head_buf = head;
  mbuf_info.head_buf_len = 1;
  EXPECT_EQ(flow_func_run_context.SetInputMbufHead(mbuf_info), FLOW_FUNC_ERR_PARAM_INVALID);

  mbuf_info.head_buf_len = 256;
  EXPECT_EQ(flow_func_run_context.SetInputMbufHead(mbuf_info), FLOW_FUNC_SUCCESS);
  EXPECT_EQ(flow_func_run_context.GetUserData(nullptr, 0), FLOW_FUNC_ERR_PARAM_INVALID);
  EXPECT_EQ(flow_func_run_context.GetUserData(&b, 0), FLOW_FUNC_ERR_PARAM_INVALID);
  EXPECT_EQ(flow_func_run_context.GetUserData(&b, sizeof(int32_t), 64), FLOW_FUNC_ERR_PARAM_INVALID);

  MOCKER(memcpy_s).stubs().will(returnValue(-1));
  EXPECT_EQ(flow_func_run_context.GetUserData(&b, sizeof(int32_t)), FLOW_FUNC_FAILED);
}

TEST_F(FlowFuncRunContextUTest, SetOutPutWithBalanceOptionsScatter) {
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
  FlowFuncRunContext flow_func_run_context(0, flow_func_param, callback_func);
  auto msg = flow_func_run_context.AllocTensorMsg({2, 2}, TensorDataType::DT_INT32);

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

  int32_t ret = flow_func_run_context.SetOutput(0, msg, options);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  auto mbuf_msg = std::dynamic_pointer_cast<MbufFlowMsg>(out_msg);
  const auto &mbuf_info = mbuf_msg->GetMbufInfo();
  auto head_buf = static_cast<const char *>(mbuf_info.head_buf) + mbuf_info.head_buf_len - sizeof(MbufHeadMsg);
  const auto *head_msg = reinterpret_cast<const MbufHeadMsg *>(head_buf);
  EXPECT_EQ(head_msg->data_label, pos_row * weight.colNum + pos_col + 1);
  EXPECT_EQ(head_msg->route_label, pos_row * weight.colNum + pos_col + 1);
}

TEST_F(FlowFuncRunContextUTest, SetOutPutWithBalanceOptionsGatherRowAffinity) {
  std::shared_ptr<FlowFuncParams> flow_func_param(new (std::nothrow) FlowFuncParams("pp", 1, 1, 0, 0));
  ASSERT_NE(flow_func_param, nullptr);
  ff::udf::AttrValue balance_attr;
  balance_attr.set_b(true);
  auto attr_impl = std::make_shared<AttrValueImpl>(balance_attr);
  std::map<std::string, std::shared_ptr<const AttrValue>> attr_map;
  attr_map.emplace("_balance_gather", attr_impl);
  flow_func_param->SetAttrMap(attr_map);
  flow_func_param->Init();
  std::shared_ptr<FlowMsg> out_msg;
  auto callback_func = [&out_msg](uint32_t out_idx, const std::shared_ptr<FlowMsg> &flow_msg) {
    out_msg = flow_msg;
    return FLOW_FUNC_SUCCESS;
  };
  FlowFuncRunContext flow_func_run_context(0, flow_func_param, callback_func);
  auto msg = flow_func_run_context.AllocTensorMsg({2, 2}, TensorDataType::DT_INT32);

  OutOptions options;
  auto balance_config = options.MutableBalanceConfig();
  balance_config->SetAffinityPolicy(AffinityPolicy::ROW_AFFINITY);
  int32_t pos_row = 1;
  int32_t pos_col = 2;
  balance_config->SetDataPos({{pos_row, pos_col}});
  BalanceWeight weight;
  weight.rowNum = 3;
  weight.colNum = 4;
  balance_config->SetBalanceWeight(weight);

  int32_t ret = flow_func_run_context.SetOutput(0, msg, options);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  auto mbuf_msg = std::dynamic_pointer_cast<MbufFlowMsg>(out_msg);
  const auto &mbuf_info = mbuf_msg->GetMbufInfo();
  auto head_buf = static_cast<const char *>(mbuf_info.head_buf) + mbuf_info.head_buf_len - sizeof(MbufHeadMsg);
  const auto *head_msg = reinterpret_cast<const MbufHeadMsg *>(head_buf);
  EXPECT_EQ(head_msg->data_label, pos_row * weight.colNum + pos_col + 1);
  EXPECT_EQ(head_msg->route_label, pos_row + 1);
}

TEST_F(FlowFuncRunContextUTest, SetOutPutWithBalanceOptionsGatherColAffinity) {
  std::shared_ptr<FlowFuncParams> flow_func_param(new (std::nothrow) FlowFuncParams("pp", 1, 1, 0, 0));
  ASSERT_NE(flow_func_param, nullptr);
  ff::udf::AttrValue balance_attr;
  balance_attr.set_b(true);
  auto attr_impl = std::make_shared<AttrValueImpl>(balance_attr);
  std::map<std::string, std::shared_ptr<const AttrValue>> attr_map;
  attr_map.emplace("_balance_gather", attr_impl);
  flow_func_param->SetAttrMap(attr_map);
  flow_func_param->Init();
  std::shared_ptr<FlowMsg> out_msg;
  auto callback_func = [&out_msg](uint32_t out_idx, const std::shared_ptr<FlowMsg> &flow_msg) {
    out_msg = flow_msg;
    return FLOW_FUNC_SUCCESS;
  };
  FlowFuncRunContext flow_func_run_context(0, flow_func_param, callback_func);
  auto msg = flow_func_run_context.AllocTensorMsg({2, 2}, TensorDataType::DT_INT32);

  OutOptions options;
  auto balance_config = options.MutableBalanceConfig();
  balance_config->SetAffinityPolicy(AffinityPolicy::COL_AFFINITY);
  int32_t pos_row = 1;
  int32_t pos_col = 2;
  balance_config->SetDataPos({{pos_row, pos_col}});
  BalanceWeight weight;
  weight.rowNum = 3;
  weight.colNum = 4;
  balance_config->SetBalanceWeight(weight);

  int32_t ret = flow_func_run_context.SetOutput(0, msg, options);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  auto mbuf_msg = std::dynamic_pointer_cast<MbufFlowMsg>(out_msg);
  const auto &mbuf_info = mbuf_msg->GetMbufInfo();
  auto head_buf = static_cast<const char *>(mbuf_info.head_buf) + mbuf_info.head_buf_len - sizeof(MbufHeadMsg);
  const auto *head_msg = reinterpret_cast<const MbufHeadMsg *>(head_buf);
  EXPECT_EQ(head_msg->data_label, pos_row * weight.colNum + pos_col + 1);
  EXPECT_EQ(head_msg->route_label, pos_col + 1);
}

TEST_F(FlowFuncRunContextUTest, SetOutPutWithBalanceOptionsMisMatch) {
  std::shared_ptr<FlowFuncParams> flow_func_param(new (std::nothrow) FlowFuncParams("pp", 1, 1, 0, 0));
  ASSERT_NE(flow_func_param, nullptr);
  ff::udf::AttrValue balance_attr;
  balance_attr.set_b(true);
  auto attr_impl = std::make_shared<AttrValueImpl>(balance_attr);
  std::map<std::string, std::shared_ptr<const AttrValue>> attr_map;
  attr_map.emplace("_balance_gather", attr_impl);
  flow_func_param->SetAttrMap(attr_map);
  flow_func_param->Init();
  auto callback_func = [](uint32_t out_idx, const std::shared_ptr<FlowMsg> &flow_msg) { return FLOW_FUNC_SUCCESS; };
  FlowFuncRunContext flow_func_run_context(0, flow_func_param, callback_func);
  auto msg = flow_func_run_context.AllocTensorMsg({2, 2}, TensorDataType::DT_INT32);

  OutOptions options;
  auto balance_config = options.MutableBalanceConfig();
  // gather not support NO_AFFINITY
  balance_config->SetAffinityPolicy(AffinityPolicy::NO_AFFINITY);
  int32_t pos_row = 1;
  int32_t pos_col = 2;
  balance_config->SetDataPos({{pos_row, pos_col}});
  BalanceWeight weight;
  weight.rowNum = 3;
  weight.colNum = 4;
  balance_config->SetBalanceWeight(weight);

  int32_t ret = flow_func_run_context.SetOutput(0, msg, options);
  EXPECT_NE(ret, FLOW_FUNC_SUCCESS);
}

TEST_F(FlowFuncRunContextUTest, AllocTensorListMsgSucc) {
  std::shared_ptr<FlowFuncParams> flow_func_param(new (std::nothrow) FlowFuncParams("pp", 1, 1, 0, 0));
  ASSERT_NE(flow_func_param, nullptr);
  ff::udf::AttrValue balance_attr;
  std::map<std::string, std::shared_ptr<const AttrValue>> attr_map;
  flow_func_param->SetAttrMap(attr_map);
  flow_func_param->Init();
  auto callback_func = [](uint32_t out_idx, const std::shared_ptr<FlowMsg> &flow_msg) { return FLOW_FUNC_SUCCESS; };
  FlowFuncRunContext flow_func_run_context(0, flow_func_param, callback_func);
  auto msg =
      flow_func_run_context.AllocTensorListMsg({{2, 2}, {4, 4}}, {TensorDataType::DT_INT32, TensorDataType::DT_INT32});
  auto list = msg->GetTensorList();
  EXPECT_EQ(list.size(), 2);
  EXPECT_EQ(list[0]->GetDataBufferSize(), 512);
  EXPECT_EQ(list[1]->GetDataBufferSize(), 512);
  auto *ptr1 = static_cast<uint8_t *>(list[0]->GetData());
  auto *ptr2 = static_cast<uint8_t *>(list[1]->GetData());
  EXPECT_EQ(ptr2 - ptr1, sizeof(RuntimeTensorDesc) + 512);
}

TEST_F(FlowFuncRunContextUTest, AllocTensorListMsgAlignInvalid) {
  std::shared_ptr<FlowFuncParams> flow_func_param(new (std::nothrow) FlowFuncParams("pp", 1, 1, 0, 0));
  ASSERT_NE(flow_func_param, nullptr);
  ff::udf::AttrValue balance_attr;
  std::map<std::string, std::shared_ptr<const AttrValue>> attr_map;
  flow_func_param->SetAttrMap(attr_map);
  flow_func_param->Init();
  auto callback_func = [](uint32_t out_idx, const std::shared_ptr<FlowMsg> &flow_msg) { return FLOW_FUNC_SUCCESS; };
  FlowFuncRunContext flow_func_run_context(0, flow_func_param, callback_func);
  auto msg = flow_func_run_context.AllocTensorListMsg({{2, 2}, {4, 4}},
                                                      {TensorDataType::DT_INT32, TensorDataType::DT_INT32}, 16);
  EXPECT_EQ(msg, nullptr);
  msg = flow_func_run_context.AllocTensorListMsg({{2, 2}, {4, 4}}, {TensorDataType::DT_INT32, TensorDataType::DT_INT32},
                                                 65);
  EXPECT_EQ(msg, nullptr);
}

TEST_F(FlowFuncRunContextUTest, SetMultiOutput) {
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
  FlowFuncRunContext flow_func_run_context(0, flow_func_param, callback_func);
  auto msg1 = flow_func_run_context.AllocTensorMsg({2, 2}, TensorDataType::DT_INT32);
  auto msg2 = flow_func_run_context.AllocTensorMsg({2, 2}, TensorDataType::DT_INT32);

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

  int32_t ret = flow_func_run_context.SetMultiOutputs(0, {msg1, msg2}, options);
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

TEST_F(FlowFuncRunContextUTest, RaiseExceptionSucc) {
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
  FlowFuncRunContext flow_func_run_context(0, flow_func_param, nullptr, 0);
  flow_func_run_context.SetInputMbufHead(info);
  MOCKER(halEschedSubmitEvent).stubs().will(returnValue(DRV_ERROR_NONE));
  flow_func_run_context.RaiseException(1000, 2);
  EXPECT_EQ(flow_func_run_context.trans_id_to_exception_raised_.size(), 1);
  EXPECT_EQ(flow_func_run_context.trans_id_to_exception_raised_.cbegin()->first, 100);
  EXPECT_EQ(flow_func_run_context.trans_id_to_exception_raised_.cbegin()->second.trans_id, 100);
  EXPECT_EQ(flow_func_run_context.trans_id_to_exception_raised_.cbegin()->second.user_context_id, 2);
  EXPECT_EQ(flow_func_run_context.trans_id_to_exception_raised_.cbegin()->second.exp_code, 1000);
  // 同一个transid第二次抛出异常不会生效
  flow_func_run_context.RaiseException(1001, 3);
  EXPECT_EQ(flow_func_run_context.trans_id_to_exception_raised_.size(), 1);
  EXPECT_EQ(flow_func_run_context.trans_id_to_exception_raised_.cbegin()->first, 100);
  EXPECT_EQ(flow_func_run_context.trans_id_to_exception_raised_.cbegin()->second.trans_id, 100);
  EXPECT_EQ(flow_func_run_context.trans_id_to_exception_raised_.cbegin()->second.user_context_id, 2);
  EXPECT_EQ(flow_func_run_context.trans_id_to_exception_raised_.cbegin()->second.exp_code, 1000);

  // transid不同可以继续抛异常
  head_msg->trans_id = 101;
  flow_func_run_context.SetInputMbufHead(info);
  flow_func_run_context.RaiseException(1001, 3);
  EXPECT_EQ(flow_func_run_context.trans_id_to_exception_raised_.size(), 2);
  EXPECT_EQ(flow_func_run_context.trans_id_to_exception_raised_.cbegin()->first, 100);
  EXPECT_EQ(flow_func_run_context.trans_id_to_exception_raised_.cbegin()->second.trans_id, 100);
  EXPECT_EQ(flow_func_run_context.trans_id_to_exception_raised_.cbegin()->second.user_context_id, 2);
  EXPECT_EQ(flow_func_run_context.trans_id_to_exception_raised_.cbegin()->second.exp_code, 1000);
  EXPECT_EQ(flow_func_run_context.trans_id_to_exception_raised_.rbegin()->first, 101);
  EXPECT_EQ(flow_func_run_context.trans_id_to_exception_raised_.rbegin()->second.trans_id, 101);
  EXPECT_EQ(flow_func_run_context.trans_id_to_exception_raised_.rbegin()->second.user_context_id, 3);
  EXPECT_EQ(flow_func_run_context.trans_id_to_exception_raised_.rbegin()->second.exp_code, 1001);
}

TEST_F(FlowFuncRunContextUTest, RaiseExceptionFailed) {
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
  FlowFuncRunContext flow_func_run_context(0, flow_func_param, nullptr, 0);
  flow_func_run_context.SetInputMbufHead(info);
  MOCKER(halEschedSubmitEvent).stubs().will(returnValue(100));
  flow_func_run_context.RaiseException(1000, 2);
  EXPECT_EQ(flow_func_run_context.trans_id_to_exception_raised_.size(), 1);
}

TEST_F(FlowFuncRunContextUTest, GetExceptionByTransIdFailed) {
  std::vector<uint32_t> input_queue_ids = {1, 2};
  std::vector<uint32_t> output_queue_ids = {3, 4};
  std::shared_ptr<FlowFuncParams> flow_func_param(
      new (std::nothrow) FlowFuncParams("pp1", input_queue_ids.size(), output_queue_ids.size(), 0, 0));
  FlowFuncRunContext flow_func_run_context(0, flow_func_param, nullptr, 0);
  UdfExceptionInfo info{};
  EXPECT_EQ(flow_func_run_context.GetExceptionByTransId(0, info), FLOW_FUNC_FAILED);
}

TEST_F(FlowFuncRunContextUTest, GetExceptionSucc) {
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
  FlowFuncRunContext flow_func_run_context(0, flow_func_param, nullptr, 0);
  UdfExceptionInfo exp_info{};
  exp_info.exp_code = -1;
  exp_info.trans_id = 1000;
  exp_info.user_context_id = 1001;
  flow_func_run_context.RecordException(exp_info);
  int32_t exp_code = 0;
  uint64_t user_context_id = 1;
  EXPECT_EQ(flow_func_run_context.GetException(exp_code, user_context_id), true);
  EXPECT_EQ(exp_code, exp_info.exp_code);
  EXPECT_EQ(user_context_id, exp_info.user_context_id);
}

TEST_F(FlowFuncRunContextUTest, SetMultiOutputParamCheck_num_mismatch) {
  std::shared_ptr<FlowFuncParams> flow_func_param(new (std::nothrow) FlowFuncParams("pp", 1, 1, 0, 0));
  ASSERT_NE(flow_func_param, nullptr);
  ff::udf::AttrValue balance_attr;
  balance_attr.set_b(true);
  auto attr_impl = std::make_shared<AttrValueImpl>(balance_attr);
  std::map<std::string, std::shared_ptr<const AttrValue>> attr_map;
  attr_map.emplace("_balance_gather", attr_impl);
  flow_func_param->SetAttrMap(attr_map);
  flow_func_param->Init();
  auto callback_func = [](uint32_t out_idx, const std::shared_ptr<FlowMsg> &flow_msg) { return FLOW_FUNC_SUCCESS; };
  FlowFuncRunContext flow_func_run_context(0, flow_func_param, callback_func);
  auto msg1 = flow_func_run_context.AllocTensorMsg({2, 2}, TensorDataType::DT_INT32);
  auto msg2 = flow_func_run_context.AllocTensorMsg({2, 2}, TensorDataType::DT_INT32);

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

  int32_t ret = flow_func_run_context.SetMultiOutputs(0, {msg1}, options);
  EXPECT_NE(ret, FLOW_FUNC_SUCCESS);
}

TEST_F(FlowFuncRunContextUTest, SetMultiOutputNoBalanceOptions) {
  std::shared_ptr<FlowFuncParams> flow_func_param(new (std::nothrow) FlowFuncParams("pp", 1, 1, 0, 0));
  ASSERT_NE(flow_func_param, nullptr);
  flow_func_param->Init();
  std::vector<std::shared_ptr<FlowMsg>> out_msgs;
  auto callback_func = [&out_msgs](uint32_t out_idx, const std::shared_ptr<FlowMsg> &flow_msg) {
    out_msgs.emplace_back(flow_msg);
    return FLOW_FUNC_SUCCESS;
  };
  FlowFuncRunContext flow_func_run_context(0, flow_func_param, callback_func);
  auto msg1 = flow_func_run_context.AllocTensorMsg({2, 2}, TensorDataType::DT_INT32);
  auto msg2 = flow_func_run_context.AllocTensorMsg({2, 2}, TensorDataType::DT_INT32);

  OutOptions options;
  int32_t ret = flow_func_run_context.SetMultiOutputs(0, {msg1, msg2}, options);
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
  EXPECT_EQ(head_msg1->data_label, 0);
  EXPECT_EQ(head_msg1->route_label, 0);
  EXPECT_EQ(head_msg2->data_label, 0);
  EXPECT_EQ(head_msg2->route_label, 0);
}

TEST_F(FlowFuncRunContextUTest, SetMultiOutputBalanceParamCheck) {
  std::shared_ptr<FlowFuncParams> flow_func_param(new (std::nothrow) FlowFuncParams("pp", 1, 1, 0, 0));
  ASSERT_NE(flow_func_param, nullptr);
  ff::udf::AttrValue balance_attr;
  balance_attr.set_b(true);
  auto attr_impl = std::make_shared<AttrValueImpl>(balance_attr);
  std::map<std::string, std::shared_ptr<const AttrValue>> attr_map;
  attr_map.emplace("_balance_scatter", attr_impl);
  flow_func_param->SetAttrMap(attr_map);
  flow_func_param->Init();
  auto callback_func = [](uint32_t out_idx, const std::shared_ptr<FlowMsg> &flow_msg) { return FLOW_FUNC_SUCCESS; };
  FlowFuncRunContext flow_func_run_context(0, flow_func_param, callback_func);
  auto msg1 = flow_func_run_context.AllocTensorMsg({2, 2}, TensorDataType::DT_INT32);
  auto msg2 = flow_func_run_context.AllocTensorMsg({2, 2}, TensorDataType::DT_INT32);

  OutOptions options;
  auto balance_config = options.MutableBalanceConfig();
  // scatter not support row affinity
  balance_config->SetAffinityPolicy(AffinityPolicy::ROW_AFFINITY);
  balance_config->SetDataPos({{1, 2}, {3, 4}});
  BalanceWeight weight;
  weight.rowNum = 5;
  weight.colNum = 5;
  balance_config->SetBalanceWeight(weight);
  int32_t ret = flow_func_run_context.SetMultiOutputs(0, {msg1, msg2}, options);
  EXPECT_NE(ret, FLOW_FUNC_SUCCESS);

  balance_config->SetAffinityPolicy(AffinityPolicy::NO_AFFINITY);
  balance_config->SetDataPos({{1, 2}, {3, 4}});
  weight.rowNum = 4;
  weight.colNum = 4;  // little than max pos
  balance_config->SetBalanceWeight(weight);
  ret = flow_func_run_context.SetMultiOutputs(0, {msg1, msg2}, options);
  EXPECT_NE(ret, FLOW_FUNC_SUCCESS);

  balance_config->SetDataPos({{-1, 0}, {1, 2}});
  ret = flow_func_run_context.SetMultiOutputs(0, {msg1, msg2}, options);
  EXPECT_NE(ret, FLOW_FUNC_SUCCESS);
  balance_config->SetDataPos({{1, 2}, {3, 4}});
  weight.rowNum = 10000;
  weight.colNum = 10000;  // too large
  balance_config->SetBalanceWeight(weight);
  ret = flow_func_run_context.SetMultiOutputs(0, {msg1, msg2}, options);
  EXPECT_NE(ret, FLOW_FUNC_SUCCESS);
  weight.rowNum = -1;  // too small
  weight.colNum = 1024;
  balance_config->SetBalanceWeight(weight);
  ret = flow_func_run_context.SetMultiOutputs(0, {msg1, msg2}, options);
  EXPECT_NE(ret, FLOW_FUNC_SUCCESS);
}

TEST_F(FlowFuncRunContextUTest, AllocTensorMsgWithAlign_RANGE_CHECK) {
  std::vector<uint32_t> input_queue_ids = {1, 2};
  std::vector<uint32_t> output_queue_ids = {3, 4};
  std::shared_ptr<FlowFuncParams> flow_func_param(
      new (std::nothrow) FlowFuncParams("pp1", input_queue_ids.size(), output_queue_ids.size(), 0, 0));
  std::shared_ptr<FlowFuncRunContext> flow_func_run_context(new (std::nothrow)
                                                                FlowFuncRunContext(0, flow_func_param, nullptr));
  EXPECT_EQ(flow_func_run_context->AllocTensorMsgWithAlign({1, 2, 3}, TensorDataType::DT_FLOAT, 0), nullptr);
  EXPECT_EQ(flow_func_run_context->AllocTensorMsgWithAlign({1, 2, 3}, TensorDataType::DT_FLOAT, 1), nullptr);
  EXPECT_EQ(flow_func_run_context->AllocTensorMsgWithAlign({1, 2, 3}, TensorDataType::DT_FLOAT, 16), nullptr);
  EXPECT_NE(flow_func_run_context->AllocTensorMsgWithAlign({1, 2, 3}, TensorDataType::DT_FLOAT, 32), nullptr);
  // can not be divide by 1024
  EXPECT_EQ(flow_func_run_context->AllocTensorMsgWithAlign({1, 2, 3}, TensorDataType::DT_FLOAT, 48), nullptr);
  EXPECT_NE(flow_func_run_context->AllocTensorMsgWithAlign({1, 2, 3}, TensorDataType::DT_FLOAT, 64), nullptr);
  EXPECT_NE(flow_func_run_context->AllocTensorMsgWithAlign({1, 2, 3}, TensorDataType::DT_FLOAT, 128), nullptr);
  EXPECT_NE(flow_func_run_context->AllocTensorMsgWithAlign({1, 2, 3}, TensorDataType::DT_FLOAT, 256), nullptr);
  EXPECT_NE(flow_func_run_context->AllocTensorMsgWithAlign({1, 2, 3}, TensorDataType::DT_FLOAT, 512), nullptr);
  EXPECT_NE(flow_func_run_context->AllocTensorMsgWithAlign({1, 2, 3}, TensorDataType::DT_FLOAT, 1024), nullptr);
  EXPECT_EQ(flow_func_run_context->AllocTensorMsgWithAlign({1, 2, 3}, TensorDataType::DT_FLOAT, 2048), nullptr);
}

TEST_F(FlowFuncRunContextUTest, AllocRawDataMsg) {
  std::vector<uint32_t> input_queue_ids = {1, 2};
  std::vector<uint32_t> output_queue_ids = {3, 4};
  std::shared_ptr<FlowFuncParams> flow_func_param(
      new (std::nothrow) FlowFuncParams("pp1", input_queue_ids.size(), output_queue_ids.size(), 0, 0));
  std::shared_ptr<FlowFuncRunContext> flow_func_run_context(new (std::nothrow)
                                                                FlowFuncRunContext(0, flow_func_param, nullptr));
  EXPECT_NE(flow_func_run_context->AllocRawDataMsg(10), nullptr);
}

TEST_F(FlowFuncRunContextUTest, ToFlowMsgSuccess) {
  auto tensor = FlowBufferFactory::AllocTensor({1, 2}, TensorDataType::DT_INT32);
  EXPECT_NE(tensor, nullptr);

  std::vector<uint32_t> input_queue_ids = {1, 2};
  std::vector<uint32_t> output_queue_ids = {3, 4};
  std::shared_ptr<FlowFuncParams> flow_func_param(
      new (std::nothrow) FlowFuncParams("pp1", input_queue_ids.size(), output_queue_ids.size(), 0, 0));
  std::shared_ptr<FlowFuncRunContext> flow_func_run_context(new (std::nothrow)
                                                                FlowFuncRunContext(0, flow_func_param, nullptr));
  auto tensor_msg = flow_func_run_context->ToFlowMsg(tensor);
  EXPECT_NE(tensor_msg, nullptr);
  auto tensor_ptr = tensor_msg->GetTensor();
  EXPECT_NE(tensor_ptr, nullptr);
  auto tensor_data = tensor_ptr->GetData();
  EXPECT_NE(tensor_data, nullptr);
  auto tensor_shape = tensor_ptr->GetShape();
  EXPECT_EQ(tensor_shape[0], 1);
  EXPECT_EQ(tensor_shape[1], 2);
}

TEST_F(FlowFuncRunContextUTest, ToFlowMsgFailed) {
  auto tensor = FlowBufferFactory::AllocTensor({1, 2}, TensorDataType::DT_INT32);
  EXPECT_NE(tensor, nullptr);

  MOCKER(halMbufGetPrivInfo).stubs().will(returnValue((int)DRV_ERROR_PARA_ERROR));

  std::vector<uint32_t> input_queue_ids = {1, 2};
  std::vector<uint32_t> output_queue_ids = {3, 4};
  std::shared_ptr<FlowFuncParams> flow_func_param(
      new (std::nothrow) FlowFuncParams("pp1", input_queue_ids.size(), output_queue_ids.size(), 0, 0));
  std::shared_ptr<FlowFuncRunContext> flow_func_run_context(new (std::nothrow)
                                                                FlowFuncRunContext(0, flow_func_param, nullptr));
  auto tensor_msg = flow_func_run_context->ToFlowMsg(tensor);
  EXPECT_EQ(tensor_msg, nullptr);
}
}  // namespace FlowFunc