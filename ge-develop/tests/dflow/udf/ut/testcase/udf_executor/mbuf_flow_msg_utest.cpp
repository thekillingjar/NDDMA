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
#include "flow_func/mbuf_flow_msg.h"
#include "mockcpp/mockcpp.hpp"
#include "common/data_utils.h"

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
class DefaultFlowMsg : public FlowMsg {
  FlowFunc::MsgType GetMsgType() const override {
    return FlowFunc::MsgType::MSG_TYPE_TENSOR_DATA;
  }

  FlowFunc::Tensor *GetTensor() const override {
    return nullptr;
  }

  int32_t GetRetCode() const override {
    return 0;
  }

  void SetRetCode(int32_t ret_code) override {}

  void SetStartTime(uint64_t start_time) override {}

  uint64_t GetStartTime() const override {
    return 0;
  }

  void SetEndTime(uint64_t end_time) override {}

  uint64_t GetEndTime() const override {
    return 0;
  }

  void SetFlowFlags(uint32_t flags) override {}

  uint32_t GetFlowFlags() const override {
    return 0;
  }

  void SetRouteLabel(uint32_t route_label) override {}
};

class DefaultTensor : public Tensor {
  const std::vector<int64_t> &GetShape() const override {
    return shape;
  }

  TensorDataType GetDataType() const override {
    return TensorDataType::DT_INT32;
  }

  void *GetData() const override {
    return nullptr;
  }

  uint64_t GetDataSize() const override {
    return 0;
  }

  uint64_t GetDataBufferSize() const override {
    return 0;
  }

  int64_t GetElementCnt() const override {
    return 0;
  }

 private:
  std::vector<int64_t> shape;
};
}  // namespace
class MbufFlowMsgUTest : public testing::Test {
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

TEST_F(MbufFlowMsgUTest, normal_test) {
  std::vector<int64_t> shape = {1, 2, 3};
  TensorDataType data_type = TensorDataType::DT_INT32;
  std::shared_ptr<MbufFlowMsg> mbuf_flow_msg = MbufFlowMsg::AllocTensorMsg(shape, data_type, 0, {});
  uint64_t expect_data_size = CalcDataSize(shape, data_type);
  uint64_t expect_element_cnt = CalcElementCnt(shape);
  EXPECT_NE(mbuf_flow_msg, nullptr);
  auto mbuf_tensor = mbuf_flow_msg->GetTensor();
  EXPECT_NE(mbuf_tensor, nullptr);
  void *data = mbuf_tensor->GetData();
  EXPECT_NE(data, nullptr);
  uint64_t element_cnt = mbuf_tensor->GetElementCnt();
  EXPECT_EQ(element_cnt, expect_element_cnt);
  uint64_t data_size = mbuf_tensor->GetDataSize();
  EXPECT_EQ(data_size, expect_data_size);
  TensorDataType type = mbuf_tensor->GetDataType();
  EXPECT_EQ(type, data_type);
  auto get_shape = mbuf_tensor->GetShape();
  EXPECT_EQ(get_shape, shape);
  mbuf_flow_msg->SetRetCode(FLOW_FUNC_ERR_DRV_ERROR);
  EXPECT_EQ(mbuf_flow_msg->GetRetCode(), FLOW_FUNC_ERR_DRV_ERROR);
}

TEST_F(MbufFlowMsgUTest, alloc_tensor_lsit_error_test) {
  std::vector<int64_t> shape = {1, 2, 3};
  TensorDataType data_type = TensorDataType::DT_INT32;
  std::shared_ptr<MbufFlowMsg> mbuf_flow_msg =
      MbufFlowMsg::AllocTensorListMsg({shape}, {data_type, data_type}, 0, {}, 512);
  EXPECT_EQ(mbuf_flow_msg, nullptr);
}

TEST_F(MbufFlowMsgUTest, set_data_label_test) {
  std::vector<int64_t> shape = {1, 2, 3};
  TensorDataType data_type = TensorDataType::DT_INT32;
  std::shared_ptr<MbufFlowMsg> mbuf_flow_msg = MbufFlowMsg::AllocTensorMsg(shape, data_type, 0, {});
  uint32_t data_label = 1000;
  mbuf_flow_msg->SetDataLabel(data_label);
  std::string debug_str = mbuf_flow_msg->DebugString();
  EXPECT_TRUE(debug_str.find("data_label=") != std::string::npos);
}

TEST_F(MbufFlowMsgUTest, unsupport_dim_num_over) {
  std::vector<int64_t> shape(kMaxDimSize + 1, 2);
  TensorDataType data_type = TensorDataType::DT_FLOAT;
  std::shared_ptr<MbufFlowMsg> mbuf_flow_msg = MbufFlowMsg::AllocTensorMsg(shape, data_type, 0, {});
  EXPECT_EQ(mbuf_flow_msg, nullptr);
}

TEST_F(MbufFlowMsgUTest, unsupport_navigate_dim) {
  std::vector<int64_t> shape = {1, -2, 3};
  TensorDataType data_type = TensorDataType::DT_INT32;
  std::shared_ptr<MbufFlowMsg> mbuf_flow_msg = MbufFlowMsg::AllocTensorMsg(shape, data_type, 0, {});
  EXPECT_EQ(mbuf_flow_msg, nullptr);
}

TEST_F(MbufFlowMsgUTest, buffer_alloc_failed) {
  std::vector<int64_t> shape = {1, 2, 3};
  TensorDataType data_type = TensorDataType::DT_INT32;

  MOCKER(halMbufAllocEx).stubs().will(returnValue((int)DRV_ERROR_OUT_OF_MEMORY));
  std::shared_ptr<MbufFlowMsg> mbuf_flow_msg = MbufFlowMsg::AllocTensorMsg(shape, data_type, 0, {});
  EXPECT_EQ(mbuf_flow_msg, nullptr);
}

TEST_F(MbufFlowMsgUTest, halMbufSetDataLen_failed) {
  std::vector<int64_t> shape = {1, 2, 3};
  TensorDataType data_type = TensorDataType::DT_INT32;

  MOCKER(halMbufSetDataLen).stubs().will(returnValue((int)DRV_ERROR_PARA_ERROR));
  std::shared_ptr<MbufFlowMsg> mbuf_flow_msg = MbufFlowMsg::AllocTensorMsg(shape, data_type, 0, {});
  EXPECT_EQ(mbuf_flow_msg, nullptr);
}

TEST_F(MbufFlowMsgUTest, halMbufGetBuffAddr_failed) {
  std::vector<int64_t> shape = {1, 2, 3};
  TensorDataType data_type = TensorDataType::DT_INT32;

  MOCKER(halMbufGetBuffAddr).stubs().will(returnValue((int)DRV_ERROR_PARA_ERROR));
  std::shared_ptr<MbufFlowMsg> mbuf_flow_msg = MbufFlowMsg::AllocTensorMsg(shape, data_type, 0, {});
  EXPECT_EQ(mbuf_flow_msg, nullptr);
}

TEST_F(MbufFlowMsgUTest, halMbufGetPrivInfo_failed) {
  std::vector<int64_t> shape = {1, 2, 3};
  TensorDataType data_type = TensorDataType::DT_INT32;

  MOCKER(halMbufGetPrivInfo).stubs().will(returnValue((int)DRV_ERROR_PARA_ERROR));
  std::shared_ptr<MbufFlowMsg> mbuf_flow_msg = MbufFlowMsg::AllocTensorMsg(shape, data_type, 0, {});
  EXPECT_EQ(mbuf_flow_msg, nullptr);
}

TEST_F(MbufFlowMsgUTest, shape_is_negative) {
  MbufImpl *mbuf_impl = new (std::nothrow) MbufImpl();
  mbuf_impl->mbuf_size = 100 + sizeof(RuntimeTensorDesc);
  mbuf_impl->tensor_desc.dtype = static_cast<int64_t>(TensorDataType::DT_INT32);
  mbuf_impl->tensor_desc.shape[0] = 1;
  mbuf_impl->tensor_desc.shape[1] = -1;

  Mbuf *buf = (Mbuf *)mbuf_impl;
  static auto mbuf_deleter = [](Mbuf *buf) { (void)halMbufFree(buf); };
  std::shared_ptr<Mbuf> mbuf_ptr(buf, mbuf_deleter);
  MbufFlowMsg mbuf_flow_msg(mbuf_ptr);
  auto ret = mbuf_flow_msg.Init();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ASSERT_NE(mbuf_flow_msg.GetTensor(), nullptr);
  ASSERT_EQ(mbuf_flow_msg.GetTensor()->GetDataSize(), 100);
}

TEST_F(MbufFlowMsgUTest, tensor_size_over_bufsize) {
  MbufImpl *mbuf_impl = new (std::nothrow) MbufImpl();
  mbuf_impl->mbuf_size = 100 + sizeof(RuntimeTensorDesc);
  mbuf_impl->tensor_desc.dtype = static_cast<int64_t>(TensorDataType::DT_INT32);
  mbuf_impl->tensor_desc.shape[0] = 1;
  mbuf_impl->tensor_desc.shape[1] = 1000;

  Mbuf *buf = (Mbuf *)mbuf_impl;
  static auto mbuf_deleter = [](Mbuf *buf) { (void)halMbufFree(buf); };
  std::shared_ptr<Mbuf> mbuf_ptr(buf, mbuf_deleter);
  MbufFlowMsg mbuf_flow_msg(mbuf_ptr);
  auto ret = mbuf_flow_msg.Init();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  // use small size
  EXPECT_EQ(mbuf_flow_msg.GetTensor()->GetDataSize(), 100);
}

TEST_F(MbufFlowMsgUTest, tensor_size_less_bufsize) {
  uint64_t mbuf_data_size = 100;
  MbufImpl *mbuf_impl = new (std::nothrow) MbufImpl();
  mbuf_impl->mbuf_size = mbuf_data_size + sizeof(RuntimeTensorDesc);
  mbuf_impl->tensor_desc.dtype = static_cast<int64_t>(TensorDataType::DT_INT32);
  mbuf_impl->tensor_desc.shape[0] = 2;
  mbuf_impl->tensor_desc.shape[1] = 2;
  mbuf_impl->tensor_desc.shape[2] = 3;

  Mbuf *buf = (Mbuf *)mbuf_impl;
  static auto mbuf_deleter = [](Mbuf *buf) { (void)halMbufFree(buf); };
  std::shared_ptr<Mbuf> mbuf_ptr(buf, mbuf_deleter);
  MbufFlowMsg mbuf_flow_msg(mbuf_ptr);
  auto ret = mbuf_flow_msg.Init();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  uint64_t expect_data_size = CalcDataSize({2, 3}, TensorDataType::DT_INT32);
  // tensor size less buf size, will set data size to real tensor size
  EXPECT_EQ(mbuf_flow_msg.GetTensor()->GetDataSize(), expect_data_size);
}

TEST_F(MbufFlowMsgUTest, retCode_not_zero) {
  MbufImpl *mbuf_impl = new (std::nothrow) MbufImpl();
  mbuf_impl->head_msg.ret_code = FLOW_FUNC_ERR_PARAM_INVALID;
  Mbuf *buf = (Mbuf *)mbuf_impl;
  static auto mbuf_deleter = [](Mbuf *buf) { (void)halMbufFree(buf); };
  std::shared_ptr<Mbuf> mbuf_ptr(buf, mbuf_deleter);
  MbufFlowMsg mbuf_flow_msg(mbuf_ptr);
  auto ret = mbuf_flow_msg.Init();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  auto tensor_data = mbuf_flow_msg.GetTensor();
  EXPECT_NE(tensor_data, nullptr);
  EXPECT_EQ(mbuf_flow_msg.GetRetCode(), FLOW_FUNC_ERR_PARAM_INVALID);
}

TEST_F(MbufFlowMsgUTest, AllocTensor_success) {
  auto tensor = FlowBufferFactory::AllocTensor({1, 2}, TensorDataType::DT_INT32);
  EXPECT_NE(tensor, nullptr);
  auto tensor_data = tensor->GetData();
  EXPECT_NE(tensor_data, nullptr);
  auto tensor_shape = tensor->GetShape();
  EXPECT_EQ(tensor_shape[0], 1);
  EXPECT_EQ(tensor_shape[1], 2);
}

TEST_F(MbufFlowMsgUTest, AllocTensor_failed) {
  MOCKER(halMbufGetPrivInfo).stubs().will(returnValue((int)DRV_ERROR_PARA_ERROR));
  auto tensor = FlowBufferFactory::AllocTensor({1, 2}, TensorDataType::DT_INT32);
  EXPECT_EQ(tensor, nullptr);
}

TEST_F(MbufFlowMsgUTest, AllocRawDataMsg_param_check) {
  int64_t raw_data_size = -1;
  std::shared_ptr<MbufFlowMsg> mbuf_flow_msg = MbufFlowMsg::AllocRawDataMsg(raw_data_size, 0);
  EXPECT_EQ(mbuf_flow_msg, nullptr);
}

TEST_F(MbufFlowMsgUTest, AllocRawDataMsg_success) {
  uint32_t raw_data_size = 100;
  std::shared_ptr<MbufFlowMsg> mbuf_flow_msg = MbufFlowMsg::AllocRawDataMsg(raw_data_size, 0);
  EXPECT_NE(mbuf_flow_msg, nullptr);
  auto tensor_data = mbuf_flow_msg->GetTensor();
  EXPECT_EQ(tensor_data, nullptr);
  const auto &mbuf_info = mbuf_flow_msg->GetMbufInfo();
  EXPECT_NE(mbuf_info.mbuf_addr, nullptr);
  EXPECT_EQ(mbuf_info.mbuf_len, raw_data_size);
  EXPECT_NE(mbuf_info.head_buf, nullptr);
  EXPECT_EQ(mbuf_info.head_buf_len, sizeof(MbufImpl::reserve_head) + sizeof(MbufImpl::head_msg));
}

TEST_F(MbufFlowMsgUTest, AllocAndGetRawDataMsg_success) {
  uint32_t raw_data_size = 100;
  std::shared_ptr<MbufFlowMsg> mbuf_flow_msg = MbufFlowMsg::AllocRawDataMsg(raw_data_size, 0);
  EXPECT_NE(mbuf_flow_msg, nullptr);
  auto tensor_data = mbuf_flow_msg->GetTensor();
  EXPECT_EQ(tensor_data, nullptr);
  void *data_ptr = nullptr;
  uint64_t data_len = 0UL;
  EXPECT_EQ(mbuf_flow_msg->GetRawData(data_ptr, data_len), FLOW_FUNC_SUCCESS);
  EXPECT_NE(data_ptr, nullptr);
  EXPECT_EQ(data_len, raw_data_size);
}

TEST_F(MbufFlowMsgUTest, raw_data_msg_init_success) {
  uint32_t raw_data_size = 100;
  MbufImpl *mbuf_impl = new (std::nothrow) MbufImpl();
  mbuf_impl->mbuf_size = raw_data_size;
  mbuf_impl->head_msg.msg_type = static_cast<uint16_t>(MsgType::MSG_TYPE_RAW_MSG);
  Mbuf *buf = (Mbuf *)mbuf_impl;
  static auto mbuf_deleter = [](Mbuf *buf) { (void)halMbufFree(buf); };
  std::shared_ptr<Mbuf> mbuf_ptr(buf, mbuf_deleter);
  MbufFlowMsg mbuf_flow_msg(mbuf_ptr);
  auto ret = mbuf_flow_msg.Init();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  auto tensor_data = mbuf_flow_msg.GetTensor();
  EXPECT_EQ(tensor_data, nullptr);
  const auto &mbuf_info = mbuf_flow_msg.GetMbufInfo();
  EXPECT_NE(mbuf_info.mbuf_addr, nullptr);
  EXPECT_EQ(mbuf_info.mbuf_len, raw_data_size);
  EXPECT_NE(mbuf_info.head_buf, nullptr);
  EXPECT_EQ(mbuf_info.head_buf_len, sizeof(MbufImpl::reserve_head) + sizeof(MbufImpl::head_msg));
}

TEST_F(MbufFlowMsgUTest, tensor_list_init_len_invalid) {
  MbufImpl *mbuf_impl = new (std::nothrow) MbufImpl();
  mbuf_impl->head_msg.ret_code = FLOW_FUNC_SUCCESS;
  mbuf_impl->head_msg.msg_type = static_cast<uint16_t>(MsgType::MSG_TYPE_TENSOR_LIST);
  mbuf_impl->mbuf_size = 1;
  Mbuf *buf = (Mbuf *)mbuf_impl;
  static auto mbuf_deleter = [](Mbuf *buf) { (void)halMbufFree(buf); };
  std::shared_ptr<Mbuf> mbuf_ptr(buf, mbuf_deleter);
  MbufFlowMsg mbuf_flow_msg(mbuf_ptr);
  auto ret = mbuf_flow_msg.Init();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
}

TEST_F(MbufFlowMsgUTest, tensor_list_init_success) {
  MbufImpl *mbuf_impl = new (std::nothrow) MbufImpl();
  mbuf_impl->head_msg.ret_code = FLOW_FUNC_SUCCESS;
  mbuf_impl->head_msg.msg_type = static_cast<uint16_t>(MsgType::MSG_TYPE_TENSOR_LIST);
  // algin size 64. real size 16 and 32
  // rtd(2*2)+64+rtd(4*2)+64
  mbuf_impl->mbuf_size = 64 + sizeof(RuntimeTensorDesc) + 64 + sizeof(RuntimeTensorDesc);
  mbuf_impl->tensor_desc.dtype = static_cast<int64_t>(TensorDataType::DT_INT32);
  mbuf_impl->tensor_desc.shape[0] = 2;
  mbuf_impl->tensor_desc.shape[1] = 2;
  mbuf_impl->tensor_desc.shape[2] = 2;
  mbuf_impl->tensor_desc.data_size = 64;
  auto *rtd = reinterpret_cast<RuntimeTensorDesc *>(&(mbuf_impl->data[64]));
  rtd->dtype = static_cast<int64_t>(TensorDataType::DT_INT32);
  rtd->shape[0] = 2;
  rtd->shape[1] = 4;
  rtd->shape[2] = 2;
  rtd->data_size = 64;
  Mbuf *buf = (Mbuf *)mbuf_impl;
  static auto mbuf_deleter = [](Mbuf *buf) { (void)halMbufFree(buf); };
  std::shared_ptr<Mbuf> mbuf_ptr(buf, mbuf_deleter);
  MbufFlowMsg mbuf_flow_msg(mbuf_ptr);
  auto ret = mbuf_flow_msg.Init();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  auto tensor_list = mbuf_flow_msg.GetTensorList();
  EXPECT_EQ(tensor_list.size(), 2);
  EXPECT_EQ(tensor_list[0]->GetShape().size(), 2);
  std::vector<int64_t> shape1 = {2, 2};
  EXPECT_EQ(tensor_list[0]->GetShape(), shape1);
  EXPECT_EQ(tensor_list[0]->GetDataSize(), 16);
  EXPECT_EQ(reinterpret_cast<MbufTensor *>(tensor_list[0])->GetDataBufferSize(), 64);
  EXPECT_EQ(tensor_list[1]->GetShape().size(), 2);
  std::vector<int64_t> shape2 = {4, 2};
  EXPECT_EQ(tensor_list[1]->GetShape(), shape2);
  EXPECT_EQ(tensor_list[1]->GetDataSize(), 32);
  EXPECT_EQ(reinterpret_cast<MbufTensor *>(tensor_list[1])->GetDataBufferSize(), 64);
}

TEST_F(MbufFlowMsgUTest, tensor_list_init_overflow1) {
  MbufImpl *mbuf_impl = new (std::nothrow) MbufImpl();
  mbuf_impl->head_msg.ret_code = FLOW_FUNC_SUCCESS;
  mbuf_impl->head_msg.msg_type = static_cast<uint16_t>(MsgType::MSG_TYPE_TENSOR_LIST);
  // algin size 64. real size 16 and 32
  // rtd(2*2)+64+rtd(4*2)+64
  mbuf_impl->mbuf_size = 64 + sizeof(RuntimeTensorDesc) + 64 + sizeof(RuntimeTensorDesc);
  mbuf_impl->tensor_desc.dtype = static_cast<int64_t>(TensorDataType::DT_INT32);
  mbuf_impl->tensor_desc.shape[0] = 2;
  mbuf_impl->tensor_desc.shape[1] = 2;
  mbuf_impl->tensor_desc.shape[2] = 2;
  mbuf_impl->tensor_desc.data_size = 64;
  auto *rtd = reinterpret_cast<RuntimeTensorDesc *>(&(mbuf_impl->data[64]));
  rtd->dtype = static_cast<int64_t>(TensorDataType::DT_INT32);
  rtd->shape[0] = 2;
  rtd->shape[1] = 4;
  rtd->shape[2] = 2;
  rtd->data_size = UINT64_MAX - 1;
  Mbuf *buf = (Mbuf *)mbuf_impl;
  static auto mbuf_deleter = [](Mbuf *buf) { (void)halMbufFree(buf); };
  std::shared_ptr<Mbuf> mbuf_ptr(buf, mbuf_deleter);
  MbufFlowMsg mbuf_flow_msg(mbuf_ptr);
  auto ret = mbuf_flow_msg.Init();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  auto tensor_list = mbuf_flow_msg.GetTensorList();
  EXPECT_EQ(tensor_list.size(), 1);
  EXPECT_EQ(tensor_list[0]->GetShape().size(), 2);
  std::vector<int64_t> shape1 = {2, 2};
  EXPECT_EQ(tensor_list[0]->GetShape(), shape1);
  EXPECT_EQ(tensor_list[0]->GetDataSize(), 16);
  EXPECT_EQ(reinterpret_cast<MbufTensor *>(tensor_list[0])->GetDataBufferSize(), 64);
}

TEST_F(MbufFlowMsgUTest, tensor_list_init_overflow2) {
  MbufImpl *mbuf_impl = new (std::nothrow) MbufImpl();
  mbuf_impl->head_msg.ret_code = FLOW_FUNC_SUCCESS;
  mbuf_impl->head_msg.msg_type = static_cast<uint16_t>(MsgType::MSG_TYPE_TENSOR_LIST);
  // algin size 64. real size 16 and 32
  // rtd(2*2)+64+rtd(4*2)+64
  mbuf_impl->mbuf_size = 64 + sizeof(RuntimeTensorDesc) + 64 + sizeof(RuntimeTensorDesc);
  mbuf_impl->tensor_desc.dtype = static_cast<int64_t>(TensorDataType::DT_INT32);
  mbuf_impl->tensor_desc.shape[0] = 2;
  mbuf_impl->tensor_desc.shape[1] = 2;
  mbuf_impl->tensor_desc.shape[2] = 2;
  mbuf_impl->tensor_desc.data_size = UINT64_MAX - 1024;
  auto *rtd = reinterpret_cast<RuntimeTensorDesc *>(&(mbuf_impl->data[64]));
  rtd->dtype = static_cast<int64_t>(TensorDataType::DT_INT32);
  rtd->shape[0] = 2;
  rtd->shape[1] = 4;
  rtd->shape[2] = 2;
  rtd->data_size = UINT64_MAX - 1;
  Mbuf *buf = (Mbuf *)mbuf_impl;
  static auto mbuf_deleter = [](Mbuf *buf) { (void)halMbufFree(buf); };
  std::shared_ptr<Mbuf> mbuf_ptr(buf, mbuf_deleter);
  MbufFlowMsg mbuf_flow_msg(mbuf_ptr);
  auto ret = mbuf_flow_msg.Init();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  auto tensor_list = mbuf_flow_msg.GetTensorList();
  EXPECT_EQ(tensor_list.size(), 0);
}

TEST_F(MbufFlowMsgUTest, default_get_trans_id) {
  DefaultFlowMsg default_flow_msg;
  EXPECT_EQ(default_flow_msg.GetTransactionId(), UINT64_MAX);
}

TEST_F(MbufFlowMsgUTest, default_get_tensor_list) {
  DefaultFlowMsg default_flow_msg;
  EXPECT_EQ(default_flow_msg.GetTensorList().empty(), true);
}

TEST_F(MbufFlowMsgUTest, default_get_raw_data) {
  DefaultFlowMsg default_flow_msg;
  void *data;
  uint64_t len = 0;
  EXPECT_EQ(default_flow_msg.GetRawData(data, len), FLOW_FUNC_ERR_NOT_SUPPORT);
}

TEST_F(MbufFlowMsgUTest, default_Reshape) {
  DefaultTensor default_tensor;
  EXPECT_EQ(default_tensor.Reshape({1, 2, 3}), FLOW_FUNC_ERR_NOT_SUPPORT);
}

TEST_F(MbufFlowMsgUTest, Reshape_success) {
  std::vector<int64_t> shape = {1, 2, 9};
  TensorDataType data_type = TensorDataType::DT_INT32;
  std::shared_ptr<MbufFlowMsg> mbuf_flow_msg = MbufFlowMsg::AllocTensorMsg(shape, data_type, 0, {});
  std::vector<int64_t> new_shape1 = {9, 2, 1};
  EXPECT_EQ(mbuf_flow_msg->GetTensor()->Reshape(new_shape1), FLOW_FUNC_SUCCESS);
  EXPECT_EQ(mbuf_flow_msg->GetTensor()->GetShape(), new_shape1);
  std::vector<int64_t> new_shape2 = {18};
  EXPECT_EQ(mbuf_flow_msg->GetTensor()->Reshape(new_shape2), FLOW_FUNC_SUCCESS);
  EXPECT_EQ(mbuf_flow_msg->GetTensor()->GetShape(), new_shape2);
  std::vector<int64_t> new_shape3 = {2, 3, 1, 3};
  EXPECT_EQ(mbuf_flow_msg->GetTensor()->Reshape(new_shape3), FLOW_FUNC_SUCCESS);
  EXPECT_EQ(mbuf_flow_msg->GetTensor()->GetShape(), new_shape3);

  std::vector<int64_t> new_shape4 = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 1, 3};
  EXPECT_EQ(mbuf_flow_msg->GetTensor()->Reshape(new_shape4), FLOW_FUNC_SUCCESS);
  EXPECT_EQ(mbuf_flow_msg->GetTensor()->GetShape(), new_shape4);
}

TEST_F(MbufFlowMsgUTest, Reshape_Failed_as_over_max_dims) {
  std::vector<int64_t> shape = {1, 2, 9};
  TensorDataType data_type = TensorDataType::DT_INT32;
  std::shared_ptr<MbufFlowMsg> mbuf_flow_msg = MbufFlowMsg::AllocTensorMsg(shape, data_type, 0, {});
  std::vector<int64_t> over_max_dims_shape = {3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3};
  EXPECT_NE(mbuf_flow_msg->GetTensor()->Reshape(over_max_dims_shape), FLOW_FUNC_SUCCESS);
  // shape not change
  EXPECT_EQ(mbuf_flow_msg->GetTensor()->GetShape(), shape);
}

TEST_F(MbufFlowMsgUTest, Reshape_Failed_as_element_not_same) {
  std::vector<int64_t> shape = {1, 2, 9};
  TensorDataType data_type = TensorDataType::DT_INT32;
  std::shared_ptr<MbufFlowMsg> mbuf_flow_msg = MbufFlowMsg::AllocTensorMsg(shape, data_type, 0, {});
  std::vector<int64_t> smaller_shape = {3, 2};
  EXPECT_NE(mbuf_flow_msg->GetTensor()->Reshape(smaller_shape), FLOW_FUNC_SUCCESS);
  std::vector<int64_t> larger_shape = {3, 2, 3, 3};
  EXPECT_NE(mbuf_flow_msg->GetTensor()->Reshape(larger_shape), FLOW_FUNC_SUCCESS);
  // shape not change
  EXPECT_EQ(mbuf_flow_msg->GetTensor()->GetShape(), shape);
}

TEST_F(MbufFlowMsgUTest, Reshape_Failed_as_dim_negative) {
  std::vector<int64_t> shape = {1, 2, 9};
  TensorDataType data_type = TensorDataType::DT_INT32;
  std::shared_ptr<MbufFlowMsg> mbuf_flow_msg = MbufFlowMsg::AllocTensorMsg(shape, data_type, 0, {});
  std::vector<int64_t> smaller_shape = {3, -2, -3};
  EXPECT_NE(mbuf_flow_msg->GetTensor()->Reshape(smaller_shape), FLOW_FUNC_SUCCESS);
  // shape not change
  EXPECT_EQ(mbuf_flow_msg->GetTensor()->GetShape(), shape);
}
}  // namespace FlowFunc