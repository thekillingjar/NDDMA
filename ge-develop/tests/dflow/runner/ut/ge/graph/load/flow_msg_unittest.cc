/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "dflow/executor/flow_msg.h"

using namespace std;
using namespace testing;

namespace ge {
class FlowMsgTest : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(FlowMsgTest, TestTensorFlowMsg) {
  auto msg = FlowBufferFactory::AllocTensorMsg({1, 1}, DT_INT32);
  // test set flow msg
  msg->SetFlowFlags(1U);
  msg->SetRetCode(2);
  msg->SetStartTime(3UL);
  msg->SetEndTime(4UL);
  msg->SetTransactionId(5UL);
  msg->SetUserData("test_user_data", 14);
  auto msg_tensor = msg->GetTensor();
  std::vector<int32_t> data0 = {static_cast<int32_t>(1)};
  msg_tensor->SetData(reinterpret_cast<const uint8_t *>(data0.data()), data0.size() * sizeof(int32_t));

  // test get flow msg
  ASSERT_EQ(msg->GetMsgType(), MsgType::MSG_TYPE_TENSOR_DATA);
  ASSERT_EQ(msg->GetFlowFlags(), 1U);
  ASSERT_EQ(msg->GetRetCode(), 2);
  ASSERT_EQ(msg->GetStartTime(), 3UL);
  ASSERT_EQ(msg->GetEndTime(), 4UL);
  ASSERT_EQ(msg->GetTransactionId(), 5UL);
  char user_data[15] = {};
  ASSERT_EQ(msg->GetUserData(user_data, 14), SUCCESS);
  ASSERT_EQ(std::string(user_data), "test_user_data");
  auto output_tensor = msg->GetTensor();
  ASSERT_EQ(output_tensor->GetSize(), data0.size() * sizeof(int32_t));
  ASSERT_EQ(reinterpret_cast<const int32_t *>(output_tensor->GetData())[0], 1);

  std::vector<int32_t> data1 = {static_cast<int32_t>(2)};
  Tensor msg1_tensor(TensorDesc(Shape({1, 1}), FORMAT_ND, DT_INT32),
                       reinterpret_cast<const uint8_t *>(data1.data()), data1.size() * sizeof(int32_t));
  auto msg1 = FlowBufferFactory::ToFlowMsg(msg1_tensor);
  // test get flow msg
  ASSERT_EQ(msg1->GetMsgType(), MsgType::MSG_TYPE_TENSOR_DATA);
  auto output1_tensor = msg1->GetTensor();
  ASSERT_EQ(output1_tensor->GetSize(), data1.size() * sizeof(int32_t));
  ASSERT_EQ(reinterpret_cast<const int32_t *>(output1_tensor->GetData())[0], 2);
}

TEST_F(FlowMsgTest, TestAllocTensor) {
  auto tensor = FlowBufferFactory::AllocTensor({1, 2}, DT_INT32);
  ASSERT_EQ(tensor->GetSize(), 2 * sizeof(int32_t));
  ASSERT_NE(tensor->GetData(), nullptr);
  ASSERT_EQ(tensor->GetTensorDesc().GetDataType(), DT_INT32);
}

TEST_F(FlowMsgTest, TestRawDataFlowMsg) {
  auto msg0 = FlowBufferFactory::AllocRawDataMsg(sizeof(int32_t) * 2);
  void *msg0_data_ptr = nullptr;
  uint64_t msg0_data_size = 0UL;
  ASSERT_EQ(msg0->GetRawData(msg0_data_ptr, msg0_data_size), SUCCESS);
  ASSERT_NE(msg0_data_ptr, nullptr);
  ASSERT_EQ(msg0_data_size, sizeof(int32_t) * 2);
  // init msg0 flow msg
  reinterpret_cast<int32_t *>(msg0_data_ptr)[0] = 1;
  reinterpret_cast<int32_t *>(msg0_data_ptr)[1] = 2;
  msg0->SetMsgType(static_cast<MsgType>(1065));

  // test get flow msg
  ASSERT_EQ(msg0->GetMsgType(), static_cast<MsgType>(1065));
  void *output0_data_ptr = nullptr;
  uint64_t output0_data_size = 0UL;
  ASSERT_EQ(msg0->GetRawData(output0_data_ptr, output0_data_size), SUCCESS);
  ASSERT_NE(output0_data_ptr, nullptr);
  ASSERT_EQ(output0_data_size, sizeof(int32_t) * 2);
  ASSERT_EQ(reinterpret_cast<int32_t *>(output0_data_ptr)[0], 1);
  ASSERT_EQ(reinterpret_cast<int32_t *>(output0_data_ptr)[1], 2);

  std::vector<int32_t> data1 = {3, 4};
  RawData msg1_raw_data = {};
  msg1_raw_data.addr = &data1[0];
  msg1_raw_data.len = data1.size() * sizeof(int32_t);
  auto msg1 = FlowBufferFactory::ToFlowMsg(msg1_raw_data);

  // test get flow msg
  ASSERT_EQ(msg1->GetMsgType(), MsgType::MSG_TYPE_RAW_MSG);
  void *output1_data_ptr = nullptr;
  uint64_t output1_data_size = 0UL;
  ASSERT_EQ(msg1->GetRawData(output1_data_ptr, output1_data_size), SUCCESS);
  ASSERT_NE(output1_data_ptr, nullptr);
  ASSERT_EQ(output1_data_size, sizeof(int32_t) * 2);
  ASSERT_EQ(reinterpret_cast<int32_t *>(output1_data_ptr)[0], 3);
  ASSERT_EQ(reinterpret_cast<int32_t *>(output1_data_ptr)[1], 4);
}

TEST_F(FlowMsgTest, TestEmptyFlowMsg) {
  auto msg = FlowBufferFactory::AllocEmptyDataMsg(MsgType::MSG_TYPE_TENSOR_DATA);
  // test get flow msg
  ASSERT_EQ(msg->GetMsgType(), MsgType::MSG_TYPE_TENSOR_DATA);
  ASSERT_EQ(msg->GetTensor(), nullptr);
}

TEST_F(FlowMsgTest, TestFlowMsgBase) {
  FlowMsgBase base;
  ASSERT_EQ(base.GetTensor(), nullptr);
  void *data_ptr = nullptr;
  uint64_t data_size = 0UL;
  ASSERT_EQ(base.GetRawData(data_ptr, data_size), UNSUPPORTED);
}

TEST_F(FlowMsgTest, TestBuildEmptyDataFlowMsg) {
  rtMbufPtr_t mbuf;
  rtMbufAlloc(&mbuf, 1025);
  EmptyDataFlowMsg empty_flow_msg;
  ASSERT_EQ(empty_flow_msg.BuildNullData(mbuf), SUCCESS);
  // test get flow msg
  ASSERT_EQ(empty_flow_msg.GetMsgType(), MsgType::MSG_TYPE_TENSOR_DATA);
  ASSERT_EQ(empty_flow_msg.GetTensor(), nullptr);
}

TEST_F(FlowMsgTest, TestBuildRawDataFlowMsg) {
  rtMbufPtr_t mbuf;
  rtMbufAlloc(&mbuf, 8);
  void *data = nullptr;
  rtMbufGetBuffAddr(mbuf, &data);
  reinterpret_cast<int32_t *>(data)[0] = 1;
  reinterpret_cast<int32_t *>(data)[1] = 2;

  RawDataFlowMsg raw_data_flow_msg;
  ASSERT_EQ(raw_data_flow_msg.BuildRawData(mbuf), SUCCESS);
  // test get flow msg
  void *data_ptr = nullptr;
  uint64_t data_size = 0UL;
  ASSERT_EQ(raw_data_flow_msg.GetRawData(data_ptr, data_size), SUCCESS);
  ASSERT_NE(data_ptr, nullptr);
  ASSERT_EQ(data_size, 8);
  ASSERT_EQ(reinterpret_cast<int32_t *>(data_ptr)[0], 1);
  ASSERT_EQ(reinterpret_cast<int32_t *>(data_ptr)[1], 2);
}

TEST_F(FlowMsgTest, TestBuildTensorFlowMsg) {
  rtMbufPtr_t mbuf;
  rtMbufAlloc(&mbuf, 1028);
  void *data = nullptr;
  rtMbufGetBuffAddr(mbuf, &data);
  reinterpret_cast<RuntimeTensorDesc *>(data)->dtype = static_cast<int64_t>(DT_INT32);
  reinterpret_cast<RuntimeTensorDesc *>(data)->format = static_cast<int64_t>(FORMAT_ND);
  reinterpret_cast<RuntimeTensorDesc *>(data)->shape[0] = 1;
  reinterpret_cast<RuntimeTensorDesc *>(data)->shape[1] = 1;
  reinterpret_cast<RuntimeTensorDesc *>(data)->data_size = 4;
  auto tensor_data = reinterpret_cast<int8_t *>(data) + sizeof(RuntimeTensorDesc);
  reinterpret_cast<int32_t *>(tensor_data)[0] = 5;

  TensorFlowMsg tensor_flow_msg;
  ASSERT_EQ(tensor_flow_msg.BuildTensor(mbuf, GeTensorDesc(GeShape({1}))), SUCCESS);
  // test get flow msg
  auto tensor = tensor_flow_msg.GetTensor();
  ASSERT_NE(tensor, nullptr);
  ASSERT_EQ(tensor->GetSize(), 4);
  auto tensor_data_ptr = tensor->GetData();
  ASSERT_NE(tensor_data_ptr, nullptr);
  ASSERT_EQ(tensor->GetSize(), 4);
  ASSERT_EQ(reinterpret_cast<int32_t *>(tensor_data_ptr)[0], 5);
}

TEST_F(FlowMsgTest, ToFlowMsgFailed) {
  Tensor tensor(TensorDesc(Shape({-1})), {1});
  // shape invalid
  ASSERT_EQ(FlowBufferFactory::ToFlowMsg(tensor), nullptr);
}
}  // namespace ge
