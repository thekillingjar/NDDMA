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
#include "utils/udf_test_helper.h"
#include "ascend_hal.h"
#include "mockcpp/mockcpp.hpp"
#include "common/data_utils.h"
#include "flow_func/mbuf_flow_msg.h"
#define private public
#include "config/global_config.h"
#include "execute/flow_model_impl.h"
#undef private

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

drvError_t halQueueDeQueueStub(unsigned int dev_id, unsigned int qid, void **mbuf) {
  Mbuf *buf = nullptr;
  (void)halMbufAllocExStub(sizeof(RuntimeTensorDesc) + 1024, 0, 0, 0, &buf);
  *mbuf = buf;
  void *data_buf = nullptr;
  halMbufGetBuffAddrStub(buf, &data_buf);
  auto tensor_desc = static_cast<RuntimeTensorDesc *>(data_buf);
  tensor_desc->dtype = static_cast<int64_t>(TensorDataType::DT_INT8);
  tensor_desc->shape[0] = 1;
  tensor_desc->shape[1] = 1024;
  return DRV_ERROR_NONE;
}

drvError_t halQueueEnQueueStub(unsigned int dev_id, unsigned int qid, void *mbuf) {
  halMbufFreeStub(static_cast<Mbuf *>(mbuf));
  return DRV_ERROR_NONE;
}

int halMbufAllocStub(uint64_t size, Mbuf **mbuf) {
  MbufImpl *mbuf_impl = new (std::nothrow) MbufImpl();
  memset_s(mbuf_impl, sizeof(MbufImpl), 0, sizeof(MbufImpl));
  mbuf_impl->mbuf_size = size;
  *mbuf = reinterpret_cast<Mbuf *>(mbuf_impl);
  return DRV_ERROR_NONE;
}

drvError_t halQueueDeQueueBuffStub(unsigned int dev_id, unsigned int qid, struct buff_iovec *vector, int timeout) {
  auto tensor_desc = static_cast<RuntimeTensorDesc *>(vector->ptr[0].iovec_base);
  tensor_desc->dtype = static_cast<int64_t>(TensorDataType::DT_INT8);
  tensor_desc->shape[0] = 1;
  tensor_desc->shape[1] = 1024;
  return DRV_ERROR_NONE;
}

drvError_t halQueuePeekStub(unsigned int dev_id, unsigned int qid, uint64_t *buf_len, int timeout) {
  *buf_len = sizeof(RuntimeTensorDesc) + 1024;
  return DRV_ERROR_NONE;
}
}  // namespace
class FlowModelImplUTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    // reset device info
    GlobalConfig::Instance().SetPhyDeviceId(-1);
    GlobalConfig::Instance().SetDeviceId(0);
  }

  static void TearDownTestCase() {
    // reset device info
    GlobalConfig::Instance().SetPhyDeviceId(-1);
    GlobalConfig::Instance().SetDeviceId(0);
  }
  virtual void SetUp() {
    back_is_on_device_ = GlobalConfig::Instance().IsOnDevice();
    GlobalConfig::Instance().SetPhyDeviceId(-1);
    MOCKER(halMbufAllocEx).defaults().will(invoke(halMbufAllocExStub));
    MOCKER(halMbufFree).defaults().will(invoke(halMbufFreeStub));
    MOCKER(halMbufGetBuffAddr).defaults().will(invoke(halMbufGetBuffAddrStub));
    MOCKER(halMbufGetBuffSize).defaults().will(invoke(halMbufGetBuffSizeStub));
    MOCKER(halMbufSetDataLen).defaults().will(invoke(halMbufSetDataLenStub));
    MOCKER(halMbufGetPrivInfo).defaults().will(invoke(halMbufGetPrivInfoStub));
    MOCKER(halMbufGetDataLen).defaults().will(invoke(halMbufGetDataLenStub));
    MOCKER(halMbufCopyRef).defaults().will(invoke(halMbufCopyRefStub));
    MOCKER(halQueueDeQueue).defaults().will(invoke(halQueueDeQueueStub));
    MOCKER(halQueueEnQueue).defaults().will(invoke(halQueueEnQueueStub));
    MOCKER(halMbufAlloc).defaults().will(invoke(halMbufAllocStub));
    MOCKER(halQueuePeek).defaults().will(invoke(halQueuePeekStub));
    MOCKER(halQueueDeQueueBuff).defaults().will(invoke(halQueueDeQueueBuffStub));
  }

  virtual void TearDown() {
    GlobalMockObject::verify();
    GlobalConfig::on_device_ = back_is_on_device_;
  }

 private:
  bool back_is_on_device_ = false;
};

TEST_F(FlowModelImplUTest, Init_test) {
  FlowModelImpl flow_model(UdfTestHelper::CreateQueueDevInfos({1, 2, 3}), UdfTestHelper::CreateQueueDevInfos({4, 5}));
  auto ret = flow_model.Init();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
}

TEST_F(FlowModelImplUTest, Init_attach_failed) {
  FlowModelImpl flow_model(UdfTestHelper::CreateQueueDevInfos({1, 2}), UdfTestHelper::CreateQueueDevInfos({3}));
  MOCKER(halQueueAttach).stubs().will(returnValue(DRV_ERROR_NONE)).then(returnValue(DRV_ERROR_INNER_ERR));
  auto ret = flow_model.Init();
  EXPECT_NE(ret, FLOW_FUNC_SUCCESS);

  MOCKER(halQueueAttach)
      .stubs()
      .will(returnValue(DRV_ERROR_NONE))
      .then(returnValue(DRV_ERROR_NONE))
      .then(returnValue(DRV_ERROR_INNER_ERR));
  ret = flow_model.Init();
  EXPECT_NE(ret, FLOW_FUNC_SUCCESS);
}

TEST_F(FlowModelImplUTest, RunFlowGraph) {
  FlowModelImpl flow_model(UdfTestHelper::CreateQueueDevInfos({1}), UdfTestHelper::CreateQueueDevInfos({3, 4}));
  auto ret = flow_model.Init();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  MbufHead head_msg;
  auto input = MbufFlowMsg::AllocTensorMsg({0}, TensorDataType::DT_INT32, 0, head_msg, true);
  std::vector<std::shared_ptr<FlowMsg>> output_msgs;
  ret = flow_model.Run({input}, output_msgs, 0);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  EXPECT_EQ(output_msgs.size(), 2);
}

TEST_F(FlowModelImplUTest, RunFlowGraphAlign) {
  std::unique_ptr<DataAligner> data_aligner;
  data_aligner.reset(new (std::nothrow) DataAligner(2, 10, 1000, false));
  FlowModelImpl flow_model(UdfTestHelper::CreateQueueDevInfos({1}), UdfTestHelper::CreateQueueDevInfos({3, 4}),
                           std::move(data_aligner));
  auto ret = flow_model.Init();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  MbufHead head_msg;
  auto input = MbufFlowMsg::AllocTensorMsg({0}, TensorDataType::DT_INT32, 0, head_msg, true);
  std::vector<std::shared_ptr<FlowMsg>> output_msgs;
  ret = flow_model.Run({input}, output_msgs, 0);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  EXPECT_EQ(output_msgs.size(), 2);
}

TEST_F(FlowModelImplUTest, RunFlowGraph_Dequeue_failed_timeout_zero) {
  MOCKER(halQueueDeQueue).stubs().will(returnValue(DRV_ERROR_INNER_ERR));
  FlowModelImpl flow_model(UdfTestHelper::CreateQueueDevInfos({1}), UdfTestHelper::CreateQueueDevInfos({3, 4}));
  auto ret = flow_model.Init();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  MbufHead head_msg;
  auto input = MbufFlowMsg::AllocTensorMsg({0}, TensorDataType::DT_INT32, 0, head_msg, true);
  std::vector<std::shared_ptr<FlowMsg>> output_msgs;
  ret = flow_model.Run({input}, output_msgs, 0);
  EXPECT_NE(ret, FLOW_FUNC_SUCCESS);
}

TEST_F(FlowModelImplUTest, RunFlowGraph_Dequeue_failed_timeout_not_zero) {
  MOCKER(halQueueDeQueue).stubs().will(returnValue(DRV_ERROR_INNER_ERR));
  FlowModelImpl flow_model(UdfTestHelper::CreateQueueDevInfos({1}), UdfTestHelper::CreateQueueDevInfos({3, 4}));
  auto ret = flow_model.Init();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  MbufHead head_msg;
  auto input = MbufFlowMsg::AllocTensorMsg({0}, TensorDataType::DT_INT32, 0, head_msg, true);
  std::vector<std::shared_ptr<FlowMsg>> output_msgs;
  ret = flow_model.Run({input}, output_msgs, 1000);
  EXPECT_NE(ret, FLOW_FUNC_SUCCESS);
}

TEST_F(FlowModelImplUTest, RunFlowGraph_on_device) {
  GlobalConfig::on_device_ = true;
  FlowModelImpl flow_model(UdfTestHelper::CreateQueueDevInfos({1}), UdfTestHelper::CreateQueueDevInfos({3, 4}));
  auto ret = flow_model.Init();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  MbufHead head_msg;
  auto input = MbufFlowMsg::AllocTensorMsg({0}, TensorDataType::DT_INT32, 0, head_msg, true);
  std::vector<std::shared_ptr<FlowMsg>> output_msgs;
  ret = flow_model.Run({input}, output_msgs, 0);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  EXPECT_EQ(output_msgs.size(), 2);
}

TEST_F(FlowModelImplUTest, RunFlowGraph_first_empty) {
  GlobalConfig::Instance().SetExitFlag(false);
  MOCKER(halQueueDeQueue).stubs().will(repeat(DRV_ERROR_QUEUE_EMPTY, 2)).then(invoke(halQueueDeQueueStub));
  FlowModelImpl flow_model(UdfTestHelper::CreateQueueDevInfos({1}), UdfTestHelper::CreateQueueDevInfos({3, 4}));
  auto ret = flow_model.Init();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  MbufHead head_msg;
  auto input = MbufFlowMsg::AllocTensorMsg({0}, TensorDataType::DT_INT32, 0, head_msg, true);
  std::vector<std::shared_ptr<FlowMsg>> output_msgs;
  ret = flow_model.Run({input}, output_msgs, 1000);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  EXPECT_EQ(output_msgs.size(), 2);
}

TEST_F(FlowModelImplUTest, RunFlowGraph_first_empty_on_device) {
  GlobalConfig::Instance().SetExitFlag(false);
  GlobalConfig::on_device_ = true;
  MOCKER(halQueueDeQueue).stubs().will(repeat(DRV_ERROR_QUEUE_EMPTY, 2)).then(invoke(halQueueDeQueueStub));
  FlowModelImpl flow_model(UdfTestHelper::CreateQueueDevInfos({1}), UdfTestHelper::CreateQueueDevInfos({3, 4}));
  auto ret = flow_model.Init();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  MbufHead head_msg;
  auto input = MbufFlowMsg::AllocTensorMsg({0}, TensorDataType::DT_INT32, 0, head_msg, true);
  std::vector<std::shared_ptr<FlowMsg>> output_msgs;
  ret = flow_model.Run({input}, output_msgs, 1000);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  EXPECT_EQ(output_msgs.size(), 2);
}

TEST_F(FlowModelImplUTest, RunFlowGraph_dequeue_empty_0_timeout) {
  MOCKER(halQueueDeQueue).stubs().will(repeat(DRV_ERROR_QUEUE_EMPTY, 2)).then(invoke(halQueueDeQueueStub));
  FlowModelImpl flow_model(UdfTestHelper::CreateQueueDevInfos({1}), UdfTestHelper::CreateQueueDevInfos({3, 4}));
  auto ret = flow_model.Init();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  MbufHead head_msg;
  auto input = MbufFlowMsg::AllocTensorMsg({0}, TensorDataType::DT_INT32, 0, head_msg, true);
  std::vector<std::shared_ptr<FlowMsg>> output_msgs;
  ret = flow_model.Run({input}, output_msgs, 0);
  EXPECT_NE(ret, FLOW_FUNC_SUCCESS);
}

TEST_F(FlowModelImplUTest, RunFlowGraph_dequeue_empty_wait_timeout) {
  MOCKER(halQueueDeQueue).stubs().will(returnValue(DRV_ERROR_QUEUE_EMPTY));
  MOCKER(halEschedWaitEvent).stubs().will(returnValue(DRV_ERROR_SCHED_WAIT_TIMEOUT));
  FlowModelImpl flow_model(UdfTestHelper::CreateQueueDevInfos({1}), UdfTestHelper::CreateQueueDevInfos({3, 4}));
  auto ret = flow_model.Init();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  MbufHead head_msg;
  auto input = MbufFlowMsg::AllocTensorMsg({0}, TensorDataType::DT_INT32, 0, head_msg, true);
  std::vector<std::shared_ptr<FlowMsg>> output_msgs;
  ret = flow_model.Run({input}, output_msgs, 2000);
  EXPECT_NE(ret, FLOW_FUNC_SUCCESS);
}

TEST_F(FlowModelImplUTest, RunFlowGraph_Dequeue_failed_raise_exception) {
  MOCKER(halQueueDeQueue).stubs().will(returnValue(DRV_ERROR_QUEUE_EMPTY));
  MOCKER(halEschedWaitEvent).stubs().will(returnValue(DRV_ERROR_SCHED_WAIT_TIMEOUT));
  std::unique_ptr<DataAligner> data_aligner;
  data_aligner.reset(new (std::nothrow) DataAligner(2, 10, 1000, false));
  FlowModelImpl flow_model(UdfTestHelper::CreateQueueDevInfos({1}), UdfTestHelper::CreateQueueDevInfos({3, 4}),
                           std::move(data_aligner));
  auto ret = flow_model.Init();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  MbufHead head_msg;
  auto input = MbufFlowMsg::AllocTensorMsg({0}, TensorDataType::DT_INT32, 0, head_msg, true);
  input->SetTransactionIdInner(2);
  flow_model.can_send_event_ = true;
  flow_model.current_trans_id_ = 2;
  flow_model.AddExceptionTransId(1);
  flow_model.AddExceptionTransId(2);
  flow_model.AddExceptionTransId(3);
  std::vector<std::shared_ptr<FlowMsg>> output_msgs;
  ret = flow_model.Run({input}, output_msgs, 1000);
  EXPECT_NE(ret, FLOW_FUNC_SUCCESS);
  flow_model.DeleteExceptionTransId(3);
}

TEST_F(FlowModelImplUTest, RunFlowGraph_dequeue_empty_while_suspend) {
  auto status_back = GlobalConfig::Instance().GetAbnormalStatus();
  GlobalConfig::Instance().SetAbnormalStatus(true);
  MOCKER(halQueueDeQueue).stubs().will(returnValue(DRV_ERROR_QUEUE_EMPTY));
  MOCKER(halEschedWaitEvent).stubs().will(returnValue(DRV_ERROR_SCHED_WAIT_TIMEOUT));
  FlowModelImpl flow_model(UdfTestHelper::CreateQueueDevInfos({1}), UdfTestHelper::CreateQueueDevInfos({3, 4}));
  auto ret = flow_model.Init();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  MbufHead head_msg;
  auto input = MbufFlowMsg::AllocTensorMsg({0}, TensorDataType::DT_INT32, 0, head_msg, true);
  std::vector<std::shared_ptr<FlowMsg>> output_msgs;
  ret = flow_model.Run({input}, output_msgs, 2000);
  EXPECT_EQ(ret, FLOW_FUNC_STATUS_REDEPLOYING);
  GlobalConfig::Instance().SetAbnormalStatus(status_back);
}

TEST_F(FlowModelImplUTest, Run_param_check) {
  FlowModelImpl flow_model(UdfTestHelper::CreateQueueDevInfos({1}), UdfTestHelper::CreateQueueDevInfos({3, 4}));
  auto ret = flow_model.Init();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  std::vector<std::shared_ptr<FlowMsg>> output_msgs;
  ret = flow_model.Run({nullptr}, output_msgs, 1000);
  EXPECT_NE(ret, FLOW_FUNC_SUCCESS);
}

TEST_F(FlowModelImplUTest, RunFlowGraph_enqueue_full) {
  MOCKER(halQueueEnQueue).stubs().will(returnValue(DRV_ERROR_QUEUE_FULL));
  FlowModelImpl flow_model(UdfTestHelper::CreateQueueDevInfos({1}), UdfTestHelper::CreateQueueDevInfos({3, 4}));
  auto ret = flow_model.Init();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  MbufHead head_msg;
  auto input = MbufFlowMsg::AllocTensorMsg({0}, TensorDataType::DT_INT32, 0, head_msg, true);
  std::vector<std::shared_ptr<FlowMsg>> output_msgs;
  ret = flow_model.Run({input}, output_msgs, 1000);
  EXPECT_EQ(ret, FLOW_FUNC_FAILED);
}

TEST_F(FlowModelImplUTest, RunFlowGraph_with_proxy) {
  GlobalConfig::Instance().SetPhyDeviceId(0);
  FlowModelImpl flow_model(UdfTestHelper::CreateQueueDevInfos({1}, Common::kDeviceTypeNpu, 1, true),
                           UdfTestHelper::CreateQueueDevInfos({3, 4}, Common::kDeviceTypeNpu, 1, true));
  auto ret = flow_model.Init();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  MbufHead head_msg;
  auto input = MbufFlowMsg::AllocTensorMsg({0}, TensorDataType::DT_INT32, 0, head_msg, true);
  std::vector<std::shared_ptr<FlowMsg>> output_msgs;
  ret = flow_model.Run({input}, output_msgs, 0);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  EXPECT_EQ(output_msgs.size(), 2);
}
}  // namespace FlowFunc