/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <thread>
#include "gtest/gtest.h"
#include "securec.h"
#include "ascend_hal.h"
#include "mockcpp/mockcpp.hpp"
#include "flow_func/mbuf_flow_msg_queue.h"
#include "flow_func/mbuf_flow_msg.h"
#include "config/global_config.h"
#include "flow_func/flow_func_config_manager.h"
#include "reader_writer/queue_wrapper.h"
#include "common/inner_error_codes.h"
#include "utils/udf_test_helper.h"
#include "common/common_define.h"

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

int halMbufGetBuffAddrStub(Mbuf *mbuf, void **buf) {
  MbufImpl *mbuf_impl = reinterpret_cast<MbufImpl *>(mbuf);
  *buf = &(mbuf_impl->tensor_desc);
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

drvError_t halQueueGetStatusStub(unsigned int dev_id, unsigned int qid, QUEUE_QUERY_ITEM query_item, unsigned int len,
                                 void *data) {
  *static_cast<int32_t *>(data) = 128;
  return DRV_ERROR_NONE;
}
}  // namespace
class MbufFlowMsgQueueUTest : public testing::Test {
 protected:
  virtual void SetUp() {
    MOCKER(halMbufAllocEx).defaults().will(invoke(halMbufAllocExStub));
    MOCKER(halMbufFree).defaults().will(invoke(halMbufFreeStub));
    MOCKER(halMbufGetBuffAddr).defaults().will(invoke(halMbufGetBuffAddrStub));
    MOCKER(halQueueDeQueue).defaults().will(invoke(halQueueDeQueueStub));
  }

  virtual void TearDown() {
    GlobalMockObject::verify();
  }

  QueueDevInfo queue_dev_info_ = UdfTestHelper::CreateQueueDevInfo(0);
  QueueDevInfo proxy_queue_dev_info_ = UdfTestHelper::CreateQueueDevInfo(1, Common::kDeviceTypeNpu, 0, true);
  std::shared_ptr<QueueWrapper> queue_wrapper_ = std::make_shared<QueueWrapper>(0, 0);
};

TEST_F(MbufFlowMsgQueueUTest, dequeue_timeout_zero_queue_empty) {
  MOCKER(halQueueDeQueue).stubs().will(returnValue(DRV_ERROR_QUEUE_EMPTY));
  MbufFlowMsgQueue mbuf_flow_msg_queue(queue_wrapper_, queue_dev_info_);
  std::shared_ptr<FlowMsg> flow_msg;
  auto ret = mbuf_flow_msg_queue.Dequeue(flow_msg, 0);
  EXPECT_NE(ret, FLOW_FUNC_SUCCESS);
}

TEST_F(MbufFlowMsgQueueUTest, dequeue_timeout_zero_failed) {
  MOCKER(halQueueDeQueue).stubs().will(returnValue(DRV_ERROR_NOT_SUPPORT));
  MbufFlowMsgQueue mbuf_flow_msg_queue(queue_wrapper_, queue_dev_info_);
  std::shared_ptr<FlowMsg> flow_msg;
  auto ret = mbuf_flow_msg_queue.Dequeue(flow_msg, 0);
  EXPECT_EQ(ret, FLOW_FUNC_ERR_QUEUE_ERROR);
}

TEST_F(MbufFlowMsgQueueUTest, dequeue_timeout_zero_mbuf_init_failed) {
  MOCKER_CPP(&MbufFlowMsg::Init).stubs().will(returnValue(FLOW_FUNC_FAILED));
  MbufFlowMsgQueue mbuf_flow_msg_queue(queue_wrapper_, queue_dev_info_);
  std::shared_ptr<FlowMsg> flow_msg;
  auto ret = mbuf_flow_msg_queue.Dequeue(flow_msg, 0);
  EXPECT_EQ(ret, FLOW_FUNC_FAILED);
}

TEST_F(MbufFlowMsgQueueUTest, dequeue_timeout_zero_success) {
  MOCKER_CPP(&MbufFlowMsg::Init).stubs().will(returnValue(FLOW_FUNC_SUCCESS));
  MbufFlowMsgQueue mbuf_flow_msg_queue(queue_wrapper_, queue_dev_info_);
  std::shared_ptr<FlowMsg> flow_msg;
  auto ret = mbuf_flow_msg_queue.Dequeue(flow_msg, 0);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
}

TEST_F(MbufFlowMsgQueueUTest, dequeue_timeout_none_zero_success) {
  MOCKER(halQueueDeQueue).stubs().will(returnValue(DRV_ERROR_QUEUE_EMPTY)).then(invoke(halQueueDeQueueStub));
  MOCKER_CPP(&MbufFlowMsg::Init).stubs().will(returnValue(FLOW_FUNC_SUCCESS));
  MbufFlowMsgQueue mbuf_flow_msg_queue(queue_wrapper_, queue_dev_info_);
  std::shared_ptr<FlowMsg> flow_msg;
  auto ret = mbuf_flow_msg_queue.Dequeue(flow_msg, -1);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
}

TEST_F(MbufFlowMsgQueueUTest, dequeue_timeout_none_zero_failed) {
  GlobalConfig::Instance().SetExitFlag(false);
  MOCKER(halQueueDeQueue).stubs().will(returnValue(DRV_ERROR_QUEUE_EMPTY));
  MOCKER_CPP(&MbufFlowMsg::Init).stubs().will(returnValue(FLOW_FUNC_SUCCESS));
  MOCKER(halQueueSubEvent).stubs().will(returnValue(DRV_ERROR_NOT_SUPPORT)).then(returnValue(DRV_ERROR_NONE));
  std::shared_ptr<FlowMsg> flow_msg;
  MbufFlowMsgQueue mbuf_flow_msg_proxy_queue(queue_wrapper_, proxy_queue_dev_info_);  // proxy queue
  auto ret = mbuf_flow_msg_proxy_queue.Dequeue(flow_msg, 1);
  EXPECT_EQ(ret, FLOW_FUNC_ERR_TIME_OUT_ERROR);
  MbufFlowMsgQueue mbuf_flow_msg_queue(queue_wrapper_, queue_dev_info_);
  ret = mbuf_flow_msg_queue.Dequeue(flow_msg, 1);
  EXPECT_EQ(ret, FLOW_FUNC_ERR_EVENT_ERROR);
  ret = mbuf_flow_msg_queue.Dequeue(flow_msg, 1);
  EXPECT_EQ(ret, FLOW_FUNC_ERR_TIME_OUT_ERROR);
  auto status = FlowFuncConfigManager::GetConfig()->GetAbnormalStatus();
  EXPECT_FALSE(status);
  GlobalConfig::Instance().SetAbnormalStatus(true);
  FlowFuncConfigManager::SetConfig(std::shared_ptr<FlowFuncConfig>(&GlobalConfig::Instance(), [](FlowFuncConfig *) {}));
  status = FlowFuncConfigManager::GetConfig()->GetAbnormalStatus();
  EXPECT_TRUE(status);
  ret = mbuf_flow_msg_queue.Dequeue(flow_msg, 1);
  EXPECT_EQ(ret, FLOW_FUNC_STATUS_REDEPLOYING);
  GlobalConfig::Instance().SetAbnormalStatus(false);
  GlobalConfig::Instance().SetExitFlag(true);
  ret = mbuf_flow_msg_queue.Dequeue(flow_msg, 1);
  EXPECT_EQ(ret, FLOW_FUNC_STATUS_EXIT);
  GlobalConfig::Instance().SetExitFlag(false);
}

TEST_F(MbufFlowMsgQueueUTest, query_depth_success) {
  MbufFlowMsgQueue mbuf_flow_msg_queue(queue_wrapper_, queue_dev_info_);
  MOCKER(halQueueGetStatus).stubs().will(returnValue(DRV_ERROR_NOT_SUPPORT)).then(invoke(halQueueGetStatusStub));
  auto q_depth = mbuf_flow_msg_queue.Depth();
  EXPECT_EQ(q_depth, 0);
  q_depth = mbuf_flow_msg_queue.Depth();
  EXPECT_EQ(q_depth, 128);
}

TEST_F(MbufFlowMsgQueueUTest, query_size_success) {
  MOCKER_CPP(&QueueWrapper::QueryQueueSize).stubs().will(returnValue(1));
  MbufFlowMsgQueue mbuf_flow_msg_queue(queue_wrapper_, queue_dev_info_);
  auto q_size = mbuf_flow_msg_queue.Size();
  EXPECT_EQ(q_size, 1);
}

TEST_F(MbufFlowMsgQueueUTest, discard_all_input_data_failed) {
  MOCKER_CPP(&QueueWrapper::DiscardMbuf).stubs().will(returnValue(HICAID_ERR_QUEUE_FAILED));
  MbufFlowMsgQueue mbuf_flow_msg_queue(queue_wrapper_, queue_dev_info_);
  mbuf_flow_msg_queue.DiscardAllInputData();
  EXPECT_FALSE(mbuf_flow_msg_queue.StatusOk());
}
}  // namespace FlowFunc