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
#include "ascend_hal.h"
#include "mockcpp/mockcpp.hpp"
#include "common/data_utils.h"
#include "reader_writer/proxy_queue_wrapper.h"
#include "common/inner_error_codes.h"

namespace FlowFunc {
namespace {
struct MbufImpl {
  uint8_t reserve_head[256];
  uint8_t data[2048];
};
}  // namespace
class ProxyQueueWrapperDataUtilUTest : public testing::Test {
 protected:
  virtual void SetUp() {}

  virtual void TearDown() {
    GlobalMockObject::verify();
  }

  ProxyQueueWrapper proxy_queue_wrapper_{0, 0};
};

TEST_F(ProxyQueueWrapperDataUtilUTest, Enqueue_param_check) {
  auto ret = proxy_queue_wrapper_.Enqueue(nullptr);
  EXPECT_NE(ret, HICAID_SUCCESS);
}

TEST_F(ProxyQueueWrapperDataUtilUTest, Enqueue_halMbufGetPrivInfo_failed) {
  MbufImpl impl;
  Mbuf *buf = (Mbuf *)&impl;
  MOCKER(halMbufGetPrivInfo).stubs().will(returnValue(static_cast<int32_t>(DRV_ERROR_INNER_ERR)));
  auto ret = proxy_queue_wrapper_.Enqueue(buf);
  EXPECT_NE(ret, HICAID_SUCCESS);
}

TEST_F(ProxyQueueWrapperDataUtilUTest, Enqueue_halMbufGetBuffAddr_failed) {
  MbufImpl impl;
  Mbuf *buf = (Mbuf *)&impl;
  MOCKER(halMbufGetPrivInfo).stubs().will(returnValue(static_cast<int32_t>(DRV_ERROR_NONE)));
  MOCKER(halMbufGetBuffAddr).stubs().will(returnValue(static_cast<int32_t>(DRV_ERROR_INNER_ERR)));
  auto ret = proxy_queue_wrapper_.Enqueue(buf);
  EXPECT_NE(ret, HICAID_SUCCESS);
}

TEST_F(ProxyQueueWrapperDataUtilUTest, Enqueue_halMbufGetDataLen_failed) {
  MbufImpl impl;
  Mbuf *buf = (Mbuf *)&impl;
  MOCKER(halMbufGetPrivInfo).stubs().will(returnValue(static_cast<int32_t>(DRV_ERROR_NONE)));
  MOCKER(halMbufGetBuffAddr).stubs().will(returnValue(static_cast<int32_t>(DRV_ERROR_NONE)));
  MOCKER(halMbufGetDataLen).stubs().will(returnValue(static_cast<int32_t>(DRV_ERROR_INNER_ERR)));
  auto ret = proxy_queue_wrapper_.Enqueue(buf);
  EXPECT_NE(ret, HICAID_SUCCESS);
}

TEST_F(ProxyQueueWrapperDataUtilUTest, Enqueue_halQueueEnQueueBuff_failed) {
  MbufImpl impl;
  Mbuf *buf = (Mbuf *)&impl;
  MOCKER(halMbufGetPrivInfo).stubs().will(returnValue(static_cast<int32_t>(DRV_ERROR_NONE)));
  MOCKER(halMbufGetBuffAddr).stubs().will(returnValue(static_cast<int32_t>(DRV_ERROR_NONE)));
  MOCKER(halMbufGetDataLen).stubs().will(returnValue(static_cast<int32_t>(DRV_ERROR_NONE)));
  MOCKER(halQueueEnQueueBuff).stubs().will(returnValue(static_cast<int32_t>(DRV_ERROR_INNER_ERR)));
  auto ret = proxy_queue_wrapper_.Enqueue(buf);
  EXPECT_NE(ret, HICAID_SUCCESS);
}

TEST_F(ProxyQueueWrapperDataUtilUTest, Enqueue_halQueueEnQueueBuff_full) {
  MbufImpl impl;
  Mbuf *buf = (Mbuf *)&impl;
  MOCKER(halMbufGetPrivInfo).stubs().will(returnValue(static_cast<int32_t>(DRV_ERROR_NONE)));
  MOCKER(halMbufGetBuffAddr).stubs().will(returnValue(static_cast<int32_t>(DRV_ERROR_NONE)));
  MOCKER(halMbufGetDataLen).stubs().will(returnValue(static_cast<int32_t>(DRV_ERROR_NONE)));
  MOCKER(halQueueEnQueueBuff).stubs().will(returnValue(static_cast<int32_t>(DRV_ERROR_QUEUE_FULL)));
  auto ret = proxy_queue_wrapper_.Enqueue(buf);
  EXPECT_EQ(ret, HICAID_ERR_QUEUE_FULL);
}

TEST_F(ProxyQueueWrapperDataUtilUTest, Enqueue_SUCCESS) {
  MbufImpl impl;
  Mbuf *buf = (Mbuf *)&impl;
  MOCKER(halMbufGetPrivInfo).stubs().will(returnValue(static_cast<int32_t>(DRV_ERROR_NONE)));
  MOCKER(halMbufGetBuffAddr).stubs().will(returnValue(static_cast<int32_t>(DRV_ERROR_NONE)));
  MOCKER(halMbufGetDataLen).stubs().will(returnValue(static_cast<int32_t>(DRV_ERROR_NONE)));
  MOCKER(halQueueEnQueueBuff).stubs().will(returnValue(static_cast<int32_t>(DRV_ERROR_NONE)));
  MOCKER(halMbufFree).stubs().will(returnValue(static_cast<int32_t>(DRV_ERROR_NONE)));
  auto ret = proxy_queue_wrapper_.Enqueue(buf);
  EXPECT_EQ(ret, HICAID_SUCCESS);
}

TEST_F(ProxyQueueWrapperDataUtilUTest, Dequeue_halQueuePeek_empty) {
  Mbuf *buf = nullptr;
  MOCKER(halQueuePeek).stubs().will(returnValue(DRV_ERROR_QUEUE_EMPTY));
  auto ret = proxy_queue_wrapper_.Dequeue(buf);
  EXPECT_EQ(ret, HICAID_ERR_QUEUE_EMPTY);
}

TEST_F(ProxyQueueWrapperDataUtilUTest, Dequeue_halQueuePeek_failed) {
  Mbuf *buf = nullptr;
  MOCKER(halQueuePeek).stubs().will(returnValue(DRV_ERROR_INNER_ERR));
  auto ret = proxy_queue_wrapper_.Dequeue(buf);
  EXPECT_NE(ret, HICAID_SUCCESS);
}

TEST_F(ProxyQueueWrapperDataUtilUTest, Dequeue_halMbufAlloc_failed) {
  Mbuf *buf = nullptr;
  MOCKER(halQueuePeek).stubs().will(returnValue(DRV_ERROR_NONE));
  MOCKER(halMbufAlloc).stubs().will(returnValue(static_cast<int32_t>(DRV_ERROR_INNER_ERR)));
  auto ret = proxy_queue_wrapper_.Dequeue(buf);
  EXPECT_NE(ret, HICAID_SUCCESS);
}

TEST_F(ProxyQueueWrapperDataUtilUTest, Dequeue_halMbufGetPrivInfo_failed) {
  Mbuf *buf = nullptr;
  MOCKER(halQueuePeek).stubs().will(returnValue(DRV_ERROR_NONE));
  MOCKER(halMbufAlloc).stubs().will(returnValue(static_cast<int32_t>(DRV_ERROR_NONE)));
  MOCKER(halMbufFree).stubs().will(returnValue(static_cast<int32_t>(DRV_ERROR_NONE)));
  MOCKER(halMbufGetPrivInfo).stubs().will(returnValue(static_cast<int32_t>(DRV_ERROR_INNER_ERR)));
  auto ret = proxy_queue_wrapper_.Dequeue(buf);
  EXPECT_NE(ret, HICAID_SUCCESS);
}

TEST_F(ProxyQueueWrapperDataUtilUTest, Dequeue_halQueueDeQueueBuff_failed) {
  Mbuf *buf = nullptr;
  MOCKER(halQueuePeek).stubs().will(returnValue(DRV_ERROR_NONE));
  MOCKER(halMbufAlloc).stubs().will(returnValue(static_cast<int32_t>(DRV_ERROR_NONE)));
  MOCKER(halMbufFree).stubs().will(returnValue(static_cast<int32_t>(DRV_ERROR_NONE)));
  MOCKER(halMbufGetPrivInfo).stubs().will(returnValue(static_cast<int32_t>(DRV_ERROR_NONE)));
  MOCKER(halQueueDeQueueBuff).stubs().will(returnValue(DRV_ERROR_INNER_ERR));
  auto ret = proxy_queue_wrapper_.Dequeue(buf);
  EXPECT_NE(ret, HICAID_SUCCESS);
}

TEST_F(ProxyQueueWrapperDataUtilUTest, Dequeue_success) {
  Mbuf *buf = nullptr;
  MOCKER(halQueuePeek).stubs().will(returnValue(DRV_ERROR_NONE));
  MOCKER(halMbufAlloc).stubs().will(returnValue(static_cast<int32_t>(DRV_ERROR_NONE)));
  MOCKER(halMbufFree).stubs().will(returnValue(static_cast<int32_t>(DRV_ERROR_NONE)));
  MOCKER(halMbufGetPrivInfo).stubs().will(returnValue(static_cast<int32_t>(DRV_ERROR_NONE)));
  MOCKER(halQueueDeQueueBuff).stubs().will(returnValue(DRV_ERROR_NONE));
  auto ret = proxy_queue_wrapper_.Dequeue(buf);
  EXPECT_EQ(ret, HICAID_SUCCESS);
}

TEST_F(ProxyQueueWrapperDataUtilUTest, DiscardMbuffOnce_success) {
  Mbuf *buf = nullptr;
  MOCKER(halQueuePeek).stubs().will(returnValue(DRV_ERROR_NONE)).then(returnValue(DRV_ERROR_QUEUE_EMPTY));
  MOCKER(halMbufAlloc).stubs().will(returnValue(static_cast<int32_t>(DRV_ERROR_NONE)));
  MOCKER(halMbufFree).stubs().will(returnValue(static_cast<int32_t>(DRV_ERROR_NONE)));
  MOCKER(halMbufGetPrivInfo).stubs().will(returnValue(static_cast<int32_t>(DRV_ERROR_NONE)));
  MOCKER(halQueueDeQueueBuff).stubs().will(returnValue(DRV_ERROR_NONE));
  auto ret = proxy_queue_wrapper_.DiscardMbuf();
  EXPECT_EQ(ret, HICAID_SUCCESS);
}

}  // namespace FlowFunc