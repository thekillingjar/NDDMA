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
#include <vector>
#include "macro_utils/dt_public_scope.h"
#include "dflow/executor/data_flow_exception_handler.h"
#include "macro_utils/dt_public_unscope.h"

using namespace std;
using namespace testing;

namespace ge {
class DataFlowExceptionHandlerTest : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(DataFlowExceptionHandlerTest, exception_not_enable) {
  DataFlowExceptionHandler handler(nullptr);
  EXPECT_EQ(handler.exception_notify_callback_, nullptr);
  InnerProcessMsgForwarding forwarding;
  EXPECT_EQ(handler.Initialize(forwarding), SUCCESS);
  EXPECT_FALSE(handler.process_thread_.joinable());
  handler.Finalize();
}

TEST_F(DataFlowExceptionHandlerTest, exception_enable) {
  std::mutex mt;
  std::condition_variable condition;
  std::set<uint64_t> exception_trans_id;
  auto func = [&mt, &condition, &exception_trans_id](const UserExceptionNotify &excep) {
    std::lock_guard<std::mutex> lk(mt);
    exception_trans_id.emplace(excep.trans_id);
    condition.notify_all();
  };
  DataFlowExceptionHandler handler(func);
  InnerProcessMsgForwarding forwarding;
  EXPECT_EQ(handler.Initialize(forwarding), SUCCESS);
  auto callback = forwarding.callbacks_[StatusQueueMsgType::EXCEPTION];
  EXPECT_NE(callback, nullptr);
  domi::SubmodelStatus request;
  auto excep = request.mutable_exception();
  excep->set_trans_id(100);
  {
    callback(request);
    std::unique_lock<std::mutex> lock(mt);
    condition.wait_for(lock, std::chrono::milliseconds(10000),
                       [&exception_trans_id]() { return !exception_trans_id.empty(); });
    ASSERT_EQ(exception_trans_id.size(), 1);
    EXPECT_EQ(*exception_trans_id.begin(), 100);
    exception_trans_id.clear();
  }

  {
    // repeat trans id exception
    callback(request);
    excep->set_trans_id(200);
    callback(request);
    std::unique_lock<std::mutex> lock(mt);
    condition.wait_for(lock, std::chrono::milliseconds(1000),
                       [&exception_trans_id]() { return !exception_trans_id.empty(); });
    ASSERT_EQ(exception_trans_id.size(), 1);
    EXPECT_EQ(*exception_trans_id.begin(), 200);
    exception_trans_id.clear();
  }
  handler.Finalize();
}

TEST_F(DataFlowExceptionHandlerTest, check_modelio_exception) {
  std::mutex mt;
  std::condition_variable condition;
  std::set<uint64_t> exception_trans_id;
  auto func = [&mt, &condition, &exception_trans_id](const UserExceptionNotify &excep) {
    std::lock_guard<std::mutex> lk(mt);
    exception_trans_id.emplace(excep.trans_id);
    condition.notify_all();
  };
  std::mutex aligner_mt;
  std::condition_variable aligner_condition;
  std::set<uint64_t> aligner_exceptions;
  auto aligner_callback = [&aligner_mt, &aligner_condition, &aligner_exceptions](uint64_t trans_id, uint32_t type) {
    std::lock_guard<std::mutex> lk(aligner_mt);
    if (type == kExceptionTypeOccured) {
      aligner_exceptions.emplace(trans_id);
    } else {
      aligner_exceptions.erase(trans_id);
    }
    aligner_condition.notify_all();
  };
  DataFlowExceptionHandler handler(func);
  handler.RegisterModelIoExpTransIdCallback(aligner_callback);
  InnerProcessMsgForwarding forwarding;
  EXPECT_EQ(handler.Initialize(forwarding), SUCCESS);
  auto callback = forwarding.callbacks_[StatusQueueMsgType::EXCEPTION];
  EXPECT_NE(callback, nullptr);

  DataFlowInfo info;
  EXPECT_FALSE(handler.TakeWaitModelIoException(info));
  uint8_t head[256] = {};
  ExchangeService::MsgInfo *msg_info = reinterpret_cast<ExchangeService::MsgInfo *>(head + (256 - 64));
  domi::SubmodelStatus request;
  auto excep = request.mutable_exception();
  excep->set_trans_id(100);
  excep->set_scope("");
  msg_info->start_time = 100;
  excep->set_exception_context(head, sizeof(head));
  {
    callback(request);
    {
      std::unique_lock<std::mutex> lock(mt);
      condition.wait_for(lock, std::chrono::milliseconds(10000),
                         [&exception_trans_id]() { return !exception_trans_id.empty(); });
    }
    {
      std::unique_lock<std::mutex> lock(aligner_mt);
      aligner_condition.wait_for(lock, std::chrono::milliseconds(1000),
                                 [&aligner_exceptions]() { return !aligner_exceptions.empty(); });
    }
    ASSERT_EQ(exception_trans_id.size(), 1);
    EXPECT_EQ(*exception_trans_id.begin(), 100);
    exception_trans_id.clear();
    aligner_exceptions.clear();
  }

  {
    // repeat trans id exception
    callback(request);
    excep->set_trans_id(200);
    excep->set_scope("not model io scope");
    msg_info->start_time = 200;
    excep->set_exception_context(head, sizeof(head));
    callback(request);
    excep->set_trans_id(300);
    excep->set_scope("");
    msg_info->start_time = 300;
    excep->set_exception_context(head, sizeof(head));
    callback(request);
    {
      std::unique_lock<std::mutex> lock(mt);
      condition.wait_for(lock, std::chrono::milliseconds(1000),
                         [&exception_trans_id]() { return exception_trans_id.size() >= 2; });
    }
    {
      std::unique_lock<std::mutex> lock(aligner_mt);
      aligner_condition.wait_for(lock, std::chrono::milliseconds(1000),
                                 [&aligner_exceptions]() { return !aligner_exceptions.empty(); });
    }
    EXPECT_EQ(exception_trans_id.size(), 2);
    EXPECT_EQ(exception_trans_id, std::set<uint64_t>({200, 300}));
    ASSERT_EQ(aligner_exceptions.size(), 1);
    ASSERT_EQ(aligner_exceptions, std::set<uint64_t>({300}));
    exception_trans_id.clear();
  }
  EXPECT_TRUE(handler.TakeWaitModelIoException(info));
  EXPECT_EQ(info.GetStartTime(), 100);
  EXPECT_TRUE(handler.TakeWaitModelIoException(info));
  EXPECT_EQ(info.GetStartTime(), 300);

  EXPECT_TRUE(handler.IsModelIoIgnoreTransId(100));
  EXPECT_FALSE(handler.IsModelIoIgnoreTransId(200));
  EXPECT_TRUE(handler.IsModelIoIgnoreTransId(300));
  handler.Finalize();
}

TEST_F(DataFlowExceptionHandlerTest, exception_over_limit) {
  std::mutex mt;
  std::condition_variable condition;
  std::set<uint64_t> exception_trans_id;
  bool has_erase = false;
  auto func = [&mt, &condition, &exception_trans_id, &has_erase](const UserExceptionNotify &excep) {
    std::lock_guard<std::mutex> lk(mt);
    if (excep.type == kExceptionTypeOccured) {
      exception_trans_id.emplace(excep.trans_id);
    } else {
      exception_trans_id.erase(excep.trans_id);
      has_erase = true;
    }
    condition.notify_all();
  };
  DataFlowExceptionHandler handler(func);
  InnerProcessMsgForwarding forwarding;
  EXPECT_EQ(handler.Initialize(forwarding), SUCCESS);
  auto callback = forwarding.callbacks_[StatusQueueMsgType::EXCEPTION];
  EXPECT_NE(callback, nullptr);
  domi::SubmodelStatus request;
  auto excep = request.mutable_exception();
  size_t i = 0;
  while (i++ < 1024) {
    excep->set_trans_id(i);
    callback(request);
  }
  {
    excep->set_trans_id(i);
    callback(request);
    std::unique_lock<std::mutex> lock(mt);
    condition.wait_for(lock, std::chrono::milliseconds(10000),
                       [i, &exception_trans_id]() { return exception_trans_id.find(i) != exception_trans_id.cend(); });
    EXPECT_TRUE(has_erase);
    EXPECT_EQ(exception_trans_id.size(), 1024);
  }
  handler.Finalize();
}
}  // namespace ge
