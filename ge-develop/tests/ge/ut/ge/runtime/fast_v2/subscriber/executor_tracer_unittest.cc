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
#include <memory>
#include "common/bg_test.h"
#include "common/global_variables/diagnose_switch.h"
#include "depends/profiler/src/profiling_test_util.h"
#include "common/model_v2_executor_test_helper.h"
#include "common/fake_node_helper.h"
#include "core/model_v2_executor_unittest.h"
#include "stub/gert_runtime_stub.h"
#include "subscriber/tracer/executor_tracer.h"
#include "runtime/model_v2_executor.h"
#include "framework/runtime/subscriber/global_tracer.h"

namespace gert {
namespace {
std::vector<std::string> TestTraceFunc(const KernelContext *) {
  return {""};
}
std::vector<std::string> TestTraceFuncWithSuperLongLog(const KernelContext *) {
  std::stringstream ss;
  for (size_t i = 0; i < MSG_LENGTH * 2; ++i) {
    ss << "a";
  }
  return {ss.str()};
}
}
class ExecutorTracerUT : public bg::BgTest {
  void SetUp() override {
    old_log_level_ = dlog_getlevel(GE_MODULE_NAME, &enable_event_);
  }
  void TearDown() override {
    dlog_setlevel(GE_MODULE_NAME, old_log_level_, enable_event_);
  }
 private:
  int old_log_level_;
  int enable_event_;
};

TEST_F(ExecutorTracerUT, GetEnableFlagOK) {
  dlog_setlevel(GE_MODULE_NAME, DLOG_WARN, 0);
  EXPECT_EQ(GlobalTracer::GetInstance()->GetEnableFlags(), 0);
  dlog_setlevel(GE_MODULE_NAME, DLOG_INFO, 0);
  EXPECT_EQ(GlobalTracer::GetInstance()->GetEnableFlags(), 1);
  dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 0);
  EXPECT_EQ(GlobalTracer::GetInstance()->GetEnableFlags(), 0);
  dlog_setlevel(GE_MODULE_NAME, DLOG_DEBUG, 0);
  EXPECT_EQ(GlobalTracer::GetInstance()->GetEnableFlags(), 1);
}

TEST_F(ExecutorTracerUT, OnExecuteEventWithTraceFuncOK) {
  REGISTER_KERNEL(KernelRegistryTest1).TracePrinter(TestTraceFunc);
  ExecutorTracer tracer(nullptr);
  dlog_setlevel(GE_MODULE_NAME, DLOG_DEBUG, 0);
  auto node_holder = FakeNodeHelper::FakeNode("test1", "test2", 1);
  EXPECT_NO_THROW(ExecutorTracer::OnExecuteEvent(0, &tracer, kExecuteEnd, &node_holder.node, 0));
}

TEST_F(ExecutorTracerUT, OnExecuteEventJustReturn) {
  ExecutorTracer tracer(nullptr);
  EXPECT_NO_THROW(ExecutorTracer::OnExecuteEvent(0, nullptr, kExecuteStart, nullptr, 0));

  auto node_holder = FakeNodeHelper::FakeNode("test1", "test2", 1);
  EXPECT_NO_THROW(ExecutorTracer::OnExecuteEvent(0, &tracer, kExecuteStart, &node_holder.node, 0));

  dlog_setlevel(GE_MODULE_NAME, DLOG_DEBUG, 0);
  EXPECT_NO_THROW(ExecutorTracer::OnExecuteEvent(0, &tracer, kExecuteStart, &node_holder.node, 0));
}

TEST_F(ExecutorTracerUT, OnExecuteEventWithoutTraceFuncOK) {
  ExecutorTracer tracer(nullptr);
  dlog_setlevel(GE_MODULE_NAME, DLOG_DEBUG, 0);
  auto node_holder = FakeNodeHelper::FakeNode("test1", "test2", 1);
  EXPECT_NO_THROW(ExecutorTracer::OnExecuteEvent(0, &tracer, kExecuteEnd, &node_holder.node, 0));
}

TEST_F(ExecutorTracerUT, TracePrinterWithSuperLongLogOK) {
  REGISTER_KERNEL(KernelRegistryTest1).TracePrinter(TestTraceFuncWithSuperLongLog);
  ExecutorTracer tracer(nullptr);
  GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().SetLevelInfo();
  auto node_holder = FakeNodeHelper::FakeNode("test1", "KernelRegistryTest1", 1);
  ExecutorTracer::OnExecuteEvent(0, &tracer, kExecuteEnd, &node_holder.node, 0);
  auto logs = runtime_stub.GetSlogStub().GetLogs();
  size_t kernel_trace_log_num = 0U;
  size_t kernel_race_log_len = 0U;
  std::string key_word = "KernelRegistryTest1";
  for (const auto &log : logs) {
    if (log.content.find("KernelTrace") != string::npos) {
      kernel_trace_log_num++;
    }
    auto index = log.content.find(key_word);
    if (index != std::string::npos) {
      kernel_race_log_len += (log.content.size() - index - key_word.size() - 1U);
    }
  }
  ASSERT_EQ(kernel_race_log_len, MSG_LENGTH * 2);
  ASSERT_TRUE(kernel_trace_log_num > 1);
  runtime_stub.Clear();
}

TEST_F(ExecutorTracerUT, TraceLogTag_Ok_OnModelStart) {
  GertRuntimeStub stub;
  ExecutorTracer tracer(nullptr);
  dlog_setlevel(GE_MODULE_NAME, DLOG_DEBUG, 0);
  stub.GetSlogStub().Clear();
  ExecutorTracer::OnExecuteEvent(0, &tracer, kModelStart, nullptr, 0);
  ASSERT_EQ(stub.GetSlogStub().GetLogs().size(), 1);
  ASSERT_EQ(stub.GetSlogStub().GetLogs().at(0).level, DLOG_INFO);
  std::string expect_log = "[KernelTrace] Start model tracing.";
  EXPECT_EQ(stub.GetSlogStub().GetLogs().at(0).content.find_last_of(expect_log),
            stub.GetSlogStub().GetLogs().at(0).content.size() - 1);
}

TEST_F(ExecutorTracerUT, TraceLogTag_Ok_OnModelEnd) {
  GertRuntimeStub stub;
  ExecutorTracer tracer(nullptr);
  dlog_setlevel(GE_MODULE_NAME, DLOG_DEBUG, 0);
  stub.GetSlogStub().Clear();
  ExecutorTracer::OnExecuteEvent(0, &tracer, kModelEnd, nullptr, 0);
  ASSERT_EQ(stub.GetSlogStub().GetLogs().size(), 1);
  ASSERT_EQ(stub.GetSlogStub().GetLogs().at(0).level, DLOG_INFO);
  std::string expect_log = "[KernelTrace] End model tracing.";
  EXPECT_EQ(stub.GetSlogStub().GetLogs().at(0).content.find_last_of(expect_log),
            stub.GetSlogStub().GetLogs().at(0).content.size() - 1);
}
}  // namespace gert
