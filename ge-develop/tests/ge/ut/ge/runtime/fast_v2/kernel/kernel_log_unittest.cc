/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel/kernel_log.h"
#include <gtest/gtest.h>
#include "stub/gert_runtime_stub.h"
#include "faker/kernel_run_context_facker.h"
namespace gert {
class KernelLogUT : public testing::Test {
 public:
  ge::graphStatus TestNullptr(KernelContext *context, void *p) {
    KERNEL_CHECK_NOTNULL(p);
    return ge::GRAPH_SUCCESS;
  }
  ge::graphStatus TestSuccess(KernelContext *context, ge::graphStatus ret) {
    KERNEL_CHECK_SUCCESS(ret);
    return ge::GRAPH_SUCCESS;
  }
};
#define ASSERT_ENDSWITH(str, str2)                                                         \
  do {                                                                                     \
    std::string expect = str2;                                                             \
    ASSERT_EQ(str.find_last_of(expect), str.size() - 1) << "The log: " << str << std::endl \
                                                        << "NOT ends with:" << expect;     \
  } while (0)

TEST_F(KernelLogUT, KLog_Success_LogError) {
  GertRuntimeStub stub;
  auto context_holder = KernelRunContextFaker().KernelName("tn").KernelType("tt").Build();
  auto context = context_holder.GetContext<KernelContext>();

  stub.GetSlogStub().Clear();
  KLOGE("Hello world");
  KLOGE("Hello world %d", 123);

  ASSERT_EQ(stub.GetSlogStub().GetLogs().size(), 2);
  ASSERT_EQ(stub.GetSlogStub().GetLogs().at(0).level, DLOG_ERROR);
  ASSERT_EQ(stub.GetSlogStub().GetLogs().at(1).level, DLOG_ERROR);
  ASSERT_ENDSWITH(stub.GetSlogStub().GetLogs().at(0).content, "[tt][tn]Hello world");
  ASSERT_ENDSWITH(stub.GetSlogStub().GetLogs().at(1).content, "[tt][tn]Hello world 123");
}
TEST_F(KernelLogUT, KLog_Success_LogDiffTypes) {
  auto context_holder = KernelRunContextFaker().KernelName("tn").KernelType("tt").Build();
  auto context = context_holder.GetContext<KernelContext>();

  GertRuntimeStub stub;
  stub.GetSlogStub().SetLevel(DLOG_DEBUG);
  KLOGE("Hello world");
  KLOGW("Hello world %d", 123);
  KLOGI("Hello world %d", 456);
  KLOGD("Hello world %d", 789);
  KLOGE("Hello world");
  KLOGW("Hello world %d", 123);
  KLOGI("Hello world %d", 456);
  KLOGD("Hello world %d", 789);

  ASSERT_EQ(stub.GetSlogStub().GetLogs().size(), 8);
  ASSERT_EQ(stub.GetSlogStub().GetLogs().at(0).level, DLOG_ERROR);
  ASSERT_EQ(stub.GetSlogStub().GetLogs().at(1).level, DLOG_WARN);
  ASSERT_EQ(stub.GetSlogStub().GetLogs().at(2).level, DLOG_INFO);
  ASSERT_EQ(stub.GetSlogStub().GetLogs().at(3).level, DLOG_DEBUG);
  ASSERT_EQ(stub.GetSlogStub().GetLogs().at(4).level, DLOG_ERROR);
  ASSERT_EQ(stub.GetSlogStub().GetLogs().at(5).level, DLOG_WARN);
  ASSERT_EQ(stub.GetSlogStub().GetLogs().at(6).level, DLOG_INFO);
  ASSERT_EQ(stub.GetSlogStub().GetLogs().at(7).level, DLOG_DEBUG);
  ASSERT_ENDSWITH(stub.GetSlogStub().GetLogs().at(0).content, "[tt][tn]Hello world");
  ASSERT_ENDSWITH(stub.GetSlogStub().GetLogs().at(1).content, "[tt][tn]Hello world 123");
  ASSERT_ENDSWITH(stub.GetSlogStub().GetLogs().at(2).content, "[tt][tn]Hello world 456");
  ASSERT_ENDSWITH(stub.GetSlogStub().GetLogs().at(3).content, "[tt][tn]Hello world 789");
  ASSERT_ENDSWITH(stub.GetSlogStub().GetLogs().at(4).content, "[tt][tn]Hello world");
  ASSERT_ENDSWITH(stub.GetSlogStub().GetLogs().at(5).content, "[tt][tn]Hello world 123");
  ASSERT_ENDSWITH(stub.GetSlogStub().GetLogs().at(6).content, "[tt][tn]Hello world 456");
  ASSERT_ENDSWITH(stub.GetSlogStub().GetLogs().at(7).content, "[tt][tn]Hello world 789");
}
TEST_F(KernelLogUT, CheckNotNull_PrintError_WhenNull) {
  GertRuntimeStub stub;
  auto context_holder = KernelRunContextFaker().KernelName("tn").KernelType("tt").Build();

  stub.GetSlogStub().Clear();
  ASSERT_NE(TestNullptr(context_holder.GetContext<KernelContext>(), nullptr), ge::GRAPH_SUCCESS);
  ASSERT_EQ(stub.GetSlogStub().GetLogs().size(), 1);
}
TEST_F(KernelLogUT, CheckNotNull_NotReturn_WhenNotNull) {
  GertRuntimeStub stub;
  auto context_holder = KernelRunContextFaker().KernelName("tn").KernelType("tt").Build();

  stub.GetSlogStub().Clear();
  ASSERT_EQ(TestNullptr(context_holder.GetContext<KernelContext>(), (void *)"Hello"), ge::GRAPH_SUCCESS);
  ASSERT_EQ(stub.GetSlogStub().GetLogs().size(), 0);
}
TEST_F(KernelLogUT, CheckSuccess_PrintError_WhenFailed) {
  GertRuntimeStub stub;
  auto context_holder = KernelRunContextFaker().KernelName("tn").KernelType("tt").Build();

  stub.GetSlogStub().Clear();
  ASSERT_NE(TestSuccess(context_holder.GetContext<KernelContext>(), ge::FAILED), ge::GRAPH_SUCCESS);
  ASSERT_EQ(stub.GetSlogStub().GetLogs().size(), 1);
}
TEST_F(KernelLogUT, CheckSuccess_NotReturn_WhenSuccess) {
  GertRuntimeStub stub;
  auto context_holder = KernelRunContextFaker().KernelName("tn").KernelType("tt").Build();

  stub.GetSlogStub().Clear();
  ASSERT_EQ(TestSuccess(context_holder.GetContext<KernelContext>(), ge::GRAPH_SUCCESS), ge::GRAPH_SUCCESS);
  ASSERT_EQ(stub.GetSlogStub().GetLogs().size(), 0);
}
}  // namespace gert