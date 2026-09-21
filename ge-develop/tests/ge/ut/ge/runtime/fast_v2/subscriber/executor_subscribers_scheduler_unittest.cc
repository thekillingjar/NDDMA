/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "runtime/subscriber/executor_subscribers_scheduler.h"
#include <gtest/gtest.h>
#include "common/bg_test.h"
#include "core/model_v2_executor_unittest.h"
#include "runtime/model_v2_executor.h"
#include "stub/gert_runtime_stub.h"
#include "core/builder/model_v2_executor_builder.h"
#include "core/model_v2_executor_unittest.h"
#include "stub/gert_runtime_stub.h"
#include "runtime/model_v2_executor.h"
#include "subscriber/dumper/executor_dumper.h"
#include "subscriber/dumper/host_executor_dumper.h"
#include "subscriber/profiler/cann_profiler_v2.h"
#include "subscriber/profiler/ge_host_profiler.h"
#include "subscriber/tracer/executor_tracer.h"
#include "hybrid/executor/cann_tracing_profiler.h"
#include "macro_utils/dt_public_scope.h"
#include "common/global_variables/diagnose_switch.h"
#include "macro_utils/dt_public_unscope.h"
namespace gert {
namespace {
class TestSubscriber {
 public:
  static void OnExecuteEvent(TestSubscriber *void_arg, ExecutorEvent event, const void *node, KernelStatus result) {
    return;
  }
};

class TestSubscriber1{
  public:
  static void OnExecuteEvent(SubExeGraphType sub_exe_graph_type, TestSubscriber1 *void_arg, ExecutorEvent event, const void *node, KernelStatus result) {
    void_arg->record_ = 100;
    return;
   }
   size_t record_{0};
   TestSubscriber1(const std::shared_ptr<const SubscriberExtendInfo> &extend_info){};
};
const auto kEnableFunc = []() -> bool { return true; };
}  // namespace

class ExecutorSubscribersSchedulerUT : public bg::BgTest {
 protected:
  void TearDown() {
    ge::diagnoseSwitch::DisableProfiling();
    ge::diagnoseSwitch::DisableDumper();
  }
};
TEST_F(ExecutorSubscribersSchedulerUT, AddProfiler_Ok) {
  ExecutorSubscribersScheduler ess;
  EXPECT_EQ(ess.GetBuiltInSubscriber<CannProfilerV2>(BuiltInSubscriberType::kCannProfilerV2), nullptr);
  ess.AddBuiltIn<CannProfilerV2>(BuiltInSubscriberType::kCannProfilerV2, 1UL, nullptr, kSubExeGraphTypeEnd, kEnableFunc);
  EXPECT_EQ(ess.GetSize(), 1UL);
  EXPECT_EQ(ess.GetBuiltInSubscriber<GeHostProfiler>(BuiltInSubscriberType::kGeProfiling), nullptr);
  ess.AddBuiltIn<GeHostProfiler>(BuiltInSubscriberType::kGeProfiling, 1UL, nullptr, kSubExeGraphTypeEnd,kEnableFunc);
  EXPECT_EQ(ess.GetSize(), 2UL);
  EXPECT_NE(ess.GetBuiltInSubscriber<GeHostProfiler>(BuiltInSubscriberType::kGeProfiling), nullptr);
}

TEST_F(ExecutorSubscribersSchedulerUT, AddDumper_Ok) {
  ExecutorSubscribersScheduler ess;
  EXPECT_EQ(ess.GetBuiltInSubscriber<ExecutorDumper>(BuiltInSubscriberType::kDumper), nullptr);
  ess.AddBuiltIn<ExecutorDumper>(BuiltInSubscriberType::kDumper, 1UL, nullptr, kSubExeGraphTypeEnd, kEnableFunc);
  EXPECT_EQ(ess.GetSize(), 1UL);
  EXPECT_NE(ess.GetBuiltInSubscriber<ExecutorDumper>(BuiltInSubscriberType::kDumper), nullptr);
  EXPECT_EQ(ess.GetBuiltInSubscriber<HostExecutorDumper>(BuiltInSubscriberType::kHostDumper), nullptr);
  ess.AddBuiltIn<HostExecutorDumper>(BuiltInSubscriberType::kHostDumper, 1UL, nullptr, kSubExeGraphTypeEnd, kEnableFunc);
  EXPECT_EQ(ess.GetSize(), 2UL);
  EXPECT_NE(ess.GetBuiltInSubscriber<HostExecutorDumper>(BuiltInSubscriberType::kHostDumper), nullptr);
}

TEST_F(ExecutorSubscribersSchedulerUT, AddBuiltInSubscriber_Failed_OutOfBounds) {
  ExecutorSubscribersScheduler ess;
  ess.AddBuiltIn<CannProfilerV2>(BuiltInSubscriberType::kNum, 1UL, nullptr, kSubExeGraphTypeEnd, kEnableFunc);
  EXPECT_EQ(ess.GetSize(), 0UL);
}
TEST_F(ExecutorSubscribersSchedulerUT, SetMultiple_Ok) {
  GertRuntimeStub stub;
  stub.GetKernelStub().AllKernelRegisteredAndSuccess();

  auto executor_pack = BuildExecutorFromSingleNode();
  ASSERT_NE(executor_pack.executor, nullptr);
  ExecutorSubscribersScheduler ess;
  const auto &extend_info = ge::MakeShared<const SubscriberExtendInfo>(
      executor_pack.executor.get(), executor_pack.exe_graph, nullptr, ge::ModelData{}, nullptr,
      SymbolsToValue{}, 0, "", nullptr, std::unordered_map<std::string, TraceAttr>{});
  ess.Init(extend_info);
  // init profiler automatically
  EXPECT_NE(ess.GetBuiltInSubscriber<GeHostProfiler>(BuiltInSubscriberType::kGeProfiling), nullptr);
  ess.AddSubscriber<TestSubscriber>();
  // CannTracer是external不是内置的
  EXPECT_EQ(ess.GetSize(), static_cast<uint64_t>(BuiltInSubscriberType::kNum) + 1U);
}

TEST_F(ExecutorSubscribersSchedulerUT, AddSubscriberMakeEnable_Ok) {
  ExecutorSubscribersScheduler ess;
  EXPECT_EQ(ess.GetBuiltInSubscriber<GeHostProfiler>(BuiltInSubscriberType::kGeProfiling), nullptr);
  EXPECT_EQ(ess.GetBuiltInSubscriber<CannProfilerV2>(BuiltInSubscriberType::kCannProfilerV2), nullptr);

  GertRuntimeStub stub;
  stub.GetKernelStub().AllKernelRegisteredAndSuccess();

  auto executor_pack = BuildExecutorFromSingleNode();;
  ASSERT_NE(executor_pack.executor, nullptr);
  const auto &extend_info = ge::MakeShared<const SubscriberExtendInfo>(
      executor_pack.executor.get(), executor_pack.exe_graph, nullptr, ge::ModelData{}, nullptr,
      SymbolsToValue{}, 0, "", nullptr, std::unordered_map<std::string, TraceAttr>{});
  ess.Init(extend_info);
  GlobalProfilingWrapper::GetInstance()->SetEnableFlags(0UL);
  GlobalDumper::GetInstance()->SetEnableFlags(0UL);
  dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 0);
  EXPECT_FALSE(ess.IsEnable());
  ess.AddSubscriber<TestSubscriber>();
  EXPECT_TRUE(ess.IsEnable());
}

TEST_F(ExecutorSubscribersSchedulerUT, ReturnSchedulerCallback_Ok_MultipleSubscribers) {
  ExecutorSubscribersScheduler ess;
  EXPECT_EQ(ess.GetBuiltInSubscriber<GeHostProfiler>(BuiltInSubscriberType::kGeProfiling), nullptr);
  ess.AddBuiltIn<GeHostProfiler>(BuiltInSubscriberType::kGeProfiling, 1UL, nullptr, kSubExeGraphTypeEnd, kEnableFunc);
  ess.AddSubscriber<TestSubscriber>();
  EXPECT_TRUE(ess.GetSubscriber(kMainExeGraph).callback ==
              reinterpret_cast<::SubscriberFunc>(ExecutorSubscribersScheduler::OnExecuteEvent));
}

TEST_F(ExecutorSubscribersSchedulerUT, AddDuplicatedBuiltInSubscriber_Ok) {
  ExecutorSubscribersScheduler ess;
  EXPECT_EQ(ess.GetBuiltInSubscriber<GeHostProfiler>(BuiltInSubscriberType::kGeProfiling), nullptr);
  ess.AddBuiltIn<GeHostProfiler>(BuiltInSubscriberType::kGeProfiling, 1UL, nullptr, kSubExeGraphTypeEnd, kEnableFunc);
  auto profiler = ess.GetBuiltInSubscriber<GeHostProfiler>(BuiltInSubscriberType::kGeProfiling);
  EXPECT_NE(profiler, nullptr);

  ess.AddBuiltIn<GeHostProfiler>(BuiltInSubscriberType::kGeProfiling, 1UL, nullptr, kSubExeGraphTypeEnd, kEnableFunc);
  auto profiler2 = ess.GetBuiltInSubscriber<GeHostProfiler>(BuiltInSubscriberType::kGeProfiling);
  EXPECT_TRUE(profiler == profiler2);
  EXPECT_EQ(ess.GetSize(), 1UL);
}
TEST_F(ExecutorSubscribersSchedulerUT, RemoveSubscriber_Ok) {
  GertRuntimeStub stub;
  stub.GetKernelStub().AllKernelRegisteredAndSuccess();

  auto executor_pack = BuildExecutorFromSingleNode();;
  ASSERT_NE(executor_pack.executor, nullptr);
  ExecutorSubscribersScheduler ess;
  const auto &extend_info = ge::MakeShared<const SubscriberExtendInfo>(
      executor_pack.executor.get(), executor_pack.exe_graph, nullptr, ge::ModelData{}, nullptr,
      SymbolsToValue{}, 0, "", nullptr, std::unordered_map<std::string, TraceAttr>{});
  ess.Init(extend_info);
  EXPECT_NE(ess.GetBuiltInSubscriber<GeHostProfiler>(BuiltInSubscriberType::kGeProfiling), nullptr);
  auto sub1 = ess.AddSubscriber<TestSubscriber>();
  auto sub2 = ess.AddSubscriber<TestSubscriber>();
  EXPECT_EQ(ess.GetSize(), static_cast<uint64_t>(BuiltInSubscriberType::kNum) + 2UL);

  ess.RemoveSubscriber(sub1);
  EXPECT_EQ(ess.GetSize(), static_cast<uint64_t>(BuiltInSubscriberType::kNum) + 1UL);

  ess.RemoveSubscriber(sub1);
  EXPECT_EQ(ess.GetSize(), static_cast<uint64_t>(BuiltInSubscriberType::kNum) + 1UL);

  ess.RemoveSubscriber(sub2);
  EXPECT_EQ(ess.GetSize(), static_cast<uint64_t>(BuiltInSubscriberType::kNum));
}
TEST_F(ExecutorSubscribersSchedulerUT, RemoveBuiltInProfiling_Ok) {
  ExecutorSubscribersScheduler ess;
  ess.AddBuiltIn<GeHostProfiler>(BuiltInSubscriberType::kGeProfiling, 1UL, nullptr, kSubExeGraphTypeEnd, kEnableFunc);
  auto profiler = ess.MutableBuiltInSubscriber<GeHostProfiler>(BuiltInSubscriberType::kGeProfiling);
  ASSERT_NE(profiler, nullptr);
  EXPECT_EQ(ess.GetSize(), 1UL);

  ess.RemoveSubscriber(profiler);
  EXPECT_EQ(ess.GetSize(), 0UL);
  EXPECT_EQ(ess.GetBuiltInSubscriber<GeHostProfiler>(BuiltInSubscriberType::kGeProfiling), nullptr);
}

TEST_F(ExecutorSubscribersSchedulerUT, RemoveBuiltInHostDumper_Ok) {
  ExecutorSubscribersScheduler ess;
  ess.AddBuiltIn<HostExecutorDumper>(BuiltInSubscriberType::kHostDumper, 1UL, nullptr, kSubExeGraphTypeEnd, kEnableFunc);
  auto dumper = ess.MutableBuiltInSubscriber<HostExecutorDumper>(BuiltInSubscriberType::kHostDumper);
  ASSERT_NE(dumper, nullptr);
  EXPECT_EQ(ess.GetSize(), 1UL);

  ess.RemoveSubscriber(dumper);
  EXPECT_EQ(ess.GetSize(), 0UL);
  EXPECT_EQ(ess.GetBuiltInSubscriber<HostExecutorDumper>(BuiltInSubscriberType::kHostDumper), nullptr);
}

TEST_F(ExecutorSubscribersSchedulerUT, RemoveBuiltInDumper_Ok) {
  GertRuntimeStub stub;
  stub.GetKernelStub().AllKernelRegisteredAndSuccess();
  auto executor_pack = BuildExecutorFromSingleNode();;
  ASSERT_NE(executor_pack.executor, nullptr);
  ExecutorSubscribersScheduler ess;
  const auto &extend_info = ge::MakeShared<const SubscriberExtendInfo>(
      executor_pack.executor.get(), executor_pack.exe_graph, nullptr, ge::ModelData{}, nullptr,
      SymbolsToValue{}, 0, "", nullptr, std::unordered_map<std::string, TraceAttr>{});
  ess.Init(extend_info);
  auto dumper = ess.MutableBuiltInSubscriber<ExecutorDumper>(BuiltInSubscriberType::kDumper);
  ASSERT_NE(dumper, nullptr);
  EXPECT_EQ(ess.GetSize(), static_cast<uint64_t>(BuiltInSubscriberType::kNum));

  ess.RemoveSubscriber(dumper);
  EXPECT_EQ(ess.GetSize(), static_cast<uint64_t>(BuiltInSubscriberType::kNum) - 1UL);
  EXPECT_EQ(ess.GetBuiltInSubscriber<ExecutorDumper>(BuiltInSubscriberType::kDumper), nullptr);
}

TEST_F(ExecutorSubscribersSchedulerUT, StartProfilingByGlobalSwitch_Ok) {
  ExecutorSubscribersScheduler ess;
  const auto &extend_info = ge::MakeShared<const SubscriberExtendInfo>(
      reinterpret_cast<ModelV2Executor *>(100), nullptr, nullptr, ge::ModelData{}, nullptr, SymbolsToValue{}, 0, "",
      nullptr, std::unordered_map<std::string, TraceAttr>{});
  ess.Init(extend_info);
  ess.IsEnable();
  EXPECT_NE(ess.GetBuiltInSubscriber<CannProfilerV2>(BuiltInSubscriberType::kCannProfilerV2), nullptr);
  EXPECT_NE(ess.GetBuiltInSubscriber<GeHostProfiler>(BuiltInSubscriberType::kGeProfiling), nullptr);
  EXPECT_EQ(ess.GetSize(), static_cast<uint64_t>(BuiltInSubscriberType::kNum));

  ge::diagnoseSwitch::EnableProfiling({ProfilingType::kGeHost});
  EXPECT_EQ(ess.GetSize(), static_cast<uint64_t>(BuiltInSubscriberType::kNum));
  EXPECT_TRUE(ess.IsEnable());

  ge::diagnoseSwitch::MutableProfiling().SetEnableFlag(0UL);
  ge::diagnoseSwitch::MutableDumper().SetEnableFlag(0UL);
  // profiler in it
  EXPECT_EQ(ess.GetSize(), static_cast<uint64_t>(BuiltInSubscriberType::kNum));
  EXPECT_FALSE(ess.IsEnable());
}

TEST_F(ExecutorSubscribersSchedulerUT, StartDumperByGlobalSwitch_Ok) {
  ge::diagnoseSwitch::MutableDumper().SetEnableFlag(0UL);
  SpaceRegistryFaker::UpdateOpImplToDefaultSpaceRegistry();
  auto executor = BuildExecutorFromSingleNode().executor;
  ASSERT_NE(executor, nullptr);
  EXPECT_EQ(executor->GetSubscribers().GetSize(), static_cast<uint64_t>(BuiltInSubscriberType::kNum));
  EXPECT_FALSE(executor->GetSubscribers().IsEnable());
  ge::diagnoseSwitch::MutableDumper().SetEnableFlag(1UL);
  EXPECT_EQ(executor->GetSubscribers().GetSize(), static_cast<uint64_t>(BuiltInSubscriberType::kNum));
  EXPECT_NE(executor->GetSubscribers().GetBuiltInSubscriber<ExecutorDumper>(BuiltInSubscriberType::kDumper), nullptr);
  EXPECT_TRUE(executor->GetSubscribers()
                  .MutableBuiltInSubscriber<ExecutorDumper>(BuiltInSubscriberType::kDumper)
                  ->IsEnable(DumpType::kDataDump));
  EXPECT_TRUE(executor->GetSubscribers().IsEnable());

  ge::diagnoseSwitch::EnableExceptionDump();
  EXPECT_TRUE(executor->GetSubscribers()
                  .MutableBuiltInSubscriber<ExecutorDumper>(BuiltInSubscriberType::kDumper)
                  ->IsEnable(DumpType::kExceptionDump));

  ge::diagnoseSwitch::MutableDumper().SetEnableFlag(0);
  EXPECT_EQ(executor->GetSubscribers().GetSize(), static_cast<uint64_t>(BuiltInSubscriberType::kNum));
  EXPECT_FALSE(executor->GetSubscribers().IsEnable());
}

TEST_F(ExecutorSubscribersSchedulerUT, GetSubcriber_GetWorkingBuiltInSubscriber_WithMultipleSubscribersEnabled) {
  ExecutorSubscribersScheduler ssh{};
  ssh.Init(nullptr);
  GlobalProfilingWrapper::GetInstance()->SetEnableFlags(
      BuiltInSubscriberUtil::BuildEnableFlags<ProfilingType>({ProfilingType::kCannHost}));
  GlobalDumper::GetInstance()->SetEnableFlags(1);
  dlog_setlevel(GE_MODULE_NAME, DLOG_INFO, 0);
  auto subscriber = ssh.GetSubscriber(kMainExeGraph);
  EXPECT_EQ(reinterpret_cast<::SubscriberFunc>(subscriber.callback),
            reinterpret_cast<::SubscriberFunc>(ExecutorSubscribersScheduler::OnExecuteEvent));
  EXPECT_EQ(ssh.GetWorkingSubscribers().size(), 3);
  dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 0);
  GlobalProfilingWrapper::GetInstance()->SetEnableFlags(0);
  GlobalDumper::GetInstance()->SetEnableFlags(0);
}

TEST_F(ExecutorSubscribersSchedulerUT, GetSubcriber_GetWorkingBuiltInSubscriber_WithSingleSubscribersEnabled) {
  ExecutorSubscribersScheduler ssh{};
  ssh.Init(nullptr);
  dlog_setlevel(GE_MODULE_NAME, DLOG_INFO, 0);
  auto subscriber = ssh.GetSubscriber(kInitExeGraph);
  EXPECT_EQ(reinterpret_cast<::SubscriberFunc>(subscriber.callback),
            reinterpret_cast<::SubscriberFunc>(ExecutorTracer::OnExecuteEvent));
  EXPECT_EQ(ssh.GetWorkingSubscribers().size(), 1);
  dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 0);
}

TEST_F(ExecutorSubscribersSchedulerUT, GetSubcriber_OnExecuteEventSuccess_DisabledFuncWorkOnMainSubgraph) {
  ExecutorSubscribersScheduler ssh{};
  auto enable_true_func = []() -> bool { return true; };
  auto enable_false_func = []() -> bool { return false; };
  ssh.AddBuiltIn<TestSubscriber1>(BuiltInSubscriberType::kDumper, 1, nullptr, kInitExeGraph, enable_true_func);
  auto subscriber = ssh.GetSubscriber(kInitExeGraph);
  subscriber.callback(kInitExeGraph, subscriber.arg, kExecuteStart, nullptr, 0);
  EXPECT_EQ(ssh.GetBuiltInSubscriber<TestSubscriber1>(BuiltInSubscriberType::kDumper)->record_, 100);
  ssh.Clear();
  ssh.AddBuiltIn<TestSubscriber1>(BuiltInSubscriberType::kDumper, 1, nullptr, kMainExeGraph, enable_false_func);
  subscriber = ssh.GetSubscriber(kInitExeGraph);
  subscriber.callback(kMainExeGraph, subscriber.arg, kExecuteStart, nullptr, 0);
  ExecutorSubscribersScheduler::OnExecuteEvent(kMainExeGraph, &ssh, kExecuteStart, nullptr, 0);
  EXPECT_EQ(ssh.GetBuiltInSubscriber<TestSubscriber1>(BuiltInSubscriberType::kDumper)->record_, 0);
}

TEST_F(ExecutorSubscribersSchedulerUT, AddBuiltIn_GetWorkingSubscribers_OnlyOverflowDumpEnabled) {
  ge::diagnoseSwitch::EnableOverflowDump();
  ExecutorSubscribersScheduler ssh{};

  ssh.Init(nullptr);
  auto subscriber = ssh.GetSubscriber(kInitExeGraph);
  EXPECT_EQ(reinterpret_cast<::SubscriberFunc>(subscriber.callback),
            reinterpret_cast<::SubscriberFunc>(ExecutorSubscribersScheduler::OnExecuteEvent));
  EXPECT_EQ(ssh.GetWorkingSubscribers().size(), 0);

  subscriber = ssh.GetSubscriber(kMainExeGraph);
  EXPECT_EQ(reinterpret_cast<::SubscriberFunc>(subscriber.callback),
            reinterpret_cast<::SubscriberFunc>(ExecutorDumper::OnExecuteEvent));
  EXPECT_EQ(ssh.GetWorkingSubscribers().size(), 1);

  subscriber = ssh.GetSubscriber(kDeInitExeGraph);
  EXPECT_EQ(reinterpret_cast<::SubscriberFunc>(subscriber.callback),
            reinterpret_cast<::SubscriberFunc>(ExecutorSubscribersScheduler::OnExecuteEvent));
  EXPECT_EQ(ssh.GetWorkingSubscribers().size(), 0);
}


}  // namespace gert
