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
#include "macro_utils/dt_public_scope.h"
#include "common/global_variables/diagnose_switch.h"
#include "depends/profiler/src/profiling_test_util.h"
#include "common/model_v2_executor_test_helper.h"
#include "runtime/subscriber/global_profiler.h"
#include "common/fake_node_helper.h"
#include "core/model_v2_executor_unittest.h"
#include "stub/gert_runtime_stub.h"
#include "subscriber/profiler/ge_host_profiler.h"
#include "runtime/model_v2_executor.h"
#include "core/executor_error_code.h"
#include "macro_utils/dt_public_unscope.h"

namespace gert {
class GeHostProfilerUT : public bg::BgTest {
 protected:
  void SetUp() {
    ge::diagnoseSwitch::DisableProfiling();
  }
  void TearDown() {
    GlobalProfilingWrapper::GetInstance()->Free();
    ge::diagnoseSwitch::DisableProfiling();
  }
};

TEST_F(GeHostProfilerUT, ProfilerInit_NotRegister_DisableProfiling) {
  GlobalProfilingWrapper::GetInstance()->SetEnableFlags(0UL);
  GertRuntimeStub stub;
  stub.GetKernelStub().AllKernelRegisteredAndSuccess();
  auto executor = BuildExecutorFromSingleNode().executor;
  ASSERT_NE(executor, nullptr);
  auto profiler =
      executor->GetSubscribers().MutableBuiltInSubscriber<GeHostProfiler>(BuiltInSubscriberType::kGeProfiling);
  EXPECT_FALSE(profiler->is_inited_);
}

TEST_F(GeHostProfilerUT, ProfilerInit_RegisterWhenModelStart) {
  GlobalProfilingWrapper::GetInstance()->SetEnableFlags(
      BuiltInSubscriberUtil::BuildEnableFlags<ProfilingType>({ProfilingType::kGeHost}));
  GertRuntimeStub stub;
  stub.GetKernelStub().AllKernelRegisteredAndSuccess();
  auto executor = BuildExecutorFromSingleNode().executor;
  ASSERT_NE(executor, nullptr);
  auto profiler =
      executor->GetSubscribers().MutableBuiltInSubscriber<GeHostProfiler>(BuiltInSubscriberType::kGeProfiling);
  profiler->is_inited_ = false;
  gert::GeHostProfiler::OnExecuteEvent(1, profiler, kModelStart, nullptr, kStatusSuccess);
  EXPECT_NE(profiler->global_prof_wrapper_->idx_to_str_[profiling::kProfilingIndexEnd], "");
  EXPECT_TRUE(profiler->is_inited_);
}

TEST_F(GeHostProfilerUT, InitAndRegister_Ok_EnableGeHostProfiling) {
  GlobalProfilingWrapper::GetInstance()->SetEnableFlags(
      BuiltInSubscriberUtil::BuildEnableFlags<ProfilingType>({ProfilingType::kGeHost}));

  GertRuntimeStub stub;
  stub.GetKernelStub().AllKernelRegisteredAndSuccess();

  auto executor = BuildExecutorFromSingleNode().executor;
  ASSERT_NE(executor, nullptr);
  auto profiler =
      executor->GetSubscribers().MutableBuiltInSubscriber<GeHostProfiler>(BuiltInSubscriberType::kGeProfiling);
  EXPECT_NE(profiler->GetGlobalProf()->idx_to_str_[profiling::kProfilingIndexEnd], "");
}

TEST_F(GeHostProfilerUT, RecordHostProfiling_Ok) {
  GlobalProfilingWrapper::GetInstance()->Init(
      BuiltInSubscriberUtil::BuildEnableFlags<ProfilingType>({ProfilingType::kGeHost}));
  GertRuntimeStub stub;
  stub.GetKernelStub().AllKernelRegisteredAndSuccess();
  auto executor = BuildExecutorFromSingleNode().executor;
  ASSERT_NE(executor, nullptr);
  auto profiler =
      executor->GetSubscribers().MutableBuiltInSubscriber<GeHostProfiler>(BuiltInSubscriberType::kGeProfiling);
  auto node_holder = FakeNodeHelper::FakeNode("test1", "test2", 1);
  profiler->Record(nullptr, kModelStart);
  profiler->Record(nullptr, kModelEnd);
  profiler->Record(&node_holder.node, kExecuteStart);
  profiler->Record(&node_holder.node, kExecuteEnd);

  GeHostProfiler::OnExecuteEvent(1, profiler, kModelStart, nullptr, kStatusSuccess);
  GeHostProfiler::OnExecuteEvent(1, profiler, kModelEnd, nullptr, kStatusSuccess);
  GeHostProfiler::OnExecuteEvent(1, profiler, kExecuteStart, &node_holder.node, kStatusSuccess);
  GeHostProfiler::OnExecuteEvent(1, profiler, kExecuteEnd, &node_holder.node, kStatusSuccess);

  EXPECT_EQ(GlobalProfilingWrapper::GetInstance()->GetRecordCount(), 8);
  GlobalProfilingWrapper::GetInstance()->Free();
}

TEST_F(GeHostProfilerUT, Record_Failed_NullArg) {
  GlobalProfilingWrapper::GetInstance()->Init(
      BuiltInSubscriberUtil::BuildEnableFlags<ProfilingType>({ProfilingType::kGeHost}));
  EXPECT_NO_THROW(GeHostProfiler::OnExecuteEvent(0, nullptr, kModelStart, nullptr, kStatusSuccess));
  EXPECT_NO_THROW(GeHostProfiler::OnExecuteEvent(0, nullptr, kModelEnd, nullptr, kStatusSuccess));
  EXPECT_NO_THROW(GeHostProfiler::OnExecuteEvent(0, nullptr, kExecuteStart, nullptr, kStatusSuccess));
  EXPECT_NO_THROW(GeHostProfiler::OnExecuteEvent(0, nullptr, kExecuteEnd, nullptr, kStatusSuccess));
  EXPECT_EQ(GlobalProfilingWrapper::GetInstance()->GetRecordCount(), 0);
}

TEST_F(GeHostProfilerUT, Record_DoNothing_NotEnableFlag) {
  auto node_holder1 = FakeNodeHelper::FakeNode("NodeName1", "InferShape");
  auto node_holder2 = FakeNodeHelper::FakeNode("NodeName2", "InferShape");
  GlobalProfilingWrapper::GetInstance()->Init(0UL);
  auto profiler = std::make_unique<GeHostProfiler>(nullptr);
  ASSERT_NE(profiler, nullptr);

  GeHostProfiler::OnExecuteEvent(1, profiler.get(), kModelStart, nullptr, kStatusSuccess);
  GeHostProfiler::OnExecuteEvent(1, profiler.get(), kModelEnd, nullptr, kStatusSuccess);
  GeHostProfiler::OnExecuteEvent(1, profiler.get(), kExecuteStart, &node_holder1.node, kStatusSuccess);
  GeHostProfiler::OnExecuteEvent(1, profiler.get(), kExecuteEnd, &node_holder1.node, kStatusSuccess);
  GeHostProfiler::OnExecuteEvent(1, profiler.get(), kExecuteStart, &node_holder2.node, kStatusSuccess);
  GeHostProfiler::OnExecuteEvent(1, profiler.get(), kExecuteEnd, &node_holder2.node, kStatusSuccess);

  EXPECT_EQ(GlobalProfilingWrapper::GetInstance()->GetRecordCount(), 0UL);
}
}
