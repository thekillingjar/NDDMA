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
#include "subscriber/profiler/cann_host_profiler.h"
#include "depends/profiler/src/profiling_auto_checker.h"
#include "core/model_v2_executor_unittest.h"
#include "common/global_variables/diagnose_switch.h"
#include "common/fake_node_helper.h"

namespace gert {
namespace {
void TestDoProf1() {
  ge::diagnoseSwitch::EnableProfiling({ProfilingType::kCannHost});
  auto executor = BuildExecutorFromSingleNodeForDump();
  auto host_profiler = executor->GetSubscribers().MutableBuiltInSubscriber<CannHostProfiler>(BuiltInSubscriberType::kCannHostProfiler);
  auto fake_node = FakeNodeHelper::FakeNode("test", "test", 10000);
  host_profiler->DoProf(kExecuteStart, &fake_node.node);
  host_profiler->DoProf(kExecuteEnd, &fake_node.node);
}

void TestDoProf2() {
  ge::diagnoseSwitch::EnableProfiling({ProfilingType::kCannHost});
  auto executor = BuildExecutorFromSingleNodeForDump();
  auto host_profiler = executor->GetSubscribers().MutableBuiltInSubscriber<CannHostProfiler>(BuiltInSubscriberType::kCannHostProfiler);
  auto &host_sch_info = host_profiler->GetHostSchData();
  auto fake_node_id = 0;
  for (size_t i = 0; i<host_sch_info.size(); ++i) {
    if(host_sch_info[i].itemId != 0) {
      fake_node_id = i;
      break;
    }
  }
  auto fake_node = FakeNodeHelper::FakeNode("test", "test", fake_node_id);
  host_profiler->DoProf(kExecuteStart, &fake_node.node);
  host_profiler->DoProf(kExecuteEnd, &fake_node.node);
}
}
class CannHostProfilerUT : public bg::BgTest {};

TEST_F(CannHostProfilerUT, Init_InitHostSchData_WhenConstruct) {
  ge::diagnoseSwitch::EnableProfiling({ProfilingType::kCannHost});
  auto executor =  BuildExecutorFromSingleNodeForDump();
  auto host_profiler = executor->GetSubscribers().GetBuiltInSubscriber<CannHostProfiler>(BuiltInSubscriberType::kCannHostProfiler);
  EXPECT_TRUE(host_profiler->IsHostProfInited());
  EXPECT_FALSE(host_profiler->GetHostSchData().empty());
}

TEST_F(CannHostProfilerUT, DoProf_ReportNothing_WhenNodeIdOutOfBound) {
  ge::EXPECT_DefaultProfilingTestWithExpectedCallTimes(TestDoProf1, 0, 0, 0, 0);
}

TEST_F(CannHostProfilerUT, DoProf_ReportApi_WhenProfilerNodeExecuteEnd) {
  ge::EXPECT_DefaultProfilingTestWithExpectedCallTimes(TestDoProf2, 1, 0, 0, 0);
}

TEST_F(CannHostProfilerUT, OnExecuteEvent_DoInit_OnModelStart) {
  ge::diagnoseSwitch::DisableProfiling();
  auto executor = BuildExecutorFromSingleNodeForDump();
  auto host_profiler =
      executor->GetSubscribers().GetBuiltInSubscriber<CannHostProfiler>(BuiltInSubscriberType::kCannHostProfiler);
  CannHostProfiler::OnExecuteEvent(kMainExeGraph, const_cast<CannHostProfiler *>(host_profiler), kModelStart, nullptr,
                                   0);
  EXPECT_TRUE(host_profiler->IsHostProfInited());
  EXPECT_TRUE(!host_profiler->GetHostSchData().empty());
}

TEST_F(CannHostProfilerUT, DoInit_RecordUnknownName_WithEmptyComputeNodeInfo) {
  ge::diagnoseSwitch::DisableProfiling();
  auto executor = BuildExecutorFromSingleNodeForDump();
  auto execution_data = const_cast<ExecutionData *>(
      static_cast<const ExecutionData *>(executor->GetExeGraphExecutor(kMainExeGraph)->GetExecutionData()));

  for (size_t i = 0UL; i < execution_data->base_ed.node_num; ++i) {
    const auto node = execution_data->base_ed.nodes[i];
    node->context.compute_node_info = nullptr;
  }
  auto host_profiler =
      executor->GetSubscribers().GetBuiltInSubscriber<CannHostProfiler>(BuiltInSubscriberType::kCannHostProfiler);

  CannHostProfiler::OnExecuteEvent(kMainExeGraph, const_cast<CannHostProfiler *>(host_profiler), kModelStart, nullptr,
                                   0);
  EXPECT_TRUE(host_profiler->IsHostProfInited());
  EXPECT_TRUE(!host_profiler->GetHostSchData().empty());
  std::hash<std::string> hs;
  for (auto &host_sch_data : host_profiler->GetHostSchData()) {
    if (host_sch_data.itemId != 0) {
      ASSERT_EQ(host_sch_data.itemId, hs("UnknownNodeName"));
    }
  }
}
}
